#!/usr/bin/env python3
"""A1's proof: direct GradeSpec vs intent+compiler, judged on graded PIXELS.

The roadmap says A1 does not ship on taste (docs/ROADMAP.md, "How A1 gets
proven"), so this script is the only thing allowed to decide it. Ten real
grading sentences, one real clip, both paths, and two questions per sentence:

    did the moment the sentence names actually move, in the right direction?
    did the moments it did NOT name stay put?

MEASURED PIXELS, NOT SPEC FIELDS. A spec with temperature +900 is not evidence
that the picture got warmer: look_mix, saturation and the tonal split can all
eat it, and a hue qualifier can move a field without touching a single pixel of
this particular clip. So every check below runs `GradeSpec.apply` over real
frames and re-measures. `check.py`-style field assertions would pass for grades
that do nothing.

WHAT IS NOT MEASURED, and why: the six texture verbs. `GradeSpec.apply`
deliberately never applies EffectSpec (it is spatial and cannot bake into a
.cube; render.py turns it into ffmpeg filters), so a pixel test of "add grain"
would be a test of ffmpeg, not of either path. No prompt here asks for texture.

LIVE MODEL CALLS. Two per prompt, sequential, with SLEEP seconds between them,
capped at CALL_BUDGET. Groq's free tier is 8000 tokens/minute and the direct
path spends ~3000 of them per call, so the sleep is what keeps the run under the
bucket rather than politeness. Token usage for BOTH paths is read off the
provider's own `usage` field — the roadmap's second criterion is that intent
must be measurably cheaper, and a guess at the cost would not settle it.

Usage:  uv run python scripts/bakeoff_intent.py [--sleep 25] [--prompts 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ragvid.compiler import compile_intent, compile_stack  # noqa: E402
from ragvid.intent import describe  # noqa: E402
from ragvid.probe import _ANALYSIS_WIDTH, _ffprobe, _grab, _stats_from_frames, _unit  # noqa: E402
from ragvid.providers import get_provider  # noqa: E402
from ragvid.refine import refine_intent, refine_spec  # noqa: E402
from ragvid.region import GradeStack  # noqa: E402
from ragvid.spec import LUMA, GradeSpec  # noqa: E402
from ragvid.vibe import ask_intent, plan_direct  # noqa: E402

SRC = REPO / "test_files" / "test.mp4"
FRAMES = 6           # enough moments to average over; each is a full decode
SLEEP = 25.0         # seconds between calls. See the docstring: 8000 TPM.
CALL_BUDGET = 25     # hard stop, counting failures. Two other agents share the key.
REFINE_BUDGET = 26   # --refine needs 20 (see REFINEMENTS); the slack is for a retry.

# A moment "moved" if it moved by more than this, in display units (0-1). 0.010
# is ~2.5 8-bit code values: comfortably visible, and far above the arithmetic
# noise, which is zero here — both paths grade the same decoded frames.
MOVE = 0.010
# A moment nobody asked about may drift by twice that before it counts as damage.
KEEP = 0.020
# Chroma is the one exception, and it is measured, not conceded: a pure warmth
# push of the size both paths choose raises measured chroma by +0.023 to +0.034
# while leaving luma flat, because (r - b) IS most of what chroma is. Judging
# `sat` at 0.020 would fail every correct answer to "warmer".
KEEP_SAT = 0.055

_EPS = 1e-8
_RAIL = 1.5 / 255.0


# ---- the prompts and what each one promises about the pixels ---------------
#
# `want` is (moment, direction) pairs the sentence explicitly asks for; `keep`
# is moments the sentence says nothing about, which a good grade leaves alone.
# Both are judged identically for both paths, so the only thing being compared
# is the grade.
PROMPTS: list[tuple[str, dict]] = [
    ("warmer",
     dict(want={"warm": "up"}, keep=("luma", "sat"))),
    ("cooler, like a cold morning",
     dict(want={"warm": "down"}, keep=("luma",))),
    ("crushed blacks",
     dict(want={"black": "down"}, keep=("white",))),
    ("moody but keep it natural",
     # "moody" is darker; "keep it natural" is the constraint that it must not
     # become a colour effect, so the cast and the chroma are the `keep` half.
     dict(want={"luma": "down"}, keep=("sat",))),
    ("make it pop, more punch",
     dict(want={"spread": "up", "sat": "up"}, keep=())),
    ("drain the colour, almost black and white",
     dict(want={"sat": "down"}, keep=("luma",))),
    ("teal shadows, warm highlights",
     # The only check here that a global temperature move cannot fake: warmth of
     # the bright half MINUS warmth of the dark half. And the global cast is the
     # `keep`, because a split tone is not supposed to be a colour cast.
     dict(want={"split": "up"}, keep=("warm", "luma"))),
    ("warm it up, but at half strength",
     # Judged twice: it must warm, and it must warm LESS than plain "warmer" did
     # on the same path. That comparison is the only test of `strength` that
     # cannot be satisfied by writing a field.
     dict(want={"warm": "up"}, keep=("luma",), weaker_than=0, moment="warm")),
    ("make it feel like a rainy night",
     dict(want={"luma": "down", "warm": "down"}, keep=())),
    ("brighter, but don't blow out the highlights",
     # `clipped` is the promise in the second half of the sentence, and it is
     # the one a naive exposure push breaks.
     dict(want={"luma": "up"}, keep=("clipped",))),
]


# ---- turn two: refining a grade the user already accepted (roadmap B7) -----
#
# THE SECOND TURN IS THE WHOLE POINT HERE, and it is judged the same way the
# first one is: apply the resulting grade to real frames and re-measure. What
# differs is the baseline. A refinement is relative, so every check below is
# against TURN ONE's measured moments, not against the source -- "a bit less
# warm" is a claim about the grade the user is already looking at.
#
# The two paths differ in SUBJECT, not transport. Direct hands the model 43
# numbers and takes 43 back (refine_spec); intent hands it the verb list and
# takes an edited list back (refine_intent), which the compiler re-reads the
# clip for. Both then produce a GradeStack, and the stack is what gets applied.
#
# (prompt, instruction, checks). `want`/`keep` are judged turn one -> turn two.
# `between` demands the moment land strictly between the source and turn one --
# the check a reset and an overshoot both fail, and neither can fake. `region`
# demands the masked part of the frame still measurably differ from the rest,
# which is the defect: 43 numbers can only describe the whole frame.
#
# Turn one is CACHED per (path, prompt), so the three refinements of "warmer"
# cost one grading call each path rather than three. Same first turn, three
# different second sentences, which is also the fairest comparison available.
REFINEMENTS: list[tuple[str, str, dict]] = [
    ("darken the top", "make it warmer",
     dict(want={"warm": "up"}, keep=(), region=True)),
    ("warmer", "a bit less warm",
     dict(want={"warm": "down"}, keep=("luma",), between="warm")),
    ("warmer", "keep that, but at half strength",
     dict(want={"warm": "down"}, keep=("luma",), between="warm")),
    # Nothing about warmth in the second sentence, so the warmth from turn one
    # has to be exactly where it was. Grain itself is spatial and unmeasurable
    # here (see the module docstring), which is why it is the perfect probe for
    # whether an unrelated refine damages what it was not asked about.
    ("warmer", "add some grain",
     dict(want={}, keep=("warm", "luma", "sat"))),
    ("moody but keep it natural", "a stop brighter",
     dict(want={"luma": "up"}, keep=("sat",))),
    ("crushed blacks", "actually, lift the blacks back up",
     dict(want={"black": "up"}, keep=("white",))),
]


# ---- measurement -----------------------------------------------------------


def moments(v: np.ndarray) -> dict[str, float]:
    """Everything the checks above can ask about, from (N, 3) display pixels."""
    hi, lo = v.max(axis=1), v.min(axis=1)
    luma = v @ LUMA
    warm = v[:, 0] - v[:, 2]
    bright = luma >= 0.5
    dark = ~bright
    return {
        "luma": float(luma.mean()),
        "warm": float(warm.mean()),
        "sat": float(np.mean((hi - lo) / np.maximum(hi, _EPS))),
        "spread": float(luma.std()),
        "black": float(np.percentile(luma, 1)),
        "white": float(np.percentile(luma, 99)),
        "clipped": float(np.mean(hi >= 1.0 - _RAIL)),
        # 0 if the clip has no pixels on one side of the crossover; the split
        # verbs would then be unmeasurable rather than passing by accident.
        "split": float(warm[bright].mean() - warm[dark].mean())
        if bright.any() and dark.any() else 0.0,
    }


def graded(spec: GradeSpec, pixels: np.ndarray) -> dict[str, float]:
    """`spec` applied to the source pixels, re-measured. Effects are not applied
    (they are spatial; see the module docstring)."""
    return moments(np.clip(spec.apply(pixels), 0.0, 1.0))


def _grade_frames(stack: GradeStack, images: list[np.ndarray]) -> list[np.ndarray]:
    """Every frame through the whole stack — base AND regional layers.

    GradeSpec.apply would be enough for a flat grade and is what the single-turn
    run uses, but a region cannot be applied to a bag of pixels: the mask is a
    function of WHERE the pixel sits. So the second turn measures images.
    """
    return [np.clip(stack.apply(im), 0.0, 1.0) for im in images]


def _moments_of(out: list[np.ndarray]) -> dict[str, float]:
    return moments(np.concatenate([f.reshape(-1, 3) for f in out]))


def _region_gap(out: list[np.ndarray]) -> float:
    """Mean luma of the top eighth of the frame minus the bottom eighth.

    The one measurement that can tell a regional grade from a flat one. A flat
    grade moves both strips together and leaves this at whatever the source's
    own composition made it; "darken the top" pushes it down.
    """
    gaps = []
    for f in out:
        luma = f @ LUMA
        h = max(1, luma.shape[0] // 8)
        gaps.append(float(luma[:h].mean() - luma[-h:].mean()))
    return float(np.mean(gaps))


def judge(before: dict, after: dict, want: dict, keep: tuple) -> list[tuple[str, bool, float]]:
    """(what was checked, did it hold, by how much) for one grade."""
    out = []
    for name, direction in want.items():
        d = after[name] - before[name]
        out.append((f"{name} {direction}", d > MOVE if direction == "up" else d < -MOVE, d))
    for name in keep:
        d = after[name] - before[name]
        out.append((f"{name} steady", abs(d) <= (KEEP_SAT if name == "sat" else KEEP), d))
    return out


# ---- the run ---------------------------------------------------------------


class Meter:
    """Counts calls and tokens by wrapping the provider's own client.

    Both paths bottom out in chat.completions.create, so one wrapper measures
    them on the same footing — and `usage` is the endpoint's count, not an
    estimate of ours.
    """

    def __init__(self, provider, budget: int):
        self.budget = budget
        self.calls = 0
        self.tokens: list[tuple[str, int, int]] = []  # (path, prompt, completion)
        self.path = "?"
        client = provider.client
        create = client.chat.completions.create

        def counted(**kwargs):
            self.calls += 1
            response = create(**kwargs)
            usage = response.usage
            self.tokens.append((self.path, usage.prompt_tokens, usage.completion_tokens))
            return response

        client.chat.completions.create = counted
        if client.chat.completions.create is not counted:
            raise SystemExit("could not instrument the client; refusing to run blind")

    def spend(self, path: str) -> None:
        if self.calls >= self.budget:
            raise SystemExit(f"call budget of {self.budget} reached — stopping, as instructed")
        self.path = path

    def total(self, path: str) -> tuple[int, int]:
        rows = [t for t in self.tokens if t[0] == path]
        return sum(r[1] for r in rows), sum(r[2] for r in rows)


def _dry_provider():
    """`--dry`: the whole harness, no network and no key.

    Exists because the live run is capped at 25 calls, so every bug in the
    decoding, measuring, judging and reporting has to be found before the first
    one is spent (it found two). The canned answer is a warm grade for both
    paths, which is the WRONG answer to most of these prompts on purpose — a dry
    run that reported all PASS would be testing nothing.
    """
    from ragvid.intent import Intent, Op
    from ragvid.providers.openai_compat import OpenAICompatProvider
    from ragvid.vibe import INTENT_SYSTEM

    warm = GradeSpec(temperature=700.0, rationale="A warm look.").model_dump_json()
    intent = Intent(ops=[Op(op="warmth", dir="up")]).model_dump_json()

    def create(model, messages, **kwargs):
        # startswith, not ==: REFINE_INTENT_SYSTEM is INTENT_SYSTEM plus a tail.
        body = intent if messages[0]["content"].startswith(INTENT_SYSTEM) else warm
        usage = type("U", (), {"prompt_tokens": len(messages[0]["content"]) // 4,
                               "completion_tokens": len(body) // 4})()
        message = type("M", (), {"content": body})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()],
                              "usage": usage})()

    completions = type("Completions", (), {"create": staticmethod(create)})()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    # The real provider class, so --dry exercises plan()'s parsing too.
    return OpenAICompatProvider("dry", "", "dry", env_var=None, client=client)


def _pause(sleep: float, first: bool) -> bool:
    if not first:
        time.sleep(sleep)
    return False


def _turn_one(path: str, prompt: str, stats, provider, images: list[np.ndarray]) -> dict:
    """Grade the clip, both paths, and measure the result through the stack."""
    if path == "direct":
        spec = plan_direct(prompt, stats, provider=provider)
        stack, intent, said = GradeStack(base=spec), None, spec.rationale
    else:
        intent = ask_intent(prompt, provider=provider)
        stack = compile_stack(intent, stats)
        said = "; ".join(describe(intent)) or "(nothing)"
    out = _grade_frames(stack, images)
    return {"stack": stack, "intent": intent, "said": said,
            "moments": _moments_of(out), "gap": _region_gap(out)}


def _turn_two(path: str, instruction: str, one: dict, stats, provider) -> tuple[GradeStack, str]:
    """The refinement. Direct edits 43 numbers, intent edits the verb list."""
    if path == "direct":
        spec = refine_spec(one["stack"].base, instruction, stats, provider=provider)
        # No layers, and that is the finding rather than an omission: refine_spec
        # returns one GradeSpec, which can only describe the whole frame.
        return GradeStack(base=spec), spec.rationale
    intent = refine_intent(one["intent"], instruction, stats, provider=provider)
    return compile_stack(intent, stats), "; ".join(describe(intent)) or "(nothing)"


def _between(name: str, src: dict, m1: dict, m2: dict) -> tuple[str, bool, float]:
    """The refined moment must land strictly BETWEEN the source and turn one.

    Both failure modes fail this and nothing else catches both: a reset lands
    back at the source, an overshoot lands past the grade it was refining, and a
    model that ignored the instruction lands exactly on turn one.
    """
    a, b, v = src[name], m1[name], m2[name]
    lo, hi = min(a, b), max(a, b)
    frac = (v - a) / (b - a) if abs(b - a) > _EPS else 0.0
    return (f"{name} between source and turn one ({frac:.0%} of the way)",
            lo + MOVE / 2 < v < hi - MOVE / 2, v - b)


def refine_bakeoff(stats, images, before, provider, meter, sleep: float) -> list[dict]:
    """Two turns per sentence, per path. See REFINEMENTS for what is checked."""
    src_gap = _region_gap(images)
    rows: list[dict] = []
    cache: dict[tuple[str, str], dict] = {}
    first = True

    for prompt, instruction, expect in REFINEMENTS:
        result = {"prompt": f"{prompt!r} then {instruction!r}"}
        for path in ("direct", "intent"):
            try:
                key = (path, prompt)
                if key not in cache:
                    first = _pause(sleep, first)
                    meter.spend(path)
                    cache[key] = _turn_one(path, prompt, stats, provider, images)
                one = cache[key]
                first = _pause(sleep, first)
                meter.spend(path)
                stack, said = _turn_two(path, instruction, one, stats, provider)
            except Exception as exc:  # a 429, a refusal, a malformed reply
                result[path] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {path:7s} FAILED: {type(exc).__name__}: {exc}")
                continue
            out = _grade_frames(stack, images)
            after = _moments_of(out)
            checks = judge(one["moments"], after, expect["want"], expect["keep"])
            if "between" in expect:
                checks.append(_between(expect["between"], before, one["moments"], after))
            if expect.get("region"):
                # THE RAW GAP UNDER-DISCRIMINATES, and the first live run showed
                # exactly how: this clip's top is its sky, so ANY global
                # darkening narrows the top-vs-bottom luma gap. The direct path
                # answered "darken the top" by darkening everything and scored
                # -0.0414 on this check with no region at all, against the verb
                # list's -0.1236 with one. So both are reported: the gap, and
                # then the part of it the LAYERS produced, which is the gap
                # minus the same grade applied flat. That second number is 0.0000
                # for a grade of 43 numbers however good it is, which is the
                # defect rather than an unfair check.
                checks.append(("region graded at turn one", one["gap"] - src_gap < -MOVE,
                               one["gap"] - src_gap))
                flat = _region_gap(_grade_frames(GradeStack(base=stack.base), images))
                checks.append(("region SURVIVES the refine", _region_gap(out) - src_gap < -MOVE,
                               _region_gap(out) - src_gap))
                checks.append(("... and the layers are what did it",
                               _region_gap(out) - flat < -MOVE, _region_gap(out) - flat))
            result[path] = {
                "turn1": one["said"], "summary": said, "checks": checks,
                "moments": after, "layers": len(stack.layers),
                "deltas": {k: after[k] - one["moments"][k] for k in after},
            }
        rows.append(result)
        _report_refine(result)
    return rows


def _report_refine(result: dict) -> None:
    print(f"\n{result['prompt']}")
    for path in ("direct", "intent"):
        r = result[path]
        if "error" in r:
            print(f"  {path:7s} ERROR {r['error']}")
            continue
        marks = "  ".join(f"{'PASS' if ok else 'FAIL'} {name} ({d:+.4f})"
                          for name, ok, d in r["checks"])
        print(f"  {path:7s} {marks}")
        print(f"          turn one: {r['turn1'][:88]}")
        print(f"          turn two: {r['summary'][:88]}   [{r['layers']} layer(s)]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sleep", type=float, default=SLEEP)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--clip", default=str(SRC))
    ap.add_argument("--dry", action="store_true", help="run the harness offline, no model")
    ap.add_argument("--refine", action="store_true",
                    help="the TWO-TURN run (roadmap B7): grade, then refine, and judge "
                         "the second frame against the first")
    args = ap.parse_args()
    if args.dry:
        args.sleep = 0.0

    width, height, duration = _ffprobe(args.clip)
    times = [duration * i / FRAMES for i in range(FRAMES)] if duration > 0 else [0.0]
    frames = [f for f in (_grab(args.clip, t) for t in times) if f is not None]
    if not frames:
        raise SystemExit(f"decoded no frames from {args.clip}")
    # The stats the model and the compiler both see, measured off exactly the
    # pixels the checks below re-measure. Any other clip would be a different
    # experiment for the compiler, which reads them.
    stats = _stats_from_frames(frames, width, height, duration)
    # probe._unit, not a division by 255: ffmpeg returns a 16-bit PNG for 10-bit
    # footage and it is the one place that knows the scale for both depths.
    pixels = np.concatenate([_unit(f) for f in frames])
    before = moments(pixels)

    print(f"clip {Path(args.clip).name}  {width}x{height} {duration:.1f}s, "
          f"{len(frames)} frames at {_ANALYSIS_WIDTH}px  "
          f"({len(pixels):,} pixels)")
    print("  source: " + "  ".join(f"{k} {v:+.4f}" for k, v in before.items()))

    budget = REFINE_BUDGET if args.refine else CALL_BUDGET
    provider = _dry_provider() if args.dry else get_provider()
    print(f"  provider {provider.name} model {provider.model}  "
          f"sleep {args.sleep:.0f}s  budget {budget} calls\n")
    meter = Meter(provider, budget)

    if args.refine:
        # A region is a function of WHERE a pixel sits, so the second turn needs
        # frames, not the flat (N, 3) block the single-turn run measures.
        images = [_unit(f).reshape(f.shape) for f in frames]
        rows = refine_bakeoff(stats, images, before, provider, meter, args.sleep)
        print("\n" + "=" * 78)
        _totals(rows, meter)
        print(f"{meter.calls} calls spent of a {budget} budget")
        out = REPO / "out" / f"bakeoff_refine{'_dry' if args.dry else ''}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"source": before, "rows": rows}, indent=2,
                                  default=str) + "\n")
        print(f"\nfull measurements -> {out.relative_to(REPO)}")
        return 0

    rows = []
    first = True
    for prompt, expect in PROMPTS[: args.prompts]:
        result = {"prompt": prompt}
        for path in ("direct", "intent"):
            if not first:
                time.sleep(args.sleep)
            first = False
            meter.spend(path)
            try:
                if path == "direct":
                    # plan_direct, NOT plan_vibe: plan_vibe routes on
                    # schema_enforced now, so against a real Groq key it would
                    # run the intent path under the "direct" label and the
                    # comparison would silently be intent against itself.
                    spec = plan_direct(prompt, stats, provider=provider)
                    summary = spec.rationale
                else:
                    intent = ask_intent(prompt, provider=provider)
                    spec = compile_intent(intent, stats)
                    summary = "; ".join(describe(intent)) or "(nothing)"
            except Exception as exc:  # a 429, a refusal, a malformed reply
                result[path] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"{prompt!r} [{path}] FAILED: {type(exc).__name__}: {exc}")
                continue
            after = graded(spec, pixels)
            result[path] = {
                "summary": summary,
                "checks": judge(before, after, expect["want"], expect["keep"]),
                "moments": after,
                "deltas": {k: after[k] - before[k] for k in after},
            }
        rows.append(result)
        _report(result, expect, rows)

    print("\n" + "=" * 78)
    _totals(rows, meter)
    # A dry run writes its own file: canned numbers overwriting a real
    # measurement is how evidence gets lost, and it happened once here.
    out = REPO / "out" / f"bakeoff_intent{'_dry' if args.dry else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"source": before, "rows": rows}, indent=2, default=float) + "\n")
    print(f"\nfull measurements -> {out.relative_to(REPO)}")
    return 0


def _report(result: dict, expect: dict, rows: list) -> None:
    print(f"\n{result['prompt']!r}")
    for path in ("direct", "intent"):
        r = result[path]
        if "error" in r:
            print(f"  {path:7s} ERROR {r['error']}")
            continue
        marks = "  ".join(
            f"{'PASS' if ok else 'FAIL'} {name} ({delta:+.4f})" for name, ok, delta in r["checks"]
        )
        print(f"  {path:7s} {marks}")
        print(f"          {r['summary'][:100]}")
        if "weaker_than" in expect:
            # The strength check: same path, same moment, against the earlier prompt.
            ref = rows[expect["weaker_than"]][path]
            if "error" not in ref:
                m = expect["moment"]
                mine = abs(r["deltas"][m])
                theirs = abs(ref["deltas"][m])
                ok = mine < theirs - MOVE / 2
                r["checks"].append((f"{m} weaker than {rows[expect['weaker_than']]['prompt']!r}",
                                    ok, mine - theirs))
                print(f"          {'PASS' if ok else 'FAIL'} half strength moved {m} "
                      f"{mine:.4f} vs {theirs:.4f} at full")


def _totals(rows: list, meter: Meter) -> None:
    for path in ("direct", "intent"):
        checks = [c for r in rows if "checks" in r.get(path, {}) for c in r[path]["checks"]]
        passed = sum(1 for _, ok, _ in checks if ok)
        failed = [r["prompt"] for r in rows
                  if "checks" in r.get(path, {}) and not all(ok for _, ok, _ in r[path]["checks"])]
        errors = [r["prompt"] for r in rows if "error" in r.get(path, {})]
        pin, pout = meter.total(path)
        n = max(1, len([r for r in rows if path in r]))
        print(f"{path:7s} {passed}/{len(checks)} checks, "
              f"{len(rows) - len(failed) - len(errors)}/{len(rows)} prompts clean, "
              f"{len(errors)} errors    "
              f"tokens {pin + pout:6d} total ({pin} in, {pout} out) = {(pin + pout) / n:.0f}/prompt")
        for prompt in failed:
            print(f"          missed: {prompt!r}")


if __name__ == "__main__":
    raise SystemExit(main())
