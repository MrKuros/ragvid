

def test_stat_notes_do_not_fire_from_unmeasured_defaults():
    """A session written before the percentile fields existed loads with
    ClipStats defaults and session.py does not migrate, so an unguarded
    threshold would feed the model instructions about footage nobody measured."""
    from ragvid.probe import ClipStats
    from ragvid.vibe import _stat_notes

    old = ClipStats(mean=dict(r=0.5, g=0.5, b=0.5), std=dict(r=0.2, g=0.2, b=0.2),
                    saturation=0.2, dominant_hue=30.0, clipped_high=0.0,
                    crushed_low=0.0, width=1920, height=1080, duration=4.0,
                    frames_sampled=10)
    assert old.p99.r == 0.0 and old.frame_variance == 0.0
    assert [n for fires, n in _stat_notes(old) if fires] == []


# ---- refining the verb list, judged on pixels (roadmap B7) -----------------
#
# Every assertion below grades a real block of pixels and re-measures it. A spec
# field moving is not evidence: look_mix, saturation and the tonal split can all
# eat a temperature push, which is why CLAUDE.md's rule is "assert on measured
# pixels". The model's reply is scripted here -- what a live model actually
# answers is scripts/bakeoff_intent.py's job, and it costs tokens.

import json

import numpy as np

from ragvid.compiler import compile_intent
from ragvid.intent import Intent, Op
from ragvid.probe import ClipStats
from ragvid.refine import refine_intent
from ragvid.spec import RGB

# A seeded block of pixels standing in for a clip, and STATS measured off it, so
# the compiler is reading the same footage the assertions re-measure. Any other
# statistics would be a different experiment for compile_intent, which consults
# them.
PIXELS = np.random.default_rng(0).random((4096, 3))


def _rgb(v) -> RGB:
    return RGB(r=float(v[0]), g=float(v[1]), b=float(v[2]))


_hi, _lo = PIXELS.max(axis=1), PIXELS.min(axis=1)
STATS = ClipStats(
    mean=_rgb(PIXELS.mean(axis=0)),
    std=_rgb(PIXELS.std(axis=0)),
    p1=_rgb(np.percentile(PIXELS, 1, axis=0)),
    p50=_rgb(np.percentile(PIXELS, 50, axis=0)),
    p99=_rgb(np.percentile(PIXELS, 99, axis=0)),
    saturation=float(np.mean((_hi - _lo) / np.maximum(_hi, 1e-8))),
    frames_sampled=6, width=640, height=360, duration=4.0,
)


class Scripted:
    """A schema-enforced provider that answers with one prepared Intent.

    `plan` raises: a refinement that reached the 43-number path would still
    produce a grade, and the point of these tests is which subject the model was
    handed, so falling back has to be loud.
    """

    name = "fake"
    model = "fake-model"
    schema_enforced = True

    def __init__(self, reply: Intent):
        self.reply = reply
        self.calls: list[str] = []

    def plan(self, system, user):
        raise AssertionError("refine_intent must never fall back to the 43-number path")

    def plan_json(self, system, user, schema):
        self.calls.append(user)
        return json.loads(self.reply.model_dump_json())


def _warmth(spec) -> float:
    """Measured r - b of the graded pixels, against the ungraded ones."""
    out = np.clip(spec.apply(PIXELS), 0.0, 1.0)
    return float((out[:, 0] - out[:, 2]).mean() - (PIXELS[:, 0] - PIXELS[:, 2]).mean())


WARM = Intent(ops=[Op(op="warmth", dir="up", amount="moderate")])


def test_less_warm_lands_between_the_source_and_the_grade_it_is_refining():
    """Neither a reset nor an overshoot. This is what "relative" has to mean:
    the model is editing a list that already says how warm the picture is, so
    "a bit less" has something to be less than."""
    full = _warmth(compile_intent(WARM, STATS))
    refined = refine_intent(
        WARM, "a bit less warm", STATS,
        provider=Scripted(Intent(ops=[Op(op="warmth", dir="up", amount="subtle")])))
    less = _warmth(compile_intent(refined, STATS))

    # Measured on this block: full +0.0207, less +0.0083 = 0.40x. UNIT["subtle"]
    # is 0.4 of "moderate", so the pixels move by the ratio the words did.
    assert 0.0 < less < full
    assert less > 0.15 * full, "a refine that lands back at the source is a reset"


def test_half_strength_measurably_holds_back_the_same_clips_full_grade():
    """The failure the last bake-off measured on the direct path: "warm it up,
    but at half strength" came back byte-identical to its own "warmer", because
    a model authoring 43 numbers from scratch has no memory of the full-strength
    grade to be half of. Here the list IS that memory."""
    full = _warmth(compile_intent(WARM, STATS))
    refined = refine_intent(WARM, "at half strength", STATS,
                            provider=Scripted(WARM.model_copy(update={"strength": "moderate"})))
    half = _warmth(compile_intent(refined, STATS))

    assert [op.model_dump() for op in refined.ops] == [op.model_dump() for op in WARM.ops]
    # look_mix is the outermost step and mixes linearly back toward the source,
    # so STRENGTH_MIX["moderate"] = 0.65 comes out as 0.65x the measured warmth
    # (+0.0207 -> +0.0135, ratio 0.652). Held back on pixels, not in a field.
    assert 0.55 * full < half < 0.75 * full


def test_an_unrelated_refine_leaves_the_other_moves_measurably_untouched():
    """The copy rule, judged where it matters. Adding grain must not move the
    warmth the user already accepted -- and because compile_intent is pure and
    grain is spatial, "untouched" here means bit-for-bit."""
    refined = refine_intent(WARM, "add some grain", STATS, provider=Scripted(
        Intent(ops=[Op(op="warmth", dir="up", amount="moderate"), Op(op="grain", dir="up")])))
    spec = compile_intent(refined, STATS)

    assert _warmth(spec) == _warmth(compile_intent(WARM, STATS))
    assert spec.effects.grain > 0.0
