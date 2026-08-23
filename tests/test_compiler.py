"""What the compiler does to actual pixels.

Every direction claim below is measured on an IMAGE, not read off a spec field.
A spec field moving is not evidence: `highlight_lift` and `offset` both read as
"the shadows went up" in the spec and only one of them leaves the highlights
alone, and the whole reason this module chooses between them is a measurement
nobody can see in the field values.

The single most important test in the file is
`test_the_same_intent_compiles_differently_on_different_footage`. If it fails,
the compiler is ignoring `ClipStats` and the architecture change is pointless --
a compiler that does not consult the measurement is a lookup table, and a lookup
table is not better than the model guessing.

No provider is constructed anywhere in this file. The compiler is a pure
function; that is the point of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragvid.compiler import compile_intent
from ragvid.intent import AMOUNTS, OPS, Intent, Op
from ragvid.lut import _grid
from ragvid.probe import ClipStats, _stats_from_frames
from ragvid.spec import LUMA, RGB, GradeSpec

# ---- the test clip --------------------------------------------------------
#
# A luma ramp plus small per-channel noise, i.e. a plausible piece of footage
# rather than uniform random RGB: real frames have correlated channels and
# modest chroma (probe.py measures saturation ~0.1-0.3 on the repo's own
# assets). It matters here because several assertions below are EXACT, and
# exactness only survives while no pixel touches a rail.
_rng = np.random.default_rng(0)
IMG = np.clip(
    _rng.uniform(0.12, 0.68, size=(20000, 1)) + _rng.normal(0.0, 0.05, size=(20000, 3)),
    0.06, 0.76,
)

# The identity grade's own output, as the oracle for "this changed nothing".
# NOT IMG itself: apply()'s saturation lerp costs ~1 ulp on arbitrary floats
# (spec.py's docstring is about exactly this), so `== IMG` would be asserting a
# property spec.py never claimed.
IDENTITY_OUT = GradeSpec.identity().apply(IMG)

SRC_L = IMG @ LUMA
_order = np.argsort(SRC_L)
DARK = _order[: len(IMG) // 10]          # darkest tenth
MID = _order[int(len(IMG) * 0.45): int(len(IMG) * 0.55)]
BRIGHT = _order[-len(IMG) // 10:]        # brightest tenth

# spec.py's tonal split masks are exactly zero on the far side of luma 0.5, and
# several "unrelated moment did not move" assertions are exact because of it.
# That is only true if the bands really do sit either side of the crossover.
assert SRC_L[DARK].max() < 0.5 < SRC_L[BRIGHT].min()


def stats_of(img: np.ndarray, **override) -> ClipStats:
    """Measure the test image with the REAL measurement code.

    Using probe.py rather than hand-written numbers keeps the compiler's input
    honest: if probe changes what it reports, these tests see it.
    """
    frame = np.round(np.asarray(img) * 255.0).astype(np.uint8)
    st = _stats_from_frames([frame], 640, 360, 4.0)
    return st.model_copy(update=override) if override else st


STATS = stats_of(IMG)


def clip_stats(**kw) -> ClipStats:
    """A hand-built ClipStats, for the tests that need footage we do not have."""
    base = dict(
        mean=RGB.of(0.4), std=RGB.of(0.15), saturation=0.25, frames_sampled=10,
        width=640, height=360, duration=4.0, p1=RGB.of(0.05), p50=RGB.of(0.4),
        p99=RGB.of(0.85), frame_variance=0.04,
    )
    return ClipStats(**{**base, **kw})


def out(intent: Intent, stats: ClipStats | None = None, img=IMG) -> np.ndarray:
    return compile_intent(intent, stats or STATS).apply(img)


def one(op: str, amount: str = "moderate", **kw) -> Intent:
    return Intent(ops=[Op(op=op, amount=amount, **kw)])


# ---- measured moments -----------------------------------------------------


def m_luma(o):      return float((o @ LUMA).mean())
def m_std(o):       return float((o @ LUMA).std())
def m_chroma(o):    return float((o.max(-1) - o.min(-1)).mean())
def m_rb(o):        return float(o[:, 0].mean() - o[:, 2].mean())
def m_rb_norm(o):   return m_rb(o) / m_luma(o)
def m_magenta(o):   return float((o[:, 0].mean() + o[:, 2].mean()) / 2 - o[:, 1].mean())
def m_dark(o):      return float((o[DARK] @ LUMA).mean())
def m_mid(o):       return float((o[MID] @ LUMA).mean())
def m_bright(o):    return float((o[BRIGHT] @ LUMA).mean())
def m_dark_teal(o): return float((o[DARK, 2] - o[DARK, 0]).mean())
def m_bright_teal(o): return float((o[BRIGHT, 2] - o[BRIGHT, 0]).mean())


# op -> what it must move, and what it must leave alone.
#   moment: must go UP for `dir: up` (all cases below use up)
#   hold:   (moment, tolerance) that must NOT move
#   less:   (moment, fraction) that may move, but by less than this fraction of
#           the primary move
CASES: dict[str, dict] = {
    # A stop of exposure is multiplicative and lands before the CDL, so it
    # cannot change the colour balance -- only the amount of light.
    "exposure": dict(moment=m_luma, hold=(m_rb_norm, 0.02)),
    # The S-curve pivots on the MEASURED median, so contrast is not allowed to
    # double as a brightness change.
    "contrast": dict(moment=m_std, hold=(m_luma, 0.04)),
    # `power` peaks around x = 0.35 and fades toward both endpoints; measured
    # ratio bright/mid = 0.71 at "moderate".
    "midtones": dict(moment=m_mid, less=(m_bright, 0.85)),
    # p99 at 0.97: no headroom, so the compiler must reach for shadow_lift,
    # whose mask is exactly 0 above luma 0.5.
    "shadows": dict(moment=m_dark, hold=(m_bright, 1e-12), stats=stats_of(IMG, p99=RGB.of(0.97))),
    "highlights": dict(moment=m_bright, hold=(m_dark, 1e-12)),
    "warmth": dict(moment=m_rb, hold=(m_luma, 0.005)),
    "tint": dict(moment=m_magenta, hold=(m_luma, 0.01)),
    # The saturation lerp is exactly luma-preserving until something clips.
    "saturation": dict(moment=m_chroma, hold=(m_luma, 1e-12)),
    "shadow_tint": dict(moment=m_dark_teal, hold=(m_bright_teal, 1e-12), kw=dict(target="teal")),
    "highlight_tint": dict(moment=m_bright_teal, hold=(m_dark_teal, 1e-12), kw=dict(target="teal")),
}

# EffectSpec is spatial: apply() ignores it by design and render.py turns it
# into ffmpeg filters, so there are no pixels to measure here. This is the only
# place in the file where a spec FIELD is the evidence, and it is a limitation
# of the test, not a choice.
EFFECT_CASES = {"grain": "grain", "glow": "glow", "vignette": "vignette",
                "softness": "softness", "denoise": "denoise", "fringe": "fringe"}

assert sorted(set(CASES) | set(EFFECT_CASES)) == sorted(OPS), "a verb with no measured test"


# ---- identity -------------------------------------------------------------


def test_an_empty_intent_is_the_identity_grade_bit_for_bit():
    """Not "close to identity". spec.py's whole guarded-step design exists
    because `x + (x-x)*1.0` costs an ulp and silently moves the LUT hash."""
    spec = compile_intent(Intent(), STATS)
    assert spec.is_identity()
    assert spec == GradeSpec.identity().model_copy(update={"rationale": spec.rationale})
    grid = _grid(33)
    assert np.array_equal(spec.apply(grid), grid)


def test_identity_survives_every_kind_of_clip():
    """An empty intent must not pick up a grade from the measurement."""
    for st in (clip_stats(), clip_stats(clipped_high=0.4, crushed_low=0.4, p99=RGB.of(1.0)),
               clip_stats(p1=RGB.of(0.0), p50=RGB.of(0.0), p99=RGB.of(0.0)),  # unmeasured
               STATS):
        assert compile_intent(Intent(), st).is_identity(), st


def test_the_rationale_is_the_sentences_and_nothing_else():
    spec = compile_intent(one("warmth", "subtle"), STATS)
    assert spec.rationale == "Warmed it up a little."


# ---- direction ------------------------------------------------------------


@pytest.mark.parametrize("op", sorted(CASES))
def test_each_verb_moves_its_own_moment_and_leaves_the_others_alone(op):
    case = CASES[op]
    st = case.get("stats", STATS)
    before = IMG
    after = out(one(op, **case.get("kw", {})), st)

    moved = case["moment"](after) - case["moment"](before)
    assert moved > 1e-4, f"{op}: measured no effect ({moved:.2e})"

    if "hold" in case:
        fn, tol = case["hold"]
        if tol < 1e-6:
            # An exact claim is only exact while nothing clips, so prove that too
            # rather than letting a rail quietly turn the tolerance into luck.
            assert after.min() > 1e-9 and after.max() < 1.0 - 1e-9, f"{op} clipped"
        assert abs(fn(after) - fn(before)) < tol, f"{op}: moved something it should not"
    if "less" in case:
        fn, frac = case["less"]
        assert abs(fn(after) - fn(before)) < frac * abs(moved), f"{op}: not selective enough"


@pytest.mark.parametrize("op", sorted(CASES))
def test_down_is_the_opposite_of_up(op):
    case = CASES[op]
    st = case.get("stats", STATS)
    kw = case.get("kw", {})
    base = case["moment"](IMG)
    assert case["moment"](out(one(op, dir="down", **kw), st)) < base < \
           case["moment"](out(one(op, dir="up", **kw), st))


@pytest.mark.parametrize("op,field", sorted(EFFECT_CASES.items()))
def test_texture_verbs_reach_their_effect(op, field):
    spec = compile_intent(one(op), STATS)
    assert getattr(spec.effects, field) > 0.0
    assert not spec.is_identity()
    # apply() must still ignore them completely: an effect is not a colour move.
    assert np.array_equal(spec.apply(IMG), IDENTITY_OUT)


# ---- monotonic magnitudes -------------------------------------------------


@pytest.mark.parametrize("op", sorted(CASES))
def test_subtle_is_less_than_moderate_is_less_than_strong(op):
    case = CASES[op]
    st = case.get("stats", STATS)
    base = case["moment"](IMG)
    got = [case["moment"](out(one(op, a, **case.get("kw", {})), st)) - base for a in AMOUNTS]
    assert got[0] < got[1] < got[2], dict(zip(AMOUNTS, got))
    assert got[0] > 0.0


@pytest.mark.parametrize("op,field", sorted(EFFECT_CASES.items()))
def test_texture_magnitudes_are_monotonic_too(op, field):
    got = [getattr(compile_intent(one(op, a), STATS).effects, field) for a in AMOUNTS]
    assert got[0] < got[1] < got[2], dict(zip(AMOUNTS, got))


def test_repeating_a_verb_compounds_it():
    twice = Intent(ops=[Op(op="warmth"), Op(op="warmth")])
    assert m_rb(out(twice)) > m_rb(out(one("warmth"))) > m_rb(IMG)


# ---- the measurement has to matter ----------------------------------------


def test_the_same_intent_compiles_differently_on_different_footage():
    """THE test. Same words, two clips, two different numbers.

    A clip already sitting in the orange part of the wheel needs less push to
    read as warmer; a blue clip has to cross neutral first and needs more.
    `dominant_hue` is the only field that knows this and nothing else in the
    project consumes it.
    """
    warm = clip_stats(dominant_hue=30.0, saturation=0.30)
    cool = clip_stats(dominant_hue=215.0, saturation=0.30)
    a = compile_intent(one("warmth"), warm)
    b = compile_intent(one("warmth"), cool)

    assert a.temperature < b.temperature
    assert b.temperature / a.temperature > 1.5, (a.temperature, b.temperature)
    # ... and it is visible in the pixels, not only in the field.
    assert m_rb(b.apply(IMG)) - m_rb(a.apply(IMG)) > 0.005

    # Cooling reverses which clip gets the bigger push, for the same reason.
    ca = compile_intent(one("warmth", dir="down"), warm)
    cb = compile_intent(one("warmth", dir="down"), cool)
    assert ca.temperature < cb.temperature


def test_a_grey_clip_gets_no_hue_correction_because_its_hue_is_meaningless():
    """probe.py measures hue chroma-weighted; below ~0.05 mean chroma the angle
    carries no information, so the compiler must fall back to the plain step."""
    grey = clip_stats(dominant_hue=30.0, saturation=0.0)
    blue_grey = clip_stats(dominant_hue=215.0, saturation=0.0)
    assert compile_intent(one("warmth"), grey).temperature == \
           compile_intent(one("warmth"), blue_grey).temperature


def test_lifting_the_shadows_picks_a_different_field_when_there_is_no_headroom():
    roomy = clip_stats(p99=RGB.of(0.7))
    blown = clip_stats(p99=RGB.of(0.99), clipped_high=0.08)
    a = compile_intent(one("shadows"), roomy)
    b = compile_intent(one("shadows"), blown)

    assert a.offset.r > 0 and a.shadow_lift == 0.0      # room at the top: whole-curve lift
    assert b.shadow_lift > 0 and b.offset.r == 0.0      # no room: masked lift only
    # And the difference is real: the roomy grade moves the highlights, the
    # blown one must not.
    assert m_bright(a.apply(IMG)) - m_bright(IMG) > 0.01
    assert abs(m_bright(b.apply(IMG)) - m_bright(IMG)) < 1e-12


def test_brightening_is_damped_by_measured_headroom():
    roomy = clip_stats(p99=RGB.of(0.6))
    blown = clip_stats(p99=RGB.of(1.0), clipped_high=0.2)
    assert compile_intent(one("exposure"), roomy).exposure > \
           3 * compile_intent(one("exposure"), blown).exposure


def test_darkening_is_damped_when_the_blacks_are_already_gone():
    clean = clip_stats(crushed_low=0.0)
    crushed = clip_stats(crushed_low=0.3)
    assert compile_intent(one("exposure", dir="down"), clean).exposure < \
           compile_intent(one("exposure", dir="down"), crushed).exposure


def test_contrast_is_scaled_by_the_measured_variance_and_pivots_on_the_median():
    flat = clip_stats(frame_variance=0.005, p50=RGB.of(0.30))
    punchy = clip_stats(frame_variance=0.15, p50=RGB.of(0.55))
    a = compile_intent(one("contrast"), flat)
    b = compile_intent(one("contrast"), punchy)
    assert a.contrast > 2 * b.contrast          # a flat clip can take more
    assert a.pivot < b.pivot                    # each rotates around its own midtone
    assert abs(a.pivot - 0.30) < 0.02 and abs(b.pivot - 0.55) < 0.02


def test_saturation_is_damped_on_footage_that_is_already_saturated():
    tame = clip_stats(saturation=0.20)
    loud = clip_stats(saturation=0.50)
    assert compile_intent(one("saturation"), tame).saturation > \
           compile_intent(one("saturation"), loud).saturation


def test_pulling_blown_highlights_down_spends_the_move_on_the_shoulder():
    """A uniform negative lift moves welded-white pixels down together and they
    stay welded. The shoulder is the only thing that puts gradient back."""
    clean = clip_stats(clipped_high=0.0)
    blown = clip_stats(clipped_high=0.2, p99=RGB.of(1.0))
    assert compile_intent(one("highlights", dir="down"), clean).highlight_rolloff == 0.0
    b = compile_intent(one("highlights", dir="down"), blown)
    assert b.highlight_rolloff > 0.1 and b.highlight_lift < 0.0


def test_a_grade_that_would_blow_the_whites_grows_a_shoulder_on_its_own():
    """match.py's argument, re-applied: solve for the peak this grade actually
    produces instead of picking a constant and hoping. `highlight_rolloff` has
    no verb precisely because it is derivable."""
    bright = clip_stats(p99=RGB.of(0.95))
    dim = clip_stats(p99=RGB.of(0.45))
    assert compile_intent(one("exposure", "strong"), bright).highlight_rolloff > 0.0
    assert compile_intent(one("exposure", "strong"), dim).highlight_rolloff == 0.0


# ---- targets --------------------------------------------------------------


def hue_patch(rgb, n=2000):
    return np.tile(np.array(rgb, dtype=np.float64), (n, 1))


def test_a_colour_target_only_touches_that_hue():
    """"drain the greens" must leave the reds where they were."""
    green, red = hue_patch([0.15, 0.6, 0.15]), hue_patch([0.6, 0.15, 0.15])
    img = np.vstack([green, red])
    o = out(one("saturation", dir="down", target="green"), img=img)
    n = len(green)
    assert m_chroma(o[:n]) < m_chroma(green) - 0.02      # the greens drained
    assert abs(m_chroma(o[n:]) - m_chroma(red)) < 0.005  # the reds did not


def test_a_colour_target_on_exposure_darkens_only_that_hue():
    blue, red = hue_patch([0.15, 0.2, 0.6]), hue_patch([0.6, 0.15, 0.15])
    img = np.vstack([blue, red])
    o = out(one("exposure", dir="down", target="blue"), img=img)
    n = len(blue)
    assert m_luma(o[:n]) < m_luma(blue) - 0.005
    assert abs(m_luma(o[n:]) - m_luma(red)) < 0.002


@pytest.mark.parametrize("target", ["red", "orange", "yellow", "green", "cyan",
                                    "teal", "blue", "purple", "magenta", "skin"])
def test_every_colour_target_reaches_the_grade(target):
    """A target the compiler silently dropped would hand the user a sentence
    describing a move that never happened."""
    assert not compile_intent(one("saturation", target=target), STATS).is_identity()
    assert not compile_intent(one("shadow_tint", target=target), STATS).is_identity()


def test_a_tint_verb_with_no_colour_uses_the_convention_it_says_it_uses():
    """describe() promises teal shadows; the numbers have to agree."""
    assert compile_intent(one("shadow_tint"), STATS) == \
           compile_intent(one("shadow_tint", target="teal"), STATS)


# ---- strength -------------------------------------------------------------


def test_strength_mixes_the_whole_look_back_toward_the_source():
    full = out(Intent(ops=[Op(op="warmth", amount="strong")], strength="full"))
    half = out(Intent(ops=[Op(op="warmth", amount="strong")], strength="moderate"))
    assert m_rb(IMG) < m_rb(half) < m_rb(full)


def test_strength_alone_does_not_grade_anything():
    """look_mix is set, but there is nothing to mix, so the pixels must not move.

    Within an ulp, not bit-for-bit: at look_mix < 1 apply() lerps the source
    against a value that has already been through the saturation lerp, and the
    two differ by ~1e-16. That is spec.py's documented cost, not a compiler
    move -- measured max deviation below is 6.9e-18.
    """
    assert np.abs(out(Intent(strength="subtle")) - IMG).max() < 1e-15


# ---- sanity ---------------------------------------------------------------


def test_every_verb_at_full_strength_still_produces_a_legal_grade():
    """Nothing the vocabulary can say may produce a NaN or an out-of-range LUT."""
    grid = _grid(17)
    for direction in ("up", "down"):
        intent = Intent(ops=[Op(op=o, dir=direction, amount="strong") for o in OPS])
        table = compile_intent(intent, STATS).apply(grid)
        assert np.all(np.isfinite(table))
        assert table.min() >= 0.0 and table.max() <= 1.0
