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


# ---- auto-balance (roadmap A6) --------------------------------------------
#
# Everything below re-probes the BALANCED PIXELS with probe.py and compares the
# measurement to the measurement of the source. A spec field is not evidence
# here for the same reason it is not evidence above, and doubly so for a
# correction: "slope 1.16 on red" says nothing about whether the cast is gone.
#
# The clips are built rather than shot because the feature's failure mode is
# specific -- neutralising footage that is deliberately monochromatic -- and the
# repo has no sodium-lit street in it.

_g = np.random.default_rng(11)


def _noise(shape, s=0.04):
    return _g.normal(0.0, s, shape)


# Blacks already on the rail and no cast: balancing this must be a near-no-op.
NEUTRAL = np.clip(_g.uniform(-0.06, 0.82, (20000, 1)) + _noise((20000, 3)), 0.0, 1.0)

# A sodium-lit street and a blue night exterior. Both are SUPPOSED to be that
# colour; a balance that neutralises them has graded the look back out of
# footage that was lit or graded on purpose, which is the one way this feature
# does harm rather than nothing.
SODIUM = np.clip(_g.beta(1.4, 6.0, (20000, 1)) * [1.0, 0.62, 0.18] + _noise((20000, 3), 0.015), 0.0, 1.0)
NIGHT = np.clip(_g.beta(2.0, 5.0, (20000, 1)) * [0.45, 0.62, 1.0] + _noise((20000, 3), 0.02), 0.0, 1.0)

# Six saturated hues in equal measure: the highest `saturation` in the file and
# no dominant hue at all. It exists to prove which field the gate reads.
_HUES = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=np.float64)
BARS = np.clip(_HUES[_g.integers(0, 6, 20000)] * _g.uniform(0.25, 0.75, (20000, 1)) + _noise((20000, 3), 0.01), 0.0, 1.0)


def cast(img, gain):
    """`img` through a per-channel gain — a lighting cast, before any grade."""
    return np.clip(img * np.array(gain, dtype=np.float64), 0.0, 1.0)


def balanced(img, intent: Intent | None = None) -> np.ndarray:
    return compile_intent(intent or Intent(), stats_of(img), balance=True).apply(img)


def m_spread(o):
    """How far apart the three channel means sit — the size of the cast."""
    mu = o.mean(axis=0)
    return float(mu.max() - mu.min())


def m_hue_strength(o):
    return stats_of(o).hue_strength


def m_crushed(o):
    return stats_of(o).crushed_low


def m_p1(o):
    return float(min(stats_of(o).p1.as_array()))


def m_p99(o):
    return float(max(stats_of(o).p99.as_array()))


# ---- it is optional and it is visible -------------------------------------


def test_balance_is_off_unless_the_caller_asks_for_it():
    """The identity contract above is the reason for the default: a balance is
    a departure the user did not request, so it cannot be this function's
    decision. Two live callers pass no flag today and must be unaffected."""
    green = cast(IMG, (1.0, 1.06, 1.0))
    assert compile_intent(Intent(), stats_of(green)).is_identity()
    assert not compile_intent(Intent(), stats_of(green), balance=True).is_identity()


def test_a_balance_says_so_in_the_rationale_before_the_look():
    """A silent correction fighting the user's own grade is worse than none.
    describe() writes the intent's sentences; the balance prepends its own, so
    the order in the sentence is the order in the pipeline."""
    fogged_and_green = cast(np.clip(IMG * 0.85 + 0.08, 0.0, 1.0), (1.0, 1.06, 1.0))
    spec = compile_intent(one("warmth"), stats_of(fogged_and_green), balance=True)
    assert spec.rationale == "Neutralised a green cast, set the black point, warmed it up."


def test_a_balance_that_did_nothing_measurable_claims_nothing():
    """SODIUM is left alone entirely, so the sentence list is empty and the
    rationale is exactly the intent's."""
    assert compile_intent(Intent(), stats_of(SODIUM), balance=True).rationale == ""
    assert compile_intent(one("warmth"), stats_of(SODIUM), balance=True).rationale == "Warmed it up."


@pytest.mark.parametrize("gain,name", [
    ((1.0, 1.06, 1.0), "green"),
    ((1.05, 1.0, 1.05), "magenta"),
    ((1.06, 1.0, 0.94), "orange"),
    ((1.0, 1.0, 1.12), "blue"),
])
def test_the_balance_names_the_cast_it_found(gain, name):
    spec = compile_intent(Intent(), stats_of(cast(IMG, gain)), balance=True)
    assert spec.rationale.startswith(f"Neutralised a{'n' if name[0] in 'aeiou' else ''} {name} cast")


# ---- it converges ---------------------------------------------------------


@pytest.mark.parametrize("gain,label,before,after", [
    # gain, name, measured hue_strength before -> after balancing the pixels
    ((1.0, 1.06, 1.0), "green fluorescent", 0.0266, 0.0000),
    ((1.05, 1.0, 1.05), "magenta", 0.0221, 0.0000),
    ((1.06, 1.0, 0.94), "tungsten", 0.0460, 0.0038),
])
def test_balance_measurably_neutralises_a_cast(gain, label, before, after):
    """The whole feature, measured on pixels. A balance that does not shrink
    the cast is the feature failing silently, which no spec-field assertion
    would catch."""
    img = cast(IMG, gain)
    out_ = balanced(img)

    assert abs(m_hue_strength(img) - before) < 0.002, "the fixture drifted"
    assert m_hue_strength(out_) < 0.15 * m_hue_strength(img), (
        f"{label}: hue_strength {m_hue_strength(img):.4f} -> {m_hue_strength(out_):.4f}")
    # ... and in the first moment too, which is what the solver actually fits.
    assert m_spread(out_) < 0.15 * m_spread(img), (
        f"{label}: channel-mean spread {m_spread(img):.4f} -> {m_spread(out_):.4f}")


def test_the_cast_correction_alone_does_not_change_the_exposure():
    """match.py's solve is aimed at the clip's own LUMA mean, so the correction
    is luma-preserving at the mean by construction -- it changes the colour and
    not the light. Measured on NEUTRAL (blacks already at 0, so the black-point
    half is a no-op and this isolates the cast half): mean luma moves 9.1e-5
    while the channel-mean spread falls from 0.0235 to 0.0000.

    The black-point half is NOT luma-preserving and is not meant to be: pulling
    a fogged black down while pinning the white point is a contrast move, and it
    darkens the clip by design (measured mean luma 0.4212 -> 0.3492 on a fogged
    IMG). That is why the two halves are tested apart.
    """
    img = cast(NEUTRAL, (1.0, 1.06, 1.0))
    out_ = balanced(img)
    assert m_spread(out_) < 0.15 * m_spread(img)
    assert abs(m_luma(out_) - m_luma(img)) < 2e-3, m_luma(out_) - m_luma(img)


# ---- it is idempotent-ish -------------------------------------------------


def test_balancing_an_already_neutral_clip_is_nearly_a_no_op():
    """Measured max deviation 8.0e-4, i.e. 0.20 of an 8-bit code value, over a
    clip whose blacks are already at 0 and whose channel means agree to 6.5e-4.
    Not exactly zero: the solve fits the measured moments, and probe.py measures
    them off a uint8 round trip."""
    out_ = balanced(NEUTRAL)
    assert np.abs(out_ - NEUTRAL).max() < 2e-3, np.abs(out_ - NEUTRAL).max()
    assert np.abs(out_ - NEUTRAL).mean() < 5e-4
    # Re-balancing the balanced clip converges rather than drifting.
    assert np.abs(balanced(out_) - out_).max() < np.abs(out_ - NEUTRAL).max() + 1e-9


# ---- it does not eat intentional colour -----------------------------------


@pytest.mark.parametrize("img,label", [(SODIUM, "sodium-lit street"), (NIGHT, "blue night exterior")])
def test_balance_leaves_deliberately_monochromatic_footage_completely_alone(img, label):
    """THE test that stops this feature being harmful. Both clips measure
    hue_strength ~0.153, above the 0.12 where the gate has already closed, so
    the grade is the identity bit-for-bit — not "close to" it."""
    st = stats_of(img)
    assert st.hue_strength > 0.12, (label, st.hue_strength)
    spec = compile_intent(Intent(), st, balance=True)
    assert spec.is_identity(), (label, spec.slope, spec.offset)
    assert np.array_equal(spec.apply(img), GradeSpec.identity().apply(img))


def test_the_gate_is_hue_strength_and_not_saturation():
    """`saturation` is HSV chroma/max and cannot fall when a frame holds two
    opposite hues, which is why probe.py added hue_strength and why the gate
    reads it. BARS is the most saturated clip in this file and has no dominant
    hue; SODIUM is less saturated and is a look. Saturation gets this backwards."""
    bars, sodium = stats_of(cast(BARS, (1.0, 1.08, 1.0))), stats_of(SODIUM)
    assert bars.saturation > sodium.saturation           # ... and yet:
    assert bars.hue_strength < 0.04 < 0.12 < sodium.hue_strength
    assert "cast" in compile_intent(Intent(), bars, balance=True).rationale
    assert "cast" not in compile_intent(Intent(), sodium, balance=True).rationale


def test_a_black_point_may_be_set_on_a_look_without_touching_its_colour():
    """The two halves are independently gated. Fogged sodium footage gets its
    blacks back and keeps every degree of its orange."""
    fogged = np.clip(SODIUM * 0.8 + 0.09, 0.0, 1.0)
    out_ = balanced(fogged)
    assert m_p1(fogged) > 0.08 and m_p1(out_) < 0.005
    a, b = stats_of(fogged), stats_of(out_)
    assert abs(a.dominant_hue - b.dominant_hue) < 0.5, (a.dominant_hue, b.dominant_hue)
    assert m_rb(out_) > m_rb(fogged)      # de-fogging raises contrast, never neutralises


# ---- black and white point ------------------------------------------------


def test_balance_pulls_a_fogged_black_point_down_and_pins_the_white_point():
    """p1 0.145 -> 0.000 with p99 unmoved to 1e-12. The white point is an
    ANCHOR, not a second target: lifted blacks are fog and always a defect,
    while a clip whose p99 sits at 0.69 may simply be a night scene, and
    stretching that to full range is what the `exposure` verb is for."""
    fog = np.clip(IMG * 0.85 + 0.08, 0.0, 1.0)
    out_ = balanced(fog)
    assert m_p1(fog) > 0.14 and m_p1(out_) < 0.002
    assert abs(m_p99(out_) - m_p99(fog)) < 1e-12, (m_p99(fog), m_p99(out_))


def test_a_black_point_already_at_zero_is_left_where_it_is():
    """Lifting a black point that is already at 0 does nothing, so the pass
    must not spend a slope on it."""
    spec = compile_intent(Intent(), stats_of(NEUTRAL), balance=True)
    assert abs(spec.offset.r) < 1e-3 and abs(spec.slope.r - 1.0) < 1e-3


def test_unmeasured_percentiles_do_not_invent_a_black_point():
    """Sessions written before probe.py grew percentiles load with p1/p50/p99
    all zero; _measure documents the same fallback for headroom."""
    spec = compile_intent(Intent(), clip_stats(p1=RGB.of(0.0), p50=RGB.of(0.0),
                                               p99=RGB.of(0.0), hue_strength=0.0), balance=True)
    assert abs(spec.slope.r - 1.0) < 1e-9 and abs(spec.offset.r) < 1e-9


# ---- rails ----------------------------------------------------------------


def test_balance_does_not_push_crushed_blacks_further_into_the_rail():
    """`_shadows` damps a shadow move by `crushed_low` because welded pixels
    move together and stay welded; the balance uses the same measurement,
    floored at 0 rather than 0.3 -- a verb the user asked for has to do
    something, an unrequested correction may correctly do nothing.

    Measured: 37.25% crushed before, 37.23% after -- it went DOWN -- with the
    green cast still removed (hue_strength 0.0185 -> 0.0001)."""
    crushed = cast(np.clip(_g.uniform(-0.35, 0.7, (20000, 1)) + _noise((20000, 3)), 0.0, 1.0),
                   (1.0, 1.07, 1.0))
    assert m_crushed(crushed) > 0.3
    out_ = balanced(crushed)
    assert m_crushed(out_) <= m_crushed(crushed) + 1e-9, (m_crushed(crushed), m_crushed(out_))
    assert m_hue_strength(out_) < 0.2 * m_hue_strength(crushed)  # and it still worked


def test_a_balanced_grade_still_grows_its_own_highlight_shoulder():
    """_protect_highlights reads slope/offset, so writing the balance into the
    CDL means the shoulder covers the composed grade for free."""
    dim = cast(np.clip(IMG * 0.55, 0.0, 1.0), (1.0, 1.08, 1.0))
    spec = compile_intent(one("exposure", "strong"), stats_of(dim), balance=True)
    o = spec.apply(dim)
    assert np.all(np.isfinite(o)) and o.max() <= 1.0


# ---- it composes ----------------------------------------------------------


@pytest.mark.parametrize("gain,label", [((1.0, 1.06, 1.0), "green"), ((1.05, 1.0, 1.05), "magenta")])
def test_the_creative_move_survives_the_balance(gain, label):
    """Balance writes slope/offset (step 2); "warmer" writes temperature (step
    3). Different fields on purpose, so the look lands on top of the correction
    instead of being cancelled by it. Measured: r-b moves +0.0243 with the
    balance on and +0.0257 with it off, i.e. 95% of the creative move survives
    on the green clip and 88% on the magenta one."""
    img = cast(IMG, gain)
    base, both = balanced(img), balanced(img, one("warmth"))
    alone = out(one("warmth"), stats_of(img), img=img)

    moved = m_rb(both) - m_rb(base)
    assert moved > 0.015, (label, moved)
    assert moved > 0.8 * (m_rb(alone) - m_rb(img)), (label, moved, m_rb(alone) - m_rb(img))


def test_balance_is_the_point_the_same_look_lands_in_the_same_place():
    """Why A6 exists, as a number. Two shots of one scene, one a little green
    and one a little magenta, given the same sentence: the green/magenta
    separation between the two results is 0.0602 without the balance and
    0.00004 with it -- 1500x smaller, which is the whole point of A6."""
    green, magenta = cast(IMG, (0.97, 1.04, 0.97)), cast(IMG, (1.04, 0.96, 1.04))
    warm = one("warmth")

    raw = abs(m_magenta(out(warm, stats_of(green), img=green)) -
              m_magenta(out(warm, stats_of(magenta), img=magenta)))
    fixed = abs(m_magenta(balanced(green, warm)) - m_magenta(balanced(magenta, warm)))
    assert raw > 0.05, raw
    assert fixed < 0.1 * raw, (raw, fixed)


def test_an_unmeasured_hue_strength_is_not_read_as_no_cast():
    """`hue_strength` defaults to 0.0 on sessions written before probe.py
    measured it, and 0.0 otherwise means "correct in full". Real footage never
    reads exactly 0 -- the flattest clip in this file is 6e-4 -- so an exact
    zero next to real chroma is missing data, and a correction nobody asked for
    does not get made on missing data."""
    st = stats_of(cast(IMG, (1.06, 1.0, 0.94)))
    assert compile_intent(Intent(), st, balance=True).rationale.startswith("Neutralised")
    legacy = st.model_copy(update={"hue_strength": 0.0})
    assert "cast" not in compile_intent(Intent(), legacy, balance=True).rationale


# ---- regions (roadmap B1) -------------------------------------------------
#
# A region op is compiled into a LAYER, never into the base. So the tests here
# measure two things the flat compiler could not have: that the base is exactly
# what it would have been without the regional op, and that the layer moves the
# pixels the sentence pointed at.


def test_a_regional_op_leaves_the_base_grade_bit_for_bit_alone():
    """The base is the grade for every pixel; a request about the top of the
    frame must not leak into it. array_equal, not allclose -- a leak of one ulp
    is a leak."""
    from ragvid.compiler import compile_stack

    plain = compile_stack(Intent(ops=[Op(op="warmth")]), STATS)
    with_region = compile_stack(
        Intent(ops=[Op(op="warmth"), Op(op="exposure", dir="down", target="top")]), STATS)
    assert with_region.base.model_dump(exclude={"rationale"}) == \
        plain.base.model_dump(exclude={"rationale"})
    assert np.array_equal(with_region.base.apply(IMG), plain.base.apply(IMG))
    assert len(with_region.layers) == 1 and plain.is_flat


def test_an_intent_with_no_region_still_compiles_flat():
    """The overwhelmingly common case, and the identity property the whole
    design rests on: no region asked for, no container overhead added."""
    from ragvid.compiler import compile_stack

    assert compile_stack(Intent(), STATS).is_flat
    assert compile_stack(Intent(ops=[Op(op="saturation", target="green")]), STATS).is_flat
    assert compile_stack(Intent(), STATS).base.is_identity()


def test_the_layer_carries_the_move_and_the_region_carries_the_place():
    from ragvid.compiler import compile_stack

    stack = compile_stack(one("exposure", dir="down", target="top"), STATS)
    layer, = stack.layers
    assert layer.spec.exposure < -0.1, "the darkening has to be IN the layer"
    assert stack.base.is_identity(), "and nowhere else"
    assert layer.region.shape == "linear" and layer.region.edge == "top"


def test_a_regional_op_darkens_only_its_own_half_of_a_real_frame():
    """Measured on pixels, through the stack -- the only evidence that counts."""
    from ragvid.compiler import compile_stack

    img = np.tile(IMG[:3600].reshape(60, 60, 3), (1, 1, 1))
    stack = compile_stack(one("exposure", dir="down", target="top"), STATS)
    got = stack.apply(img)
    top = (got[:20] @ LUMA).mean() - (img[:20] @ LUMA).mean()
    bottom = (got[-20:] @ LUMA).mean() - (img[-20:] @ LUMA).mean()
    assert top < -0.02, f"the top only moved {top:+.4f}"
    assert bottom == 0.0, f"the bottom moved {bottom:+.4f}"


def test_two_regions_become_two_layers_in_the_order_asked_for():
    from ragvid.compiler import compile_stack

    stack = compile_stack(Intent(ops=[
        Op(op="exposure", dir="down", target="bottom"),
        Op(op="warmth", target="left"),
    ]), STATS)
    assert [l.region.edge for l in stack.layers] == ["bottom", "left"]


def test_two_ops_on_the_same_region_share_one_layer():
    """One region, one mask, one .cube -- and the two moves compose inside the
    layer exactly as two global moves compose inside the base."""
    from ragvid.compiler import compile_stack

    stack = compile_stack(Intent(ops=[
        Op(op="exposure", dir="down", target="top"),
        Op(op="saturation", dir="down", target="top"),
    ]), STATS)
    layer, = stack.layers
    assert layer.spec.exposure < 0 and layer.spec.saturation < 1.0


def test_a_region_on_a_texture_verb_is_dropped_rather_than_promised():
    """grain/glow/vignette are already ffmpeg filters, so a region on one cannot
    be honoured. It is cleared at the Intent boundary, which is what keeps
    describe() from reporting a move that never happened."""
    from ragvid.compiler import compile_stack

    stack = compile_stack(one("grain", target="top"), STATS)
    assert stack.is_flat and stack.base.effects.grain > 0
    assert "top" not in stack.base.rationale


def test_a_regional_verb_never_looks_its_region_up_as_a_hue():
    """`_exposure` routes a colour target onto the hue bands. A region target is
    a different question and must not reach that table -- it used to raise."""
    from ragvid.compiler import compile_stack

    for target in ("top", "bottom", "left", "right", "center", "edges"):
        stack = compile_stack(one("exposure", dir="down", target=target), STATS)
        assert not stack.layers[0].spec.has_hue_qualifiers()


def test_the_rationale_names_the_regional_move_too():
    """The sentence list is the UI. A move that landed in a layer going
    unmentioned is the same silent drop, one level up."""
    stack_rationale = compile_intent(
        Intent(ops=[Op(op="warmth"), Op(op="exposure", dir="down", target="top")]),
        STATS).rationale
    assert stack_rationale == "Warmed it up, darkened the top."


def test_strength_reaches_the_layers_as_well_as_the_base():
    """"half strength" modifies everything that was asked for, including the
    half of it that was about a corner of the frame."""
    from ragvid.compiler import compile_stack

    stack = compile_stack(Intent(ops=[Op(op="exposure", dir="down", target="top")],
                                strength="subtle"), STATS)
    assert stack.base.look_mix == stack.layers[0].spec.look_mix == 0.4


def test_compile_intent_is_exactly_the_base_of_compile_stack():
    """vibe.py and every provider path still return one GradeSpec, so the two
    entry points must not be able to disagree."""
    from ragvid.compiler import compile_stack

    intent = Intent(ops=[Op(op="warmth"), Op(op="contrast", target="center")])
    for balance in (False, True):
        assert compile_intent(intent, STATS, balance=balance) == \
            compile_stack(intent, STATS, balance=balance).base


def test_the_balance_runs_once_for_the_clip_and_not_once_per_region():
    """Auto-balance is a property of the footage, not of a corner of it. A
    second copy inside a layer would apply the cast correction twice wherever
    the mask is on."""
    from ragvid.compiler import compile_stack

    cast = clip_stats(mean=RGB(r=0.45, g=0.40, b=0.34), dominant_hue=30.0,
                      hue_strength=0.03, saturation=0.2)
    stack = compile_stack(one("exposure", dir="down", target="top"), cast, balance=True)
    assert not stack.base.is_identity(), "the balance has to be in the base"
    assert stack.layers[0].spec.slope == RGB.of(1.0)
    assert stack.layers[0].spec.offset == RGB.of(0.0)


# ---- semantic regions (roadmap B2) ----------------------------------------
#
# B2 added NO code to compiler.py, deliberately, and these are the tests that
# say so. "the sky" has to group, order and compile through exactly the path
# "the top" does; if any of this needed a branch in the compiler, the feature
# was the wrong shape and it would be a second implementation of layering to
# keep in step with the first.


def test_a_semantic_op_compiles_into_a_layer_like_a_geometric_one():
    from ragvid.compiler import compile_stack

    stack = compile_stack(one("exposure", dir="down", target="sky"), STATS)
    layer, = stack.layers
    assert stack.base.is_identity(), "the darkening belongs in the layer and nowhere else"
    assert layer.spec.exposure < -0.1
    assert layer.region.shape == "semantic" and layer.region.subject == "sky"
    assert layer.region.needs_frame


def test_the_two_kinds_of_region_compile_to_the_same_numbers():
    """The only difference between "darken the top" and "darken the sky" is
    which pixels the mask covers. If the compiled GradeSpecs ever differ, the
    compiler learned about mask sources and it should not have."""
    from ragvid.compiler import compile_stack

    top = compile_stack(one("exposure", dir="down", target="top"), STATS)
    sky = compile_stack(one("exposure", dir="down", target="sky"), STATS)
    assert top.layers[0].spec.model_dump(exclude={"rationale"}) == \
        sky.layers[0].spec.model_dump(exclude={"rationale"})   # only the sentence differs
    assert top.layers[0].region != sky.layers[0].region


def test_a_geometric_and_a_semantic_region_are_two_layers_in_the_order_asked_for():
    """Layer order is evaluation order, so it has to be the order the person
    said things in -- across both mask sources, from one dict.fromkeys pass."""
    from ragvid.compiler import compile_stack

    stack = compile_stack(Intent(ops=[
        Op(op="exposure", dir="down", target="sky"),
        Op(op="warmth", target="bottom"),
        Op(op="saturation", dir="down", target="sky"),
    ]), STATS)
    assert [l.region.subject or l.region.edge for l in stack.layers] == ["sky", "bottom"]
    assert stack.layers[0].spec.exposure < 0 and stack.layers[0].spec.saturation < 1.0


def test_a_semantic_verb_never_looks_its_subject_up_as_a_hue():
    """Inside a layer the target has already answered "which pixels", so the
    verb acts on all of them. Without that reset `_saturation` would look "sky"
    up in the hue-band table and raise."""
    from ragvid.compiler import compile_stack

    stack = compile_stack(one("saturation", dir="down", target="foliage"), STATS)
    layer, = stack.layers
    assert layer.spec.saturation < 0.95
    assert layer.spec.hue_green.model_dump() == GradeSpec.identity().hue_green.model_dump()


def test_the_base_is_bit_for_bit_untouched_by_a_semantic_op():
    from ragvid.compiler import compile_stack

    plain = compile_stack(Intent(ops=[Op(op="warmth")]), STATS)
    withsky = compile_stack(
        Intent(ops=[Op(op="warmth"), Op(op="exposure", dir="down", target="sky")]), STATS)
    assert np.array_equal(withsky.base.apply(IMG), plain.base.apply(IMG))


def test_the_rationale_names_the_semantic_move_too():
    """A move that landed in a layer going unmentioned is the same silent-drop
    bug one level up -- the list is what the UI shows."""
    from ragvid.compiler import compile_stack

    stack = compile_stack(Intent(ops=[
        Op(op="warmth"), Op(op="exposure", dir="down", target="sky")]), STATS)
    assert "sky" in stack.base.rationale
    assert "darkened the sky" in stack.layers[0].spec.rationale
