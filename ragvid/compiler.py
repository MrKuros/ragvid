"""Intent + measured statistics -> GradeStack. Pure, deterministic, no model.

`compile_stack` is the whole answer -- a base GradeSpec plus a layer per region
named (roadmap B1) -- and `compile_intent` is its base, which is the whole
answer too for every intent that names no region.

THE LOAD-BEARING CLAIM: this file has to be BETTER than the model guessing, not
merely cheaper. It gets there one way — by consulting the measurement. "Moderate
warm" on footage whose `dominant_hue` already sits at 30 degrees is a smaller
number than the same request on a clip sitting at 210, because crossing neutral
costs the whole way. A compiler that ignored `stats` would be a lookup table,
and a lookup table is not worth an architecture change.

match.py is the pattern being extended: it solves slope/offset in closed form
from ClipStats with no LLM in the loop. The difference is only that match.py's
target is another clip's histogram and this one's is a sentence.

EVALUATION ORDER DECIDES WHICH FIELD A VERB WRITES. spec.py pins the order and
this file has to respect it, so the choice is made per verb and the reason is in
a comment next to the mapping:

  * "brighter" -> `exposure`, not `slope`. Exposure is step 1, BEFORE the CDL,
    which is precisely what keeps `offset` an absolute lift for the shadow verb
    to use. Using slope for brightness would also collide with the colour tools,
    which is what slope is for (match.py).
  * "lift the shadows" -> `offset` OR `shadow_lift`, decided by measured
    headroom: offset moves the whole curve (step 2), shadow_lift is masked to
    luma < 0.5 by construction (step 6).
  * "pull the highlights down" on already-blown footage -> `highlight_rolloff`
    (step 7) before `highlight_lift`, because a uniform negative lift moves
    welded-white pixels down together and they stay welded.
  * auto-balance -> `slope`/`offset`, i.e. the CDL at step 2. A technical
    correction goes where the industry standard already puts one, and it is the
    only field pair an ASC CDL export can carry off the machine. `_balance` has
    the full argument, including why not `exposure` and not temperature/tint.

Every number below is one of: a step size (how much "moderate" means for that
axis), or a measured scaling of it. The step sizes are the taste in this file
and are the only thing that should ever need tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ragvid.intent import DEFAULT_TINT, OPS, REGIONS, STRENGTH_MIX, Intent, Op, describe
from ragvid.match import match_reference
from ragvid.region import GradeStack, Layer, for_target
from ragvid.spec import LUMA, RGB, GradeSpec

if TYPE_CHECKING:  # probe imports nothing from us; keeps the dependency one-way
    from ragvid.probe import ClipStats

# "moderate" is 1.0 by definition; the other two are its multiples. Monotonicity
# of subtle < moderate < strong is therefore structural, not a property that has
# to be maintained per verb.
UNIT = {"subtle": 0.4, "moderate": 1.0, "strong": 2.0}

# What "moderate" means on each axis, in that axis's own units. Chosen to land
# inside the ranges vibe.py's prompt already declares sane (and which
# tests/test_providers.py pins against spec.sanitize()), with "strong" = 2x
# still inside them.
STEP_EXPOSURE = 0.35      # stops
STEP_CONTRAST = 0.22      # S-curve strength
STEP_MIDTONES = 0.18      # gamma exponent delta
STEP_SHADOWS = 0.05       # absolute lift, display space
STEP_HIGHLIGHTS = 0.05
STEP_WARMTH = 800.0       # Kelvin-ish, TEMP_FULL is 3000
STEP_TINT = 0.12          # green/magenta axis
STEP_SATURATION = 0.30    # fractional, applied multiplicatively
STEP_SPLIT_TINT = 0.06    # tonal-split tint vector length
STEP_BAND_SAT = 0.35      # hue qualifier saturation, fractional
STEP_BAND_LUM = 0.05      # hue qualifier luma offset
STEP_EFFECT = 0.25        # every EffectSpec knob except fringe
STEP_FRINGE = 0.15        # narrower: fringe reads as a defect twice as fast

# Hue is ill-conditioned near the neutral axis, so `dominant_hue` is only
# trusted in proportion to how much colour there is to have a hue (probe.py
# measures it chroma-weighted for the same reason). Below 0.05 mean chroma the
# measurement carries no information and the compiler falls back to the
# unmodified step.
_HUE_CONF_FLOOR = 0.05
_HUE_CONF_SPAN = 0.15

# How far a measured bias may move a step, either way. 0.4 = a clip already all
# the way warm gets 60% of the push toward warm and 140% of the push away from
# it. Bounded on purpose: the measurement scales the request, it never overrides
# it, and a verb must never come out at zero or reversed.
_BIAS_GAIN = 0.4


@dataclass(frozen=True)
class _Measured:
    """The parts of ClipStats this file actually reads, pre-derived once.

    Every field here is measured. Nothing in this dataclass is a preference.
    """

    mean: np.ndarray
    p1: np.ndarray
    p99: np.ndarray
    luma_p50: float
    headroom: float      # 1 - brightest measured white; 0 = no room to push up
    floor: float         # darkest measured black; 0 = sitting on the rail
    clipped: float
    crushed: float
    var: float
    sat: float
    warm_bias: float     # -1..1, +1 = this clip is ALREADY orange
    magenta_bias: float  # -1..1, +1 = this clip is ALREADY magenta


def _measure(stats: "ClipStats") -> _Measured:
    p1, p50, p99 = stats.p1.as_array(), stats.p50.as_array(), stats.p99.as_array()
    # Sessions written before probe.py grew percentiles load with p1/p50/p99 all
    # zero and session.py does not migrate them. An all-zero p99 therefore means
    # "unmeasured", not "black clip": headroom falls back to 1.0 (push freely,
    # i.e. exactly what the compiler would do without the measurement) rather
    # than to 0.0, which would silently gag every brightening verb.
    measured_white = float(p99.max())
    headroom = 1.0 - measured_white if measured_white > 0.0 else 1.0

    conf = float(np.clip((stats.saturation - _HUE_CONF_FLOOR) / _HUE_CONF_SPAN, 0.0, 1.0))
    h = math.radians(stats.dominant_hue)
    return _Measured(
        mean=stats.mean.as_array(),
        p1=p1,
        p99=p99,
        luma_p50=float(p50 @ LUMA),
        headroom=headroom,
        floor=float(p1.min()),
        clipped=stats.clipped_high,
        crushed=stats.crushed_low,
        var=stats.frame_variance,
        sat=stats.saturation,
        # 30 deg is orange (the warm end of the wheel), 300 deg is magenta.
        warm_bias=conf * math.cos(h - math.radians(30.0)),
        magenta_bias=conf * math.cos(h - math.radians(300.0)),
    )


# ---- hue qualifier routing -------------------------------------------------

# A colour name -> the qualifier bands that carry it, with weights. spec.py's
# six bands are an exact partition of unity 60 degrees apart, so a hue that
# falls between two centres is expressed as a weighted pair rather than snapped
# to the nearer one — the same interpolation apply() does per pixel.
_BANDS: dict[str, tuple[tuple[str, float], ...]] = {
    "red": (("hue_red", 1.0),),
    "orange": (("hue_red", 0.5), ("hue_yellow", 0.5)),          # ~30 deg
    "skin": (("hue_red", 0.6), ("hue_yellow", 0.4)),            # ~20-25 deg
    "yellow": (("hue_yellow", 1.0),),
    "green": (("hue_green", 1.0),),
    "cyan": (("hue_cyan", 1.0),),
    "teal": (("hue_cyan", 0.7), ("hue_blue", 0.3)),             # ~190 deg
    "blue": (("hue_blue", 1.0),),
    "purple": (("hue_blue", 0.5), ("hue_magenta", 0.5)),        # ~270 deg
    "magenta": (("hue_magenta", 1.0),),
}

# Direction of each colour as an RGB push, for the tonal-split tints. Magnitude
# is irrelevant here (spec.py strips the luma out of a tint before adding it),
# so these are shapes, not amounts.
_TINT_VECTORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, -0.5, -0.5),
    "orange": (1.0, 0.2, -1.0),
    "skin": (1.0, 0.2, -0.8),
    "yellow": (0.7, 0.7, -1.0),
    "green": (-0.5, 1.0, -0.5),
    "cyan": (-1.0, 0.5, 0.7),
    "teal": (-1.0, 0.2, 1.0),
    "blue": (-0.5, -0.5, 1.0),
    "purple": (0.2, -1.0, 1.0),
    "magenta": (1.0, -1.0, 1.0),
}


def _band_sat(d: dict, target: str, k: float) -> None:
    """Saturation on one hue family. Multiplicative so repeats compose."""
    for field, w in _BANDS[target]:
        u = abs(k) * STEP_BAND_SAT * w
        d[field]["sat"] *= (1.0 + u) if k > 0 else 1.0 / (1.0 + u)


def _band_lum(d: dict, target: str, k: float) -> None:
    for field, w in _BANDS[target]:
        d[field]["lum"] += k * w


# ---- the verbs -------------------------------------------------------------
#
# Each takes the working spec dict, a SIGNED magnitude `k` (+/- UNIT[amount]),
# the operation, and the measurement. Each is responsible for saying in a
# comment WHICH GradeSpec field it chose and WHY the measurement moved it.


def _exposure(d: dict, k: float, item: Op, m: _Measured) -> None:
    # With a colour target this is "darken the reds", which is a qualifier's
    # luma offset, not a global exposure move.
    if item.target:
        return _band_lum(d, item.target, k * STEP_BAND_LUM)

    stops = k * STEP_EXPOSURE
    if stops > 0:
        # Measured headroom decides the push. A clip whose p99 is already at 1.0
        # has nowhere to go: another stop only welds more pixels to white, and
        # the shoulder added at the end of compile_intent cannot invent detail
        # that the extra stop destroyed.
        stops *= float(np.clip(m.headroom / 0.25, 0.3, 1.0))
    else:
        # The same argument at the other rail: crushed_low pixels are already at
        # 0 and darkening only buries more of them.
        stops *= float(np.clip(1.0 - 2.0 * m.crushed, 0.4, 1.0))
    d["exposure"] += stops


def _contrast(d: dict, k: float, item: Op, m: _Measured) -> None:
    c = k * STEP_CONTRAST
    # frame_variance IS the measurement of how contrasty the clip already is.
    # Thresholds are vibe.py's, which today only reach the model as prose; here
    # they change the number instead of asking a model to.
    if 0.0 < m.var < 0.02:
        c *= 1.5   # flat: it can take more than usual
    elif m.var > 0.09:
        c *= 0.5   # already contrasty (or the clip spans a cut)
    d["contrast"] += c

    # `pivot` has no verb, and this is why: the right pivot is measurable. Put
    # the S-curve's fixed point on the clip's own median luma so that adding
    # contrast does not double as a brightness change (spec.py: the curve fixes
    # 0, 1 and pivot). Left at the 0.435 default when p50 is unmeasured.
    if m.luma_p50 > 0.0:
        d["pivot"] = float(np.clip(m.luma_p50, 0.25, 0.65))


def _midtones(d: dict, k: float, item: Op, m: _Measured) -> None:
    e = abs(k) * STEP_MIDTONES
    # power > 1 DARKENS (it is an exponent on values in [0,1]), so the sign is
    # inverted here rather than in the vocabulary.
    if m.luma_p50 > 0.0:
        room = (1.0 - m.luma_p50) if k > 0 else m.luma_p50
        e *= float(np.clip(room / 0.5, 0.4, 1.0))  # little mid left to move -> smaller push
    _mul_rgb(d, "power", 1.0 / (1.0 + e) if k > 0 else (1.0 + e))


def _shadows(d: dict, k: float, item: Op, m: _Measured) -> None:
    u = k * STEP_SHADOWS
    if k > 0:
        # Blacks that already sit at 0.08 go milky fast; blacks on the rail have
        # the full move available.
        u *= float(np.clip(1.0 - m.floor / 0.10, 0.4, 1.0))
    else:
        # Nothing left to crush.
        u *= float(np.clip(1.0 - 2.0 * m.crushed, 0.3, 1.0))

    if k > 0 and m.headroom > 0.1:
        # Room at the top: `offset` is the honest filmic fade — it lifts the
        # whole curve inside the CDL (step 2), and it is the field an ASC CDL
        # export can actually carry off the machine.
        _add_rgb(d, "offset", (u, u, u))
    else:
        # No headroom: `offset` would push the highlights into the clip as well.
        # `shadow_lift` is exactly 0 above luma 0.5 by construction (step 6), so
        # it buys the same shadow move without spending the highlights.
        d["shadow_lift"] += u


def _highlights(d: dict, k: float, item: Op, m: _Measured) -> None:
    u = k * STEP_HIGHLIGHTS
    if k < 0 and m.clipped > 0.01:
        # The highlights are welded to 1.0. A uniform negative lift moves them
        # down together and they stay welded — flat grey instead of flat white.
        # The shoulder is the only tool that turns that area back into gradient,
        # so spend most of the request there.
        d["highlight_rolloff"] = min(0.5, d["highlight_rolloff"] + 0.3 * abs(k))
        u *= 0.5
    elif k > 0:
        u *= float(np.clip(m.headroom / 0.25, 0.3, 1.0))  # same headroom rule as exposure
    d["highlight_lift"] += u


def _warmth(d: dict, k: float, item: Op, m: _Measured) -> None:
    # The measurement that matters, and the reason this module exists: a clip
    # already sitting in the orange part of the wheel needs LESS push to read as
    # warmer, and MORE to read as cooler, because cooling it has to cross
    # neutral first. `dominant_hue` is the only field that knows this, and today
    # nothing consumes it.
    d["temperature"] += k * STEP_WARMTH * (1.0 - _BIAS_GAIN * m.warm_bias * _sign(k))


def _tint(d: dict, k: float, item: Op, m: _Measured) -> None:
    # Same argument as _warmth, on the green/magenta axis. up = magenta.
    d["tint"] += k * STEP_TINT * (1.0 - _BIAS_GAIN * m.magenta_bias * _sign(k))


def _saturation(d: dict, k: float, item: Op, m: _Measured) -> None:
    if item.target:
        return _band_sat(d, item.target, k)  # "drain the greens"

    u = abs(k) * STEP_SATURATION
    if k > 0:
        # Measured chroma decides how far a boost is worth taking. Above ~0.35
        # mean chroma the extra saturation mostly clips (vibe.py says the same
        # thing to the model in prose); below ~0.06 the clip is nearly
        # monochrome and a small multiplier does not read at all.
        if m.sat > 0.35:
            u *= 0.5
        elif 0.0 < m.sat < 0.06:
            u *= 1.5
    # Multiplicative in both directions so two "less saturated" ops compose and
    # the result can never go negative.
    d["saturation"] *= (1.0 + u) if k > 0 else 1.0 / (1.0 + u)


def _split_tint(field: str):
    def apply_tint(d: dict, k: float, item: Op, m: _Measured) -> None:
        # NO measured scaling, deliberately. The only measurement that bears on
        # a split tint is how much of it the final clip truncates on a clip
        # whose blacks are already at the rail (spec.py measures 6.1% of grid
        # points affected) — and scaling the push UP cannot recover a truncated
        # channel. The fix for that is lifting the shadows, which is a separate
        # verb the user has to actually ask for.
        colour = item.target or DEFAULT_TINT[item.op]
        _add_rgb(d, field, np.array(_TINT_VECTORS[colour]) * (k * STEP_SPLIT_TINT))

    return apply_tint


def _effect(name: str, step: float = STEP_EFFECT):
    def apply_effect(d: dict, k: float, item: Op, m: _Measured) -> None:
        # NO measured scaling either, and this one is a limitation rather than a
        # choice: probe.py samples 8-bit stills at 256px, where its own docstring
        # warns that frame_variance and the clipped/crushed counts are proxies.
        # Nothing in ClipStats can tell real grain from fine detail, so there is
        # no measurement to consult. Note that `dir: down` on denoise/glow/grain
        # goes negative and sanitize() clamps it back to 0 — the spec has no
        # negative grain, so "less grain than none" is correctly a no-op.
        d["effects"][name] += k * step

    return apply_effect


_COMPILERS = {
    "exposure": _exposure,
    "contrast": _contrast,
    "midtones": _midtones,
    "shadows": _shadows,
    "highlights": _highlights,
    "warmth": _warmth,
    "tint": _tint,
    "saturation": _saturation,
    "shadow_tint": _split_tint("shadow_tint"),
    "highlight_tint": _split_tint("highlight_tint"),
    "grain": _effect("grain"),
    "glow": _effect("glow"),
    "vignette": _effect("vignette"),
    "softness": _effect("softness"),
    "denoise": _effect("denoise"),
    "fringe": _effect("fringe", STEP_FRINGE),
}
# A verb the compiler does not implement would silently do nothing, which is the
# worst possible failure mode here: the user gets a sentence back describing a
# move that never happened.
assert set(_COMPILERS) == set(OPS), sorted(set(OPS) ^ set(_COMPILERS))


# ---- auto-balance ----------------------------------------------------------
#
# WHY IT EXISTS. Every verb above starts from wherever the clip happens to sit.
# Two shots of the same scene, one a little green and one a little magenta, get
# the same look applied on top of two different starting points and come out
# different. A colourist balances first and grades second for exactly that
# reason, and the balance is closed form from ClipStats — no model, no taste.
#
# WHICH FIELDS: `slope`/`offset`, the CDL at step 2, for both halves of it.
#   * NOT `exposure` (step 1). Exposure sits before the CDL precisely so that
#     `offset` stays an ABSOLUTE lift for the shadow verb, and spending it here
#     would rescale every offset written after it.
#   * NOT `temperature`/`tint` (step 3). Those are the colour verbs' fields, and
#     match.py already gives the argument: the per-channel slope carries the
#     colour shift, so a temperature push on top would double it. Leaving them
#     at identity is what keeps "warmer" meaning the same thing whether or not the
#     balance ran — which is the composition property this pass has to have.
#   * NOT the tonal split (step 6). Those masks are luma-gated and are the
#     creative shadow/highlight tools. A black point is not a shadow tint.
# Two things fall out for free: `_protect_highlights` already reads slope/offset
# and so grows a shoulder over the composed grade, and an ASC CDL export
# (roadmap A8) is the one interchange format that carries the balance with it.
#
# The solve itself is match.py's, unmodified. Auto-balance is a reference match
# where the reference is this clip with the colour taken out of it.

# WHERE A CAST STOPS BEING A CAST AND STARTS BEING THE LOOK. This is the whole
# risk of the pass: a sodium-lit night exterior is SUPPOSED to be orange, and
# neutralising it grades the look back out of footage that was lit or graded
# deliberately. `hue_strength` is the only field that can tell the two apart —
# it is the resultant length of the chroma-weighted hue vector, so it is large
# only when the frame is BOTH colourful AND agrees with itself about which
# colour. `saturation` cannot do this job: it is HSV chroma/max, and probe.py
# says why. Measured with probe.py over this repo's own files and over synthetic
# casts (scratch script, 40k-pixel frames):
#
#   neutral ramp, no cast                        0.0004
#   teal-and-orange grade, two opposite hues     0.0015   <- saturation 0.42
#   assets/sample.mp4 (testsrc2 colour bars)     0.021    <- saturation 0.97
#   green fluorescent cast, G gain 1.06          0.030
#   test_files/ref_tvd.png                       0.030
#   green fluorescent cast, G gain 1.10          0.049
#   tungsten cast, R/B +/-6%                     0.052
#   test_files/ironman.gif                       0.061
#   tungsten cast, R/B +/-10%                    0.087
#   ---------------------------------------------------- casts above, looks below
#   sodium-lit street                            0.153
#   blue night exterior                          0.154
#   test_files/giphy.gif (teal look)             0.169
#   test_files/test.mp4 (warm look)              0.189
#   test_files/higher_def.gif (blue look)        0.257
#   assets/ref_warm.png (solid brown)            0.446
#
# The two rows either side of the line are 0.087 and 0.153, so the gate is a
# ramp across that gap rather than a threshold in it: full correction at or
# below 0.04, none at or above 0.12, linear between. That leaves the nearest
# real look (sodium, 0.153) 1.28x clear of any correction at all, and still
# gives every synthetic cast up to a 10% channel error 40-100% of its fix. A
# gently graded clip landing mid-ramp is half-corrected, never reversed.
#
# Note this also bounds the SIZE of the correction, without a second constant:
# hue_strength equals mean absolute chroma when the hue is consistent, so a
# clip the gate lets through by definition has only a small cast to remove.
_CAST_FULL = 0.04
_CAST_NONE = 0.12

# Below one 8-bit code value the balance did something nobody can see, and
# saying so in the rationale would be a lie of emphasis rather than a report.
_SAY = 1.0 / 255.0

# probe.py's dominant_hue in degrees -> a word for the rationale. The angles are
# _TINT_VECTORS' own vocabulary, so the sentence a balance writes and the colour
# a verb takes as a `target` mean the same thing. Nearest-angle on the circle:
# a warm cast at 33 degrees has to read as "orange", which snapping to spec.py's
# six 60-degree band centres cannot say.
_CAST_NAMES = {"red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
               "cyan": 180.0, "blue": 240.0, "purple": 270.0, "magenta": 300.0}


def _hue_name(deg: float) -> str:
    return min(_CAST_NAMES, key=lambda n: abs((deg - _CAST_NAMES[n] + 180.0) % 360.0 - 180.0))


def _balance(d: dict, stats: "ClipStats", m: _Measured) -> list[str]:
    """Neutralise the clip from its measurements. Writes slope/offset only.

    Takes `stats` and not just `_Measured` because the solve is match.py's and
    match.py's input is a ClipStats — the point is to reuse that solver rather
    than re-derive it here.

    Runs BEFORE the verbs and ASSIGNS rather than composes: the balance is the
    base the creative grade sits on, and every verb above either adds into
    `offset` (the shadow verb, correctly, on top of the corrected black point)
    or lives in a later step entirely.

    Returns the sentences describing what it actually did, in describe()'s voice
    and lower case, so that a balance can never happen silently.
    """
    said: list[str] = []
    slope, offset = np.ones(3), np.zeros(3)

    # -- the cast --
    # hue_strength defaults to 0.0 on a session written before probe.py measured
    # it (ClipStats keeps a default for every field added after `duration`), and
    # 0.0 reads as "correct this cast in full" — on footage that could be a
    # sodium street. Measured footage never reads exactly 0: the flattest clip
    # in the table above is 0.0004. So an exact 0 next to any chroma at all is
    # UNMEASURED, and the safe direction for a correction nobody asked for is to
    # not make it. _HUE_CONF_FLOOR is reused because it already means "below
    # this there is not enough chroma for a hue to exist".
    unmeasured = stats.hue_strength == 0.0 and stats.saturation > _HUE_CONF_FLOOR
    conf = 0.0 if unmeasured else float(
        np.clip((_CAST_NONE - stats.hue_strength) / (_CAST_NONE - _CAST_FULL), 0.0, 1.0))
    if conf > 0.0:
        # Grey world, solved by match_reference against a reference that is this
        # clip with the colour taken out: every channel mean on the clip's own
        # LUMA mean, every channel spread on its LUMA spread. Anchoring the grey
        # on luma rather than on the mean of the three means is spec.py's rule
        # for a tint (luma-stripped, so colour and brightness stay independent
        # axes) applied to the balance — at the mean this correction is exactly
        # luma-preserving, so it changes the colour and not the exposure.
        neutral = stats.model_copy(update={
            "mean": RGB.of(float(m.mean @ LUMA)),
            "std": RGB.of(float(stats.std.as_array() @ LUMA)),
        })
        fit = match_reference(stats, neutral)
        # Lerp the whole transform toward identity by the confidence, not the
        # slope alone: conf = 0 has to be a bit-for-bit no-op, and a half-trusted
        # cast has to be half-corrected rather than corrected then partly undone.
        slope = 1.0 + (fit.slope.as_array() - 1.0) * conf
        offset = fit.offset.as_array() * conf
        if float(np.max(np.abs(m.mean * slope + offset - m.mean))) > _SAY:
            name = _hue_name(stats.dominant_hue)
            said.append(f"neutralised {'an' if name[0] in 'aeiou' else 'a'} {name} cast")

    # -- the black point --
    # p1/p99 carried through the cast correction: where the real range sits once
    # the colour is off it. min/max ACROSS channels, because this move is
    # achromatic by construction — a per-channel percentile stretch is a second
    # cast correction, and it would run without the evidence the gate above
    # demands of the first one.
    b0 = float((m.p1 * slope + offset).min())
    w0 = float((m.p99 * slope + offset).max())
    # `crushed_low` for the same reason `_shadows` consults it: pixels already on
    # the rail are welded, and pulling the black point down only welds more.
    # Floored at 0 rather than _shadows' 0.3 — a verb the user asked for has to
    # produce something, an unrequested correction may correctly do nothing.
    # In practice the two measurements are not independent: p1 is a per-channel
    # 1st percentile, so crushed_low much above 0.03 already forces some
    # channel's p1 onto the rail and b0 with it. This is the belt to that braces.
    pull = float(np.clip(1.0 - 2.0 * m.crushed, 0.0, 1.0))
    # No measured range: a near-solid frame, or the all-zero p1/p50/p99 of a
    # session written before probe.py grew percentiles (_measure documents the
    # same fallback for headroom). Either way there is no black point to find.
    if w0 - b0 > 0.05:
        target = b0 * (1.0 - pull)
        # Move the black point to `target` and PIN the white point. w0 is an
        # anchor, not a second target: blacks sitting at 0.08 are fog and always
        # a defect, whereas a clip whose p99 is 0.45 is a night scene, and
        # stretching that to full range is the creative call `exposure` and
        # `contrast` are the verbs for. Auto-levels that invents range is the
        # same harm as neutralising a sodium street, one axis over.
        g = (w0 - target) / (w0 - b0)
        slope, offset = slope * g, offset * g + (target - g * b0)
        if abs(target - b0) > _SAY:
            said.append("set the black point")

    for c, s, o in zip("rgb", slope, offset):
        d["slope"][c], d["offset"][c] = float(s), float(o)
    return said


# ---- the compiler ----------------------------------------------------------


def compile_stack(intent: Intent, stats: "ClipStats", balance: bool = False) -> GradeStack:
    """Typed verbs + measured statistics -> a GradeStack. No model, no I/O.

    An empty Intent compiles to a FLAT stack whose base is the identity grade
    bit-for-bit; that is the property the whole design rests on, since it means
    an Intent only ever describes DEPARTURES from the source.

    `balance` runs the auto-balance pass first (roadmap A6): neutralise the clip
    from its own measurements, then apply the creative look on top of a known
    starting point. It is OFF by default and the identity property above is the
    reason — a balance is a departure the user did not ask for, so it has to be
    a caller's decision and not this function's. When it is on it announces
    itself in `rationale` ahead of the intent's own sentences, because a silent
    correction fighting the user's grade is worse than no correction at all.

    REGIONS SPLIT THE INTENT IN TWO (roadmap B1). Ops with no region target
    compile into the base exactly as they always did; ops that name one are
    grouped by region — in first-mention order — and each group compiles into a
    layer of its own, from a FRESH identity spec. So a layer's GradeSpec holds
    one correction, and the base is bit-for-bit what it would have been if the
    regional ops had not been asked for. The balance is deliberately not
    repeated per layer: it is a property of the clip, not of a corner of it.
    """
    m = _measure(stats)
    d = GradeSpec.identity().model_dump()

    # Before the verbs: they add into the same CDL and later steps, so the look
    # lands on top of the corrected base rather than beside it.
    said = _balance(d, stats, m) if balance else []
    _run_ops([o for o in intent.ops if o.target not in REGIONS], d, m, intent.strength)

    # The model no longer writes the rationale — the sentences ARE the intent,
    # so they cannot drift from what the numbers do. Every op is named here,
    # regional ones included: the list is what the UI shows, and a move the user
    # asked for going unmentioned because it landed in a layer would be the same
    # silent-drop bug one level up.
    text = ", ".join(said + describe(intent))
    d["rationale"] = (text[0].upper() + text[1:] + ".") if text else ""

    layers = []
    # dict.fromkeys, not a set: layer order is evaluation order (region.py), so
    # it has to be the order the person said things in and nothing else.
    for target in dict.fromkeys(o.target for o in intent.ops if o.target in REGIONS):
        ops = [o for o in intent.ops if o.target == target]
        ld = GradeSpec.identity().model_dump()
        _run_ops(ops, ld, m, intent.strength)
        ld["rationale"] = ", ".join(describe(Intent(ops=ops)))
        layers.append(Layer(region=for_target(target), spec=GradeSpec(**ld).sanitize()))

    return GradeStack(base=GradeSpec(**d).sanitize(), layers=layers)


def compile_intent(intent: Intent, stats: "ClipStats", balance: bool = False) -> GradeSpec:
    """The base grade of `compile_stack` — the whole answer whenever no op names
    a region, and every consumer that only speaks GradeSpec still gets one."""
    return compile_stack(intent, stats, balance=balance).base


def _run_ops(ops: list[Op], d: dict, m: _Measured, strength: str) -> None:
    """Compile a group of ops into one spec dict, then close it out.

    Shared by the base and by every layer so that a regional correction is
    computed by exactly the code path a global one is — the alternative is two
    implementations of "moderate warm" that drift.
    """
    for item in ops:
        k = UNIT[item.amount] * (1.0 if item.dir == "up" else -1.0)
        # A region target has ALREADY answered "which pixels", so inside the
        # group the verb acts on all of them. Without this, `_exposure` would
        # look "top" up in the hue-band table and raise.
        if item.target in REGIONS:
            item = item.model_copy(update={"target": ""})
        _COMPILERS[item.op](d, k, item, m)

    _protect_highlights(d, m)

    # look_mix last: it is the outermost operation in apply() (step 9) and it
    # scales everything above it, exactly as the word "strength" implies. A
    # layer gets the same mix as the base: strength is "how much of what I asked
    # for", and half of a look is half of its regional half too.
    d["look_mix"] = STRENGTH_MIX[strength]


def _protect_highlights(d: dict, m: _Measured) -> None:
    """Add exactly as much shoulder as the grade we just built needs.

    match.py's argument, applied to a different input: a grade that pushes the
    measured white point above 1.0 hard-clips it, and the detail is gone for
    good in the baked .cube. Solve for the peak this grade actually produces
    rather than picking a constant.

    ponytail: the estimate ignores `power` and the white-balance gains, both
    within a few percent over the step sizes above, and it only ever ADDS
    shoulder. It is bounded below by whatever the highlights verb already asked
    for, and it is exactly 0.0 for an untouched spec — which is what keeps the
    identity path bit-for-bit.
    """
    slope = np.array([d["slope"][c] for c in "rgb"])
    offset = np.array([d["offset"][c] for c in "rgb"])
    peak = float(np.max(m.p99 * (2.0 ** d["exposure"]) * slope + offset))
    peak += max(0.0, d["highlight_lift"])
    d["highlight_rolloff"] = max(d["highlight_rolloff"], float(np.clip((peak - 1.0) / 2.0, 0.0, 1.0)))


# ---- small helpers ---------------------------------------------------------


def _sign(k: float) -> float:
    return 1.0 if k > 0 else -1.0


def _add_rgb(d: dict, field: str, vec) -> None:
    for c, v in zip("rgb", vec):
        d[field][c] += float(v)


def _mul_rgb(d: dict, field: str, factor: float) -> None:
    for c in "rgb":
        d[field][c] *= float(factor)
