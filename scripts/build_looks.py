#!/usr/bin/env python3
"""Rebuild the looks corpus in ragvid/look_corpus/ — reproducible, no LLM, no hand-typed numbers.

Every number in every corpus entry is MEASURED, never authored. The pipeline is:

    base still  --ffmpeg look filter-->  reference still
         |                                    |
    probe_image                          probe_image
         |                                    |
         +---- match.match_reference() -------+      slope/offset/sat, closed form
                        |
                 fit_hue_bands()                     6 sat + 6 lum, one lstsq
                        |
                 fit_tonal_split()                   2 tints + 2 lifts, closed form
                        |
                 recover_effect()                    spatial, from a still pair
                        |
                 factor_exposure()                   exact re-parameterisation
                        |
                   GradeSpec

Each solver fits the residual the previous ones left, in apply()'s own
evaluation order, so no solver is ever handed a target another one has already
claimed. Measured over the corpus, median RMSE against the reference:
match_reference alone 4.45 code values -> 3.85 after the hue bands -> 3.16
after the tonal split. A solver that does not move that number does not belong
here.

The base still is a 6-frame tile montage of test_files/test.mp4 — real footage
already in the repo, with photographic statistics (mean 0.58/0.51/0.39, sat
0.36), not a synthetic ramp. Both stills come from the same pixels, so the
solved spec isolates the LOOK and nothing about the scene.

What IS hand-written here: a name, mood words, and the ffmpeg filter chain that
stands in for "a still shot on that stock". That is the STIMULUS — the
equivalent of choosing which frame to reference — and it is recorded verbatim in
each entry's `provenance` so any claim in the corpus can be re-derived from
pixels. The GradeSpec itself is never touched by hand. For the four spatial
effects the stimulus is render.py's own _effect_filters output, so a recovered
strength round-trips through the renderer that will actually apply it.

ONE CAVEAT, STATED HERE RATHER THAN BURIED: the base still is a 3x2 montage,
so every spatial measurement is made at the montage's geometry. Grain, softness
and glow are local and do not care. `vignette` does -- its ffmpeg angle is
relative to the frame, and the montage is 8:3, not 16:9 -- so the recovered
vignette strength is exact for the stimulus and approximate for a real frame.
It is a measured number either way, which is the rule; it is just measured
somewhere slightly other than where it will be used.

Stills land in out/ (gitignored, regenerable); only ragvid/look_corpus/*.json is committed.

Usage:  uv run python scripts/build_looks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ragvid.match import match_reference  # noqa: E402
from ragvid.probe import probe_image  # noqa: E402
from ragvid.render import _effect_filters  # noqa: E402
from ragvid.spec import (  # noqa: E402
    HUE_CENTERS, HUE_CHROMA_GATE, HUE_FIELDS, HUE_HALFWIDTH, LUMA, SPLIT_CROSSOVER,
    EffectSpec, GradeSpec, HueBand, RGB, _hue_chroma, _smoothstep,
)

# Ridge (Tikhonov) weight for the hue solve, as a fraction of the largest
# column norm of the design matrix. See fit_hue_bands for why it is not zero,
# and for the sweep that picked this value.
HUE_RIDGE = 0.03

SRC = REPO / "test_files" / "test.mp4"
WORK = REPO / "out" / "looks_src"
CORPUS = REPO / "ragvid" / "look_corpus"

# 6 frames of the source, tiled into one still: a single probe_image call then
# sees six moments of the clip instead of one possibly-atypical frame.
BASE_VF = "select='not(mod(n,90))',scale=320:-2,tile=3x2"

# name -> (mood words, ffmpeg filter chain applied to the base still).
# The words describe the INTENT; the numbers in the entry are measured from the
# result. If a word and its measured spec ever disagree, the spec is right.
LOOKS: dict[str, tuple[str, str]] = {
    "golden hour": (
        "warm sunny amber golden summer nostalgic soft romantic evening sunset",
        "colortemperature=temperature=4200,eq=saturation=1.12:brightness=0.03",
    ),
    "teal and orange": (
        "cinematic blockbuster action punchy contrast teal orange trailer hollywood",
        "colorbalance=rs=-0.12:bs=0.18:rh=0.18:bh=-0.14,eq=saturation=1.25:contrast=1.15",
    ),
    "bleach bypass": (
        "gritty harsh desaturated washed bleak war documentary raw brutal",
        "eq=saturation=0.35:contrast=1.45,curves=preset=increase_contrast",
    ),
    "moonlight": (
        "cool blue night moody dark nocturnal cold lonely dream",
        "colortemperature=temperature=12000,eq=brightness=-0.09:saturation=0.8:gamma=0.85",
    ),
    "vintage film": (
        "faded retro nostalgic seventies analog grainy old warm milky",
        "curves=preset=vintage",
    ),
    "film noir": (
        "monochrome black white dramatic contrast shadows crime classic stark",
        "eq=saturation=0.05:contrast=1.5,curves=preset=strong_contrast",
    ),
    "pastel dream": (
        "soft airy light pale pastel gentle romantic ethereal bright dreamy",
        "eq=contrast=0.72:brightness=0.1:saturation=0.85,curves=preset=lighter",
    ),
    "cyberpunk neon": (
        "neon magenta purple futuristic night city electric synthwave vivid",
        "colorbalance=rh=0.2:bh=0.25:gs=-0.1,eq=saturation=1.5:contrast=1.2:brightness=-0.04",
    ),
    "sun bleached desert": (
        "hot dusty arid bright harsh sand western daylight faded",
        "colortemperature=temperature=4800,eq=brightness=0.12:contrast=0.85:saturation=0.7",
    ),
    "sickly green": (
        "green eerie sickly institutional uneasy horror clinical toxic",
        "colorbalance=gm=0.18:gh=0.12:rm=-0.08,eq=saturation=0.9:contrast=1.1",
    ),
    "cold steel": (
        "cold clinical sterile blue grey modern corporate hard technical",
        "colortemperature=temperature=9000,eq=saturation=0.6:contrast=1.2",
    ),
    "candlelit interior": (
        "cozy warm intimate amber candlelight indoor low soft quiet",
        "colortemperature=temperature=3400,eq=brightness=-0.06:saturation=1.05:gamma=0.9",
    ),
    "punchy commercial": (
        "vivid bold punchy saturated clean crisp advertising energetic",
        "eq=saturation=1.4:contrast=1.25,curves=preset=medium_contrast",
    ),
    "flat log": (
        "flat neutral ungraded low contrast soft milky raw baseline",
        "eq=contrast=0.6:saturation=0.75:brightness=0.05",
    ),
    "rose gold": (
        "pink rose warm soft romantic blush feminine gentle glow",
        "colorbalance=rm=0.12:rh=0.14:bm=0.06:gm=-0.04,eq=saturation=1.05:brightness=0.04",
    ),
    "underwater": (
        "cyan aqua underwater deep cool submerged blue green murky",
        "colorbalance=bm=0.18:gm=0.12:rm=-0.16,eq=saturation=1.1:brightness=-0.05",
    ),
    "cross process": (
        "experimental lomo skewed chemical indie quirky music video",
        "curves=preset=cross_process",
    ),
    "high key white": (
        "bright white clean airy studio minimal fresh overexposed",
        "eq=brightness=0.18:contrast=0.8:saturation=0.9,curves=preset=lighter",
    ),

    # --- hue-qualifier stimuli ---------------------------------------------
    # selectivecolor moves reds/yellows/greens/cyans/blues/magentas
    # independently, which is exactly the axis the six HUE_CENTERS bands span.
    # A global CDL cannot follow it, so the residual it leaves is hue-shaped and
    # fit_hue_bands can see it.
    "autumn leaves": (
        "autumn fall october red orange leaves harvest woodland crisp turning",
        "selectivecolor=reds=0.2 -0.15 -0.2 0:yellows=0.15 -0.1 -0.25 0"
        ":greens=-0.1 0.05 0.3 0.05,eq=saturation=1.1",
    ),
    "sodium street lamp": (
        "night street urban sodium lamp orange grimy city sidewalk late",
        "selectivecolor=yellows=-0.3 -0.1 0.45 -0.05:reds=-0.15 0 0.25 0"
        ":blues=0.25 0.1 -0.3 0.15:cyans=0.2 0.05 -0.2 0.1,eq=contrast=1.1",
    ),
    "emerald forest": (
        "forest green lush jungle woodland nature verdant damp mossy",
        "selectivecolor=greens=-0.3 0.1 0.25 -0.1:cyans=-0.2 0.05 0.15 -0.05"
        ":reds=0.1 0.05 0.05 0.1:yellows=-0.15 0 0.2 0,eq=saturation=1.15",
    ),
    "acid wash": (
        "acid trippy psychedelic clashing bold graphic loud lomo poster",
        "selectivecolor=cyans=-0.35 0.2 -0.15 -0.1:magentas=0.3 -0.3 0.2 -0.1"
        ":blues=-0.25 0.15 0.25 -0.1:greens=0.2 -0.25 -0.1 0,eq=saturation=1.3",
    ),

    # --- spatial-effect stimuli --------------------------------------------
    # The third element is the effect stimulus. It is rendered by render.py's
    # OWN _effect_filters, so the recovered number round-trips through the
    # actual renderer rather than through a lookalike chain written here.
    "super 8": (
        "grainy super8 home movie amateur retro seventies memory flicker",
        "curves=preset=vintage,eq=saturation=0.9:contrast=1.05",
        {"grain": 0.45},
    ),
    "dream haze": (
        "hazy dreamy bloom halation glow soft romantic memory ethereal",
        "eq=brightness=0.05:contrast=0.9:saturation=1.05",
        {"glow": 0.35},
    ),
    "old lens": (
        "vintage lens falloff dark corners tunnel antique portrait heirloom",
        "colortemperature=temperature=4600,eq=saturation=0.95",
        {"vignette": 0.55},
    ),
    "soft focus portrait": (
        "soft focus beauty portrait flattering diffusion gentle glamour",
        "eq=contrast=0.95:saturation=1.05:brightness=0.02",
        {"softness": 0.3},
    ),
}

# NOT IN THE CORPUS, DELIBERATELY: look_mix.
#
# Every corpus entry is a look at FULL strength -- that is what "a look" means.
# look_mix is the user's strength dial (the UI slider), a property of how much
# of a look someone wants, not a property of the look itself. A corpus entry
# with look_mix < 1 would be the same look with the volume turned down, and it
# would teach the model to emit a half-strength grade when the user asked for a
# full-strength one. The measured coverage table will therefore always read
# `look_mix 0/N`, and that is correct. Do not "fix" it by inventing values.


def ff(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        check=True, capture_output=True,
    )


def slug(name: str) -> str:
    return name.replace(" ", "_")


def pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def fit_tonal_split(spec: GradeSpec, base_px: np.ndarray, ref_px: np.ndarray) -> GradeSpec:
    """Solve the shadow/highlight tint + lift from what the CDL match left over.

    match_reference solves a single global CDL, so on its own it produces a
    corpus in which all four tonal-split fields are 0.0 in every entry — which
    would teach the model that "teal shadows, orange highlights" is not a thing
    the spec can express. That is the opposite of grounding.

    So fit them, still without inventing a number. The residual after the CDL is
    r = ref - spec.apply(base). apply() adds w_s*d_s + w_h*d_h at step 6 with the
    SAME masks used here, and the two masks have DISJOINT support, so the least-
    squares solution decouples per band to d = sum(w*r)/sum(w*w) per channel —
    closed form, no iteration, no fitting library.

    Splitting d into the spec's two knobs is exact, not approximate. apply()
    contributes `t - L(t) + lift`; with lift = L(d) and t = d - L(d) we get
    L(t) = L(d) - L(d) = 0 (LUMA sums to 1), so the contribution is exactly d.

    READ THE RESULT AS A RESIDUAL, NOT AS COLOUR DESIGN. `teal and orange`
    comes out with a WARM shadow_tint, which looks backwards until you notice
    the CDL that precedes it already over-rotates the shadows teal (its solved
    saturation is 1.87) and this term pulls them back. The composite renders the
    reference; the field in isolation does not describe the look. Measured over
    all 26 entries it lowers RMSE against the reference in every single one --
    median 3.85 -> 3.16 code values, best 6.50 -> 3.27 (golden hour) -- and it
    still does so with fit_hue_bands running ahead of it, which is the test that
    matters: two solvers that were both just absorbing the same residual would
    show the second one earning nothing.
    """
    got = spec.apply(base_px).reshape(-1, 3)
    resid = ref_px.reshape(-1, 3) - got
    L = (np.clip(got, 0.0, 1.0) @ LUMA)[:, None]
    masks = (
        1.0 - _smoothstep(np.clip(L / SPLIT_CROSSOVER, 0.0, 1.0)),
        _smoothstep(np.clip((L - SPLIT_CROSSOVER) / (1.0 - SPLIT_CROSSOVER), 0.0, 1.0)),
    )
    update = {}
    for w, tint_field, lift_field in zip(masks, ("shadow_tint", "highlight_tint"),
                                         ("shadow_lift", "highlight_lift")):
        denom = float((w * w).sum())
        # A frame with no true shadows (or no highlights) leaves that band
        # unconstrained; leave it at identity rather than dividing by ~0.
        if denom < 1.0:
            continue
        d = (w * resid).sum(axis=0) / denom
        lift = float(d @ LUMA)
        t = d - lift
        update[tint_field] = RGB(r=t[0], g=t[1], b=t[2])
        update[lift_field] = lift
    return spec.model_copy(update=update).sanitize() if update else spec


def fit_hue_bands(spec: GradeSpec, base_px: np.ndarray, ref_px: np.ndarray) -> GradeSpec:
    """Solve the six hue qualifiers as ONE linear least-squares problem.

    A global CDL is hue-blind by construction, so a reference that moved reds
    and yellows in opposite directions leaves a residual that no slope/offset
    can absorb -- and with no solver for it every corpus entry would report all
    six bands at identity, teaching the model that "warm the skin tones, leave
    the sky alone" is not a thing the spec can say.

    It is linear, and exactly so. At step 5 apply() computes

        s = 1 + gate * (sum_i w_i * sat_i - 1)      o = gate * sum_i w_i * lum_i
        out = L + (x - L) * s + o

    The six band weights are an EXACT partition of unity (that is why
    HUE_HALFWIDTH equals the band spacing -- see spec.py), so sum_i w_i == 1 and
    `sum_i w_i * sat_i - 1` collapses to `sum_i w_i * (sat_i - 1)`. Substituting
    and using L + (x - L) == x:

        out = x + sum_i [gate * w_i * (x - L)] * (sat_i - 1)
                + sum_i [gate * w_i] * lum_i

    which is `x` plus a linear combination of 12 known per-pixel columns. So the
    12 unknowns fall out of a single np.linalg.lstsq against the residual. No
    iteration, no fitting library, no line search.

    THE MASKS AND THE GATE ARE apply()'s OWN. `x` here is
    spec.through_saturation(base) -- literally the array apply() hands to step 5
    -- and w/gate are rebuilt from spec.HUE_CENTERS / HUE_HALFWIDTH /
    HUE_CHROMA_GATE via spec._hue_chroma. A solver that skipped the chroma gate
    would hand back bands that tint every grey the renderer refuses to touch,
    i.e. a corpus describing a look the renderer does not produce.

    A band with near-zero weight mass in this stimulus is UNCONSTRAINED (a still
    with no cyan in it says nothing about the cyan band), and its column is
    dropped so it stays at identity -- the same call fit_tonal_split makes when
    a mask's denominator is tiny.

    The residual is measured in OUTPUT space while the columns live in step-5
    space, so the downstream steps (tonal split, rolloff, contrast, mix, clip)
    are treated as pass-through. That is an approximation, not an identity, and
    it is why the build prints the RMSE before and after: the solver only earns
    its place if the measured error actually drops. It does — median 4.45 -> 3.85
    code values over the corpus, and it drops on 26 entries out of 26.

    WHY THE RIDGE. Weight mass alone does not say a band is constrained. A
    near-monochrome stimulus (`film noir`, solved saturation 0.216) has weight in
    every band and chroma in none, so the `x - L` factor in the sat columns goes
    to ~0 and the plain normal equations answer with enormous coefficients:
    unregularised, 10 of 156 bands came back needing |sat - 1| > 1 and 7 needing
    |lum| > 0.2, several pinned on sanitize()'s own rails — a solver whose answer
    has to be clamped has left the range the spec claims to be valid over.
    Tikhonov shrinkage toward IDENTITY (sat = 1, lum = 0 is the origin in this
    parameterisation, which is why the solve is written in deltas) is the
    textbook fix and costs 12 extra rows.

    HUE_RIDGE was measured, not guessed. Sweeping it over 0 .. 0.1 on the whole
    corpus, mean RMSE runs 4.10 / 4.10 / 4.10 / 4.10 / 4.10 / 4.11 / 4.13 / 4.23
    at 0 / 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-2 / 1e-1, while the count of
    railed bands runs 10 / 6 / 3 / 1 / 0 / 0 / 0 / 0. 0.03 is the largest value
    still within 1% of the unregularised error (+0.7%); 0.1 costs 3.2% and
    visibly flattens the bands. Accuracy is flat across three decades here
    because the extreme coefficients were buying almost nothing.
    """
    x = spec.through_saturation(base_px).reshape(-1, 3)
    resid = ref_px.reshape(-1, 3) - spec.apply(base_px).reshape(-1, 3)

    h, c = _hue_chroma(x)
    d = ((h[:, None] - HUE_CENTERS + 180.0) % 360.0) - 180.0
    w = _smoothstep(np.clip(1.0 - np.abs(d) / HUE_HALFWIDTH, 0.0, 1.0))  # (N, 6)
    gw = _smoothstep(np.clip(c / HUE_CHROMA_GATE, 0.0, 1.0))[:, None] * w
    chroma = x - (x @ LUMA)[:, None]

    # One row per (pixel, channel): the sat columns act on the chroma vector,
    # the lum columns are a flat per-pixel offset shared by all three channels.
    n = x.shape[0]
    A = np.empty((n * 3, 12))
    for i in range(6):
        A[:, i] = (gw[:, i : i + 1] * chroma).reshape(-1)
        A[:, 6 + i] = np.repeat(gw[:, i], 3)

    # Weight mass in pixel-equivalents. Below one full-weight pixel the band is
    # not in the stimulus at all; leave it at identity rather than letting the
    # pseudo-inverse read noise.
    keep = gw.sum(axis=0) >= 1.0
    if not keep.any():
        return spec
    cols = np.concatenate([keep, keep])
    Ak = A[:, cols]
    b = np.concatenate([resid.reshape(-1), np.zeros(Ak.shape[1])])
    ridge = HUE_RIDGE * float(np.linalg.norm(Ak, axis=0).max())
    Ak = np.vstack([Ak, ridge * np.eye(Ak.shape[1])])
    sol = np.zeros(12)
    sol[cols] = np.linalg.lstsq(Ak, b, rcond=None)[0]

    update = {
        f: HueBand(sat=1.0 + sol[i], lum=sol[6 + i])
        for i, f in enumerate(HUE_FIELDS)
        if keep[i]
    }
    return spec.model_copy(update=update).sanitize()


def factor_exposure(spec: GradeSpec, base_px: np.ndarray) -> GradeSpec:
    """Move the common gain out of `slope` and into `exposure`. EXACT, not a fit.

    match_reference puts every bit of gain in the per-channel slope, so a corpus
    built from it reports exposure = 0.0 in every single entry -- and a model
    grounded on that never reaches for the one field a human actually names
    ("half a stop brighter"). But exposure is not a separate degree of freedom:
    apply() does `x * 2**exposure` and then `x * slope`, so

        (2**e) * (slope / 2**e) == slope

    for any e. Choosing e = log2(geometric mean of slope) is the choice that
    leaves the residual slope at geometric mean 1.0, i.e. pure colour balance
    with the brightness factored out. Nothing is re-fitted and nothing moves,
    and this function asserts exactly that: the re-parameterised spec must
    render the base still to within 1e-12 of what it rendered before. Measured
    over the corpus the largest drift is 1.1e-15, i.e. floating-point dust. An
    assertion failure here means the factoring is wrong, not that the tolerance
    is tight.
    """
    s = spec.slope.as_array()
    if np.any(s <= 0.0):
        return spec  # a dead channel has no logarithm; leave it alone
    e = float(np.mean(np.log2(s)))
    out = spec.model_copy(update={
        "exposure": e,
        "slope": RGB(r=s[0] / 2.0**e, g=s[1] / 2.0**e, b=s[2] / 2.0**e),
    }).sanitize()
    drift = float(np.abs(out.apply(base_px) - spec.apply(base_px)).max())
    assert drift < 1e-12, f"exposure factoring is not exact: {drift:.3e}"
    return out


# ---- spatial effects ------------------------------------------------------
#
# Effects are SPATIAL, so GradeSpec.apply() cannot see them and a CDL solver
# cannot either. They are still measurable from a pair of stills, which keeps
# the derived-never-authored rule intact: render the graded still twice, once
# with the effect and once without, and read the ratio of a statistic the
# effect is known to move.
#
# The stimulus is render.py's OWN _effect_filters output, so a recovered value
# written into the corpus round-trips through the real renderer. Measured
# round-trip error (stimulus -> recovered): grain 0.45 -> 0.460, glow 0.35 ->
# 0.363, vignette 0.55 -> 0.545, softness 0.30 -> 0.322. Cross-talk between the
# four metrics is small and one-directional: grain reads as -0.093 of softness
# (grain IS high-frequency detail, so a sharpen-shaped metric sees it), and
# nothing else leaks past 0.036.
#
# LEFT AT IDENTITY ON PURPOSE, because they cannot be recovered honestly here:
#   denoise -- it is the INVERSE of grain and shares its metric, so on a clean
#              source still there is nothing for it to remove and the high-
#              frequency ratio barely moves. It needs a noisy plate as the base,
#              which this montage is not.
#   fringe  -- rgbashift moves whole pixels, so a per-channel registration
#              measurement is what would recover it; the luma metrics here are
#              blind to it by construction.
# No number is written for either. An unmeasurable field stays at identity.


def fx_chain(**kw) -> str:
    pre, post = _effect_filters(EffectSpec(**kw))
    return ",".join([*pre, *post])


def luma(px: np.ndarray) -> np.ndarray:
    return px @ LUMA


def _hf(y: np.ndarray) -> float:
    """High-frequency energy: RMS of the 4-neighbour Laplacian. Grain adds it."""
    lap = 4.0 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    return float(np.sqrt((lap**2).mean()))


def _grad(y: np.ndarray) -> float:
    """Gradient energy. Blur lowers it, sharpening raises it."""
    return float(np.sqrt((np.diff(y, axis=0) ** 2).mean() + (np.diff(y, axis=1) ** 2).mean()))


def _corner(y: np.ndarray) -> float:
    """Corner brightness / centre brightness. A vignette drives this down."""
    h, w = y.shape
    ch, cw = h // 4, w // 4
    corners = np.concatenate([
        y[:ch, :cw].ravel(), y[:ch, -cw:].ravel(),
        y[-ch:, :cw].ravel(), y[-ch:, -cw:].ravel(),
    ])
    centre = y[h // 2 - ch : h // 2 + ch, w // 2 - cw : w // 2 + cw]
    return float(corners.mean() / max(centre.mean(), 1e-6))


def _bloom(y: np.ndarray) -> float:
    """Upper-tone energy. The glow chain screens blurred highlights back on,
    which lifts everything in the neighbourhood of a highlight."""
    return float(np.maximum(y - 0.5, 0.0).mean())


EFFECT_METRIC = {"grain": _hf, "softness": _grad, "vignette": _corner, "glow": _bloom}

# Calibration ladder per effect. 0.0 is included analytically (ratio 1.0 by
# definition) rather than rendered, because an all-identity EffectSpec produces
# an empty filter chain.
EFFECT_GRID = {
    "grain": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "glow": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "vignette": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # Denser than the rest, and deliberately so: gblur sigma is 6*v, so the
    # gradient ratio collapses inside the first tenth of the range and a coarse
    # ladder inverts it badly (measured: 5 rungs gave a 0.133 round-trip error
    # at v=0.3, 9 rungs give 0.020). No stimulus sits on a rung.
    "softness": [-1.0, -0.6, -0.35, -0.15, 0.0, 0.15, 0.35, 0.6, 1.0],
}


def calibrate_effects(base_png: Path) -> dict:
    """Metric ratio as a function of effect strength, measured on the base still.

    Deliberately measured on the UNGRADED base, once, and then used to recover
    values off GRADED stills. Calibrating per-look would be circular -- the
    ladder would pass through the stimulus point by construction and the
    round-trip error would be meaningless. This way the printed round-trip error
    is a real measure of how much a grade perturbs the recovery.
    """
    base_y = luma(pixels(base_png))
    cal = {}
    for name, grid in EFFECT_GRID.items():
        metric = EFFECT_METRIC[name]
        ref = metric(base_y)
        ratios = []
        for v in grid:
            if v == 0.0:
                ratios.append(1.0)
                continue
            still = WORK / f"_cal_{name}_{v}.png"
            ff(["-i", str(base_png), "-vf", fx_chain(**{name: v}), "-frames:v", "1", str(still)])
            ratios.append(metric(luma(pixels(still))) / ref)
        r = np.array(ratios)
        assert np.all(np.diff(r) > 0) or np.all(np.diff(r) < 0), (
            f"{name} calibration is not monotone: {ratios} -- it cannot be inverted"
        )
        cal[name] = (np.array(grid, dtype=np.float64), r)
    return cal


def recover_effect(name: str, cal: dict, ctrl_y: np.ndarray, fx_y: np.ndarray) -> float:
    """Invert the calibration ladder for one effect. np.interp clamps outside
    the measured range, which saturates rather than extrapolating nonsense."""
    grid, ratios = cal[name]
    metric = EFFECT_METRIC[name]
    r = metric(fx_y) / metric(ctrl_y)
    o = np.argsort(ratios)
    return float(np.interp(r, ratios[o], grid[o]))


def fit_error(spec, base_px: np.ndarray, ref_px: np.ndarray) -> dict:
    """How well the solved spec actually reproduces the reference, in 8-bit code
    values. This is the honesty number: match_reference solves a LINEAR CDL, so
    a reference made with a gamma or curve preset cannot be matched exactly and
    the residual belongs in the corpus where a reader can see it."""
    got = spec.apply(base_px).reshape(-1, 3)
    ref = ref_px.reshape(-1, 3)
    return {
        "rmse_cv": float(np.sqrt(((got - ref) ** 2).mean()) * 255.0),
        "mean_cv": float(np.abs(got.mean(0) - ref.mean(0)).max() * 255.0),
        "std_cv": float(np.abs(got.std(0) - ref.std(0)).max() * 255.0),
    }


def coverage(specs: dict) -> str:
    """Which capability groups the corpus actually exercises. The reason this
    exists: before the hue/exposure/effects solvers the answer was 2 of 6, and
    nothing in the build said so."""
    n = len(specs)
    ident = GradeSpec.identity()
    groups = {
        "tonal split": lambda sp: sp._has_tonal_split(),
        "highlight_rolloff": lambda sp: sp.highlight_rolloff > 1e-9,
        "hue qualifiers": lambda sp: sp.has_hue_qualifiers(),
        "exposure": lambda sp: abs(sp.exposure) > 1e-9,
        "effects": lambda sp: sp.effects != ident.effects,
        "look_mix": lambda sp: sp.look_mix < 1.0,
    }
    return "\n".join(
        f"  {g:20s} {sum(1 for sp in specs.values() if f(sp)):2d}/{n}"
        for g, f in groups.items()
    )


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing base footage: {SRC}")
    WORK.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(exist_ok=True)

    base = WORK / "_base.png"
    ff(["-i", str(SRC), "-vf", BASE_VF, "-frames:v", "1", str(base)])
    neutral = probe_image(str(base))
    base_px = pixels(base)
    cal = calibrate_effects(base)

    specs, rmse, crosstalk = {}, [], {}
    for name, entry in LOOKS.items():
        tags, vf = entry[0], entry[1]
        fx_stim = entry[2] if len(entry) > 2 else {}

        # The colour solve always runs against the effect-FREE still: a vignette
        # or a grain field moves the very moments match_reference reads, and
        # letting it do so would smear a spatial effect into the CDL.
        still = WORK / f"{slug(name)}.png"
        ff(["-i", str(base), "-vf", vf, "-frames:v", "1", str(still)])
        ref = probe_image(str(still))
        ref_px = pixels(still)

        cdl = match_reference(neutral, ref)
        hue = fit_hue_bands(cdl, base_px, ref_px)
        spec = fit_tonal_split(hue, base_px, ref_px)

        err = fit_error(spec, base_px, ref_px)
        err["rmse_cv_cdl_only"] = fit_error(cdl, base_px, ref_px)["rmse_cv"]
        err["rmse_cv_after_hue"] = fit_error(hue, base_px, ref_px)["rmse_cv"]

        prov_extra = {}
        if fx_stim:
            chain = fx_chain(**fx_stim)
            fx_still = WORK / f"{slug(name)}_fx.png"
            ff(["-i", str(still), "-vf", chain, "-frames:v", "1", str(fx_still)])
            ctrl_y, fx_y = luma(ref_px), luma(pixels(fx_still))
            # Recover ALL four, not just the one in the stimulus: the leak into
            # the other three is the honest measure of how separable these
            # metrics are. Only the declared effect is written to the corpus --
            # which effect a still contains is part of the STIMULUS, exactly as
            # the ffmpeg chain is; only its strength is measured.
            all_got = {k: recover_effect(k, cal, ctrl_y, fx_y) for k in EFFECT_METRIC}
            crosstalk[name] = all_got
            got = {k: all_got[k] for k in fx_stim}
            spec = spec.model_copy(update={"effects": EffectSpec(**got)}).sanitize()
            prov_extra["effect_stimulus"] = chain
            prov_extra["effect_roundtrip"] = {
                k: {"stimulus": v, "recovered": got[k], "error": abs(got[k] - v)}
                for k, v in fx_stim.items()
            }

        # EXACT re-parameterisation, last, so nothing downstream re-fits it.
        spec = factor_exposure(spec, base_px)
        specs[name] = spec
        rmse.append((name, err["rmse_cv_cdl_only"], err["rmse_cv_after_hue"], err["rmse_cv"]))

        entry_json = {
            "name": name,
            "tags": tags.split(),
            "provenance": {
                "base": f"{SRC.relative_to(REPO)} -> {BASE_VF}",
                "reference": vf,
                "derived_by": [
                    "match.match_reference(probe_image(base), probe_image(reference))",
                    "build_looks.fit_hue_bands (12-unknown lstsq on the post-CDL residual)",
                    "build_looks.fit_tonal_split (per-band closed form on the residual)",
                    "build_looks.recover_effect (metric ratio inverted through "
                    "calibrate_effects' measured ladder)"
                    if fx_stim else
                    "build_looks.recover_effect: not run, no spatial stimulus "
                    "(effects stay at identity)",
                    "build_looks.factor_exposure (exact slope -> exposure re-parameterisation)",
                ],
                "measured_base": {
                    "mean": neutral.mean.model_dump(),
                    "std": neutral.std.model_dump(),
                    "saturation": neutral.saturation,
                },
                "measured_ref": {
                    "mean": ref.mean.model_dump(),
                    "std": ref.std.model_dump(),
                    "saturation": ref.saturation,
                    "dominant_hue": ref.dominant_hue,
                },
                "fit_error": err,
                **prov_extra,
            },
            "spec": spec.model_dump(),
        }
        out = CORPUS / f"{slug(name)}.json"
        out.write_text(json.dumps(entry_json, indent=2) + "\n")
        s_ = spec.slope
        print(
            f"{name:22s} exp {spec.exposure:+.3f} slope r{s_.r:.3f} g{s_.g:.3f} b{s_.b:.3f}"
            f"  sat {spec.saturation:.3f}"
            f"  RMSE {err['rmse_cv_cdl_only']:5.2f} -> {err['rmse_cv_after_hue']:5.2f}"
            f" -> {err['rmse_cv']:5.2f} cv"
        )

    stale = {p.stem for p in CORPUS.glob("*.json")} - {slug(n) for n in LOOKS}
    for name in stale:
        (CORPUS / f"{name}.json").unlink()
        print(f"removed stale entry {name}")

    a = np.array([[r[1], r[2], r[3]] for r in rmse])
    print(f"\nRMSE median  CDL {np.median(a[:, 0]):.2f} -> +hue {np.median(a[:, 1]):.2f}"
          f" -> +split {np.median(a[:, 2]):.2f} code values")
    if crosstalk:
        print("\neffect cross-talk (recovered value for every metric; only the "
              "stimulus one is written):")
        for name, got in crosstalk.items():
            print(f"  {name:22s} " + "  ".join(f"{k} {v:+.3f}" for k, v in got.items()))
    print(f"\ncoverage over {len(LOOKS)} entries:\n{coverage(specs)}")
    print(f"\n{len(LOOKS)} entries -> {CORPUS.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
