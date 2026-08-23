"""Measured oracles for the grade math.

Every test here MEASURES a value. "No exception raised" is not evidence: every
serious bug this project has shipped (colour space 4x too dark, export using
the wrong grade, 0-byte GIF, dead undo) passed a green suite.

Two failure modes drive the design of this file:

  * atol traps. `L + (x-L)*1.0` and `src + (x-src)*1.0` cost about 1 ulp, so an
    unguarded no-op step still passes an atol=1e-6 assertion while silently
    changing the identity LUT. Hence the bit-for-bit hash gate and the
    np.array_equal assertions -- tolerances are only used where the maths
    genuinely produces a non-zero difference.
  * ordering. Several steps commute *approximately*, so a swapped pair still
    looks plausible. The order tests below pin each swap to a number that only
    comes out right in the documented order.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ragvid.lut import _grid
from ragvid.spec import (
    HUE_CENTERS,
    HUE_CHROMA_GATE,
    HUE_HALFWIDTH,
    HUE_FIELDS,
    LUMA,
    SPLIT_CROSSOVER,
    EffectSpec,
    GradeSpec,
    HueBand,
    RGB,
    _rolloff,
    _smoothstep,
)

# Baseline computed on main, before the expressive-grade work. Hard-coded on
# purpose: it is the gate, so it must not be derived from the code under test.
IDENTITY_SHA256 = "517467be3ba6b7a8afe71a05c847061dc597f0ea92e41b422164b579fbc74291"


def luma(x):
    return np.sum(np.asarray(x) * LUMA, axis=-1)


def chroma(x):
    x = np.asarray(x)
    return x.max(axis=-1) - x.min(axis=-1)


def greys(n=101, lo=0.05, hi=0.95):
    v = np.linspace(lo, hi, n)
    return np.repeat(v[:, None], 3, axis=1)


HUE_WHEEL = np.array([  # pure hues at 0/60/120/180/240/300 degrees
    [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
])


# ---- 1. the hard gate -----------------------------------------------------


def test_identity_lut_is_bit_for_bit_the_baseline():
    """GradeSpec.identity() over the 33^3 grid must hash to the pre-work value.

    Not "close to": identical. Every saved session's .cube is regenerated from
    its spec, so a 1-ulp drift here silently re-grades everything a user ever
    saved.
    """
    grid = _grid(33)
    table = GradeSpec.identity().apply(grid)
    digest = hashlib.sha256(
        np.ascontiguousarray(table.astype(np.float64)).tobytes()
    ).hexdigest()
    assert digest == IDENTITY_SHA256
    assert np.abs(table - grid).max() == 0.0
    assert np.array_equal(table, grid)


@pytest.mark.parametrize("kwargs", [
    {"exposure": 3.0},
    {"look_mix": 0.5},
    {"highlight_rolloff": 0.2},
    {"shadow_tint": RGB(r=0.0, g=0.1, b=0.1)},
    {"highlight_tint": RGB.of(0.05)},
    {"shadow_lift": 0.05},
    {"highlight_lift": -0.05},
    {"hue_red": HueBand(sat=0.8)},
    {"hue_magenta": HueBand(lum=0.02)},
    {"effects": EffectSpec(grain=0.4)},
])
def test_is_identity_covers_every_new_field(kwargs):
    """A field missing from is_identity() makes test_match.py assert nothing."""
    assert GradeSpec.identity().is_identity()
    assert not GradeSpec(**kwargs).is_identity()


def test_every_new_step_is_guarded_bitwise():
    """Each new op, set to its own no-op value, must leave the grid untouched
    BITWISE -- i.e. the step was skipped, not merely applied harmlessly."""
    grid = _grid(17)
    for kwargs in (
        {"exposure": 0.0},
        {"look_mix": 1.0},
        {"highlight_rolloff": 0.0},
        {"shadow_tint": RGB.of(0.0), "highlight_tint": RGB.of(0.0)},
        {"shadow_lift": 0.0, "highlight_lift": 0.0},
        {f: HueBand(sat=1.0, lum=0.0) for f in HUE_FIELDS},
    ):
        out = GradeSpec(**kwargs).apply(grid)
        assert np.array_equal(out, grid), f"{kwargs} is not bitwise a no-op"


def test_unguarded_hue_pass_would_actually_drift():
    """Proof the guard above is load-bearing rather than decorative: the luma
    round trip at sat=1 is NOT the identity in floating point.

    (It happens to be exact on the 33^3 grid's tidy binary fractions, which is
    precisely the trap: a spot check on the LUT grid would say "no drift" while
    real footage drifts on 1 sample in 100.)
    """
    x = np.random.default_rng(0).random((100000, 3))
    L = np.sum(x * LUMA, axis=-1, keepdims=True)
    unguarded = L + (x - L) * 1.0
    moved = int(np.count_nonzero(unguarded != x))
    assert moved > 0, "if this ever becomes 0 the guard argument needs revisiting"
    assert np.abs(unguarded - x).max() < 1e-15  # invisible to atol=1e-6
    # the guarded code path, by contrast, adds nothing: a band at its no-op
    # value is bitwise identical to not mentioning the band at all
    assert np.array_equal(
        GradeSpec(hue_red=HueBand(sat=1.0)).apply(x), GradeSpec().apply(x)
    )


# ---- 2. one op at a time, measured ----------------------------------------


def test_exposure_is_exact_stops_and_nothing_else():
    x = np.array([[0.25, 0.25, 0.25], [0.1, 0.2, 0.3]])
    out = GradeSpec(exposure=1.0).apply(x)
    assert out[0].tolist() == [0.5, 0.5, 0.5]              # exactly one stop
    assert out[1].tolist() == [0.2, 0.4, 0.6]
    # a stop is a pure gain: channel ratios (hence hue) are untouched
    assert np.abs(out[1] / out[1].sum() - x[1] / x[1].sum()).max() < 1e-16
    assert GradeSpec(exposure=-1.0).apply(x)[0].tolist() == [0.125, 0.125, 0.125]


def test_shadow_tint_colours_shadows_only_and_never_brightness():
    """Tints are luma-stripped, so tint and lift are exactly independent."""
    g = greys()
    spec = GradeSpec(shadow_tint=RGB(r=0.0, g=0.03, b=0.03))
    out = spec.apply(g)

    # brightness is untouched, exactly (L is linear; the tint's luma is removed)
    assert np.abs(luma(out) - luma(g)).max() < 1e-15

    dark, light = g[:, 0] < 0.2, g[:, 0] >= 0.5
    # low tones gain measurable chroma, exactly mask * tint spread...
    assert chroma(out[dark]).min() > 0.015
    assert chroma(out[0]) == pytest.approx(0.02916, abs=1e-5)   # v=0.05
    w_s = 1.0 - _smoothstep(np.clip(g[:, 0] / SPLIT_CROSSOVER, 0, 1))
    assert np.abs(chroma(out) - w_s * 0.03).max() < 1e-15
    # ...and everything at or above the crossover is bitwise untouched
    assert np.array_equal(out[light], g[light])
    assert chroma(out[light]).max() == 0.0

    # the sign is right too: a teal tint moves g/b up and r down
    d = out[0] - g[0]
    assert d[0] < 0 < d[1] and d[1] == pytest.approx(d[2], abs=1e-15)


def test_highlight_tint_is_the_mirror_image():
    g = greys()
    out = GradeSpec(highlight_tint=RGB(r=0.04, g=0.0, b=0.0)).apply(g)
    assert np.abs(luma(out) - luma(g)).max() < 1e-15
    dark = g[:, 0] <= 0.5
    assert np.array_equal(out[dark], g[dark])
    assert chroma(out[g[:, 0] > 0.85]).min() > 0.02


def test_lift_moves_brightness_only():
    g = greys()
    out = GradeSpec(shadow_lift=0.05).apply(g)
    d = out - g
    # identical on every channel => pure luma move, zero chroma crosstalk
    assert np.abs(d[:, 0] - d[:, 1]).max() == 0.0
    assert np.abs(d[:, 1] - d[:, 2]).max() == 0.0
    assert np.abs(chroma(out) - chroma(g)).max() == 0.0
    # and the amount is exactly the mask times the lift
    L = g[:, 0]
    w_s = 1.0 - _smoothstep(np.clip(L / SPLIT_CROSSOVER, 0, 1))
    assert np.abs(d[:, 0] - w_s * 0.05).max() < 1e-15
    assert d[L >= 0.5, 0].max() == 0.0


def test_split_masks_are_disjoint_at_the_crossover():
    """Midtones receive exactly zero of either mask, so double-counting is
    structurally impossible rather than merely small."""
    mid = np.array([[0.5, 0.5, 0.5]])
    spec = GradeSpec(shadow_lift=0.2, highlight_lift=-0.2,
                     shadow_tint=RGB.of(0.2), highlight_tint=RGB(r=0.2, g=0, b=0))
    assert np.array_equal(spec.apply(mid), mid)

    L = np.linspace(0, 1, 20001)
    w_s = 1.0 - _smoothstep(np.clip(L / SPLIT_CROSSOVER, 0, 1))
    w_h = _smoothstep(np.clip((L - SPLIT_CROSSOVER) / (1 - SPLIT_CROSSOVER), 0, 1))
    assert (w_s + w_h).max() == pytest.approx(1.0, abs=1e-12)
    assert int(np.count_nonzero((w_s > 0) & (w_h > 0))) == 0


def test_hue_band_moves_its_own_band_and_leaves_the_others_alone():
    spec = GradeSpec(hue_red=HueBand(sat=0.5))
    out = spec.apply(HUE_WHEEL)
    # red halves its chroma, exactly
    assert chroma(out[0]) == pytest.approx(0.5, abs=1e-12)
    # the other five pure hues sit at weight 0 for the red band
    assert np.abs(out[1:] - HUE_WHEEL[1:]).max() < 1e-15
    # ...and the move is luma-preserving: `lum` is the only brightness axis
    assert abs(luma(out[0]) - luma(HUE_WHEEL[0])) < 1e-15


def test_hue_band_lum_moves_brightness_only():
    wheel = HUE_WHEEL * 0.6 + 0.1   # same hues, but with headroom to move into
    out = GradeSpec(hue_green=HueBand(lum=0.05)).apply(wheel)
    d = out[2] - wheel[2]
    assert luma(out[2]) - luma(wheel[2]) == pytest.approx(0.05, abs=1e-12)
    assert np.abs(d - d[0]).max() < 1e-15                  # equal on all channels
    assert np.abs(chroma(out[2]) - chroma(wheel[2])) < 1e-15   # saturation untouched
    assert np.abs(out[[0, 1, 3, 4, 5]] - wheel[[0, 1, 3, 4, 5]]).max() < 1e-15


def test_chroma_gate_protects_the_neutral_axis():
    """Without the gate a qualifier tints every grey and the grade reads as a
    cast. The gate value is smoothstep(c/0.15), measured exactly."""
    spec = GradeSpec(hue_red=HueBand(sat=0.5))
    grey = np.array([[0.5, 0.5, 0.5]])
    assert np.abs(spec.apply(grey) - grey).max() < 1e-15   # pure grey: untouched

    for c in (0.02, 0.05, 0.10, 0.30):
        px = np.array([[0.5 + c / 2, 0.5 - c / 2, 0.5 - c / 2]])  # red-ish hue
        got = chroma(spec.apply(px))[0] / chroma(px)[0]
        gate = _smoothstep(min(c / HUE_CHROMA_GATE, 1.0))
        assert got == pytest.approx(1.0 + gate * (0.5 - 1.0), abs=1e-12)
    # and the gate really does attenuate: near-neutral moves ~4% where a
    # saturated pixel moves 50%
    near = np.array([[0.51, 0.49, 0.49]])
    assert 1.0 - chroma(spec.apply(near))[0] / chroma(near)[0] < 0.05


# ---- 3. highlight rolloff -------------------------------------------------


def _pinned_fraction(rolloff: float, slope: float = 1.6, n: int = 4096) -> float:
    ramp = np.repeat(np.linspace(0.0, 1.0, n)[:, None], 3, axis=1)
    out = GradeSpec(slope=RGB.of(slope), highlight_rolloff=rolloff).apply(ramp)
    return float(np.count_nonzero(out[:, 0] == 1.0)) / n


def test_rolloff_zero_is_todays_hard_clip_exactly():
    """37.5% of a 4096-step ramp welded to pure white at slope=1.6. This is the
    behaviour rolloff exists to fix, and rolloff=0 must preserve it byte for
    byte -- otherwise every existing saved grade shifts."""
    assert _pinned_fraction(0.0) == 1536 / 4096 == 0.375
    assert _pinned_fraction(0.0, slope=1.3) == pytest.approx(0.2307, abs=5e-4)


def test_rolloff_recovers_the_clipped_highlights():
    pinned = {r: _pinned_fraction(r) for r in (0.0, 0.3, 0.6, 1.0)}
    assert pinned[0.0] == 0.375
    assert pinned[1.0] == 0.0
    # monotonically fewer welded pixels as the shoulder opens
    vals = [pinned[r] for r in (0.0, 0.3, 0.6, 1.0)]
    assert vals == sorted(vals, reverse=True)
    assert pinned[0.3] < 0.15, pinned

    # the recovered range is real detail, not a flat shelf: above the old clip
    # point the ramp is still strictly increasing
    ramp = np.repeat(np.linspace(0.0, 1.0, 4096)[:, None], 3, axis=1)
    out = GradeSpec(slope=RGB.of(1.6), highlight_rolloff=0.6).apply(ramp)[:, 0]
    top = out[ramp[:, 0] >= 0.625]
    assert np.all(np.diff(top) > 0)
    assert len(np.unique(top)) == len(top)


@pytest.mark.parametrize("r", [0.1, 0.3, 0.6, 1.0])
def test_shoulder_is_strictly_monotone_and_caps_at_one(r):
    """A non-monotone curve baked into a .cube INVERTS highlights."""
    x = np.linspace(0.0, 4.0, 400001)
    y = _rolloff(x, r)
    assert np.all(np.diff(y) >= 0.0)
    assert y.max() == pytest.approx(1.0, abs=1e-9)
    assert y.max() <= 1.0 + 1e-12
    # the unavoidable white loss: no monotone shoulder can map 1 -> 1
    expect = {0.1: 0.976, 0.3: 0.928, 0.6: 0.856, 1.0: 0.760}[r]
    assert _rolloff(np.array([1.0]), r)[0] == pytest.approx(expect, abs=5e-4)
    # C1 at the knee: slope 1.0 coming in
    knee = 1.0 - 0.5 * r
    h = 1e-6
    slope = (_rolloff(np.array([knee + h]), r)[0] - knee) / h
    assert slope == pytest.approx(1.0, abs=1e-4)


# ---- 4. look_mix ----------------------------------------------------------

LOOK = dict(
    slope=RGB(r=1.15, g=1.0, b=0.9), offset=RGB(r=-0.01, g=0.0, b=0.02),
    power=RGB.of(0.95), saturation=1.2, temperature=800.0, tint=-0.2,
    contrast=0.35, exposure=0.3, highlight_rolloff=0.4,
    shadow_tint=RGB(r=0.0, g=0.04, b=0.06), highlight_tint=RGB(r=0.05, g=0.01, b=0.0),
    shadow_lift=0.02, highlight_lift=-0.03,
    hue_red=HueBand(sat=0.8, lum=0.01), hue_cyan=HueBand(sat=1.25),
)


def test_look_mix_half_lands_halfway():
    grid = _grid(33)
    full = GradeSpec(**LOOK, look_mix=1.0).apply(grid)
    half = GradeSpec(**LOOK, look_mix=0.5).apply(grid)
    expected = grid + (full - grid) * 0.5
    assert np.abs(half - expected).max() < 1e-12
    # and it is genuinely a blend, not a no-op in either direction
    assert np.abs(full - grid).max() > 0.05
    assert np.abs(half - grid).max() > 0.02
    assert np.array_equal(GradeSpec(**LOOK, look_mix=0.0).apply(grid), np.clip(grid, 0, 1))


def test_look_mix_one_is_bit_for_bit_the_unmixed_result():
    """The lerp costs ~1 ulp, so this catches a missing guard that atol would
    wave through."""
    grid = _grid(33)
    assert np.array_equal(
        GradeSpec(**LOOK, look_mix=1.0).apply(grid), GradeSpec(**LOOK).apply(grid)
    )
    assert np.array_equal(GradeSpec(look_mix=1.0).apply(grid), grid)

    # proof the guard matters: the unguarded lerp does move bits
    out = GradeSpec(**LOOK).apply(grid)
    s = np.clip(grid, 0.0, 1.0)
    unguarded = np.clip(s + (out - s) * 1.0, 0.0, 1.0)
    assert int(np.count_nonzero(unguarded != out)) > 0


# ---- 5. evaluation order --------------------------------------------------
# Each of these is a swap that "looks fine" and produces a different number.


def test_exposure_runs_before_the_cdl_so_offset_stays_absolute():
    x = np.array([[0.2, 0.2, 0.2]])
    out = GradeSpec(exposure=1.0, offset=RGB.of(0.1)).apply(x)[0, 0]
    assert out == pytest.approx(0.5, abs=1e-15)   # (0.2*2) + 0.1
    assert out != pytest.approx(0.6, abs=1e-3)    # (0.2+0.1)*2, the swapped order


def test_hue_qualifiers_run_before_the_tonal_split():
    """If the split tinted first, "teal shadows" would make the cyan qualifier
    fire on every shadow in the frame -- a feedback loop."""
    teal = RGB(r=0.0, g=0.05, b=0.05)
    split_only = GradeSpec(shadow_tint=teal)
    both = GradeSpec(shadow_tint=teal, hue_cyan=HueBand(sat=0.0))
    g = greys(51, 0.05, 0.45)  # shadows only
    # the qualifier cannot see the tint the split has not injected yet
    assert np.abs(both.apply(g) - split_only.apply(g)).max() < 1e-15
    # sanity: that qualifier is not inert -- it flattens a real cyan
    cyan = np.array([[0.2, 0.6, 0.6]])
    assert chroma(both.apply(cyan))[0] < 0.02 < chroma(cyan)[0]


def test_rolloff_runs_before_contrast_not_at_the_end():
    """The clip that destroys highlights is the one INSIDE the contrast step.
    A shoulder only at the final return would leave it untouched."""
    ramp = np.repeat(np.linspace(0.0, 1.0, 4096)[:, None], 3, axis=1)
    spec = GradeSpec(slope=RGB.of(1.6), contrast=0.5, highlight_rolloff=0.6)
    out = spec.apply(ramp)[:, 0]
    assert np.count_nonzero(out == 1.0) == 0
    assert np.all(np.diff(out) > 0)          # injective: no welded shelf
    # a shoulder applied after the contrast clip would collapse the top 37.5%
    # to a single value; measure that it did not
    assert len(np.unique(out[ramp[:, 0] >= 0.625])) == int(np.count_nonzero(ramp[:, 0] >= 0.625))


def test_saturation_before_rolloff_catches_what_saturation_blows_out():
    """saturation > 1 can itself drive a channel above 1, which is why the
    shoulder sits after it."""
    px = np.array([[0.95, 0.5, 0.5]])
    hard = GradeSpec(saturation=1.8).apply(px)[0, 0]
    soft = GradeSpec(saturation=1.8, highlight_rolloff=0.6).apply(px)[0, 0]
    assert hard == 1.0                       # welded by the clip
    assert soft < 0.95                       # the shoulder caught it
    assert GradeSpec(saturation=1.8).apply(px)[0, 1] < 0.5


# ---- 6a. partition of unity ----------------------------------------------


def _band_weights(h_deg: np.ndarray) -> np.ndarray:
    d = ((h_deg[..., None] - HUE_CENTERS + 180.0) % 360.0) - 180.0
    return _smoothstep(np.clip(1.0 - np.abs(d) / HUE_HALFWIDTH, 0.0, 1.0))


def test_hue_band_weights_are_an_exact_partition_of_unity():
    h = np.linspace(-720.0, 1080.0, 200001)   # well past one wrap in both directions
    w = _band_weights(h)
    err = float(np.abs(w.sum(axis=-1) - 1.0).max())
    assert err < 1e-12, err
    assert int((w > 0).sum(axis=-1).max()) <= 2
    assert w.min() >= 0.0


def test_all_bands_equal_reduces_to_a_plain_saturation_move():
    """Functional consequence of sum(w)==1: with every band set the same, the
    qualifier pass is exactly a (gated) global saturation, no hue structure."""
    sat = 1.4
    spec = GradeSpec(**{f: HueBand(sat=sat) for f in HUE_FIELDS})
    x = HUE_WHEEL  # chroma 1.0 everywhere => gate == 1
    out = spec.apply(x)
    L = np.sum(x * LUMA, axis=-1, keepdims=True)
    assert np.abs(out - np.clip(L + (x - L) * sat, 0, 1)).max() < 1e-12


# ---- 6b. nothing non-finite ever reaches the LUT --------------------------

ADVERSARIAL = [
    dict(offset=RGB.of(-0.9), power=RGB.of(0.4)),                   # neg^fractional
    dict(offset=RGB(r=-1.0, g=0.5, b=-0.3), power=RGB(r=0.05, g=7.5, b=0.5)),
    dict(power=RGB.of(0.0)),                                        # division blowup
    dict(power=RGB.of(-3.0), slope=RGB.of(-2.0)),
    dict(slope=RGB.of(1e12), exposure=99.0),
    dict(slope=RGB.of(0.0), saturation=-4.0),
    dict(saturation=float("inf"), contrast=-2.0),
    dict(exposure=-40.0, highlight_rolloff=5.0, look_mix=-1.0),
    dict(highlight_rolloff=1.0, highlight_lift=9.0, shadow_lift=-9.0),
    dict(shadow_tint=RGB.of(9.0), highlight_tint=RGB.of(-9.0)),
    dict(hue_red=HueBand(sat=-4.0, lum=7.0), hue_cyan=HueBand(sat=1e9, lum=-1e9)),
    dict(slope=RGB(r=float("nan"), g=1.0, b=float("inf")), pivot=0.0),
    dict(pivot=float("nan"), contrast=1.0, power=RGB.of(float("nan"))),
    dict(**LOOK),
]


@pytest.mark.parametrize("kwargs", ADVERSARIAL, ids=range(len(ADVERSARIAL)))
def test_sanitized_specs_never_produce_nan_or_out_of_range(kwargs):
    spec = GradeSpec(**kwargs).sanitize()
    out = spec.apply(_grid(17))
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_random_sanitized_specs_stay_finite():
    """Fuzz across every field at once -- the adversarial list above is chosen
    by hand and cannot cover the combinations."""
    rng = np.random.default_rng(20260820)
    grid = _grid(9)
    worst = 0.0
    for _ in range(300):
        def w():  # wild: heavy tails plus the occasional non-finite
            v = rng.standard_cauchy() * 3.0
            return float(rng.choice([v, np.inf, -np.inf, np.nan], p=[0.94, 0.02, 0.02, 0.02]))
        spec = GradeSpec(
            slope=RGB(r=w(), g=w(), b=w()), offset=RGB(r=w(), g=w(), b=w()),
            power=RGB(r=w(), g=w(), b=w()), saturation=w(), temperature=w() * 1000,
            tint=w(), contrast=w(), pivot=w(), exposure=w(), look_mix=w(),
            highlight_rolloff=w(),
            shadow_tint=RGB(r=w(), g=w(), b=w()), highlight_tint=RGB(r=w(), g=w(), b=w()),
            shadow_lift=w(), highlight_lift=w(),
            **{f: HueBand(sat=w(), lum=w()) for f in HUE_FIELDS},
        ).sanitize()
        out = spec.apply(grid)
        assert np.all(np.isfinite(out)), spec.model_dump()
        assert out.min() >= 0.0 and out.max() <= 1.0
        worst = max(worst, float(out.max()))
    assert worst > 0.5  # the fuzz actually reached the top of the range


def test_effects_are_never_part_of_the_colour_transform():
    """A 3D LUT cannot express a blur; apply() must ignore effects entirely."""
    grid = _grid(17)
    loud = EffectSpec(denoise=1.0, glow=1.0, softness=-1.0, grain=1.0,
                      vignette=1.0, fringe=1.0)
    assert np.array_equal(GradeSpec(effects=loud).apply(grid), grid)
    assert np.array_equal(
        GradeSpec(**LOOK, effects=loud).apply(grid), GradeSpec(**LOOK).apply(grid)
    )


def test_look_mix_fades_the_spatial_effects_too():
    """The UI calls look_mix "Strength of the whole look", but effects live in
    the ffmpeg chain, outside apply() -- so at Strength 0% grain, glow and
    vignette used to stay at full force in the preview AND the export."""
    from ragvid.spec import EffectSpec

    fx = EffectSpec(vignette=1.0, grain=0.6, glow=0.5, softness=-0.4)
    assert GradeSpec(effects=fx, look_mix=1.0).render_effects() == fx
    assert GradeSpec(effects=fx, look_mix=0.0).render_effects() == EffectSpec()
    half = GradeSpec(effects=fx, look_mix=0.5).render_effects()
    assert half.vignette == 0.5 and half.grain == 0.3 and half.softness == -0.2


def test_is_identity_sees_every_field_including_pivot():
    assert GradeSpec().is_identity()
    assert not GradeSpec(pivot=0.9).is_identity()


def test_identity_tolerance_matches_the_apply_guard():
    """is_identity() and the exposure guard must agree, or there is a window
    where a spec reports identity but no longer bakes the identity grid."""
    import numpy as np

    from ragvid.lut import _grid

    g = _grid(9)
    for e in (1e-13, 1e-12, 5e-10):
        s = GradeSpec(exposure=e)
        assert s.is_identity() and np.array_equal(s.apply(g), g)


# ---- regions: the container around the spec (roadmap B1) -------------------
#
# A GradeSpec is the currency of ONE correction and cannot say which pixels;
# region.GradeStack is the container that can. These tests live here because
# the stack's evaluation order is spec math -- and because the identity gate
# above is exactly what a container is most likely to break.


def test_a_flat_stack_is_bit_for_bit_its_base():
    """No regions => byte-identical output to before regions existed.

    array_equal, not allclose: a lerp against a mask of 1.0 costs ~1 ulp, so an
    unguarded container would pass an atol test while shifting every saved
    grade -- the same trap the module docstring describes for apply()'s steps.
    """
    from ragvid.region import GradeStack

    img = np.random.default_rng(7).random((16, 24, 3))
    for spec in (GradeSpec.identity(), GradeSpec(contrast=0.4, temperature=-1200.0)):
        assert np.array_equal(GradeStack(base=spec).apply(img), spec.apply(img))


def test_the_identity_stack_still_hashes_to_the_identity_lut():
    """The bit-for-bit gate, through the container this time. If a stack with no
    layers ever stops reproducing this hash, every grade on disk has shifted."""
    from ragvid.region import GradeStack

    grid = _grid(33).reshape(1, -1, 3)   # (h, w, 3): a stack grades images
    table = GradeStack(base=GradeSpec.identity()).apply(grid).reshape(-1, 3)
    assert hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest() == (
        "517467be3ba6b7a8afe71a05c847061dc597f0ea92e41b422164b579fbc74291"
    )


def test_a_full_frame_region_equals_the_same_grade_applied_globally():
    """A mask of 1 everywhere must reproduce the global grade.

    Not bit-for-bit, and deliberately so: the lerp `x + (g-x)*1.0` is the ulp
    the docstring warns about, and paying it is what keeps the mask a plain
    array instead of a special case. Measured max error 1.1e-16 -- under half a
    code value at 53 bits, let alone at 8.
    """
    from ragvid.region import GradeStack, Layer, Region

    img = np.random.default_rng(3).random((12, 20, 3))
    spec = GradeSpec(exposure=-0.4, saturation=1.3, contrast=0.25)
    full = Region(shape="linear", edge="top", extent=9.0, softness=0.0)
    assert full.mask(20, 12).min() == 1.0
    stacked = GradeStack(base=GradeSpec.identity(),
                         layers=[Layer(region=full, spec=spec)]).apply(img)
    assert np.abs(stacked - spec.apply(img)).max() < 1e-15


def test_an_empty_region_changes_nothing_at_all():
    """The other rail: a mask of 0 is bit-exact, because x + (g-x)*0 == x.

    Measured against `base.apply`, not against the source: apply() is only
    bit-exact on the LUT grid (its saturation step is `L + (x-L)*1.0`, which is
    exact when L lands on the value and ~1 ulp off when it does not), so the
    source is the wrong oracle for anything but the grid.
    """
    from ragvid.region import GradeStack, Layer, Region

    img = np.random.default_rng(4).random((8, 8, 3))
    base = GradeSpec(contrast=0.3)
    none = Region(shape="linear", edge="top", extent=-9.0, softness=0.0)
    assert none.mask(8, 8).max() == 0.0
    out = GradeStack(base=base,
                     layers=[Layer(region=none, spec=GradeSpec(exposure=2.0))]).apply(img)
    assert np.array_equal(out, base.apply(img))


def test_the_region_is_where_it_says_it_is():
    """"darken the top" must darken the top and leave the bottom exactly alone."""
    from ragvid.region import GradeStack, Layer, for_target

    img = np.full((90, 60, 3), 0.5)
    stack = GradeStack(base=GradeSpec.identity(),
                       layers=[Layer(region=for_target("top"),
                                     spec=GradeSpec(exposure=-0.35))])
    out = stack.apply(img)
    top, bottom = out[:30].mean(), out[-30:].mean()
    assert top < 0.45, f"the top barely moved: {top:.4f} from 0.5"
    # EXACTLY unchanged, not approximately: the falloff of "top" ends at 0.6 of
    # the frame, so the whole bottom third has a mask of 0 and the lerp there is
    # bit-exact. A region whose name does not survive this is mis-shaped.
    assert np.array_equal(out[-30:], img[-30:]), f"the bottom moved to {bottom:.4f}"


def test_the_soft_edge_is_monotonic_with_no_step():
    """A hard edge bands badly through an 8-bit encode, so the falloff has to be
    a real ramp: monotone, and with no jump big enough to read as a contour.

    The bound is the smoothstep's own maximum slope (1.5) over the ramp's height
    in pixels -- a step edge would be 1.0 in a single row and fail it by 30x.
    """
    from ragvid.region import for_target

    m = for_target("top").mask(40, 200)
    col = m[:, 0]
    assert col[0] == 1.0 and col[-1] == 0.0
    d = np.diff(col)
    assert np.all(d <= 0.0), "the mask must fall monotonically from the top"
    # The ramp spans `softness` (0.4) of the frame, and smoothstep's steepest
    # slope is 1.5, so no single row may move more than 1.5/(0.4*200). A step
    # edge would move 1.0 in one row and miss it by 50x.
    assert np.abs(d).max() < 1.5 / (0.4 * 200) + 1e-12
    # ... and it is a genuine ramp, not a step with rounded corners: 40% of the
    # frame is strictly between the two plateaus.
    assert 0.3 < np.count_nonzero((col > 0.01) & (col < 0.99)) / len(col) < 0.6


def test_overlapping_regions_compose_in_list_order():
    """Two layers over the same pixels compose as lerps, in the order asked for.

    Documented order = evaluation order: layer i grades the result of 0..i-1.
    So the reference here is the composition itself, and swapping the two must
    produce a different (not merely rounded) picture.
    """
    from ragvid.region import GradeStack, Layer, for_target

    img = np.full((80, 80, 3), 0.5)
    a = Layer(region=for_target("top"), spec=GradeSpec(exposure=-0.5))
    b = Layer(region=for_target("left"), spec=GradeSpec(saturation=0.0))
    ab = GradeStack(base=GradeSpec.identity(), layers=[a, b]).apply(img)
    ba = GradeStack(base=GradeSpec.identity(), layers=[b, a]).apply(img)

    # Where only one mask reaches, order cannot matter.
    assert np.abs(ab[5, -5] - ba[5, -5]).max() < 1e-12    # top right: `a` only
    assert np.abs(ab[-5, 5] - ba[-5, 5]).max() < 1e-12    # bottom left: `b` only
    # Explicit reference for the overlap, spelled out the way the docstring says.
    m_a = for_target("top").mask(80, 80)[..., None]
    m_b = for_target("left").mask(80, 80)[..., None]
    want = img + (a.spec.apply(img) - img) * m_a
    want = want + (b.spec.apply(want) - want) * m_b
    assert np.abs(ab - want).max() < 1e-12


def test_a_radial_region_is_an_ellipse_around_its_centre():
    from ragvid.region import Region, for_target

    m = for_target("center").mask(100, 100)
    assert m[50, 50] == 1.0 and m[0, 0] < 0.01
    # Wider than tall by construction (rx 0.6, ry 0.75), so the mask reaches
    # further down the frame than across it.
    assert m[50, 5] < m[5, 50]
    off = Region(shape="radial", cx=0.2, cy=0.2, rx=0.15, ry=0.15, softness=0.4)
    o = off.mask(100, 100)
    assert o[20, 20] == 1.0 and o[80, 80] == 0.0
    # Symmetric about the centre it was given, and falling away from it. Column
    # 10 and column 29 are the pair equidistant from x = 0.2 once pixel centres
    # are accounted for (10.5/100 and 29.5/100).
    assert o[20, 10] == pytest.approx(o[20, 29]) and o[10, 20] == pytest.approx(o[29, 20])
    assert o[20, 20] == 1.0 > o[20, 30] > o[20, 33] > o[20, 36] == 0.0


def test_invert_swaps_inside_and_outside_exactly():
    from ragvid.region import for_target

    inside = for_target("center").mask(60, 40)
    outside = for_target("edges").mask(60, 40)
    assert np.array_equal(inside + outside, np.ones_like(inside))


def test_a_mask_is_resolution_independent():
    """The same Region has to mask a 4K export and a 480p preview identically,
    or the frame someone approved is not the frame they get."""
    from ragvid.region import for_target

    def half_crossing(h: int) -> float:
        col = for_target("top").mask(4, h)[:, 0]
        return float(np.argmax(col < 0.5)) / h

    # The 50% point, as a fraction of the frame. One row of the coarser mask is
    # 1/24, and the two must agree inside that.
    assert abs(half_crossing(24) - half_crossing(240)) < 1 / 24


def test_a_stack_refuses_a_flat_array_of_pixels():
    """GradeSpec.apply takes any (..., 3) because a colour map does not care how
    the pixels are arranged. A region does -- that is the whole point."""
    from ragvid.region import GradeStack

    with pytest.raises(ValueError):
        GradeStack(base=GradeSpec.identity()).apply(np.zeros((10, 3)))
