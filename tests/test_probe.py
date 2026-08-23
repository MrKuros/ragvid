"""Statistics are checked against values derived by hand, not against a golden
run of the code under test."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from PIL import Image

from ragvid.probe import (
    ClipStats,
    linear_to_srgb,
    probe_image,
    probe_video,
    srgb_to_linear,
)

SAMPLE = "assets/sample.mp4"


# --- reference implementations, written independently of probe.py ----------


def ref_disp(v8: int) -> float:
    """Stats live in display space, so the expected value is just the code value."""
    return v8 / 255.0


def ref_sat(rgb8) -> float:
    d = [ref_disp(v) for v in rgb8]
    hi, lo = max(d), min(d)
    return (hi - lo) / max(hi, 1e-8)


def solid(color, size=(16, 12)) -> Image.Image:
    return Image.new("RGB", size, tuple(color))


# --- transfer function ------------------------------------------------------


def test_srgb_known_points():
    # Published sRGB values; a naive 2.2 power gets 0.5 -> 0.2176 and fails.
    assert srgb_to_linear(0.0) == pytest.approx(0.0)
    assert srgb_to_linear(1.0) == pytest.approx(1.0)
    assert srgb_to_linear(0.5) == pytest.approx(0.21404114, abs=1e-7)
    assert srgb_to_linear(0.04045) == pytest.approx(0.0031308, abs=1e-7)
    assert srgb_to_linear(0.02) == pytest.approx(0.02 / 12.92)  # linear segment


def test_srgb_roundtrip():
    x = np.linspace(0.0, 1.0, 4097)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-12)


def test_srgb_monotonic_and_bounded():
    x = np.linspace(0.0, 1.0, 1000)
    y = srgb_to_linear(x)
    assert np.all(np.diff(y) > 0)
    assert y.min() >= 0.0 and y.max() <= 1.0


# --- probe_image ------------------------------------------------------------


def test_solid_color_matches_analytic(tmp_path):
    c = (200, 120, 40)
    p = tmp_path / "solid.png"
    solid(c).save(p)
    s = probe_image(str(p))

    assert s.mean.r == pytest.approx(ref_disp(c[0]))
    assert s.mean.g == pytest.approx(ref_disp(c[1]))
    assert s.mean.b == pytest.approx(ref_disp(c[2]))
    assert (s.std.r, s.std.g, s.std.b) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert s.saturation == pytest.approx(ref_sat(c))
    assert (s.width, s.height, s.frames_sampled, s.duration) == (16, 12, 1, 0.0)


def test_grey_has_zero_saturation(tmp_path):
    p = tmp_path / "grey.png"
    solid((128, 128, 128)).save(p)
    assert probe_image(str(p)).saturation == pytest.approx(0.0)


def test_two_tone_std_is_half_the_gap(tmp_path):
    """Half black, half a known color: mean is the midpoint, std the half-gap."""
    a, b = (0, 0, 0), (255, 180, 90)
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    arr[:, 10:] = b
    p = tmp_path / "two.png"
    Image.fromarray(arr).save(p)
    s = probe_image(str(p))

    for ch, name in enumerate("rgb"):
        hi = ref_disp(b[ch])
        assert getattr(s.mean, name) == pytest.approx(hi / 2)
        assert getattr(s.std, name) == pytest.approx(hi / 2)


def test_stats_are_display_space_not_linear(tmp_path):
    """Pins the space. match.py fits a display-space transform to these moments,
    so a linearized mean here silently produces a wrong grade."""
    p = tmp_path / "grey.png"
    solid((128, 128, 128)).save(p)
    m = probe_image(str(p)).mean
    assert m.r == pytest.approx(128 / 255.0)
    # The linear-light answer would have been srgb_to_linear(0.502) = 0.216.
    assert m.r > 2 * srgb_to_linear(128 / 255.0)


def test_greyscale_and_rgba_inputs(tmp_path):
    g = tmp_path / "g.png"
    Image.new("L", (8, 8), 128).save(g)
    assert probe_image(str(g)).mean.r == pytest.approx(ref_disp(128))

    a = tmp_path / "a.png"
    Image.new("RGBA", (8, 8), (10, 20, 30, 0)).save(a)
    assert probe_image(str(a)).mean.g == pytest.approx(ref_disp(20))


# --- probe_video ------------------------------------------------------------


def make_video(path, frames, fps=1):
    """Encode uint8 RGB frames losslessly so decoded values match the source."""
    d = path.parent / "src"
    d.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(d / f"f{i:03d}.png")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
         "-i", str(d / "f%03d.png"), "-c:v", "libx264rgb", "-qp", "0",
         "-pix_fmt", "rgb24", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def frame(color, size=(64, 48)):
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def test_median_rejects_outlier_frame(tmp_path):
    """One blown-out white frame among nine grey ones must not move the mean.

    This is the whole reason for sampling >1 frame and for taking a median.
    """
    frames = [frame((128, 128, 128)) for _ in range(9)]
    frames.insert(4, frame((255, 255, 255)))
    s = probe_video(make_video(tmp_path / "v.mp4", frames), n_frames=10)

    assert s.frames_sampled == 10
    # Median -> the grey frame's value. A mean would land near 0.29.
    assert s.mean.r == pytest.approx(ref_disp(128), abs=2e-3)
    assert s.std.r == pytest.approx(0.0, abs=2e-3)


def test_video_stats_match_still(tmp_path):
    c = (200, 120, 40)
    frames = [frame(c) for _ in range(6)]
    s = probe_video(make_video(tmp_path / "v.mp4", frames), n_frames=4)

    assert s.frames_sampled == 4
    assert s.mean.r == pytest.approx(ref_disp(c[0]), abs=2e-3)
    assert s.mean.b == pytest.approx(ref_disp(c[2]), abs=2e-3)
    assert s.saturation == pytest.approx(ref_sat(c), abs=5e-3)


def test_reports_source_dimensions_not_analysis_size(tmp_path):
    """Frames are downscaled for speed; the reported size is the real one."""
    s = probe_video(make_video(tmp_path / "v.mp4", [frame((50, 50, 50), (700, 400))] * 3), n_frames=2)
    assert (s.width, s.height) == (700, 400)


def test_n_frames_is_respected(tmp_path):
    path = make_video(tmp_path / "v.mp4", [frame((90, 90, 90)) for _ in range(8)])
    assert probe_video(path, n_frames=1).frames_sampled == 1
    assert probe_video(path, n_frames=3).frames_sampled == 3


def test_real_sample_clip():
    s = probe_video(SAMPLE, n_frames=5)
    assert isinstance(s, ClipStats)
    assert (s.width, s.height) == (640, 360)
    assert s.duration == pytest.approx(4.0, abs=0.2)
    assert s.frames_sampled == 5
    assert 0.0 <= s.saturation <= 1.0
    for name in "rgb":
        assert 0.0 <= getattr(s.mean, name) <= 1.0


def test_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        probe_video(str(tmp_path / "nope.mp4"))


# --- 16-bit sampling (roadmap A2) -------------------------------------------
#
# probe.py samples rgb48le rather than 8-bit PNG. The depth is only visible
# through a non-linear transform (see the module docstring's log measurements),
# but getting the SCALE wrong would be visible everywhere, so that is what these
# pin: swscale expands an 8-bit source by left shift, so full white arrives as
# 65280 and not 65535, and dividing by the wrong one puts a 0.4% gain error on
# every statistic in the project.


def test_grab_returns_16_bit_frames(tmp_path):
    from ragvid.probe import _grab

    path = make_video(tmp_path / "v.mp4", [frame((10, 200, 30))] * 2)
    f = _grab(path, 0.0)
    assert f.dtype == np.uint16 and f.shape[1:] == (256, 3)
    assert f.max() > 255  # not an 8-bit array that merely got widened


def test_sampled_code_values_survive_the_16_bit_expansion(tmp_path):
    """A lossless 0-255 ramp must come back as the code values that went in."""
    from ragvid.probe import _grab, _unit

    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (64, 1, 3))
    path = make_video(tmp_path / "ramp.mp4", [ramp] * 2)
    v = _unit(_grab(path, 0.0)).reshape(64, 256, 3)

    assert np.abs(v[0, :, 0] - np.arange(256) / 255.0).max() < 0.5 / 255.0
    # The two ends are the ones a wrong divisor moves furthest.
    assert v[..., 0].min() == pytest.approx(0.0, abs=1e-4)
    assert v[..., 0].max() == pytest.approx(1.0, abs=1e-4)


def test_pure_white_still_reads_as_clipped(tmp_path):
    """/65535 would report white as 0.9961 and clipped_high as 0.0."""
    s = probe_video(make_video(tmp_path / "w.mp4", [frame((255, 255, 255))] * 3), n_frames=2)
    assert s.mean.r == pytest.approx(1.0, abs=1e-4)
    assert s.clipped_high == pytest.approx(1.0)

    s = probe_video(make_video(tmp_path / "b.mp4", [frame((0, 0, 0))] * 3), n_frames=2)
    assert s.mean.r == pytest.approx(0.0, abs=1e-4)
    assert s.crushed_low == pytest.approx(1.0)


# --- hue_strength (roadmap A5) ----------------------------------------------


def test_hue_strength_is_zero_on_grey_and_equals_chroma_on_one_colour(tmp_path):
    p = tmp_path / "grey.png"
    solid((128, 128, 128)).save(p)
    assert probe_image(str(p)).hue_strength == pytest.approx(0.0, abs=1e-9)

    c = (200, 120, 40)
    p = tmp_path / "c.png"
    solid(c).save(p)
    s = probe_image(str(p))
    # One hue everywhere: the resultant length IS the mean absolute chroma.
    assert s.hue_strength == pytest.approx((c[0] - c[2]) / 255.0, abs=1e-6)


def test_hue_strength_collapses_where_saturation_cannot(tmp_path):
    """Half orange, half the opposite hue. `dominant_hue` is meaningless here
    and `saturation` -- which compiler.py substitutes for confidence -- does not
    notice at all, because HSV saturation is per pixel and both halves are
    saturated. Only the resultant length says the mean hue means nothing."""
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    arr[:, :10] = (200, 120, 40)   # orange, ~30 deg
    arr[:, 10:] = (40, 120, 200)   # its opposite, ~210 deg
    p = tmp_path / "opposed.png"
    Image.fromarray(arr).save(p)
    s = probe_image(str(p))

    assert s.saturation > 0.5          # both halves are saturated pixels
    assert s.hue_strength < 0.01       # ...and the mean hue carries no information


# --- cut detection (roadmap A7) ---------------------------------------------
#
# Measurement only: nothing grades differently because of `cuts`. The false
# positives asserted below are the ones documented at _CUT_TV; they are asserted
# rather than described so that a future detector cannot quietly reintroduce a
# worse one.

_H, _W = 48, 64


def _plate(seed, lo, hi, tint, width=_W):
    """A deterministic textured frame, in float [0, 1]."""
    r = np.random.default_rng(seed)
    y = np.linspace(0, 1, _H)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    g = lo + (hi - lo) * (0.5 + 0.5 * np.sin(6 * x + 2 * y + r.uniform(0, 6))) * (0.6 + 0.4 * y)
    g = g + r.normal(0.0, 0.02, (_H, width))
    return np.clip(np.stack([g * t for t in tint], -1), 0.0, 1.0)


def _u8(img):
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


SHOT_A = _plate(1, 0.10, 0.75, (1.00, 0.95, 0.85))
SHOT_B = _plate(2, 0.02, 0.30, (0.70, 0.85, 1.00))


def _shot(plate, n, start=0):
    """`n` frames of one continuous take: a slow drift, so no two are equal."""
    return [_u8(np.roll(plate, (0, start + i), axis=(0, 1)) * (1 + 0.004 * i)) for i in range(n)]


def test_a_splice_between_two_sources_is_detected(tmp_path):
    frames = _shot(SHOT_A, 20) + _shot(SHOT_B, 20)
    s = probe_video(make_video(tmp_path / "cut.mp4", frames, fps=10), n_frames=10)
    assert s.cuts == 1


def test_one_continuous_shot_is_not_a_cut(tmp_path):
    s = probe_video(make_video(tmp_path / "one.mp4", _shot(SHOT_A, 40), fps=10), n_frames=10)
    assert s.cuts == 0


def test_a_whip_pan_is_not_a_cut(tmp_path):
    """The case a frame-difference detector gets wrong: the plate is six screens
    wide and the window crosses all of it, so between adjacent SAMPLES not one
    pixel is shared -- and the luma histogram barely moves."""
    wide = np.concatenate([_plate(3 + k, 0.08, 0.80, (1.0, 0.9, 0.8)) for k in range(6)], axis=1)
    step = (wide.shape[1] - _W) / 39.0
    frames = [_u8(wide[:, int(i * step): int(i * step) + _W]) for i in range(40)]
    s = probe_video(make_video(tmp_path / "pan.mp4", frames, fps=10), n_frames=10)
    assert s.cuts == 0


def test_a_hard_flash_is_a_known_false_positive(tmp_path):
    """Documented at _CUT_TV, and not fixable at one sample per second: a sample
    that lands on a blown frame differs from both its neighbours, so it is
    counted twice. The median still protects every OTHER statistic from it."""
    frames = _shot(SHOT_A, 40)
    frames[20:24] = [np.full((_H, _W, 3), 252, np.uint8)] * 4
    s = probe_video(make_video(tmp_path / "flash.mp4", frames, fps=10), n_frames=10)
    assert s.cuts == 2
    assert s.mean.r == pytest.approx(probe_video(
        make_video(tmp_path / "clean.mp4", _shot(SHOT_A, 40), fps=10), n_frames=10).mean.r, abs=0.02)


def test_a_still_and_the_real_sample_report_no_cuts(tmp_path):
    p = tmp_path / "solid.png"
    solid((100, 100, 100)).save(p)
    assert probe_image(str(p)).cuts == 0
    assert probe_video(SAMPLE, n_frames=8).cuts == 0


def test_16_bit_sampling_beats_8_bit_on_log_footage(tmp_path):
    """The measurement behind A2, at test size.

    Reconstructs the old 8-bit PNG path and runs both over an underexposed
    S-Log3 clip through logspace's own conversion .cube -- the case where a
    non-linear transform runs before the statistics. The full-size version of
    this measurement is quoted in probe.py's docstring.
    """
    import io

    from ragvid.logspace import bake_conversion, lin_to_log
    from ragvid.probe import (
        _analysis_vf,
        _grab,
        _run,
        _stats_from_frames,
        _unit,
        ffmpeg,
    )

    def grab8(path, t, lut):
        png = _run([ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
                    "-frames:v", "1", "-vf", _analysis_vf(lut),
                    "-f", "image2pipe", "-c:v", "png", "-"])
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    # Scene linear 0.0006..0.32 -- everything below mid grey, i.e. the band a
    # log curve spends its code values on and 8 bits cannot hold.
    y = np.linspace(0, 1, 216)[:, None]
    x = np.linspace(0, 1, 384)[None, :]
    lin = 0.0008 * (375.0 ** (y * 0.6 + x * 0.4))
    log = np.clip(lin_to_log("slog3", np.stack([lin * 1.06, lin, lin * 0.85], -1)), 0, 1)

    # 10-bit source: an 8-bit file could not carry the shadow detail either.
    raw = tmp_path / "log.mkv"
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
         "-s", "384x216", "-framerate", "2", "-i", "-", "-c:v", "ffv1",
         "-pix_fmt", "gbrp10le", str(raw)], stdin=subprocess.PIPE)
    for _ in range(4):
        p.stdin.write(np.round(log * 65535).astype("<u2").tobytes())
    p.stdin.close()
    assert p.wait() == 0

    cube = bake_conversion("slog3", str(tmp_path / "slog3.cube"))
    hi = _stats_from_frames([_grab(str(raw), 0.0, cube)], 384, 216, 2.0)
    lo = _stats_from_frames([grab8(str(raw), 0.0, cube)], 384, 216, 2.0)

    # 8 bits cannot represent the converted shadows, so it reports blacks that
    # are not there and a floor that has snapped onto the 1/255 grid.
    assert lo.crushed_low > 1.5 * hi.crushed_low
    assert lo.p1.r == pytest.approx(round(lo.p1.r * 255) / 255.0, abs=1e-9)
    assert hi.p1.r > lo.p1.r

    def levels(f):
        return len(np.unique(_unit(f) @ np.array([0.2126, 0.7152, 0.0722])))

    assert levels(_grab(str(raw), 0.0, cube)) > 10 * levels(grab8(str(raw), 0.0, cube))
