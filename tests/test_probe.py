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
