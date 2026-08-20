import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ragvid import render

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "assets" / "sample.mp4")
W, H = 640, 360


def identity_cube(path: Path, size: int = 17) -> str:
    """A .cube that must be a no-op. Written here rather than imported from
    ragvid.lut so this file tests render.py and nothing else."""
    g = np.linspace(0.0, 1.0, size)
    lines = [f"LUT_3D_SIZE {size}", ""]
    # .cube varies red fastest
    for b in g:
        for gg in g:
            for r in g:
                lines.append(f"{r:.6f} {gg:.6f} {b:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def ffprobe(video: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-of", "json", video],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


# ---- escaping -------------------------------------------------------------

def test_escape_path_escapes_specials():
    assert render.escape_path("/a/b.cube") == "/a/b.cube"
    assert render.escape_path("/a b,c.cube") == r"/a b\,c.cube"
    assert render.escape_path("/a:b.cube") == r"/a\\:b.cube"


@pytest.mark.parametrize("name", [
    "plain.cube",
    "with space.cube",
    "wi,th comma.cube",
    "co:lon.cube",
    "br[ack]et.cube",
])
def test_preview_with_awkward_cube_filename(tmp_path, name):
    """The escaping has to survive ffmpeg's real filtergraph parser."""
    cube = identity_cube(tmp_path / name)
    out = str(tmp_path / "sheet.png")
    render.render_preview(SAMPLE, cube, out, n_frames=1)
    assert Image.open(out).size == (W, H)


# ---- preview --------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3, 5])
def test_preview_is_n_frames_wide(tmp_path, n):
    out = str(tmp_path / f"sheet{n}.png")
    assert render.render_preview(SAMPLE, None, out, n_frames=n) == out
    assert Image.open(out).size == (W * n, H)


def test_preview_frames_are_distinct(tmp_path):
    """Evenly spaced, not three copies of frame 0."""
    out = tmp_path / "sheet.png"
    render.render_preview(SAMPLE, None, str(out), n_frames=3)
    a = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    tiles = [a[:, i * W:(i + 1) * W] for i in range(3)]
    assert np.abs(tiles[0] - tiles[1]).mean() > 1.0
    assert np.abs(tiles[1] - tiles[2]).mean() > 1.0


def test_identity_lut_preview_is_near_identical(tmp_path):
    cube = identity_cube(tmp_path / "identity.cube")
    plain, graded = tmp_path / "plain.png", tmp_path / "graded.png"
    render.render_preview(SAMPLE, None, str(plain), n_frames=3)
    render.render_preview(SAMPLE, cube, str(graded), n_frames=3)
    a = np.asarray(Image.open(plain).convert("RGB"), dtype=float)
    b = np.asarray(Image.open(graded).convert("RGB"), dtype=float)
    assert a.shape == b.shape
    assert np.abs(a - b).mean() < 1.5


def test_preview_rejects_zero_frames(tmp_path):
    with pytest.raises(ValueError):
        render.render_preview(SAMPLE, None, str(tmp_path / "x.png"), n_frames=0)


# ---- full render ----------------------------------------------------------

def test_render_video_keeps_duration_and_audio(tmp_path):
    cube = identity_cube(tmp_path / "identity.cube")
    out = str(tmp_path / "out.mp4")
    assert render.render_video(SAMPLE, cube, out) == out

    src, dst = ffprobe(SAMPLE), ffprobe(out)
    assert abs(float(src["format"]["duration"]) - float(dst["format"]["duration"])) < 0.15

    audio = [s for s in dst["streams"] if s["codec_type"] == "audio"]
    src_audio = [s for s in src["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == 1
    # -c:a copy: same codec, same sample rate, bit-identical stream
    assert audio[0]["codec_name"] == src_audio[0]["codec_name"]
    assert audio[0]["sample_rate"] == src_audio[0]["sample_rate"]

    v = [s for s in dst["streams"] if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (W, H)


def test_render_video_identity_lut_is_near_identical(tmp_path):
    cube = identity_cube(tmp_path / "identity.cube")
    out = str(tmp_path / "out.mp4")
    render.render_video(SAMPLE, cube, out)
    a_png, b_png = tmp_path / "a.png", tmp_path / "b.png"
    render.render_preview(SAMPLE, None, str(a_png), n_frames=3)
    render.render_preview(out, None, str(b_png), n_frames=3)
    a = np.asarray(Image.open(a_png).convert("RGB"), dtype=float)
    b = np.asarray(Image.open(b_png).convert("RGB"), dtype=float)
    assert np.abs(a - b).mean() < 6.0  # x264 re-encode noise dominates


# ---- errors & hw detection ------------------------------------------------

def test_failure_raises_with_stderr(tmp_path):
    with pytest.raises(render.FFmpegError) as e:
        render.render_preview(str(tmp_path / "nope.mp4"), None, str(tmp_path / "o.png"))
    assert "No such file" in str(e.value)


def test_bad_cube_raises(tmp_path):
    bad = tmp_path / "bad.cube"
    bad.write_text("this is not a cube\n")
    with pytest.raises(render.FFmpegError):
        render.render_preview(SAMPLE, str(bad), str(tmp_path / "o.png"), n_frames=1)


def test_detect_hw_encoder_is_cached_and_usable(tmp_path):
    enc = render.detect_hw_encoder()
    assert enc is None or enc in {e for e, _, _ in render._HW_CANDIDATES}
    assert render.detect_hw_encoder() is enc  # cached: no second trial encode
    assert render.detect_hw_encoder.cache_info().hits >= 1


def test_gpu_render_works_or_falls_back(tmp_path):
    cube = identity_cube(tmp_path / "identity.cube")
    out = str(tmp_path / "gpu.mp4")
    if render.detect_hw_encoder() is None:
        with pytest.warns(UserWarning, match="libx264"):
            render.render_video(SAMPLE, cube, out, gpu=True)
    else:
        render.render_video(SAMPLE, cube, out, gpu=True)
    assert [s for s in ffprobe(out)["streams"] if s["codec_type"] == "audio"]
