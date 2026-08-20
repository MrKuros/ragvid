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
    assert enc is None or enc in {e for e, _, _ in render.hw_encoders()}
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


# ---- GIF / no-audio inputs ------------------------------------------------
# Regression: exporting to .gif failed outright. render_video pinned
# -pix_fmt yuv420p (needed so lut3d's yuv444p output doesn't produce H.264
# files that hardware decoders reject), but the GIF encoder takes pal8, so
# ffmpeg died with "Invalid argument" in the filter chain and left a 0-byte
# file behind while still exiting 0 from the caller's point of view.

SAMPLE_GIF = str(ROOT / "assets" / "sample.gif")


def test_gif_input_renders_to_mp4(tmp_path):
    """A GIF carries no audio; -map 0:a:0? must tolerate the missing stream."""
    out = str(tmp_path / "out.mp4")
    render.render_video(SAMPLE_GIF, identity_cube(tmp_path / "id.cube"), out)
    assert Path(out).stat().st_size > 0
    streams = ffprobe(out)["streams"]
    assert [s["codec_type"] for s in streams] == ["video"]
    assert streams[0]["pix_fmt"] == "yuv420p"


def test_gif_output_is_a_real_animated_gif(tmp_path):
    out = str(tmp_path / "out.gif")
    render.render_video(SAMPLE_GIF, identity_cube(tmp_path / "id.cube"), out)
    assert Path(out).stat().st_size > 0
    assert ffprobe(out)["streams"][0]["codec_name"] == "gif"
    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) > 1, "collapsed to a single frame"


def test_gif_output_still_applies_the_grade(tmp_path):
    """The 256-colour palette pass must not quantize the grade away."""
    from ragvid.lut import bake_cube
    from ragvid.spec import GradeSpec, RGB

    cube = bake_cube(GradeSpec(slope=RGB(r=1.6, g=1.0, b=1.0)), str(tmp_path / "red.cube"))
    graded, plain = str(tmp_path / "g.gif"), str(tmp_path / "p.gif")
    render.render_video(SAMPLE_GIF, cube, graded)
    render.render_video(SAMPLE_GIF, identity_cube(tmp_path / "id.cube"), plain)

    def mean_rgb(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"), float).reshape(-1, 3).mean(axis=0)

    g, pl = mean_rgb(graded), mean_rgb(plain)
    assert g[0] > pl[0] + 2.0, f"red should survive the palette pass: {pl} -> {g}"


# ---- odd dimensions -------------------------------------------------------
# Regression: libx264 at yuv420p refuses odd width/height outright ("height not
# divisible by 2"), so exporting a 720x405 source to .mp4 failed at the very
# last step, after the user had already waited through the encode. Plenty of
# real sources are odd -- GIFs, hand-cropped clips.


def test_odd_dimension_source_exports(tmp_path):
    """An odd-sized input must still produce a playable H.264 file."""
    odd = str(tmp_path / "odd.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=641x405:rate=10:duration=1",
         "-c:v", "libx264rgb", "-pix_fmt", "rgb24", odd],
        check=True,
    )
    out = str(tmp_path / "out.mp4")
    render.render_video(odd, identity_cube(tmp_path / "id.cube"), out)

    stream = ffprobe(out)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    # cropped down to the nearest even size, never up and never resampled
    assert (stream["width"], stream["height"]) == (640, 404)


def test_even_dimension_source_is_not_cropped(tmp_path):
    """The crop must be a no-op when it isn't needed."""
    out = str(tmp_path / "out.mp4")
    render.render_video(SAMPLE, identity_cube(tmp_path / "id.cube"), out)
    stream = ffprobe(out)["streams"][0]
    assert (stream["width"], stream["height"]) == (W, H)
