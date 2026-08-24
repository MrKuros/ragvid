import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ragvid import render
from ragvid.spec import EffectSpec

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


# ---- the grade that reaches the pixels ------------------------------------
# Everything above tests plumbing. This measures the actual thing a user sees:
# a .cube baked from a spec, pushed through ffmpeg's own lut3d, read back as
# 8-bit code values. It exists because "the LUT math is right" and "the render
# is right" have been separately true here while the pair was wrong.

def ramp_png(path: Path, w: int = 256, h: int = 32) -> str:
    """A horizontal 0..255 grey ramp -- one pixel column per code value."""
    row = np.arange(w, dtype=np.uint8)
    Image.fromarray(np.repeat(np.repeat(row[None, :, None], h, 0), 3, 2)).save(path)
    return str(path)


def _pinned_white_fraction(tmp_path: Path, rolloff: float) -> float:
    from ragvid.lut import bake_cube
    from ragvid.spec import GradeSpec, RGB

    cube = bake_cube(
        GradeSpec(slope=RGB.of(1.6), highlight_rolloff=rolloff),
        str(tmp_path / f"r{rolloff}.cube"),
    )
    out = str(tmp_path / f"r{rolloff}.png")
    render.render_frame(ramp_png(tmp_path / "ramp.png"), cube, out)
    a = np.asarray(Image.open(out).convert("RGB"), dtype=np.uint8)
    return float(np.count_nonzero(a[..., 0] == 255)) / a[..., 0].size


def test_rolloff_survives_the_round_trip_through_ffmpeg(tmp_path):
    """slope=1.6 welds 37.5% of a ramp to pure white; the shoulder recovers it.

    Measured through the real filter chain, so it also covers ffmpeg's
    tetrahedral interpolation of the baked cube, not just spec.apply().
    """
    hard = _pinned_white_fraction(tmp_path, 0.0)
    soft = _pinned_white_fraction(tmp_path, 1.0)
    assert hard == pytest.approx(0.375, abs=0.01), hard   # today's behaviour, intact
    assert soft == 0.0, soft

    # and the recovered range carries real detail rather than a flat shelf.
    # Above the old clip point (input 160/255) the hard-clipped render holds a
    # single code value; the shoulder holds a monotone gradient instead.
    def top_levels(name):
        a = np.asarray(Image.open(tmp_path / name).convert("RGB"), dtype=int)
        return a[0, 160:, 0]

    hard_top, soft_top = top_levels("r0.0.png"), top_levels("r1.0.png")
    assert list(np.unique(hard_top)) == [255]             # 96 columns, one value
    assert np.all(np.diff(soft_top) >= 0)                 # monotone: no inversion
    assert len(np.unique(soft_top)) > 20, np.unique(soft_top)
    assert soft_top.max() < 255                           # the unavoidable white loss


def test_baked_grade_matches_spec_apply_through_ffmpeg(tmp_path):
    """The whole path -- spec -> .cube -> ffmpeg lut3d -> 8-bit -- within a few
    code values of the exact math. Catches a wrong-way LUT axis order or a
    colour-range conversion sneaking into the filter chain."""
    from ragvid.lut import bake_cube
    from ragvid.spec import GradeSpec, HueBand, RGB

    spec = GradeSpec(slope=RGB(r=1.15, g=1.0, b=0.9), saturation=1.2,
                     contrast=0.3, exposure=0.2, highlight_rolloff=0.3,
                     shadow_tint=RGB(r=0.0, g=0.03, b=0.05),
                     hue_red=HueBand(sat=0.7), hue_blue=HueBand(sat=1.25))
    cube = bake_cube(spec, str(tmp_path / "look.cube"))

    src = tmp_path / "src.png"
    rng = np.random.default_rng(3)
    pix = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(pix).save(src)

    out = str(tmp_path / "graded.png")
    render.render_frame(str(src), cube, out)
    got = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    want = spec.apply(pix.astype(float) / 255.0) * 255.0

    err = np.abs(got - want).max()
    assert err < 6.0, f"max {err:.2f} code values off the exact grade"
    assert np.abs(got - want).mean() < 1.0


# ---- effects filtergraph ---------------------------------------------------
# Was a `python -m ragvid.render` self-check while tests/ belonged to another
# agent this build; moved here on integration so it runs in CI. ~0.9s total.


def test_no_effects_leaves_the_lut3d_node_alone():
    """The bare-lut3d string is asserted elsewhere (test_platform); this asserts
    the effects wrapper cannot perturb it, which is what keeps every existing
    render path byte-identical for a grade that uses no effects."""
    bare = render._lut_filter("/tmp/x.cube")
    assert render._vf(None, bare) == bare
    assert render._vf(EffectSpec(), bare) == bare


@pytest.mark.parametrize("name", list(EffectSpec.model_fields))
@pytest.mark.parametrize("value", (0.6, -0.6))
def test_every_effect_fragment_parses_inside_a_real_graph(name, value):
    """String equality cannot tell you a fragment is grammatical once spliced:
    glow's split/blend is three chains pretending to be one filter, and a
    malformed one only surfaces when ffmpeg refuses the whole -vf."""
    bare = render._lut_filter("/tmp/x.cube")
    chain = render._vf(EffectSpec(**{name: value}), bare,
                       "crop=trunc(iw/2)*2:trunc(ih/2)*2")
    chain = chain.replace(bare, "null")  # no cube on disk here
    render._run(["-f", "lavfi", "-i", "testsrc2=s=64x64:d=0.2", "-vf", chain,
                 "-frames:v", "2", "-f", "null", "-"], timeout=60)


# ---- preview must equal the export, per tile ------------------------------

@pytest.mark.parametrize("effects", [
    EffectSpec(vignette=0.8),
    EffectSpec(glow=0.8),
    EffectSpec(softness=0.6),
    EffectSpec(fringe=0.6),
])
def test_preview_tile_matches_the_single_frame_render(tmp_path, effects):
    """Every spatial effect must see ONE frame, not the hstacked sheet.

    Filtering after the stack put a single vignette across the whole sheet and
    bled glow/softness/fringe over the tile seams, so the contact sheet showed a
    look the export could not produce. Measured max |tile - frame| was 96-255
    code values; it must be 0.
    """
    n = 3
    duration = render.probe_duration(SAMPLE)
    sheet_png = tmp_path / "sheet.png"
    render.render_preview(SAMPLE, None, str(sheet_png), effects, n_frames=n)
    sheet = np.asarray(Image.open(sheet_png).convert("RGB")).astype(int)
    w = sheet.shape[1] // n
    for i in range(n):
        frame_png = tmp_path / f"f{i}.png"
        render.render_frame(SAMPLE, None, str(frame_png), effects,
                            at=duration * (i + 0.5) / n)
        frame = np.asarray(Image.open(frame_png).convert("RGB")).astype(int)
        assert np.abs(sheet[:, i * w:(i + 1) * w] - frame).max() == 0


def test_progress_survives_an_ffmpeg_that_floods_stderr(tmp_path):
    """ffmpeg's stderr must never be a pipe we leave undrained.

    _run_with_progress reads stdout for -progress and used to read stderr only
    after that loop ended. Any render whose stderr exceeds the 64 KiB pipe
    buffer therefore blocked forever: ffmpeg stalled writing stderr, we stalled
    reading stdout. The server serialises exports, so one such job also blocked
    every later export for the life of the process.

    Volume comes from -loglevel trace rather than from a deliberately corrupted
    file. Corruption looked realistic but measured the DECODER'S TOLERANCE, not
    our pipe handling: the same bytes make ffmpeg 9 emit ~190 KB of warnings and
    make older builds refuse the file outright with "error reading header", so
    the test passed locally and failed in CI. trace output is a documented
    loglevel that behaves the same everywhere -- measured 161 KB here, 2.5x the
    buffer.

    Run on a thread with a deadline: a regression must FAIL, not hang CI.
    """
    import threading

    seen = []
    box = {}

    def go():
        try:
            box["stderr"] = render._run_with_progress(
                ["-loglevel", "trace", "-i", SAMPLE, "-map", "0:v:0",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 str(tmp_path / "out.mp4")],
                render.probe_duration(SAMPLE), seen.append,
            )
        except BaseException as exc:  # surfaced by the assertions below
            box["error"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=45)   # normal run is under a second; 45s is a deadlock, not slowness

    assert not t.is_alive(), "render deadlocked on an undrained stderr pipe"
    assert "error" not in box, box.get("error")
    assert len(box["stderr"]) > 65536, (
        f"only {len(box['stderr'])} bytes of stderr -- too small to prove the "
        "pipe buffer was exceeded, so this test would pass even if it regressed"
    )
    assert seen and seen[-1] == 1.0


def test_timeout_is_typed_and_leaves_no_partial_file(tmp_path):
    """An expired cap must raise FFmpegError, not a bare TimeoutExpired, and
    must not leave a truncated file that looks like a finished render."""
    from ragvid.errors import FFmpegError

    src = ROOT / "test_files" / "test.mp4"
    if not src.exists():
        pytest.skip("no test_files/test.mp4")
    out = tmp_path / "o.mp4"
    args = ["-i", str(src), "-map", "0:v:0", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(out)]
    with pytest.raises(FFmpegError):
        render._run(args, timeout=0.5)
    assert not out.exists()


def test_run_has_no_default_timeout():
    """A render is as long as the clip is. The old 600s default killed any
    export past ~7 minutes of 1080p source."""
    import inspect

    assert inspect.signature(render._run).parameters["timeout"].default is None


# ---- output bit depth ------------------------------------------------------
# The -pix_fmt pin used to be yuv420p unconditionally, so a 10-bit log source
# was graded and delivered at 8 bits -- which throws away the whole reason for
# shooting log. 4:2:0 is still forced (lut3d negotiates yuv444p and libx264 will
# happily write a file players reject); only the depth follows the source now.


def _source(tmp_path: Path, pix_fmt: str) -> str:
    out = str(tmp_path / f"src_{pix_fmt}.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=320x180:rate=10:duration=1",
         "-c:v", "libx264", "-pix_fmt", pix_fmt, out],
        check=True,
    )
    return out


@pytest.mark.parametrize("src_pix_fmt,want_pix_fmt,want_profile", [
    ("yuv420p", "yuv420p", "High"),
    ("yuv420p10le", "yuv420p10le", "High 10"),
    ("yuv422p10le", "yuv420p10le", "High 10"),   # depth kept, chroma still forced to 4:2:0
])
def test_output_depth_follows_the_source(tmp_path, src_pix_fmt, want_pix_fmt, want_profile):
    out = str(tmp_path / "out.mp4")
    render.render_video(_source(tmp_path, src_pix_fmt),
                        identity_cube(tmp_path / "id.cube"), out)
    stream = ffprobe(out)["streams"][0]
    assert stream["pix_fmt"] == want_pix_fmt
    assert stream["profile"] == want_profile


def test_source_bit_depth_reads_ffprobe(tmp_path):
    assert render._source_bit_depth(_source(tmp_path, "yuv420p")) == 8
    assert render._source_bit_depth(_source(tmp_path, "yuv420p10le")) == 10


def test_source_bit_depth_defaults_to_8_when_unprobeable(tmp_path):
    """An unreadable source must fall back to today's behaviour, not raise --
    render_video's own error handling is what should report the bad input."""
    assert render._source_bit_depth(str(tmp_path / "nope.mp4")) == 8


def test_gpu_export_stays_8_bit(tmp_path):
    """No H.264 hardware encoder here can do 10 bits -- NVENC's profiles stop at
    high444p, QSV takes nv12 only, AMF is 8-bit out -- and vf_suffix is empty for
    all of them except VAAPI, so the depth branch has to gate on the encoder
    name. Asking nvenc for -profile:v high10 fails the render outright."""
    src = _source(tmp_path, "yuv420p10le")
    out = str(tmp_path / "gpu.mp4")
    cube = identity_cube(tmp_path / "id.cube")
    if render.detect_hw_encoder() is None:
        with pytest.warns(UserWarning, match="libx264"):
            render.render_video(src, cube, out, gpu=True)
        assert ffprobe(out)["streams"][0]["pix_fmt"] == "yuv420p10le"  # fell back to CPU
    else:
        render.render_video(src, cube, out, gpu=True)
        assert ffprobe(out)["streams"][0]["pix_fmt"] == "yuv420p"


# ---- regions: a grade over part of the frame (roadmap B1) -----------------
#
# A region is spatial, so like `effects` it composites AROUND the lut3d node
# rather than inside it. Everything here measures pixels: a filter graph that
# parses is not evidence that the mask landed where the sentence said.


def darken_cube(path: Path, stops: float = -1.0, size: int = 17) -> str:
    """A .cube that halves (or doubles) every value. Written here for the same
    reason identity_cube is: this file tests render.py and nothing else."""
    g = np.linspace(0.0, 1.0, size)
    k = 2.0 ** stops
    lines = [f"LUT_3D_SIZE {size}", ""]
    for b in g:
        for gg in g:
            for r in g:
                lines.append(" ".join(f"{min(v * k, 1.0):.6f}" for v in (r, gg, b)))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def region_layer(tmp_path: Path, target: str = "top", stops: float = -1.0):
    """[(cube, mask png)] for one region, exactly as Project.bake_layers builds it."""
    from ragvid.region import for_target

    cube = darken_cube(tmp_path / f"{target}.cube", stops)
    mask = for_target(target).write_png(tmp_path / f"{target}.png", W, H)
    return [(cube, mask)]


def test_no_layers_leaves_the_lut3d_node_alone():
    """The bare-lut3d string is asserted in test_platform; this asserts the
    region wrapper cannot perturb it. A grade with no regions must produce the
    byte-identical filter chain it produced before regions existed."""
    bare = render._lut_filter("/tmp/x.cube")
    assert render._vf(None, bare, layers=None) == bare
    assert render._vf(None, bare, layers=[]) == bare
    assert render._region_filters(None) == [] and render._region_filters([]) == []


def test_a_region_fragment_is_grammatical_inside_a_real_graph(tmp_path):
    """String equality cannot tell you a fragment is chainable once spliced: the
    composite is five chains pretending to be one filter, and a malformed one
    only surfaces when ffmpeg refuses the whole -vf."""
    layers = region_layer(tmp_path)
    chain = render._vf(EffectSpec(glow=0.5), render._lut_filter(layers[0][0]),
                       "crop=trunc(iw/2)*2:trunc(ih/2)*2", layers=layers)
    render._run(["-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:d=0.2", "-vf", chain,
                 "-frames:v", "2", "-f", "null", "-"], timeout=60)


def test_awkward_mask_filename_survives_the_filtergraph_parser(tmp_path):
    """Same escaping contract the cube path has: the mask is a file path inside
    a filter option, so a comma or a colon in it must not split the graph."""
    from ragvid.region import for_target
    from ragvid.platform import is_windows

    # Windows forbids ':' in a filename -- but every path there already carries
    # one in the drive letter, so the colon half of the contract is exercised
    # anyway and only the comma has to be spelled out by hand.
    d = tmp_path / ("wei,rd dir" if is_windows() else "wei,rd: dir")
    layers = [(darken_cube(tmp_path / "c.cube"),
               for_target("top").write_png(d / "ma,sk.png", W, H))]
    out = str(tmp_path / "f.png")
    render.render_frame(SAMPLE, layers[0][0], out, at=1.0, layers=layers)
    assert Image.open(out).size == (W, H)


def test_the_region_darkens_the_top_and_leaves_the_bottom_alone(tmp_path):
    """The measurement the feature exists for. A flat grey source, so the mask
    is readable straight off the rendered frame."""
    grey = tmp_path / "grey.mp4"
    render._run(["-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:d=1:r=10",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(grey)], timeout=60)
    out = tmp_path / "f.png"
    render.render_frame(str(grey), None, str(out), at=0.5,
                        layers=region_layer(tmp_path, "top", stops=-0.5))
    got = np.asarray(Image.open(out).convert("RGB")).astype(float)

    top, bottom = got[:H // 3].mean(), got[-(H // 3):].mean()
    assert top < 100.0, f"the top barely moved: {top:.2f} of a 128 source"
    assert abs(bottom - 128.0) <= 1.0, f"the bottom moved to {bottom:.2f}"


def test_the_masked_edge_is_soft_all_the_way_through_ffmpeg(tmp_path):
    """The falloff has to survive the 8-bit mask, the alpha merge and the blend.

    A hard edge bands visibly; so does a mask quantised to a handful of levels.
    Measured through the real filter chain, not in numpy.
    """
    # .mkv, not .mp4: MP4 only learned to carry FFV1 recently, so an .mp4 here
    # passes on a new local ffmpeg and fails CI's with "Could not find tag for
    # codec ffv1". Matroska has always taken it, which is what the sibling
    # tests below already use.
    grey = tmp_path / "grey.mkv"
    render._run(["-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:d=1:r=10",
                 "-c:v", "ffv1", str(grey)], timeout=60)
    out = tmp_path / "f.png"
    render.render_frame(str(grey), None, str(out), at=0.5,
                        layers=region_layer(tmp_path, "top", stops=-1.0))
    col = np.asarray(Image.open(out).convert("RGB")).astype(float)[:, W // 2, 0]

    d = np.diff(col)
    assert (d >= -1.0).all(), "the ramp must not reverse"
    assert d.max() <= 4.0, f"a single row jumps {d.max():.0f} code values"
    assert np.count_nonzero((col > 65) & (col < 127)) > H // 5, "the ramp is too short to be soft"


def test_the_still_path_and_the_video_path_composite_identically(tmp_path):
    """A region built one way for the preview and another for the export is the
    same bug as filtering the contact sheet after stacking it.

    Both paths are compared at FULL chroma resolution, because the export's own
    4:2:0 already costs 23 code values on testsrc2's hard colour edges with NO
    region present at all -- measuring through it would report the pixel format,
    not the composite. At full chroma the no-region control is exactly 0.0 and
    the region path is 2.0 max / 0.11 mean, which is the 8-bit rounding of
    blending in YUV on one path and RGB on the other.
    """
    layers = region_layer(tmp_path, "top")
    cube = layers[0][0]
    still_png = tmp_path / "still.png"
    render.render_frame(SAMPLE, cube, str(still_png), at=1.5, layers=layers)
    still = np.asarray(Image.open(still_png).convert("RGB")).astype(float)

    vf = render._vf(None, render._lut_filter(cube),
                    "crop=trunc(iw/2)*2:trunc(ih/2)*2", layers=layers)
    mkv = tmp_path / "v.mkv"
    render._run(["-ss", "1.5", "-i", SAMPLE, "-vf", vf, "-frames:v", "1",
                 "-c:v", "ffv1", "-pix_fmt", "gbrp", str(mkv)], timeout=120)
    vid_png = tmp_path / "v.png"
    render._run(["-i", str(mkv), "-frames:v", "1", "-update", "1", str(vid_png)], timeout=60)
    vid = np.asarray(Image.open(vid_png).convert("RGB")).astype(float)

    err = np.abs(still - vid)
    assert err.max() <= 4.0, f"still vs video path: {err.max():.0f} code values"
    assert err.mean() < 0.6


def test_the_preview_tile_composites_the_region_like_the_single_frame(tmp_path):
    """Per-tile again, this time for regions: the mask is in frame-relative
    coordinates, so one applied to the hstacked sheet would smear every region
    across three frames -- exactly what the effects fragment used to do."""
    layers = region_layer(tmp_path, "center")
    cube = layers[0][0]
    n = 3
    duration = render.probe_duration(SAMPLE)
    sheet_png = tmp_path / "sheet.png"
    render.render_preview(SAMPLE, cube, str(sheet_png), n_frames=n, layers=layers)
    sheet = np.asarray(Image.open(sheet_png).convert("RGB")).astype(int)
    w = sheet.shape[1] // n
    for i in range(n):
        frame_png = tmp_path / f"f{i}.png"
        render.render_frame(SAMPLE, cube, str(frame_png), at=duration * (i + 0.5) / n,
                            layers=layers)
        frame = np.asarray(Image.open(frame_png).convert("RGB")).astype(int)
        assert np.abs(sheet[:, i * w:(i + 1) * w] - frame).max() == 0


def test_a_mask_sized_from_ffprobe_survives_a_frame_the_decoder_rotated(tmp_path):
    """alphamerge REFUSES a mask whose size differs from the frame.

    The mask is written at the size ffprobe reported and the frame is what the
    decoder produced, and those disagree on any clip carrying rotation side data
    -- a phone video reports 1920x1080 and decodes to 1080x1920. Without the
    scale2ref that recovers it, the export fails outright with "Input frame
    sizes do not match" rather than looking slightly wrong. Simulated here by
    handing the chain a transposed mask directly.
    """
    from ragvid.region import for_target

    layers = [(darken_cube(tmp_path / "c.cube"),
               for_target("top").write_png(tmp_path / "m.png", H, W))]  # transposed
    out = tmp_path / "f.png"
    render.render_frame(SAMPLE, None, str(out), at=1.0, layers=layers)
    got = np.asarray(Image.open(out).convert("RGB")).astype(float)
    assert got.shape[:2] == (H, W)
    assert got[:60].mean() < got[-60:].mean(), "the rescaled mask still darkens the top"


def test_odd_dimensions_still_export_with_a_region(tmp_path):
    """The even-dimension crop runs AFTER the composite, so the mask sees the
    source's real size. Getting that order wrong is an export that dies at the
    very last step on a hand-cropped clip."""
    odd = tmp_path / "odd.mkv"
    render._run(["-f", "lavfi", "-i", "color=c=red:s=321x241:d=0.5:r=10",
                 "-c:v", "ffv1", "-pix_fmt", "gbrp", str(odd)], timeout=60)
    from ragvid.region import for_target

    layers = [(darken_cube(tmp_path / "c.cube"),
               for_target("center").write_png(tmp_path / "m.png", 321, 241))]
    out = tmp_path / "odd.mp4"
    render.render_video(str(odd), layers[0][0], str(out), layers=layers)
    assert ffprobe(str(out))["streams"][0]["width"] == 320


def test_a_region_does_not_drag_a_10_bit_export_down_to_8(tmp_path):
    """`overlay=format=auto` is what keeps the depth. The alternative graphs
    considered here (maskedmerge over gbrp) would silently truncate exactly the
    log footage the 10-bit path was added for."""
    src = tmp_path / "s10.mp4"
    render._run(["-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:d=1:r=10", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p10le", "-profile:v", "high10", str(src)], timeout=60)
    from ragvid.region import for_target

    layers = [(darken_cube(tmp_path / "c.cube"),
               for_target("top").write_png(tmp_path / "m.png", W, H))]
    out = tmp_path / "o.mp4"
    render.render_video(str(src), layers[0][0], str(out), layers=layers)
    assert ffprobe(str(out))["streams"][0]["pix_fmt"] == "yuv420p10le"


def test_two_regions_composite_in_order_through_ffmpeg(tmp_path):
    """The numpy stack composes layer-on-layer; the filter chain must do the
    same, or the export disagrees with everything else in the project."""
    from ragvid.region import for_target

    grey = tmp_path / "grey.mkv"
    render._run(["-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:d=1:r=10",
                 "-c:v", "ffv1", str(grey)], timeout=60)
    layers = [
        (darken_cube(tmp_path / "a.cube"), for_target("top").write_png(tmp_path / "a.png", W, H)),
        (darken_cube(tmp_path / "b.cube"), for_target("left").write_png(tmp_path / "b.png", W, H)),
    ]
    out = tmp_path / "f.png"
    render.render_frame(str(grey), None, str(out), layers=layers)
    got = np.asarray(Image.open(out).convert("RGB")).astype(float)[..., 0]
    tl, tr, bl, br = got[5, 5], got[5, -5], got[-5, 5], got[-5, -5]
    assert br == 128.0, "neither region reaches the bottom right"
    assert tr == bl == 64.0, "one region each: half of 128"
    assert tl == 32.0, f"both regions, composed: expected 32, got {tl}"


# ---- semantic regions (roadmap B2) ----------------------------------------
#
# render.py knows NOTHING about B2 and these tests exist to keep it that way. A
# semantic mask arrives as the same 8-bit greyscale PNG a geometric one does,
# so the filter chain is unchanged and the only new risk is that the mask's own
# shape -- a soft blob rather than a full-width ramp -- trips something in it.
#
# No model is loaded: segment.class_prob is stubbed with a probability field.
# See tests/test_segment.py's docstring for why no test may download it.


def semantic_layer(tmp_path: Path, monkeypatch, stops: float = -1.0):
    """[(cube, mask png)] for one semantic region, as bake_layers will build it.

    The frame is passed to write_png -- that is the one wiring difference B2
    introduces, and it is the reason Project.bake_layers has to decode a frame.
    """
    from ragvid import segment
    from ragvid.region import for_target

    p = np.full((128, 128), 0.02)
    p[: int(128 * 0.55)] = 0.98
    monkeypatch.setattr(segment, "class_prob", lambda rgb, name: p)
    cube = darken_cube(tmp_path / "sky.cube", stops)
    mask = for_target("sky").write_png(tmp_path / "sky.png", W, H, frame=np.zeros((H, W, 3)))
    return [(cube, mask)]


def test_a_semantic_mask_darkens_its_subject_and_leaves_the_rest_alone(tmp_path, monkeypatch):
    """Measured on real pixels through the real chain, not on a filter string."""
    # .mkv, not .mp4: MP4 only learned to carry FFV1 recently, so an .mp4 here
    # passes on a new local ffmpeg and fails CI's with "Could not find tag for
    # codec ffv1". Matroska has always taken it, which is what the sibling
    # tests below already use.
    grey = tmp_path / "grey.mkv"
    render._run(["-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:d=1:r=10",
                 "-c:v", "ffv1", str(grey)], timeout=60)
    out = tmp_path / "f.png"
    render.render_frame(str(grey), None, str(out), at=0.5,
                        layers=semantic_layer(tmp_path, monkeypatch))
    got = np.asarray(Image.open(out).convert("RGB")).astype(float)
    assert got[:40].mean() < 70.0, "the subject was not darkened"
    assert abs(got[-40:].mean() - 128.0) <= 1.0, "everything else moved"


def test_the_semantic_edge_is_soft_all_the_way_through_ffmpeg(tmp_path, monkeypatch):
    """Same claim B1's mask makes, on a mask that came from a model instead of
    from geometry. The feather is measured in region.py at 13.2 code values per
    row on the float mask and 14 on the 8-bit PNG; here it is measured on the
    graded frame, where the layer's own -1 stop halves the step."""
    # .mkv, not .mp4: MP4 only learned to carry FFV1 recently, so an .mp4 here
    # passes on a new local ffmpeg and fails CI's with "Could not find tag for
    # codec ffv1". Matroska has always taken it, which is what the sibling
    # tests below already use.
    grey = tmp_path / "grey.mkv"
    render._run(["-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:d=1:r=10",
                 "-c:v", "ffv1", str(grey)], timeout=60)
    out = tmp_path / "f.png"
    render.render_frame(str(grey), None, str(out), at=0.5,
                        layers=semantic_layer(tmp_path, monkeypatch))
    col = np.asarray(Image.open(out).convert("RGB")).astype(float)[:, W // 2, 0]
    d = np.diff(col)
    assert (d >= -1.0).all(), "the ramp must not reverse"
    assert d.max() <= 8.0, f"a single row jumps {d.max():.0f} code values"
    assert np.count_nonzero((col > 70) & (col < 125)) > 6, "the ramp is too short to be soft"


def test_the_still_path_and_the_video_path_composite_a_semantic_mask_identically(tmp_path, monkeypatch):
    """The rule this project keeps breaking: the preview must match the export.

    Measured on the real model at 1280x720 with a real sky: max 0.0 code values,
    mean 0.000 -- a semantic mask is a PNG like any other, so it inherits B1's
    composite exactly. Compared at full chroma for B1's reason: 4:2:0 costs 23
    code values on testsrc2's colour edges with no region present at all.
    """
    layers = semantic_layer(tmp_path, monkeypatch)
    still_png = tmp_path / "still.png"
    render.render_frame(SAMPLE, None, str(still_png), at=1.5, layers=layers)
    still = np.asarray(Image.open(still_png).convert("RGB")).astype(float)

    vf = render._vf(None, None, "crop=trunc(iw/2)*2:trunc(ih/2)*2", layers=layers)
    mkv = tmp_path / "v.mkv"
    render._run(["-ss", "1.5", "-i", SAMPLE, "-vf", vf, "-frames:v", "1",
                 "-c:v", "ffv1", "-pix_fmt", "gbrp", str(mkv)], timeout=120)
    vid_png = tmp_path / "v.png"
    render._run(["-i", str(mkv), "-frames:v", "1", "-update", "1", str(vid_png)], timeout=60)
    vid = np.asarray(Image.open(vid_png).convert("RGB")).astype(float)

    err = np.abs(still - vid)
    assert err.max() <= 4.0, f"still vs video path: {err.max():.0f} code values"
    assert err.mean() < 0.6
