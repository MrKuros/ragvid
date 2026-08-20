"""Cross-module tests: no mocks, real ffmpeg, real assets.

Every other test file mocks its neighbours, which is how a color-space mismatch
between probe.py and spec.GradeSpec survived six green suites. These run the
whole offline path.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
from PIL import Image

from ragvid import cli
from ragvid.lut import bake_cube, read_cube
from ragvid.match import match_reference
from ragvid.probe import probe_image, probe_video
from ragvid.session import Session
from ragvid.spec import GradeSpec

# Absolute: several of these chdir into a tmp cwd.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = str(_ROOT / "assets" / "sample.mp4")
REF = str(_ROOT / "assets" / "ref_warm.png")


def _mean(arr8: np.ndarray) -> np.ndarray:
    return (np.asarray(arr8, dtype=np.float64).reshape(-1, 3) / 255.0).mean(axis=0)


def _ffprobe_field(path: str, stream: str, entries: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", entries, "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _audio_md5(path: str) -> str:
    """MD5 of the raw audio packets — differs the moment anything re-encodes."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_reference_match_actually_lands_on_the_reference():
    """The regression that matters: probe's moments and GradeSpec.apply must be
    in the same color space, or the fit is solved against the wrong numbers."""
    src, ref = probe_video(SAMPLE, n_frames=5), probe_image(REF)
    spec = match_reference(src, ref)

    # No channel may be pinned to the clamp — that is the symptom of a bad fit.
    assert np.all(spec.slope.as_array() > 0.1), spec.slope

    graded = spec.apply(np.random.default_rng(0).normal(
        src.mean.as_array(), src.std.as_array(), size=(20000, 3)).clip(0, 1))
    assert np.allclose(graded.mean(axis=0), ref.mean.as_array(), atol=0.06), graded.mean(axis=0)


def test_matched_grade_through_real_ffmpeg_moves_toward_reference(tmp_path):
    """probe -> match -> bake_cube -> ffmpeg lut3d, measured on real pixels."""
    src, ref = probe_video(SAMPLE, n_frames=5), probe_image(REF)
    cube = bake_cube(match_reference(src, ref), str(tmp_path / "m.cube"))

    from ragvid.render import render_preview

    before = np.asarray(Image.open(render_preview(SAMPLE, None, str(tmp_path / "a.png"))).convert("RGB"))
    after = np.asarray(Image.open(render_preview(SAMPLE, cube, str(tmp_path / "b.png"))).convert("RGB"))

    target = ref.mean.as_array()
    d_before = np.linalg.norm(_mean(before) - target)
    d_after = np.linalg.norm(_mean(after) - target)
    assert d_after < d_before / 2, (d_before, d_after)


def test_identity_spec_bakes_to_an_identity_cube(tmp_path):
    size, table = read_cube(bake_cube(GradeSpec.identity(), str(tmp_path / "i.cube"), size=17))
    grid = np.linspace(0, 1, size)
    # .cube data order is red-fastest.
    expect = np.array([[r, g, b] for b in grid for g in grid for r in grid])
    assert np.allclose(table, expect, atol=1e-6)


def test_cli_offline_round_trip(tmp_path, monkeypatch, capsys):
    """grade --ref / spec / export / reset, for real, with no LLM in the loop."""
    monkeypatch.chdir(tmp_path)

    assert cli.main(["grade", SAMPLE, "--ref", REF]) == 0
    assert cli.main(["spec"]) == 0
    assert '"slope"' in capsys.readouterr().out

    s = Session.load()
    assert s.source == SAMPLE and len(s.specs) == 1

    out = tmp_path / "out.mp4"
    assert cli.main(["export", str(out)]) == 0
    assert out.stat().st_size > 0

    # Export must stay playable and must not re-encode audio.
    assert _ffprobe_field(str(out), "v:0", "stream=pix_fmt") == "yuv420p"
    assert _ffprobe_field(str(out), "a:0", "stream=codec_name") == "aac"
    assert _audio_md5(str(out)) == _audio_md5(SAMPLE)

    # reset at the floor is a no-op, not an error
    assert cli.main(["reset"]) == 0
