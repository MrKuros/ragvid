"""CLI tests: argument parsing, exit codes and what gets printed.

The orchestration itself now lives in ragvid.project.Project (see
tests/test_project.py); these drive it through main() to prove the front end
wires argv to it correctly. Everything that touches ffmpeg or an API is
monkeypatched at its defining module -- cli.py no longer imports any of it, and
Project imports its dependencies inside the methods that use them.
"""

import json
from pathlib import Path

import pytest

from ragvid import cli
from ragvid.probe import ClipStats
from ragvid.session import Session
from ragvid.spec import RGB, GradeSpec

STATS = ClipStats(
    mean=RGB(r=0.2, g=0.3, b=0.4),
    std=RGB.of(0.1),
    saturation=0.25,
    frames_sampled=10,
    width=640,
    height=360,
    duration=4.0,
)


def _art(name: str) -> str:
    """An artifact ragvid writes, under the project dir (cwd for the CLI)."""
    return str(Path.cwd() / ".ragvid" / name)


def _source() -> str:
    """What Project.create stores as the source: absolute, resolved."""
    return str(Path.cwd() / "clip.mp4")


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """Run in a temp cwd with every heavy dependency recorded, not executed."""
    monkeypatch.chdir(tmp_path)
    # Project.create/plan_from_reference reject paths that don't exist, so the
    # inputs have to be real files even though nothing ever reads them.
    (tmp_path / "clip.mp4").write_bytes(b"")
    (tmp_path / "ref.png").write_bytes(b"")
    seen = {}

    def rec(name, ret):
        def f(*a, **kw):
            seen[name] = (a, kw)
            return ret
        return f

    # Patch where each function is defined: Project imports them inside its
    # methods, so there is no module-level name to shadow.
    monkeypatch.setattr("ragvid.probe.probe_video", rec("probe_video", STATS))
    monkeypatch.setattr("ragvid.probe.probe_image", rec("probe_image", STATS))
    monkeypatch.setattr("ragvid.match.match_reference", rec("match_reference", GradeSpec(temperature=800, rationale="matched")))
    monkeypatch.setattr("ragvid.vibe.plan_vibe", rec("plan_vibe", GradeSpec(contrast=0.4, rationale="gloomy")))
    monkeypatch.setattr("ragvid.refine.refine_spec", rec("refine_spec", GradeSpec(contrast=0.1, rationale="less")))
    monkeypatch.setattr("ragvid.lut.bake_cube", rec("bake_cube", None))
    monkeypatch.setattr("ragvid.render.render_preview", rec("render_preview", None))
    monkeypatch.setattr("ragvid.render.render_video", rec("render_video", "out.mp4"))
    return seen


def test_grade_vibe(calls, capsys):
    assert cli.main(["grade", "clip.mp4", "--vibe", "gloomy"]) == 0

    assert calls["plan_vibe"][0][:2] == ("gloomy", STATS)
    assert "match_reference" not in calls
    assert calls["bake_cube"][0][1] == _art("current.cube")
    assert calls["render_preview"][0][:3] == (_source(), _art("current.cube"), _art("preview.png"))

    s = Session.load()
    assert s.source == _source()
    assert s.spec.contrast == 0.4
    assert s.stats == STATS

    out = capsys.readouterr().out
    assert "gloomy" in out and _art("preview.png") in out


def test_grade_ref_never_calls_llm(calls):
    assert cli.main(["grade", "clip.mp4", "--ref", "ref.png"]) == 0
    assert calls["match_reference"][0] == (STATS, STATS)
    assert calls["probe_image"][0][0] == "ref.png"
    assert "plan_vibe" not in calls
    assert Session.load().spec.temperature == 800


def test_grade_requires_exactly_one_source(calls):
    with pytest.raises(SystemExit):
        cli.main(["grade", "clip.mp4"])
    with pytest.raises(SystemExit):
        cli.main(["grade", "clip.mp4", "--vibe", "x", "--ref", "y.png"])


def test_refine_uses_cached_stats_and_pushes(calls):
    cli.main(["grade", "clip.mp4", "--vibe", "gloomy"])
    assert cli.main(["refine", "less contrast"]) == 0

    current, instruction, stats = calls["refine_spec"][0][:3]
    assert current.contrast == 0.4  # the spec grade produced
    assert instruction == "less contrast"
    assert stats == STATS
    assert "probe_video" in calls and calls["probe_video"][0][0] == "clip.mp4"

    s = Session.load()
    assert len(s.specs) == 2 and s.spec.contrast == 0.1


def test_refine_without_session_is_an_error(calls, capsys):
    assert cli.main(["refine", "warmer"]) == 1
    err = capsys.readouterr().err
    assert "ragvid grade" in err and "Traceback" not in err


def test_spec_prints_json(calls, capsys):
    cli.main(["grade", "clip.mp4", "--vibe", "gloomy"])
    capsys.readouterr()
    assert cli.main(["spec"]) == 0
    assert json.loads(capsys.readouterr().out)["contrast"] == 0.4


def test_reset_steps_back(calls, capsys):
    cli.main(["grade", "clip.mp4", "--vibe", "gloomy"])
    cli.main(["refine", "less contrast"])
    assert cli.main(["reset"]) == 0
    assert Session.load().spec.contrast == 0.4

    # at the floor it says so and stays put, without failing
    capsys.readouterr()
    assert cli.main(["reset"]) == 0
    assert "nothing to step back to" in capsys.readouterr().out
    assert Session.load().spec.contrast == 0.4


def test_export(calls, capsys):
    cli.main(["grade", "clip.mp4", "--vibe", "gloomy"])
    capsys.readouterr()
    assert cli.main(["export", "out.mp4", "--gpu"]) == 0

    args, kw = calls["render_video"]
    assert args[0] == _source()
    assert args[2] == "out.mp4"
    # NOT the shared .ragvid/current.cube: export bakes to a private temporary
    # LUT, because anything rendering a frame re-bakes current.cube in place and
    # would hand ffmpeg a different grade mid-encode.
    assert args[1] != str(_art("current.cube"))
    assert args[1].endswith(".cube")
    # No progress bar: stderr is captured, so it is not a tty.
    assert kw == {"gpu": True, "progress": None}
    assert "out.mp4" in capsys.readouterr().out


def test_export_defaults_to_cpu(calls):
    cli.main(["grade", "clip.mp4", "--vibe", "gloomy"])
    cli.main(["export", "out.mp4"])
    assert calls["render_video"][1] == {"gpu": False, "progress": None}


def test_provider_passthrough(calls, monkeypatch):
    got = {}
    monkeypatch.setattr(
        "ragvid.providers.base.get_provider",
        lambda name=None, model=None: got.setdefault("p", (name, model)),
    )
    cli.main(["--provider", "anthropic", "--model", "m1", "grade", "clip.mp4", "--vibe", "x"])
    assert got["p"] == ("anthropic", "m1")
    assert calls["plan_vibe"][1]["provider"] == ("anthropic", "m1")


def test_no_provider_flag_passes_none(calls):
    cli.main(["grade", "clip.mp4", "--vibe", "x"])
    assert calls["plan_vibe"][1]["provider"] is None


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main(["bogus"])
