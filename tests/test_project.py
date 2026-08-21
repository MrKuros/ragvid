"""The facade a GUI drives. No CLI, no argv, no cwd assumptions, no network.

Everything below patches probe_video/probe_image at their defining module,
because Project imports its dependencies inside the methods that use them --
deliberately, so a plain `spec` read never drags numpy or ffmpeg in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragvid.errors import InputError, SessionNotFound
from ragvid.probe import ClipStats
from ragvid.project import Project
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

# A reference that is measurably warmer/flatter than STATS, so match_reference
# has something real to solve for.
REF_STATS = STATS.model_copy(update={"mean": RGB(r=0.5, g=0.3, b=0.2), "std": RGB.of(0.15)})


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Nothing in this file may reach a provider. Free-tier tokens are finite."""
    def boom(*a, **kw):
        raise AssertionError("a provider was constructed; this path must stay offline")

    monkeypatch.setattr("ragvid.providers.get_provider", boom)
    monkeypatch.setattr("ragvid.providers.base.get_provider", boom)


@pytest.fixture
def clip(tmp_path, monkeypatch):
    """A stand-in video file with probing faked out."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    (tmp_path / "ref.png").write_bytes(b"")  # probe_image is faked; only existence matters
    monkeypatch.setattr("ragvid.probe.probe_video",
                        lambda path, n_frames=10, input_lut=None: STATS)
    monkeypatch.setattr("ragvid.probe.probe_image", lambda path: REF_STATS)
    return video


@pytest.fixture
def project(clip, tmp_path):
    return Project.create(clip, root=tmp_path / "proj")


# ---- lifecycle ------------------------------------------------------------


def test_create_probes_once_and_caches_the_stats(clip, tmp_path):
    p = Project.create(clip, root=tmp_path / "proj")

    assert p.stats == STATS
    assert p.source == str(clip.resolve())
    assert p.root == tmp_path / "proj"
    assert p.is_planned is False and p.history == []


def test_create_passes_n_frames_through(clip, tmp_path, monkeypatch):
    seen = {}

    def fake(path, n_frames=10, input_lut=None):
        seen["n"] = n_frames
        return STATS

    monkeypatch.setattr("ragvid.probe.probe_video", fake)
    Project.create(clip, root=tmp_path / "proj", n_frames=4)
    assert seen["n"] == 4


def test_create_defaults_root_to_the_videos_own_folder(clip):
    """Opening a clip from a file picker must not need a second question."""
    assert Project.create(clip).root == clip.parent


def test_create_with_a_missing_file_is_an_input_error(tmp_path):
    with pytest.raises(InputError) as info:
        Project.create(tmp_path / "nope.mp4")
    assert info.value.reason == "no such file"
    assert str(tmp_path / "nope.mp4") in str(info.value)


def test_exists_and_open_round_trip(project, tmp_path):
    root = tmp_path / "proj"
    assert Project.exists(root) is False  # nothing saved until something is pushed

    project.set_spec(GradeSpec(contrast=0.4, rationale="gloomy"))
    assert Project.exists(root) is True

    reopened = Project.open(root)
    assert reopened.source == project.source
    assert reopened.stats == STATS
    assert reopened.spec.contrast == 0.4
    assert reopened.root == root


def test_open_without_a_project_raises(tmp_path):
    with pytest.raises(SessionNotFound, match="run 'ragvid grade' first"):
        Project.open(tmp_path / "empty")


def test_two_projects_do_not_share_state(clip, tmp_path):
    """The reason root exists at all: a GUI has several open at once."""
    a = Project.create(clip, root=tmp_path / "a")
    b = Project.create(clip, root=tmp_path / "b")
    a.set_spec(GradeSpec(contrast=0.4))
    b.set_spec(GradeSpec(contrast=-0.4))

    assert Project.open(tmp_path / "a").spec.contrast == 0.4
    assert Project.open(tmp_path / "b").spec.contrast == -0.4


# ---- paths ----------------------------------------------------------------


def test_artifact_paths_live_under_one_deletable_folder(project, tmp_path):
    root = tmp_path / "proj"
    assert project.state_dir == root / ".ragvid" == Session.dir(root)
    assert project.cube_path == root / ".ragvid" / "current.cube"
    assert project.preview_path == root / ".ragvid" / "preview.png"


# ---- planning -------------------------------------------------------------


def test_plan_from_reference_is_offline_and_moves_toward_the_reference(project, tmp_path):
    """No LLM (the autouse fixture would fail the test), and a real fit."""
    spec = project.plan_from_reference(tmp_path / "ref.png")

    assert spec is project.spec
    assert project.is_planned and len(project.history) == 1
    # STATS is blue-ish and flat, REF_STATS is warm and contrastier.
    assert spec.offset.r > spec.offset.b
    assert 1.0 < spec.slope.r < 2.0    # a real fit, not pinned to the clamp
    assert Project.open(project.root).spec == spec  # persisted, not just in memory


def test_plan_from_reference_with_a_missing_image_is_an_input_error(project):
    with pytest.raises(InputError, match="no such file"):
        project.plan_from_reference("/nope/ref.png")
    assert project.is_planned is False


def test_plan_from_vibe_uses_the_given_provider_and_cached_stats(project):
    """The LLM is a fake object: the call shape is what is under test."""
    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.user = None

        def plan(self, system, user):
            self.user = user
            return GradeSpec(contrast=0.4, rationale="gloomy")

    fake = FakeProvider()
    spec = project.plan_from_vibe("gloomy", provider=fake)

    from ragvid.vibe import format_stats

    assert spec.rationale == "gloomy" and project.spec == spec
    assert "gloomy" in fake.user
    assert format_stats(STATS) in fake.user  # calibrated to the cached stats


def test_refine_never_reprobes(project, monkeypatch):
    monkeypatch.setattr("ragvid.probe.probe_video", lambda *a, **kw: pytest.fail("re-probed"))
    project.set_spec(GradeSpec(contrast=0.4))

    seen = {}
    monkeypatch.setattr(
        "ragvid.refine.refine_spec",
        lambda spec, instruction, stats, provider=None: seen.update(
            spec=spec, instruction=instruction, stats=stats, provider=provider
        ) or GradeSpec(contrast=0.1),
    )
    assert project.refine("less contrast").contrast == 0.1
    assert seen["instruction"] == "less contrast"
    assert seen["stats"] == STATS and seen["spec"].contrast == 0.4
    assert len(project.history) == 2


# ---- editing and history --------------------------------------------------


def test_set_spec_pushes_history_and_saves(project, tmp_path):
    """The slider path: a direct set is an edit like any other, so undo works."""
    project.set_spec(GradeSpec(contrast=0.4))
    project.set_spec(GradeSpec(contrast=0.5))

    assert [s.contrast for s in project.history] == [0.4, 0.5]  # oldest first
    assert project.spec.contrast == 0.5
    assert Project.open(tmp_path / "proj").spec.contrast == 0.5


def test_history_is_a_copy_not_the_live_list(project):
    project.set_spec(GradeSpec(contrast=0.4))
    project.history.append(GradeSpec(contrast=9.0))
    assert len(project.history) == 1


def test_undo_steps_back_through_every_grade(project, tmp_path):
    assert project.can_undo is False        # nothing graded yet

    project.set_spec(GradeSpec(contrast=0.4))
    assert project.can_undo is True         # a single grade IS undoable
    project.set_spec(GradeSpec(contrast=0.5))

    assert project.undo() is True
    assert project.spec.contrast == 0.4
    assert Project.open(tmp_path / "proj").spec.contrast == 0.4  # the undo persisted

    # undoing the first grade lands on the ungraded clip, and persists
    assert project.undo() is True
    assert project.is_planned is False
    assert len(project.history) == 0
    assert Project.open(tmp_path / "proj").is_planned is False

    # only now is there nothing left; it refuses rather than raising
    assert project.undo() is False


# ---- interop --------------------------------------------------------------


def test_to_dict_is_json_serializable_and_describes_the_whole_ui_state(project, tmp_path):
    import json

    before = project.to_dict()
    assert before["spec"] is None and before["history_depth"] == 0
    assert before["can_undo"] is False

    project.set_spec(GradeSpec(contrast=0.4))
    project.set_spec(GradeSpec(contrast=0.5))
    d = project.to_dict()

    assert d["source"] == str((tmp_path / "clip.mp4").resolve())
    assert d["root"] == str(tmp_path / "proj")
    assert d["spec"]["contrast"] == 0.5
    assert d["history_depth"] == 2 and d["can_undo"] is True
    assert d["stats"]["saturation"] == 0.25
    assert d["cube"] == str(project.cube_path)
    assert d["preview"] == str(project.preview_path)
    assert json.loads(json.dumps(d)) == d  # a UI has to be able to send this


def test_repr_names_the_clip_and_the_depth(project):
    project.set_spec(GradeSpec(contrast=0.4))
    assert repr(project) == "<Project clip.mp4 grades=1>"


# ---- unplanned project ----------------------------------------------------
# A GUI can press Export before Grade; the CLI never can, because every command
# path loads a session that already holds at least one spec. Without the guard
# on Project.spec these surface a bare IndexError from the guts of Session.


def test_unplanned_project_raises_typed_error_not_indexerror(project, tmp_path):
    from ragvid import NoGrade, RagvidError

    assert project.is_planned is False
    for label, call in (
        ("spec", lambda: project.spec),
        ("bake", project.bake),
        ("preview", project.preview),
        ("export", lambda: project.export(tmp_path / "out.mp4")),
    ):
        with pytest.raises(NoGrade) as excinfo:
            call()
        assert isinstance(excinfo.value, RagvidError), label
        assert "plan one" in str(excinfo.value), label


def test_spec_is_reachable_once_planned(project, tmp_path):
    project.plan_from_reference(tmp_path / "ref.png")
    assert project.is_planned is True
    assert project.spec is not None


def test_export_does_not_share_its_lut_with_live_rendering(project, tmp_path, monkeypatch):
    """Export must bake to a private LUT, not the shared cube_path.

    Anything that renders a frame re-bakes cube_path in place, so sharing it
    means a scrubber move or slider drag mid-export hands ffmpeg a different
    grade -- and the finished file matches neither what the user approved nor
    what they asked for. This is a real bug that shipped and was measured
    (saturation 2.5 approved, slider nudged to 0, greyscale file out).
    """
    from ragvid.spec import GradeSpec

    project.set_spec(GradeSpec(saturation=2.5))
    seen = {}

    def fake_render(video, cube, out, effects=None, gpu=False, progress=None, input_lut=None):
        seen["cube"] = Path(cube)
        seen["exists_during"] = Path(cube).is_file()
        Path(out).write_bytes(b"x")
        return str(out)

    monkeypatch.setattr("ragvid.render.render_video", fake_render)
    project.export(tmp_path / "out.mp4")

    assert seen["exists_during"], "the LUT must exist while ffmpeg reads it"
    assert seen["cube"] != project.cube_path, (
        f"export baked to the shared {project.cube_path.name}; a concurrent "
        "frame render would overwrite it mid-encode"
    )
    assert not seen["cube"].exists(), "the private LUT should be cleaned up after"


def test_bake_escalates_lut_size_for_hue_qualifiers(project, tmp_path):
    """Project.bake is the ONLY route the app takes to a cube, so a hard-coded
    size here made the 65^3 escalation dead everywhere but the tests."""
    from ragvid.spec import HueBand

    p = project
    p.set_spec(GradeSpec(hue_green=HueBand(sat=0.3)))
    assert _lut_size(p.bake()) == 65
    p.set_spec(GradeSpec(saturation=1.3))
    assert _lut_size(p.bake()) == 33


def _lut_size(path) -> int:
    for line in Path(path).read_text().splitlines():
        if line.startswith("LUT_3D_SIZE"):
            return int(line.split()[1])
    raise AssertionError("no LUT_3D_SIZE")


def test_available_providers_tracks_the_catalog(monkeypatch):
    """The documented dropdown helper must not drift from the real catalog.

    It returned a hard-coded ("groq", "anthropic") and kept returning it after
    nine more providers landed, so a UI built on docs/ARCHITECTURE.md's
    "Provider dropdown -> available_providers()" row offered two of eleven.
    """
    from ragvid import available_providers
    from ragvid.providers.base import CATALOG

    got = list(available_providers())
    assert set(got) >= set(CATALOG), "a catalogued provider is missing from the dropdown"

    # `custom` appears only when there is somewhere for it to point.
    monkeypatch.delenv("RAGVID_BASE_URL", raising=False)
    assert "custom" not in available_providers()
    monkeypatch.setenv("RAGVID_BASE_URL", "http://localhost:1234/v1")
    assert "custom" in available_providers()
