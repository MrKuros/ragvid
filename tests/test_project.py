"""The facade a GUI drives. No CLI, no argv, no cwd assumptions, no network.

Everything below patches probe_video/probe_image at their defining module,
because Project imports its dependencies inside the methods that use them --
deliberately, so a plain `spec` read never drags numpy or ffmpeg in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragvid.errors import InputError, SessionNotFound
from ragvid.probe import ClipStats
from ragvid.project import Project
from ragvid.session import Session
from ragvid.intent import Intent, Op
from ragvid.spec import LUMA, RGB, GradeSpec

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


# ---- refine: editing the verb list (roadmap B7) ----------------------------
#
# The defect these pin: refine used to hand the model 43 numbers and take 43
# back, and 43 numbers can only describe the whole frame -- so a regional grade
# lost its region on the next sentence. Judged by grading a real frame through
# the stack and comparing the masked part with the rest, because a layer still
# being in the session is not proof that any pixel of it moved.


class FakeSchemaProvider:
    """A provider that can constrain decoding, so Project.refine routes to the
    verb list.

    `plan` answers with a plausible flat warm spec instead of raising, on
    purpose: that is exactly what the 43-number path returned before this
    change, so the test below fails on the old code by MEASURING a flattened
    frame rather than by noticing which method got called.
    """

    name = "fake"
    schema_enforced = True

    def __init__(self, reply):
        self.reply = reply
        self.used: list[str] = []

    def plan(self, system, user):
        self.used.append("plan")
        return GradeSpec(temperature=800.0, rationale="Warmer.")

    def plan_json(self, system, user, schema):
        self.used.append("plan_json")
        return json.loads(self.reply.model_dump_json())


DARK_TOP = [Op(op="exposure", dir="down", amount="strong", target="top")]


def _top_vs_bottom(image) -> float:
    """Luma of the masked strip minus luma of the untouched one."""
    luma = image @ LUMA
    return float(luma[:8].mean() - luma[-8:].mean())


def test_a_regional_grade_survives_a_refine(project):
    """"darken the top", then "make it warmer". The region has to still be there
    afterwards, and the masked area still measurably darker than the rest."""
    frame = np.full((64, 64, 3), 0.5)
    project.set_intent(Intent(ops=list(DARK_TOP)), label="darken the top")
    first = project.stack.apply(frame)
    gap = _top_vs_bottom(first)
    assert gap < -0.05, "the region did nothing to begin with; nothing to preserve"

    p = FakeSchemaProvider(Intent(ops=DARK_TOP + [Op(op="warmth", dir="up")]))
    project.refine("make it warmer", provider=p)
    second = project.stack.apply(frame)

    # PIXELS FIRST, and in this order deliberately: on the pre-change code the
    # 43-number path answers with a flat spec and this measurement is what
    # catches it (the masked strip and the rest of the frame come back equal),
    # not the bookkeeping assertions below it.
    # Measured: the masked strip sits -0.1922 luma below the untouched one
    # before the refine and -0.1928 after; on the pre-change code it was 0.0000.
    assert _top_vs_bottom(second) < -0.05
    assert abs(_top_vs_bottom(second) - gap) < 0.02
    # ... and the refine actually happened, everywhere.
    warm = lambda im: float((im[..., 0] - im[..., 2]).mean())  # noqa: E731
    assert warm(second) > warm(first) + 0.01

    assert p.used == ["plan_json"]                       # the verbs, not the floats
    assert [(o.op, o.target) for o in project.intent.ops] == [
        ("exposure", "top"), ("warmth", "")]
    assert len(project.layers) == 1


def test_refine_falls_back_to_the_43_number_path_below_the_schema_rung(project):
    """A capability test, exactly as in plan_vibe. An endpoint that cannot
    constrain decoding still refines, and still flattens while doing it -- that
    is the documented deal, not a regression."""
    class Weak:
        name = "fake"
        schema_enforced = False

        def plan(self, system, user):
            return GradeSpec(temperature=800.0, rationale="Warmer.")

        def plan_json(self, system, user, schema):
            raise AssertionError("an endpoint without schema support must not be asked")

    project.set_intent(Intent(ops=list(DARK_TOP)), label="darken the top")
    project.refine("make it warmer", provider=Weak())

    assert project.spec.temperature == 800.0
    assert project.layers == []       # 43 numbers describe the whole frame
    assert project.intent is None     # ... and no list of verbs describes them


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

    def fake_render(video, cube, out, effects=None, gpu=False, progress=None,
                    input_lut=None, layers=None):
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


# ---- log footage ----------------------------------------------------------


def test_a_format_name_bakes_its_own_conversion(project, tmp_path):
    """The thing a person can actually answer is what they shot, not where the
    vendor's .cube is."""
    lut = project.set_input_lut("slog3")

    assert lut == str(tmp_path / "proj" / ".ragvid" / "log_slog3.cube")
    assert Path(lut).read_text().startswith('TITLE "ragvid slog3 to Rec.709"')
    assert project.input_format == "slog3"
    # Derived data: it lives with the rest of the project's artifacts and comes
    # back on reopen without anything being copied out of the way.
    assert Project.open(tmp_path / "proj").input_format == "slog3"


def test_a_vendor_cube_still_works_and_reports_no_format(project, tmp_path):
    """People who do have their camera's LUT must keep working unchanged."""
    cube = tmp_path / "vendor.cube"
    cube.write_text("LUT_3D_SIZE 2\n")

    assert project.set_input_lut(cube) == str(cube.resolve())
    assert project.input_format is None


def test_an_unknown_format_name_is_an_input_error(project):
    with pytest.raises(InputError):
        project.set_input_lut("slog9")
    assert project.input_lut is None


def test_setting_a_format_reprobes_through_the_generated_lut(project, monkeypatch):
    """Same rule as a vendor LUT: the cached stats have to describe the image
    the grade will land on."""
    seen = {}
    monkeypatch.setattr(
        "ragvid.probe.probe_video",
        lambda path, n_frames=10, input_lut=None: seen.update(lut=input_lut) or STATS,
    )
    project.set_input_lut("vlog")
    assert seen["lut"].endswith("log_vlog.cube")


def test_opening_a_clip_applies_a_detected_format_and_nothing_otherwise(clip, tmp_path,
                                                                       monkeypatch):
    """detect() answers None for most clips on purpose, and None must change
    nothing at all -- a wrong technical LUT is worse than no LUT."""
    monkeypatch.setattr("ragvid.logspace.detect", lambda path: "logc3")
    assert Project.create(clip, root=tmp_path / "a").input_format == "logc3"

    monkeypatch.setattr("ragvid.logspace.detect", lambda path: None)
    quiet = Project.create(clip, root=tmp_path / "b")
    assert quiet.input_format is None and quiet.input_lut is None

    # An explicit choice is never second-guessed by the metadata.
    monkeypatch.setattr("ragvid.logspace.detect", lambda path: "logc3")
    assert Project.create(clip, root=tmp_path / "c", input_lut="nlog").input_format == "nlog"


# ---- intent (roadmap C3/C4) ------------------------------------------------


def _intent(amount="moderate"):
    from ragvid.intent import Intent, Op

    return Intent(ops=[Op(op="warmth", dir="up", amount=amount)], strength="full")


def test_plan_from_vibe_keeps_the_verbs_beside_the_spec(project):
    """A schema-enforcing endpoint takes the intent path, and what it emitted
    has to survive the call -- it cannot be recovered from 43 floats."""
    class FakeIntentProvider:
        name = "fake"
        schema_enforced = True

        def plan(self, system, user):
            raise AssertionError("a schema endpoint must not take the direct path")

        def plan_json(self, system, user, schema):
            return {"ops": [{"op": "warmth", "dir": "up", "amount": "moderate",
                             "target": ""}], "strength": "full"}

    spec = project.plan_from_vibe("warmer", provider=FakeIntentProvider())
    assert spec.temperature > 0.0
    assert project.intent == _intent()
    # ... and out the other side of a save/load, which is what a UI reads back.
    assert Project.open(project.root).intent == _intent()


def test_set_intent_recompiles_rather_than_nudging_a_field(project):
    project.set_intent(_intent("subtle"))
    subtle = project.spec.temperature

    project.set_intent(_intent("strong"))
    assert project.spec.temperature > subtle       # the verb moved, on its own axis
    assert project.spec.saturation == 1.0          # and nothing else did
    assert project.intent.ops[0].amount == "strong"
    assert len(project.history) == 2               # every edit is one undo step


def test_undo_restores_the_intent_with_the_spec(project):
    project.set_intent(_intent("subtle"))
    project.set_intent(_intent("strong"))
    assert project.undo() is True
    assert project.intent == _intent("subtle")
    assert project.revert_to(-1) is True
    assert project.intent is None and project.is_planned is False


def test_a_raw_spec_has_no_verbs_behind_it(project):
    """set_spec must NOT carry the previous intent forward: 43 numbers are not
    described by any verb list, and a stale sentence claiming otherwise is
    exactly the drift the intent path exists to remove."""
    project.set_intent(_intent())
    project.set_spec(GradeSpec(contrast=0.4))
    assert project.intent is None
    assert project.to_dict()["intent"] is None


def test_the_intent_view_is_sentences_without_their_adverbs(project):
    """The magnitude is the control next to the sentence; printing it in the
    sentence as well means the two disagree for as long as a drag lasts."""
    from ragvid.project import intent_view

    project.set_intent(_intent("subtle"))
    view = project.to_dict()["intent"]
    assert view == {"strength": "full",
                    "ops": [{"op": "warmth", "dir": "up", "amount": "subtle",
                             "target": "", "text": "warmed it up"}]}
    assert intent_view(None) is None


def test_auto_balance_is_on_by_default_and_recompiles_when_switched(project):
    """The library keeps balance off so an empty Intent is the identity grade;
    the UI layer is where "on" belongs, and this is that layer."""
    # STATS has no measured hue_strength, which the balance pass correctly reads
    # as "unmeasured, correct nothing". Give it one, so there is a cast to remove.
    project.session.stats = STATS.model_copy(update={"hue_strength": 0.05})
    project.set_intent(_intent())
    assert project.auto_balance is True
    balanced = project.spec

    assert project.set_auto_balance(False) is True
    assert project.spec != balanced                # the switch is visible at once
    assert project.intent == _intent()             # ... and the verbs are untouched
    assert len(project.history) == 2

    assert project.set_auto_balance(False) is False   # no-op, no history step
    assert Project.open(project.root).auto_balance is False


def test_turning_balance_off_leaves_a_spec_that_has_no_verbs_alone(project):
    """There is nothing to re-compile from 43 numbers, and pretending otherwise
    would silently replace a grade the user made by hand."""
    project.set_spec(GradeSpec(contrast=0.4))
    assert project.set_auto_balance(False) is True
    assert project.spec.contrast == 0.4 and len(project.history) == 1


# ---- regions (roadmap B1) -------------------------------------------------
#
# The base spec and its regional layers are two halves of one grade, kept in
# parallel lists exactly as `labels` and `intents` are. Every test here exists
# because a parallel list is a chance to desync.


def _regional() -> "Intent":
    from ragvid.intent import Intent, Op

    return Intent(ops=[Op(op="warmth"), Op(op="exposure", dir="down", target="top")])


def test_a_regional_intent_lands_in_the_stack_not_in_the_spec(project):
    project.set_intent(_regional())
    assert len(project.layers) == 1
    assert project.stack.base == project.spec
    assert not project.stack.is_flat
    assert project.layers[0].region.edge == "top"


def test_a_flat_grade_has_no_layers_and_says_so(project):
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]))
    assert project.layers == [] and project.stack.is_flat


def test_layers_survive_a_save_and_reopen(project):
    project.set_intent(_regional())
    reopened = Project.open(project.root)
    assert reopened.stack == project.stack
    assert reopened.layers[0].region.model_dump() == project.layers[0].region.model_dump()


def test_undo_takes_the_layers_with_it(project):
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]))
    project.set_intent(_regional())
    assert len(project.layers) == 1
    project.undo()
    assert project.layers == [], "the layers of the undone grade are still on screen"
    assert len(project.session.layers) == len(project.session.specs) == 1


# ---- browsing the history instead of destroying it -------------------------
#
# `revert_to` truncates, so using it to LOOK at an old step threw away every
# step after it, with no redo -- undo is `session.pop`, so a truncation has
# nothing to undo it with. `restore` appends instead. These pin the difference.


def test_restoring_an_old_step_deletes_nothing(project):
    """The whole point. Three steps, go back to the first, and end with FOUR --
    the first one's grade is now also the last one's, and the two in between
    are exactly where they were."""
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]), label="warmer")
    project.set_intent(Intent(ops=[Op(op="contrast")]), label="punchier")
    project.set_intent(Intent(ops=[Op(op="grain")]), label="grainier")

    project.restore(0)

    labels = [s["label"] for s in project.steps]
    assert len(labels) == 4, labels
    assert labels[:3] == ["warmer", "punchier", "grainier"], "an earlier step went missing"
    assert labels[3] == "back to: warmer"
    # ...and the grade really is the old one, not merely a row saying so.
    assert project.spec == project.history[0]
    assert project.steps[3]["current"] is True


def test_a_restored_step_does_not_share_its_regions_with_the_original(project):
    """Region is a MUTABLE model. compiler._spare already documents the trap --
    "two layers sharing one would let an edit to either reach the other" -- and
    an aliased restore is that bug with the two ends a history apart. It would
    corrupt silently: nothing raises, the old step just quietly changes."""
    project.set_intent(_regional())
    project.set_intent(Intent(ops=[Op(op="contrast")]))
    before = project.session.layers[0][0].region.extent

    project.restore(0)
    project.session.layers[-1][0].region.extent = 0.123

    assert project.session.layers[0][0].region.extent == before, \
        "editing the restored step reached back into the original"


def test_deleting_a_step_removes_only_that_step(project):
    from ragvid.intent import Intent, Op

    for label in ("one", "two", "three"):
        project.set_intent(Intent(ops=[Op(op="warmth")]), label=label)

    assert project.delete_step(1) is True

    assert [s["label"] for s in project.steps] == ["one", "three"]
    # The four parallel arrays are the invariant this project keeps by
    # convention rather than by construction, so a new mutator has to be held
    # to it the same way revert and reset are, just below.
    n = len(project.session.specs)
    assert len(project.session.labels) == n
    assert len(project.session.intents) == n
    assert len(project.session.layers) == n
    assert project.delete_step(9) is False and project.delete_step(-1) is False


def test_deleting_the_current_step_lands_on_the_one_before_it(project):
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]), label="one")
    project.set_intent(Intent(ops=[Op(op="contrast")]), label="two")
    was = project.history[0]

    project.delete_step(1)

    assert project.spec == was
    assert [s["label"] for s in project.steps] == ["one"]


def test_every_step_carries_its_own_verbs_so_a_reader_need_not_restore_it(project):
    """A UI showing "what did step 2 do?" had nothing to answer with: `steps`
    carried only a label and a rationale, and `intent` was the CURRENT step's
    alone."""
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]))
    project.set_intent(Intent(ops=[Op(op="contrast"), Op(op="grain")]))

    steps = project.steps
    assert [o["op"] for o in steps[0]["intent"]["ops"]] == ["warmth"]
    assert [o["op"] for o in steps[1]["intent"]["ops"]] == ["contrast", "grain"]
    # ...and every op carries the English the row is labelled with.
    assert all(o["text"] for s in steps for o in s["intent"]["ops"])
    # A grade nobody compiled from verbs reports None, exactly as the current
    # one does -- one branch handles both.
    project.set_spec(GradeSpec(contrast=0.4))
    assert project.steps[-1]["intent"] is None


def test_a_tweak_says_it_is_one_so_the_history_can_fold_it_away(project):
    """A slider drag, an item switched off and the balance toggle each push a
    real step -- that is what makes them undoable -- but they are adjustments OF
    the prompt above them. Without a flag saying so, four words somebody typed
    end up buried under "adjusted, adjusted, adjusted".

    Asserts the BEHAVIOUR of each path rather than the TWEAK_LABELS tuple, so a
    default label that changes without the tuple is caught here.
    """
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]), label="cold but colorful")
    project.set_intent(Intent(ops=[Op(op="warmth", amount="strong")]))   # a slider
    project.set_spec(GradeSpec(contrast=0.2))                            # the direct slider
    project.set_auto_balance(not project.auto_balance)                   # the balance row

    flags = [(s["label"], s["tweak"]) for s in project.steps]
    assert flags[0] == ("cold but colorful", False), flags
    assert all(t for _, t in flags[1:]), f"a tweak is being listed as a prompt: {flags}"


def test_deleting_a_row_takes_the_tweaks_folded_into_it(project):
    """One row is one prompt plus its run of tweaks, so deleting the row deletes
    the run -- in one call, so it is one save rather than a race of N."""
    from ragvid.intent import Intent, Op

    project.set_intent(Intent(ops=[Op(op="warmth")]), label="first")
    project.set_intent(Intent(ops=[Op(op="warmth", amount="strong")]))   # tweak of it
    project.set_intent(Intent(ops=[Op(op="contrast")]), label="second")

    assert project.delete_step(0, 2) is True

    assert [s["label"] for s in project.steps] == ["second"]
    n = len(project.session.specs)
    assert len(project.session.labels) == len(project.session.intents) == n


def test_revert_and_reset_keep_the_parallel_lists_in_step(project):
    from ragvid.intent import Intent, Op

    project.set_intent(_regional())
    project.set_intent(Intent(ops=[Op(op="contrast")]))
    project.set_intent(_regional())
    project.revert_to(0)
    assert len(project.session.layers) == len(project.session.specs) == 1
    assert len(project.layers) == 1
    project.reset()
    assert project.session.layers == []


def test_a_hand_edited_spec_is_a_flat_grade(project):
    """43 numbers describe the whole frame, so a spec arriving that way IS the
    whole grade. Carrying layers forward would keep correcting a corner of an
    image nothing in the incoming spec knows about."""
    project.set_intent(_regional())
    project.set_spec(GradeSpec(contrast=0.4))
    assert project.layers == [] and project.stack.is_flat


def test_a_session_written_before_regions_still_opens(project, tmp_path):
    """Every field added after the first sessions were written keeps a default;
    `layers` is no different, and an older session's grades WERE flat."""
    import json

    project.set_spec(GradeSpec(contrast=0.2))
    path = Session.path(project.root)
    raw = json.loads(path.read_text())
    del raw["layers"]
    path.write_text(json.dumps(raw))
    reopened = Project.open(project.root)
    assert reopened.layers == [] and reopened.spec.contrast == 0.2


def test_to_dict_hands_a_ui_the_regions_it_has_to_draw(project):
    """A UI that ignores the layers -- a WebGL preview especially -- draws a
    picture the export will not produce."""
    project.set_intent(_regional())
    d = project.to_dict()
    assert d["layers"][0]["region"]["edge"] == "top"
    assert d["layers"][0]["spec"]["exposure"] < 0
    assert "darkened the top" in d["spec"]["rationale"].lower()


def test_bake_layers_writes_one_cube_and_one_mask_per_layer(project):
    """A region cannot bake into a .cube, so the export grows a per-region LUT
    plus a mask. The mask is at the SOURCE's resolution so one file serves the
    still, the contact sheet and the export."""
    from PIL import Image

    project.set_intent(_regional())
    made = project.bake_layers()
    assert len(made) == 1
    cube, mask = made[0]
    assert Path(cube).is_file() and Path(mask).is_file()
    assert Image.open(mask).size == (STATS.width, STATS.height)


def test_a_flat_grade_bakes_no_layer_artifacts_at_all(project):
    project.set_spec(GradeSpec(contrast=0.2))
    assert project.bake_layers() == []


def test_the_render_calls_are_handed_the_layers(project, tmp_path, monkeypatch):
    """The one thing a preview must never do is show a look the export cannot
    produce, so every render path has to receive the same layer list."""
    seen = {}

    def fake(name):
        def go(*a, **kw):
            seen[name] = kw.get("layers")
            out = Path(a[2])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"x")
            return str(out)
        return go

    monkeypatch.setattr("ragvid.render.render_preview", fake("preview"))
    monkeypatch.setattr("ragvid.render.render_frame", fake("frame"))
    monkeypatch.setattr("ragvid.render.render_video",
                        lambda video, cube, out, effects=None, gpu=False, progress=None,
                        input_lut=None, layers=None: seen.update(video=layers)
                        or Path(out).write_bytes(b"x"))
    project.set_intent(_regional())
    project.preview()
    project.frame(at=0.0)
    project.export(tmp_path / "out.mp4")
    assert [len(seen[k]) for k in ("preview", "frame", "video")] == [1, 1, 1]
    # ... and the ungraded half of a before/after compare gets none of it.
    project.preview(graded=False)
    assert seen["preview"] is None


# ---- the semantic-mask precondition ----------------------------------------
# Mocked at the `segment` boundary in every direction: no test here may install
# an extra, download 15 MB, or run inference.


def _sky() -> "Intent":
    from ragvid.intent import Intent, Op

    return Intent(ops=[Op(op="warmth"), Op(op="exposure", dir="down", target="sky")])


def test_a_semantic_grade_is_refused_before_it_reaches_the_history(project, monkeypatch):
    """The failure has to land BEFORE _push, not inside the render.

    Pushed first, the session held a grade that could not be drawn and the only
    escape was guessing at undo. So the assertions are about what did NOT
    happen: no spec, no layer, no label.
    """
    from ragvid.intent import Intent, Op
    from ragvid.segment import SegmentUnavailable

    monkeypatch.setattr("ragvid.segment.have_runtime", lambda: False)
    project.set_intent(Intent(ops=[Op(op="warmth")]), label="warmer")

    with pytest.raises(SegmentUnavailable) as caught:
        project.set_intent(_sky(), label="the sky moody")

    assert caught.value.needs_install is True
    assert "ragvid[masks]" in caught.value.hint
    assert len(project.history) == 1, "the unrenderable grade landed anyway"
    assert project.layers == []
    assert [s["label"] for s in project.steps] == ["warmer"]
    project.bake_layers()          # still renderable, which is the actual fix


def test_the_weights_being_absent_is_the_same_refusal_with_a_different_fix(
        project, monkeypatch, tmp_path):
    from ragvid.segment import SegmentUnavailable

    monkeypatch.setattr("ragvid.segment.have_runtime", lambda: True)
    monkeypatch.setattr("ragvid.segment.model_path", lambda: tmp_path / "absent.onnx")

    with pytest.raises(SegmentUnavailable) as caught:
        project.set_intent(_sky())
    assert caught.value.needs_install is False
    assert project.history == []


def test_the_guard_lets_a_semantic_grade_through_once_the_model_is_there(
        project, monkeypatch, tmp_path):
    """The guard must be exactly `needs_frame`, not "any semantic word ever".
    With the model present the grade lands like any other."""
    weights = tmp_path / "there.onnx"
    weights.write_bytes(b"not a real model; nothing here loads it")
    monkeypatch.setattr("ragvid.segment.have_runtime", lambda: True)
    monkeypatch.setattr("ragvid.segment.model_path", lambda: weights)

    project.set_intent(_sky())
    assert len(project.layers) == 1
    assert project.layers[0].region.shape == "semantic"


def test_a_geometric_region_never_asks_about_the_model(project, monkeypatch):
    """"the top" needs no model, and must not be gated on one -- the guard reads
    GradeStack.needs_frame rather than "does this grade have layers"."""
    def boom() -> bool:
        raise AssertionError("a geometric region asked whether segmentation is ready")

    monkeypatch.setattr("ragvid.segment.have_runtime", boom)
    project.set_intent(_regional())
    assert len(project.layers) == 1


def test_a_protect_that_names_a_thing_hits_the_same_refusal(project, monkeypatch):
    """B6's protect reaches the model through the back door: "darken the top but
    not the person" is a GEOMETRIC region whose `exclude` is semantic. The guard
    keys off GradeStack.needs_frame, which is recursive through `exclude`, so it
    covers this without knowing the verb exists -- verified, not assumed."""
    from ragvid.intent import Intent, Op
    from ragvid.segment import SegmentUnavailable

    monkeypatch.setattr("ragvid.segment.have_runtime", lambda: False)
    intent = Intent(ops=[Op(op="exposure", dir="down", target="top"),
                         Op(op="protect", target="person")])
    with pytest.raises(SegmentUnavailable):
        project.set_intent(intent)
    assert project.history == []
