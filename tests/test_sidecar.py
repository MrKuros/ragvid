"""The two sidecars written beside an export.

A .cube cannot hold EffectSpec, so a look that leaves ragvid as a cube alone
loses grain/vignette/glow with nothing to say so. look.json is the lossless
record; the .cdl is the portable one and must NAME what it could not carry.
Nothing here touches a provider or ffmpeg.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from ragvid.compiler import compile_stack
from ragvid.errors import InputError
from ragvid.intent import Intent, Op
from ragvid.probe import ClipStats, _stats_from_frames
from ragvid.project import Project
from ragvid.sidecar import read_look, write_cdl, write_look
from ragvid.spec import RGB, EffectSpec, GradeSpec, HueBand

NS = "{urn:ASC:CDL:v1.2}"

# Deliberately off identity in every category the CDL cannot express, so the
# round-trip and the drop list both have something real to fail on.
RICH = GradeSpec(
    slope=RGB(r=1.2, g=1.0, b=0.85),
    offset=RGB(r=0.02, g=0.0, b=-0.01),
    power=RGB(r=0.95, g=1.0, b=1.1),
    saturation=1.35,
    temperature=1200.0,
    tint=-0.2,
    contrast=0.35,
    exposure=0.5,
    look_mix=0.8,
    highlight_rolloff=0.3,
    shadow_tint=RGB(r=0.0, g=0.02, b=0.06),
    highlight_lift=0.03,
    hue_blue=HueBand(sat=0.6, lum=-0.05),
    effects=EffectSpec(grain=0.4, vignette=0.25),
    rationale="cold night exterior",
)


def _cc(path) -> ET.Element:
    return ET.parse(path).getroot().find(f"{NS}ColorDecision/{NS}ColorCorrection")


def _floats(el) -> list[float]:
    return [float(v) for v in el.text.split()]


# ---- look.json ------------------------------------------------------------


def test_look_json_round_trips_the_spec_field_for_field(tmp_path):
    """The whole justification for the file: everything the cube drops survives."""
    write_look(RICH, tmp_path / "out.look.json")
    assert read_look(tmp_path / "out.look.json")[0].base == RICH
    assert read_look(tmp_path / "out.look.json")[0].is_flat, "no regions were asked for"


def test_look_json_header_names_what_the_cube_does_not_carry(tmp_path):
    raw = json.loads(Path(write_look(RICH, tmp_path / "o.look.json")).read_text())
    assert raw["format"] == "ragvid-look" and raw["version"] == 3
    assert "effects" in raw["note"] and "3D LUT" in raw["note"]
    assert "layers" not in raw, "a flat grade writes no layer list"
    assert raw["spec"]["effects"]["grain"] == 0.4


# ---- .cdl -----------------------------------------------------------------


def test_cdl_is_valid_xml_with_the_asc_element_names(tmp_path):
    cc = _cc(write_cdl(RICH, tmp_path / "out.cdl"))
    assert cc is not None, "no ColorDecision/ColorCorrection"
    assert cc.get("id") == "out"
    assert [e.tag for e in cc.find(f"{NS}SOPNode")] == [
        f"{NS}Slope", f"{NS}Offset", f"{NS}Power",
    ]
    assert _floats(cc.find(f"{NS}SatNode/{NS}Saturation")) == [1.35]


def test_exposure_is_folded_into_slope_not_dropped(tmp_path):
    """spec.py applies exposure immediately before the CDL, so
    (x * 2**E) * slope + offset == x * (slope * 2**E) + offset exactly."""
    spec = GradeSpec(slope=RGB(r=1.5, g=1.0, b=0.5), offset=RGB.of(0.1), exposure=1.0)
    cc = _cc(write_cdl(spec, tmp_path / "e.cdl"))
    assert _floats(cc.find(f"{NS}SOPNode/{NS}Slope")) == [3.0, 2.0, 1.0]
    assert _floats(cc.find(f"{NS}SOPNode/{NS}Offset")) == [0.1, 0.1, 0.1]
    assert _floats(cc.find(f"{NS}SOPNode/{NS}Power")) == [1.0, 1.0, 1.0]


def test_the_folded_slope_reproduces_what_apply_actually_does(tmp_path):
    """The fold checked against the renderer, not against my own arithmetic:
    for a CDL-only grade, folding exposure into slope must be pixel-identical."""
    import numpy as np

    spec = GradeSpec(slope=RGB(r=1.5, g=1.0, b=0.5), offset=RGB.of(0.1),
                     power=RGB.of(0.9), exposure=0.7)
    slope = _floats(_cc(write_cdl(spec, tmp_path / "f.cdl")).find(f"{NS}SOPNode/{NS}Slope"))
    folded = spec.model_copy(update={
        "slope": RGB(r=slope[0], g=slope[1], b=slope[2]), "exposure": 0.0})
    grid = np.random.default_rng(0).random((512, 3))
    assert np.allclose(spec.apply(grid), folded.apply(grid), atol=1e-9)


def test_description_names_every_field_the_cdl_cannot_carry(tmp_path):
    text = _cc(write_cdl(RICH, tmp_path / "d.cdl")).find(f"{NS}Description").text
    for field in ("temperature", "tint", "contrast", "look_mix", "highlight_rolloff",
                  "shadow_tint.b", "highlight_lift", "hue_blue.sat", "effects.grain",
                  "effects.vignette"):
        assert field in text, f"{field} was dropped silently"
    assert "cold night exterior" in text
    assert "look.json" in text


def test_description_reports_exposure_as_folded_never_as_dropped(tmp_path):
    text = _cc(write_cdl(RICH, tmp_path / "x.cdl")).find(f"{NS}Description").text
    dropped = text.split("NOT represented here:")[1]
    assert "exposure" not in dropped
    assert "folded into slope" in text


def test_identity_grade_claims_no_loss(tmp_path):
    text = _cc(write_cdl(GradeSpec.identity(), tmp_path / "i.cdl")).find(f"{NS}Description").text
    assert "NOT represented" not in text
    assert "Nothing in this grade is lost" in text


def test_a_new_spec_field_shows_up_in_the_drop_list_automatically(tmp_path):
    """The drop list is diffed against identity, not hand-written, so it cannot
    quietly stop being true when GradeSpec grows a field."""
    from ragvid.sidecar import _dropped

    assert _dropped(GradeSpec.identity()) == []
    assert _dropped(GradeSpec(pivot=0.5)) == ["pivot=0.500000"]


# ---- export wiring --------------------------------------------------------

STATS = ClipStats(mean=RGB(r=0.2, g=0.3, b=0.4), std=RGB.of(0.1), saturation=0.25,
                  frames_sampled=10, width=640, height=360, duration=4.0)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project whose probe and render are both faked -- no ffmpeg, no network."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr("ragvid.probe.probe_video",
                        lambda path, n_frames=10, input_lut=None: STATS)
    monkeypatch.setattr(
        "ragvid.render.render_video",
        lambda video, cube, out, effects=None, gpu=False, progress=None,
        input_lut=None, layers=None: Path(out).write_bytes(b"x"),
    )
    p = Project.create(video, root=tmp_path / "proj")
    p.set_spec(RICH)
    return p


def test_export_writes_the_video_and_both_sidecars(project, tmp_path):
    out = project.export(tmp_path / "graded.mp4")
    assert out.is_file()
    look = tmp_path / "graded.look.json"
    cdl = tmp_path / "graded.cdl"
    assert look.is_file() and cdl.is_file()
    assert read_look(look)[0].base == RICH
    assert _floats(_cc(cdl).find(f"{NS}SOPNode/{NS}Slope"))[0] == pytest.approx(1.2 * 2 ** 0.5)


# ---- regions: look.json is now the ONLY lossless format --------------------
#
# A .cube is a per-pixel colour map and a region is an address, so a regional
# grade cannot round-trip through one at all. That is why A8 shipped first.


def _regional_stack():
    from ragvid.region import GradeStack, Layer, for_target

    return GradeStack(base=RICH, layers=[
        Layer(region=for_target("top"),
              spec=GradeSpec(exposure=-0.35, rationale="darkened the top")),
        Layer(region=for_target("center"), spec=GradeSpec(saturation=1.4)),
    ])


def test_look_json_round_trips_the_regions_too(tmp_path):
    """The whole justification for the file, one category larger than before."""
    stack = _regional_stack()
    write_look(stack, tmp_path / "r.look.json")
    assert read_look(tmp_path / "r.look.json")[0] == stack


def test_the_layer_geometry_survives_field_for_field(tmp_path):
    """A region that reloads with a different softness is a different picture,
    and nothing in the file would say so."""
    stack = _regional_stack()
    back, _ = read_look(write_look(stack, tmp_path / "r.look.json"))
    for got, want in zip(back.layers, stack.layers):
        assert got.region.model_dump() == want.region.model_dump()
        assert got.spec == want.spec


def test_a_bare_spec_still_writes_a_flat_look(tmp_path):
    """Most grades are flat and a caller holding one correction should not have
    to wrap it."""
    assert read_look(write_look(RICH, tmp_path / "f.look.json"))[0].base == RICH


def test_the_cdl_names_the_regions_it_cannot_carry(tmp_path):
    """A CDL is one correction for every pixel by definition. A silent drop here
    is the same data-loss bug as the cube's missing effects."""
    desc = _cc(write_cdl(_regional_stack(), tmp_path / "r.cdl")).find(f"{NS}Description").text
    assert "region grade" in desc
    assert "darkened the top" in desc
    assert "look.json" in desc


def test_the_cdl_of_a_regional_grade_still_carries_the_base(tmp_path):
    """Naming what it dropped must not stop it exporting what it can."""
    cc = _cc(write_cdl(_regional_stack(), tmp_path / "r.cdl"))
    assert _floats(cc.find(f"{NS}SOPNode/{NS}Slope"))[0] == pytest.approx(1.2 * 2 ** 0.5)


# ---- a look that survives being picked by a human --------------------------
#
# read_look parses a file the USER chose, so every failure here is something a
# picker has to explain rather than a traceback: same funnel Session.load uses.


@pytest.mark.parametrize("body,why", [
    ("not json at all", "garbage"),
    ('{"format": "ragvid-look", "version": 3}', "no spec key"),
    ('{"format": "ragvid-look", "version": 3, "spec": {"saturation": "blue"}}', "bad spec"),
    ('{"format": "ragvid-look", "version": 3, "spec": {}, "layers": [{"nope": 1}]}', "bad layer"),
    ('["a", "list"]', "not an object"),
    ('{"format": "resolve-powergrade", "version": 1, "spec": {}}', "another tool's file"),
    ('{"format": "ragvid-look", "version": 99, "spec": {}}', "from the future"),
])
def test_an_unreadable_look_is_an_input_error_not_a_traceback(tmp_path, body, why):
    """400 and reopen the picker, not 500. Every one of these reached the caller
    as a JSONDecodeError, a KeyError or a pydantic ValidationError before."""
    p = tmp_path / "bad.look.json"
    p.write_text(body)
    with pytest.raises(InputError) as exc:
        read_look(p)
    assert exc.value.path == str(p) and exc.value.reason


def test_a_missing_look_file_is_an_input_error_too(tmp_path):
    with pytest.raises(InputError):
        read_look(tmp_path / "nothing.look.json")


def test_a_look_from_the_future_names_its_version(tmp_path):
    """Rejected rather than best-guessed: a key this reader does not know about
    is a part of the look it would drop silently, which is the data-loss bug
    this module exists to close."""
    p = tmp_path / "v99.look.json"
    p.write_text(json.dumps({"format": "ragvid-look", "version": 99,
                             "spec": json.loads(RICH.model_dump_json())}))
    with pytest.raises(InputError, match="99"):
        read_look(p)


# ---- the intent rides along, because it is the only portable part ----------


INTENT = Intent(ops=[Op(op="warmth", dir="up", amount="moderate"),
                     Op(op="contrast", dir="up", amount="moderate")])


def test_the_intent_round_trips(tmp_path):
    back = read_look(write_look(RICH, tmp_path / "i.look.json", INTENT))[1]
    assert back == INTENT


def test_a_look_with_no_intent_reads_back_as_none(tmp_path):
    """A photo match or a hand-edited spec has no verbs behind it, and None is
    the honest answer rather than an empty Intent that would compile to
    identity and claim the grade did nothing."""
    assert read_look(write_look(RICH, tmp_path / "n.look.json"))[1] is None


def test_a_version_2_look_still_loads_and_has_no_intent(tmp_path):
    """Hand-written rather than produced by this code, because the point is a
    file THIS version can no longer write. Nothing to migrate: no `intent` key
    and no verbs behind the grade are the same grade."""
    p = tmp_path / "v2.look.json"
    p.write_text(json.dumps({
        "format": "ragvid-look", "version": 2, "note": "old",
        "spec": json.loads(RICH.model_dump_json()),
        "layers": [json.loads(l.model_dump_json()) for l in _regional_stack().layers],
    }))
    stack, intent = read_look(p)
    assert intent is None
    assert stack.base == RICH and len(stack.layers) == 2


def test_export_carries_the_intent_into_the_sidecar(project, tmp_path):
    """The wiring, end to end: the Project holds the Intent, and before this it
    was written into session.json and nowhere a second machine could read it."""
    project.set_intent(INTENT)
    project.export(tmp_path / "graded.mp4")
    assert read_look(tmp_path / "graded.look.json")[1] == INTENT


# ---- the measured claim: a look is portable, its numbers are not -----------
#
# Every number in `spec` was derived from the exported clip's ClipStats, the
# auto-balance cast correction included. Handing them to differently lit footage
# applies one clip's ANSWERS to another clip's question -- a LUT copy wearing a
# better name. The verbs carry no measurement, so they survive the trip.
#
# Measured on PIXELS. The reference for "what the sentence means on clip B" is
# not clip B graded through the intent (that would be circular) but the SAME
# SENTENCE on cast-free footage: auto-balance exists to make those two the same
# picture, so the distance to it is exactly how much of clip A's lighting leaked
# across. `dist` is on measurements, not on spec fields.

_x = np.random.default_rng(0)
_RAMP = np.clip(_x.uniform(0.12, 0.68, (20000, 1)) + _x.normal(0.0, 0.05, (20000, 3)), 0.06, 0.76)


def _cast(img, gain):
    return np.clip(img * np.array(gain, dtype=np.float64), 0.0, 1.0)


def _stats(img) -> ClipStats:
    return _stats_from_frames([np.round(img * 255.0).astype(np.uint8)], 640, 360, 4.0)


def _graded(img):
    return compile_stack(INTENT, _stats(img), balance=True).base.apply(img)


def _warmth(o):
    return float(o[:, 0].mean() - o[:, 2].mean())


def test_the_intent_in_a_look_beats_its_numbers_on_a_differently_cast_clip(tmp_path):
    """The whole justification for the `intent` key, on real pixels.

    Clip A is green-fluorescent, clip B is tungsten -- opposite casts, so clip
    A's balance correction is roughly clip B's cast again. Measured:

        the sentence on cast-free footage   R-B  0.0158   hue_strength 0.0150
        clip B through the look's INTENT    R-B  0.0195   hue_strength 0.0186
        clip B through the look's NUMBERS   R-B  0.0856   hue_strength 0.0875

    The copied numbers overshoot the warmth the sentence asked for by 5.4x, and
    land 10.2x further from it in RMSE (0.0355 against 0.0035). The numbers path
    is exactly what a version-2 look could do, so this is the before/after.
    """
    clip_a = _cast(_RAMP, (1.0, 1.06, 1.0))     # green fluorescent
    clip_b = _cast(_RAMP, (1.06, 1.0, 0.94))    # tungsten
    assert _stats(clip_a).hue_strength > 0.02 and _stats(clip_b).hue_strength > 0.02, \
        "the fixtures must actually be cast, and differently"

    stack_a = compile_stack(INTENT, _stats(clip_a), balance=True)
    look = write_look(stack_a, tmp_path / "a.look.json", INTENT)
    stack_back, intent_back = read_look(look)

    target = _graded(_RAMP)                            # the sentence, no cast to fight
    via_intent = compile_stack(intent_back, _stats(clip_b), balance=True).base.apply(clip_b)
    via_numbers = stack_back.base.apply(clip_b)        # what a version-2 look can do

    rmse = lambda o: float(np.sqrt(((o - target) ** 2).mean()))
    assert rmse(via_intent) < 0.25 * rmse(via_numbers), (
        f"intent {rmse(via_intent):.4f} vs numbers {rmse(via_numbers):.4f}")
    # ... and in the one moment the sentence actually names.
    assert abs(_warmth(via_intent) - _warmth(target)) < 0.25 * abs(
        _warmth(via_numbers) - _warmth(target)), (
        f"R-B target {_warmth(target):.4f}, intent {_warmth(via_intent):.4f}, "
        f"numbers {_warmth(via_numbers):.4f}")
    # The cast clip A was lit under must not survive into clip B's grade.
    assert _stats(via_intent).hue_strength < 0.5 * _stats(via_numbers).hue_strength


def test_apply_look_takes_the_intent_path_when_there_is_one(project, tmp_path):
    """Project.apply_look is where server.py's three lines land. An Intent goes
    through set_intent, so the grade is re-derived from THIS clip's stats."""
    look = write_look(RICH, tmp_path / "p.look.json", INTENT)
    project.apply_look(look)
    assert project.intent == INTENT
    assert project.spec != RICH, "the numbers were re-compiled, not copied"


def test_apply_look_without_an_intent_flattens_and_says_nothing_it_cannot(project, tmp_path):
    """Honest degradation, in refine_spec's terms: 44 numbers can only describe
    the whole frame, so the regional layers go. The alternative on such a file
    is not applying it at all."""
    look = write_look(_regional_stack(), tmp_path / "f2.look.json")
    project.apply_look(look)
    assert project.intent is None
    assert project.layers == []
    assert project.spec == RICH.sanitize()


@pytest.mark.parametrize("name", ["nope.json", "look.cube"])
def test_apply_look_refuses_what_it_cannot_read(project, tmp_path, name):
    """Checked at the moment the file is picked, the way _check_lut is."""
    if name.endswith(".cube"):
        (tmp_path / name).write_text("LUT_3D_SIZE 2\n")
    with pytest.raises(InputError):
        project.apply_look(tmp_path / name)
