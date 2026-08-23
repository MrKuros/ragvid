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

import pytest

from ragvid.probe import ClipStats
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
    assert read_look(tmp_path / "out.look.json").base == RICH
    assert read_look(tmp_path / "out.look.json").is_flat, "no regions were asked for"


def test_look_json_header_names_what_the_cube_does_not_carry(tmp_path):
    raw = json.loads(Path(write_look(RICH, tmp_path / "o.look.json")).read_text())
    assert raw["format"] == "ragvid-look" and raw["version"] == 2
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
    assert read_look(look).base == RICH
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
    assert read_look(tmp_path / "r.look.json") == stack


def test_the_layer_geometry_survives_field_for_field(tmp_path):
    """A region that reloads with a different softness is a different picture,
    and nothing in the file would say so."""
    stack = _regional_stack()
    back = read_look(write_look(stack, tmp_path / "r.look.json"))
    for got, want in zip(back.layers, stack.layers):
        assert got.region.model_dump() == want.region.model_dump()
        assert got.spec == want.spec


def test_a_bare_spec_still_writes_a_flat_look(tmp_path):
    """Most grades are flat and a caller holding one correction should not have
    to wrap it."""
    assert read_look(write_look(RICH, tmp_path / "f.look.json")).base == RICH


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
