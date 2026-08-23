"""Sidecars written next to an export: the lossless record and the portable one.

A `.cube` is a per-pixel colour map, so `EffectSpec` (denoise, glow, softness,
grain, vignette, fringe) cannot be in it — 6 of the spec's 43 numbers are
spatial and live only in render.py's filter chain. Hand someone the cube alone
and those six vanish with nothing to say they ever existed. That is the bug
these two files close, from opposite ends:

    look.json   the whole GradeSpec verbatim. Lossless, ragvid-only, and the
                only thing that round-trips.
    .cdl        ASC CDL. Lossy — slope/offset/power/saturation are the only
                fields that map — but every grading tool on the planet reads
                it. What does not map is NAMED in the <Description>, which is
                part of the CDL schema and so survives the trip into Resolve.

Exposure is the one field that looks lossy and is not. spec.py applies it as
`x *= 2**exposure` immediately BEFORE the CDL (`through_saturation`, step 1),
so `(x * 2**E) * slope + offset == x * (slope * 2**E) + offset` exactly. It is
folded into the emitted slope rather than reported as dropped.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .spec import GradeSpec

LOOK_FORMAT = "ragvid-look"
LOOK_VERSION = 1

CUBE_NOTE = (
    "Lossless record of the grade. The accompanying .cube carries the 37 colour "
    "fields only; `effects` (denoise, glow, softness, grain, vignette, fringe) is "
    "spatial and cannot exist in a 3D LUT, so it is applied as an ffmpeg filter "
    "chain at render time and is present in this file alone."
)

# Everything ASC CDL can actually express. `exposure` is here because it folds
# into slope exactly; `rationale` because it is prose, not a number.
_CDL_FIELDS = frozenset(
    ["saturation", "exposure", "rationale"]
    + [f"{f}.{c}" for f in ("slope", "offset", "power") for c in "rgb"]
)

_NS = "urn:ASC:CDL:v1.2"


def write_look(spec: GradeSpec, path: str | Path) -> str:
    """Write the full spec as JSON. Read it back with `read_look`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "format": LOOK_FORMAT,
                "version": LOOK_VERSION,
                "note": CUBE_NOTE,
                "spec": json.loads(spec.model_dump_json()),
            },
            indent=2,
        )
    )
    return str(out)


def read_look(path: str | Path) -> GradeSpec:
    """The inverse of `write_look`. No format-version branching until there is
    a second version to branch on."""
    return GradeSpec.model_validate(json.loads(Path(path).read_text())["spec"])


def write_cdl(spec: GradeSpec, path: str | Path) -> str:
    """Write an ASC CDL ColorDecisionList.

    Only slope/offset/power/saturation map. Everything else ragvid does is
    listed by name and value in <Description> — a silent drop here is the same
    data-loss bug as the cube's missing effects, one file further along.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    slope = [v * 2.0 ** spec.exposure for v in spec.slope.as_array()]

    root = ET.Element("ColorDecisionList", xmlns=_NS)
    cc = ET.SubElement(ET.SubElement(root, "ColorDecision"), "ColorCorrection", id=out.stem)
    ET.SubElement(cc, "Description").text = _describe(spec)
    sop = ET.SubElement(cc, "SOPNode")
    ET.SubElement(sop, "Slope").text = _fmt(slope)
    ET.SubElement(sop, "Offset").text = _fmt(spec.offset.as_array())
    ET.SubElement(sop, "Power").text = _fmt(spec.power.as_array())
    sat = ET.SubElement(cc, "SatNode")
    ET.SubElement(sat, "Saturation").text = _fmt([spec.saturation])

    ET.indent(root)
    ET.ElementTree(root).write(str(out), encoding="utf-8", xml_declaration=True)
    return str(out)


def _fmt(values) -> str:
    # %.6f, the precision lut.py writes its table at, so the two artifacts of
    # one grade do not disagree in the seventh decimal.
    return " ".join(f"{float(v):.6f}" for v in values)


def _describe(spec: GradeSpec) -> str:
    parts = []
    if spec.rationale:
        parts.append(f'ragvid look: "{spec.rationale}".')
    parts.append("ASC CDL carries slope/offset/power/saturation only.")
    if spec.exposure:
        parts.append(f"exposure {spec.exposure:+.3f} stops is folded into slope.")
    dropped = _dropped(spec)
    if dropped:
        parts.append("NOT represented here: " + ", ".join(dropped) + ".")
        parts.append("The complete look is in the accompanying .look.json.")
    else:
        parts.append("Nothing in this grade is lost by the conversion.")
    return " ".join(parts)


def _dropped(spec: GradeSpec) -> list[str]:
    """Non-CDL fields that are off identity, as `field=value` strings.

    Diffed against `GradeSpec.identity()` rather than listed by hand: a hard-
    coded list silently stops being true the day a field is added to GradeSpec,
    and a stale honesty note is worse than no note.
    """
    identity = GradeSpec.identity().model_dump()
    out = []

    def walk(cur: dict, ref: dict, prefix: str = "") -> None:
        for k, v in cur.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(v, ref[k], f"{key}.")
            elif key not in _CDL_FIELDS and v != ref[k]:
                out.append(f"{key}={v:.6f}" if isinstance(v, (int, float)) else f"{key}={v!r}")

    walk(spec.model_dump(), identity)
    return out
