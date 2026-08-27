"""Sidecars written next to an export: the lossless record and the portable one.

A `.cube` is a per-pixel colour map, so `EffectSpec` (denoise, glow, softness,
grain, vignette, fringe) cannot be in it — 6 of the spec's 44 numbers are
spatial and live only in render.py's filter chain. Nor can a REGION: a lookup
table indexed by colour has nowhere to put an address, so a per-region grade
(roadmap B1) is the same category of loss one step larger. Hand someone the
cube alone and all of it vanishes with nothing to say it ever existed. That is
the bug these two files close, from opposite ends:

    look.json   the whole GradeStack verbatim — base spec plus every
                (region, spec) layer — AND the Intent it was compiled from.
                Lossless, ragvid-only, and the only thing that round-trips. The
                numbers describe the clip they were measured on; the Intent is
                the part that travels to a different one (see `write_look`).
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

from .errors import InputError
from .intent import Intent
from .region import GradeStack, Layer
from .spec import GradeSpec

LOOK_FORMAT = "ragvid-look"
# 2 added `layers`. Version 1 files still read: they have no regions, which is
# exactly what an empty layer list means, so there is nothing to migrate.
# 3 added `intent`. Versions 1-2 still read: no key means no Intent, which is
# the same honest None a photo match or a hand-edited spec produces.
LOOK_VERSION = 3

CUBE_NOTE = (
    "Lossless record of the grade. The accompanying .cube carries the 38 colour "
    "fields of the BASE grade only; `effects` (denoise, glow, softness, grain, "
    "vignette, fringe) and `layers` (per-region grades) are spatial and cannot "
    "exist in a 3D LUT, so they are applied as an ffmpeg filter chain at render "
    "time and are present in this file alone."
)

# Everything ASC CDL can actually express. `exposure` is here because it folds
# into slope exactly; `rationale` because it is prose, not a number.
_CDL_FIELDS = frozenset(
    ["saturation", "exposure", "rationale"]
    + [f"{f}.{c}" for f in ("slope", "offset", "power") for c in "rgb"]
)

_NS = "urn:ASC:CDL:v1.2"


def write_look(
    look: GradeStack | GradeSpec, path: str | Path, intent: Intent | None = None
) -> str:
    """Write the full grade as JSON. Read it back with `read_look`.

    Takes a bare GradeSpec too, and means the flat stack by it — most grades are
    flat and a caller holding one correction should not have to wrap it.

    `spec` stays the base grade under its original key so a version-1 reader
    keeps working; `layers` is additive and absent when there are none.

    `intent` is the only PORTABLE thing in the file. Every number in `spec` was
    derived from the source clip's measured ClipStats — including its
    auto-balance cast correction — so handing those numbers to a differently lit
    clip copies one clip's ANSWERS onto another clip's question. The verbs
    survive the trip because they contain no measurement: re-compiled against
    the new clip's stats they mean the same sentence and produce different
    numbers, which is the whole point of the intent path. None when the grade
    came from a photo match, a hand-edited spec or a direct-path provider.
    """
    stack = look if isinstance(look, GradeStack) else GradeStack(base=look)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "format": LOOK_FORMAT,
        "version": LOOK_VERSION,
        "note": CUBE_NOTE,
        "spec": json.loads(stack.base.model_dump_json()),
        "intent": json.loads(intent.model_dump_json()) if intent else None,
    }
    if stack.layers:
        body["layers"] = [json.loads(l.model_dump_json()) for l in stack.layers]
    out.write_text(json.dumps(body, indent=2))
    return str(out)


def read_look(path: str | Path) -> tuple[GradeStack, Intent | None]:
    """The inverse of `write_look`: `(stack, intent)`. `stack.base` is the spec.

    No format-version branching downwards: a version-1 file has no `layers` key
    and a grade with no regions has an empty one, and those are the same grade;
    a version-2 file has no `intent` key and a grade with no verbs behind it has
    None, and those are the same grade too. A version from the FUTURE is
    rejected rather than best-guessed — a key this reader does not know about is
    a part of the look it would drop silently, which is the exact data-loss bug
    this whole module exists to close.

    Everything here parses a file the user picked, so every failure is an
    InputError and a 400 rather than a traceback: this is the same funnel
    Session.load uses, for the same reason. `ValueError` covers both
    json's JSONDecodeError and pydantic's ValidationError.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
        if not isinstance(raw, dict) or raw.get("format") != LOOK_FORMAT:
            raise InputError(str(p), "not a ragvid look file")
        if int(raw.get("version") or 1) > LOOK_VERSION:
            raise InputError(
                str(p), f"look version {raw['version']} — written by a newer ragvid"
            )
        raw_intent = raw.get("intent")
        return (
            GradeStack(
                base=GradeSpec.model_validate(raw["spec"]),
                layers=[Layer.model_validate(l) for l in raw.get("layers") or []],
            ),
            Intent.model_validate(raw_intent) if raw_intent else None,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise InputError(str(p), f"unreadable look file ({exc})") from exc


def write_cdl(look: GradeStack | GradeSpec, path: str | Path) -> str:
    """Write an ASC CDL ColorDecisionList for the BASE grade.

    Only slope/offset/power/saturation map. Everything else ragvid does is
    listed by name and value in <Description> — a silent drop here is the same
    data-loss bug as the cube's missing effects, one file further along. A CDL
    is one correction for every pixel by definition, so regional layers are
    named and counted there rather than approximated into the base.
    """
    stack = look if isinstance(look, GradeStack) else GradeStack(base=look)
    spec = stack.base
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    slope = [v * 2.0 ** spec.exposure for v in spec.slope.as_array()]

    root = ET.Element("ColorDecisionList", xmlns=_NS)
    cc = ET.SubElement(ET.SubElement(root, "ColorDecision"), "ColorCorrection", id=out.stem)
    ET.SubElement(cc, "Description").text = _describe(stack)
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


def _describe(stack: GradeStack) -> str:
    spec = stack.base
    parts = []
    if spec.rationale:
        parts.append(f'ragvid look: "{spec.rationale}".')
    parts.append("ASC CDL carries slope/offset/power/saturation only.")
    if spec.exposure:
        parts.append(f"exposure {spec.exposure:+.3f} stops is folded into slope.")
    dropped = _dropped(spec)
    # Regions are not a field, so _dropped's identity diff cannot see them.
    # They are still the largest thing a CDL leaves behind, so they are named
    # here by the sentence that produced them.
    dropped += [f"region grade ({l.spec.rationale or l.region.shape})" for l in stack.layers]
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
