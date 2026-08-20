"""Conversational refinement: adjust an existing GradeSpec by a plain-English request."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragvid.spec import GradeSpec
from ragvid.vibe import SYSTEM, format_stats

if TYPE_CHECKING:
    from ragvid.probe import ClipStats

REFINE_RULES = """\

YOU ARE NOW REFINING AN EXISTING GRADE, NOT CREATING ONE.
You will be given the current spec as JSON plus a short adjustment request. Return the
FULL spec — every field — not a diff and not a patch.

- Copy every field of the current spec through unchanged, except the ones the request
  actually implies. Fields the user did not mention must come back byte-for-byte
  identical. Never silently reset a field to identity.
- Change the fields you do touch by a perceptible but modest amount: roughly one nudge
  of a colorist's wheel, not a re-grade. "Less blue" moves temperature a few hundred
  Kelvin warmer, it does not jump to +2000.
- Apply the change relative to the CURRENT values, and stay inside the sane ranges. If a
  field is already at the edge of its range and the user asks for more of it, move it
  only slightly and say so in the rationale.
- If the request is vague ("better", "more cinematic", "less flat"), make one small
  coherent change rather than several large ones.
- Update `rationale` to describe the adjusted look."""


def refine_spec(
    current: GradeSpec, instruction: str, stats: "ClipStats", provider=None
) -> GradeSpec:
    """Return `current` adjusted by `instruction`, still calibrated to this clip."""
    if provider is None:
        from ragvid.providers import get_provider

        provider = get_provider()

    user = (
        f"{format_stats(stats)}\n\n"
        "The CURRENT grade spec, already applied to this clip:\n"
        f"{current.model_dump_json(indent=2)}\n\n"
        f'The user\'s adjustment request: "{instruction}"\n\n'
        "Return the full modified spec."
    )
    return provider.plan(SYSTEM + REFINE_RULES, user).sanitize()
