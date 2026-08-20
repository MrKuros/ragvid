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

THE COPY RULE, WHICH MATTERS MORE THAN EVERY OTHER RULE HERE. The spec has 43 numbers
and the request is about one or two of them. Work like this:

  1. Transcribe the current spec you were given, verbatim, field for field.
  2. Change ONLY the one or two fields the request actually names or implies.
  3. Rewrite `rationale`.

Every other number must come back byte-for-byte identical to the JSON you were handed —
same value, same sign, same digits. Do NOT round it, do NOT tidy it, and above all do
NOT reset it to its identity value because it looks untouched: a field sitting at a
non-identity number is there on purpose, from an earlier turn the user already accepted.
Silently zeroing one is the failure mode of this task. If a field is not part of the
request, you are copying, not deciding. When in doubt, copy.

This applies with full force to the fields that are easy to forget because they are
nested or usually zero: shadow_tint, highlight_tint, shadow_lift, highlight_lift, the
six hue bands (hue_red, hue_yellow, hue_green, hue_cyan, hue_blue, hue_magenta), every
key inside effects, exposure, highlight_rolloff, look_mix and pivot. If the current spec
has effects.grain 0.25 and the user says "cooler", the answer still has grain 0.25.

- Change the fields you do touch by a perceptible but modest amount: roughly one nudge
  of a colorist's wheel, not a re-grade. "Less blue" moves temperature a few hundred
  Kelvin warmer, it does not jump to +2000.
- Apply the change relative to the CURRENT values, and stay inside the sane ranges. If a
  field is already at the edge of its range and the user asks for more of it, move it
  only slightly and say so in the rationale.
- If the request is vague ("better", "more cinematic", "less flat"), make one small
  coherent change rather than several large ones.
- "Brighter" is one field, not four: move exposure (or slope), and if that takes exposure
  past +0.3 or slope past 1.1, raise highlight_rolloff with it to protect the highlights.
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
