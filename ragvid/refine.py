"""Conversational refinement: adjust a grade that already exists, in words.

TWO FUNCTIONS, AND WHAT DIFFERS IS THE SUBJECT, NOT THE TRANSPORT (roadmap B7).

`refine_intent` edits the VERB LIST. The model is handed the ops the user already
accepted -- "darkened the top, warmed it up" -- and returns an edited list, which
compiler.py re-compiles against the clip. Editing five words is dramatically more
tractable than editing forty-three floats, and it is the only version that can
keep a region: the region lives on the op's `target`, so copying the op copies
where it applies. It is also the only version that can be RELATIVE, because the
list is the memory of what the full-strength grade actually was -- the bake-off's
"warm it up, but at half strength" came back byte-identical to its own "warmer"
on the direct path for exactly that reason (vibe.py's docstring has the numbers).

`refine_spec` is the fallback, for an endpoint that cannot constrain decoding to
the Intent schema and for a grade that has no Intent behind it at all (a photo
match, a hand-edited spec). It hands the model all 44 numbers and takes 44 back,
so it FLATTENS: 44 numbers can only describe the whole frame, and any regional
layer is dropped. That is the defect the intent path exists to fix, and it stays
here because on those endpoints the alternative is no refinement at all.

MEASURED, not argued (one two-turn run of scripts/bakeoff_intent.py --refine,
gpt-oss-120b, six sentences on test_files/test.mp4, every second-turn grade
applied to real frames and re-measured): the verb list took 15/16 checks and 5/6
sentences clean at 4116 tokens a sentence; the 43-number path took 13/16 and 4/6
at 8332. The two it alone got right are the two a fresh set of 44 numbers cannot
express, both of them relative:

  * "a bit less warm" moved the direct path's measured warmth by -0.0040 -- it
    landed 89% of the way back at its own full-strength grade, i.e. it barely
    moved -- against -0.0112 and 40% for the verb list.
  * "a stop brighter" after "moody but keep it natural" cost the direct path
    0.077 of measured chroma while overshooting luma by +0.379, where the verb
    list flipped one op's direction and moved luma +0.097 with chroma flat.

And the region: "darken the top" then "make it warmer" kept its layer here (top
minus bottom luma -0.1236 against the source at turn one, -0.1230 after the
refine, one layer). The direct path had no layer to keep.

The older note said an Intent could not carry a refinement because it describes
departures from the SOURCE and compile_intent starts from identity. That is still
true and is not an obstacle: the current Intent IS the departure from the source,
so re-compiling an edited copy of it reproduces every move the user accepted
rather than discarding them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragvid.errors import ProviderError
from ragvid.intent import Intent
from ragvid.spec import HUE_FIELDS, GradeSpec
from ragvid.vibe import REFINE_INTENT_SYSTEM, SYSTEM, format_stats

if TYPE_CHECKING:
    from ragvid.probe import ClipStats

REFINE_RULES = """\

YOU ARE NOW REFINING AN EXISTING GRADE, NOT CREATING ONE.
You will be given the current spec as JSON plus a short adjustment request. Return the
FULL spec — every field — not a diff and not a patch.

THE COPY RULE, WHICH MATTERS MORE THAN EVERY OTHER RULE HERE. The spec has 44 numbers
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
key inside effects, exposure, highlight_rolloff, look_mix, pivot and contrast_balance. If the current spec
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
    out = provider.plan(SYSTEM + REFINE_RULES, user).sanitize()
    # Carry the hue rotations across by hand. They are deliberately absent from
    # GradeSpec.llm_json_schema (the direct path has no words for them, see
    # spec.llm_json_schema), so a spec that arrived with rotations -- graded on
    # the intent path, then refined after the provider was swapped for one that
    # cannot constrain decoding -- would come back with every one of them reset
    # to 0. A refine is supposed to adjust a grade, never to silently drop a
    # part of it the model was not asked about.
    return out.model_copy(update={
        f: getattr(out, f).model_copy(update={"rot": getattr(current, f).rot})
        for f in HUE_FIELDS
    })


def refine_intent(
    current: Intent, instruction: str, stats: "ClipStats", provider=None
) -> Intent:
    """Return `current` with `instruction` applied to its VERBS, not its numbers.

    The Intent comes back, not a spec: the caller re-compiles it (project.refine
    goes through set_intent), which is what re-derives every number from this
    clip and rebuilds the regional layers. A move nobody touched therefore lands
    on exactly the value it had, because compile_intent is pure.

    `stats` IS NOT SHOWN TO THE MODEL, deliberately, for the reason spelled out
    above plan_intent: the measurement is what compiler.py consults, and a model
    choosing between "subtle" and "moderate" is not helped by knowing the median
    luma. It stays in the signature because it makes this interchangeable with
    refine_spec at the one call site that picks between them.

    WHAT THIS CANNOT DO, and it is a real limit rather than a gap to fill later:
    a request outside the vocabulary ("crop it", "sharpen just her face") has no
    op to become, so the prompt tells the model to return the list unchanged.
    The user then sees a frame that did not move, which is the honest answer --
    an invented op would change the picture for a reason nobody asked for. The
    fallback for a genuinely unsayable request is another verb, not another
    transport.

    DELETING AN OP IS NOT THE SAME AS SHRINKING IT, and the prompt says so
    explicitly: "subtle" still moves the picture. "less grain" wants a smaller
    grain op, "take the grain out" wants no grain op at all, and only the second
    one can return the clip to where it started on that axis.
    """
    if provider is None:
        from ragvid.providers import get_provider

        provider = get_provider()

    user = (
        "The moves already applied to this clip, which the user has accepted:\n"
        f"{current.model_dump_json(indent=2)}\n\n"
        f'The user\'s adjustment request: "{instruction}"\n\n'
        "Return the whole edited list."
    )
    # Same call-and-judge as vibe.ask_intent, and duplicated on purpose: eight
    # lines of local code beat importing a private helper across modules, and a
    # half-answer has to raise here for the same reason it does there -- every
    # field of Intent has a default, so it would otherwise compile to a grade
    # that quietly does less than the user already had.
    raw = provider.plan_json(REFINE_INTENT_SYSTEM, user, Intent.llm_json_schema())
    try:
        return Intent(**raw)
    except Exception as exc:  # pydantic ValidationError: a verb outside the vocabulary
        raise ProviderError(provider.name, f"returned JSON that is not an intent: {exc}") from exc
