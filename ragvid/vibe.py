"""Turn a mood word into a GradeSpec, calibrated to the measured footage.

TWO PATHS LIVE HERE AND plan_vibe PICKS BETWEEN THEM. The default is intent ->
compiler (roadmap A1): the model emits typed verbs from a closed vocabulary and
compiler.py turns them into numbers by reading the clip. The older path, where
the model authors all 43 numbers itself, stays as the fallback for endpoints
that cannot constrain decoding to the Intent schema.

WHY THAT WAY ROUND, measured rather than argued (scripts/bakeoff_intent.py,
gpt-oss-120b, ten sentences on test_files/test.mp4, each resulting spec applied
to real frames and the moments re-measured): intent 23/23 checks, 10/10 prompts
clean, 2010 tokens per prompt. Direct 21/23, 8/10, 5131 tokens. The direct
path's two misses are the two it cannot fix by trying harder -- it darkened a
clip by 0.13 luma that nobody asked to darken, and it answered "warm it up, but
at half strength" with the byte-identical grade it gave "warmer", because a
model with no memory of the full-strength look has nothing to be half of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragvid import looks
from ragvid.compiler import compile_intent
from ragvid.errors import ProviderError
from ragvid.intent import Intent
from ragvid.spec import GradeSpec

if TYPE_CHECKING:  # probe imports nothing from us; this keeps the dependency one-way
    from ragvid.probe import ClipStats

# Ranges are written `field [low..high]` — one uniform, machine-readable form, because
# tests/test_providers.py parses them back out and asserts every one sits INSIDE the
# corresponding spec.sanitize() clamp. Prompt and clamp used to disagree (the prompt
# said slope 0.7-1.4 while sanitize allowed 0-8); that test is what keeps them honest.
# Do not write any other bracketed `[a..b]` in this string.
SYSTEM = """\
You are a colorist. You output exactly one JSON object describing a color grade: 43
numbers plus a one-line rationale. You never see the footage — you are given measured
statistics of the clip and a description of the look that is wanted.

THE IDENTITY GRADE (leaves the image completely unchanged) is:
  slope {r:1,g:1,b:1}, offset {r:0,g:0,b:0}, power {r:1,g:1,b:1},
  saturation 1.0, temperature 0, tint 0, contrast 0, pivot 0.435,
  exposure 0, highlight_rolloff 0, look_mix 1.0,
  shadow_tint {r:0,g:0,b:0}, highlight_tint {r:0,g:0,b:0},
  shadow_lift 0, highlight_lift 0,
  hue_red, hue_yellow, hue_green, hue_cyan, hue_blue, hue_magenta all {sat:1,lum:0},
  effects {denoise:0, glow:0, softness:0, grain:0, vignette:0, fringe:0}

MOST FIELDS MUST COME BACK AT IDENTITY. There are 43 knobs and a good grade moves three
to eight of them. Emit every other field at exactly the identity value listed above —
not near it, at it. A grade is a nudge, not a repaint. Reaching for a knob you were not
asked for is the worst mistake available to you here; leaving one alone is never wrong.
In particular the hue qualifiers and the effects are OFF by default: touch them only
when the request is explicitly about one hue, or explicitly about texture.

PARAMETERS, in the order they are applied to the image. Ranges are written
low..high in brackets — stay inside them.

TONE — brightness and contrast. This is where most looks live.
1. exposure [-1.5..1.5] — overall brightness in photographic stops, applied first.
   +1 is twice as bright, -1 half. The cleanest way to make an image brighter or darker.
2. slope [0.7..1.4] — per-channel gain (r, g, b; the range applies to each channel), the
   signal is multiplied by it. >1 brightens that channel, <1 darkens it. Biggest effect
   on highlights. Prefer exposure for a neutral brightness change and slope when the
   channels must move by different amounts.
3. offset [-0.1..0.1] — per-channel lift (r, g, b), added after slope. POSITIVE lifts
   blacks (milky, faded, filmic); negative crushes them. Biggest effect on shadows.
4. power [0.7..1.4] — per-channel gamma exponent (r, g, b). >1 DARKENS midtones, <1
   BRIGHTENS them (the relationship is inverted). Must stay above 0.
5. highlight_rolloff [0..0.6] — soft shoulder that bends highlights over instead of
   clipping them flat. 0 is a hard clip. IF YOU RAISE slope OR exposure ABOVE ABOUT 1.1
   OR +0.3, RAISE THIS TO 0.2-0.4: a hard clip welds every bright pixel to pure white and
   the detail is gone for good. The cost is that a shoulder pulls pure white slightly
   down (at 0.3, white lands at 0.93), so leave it at 0 when you are not pushing up.
6. contrast [-0.6..0.8] — S-curve strength. POSITIVE = more contrast (deeper blacks and
   brighter highlights), negative = flatter and softer.
7. pivot [0.25..0.65] — the tone the contrast S-curve rotates around. Default 0.435.
   Usually leave it alone; only move it when the contrast move lands at the wrong
   brightness. RAISE it to let the curve darken the image, LOWER it to lighten
   (measured: contrast +0.5 on a mid-grey clip gives mean luma 0.557 at pivot 0.25
   and 0.483 at pivot 0.65). With negative contrast the effect reverses.

COLOUR — global cast and intensity.
8. temperature [-2500..2500] — blue/orange axis, in Kelvin-like units. NEGATIVE = cooler
   and bluer, POSITIVE = warmer and more orange. Use this for a cast, not per-channel
   slope, unless the cast has to differ between shadows and highlights.
9. tint [-0.5..0.5] — green/magenta axis. NEGATIVE = greener, POSITIVE = more magenta.
10. saturation [0..1.8] — 1.0 unchanged, 0 greyscale, above 1 more colorful.

TONAL SPLIT — different colour in shadows than in highlights. This is what makes
"teal shadows, warm highlights" possible; a global temperature move cannot do it.
11. shadow_tint [-0.12..0.12] — colour added to the darker half only (r, g, b). It is
    luma-stripped, so it changes colour and never brightness. Teal shadows are about
    r -0.05, g 0.02, b 0.05.
12. highlight_tint [-0.12..0.12] — the same for the brighter half (r, g, b).
13. shadow_lift [-0.1..0.1] — brightness of the shadows only. Positive fades the blacks.
14. highlight_lift [-0.1..0.1] — brightness of the highlights only.

HUE — six qualifiers, each an object {sat, lum}, targeting one hue family:
hue_red, hue_yellow, hue_green, hue_cyan, hue_blue, hue_magenta.
15. hue_red.sat [0.5..1.5] — saturation multiplier for pixels of that hue. 1 = unchanged.
16. hue_red.lum [-0.1..0.1] — brightness offset for pixels of that hue. 0 = unchanged.
    The same two ranges apply to all six bands. They act only on pixels that already
    carry that hue with real chroma — greys and skies of another hue are untouched.
    Use at most one or two bands, for a request that names a colour ("drain the greens",
    "keep the skin, kill everything else"). Leave the rest at sat 1, lum 0.

TEXTURE — spatial effects, in `effects`. These are NOT part of the colour transform;
the renderer applies them separately. All default to 0 and most grades leave them there.
17. effects.denoise [0..0.6] — removes grain and sensor noise. Costs fine detail.
18. effects.glow [0..0.5] — bloom around highlights. Dreamy, hazy, romantic.
19. effects.softness [-0.5..0.5] — POSITIVE blurs, NEGATIVE sharpens.
20. effects.grain [0..0.5] — film grain. The usual answer for "filmic", "16mm", "vintage".
21. effects.vignette [-0.5..0.5] — POSITIVE darkens the corners, negative brightens them.
22. effects.fringe [-0.3..0.3] — chromatic aberration at the edges. Cheap-lens, VHS, dream.

STRENGTH
23. look_mix [0.6..1] — how much of the whole grade survives, mixed back toward the
    untouched source. 1 is the full look. Drop to 0.7 only when the user asks for the
    look to be subtler; it is not a substitute for choosing smaller numbers.

HOW TO CHOOSE VALUES:
- Read the measured statistics first and grade RELATIVE to them: footage that is already
  dark needs little negative exposure, footage that is already blue needs little negative
  temperature, and footage that is already flat (low std) needs more contrast than
  footage that is already contrasty.
- Obey the notes attached to the statistics. They describe what this clip cannot take.
- Prefer the smallest set of moves that reads as the requested look. Two well-chosen
  numbers beat ten small ones, and the ten small ones drift.
- Keep the image legal: do not drive saturation to 0 unless greyscale is asked for, and
  do not push slope, exposure or offset so far that highlights blow out or blacks crush.

Also fill in `rationale`: one short sentence, addressed to the user, describing the look
you made. It is not used in the math.

Return only the JSON object."""


def format_stats(stats: "ClipStats") -> str:
    """The measured footage, as prompt text. Keep it dense — it is in every request.

    This is the ONLY place ClipStats becomes prompt text. The numbers alone are not
    enough for a small model: the notes below turn the two measurements that constrain
    a grade the hardest — no headroom at the top, nothing left at the bottom — into an
    instruction. They are emitted only when they actually fire, so a clean clip pays
    nothing for them.
    """
    m, s, p1, p50, p99 = stats.mean, stats.std, stats.p1, stats.p50, stats.p99
    text = (
        "Measured statistics for this clip (sRGB display space, 0-1):\n"
        f"  mean RGB:    r={m.r:.4f} g={m.g:.4f} b={m.b:.4f}\n"
        f"  std RGB:     r={s.r:.4f} g={s.g:.4f} b={s.b:.4f}\n"
        f"  p1  (black): r={p1.r:.4f} g={p1.g:.4f} b={p1.b:.4f}\n"
        f"  p50 (median):r={p50.r:.4f} g={p50.g:.4f} b={p50.b:.4f}\n"
        f"  p99 (white): r={p99.r:.4f} g={p99.g:.4f} b={p99.b:.4f}\n"
        f"  saturation:  {stats.saturation:.4f}   dominant hue: {stats.dominant_hue:.0f} deg\n"
        f"  clipped high: {stats.clipped_high:.1%} of pixels   "
        f"crushed low: {stats.crushed_low:.1%}\n"
        f"  luma variance within a frame: {stats.frame_variance:.4f}\n"
        f"  source:      {stats.width}x{stats.height}, {stats.duration:.2f}s, "
        f"{stats.frames_sampled} frames sampled"
    )
    notes = [note for fires, note in _stat_notes(stats) if fires]
    if notes:
        text += "\n\nWhat these measurements mean for this grade:\n" + "\n".join(
            f"- {n}" for n in notes
        )
    return text


def _stat_notes(st: "ClipStats") -> list[tuple[bool, str]]:
    """(fires?, instruction) for each measurement that can constrain a grade."""
    return [
        (st.clipped_high > 0.01,
         "The highlights are ALREADY at the rail, so there is no headroom left: protect "
         "them, do not push. Keep exposure and slope at or below 1.0, and set "
         "highlight_rolloff 0.2-0.4 to bend the blown area back into detail."),
        (st.crushed_low > 0.02,
         "The blacks are ALREADY crushed: negative offset would only destroy more of "
         "them. If the look wants depth, get it from contrast; if it wants detail back, "
         "use a small positive offset or shadow_lift."),
        # `> 0` guards, not decoration: a session written before these fields
        # existed loads with ClipStats defaults (p99 = 0, frame_variance = 0)
        # and session.py does not migrate, so an unguarded threshold fires from
        # a default and feeds the model an instruction about footage nobody
        # measured. A genuinely all-black clip reading as "unmeasured" is the
        # correct trade.
        (0.0 < max(st.p99.r, st.p99.g, st.p99.b) and
         st.p99.r < 0.75 and st.p99.g < 0.75 and st.p99.b < 0.75,
         "There is no real white in this clip (p99 well under 1) — it is hazy or "
         "under-exposed, so exposure, slope and contrast all have room to work."),
        (0.0 < st.frame_variance < 0.02,
         "Very low luma variance: the image is FLAT and can take more contrast than "
         "usual."),
        (st.frame_variance > 0.09,
         "High luma variance: the image is already contrasty (or the clip spans very "
         "different shots). Add contrast sparingly."),
        (st.saturation > 0.35,
         "This footage is already saturated; raising saturation further will clip the "
         "colour. Prefer the tonal split or a hue qualifier for a colour move."),
        (st.saturation < 0.06,
         "This footage is nearly monochrome, so colour moves will barely show. A cast "
         "has to come from the tonal split or from temperature, not from saturation."),
    ]


def plan_vibe(vibe: str, stats: "ClipStats", provider=None,
              balance: bool = False) -> tuple[GradeSpec, Intent | None]:
    """Plan a grade for `vibe`, from whichever path this endpoint can run — see
    the module docstring for which is default and what decided it.

    Returns (spec, intent). The Intent is None on the direct path and that is
    not a degenerate case to paper over: the model authored numbers there, so
    there are no verbs to show, and a caller that wants to say what the grade
    did falls back to `spec.rationale`. The spec is still the only currency
    anything downstream renders, bakes or exports.

    The routing test is a CAPABILITY, not a preference: the intent path needs
    constrained decoding against Intent's schema (ask_intent's ponytail note),
    which is exactly the endpoints sitting at the "json_schema" rung. Everything
    below that rung, and the Anthropic provider (a different SDK, no
    `structured` attribute), gets the direct path, which works there today.
    Nothing else in ragvid changes: both paths return a sanitized GradeSpec, and
    the spec is still the only currency session, history and the LUT ever see.
    """
    if provider is None:
        from ragvid.providers import get_provider

        provider = get_provider()
    if getattr(provider, "schema_enforced", False):
        return plan_intent(vibe, stats, provider, balance=balance)
    # `balance` is silently dropped here, and there is nowhere honest to put it:
    # auto-balance is a compiler pass and the direct path has no compiler. An
    # endpoint that cannot constrain decoding gets an unbalanced grade, which is
    # what it got before this flag existed.
    return plan_direct(vibe, stats, provider), None


def plan_direct(vibe: str, stats: "ClipStats", provider=None) -> GradeSpec:
    """The model authors all 43 numbers itself. Was the only path; now the
    fallback, and still what refine.py builds on."""
    if provider is None:
        from ragvid.providers import get_provider

        provider = get_provider()

    # Retrieval, additive: measured neighbouring looks from the corpus. Goes in
    # the USER message, not SYSTEM -- the hits depend on `vibe`, so putting them
    # in the system prompt would make it un-cacheable and mix data into
    # instructions. Empty string when nothing in the corpus overlaps the vibe.
    examples = looks.ground(vibe)

    user = "\n\n".join(
        part for part in (
            format_stats(stats),
            f'The look the user asked for: "{vibe}"',
            examples,
            # The starting point has to agree with whether retrieval fired. Telling
            # the model to start from identity right after handing it a measured
            # neighbouring grade is a direct contradiction, and a 20B model
            # resolves contradictions by ignoring one of them at random.
            ("Grade this specific footage toward that look, starting from the closest "
             "reference look above and moving only what this request and this footage "
             "require. Return the full spec."
             if examples else
             "Grade this specific footage toward that look, starting from identity and "
             "moving only what the look requires. Return the full spec."),
        ) if part
    )
    return provider.plan(SYSTEM, user).sanitize()


# ---- the intent path (roadmap A1) -----------------------------------------

# NO DIGIT APPEARS IN THIS STRING, and tests/test_providers.py asserts it. The
# whole claim of this path is that the model never emits a magnitude; a prompt
# that quotes one — a range, a Kelvin figure, a default — hands it a number to
# copy, and we are back to SYSTEM above, which spends most of its ~1700 tokens
# telling a model how far to move 43 knobs it should not be holding.
#
# NO CLIP STATISTICS EITHER, deliberately, which is why plan_intent does not
# take a stats argument for the prompt's sake. The measurement is what
# compiler.py consults; showing it to the model only invites it to reason about
# magnitude in prose, and it is the single largest block of tokens in the direct
# path's request (format_stats is ~200 tokens, every call).
#
# AND NO RETRIEVAL. looks.ground() exists because "author 43 numbers from
# nothing" is a job a 20B model does badly, so the corpus collapses it into
# "edit these 43 numbers a bit" (looks.py docstring). There is nothing here to
# collapse: the model picks from sixteen words. A corpus entry is a full 43-float
# GradeSpec — ~350 tokens of exactly the numbers this path exists to stop the
# model reading. It would be paying tokens to re-tempt it.
#
# The brightness sentence under HOW TO CHOOSE is there because of a measured
# failure, not a hunch. The first version of that line offered "moody" as deeper
# shadows, cooler and less colour — no exposure — and the model followed it
# exactly: "moody but keep it natural" and "make it feel like a rainy night"
# both came back without an `exposure` op, moving mean luma -0.006 and -0.009
# where the sentence asked for dark. An example in this prompt is not
# illustration, it is instruction, and anything it leaves out the model leaves
# out too.
INTENT_SYSTEM = """\
You are a colorist. You output exactly one JSON object listing the moves a grade needs.
You choose WORDS, and never numbers. Every number in the finished grade is computed from
measured statistics of this particular clip, by code you do not see; your words decide
which axis moves, which way, and roughly how far. There is no field here that takes a
magnitude, and inventing one is the only way to get this task wrong.

Return {"ops": [...], "strength": ...}. Each op has four fields:

  op      which move — one of the verbs below, spelled exactly as it is written there.
  dir     "up" or "down" along that verb's own axis. Every verb below says what up means.
  amount  "subtle", "moderate" or "strong". "moderate" is the normal answer, and the
          right one whenever the sentence just names a change ("warmer"). "subtle" is
          for "a touch", "slightly", "a hint of", "just barely". "strong" is for
          "really", "very", "heavily", "way more", "push it".
  target  "" — the whole picture — unless the sentence names a colour or a part of
          the frame. See TARGET below.

VERBS — TONE. Most looks live here.
  exposure    up = brighter overall, down = darker overall.
  contrast    up = punchier: deeper blacks and brighter whites. down = flatter, softer.
  midtones    up = brighter mids, down = darker mids, without moving black or white.
  shadows     up = lifted, faded, milky, filmic blacks. down = crushed, deep, inky blacks.
  highlights  up = brighter whites. down = pulled back, recovered, protected highlights.

VERBS — COLOUR.
  warmth      up = warmer, oranger, golden, sunlit. down = cooler, bluer, icier.
              This is the verb for every temperature word in the language. The tint
              verbs are not a substitute for it.
  tint        up = magenta or pink, down = green. That axis only, and it is rarely asked
              for by name.
  saturation  up = richer, more colourful. down = drained, muted, washed out; down with
              amount "strong" is as close to black and white as this vocabulary goes.
  shadow_tint     puts a colour into the DARK half of the picture only.
  highlight_tint  puts a colour into the BRIGHT half only.
              These two are how a split-tone look is said — "teal shadows, warm
              highlights" is one of each — and a single warmth move cannot do it. dir up
              adds the colour named in target, down takes that colour out.

VERBS — TEXTURE. Spatial, applied outside the colour transform, and OFF unless the
sentence is actually about texture: film, grain, dreamy, hazy, soft, sharp, vintage,
VHS, clean, noisy.
  grain     up = more film grain.
  glow      up = more bloom around the highlights.
  vignette  up = darker corners.
  softness  up = softer, down = sharper.
  denoise   up = cleaner, less sensor noise. Costs fine detail.
  fringe    up = more chromatic aberration at the edges. Cheap lens, VHS, dream.

TARGET, in full. It answers WHICH PIXELS, and takes either a colour or a place.

COLOURS: red, orange, yellow, green, cyan, teal, blue, magenta, purple, skin.
  - on saturation and exposure a colour SELECTS which pixels move, by the colour they
    already are: "drain the greens", "darken the reds", "keep the skin, kill the rest".
  - on shadow_tint and highlight_tint a colour NAMES THE COLOUR being added.
  Every other verb ignores a colour target. Leave it "" on them.

PLACES: top, bottom, left, right, center, edges. A place means the move applies only to
that part of the frame and the rest of the picture is left alone — "darken the top",
"warm the left side", "brighten the middle", "drop the edges off". Use a place ONLY when
the sentence actually points at one; a look that is about the whole picture takes "".
  - center is the middle of the frame, edges is everything but the middle.
  - places work on every TONE and COLOUR verb. The TEXTURE verbs cannot take one, because
    they are applied to the whole frame after the colour: leave their target "".

THINGS: sky, foliage, person, water, buildings. A thing means the move applies only to
the pixels that are part of it, wherever in the frame they sit — "make the sky moody",
"drop the buildings back", "warm the water up". Name one ONLY when the sentence names
it; a thing this clip does not contain grades nothing at all.
  - things work on every TONE and COLOUR verb, exactly as places do, and the TEXTURE
    verbs cannot take one for the same reason.
  - skin is in the COLOURS list above and not here, on purpose: it is a hue family, so it
    follows a face through movement and through a cut, which an outline drawn per frame
    does not.

STRENGTH is how much of the whole look survives against the untouched footage: "full",
"strong", "moderate" or "subtle". Use "full" unless the user asked for the look itself to
be held back — "half strength", "dial it back", "a subtle version", "just a hint of it".
It is not a substitute for choosing a smaller amount on one op.

HOW TO CHOOSE:
- The fewest ops that read as the look. A real grade is three to eight moves, and a
  sentence naming one thing is one op. Reaching for a verb you were not asked for is the
  worst mistake available to you here; leaving one out is never wrong.
- A mood word does imply moves, and naming them is the job. Start with brightness: a
  mood is usually a BRIGHTNESS word before it is a colour one, so "moody", "night",
  "dusk", "overcast", "gloomy" and "sombre" all need an exposure op pointing down, and
  then the rest — deeper shadows, cooler, a little less colour. Name the moves, never
  the mood, and never leave the picture at its original brightness when the sentence
  asked for a darker one.
- One op per verb. Two warmth ops in one list compose into a bigger push than either of
  them asked for; say it once, with the right amount.
- Texture verbs, colour targets and place targets stay out of a grade that did not ask
  for them.
- Order does not matter. The compiler decides the order the moves are applied in.
- You are not told what this footage looks like, and you do not need to be. "Warmer" on
  an already-orange clip and on a blue one is the same request; how far it actually goes
  is measured from the clip, not guessed by you.

Return only the JSON object."""

# REFINEMENT IS THE SAME JOB WITH A DIFFERENT SUBJECT, so this is INTENT_SYSTEM
# plus a tail rather than a second prompt. Two consequences, both wanted: the
# verb vocabulary is defined once, so intent.py growing a target word is a
# prompt edit here and nothing else (the guard test in tests/test_providers.py
# iterates OPS/AMOUNTS/STRENGTHS/TARGETS over BOTH strings), and the tail
# inherits the no-digit rule for free -- a magnitude quoted here would be a
# number for the model to copy, which is the one thing this path exists to stop.
#
# NO CLIP STATISTICS HERE EITHER, for the reason above plan_intent: the numbers
# come from compiler.py re-reading the clip, and the model is choosing between
# "subtle" and "moderate", which no measurement helps it do.
#
# The last paragraph is the one that costs something to leave out. The verb list
# cannot say "crop it" or "sharpen just her face", and a model asked to return a
# list will always return a list -- so the honest answer to an unsayable request
# has to be spelled out, or it comes back as an invented op the user never asked
# for and cannot attribute.
REFINE_INTENT_SYSTEM = INTENT_SYSTEM + """

YOU ARE NOW EDITING A LIST OF MOVES THAT ALREADY EXISTS, NOT WRITING A NEW ONE.
You are given the moves already applied to this clip -- the same JSON shape you return
-- and one short adjustment request. Return the WHOLE edited list, not a diff.

THE COPY RULE, WHICH MATTERS MORE THAN EVERY OTHER RULE HERE. Every op the request does
not talk about comes back exactly as it was handed to you: same op, same dir, same
amount, same target. Copy it, do not re-judge it. It is in the list because the user
asked for it on an earlier turn and kept it, and dropping one because it looks unrelated
to the new sentence is the failure mode of this task. The target is part of the copy: an
op aimed at a part of the frame, or at one colour, still names that part or that colour
after a request about something else entirely. When in doubt, copy.

HOW TO MAKE THE EDIT THE REQUEST ASKS FOR:
- MORE OR LESS OF A MOVE THAT IS ALREADY THERE -- keep that op, and move its amount
  along the ladder subtle - moderate - strong. "a bit less warm" on a moderate warmth op
  pointing up leaves it pointing up and makes it subtle. That ladder has those rungs and
  no others, so an op already at strong that is asked for even more stays at strong.
- THE OPPOSITE OF A MOVE THAT IS ALREADY THERE -- flip its dir, and put its amount back
  to moderate unless the sentence says how far to go.
- SOMETHING THE LIST DOES NOT MENTION AT ALL -- add one new op for it, chosen exactly as
  you would have chosen it in a fresh grade.
- A MOVE TO STOP ENTIRELY -- "take the grain out", "lose the vignette", "forget the
  warmth", "leave the top alone" -- delete that op from the list. Deleting is NOT the
  same as making it subtle: a subtle op still changes the picture, and somebody asking
  for a move to go away wants it gone. Shrink the amount when the sentence asks for
  less of something; delete the op when it asks for none of it.
- THE WHOLE LOOK HELD BACK, OR LET OUT AGAIN -- "half strength", "dial it back", "a
  subtle version of that", "go all in" -- change `strength` and leave every op alone.
  strength is a ladder too, from full down through strong and moderate to subtle, and it
  is read against the list you were given: dialling back a look that is already held
  back lands one rung further down, not back at the top.

IF THE REQUEST CANNOT BE SAID IN THIS VOCABULARY AT ALL -- a crop, a speed change, text
on screen, one object in the frame, a different clip -- return the list you were given,
complete and unchanged. A move invented to look responsive is worse than no move at all:
the picture changes for a reason nobody asked for, and the user cannot tell which of
their earlier moves did it."""


def plan_intent(vibe: str, stats: "ClipStats", provider=None,
                balance: bool = False) -> tuple[GradeSpec, Intent]:
    """Plan a grade for `vibe` the roadmap-A1 way: model -> Intent -> compiler.

    Same signature and same return type as plan_vibe, so the two are swappable
    at the call site. The difference is where the numbers come from: plan_vibe's
    provider authors all 43, this one authors none of them — `stats` is read by
    compiler.compile_intent, never shown to the model.

    BOTH come back, because the Intent is not an implementation detail of the
    spec. It is the only human-readable record of what was asked for, and it
    cannot be recovered afterwards: 43 floats do not say which four words made
    them. Roadmap C3/C4 renders it as a list of sentences with a strength
    control each, and re-compiles through this same compiler when one moves.
    """
    intent = ask_intent(vibe, provider)
    return compile_intent(intent, stats, balance=balance), intent


def ask_intent(vibe: str, provider=None) -> Intent:
    """The model call, and the only part of the intent path that can fail.

    Provider.plan_json does the talking: Intent is not a GradeSpec, so plan()
    cannot carry it, and the two SDKs constrain output differently (Anthropic's
    `output_config`, everyone else's `response_format`). Only the provider knows
    which it speaks, and it is also the one place a bad reply is judged — a
    half-answer here would compile to the identity grade and look like a model
    that did nothing rather than an endpoint that answered wrong.

    Callers route on `schema_enforced` first (plan_vibe does); plan_json raises
    rather than degrade to prompt-coaxed JSON.
    """
    if provider is None:
        from ragvid.providers import get_provider

        provider = get_provider()

    raw = provider.plan_json(
        INTENT_SYSTEM, f'The look the user asked for: "{vibe}"', Intent.llm_json_schema()
    )
    try:
        return Intent(**raw)
    except Exception as exc:  # pydantic ValidationError: a verb outside the vocabulary
        raise ProviderError(provider.name, f"returned JSON that is not an intent: {exc}") from exc
