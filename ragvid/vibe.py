"""Turn a mood word into a GradeSpec, calibrated to the measured footage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragvid import looks
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
   brightness. Lower it to let the curve darken the image, raise it to lighten.

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
        (st.p99.r < 0.75 and st.p99.g < 0.75 and st.p99.b < 0.75,
         "There is no real white in this clip (p99 well under 1) — it is hazy or "
         "under-exposed, so exposure, slope and contrast all have room to work."),
        (st.frame_variance < 0.02,
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


def plan_vibe(vibe: str, stats: "ClipStats", provider=None) -> GradeSpec:
    """Plan a grade for `vibe`, calibrated to this clip's measured statistics."""
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
            "Grade this specific footage toward that look, starting from identity and moving "
            "only what the look requires. Return the full spec.",
        ) if part
    )
    return provider.plan(SYSTEM, user).sanitize()
