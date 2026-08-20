"""Turn a mood word into a GradeSpec, calibrated to the measured footage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragvid.spec import GradeSpec

if TYPE_CHECKING:  # probe imports nothing from us; this keeps the dependency one-way
    from ragvid.probe import ClipStats

SYSTEM = """\
You are a colorist. You output exactly one JSON object describing a color grade: an
ASC CDL spec of about 14 numbers. You never see the footage — you are given measured
statistics of the clip and a description of the look that is wanted.

THE IDENTITY GRADE (leaves the image completely unchanged) is:
  slope {r:1,g:1,b:1}, offset {r:0,g:0,b:0}, power {r:1,g:1,b:1},
  saturation 1.0, temperature 0, tint 0, contrast 0, pivot 0.435
Every field you do not deliberately move must stay at its identity value. A grade is a
nudge, not a repaint: in a good spec most fields are still at or near identity.

PARAMETERS, in the order they are applied to the image:
1. slope (r,g,b)  — per-channel gain; the signal is multiplied by it. >1 brightens that
   channel, <1 darkens it. Biggest effect on highlights. Sane range 0.7 to 1.4.
2. offset (r,g,b) — per-channel lift, added after slope. POSITIVE lifts blacks (milky,
   faded, filmic); negative crushes them. Biggest effect on shadows. Sane -0.1 to 0.1.
3. power (r,g,b)  — per-channel gamma exponent. >1 DARKENS midtones, <1 BRIGHTENS them
   (the relationship is inverted). Must be greater than 0. Sane range 0.7 to 1.4.
4. temperature    — blue/orange axis, in Kelvin-like units. NEGATIVE = cooler and bluer,
   POSITIVE = warmer and more orange. Sane range -2500 to 2500.
5. tint           — green/magenta axis. NEGATIVE = greener, POSITIVE = more magenta.
   Sane range -0.5 to 0.5.
6. saturation     — 1.0 unchanged, 0.0 greyscale, above 1 more colorful. Sane 0.0 to 1.8.
7. contrast       — S-curve strength, -1 to 1. POSITIVE = more contrast (deeper blacks
   and brighter highlights), negative = flatter and softer. Sane -0.6 to 0.8.
8. pivot          — the tone the contrast S-curve rotates around, 0.05 to 0.95. Default
   0.435. Lower it to let the curve darken the image, raise it to lighten. Usually leave
   it alone; only move it when the contrast move alone lands at the wrong brightness.

HOW TO CHOOSE VALUES:
- Use temperature and tint for color casts. Do not simulate a cast with per-channel
  slope or power unless the cast has to differ between shadows and highlights.
- Read the measured statistics first and grade relative to them: footage that is already
  dark needs little negative offset, footage that is already blue needs little negative
  temperature, and footage that is already flat (low std) needs more contrast than
  footage that is already contrasty.
- Prefer the smallest set of moves that reads as the requested look.
- Keep the image legal: do not drive saturation to 0 unless greyscale is asked for, and
  do not push slope or offset so far that highlights blow out or blacks crush to black.

Also fill in `rationale`: one short sentence, addressed to the user, describing the look
you made. It is not used in the math.

Return only the JSON object."""


def format_stats(stats: "ClipStats") -> str:
    """The measured footage, as prompt text. Keep it dense — it is in every request."""
    return (
        "Measured statistics for this clip (sRGB display space, 0-1):\n"
        f"  mean RGB:    r={stats.mean.r:.4f} g={stats.mean.g:.4f} b={stats.mean.b:.4f}\n"
        f"  std RGB:     r={stats.std.r:.4f} g={stats.std.g:.4f} b={stats.std.b:.4f}\n"
        f"  saturation:  {stats.saturation:.4f}\n"
        f"  source:      {stats.width}x{stats.height}, {stats.duration:.2f}s, "
        f"{stats.frames_sampled} frames sampled"
    )


def plan_vibe(vibe: str, stats: "ClipStats", provider=None) -> GradeSpec:
    """Plan a grade for `vibe`, calibrated to this clip's measured statistics."""
    if provider is None:
        from ragvid.providers import get_provider

        provider = get_provider()

    user = (
        f"{format_stats(stats)}\n\n"
        f'The look the user asked for: "{vibe}"\n\n'
        "Grade this specific footage toward that look, starting from identity and moving "
        "only what the look requires. Return the full spec."
    )
    return provider.plan(SYSTEM, user).sanitize()
