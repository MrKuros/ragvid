"""The grade spec — the contract every other module codes against.

A GradeSpec is 44 numbers describing a color grade. It is the *only* thing an
LLM ever produces in this project; no pixels leave the machine (see
docs/ARCHITECTURE.md rule 2). The .cube
LUT is a derived artifact baked from a spec, which is why refinement ("make it
less blue") is possible at all — you can adjust numbers, not a baked LUT.

The core is ASC CDL (slope/offset/power per channel + saturation), an industry
interchange standard, plus temperature/tint/contrast, an exposure stop, a
tonal split (shadow/highlight tint + lift), six hue qualifiers, a highlight
shoulder and a whole-look mix.

`effects` is the odd one out: it is SPATIAL (blur, grain, vignette...), so it
cannot live in a 3D LUT at all and `apply()` ignores it completely. render.py
wires it into ffmpeg filters.

A GradeSpec is therefore the currency of ONE correction, for every pixel in the
frame. It answers WHAT to do to a colour and has no place to answer WHICH
PIXELS -- a lookup table indexed by colour has nowhere to put an address. That
second question belongs to region.py, whose GradeStack wraps a base GradeSpec
plus an ordered list of (Region, GradeSpec) layers. Nothing here knows about it,
deliberately: a grade with no regions is a stack with no layers, and this file
stays byte-identical for it.

EVALUATION ORDER IS LOAD-BEARING. Reviewers and reimplementations must match:

    0. src         the input, captured for the look_mix lerp at step 9
    1. exposure    x *= 2**exposure      before CDL, so `offset` stays an
                                         ABSOLUTE lift rather than being
                                         scaled by exposure
    2. CDL         x*slope + offset; clip(x, 0, None); x **= clip(power, 1e-6, None)
    3. white bal   temperature/tint applied as per-channel gains
    4. saturation  luma + (x - luma) * saturation
                   (1-4 also live behind `through_saturation`, so a solver can
                   fit against the exact array step 5 receives)
    5. hue qualifiers  six smoothstep bands, chroma-gated, luma-preserving
    6. tonal split shadow/highlight masks; tints are luma-stripped so tint and
                   lift are exactly independent axes
    7. rolloff     soft highlight shoulder (extended Reinhard)
    8. contrast    S-curve around `pivot`, aimed at one end of the tone
                   scale by `contrast_balance` (0 = the symmetric curve)
    9. look_mix    out = clip(src,0,1) + (out - clip(src,0,1)) * look_mix
   10. clip to [0, 1]

THE CONSTRAINT THAT PINS THIS ORDER. `_smoothstep(u) = u*u*(3-2*u)` has
derivative `6u(1-u)`, which goes NEGATIVE for u > 1 — feed the S-curve an
out-of-range value and it is non-monotonic, i.e. brighter input maps to darker
output, permanently baked into the .cube. That is why `_s_curve` clips its
input and why the clip is NOT removable. Consequence: every op that can push a
value above 1 sits before the soft clip, and the soft clip sits immediately
before contrast.

  - Rolloff is at 7, NOT at the end. The clip that actually destroys highlights
    is the one inside the contrast step; a shoulder only at the final return
    would leave it untouched. It is also after saturation, because saturation
    > 1 can itself drive a channel above 1.
  - `contrast_balance` is a PARAMETER of step 8, not a step of its own, and
    three independent constraints force that. (a) The asymmetry is defined in
    `u`, the pivot-warped coordinate, which only exists inside _s_curve; a
    standalone toe step would need its own warp -- a second implementation of
    the pivot semantics, free to drift, so moving `pivot` would silently
    retarget the toe. That is exactly what SPLIT_CROSSOVER is a separate
    constant to avoid, one axis over. (b) It consumes _smoothstep, so it must
    receive the clipped input, which only exists after the soft clip at 7.
    (c) It cannot sit after 8, because the clip that actually destroys
    highlights is the one INSIDE step 8.
  - Hue qualifiers come BEFORE the tonal split. A qualifier reads a hue angle,
    so if the split injected its tint first, "teal shadows" would make the cyan
    qualifier fire on every shadow in the frame.
  - The tonal split comes AFTER global saturation: its tints are luma-stripped
    so saturation cannot attenuate them, and its highlight lift is the last
    thing that can exceed 1 — which is what the soft clip at 7 is there to catch.

EVERY new step is guarded by an explicit identity check that skips it entirely.
That is correctness, not micro-optimisation: `L + (x-L)*1.0` and
`src + (x-src)*1.0` are not bitwise identities in floating point (~1 ulp), so
an unguarded step would make `GradeSpec.identity().apply(grid) != grid` while
still passing an atol=1e-6 test. The identity LUT hash gate would break
silently and every saved grade would shift.

All operations take and return float arrays in [0, 1] display space, shape
(..., 3). Values are clipped to [0, 1] only at the very end.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

# Rec.709 luma coefficients — used for the saturation step.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Scaling constants mapping the human-facing temperature/tint units onto
# channel gains. TEMP_FULL is the Kelvin shift that corresponds to a full-
# strength push; chosen so typical LLM outputs (+/-2000) land in a sane range.
TEMP_FULL = 3000.0
TEMP_GAIN = 0.12
TINT_GAIN = 0.10

# Luma at which the shadow and highlight masks meet. Deliberately NOT
# `self.pivot`: coupling them would make a contrast-pivot move silently
# retarget the split masks.
SPLIT_CROSSOVER = 0.5

# Hue qualifier band centres and half-width, in degrees.
#
# HUE_HALFWIDTH == the band spacing is not arbitrary. At any hue exactly two
# bands are active with pre-smoothstep weights u and 1-u, and
# _smoothstep(u) + _smoothstep(1-u) == 1 algebraically, so the six weights are
# an EXACT partition of unity (measured: within 2e-15 of 1 over 1e4 hues). No
# hue is unweighted, none is double-counted, and interpolation between adjacent
# band settings is C1. Wrapping is free — it lives in the mod 360.
HUE_CENTERS = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], dtype=np.float64)
HUE_HALFWIDTH = 60.0

# Chroma below which hue qualifiers fade out. Mandatory, not a nicety: hue is
# ill-conditioned near the neutral axis (d(hue)/d(channel) ~ 1/chroma) and the
# neutral axis is the diagonal of the LUT cube. Without this gate a qualifier
# tints every grey in the image and the whole grade reads as a colour cast.
HUE_CHROMA_GATE = 0.15

# The six band fields, in HUE_CENTERS order.
HUE_FIELDS = ("hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta")

# Width of the extended-Reinhard shoulder, in knee-relative units.
SHOULDER_SPAN = 5.0

_TOL = 1e-12
# Everything that decides "is this field still at identity?" uses ONE tolerance,
# so a value can never be identity-enough to skip a step but not identity-enough
# to report as identity (or the reverse).
_ID_TOL = 1e-9


class RGB(BaseModel):
    """A per-channel triple.

    Deliberately an object with required r/g/b rather than a 3-element array:
    Groq's strict json_schema does not enforce array minItems during
    constrained generation (models reliably emit 1-element arrays and the
    request then fails validation), whereas required object properties are
    structural and are enforced. Verified against gpt-oss-120b / 20b.
    """

    r: float
    g: float
    b: float

    def as_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b], dtype=np.float64)

    @classmethod
    def of(cls, v: float) -> "RGB":
        return cls(r=v, g=v, b=v)


class HueBand(BaseModel):
    """One hue qualifier. Same object-not-array reasoning as RGB."""

    sat: float = Field(1.0, description="Saturation multiplier for this hue. 1.0 = unchanged.")
    lum: float = Field(0.0, description="Luma offset for this hue. 0.0 = unchanged.")


class EffectSpec(BaseModel):
    """Spatial effects. NEVER applied by GradeSpec.apply and never baked into
    the LUT — a 3D LUT is a per-pixel colour map and cannot express a blur.
    render.py turns these into ffmpeg filters."""

    denoise: float = Field(0.0, description="Temporal/spatial denoise strength, 0..1. 0 = off.")
    glow: float = Field(0.0, description="Bloom on highlights, 0..1. 0 = off.")
    softness: float = Field(0.0, description="Positive blurs, negative sharpens. -1..1. 0 = off.")
    grain: float = Field(0.0, description="Film grain amount, 0..1. 0 = off.")
    vignette: float = Field(0.0, description="Corner darkening, -1..1. Negative brightens corners. 0 = off.")
    fringe: float = Field(0.0, description="Chromatic aberration, -1..1. 0 = off.")


class GradeSpec(BaseModel):
    """A complete color grade. Serializes to the session file verbatim."""

    slope: RGB = Field(default_factory=lambda: RGB.of(1.0), description="Per-channel gain (ASC CDL slope). 1.0 = unchanged.")
    offset: RGB = Field(default_factory=lambda: RGB.of(0.0), description="Per-channel lift (ASC CDL offset). 0.0 = unchanged. Positive lifts blacks.")
    power: RGB = Field(default_factory=lambda: RGB.of(1.0), description="Per-channel gamma (ASC CDL power). 1.0 = unchanged. Must be > 0.")
    saturation: float = Field(1.0, description="1.0 = unchanged, 0.0 = greyscale, >1 more saturated.")
    temperature: float = Field(0.0, description="Kelvin shift. Negative = cooler/bluer, positive = warmer/oranger. Typical range -3000..3000.")
    tint: float = Field(0.0, description="Green/magenta axis. Negative = greener, positive = magenta. Typical range -1..1.")
    contrast: float = Field(0.0, description="S-curve strength, -1..1. 0 = unchanged, positive = more contrast.")
    pivot: float = Field(0.435, description="Tonal pivot the contrast S-curve rotates around.")
    contrast_balance: float = Field(0.0, description="Which end of the tone scale `contrast` acts on, -1..1. 0 = evenly, the symmetric S-curve. Negative puts it in the shadows (a crushed toe, highlights left alone), positive in the highlights. Does nothing on its own -- it scales `contrast`.")

    exposure: float = Field(0.0, description="Photographic stops applied before the CDL. 0.0 = unchanged, +1 = twice as bright.")
    look_mix: float = Field(1.0, description="0..1 blend of the whole graded result back toward the source. 1.0 = full look.")
    highlight_rolloff: float = Field(0.0, description="Soft highlight shoulder, 0..1. 0 = hard clip. Higher rolls off sooner and pulls whites below 1.")

    shadow_tint: RGB = Field(default_factory=lambda: RGB.of(0.0), description="Colour pushed into shadows only. 0 = none. Luma-stripped, so it changes colour and not brightness -- exactly, until a channel is driven outside [0,1] and the final clip truncates it.")
    highlight_tint: RGB = Field(default_factory=lambda: RGB.of(0.0), description="Colour pushed into highlights only. 0 = none. Luma-stripped, so it changes colour and not brightness -- exactly, until a channel is driven outside [0,1] and the final clip truncates it.")
    shadow_lift: float = Field(0.0, description="Brightness offset applied to shadows only. 0 = unchanged.")
    highlight_lift: float = Field(0.0, description="Brightness offset applied to highlights only. 0 = unchanged.")

    hue_red: HueBand = Field(default_factory=HueBand, description="Qualifier for hues near 0 degrees (red).")
    hue_yellow: HueBand = Field(default_factory=HueBand, description="Qualifier for hues near 60 degrees (yellow).")
    hue_green: HueBand = Field(default_factory=HueBand, description="Qualifier for hues near 120 degrees (green).")
    hue_cyan: HueBand = Field(default_factory=HueBand, description="Qualifier for hues near 180 degrees (cyan).")
    hue_blue: HueBand = Field(default_factory=HueBand, description="Qualifier for hues near 240 degrees (blue).")
    hue_magenta: HueBand = Field(default_factory=HueBand, description="Qualifier for hues near 300 degrees (magenta).")

    effects: EffectSpec = Field(default_factory=EffectSpec, description="Spatial effects. Not part of the colour transform and never baked into the LUT.")

    rationale: str = Field("", description="One sentence explaining the look, for the user. Not used in math.")

    # ---- construction -----------------------------------------------------

    @classmethod
    def identity(cls) -> "GradeSpec":
        """A grade that must round-trip an image unchanged. Used as a test oracle."""
        return cls()

    def is_identity(self, tol: float = _ID_TOL) -> bool:
        i = GradeSpec.identity()
        # Plain field-wise comparison. Note look_mix == 0 also returns the
        # source, but a zero mix *with other fields set* is a state worth
        # reporting as non-identity, so it is not special-cased.
        return (
            np.allclose(self.slope.as_array(), i.slope.as_array(), atol=tol)
            and np.allclose(self.offset.as_array(), i.offset.as_array(), atol=tol)
            and np.allclose(self.power.as_array(), i.power.as_array(), atol=tol)
            and abs(self.saturation - 1.0) < tol
            and abs(self.temperature) < tol
            and abs(self.tint) < tol
            and abs(self.contrast) < tol
            and abs(self.pivot - i.pivot) < tol
            and abs(self.contrast_balance) < tol
            and abs(self.exposure) < tol
            and abs(self.look_mix - 1.0) < tol
            and abs(self.highlight_rolloff) < tol
            and np.allclose(self.shadow_tint.as_array(), 0.0, atol=tol)
            and np.allclose(self.highlight_tint.as_array(), 0.0, atol=tol)
            and abs(self.shadow_lift) < tol
            and abs(self.highlight_lift) < tol
            and not self.has_hue_qualifiers(tol)
            and self.effects == i.effects
        )

    def render_effects(self) -> EffectSpec:
        """`self.effects` scaled by look_mix — what a renderer should actually apply.

        look_mix is documented as "how much of the whole grade survives", and the
        UI's always-visible Strength slider says "Strength of the whole look".
        But effects are spatial: they live in the ffmpeg chain, not in apply(),
        so the lerp inside apply() could never reach them. Measured before this
        existed: Strength 0% still left a mean 42.9/255 deviation on screen and
        in the export, with the vignette fully present. Scaling each effect's
        magnitude is monotone in look_mix and reaches exactly the source at 0,
        which is the whole contract of the knob.
        """
        m = self.look_mix
        if m >= 1.0:
            return self.effects
        return EffectSpec(**{k: v * m for k, v in self.effects.model_dump().items()})

    def has_hue_qualifiers(self, tol: float = _ID_TOL) -> bool:
        """True if any hue band is off identity. lut.py uses this to pick a
        LUT size; apply() uses it to skip the band pass entirely."""
        return any(
            abs(b.sat - 1.0) > tol or abs(b.lum) > tol
            for b in (getattr(self, f) for f in HUE_FIELDS)
        )

    def _has_tonal_split(self, tol: float = _ID_TOL) -> bool:
        return (
            np.any(np.abs(self.shadow_tint.as_array()) > tol)
            or np.any(np.abs(self.highlight_tint.as_array()) > tol)
            or abs(self.shadow_lift) > tol
            or abs(self.highlight_lift) > tol
        )

    # ---- evaluation -------------------------------------------------------

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply this grade to an array of shape (..., 3), float in [0, 1].

        Pure and vectorized: used both to bake the LUT and to unit-test the
        math directly without going through ffmpeg. `self.effects` is spatial
        and is deliberately not touched here.
        """
        x = np.asarray(rgb, dtype=np.float64)
        if x.shape[-1] != 3:
            raise ValueError(f"expected trailing dimension 3, got shape {x.shape}")

        # 0. Keep the input for the look_mix lerp at step 9.
        src = x

        # 1-4. Exposure, CDL, white balance, saturation.
        x = self.through_saturation(x)

        # 5. Hue qualifiers. Guarded: the luma round trip inside costs ~1 ulp,
        #    so an unguarded pass would break the identity hash silently.
        if self.has_hue_qualifiers():
            bands = [getattr(self, f) for f in HUE_FIELDS]
            x = _apply_hue_bands(
                x,
                np.array([b.sat for b in bands], dtype=np.float64),
                np.array([b.lum for b in bands], dtype=np.float64),
            )

        # 6. Tonal split. Exact even unguarded (x + 0.0 == x); guarded anyway to
        #    skip a whole luma pass.
        if self._has_tonal_split():
            x = _apply_tonal_split(
                x,
                self.shadow_tint.as_array(), self.shadow_lift,
                self.highlight_tint.as_array(), self.highlight_lift,
            )

        # 7. Highlight shoulder. At rolloff 0 this inserts literally no code and
        #    the hard clip inside step 8 stays the sole range enforcement — that
        #    is what makes rolloff=0 bit-for-bit identical to the old behaviour.
        if self.highlight_rolloff > 1e-9:  # epsilon, not 0: knee -> 1 divides by ~0
            x = _rolloff(x, self.highlight_rolloff)

        # 8. Contrast S-curve around the pivot. Its np.clip is now effectively
        #    only the floor guard when rolloff is on, but it is NOT removable:
        #    _s_curve is non-monotonic on out-of-range input.
        # `contrast_balance` needs NO guard of its own, and that is the whole
        # reason it is coupled multiplicatively rather than additively: it only
        # ever scales `contrast`, so contrast == 0 already skips it and the
        # identity grid stays bit-for-bit. An additive balance would be a
        # second thing that can move the picture with contrast at 0, and would
        # need its own guard and its own row in the bitwise-no-op test.
        if abs(self.contrast) > 1e-9:
            x = _s_curve(np.clip(x, 0.0, 1.0), self.contrast, self.pivot,
                         self.contrast_balance)

        # 9. Mix the whole look back toward the source. Outermost by design.
        #    Guarded: src + (x - src) * 1.0 is not a bitwise identity.
        if self.look_mix < 1.0:
            s = np.clip(src, 0.0, 1.0)
            x = s + (x - s) * self.look_mix

        return np.clip(x, 0.0, 1.0)

    def through_saturation(self, rgb: np.ndarray) -> np.ndarray:
        """Steps 1-4 of apply(): exposure, CDL, white balance, saturation.

        Split out of apply() so a solver can see the exact array apply() hands
        to the hue qualifiers at step 5 -- scripts/build_looks.py fits the six
        bands against it. A re-implementation of these four steps in the solver
        would be free to drift out of step with the renderer, and the corpus
        would then describe a look the renderer does not produce.
        """
        x = np.asarray(rgb, dtype=np.float64)

        # 1. Exposure, in stops, before the CDL so `offset` stays an absolute lift.
        # _ID_TOL, not _TOL: the guard must agree with is_identity(), or there
        # is a window (1e-12 < |exposure| < 1e-9) where is_identity() says True
        # while apply() no longer reproduces the identity grid bit-for-bit --
        # exactly the silent hash-gate corruption class.
        if abs(self.exposure) > _ID_TOL:
            x = x * (2.0 ** self.exposure)

        # 2. CDL. Clamp before the power step: a fractional power of a negative
        #    number is NaN, and offset can legitimately push values below zero.
        x = x * self.slope.as_array() + self.offset.as_array()
        x = np.clip(x, 0.0, None)
        power = np.clip(self.power.as_array(), 1e-6, None)
        x = np.power(x, power)

        # 3. White balance as channel gains.
        x = x * self._wb_gains()

        # 4. Saturation around Rec.709 luma.
        luma = np.sum(x * LUMA, axis=-1, keepdims=True)
        return luma + (x - luma) * self.saturation

    def _wb_gains(self) -> np.ndarray:
        t = self.temperature / TEMP_FULL
        gains = np.array([1.0 + t * TEMP_GAIN, 1.0, 1.0 - t * TEMP_GAIN], dtype=np.float64)
        gains[1] -= self.tint * TINT_GAIN  # +tint = magenta = pull green down
        return np.clip(gains, 1e-6, None)

    # ---- LLM interop ------------------------------------------------------

    @staticmethod
    def llm_json_schema() -> dict:
        """Strict JSON schema for constrained decoding.

        `rationale` is included so the model explains itself in-band; every
        property is required and additionalProperties is False, which is what
        Groq's strict mode actually enforces. A property that is present but
        NOT required is rejected outright, so the two lists must stay in step.
        """
        num = {"type": "number"}
        rgb = {
            "type": "object",
            "properties": {"r": num, "g": num, "b": num},
            "required": ["r", "g", "b"],
            "additionalProperties": False,
        }
        hue = {
            "type": "object",
            "properties": {"sat": num, "lum": num},
            "required": ["sat", "lum"],
            "additionalProperties": False,
        }
        effect_keys = list(EffectSpec.model_fields)
        effects = {
            "type": "object",
            "properties": {k: num for k in effect_keys},
            "required": effect_keys,
            "additionalProperties": False,
        }
        props = {
            "slope": rgb, "offset": rgb, "power": rgb,
            "saturation": num, "temperature": num, "tint": num,
            "contrast": num, "pivot": num, "contrast_balance": num,
            "exposure": num, "look_mix": num, "highlight_rolloff": num,
            "shadow_tint": rgb, "highlight_tint": rgb,
            "shadow_lift": num, "highlight_lift": num,
            "hue_red": hue, "hue_yellow": hue, "hue_green": hue,
            "hue_cyan": hue, "hue_blue": hue, "hue_magenta": hue,
            "effects": effects,
            "rationale": {"type": "string"},
        }
        return {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        }

    def sanitize(self) -> "GradeSpec":
        """Clamp LLM output into physically sensible ranges.

        A schema guarantees *shape*, never *sanity* — a model can legally emit
        power=0 (division blowup) or saturation=-4. Called on every LLM result.
        """
        d = self.model_dump()
        d["power"] = {k: float(np.clip(v, 0.05, 8.0)) for k, v in d["power"].items()}
        d["slope"] = {k: float(np.clip(v, 0.0, 8.0)) for k, v in d["slope"].items()}
        d["offset"] = {k: float(np.clip(v, -1.0, 1.0)) for k, v in d["offset"].items()}
        d["saturation"] = float(np.clip(d["saturation"], 0.0, 4.0))
        d["temperature"] = float(np.clip(d["temperature"], -6000.0, 6000.0))
        d["tint"] = float(np.clip(d["tint"], -2.0, 2.0))
        d["contrast"] = float(np.clip(d["contrast"], -1.0, 1.0))
        d["pivot"] = float(np.clip(d["pivot"], 0.05, 0.95))
        d["contrast_balance"] = float(np.clip(d["contrast_balance"], -1.0, 1.0))

        d["exposure"] = float(np.clip(d["exposure"], -4.0, 4.0))
        d["look_mix"] = float(np.clip(d["look_mix"], 0.0, 1.0))
        d["highlight_rolloff"] = float(np.clip(d["highlight_rolloff"], 0.0, 1.0))

        # Tints are added straight onto pixel values, so their sane range is
        # much tighter than a gain's.
        for k in ("shadow_tint", "highlight_tint"):
            d[k] = {c: float(np.clip(v, -0.5, 0.5)) for c, v in d[k].items()}
        d["shadow_lift"] = float(np.clip(d["shadow_lift"], -0.5, 0.5))
        d["highlight_lift"] = float(np.clip(d["highlight_lift"], -0.5, 0.5))

        for f in HUE_FIELDS:
            d[f] = {
                "sat": float(np.clip(d[f]["sat"], 0.0, 4.0)),
                "lum": float(np.clip(d[f]["lum"], -0.5, 0.5)),
            }

        e = d["effects"]
        for k in ("denoise", "glow", "grain"):
            e[k] = float(np.clip(e[k], 0.0, 1.0))
        for k in ("softness", "vignette", "fringe"):
            e[k] = float(np.clip(e[k], -1.0, 1.0))

        # Non-finite input (a model emitting 1e400, or NaN via a hand-edited
        # session) would survive every clip above and poison the LUT.
        return GradeSpec(**_scrub(d, GradeSpec.identity().model_dump()))


def _scrub(d: dict, ident: dict) -> dict:
    """Replace any non-finite number with its identity value, recursively."""
    for k, v in d.items():
        if isinstance(v, dict):
            _scrub(v, ident[k])
        elif isinstance(v, float) and not np.isfinite(v):
            d[k] = ident[k]
    return d


# ---- tonal split ----------------------------------------------------------


def _apply_tonal_split(
    x: np.ndarray, s_tint: np.ndarray, s_lift: float,
    h_tint: np.ndarray, h_lift: float,
) -> np.ndarray:
    """Shadow/highlight tint + lift with two disjoint smoothstep masks.

    Luma is a LINEAR functional, so subtracting a tint's own luma makes it
    exactly chroma-only:  L(x + (t - L(t))) == L(x)  and  L(x + lift) ==
    L(x) + lift. Tint moves colour and nothing else; lift moves brightness and
    nothing else. Two knobs, two axes, no crosstalk — exactly, not approximately.
    """
    # Clip first so a blown highlight reads as 1.0 rather than running off the
    # top of the mask.
    L = np.sum(np.clip(x, 0.0, 1.0) * LUMA, axis=-1, keepdims=True)
    lo, hi = SPLIT_CROSSOVER, 1.0 - SPLIT_CROSSOVER  # both 0.5; constants, never 0
    w_s = 1.0 - _smoothstep(np.clip(L / lo, 0.0, 1.0))       # 1 at black, 0 at L >= 0.5
    w_h = _smoothstep(np.clip((L - SPLIT_CROSSOVER) / hi, 0.0, 1.0))  # 0 at L <= 0.5
    ts = s_tint - float(s_tint @ LUMA)
    th = h_tint - float(h_tint @ LUMA)
    return x + w_s * (ts + s_lift) + w_h * (th + h_lift)


# ---- hue qualifiers -------------------------------------------------------


def _hue_chroma(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hue in degrees, chroma) via standard HSV sextants. Branchless, and
    0/0 is impossible: the divisor is forced to 1 wherever chroma is zero."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=-1)
    c = mx - x.min(axis=-1)
    safe = np.where(c > _TOL, c, 1.0)
    h = 60.0 * np.where(
        mx == r,
        ((g - b) / safe) % 6.0,
        np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    return np.where(c > _TOL, h, 0.0), c


def _apply_hue_bands(x: np.ndarray, sats: np.ndarray, lums: np.ndarray) -> np.ndarray:
    """Six smoothstepped triangular hue bands, chroma-gated, luma-preserving."""
    h, c = _hue_chroma(x)
    d = ((h[..., None] - HUE_CENTERS + 180.0) % 360.0) - 180.0
    w = _smoothstep(np.clip(1.0 - np.abs(d) / HUE_HALFWIDTH, 0.0, 1.0))  # (..., 6)
    gate = _smoothstep(np.clip(c / HUE_CHROMA_GATE, 0.0, 1.0))

    # sum(w) == 1, so this interpolates between adjacent bands rather than
    # accumulating them.
    s = 1.0 + gate * ((w * sats).sum(axis=-1) - 1.0)
    o = gate * (w * lums).sum(axis=-1)
    L = np.sum(x * LUMA, axis=-1, keepdims=True)
    return L + (x - L) * s[..., None] + o[..., None]


# ---- highlight rolloff ----------------------------------------------------


def _rolloff(x: np.ndarray, rolloff: float) -> np.ndarray:
    """Extended-Reinhard highlight shoulder: monotone, C1 at the knee, max 1.

    NO soft shoulder can map 1 -> 1. If f is monotone, f(x) == x on [0,1] and
    f <= 1 everywhere, then f is identically 1 above 1 — that IS hard clipping.
    A real shoulder must pull legal whites below 1, so rolloff MUST default to
    0 and 0 MUST mean today's hard clip. Measured f(1) = 0.976 / 0.928 / 0.856
    / 0.760 at rolloff 0.1 / 0.3 / 0.6 / 1.0. That white loss is unavoidable.

    g'(y) = (1 + 2y/W^2 + y^2/W^2) / (1+y)^2 > 0 everywhere, so the curve is
    strictly increasing. This matters more than it looks: a 3D LUT baked from a
    non-monotone curve INVERTS highlights.

    Motivation, measured on a 4096-step ramp: slope=1.3 — inside the 0.7..1.4
    range the model is told is sane — pins 23.1% of the ramp at pure white;
    slope=1.6 pins 37.5%. Every "brighter" prompt has been doing this.
    """
    knee = 1.0 - 0.5 * float(rolloff)
    s = 1.0 - knee  # > 0: the caller gated on rolloff > 0
    y = np.maximum(x - knee, 0.0) / s  # maximum() first: y = -1 would divide badly
    g = y * (1.0 + y / SHOULDER_SPAN**2) / (1.0 + y)
    return np.where(x <= knee, x, knee + s * np.minimum(g, 1.0))


# ---- contrast helpers -----------------------------------------------------


def _pivot_warp(x: np.ndarray, pivot: float) -> np.ndarray:
    """Map [0,1] to [0,1] piecewise-linearly so `pivot` lands on 0.5."""
    p = float(np.clip(pivot, 1e-3, 1 - 1e-3))
    return np.where(x < p, 0.5 * x / p, 0.5 + 0.5 * (x - p) / (1 - p))


def _pivot_unwarp(u: np.ndarray, pivot: float) -> np.ndarray:
    """Inverse of _pivot_warp."""
    p = float(np.clip(pivot, 1e-3, 1 - 1e-3))
    return np.where(u < 0.5, u * 2.0 * p, p + (u - 0.5) * 2.0 * (1 - p))


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3.0 - 2.0 * u)


def _inv_smoothstep(u: np.ndarray) -> np.ndarray:
    """Analytic inverse of smoothstep — the contrast-*reducing* direction."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(np.clip(1.0 - 2.0 * u, -1.0, 1.0)) / 3.0)


def _s_curve(x: np.ndarray, contrast: float, pivot: float,
             balance: float = 0.0) -> np.ndarray:
    """Monotonic S-curve fixing 0, 1 and `pivot`, optionally aimed at one end.

    Blends identity toward smoothstep (contrast > 0) or its inverse
    (contrast < 0), in a space warped so the pivot sits at 0.5.

    `balance` sweeps the blend WEIGHT across that warped scale: -1 puts the
    whole S in the shadows (a crushed toe, the shoulder left alone), +1 in the
    highlights, 0 is the symmetric curve this function has always been. It is
    the answer to "crush the blacks but keep the highlights soft", which one
    symmetric knob cannot express -- measured at pivot 0.435 on a 20001-step
    ramp, contrast 0.44 moves y(0.08) 0.0800 -> 0.0539 but drags y(0.95)
    0.9500 -> 0.9692 with it, where balance -1 reaches y(0.08) 0.0326 and
    leaves y(0.95) at 0.9517.

    MULTIPLICATIVE, not additive, for three separate reasons:

      * `sign(s) == sign(contrast)` everywhere, so the smoothstep /
        inv-smoothstep choice stays a SCALAR branch. A strength that crossed
        zero mid-curve would switch targets per element and put a C0 kink in
        the transfer function.
      * balance alone cannot move a pixel, so `apply()`'s existing
        `abs(contrast) > 1e-9` guard still covers this step exactly and the
        identity grid stays bit-for-bit. An additive balance needs a second
        guard and a second row in the bitwise-no-op test.
      * it never parks `_inv_smoothstep` at an endpoint. That matters: the
        inverse goes as sqrt(u/3) near 0, so its derivative is unbounded and
        33^3 reconstruction of NEGATIVE contrast is already poor -- measured
        6.91 code values at contrast -1.0, and 65^3 only gets it to 4.89. An
        additive form reaches lo > 0 > hi, which buys ~10% of midtone slope
        and costs 20x the LUT error.

    Monotone over the whole clamped space: swept in _self_check, 0
    non-monotone of 8405 (contrast x balance on 41x41, five pivots).
    """
    u = _pivot_warp(np.clip(x, 0.0, 1.0), pivot)
    c = float(np.clip(contrast, -1.0, 1.0))
    b = float(np.clip(balance, -1.0, 1.0))
    target = _smoothstep(u) if c > 0 else _inv_smoothstep(u)
    if b == 0.0:
        # Bit-for-bit the curve this was before balance existed. `abs(c) *
        # np.ones_like(u)` is not the same array as `abs(c)` in the last ulp,
        # so the scalar path is a guard, not a shortcut.
        return _pivot_unwarp(u + (target - u) * abs(c), pivot)
    s = np.clip(c * (1.0 + b * (2.0 * u - 1.0)), -1.0, 1.0)
    return _pivot_unwarp(u + (target - u) * np.abs(s), pivot)


# ---- self-check -----------------------------------------------------------


def _self_check() -> None:
    """`python -m ragvid.spec` — the properties the evaluation order depends on.

    Not a substitute for the test suite; these are the claims that are easy to
    break with an innocent-looking reordering and that a green suite would not
    catch (see the atol=1e-6 trap in the module docstring).
    """
    import hashlib

    from .lut import _grid

    grid = _grid(33)
    table = GradeSpec.identity().apply(grid)
    # The hard gate: identity must be BIT-identical, not merely close.
    assert hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest() == (
        "517467be3ba6b7a8afe71a05c847061dc597f0ea92e41b422164b579fbc74291"
    ), "identity LUT changed -- every saved grade just shifted"
    assert np.array_equal(table, grid)
    assert GradeSpec.identity().is_identity()
    assert not GradeSpec(exposure=3.0).is_identity()
    assert not GradeSpec(effects=EffectSpec(grain=0.4)).is_identity()

    # Hue band weights are an exact partition of unity, so no hue is unweighted
    # and none is double-counted.
    h = np.linspace(0.0, 360.0, 10001)
    d = ((h[..., None] - HUE_CENTERS + 180.0) % 360.0) - 180.0
    assert np.abs(_smoothstep(np.clip(1.0 - np.abs(d) / HUE_HALFWIDTH, 0, 1)).sum(-1) - 1).max() < 1e-14

    # Tonal split masks: sum <= 1 and DISJOINT support, so midtones cannot be
    # double-counted even in principle.
    L = np.linspace(0.0, 1.0, 10001)
    w_s = 1.0 - _smoothstep(np.clip(L / SPLIT_CROSSOVER, 0, 1))
    w_h = _smoothstep(np.clip((L - SPLIT_CROSSOVER) / (1 - SPLIT_CROSSOVER), 0, 1))
    assert (w_s + w_h).max() <= 1.0 and not np.any((w_s > 0) & (w_h > 0))

    # Tint is exactly chroma-only, lift exactly luma-only. No crosstalk.
    rng = np.random.default_rng(0)
    x = rng.random((5000, 3))
    only_tint = _apply_tonal_split(x, np.array([0.1, -0.05, 0.2]), 0.0, np.array([-0.1, 0.0, 0.3]), 0.0)
    assert np.abs((only_tint - x) @ LUMA).max() < 1e-15
    # ... but only while the result stays in gamut. apply()'s final clip
    # truncates whatever left [0,1], and truncation is not luma-preserving:
    # measured on the 65^3 grid, 6.1% of grid points clip and their luma error
    # reaches 0.0198 (5 code values). That is the price of a legal image, not a
    # bug in the split -- which is why the guarantee is stated on the helper.
    clipped = np.clip(only_tint, 0.0, 1.0)
    assert np.abs((clipped - np.clip(x, 0, 1)) @ LUMA).max() > 1e-9
    only_lift = _apply_tonal_split(x, np.zeros(3), 0.07, np.zeros(3), -0.03) - x
    assert np.abs(only_lift - (only_lift @ LUMA)[:, None]).max() < 1e-15

    # A non-monotone CURVE inverts tone in the baked .cube just as a
    # non-monotone shoulder would. The asymmetric strength makes the derivative
    # argument non-obvious, so it is swept rather than argued.
    sweep = np.linspace(0.0, 1.0, 40001)
    worst = 0.0
    for c in np.linspace(-1.0, 1.0, 41):
        for b in np.linspace(-1.0, 1.0, 41):
            for pv in (0.05, 0.25, 0.435, 0.65, 0.95):
                worst = min(worst, float(np.diff(_s_curve(sweep, c, pv, b)).min()))
    assert worst >= -1e-15, f"the contrast curve is non-monotone somewhere: {worst}"

    # ...and balance 0 must be BIT-for-bit the curve from before it existed, or
    # every saved grade shifted underneath the people who saved them.
    z = np.random.default_rng(0).random(200000)
    for c in (-1.0, -0.6, -0.22, 0.22, 0.5, 1.0):
        for pv in (0.25, 0.435, 0.65):
            assert np.array_equal(_s_curve(z, c, pv), _s_curve(z, c, pv, 0.0)), (c, pv)

    # A non-monotone shoulder would INVERT highlights in the baked .cube.
    ramp = np.linspace(0.0, 4.0, 400001)
    for r, expect in ((0.1, 0.976), (0.3, 0.928), (0.6, 0.856), (1.0, 0.760)):
        y = _rolloff(ramp, r)
        assert np.all(np.diff(y) >= 0.0) and abs(y.max() - 1.0) < 1e-12
        assert abs(_rolloff(np.array([1.0]), r)[0] - expect) < 5e-4  # unavoidable white loss

    # Nothing non-finite may reach the LUT, at any sanitized setting.
    hot = GradeSpec(
        slope=RGB(r=9e9, g=-3.0, b=float("nan")), offset=RGB.of(-5.0), power=RGB.of(0.0),
        saturation=float("inf"), exposure=99.0, contrast=-2.0, contrast_balance=9.0,
        highlight_rolloff=5.0,
        look_mix=-1.0, shadow_tint=RGB.of(9.0), highlight_lift=float("-inf"),
        hue_red=HueBand(sat=-4.0, lum=7.0), effects=EffectSpec(glow=float("nan")),
    ).sanitize()
    assert np.all(np.isfinite(hot.apply(grid)))
    assert hot.apply(grid).min() >= 0.0 and hot.apply(grid).max() <= 1.0
    print("spec self-check OK")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
