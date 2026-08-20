"""The grade spec — the contract every other module codes against.

A GradeSpec is ~14 numbers describing a color grade. It is the *only* thing an
LLM ever produces in this project; pixels are never sent to a model. The .cube
LUT is a derived artifact baked from a spec, which is why refinement ("make it
less blue") is possible at all — you can adjust numbers, not a baked LUT.

The core is ASC CDL (slope/offset/power per channel + saturation), an industry
interchange standard, plus temperature/tint/contrast.

EVALUATION ORDER IS LOAD-BEARING. Reviewers and reimplementations must match:

    1. CDL       out = (x * slope + offset) ** power     [x clamped >= 0 first]
    2. white bal temperature/tint applied as per-channel gains
    3. saturation  lerp between luma and color
    4. contrast  S-curve around `pivot`

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
    rationale: str = Field("", description="One sentence explaining the look, for the user. Not used in math.")

    # ---- construction -----------------------------------------------------

    @classmethod
    def identity(cls) -> "GradeSpec":
        """A grade that must round-trip an image unchanged. Used as a test oracle."""
        return cls()

    def is_identity(self, tol: float = 1e-9) -> bool:
        i = GradeSpec.identity()
        return (
            np.allclose(self.slope.as_array(), i.slope.as_array(), atol=tol)
            and np.allclose(self.offset.as_array(), i.offset.as_array(), atol=tol)
            and np.allclose(self.power.as_array(), i.power.as_array(), atol=tol)
            and abs(self.saturation - 1.0) < tol
            and abs(self.temperature) < tol
            and abs(self.tint) < tol
            and abs(self.contrast) < tol
        )

    # ---- evaluation -------------------------------------------------------

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply this grade to an array of shape (..., 3), float in [0, 1].

        Pure and vectorized: used both to bake the LUT and to unit-test the
        math directly without going through ffmpeg.
        """
        x = np.asarray(rgb, dtype=np.float64)
        if x.shape[-1] != 3:
            raise ValueError(f"expected trailing dimension 3, got shape {x.shape}")

        # 1. CDL. Clamp before the power step: a fractional power of a negative
        #    number is NaN, and offset can legitimately push values below zero.
        x = x * self.slope.as_array() + self.offset.as_array()
        x = np.clip(x, 0.0, None)
        power = np.clip(self.power.as_array(), 1e-6, None)
        x = np.power(x, power)

        # 2. White balance as channel gains.
        x = x * self._wb_gains()

        # 3. Saturation around Rec.709 luma.
        luma = np.sum(x * LUMA, axis=-1, keepdims=True)
        x = luma + (x - luma) * self.saturation

        # 4. Contrast S-curve around the pivot.
        if abs(self.contrast) > 1e-9:
            x = _s_curve(np.clip(x, 0.0, 1.0), self.contrast, self.pivot)

        return np.clip(x, 0.0, 1.0)

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
        Groq's strict mode actually enforces.
        """
        num = {"type": "number"}
        rgb = {
            "type": "object",
            "properties": {"r": num, "g": num, "b": num},
            "required": ["r", "g", "b"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "slope": rgb, "offset": rgb, "power": rgb,
                "saturation": num, "temperature": num, "tint": num,
                "contrast": num, "pivot": num,
                "rationale": {"type": "string"},
            },
            "required": ["slope", "offset", "power", "saturation", "temperature",
                         "tint", "contrast", "pivot", "rationale"],
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
        return GradeSpec(**d)


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


def _s_curve(x: np.ndarray, contrast: float, pivot: float) -> np.ndarray:
    """Monotonic S-curve fixing 0, 1 and `pivot`.

    Blends identity toward smoothstep (contrast > 0) or its inverse
    (contrast < 0), in a space warped so the pivot sits at 0.5.
    """
    u = _pivot_warp(np.clip(x, 0.0, 1.0), pivot)
    c = float(np.clip(contrast, -1.0, 1.0))
    target = _smoothstep(u) if c > 0 else _inv_smoothstep(u)
    u2 = u + (target - u) * abs(c)
    return _pivot_unwarp(u2, pivot)
