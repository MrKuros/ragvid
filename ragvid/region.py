"""Geometric regions, and the container that lets a grade apply to part of a frame.

THE LOAD-BEARING CONSTRAINT: ONE mask, generated ONCE, in numpy. `mask()` is
the only place the geometry exists. render.py never re-derives it in an ffmpeg
expression -- it is handed the PNG this module writes and blends with it. That
is not an optimisation, it is the reason the preview and the export agree: two
implementations of the same falloff (a `geq` expression and a numpy array)
would be two chances to disagree, and this project has already shipped that bug
once (see render.render_preview's comment about grading each tile before
stacking).

A GradeSpec is a per-pixel colour map: it can say WHAT to do and never WHERE.
So a region cannot live inside one, exactly as `EffectSpec` cannot -- same
category, same consequence, and it cannot bake into a .cube either. What lands
here instead is the outer container:

    GradeStack = base GradeSpec (the whole frame)
               + ordered [ (Region, GradeSpec) ] layers

EVALUATION ORDER, and it is as load-bearing as spec.py's:

    0. base.apply(x)                     the whole frame, as before
    1..n  for each layer, IN LIST ORDER:
          x = x + (layer.spec.apply(x) - x) * layer.region.mask()

Two things follow, both deliberate:

  * A layer grades the ACCUMULATED result, not the source. So "darken the top"
    means one more correction on top of the look, which is what a colourist's
    node graph does and what makes each layer's GradeSpec readable on its own --
    it holds one correction, not a copy of the base with one field changed.
  * Overlapping regions compose in list order and the later one wins in
    proportion to its mask. Composition is a lerp, so it is continuous: nothing
    steps at an overlap boundary.

IDENTITY IS PRESERVED BY CONSTRUCTION. With no layers, `apply` is literally
`base.apply`, so a flat stack is bit-for-bit today's output -- no lerp, no
`x + (g-x)*1.0` ulp (spec.py's docstring explains why that matters). Inside the
lerp, a mask of exactly 0 gives `x + (g-x)*0 == x` bitwise; a mask of exactly 1
does NOT give `g` bitwise, and that ~1 ulp is the price of the mask being a
first-class array rather than a special case. Measured on a full-frame mask
over 64x64 random pixels: max |stack - global| = 5.6e-16, i.e. 1.4e-13 of one
8-bit code value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .spec import GradeSpec, _smoothstep

# Shapes a SENTENCE can specify. Not a list of everything a mask could be: a
# person with no pointing device can say which edge, roughly how far in, and
# how soft -- and nothing more. Anything needing a bezier is a different UI.
Shape = Literal["linear", "radial"]
Edge = Literal["top", "bottom", "left", "right"]

# Below this a divisor is zero. Softness 0 is a legal request (a hard edge),
# and it must produce a step within one pixel rather than a NaN.
_EPS = 1e-6


class Region(BaseModel):
    """Where a correction applies, in frame-relative coordinates.

    Everything is a fraction of the frame, never pixels: the same Region has to
    mask a 4K export and a 480p preview identically, and the preview is where
    the user decides whether the grade is right.

    `shape` selects which fields are read -- `edge`/`extent` for linear,
    `cx`/`cy`/`rx`/`ry` for radial. A discriminated union would be tidier and
    would cost a second model, a discriminator and a serialisation branch to
    say the same thing.
    """

    shape: Shape = "linear"

    # linear: the side that is at FULL strength, and how far in the falloff is
    # centred. extent 0.5 = the ramp's midpoint sits across the middle of the
    # frame, so "the top" is the top half.
    edge: Edge = "top"
    extent: float = Field(0.5, description="Linear only. Fraction of the frame from `edge` to the middle of the falloff.")

    # radial: centre and radii, both fractions of the frame's own width/height,
    # so an ellipse on a 16:9 frame stays an ellipse on a 4:3 one.
    cx: float = Field(0.5, description="Radial only. Centre X, 0..1 across the frame.")
    cy: float = Field(0.5, description="Radial only. Centre Y, 0..1 down the frame.")
    rx: float = Field(0.5, description="Radial only. X radius where the mask reaches 0.")
    ry: float = Field(0.5, description="Radial only. Y radius where the mask reaches 0.")

    softness: float = Field(0.5, description="Width of the falloff, as a fraction of the region. 0 = hard edge.")
    invert: bool = Field(False, description="Swap inside and outside.")

    def mask(self, width: int, height: int) -> np.ndarray:
        """(height, width) float64 in [0, 1]. 1 = fully inside the region.

        Sampled at PIXEL CENTRES, which is what makes the array and the PNG
        written from it describe the same geometry as the pixels ffmpeg blends.

        The falloff is spec._smoothstep -- the same C1 curve the tonal split's
        masks use, deliberately reused rather than reinvented. A linear ramp
        has a gradient discontinuity at both ends of the falloff and an 8-bit
        encode makes that visible as a pair of bands (a Mach band is a gradient
        artifact, not a value one).
        """
        if width < 1 or height < 1:
            raise ValueError(f"mask needs a positive size, got {width}x{height}")
        x = (np.arange(width) + 0.5) / width
        y = ((np.arange(height) + 0.5) / height)[:, None]
        s = max(float(self.softness), _EPS)

        if self.shape == "radial":
            r = np.hypot(
                (x - self.cx) / max(abs(float(self.rx)), _EPS),
                (y - self.cy) / max(abs(float(self.ry)), _EPS),
            )
            t = (1.0 - r) / s  # 1 at the centre, 0 at r == 1
        else:
            u = {"top": y, "bottom": 1.0 - y, "left": x, "right": 1.0 - x}[self.edge]
            t = (float(self.extent) + 0.5 * s - u) / s

        m = _smoothstep(np.clip(t, 0.0, 1.0))
        m = np.broadcast_to(m, (height, width))
        return (1.0 - m) if self.invert else np.ascontiguousarray(m)

    def write_png(self, path: str | Path, width: int, height: int) -> str:
        """Write the mask as an 8-bit greyscale PNG for render.py's filter chain.

        8-bit, not 16: it becomes an ffmpeg alpha plane, and every alpha format
        overlay can blend with is 8-bit anyway. The quantisation costs a
        measured 0.58 code values worst case on a half-frame ramp, and rounding
        a monotone function stays monotone, so the falloff does not gain a step.
        """
        from PIL import Image  # pillow is already a dependency (probe.py reads stills with it)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = (self.mask(width, height) * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out)
        return str(out)


class Layer(BaseModel):
    """One correction and where it lands. `spec` is a whole GradeSpec because a
    region's correction is not a lesser thing than a global one -- it is the
    same 43 numbers pointed at fewer pixels."""

    region: Region
    spec: GradeSpec


class GradeStack(BaseModel):
    """A base grade plus ordered regional layers. The thing a renderer needs.

    `base` alone is what every consumer that predates regions still gets, and
    `layers == []` is the overwhelmingly common case -- which is why this is a
    container around GradeSpec rather than a replacement for it.
    """

    base: GradeSpec
    layers: list[Layer] = Field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        """True when this stack is exactly one GradeSpec, i.e. today's grade."""
        return not self.layers

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Grade an IMAGE, shape (h, w, 3), float in [0, 1].

        Note the shape requirement: GradeSpec.apply takes any (..., 3) because a
        colour map does not care how the pixels are arranged. This one does --
        that is the whole point of a region, and it is why a stack cannot bake
        into a .cube.
        """
        x = np.asarray(rgb, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"expected an (h, w, 3) image, got shape {x.shape}")
        out = self.base.apply(x)
        h, w = x.shape[:2]
        for layer in self.layers:
            m = layer.region.mask(w, h)[..., None]
            # x + (g - x) * m, not a np.where on m: the lerp is exact at m == 0
            # and continuous everywhere, which is what "soft edge" means.
            out = out + (layer.spec.apply(out) - out) * m
        return out


# ---- the vocabulary a sentence can reach -----------------------------------
#
# intent.Target's region words -> geometry. The numbers here are the taste in
# this file and the only thing that should need tuning. They are NOT measured
# and cannot be: nothing in ClipStats knows where the interesting half of the
# frame is (that is roadmap B2, a segmentation model, explicitly not this).
#
# extent 0.4 with softness 0.4 puts the ramp between 0.2 and 0.6: the near
# quarter is at full strength, the far 40% is untouched, and the falloff spans
# the frame's own middle. The constraint that fixes it is "the bottom third of
# 'darken the top' must be EXACTLY unchanged" -- a region a person can point at
# in the result. A wider ramp (0.45/0.5, the first pass here) reached 0.70 and
# left a measured 0.6% of the grade in the bottom third, which is a region that
# does not mean what its name says.
_FOR_TARGET: dict[str, Region] = {
    "top": Region(shape="linear", edge="top", extent=0.4, softness=0.4),
    "bottom": Region(shape="linear", edge="bottom", extent=0.4, softness=0.4),
    "left": Region(shape="linear", edge="left", extent=0.4, softness=0.4),
    "right": Region(shape="linear", edge="right", extent=0.4, softness=0.4),
    # Wider than tall: a subject sits in the middle of the frame, and a circle
    # on a 16:9 frame either misses their shoulders or reaches the ceiling.
    "center": Region(shape="radial", cx=0.5, cy=0.5, rx=0.6, ry=0.75, softness=0.7),
    "edges": Region(shape="radial", cx=0.5, cy=0.5, rx=0.6, ry=0.75, softness=0.7, invert=True),
}


def for_target(target: str) -> Region | None:
    """The Region a `intent.Op.target` names, or None if it names a colour.

    Returns a copy: Regions are mutable models and a caller editing one must not
    edit the vocabulary.
    """
    r = _FOR_TARGET.get(target)
    return r.model_copy(deep=True) if r else None


def _self_check() -> None:
    """`python -m ragvid.region` -- the properties the mask contract rests on."""
    r = for_target("top")
    m = r.mask(64, 96)
    col = m[:, 0]
    assert np.all(np.diff(col) <= 0.0), "the top mask must fall monotonically"
    assert col[0] == 1.0 and col[-1] == 0.0
    assert np.allclose(m, m[:, :1]), "a linear top mask cannot vary across x"

    # No step anywhere: the largest jump between adjacent rows is bounded by the
    # smoothstep's own maximum slope (1.5) over the ramp's height in pixels.
    assert np.abs(np.diff(col)).max() < 1.5 / (r.softness * 96) + 1e-12

    inv = for_target("edges").mask(80, 60)
    assert inv[30, 40] == 0.0 and inv[0, 0] > 0.9

    # A flat stack is bit-for-bit its base.
    img = np.random.default_rng(0).random((12, 16, 3))
    spec = GradeSpec(contrast=0.3, temperature=900.0)
    assert np.array_equal(GradeStack(base=spec).apply(img), spec.apply(img))

    # A full-frame region equals the same grade applied globally.
    full = Region(shape="linear", edge="top", extent=9.0, softness=_EPS)
    assert full.mask(16, 12).min() == 1.0
    stacked = GradeStack(base=GradeSpec.identity(), layers=[Layer(region=full, spec=spec)]).apply(img)
    assert np.abs(stacked - spec.apply(img)).max() < 1e-15
    print("region self-check OK")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
