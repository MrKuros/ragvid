"""Regions, and the container that lets a grade apply to part of a frame.

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

TWO MASK SOURCES, ONE CONTRACT (roadmap B1 then B2). A mask comes either from
GEOMETRY -- linear and radial, computed analytically from width and height --
or from CONTENT, a class name resolved by the local segmentation model in
segment.py. They are the same object with a different `shape`, because
everything downstream of `mask()` is identical: one float array, one 8-bit PNG,
one lerp. A second container would have bought a second serialisation, a second
render path and a second chance for the preview to disagree with the export.

The cost of the second source is one signature change and it is the only place
the two differ: a semantic mask needs the PICTURE, and `mask(width, height)`
does not have it. See Region.mask for why the frame is an optional trailing
argument rather than a second method or a required parameter.

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

from .segment import CLASSES
from .spec import GradeSpec, _smoothstep

# Shapes a SENTENCE can specify. Not a list of everything a mask could be: a
# person with no pointing device can say which edge, roughly how far in, and
# how soft -- and nothing more. Anything needing a bezier is a different UI.
# "semantic" is the one addition that is not geometry at all: the shape of the
# region is whatever the picture contains (segment.py).
Shape = Literal["linear", "radial", "semantic"]
Edge = Literal["top", "bottom", "left", "right"]

# Below this a divisor is zero. Softness 0 is a legal request (a hard edge),
# and it must produce a step within one pixel rather than a NaN.
_EPS = 1e-6

# Feather for a semantic mask, as a fraction of the frame's SHORT side, at
# softness 1.0. A frame-relative radius, not a pixel one, for the reason the
# whole class is frame-relative: the same Region has to mask a 4K export and a
# 480p preview identically.
#
# The number is measured, not chosen. segment.py returns a 128x128 grid, so a
# confident boundary upsampled to 720p is a step over ~5 px and the raw mask has
# a MAX SINGLE-ROW STEP OF 57.6 CODE VALUES across a horizon -- a visible
# staircase once it is written as an 8-bit PNG. Blur radius against that step,
# measured on the same frame at 1280x720:
#
#     radius (short side)   0.5%   1.0%   1.5%   2.0%   3.0%
#     max step, code values 38.2   24.9   16.6   13.2    8.5
#
# The default region softness is 0.5, so _FEATHER_FRAC = 0.04 puts the default
# at 2.0% and 13.2 code values -- the same neighbourhood as B1's geometric
# default (1.5/(0.4*96) = 10 code values on a 96-row mask). Going wider buys
# little and starts bleeding the sky grade into the treeline.
_FEATHER_FRAC = 0.04


def _resize(a: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resample of a float field to (height, width).

    Bilinear and not bicubic for the reason render.py's scale2ref gives: the
    input is a monotone probability ramp and bicubic overshoots at both ends of
    one, which puts a step back into the falloff we are here to remove.

    PIL's mode "F" carries float32 exactly, so this does not round-trip through
    8 bits -- write_png quantises once, at the end, and nowhere else.
    """
    from PIL import Image

    im = Image.fromarray(np.asarray(a, dtype=np.float32), mode="F")
    return np.asarray(im.resize((width, height), Image.BILINEAR), dtype=np.float64)


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur, radius `r` pixels, edges extended.

    Cumsum rather than a convolution or a dependency: it is O(n) in the frame
    regardless of radius, and the radius here is 2% of the short side (22 px on
    a 1080p frame), where a naive kernel is 45x the work per axis. scipy would
    do it in one line and is 30 MB of dependency this project does not have.

    One pass, not two. A single box gives a piecewise-linear ramp, and
    _smoothstep on top of it is C1 -- two boxes would be smoother and would
    widen the ramp by another radius for no measured benefit.
    """
    if r < 1:
        return a
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
        zero = np.zeros_like(np.take(c, [0], axis))
        c = np.concatenate([zero, c], axis=axis)
        n = c.shape[axis]
        a = (np.take(c, range(k, n), axis) - np.take(c, range(0, n - k), axis)) / k
    return a


class Region(BaseModel):
    """Where a correction applies, in frame-relative coordinates.

    Everything is a fraction of the frame, never pixels: the same Region has to
    mask a 4K export and a 480p preview identically, and the preview is where
    the user decides whether the grade is right.

    `shape` selects which fields are read -- `edge`/`extent` for linear,
    `cx`/`cy`/`rx`/`ry` for radial, `subject` for semantic. A discriminated
    union would be tidier and would cost a second model, a discriminator and a
    serialisation branch to say the same thing.
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

    # semantic: which segment.CLASSES word the mask comes from. Named
    # `subject` and not `class`/`cls`: the value is what a person points at in
    # the picture, and the two obvious spellings are a Python keyword and the
    # conventional name of a method's first argument.
    subject: str = Field("", description="Semantic only. A segment.CLASSES word: sky, foliage, person, water, buildings.")

    softness: float = Field(0.5, description="Width of the falloff, as a fraction of the region. 0 = hard edge.")
    invert: bool = Field(False, description="Swap inside and outside.")

    @property
    def needs_frame(self) -> bool:
        """True when mask() cannot be answered from width and height alone.

        Exists so a caller (project.bake_layers, a UI) can find out whether it
        must supply a frame WITHOUT reading `shape` and reimplementing this
        file's knowledge of which shapes are which.
        """
        return self.shape == "semantic"

    def mask(self, width: int, height: int, frame: np.ndarray | None = None) -> np.ndarray:
        """(height, width) float64 in [0, 1]. 1 = fully inside the region.

        Sampled at PIXEL CENTRES, which is what makes the array and the PNG
        written from it describe the same geometry as the pixels ffmpeg blends.

        The falloff is spec._smoothstep -- the same C1 curve the tonal split's
        masks use, deliberately reused rather than reinvented. A linear ramp
        has a gradient discontinuity at both ends of the falloff and an 8-bit
        encode makes that visible as a pair of bands (a Mach band is a gradient
        artifact, not a value one). The semantic branch reuses it too, on the
        blurred probability field rather than on a distance.

        `frame` IS THE SIGNATURE PROBLEM B2 POSES, and this is the answer.
        A semantic mask cannot be computed from a size -- it needs the picture --
        while every geometric one is fully determined by width and height. Three
        shapes were available:

          * a second method, `mask_from(frame)`. Rejected: every caller then
            branches on `shape` to decide which to call, which is precisely the
            knowledge this class exists to hold. render.py, GradeStack.apply and
            project.bake_layers would each carry a copy of it.
          * `frame` required. Rejected: it forces `None` through six geometric
            call sites for the overwhelmingly common case, and a caller that has
            no frame handy (baking a mask for a still, sizing a UI overlay) has
            to invent one.
          * optional trailing argument, and a semantic Region with no frame
            RAISES. Chosen. Geometric callers are untouched, the common case
            stays two arguments, and the one thing that must never happen --
            a semantic region silently resolving to all-ones or all-zeros and
            grading the wrong pixels -- is a loud failure instead. `needs_frame`
            lets a caller ask in advance rather than catch.

        `frame` is an (h, w, 3) image, uint8 or float in [0, 1]. Its own size is
        irrelevant to the result's size: the model resizes to 512x512 regardless
        and the probability field is resampled to (height, width) here, so the
        mask stays frame-relative exactly as the geometric ones are.
        """
        if width < 1 or height < 1:
            raise ValueError(f"mask needs a positive size, got {width}x{height}")
        if self.shape == "semantic":
            return self._semantic_mask(width, height, frame)
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

    def _semantic_mask(self, width: int, height: int, frame: np.ndarray | None) -> np.ndarray:
        """segment.py's 128x128 probability grid -> a feathered frame-sized mask.

        Order is resample, DECIDE, blur, smoothstep, and each step earns its
        place:

          * resample first, so the class boundary lands between grid cells
            rather than snapping to one -- deciding at 128x128 and enlarging
            afterwards throws away that eighth of a cell for nothing.
          * decide at segment.DECIDED, because a probability is not a mask.
            Softmax never reaches 0 or 1: measured on a synthetic sky, using the
            raw probability left NO pixel at exactly 0 and none at exactly 1, so
            every pixel in the frame carried a sliver of the layer. That breaks
            the property B1 shipped on -- "the bottom third of 'darken the top'
            is EXACTLY unchanged" -- and region.py's identity note depends on a
            mask of exactly 0 lerping bitwise. After the gate, on that same
            frame, 42.2% is bitwise untouched, 53.3% is fully graded and 4.6%
            is in the falloff.
          * box blur, because a decided 128-grid boundary blown up to a render
            is a staircase: 57.6 code values of single-step measured across a
            horizon at 720p, against 13.2 with the default feather.
          * _smoothstep last, because a box blur leaves a piecewise-linear ramp
            with a gradient kink at each end, which is the same Mach-band
            argument the geometric falloff already makes.
        """
        if frame is None:
            raise ValueError(
                f"a semantic region ({self.subject or '?'}) needs the frame it masks; "
                "pass mask(width, height, frame=rgb)"
            )
        from .segment import DECIDED, class_prob

        p = _resize(class_prob(frame, self.subject), width, height)
        r = int(round(max(float(self.softness), 0.0) * _FEATHER_FRAC * min(width, height)))
        m = _smoothstep(_box_blur((p > DECIDED).astype(np.float64), r))
        return (1.0 - m) if self.invert else np.ascontiguousarray(m)

    def write_png(self, path: str | Path, width: int, height: int,
                  frame: np.ndarray | None = None) -> str:
        """Write the mask as an 8-bit greyscale PNG for render.py's filter chain.

        8-bit, not 16: it becomes an ffmpeg alpha plane, and every alpha format
        overlay can blend with is 8-bit anyway. The quantisation costs a
        measured 0.58 code values worst case on a half-frame ramp, and rounding
        a monotone function stays monotone, so the falloff does not gain a step.
        """
        from PIL import Image  # pillow is already a dependency (probe.py reads stills with it)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = (self.mask(width, height, frame) * 255.0).round().astype(np.uint8)
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

    @property
    def needs_frame(self) -> bool:
        """True when baking this stack's masks requires a decoded frame.

        The question project.bake_layers has to answer before it can write the
        PNGs, asked once of the whole stack so a caller does not walk the layers
        itself.
        """
        return any(l.region.needs_frame for l in self.layers)

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
            # The SOURCE frame, not `out`: segmentation reads the picture, and
            # the picture is what the camera captured. Feeding it the graded
            # result would make the mask depend on the grade, so "make the sky
            # moody" would move its own mask as it took effect.
            m = layer.region.mask(w, h, frame=x)[..., None]
            # x + (g - x) * m, not a np.where on m: the lerp is exact at m == 0
            # and continuous everywhere, which is what "soft edge" means.
            out = out + (layer.spec.apply(out) - out) * m
        return out


# ---- the vocabulary a sentence can reach -----------------------------------
#
# intent.Target's region words -> a Region. The numbers here are the taste in
# this file and the only thing that should need tuning. They are NOT measured
# and cannot be: nothing in ClipStats knows where the interesting half of the
# frame is. That is what the semantic entries below answer instead -- the model
# knows, and it is the only thing in the project that does.
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
    # The semantic half (roadmap B2). One entry per segment.CLASSES word, built
    # from the vocabulary rather than listed again here: a word the model can
    # emit and this table cannot resolve compiles to a silent no-op, and the
    # only way to be sure the two lists agree is not to have two lists.
    # softness stays at the class default -- see _FEATHER_FRAC for the 13.2
    # code values that buys.
    **{word: Region(shape="semantic", subject=word) for word in CLASSES},
}


def for_target(target: str) -> Region | None:
    """The Region a `intent.Op.target` names, or None if it names a colour.

    One lookup for both mask sources: "the top" and "the sky" are the same kind
    of answer to "which pixels", so the compiler resolves them through one call
    and never learns that one of them needs a model.

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

    # Every semantic word resolves, and refuses to answer without a frame --
    # the failure that would otherwise be a grade on the wrong pixels.
    for word in CLASSES:
        r = for_target(word)
        assert r is not None and r.needs_frame and r.subject == word
        try:
            r.mask(16, 12)
        except ValueError as e:
            assert word in str(e)
        else:  # pragma: no cover
            raise AssertionError(f"{word} masked without a frame")

    # The feather is a real low-pass: a step blurred at radius r spans ~2r+1.
    step = np.zeros((64, 64)); step[:, 32:] = 1.0
    assert np.abs(np.diff(_box_blur(step, 5)[32])).max() <= 1.0 / 11 + 1e-12
    assert np.array_equal(_box_blur(step, 0), step), "softness 0 stays a hard edge"

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
