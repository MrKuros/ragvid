"""Closed-form reference matching — linear color transfer, no LLM.

Per channel, in DISPLAY space — the space GradeSpec.apply and the baked .cube
operate in, and therefore the space probe.py reports its moments in:
    slope  = std_ref / std_src
    offset = aim_ref - slope * mean_src
which is exactly ASC CDL slope/offset with power = 1 (`aim_ref` is mean_ref
pushed back through the saturation lerp that apply() runs after the CDL).
Saturation is the ratio of the two chroma measures. temperature/tint stay at 0: the per-channel slope
already carries the color shift, so a temperature push on top would double it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ragvid.spec import LUMA, RGB, GradeSpec

if TYPE_CHECKING:  # pragma: no cover
    from ragvid.probe import ClipStats

# A channel flatter than this carries no scale information (solid color, blown
# highlight, letterbox bar): match its mean and leave the slope alone.
FLAT_STD = 1e-4
SLOPE_RANGE = (0.1, 4.0)
SAT_RANGE = (0.2, 3.0)


def match_reference(src: "ClipStats", ref: "ClipStats") -> GradeSpec:
    """Build a GradeSpec that moves `src`'s distribution onto `ref`'s."""
    m_src, s_src = src.mean.as_array(), src.std.as_array()
    m_ref, s_ref = ref.mean.as_array(), ref.std.as_array()

    flat = s_src < FLAT_STD
    slope = np.clip(np.divide(s_ref, s_src, out=np.ones(3), where=~flat), *SLOPE_RANGE)
    sat = 1.0
    if src.saturation > FLAT_STD:
        sat = float(np.clip(ref.saturation / src.saturation, *SAT_RANGE))

    # apply() runs the saturation lerp AFTER the CDL, which drags the channel
    # means back toward luma -- solving offset against m_ref directly lands the
    # *pre*-saturation mean there and the graded result short of it (measured:
    # ~14 code values median over random src/ref pairs). Aim the CDL at the
    # pre-saturation point that the lerp then maps onto m_ref.
    aim = m_ref
    if abs(sat) > 1e-6:
        luma_ref = float(m_ref @ LUMA)
        aim = luma_ref + (m_ref - luma_ref) / sat
    offset = aim - slope * m_src

    # Protect the highlights this grade is about to create. A reference match
    # routinely lands slope well above 1 -- matching a bright reference onto
    # darker footage is the ordinary case, not the extreme one -- and with
    # highlight_rolloff at its default 0 the result hard-clips. Measured on
    # test_files/ref_tvd.png against ironman.gif: slope 1.43, and 28.9% of a
    # 4096-step luminance ramp pinned at exactly 1.0, i.e. a third of the tonal
    # range flattened to detail-free white.
    #
    # power is 1 and temperature/tint are 0 on this path, so the largest value
    # the CDL can hand downstream is simply max(slope + offset). The shoulder
    # maps inputs up to 1 + 2*rolloff back into range (knee = 1 - rolloff/2,
    # white point = knee + 5*(1 - knee)), so solve for the peak this grade
    # actually produces rather than picking a constant and hoping.
    peak = float(np.max(slope + offset))
    rolloff = float(np.clip((peak - 1.0) / 2.0, 0.0, 1.0))

    warmth = "warmer" if slope[0] > slope[2] else "cooler"
    return GradeSpec(
        slope=RGB(r=slope[0], g=slope[1], b=slope[2]),
        offset=RGB(r=offset[0], g=offset[1], b=offset[2]),
        highlight_rolloff=rolloff,
        saturation=sat,
        rationale=(
            f"Matched reference: {warmth}, "
            f"{'brighter' if m_ref.mean() > m_src.mean() else 'darker'}, "
            f"{'more' if sat > 1 else 'less'} saturated."
        ),
    ).sanitize()
