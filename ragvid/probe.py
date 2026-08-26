"""Frame sampling and clip statistics.

Statistics are computed in DISPLAY space (sRGB code values / 255), because that
is the space the grade is applied in: spec.GradeSpec.apply, the baked .cube and
ffmpeg's lut3d all take and return display values in [0, 1]. match.py solves a
slope/offset from these moments and hands them straight to a GradeSpec, so the
moments have to live in the same space as the transform or the fit is simply
wrong (measurably so: on assets/ref_warm.png the linear-light fit hit match.py's
slope floor on two channels and came out ~4x too dark).

Linear light is the right space for *photometry*; it is the wrong space for
"what number do I feed a display-space LUT". srgb_to_linear/linear_to_srgb stay
public for callers that do want light.

Across frames we take the MEDIAN, not the mean: that is the entire reason for
sampling more than one frame. A single flash, cut, or fade-to-black frame moves
a mean and does not move a median.

Frames are sampled at 16 BITS PER CHANNEL (`rgb48le` over the pipe), not 8, and
DEPTH IS THE ONLY THING THAT CHANGED -- the analysis downscale to 256px stays,
for the reason given at _ANALYSIS_WIDTH.

The depth is invisible on ordinary 8-bit Rec.709 footage: measured against the
old PNG path, every moment agrees to 0.47 of an 8-bit code value and mean/std
to 0.01, i.e. quantisation and nothing else. It stops being invisible the moment
a non-linear transform runs before the statistics, which is exactly what
`input_lut` is. Measured on an underexposed synthetic S-Log3 clip (scene linear
0.0006-0.32, 10-bit) through logspace.bake_conversion's own .cube:

    crushed_low   0.0566 at 8 bits vs 0.0277 at 16 -- 2.0x over-reported
    p1            0.00392 on all three channels at 8 bits (that is 1/255: the
                  measurement has snapped to the grid) vs 0.0057/0.0054/0.0041
    saturation    5.7% high;  mean 2.0% high
    distinct luma levels surviving the LUT: 723 vs 36842

crushed_low and p1 are the two fields compiler.py consults to decide how far a
shadow verb may push, so at 8 bits the compiler was being told the blacks were
twice as dead as they are. On a 10-bit log source with no LUT at all the gap is
smaller but still real -- p1 out by 0.87 of a code value, 3.4% -- because 8-bit
sampling throws two bits away at the decode.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ragvid.platform import ffmpeg, ffprobe
from ragvid.spec import HUE_CENTERS, HUE_HALFWIDTH, LUMA, RGB, _smoothstep

# sRGB piecewise transfer function constants (IEC 61966-2-1).
_A = 0.055
_LIN_CUT = 0.0031308
_SRGB_CUT = 0.04045

# Frames are scaled to this width before analysis; statistics do not need
# full resolution and this keeps a 4K clip as cheap as a 480p one.
_ANALYSIS_WIDTH = 256

_EPS = 1e-8

# itemsize -> the value that means 1.0. uint8 frames come from PIL (probe_image)
# and from callers holding arrays already; uint16 frames come from _grab, i.e.
# from ffmpeg, and 65535 is NOT their full scale.
#
# swscale expands an N-bit source to 16 bits by LEFT SHIFT, not by replication:
# an 8-bit 255 arrives as 255 << 8 = 65280. Measured over a lossless 0-255 ramp,
# /65280 reproduces the source code values to 4.6e-5 (0.012 of an 8-bit code
# value); /65535 is 0.0038 low at white, which is a 0.4% GAIN error on every
# statistic -- not quantisation, just wrong, and it silently reports pure white
# as unclipped. A 10-bit source overshoots the other way by 0.001 (0.25 of a
# code value), hence the clip. tests/test_probe.py pins both ends so a change in
# swscale's convention fails loudly instead of shifting every grade 0.4%.
_FULL_SCALE = {1: 255.0, 2: 65280.0}


def srgb_to_linear(x: np.ndarray | float) -> np.ndarray:
    """sRGB display values in [0,1] -> linear light. Piecewise, not a 2.2 power."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= _SRGB_CUT, x / 12.92, ((x + _A) / (1 + _A)) ** 2.4)


def linear_to_srgb(x: np.ndarray | float) -> np.ndarray:
    """Linear light -> sRGB display values in [0,1]. Inverse of srgb_to_linear."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    return np.where(x <= _LIN_CUT, x * 12.92, (1 + _A) * x ** (1 / 2.4) - _A)


# Values within this of the ends of the range count as clipped / crushed. One
# 8-bit code value, so it means "already at the rail", not "nearly there".
# Deliberately still 8-bit-sized now that sampling is 16-bit: a 16-bit rail
# would be 1/65535, which no real footage sits inside, and clipped_high /
# crushed_low are read by compiler.py and vibe.py with their present meaning.
_RAIL = 1.5 / 255.0

# Cut detection. Luma histograms of adjacent samples, compared by total
# variation distance (half the L1 difference: 0 = identical, 1 = disjoint, and
# bounded either way whatever the frame size, which is what lets the threshold
# be a constant rather than something tuned per clip).
#
# A HISTOGRAM, not a frame difference, because a frame difference is what the
# naive detector gets wrong. Measured on synthetic 6s clips at n_frames=10:
#
#   two sources spliced      0.63  <- the thing being detected
#   dissolve over 12 frames  0.49
#   3-stop lighting change   0.74
#   whip pan, content fully replaced between samples   0.22   <- 1.00 by frame
#   slow pan                                           0.25      difference
#   continuous shot                                    0.03
#   assets/sample.mp4                                  0.08
#
# 0.35 sits 1.4x above the worst pan and 1.8x below a real splice. The two
# KNOWN false positives are in that table and are not fixable at this sample
# rate: a hard flash reads 1.00 (and, being one sample wide, is counted TWICE,
# once entering and once leaving), and a large exposure change reads as a cut
# because at 1s between samples there is nothing to distinguish a light coming
# on from a splice to a brighter shot. Both are cases where one look across the
# span is questionable anyway, so over-reporting is the safe direction.
_HIST_BINS = 32
_CUT_TV = 0.35


class ClipStats(BaseModel):
    """Measured description of a clip. Every field after `duration` was added
    later and MUST keep a default: three test modules construct ClipStats by
    keyword, and old session files on disk do not have the new keys either.
    """

    mean: RGB  # per-channel mean, sRGB display space, 0-1
    std: RGB  # per-channel std, sRGB display space
    saturation: float  # mean chroma 0-1
    frames_sampled: int
    width: int
    height: int
    duration: float

    # Percentiles say what mean/std cannot: whether the blacks sit at 0 or at
    # 0.08, and how far the real white point is from 1.
    p1: RGB = Field(default_factory=lambda: RGB.of(0.0))  # per-channel 1st percentile
    p50: RGB = Field(default_factory=lambda: RGB.of(0.0))  # per-channel median
    p99: RGB = Field(default_factory=lambda: RGB.of(0.0))  # per-channel 99th percentile

    clipped_high: float = 0.0  # fraction of pixels with any channel at the top rail
    crushed_low: float = 0.0  # fraction of pixels with any channel at the bottom rail
    dominant_hue: float = 0.0  # chroma-weighted circular mean hue, degrees 0-360
    frame_variance: float = 0.0  # spatial variance of luma within a frame

    # How much to BELIEVE dominant_hue: the length of the chroma-weighted mean
    # hue vector whose angle dominant_hue already is (circular statistics' R).
    # compiler._measure needs this and does not have it, so it substitutes
    # `saturation` -- but `saturation` is HSV chroma/max, which is large on a
    # dark colourful pixel that carries almost no chroma, and it cannot fall at
    # all when the frame holds equal amounts of two opposite hues. Both cases
    # make dominant_hue meaningless and only this number says so. 0 = no usable
    # hue; equals mean absolute chroma when every pixel agrees on the hue.
    hue_strength: float = 0.0

    # dominant_hue and hue_strength, per hue qualifier band, in HUE_CENTERS
    # order. Six angles in degrees and the six vector lengths that say how much
    # to believe them -- a band holding no chroma reports strength 0, and the
    # compiler must not bias a rotation on an angle nobody measured. Default
    # empty rather than six zeros: a session written before this existed would
    # otherwise claim every band sits at red with confidence, which is a
    # measurement, not a missing one. Consumers check the length.
    band_hue: list[float] = Field(default_factory=list)
    band_strength: list[float] = Field(default_factory=list)

    # Number of adjacent SAMPLE PAIRS that do not look like the same shot. A
    # lower bound on the cuts in the clip, never a shot list: at n_frames=10
    # the samples are whole seconds apart, so this answers "is this one
    # continuous shot" and nothing finer. Nonzero means one look is about to be
    # applied across a cut. Nothing acts on it yet (roadmap A7 is measurement
    # only); compiler.py currently infers the same thing from frame_variance,
    # which conflates "spans a cut" with "is contrasty".
    cuts: int = 0


def _frame_stats(rgb: np.ndarray) -> dict:
    """Statistics for one uint8 or uint16 RGB frame, in display space.

    NOTE: the caller has already downscaled the frame to _ANALYSIS_WIDTH px, so
    anything sensitive to spatial frequency -- frame_variance especially, and
    the clipped/crushed counts to a lesser degree -- is a PROXY for the full-res
    value, not the absolute truth. Downscaling averages neighbours, which pulls
    a few isolated blown pixels back below the rail and shrinks fine-detail
    variance. Good enough to rank clips and to tell a model "this is flat";
    not good enough to certify legal levels for delivery.
    """
    v = _unit(rgb)
    hi = v.max(axis=1)
    lo = v.min(axis=1)
    chroma = hi - lo

    # Chroma-weighted circular mean hue. Returned as a vector, not an angle:
    # angles do not median across frames (179 and -179 average to 0, the
    # opposite hue), so the wrap-around has to be resolved after the median.
    h = np.radians(_hue_deg(v, hi, chroma))
    hue_vec = np.array([np.sum(chroma * np.cos(h)), np.sum(chroma * np.sin(h))]) / len(v)

    # The same thing again, once per hue qualifier band. A band rotation needs
    # to know where THAT band's pixels actually sit -- "warm the greens" on a
    # shot whose greens are already yellow-green needs less rotation than on one
    # whose greens are pure -- and one global angle cannot say. The six weights
    # are spec._apply_hue_bands' own, so a band measured here is exactly the set
    # of pixels the band setting will move; deriving them separately would let
    # the two drift apart silently.
    #
    # Chroma weighting is not optional: _hue_deg returns 0.0 for a neutral
    # pixel, so an unweighted mean would file every grey in the frame under RED.
    d = ((np.degrees(h)[:, None] - HUE_CENTERS + 180.0) % 360.0) - 180.0
    cw = chroma[:, None] * _smoothstep(np.clip(1.0 - np.abs(d) / HUE_HALFWIDTH, 0.0, 1.0))
    band_vec = np.stack(
        [(cw * np.cos(h)[:, None]).sum(axis=0), (cw * np.sin(h)[:, None]).sum(axis=0)],
        axis=1,
    ) / len(v)  # (6, 2), vectors for the same reason hue_vec is one

    luma = v @ LUMA
    return {
        "mean": v.mean(axis=0),
        "std": v.std(axis=0),
        "saturation": float(np.mean(chroma / np.maximum(hi, _EPS))),
        "p1": np.percentile(v, 1, axis=0),
        "p50": np.percentile(v, 50, axis=0),
        "p99": np.percentile(v, 99, axis=0),
        "clipped_high": float(np.mean(hi >= 1.0 - _RAIL)),
        "crushed_low": float(np.mean(lo <= _RAIL)),
        "hue_vec": hue_vec,
        "band_vec": band_vec,
        "frame_variance": float(luma.var()),
    }


def _unit(rgb: np.ndarray) -> np.ndarray:
    """Integer RGB frame of any shape -> (N, 3) float64 display values in [0, 1]."""
    v = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / _FULL_SCALE[rgb.dtype.itemsize]
    return np.clip(v, 0.0, 1.0, out=v)


def _cut_distances(frames: list[np.ndarray]) -> list[float]:
    """Total variation distance between each adjacent pair's luma histogram.

    Bounded in [0, 1] whatever the frame size, which is what lets _CUT_TV be a
    constant rather than something tuned per clip.
    """
    h = [_luma_hist(f) for f in frames]
    return [float(np.abs(a - b).sum()) / 2.0 for a, b in zip(h, h[1:])]


def _luma_hist(frame: np.ndarray) -> np.ndarray:
    v = _unit(frame)
    return np.histogram(v @ LUMA, bins=_HIST_BINS, range=(0.0, 1.0))[0] / len(v)


def _hue_deg(v: np.ndarray, hi: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """HSV hue in degrees for (N, 3) display-space pixels. 0 where chroma is 0."""
    r, g, b = v[:, 0], v[:, 1], v[:, 2]
    safe = np.where(chroma > _EPS, chroma, 1.0)  # never 0/0
    h = 60.0 * np.where(
        hi == r,
        ((g - b) / safe) % 6.0,
        np.where(hi == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    return np.where(chroma > _EPS, h, 0.0)


def _stats_from_frames(
    frames: list[np.ndarray], width: int, height: int, duration: float
) -> ClipStats:
    per = [_frame_stats(f) for f in frames]
    med = {k: np.median([p[k] for p in per], axis=0) for k in per[0]}
    hx, hy = med["hue_vec"]
    # np.median with axis=0 over a list of (6, 2) is elementwise, and the six
    # arctan2s are taken AFTER it for the same reason dominant_hue's one is.
    bx, by = med["band_vec"][:, 0], med["band_vec"][:, 1]
    return ClipStats(
        mean=RGB(r=med["mean"][0], g=med["mean"][1], b=med["mean"][2]),
        std=RGB(r=med["std"][0], g=med["std"][1], b=med["std"][2]),
        saturation=float(med["saturation"]),
        frames_sampled=len(frames),
        width=width,
        height=height,
        duration=duration,
        p1=RGB(r=med["p1"][0], g=med["p1"][1], b=med["p1"][2]),
        p50=RGB(r=med["p50"][0], g=med["p50"][1], b=med["p50"][2]),
        p99=RGB(r=med["p99"][0], g=med["p99"][1], b=med["p99"][2]),
        clipped_high=float(med["clipped_high"]),
        crushed_low=float(med["crushed_low"]),
        dominant_hue=float(np.degrees(np.arctan2(hy, hx)) % 360.0),
        frame_variance=float(med["frame_variance"]),
        hue_strength=float(np.hypot(hx, hy)),
        band_hue=[float(a) for a in np.degrees(np.arctan2(by, bx)) % 360.0],
        band_strength=[float(a) for a in np.hypot(bx, by)],
        cuts=int(sum(d > _CUT_TV for d in _cut_distances(frames))),
    )


def _run(cmd: list[str], timeout: float = 60.0) -> bytes:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr.decode('utf-8', 'replace')[-500:]}")
    return p.stdout


def _ffprobe(path: str) -> tuple[int, int, float]:
    out = _run([
        ffprobe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", path,
    ])
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    s = streams[0]
    try:
        duration = float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    return int(s["width"]), int(s["height"]), duration


def _grab(path: str, t: float, input_lut: str | None = None) -> np.ndarray | None:
    """Decode one frame at time `t` as a scaled-down uint16 RGB array.

    `input_lut` is applied BEFORE the downscale, not after. A LUT is non-linear,
    so LUT-then-average and average-then-LUT are different numbers, and the
    whole point of measuring here is to report what the grade will actually see.
    Ten full-size LUT applications is a price worth paying for that.

    rgb48le rather than PNG, because PIL silently drops a 16-bit RGB PNG back to
    8 bits on load, and 8 bits is what the lut3d above runs in. The frame size
    PNG used to carry comes back for free: the scale filter pins the width, so
    the height is the only unknown and the buffer length names it.
    """
    raw = _run([
        # -nostdin: without it ffmpeg reads the terminal and eats the user's keystrokes.
        ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
        "-frames:v", "1", "-vf", _analysis_vf(input_lut),
        "-pix_fmt", "rgb48le", "-f", "rawvideo", "-",
    ])
    if len(raw) < _ANALYSIS_WIDTH * 6:  # not even one row
        return None
    return np.frombuffer(raw, dtype="<u2").reshape(-1, _ANALYSIS_WIDTH, 3)


def _analysis_vf(input_lut: str | None) -> str:
    from .render import _lut_filter

    scale = f"scale={_ANALYSIS_WIDTH}:-2"
    return f"{_lut_filter(input_lut)},{scale}" if input_lut else scale


def probe_video(path: str, n_frames: int = 10, input_lut: str | None = None) -> ClipStats:
    """Sample `n_frames` evenly spaced frames and summarize them.

    Pass `input_lut` for log footage: the stats then describe the converted
    image, which is what the grade is applied to. Measuring the raw log instead
    tells the model the clip is flat and grey, and it answers by inventing a
    contrast push -- a guess at a conversion that already exists as a LUT.
    """
    width, height, duration = _ffprobe(path)
    n = max(1, n_frames)
    # Bin starts, not bin centers: ffmpeg's input seek discards frames whose
    # timestamp precedes the seek target, so a center past the final frame's
    # PTS (e.g. 9.5s on a 10s clip whose last frame is at 9.0s) decodes
    # nothing. Bin starts are still evenly spaced and always land on a frame.
    times = [duration * i / n for i in range(n)] if duration > 0 else [0.0]

    frames = [f for f in (_grab(path, t, input_lut) for t in times) if f is not None]
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return _stats_from_frames(frames, width, height, duration)


def probe_image(path: str) -> ClipStats:
    """Same statistics for a single still."""
    img = Image.open(path).convert("RGB")
    return _stats_from_frames([np.asarray(img)], img.width, img.height, 0.0)
