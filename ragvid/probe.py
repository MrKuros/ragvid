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
"""

from __future__ import annotations

import io
import json
import subprocess

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ragvid.platform import ffmpeg, ffprobe
from ragvid.spec import LUMA, RGB

# sRGB piecewise transfer function constants (IEC 61966-2-1).
_A = 0.055
_LIN_CUT = 0.0031308
_SRGB_CUT = 0.04045

# Frames are scaled to this width before analysis; statistics do not need
# full resolution and this keeps a 4K clip as cheap as a 480p one.
_ANALYSIS_WIDTH = 256

_EPS = 1e-8


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
_RAIL = 1.5 / 255.0


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


def _frame_stats(rgb8: np.ndarray) -> dict:
    """Statistics for one uint8 RGB frame, in display space.

    NOTE: the caller has already downscaled the frame to _ANALYSIS_WIDTH px, so
    anything sensitive to spatial frequency -- frame_variance especially, and
    the clipped/crushed counts to a lesser degree -- is a PROXY for the full-res
    value, not the absolute truth. Downscaling averages neighbours, which pulls
    a few isolated blown pixels back below the rail and shrinks fine-detail
    variance. Good enough to rank clips and to tell a model "this is flat";
    not good enough to certify legal levels for delivery.
    """
    v = np.asarray(rgb8, dtype=np.float64).reshape(-1, 3) / 255.0
    hi = v.max(axis=1)
    lo = v.min(axis=1)
    chroma = hi - lo

    # Chroma-weighted circular mean hue. Returned as a vector, not an angle:
    # angles do not median across frames (179 and -179 average to 0, the
    # opposite hue), so the wrap-around has to be resolved after the median.
    h = np.radians(_hue_deg(v, hi, chroma))
    hue_vec = np.array([np.sum(chroma * np.cos(h)), np.sum(chroma * np.sin(h))]) / len(v)

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
        "frame_variance": float(luma.var()),
    }


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
    """Decode one frame at time `t` as a scaled-down uint8 RGB array.

    `input_lut` is applied BEFORE the downscale, not after. A LUT is non-linear,
    so LUT-then-average and average-then-LUT are different numbers, and the
    whole point of measuring here is to report what the grade will actually see.
    Ten full-size LUT applications is a price worth paying for that.
    """
    # ponytail: PNG over the pipe instead of rawvideo so PIL reports the frame
    # size — scale=256:-2 gives a height we would otherwise have to predict.
    png = _run([
        # -nostdin: without it ffmpeg reads the terminal and eats the user's keystrokes.
        ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
        "-frames:v", "1", "-vf", _analysis_vf(input_lut),
        "-f", "image2pipe", "-c:v", "png", "-",
    ])
    if not png:
        return None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


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
