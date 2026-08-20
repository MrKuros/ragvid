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
from pydantic import BaseModel

from ragvid.platform import ffmpeg, ffprobe
from ragvid.spec import RGB

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


class ClipStats(BaseModel):
    mean: RGB  # per-channel mean, sRGB display space, 0-1
    std: RGB  # per-channel std, sRGB display space
    saturation: float  # mean chroma 0-1
    frames_sampled: int
    width: int
    height: int
    duration: float


def _frame_stats(rgb8: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(mean, std, saturation) for one uint8 RGB frame, in display space."""
    v = np.asarray(rgb8, dtype=np.float64).reshape(-1, 3) / 255.0
    hi = v.max(axis=1)
    lo = v.min(axis=1)
    sat = float(np.mean((hi - lo) / np.maximum(hi, _EPS)))
    return v.mean(axis=0), v.std(axis=0), sat


def _stats_from_frames(
    frames: list[np.ndarray], width: int, height: int, duration: float
) -> ClipStats:
    per = [_frame_stats(f) for f in frames]
    means = np.median([p[0] for p in per], axis=0)
    stds = np.median([p[1] for p in per], axis=0)
    sat = float(np.median([p[2] for p in per]))
    return ClipStats(
        mean=RGB(r=means[0], g=means[1], b=means[2]),
        std=RGB(r=stds[0], g=stds[1], b=stds[2]),
        saturation=sat,
        frames_sampled=len(frames),
        width=width,
        height=height,
        duration=duration,
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


def _grab(path: str, t: float) -> np.ndarray | None:
    """Decode one frame at time `t` as a scaled-down uint8 RGB array."""
    # ponytail: PNG over the pipe instead of rawvideo so PIL reports the frame
    # size — scale=256:-2 gives a height we would otherwise have to predict.
    png = _run([
        # -nostdin: without it ffmpeg reads the terminal and eats the user's keystrokes.
        ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
        "-frames:v", "1", "-vf", f"scale={_ANALYSIS_WIDTH}:-2",
        "-f", "image2pipe", "-c:v", "png", "-",
    ])
    if not png:
        return None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def probe_video(path: str, n_frames: int = 10) -> ClipStats:
    """Sample `n_frames` evenly spaced frames and summarize them."""
    width, height, duration = _ffprobe(path)
    n = max(1, n_frames)
    # Bin starts, not bin centers: ffmpeg's input seek discards frames whose
    # timestamp precedes the seek target, so a center past the final frame's
    # PTS (e.g. 9.5s on a 10s clip whose last frame is at 9.0s) decodes
    # nothing. Bin starts are still evenly spaced and always land on a frame.
    times = [duration * i / n for i in range(n)] if duration > 0 else [0.0]

    frames = [f for f in (_grab(path, t) for t in times) if f is not None]
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return _stats_from_frames(frames, width, height, duration)


def probe_image(path: str) -> ClipStats:
    """Same statistics for a single still."""
    img = Image.open(path).convert("RGB")
    return _stats_from_frames([np.asarray(img)], img.width, img.height, 0.0)
