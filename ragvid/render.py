"""ffmpeg wrappers: contact-sheet previews, full renders, hw-encoder probing.

Everything shells out to ffmpeg with an argv list (never a shell string), so the
only escaping that matters is ffmpeg's own filtergraph syntax -- see
`escape_path`.
"""

from __future__ import annotations

from pathlib import Path
import functools
import re
import subprocess
import warnings

from .errors import FFmpegError

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def _run(args: list[str], timeout: float | None = 600) -> str:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-y", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise FFmpegError(proc.returncode, args, proc.stderr)
    return proc.stderr


def _run_with_progress(args: list[str], duration: float, progress) -> str:
    """Same as _run, but stream ffmpeg's progress and report 0.0 -> 1.0.

    `-progress pipe:1` writes machine-readable key=value blocks to stdout, which
    is far more reliable to parse than the human status line on stderr. Without
    a known duration there is no fraction to report, so we fall back to _run.
    """
    if duration <= 0:
        return _run(args)

    proc = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    # ffmpeg's final out_time_us and its progress=end marker both land on 1.0,
    # so report monotonically and only on change -- a consumer should never see
    # the same fraction twice, least of all the terminal one.
    last = -1.0

    def report(fraction: float) -> None:
        nonlocal last
        fraction = max(0.0, min(1.0, fraction))
        if fraction > last:
            last = fraction
            progress(fraction)

    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.strip("-").isdigit():
            report(int(value) / 1e6 / duration)
        elif key == "progress" and value == "end":
            report(1.0)
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise FFmpegError(proc.returncode, args, stderr)
    return stderr


def escape_path(path: str) -> str:
    """Escape a path for use as a filter option value inside a filtergraph.

    Two escaping levels apply, in this order (ffmpeg's "Notes on filtergraph
    escaping"): the filter's own option parser, where ``:`` separates options,
    then the filtergraph parser, where ``, ; [ ]`` separate filters. So the
    filtergraph specials get escaped twice over -- once as themselves, once as
    the backslash the first pass added.
    """
    s = re.sub(r"([\\'])", r"\\\1", str(path))   # level 1: option value
    s = s.replace(":", r"\:")
    return re.sub(r"([\\'\[\],;])", r"\\\1", s)  # level 2: filtergraph


def _lut_filter(cube: str | None) -> str | None:
    # file= explicitly, not the positional form: ffmpeg splits a positional
    # filter arg on its first "=", so a path containing one (dir named "a=b")
    # is parsed as an option name instead of the file -- and is an option
    # injection primitive besides. Named key => the whole rest is the value.
    return None if cube is None else f"lut3d=file={escape_path(cube)}"


def probe_duration(video: str) -> float:
    """Container duration in seconds, 0.0 if ffprobe can't say."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video],
        capture_output=True, text=True,
    )
    try:
        return max(float(proc.stdout.strip()), 0.0)
    except ValueError:
        return 0.0


# ---- hardware encoders ----------------------------------------------------

# (encoder, args before -i, filter chain suffix). Ordered by preference.
_HW_CANDIDATES: list[tuple[str, list[str], str]] = [
    ("h264_nvenc", [], ""),
    ("h264_qsv", [], ""),
    ("h264_vaapi", ["-vaapi_device", "/dev/dri/renderD128"], "format=nv12,hwupload"),
    ("h264_amf", [], ""),
]


@functools.lru_cache(maxsize=1)
def detect_hw_encoder() -> str | None:
    """First hardware H.264 encoder that survives a 1-frame trial encode.

    `ffmpeg -encoders` lists everything compiled in, which says nothing about
    the hardware actually being present, so we encode a real frame instead.
    """
    for enc, pre, vf in _HW_CANDIDATES:
        args = [*pre, "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1"]
        if vf:
            args += ["-vf", vf]
        args += ["-frames:v", "1", "-c:v", enc, "-f", "null", "-"]
        try:
            _run(args, timeout=30)
        except (FFmpegError, subprocess.TimeoutExpired, OSError):
            continue
        return enc
    return None


def _encoder_args(gpu: bool) -> tuple[list[str], str, str]:
    """-> (args before -i, encoder name, filter chain suffix)."""
    if not gpu:
        return [], "libx264", ""
    enc = detect_hw_encoder()
    if enc is None:
        warnings.warn("no usable hardware encoder found; falling back to libx264", stacklevel=3)
        return [], "libx264", ""
    pre, vf = next((p, v) for e, p, v in _HW_CANDIDATES if e == enc)
    return list(pre), enc, vf


# ---- rendering ------------------------------------------------------------

def render_preview(video: str, cube: str | None, out_png: str, n_frames: int = 3) -> str:
    """Contact sheet: n_frames evenly spaced, LUT applied, hstacked into one PNG.

    One ffmpeg process with n seek-before-input inputs, so cost is independent
    of clip length -- this runs on every refine.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    duration = probe_duration(video)

    args: list[str] = []
    for i in range(n_frames):
        args += ["-ss", f"{duration * (i + 0.5) / n_frames:.3f}", "-i", video]

    chain = [f"[{i}:v]" for i in range(n_frames)]
    graph = "".join(chain)
    graph += f"hstack=inputs={n_frames}" if n_frames > 1 else "null"
    if lut := _lut_filter(cube):
        graph += "," + lut

    _run([*args, "-filter_complex", graph, "-frames:v", "1", "-update", "1", out_png], timeout=120)
    return out_png


def render_frame(video: str, cube: str | None, out_png: str, at: float = 0.0) -> str:
    """One frame at `at` seconds, LUT applied. The check-before-you-render call.

    Seeking before -i makes this independent of clip length, so scrubbing a
    two-hour film costs the same as scrubbing a two-second one -- which is what
    lets a UI re-render on every drag of a scrubber.
    """
    args = ["-ss", f"{max(0.0, at):.3f}", "-i", video]
    if lut := _lut_filter(cube):
        args += ["-vf", lut]
    _run([*args, "-frames:v", "1", "-update", "1", out_png], timeout=120)
    return out_png


def _render_gif(video: str, cube: str, out_path: str, progress=None) -> str:
    """Render to an animated GIF.

    GIF is not just "another container": the encoder takes pal8, so the
    -pix_fmt yuv420p pin that the H.264 path needs is an invalid argument here
    and ffmpeg fails in the filter chain. It also has no audio, and a naive
    encode quantizes to a fixed 256-colour palette, which wrecks exactly the
    subtle tonal shifts a grading tool exists to produce.

    So: derive an optimal palette from the *graded* frames (stats_mode=diff
    weights pixels that actually change between frames) and map through it with
    dithering, in one pass over the input.
    """
    lut = _lut_filter(cube)
    fc = (
        f"[0:v]{lut},split[g1][g2];"
        "[g1]palettegen=stats_mode=diff[pal];"
        "[g2][pal]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    args = ["-i", video, "-filter_complex", fc, "-loop", "0", "-an", out_path]
    if progress:
        _run_with_progress(args, probe_duration(video), progress)
    else:
        _run(args)
    return out_path


def render_video(video: str, cube: str, out_path: str, gpu: bool = False,
                 progress=None) -> str:
    """Full render. Audio is stream-copied, never re-encoded.

    `progress` is an optional callable taking a 0.0-1.0 float, called as ffmpeg
    works. This is the only operation slow enough for a UI to need a bar.
    """
    if Path(out_path).suffix.lower() == ".gif":
        return _render_gif(video, cube, out_path, progress=progress)
    pre, enc, vf_suffix = _encoder_args(gpu)
    vf = _lut_filter(cube)
    # H.264 at 4:2:0 needs even dimensions, and plenty of real sources are odd
    # (a 720x405 GIF, anything cropped by hand) -- libx264 refuses outright with
    # "height not divisible by 2" and the export fails at the very last step.
    # Crop rather than scale: losing at most one row/column beats resampling
    # every frame. A no-op when the dimensions are already even, so it costs
    # nothing to apply unconditionally and saves probing for the size.
    vf = f"{vf},crop=trunc(iw/2)*2:trunc(ih/2)*2"
    if vf_suffix:
        vf = f"{vf},{vf_suffix}"
    args = [*pre, "-i", video, "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf, "-c:v", enc]
    if not vf_suffix:
        # lut3d negotiates yuv444p, which libx264 happily encodes into a file most
        # players and hardware decoders reject. Pin 4:2:0. The hw paths already
        # upload their own format (nv12) so they must not get -pix_fmt.
        args += ["-pix_fmt", "yuv420p"]
    full = [*args, "-c:a", "copy", out_path]
    if progress:
        _run_with_progress(full, probe_duration(video), progress)
    else:
        _run(full)
    return out_path
