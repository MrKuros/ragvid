"""ffmpeg wrappers: contact-sheet previews, full renders, hw-encoder probing.

Everything shells out to ffmpeg with an argv list (never a shell string), so the
only escaping that matters is ffmpeg's own filtergraph syntax -- see
`escape_path`.

Every render entry point takes an optional `effects` (a spec.EffectSpec). That
is the half of a look a 3D LUT cannot carry, because it reads neighbouring
pixels -- so it is built as ffmpeg filters *around* the lut3d node rather than
baked into the cube. Preview, frame and export all take it, and must: a preview
that drops the effects shows a look the export will not produce.
"""

from __future__ import annotations

from pathlib import Path
import functools
import math
import re
import subprocess
import warnings

from .errors import FFmpegError
from .platform import ffmpeg, ffprobe, hw_encoders, is_windows


def _run(args: list[str], timeout: float | None = 600) -> str:
    proc = subprocess.run(
        [ffmpeg(), "-hide_banner", "-nostdin", "-y", *args],
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
        [ffmpeg(), "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *args],
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

    On Windows the separators are flipped to ``/`` first. Win32 accepts forward
    slashes in every file API, and it keeps a drive-letter path down to one
    escaped colon (``C\\:/Users/...``, the form ffmpeg's own docs use) instead of
    a run of eight backslashes per separator that then has to survive
    subprocess.list2cmdline as well. Gated on the *host*, never on the shape of
    the string: ``\\`` is an ordinary character in a POSIX filename and
    rewriting it there would point ffmpeg at a file that does not exist.
    """
    raw = str(path).replace("\\", "/") if is_windows() else str(path)
    s = re.sub(r"([\\'])", r"\\\1", raw)         # level 1: option value
    s = s.replace(":", r"\:")
    return re.sub(r"([\\'\[\],;])", r"\\\1", s)  # level 2: filtergraph


def _lut_filter(cube: str | None) -> str | None:
    # file= explicitly, not the positional form: ffmpeg splits a positional
    # filter arg on its first "=", so a path containing one (dir named "a=b")
    # is parsed as an option name instead of the file -- and is an option
    # injection primitive besides. Named key => the whole rest is the value.
    return None if cube is None else f"lut3d=file={escape_path(cube)}"


def _effect_filters(effects) -> tuple[list[str], list[str]]:
    """An EffectSpec as ffmpeg filter fragments: (before lut3d, after lut3d).

    These are the parts of a look a 3D LUT cannot hold, because they read
    neighbouring pixels. Denoise is the only one that belongs *before* the LUT
    -- grading a clean plate stops the grade from amplifying sensor noise. The
    rest act on the graded image, ordered the way the physical stack is:
    glow (lens flare) -> softness (focus) -> fringe (lens dispersion) ->
    grain (film stock) -> vignette (lens falloff). Grain sits after softness so
    a blur cannot wash it back out again.

    Each fragment is chainable at both ends -- glow's split/blend is written as
    ``split[..];[..]..[..];[..][..]blend`` so that joining fragments with "," and
    splicing the result into a larger graph stays grammatical.
    """
    if effects is None:
        return [], []
    pre: list[str] = []
    post: list[str] = []

    if (v := effects.denoise) > 0:
        # Scaled so denoise~=0.35 lands on hqdn3d's own defaults (4:3:6:4.5) and
        # 1.0 is as far as it goes before detail starts to smear. Measured on a
        # noisy plate: the defaults only move RMSE-vs-clean 9.56 -> 9.54, i.e.
        # they are near-invisible, so a linear scale off them would make every
        # setting below "1.0" do nothing at all.
        pre.append(f"hqdn3d={12 * v:.3f}:{9 * v:.3f}:{9 * v:.3f}:{7 * v:.3f}")

    if (v := effects.glow) > 0:
        # Isolate highlights (curve crushes everything under 0.6), blur them,
        # screen the blur back over the untouched image. all_opacity meters it,
        # so glow=0.1 is a sheen and glow=1.0 is a halation bloom.
        post.append(
            "split[rvg0][rvg1];"
            f"[rvg1]curves=all=0/0 0.6/0 1/1,gblur=sigma={2 + 18 * v:.3f}[rvg2];"
            f"[rvg0][rvg2]blend=all_mode=screen:all_opacity={min(v, 1.0):.3f}"
        )

    if (v := effects.softness) > 0:
        post.append(f"gblur=sigma={6 * v:.3f}")
    elif v < 0:
        # cas is a contrast-adaptive sharpen: it does not ring on edges the way
        # unsharp does, and its strength is already normalised to 0..1.
        post.append(f"cas=strength={min(-v, 1.0):.3f}")

    if (v := effects.fringe) != 0:
        # rgbashift shifts by whole pixels, so anything non-zero is at least 1 --
        # rounding a small value to 0 would silently drop the effect.
        n = max(1, round(abs(v) * 8)) * (1 if v > 0 else -1)
        post.append(f"rgbashift=rh={n}:bh={-n}")

    if (v := effects.grain) > 0:
        # allf=t+u: temporal (a new pattern every frame, or it reads as dirt on
        # the lens) and uniform (gaussian noise clumps and looks like blocking).
        post.append(f"noise=alls={max(1, round(v * 40))}:allf=t+u")

    if (v := effects.vignette) != 0:
        # mode=backward inverts the falloff, so a negative value brightens the
        # corners instead of darkening them.
        post.append(f"vignette=a={abs(v) * math.pi / 4:.4f}"
                    + ("" if v > 0 else ":mode=backward"))

    return pre, post


def _vf(effects, lut: str | None, *extra: str) -> str:
    """Compose one filter chain: effects around `lut`, then `extra` at the end.

    Built *around* _lut_filter's bare node rather than inside it, the same way
    the odd-dimension crop and the hw-encoder suffix already are.
    """
    pre, post = _effect_filters(effects)
    parts = [*pre, *([lut] if lut else []), *post, *(e for e in extra if e)]
    return ",".join(parts)


def probe_duration(video: str) -> float:
    """Container duration in seconds, 0.0 if ffprobe can't say."""
    proc = subprocess.run(
        [ffprobe(), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video],
        capture_output=True, text=True,
    )
    try:
        return max(float(proc.stdout.strip()), 0.0)
    except ValueError:
        return 0.0


# ---- hardware encoders ----------------------------------------------------

@functools.lru_cache(maxsize=1)
def detect_hw_encoder() -> str | None:
    """First hardware H.264 encoder that survives a 1-frame trial encode.

    `ffmpeg -encoders` lists everything compiled in, which says nothing about
    the hardware actually being present, so we encode a real frame instead. The
    candidate list is platform-scoped (see platform.hw_encoders) so we never pay
    for a trial encode that could not possibly work here.
    """
    for enc, pre, vf in hw_encoders():
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
    pre, vf = next(((p, v) for e, p, v in hw_encoders() if e == enc), ([], ""))
    return list(pre), enc, vf


# ---- rendering ------------------------------------------------------------

def render_preview(video: str, cube: str | None, out_png: str, effects=None,
                   n_frames: int = 3) -> str:
    """Contact sheet: n_frames evenly spaced, LUT + effects, hstacked into one PNG.

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
    if vf := _vf(effects, _lut_filter(cube)):
        graph += "," + vf

    _run([*args, "-filter_complex", graph, "-frames:v", "1", "-update", "1", out_png], timeout=120)
    return out_png


def render_frame(video: str, cube: str | None, out_png: str, effects=None,
                 at: float = 0.0) -> str:
    """One frame at `at` seconds, LUT + effects. The check-before-you-render call.

    Seeking before -i makes this independent of clip length, so scrubbing a
    two-hour film costs the same as scrubbing a two-second one -- which is what
    lets a UI re-render on every drag of a scrubber.
    """
    args = ["-ss", f"{max(0.0, at):.3f}", "-i", video]
    if vf := _vf(effects, _lut_filter(cube)):
        args += ["-vf", vf]
    _run([*args, "-frames:v", "1", "-update", "1", out_png], timeout=120)
    return out_png


def _render_gif(video: str, cube: str, out_path: str, effects=None, progress=None) -> str:
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
    fc = (
        f"[0:v]{_vf(effects, _lut_filter(cube))},split[g1][g2];"
        "[g1]palettegen=stats_mode=diff[pal];"
        "[g2][pal]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    args = ["-i", video, "-filter_complex", fc, "-loop", "0", "-an", out_path]
    if progress:
        _run_with_progress(args, probe_duration(video), progress)
    else:
        _run(args)
    return out_path


def render_video(video: str, cube: str, out_path: str, effects=None,
                 gpu: bool = False, progress=None) -> str:
    """Full render. Audio is stream-copied, never re-encoded.

    `progress` is an optional callable taking a 0.0-1.0 float, called as ffmpeg
    works. This is the only operation slow enough for a UI to need a bar.
    """
    if Path(out_path).suffix.lower() == ".gif":
        return _render_gif(video, cube, out_path, effects, progress=progress)
    pre, enc, vf_suffix = _encoder_args(gpu)
    # H.264 at 4:2:0 needs even dimensions, and plenty of real sources are odd
    # (a 720x405 GIF, anything cropped by hand) -- libx264 refuses outright with
    # "height not divisible by 2" and the export fails at the very last step.
    # Crop rather than scale: losing at most one row/column beats resampling
    # every frame. A no-op when the dimensions are already even, so it costs
    # nothing to apply unconditionally and saves probing for the size.
    vf = _vf(effects, _lut_filter(cube), "crop=trunc(iw/2)*2:trunc(ih/2)*2", vf_suffix)
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


if __name__ == "__main__":  # pragma: no cover
    # ponytail: self-check instead of a test file -- tests/ belongs to another
    # agent this build. Run with `python -m ragvid.render`.
    from .spec import EffectSpec

    bare = _lut_filter("/tmp/x.cube")
    assert _vf(None, bare) == bare, "no effects must leave the lut3d node alone"
    assert _vf(EffectSpec(), bare) == bare, "a default EffectSpec must be a no-op"

    # Every fragment must be grammatical *spliced into a real graph*, which is
    # the part string equality cannot tell you -- glow's split/blend is three
    # chains pretending to be one filter.
    for name in EffectSpec.model_fields:
        for value in (0.6, -0.6):
            chain = _vf(EffectSpec(**{name: value}), bare,
                        "crop=trunc(iw/2)*2:trunc(ih/2)*2")
            chain = chain.replace(bare, "null")  # no cube on disk here
            _run(["-f", "lavfi", "-i", "testsrc2=s=64x64:d=0.2", "-vf", chain,
                  "-frames:v", "2", "-f", "null", "-"], timeout=60)
            print(f"ok  {name}={value:+.1f}  {chain}")
    print("all effect fragments parse")
