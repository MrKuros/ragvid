"""Where Linux, macOS and Windows actually differ. One module, four questions.

Only four things in ragvid are platform-shaped: where per-user data lives, which
hardware H.264 encoders can plausibly exist, how to find ffmpeg when it is not on
PATH, and whether a filtergraph path needs its separators flipped. Everything
else is pathlib and is portable already.

Every function reads `sys.platform` and the environment *at call time*. That is
deliberate: it is what makes the other two platforms testable from a single host
— monkeypatch `sys.platform` and call again. (`sys.platform` rather than
`os.name`: pathlib picks its flavour from `os.name` at instantiation, so a test
that patched it would start handing WindowsPath objects to a Linux filesystem.)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .errors import FFmpegNotFound


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


# ---- data directory -------------------------------------------------------


def data_dir() -> Path:
    """Per-user application data root, native to the host.

    XDG on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on
    Windows. No dependency: the branch is three lines and the conventions do not
    move.
    """
    if is_windows():
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    elif is_macos():
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "ragvid"


# ---- hardware encoders ----------------------------------------------------

# (encoder, args before -i, filter chain suffix)
_NVENC = ("h264_nvenc", [], "")
_QSV = ("h264_qsv", [], "")
_AMF = ("h264_amf", [], "")
_VAAPI = ("h264_vaapi", ["-vaapi_device", "/dev/dri/renderD128"], "format=nv12,hwupload")
_VIDEOTOOLBOX = ("h264_videotoolbox", [], "")


def hw_encoders() -> list[tuple[str, list[str], str]]:
    """H.264 hardware encoders that could exist here, in preference order.

    VAAPI is a Linux kernel interface and `/dev/dri/renderD128` is a Linux device
    node, so probing it elsewhere costs a subprocess and can never succeed.
    macOS has exactly one answer, VideoToolbox, and without this list it had
    none at all. Windows keeps the three vendor encoders.

    The trial encode in render.detect_hw_encoder still decides correctness; this
    only stops us asking impossible questions.
    """
    if is_macos():
        return [_VIDEOTOOLBOX]
    if is_windows():
        return [_NVENC, _QSV, _AMF]
    return [_NVENC, _QSV, _VAAPI, _AMF]


# ---- ffmpeg / ffprobe discovery -------------------------------------------

# A macOS app launched from Finder inherits a PATH that has never seen a shell
# profile, so Homebrew's prefixes are invisible to it. Both are checked because
# /opt/homebrew is Apple silicon and /usr/local is Intel.
_MAC_EXTRA = ("/opt/homebrew/bin", "/usr/local/bin")


def _install_hint() -> str:
    if is_macos():
        return "install it with 'brew install ffmpeg'"
    if is_windows():
        return "install it with 'winget install Gyan.FFmpeg' or 'choco install ffmpeg'"
    return "install it with your package manager (pacman -S ffmpeg, apt install ffmpeg, ...)"


def find_binary(name: str) -> str:
    """Full path to `ffmpeg` or `ffprobe`, or FFmpegNotFound naming the fix.

    `RAGVID_FFMPEG` / `RAGVID_FFPROBE` win outright — the escape hatch for a
    static build, a sandboxed app bundle, or two ffmpegs on one machine.

    ponytail: not cached. shutil.which is a handful of stats beside a render that
    takes hundreds of milliseconds, and a cache would freeze the env override for
    the life of the process.
    """
    env_var = f"RAGVID_{name.upper()}"
    override = os.environ.get(env_var)
    if override:
        # which() resolves a bare name, a relative path or an absolute one, and
        # checks the executable bit -- so a typo'd override fails here, loudly,
        # rather than as a confusing ffmpeg error later.
        found = shutil.which(override)
        if found:
            return found
        raise FFmpegNotFound(name, env_var, f"{env_var} is set to {override!r}, which is not executable")

    # shutil.which applies PATHEXT on Windows, so plain "ffmpeg" finds ffmpeg.exe.
    found = shutil.which(name)
    if not found and is_macos():
        found = shutil.which(name, path=os.pathsep.join(_MAC_EXTRA))
    if found:
        return found
    raise FFmpegNotFound(name, env_var, _install_hint())


def ffmpeg() -> str:
    return find_binary("ffmpeg")


def ffprobe() -> str:
    return find_binary("ffprobe")
