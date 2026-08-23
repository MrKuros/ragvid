"""Portability: macOS and Windows behaviour, pinned from a Linux host.

Every one of these fixes is for a platform we cannot boot, so the whole file
works the same way: monkeypatch `sys.platform` and assert what ragvid then
decides. That works because ragvid.platform reads it at call time and caches
nothing.

`sys.platform`, not `os.name`: pathlib chooses PosixPath vs WindowsPath from
`os.name` every time a Path is constructed, so patching that one would have the
test suite building WindowsPath objects on a Linux filesystem -- including
inside pytest itself. sys.platform is read by the code under test and by
shutil.which (for PATHEXT), which is exactly the coverage we want.
"""

from __future__ import annotations

import errno
import socket
import stat
import sys
from pathlib import Path

import pytest

from ragvid import render, server
from ragvid import platform as plat
from ragvid.errors import FFmpegNotFound

WINDOWS, MACOS, LINUX = "win32", "darwin", "linux"


@pytest.fixture
def as_host(monkeypatch):
    """`as_host(WINDOWS)` -> the rest of the test runs as if on that platform."""
    return lambda host: monkeypatch.setattr(sys, "platform", host)


def test_the_host_probes_agree_with_the_simulated_host(as_host):
    as_host(WINDOWS)
    assert (plat.is_windows(), plat.is_macos()) == (True, False)
    as_host(MACOS)
    assert (plat.is_windows(), plat.is_macos()) == (False, True)
    as_host(LINUX)
    assert (plat.is_windows(), plat.is_macos()) == (False, False)


# ---- 1. filtergraph paths -------------------------------------------------
# A Windows LUT path carries both filtergraph metacharacters at once: the drive
# colon separates filter *options*, and the separators are backslashes, which
# are the escape character itself. Escaping the backslashes is grammatically
# correct but yields eight-deep runs; forward slashes -- which Win32 accepts
# everywhere -- reduce the whole thing to one escaped colon.


def test_windows_lut_path_becomes_forward_slashes_with_an_escaped_drive_colon(as_host):
    as_host(WINDOWS)
    win = r"C:\Users\x\AppData\Roaming\ragvid\work\.ragvid\current.cube"
    assert render.escape_path(win) == r"C\\:/Users/x/AppData/Roaming/ragvid/work/.ragvid/current.cube"
    # And that is what actually reaches ffmpeg.
    assert render._lut_filter(win) == (
        r"lut3d=file=C\\:/Users/x/AppData/Roaming/ragvid/work/.ragvid/current.cube"
    )


def test_windows_paths_with_spaces_and_unc_shares_survive(as_host):
    as_host(WINDOWS)
    assert render.escape_path(r"C:\Users\Jo Smith\g.cube") == r"C\\:/Users/Jo Smith/g.cube"
    # UNC: no drive letter, so nothing to escape at all once flipped.
    assert render.escape_path(r"\\nas\media\g.cube") == "//nas/media/g.cube"


def test_macos_paths_pass_through_untouched(as_host):
    as_host(MACOS)
    mac = "/Users/x/Library/Application Support/ragvid/work/.ragvid/current.cube"
    assert render.escape_path(mac) == mac  # a space is not a filtergraph special
    assert render.escape_path("/Users/x/My Movies/a,b.cube") == r"/Users/x/My Movies/a\,b.cube"


def test_a_posix_backslash_is_escaped_not_rewritten(as_host):
    r"""On Linux and macOS `\` is an ordinary filename character.

    Flipping it to `/` would point ffmpeg at a directory that does not exist, so
    the rewrite has to be gated on the host and never on the shape of the string.
    tests/test_robustness.py grades a real file called `back\slash 'q' [b],s;.cube`
    through ffmpeg, which is the end-to-end half of this assertion.
    """
    for host in (LINUX, MACOS):
        as_host(host)
        assert render.escape_path("/tmp/back\\slash.cube") == r"/tmp/back\\\\slash.cube"
        assert render.escape_path("/tmp/a:b.cube") == r"/tmp/a\\:b.cube"


def test_the_escaped_form_round_trips_through_a_real_ffmpeg(tmp_path):
    """The Windows shape cannot be created here, but its grammar can be checked.

    `C\\:/x` is a colon escaped for the option parser and then for the
    filtergraph parser. Build a file whose *name* needs exactly that treatment
    and confirm ffmpeg opens it -- if the double escape were wrong, ffmpeg would
    read the tail as another filter option and fail.
    """
    from tests.test_render import identity_cube  # real ffmpeg, real parser

    cube = identity_cube(tmp_path / "drive:c.cube")
    assert render.escape_path(cube).endswith(r"drive\\:c.cube")
    render.render_preview("assets/sample.mp4", cube, str(tmp_path / "o.png"), n_frames=1)
    assert (tmp_path / "o.png").stat().st_size > 0


# ---- 2. binary discovery --------------------------------------------------


def _fake_exe(path: Path) -> Path:
    """A do-nothing executable, on whatever OS is running the suite.

    Windows resolves executables through PATHEXT and has no shebang, so an
    extensionless /bin/sh script is not an executable there at all -- shutil.which
    returns None and the override tests fail for a reason that has nothing to do
    with the code under test.
    """
    if plat.is_windows():
        path = path.with_suffix(".cmd")
        path.write_text("@exit /b 0\r\n")
    else:
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_env_override_wins_over_path(tmp_path, monkeypatch):
    mine = _fake_exe(tmp_path / "my-ffmpeg")
    monkeypatch.setenv("RAGVID_FFMPEG", str(mine))
    assert plat.ffmpeg() == str(mine)


def test_a_bogus_override_fails_immediately_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGVID_FFPROBE", str(tmp_path / "not-here"))
    with pytest.raises(FFmpegNotFound) as e:
        plat.ffprobe()
    assert e.value.binary == "ffprobe" and e.value.env_var == "RAGVID_FFPROBE"
    assert "not executable" in str(e.value)


@pytest.mark.parametrize("host,fragment", [
    (MACOS, "brew install ffmpeg"),
    (WINDOWS, "winget"),
    (LINUX, "package manager"),
])
def test_missing_ffmpeg_is_typed_and_names_the_install_command(host, fragment, as_host, monkeypatch):
    """Not a bare FileNotFoundError from subprocess: a UI has to be able to
    branch on it, and the user has to be told what to install."""
    as_host(host)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("RAGVID_FFMPEG", raising=False)
    monkeypatch.setattr(plat, "_MAC_EXTRA", ())
    with pytest.raises(FFmpegNotFound) as e:
        plat.ffmpeg()
    assert e.value.binary == "ffmpeg"
    assert fragment in str(e.value)
    assert "RAGVID_FFMPEG" in str(e.value)


def test_macos_falls_back_to_the_homebrew_prefix(tmp_path, as_host, monkeypatch):
    """A macOS app launched from Finder has a PATH that never saw a shell
    profile, so /opt/homebrew/bin is invisible and ffmpeg 'is not installed'."""
    _fake_exe(tmp_path / "ffmpeg")
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("RAGVID_FFMPEG", raising=False)
    monkeypatch.setattr(plat, "_MAC_EXTRA", (str(tmp_path),))
    as_host(MACOS)
    assert plat.ffmpeg() == str(tmp_path / "ffmpeg")
    as_host(LINUX)  # ...and only on macOS
    with pytest.raises(FFmpegNotFound):
        plat.ffmpeg()


def test_discovery_asks_for_the_bare_name_so_windows_pathext_applies(monkeypatch):
    """On Windows the binary is ffmpeg.exe. shutil.which appends every PATHEXT
    entry itself, so find_binary must hand it the bare name and never a
    hand-built \'ffmpeg.exe\' that would then be wrong everywhere else.

    (Asserted by delegation rather than by faking sys.platform inside
    shutil.which: that branch calls into _winapi, which does not exist here.)
    """
    import shutil

    asked: list[str] = []
    monkeypatch.delenv("RAGVID_FFMPEG", raising=False)
    monkeypatch.setattr(shutil, "which", lambda cmd, **kw: asked.append(cmd) or "/usr/bin/ffmpeg")
    assert plat.ffmpeg() == "/usr/bin/ffmpeg"
    assert asked == ["ffmpeg"]


def test_render_and_probe_both_go_through_discovery(tmp_path, monkeypatch):
    """Two modules shell out to ffmpeg. Both must honour the override, or a Mac
    that needs it works for previews and fails on the very first probe."""
    from ragvid import probe

    calls: list[str] = []
    noop = str(_fake_exe(tmp_path / "noop"))  # /bin/true does not exist on Windows
    monkeypatch.setattr(plat, "find_binary", lambda name: calls.append(name) or noop)
    render.probe_duration("x.mp4")
    with pytest.raises(Exception):
        probe._ffprobe("x.mp4")  # it prints nothing, so this is not JSON
    assert calls == ["ffprobe", "ffprobe"]


# ---- 3. hardware encoders -------------------------------------------------


def test_vaapi_is_offered_on_linux_only(as_host):
    as_host(LINUX)
    linux = dict((e, pre) for e, pre, _ in plat.hw_encoders())
    assert linux["h264_vaapi"] == ["-vaapi_device", "/dev/dri/renderD128"]
    for host in (MACOS, WINDOWS):
        as_host(host)
        assert "h264_vaapi" not in {e for e, _, _ in plat.hw_encoders()}


def test_macos_gets_videotoolbox_and_nothing_else(as_host):
    as_host(MACOS)
    assert [e for e, _, _ in plat.hw_encoders()] == ["h264_videotoolbox"]
    for host in (LINUX, WINDOWS):
        as_host(host)
        assert "h264_videotoolbox" not in {e for e, _, _ in plat.hw_encoders()}


def test_windows_offers_the_three_vendor_encoders(as_host):
    as_host(WINDOWS)
    assert [e for e, _, _ in plat.hw_encoders()] == ["h264_nvenc", "h264_qsv", "h264_amf"]


@pytest.mark.parametrize("host,expected", [
    (MACOS, ["h264_videotoolbox"]),
    (WINDOWS, ["h264_nvenc", "h264_qsv", "h264_amf"]),
    (LINUX, ["h264_nvenc", "h264_qsv", "h264_vaapi", "h264_amf"]),
])
def test_the_trial_encode_only_probes_possible_encoders(host, expected, as_host, monkeypatch):
    """The probe is one ffmpeg subprocess per candidate. Asking macOS about
    /dev/dri is a guaranteed-failing process launch on the export path."""
    tried: list[str] = []

    def fake_run(args, timeout=None):
        tried.append(args[args.index("-c:v") + 1])
        assert host == LINUX or "/dev/dri/renderD128" not in args
        raise render.FFmpegError(1, args, "no such device")

    as_host(host)
    monkeypatch.setattr(render, "_run", fake_run)
    render.detect_hw_encoder.cache_clear()
    try:
        assert render.detect_hw_encoder() is None
        assert tried == expected
    finally:
        render.detect_hw_encoder.cache_clear()


def test_an_encoder_outside_the_current_list_does_not_raise(as_host, monkeypatch):
    """detect_hw_encoder is cached for the whole process while the candidate
    list is recomputed per call, so the two can disagree. _encoder_args looked
    the encoder up with a bare next() -- StopIteration, mid-export."""
    as_host(MACOS)
    monkeypatch.setattr(render, "detect_hw_encoder", lambda: "h264_vaapi")
    pre, enc, vf = render._encoder_args(gpu=True)
    assert (pre, enc, vf) == ([], "h264_vaapi", "")


# ---- 4. data directory ----------------------------------------------------


def test_linux_uses_xdg(as_host, monkeypatch, tmp_path):
    as_host(LINUX)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    assert plat.data_dir() == tmp_path / "share" / "ragvid"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert plat.data_dir() == Path.home() / ".local" / "share" / "ragvid"


def test_macos_uses_application_support_not_xdg(as_host, monkeypatch, tmp_path):
    as_host(MACOS)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))  # must be ignored
    assert plat.data_dir() == Path.home() / "Library" / "Application Support" / "ragvid"


def test_windows_uses_appdata(as_host, monkeypatch, tmp_path):
    as_host(WINDOWS)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))  # must be ignored
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert plat.data_dir() == tmp_path / "Roaming" / "ragvid"
    monkeypatch.delenv("APPDATA")
    assert plat.data_dir() == Path.home() / "AppData" / "Roaming" / "ragvid"


@pytest.mark.parametrize("host", [LINUX, MACOS, WINDOWS])
def test_the_server_work_dir_follows_the_platform(host, as_host, monkeypatch, tmp_path):
    """Uploads land here and it is the default project root, so a wrong answer
    scatters files somewhere the OS will not back up or clean."""
    as_host(host)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert server._work_dir() == plat.data_dir() / "work"
    assert server._work_dir().parent.name == "ragvid"


# ---- 5. server binding ----------------------------------------------------


@pytest.mark.parametrize("host,reuse", [(LINUX, True), (MACOS, True), (WINDOWS, False)])
def test_so_reuseaddr_is_off_on_windows(host, reuse, as_host):
    """POSIX SO_REUSEADDR waives TIME_WAIT. Windows SO_REUSEADDR lets bind()
    succeed on a port another process is actively listening on, which would make
    the port walk below hand back a port we are only half-holding."""
    as_host(host)
    httpd, port = server._bind(0)
    try:
        assert httpd.allow_reuse_address is reuse
    finally:
        httpd.server_close()


def test_bind_walks_past_a_port_in_use():
    """A second `ragvid serve` must move to the next port, not share one."""
    first, _ = server._bind(0)  # 0 -> an ephemeral port, guaranteed free
    port = first.server_address[1]
    try:
        second, next_port = server._bind(port)
        second.server_close()
        assert next_port > port
    finally:
        first.server_close()


def test_bind_reraises_errors_that_are_not_a_busy_port(monkeypatch):
    def refuse(*a, **kw):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(server, "_Server", refuse)
    with pytest.raises(OSError) as e:
        server._bind(1)
    assert e.value.errno == errno.EACCES


def test_bind_treats_the_raw_winsock_code_as_a_busy_port(monkeypatch):
    """CPython usually maps WSAEADDRINUSE to EADDRINUSE, but exposes the raw
    10048 as .winerror; a build that leaves errno unmapped must not turn a busy
    port into a hard failure."""
    calls: list[int] = []
    real = server._Server

    def flaky(addr, handler):
        calls.append(addr[1])
        if len(calls) == 1:
            exc = OSError(0, "address in use")
            exc.winerror = 10048
            raise exc
        return real(addr, handler)

    probe_srv, _ = server._bind(0)
    base = probe_srv.server_address[1]  # an ephemeral port, then handed back
    probe_srv.server_close()
    monkeypatch.setattr(server, "_Server", flaky)
    httpd, port = server._bind(base)
    httpd.server_close()
    assert calls[0] == base and port > base


def test_the_server_never_leaves_loopback(as_host):
    """Same on every platform: the API opens local files by absolute path."""
    for host in (LINUX, MACOS, WINDOWS):
        as_host(host)
        httpd, _ = server._bind(0)
        try:
            assert httpd.server_address[0] == "127.0.0.1"
            assert httpd.socket.family == socket.AF_INET
        finally:
            httpd.server_close()


# ---- 6. .cube newlines ----------------------------------------------------


def test_baked_cubes_are_lf_on_every_platform(tmp_path):
    """Text mode on Windows would translate every \\n to \\r\\n. ffmpeg accepts
    either (see below), but the same grade should be the same bytes anywhere."""
    from ragvid.lut import bake_cube
    from ragvid.spec import GradeSpec

    out = Path(bake_cube(GradeSpec(), str(tmp_path / "g.cube"), size=4))
    assert b"\r" not in out.read_bytes()


def test_a_crlf_cube_is_still_readable_and_still_grades(tmp_path):
    """Cubes written by Windows tools -- or by an older ragvid on Windows -- are
    CRLF. Both our parser and ffmpeg's have to cope."""
    from ragvid.lut import bake_cube, read_cube
    from ragvid.spec import GradeSpec

    lf = Path(bake_cube(GradeSpec(saturation=0.0), str(tmp_path / "lf.cube"), size=8))
    crlf = tmp_path / "crlf.cube"
    crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))

    assert read_cube(str(crlf))[0] == 8
    out = tmp_path / "o.png"
    render.render_preview("assets/sample.mp4", str(crlf), str(out), n_frames=1)

    import numpy as np
    from PIL import Image

    px = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    assert np.abs(px[..., 0] - px[..., 1]).mean() < 1.0  # greyscale => the LUT applied
