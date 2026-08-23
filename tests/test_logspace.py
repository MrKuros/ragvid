"""Log transfer functions, the conversion .cube, and camera detection.

No provider is involved anywhere in this module -- logspace.py is pure numpy
plus two ffprobe/ffmpeg calls, which is the point of shipping it as a library
before anything wires it into a session.
"""

import numpy as np
import pytest
from PIL import Image

from ragvid import logspace, render
from ragvid.lut import read_cube

# What the vendors publish for 18% scene grey, and what 100% diffuse white
# lands on. The second column is the fact that makes ungraded log look washed
# out: white is nowhere near 1.0.
MID_GREY = {
    "slog3": (0.4105571848, 0.5960273437),
    "vlog": (0.4233114488, 0.5991177002),
    "clog3": (0.3433893704, 0.5802777942),
    "logc3": (0.3910068320, 0.5706315581),
    "nlog": (0.3636677701, 0.6050830890),
}

# Where each curve's segments meet, on the linear side.
BREAKPOINTS = {
    "slog3": [0.01125],
    "vlog": [0.01],
    "clog3": [-0.0126, 0.0126],   # Canon publishes +-0.014 of x, and x = reflectance / 0.9
    "logc3": [0.010591],
    "nlog": [0.328],
}


def _neutral(table: np.ndarray, size: int) -> np.ndarray:
    """The r == g == b axis of a .cube table. Red varies fastest, so grid point
    (i, i, i) is entry i * (1 + size + size**2)."""
    return table[:: 1 + size + size**2, 0][:size]


def test_names_match_the_implementations():
    assert set(logspace.NAMES) == set(MID_GREY) == set(BREAKPOINTS)
    assert len(logspace.NAMES) == 5


@pytest.mark.parametrize("name", logspace.NAMES)
def test_mid_grey_and_white_land_where_the_spec_says(name):
    grey, white = MID_GREY[name]
    # 1e-9, not a loose atol: these are the vendors' own published numbers to
    # full precision, so a single mistyped digit anywhere in a curve fails here.
    assert float(logspace.lin_to_log(name, 0.18)) == pytest.approx(grey, abs=1e-9)
    assert float(logspace.lin_to_log(name, 1.0)) == pytest.approx(white, abs=1e-9)


@pytest.mark.parametrize("name", logspace.NAMES)
def test_round_trip(name):
    """log -> lin -> log, and lin -> log -> lin, across the whole range.

    Four of the five are exact to 1e-13 in both directions. N-Log is not, and not
    because of this code: its published segments step 1.27e-4 apart at x = 0.328,
    so there is a band of log values no N-Log encoder can emit and nothing can
    invert into it. Pinned rather than hidden -- a genuinely mistyped constant
    moves these by orders of magnitude, not by 1e-4.
    """
    tol = {"nlog": 5e-4}.get(name, 1e-9)

    y = np.linspace(0.0, 1.0, 20001)
    assert np.abs(logspace.lin_to_log(name, logspace.log_to_lin(name, y)) - y).max() < tol

    lin = np.linspace(-0.004, 20.0, 20001)
    assert np.abs(logspace.log_to_lin(name, logspace.lin_to_log(name, lin)) - lin).max() < tol


@pytest.mark.parametrize("name", logspace.NAMES)
def test_encode_is_monotone(name):
    """A non-monotone log curve inverts tones in the baked cube forever -- the
    same failure mode ARCHITECTURE.md documents for _s_curve's clip."""
    step = np.diff(logspace.lin_to_log(name, np.linspace(0.0, 20.0, 200001)))
    assert step.min() > 0.0   # N-Log's break steps UP, so even that one holds


@pytest.mark.parametrize("name", logspace.NAMES)
def test_segments_meet_at_the_breakpoints(name):
    for cut in BREAKPOINTS[name]:
        lo = float(logspace.lin_to_log(name, cut - 1e-9))
        hi = float(logspace.lin_to_log(name, cut + 1e-9))
        # N-Log is discontinuous by 1.27e-4 in Nikon's published spec (0.13 of a
        # 10-bit code value); everything else agrees to well inside 1e-6.
        assert abs(lo - hi) < (2e-4 if name == "nlog" else 1e-6), f"{name} at {cut}"


def test_unknown_format_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown log format"):
        logspace.lin_to_log("prores", 0.18)


# ---- the baked conversion --------------------------------------------------


@pytest.mark.parametrize("name", logspace.NAMES)
def test_bake_conversion_writes_a_readable_cube(tmp_path, name):
    path = logspace.bake_conversion(name, str(tmp_path / f"{name}.cube"), size=17)
    size, table = read_cube(path)
    assert size == 17
    assert table.shape == (17**3, 3)
    assert table.min() >= 0.0 and table.max() <= 1.0
    assert np.all(np.diff(_neutral(table, 17)) >= 0), "a dip inverts tones forever"


def test_bake_conversion_rejects_a_degenerate_size(tmp_path):
    with pytest.raises(ValueError):
        logspace.bake_conversion("slog3", str(tmp_path / "x.cube"), size=1)


@pytest.mark.parametrize("name", logspace.NAMES)
def test_conversion_restores_contrast_through_ffmpeg(tmp_path, name):
    """The measurement that matters: real log pixels, ffmpeg's own lut3d.

    A linear ramp is encoded with this module's own lin_to_log, so the input is
    genuine log rather than a stand-in. Ungraded it is flat: measured std
    0.094-0.112 and p99 0.569-0.604, i.e. white sits two thirds of the way up the
    range, which is exactly why log looks washed out. Through the conversion the
    same ramp measures std 0.242-0.243 and p99 0.953-0.961.

    Mid grey is the sharper check: all five formats land 18% scene grey within
    0.408-0.412 after conversion, against Rec.709's own 0.409 -- from five
    unrelated sets of published constants (0.3434 to 0.4233 in their own log
    spaces). Agreement on one number is what says the transfer functions are
    right, not merely self-consistent.
    """
    lin = np.linspace(0.0, 1.0, 1024)
    logv = np.clip(np.asarray(logspace.lin_to_log(name, lin)), 0.0, 1.0)
    frame = np.repeat(np.repeat((logv * 255 + 0.5).astype(np.uint8)[None, :, None], 64, 0), 3, 2)
    src = tmp_path / "log.png"
    Image.fromarray(frame).save(src)

    cube = logspace.bake_conversion(name, str(tmp_path / f"{name}.cube"))
    out = tmp_path / "rec709.png"
    render.render_frame(str(src), cube, str(out))

    before = np.asarray(Image.open(src).convert("RGB"), float) / 255.0
    after = np.asarray(Image.open(out).convert("RGB"), float) / 255.0

    assert before.std(axis=(0, 1)).mean() < 0.12          # flat, as shot
    assert after.std(axis=(0, 1)).mean() > 0.22           # contrast restored
    assert np.percentile(after, 99) > 0.95                # white reaches white
    assert np.percentile(before, 99) < 0.62               # and did not, before

    grey = after[0, int(np.argmin(np.abs(lin - 0.18))), 0]
    assert grey == pytest.approx(0.409, abs=0.01), f"{name} mid grey {grey}"


def test_shoulder_keeps_overexposure_distinct(tmp_path):
    """Two stops over white must survive as separate values, not one shelf.

    S-Log3 code 0.596 is already scene linear 1.0, and code 1.0 is linear 38.4.
    A conversion that stopped at 1.0 would map every one of those to pure white.
    The shoulder puts linear 1.0 at 0.96 instead and keeps climbing above it --
    which is the whole reason the top of a log ramp is recoverable at all.
    """
    size = 65
    _, table = read_cube(logspace.bake_conversion("slog3", str(tmp_path / "s.cube"), size=size))
    neutral = _neutral(table, size)

    def at(scene_linear):
        return neutral[round(float(logspace.lin_to_log("slog3", scene_linear)) * (size - 1))]

    assert at(1.0) == pytest.approx(0.96, abs=0.02)     # white, not clipped white
    assert at(1.0) < at(2.0) < at(4.0) <= 1.0           # two stops over, still separated


# ---- detection -------------------------------------------------------------


def _clip(tmp_path, tag: str | None) -> str:
    import subprocess

    out = str(tmp_path / "clip.mp4")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=10:duration=0.3", "-c:v", "libx264"]
    if tag:
        args += ["-metadata:s:v:0", f"encoder={tag}"]
    subprocess.run([*args, "-pix_fmt", "yuv420p", out], check=True)
    return out


@pytest.mark.parametrize("tag,want", [
    ("Sony S-Log3 / S-Gamut3.Cine", "slog3"),
    ("Panasonic V-Log", "vlog"),
    ("Canon Log 3", "clog3"),
    ("ARRI LogC", "logc3"),
    ("Nikon N-Log", "nlog"),
])
def test_detect_reads_an_explicit_curve_name(tmp_path, tag, want):
    assert logspace.detect(_clip(tmp_path, tag)) == want


@pytest.mark.parametrize("tag", [None, "Sony", "Canon EOS R5", "Lavc libx264"])
def test_detect_refuses_to_guess(tmp_path, tag):
    """A camera MAKE is not a picture profile. The same body shoots Rec.709 and
    log into the same container, so guessing from "Sony" would mislabel most
    clips -- and a wrong technical LUT is worse than none, because it bakes a
    wrong contrast curve under every creative grade that follows."""
    assert logspace.detect(_clip(tmp_path, tag)) is None


def test_detect_on_a_missing_file_is_none(tmp_path):
    assert logspace.detect(str(tmp_path / "nope.mp4")) is None
