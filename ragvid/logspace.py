"""Camera log curves, and the .cube that converts one to Rec.709.

Log footage exists because a sensor sees more range than a display shows, so the
camera spends most of its code values on the shadows and compresses the rest.
The consequence people actually meet: diffuse white is NOT 1.0 in a log file.
S-Log3 puts 100% white at 0.596 and 18% grey at 0.411, so ungraded log looks
flat and washed out -- and grading on top of it grades the wrong tones, because
every curve in spec.py assumes display values.

Today the fix is to hand ragvid the camera vendor's own .cube (`input_lut`).
Most people do not have that file and do not know they need it. This module
generates it instead: five published transfer functions, a conversion LUT baked
from them, and a conservative guess at which one a clip was shot with.

Nothing here touches a GradeSpec. A log conversion is a TECHNICAL transform, and
render._vf already runs `input_lut` before the grade for exactly that reason: a
creative look sits on top of the conversion, never mixed into it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np

from .lut import _grid
from .platform import ffprobe

# ---- the transfer functions ------------------------------------------------
#
# Each pair is scene linear (1.0 = 100% diffuse white, 0.18 = mid grey) <-> the
# normalised log code value in [0, 1]. Constants are the published ones, at the
# precision the vendor publishes them; every pair is asserted continuous at its
# breakpoint by tests/test_logspace.py, with the one measured exception noted on
# N-Log below.


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# Sony S-Log3. Sony, "Technical Summary for S-Gamut3.Cine/S-Log3 and
# S-Gamut3/S-Log3" v1.01, appendix "S-Log3 Formula".
# https://pro.sony/s3/cms-static-content/uploadfile/06/1237494271406.pdf
# (identical constants in Sony's own ACES IDT, IDT.Sony.SLog3_SGamut3.ctl).
# Full range, 0-1023, never legal range. Reflectance 0.18 -> 0.4105572, which is
# 420/1023 exactly; 1.0 -> 0.5960273. Continuous to 1.2e-11 at the cut.
_SLOG3_CUT_LIN = 0.01125000
_SLOG3_CUT_LOG = 171.2102946929 / 1023.0


def _lin_to_slog3(x):
    x = _a(x)
    log = (420.0 + np.log10(np.maximum(x, 1e-12) / 0.19 + 0.01 / 0.19) * 261.5) / 1023.0
    lin = (x * (171.2102946929 - 95.0) / _SLOG3_CUT_LIN + 95.0) / 1023.0
    return np.where(x >= _SLOG3_CUT_LIN, log, lin)


def _slog3_to_lin(y):
    y = _a(y)
    log = 10.0 ** ((y * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    lin = (y * 1023.0 - 95.0) * _SLOG3_CUT_LIN / (171.2102946929 - 95.0)
    return np.where(y >= _SLOG3_CUT_LOG, log, lin)


# Panasonic V-Log. Panasonic, "V-Log/V-Gamut REFERENCE MANUAL" rev. 1.0
# (28 Nov 2014), section 3 "V-Log Formula".
# https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf
# Full range. 0.18 -> 0.4233114 (Panasonic's CV 433), 1.0 -> 0.5991177. The two
# segments differ by 3.1e-7 at the cut -- 0.0003 of a 10-bit code value, because
# 5.6 and 0.125 are the log branch's tangent rounded.
_VLOG_B, _VLOG_C, _VLOG_D = 0.00873, 0.241514, 0.598206
_VLOG_CUT_LIN, _VLOG_CUT_LOG = 0.01, 0.181


def _lin_to_vlog(x):
    x = _a(x)
    log = _VLOG_C * np.log10(np.maximum(x, 0.0) + _VLOG_B) + _VLOG_D
    return np.where(x >= _VLOG_CUT_LIN, log, 5.6 * x + 0.125)


def _vlog_to_lin(y):
    y = _a(y)
    log = 10.0 ** ((y - _VLOG_D) / _VLOG_C) - _VLOG_B
    return np.where(y >= _VLOG_CUT_LOG, log, (y - 0.125) / 5.6)


# Canon Log 3. Canon's own x is SCENE LINEAR normalised so that 1.0 is 90%
# reflectance white, not reflectance itself -- so this module's argument is
# divided by 0.9 first. Without that step 18% grey lands on 0.331 instead of
# Canon's published 0.343 (code value 351), which is the whole curve shifted a
# third of a stop. Three segments, because Canon gives negative scene values
# their own branch rather than clipping them. Reflectance 0.18 -> 0.34339
# (Canon's 34.3% / CV 351), 0.90 -> 0.56447 (Canon's 56.4% / CV 577).
#
# These are the white paper's own "Full %" constants, matching Canon's 2020 IDT
# and OCIO's CANON_CLOG3. The 2016 IDT publishes a different-looking set
# (0.42889912 / 0.07623209 / ...) which is the SAME curve expressed before a
# legal-range rescale -- the two agree to 8e-9 once scaled, and this one is
# directly the normalised code value, so it needs no post-scale at all.
_CLOG3_WHITE = 0.9
_CLOG3_K = 14.98325
_CLOG3_CUT_LIN = 0.014                                     # in Canon's x, not reflectance
_CLOG3_CUT_LOG_LO = 1.9754798 * -_CLOG3_CUT_LIN + 0.12512219
_CLOG3_CUT_LOG_HI = 1.9754798 * _CLOG3_CUT_LIN + 0.12512219


def _lin_to_clog3(x):
    z = _a(x) / _CLOG3_WHITE
    lo = -0.36726845 * np.log10(1.0 - _CLOG3_K * np.minimum(z, 0.0)) + 0.12783901
    hi = 0.36726845 * np.log10(_CLOG3_K * np.maximum(z, 0.0) + 1.0) + 0.12240537
    return np.where(z <= -_CLOG3_CUT_LIN, lo,
                    np.where(z <= _CLOG3_CUT_LIN, 1.9754798 * z + 0.12512219, hi))


def _clog3_to_lin(y):
    y = _a(y)
    lo = (1.0 - 10.0 ** ((0.12783901 - y) / 0.36726845)) / _CLOG3_K
    hi = (10.0 ** ((y - 0.12240537) / 0.36726845) - 1.0) / _CLOG3_K
    z = np.where(y <= _CLOG3_CUT_LOG_LO, lo,
                 np.where(y <= _CLOG3_CUT_LOG_HI, (y - 0.12512219) / 1.9754798, hi))
    return z * _CLOG3_WHITE


# ARRI LogC3 at EI 800. ARRI, "ALEXA Log C Curve - Usage in VFX" (2017-03),
# appendix table "Log C values and exposure values", row EI 800.
# https://www.arri.com/resource/blob/31918/66f56e6abb6e5b6553929edf9aa7483e/2017-03-alexa-logc-curve-in-vfx-data.pdf
# LogC3 is EI-dependent; EI 800 is the native sensitivity and the one every
# "LogC" .cube on the internet means. This is the LINEAR SCENE EXPOSURE table
# (a = 5.555556 = 1/0.18), not the sensor-signal table for the same EI, which
# shares c/d and looks deceptively similar but takes sensor signal as its input.
# 0.18 -> 0.3910068, ARRI's stated design target of 400/1023. The 2.5e-7 step at
# the cut is ARRI's 6-decimal printing of e and f, not a kink in the curve: the
# exact tangent is e = 5.36767359, f = 0.09280855.
_LOGC_A, _LOGC_B, _LOGC_C = 5.555556, 0.052272, 0.247190
_LOGC_D, _LOGC_E, _LOGC_F = 0.385537, 5.367655, 0.092809
_LOGC_CUT_LIN = 0.010591
_LOGC_CUT_LOG = _LOGC_E * _LOGC_CUT_LIN + _LOGC_F


def _lin_to_logc3(x):
    x = _a(x)
    log = _LOGC_C * np.log10(_LOGC_A * np.maximum(x, 0.0) + _LOGC_B) + _LOGC_D
    return np.where(x > _LOGC_CUT_LIN, log, _LOGC_E * x + _LOGC_F)


def _logc3_to_lin(y):
    y = _a(y)
    log = (10.0 ** ((y - _LOGC_D) / _LOGC_C) - _LOGC_B) / _LOGC_A
    return np.where(y > _LOGC_CUT_LOG, log, (y - _LOGC_F) / _LOGC_E)


# Nikon N-Log. Published in 10-bit code values, hence the /1023. Reflectance
# 0.18 -> 0.36367 (CV 372), 1.0 -> 619/1023 exactly, by construction: ln(1) = 0.
#
# N-Log is the one curve of the five that is genuinely discontinuous, and it is
# a defect in the specification rather than a rounding artefact. At x = 0.328 the
# cube-root segment gives 0.441531 and the logarithmic segment 0.441631: a step
# UP of 1.27e-4, or 0.13 of a 10-bit code value. Monotone, so nothing inverts;
# reproduced rather than smoothed, because the point of this module is to match
# what the camera actually wrote.
#
# The decode cut, though, is moved: Nikon's published inverse switches at
# 452/1023 = 0.441838, which sits ABOVE both branch values, so every log value
# in [0.441631, 0.441838] -- legitimate encoder output -- gets decoded by the
# wrong branch, for a measured round-trip error of 0.09%. Cutting at the log
# segment's own value at x = 0.328 instead makes the pair exactly invertible and
# touches a 2e-4 window nothing else can reach. Deliberate deviation, and the
# only one in this module.
_NLOG_CUT_LIN = 0.328
_NLOG_CUT_LOG = (150.0 * math.log(_NLOG_CUT_LIN) + 619.0) / 1023.0


def _lin_to_nlog(x):
    x = _a(x)
    log = (150.0 * np.log(np.maximum(x, 1e-12)) + 619.0) / 1023.0
    # No clamp at 0: N-Log's toe legitimately reaches x = -0.0075, and np.cbrt
    # takes negatives. Clamping here cost 0.124 of round-trip error at y = 0.
    root = 650.0 * np.cbrt(x + 0.0075) / 1023.0
    return np.where(x > _NLOG_CUT_LIN, log, root)


def _nlog_to_lin(y):
    y = _a(y)
    log = np.exp((y * 1023.0 - 619.0) / 150.0)
    root = (y * 1023.0 / 650.0) ** 3 - 0.0075
    return np.where(y >= _NLOG_CUT_LOG, log, root)


# name -> (linear -> log, log -> linear). Five fixed curves that have not changed
# in a decade, so a dict beats a registry.
_FORMATS = {
    "slog3": (_lin_to_slog3, _slog3_to_lin),
    "vlog": (_lin_to_vlog, _vlog_to_lin),
    "clog3": (_lin_to_clog3, _clog3_to_lin),
    "logc3": (_lin_to_logc3, _logc3_to_lin),
    "nlog": (_lin_to_nlog, _nlog_to_lin),
}

NAMES: tuple[str, ...] = tuple(_FORMATS)


def _pick(name: str):
    try:
        return _FORMATS[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown log format {name!r}; known: {', '.join(NAMES)}") from None


def lin_to_log(name: str, x) -> np.ndarray:
    """Scene linear (0.18 = mid grey, 1.0 = diffuse white) -> log code value 0-1."""
    return _pick(name)[0](x)


def log_to_lin(name: str, y) -> np.ndarray:
    """Log code value 0-1 -> scene linear. The inverse of lin_to_log."""
    return _pick(name)[1](y)


# ---- log -> Rec.709 --------------------------------------------------------

# Above this scene-linear value the highlights are rolled off instead of clipped.
# A log file's headroom is enormous -- S-Log3 code 0.596 is already linear 1.0,
# and code 1.0 is linear 38.4 -- so a conversion that stops at 1.0 welds the top
# third of every log ramp to pure white.
_KNEE = 0.8


def _shoulder(x: np.ndarray) -> np.ndarray:
    """Soft-clip scene linear into [0, 1]. Identity below _KNEE, asymptotic above.

    k + (1-k)(1 - exp(-(x-k)/(1-k))): monotone everywhere, C1 at the knee, and it
    never reaches 1.0, so nothing is welded flat. Identity below the knee is the
    load-bearing half -- it keeps 0.18 at 0.18, which is what puts mid grey where
    Rec.709 expects it instead of somewhere a "corrected" exposure invented.
    """
    over = x - _KNEE
    return np.where(x <= _KNEE, x, _KNEE + (1.0 - _KNEE) * (1.0 - np.exp(-over / (1.0 - _KNEE))))


def _rec709_oetf(x: np.ndarray) -> np.ndarray:
    """Scene linear -> Rec.709 display values. ITU-R BT.709-6, section 1.2.

    (The two segments differ by 2.4e-4 at E = 0.018; that step is in the
    Recommendation itself, not introduced here.)
    """
    x = np.clip(x, 0.0, None)
    return np.where(x < 0.018, 4.5 * x, 1.099 * x ** 0.45 - 0.099)


def bake_conversion(name: str, out_path: str, size: int = 33) -> str:
    """Write a .cube taking `name`'s log space to Rec.709 display. -> out_path.

    log code value -> scene linear -> highlight shoulder -> BT.709 OETF. That is
    the same three steps a vendor's own "log to Rec.709" LUT performs, and it
    deliberately stops there: the remaining ~1.2 system gamma that makes an image
    look right is the viewing display's job, not the transform's. Adding it here
    would bake a second contrast curve under every creative grade.

    33 is enough. lut.py escalates to 65 for hue qualifiers because their
    gradient kinks make trilinear reconstruction only first-order accurate; this
    transform is smooth and per-channel, with no kinks in RGB at all, so the
    extra 8x file buys nothing.
    """
    if size < 2:
        raise ValueError(f"LUT size must be >= 2, got {size}")
    to_lin = _pick(name)[1]
    table = np.clip(_rec709_oetf(_shoulder(to_lin(_grid(size)))), 0.0, 1.0)
    if d := os.path.dirname(out_path):
        os.makedirs(d, exist_ok=True)
    # Same header and float format as lut.bake_cube, and newline="\n" for the
    # same reason: a .cube is an interchange file and must not depend on the
    # host's line endings. Not shared with bake_cube because that one takes a
    # GradeSpec, and a log conversion is not expressible as one.
    with open(out_path, "w", newline="\n") as f:
        f.write(
            f'TITLE "ragvid {name} to Rec.709"\nLUT_3D_SIZE {size}\n'
            "DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n"
        )
        np.savetxt(f, table, fmt="%.6f")
    return out_path


# ---- detection -------------------------------------------------------------

# Substrings that NAME a curve, in the order they are tested. Nothing here
# matches a camera make on its own -- see detect().
_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("slog3", ("s-log3", "slog3", "s-log 3", "sgamut3", "s-gamut3")),
    ("clog3", ("canon log 3", "clog3", "c-log3", "canonlog3")),
    ("vlog", ("v-log", "vlog", "v-gamut", "vgamut")),
    ("logc3", ("logc3", "log-c3", "logc", "log-c", "arri")),
    ("nlog", ("n-log", "nlog")),
)


def detect(path: str) -> str | None:
    """Guess the log format from ffprobe metadata, or None when it cannot tell.

    Conservative on purpose, and it will usually return None. A camera MAKE is
    not evidence of a picture profile: the same Sony body shoots Rec.709 and
    S-Log3 into the same file format, so "Sony" would mislabel the majority of
    clips. Nor is there an in-band signal to fall back on -- ffmpeg's
    AVColorTransferCharacteristic (ITU-T H.273) has no code point for any camera
    log curve, so `color_transfer` on a real S-Log3 file reads "unknown". Only an
    explicit name in a tag counts.

    That asymmetry is deliberate: a wrong technical LUT is worse than none. It
    bakes a wrong contrast curve underneath every creative grade that follows,
    and the person has no way to see that the conversion, rather than the look,
    is what is off. None means "ask the user", which is a good outcome.

    "logc"/"arri" are matched together because ARRI ships no other log curve on
    the cameras that write these tags; LogC4 (ALEXA 35) is not implemented here
    and would need its own marker before that shortcut stops being true.
    """
    proc = subprocess.run(
        [ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=color_transfer,color_primaries,color_space:stream_tags:format_tags",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None

    values: list[str] = []
    for stream in info.get("streams") or []:
        values += [str(v) for v in stream.values() if not isinstance(v, dict)]
        values += [str(v) for v in (stream.get("tags") or {}).values()]
    values += [str(v) for v in ((info.get("format") or {}).get("tags") or {}).values()]
    blob = " ".join(values).lower()

    for name, markers in _MARKERS:
        if any(m in blob for m in markers):
            return name
    return None
