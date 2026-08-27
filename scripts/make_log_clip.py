"""Derive a 10-bit S-Log3 clip from the Rec.709 fixture, then check ragvid's own
conversion brings it back.

The five log transforms in logspace.py have only ever been proven against
generated ramps. This puts real image content through one: encode the fixture to
S-Log3, then ask ragvid to convert it back, and measure what came out against
what went in. It is not a substitute for real camera output -- the gamut is still
Rec.709 and the sensor noise is still 8-bit -- and the report says so.
"""
import subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ragvid.logspace import bake_conversion, lin_to_log
from ragvid.lut import _grid

SP = REPO / "out" / "log"
SP.mkdir(parents=True, exist_ok=True)
SRC = str(REPO / "test_files" / "test.mp4")
OUT = SP / "test_slog3_10bit.mkv"


def rec709_eotf_inv(y):
    """Rec.709 display -> scene linear. The inverse of logspace._rec709_oetf,
    segment for segment, including the 2.4e-4 step the Recommendation itself has
    at E = 0.018 (y = 0.081)."""
    y = np.clip(y, 0.0, 1.0)
    return np.where(y < 4.5 * 0.018, y / 4.5, ((y + 0.099) / 1.099) ** (1.0 / 0.45))


def write_cube(table, path, title):
    n = round(len(table) ** (1 / 3))
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\n'
                "DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        np.savetxt(f, table, fmt="%.6f")


SIZE = 65   # the ENCODE side wants headroom: S-Log3 packs the shadows hard
grid = _grid(SIZE)
enc = np.clip(lin_to_log("slog3", rec709_eotf_inv(grid)), 0.0, 1.0)
write_cube(enc, SP / "rec709_to_slog3.cube", "rec709 to S-Log3 (test fixture only)")
bake_conversion("slog3", str(SP / "slog3_to_rec709.cube"), size=33)

print("round trip, in numpy, before any video is touched:")
back_path = SP / "slog3_to_rec709.cube"
from ragvid.logspace import _pick, _rec709_oetf, _shoulder
to_lin = _pick("slog3")[1]
back = _rec709_oetf(_shoulder(to_lin(enc)))
err = np.abs(back - grid)
below = grid.max(axis=1) <= 0.91            # under the highlight shoulder
print(f"  all points          max |err| {err.max():.6f}  = {err.max()*255:.3f} code values")
print(f"  below the shoulder  max |err| {err[below].max():.6f}  = {err[below].max()*255:.3f} code values")
# 18% SCENE LINEAR is the number logspace.py pins ("all five land 18% grey on
# 0.408-0.412"), and in Rec.709 DISPLAY that sits at 0.409 -- so the check is
# display 0.409 -> linear 0.18 -> the S-Log3 code the docstring promises.
disp = np.array([[0.409, 0.409, 0.409]])
gl = rec709_eotf_inv(disp)
print(f"  Rec.709 display 0.409 -> scene linear {gl[0,0]:.4f} (want 0.18) "
      f"-> S-Log3 code {lin_to_log('slog3', gl)[0,0]:.4f} (want 0.408-0.412)")

print("\nencoding the fixture to 10-bit S-Log3 (FFV1 in .mkv: MP4 only learned ffv1 recently)")
subprocess.run([
    "ffmpeg", "-v", "error", "-y", "-i", SRC,
    "-vf", f"format=gbrp16le,lut3d=file={SP/'rec709_to_slog3.cube'},format=yuv420p10le",
    "-c:v", "ffv1", "-level", "3", "-an", "-t", "5", str(OUT)], check=True)
print(f"  {OUT}  {OUT.stat().st_size/1e6:.1f} MB")

meta = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=codec_name,pix_fmt,width,height", "-of", "csv=p=0", str(OUT)],
                      capture_output=True, text=True).stdout.strip()
print(f"  {meta}")

# what ragvid measures off it, both raw and through its own conversion
from ragvid.probe import probe_video
raw = probe_video(str(OUT))
conv = probe_video(str(OUT), input_lut=str(back_path))
print(f"\nprobe of the log clip, NO conversion: luma p50 {raw.p50:.4f}  p1 {raw.p1:.4f}  "
      f"p99 {raw.p99:.4f}  crushed_low {raw.crushed_low:.5f}")
if conv:
    print(f"probe THROUGH the S-Log3 conversion: luma p50 {conv.p50:.4f}  p1 {conv.p1:.4f}  "
          f"p99 {conv.p99:.4f}  crushed_low {conv.crushed_low:.5f}")
orig = probe_video(SRC)
print(f"probe of the ORIGINAL Rec.709 clip:  luma p50 {orig.p50:.4f}  p1 {orig.p1:.4f}  "
      f"p99 {orig.p99:.4f}  crushed_low {orig.crushed_low:.5f}")
