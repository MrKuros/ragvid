"""Bake a GradeSpec into a .cube 3D LUT (and read one back).

The .cube ordering rule: RED varies fastest, then green, then blue. So the
entry at index i is the grid point (i % N, (i // N) % N, i // N**2).
"""

from __future__ import annotations

import os

import numpy as np

from .spec import GradeSpec


def _grid(size: int) -> np.ndarray:
    """(size**3, 3) identity grid in [0,1], red fastest."""
    i = np.arange(size**3)
    return np.stack([i % size, (i // size) % size, i // size**2], axis=-1) / (size - 1)


def bake_cube(spec: GradeSpec, out_path: str, size: int = 33) -> str:
    if size < 2:
        raise ValueError(f"LUT size must be >= 2, got {size}")
    # Hue qualifiers have gradient KINKS on the planes r=g, g=b, r=b, so their
    # output is C0-but-not-C1 in RGB and trilinear/tetrahedral reconstruction is
    # only FIRST-order accurate there -- error ~ 1/n, not 1/n**2. Measured max
    # error against exact apply(), in 8-bit code values: mild qualifiers 2.60 at
    # 33^3 vs 1.15 at 65^3; extreme 4.90 vs 2.28. Widening the bands does not
    # help; it is intrinsic to hue selection. So escalate only when a band is
    # actually in use -- raising the default would 8x every .cube (0.7 -> 5.6 MB)
    # for grades that never touch a qualifier. `size == 33` means "the default";
    # an explicitly requested size is always honoured.
    if size == 33 and spec.has_hue_qualifiers():
        size = 65
    table = spec.apply(_grid(size))
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    # newline="\n": a .cube is an interchange file. Text mode on Windows would
    # emit CRLF -- harmless to ffmpeg and to read_cube, but it makes the same
    # grade a different file on a different host for no reason at all.
    with open(out_path, "w", newline="\n") as f:
        f.write(
            f'TITLE "ragvid"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n'
        )
        np.savetxt(f, table, fmt="%.6f")
    return out_path


def read_cube(path: str) -> tuple[int, np.ndarray]:
    size = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if line[0].isalpha():  # header keyword; data lines start with a digit/-/.
                key, *rest = line.split()
                if key.upper() == "LUT_3D_SIZE":
                    size = int(rest[0])
                continue
            rows.append([float(v) for v in line.split()])
    if size is None:
        raise ValueError(f"{path}: no LUT_3D_SIZE")
    if len(rows) != size**3:
        raise ValueError(f"{path}: expected {size**3} entries, got {len(rows)}")
    return size, np.array(rows, dtype=np.float64)
