import numpy as np
import pytest

from ragvid.lut import bake_cube, read_cube
from ragvid.spec import RGB, GradeSpec


def ref_grid(size):
    """Independent (no _grid) red-fastest grid, built with explicit loops."""
    return np.array(
        [
            [r / (size - 1), g / (size - 1), b / (size - 1)]
            for b in range(size)
            for g in range(size)
            for r in range(size)
        ]
    )


def test_identity_lut_is_the_grid(tmp_path):
    size, table = read_cube(bake_cube(GradeSpec.identity(), str(tmp_path / "id.cube"), size=17))
    assert size == 17
    assert np.allclose(table, ref_grid(17), atol=1e-6)


def test_round_trip(tmp_path):
    spec = GradeSpec(saturation=0.4, temperature=1500, contrast=0.3)
    p = bake_cube(spec, str(tmp_path / "rt.cube"), size=9)
    size, table = read_cube(p)
    assert size == 9
    assert np.allclose(table, spec.apply(ref_grid(9)), atol=1e-6)
    # re-baking the parsed values would give the same file
    assert np.allclose(read_cube(p)[1], table)


def test_red_varies_fastest(tmp_path):
    """Red-only spec: the parsed table must show red on the fast axis."""
    n = 5
    spec = GradeSpec(slope=RGB(r=0.5, g=1.0, b=1.0))
    size, table = read_cube(bake_cube(spec, str(tmp_path / "r.cube"), size=n))
    assert size == n

    expected = spec.apply(ref_grid(n))
    assert np.allclose(table, expected, atol=1e-6)

    # Explicit index arithmetic, independent of the above: entry i is grid
    # point (i%n, (i//n)%n, i//n**2), and only red was touched.
    for i, (r, g, b) in enumerate(table):
        assert r == pytest.approx(0.5 * (i % n) / (n - 1), abs=1e-6)
        assert g == pytest.approx(((i // n) % n) / (n - 1), abs=1e-6)
        assert b == pytest.approx((i // n**2) / (n - 1), abs=1e-6)

    # Sanity on the failure mode this test exists for: if blue were fastest the
    # first n entries would share a red value and differ in blue.
    assert len(set(table[:n, 0])) == n
    assert len(set(table[:n, 2])) == 1


def test_header_and_line_count(tmp_path):
    n = 4
    p = bake_cube(GradeSpec.identity(), str(tmp_path / "h.cube"), size=n)
    lines = [ln.strip() for ln in open(p) if ln.strip()]
    assert lines[0].startswith("TITLE ")
    assert lines[1] == f"LUT_3D_SIZE {n}"
    assert lines[2].split() == ["DOMAIN_MIN", "0.0", "0.0", "0.0"]
    assert lines[3].split() == ["DOMAIN_MAX", "1.0", "1.0", "1.0"]
    assert len(lines) == n**3 + 4
    for ln in lines[4:]:
        vals = [float(v) for v in ln.split()]
        assert len(vals) == 3 and all(0.0 <= v <= 1.0 for v in vals)


def test_bad_size(tmp_path):
    with pytest.raises(ValueError):
        bake_cube(GradeSpec.identity(), str(tmp_path / "x.cube"), size=1)


def test_read_rejects_truncated(tmp_path):
    p = tmp_path / "bad.cube"
    p.write_text("LUT_3D_SIZE 4\n0.0 0.0 0.0\n")
    with pytest.raises(ValueError):
        read_cube(str(p))
