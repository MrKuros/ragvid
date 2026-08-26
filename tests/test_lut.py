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


# ---- reconstruction error -------------------------------------------------
# A .cube is only sampled at its grid points; ffmpeg reconstructs everything in
# between by interpolation. Hue qualifiers have gradient KINKS on the planes
# r=g, g=b and r=b, so their output is C0-but-not-C1 in RGB and interpolation
# is only FIRST-order accurate there -- error ~ 1/n, not 1/n^2. The worst points
# sit at chroma ~ 0.38, i.e. AT the sextant kinks, which is why this test uses
# RANDOM RGB points: a sweep along a saturated-hue ramp never visits them and
# would happily report a clean LUT that is visibly wrong in the shot.

from ragvid.spec import HUE_FIELDS, HueBand  # noqa: E402


def trilinear(table, size, pts):
    """Reconstruct like a LUT consumer does. `table` is red-fastest, so the
    reshaped cube is indexed [blue, green, red]."""
    cube = table.reshape(size, size, size, 3)
    c = np.clip(pts, 0.0, 1.0) * (size - 1)
    i0 = np.minimum(np.floor(c).astype(int), size - 2)
    f = c - i0
    out = np.zeros_like(pts)
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = ((f[:, 2] if db else 1 - f[:, 2])
                     * (f[:, 1] if dg else 1 - f[:, 1])
                     * (f[:, 0] if dr else 1 - f[:, 0]))
                out += w[:, None] * cube[i0[:, 2] + db, i0[:, 1] + dg, i0[:, 0] + dr]
    return out


QUALIFIED = GradeSpec(
    slope=RGB(r=1.1, g=1.0, b=0.95), saturation=1.15, contrast=0.3,
    exposure=0.2, highlight_rolloff=0.3,
    shadow_tint=RGB(r=0.0, g=0.03, b=0.05), highlight_lift=-0.02,
    hue_red=HueBand(sat=0.65, lum=0.02), hue_cyan=HueBand(sat=1.30),
    hue_blue=HueBand(sat=1.2, lum=-0.03),
)


def _max_err_code_values(spec, path, size=None, n=100_000, seed=7):
    p = bake_cube(spec, path) if size is None else bake_cube(spec, path, size=size)
    baked_size, table = read_cube(p)
    pts = np.random.default_rng(seed).random((n, 3))
    err = np.abs(trilinear(table, baked_size, pts) - spec.apply(pts)).max()
    return baked_size, err * 255.0


def test_lut_reconstruction_error_with_hue_qualifiers(tmp_path):
    """Measured over 1e5 random points, not a hue sweep."""
    size, err = _max_err_code_values(QUALIFIED, str(tmp_path / "q.cube"))
    assert size == 65, "a qualified grade must bake at 65^3"
    assert err < 6.0, f"max reconstruction error {err:.2f}/255"


def test_thirty_three_would_not_have_been_enough(tmp_path):
    """The escalation is not superstition: the same grade sampled at 33^3 is
    measurably worse. Tables are built directly here rather than through
    bake_cube, which escalates whenever a qualifier is present."""
    pts = np.random.default_rng(7).random((100_000, 3))
    exact = QUALIFIED.apply(pts)
    errs = {}
    for n in (33, 65):
        table = QUALIFIED.apply(ref_grid(n))
        errs[n] = np.abs(trilinear(table, n, pts) - exact).max() * 255.0
    # first-order accuracy at the sextant kinks: halving the step should roughly
    # halve the error, and does
    assert errs[33] > errs[65] * 1.6, f"33^3 {errs[33]:.2f} vs 65^3 {errs[65]:.2f}"
    assert errs[65] < 6.0 < errs[33] * 100  # keep the real numbers in the failure text

    # ...and it gets worse fast with band strength, which is why the default
    # cannot simply stay at 33
    extreme = QUALIFIED.model_copy(update={
        "hue_red": HueBand(sat=0.30), "hue_cyan": HueBand(sat=1.60),
        "hue_green": HueBand(sat=1.55, lum=0.05),
    })
    ex_exact = extreme.apply(pts)
    e33 = np.abs(trilinear(extreme.apply(ref_grid(33)), 33, pts) - ex_exact).max() * 255
    e65 = np.abs(trilinear(extreme.apply(ref_grid(65)), 65, pts) - ex_exact).max() * 255
    assert e33 > errs[33], f"extreme {e33:.2f} vs mild {errs[33]:.2f}"
    assert e65 < 6.0, f"extreme at 65^3: {e65:.2f}/255"


def test_a_hue_rotation_is_cheaper_to_bake_than_what_already_ships():
    """The bound on MAX_BAND_ROT is a colour argument, so this is the check that
    it is not ALSO buying a precision problem. Measured over 2e5 random points,
    max error in 8-bit code values:

                                    33^3    65^3
        today's QUALIFIED           2.63    1.28
        worst shipping band sat     3.42    1.52
        rot 12 (one "warmer")       2.07    1.03
        rot 30 (MAX_BAND_ROT)       2.20    1.09
        rot 30 on all six bands     2.06    1.08

    Rotation is the cheapest of the three band fields, and it escalates to 65^3
    anyway because it is a qualifier -- so B3b costs nothing in LUT accuracy."""
    from ragvid.compiler import MAX_BAND_ROT

    pts = np.random.default_rng(7).random((200_000, 3))

    def err(spec, n):
        return np.abs(trilinear(spec.apply(ref_grid(n)), n, pts) - spec.apply(pts)).max() * 255

    turned = GradeSpec(**{f: HueBand(rot=MAX_BAND_ROT) for f in HUE_FIELDS})
    assert err(turned, 33) < err(QUALIFIED, 33)
    assert err(turned, 65) < 1.5


def test_no_qualifier_grade_is_accurate_at_the_default_size(tmp_path):
    spec = GradeSpec(slope=RGB(r=1.2, g=1.0, b=0.85), saturation=1.3,
                     contrast=0.4, exposure=0.3, highlight_rolloff=0.4,
                     shadow_tint=RGB(r=0.0, g=0.04, b=0.06), shadow_lift=0.02)
    size, err = _max_err_code_values(spec, str(tmp_path / "n.cube"))
    assert size == 33, "no qualifier => no 8x file-size tax"
    assert err < 6.0, f"max reconstruction error {err:.2f}/255"


def test_size_escalates_only_for_hue_qualifiers(tmp_path):
    plain = GradeSpec(saturation=1.4, contrast=0.3, exposure=0.5)
    assert read_cube(bake_cube(plain, str(tmp_path / "p.cube")))[0] == 33
    assert not plain.has_hue_qualifiers()

    for field in ("hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta"):
        for band in (HueBand(sat=1.05), HueBand(lum=0.01), HueBand(rot=1.0)):
            spec = GradeSpec(**{field: band})
            assert spec.has_hue_qualifiers(), field
            assert read_cube(bake_cube(spec, str(tmp_path / f"{field}.cube")))[0] == 65

    # an explicitly requested size is honoured, qualifiers or not
    assert read_cube(bake_cube(QUALIFIED, str(tmp_path / "x.cube"), size=17))[0] == 17


def test_baked_luts_are_always_finite_and_in_range(tmp_path):
    """Adversarial but sanitize()-legal specs, straight through the writer:
    a NaN in a .cube makes ffmpeg fail at render time, long after the grade."""
    from ragvid.spec import EffectSpec

    specs = [
        GradeSpec(offset=RGB.of(-1.0), power=RGB.of(0.05)),       # neg ^ fractional
        GradeSpec(power=RGB.of(0.0), slope=RGB.of(8.0)),          # zero power
        GradeSpec(slope=RGB.of(1e9), exposure=99.0, highlight_rolloff=5.0),
        GradeSpec(saturation=float("inf"), contrast=-2.0, look_mix=-1.0,
                  shadow_tint=RGB.of(9.0), highlight_lift=float("-inf"),
                  hue_red=HueBand(sat=-4.0, lum=7.0),
                  effects=EffectSpec(glow=float("nan"))),
        GradeSpec(pivot=float("nan"), power=RGB.of(float("nan")), contrast=1.0),
    ]
    for i, raw in enumerate(specs):
        p = bake_cube(raw.sanitize(), str(tmp_path / f"hot{i}.cube"), size=17)
        _, table = read_cube(p)                      # would raise on "nan" rows
        assert np.all(np.isfinite(table)), i
        assert table.min() >= 0.0 and table.max() <= 1.0
        assert "nan" not in open(p).read().lower()
