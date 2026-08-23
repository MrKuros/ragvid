"""index.html's <script>, run under node against a stubbed DOM.

There is no browser here, so the things a browser would prove -- that the
shader draws, that the decoder hands over pixels -- are out of reach. Everything
*around* the shader is plain JavaScript and is exactly the part that fails
quietly: whether the canvas is used at all, which cube goes into which texture
unit, and what each failure falls back to. That is what runs here.

Two other checks stand in for the browser:

* the GLSL is handed to `glslangValidator` when the host has one, so a typo in
  the shader is a test failure rather than a black rectangle on someone's
  machine;
* `test_the_harness_fails_against_the_previous_page` runs the identical harness
  against the committed index.html and demands that it FAIL. A harness that
  passes both ways is testing nothing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ragvid import server

HARNESS = Path(__file__).parent / "page_harness.js"
MARKER = "// PAGE_SOURCE_HERE"


def _script(page: str) -> str:
    """The page's one <script> block, without its tags."""
    found = re.search(r"<script>\n(.*)</script>", page, re.S)
    assert found, "index.html no longer has a single <script> block"
    return found.group(1)


def _run_harness(page: str, tmp_path: Path) -> subprocess.CompletedProcess:
    harness = HARNESS.read_text()
    assert MARKER in harness
    js = tmp_path / "harness.js"
    js.write_text(harness.replace(MARKER, _script(page)))
    return subprocess.run(["node", str(js)], capture_output=True, text=True, timeout=60)


@pytest.fixture(autouse=True)
def _needs_node():
    if not shutil.which("node"):
        pytest.skip("node is not installed")


def test_the_page_runs_and_its_live_preview_behaves(tmp_path):
    proc = _run_harness(server.INDEX.read_text(), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "page harness ok" in proc.stdout


def test_the_harness_fails_against_the_previous_page(tmp_path):
    """The check on the check. These assertions did not exist before this
    change, so the same harness must reject the page that came before it --
    otherwise it is asserting nothing about what was added."""
    git = subprocess.run(["git", "show", "HEAD:ragvid/web/index.html"],
                         capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    if git.returncode != 0:
        pytest.skip("no committed index.html to compare against")
    if "parseCube" in git.stdout:
        pytest.skip("the live preview is already committed; nothing older to fail against")
    proc = _run_harness(git.stdout, tmp_path)
    assert proc.returncode != 0, "the harness passed on the pre-change page — it tests nothing"


# ---- the shader itself ------------------------------------------------------


def _shader(name: str) -> str:
    page = server.INDEX.read_text()
    found = re.search(rf"const {name} = `(.*?)`;", page, re.S)
    assert found, f"{name} is not in index.html"
    # One template substitution, and it is the clipping threshold.
    return found.group(1).replace("${RAIL}", repr(1.5 / 255))


@pytest.mark.parametrize("name,suffix", [("GL_VS", ".vert"), ("GL_FS", ".frag")])
def test_the_shaders_compile(name, suffix, tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / f"s{suffix}"
    path.write_text(_shader(name))
    proc = subprocess.run(["glslangValidator", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_shader_interpolates_tetrahedrally_not_trilinearly():
    """ffmpeg's lut3d defaults to tetrahedral and render.py passes no interp=,
    so hardware texture filtering -- which is trilinear -- would be the wrong
    reconstruction. Measured over all 16.7M 8-bit colours on a 65^3 grade the
    two differ by at most 2 code values, but 'small' is not 'the same', and a
    float LUT texture is not linearly filterable in core WebGL2 anyway.
    """
    fs = _shader("GL_FS")
    assert "texelFetch" in fs, "the shader must address texels, not filter between them"
    # One declaration and six four-corner blends -- the six tetrahedra.
    assert fs.count("c000") == 7, "the six-tetrahedron split is not intact"
    for corner in ("c100", "c010", "c001", "c110", "c101", "c011", "c111"):
        assert corner in fs, f"{corner} is never sampled"


def test_the_clipping_test_is_the_one_clipstats_uses():
    """C5's marks and `ClipStats.clipped_high` / `crushed_low` must be the same
    question asked twice, or the warning points at pixels the numbers do not."""
    from ragvid import probe

    fs = _shader("GL_FS")
    rail = repr(probe._RAIL)
    assert f"1.0 - {rail}" in fs, "the top rail is not probe._RAIL"
    assert f"<= {rail}" in fs, "the bottom rail is not probe._RAIL"
    # clipped_high is a MAX over channels at the top, crushed_low a MIN at the
    # bottom -- swapping them marks a saturated red as crushed.
    assert "max(c.r, max(c.g, c.b)) >= 1.0" in fs
    assert "min(c.r, min(c.g, c.b)) <=" in fs


# ---- agreement with the export ----------------------------------------------
# The one thing that decides whether the live preview is a feature or a bug. A
# browser cannot be run here, so the shader's own text is parsed out of the page
# and evaluated in numpy against what ffmpeg's lut3d actually produces. Change a
# weight, a corner or a comparison in the GLSL and this fails.

FS_COND = re.compile(r"if \(([^)]*)\)")
FS_RETURN = re.compile(r"return\s+([^;]+);", re.S)


def _tetrahedra():
    """(five conditions, six expressions) lifted out of the shader's lut3d()."""
    fs = _shader("GL_FS")
    body = fs[fs.index("vec3 lut3d("):fs.index("void main(")]
    conds, exprs = FS_COND.findall(body), [" ".join(e.split()) for e in FS_RETURN.findall(body)]
    assert len(conds) == 5 and len(exprs) == 6, (conds, exprs)
    return conds, exprs


def _eval(src, d, c):
    """One GLSL expression over vec3s, as numpy. `d.r` and `c101` are the whole
    vocabulary; nothing else in these six lines is a name."""
    py = re.sub(r"\bd\.([rgb])\b", r'd["\1"]', src)
    py = re.sub(r"\bc(\d{3})\b", r'c["\1"]', py)
    return eval(py, {"__builtins__": {}}, {"d": d, "c": c})  # noqa: S307 -- our own file


def apply_shader(x, table, size):
    """What the fragment shader computes, from the fragment shader's own source."""
    import numpy as np

    conds, exprs = _tetrahedra()
    grid = np.clip(x, 0.0, 1.0) * (size - 1)
    p = np.floor(grid).astype(np.int64)
    q = np.minimum(p + 1, size - 1)
    dv = grid - p
    d = {k: dv[:, i:i + 1] for i, k in enumerate("rgb")}

    def corner(rx, gy, bz):
        i = ((q if rx else p)[:, 0] + (q if gy else p)[:, 1] * size
             + (q if bz else p)[:, 2] * size * size)
        return table[i]

    c = {f"{r}{g}{b}": corner(r, g, b)
         for r in (0, 1) for g in (0, 1) for b in (0, 1)}
    t = [_eval(cond, d, c) for cond in conds]
    out = np.empty_like(c["000"])
    # The if/else-if/else nesting the six live in, which is structure rather
    # than text: two three-way splits, the first gated on `d.r > d.g`.
    picks = [(t[0] & t[1], 0), (t[0] & ~t[1] & t[2], 1), (t[0] & ~t[1] & ~t[2], 2),
             (~t[0] & t[3], 3), (~t[0] & ~t[3] & t[4], 4), (~t[0] & ~t[3] & ~t[4], 5)]
    for mask, i in picks:
        out = np.where(mask, _eval(exprs[i], d, c), out)
    return out


def _frame_rgb(video, at, cube=None):
    """One frame as uint8 RGB, optionally through ffmpeg's own lut3d."""
    import numpy as np

    vf = ["-vf", f"lut3d=file={cube}"] if cube else []
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-ss", str(at), "-i", str(video), *vf, "-frames:v", "1",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 3)


@pytest.mark.parametrize("name", ["identity", "strong"])
def test_the_shader_agrees_with_ffmpeg_on_the_same_frame(name, tmp_path):
    """Worst case, per channel, in 8-bit code values: 1.

    Both grades, on a real frame. `strong` turns on hue qualifiers, which is
    when lut.py escalates to 65^3 and when the interpolation has kinks to get
    wrong. Measured separately over ALL 16.7M 8-bit colours rather than one
    frame's gamut, the same number holds -- and ffmpeg's own trilinear differs
    from its own tetrahedral by up to 2 there, which is why the shader spells
    the tetrahedra out instead of letting the sampler filter.
    """
    import numpy as np

    from ragvid.lut import bake_cube, read_cube
    from ragvid.spec import RGB, GradeSpec, HueBand

    specs = {
        "identity": GradeSpec(),
        "strong": GradeSpec(
            saturation=1.45, temperature=1800.0, contrast=0.55, exposure=-0.4,
            shadow_tint=RGB(r=-0.05, g=0.0, b=0.09),
            highlight_tint=RGB(r=0.07, g=0.02, b=-0.05),
            hue_red=HueBand(sat=1.6, lum=0.08), hue_cyan=HueBand(sat=0.25, lum=-0.1),
            hue_green=HueBand(sat=0.4, lum=-0.05)),
    }
    spec = specs[name]
    cube = str(tmp_path / f"{name}.cube")
    bake_cube(spec, cube)
    size, table = read_cube(cube)
    assert size == (65 if name == "strong" else 33)

    source = _frame_rgb("assets/sample.mp4", 1.0)
    ffmpeg_out = _frame_rgb("assets/sample.mp4", 1.0, cube).astype(np.int16)
    shader_out = apply_shader(source.astype(np.float64) / 255.0, table, size)
    shader_8 = np.clip(np.rint(shader_out * 255), 0, 255).astype(np.int16)

    worst = int(np.abs(shader_8 - ffmpeg_out).max())
    assert worst <= 1, f"{name}: worst-case disagreement {worst} code values"


def test_the_marks_are_clipstats_own_counts(tmp_path):
    """C5 must point at exactly the pixels `clipped_high` and `crushed_low`
    count, or the warning and the numbers the compiler reads disagree."""
    import numpy as np

    from ragvid import probe
    from ragvid.lut import bake_cube
    from ragvid.spec import GradeSpec

    # A grade that really does blow and crush something to count.
    cube = str(tmp_path / "hard.cube")
    bake_cube(GradeSpec(contrast=0.9, exposure=0.6, saturation=1.4), cube)
    frame = _frame_rgb("assets/sample.mp4", 1.0, cube)

    v = frame.astype(np.float64) / 255.0
    rail = probe._RAIL
    blown = v.max(axis=1) >= 1.0 - rail          # the shader's first branch
    crushed = v.min(axis=1) <= rail              # ...and its second

    stats = probe._frame_stats(frame)
    assert blown.mean() == pytest.approx(stats["clipped_high"])
    assert crushed.mean() == pytest.approx(stats["crushed_low"])
    assert blown.any() and crushed.any(), "the fixture marks nothing; it proves nothing"
