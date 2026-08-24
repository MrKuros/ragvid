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

from ragvid import render, server

HARNESS = Path(__file__).parent / "page_harness.js"
MARKER = "// PAGE_SOURCE_HERE"


def _script(page: str) -> str:
    """The page's one <script> block, without its tags."""
    found = re.search(r"<script>\n(.*)</script>", page, re.S)
    assert found, "index.html no longer has a single <script> block"
    return found.group(1)


def _run_harness(page: str, tmp_path: Path) -> subprocess.CompletedProcess:
    harness = HARNESS.read_text(encoding="utf-8")
    assert MARKER in harness
    js = tmp_path / "harness.js"
    # encoding= on both, and neither is optional on Windows: the page carries
    # emoji, Python's text mode there defaults to cp1252, and the failure is a
    # UnicodeEncodeError on the write and a UnicodeDecodeError inside
    # subprocess's reader THREAD -- which surfaces as `proc.stdout is None`
    # rather than as an exception this call can catch.
    js.write_text(harness.replace(MARKER, _script(page)), encoding="utf-8")
    return subprocess.run(["node", str(js)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)


@pytest.fixture(autouse=True)
def _needs_node():
    if not shutil.which("node"):
        pytest.skip("node is not installed")


def test_the_page_runs_and_its_live_preview_behaves(tmp_path):
    proc = _run_harness(server.INDEX.read_text(encoding="utf-8"), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "page harness ok" in proc.stdout


def test_the_harness_fails_against_the_previous_page(tmp_path):
    """The check on the check. These assertions did not exist before this
    change, so the same harness must reject the page that came before it --
    otherwise it is asserting nothing about what was added."""
    git = subprocess.run(["git", "show", "HEAD:ragvid/web/index.html"],
                         capture_output=True, text=True, encoding="utf-8",
                         cwd=Path(__file__).parent.parent)
    if git.returncode != 0:
        pytest.skip("no committed index.html to compare against")
    # The marker is the newest thing the harness asserts about, not the oldest:
    # once bannerGet is committed there is nothing older here to fail against
    # and the next change has to move this line to its own marker.
    if "bannerGet" in git.stdout:
        pytest.skip("the segmentation row is already committed; nothing older to fail against")
    proc = _run_harness(git.stdout, tmp_path)
    assert proc.returncode != 0, "the harness passed on the pre-change page — it tests nothing"


# ---- the shader itself ------------------------------------------------------


def _shader(name: str) -> str:
    page = server.INDEX.read_text(encoding="utf-8")
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


def _frame_rgb(video, at, cube=None, vf=None):
    """One frame as uint8 RGB, optionally through ffmpeg's own filter chain.

    `cube` is the one-node shorthand the base-cube checks use; `vf` is a whole
    chain, which is how the regional checks feed in `render._vf`'s own output
    rather than a second spelling of it.
    """
    import numpy as np

    # render._lut_filter, not a second spelling of it: a Windows cube path is
    # `C:\...`, and an unescaped drive colon separates filter OPTIONS -- ffmpeg
    # rejects the whole -vf with EINVAL. Reusing the library's escaping is also
    # the point, since what this test compares against is the library's output.
    chain = vf if vf else (render._lut_filter(str(cube)) if cube else None)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-ss", str(at), "-i", str(video), *(["-vf", chain] if chain else []),
         "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 3)


def _frame_size(video, at=0.0):
    """(width, height) as the DECODER produces it, not as ffprobe reports it.

    Those differ on a clip with rotation side data, and the mask has to be
    written at the size the pixels actually arrive in or scale2ref resamples it
    -- which is a second geometry and the whole point of this file is that there
    is one.
    """
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", str(at), "-i", str(video),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, check=True, text=True)
    w, h = (int(v) for v in probe.stdout.strip().split(","))
    assert len(out.stdout) == w * h * 3, "the decoder disagrees with ffprobe on this clip"
    return w, h


# ---- the regional composite -------------------------------------------------
# The shader's masks and its layer lerp, evaluated in numpy the same way its
# tetrahedra already are. `regionMask` is a transcription rather than a parse --
# vec2 swizzles are more machinery to translate than the six lines are worth --
# so it is pinned three ways: the GLSL's own text below, an equality check
# against `Region.mask` (the one true geometry), and the ffmpeg comparison.

GLSL_MASK_LINES = (
    "float s = max(b.y, 1e-6);",
    "t = (1.0 - length(d)) / s;",
    "float u = dot(p, a.xy) + a.z;",
    "t = (a.w + 0.5 * s - u) / s;",
    "t = clamp(t, 0.0, 1.0);",
    "float m = t * t * (3.0 - 2.0 * t);",
    "return b.z > 0.5 ? 1.0 - m : m;",
)


def _gl_shapes():
    """The shapes index.html's gate will draw, from the page's own allow-list."""
    page = server.INDEX.read_text(encoding="utf-8")
    found = re.search(r"const GL_SHAPES = new Set\(\[(.*?)\]\)", page)
    assert found, "index.html no longer allow-lists the shapes it can compute"
    return set(re.findall(r'"(\w+)"', found.group(1)))


GL_SHAPES = _gl_shapes()


def _gl_edge():
    """index.html's `edge` -> (dir.x, dir.y, offset) packing, from the page."""
    page = server.INDEX.read_text(encoding="utf-8")
    found = re.search(r"const GL_EDGE = \{(.*?)\};", page, re.S)
    assert found, "index.html no longer packs the linear edges"
    return {k: [float(n) for n in v.split(",")]
            for k, v in re.findall(r"(\w+): \[([^\]]+)\]", found.group(1))}


def pack_region(region):
    """glSetRegion()'s two vec4s, from a `Region`."""
    if region.shape == "radial":
        a = [region.cx, region.cy, region.rx, region.ry]
    else:
        a = [*_gl_edge()[region.edge], region.extent]
    return a, [1.0 if region.shape == "radial" else 0.0,
               region.softness, 1.0 if region.invert else 0.0, 0.0]


def shader_mask(region, w, h):
    """regionMask() over a (h, w) frame, at the uv the rasteriser hands it."""
    import numpy as np

    a, b = pack_region(region)
    x = (np.arange(w) + 0.5) / w                 # uv.x at the fragment centre
    y = ((np.arange(h) + 0.5) / h)[:, None]      # uv.y, 0 at the top row
    s = max(b[1], 1e-6)
    if b[0] > 0.5:
        d = np.hypot((x - a[0]) / max(abs(a[2]), 1e-6),
                     (y - a[1]) / max(abs(a[3]), 1e-6))
        t = (1.0 - d) / s
    else:
        u = x * a[0] + y * a[1] + a[2]
        t = (a[3] + 0.5 * s - u) / s
    t = np.clip(t, 0.0, 1.0)
    m = np.broadcast_to(t * t * (3.0 - 2.0 * t), (h, w))
    return (1.0 - m) if b[2] > 0.5 else m


def read_lut(path):
    """`lut.read_cube` in apply_shader's argument order: (table, size)."""
    from ragvid.lut import read_cube

    size, table = read_cube(path)
    return table, size


def apply_stack(x, w, h, base, layers):
    """The whole fragment shader: base cube, then each layer through its mask.

    `x` is (w*h, 3) in row-major order, `base` is (table, size) and each layer
    is (table, size, Region). This is GL_FS's main() and
    `render._region_filters`' chain, which are the same chain.
    """
    out = apply_shader(x, *base)
    for table, size, region in layers:
        m = shader_mask(region, w, h).reshape(-1, 1)
        out = out + (apply_shader(out, table, size) - out) * m
    return out


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


def test_the_shader_declares_the_slots_index_html_thinks_it_has():
    """GL_MAX_LAYERS lives in JS and the array size lives in GLSL. They are one
    number in two languages, and a stack past the smaller one is either a
    dropped layer or a link failure."""
    page = server.INDEX.read_text(encoding="utf-8")
    n = int(re.search(r"const GL_MAX_LAYERS = (\d+);", page).group(1))
    fs = _shader("GL_FS")
    assert f"uniform sampler3D lay[{n}];" in fs
    assert f"uniform vec4 rA[{n}], rB[{n}];" in fs
    # One composite line per slot, and no more: an unrolled loop that stops
    # short would silently drop the last layer.
    assert fs.count("c = mix(c, lut3d(lay[") == n
    for i in range(n):
        assert f"if (nLayers > {i}) c = mix(c, lut3d(lay[{i}], layN[{i}], c), " \
               f"regionMask(rA[{i}], rB[{i}], uv));" in fs

    # region._FOR_TARGET is the whole vocabulary compile_stack can reach. The
    # shader needs a slot for every ANALYTIC word in it, or the common case
    # falls back; the semantic ones (roadmap B2) have no closed form and are
    # turned away by cubeIsTheWholeGrade instead.
    from ragvid import region as region_mod

    analytic = [r for r in region_mod._FOR_TARGET.values() if r.shape in GL_SHAPES]
    assert n >= len(analytic), f"{len(analytic)} geometric region words, {n} slots"


def test_the_shaders_mask_is_region_pys_own_geometry():
    """Worst case over every region the vocabulary can produce, plus the awkward
    ones (softness 0, inverted, off-centre): 0.0 -- the transcription and
    `Region.mask` are the same arithmetic in the same order.

    This is the check that keeps the shader from being a SECOND implementation
    of the falloff, which is the bug region.py's module docstring exists about.
    """
    import numpy as np

    from ragvid.region import Region, _FOR_TARGET

    # Only the geometric ones: a semantic mask comes out of segment.py and has
    # no closed form for the shader to be checked against.
    regions = [r.model_copy(deep=True) for r in _FOR_TARGET.values()
               if r.shape in GL_SHAPES] + [
        Region(shape="linear", edge="bottom", extent=0.0, softness=0.0),
        Region(shape="linear", edge="left", extent=0.9, softness=1.0, invert=True),
        Region(shape="linear", edge="right", extent=0.25, softness=0.05),
        Region(shape="radial", cx=0.2, cy=0.8, rx=0.15, ry=0.4, softness=0.0),
        Region(shape="radial", cx=0.5, cy=0.5, rx=0.0, ry=0.0, softness=0.3),
    ]
    worst = 0.0
    for r in regions:
        for w, h in [(320, 180), (17, 5)]:      # 16:9, and an odd little one
            worst = max(worst, float(np.abs(shader_mask(r, w, h) - r.mask(w, h)).max()))
    assert worst == 0.0, f"the shader's mask differs from Region.mask by {worst}"

    # ...and the GLSL those numpy lines were transcribed from is still that.
    fs = _shader("GL_FS")
    for line in GLSL_MASK_LINES:
        assert line in fs, f"the shader no longer contains: {line}"


def test_the_soft_edge_stays_soft_through_the_shader(tmp_path):
    """A region's whole point is that it has no visible boundary. Sampled down
    the column that crosses one, the composited output must fall monotonically
    from the graded plateau to the untouched one -- and by no more per row than
    the smoothstep's own maximum slope allows.
    """
    import numpy as np

    from ragvid.lut import bake_cube
    from ragvid.region import Region
    from ragvid.spec import GradeSpec

    base = str(tmp_path / "base.cube")
    dark = str(tmp_path / "dark.cube")
    bake_cube(GradeSpec(), base)
    bake_cube(GradeSpec(exposure=-1.0), dark)

    w, h = 8, 240
    region = Region(shape="linear", edge="top", extent=0.4, softness=0.4)
    flat = np.full((h * w, 3), 0.5)
    out = apply_stack(flat, w, h, read_lut(base), [(*read_lut(dark), region)])
    col = out.reshape(h, w, 3)[:, 0, 0]

    assert np.all(np.diff(col) >= -1e-12), "the falloff is not monotonic"
    assert col[0] < 0.5 - 0.1, "the top is not graded"
    assert abs(col[-1] - 0.5) < 2e-3, "the bottom is not left alone"
    # Whole-frame amplitude times the smoothstep's max slope (1.5) over the
    # ramp's height in rows. A step anywhere -- a `where` instead of a lerp, a
    # mask quantised to 8 bits -- breaks this before it breaks the eye.
    span = float(col[-1] - col[0])
    assert np.abs(np.diff(col)).max() < span * 1.5 / (region.softness * h) + 1e-9


def test_overlapping_layers_compose_in_the_stacks_order(tmp_path):
    """Worst case against `GradeStack.apply`, which is the definition: 1.8 in
    8-bit code values -- the 33^3 cube's own approximation of these two specs,
    with the layers themselves adding nothing. `GradeStack.apply` evaluates the
    spec exactly; the shader reads a baked LUT, so a difference of this size is
    the bake, not the composite.

    Then the same two layers in the other order, which must NOT agree -- an
    implementation that composited them as a set rather than a sequence would
    pass the first assertion and fail this one. Measured: 24 code values apart,
    i.e. more than an order of magnitude above the bake's own error.
    """
    import numpy as np

    from ragvid.lut import bake_cube
    from ragvid.region import GradeStack, Layer, Region
    from ragvid.spec import GradeSpec

    w, h = 48, 32
    top = Region(shape="linear", edge="top", extent=0.4, softness=0.4)
    mid = Region(shape="radial", cx=0.5, cy=0.5, rx=0.6, ry=0.75, softness=0.7)
    specs = [GradeSpec(exposure=0.5, saturation=0.2), GradeSpec(contrast=0.6, temperature=-1500.0)]
    base_spec = GradeSpec(saturation=1.2, exposure=-0.2)

    cubes = []
    for i, s in enumerate([base_spec, *specs]):
        p = str(tmp_path / f"c{i}.cube")
        bake_cube(s, p)
        cubes.append(read_lut(p))

    img = np.random.default_rng(7).random((h, w, 3))
    flat = img.reshape(-1, 3)

    layers = [Layer(region=top, spec=specs[0]), Layer(region=mid, spec=specs[1])]
    want = GradeStack(base=base_spec, layers=layers).apply(img).reshape(-1, 3)
    got = apply_stack(flat, w, h, cubes[0], [(*cubes[1], top), (*cubes[2], mid)])
    worst = float(np.abs(got - want).max()) * 255
    assert worst < 2.0, f"the composite is {worst:.3f} code values off the stack"

    # The control: the same arithmetic through the same LUTs, layers reversed.
    swapped = apply_stack(flat, w, h, cubes[0], [(*cubes[2], mid), (*cubes[1], top)])
    out_of_order = float(np.abs(swapped - want).max()) * 255
    assert out_of_order > 10 * worst, (
        f"reversing the layers moved only {out_of_order:.2f} code values -- this "
        f"check cannot tell an ordered composite from an unordered one")


def test_the_shader_agrees_with_ffmpeg_with_regions_active(tmp_path):
    """Worst case, per channel, in 8-bit code values, on a real frame with two
    overlapping regional layers and deliberately strong grades: **6**, mean
    0.39, 95.1% of samples within 1.

    Against 1 for the base cube alone, and the extra five are NOT the mask.
    Measured by taking the terms away one at a time:

      * the masks agree with `Region.mask` exactly (0.0 -- the test above), and
        a single layer whose mask is 255 everywhere still disagrees by 2, so
        the geometry contributes none of it;
      * `format=argb,format=rgb24` on this clip, with no grade at all, is
        already 2 code values away from a direct yuv420p->rgb24 decode. The
        region chain forces argb (movie/alphamerge need an alpha plane), so
        that swscale round trip is inside the measurement and inside the
        export;
      * the remaining 4 is the base cube's own 1 code value AMPLIFIED. ffmpeg
        hands each filter an 8-bit buffer, so the layer's lut3d re-quantises
        and then multiplies whatever error it was given by its own slope --
        these two layers run contrast 0.4 and saturation 1.3. The shader stays
        in float and rounds once, at the end, so it is the more accurate of the
        two; the disagreement grows with layer count and layer strength, not
        with mask softness.

    So: 4 without swscale in the way, 2 with one layer, 1 with none. Modelling
    ffmpeg's per-node rounding in the numpy port does not close the gap, which
    is what rules the shader out as the source of it.
    """
    import numpy as np

    from ragvid import render
    from ragvid.lut import bake_cube
    from ragvid.region import Region
    from ragvid.spec import RGB, GradeSpec

    w, h = _frame_size("assets/sample.mp4")
    regions = [
        Region(shape="linear", edge="top", extent=0.4, softness=0.4),
        Region(shape="radial", cx=0.5, cy=0.5, rx=0.6, ry=0.75, softness=0.7, invert=True),
    ]
    specs = [
        GradeSpec(exposure=-0.6, saturation=0.35, shadow_tint=RGB(r=0.0, g=0.02, b=0.08)),
        GradeSpec(contrast=0.4, temperature=1400.0, saturation=1.3),
    ]
    base_spec = GradeSpec(saturation=1.25, contrast=0.3, exposure=0.15)

    base = str(tmp_path / "base.cube")
    bake_cube(base_spec, base)
    made, tables = [], []
    for i, (r, s) in enumerate(zip(regions, specs)):
        cube = str(tmp_path / f"layer{i}.cube")
        bake_cube(s, cube)
        made.append((cube, r.write_png(tmp_path / f"layer{i}.png", w, h)))
        tables.append((*read_lut(cube), r))

    # render.py's own chain, not a second spelling of it.
    vf = render._vf(None, render._lut_filter(base), layers=made)
    ffmpeg_out = _frame_rgb("assets/sample.mp4", 1.0, vf=vf).astype(np.int16)
    source = _frame_rgb("assets/sample.mp4", 1.0)
    shader_out = apply_stack(source.astype(np.float64) / 255.0, w, h,
                             read_lut(base), tables)
    shader_8 = np.clip(np.rint(shader_out * 255), 0, 255).astype(np.int16)

    diff = np.abs(shader_8 - ffmpeg_out)
    worst, mean, within = int(diff.max()), float(diff.mean()), float((diff <= 1).mean())
    assert worst <= 6, f"worst-case disagreement with regions: {worst} code values"
    # The max alone cannot tell a rounding tail from a wrong picture: a mask off
    # by a pixel, a layer applied in the wrong order or a mask that is silently
    # 1 everywhere all move the bulk of the frame, not its worst pixel.
    assert mean < 0.5, f"mean disagreement {mean:.3f} code values"
    assert within > 0.94, f"only {within:.1%} of samples within 1 code value"
