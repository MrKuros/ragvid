"""Semantic regions (roadmap B2): a mask whose source is the picture.

The geometry lives in tests/test_spec.py, where it landed with B1. This file is
only about the second mask source and the one thing that distinguishes it: it
needs the frame, and everything downstream must not be able to tell.

segment.class_prob is stubbed throughout. No test downloads the model -- see
tests/test_segment.py's docstring. The numbers quoted in the assertions were
measured against the REAL model on a 1280x720 outdoor frame and are reproduced
here on a synthetic probability field of the same shape, so a regression in the
feather or the gate fails without a 15 MB download.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragvid import segment
from ragvid.region import GradeStack, Layer, Region, for_target
from ragvid.spec import GradeSpec

GRID = 128


def horizon(at: float = 0.5) -> np.ndarray:
    """The model's 128x128 grid: this class fills the frame down to `at`."""
    p = np.full((GRID, GRID), 0.02)
    p[: int(GRID * at)] = 0.98
    return p


def stub_prob(monkeypatch, field: np.ndarray):
    monkeypatch.setattr(segment, "class_prob", lambda rgb, name: field)


def frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.zeros((h, w, 3))


# ---- the signature: a semantic region needs the picture --------------------


def test_a_semantic_region_refuses_to_mask_without_a_frame(tmp_path):
    """The failure that must never be silent. A semantic region resolving to
    all-ones or all-zeros because nobody passed a frame is a grade on the wrong
    pixels with a sentence claiming otherwise -- the exact class of bug this
    project's preview/export rules exist to prevent."""
    with pytest.raises(ValueError, match="sky"):
        for_target("sky").mask(64, 48)
    with pytest.raises(ValueError, match="needs the frame"):
        for_target("person").write_png(tmp_path / "m.png", 64, 48)


def test_needs_frame_lets_a_caller_ask_instead_of_catching():
    """project.bake_layers has to decide whether to decode a frame BEFORE it
    writes the PNGs, and it must not do that by reading `shape` itself."""
    assert for_target("sky").needs_frame is True
    assert for_target("top").needs_frame is False
    assert for_target("center").needs_frame is False

    flat = GradeStack(base=GradeSpec.identity())
    assert flat.needs_frame is False
    geo = GradeStack(base=GradeSpec.identity(),
                     layers=[Layer(region=for_target("top"), spec=GradeSpec.identity())])
    assert geo.needs_frame is False
    sem = GradeStack(base=GradeSpec.identity(),
                     layers=[Layer(region=for_target("top"), spec=GradeSpec.identity()),
                             Layer(region=for_target("sky"), spec=GradeSpec.identity())])
    assert sem.needs_frame is True


def test_the_frame_argument_leaves_the_geometric_shapes_exactly_as_they_were():
    """B1's masks are fully determined by width and height, so the new argument
    has to be inert for them -- including when a caller passes one anyway."""
    for word in ("top", "bottom", "left", "right", "center", "edges"):
        a = for_target(word).mask(97, 61)
        b = for_target(word).mask(97, 61, frame=frame(61, 97))
        assert np.array_equal(a, b), word


# ---- the mask is where it says it is ---------------------------------------


def test_the_mask_lands_on_the_class_and_nowhere_else(monkeypatch):
    """Measured on the real model, synthetic 1280x720 outdoor frame:
    the "sky" mask has mean 1.0000 over the top third and 0.0000 over the
    bottom third, covering 55.6% of the frame. Reproduced here against a
    probability field with its horizon in the same place."""
    stub_prob(monkeypatch, horizon(0.55))
    m = for_target("sky").mask(1280, 720, frame=frame())
    h = 720
    assert m[: h // 3].mean() == pytest.approx(1.0, abs=1e-9)
    assert m[-h // 3:].mean() == pytest.approx(0.0, abs=1e-9)
    assert 0.5 < m.mean() < 0.6


def test_the_mask_reaches_exactly_0_and_exactly_1(monkeypatch):
    """Softmax never does. Measured on the real model, using the raw probability
    as the mask left NO pixel at exactly 0 and none at exactly 1, so every pixel
    in the frame carried a sliver of the layer -- which breaks the property B1
    shipped on ("the bottom third of 'darken the top' is EXACTLY unchanged").
    Gating at segment.DECIDED before the feather is what buys it back: 53.3% of
    that frame ends up bitwise untouched, 42.2% fully graded, 4.6% in the
    falloff."""
    stub_prob(monkeypatch, horizon(0.55))
    m = for_target("sky").mask(1280, 720, frame=frame())
    assert (m == 1.0).mean() > 0.4
    assert (m == 0.0).mean() > 0.3
    assert 0.01 < ((m > 0.0) & (m < 1.0)).mean() < 0.15, "the feather is the boundary, not the frame"


def test_a_layer_changes_nothing_at_all_outside_its_mask(monkeypatch):
    """Bitwise, not approximately. `x + (g-x)*0.0` is exact, and that exactness
    is the entire reason a mask of 0 is worth having over a small number."""
    stub_prob(monkeypatch, horizon(0.5))
    rgb = np.random.default_rng(4).random((180, 320, 3))
    base = GradeSpec(contrast=0.2).sanitize()
    stack = GradeStack(base=base, layers=[
        Layer(region=for_target("sky"), spec=GradeSpec(exposure=-1.0).sanitize())])
    out = stack.apply(rgb)
    m = for_target("sky").mask(320, 180, frame=rgb)
    outside = m == 0.0
    assert outside.any()
    assert np.array_equal(out[outside], base.apply(rgb)[outside])


def test_invert_swaps_the_subject_for_everything_else(monkeypatch):
    """"everything except the sky" is one flag, not a second class list."""
    stub_prob(monkeypatch, horizon(0.5))
    r = for_target("sky")
    a = r.mask(320, 180, frame=frame(180, 320))
    r.invert = True
    b = r.mask(320, 180, frame=frame(180, 320))
    assert np.allclose(a + b, 1.0)


def test_the_grade_lands_on_the_subject_and_the_rest_is_untouched(monkeypatch):
    """"Make the sky moody" as a measurement rather than as a spec diff.

    Measured on the real model at 1280x720, exposure -0.7 / saturation 0.75:
    mask==1 pixels went 159.0 -> 96.0 code values (-63.1) and mask==0 pixels
    went 79.2 -> 79.2, max change 7e-15 code values.
    """
    stub_prob(monkeypatch, horizon(0.5))
    rgb = np.full((180, 320, 3), 0.6)
    stack = GradeStack(base=GradeSpec.identity(), layers=[
        Layer(region=for_target("sky"),
              spec=GradeSpec(exposure=-0.7, saturation=0.75).sanitize())])
    out = stack.apply(rgb)
    m = for_target("sky").mask(320, 180, frame=rgb)
    inside, outside = m == 1.0, m == 0.0
    assert (out[inside].mean() - rgb[inside].mean()) * 255 < -30.0
    assert abs(out[outside].mean() - rgb[outside].mean()) * 255 < 1e-9


# ---- the edge --------------------------------------------------------------


def test_the_feathered_edge_has_no_visible_step(monkeypatch):
    """A decided 128x128 boundary blown up to a render is a staircase: measured
    57.6 code values of max single step across a horizon at 720p with no
    feather. The default feather brings it to 13.2, which is the same
    neighbourhood as B1's geometric default (1.5/(0.4*96) = 10 code values on a
    96-row mask).

    Measured across ROWS because that is the direction the boundary runs; the
    same frame measures 5.9 across columns.
    """
    stub_prob(monkeypatch, horizon(0.55))
    m = for_target("sky").mask(1280, 720, frame=frame())
    step = np.abs(np.diff(m, axis=0)).max() * 255.0
    assert step < 16.0, f"a single row jumps {step:.1f} code values"

    raw = Region(shape="semantic", subject="sky", softness=0.0).mask(1280, 720, frame=frame())
    assert np.abs(np.diff(raw, axis=0)).max() * 255.0 > 40.0, \
        "softness 0 must still be a hard edge -- otherwise this test proves nothing"


def test_the_falloff_is_monotone_and_crosses_once(monkeypatch):
    """A mask that dips back up inside its own falloff is a visible ring. The
    box blur then smoothstep is monotone by construction; this is the check that
    the composition stayed that way."""
    stub_prob(monkeypatch, horizon(0.5))
    col = for_target("sky").mask(400, 400, frame=frame(400, 400))[:, 200]
    assert np.all(np.diff(col) <= 1e-12)
    assert col[0] == 1.0 and col[-1] == 0.0


def test_the_8_bit_png_keeps_the_edge_soft(monkeypatch, tmp_path):
    """The mask reaches ffmpeg as an 8-bit greyscale PNG, so the falloff has to
    survive quantisation. Measured on the real model: 14 code values of max
    single step against 13.2 in float."""
    from PIL import Image

    stub_prob(monkeypatch, horizon(0.55))
    path = for_target("sky").write_png(tmp_path / "sky.png", 1280, 720, frame=frame())
    arr = np.asarray(Image.open(path)).astype(int)
    assert arr.shape == (720, 1280)
    assert np.abs(np.diff(arr, axis=0)).max() <= 16
    assert arr.max() == 255 and arr.min() == 0


# ---- frame-relative, like every other Region -------------------------------


def test_the_same_region_masks_a_4k_export_and_a_preview_identically(monkeypatch):
    """The invariant the whole class rests on. The model runs at 512x512
    regardless, so the answer is resampled to the render's size and the covered
    FRACTION must not move -- the preview is where the user decides whether the
    grade is right."""
    stub_prob(monkeypatch, horizon(0.55))
    small = for_target("sky").mask(640, 360, frame=frame(360, 640))
    big = for_target("sky").mask(3840, 2160, frame=frame(2160, 3840))
    assert small.mean() == pytest.approx(big.mean(), abs=0.005)
    # And the feather scales with the frame rather than staying a pixel count.
    assert np.abs(np.diff(small, axis=0)).max() > np.abs(np.diff(big, axis=0)).max()


def test_a_frame_of_a_different_size_than_the_mask_is_fine(monkeypatch):
    """The frame is what the model reads; width/height are what the render
    needs. Tying them together would mean re-decoding at every output size."""
    stub_prob(monkeypatch, horizon(0.5))
    m = for_target("sky").mask(200, 100, frame=frame(719, 1281))
    assert m.shape == (100, 200)


# ---- the vocabulary --------------------------------------------------------


def test_every_semantic_word_resolves_to_a_semantic_region():
    """A word the model can emit that this table cannot resolve compiles to a
    silent no-op with a sentence attached."""
    for word in segment.CLASSES:
        r = for_target(word)
        assert r is not None and r.shape == "semantic" and r.subject == word


def test_a_region_round_trips_through_json():
    """session.json holds the layers, so a semantic Region has to survive
    serialisation with its subject intact -- undo and reload go through it."""
    r = for_target("foliage")
    assert Region.model_validate(r.model_dump()) == r
    assert '"subject":"foliage"' in r.model_dump_json().replace(" ", "")


# ---- exclusions (roadmap B6) -----------------------------------------------
#
# A protect reaches the picture as `Region.exclude`, so these are the mask
# arithmetic B6 rests on. Measured on the mask, which is the only thing render.py
# and GradeStack.apply ever see.


def test_an_exclusion_is_a_set_difference_and_nothing_else():
    """The composition rule, as arithmetic. Multiplication by (1 - the excluded
    mask) can only ever REMOVE coverage, is exactly 0 where the exclusion is
    exactly 1, and is bit-for-bit the unexcluded mask where it is exactly 0."""
    top = for_target("top")
    cut = for_target("top")
    cut.exclude = [for_target("bottom")]
    a, b = top.mask(32, 96), cut.mask(32, 96)
    assert np.all(b <= a + 1e-12)
    assert b[0, 0] == a[0, 0] == 1.0, "the far end of the frame is untouched"
    assert b[-1, 0] == 0.0
    # And bitwise where the exclusion is exactly zero, not merely close.
    off = for_target("bottom").mask(32, 96) == 0.0
    assert off.any() and np.array_equal(b[off], a[off])


def test_two_exclusions_compose_in_either_order():
    """(1-a)(1-b) is symmetric, which is what makes "and don't touch that
    either" mean one thing. A rule with an order would make the sentence's word
    order load-bearing, which no person expects of it.

    allclose and not array_equal, and the difference is the whole reason this
    test says what it says: float multiplication commutes but does not
    ASSOCIATE, so (m*a)*b and (m*b)*a differ by up to a measured 5.6e-17 here --
    1.4e-14 of one 8-bit code value, i.e. the same order as region.py's
    documented full-frame-mask ulp and nothing a mask can carry."""
    def m(*words):
        r = for_target("center")
        r.exclude = [for_target(w) for w in words]
        return r.mask(64, 48)

    assert np.abs(m("top", "left") - m("left", "top")).max() < 1e-15
    assert np.all(m("top", "left") <= m("top") + 1e-12)


def test_an_exclusion_survives_the_invert_it_sits_outside_of(monkeypatch):
    """`invert` is part of the region's OWN shape and the exclusion is applied
    to whatever that leaves. So "everything except the sky, minus the person"
    spares the person -- the failure would be an exclusion that inverts with
    the region and grades exactly the pixels it promised to spare."""
    stub_prob(monkeypatch, horizon(0.5))
    r = for_target("sky")
    r.invert = True
    r.exclude = [for_target("top")]
    m = r.mask(320, 180, frame=frame(180, 320))
    assert m[:20].max() == 0.0, "the top is spared on BOTH sides of the invert"
    assert m[-1, 0] == 1.0, "and the bottom, which neither covers, is fully in"


def test_outside_is_the_whole_frame_minus_the_union():
    """What a protect leaves for the ops that named no region of their own."""
    from ragvid.region import outside

    m = outside([for_target("top"), for_target("bottom")]).mask(32, 96)
    a = for_target("top").mask(32, 96)
    b = for_target("bottom").mask(32, 96)
    assert np.allclose(m, (1.0 - a) * (1.0 - b))
    assert m[0, 0] == 0.0 and m[-1, 0] == 0.0
    # "everything except the edges" is the middle, not the edges again: outside()
    # flips whatever invert the vocabulary already carries rather than setting it.
    assert outside([for_target("edges")]).mask(80, 60)[30, 40] == 1.0
    assert outside([for_target("center")]).mask(80, 60)[30, 40] == 0.0


def test_needs_frame_is_recursive_through_the_exclusions():
    """project.bake_layers decides whether to decode a frame from this ONE
    property. A geometric region protecting a person needs the picture, and a
    caller that answered from `shape` alone would hand mask() no frame and get
    a ValueError where it wanted a mask."""
    r = for_target("top")
    assert r.needs_frame is False
    r.exclude = [for_target("person")]
    assert r.needs_frame is True
    assert GradeStack(base=GradeSpec.identity(),
                      layers=[Layer(region=r, spec=GradeSpec.identity())]).needs_frame

    with pytest.raises(ValueError, match="person"):
        r.mask(64, 48)


def test_an_excluded_region_round_trips_through_json():
    """session.json holds the layers, so a protect has to survive undo and
    reload with the thing it spares intact."""
    r = for_target("top")
    r.exclude = [for_target("person")]
    assert Region.model_validate(r.model_dump()) == r
    assert '"subject":"person"' in r.model_dump_json().replace(" ", "")


def test_a_region_with_no_exclusions_is_bit_for_bit_what_b1_shipped():
    """The identity property one level down: adding a field to Region must not
    have moved a single mask that does not use it."""
    for word in ("top", "left", "center", "edges"):
        r = for_target(word)
        assert r.exclude == []
        m = r.mask(64, 48)
        assert np.array_equal(m, r.model_copy(deep=True).mask(64, 48))
