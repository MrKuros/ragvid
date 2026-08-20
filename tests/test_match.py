import numpy as np
import pytest

from ragvid.match import match_reference
from ragvid.spec import RGB

try:  # probe.py may land after this module; it only needs to quack right.
    from ragvid.probe import ClipStats
except ImportError:  # pragma: no cover
    from pydantic import BaseModel

    class ClipStats(BaseModel):
        mean: RGB
        std: RGB
        saturation: float
        frames_sampled: int
        width: int
        height: int
        duration: float


def stats(mean, std, sat=0.3):
    return ClipStats(
        mean=RGB(r=mean[0], g=mean[1], b=mean[2]),
        std=RGB(r=std[0], g=std[1], b=std[2]),
        saturation=sat,
        frames_sampled=10,
        width=640,
        height=360,
        duration=4.0,
    )


def sample(st, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(st.mean.as_array(), st.std.as_array(), size=(n, 3)), 0.0, 1.0)


def test_identity_when_ref_equals_src():
    s = stats([0.4, 0.45, 0.5], [0.1, 0.12, 0.09])
    assert match_reference(s, s).is_identity()


def test_roundtrip_lands_on_reference():
    src = stats([0.30, 0.35, 0.42], [0.05, 0.06, 0.07], sat=0.3)
    ref = stats([0.50, 0.45, 0.36], [0.10, 0.09, 0.11], sat=0.3)  # same sat -> pure CDL
    out = match_reference(src, ref).apply(sample(src))
    assert np.allclose(out.mean(axis=0), ref.mean.as_array(), atol=0.01)
    assert np.allclose(out.std(axis=0), ref.std.as_array(), atol=0.01)


def test_moves_toward_reference_with_saturation_change():
    src = stats([0.25, 0.30, 0.45], [0.04, 0.05, 0.08], sat=0.20)
    ref = stats([0.55, 0.50, 0.38], [0.09, 0.10, 0.06], sat=0.35)
    data = sample(src)
    out = match_reference(src, ref).apply(data)
    target = ref.mean.as_array()
    before = np.linalg.norm(data.mean(axis=0) - target)
    after = np.linalg.norm(out.mean(axis=0) - target)
    assert after < before / 3, (before, after)
    # ref is the more saturated of the two, so chroma must go up. (It fights the
    # std match: saturation runs after the CDL and re-widens the channels, which
    # is why the exact std assertion lives in the equal-saturation test above.)
    chroma = lambda a: float(np.mean(a.max(-1) - a.min(-1)))
    assert chroma(out) > chroma(data)


@pytest.mark.parametrize("std", [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
def test_flat_source_does_not_blow_up(std):
    src = stats([0.5, 0.5, 0.5], std, sat=0.0)
    ref = stats([0.3, 0.4, 0.6], [0.1, 0.1, 0.1], sat=0.4)
    spec = match_reference(src, ref)
    for arr in (spec.slope.as_array(), spec.offset.as_array()):
        assert np.all(np.isfinite(arr))
    assert spec.slope.r == 1.0 and spec.saturation == 1.0
    assert np.all(np.isfinite(spec.apply(sample(src))))


def _pinned_white(spec):
    """Fraction of a luminance ramp this grade flattens to exactly 1.0."""
    ramp = np.linspace(0.0, 1.0, 4096)[:, None].repeat(3, axis=1)
    return float((spec.apply(ramp).max(axis=-1) >= 1.0 - 1e-9).mean())


def test_bright_reference_does_not_hard_clip_the_highlights():
    """Matching a bright reference must not flatten the top of the range to paper white.

    Regression: this path runs with no LLM at all, so nothing downstream was ever
    going to notice. Before highlight_rolloff was solved for here, matching
    ref_tvd.png onto ironman.gif produced slope 1.43 and pinned 28.9% of a
    4096-step ramp at exactly 1.0 -- a third of the tonal range as detail-free
    white, in the one code path that is meant to be pure measurement.
    """
    src = stats((0.25, 0.22, 0.20), (0.12, 0.12, 0.12))
    ref = stats((0.55, 0.55, 0.58), (0.20, 0.20, 0.20))
    spec = match_reference(src, ref)

    assert spec.slope.as_array().max() > 1.0, "test needs a brightening match to be meaningful"
    assert spec.highlight_rolloff > 0.0
    # Not exactly zero: the shoulder is solved so the PEAK input lands exactly on
    # 1.0, so the single topmost ramp sample legitimately reads as white. What
    # must not survive is a whole flattened region -- 28.9% before, 1 sample now.
    assert _pinned_white(spec) < 0.01
    # White loss is the unavoidable price, and it is real: no monotone shoulder
    # can be the identity on [0,1] and still cap at 1 (that IS hard clipping), so
    # legal whites must come down. Bound it rather than pretending it is free.
    assert spec.apply(np.ones((1, 3))).max() > 0.85


def test_darkening_match_leaves_rolloff_alone():
    """No overshoot, no shoulder -- and therefore no white loss to pay for."""
    src = stats((0.60, 0.60, 0.60), (0.20, 0.20, 0.20))
    ref = stats((0.30, 0.30, 0.30), (0.10, 0.10, 0.10))
    spec = match_reference(src, ref)

    assert spec.slope.as_array().max() <= 1.0
    assert spec.highlight_rolloff == 0.0
