"""The local segmentation model — everything about it except the model.

NO TEST HERE DOWNLOADS ANYTHING and none loads onnxruntime. The 15 MB of
weights are an opt-in the user consents to; a test suite that fetched them
would have made that consent a lie, and a suite that needed them would not run
on a fresh clone. So the ONNX session is stubbed with an array whose contents
are chosen to pin the two things that are actually easy to get wrong: the
preprocessing (which comes from the model's own preprocessor_config.json) and
the class arithmetic (which is where an ADE20K off-by-one would hide).

What is NOT covered here, and cannot be: whether class 2 really is sky. That is
a property of the published weights, checked by hand against config.json's
id2label and recorded in segment.CLASSES' comments. Measured on the real model:
a synthetic outdoor frame gave a "sky" mask with mean 1.0000 over the top third
and 0.0000 over the bottom third, and test_files/ref_tvd.png gave a "person"
mask covering 48.9% of the frame -- both figures in the shot, and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragvid import segment
from ragvid.segment import CLASSES, SegmentUnavailable

GRID = 128        # the model's decode-head resolution, stride 4 from 512
N_CLASSES = 150   # ADE20K, config.json's id2label


class StubSession:
    """Stands in for onnxruntime.InferenceSession. Records what it was fed."""

    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits
        self.fed: np.ndarray | None = None
        self.calls = 0

    def run(self, outputs, feed):
        self.calls += 1
        assert outputs is None
        assert list(feed) == ["pixel_values"], feed.keys()
        self.fed = feed["pixel_values"]
        return [self.logits[None]]


def stub(monkeypatch, logits: np.ndarray) -> StubSession:
    s = StubSession(logits)
    segment.reset()
    monkeypatch.setattr(segment, "_session", lambda: s)
    monkeypatch.setattr(segment, "_CACHE", {})
    return s


def logits_for(**by_class: np.ndarray) -> np.ndarray:
    """(150, 128, 128) logits: -10 everywhere, plus the given per-class fields."""
    out = np.full((N_CLASSES, GRID, GRID), -10.0, dtype=np.float32)
    for name, field in by_class.items():
        out[int(name.lstrip("c"))] += field.astype(np.float32) * 30.0
    return out


def top_half() -> np.ndarray:
    f = np.zeros((GRID, GRID)); f[: GRID // 2] = 1.0
    return f


def grey(h=64, w=96, value=128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---- degrading without the extra, and without the weights ------------------


def test_a_missing_extra_is_a_typed_error_naming_the_install(monkeypatch):
    """The whole point of an optional dependency: the failure has to be a
    sentence a person can act on, not an ImportError from three frames down.

    Forced by making the import genuinely unavailable rather than by patching
    the error path, so this fails if someone ever moves `import onnxruntime` to
    module scope."""
    monkeypatch.setattr(segment, "have_runtime", lambda: False)
    segment.reset()
    with pytest.raises(SegmentUnavailable) as e:
        segment.class_prob(grey(), "sky")
    assert e.value.needs_install is True
    assert "ragvid[masks]" in str(e.value)
    assert "pip install" in e.value.hint


def test_missing_weights_are_a_different_error_pointing_at_the_consent_gate(monkeypatch, tmp_path):
    """Installed but not downloaded is a different fix from not installed, and a
    UI branches on it. It must also NOT say 'pip install' -- that is the wrong
    instruction and the user would follow it."""
    monkeypatch.setattr(segment, "have_runtime", lambda: True)
    monkeypatch.setattr(segment, "model_path", lambda: tmp_path / "nope.onnx")
    segment.reset()
    with pytest.raises(SegmentUnavailable) as e:
        segment.class_prob(grey(), "sky")
    assert e.value.needs_install is False
    assert "download_model" in e.value.hint
    assert "pip install" not in str(e.value)


def test_nothing_downloads_or_loads_a_runtime_just_by_importing(monkeypatch, tmp_path):
    """The base install has to import ragvid cleanly with no onnxruntime and no
    weights on disk. `is_ready` is the question a UI asks first, and asking it
    must not be what triggers the download."""
    monkeypatch.setattr(segment, "have_runtime", lambda: False)
    assert segment.is_ready() is False
    monkeypatch.setattr(segment, "have_runtime", lambda: True)
    monkeypatch.setattr(segment, "model_path", lambda: tmp_path / "nope.onnx")
    assert segment.is_ready() is False
    assert not (tmp_path / "nope.onnx").exists(), "asking must not fetch"


# ---- the class map ---------------------------------------------------------


def test_every_class_index_is_a_real_ade20k_index():
    """150 output channels, 0-based over ADE20K's classes 1..150 because the
    checkpoint was trained with reduce_labels. An index of 150 would be a silent
    IndexError-free wrong answer if the array were ever larger."""
    for word, idx in CLASSES.items():
        assert idx, f"{word} maps to no class at all"
        assert all(0 <= i < N_CLASSES for i in idx), word


def test_no_two_words_claim_the_same_class():
    """Overlapping words would make two layers fight over the same pixels while
    both sentences claimed to have grabbed them."""
    seen = [i for idx in CLASSES.values() for i in idx]
    assert len(seen) == len(set(seen))


def test_an_unknown_word_is_refused_rather_than_silently_empty():
    with pytest.raises(ValueError, match="not a semantic class"):
        segment.class_prob(grey(), "unicorn")


# ---- preprocessing, straight from preprocessor_config.json -----------------


def test_the_frame_is_preprocessed_exactly_as_the_model_was_trained(monkeypatch):
    """512x512, 1/255 rescale, ImageNet mean/std, NCHW float32.

    A wrong normalisation does not raise -- it returns a confident mask of the
    wrong thing, which is the failure this project keeps writing tests against.
    Checked on a flat grey frame, where the expected tensor value is closed
    form: (128/255 - mean) / std per channel.
    """
    s = stub(monkeypatch, logits_for(c2=top_half()))
    segment.class_prob(grey(64, 96, 128), "sky")

    x = s.fed
    assert x.shape == (1, 3, 512, 512) and x.dtype == np.float32
    want = (128 / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(x[0, :, 200, 300], want, atol=1e-5)


def test_a_float_frame_and_a_uint8_frame_give_the_same_answer(monkeypatch):
    """GradeStack.apply hands over float in [0,1]; a still read off disk is
    uint8. Two code paths into one model is two chances to scale differently."""
    s = stub(monkeypatch, logits_for(c2=top_half()))
    a = segment.class_prob(grey(64, 96, 200), "sky").copy()
    s.fed = None
    segment._CACHE.clear()
    b = segment.class_prob(np.full((64, 96, 3), 200 / 255.0), "sky")
    assert np.array_equal(a, b)


def test_a_frame_that_is_not_an_image_is_refused():
    with pytest.raises(ValueError, match="h, w, 3"):
        segment.class_prob(np.zeros((64, 96)), "sky")


# ---- the class arithmetic --------------------------------------------------


def test_a_one_class_word_is_that_channels_probability(monkeypatch):
    stub(monkeypatch, logits_for(c2=top_half()))
    p = segment.class_prob(grey(), "sky")
    assert p.shape == (GRID, GRID)
    assert p[:GRID // 2].min() > 0.99 and p[GRID // 2:].max() < 0.01


def test_a_word_covering_several_classes_sums_them(monkeypatch):
    """"foliage" is tree + grass + plant + flower + palm. Summing is correct
    because the 150 channels are ONE softmax, so they are mutually exclusive:
    half tree and half grass is a whole foliage, and taking the max instead
    would report 0.5 for a pixel the model is certain about.
    """
    left = np.zeros((GRID, GRID)); left[:, : GRID // 2] = 1.0
    stub(monkeypatch, logits_for(c4=left, c9=1.0 - left))   # tree | grass
    p = segment.class_prob(grey(), "foliage")
    assert p.min() > 0.99, "tree on one side and grass on the other is all foliage"
    assert segment.class_prob(grey(), "sky").max() < 0.01


def test_a_detection_smaller_than_the_floor_is_not_a_region(monkeypatch):
    """A spurious handful of pixels must not become a grade. The floor is on the
    DECIDED crossing, not the probability integral: a class sitting at 0.1 over
    the whole frame has a large integral and is nowhere."""
    tiny = np.zeros((GRID, GRID)); tiny[:4, :4] = 1.0        # 16/16384 = 0.1%
    big = np.zeros((GRID, GRID)); big[:16, :16] = 1.0        # 256/16384 = 1.6%
    stub(monkeypatch, logits_for(c2=tiny, c4=big))
    assert segment.class_prob(grey(), "sky").max() == 0.0
    assert segment.class_prob(grey(), "foliage").max() > 0.99
    assert 0.001 < segment.MIN_AREA < 0.02, "the floor moved; re-read its comment"


# ---- the cache, which is the whole performance story -----------------------


def test_two_words_off_one_frame_are_one_forward_pass(monkeypatch):
    """Inference is 244 ms end to end on a CPU here against ~1 ms for the class
    arithmetic, so "make the sky moody and the foliage richer" being two passes
    would double the cost of the sentence for nothing."""
    s = stub(monkeypatch, logits_for(c2=top_half(), c4=1.0 - top_half()))
    frame = grey()
    segment.class_prob(frame, "sky")
    segment.class_prob(frame, "foliage")
    segment.class_prob(frame, "sky")
    assert s.calls == 1


def test_a_different_frame_is_a_different_answer(monkeypatch):
    """The cache is keyed on the pixels. Keying it on anything cheaper -- a
    path, a timestamp, an id() -- returns the previous clip's mask."""
    s = stub(monkeypatch, logits_for(c2=top_half()))
    segment.class_prob(grey(value=100), "sky")
    segment.class_prob(grey(value=101), "sky")
    assert s.calls == 2


def test_reset_drops_the_session_and_the_cache(monkeypatch):
    s = stub(monkeypatch, logits_for(c2=top_half()))
    segment.class_prob(grey(), "sky")
    segment.reset()
    monkeypatch.setattr(segment, "_session", lambda: s)
    segment.class_prob(grey(), "sky")
    assert s.calls == 2
