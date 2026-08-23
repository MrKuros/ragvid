"""Semantic masks — the second kind of answer to "which pixels".

region.py answers it with geometry: "the top", "the middle". That is where a
thing SITS. This file answers it with content: "the sky", "the person". That is
what a thing IS, and no arrangement of ramps and ellipses can express it.

THE ARCHITECTURE DECISION THIS FILE EMBODIES. The standing rule was "no pixels
reach a model, ever". This file hands a frame to a model, so the rule was
restated in the same commit as **"no pixels leave the machine"** — see
docs/ARCHITECTURE.md rule 2, which records what the old rule was and why it
stopped fitting. The restatement is honest only because everything here is
local: an ONNX file on disk, onnxruntime on the CPU, no network at inference
time and no network at all after the one download the user consented to.

CONSENT IS A FUNCTION CALL, NOT A PROMPT. Core modules have no I/O policy
(ARCHITECTURE rule 3), so nothing here prints, asks or downloads on its own.
`class_prob` raises `SegmentUnavailable` until `download_model()` has been
called, and `download_model()` is only ever called by a front end that has
asked. Weights are not bundled: the base wheel stays ~5 MB, the model is
15,142,812 bytes, and `pip install ragvid[masks]` adds onnxruntime (~64 MB
installed) for the people who want it.

THE MODEL: SegFormer-B0 fine-tuned on ADE20K, exported to ONNX
(`optimum/segformer-b0-finetuned-ade-512-512`). B0 is the smallest of the
family and the only one that fits the "one more optional download" budget.
150 classes, of which five are worth a vocabulary word — see CLASSES.

WHAT THIS RETURNS, AND WHY IT IS NOT A MASK. `class_prob` returns the model's
own 128x128 probability grid, not a frame-sized mask. Sizing, feathering and
inversion are the mask contract and belong to region.Region, which already owns
them for the geometric shapes. Splitting it here means there is exactly ONE
place that turns a field in [0,1] into a mask, which is the same argument
region.py's docstring makes about the geometry.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from .errors import RagvidError
from .platform import data_dir

# ---- the model -------------------------------------------------------------

MODEL_URL = (
    "https://huggingface.co/optimum/segformer-b0-finetuned-ade-512-512"
    "/resolve/main/model.onnx"
)
MODEL_BYTES = 15_142_812
MODEL_SHA256 = "3a89102115fe3c16230502437b894844ba50cde6f7c800f9884e87c360bcbfc9"

# Preprocessing, read from the model's OWN preprocessor_config.json rather than
# assumed (SegformerImageProcessor, resample=2 i.e. PIL bilinear):
#   size {height: 512, width: 512}, do_resize true
#   rescale_factor 1/255, do_normalize true
#   image_mean [0.485, 0.456, 0.406], image_std [0.229, 0.224, 0.225]
# i.e. ImageNet statistics on 0..1 RGB, NCHW float32.
#
# `do_reduce_labels: true` is about TRAINING TARGETS, not about the output: it
# drops ADE20K's class 0 ("other") from the ground truth, which is why the
# model's 150 output channels are ADE20K classes 1..150 and why config.json's
# id2label is 0-based over them. The indices in CLASSES are id2label's, taken
# from config.json directly — this is the off-by-one the model invites.
_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The vocabulary a sentence can reach, and the config.json id2label entries each
# word covers. A word maps to SEVERAL classes whenever English is coarser than
# ADE20K, which is most of the time — "the foliage" is not one class and a
# person saying it does not mean the model's split between a tree and the grass
# under it. Summing is the correct combination: the 150 channels are one
# softmax, so they are mutually exclusive events and P(foliage) = sum P(class).
#
# Deliberately NOT here: `skin`. It is already an intent target, resolved as a
# hue family by the qualifier bands, and a hue qualifier follows a face through
# a cut for free where a per-frame matte does not (see the per-frame note at the
# bottom of this file). ADE20K has no skin class either.
CLASSES: dict[str, tuple[int, ...]] = {
    "sky": (2,),                                    # sky
    "foliage": (4, 9, 17, 66, 72),                  # tree, grass, plant, flower, palm
    "person": (12,),                                # person
    "water": (21, 26, 60, 113, 128),                # water, sea, river, waterfall, lake
    "buildings": (1, 25, 48, 84),                   # building, house, skyscraper, tower
}

# Below this fraction of the frame, a detection is not a region. Measured
# justification rather than a guess: region.Region's default feather for a
# semantic mask is a box blur of radius 2% of the short side, so a blob smaller
# than about (4% of the short side)^2 ~= 0.16% of the frame is already erased by
# the feather. The floor sits just above that, so it removes exactly the cases
# where the alternative was a grade applied to a smear of nothing.
MIN_AREA = 0.005

# Where a class stops being a maybe and starts being a region. 0.5 of a single
# softmax is exactly "more likely this than everything else put together", i.e.
# argmax generalised to a word covering several classes. It lives here rather
# than in region.py because it is a property of the classifier, and both files
# need the same number: region.py gates the mask on it and MIN_AREA is measured
# against it.
DECIDED = 0.5


class SegmentUnavailable(RagvidError):
    """A semantic region was asked for and no local model can answer it.

    Two distinguishable causes, because the fix differs and a UI has to say
    which: `needs_install` means the optional extra is not installed, otherwise
    the weights have simply never been downloaded and `download_model()` is the
    consent gate that fixes it. Both carry a `hint` that names the exact next
    action, so nothing has to be recovered by parsing the message.
    """

    def __init__(self, reason: str, needs_install: bool, hint: str) -> None:
        self.reason = reason
        self.needs_install = needs_install
        self.hint = hint
        super().__init__(f"{reason} — {hint}")


def model_path() -> Path:
    """Where the weights live. Beside every other per-user ragvid artifact.

    Not in the package directory: a wheel is read-only on a system install, and
    a 15 MB download that lands in site-packages is invisible to anyone trying
    to reclaim disk.
    """
    return data_dir() / "models" / "segformer-b0-ade-512.onnx"


def have_runtime() -> bool:
    """True when `pip install ragvid[masks]` has been done."""
    from importlib.util import find_spec

    return find_spec("onnxruntime") is not None


def is_ready() -> bool:
    """True when a semantic region can actually be resolved right now."""
    return have_runtime() and model_path().is_file()


def require() -> None:
    """Raise the typed, actionable failure. The whole degradation path.

    Public because project.py and server.py both need the *typed* answer, not
    the bool `is_ready()` gives: the two causes carry different `needs_install`
    and different hints, and rebuilding that at each caller would duplicate both
    hint strings and let them drift.
    """
    if not have_runtime():
        raise SegmentUnavailable(
            "semantic masks need the optional segmentation runtime",
            needs_install=True,
            hint="run: pip install 'ragvid[masks]'",
        )
    if not model_path().is_file():
        raise SegmentUnavailable(
            f"the segmentation model is not downloaded ({MODEL_BYTES / 1e6:.0f} MB)",
            needs_install=False,
            hint="call ragvid.segment.download_model() to fetch it once, from "
                 "huggingface.co; it then runs entirely offline",
        )


def download_model(progress: Callable[[float], None] | None = None) -> Path:
    """Fetch the weights. THE CONSENT GATE — nothing else here ever downloads.

    Writes to a `.part` and renames, so an interrupted download can never be
    mistaken for a model: a truncated ONNX file fails inside onnxruntime with a
    protobuf parse error, which is the least diagnosable failure available.
    The hash is checked for the same reason and not for security theatre — the
    file is served over TLS; what this catches is a proxy's error page saved
    under the model's name.
    """
    out = model_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        return out
    part = out.with_suffix(".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(MODEL_URL) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or MODEL_BYTES)
        seen = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
            seen += len(chunk)
            if progress:
                progress(min(seen / total, 1.0))
    if digest.hexdigest() != MODEL_SHA256:
        part.unlink(missing_ok=True)
        raise SegmentUnavailable(
            "the downloaded segmentation model did not match its checksum",
            needs_install=False,
            hint="check the network path and retry download_model()",
        )
    part.replace(out)
    return out


_SESSION = None


def _session():
    """The ONNX session, built once. Module-level and not lru_cache'd so that a
    test can replace it, and so `reset()` is one assignment.

    onnxruntime is imported HERE, never at module scope: `ragvid.intent` imports
    this file for the class vocabulary, so a base install with no extra must be
    able to import it and only fail when a semantic mask is actually asked for.
    """
    global _SESSION
    if _SESSION is None:
        require()
        import onnxruntime as ort  # noqa: PLC0415 — deliberate, see above

        _SESSION = ort.InferenceSession(
            str(model_path()), providers=["CPUExecutionProvider"]
        )
    return _SESSION


def reset() -> None:
    """Drop the session and the frame cache. For tests and for a settings change."""
    global _SESSION
    _SESSION = None
    _CACHE.clear()


# ---- inference -------------------------------------------------------------

# digest -> (150, 128, 128) float32 softmax. Inference is 0.17 s on this CPU
# against ~1 ms for everything else in this file, so the cache is the whole
# performance story: "make the sky moody and the foliage richer" is two words
# off one frame and must not be two forward passes.
#
# ponytail: clear-all at 2 entries rather than an LRU. A mask is baked from one
# sampled frame per grade (see the note at the bottom), so the working set is
# one; an LRU here would be a data structure guarding a case that does not
# arise. Swap it for one if masks ever go per-frame.
_CACHE: dict[bytes, np.ndarray] = {}
_CACHE_MAX = 2


def _softmax_probs(rgb: np.ndarray) -> np.ndarray:
    """One forward pass -> (150, 128, 128) class probabilities.

    The 128x128 is the model's own decode-head resolution (stride 4 from the
    512x512 input), not a choice made here. Upsampling logits before the softmax
    would be a different and more expensive answer to the same question, and
    region.py has to resample to the render's size regardless.
    """
    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"expected an (h, w, 3) image, got shape {x.shape}")
    if x.dtype != np.uint8:
        x = (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    key = hashlib.blake2b(x.tobytes(), digest_size=16).digest()
    if (hit := _CACHE.get(key)) is not None:
        return hit

    from PIL import Image  # already a dependency (probe.py reads stills with it)

    # PIL BILINEAR, because preprocessor_config.json says resample=2, which is
    # PIL's own enum value for it. A different filter here is a different model.
    small = Image.fromarray(x).resize((_SIZE, _SIZE), Image.BILINEAR)
    arr = (np.asarray(small, dtype=np.float32) / 255.0 - _MEAN) / _STD
    logits = _session().run(None, {"pixel_values": arr.transpose(2, 0, 1)[None]})[0][0]

    logits = logits - logits.max(axis=0, keepdims=True)  # stable softmax
    e = np.exp(logits, dtype=np.float32)
    p = e / e.sum(axis=0, keepdims=True)

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = p
    return p


def class_prob(rgb: np.ndarray, name: str) -> np.ndarray:
    """P(`name`) per cell of the model's 128x128 grid, float32 in [0, 1].

    `rgb` is an (h, w, 3) image, uint8 or float in [0, 1]. Aspect ratio is not
    preserved — the model was trained on a square 512x512 resize and reproducing
    its preprocessing exactly matters more than the geometry looking right.

    Returns all zeros when the class covers less than MIN_AREA of the frame, so
    a spurious detection compiles to a layer that changes nothing rather than to
    a grade on a handful of pixels.
    """
    if name not in CLASSES:
        raise ValueError(f"{name!r} is not a semantic class; have {sorted(CLASSES)}")
    p = _softmax_probs(rgb)
    out = p[list(CLASSES[name])].sum(axis=0)
    # Area is measured on the DECIDED crossing, not on the probability sum: a
    # class sitting at 0.1 over the whole frame has a large integral and is
    # nowhere.
    return out if (out > DECIDED).mean() >= MIN_AREA else np.zeros_like(out)


# PER-FRAME OR ONCE? Once, from one sampled frame, and the number is why.
# Measured on this CPU: 0.17 s per forward pass at 512x512. A 10-minute clip at
# 24 fps is 14,400 frames = 41 minutes of inference bolted onto a render that
# measures 1.46x realtime, i.e. the segmentation would cost ~4x the entire
# export. That is not a tuning problem, and the ffmpeg side is worse: a moving
# mask means writing one PNG per frame and feeding an image sequence into the
# alphamerge, which is a second mask pipeline to keep in step with this one.
# So the mask is baked once, exactly as a geometric one is, and the honest
# limitation is that a subject leaving frame keeps its grade. Revisit when a
# clip is short enough or a GPU provider is present — not before.
