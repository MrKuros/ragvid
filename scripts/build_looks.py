#!/usr/bin/env python3
"""Rebuild the looks corpus in looks/ — reproducible, no LLM, no hand-typed numbers.

Every number in every corpus entry is MEASURED, never authored. The pipeline is:

    base still  --ffmpeg look filter-->  reference still
         |                                    |
    probe_image                          probe_image
         |                                    |
         +---- match.match_reference() -------+
                        |
                   GradeSpec  (slope/offset/saturation solved in closed form)

The base still is a 6-frame tile montage of test_files/test.mp4 — real footage
already in the repo, with photographic statistics (mean 0.58/0.51/0.39, sat
0.36), not a synthetic ramp. Both stills come from the same pixels, so the
solved spec isolates the LOOK and nothing about the scene.

What IS hand-written here: a name, mood words, and the ffmpeg filter chain that
stands in for "a still shot on that stock". That is the STIMULUS — the
equivalent of choosing which frame to reference — and it is recorded verbatim in
each entry's `provenance` so any claim in the corpus can be re-derived from
pixels. The GradeSpec itself is never touched by hand.

Stills land in out/ (gitignored, regenerable); only looks/*.json is committed.

Usage:  uv run python scripts/build_looks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ragvid.match import match_reference  # noqa: E402
from ragvid.probe import probe_image  # noqa: E402
from ragvid.spec import LUMA, SPLIT_CROSSOVER, GradeSpec, RGB, _smoothstep  # noqa: E402

SRC = REPO / "test_files" / "test.mp4"
WORK = REPO / "out" / "looks_src"
CORPUS = REPO / "looks"

# 6 frames of the source, tiled into one still: a single probe_image call then
# sees six moments of the clip instead of one possibly-atypical frame.
BASE_VF = "select='not(mod(n,90))',scale=320:-2,tile=3x2"

# name -> (mood words, ffmpeg filter chain applied to the base still).
# The words describe the INTENT; the numbers in the entry are measured from the
# result. If a word and its measured spec ever disagree, the spec is right.
LOOKS: dict[str, tuple[str, str]] = {
    "golden hour": (
        "warm sunny amber golden summer nostalgic soft romantic evening sunset",
        "colortemperature=temperature=4200,eq=saturation=1.12:brightness=0.03",
    ),
    "teal and orange": (
        "cinematic blockbuster action punchy contrast teal orange trailer hollywood",
        "colorbalance=rs=-0.12:bs=0.18:rh=0.18:bh=-0.14,eq=saturation=1.25:contrast=1.15",
    ),
    "bleach bypass": (
        "gritty harsh desaturated washed bleak war documentary raw brutal",
        "eq=saturation=0.35:contrast=1.45,curves=preset=increase_contrast",
    ),
    "moonlight": (
        "cool blue night moody dark nocturnal cold lonely dream",
        "colortemperature=temperature=12000,eq=brightness=-0.09:saturation=0.8:gamma=0.85",
    ),
    "vintage film": (
        "faded retro nostalgic seventies analog grainy old warm milky",
        "curves=preset=vintage",
    ),
    "film noir": (
        "monochrome black white dramatic contrast shadows crime classic stark",
        "eq=saturation=0.05:contrast=1.5,curves=preset=strong_contrast",
    ),
    "pastel dream": (
        "soft airy light pale pastel gentle romantic ethereal bright dreamy",
        "eq=contrast=0.72:brightness=0.1:saturation=0.85,curves=preset=lighter",
    ),
    "cyberpunk neon": (
        "neon magenta purple futuristic night city electric synthwave vivid",
        "colorbalance=rh=0.2:bh=0.25:gs=-0.1,eq=saturation=1.5:contrast=1.2:brightness=-0.04",
    ),
    "sun bleached desert": (
        "hot dusty arid bright harsh sand western daylight faded",
        "colortemperature=temperature=4800,eq=brightness=0.12:contrast=0.85:saturation=0.7",
    ),
    "sickly green": (
        "green eerie sickly institutional uneasy horror clinical toxic",
        "colorbalance=gm=0.18:gh=0.12:rm=-0.08,eq=saturation=0.9:contrast=1.1",
    ),
    "cold steel": (
        "cold clinical sterile blue grey modern corporate hard technical",
        "colortemperature=temperature=9000,eq=saturation=0.6:contrast=1.2",
    ),
    "candlelit interior": (
        "cozy warm intimate amber candlelight indoor low soft quiet",
        "colortemperature=temperature=3400,eq=brightness=-0.06:saturation=1.05:gamma=0.9",
    ),
    "punchy commercial": (
        "vivid bold punchy saturated clean crisp advertising energetic",
        "eq=saturation=1.4:contrast=1.25,curves=preset=medium_contrast",
    ),
    "flat log": (
        "flat neutral ungraded low contrast soft milky raw baseline",
        "eq=contrast=0.6:saturation=0.75:brightness=0.05",
    ),
    "rose gold": (
        "pink rose warm soft romantic blush feminine gentle glow",
        "colorbalance=rm=0.12:rh=0.14:bm=0.06:gm=-0.04,eq=saturation=1.05:brightness=0.04",
    ),
    "underwater": (
        "cyan aqua underwater deep cool submerged blue green murky",
        "colorbalance=bm=0.18:gm=0.12:rm=-0.16,eq=saturation=1.1:brightness=-0.05",
    ),
    "cross process": (
        "experimental lomo skewed chemical indie quirky music video",
        "curves=preset=cross_process",
    ),
    "high key white": (
        "bright white clean airy studio minimal fresh overexposed",
        "eq=brightness=0.18:contrast=0.8:saturation=0.9,curves=preset=lighter",
    ),
}


def ff(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        check=True, capture_output=True,
    )


def slug(name: str) -> str:
    return name.replace(" ", "_")


def pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def fit_tonal_split(spec: GradeSpec, base_px: np.ndarray, ref_px: np.ndarray) -> GradeSpec:
    """Solve the shadow/highlight tint + lift from what the CDL match left over.

    match_reference solves a single global CDL, so on its own it produces a
    corpus in which all four tonal-split fields are 0.0 in every entry — which
    would teach the model that "teal shadows, orange highlights" is not a thing
    the spec can express. That is the opposite of grounding.

    So fit them, still without inventing a number. The residual after the CDL is
    r = ref - spec.apply(base). apply() adds w_s*d_s + w_h*d_h at step 6 with the
    SAME masks used here, and the two masks have DISJOINT support, so the least-
    squares solution decouples per band to d = sum(w*r)/sum(w*w) per channel —
    closed form, no iteration, no fitting library.

    Splitting d into the spec's two knobs is exact, not approximate. apply()
    contributes `t - L(t) + lift`; with lift = L(d) and t = d - L(d) we get
    L(t) = L(d) - L(d) = 0 (LUMA sums to 1), so the contribution is exactly d.

    READ THE RESULT AS A RESIDUAL, NOT AS COLOUR DESIGN. `teal and orange`
    comes out with a WARM shadow_tint, which looks backwards until you notice
    the CDL that precedes it already over-rotates the shadows teal (its solved
    saturation is 1.87) and this term pulls them back. The composite renders the
    reference; the field in isolation does not describe the look. Measured over
    all 18 entries it lowers RMSE against the reference in every single one:
    median 5.84 -> 4.87 code values, best 7.52 -> 4.27 (golden hour).
    """
    got = spec.apply(base_px).reshape(-1, 3)
    resid = ref_px.reshape(-1, 3) - got
    L = (np.clip(got, 0.0, 1.0) @ LUMA)[:, None]
    masks = (
        1.0 - _smoothstep(np.clip(L / SPLIT_CROSSOVER, 0.0, 1.0)),
        _smoothstep(np.clip((L - SPLIT_CROSSOVER) / (1.0 - SPLIT_CROSSOVER), 0.0, 1.0)),
    )
    update = {}
    for w, tint_field, lift_field in zip(masks, ("shadow_tint", "highlight_tint"),
                                         ("shadow_lift", "highlight_lift")):
        denom = float((w * w).sum())
        # A frame with no true shadows (or no highlights) leaves that band
        # unconstrained; leave it at identity rather than dividing by ~0.
        if denom < 1.0:
            continue
        d = (w * resid).sum(axis=0) / denom
        lift = float(d @ LUMA)
        t = d - lift
        update[tint_field] = RGB(r=t[0], g=t[1], b=t[2])
        update[lift_field] = lift
    return spec.model_copy(update=update).sanitize() if update else spec


def fit_error(spec, base_px: np.ndarray, ref_px: np.ndarray) -> dict:
    """How well the solved spec actually reproduces the reference, in 8-bit code
    values. This is the honesty number: match_reference solves a LINEAR CDL, so
    a reference made with a gamma or curve preset cannot be matched exactly and
    the residual belongs in the corpus where a reader can see it."""
    got = spec.apply(base_px).reshape(-1, 3)
    ref = ref_px.reshape(-1, 3)
    return {
        "rmse_cv": float(np.sqrt(((got - ref) ** 2).mean()) * 255.0),
        "mean_cv": float(np.abs(got.mean(0) - ref.mean(0)).max() * 255.0),
        "std_cv": float(np.abs(got.std(0) - ref.std(0)).max() * 255.0),
    }


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing base footage: {SRC}")
    WORK.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(exist_ok=True)

    base = WORK / "_base.png"
    ff(["-i", str(SRC), "-vf", BASE_VF, "-frames:v", "1", str(base)])
    neutral = probe_image(str(base))
    base_px = pixels(base)

    for name, (tags, vf) in LOOKS.items():
        still = WORK / f"{slug(name)}.png"
        ff(["-i", str(base), "-vf", vf, "-frames:v", "1", str(still)])
        ref = probe_image(str(still))
        ref_px = pixels(still)
        cdl = match_reference(neutral, ref)
        spec = fit_tonal_split(cdl, base_px, ref_px)
        err = fit_error(spec, base_px, ref_px)
        err["rmse_cv_cdl_only"] = fit_error(cdl, base_px, ref_px)["rmse_cv"]

        entry = {
            "name": name,
            "tags": tags.split(),
            "provenance": {
                "base": f"{SRC.relative_to(REPO)} -> {BASE_VF}",
                "reference": vf,
                "derived_by": "match.match_reference(probe_image(base), probe_image(reference))",
                "measured_base": {
                    "mean": neutral.mean.model_dump(),
                    "std": neutral.std.model_dump(),
                    "saturation": neutral.saturation,
                },
                "measured_ref": {
                    "mean": ref.mean.model_dump(),
                    "std": ref.std.model_dump(),
                    "saturation": ref.saturation,
                    "dominant_hue": ref.dominant_hue,
                },
                "fit_error": err,
            },
            "spec": spec.model_dump(),
        }
        out = CORPUS / f"{slug(name)}.json"
        out.write_text(json.dumps(entry, indent=2) + "\n")
        s = spec.slope
        print(
            f"{name:22s} slope r{s.r:.3f} g{s.g:.3f} b{s.b:.3f}  sat {spec.saturation:.3f}"
            f"  RMSE {err['rmse_cv_cdl_only']:5.2f} -> {err['rmse_cv']:5.2f} cv"
        )

    print(f"\n{len(LOOKS)} entries -> {CORPUS.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
