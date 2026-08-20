#!/usr/bin/env python3
"""Every call a front end needs, in the order a front end would make them.

Runnable end to end without an API key -- the reference-match path is closed
form and never touches the network. Pass --vibe to also exercise the LLM path.

    python examples/api_tour.py
    python examples/api_tour.py --vibe "gloomy"

This doubles as the acceptance test for "is the API actually UI-ready": if a
step here needs something that isn't on Project, the facade is incomplete.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from ragvid import (
    GradeSpec,
    InputError,
    Project,
    ProviderNotConfigured,
    RagvidError,
    RateLimited,
    available_providers,
)

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default=str(REPO / "assets" / "sample.mp4"))
    ap.add_argument("--reference", default=str(REPO / "assets" / "ref_bars.png"))
    ap.add_argument("--vibe", help="also run the LLM path with this mood (needs a key)")
    args = ap.parse_args()

    if not Path(args.clip).exists():
        print(f"no clip at {args.clip} — run `uv run pytest -q` once to generate fixtures,"
              " or pass --clip", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        return tour(args, root=Path(tmp))


def tour(args, root: Path) -> int:
    say("providers a UI could offer", list(available_providers()))

    # ---- open or create ---------------------------------------------------
    # `exists` lets a UI choose between "resume" and "new" without catching.
    say("project already here?", Project.exists(root))
    project = Project.create(args.clip, root=root)
    say("created", repr(project))
    say("measured stats", {
        "mean": [round(v, 3) for v in (project.stats.mean.r, project.stats.mean.g, project.stats.mean.b)],
        "saturation": round(project.stats.saturation, 3),
        "frames_sampled": project.stats.frames_sampled,
    })

    # ---- plan: offline path -----------------------------------------------
    spec = project.plan_from_reference(args.reference)
    say("plan_from_reference", spec.rationale)

    # ---- plan: LLM path (optional) ----------------------------------------
    if args.vibe:
        try:
            say("plan_from_vibe", project.plan_from_vibe(args.vibe).rationale)
            say("refine", project.refine("warmer and brighter").rationale)
        except ProviderNotConfigured as exc:
            say("skipped LLM path", f"{exc.env_var} not set")
        except RateLimited as exc:
            say("skipped LLM path", f"rate limited, retry in {exc.retry_after or '?'}s")

    # ---- direct edit: the slider path -------------------------------------
    # A GUI mutating one control does exactly this. It lands in history like
    # any other edit, so undo covers a dragged slider too.
    nudged = project.spec.model_copy(update={"contrast": 0.35})
    project.set_spec(nudged)
    say("set_spec (slider)", f"contrast -> {project.spec.contrast}")

    # ---- history / undo ---------------------------------------------------
    say("history depth", len(project.history))
    say("can_undo", project.can_undo)
    project.undo()
    say("after undo, contrast", project.spec.contrast)

    # ---- artifacts --------------------------------------------------------
    say("baked LUT", project.bake())
    say("preview", project.preview())

    out = Path(root) / "export.mp4"
    project.export(out, progress=bar())
    say("exported", f"{out} ({out.stat().st_size // 1024} KB)")

    # ---- what a view would render -----------------------------------------
    state = project.to_dict()
    state["stats"] = "<ClipStats>"      # too long to print in full
    state["spec"] = "<GradeSpec>"
    say("to_dict", json.dumps(state, indent=2))

    # ---- reopening --------------------------------------------------------
    reopened = Project.open(root)
    say("reopened", f"{reopened!r}, same spec: {reopened.spec == project.spec}")

    # ---- errors are typed -------------------------------------------------
    try:
        Project.create("/nope/missing.mp4", root=root)
    except InputError as exc:
        say("InputError", f"path={exc.path!r} reason={exc.reason!r}")

    try:
        Project.open(Path(root) / "empty")
    except RagvidError as exc:
        say(type(exc).__name__, str(exc))

    return 0


def bar():
    """The progress callback shape a UI would wire to a progress widget."""
    def draw(fraction: float) -> None:
        filled = int(fraction * 24)
        end = "\n" if fraction >= 1.0 else ""
        print(f"\r  export [{'#' * filled}{'.' * (24 - filled)}] {fraction * 100:3.0f}%",
              end=end, flush=True)
    return draw


def say(label: str, value) -> None:
    print(f"\n\033[36m{label}\033[0m\n  {value}")


if __name__ == "__main__":
    raise SystemExit(main())
