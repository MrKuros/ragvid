"""ragvid command line. See README for the command table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .lut import bake_cube
from .match import match_reference
from .probe import probe_image, probe_video
from .refine import refine_spec
from .render import render_preview, render_video
from .session import SESSION_DIR, Session
from .vibe import plan_vibe

CUBE = str(Path(SESSION_DIR) / "current.cube")
PREVIEW = str(Path(SESSION_DIR) / "preview.png")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragvid", description="Describe the vibe, get the grade.")
    p.add_argument("--provider", help="LLM provider (default: $RAGVID_PROVIDER or groq)")
    p.add_argument("--model", help="override the provider's default model")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="analyse a clip and plan a grade")
    g.add_argument("video")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--vibe", help="mood word or phrase, e.g. 'gloomy'")
    src.add_argument("--ref", help="reference image to match (offline, no LLM)")

    r = sub.add_parser("refine", help="adjust the current grade in words")
    r.add_argument("instruction")

    sub.add_parser("spec", help="print the current grade spec as JSON")
    sub.add_parser("reset", help="step back to the previous spec")

    e = sub.add_parser("export", help="render the full video")
    e.add_argument("out")
    e.add_argument("--gpu", action="store_true", help="use a hardware encoder if one is available")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return globals()[f"_cmd_{args.cmd}"](args)
    except Exception as exc:  # readable message, not a traceback
        print(f"ragvid: {exc}", file=sys.stderr)
        return 1


# ---- commands -------------------------------------------------------------


def _cmd_grade(args) -> int:
    stats = probe_video(args.video)
    if args.ref:
        spec = match_reference(stats, probe_image(args.ref))
    else:
        spec = plan_vibe(args.vibe, stats, provider=_provider(args))
    session = Session.create(args.video, stats)
    session.push(spec)
    return _apply(session)


def _cmd_refine(args) -> int:
    session = Session.load()
    # cached stats — refine never re-probes the video
    session.push(refine_spec(session.spec, args.instruction, session.stats, provider=_provider(args)))
    return _apply(session)


def _cmd_spec(args) -> int:
    print(Session.load().spec.model_dump_json(indent=2))
    return 0


def _cmd_reset(args) -> int:
    session = Session.load()
    if not session.pop():
        print("already at the first grade — nothing to step back to")
        return 0
    return _apply(session)


def _cmd_export(args) -> int:
    session = Session.load()
    bake_cube(session.spec, CUBE)
    print(render_video(session.source, CUBE, args.out, gpu=args.gpu))
    return 0


# ---- shared ---------------------------------------------------------------


def _apply(session: Session) -> int:
    """Persist, bake the LUT, refresh the preview, say what happened."""
    session.save()
    bake_cube(session.spec, CUBE)
    render_preview(session.source, CUBE, PREVIEW)
    print(session.spec.rationale or "(no rationale)")
    print(f"preview: {PREVIEW}")
    return 0


def _provider(args):
    # ponytail: None lets vibe/refine pick their own default; only build one
    # when the user actually asked for a specific provider or model.
    if not (args.provider or args.model):
        return None
    from .providers.base import get_provider

    return get_provider(args.provider, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
