"""ragvid command line.

Deliberately thin: every command is a few calls into `ragvid.project.Project`,
which is the same API the GUI uses. Argument parsing, printing and exit codes
live here and nowhere else, so the two front ends cannot drift on grading.

They are not equals on credentials: API keys are entered, changed and cleared
only in the GUI's Settings panel (`ragvid serve`). The CLI reads keys, never
writes them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import RagvidError
from .project import Project


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

    s = sub.add_parser("serve", help="open the local web UI")
    s.add_argument("--port", type=int, default=8765, help="default 8765; the next free port if taken")
    s.add_argument("--root", help="where project state lives (default: the ragvid work dir)")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return globals()[f"_cmd_{args.cmd}"](args)
    except RagvidError as exc:  # known failure: readable message, no traceback
        print(f"ragvid: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # unknown: still readable, but it is a bug
        print(f"ragvid: {exc}", file=sys.stderr)
        return 1


# ---- commands -------------------------------------------------------------
# The CLI keeps its state in the working directory; a GUI passes a real project
# folder instead. That is the only difference between the two front ends.


def _cmd_grade(args) -> int:
    """Grade a clip, ADDING to whatever history is already in this directory.

    It used to be Project.create every time, whose first push saves over
    session.json -- so a second `ragvid grade` in the same folder destroyed the
    first one's history, and re-probed the clip that the cached ClipStats exists
    to avoid. Grading twice is a normal thing to do; losing the first one is not.
    """
    root = Path.cwd()
    project = None
    if Project.exists(root):
        found = Project.open(root)
        # Same clip -> keep its history. A different clip in a directory that
        # already holds a project is a new project, and create() is right.
        if Path(found.source).resolve() == Path(args.video).expanduser().resolve():
            project = found
    if project is None:
        project = Project.create(args.video, root=root)
    if args.ref:
        project.plan_from_reference(args.ref)
    else:
        project.plan_from_vibe(args.vibe, provider=_provider(args))
    return _report(project)


def _cmd_refine(args) -> int:
    project = Project.open()
    project.refine(args.instruction, provider=_provider(args))
    return _report(project)


def _cmd_spec(args) -> int:
    print(Project.open().spec.model_dump_json(indent=2))
    return 0


def _cmd_reset(args) -> int:
    project = Project.open()
    if not project.undo():
        print("nothing to step back to — the clip is ungraded")
        return 0
    if not project.is_planned:
        # Stepped back off the first grade; there is no preview to render.
        print("back to the original — no look applied")
        return 0
    return _report(project)


def _cmd_serve(args) -> int:
    from .server import serve

    serve(port=args.port, root=args.root, open_browser=not args.no_browser)
    return 0


def _cmd_export(args) -> int:
    project = Project.open()
    out = project.export(args.out, gpu=args.gpu, progress=_bar() if sys.stderr.isatty() else None)
    print(out)
    return 0


# ---- presentation ---------------------------------------------------------


def _report(project: Project) -> int:
    preview = project.preview()
    print(project.spec.rationale or "(no rationale)")
    print(f"preview: {preview}")
    return 0


def _bar():
    """Single-line progress bar on stderr, so stdout stays pipeable."""
    width = 28

    def draw(fraction: float) -> None:
        filled = int(fraction * width)
        sys.stderr.write(f"\r  [{'#' * filled}{'.' * (width - filled)}] {fraction * 100:3.0f}%")
        if fraction >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return draw


def _provider(args):
    # ponytail: None lets vibe/refine pick their own default; only build one
    # when the user actually asked for a specific provider or model.
    if not (args.provider or args.model):
        return None
    from .providers.base import get_provider

    return get_provider(args.provider, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
