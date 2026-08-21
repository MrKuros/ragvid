"""ragvid command line.

Deliberately thin: every command is a few calls into `ragvid.project.Project`,
which is the same API a GUI would use. Argument parsing, printing and exit codes
live here and nowhere else, so the two front ends can never drift apart.
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

    c = sub.add_parser("config", help="providers and API keys")
    c.add_argument("--use", metavar="PROVIDER", help="make this provider the one grades use")
    c.add_argument("--set-model", metavar="MODEL", dest="set_model",
                   help="model for the chosen provider (empty string restores the default)")
    c.add_argument("--set-key", metavar="PROVIDER", dest="set_key",
                   help="store an API key, read from a prompt or from stdin")
    c.add_argument("--clear-key", metavar="PROVIDER", dest="clear_key",
                   help="forget the stored key for a provider")
    # Deliberately accepted so it can be REFUSED with a reason -- see _cmd_config.
    c.add_argument("--key", help=argparse.SUPPRESS)

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
    project = Project.create(args.video, root=Path.cwd())
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


# ---- config ---------------------------------------------------------------

ARGV_KEY_REFUSAL = (
    "an API key must never be typed as an argument — every user on this machine "
    "can read it out of `ps` while the command runs, and your shell has already "
    "written it to its history file.\n"
    "  Run 'ragvid config --set-key PROVIDER' and paste it at the prompt, or pipe "
    "it in:  cat key.txt | ragvid config --set-key PROVIDER"
)


def _looks_like_a_key(value: str) -> bool:
    return len(value) > 24 or value.startswith(("sk-", "gsk_", "sk_", "xai-"))


def _cmd_config(args) -> int:
    from ragvid import settings
    from ragvid.providers.base import describe, info_for, load_env

    load_env()
    if args.key is not None:
        print(f"ragvid: {ARGV_KEY_REFUSAL}", file=sys.stderr)
        return 2

    changed = False
    if args.use or args.set_model is not None:
        name = (args.use or "").strip().lower()
        if name:
            info_for(name)  # unknown name -> a clear error, before anything is saved
        settings.select(provider=name or None, model=args.set_model)
        changed = True

    for name, action in (("set_key", "set"), ("clear_key", "clear")):
        target = getattr(args, name)
        if not target:
            continue
        target = target.strip().lower()
        if _looks_like_a_key(target):
            print(f"ragvid: {ARGV_KEY_REFUSAL}", file=sys.stderr)
            return 2
        info_for(target)
        if action == "clear":
            settings.clear_key(target)
        else:
            settings.set_key(target, _read_key(target))
        changed = True

    _print_providers(describe())
    if changed:
        print(f"\nsaved to {settings.path()}")
    return 0


def _read_key(provider: str) -> str:
    """From a prompt (never echoed) or from a pipe. Never from argv."""
    import getpass

    if sys.stdin.isatty():
        key = getpass.getpass(f"Paste the API key for {provider} (it will not be shown): ")
    else:
        key = sys.stdin.readline()
    key = key.strip()
    if not key:
        raise RagvidError("no key given — nothing was saved")
    return key


def _print_providers(rows: list[dict]) -> None:
    print("  provider     key                          schema      model")
    for p in rows:
        if not p["needs_key"]:
            key = "not needed"
        elif p["source"] == "settings":
            key = f"saved here {p['hint']}"
        elif p["source"] == "environment":
            key = f"from {p['env_var']} {p['hint']}"
        else:
            key = f"none ({p['env_var']})"
        schema = "enforced" if p["enforces_schema"] else "best effort"
        mark = "->" if p["active"] else "  "
        print(f"{mark} {p['name']:<12} {key:<28} {schema:<11} {p['model']}")
    print("\n  -> is the one in use.  'enforced' means the provider guarantees a "
          "complete\n     grade; 'best effort' providers sometimes leave fields out, "
          "and ragvid\n     tells you instead of guessing.\n"
          "  Set a key with: ragvid config --set-key groq")


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
