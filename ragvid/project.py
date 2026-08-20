"""The programmatic API. Everything a front end needs, and nothing about how
that front end presents it.

`Project` owns the whole flow — probe, plan, refine, bake, preview, export — so
a CLI, a desktop app and a web service all drive the same code path instead of
each reimplementing the orchestration. It never prints, never reads argv, never
calls sys.exit, and never assumes the current working directory.

    p = Project.create("clip.mp4", root="~/grades/clip")
    p.plan_from_vibe("gloomy")          # or plan_from_reference("still.png")
    p.refine("less blue")
    p.set_spec(p.spec.model_copy(update={"contrast": 0.4}))   # a slider moved
    p.export("out.mp4", progress=lambda f: bar.set(f))

Long operations take an optional `progress` callable so a UI can show a bar;
everything else is fast enough to call synchronously.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Iterable

from .errors import InputError, NoGrade
from .session import Session
from .spec import GradeSpec

# Everything the project writes lives under root/SESSION_DIR, so a UI can show
# one folder per project and delete it to clean up.
CUBE_NAME = "current.cube"
PREVIEW_NAME = "preview.png"
SOURCE_PREVIEW_NAME = "source.png"

ProgressFn = Callable[[float], None]


class Project:
    """One clip, its grade history, and the artifacts derived from them."""

    def __init__(self, session: Session, root: Path) -> None:
        self.session = session
        self.root = Path(root)

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    def create(cls, video: str | Path, root: str | Path | None = None,
               n_frames: int = 10) -> "Project":
        """Probe `video` and start a new project.

        `root` defaults to the video's own directory, so opening a clip from a
        file picker does something sensible without asking the user where state
        should live.
        """
        from .probe import probe_video

        video = Path(video).expanduser()
        if not video.exists():
            raise InputError(str(video), "no such file")
        root = Path(root).expanduser() if root else video.parent
        stats = probe_video(str(video), n_frames=n_frames)
        session = Session.create(str(video.resolve()), stats)
        return cls(session, root)

    @classmethod
    def open(cls, root: str | Path | None = None) -> "Project":
        """Load an existing project. Raises SessionNotFound if there isn't one."""
        root = Path(root).expanduser() if root else Path.cwd()
        return cls(Session.load(root), root)

    @classmethod
    def exists(cls, root: str | Path | None = None) -> bool:
        """Cheap check so a UI can decide between 'open' and 'new' without
        catching an exception."""
        root = Path(root).expanduser() if root else Path.cwd()
        return Session.path(root).is_file()

    def save(self) -> None:
        self.session.save(self.root)

    # ---- state (read-only views for a UI) ---------------------------------

    @property
    def source(self) -> str:
        return self.session.source

    @property
    def stats(self):
        """Cached ClipStats. Never re-probed — this is what keeps refine fast."""
        return self.session.stats

    @property
    def spec(self) -> GradeSpec:
        """The current grade. Raises NoGrade if nothing has been planned.

        Guarded here rather than in each artifact method because everything that
        needs a spec reads it through this one property.
        """
        if not self.session.specs:
            raise NoGrade()
        return self.session.spec

    @property
    def history(self) -> list[GradeSpec]:
        """Oldest first, current last. A UI can render this as an undo stack."""
        return list(self.session.specs)

    @property
    def can_undo(self) -> bool:
        # Any grade at all can be undone, including the first -- undoing back to
        # the ungraded clip is exactly what someone means by "undo that".
        return len(self.session.specs) > 0

    @property
    def steps(self) -> list[dict]:
        """What the user asked for, oldest first. A UI renders this as history.

        Each entry is what *they* typed, not the model's rationale -- that is
        what someone recognises when scanning back over what they tried.
        """
        return [
            {"index": i,
             "label": self.session.labels[i] if i < len(self.session.labels) else "",
             "rationale": spec.rationale,
             "current": i == len(self.session.specs) - 1}
            for i, spec in enumerate(self.session.specs)
        ]

    def revert_to(self, index: int) -> bool:
        """Drop every step after `index`. -1 goes back to the ungraded clip.

        This is what clicking an entry in the history does; undo is just
        revert_to(len - 2).
        """
        if index < -1 or index >= len(self.session.specs):
            return False
        del self.session.specs[index + 1:]
        del self.session.labels[index + 1:]
        self.save()
        return True

    @property
    def is_planned(self) -> bool:
        return bool(self.session.specs)

    # ---- paths a UI needs to display --------------------------------------

    @property
    def state_dir(self) -> Path:
        return Session.dir(self.root)

    @property
    def cube_path(self) -> Path:
        return self.state_dir / CUBE_NAME

    @property
    def preview_path(self) -> Path:
        return self.state_dir / PREVIEW_NAME

    @property
    def duration(self) -> float:
        """Seconds. The range a scrubber spans."""
        from .render import probe_duration

        return probe_duration(self.source)

    @property
    def source_preview_path(self) -> Path:
        """Ungraded counterpart of `preview_path`, for before/after compare."""
        return self.state_dir / SOURCE_PREVIEW_NAME

    # ---- planning ---------------------------------------------------------

    def plan_from_vibe(self, vibe: str, provider=None) -> GradeSpec:
        """Ask the LLM for a grade matching `vibe`, calibrated to this clip."""
        from .vibe import plan_vibe

        return self._push(plan_vibe(vibe, self.stats, provider=provider), label=vibe)

    def plan_from_reference(self, image: str | Path) -> GradeSpec:
        """Match a reference image. Closed form — no model, no network."""
        from .match import match_reference
        from .probe import probe_image

        image = Path(image).expanduser()
        if not image.exists():
            raise InputError(str(image), "no such file")
        return self._push(match_reference(self.stats, probe_image(str(image))),
                          label=f"photo: {image.name}")

    def refine(self, instruction: str, provider=None) -> GradeSpec:
        """Adjust the current grade in words. Uses cached stats, never re-probes."""
        from .refine import refine_spec

        return self._push(refine_spec(self.spec, instruction, self.stats, provider=provider),
                          label=instruction)

    def set_spec(self, spec: GradeSpec, label: str = "manual adjustment") -> GradeSpec:
        """Set the grade directly — the path a slider or numeric field uses.

        Pushes onto the history like any other edit, so undo works the same for
        a dragged slider as for a refine.

        sanitize() here and not in server.py: this is the funnel every "a spec
        arrived from outside" path goes through (HTTP body, a future desktop UI,
        a hand-edited session), and effects.* in particular is unbounded on the
        way in -- glow=50 is a legal float and a gblur sigma that hangs ffmpeg.
        Sanitizing at the caller would leave every sibling caller unguarded.
        """
        return self._push(spec.sanitize(), label=label)

    def reset(self) -> bool:
        """Discard every grade, back to the ungraded clip. False if there was
        nothing to discard.

        Distinct from undo, which steps back one edit. This is the "start over"
        a user reaches for when the look has wandered somewhere they don't want
        and stepping back one at a time is not worth it.
        """
        if not self.session.specs:
            return False
        self.session.specs.clear()
        self.session.labels.clear()
        self.save()
        return True

    def undo(self) -> bool:
        """Step back one grade. False if already at the first."""
        if not self.session.pop():
            return False
        self.save()
        return True

    def _push(self, spec: GradeSpec, label: str = "") -> GradeSpec:
        self.session.push(spec, label)
        self.save()
        return spec

    # ---- artifacts --------------------------------------------------------

    def bake(self, path: str | Path | None = None, size: int = 33) -> Path:
        """Write the current grade as a .cube LUT."""
        from .lut import bake_cube

        out = Path(path) if path else self.cube_path
        out.parent.mkdir(parents=True, exist_ok=True)
        bake_cube(self.spec, str(out), size=size)
        return out

    def preview(self, n_frames: int = 3, path: str | Path | None = None,
                graded: bool = True) -> Path:
        """Render a contact sheet. Sub-second; safe to call on every keystroke
        of a refine box.

        `graded=False` renders the untouched source through the same framing, so
        a UI can hold-to-compare against an image that differs only by the grade
        — comparing against a differently-sampled frame would be misleading.
        """
        from .render import render_preview

        # graded=False means the untouched source: no LUT and no effects
        # either, or the "before" half of a compare would already be softened,
        # grained and vignetted.
        cube = str(self.bake()) if graded else None  # ungraded needs no spec
        effects = self.spec.effects if graded else None
        out = Path(path) if path else (
            self.preview_path if graded else self.state_dir / SOURCE_PREVIEW_NAME
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        render_preview(self.source, cube, str(out), effects, n_frames=n_frames)
        return out

    def frame(self, at: float = 0.0, graded: bool = True,
              path: str | Path | None = None) -> Path:
        """Render a single frame at `at` seconds — the check-before-you-render
        call, and what a scrubber drives.

        Cheap regardless of clip length (ffmpeg seeks before decoding), so a UI
        can re-render on every drag rather than only on release. Exists so a
        full export is never the way to find out whether a grade works.
        """
        from .render import render_frame

        cube = str(self.bake()) if graded else None
        effects = self.spec.effects if graded else None
        name = "frame.png" if graded else "frame_source.png"
        out = Path(path) if path else self.state_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        render_frame(self.source, cube, str(out), effects, at=at)
        return out

    def export(self, out_path: str | Path, gpu: bool = False,
               progress: ProgressFn | None = None) -> Path:
        """Render the full clip. The slow one — pass `progress` for a bar.

        Bakes to a PRIVATE temporary LUT rather than `cube_path`. An export runs
        for minutes while the rest of the app stays live, and anything that
        renders a frame re-bakes `cube_path` in place — so sharing it means a
        scrubber move or a slider drag mid-export hands ffmpeg a different grade
        and the finished file silently matches neither what was approved nor
        what was asked for. Measured, not theorised: exporting at saturation 2.5
        and then nudging the slider to 0 produced a greyscale file.

        `spec.effects` rides along as an ffmpeg filter chain -- it is spatial, so
        it is not in the cube and would otherwise be silently dropped from the
        one render that matters.

        Note this makes the *file* safe, not the spec: `self.spec` is read when
        the render starts, so a caller that wants the grade frozen at the moment
        the user pressed Export must snapshot the project itself.
        """
        from .render import render_video

        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ragvid-export-") as tmp:
            cube = self.bake(Path(tmp) / "export.cube")
            render_video(self.source, str(cube), str(out), self.spec.effects,
                         gpu=gpu, progress=progress)
        return out

    # ---- interop ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Everything a UI needs to render its state, JSON-serializable."""
        return {
            "source": self.source,
            "root": str(self.root),
            "duration": self.duration,
            "spec": self.spec.model_dump() if self.is_planned else None,
            "history_depth": len(self.session.specs),
            "steps": self.steps,
            "can_undo": self.can_undo,
            "stats": self.stats.model_dump(),
            "cube": str(self.cube_path),
            "preview": str(self.preview_path),
        }

    def __repr__(self) -> str:
        return f"<Project {Path(self.source).name} grades={len(self.session.specs)}>"


def available_providers() -> Iterable[str]:
    """Provider names a UI can offer in a dropdown."""
    return ("groq", "anthropic")
