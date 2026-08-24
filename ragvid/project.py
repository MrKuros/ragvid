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
from .intent import Intent, describe
from .region import GradeStack, Layer
from .session import Session
from .spec import GradeSpec

# Everything the project writes lives under root/SESSION_DIR, so a UI can show
# one folder per project and delete it to clean up.
def _check_lut(path: str | Path | None) -> str | None:
    """Validate a technical LUT path here, not at render time.

    ffmpeg would otherwise report a missing or malformed .cube from inside a
    filter graph, minutes into an export, as a wall of filter syntax. Checking
    on the way in turns that into one sentence at the moment the file is picked.
    """
    if path is None or (isinstance(path, str) and not path.strip()):
        return None
    lut = Path(path).expanduser()
    if not lut.is_file():
        raise InputError(str(lut), "no such LUT file")
    if lut.suffix.lower() != ".cube":
        raise InputError(str(lut), "not a .cube LUT")
    return str(lut.resolve())


def _resolve_lut(value: str | Path | None, root: Path) -> tuple[str | None, str | None]:
    """A .cube path or a `logspace` format name -> (lut path, format name).

    A generated conversion is derived data, so it lands in the session dir beside
    current.cube rather than in a shared cache: measured 67 ms to bake a 33^3
    cube against a re-probe measured in seconds, so it is re-generated every time
    a format is chosen and there is nothing to invalidate or clean up. Deleting
    .ragvid resets it exactly like every other artifact in there.
    """
    from . import logspace

    name = value.strip().lower() if isinstance(value, str) else ""
    if name in logspace.NAMES:
        return logspace.bake_conversion(name, str(Session.dir(root) / f"log_{name}.cube")), name
    return _check_lut(value), None


def balance_text(stats) -> str:
    """What auto-balance actually does to THIS clip, in its own words.

    Read straight out of the compiler rather than parsed out of a finished
    rationale: an empty Intent compiles to nothing but the balance, so the
    rationale of that compile IS the balance's report. Pure, cheap, and it
    cannot drift from the numbers. "" when there is nothing to correct.
    """
    from .compiler import compile_intent

    said = compile_intent(Intent(), stats, balance=True).rationale
    return said[:1].lower() + said[1:-1] if said else ""


def intent_view(intent: Intent | None) -> dict | None:
    """An Intent as a UI renders it, or None when there is no Intent at all.

    One entry per op carrying both halves a control needs: `text`, the English
    phrase, and the op's own four fields so the UI can send an edited copy back
    without knowing the vocabulary.

    The phrases come from describe() with every amount forced to "moderate",
    which is exactly the phrase with no adverb on the end ("cooled it down",
    not "cooled it down a little"). The magnitude is the control sitting next
    to the sentence; printing it in the sentence as well is the same fact
    twice, and the two disagree for as long as a drag lasts.
    """
    if intent is None:
        return None
    bare = Intent(ops=[op.model_copy(update={"amount": "moderate"}) for op in intent.ops])
    return {
        "strength": intent.strength,
        "ops": [{**op.model_dump(), "text": text}
                for op, text in zip(intent.ops, describe(bare))],
    }


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
               n_frames: int = 10, input_lut: str | Path | None = None) -> "Project":
        """Probe `video` and start a new project.

        `root` defaults to the video's own directory, so opening a clip from a
        file picker does something sensible without asking the user where state
        should live.

        `input_lut` is for log footage — a `.cube` path or a format name; see
        `set_input_lut`. Left unset, the clip's own metadata is asked, which
        answers for almost nothing (`logspace.detect`) and silently does the
        right thing when it does answer.
        """
        from . import logspace
        from .probe import probe_video

        video = Path(video).expanduser()
        if not video.exists():
            raise InputError(str(video), "no such file")
        root = Path(root).expanduser() if root else video.parent
        # Only when the caller said nothing: an explicit choice always wins, and
        # detect() returns None for most clips on purpose -- a wrong technical
        # LUT bakes a wrong contrast curve under every grade that follows.
        if input_lut is None:
            input_lut = logspace.detect(str(video))
        lut, fmt = _resolve_lut(input_lut, root)
        stats = probe_video(str(video), n_frames=n_frames, input_lut=lut)
        session = Session.create(str(video.resolve()), stats, input_lut=lut,
                                 input_format=fmt)
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
    def stack(self) -> GradeStack:
        """The current grade INCLUDING its regional layers (roadmap B1).

        `self.spec` is the base — one correction for every pixel — and is still
        what every consumer that predates regions reads. This is the whole
        answer, and it is what the renderer and the `look.json` sidecar take.
        Flat (`stack.is_flat`) for every grade that names no region, which is
        almost all of them.
        """
        return GradeStack(base=self.spec, layers=list(self.session.region_layers))

    @property
    def layers(self) -> list[Layer]:
        """The current grade's regional layers, oldest first. [] when flat."""
        return list(self.session.region_layers) if self.session.specs else []

    @property
    def intent(self) -> Intent | None:
        """The typed verbs the current grade was compiled from, or None.

        None is the honest and common answer — a photo match, a refine, a
        direct-path provider — and every consumer has to degrade to
        `spec.rationale` rather than render an empty list.
        """
        return self.session.intent if self.session.specs else None

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
        del self.session.intents[index + 1:]
        del self.session.layers[index + 1:]
        self.save()
        return True

    @property
    def input_lut(self) -> str | None:
        """The technical LUT applied before the grade, or None."""
        return self.session.input_lut

    @property
    def input_format(self) -> str | None:
        """Which `logspace` format that LUT was generated from, or None when it
        is a vendor file the user supplied. A UI shows one or the other."""
        return self.session.input_format

    def set_input_lut(self, lut: str | Path | None, n_frames: int = 10) -> str | None:
        """Set (or clear) the log-to-Rec.709 conversion and RE-PROBE.

        `lut` is either a path to a camera vendor's `.cube` or one of
        `logspace.NAMES` ("slog3", "vlog", "clog3", "logc3", "nlog"), in which
        case ragvid bakes the transform itself. Naming the format is the path
        most people can actually take: they know what they shot, they rarely
        have the vendor's file.

        Re-probing is not an optimisation to skip. Every number the model is
        given describes the image the grade will land on, and a conversion LUT
        changes all of them -- measured on simulated S-Log3, mean 0.34 -> 0.49
        and std 0.24 -> 0.48. Keeping the old stats would describe a clip that
        no longer exists and quietly aim the whole grade at the wrong tones.

        The existing grade is deliberately left alone: it is the user's, and
        silently rewriting it is worse than letting them see it applied to the
        converted image and adjust.
        """
        from .probe import probe_video

        path, fmt = _resolve_lut(lut, self.root)
        if path == self.session.input_lut:
            return path
        self.session.input_lut = path
        self.session.input_format = fmt
        self.session.stats = probe_video(self.source, n_frames=n_frames, input_lut=path)
        # Saved here, unlike the old version: the stats this just replaced are
        # cached in session.json, so not writing them left a reopened project
        # measuring the log image while rendering the converted one.
        self.save()
        return path

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
        """Ask the LLM for a grade matching `vibe`, calibrated to this clip.

        Returns the spec, as every planning method here does. The Intent that
        produced it (None on the direct path) is kept beside it in the session
        and read back through `self.intent` — it is what "what it did" is drawn
        from, and what `set_intent` re-compiles when one item is adjusted.
        """
        from .compiler import compile_stack
        from .vibe import plan_vibe

        spec, intent = plan_vibe(vibe, self.stats, provider=provider,
                                 balance=self.session.auto_balance)
        # plan_vibe returns the base spec, which is all a flat grade has. When
        # the model named a region the layers are in the compile it already ran,
        # so re-run it: compile_intent IS compile_stack().base, the function is
        # pure and takes microseconds, and re-deriving beats widening a return
        # type that three provider paths and vibe.py's direct path all share.
        layers = compile_stack(intent, self.stats,
                               balance=self.session.auto_balance).layers if intent else []
        return self._push(spec, label=vibe, intent=intent, layers=layers)

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
        """Adjust the current grade in words. Uses cached stats, never re-probes.

        TWO PATHS, picked the way plan_vibe picks: a capability test, not a
        preference. With an Intent behind the current grade AND an endpoint that
        can constrain decoding, the model edits the VERB LIST (refine_intent) and
        set_intent re-compiles it. Regions survive that for free -- a region is
        the op's `target`, so copying the op copies where it applies -- and so
        does everything else the user accepted, because the edited list still
        names it. "A stop brighter" and "half strength" are edits to an op's
        amount and to intent.strength, which is roadmap B7.

        Otherwise it falls back to refine_spec, which hands the model 44 numbers
        and takes 44 back. That FLATTENS: 44 numbers can only describe the whole
        frame, so regional layers are dropped exactly as in set_spec. It is what
        a photo match, a hand-edited spec and a non-schema endpoint get, and on
        those the alternative is no refinement at all.

        The provider is resolved here rather than inside refine_*, because the
        routing question is `schema_enforced` and only an instance answers it.
        Resolved ONLY when there is an Intent to route: with none, the answer is
        refine_spec whatever the endpoint can do, and constructing a provider to
        learn that would make the no-Intent path fail on a missing key it never
        needed.
        """
        from .refine import refine_intent, refine_spec

        intent = self.intent
        if intent is not None:
            if provider is None:
                from .providers import get_provider

                provider = get_provider()
            if getattr(provider, "schema_enforced", False):
                return self.set_intent(
                    refine_intent(intent, instruction, self.stats, provider=provider),
                    label=instruction,
                )
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

        No Intent goes with it, ever. A spec that arrived as 44 numbers is not
        described by any list of verbs, and carrying the previous one forward
        would leave sentences on screen claiming things the numbers no longer
        do. Editing a grade that HAS an Intent goes through set_intent instead.

        No regional layers either, and for the same reason: 44 numbers describe
        the whole frame, so a spec that arrived this way IS the whole grade.
        Carrying layers forward would keep applying a correction to a corner of
        an image nothing in the incoming spec knows about. set_intent is the
        path that keeps them, because verbs are what regions attach to.
        """
        return self._push(spec.sanitize(), label=label)

    def set_intent(self, intent: Intent, label: str = "adjusted") -> GradeSpec:
        """Re-compile the grade from edited verbs — the per-item strength path.

        Deliberately not set_spec(): the person adjusted "cooled it down", not
        `temperature`. Re-compiling re-derives every number from this clip's
        measurements, so an item that did not move lands on exactly the value it
        had (compile_intent is pure) and the one that did moves along its own
        axis rather than along whichever spec field happened to carry it.
        """
        from .compiler import compile_stack

        stack = compile_stack(intent, self.stats, balance=self.session.auto_balance)
        return self._push(stack.base, label=label, intent=intent, layers=stack.layers)

    @property
    def auto_balance(self) -> bool:
        """Whether the clip is neutralised from its measurements before the look."""
        return self.session.auto_balance

    def set_auto_balance(self, on: bool) -> bool:
        """Turn auto-balance on or off. True if it changed.

        Re-compiles the current grade when there is an Intent to re-compile it
        from, so the toggle is something you SEE rather than a preference that
        takes effect on the next prompt. There is nothing to re-compile for a
        spec that came from anywhere else, and it is left alone -- turning the
        setting off cannot undo a balance that is already baked into 44 numbers
        nobody can decompose.
        """
        if on == self.session.auto_balance:
            return False
        self.session.auto_balance = on
        intent = self.intent
        if intent is not None:
            self.set_intent(intent, label="balance on" if on else "balance off")
        else:
            self.save()
        return True

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
        self.session.intents.clear()
        self.session.layers.clear()
        self.save()
        return True

    def undo(self) -> bool:
        """Step back one grade. False if already at the first."""
        if not self.session.pop():
            return False
        self.save()
        return True

    def _push(self, spec: GradeSpec, label: str = "", intent: Intent | None = None,
              layers: list[Layer] | None = None) -> GradeSpec:
        """The one funnel every grade lands through. Guarded here for that reason.

        A semantic layer needs the local segmentation model, and until this
        guard existed the failure landed at RENDER time -- after the grade was
        already on the history. The session then held a look that could not be
        drawn, and the only way out was guessing at undo. Refusing before the
        push leaves the session exactly as it was, still rendering.

        `segment.require()` rather than `is_ready()` because the two causes
        need different fixes and only that function knows which one applies;
        rebuilding its SegmentUnavailable here would duplicate both hints.
        """
        if layers and GradeStack(base=spec, layers=layers).needs_frame:
            from . import segment

            segment.require()
        self.session.push(spec, label, intent, layers)
        self.save()
        return spec

    # ---- artifacts --------------------------------------------------------

    def bake(self, path: str | Path | None = None, size: int | None = None) -> Path:
        """Write the current grade as a .cube LUT.

        `size=None` means "let bake_cube choose", which is what makes the 65^3
        escalation for hue qualifiers reachable at all: this method is the ONLY
        route the app takes to a cube (preview, frame, export, /media/cube), so
        hard-coding 33 here left every shipping path on the coarse grid while
        the tests -- which call bake_cube directly -- kept passing.
        """
        from .lut import bake_cube

        out = Path(path) if path else self.cube_path
        out.parent.mkdir(parents=True, exist_ok=True)
        bake_cube(self.spec, str(out), size=size)
        return out

    def bake_layers(self, dir: str | Path | None = None) -> list[tuple[str, str]]:
        """Bake the regional layers into [(cube, mask PNG)] for render.py.

        One .cube per layer plus one mask, because a region is spatial and a
        .cube has no way to hold it (docs/ARCHITECTURE.md: the same seam
        `effects` already sits on, one file further along). [] for a flat grade,
        which is what keeps every existing render byte-identical.

        The mask is written at the SOURCE's measured resolution, not the
        preview's: every render path here decodes full frames and only crops
        afterwards, so one mask fits the still, the contact sheet and the export
        — which is the cheapest possible way to guarantee they composite the
        same. `dir` defaults to the session dir; export passes its private temp
        dir for the same reason it bakes a private cube there.
        """
        from .lut import bake_cube

        out_dir = Path(dir) if dir else self.state_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # A semantic region's mask comes from the picture, so one frame is
        # decoded for the whole stack -- once, because inference is 244 ms.
        frame = self._mask_frame() if self.stack.needs_frame else None
        made = []
        for i, layer in enumerate(self.layers):
            cube = out_dir / f"layer{i}.cube"
            bake_cube(layer.spec, str(cube))
            mask = layer.region.write_png(out_dir / f"layer{i}.png",
                                          self.stats.width, self.stats.height,
                                          frame)
            made.append((str(cube), mask))
        return made

    def _mask_frame(self):
        """One UNGRADED frame from the middle of the clip, for semantic masks.

        Ungraded because segmentation reads the picture the camera captured --
        GradeStack.apply passes the source frame for the same reason. Its own
        path, not `frame_source.png`, so a compare render is not clobbered.
        """
        import numpy as np
        from PIL import Image

        png = self.frame(at=self.duration / 2.0, graded=False,
                         path=self.state_dir / "mask_frame.png")
        return np.asarray(Image.open(png).convert("RGB"))

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
        effects = self.spec.render_effects() if graded else None
        out = Path(path) if path else (
            self.preview_path if graded else self.state_dir / SOURCE_PREVIEW_NAME
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        render_preview(self.source, cube, str(out), effects, n_frames=n_frames,
                       input_lut=self.input_lut,
                       layers=self.bake_layers() if graded else None)
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
        effects = self.spec.render_effects() if graded else None
        name = "frame.png" if graded else "frame_source.png"
        out = Path(path) if path else self.state_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        render_frame(self.source, cube, str(out), effects, at=at,
                     input_lut=self.input_lut,
                     layers=self.bake_layers() if graded else None)
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

        `spec.effects` and the regional layers both ride along as ffmpeg filter
        chains -- they are spatial, so they are not in the cube and would
        otherwise be silently dropped from the one render that matters. That
        covers the exported *file*; it does not cover the cube once it leaves
        ragvid, which is why two sidecars land beside the video:
        `<stem>.look.json` (the whole STACK, lossless, the only round-trip
        format now that a grade can be spatial) and `<stem>.cdl` (ASC CDL, lossy
        but universal, with everything it cannot carry named in its
        Description).

        Note this makes the *file* safe, not the spec: `self.spec` is read when
        the render starts, so a caller that wants the grade frozen at the moment
        the user pressed Export must snapshot the project itself.
        """
        from .render import render_video
        from .sidecar import write_cdl, write_look

        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ragvid-export-") as tmp:
            cube = self.bake(Path(tmp) / "export.cube")
            render_video(self.source, str(cube), str(out), self.spec.render_effects(),
                         gpu=gpu, progress=progress, input_lut=self.input_lut,
                         layers=self.bake_layers(tmp))
        # with_name, not with_suffix: a multi-dot suffix is not accepted there.
        write_look(self.stack, out.with_name(out.stem + ".look.json"))
        write_cdl(self.stack, out.with_name(out.stem + ".cdl"))
        return out

    # ---- interop ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Everything a UI needs to render its state, JSON-serializable."""
        return {
            "source": self.source,
            "root": str(self.root),
            "input_lut": self.input_lut,
            "input_format": self.input_format,
            "duration": self.duration,
            "spec": self.spec.model_dump() if self.is_planned else None,
            # The regional half of the grade (roadmap B1). A UI that ignores it
            # draws a picture the export will not produce -- and a WebGL preview
            # in particular has to build the same mask this list describes.
            "layers": [l.model_dump() for l in self.layers],
            "intent": intent_view(self.intent),
            "auto_balance": self.auto_balance,
            "balance": balance_text(self.stats),
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
    """Provider names a UI can offer in a dropdown.

    Reads the live catalog rather than a hard-coded pair. This used to return
    ("groq", "anthropic") and kept returning it after nine more providers
    landed, so any UI built on the documented dropdown helper silently offered
    two of eleven.

    Includes the `custom` entry only when RAGVID_BASE_URL actually points
    somewhere, matching what get_provider() will accept.
    """
    from .providers.base import catalog

    return tuple(p.name for p in catalog())
