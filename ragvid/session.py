"""Session state: <root>/.ragvid/session.json.

Holds the source path, the cached ClipStats (so `refine` never re-probes the
video — that is what keeps the refine loop sub-second) and the spec history.
Last spec in the list is the current one.

`root` is explicit everywhere and defaults to the working directory. A CLI wants
cwd; a GUI has several projects open at once and must be able to say which.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import SessionCorrupt, SessionNotFound
from .intent import Intent
from .region import Layer
from .spec import GradeSpec

if TYPE_CHECKING:  # ponytail: import at runtime only in load(), so session.py
    from .probe import ClipStats  # never drags numpy/ffmpeg in for a `spec` call.

SESSION_DIR = ".ragvid"
SESSION_FILE = "session.json"

# Kept as an alias so existing callers that caught NoSession still work.
NoSession = SessionNotFound


@dataclass
class Session:
    source: str
    stats: "ClipStats"
    # A camera vendor's log-to-Rec.709 LUT, applied before anything else. Stored
    # as a path rather than copied in: it belongs to the shoot, not the session,
    # and the same file is normally shared by every clip from that camera.
    input_lut: str | None = None
    # Which logspace format that LUT was baked from, when it was ragvid that
    # baked it. None means the file is the user's own and ragvid knows nothing
    # about what it does -- the two are stored separately because only one of
    # them can be re-generated.
    input_format: str | None = None
    specs: list[GradeSpec] = field(default_factory=list)
    # What the user actually asked for, one per spec. The spec's own `rationale`
    # is the model explaining itself; this is the person's own words, which is
    # what they recognise when scanning back through what they tried.
    labels: list[str] = field(default_factory=list)
    # The typed verbs a spec was compiled from, one per spec, parallel to
    # `labels` and handled exactly the same way -- including the same
    # backward-compatibility problem, since sessions on disk predate the key.
    # None is a real value here, not a gap: a photo match, a refine, a moved
    # slider and a direct-path provider all produce a spec that no Intent
    # describes, and pretending otherwise would put sentences on screen that
    # nothing compiled.
    intents: list[Intent | None] = field(default_factory=list)
    # The regional half of each grade (roadmap B1), one list per spec and
    # parallel to it exactly as `labels` and `intents` are. Kept BESIDE the spec
    # rather than inside it because a GradeSpec is the currency of one
    # correction and a region is a second question -- which pixels -- that a
    # per-pixel colour map has no place to answer. Empty is the common case and
    # means today's flat grade; `Project.stack` reassembles the two halves into
    # the region.GradeStack a renderer wants.
    layers: list[list[Layer]] = field(default_factory=list)
    # Neutralise the clip's measured cast and black point before the creative
    # look (roadmap A6). A property of the PROJECT, not of one grade: it is the
    # base every look in this session sits on, so it lives here rather than
    # once per spec. On by default -- compiler.compile_intent defaults it off so
    # that an empty Intent stays the identity grade bit-for-bit, which makes the
    # choice a caller's, and this is the caller.
    auto_balance: bool = True

    @property
    def spec(self) -> GradeSpec:
        return self.specs[-1]

    @property
    def intent(self) -> Intent | None:
        return self.intents[-1] if self.intents else None

    @property
    def region_layers(self) -> list[Layer]:
        """The current grade's regional layers, [] when it is a flat grade."""
        return self.layers[-1] if self.layers else []

    def push(self, spec: GradeSpec, label: str = "", intent: Intent | None = None,
             layers: list[Layer] | None = None) -> None:
        self.specs.append(spec)
        self.labels.append(label)
        self.intents.append(intent)
        self.layers.append(list(layers or []))

    def pop(self) -> bool:
        """Step back one step. False only when there is nothing left.

        Popping the *first* grade is allowed and lands on the ungraded clip.
        The old floor of one spec meant undo did nothing after a first grade --
        exactly when someone is most likely to want it -- and it only existed
        because an empty spec list used to be treated as corruption.
        """
        if not self.specs:
            return False
        self.specs.pop()
        if self.labels:
            self.labels.pop()
        if self.intents:
            self.intents.pop()
        if self.layers:
            self.layers.pop()
        return True

    # ---- persistence ------------------------------------------------------

    @staticmethod
    def dir(root: str | Path | None = None) -> Path:
        return Path(root or Path.cwd()) / SESSION_DIR

    @classmethod
    def path(cls, root: str | Path | None = None) -> Path:
        return cls.dir(root) / SESSION_FILE

    def save(self, root: str | Path | None = None) -> None:
        """Write the session so a reader sees the old one or the new one, never
        a half of either.

        ATOMIC, and it was not: `Path.write_text` opens "w", which TRUNCATES the
        file before json.dumps' result is handed to write(). A crash, a full
        disk or a serialisation error mid-save therefore left a zero-byte or
        half-written session.json -- which `load` can only report as
        SessionCorrupt, with no repair path and no backup. That is somebody's
        whole grade.

        settings.save has done this correctly since it was written, for the file
        holding an API KEY. This is the same two steps for the file holding the
        WORK -- and only those two: settings' 0700/0600 fchmod dance exists
        because that file is a credential, while a session is a path and some
        44-number specs.

        os.getpid() in the temp name for settings' own reason: two `ragvid
        serve` instances on one root, or a serve plus a CLI in one cwd, collide
        -- and today they do it with no temp file at all, so the loser's write
        can interleave with the winner's. os.replace is atomic on POSIX and over
        an existing file on Windows, which the platform.py matrix needs.
        """
        path = self.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{SESSION_FILE}.{os.getpid()}.tmp")
        body = {
            "source": self.source,
            "input_lut": self.input_lut,
            "input_format": self.input_format,
            "stats": json.loads(self.stats.model_dump_json()),
            "specs": [json.loads(s.model_dump_json()) for s in self.specs],
            "labels": self.labels,
            "intents": [i.model_dump() if i else None for i in self.intents],
            "layers": [[json.loads(l.model_dump_json()) for l in ls]
                       for ls in self.layers],
            "auto_balance": self.auto_balance,
        }
        try:
            # Serialise BEFORE the file exists, so a ValidationError leaves the
            # previous session untouched rather than a temp file to clean up.
            tmp.write_text(json.dumps(body, indent=2))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def create(cls, source: str, stats: "ClipStats", input_lut: str | None = None,
               input_format: str | None = None) -> "Session":
        return cls(source=source, stats=stats, input_lut=input_lut,
                   input_format=input_format)

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Session":
        from .probe import ClipStats

        path = cls.path(root)
        if not path.is_file():
            raise SessionNotFound(str(Path(root or Path.cwd())))
        try:
            raw = json.loads(path.read_text())
            # An empty spec list is a legitimate state, not damage: a clip is
            # open and nothing is graded yet, which is what Project.reset()
            # leaves behind. Project.spec raises NoGrade for it, so the old
            # "reject empty" guard would now reject a state the API creates.
            specs = [GradeSpec(**s) for s in raw["specs"]]
            # labels arrived after the first sessions were written; backfill so
            # an older session still loads.
            labels = list(raw.get("labels") or [])
            labels += [""] * (len(specs) - len(labels))
            # Same backfill for the same reason: every session written before
            # the intent path has no `intents` key, and None per spec is the
            # honest answer for grades nobody compiled from verbs.
            intents = list(raw.get("intents") or [])
            intents += [None] * (len(specs) - len(intents))
            intents = [Intent(**i) if i else None for i in intents[:len(specs)]]
            # Same backfill again, for the same reason: every session written
            # before regions has no `layers` key, and an empty list per spec is
            # the honest answer -- that grade WAS flat.
            layers = [[Layer(**l) for l in ls] for ls in (raw.get("layers") or [])]
            layers += [[] for _ in range(len(specs) - len(layers))]
            # .get, not ["input_lut"]: sessions written before log support have
            # no such key and must still open. Same for input_format, which
            # arrived later still -- an older session with a vendor .cube set
            # loads with format None, which is exactly what it means.
            return cls(source=raw["source"], stats=ClipStats(**raw["stats"]),
                       input_lut=raw.get("input_lut") or None,
                       input_format=raw.get("input_format") or None,
                       specs=specs, labels=labels[:len(specs)], intents=intents,
                       layers=layers[:len(specs)],
                       # Missing key -> the default, same as every field added
                       # after the first sessions were written. An older session
                       # opens balanced, which is what a new one would do.
                       auto_balance=bool(raw.get("auto_balance", True)))
        # ValueError covers JSONDecodeError and pydantic's ValidationError; TypeError
        # covers a session.json whose shape is wrong rather than merely incomplete.
        # Distinguished from "missing" above because the advice differs: a corrupt
        # file is worth reporting, a missing one just means grade something first.
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SessionCorrupt(str(path), str(exc)) from exc
