"""Session state: <root>/.ragvid/session.json.

Holds the source path, the cached ClipStats (so `refine` never re-probes the
video — that is what keeps the refine loop sub-second) and the spec history.
Last spec in the list is the current one.

`root` is explicit everywhere and defaults to the working directory. A CLI wants
cwd; a GUI has several projects open at once and must be able to say which.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import SessionCorrupt, SessionNotFound
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
    specs: list[GradeSpec] = field(default_factory=list)
    # What the user actually asked for, one per spec. The spec's own `rationale`
    # is the model explaining itself; this is the person's own words, which is
    # what they recognise when scanning back through what they tried.
    labels: list[str] = field(default_factory=list)

    @property
    def spec(self) -> GradeSpec:
        return self.specs[-1]

    def push(self, spec: GradeSpec, label: str = "") -> None:
        self.specs.append(spec)
        self.labels.append(label)

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
        return True

    # ---- persistence ------------------------------------------------------

    @staticmethod
    def dir(root: str | Path | None = None) -> Path:
        return Path(root or Path.cwd()) / SESSION_DIR

    @classmethod
    def path(cls, root: str | Path | None = None) -> Path:
        return cls.dir(root) / SESSION_FILE

    def save(self, root: str | Path | None = None) -> None:
        path = self.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "source": self.source,
                    "input_lut": self.input_lut,
                    "stats": json.loads(self.stats.model_dump_json()),
                    "specs": [json.loads(s.model_dump_json()) for s in self.specs],
                    "labels": self.labels,
                },
                indent=2,
            )
        )

    @classmethod
    def create(cls, source: str, stats: "ClipStats",
               input_lut: str | None = None) -> "Session":
        return cls(source=source, stats=stats, input_lut=input_lut)

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
            # .get, not ["input_lut"]: sessions written before log support have
            # no such key and must still open.
            return cls(source=raw["source"], stats=ClipStats(**raw["stats"]),
                       input_lut=raw.get("input_lut") or None,
                       specs=specs, labels=labels[:len(specs)])
        # ValueError covers JSONDecodeError and pydantic's ValidationError; TypeError
        # covers a session.json whose shape is wrong rather than merely incomplete.
        # Distinguished from "missing" above because the advice differs: a corrupt
        # file is worth reporting, a missing one just means grade something first.
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SessionCorrupt(str(path), str(exc)) from exc
