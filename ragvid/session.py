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
    specs: list[GradeSpec] = field(default_factory=list)

    @property
    def spec(self) -> GradeSpec:
        return self.specs[-1]

    def push(self, spec: GradeSpec) -> None:
        self.specs.append(spec)

    def pop(self) -> bool:
        """Step back one spec. False if there is nothing left to step back to."""
        if len(self.specs) <= 1:
            return False
        self.specs.pop()
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
                    "stats": json.loads(self.stats.model_dump_json()),
                    "specs": [json.loads(s.model_dump_json()) for s in self.specs],
                },
                indent=2,
            )
        )

    @classmethod
    def create(cls, source: str, stats: "ClipStats") -> "Session":
        return cls(source=source, stats=stats)

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Session":
        from .probe import ClipStats

        path = cls.path(root)
        if not path.is_file():
            raise SessionNotFound(str(Path(root or Path.cwd())))
        try:
            raw = json.loads(path.read_text())
            specs = [GradeSpec(**s) for s in raw["specs"]]
            if not specs:  # .spec would IndexError later, far from the cause
                raise ValueError("session has no specs")
            return cls(source=raw["source"], stats=ClipStats(**raw["stats"]), specs=specs)
        # ValueError covers JSONDecodeError and pydantic's ValidationError; TypeError
        # covers a session.json whose shape is wrong rather than merely incomplete.
        # Distinguished from "missing" above because the advice differs: a corrupt
        # file is worth reporting, a missing one just means grade something first.
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SessionCorrupt(str(path), str(exc)) from exc
