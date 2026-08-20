"""Session state: .ragvid/session.json in the current working directory.

Holds the source path, the cached ClipStats (so `refine` never re-probes the
video — that is what keeps the refine loop sub-second) and the spec history.
Last spec in the list is the current one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .spec import GradeSpec

if TYPE_CHECKING:  # ponytail: import at runtime only in load(), so session.py
    from .probe import ClipStats  # never drags numpy/ffmpeg in for a `spec` call.

SESSION_DIR = ".ragvid"
SESSION_FILE = "session.json"


class NoSession(RuntimeError):
    """Raised by Session.load() when there is nothing to load."""


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
    def path() -> Path:
        return Path(SESSION_DIR) / SESSION_FILE

    def save(self) -> None:
        self.path().parent.mkdir(parents=True, exist_ok=True)
        self.path().write_text(
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
    def load(cls) -> "Session":
        from .probe import ClipStats

        try:
            raw = json.loads(cls.path().read_text())
            specs = [GradeSpec(**s) for s in raw["specs"]]
            if not specs:  # .spec would IndexError later, far from the cause
                raise ValueError("session has no specs")
            return cls(source=raw["source"], stats=ClipStats(**raw["stats"]), specs=specs)
        # ValueError covers JSONDecodeError and pydantic's ValidationError; TypeError
        # covers a session.json whose shape is wrong rather than merely incomplete.
        # A corrupt session and a missing one want the same advice: re-run grade.
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise NoSession("no session here — run 'ragvid grade' first") from exc
