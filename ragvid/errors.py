"""Typed errors.

Every failure a caller can reasonably act on gets its own class, because a GUI
needs to *branch* on failures, not print them: a missing API key opens settings,
a rate limit shows a retry timer, a bad input file reopens the picker. Anything
that only ever gets shown as text can stay a plain RagvidError.

Each class carries the fields a caller needs to build that response, so nothing
has to be recovered by parsing a message string.
"""

from __future__ import annotations


class RagvidError(Exception):
    """Base for everything ragvid raises deliberately.

    Catch this to be sure you are handling a known failure rather than
    swallowing a bug.
    """


# ---- input ----------------------------------------------------------------


class InputError(RagvidError):
    """The source clip or reference image could not be used."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


# ---- session --------------------------------------------------------------


class SessionNotFound(RagvidError):
    """No session at this location — nothing has been graded here yet."""

    def __init__(self, root: str) -> None:
        self.root = root
        # Name the command, not the concept -- the message is most often read by
        # someone who just typed something in a terminal.
        super().__init__(f"no session in {root} — run 'ragvid grade' first")


class NoGrade(RagvidError):
    """The project exists but nothing has been planned yet.

    Reachable from a GUI in a way it never is from the CLI: open a clip, then
    press Export before pressing Grade. Typed so that button can be disabled or
    the failure explained, rather than surfacing an IndexError from the guts.
    """

    def __init__(self) -> None:
        super().__init__("no grade yet — plan one with a vibe or a reference first")


class SessionCorrupt(RagvidError):
    """A session file exists but could not be read."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"session at {path} is unreadable ({reason})")


# ---- rendering ------------------------------------------------------------


class FFmpegNotFound(RagvidError):
    """ffmpeg or ffprobe is not installed, or not visible to this process.

    Distinct from FFmpegError: nothing ran, so there is no stderr to show and no
    bug to report — the user has to install something. Common enough on macOS,
    where an app launched from Finder never sees Homebrew's /opt/homebrew/bin,
    that it deserves better than a bare FileNotFoundError.
    """

    def __init__(self, binary: str, env_var: str, hint: str) -> None:
        self.binary = binary
        self.env_var = env_var
        self.hint = hint
        super().__init__(f"{binary} not found — {hint}, or set {env_var} to its full path")


class FFmpegError(RagvidError):
    """ffmpeg exited non-zero. Carries its stderr for display."""

    def __init__(self, returncode: int, args: list[str], stderr: str) -> None:
        self.returncode = returncode
        self.args_used = args
        self.stderr = stderr
        super().__init__(
            f"ffmpeg exited {returncode}\n  args: {' '.join(args)}\n  stderr:\n{stderr.strip()}"
        )


# ---- planning / providers -------------------------------------------------


class ProviderError(RagvidError):
    """The LLM layer failed."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ProviderNotConfigured(ProviderError):
    """No credentials. A GUI should route this to its settings screen."""

    def __init__(self, provider: str, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(
            provider,
            f"no API key — add one in Settings under 'ragvid serve', or set {env_var}",
        )


class RateLimited(ProviderError):
    """Provider refused for quota reasons.

    `retry_after` is seconds when the provider told us, else None — a GUI can
    show a countdown instead of a bare failure.
    """

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        wait = f" — retry in {retry_after:.0f}s" if retry_after else ""
        super().__init__(provider, f"rate limit reached{wait}")
