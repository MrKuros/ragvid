"""Settings the user typed into ragvid: API keys, and which provider to use.

One JSON file at `platform.data_dir()/settings.json`, owner-readable only:

    {"provider": "groq", "model": null, "keys": {"groq": "gsk_..."}}

PRECEDENCE — settings.json beats the process environment (and therefore .env).

    settings.json  >  os.environ (which .env has already been folded into)  >  default

That order is deliberate and it is the opposite of what a server would do.
"I pasted my key into the box and nothing happened" is a far worse failure than
"my shell's GROQ_API_KEY was ignored": the first looks like a broken program,
the second only bites someone who set two keys and forgot. A shell variable
still works fine on its own — it is only overridden when this app holds a key
for the same provider, which only happens because someone deliberately typed
one in here.

The file holds credentials, so:
  * the directory is 0700 and the file is 0600, and both are made that way
    BEFORE any secret reaches the disk (see `save`);
  * a key is never returned by an API route, never logged, and never put into
    an exception message — callers get `hint()`, which is "…" plus four digits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .platform import data_dir

FILENAME = "settings.json"


def path() -> Path:
    # Read at call time, never cached: data_dir() reads the environment, which
    # is what makes this testable (and what makes XDG_DATA_HOME work at all).
    return data_dir() / FILENAME


def load() -> dict:
    """The whole settings dict, or {} if there is nothing usable on disk.

    A damaged file is not an error worth crashing over — it means "no settings",
    and the next `save` rewrites it.
    """
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict) -> None:
    """Write the settings atomically, never letting a secret exist world-readable.

    The order matters and is the whole point of this function:
      1. the directory is created 0700 and chmod'd 0700 (mkdir's mode is masked
         by the process umask, chmod is not);
      2. the temp file is opened with os.open(..., O_CREAT|O_WRONLY|O_TRUNC,
         0o600) and immediately fchmod'd 0600 — O_CREAT does NOT change the mode
         of a file that already exists, and the plain open()-then-chmod dance
         leaves a window in which the key is on disk under the umask's mode;
      3. only then is anything written;
      4. os.replace swaps it in, so a reader sees either the old file or the
         new one — never a truncated one.
    """
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    tmp = target.with_name(f".{FILENAME}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)  # the file is still empty here
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---- keys -----------------------------------------------------------------


def key(provider: str, env_var: str | None) -> str | None:
    """The API key for `provider`: settings.json first, then the environment."""
    stored = load().get("keys", {}).get(provider)
    if stored:
        return stored
    return os.environ.get(env_var) if env_var else None


def set_key(provider: str, value: str) -> None:
    value = value.strip()
    if not value:
        raise ValueError("empty key")
    data = load()
    data.setdefault("keys", {})[provider] = value
    save(data)


def clear_key(provider: str) -> None:
    """Remove the key outright — not blank it in place, which would leave the
    old bytes in the file and read back as "configured, empty"."""
    data = load()
    if data.get("keys", {}).pop(provider, None) is not None:
        save(data)


def hint(provider: str, env_var: str | None) -> str | None:
    """A safe echo of the configured key: "…" plus its last four characters.

    The only shape of a key that ever leaves this module.
    """
    value = key(provider, env_var)
    return f"…{value[-4:]}" if value else None


def source(provider: str, env_var: str | None) -> str | None:
    """Where the key came from: "settings", "environment", or None."""
    if load().get("keys", {}).get(provider):
        return "settings"
    if env_var and os.environ.get(env_var):
        return "environment"
    return None


# ---- the active provider --------------------------------------------------


def selected() -> tuple[str | None, str | None]:
    """(provider, model) as chosen in this app, either possibly None."""
    data = load()
    return data.get("provider") or None, data.get("model") or None


def select(provider: str | None = None, model: str | None = None) -> None:
    """Set the active provider and/or model. An empty model clears the override."""
    data = load()
    if provider:
        data["provider"] = provider
    if model is not None:
        data["model"] = model or None
    save(data)


def _selfcheck() -> None:
    """`python -m ragvid.settings` — the security-shaped half: file mode, atomic
    replace, clear-not-blank, and precedence."""
    import stat
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["XDG_DATA_HOME"] = tmp
        os.environ["HOME"] = tmp  # macOS/Windows branches of data_dir()
        os.environ["APPDATA"] = tmp
        set_key("groq", "gsk_secret")
        mode = stat.S_IMODE(path().stat().st_mode)
        assert mode == 0o600, oct(mode)
        assert key("groq", "GROQ_API_KEY") == "gsk_secret"
        assert hint("groq", "GROQ_API_KEY") == "…cret"
        os.environ["GROQ_API_KEY"] = "from_env"
        assert key("groq", "GROQ_API_KEY") == "gsk_secret"  # settings wins
        clear_key("groq")
        assert "gsk_secret" not in path().read_text()
        assert key("groq", "GROQ_API_KEY") == "from_env"  # env still works
        assert not [p for p in path().parent.iterdir() if p.name.endswith(".tmp")]
    print("settings selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
