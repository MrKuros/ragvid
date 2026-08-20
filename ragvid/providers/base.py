"""Provider protocol + selection. The only thing a provider ever returns is a GradeSpec."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragvid.errors import ProviderError
from ragvid.spec import GradeSpec

# Repo root .env — ragvid/providers/base.py -> parents[2].
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

DEFAULTS = {"groq": "openai/gpt-oss-120b", "anthropic": "claude-opus-5"}


def load_env(path: Path = ENV_PATH) -> None:
    """Populate os.environ from a .env file, never overriding what's already set.

    ponytail: 8-line parser instead of python-dotenv. Handles KEY=value and #comments;
    no export/multiline/interpolation. Add the dep if we ever need those.
    """
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@runtime_checkable
class Provider(Protocol):
    name: str

    def plan(self, system: str, user: str) -> GradeSpec: ...


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    """Build the configured provider. Reads RAGVID_PROVIDER / RAGVID_MODEL (and .env)."""
    load_env()
    name = (name or os.environ.get("RAGVID_PROVIDER") or "groq").strip().lower()
    model = model or os.environ.get("RAGVID_MODEL") or DEFAULTS.get(name)

    if name == "groq":
        from ragvid.providers.groq import GroqProvider

        return GroqProvider(model=model)
    if name == "anthropic":
        from ragvid.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model)
    raise ProviderError(str(name), "unknown provider; expected 'groq' or 'anthropic'")
