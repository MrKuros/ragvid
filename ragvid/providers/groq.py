"""Groq, as a name. The code is ragvid.providers.openai_compat, which every
OpenAI-compatible endpoint now shares; Groq is one row of the catalog.

Kept as its own module because Groq is the default and the documented starting
point, so `from ragvid.providers.groq import GroqProvider` should keep working.
"""

from __future__ import annotations

import time  # noqa: F401  -- tests monkeypatch ragvid.providers.groq.time.sleep

from ragvid.providers.base import CATALOG
from ragvid.providers.openai_compat import (  # noqa: F401  -- re-exported
    ATTEMPTS_PER_MODEL,
    MAX_SLEEP,
    OpenAICompatProvider,
    parse_reset,
)

_INFO = CATALOG["groq"]
BASE_URL = _INFO.base_url
DEFAULT_MODEL = _INFO.model
FALLBACK_MODEL = _INFO.fallback


def GroqProvider(model: str | None = None, client=None) -> OpenAICompatProvider:
    """The catalog's Groq row, built."""
    return OpenAICompatProvider(
        name=_INFO.name, base_url=BASE_URL, model=model or DEFAULT_MODEL,
        env_var=_INFO.env_var, fallback_model=FALLBACK_MODEL,
        structured=_INFO.structured, client=client,
    )
