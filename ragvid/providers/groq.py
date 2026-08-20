"""Groq provider. OpenAI-compatible endpoint, strict json_schema constrained decoding."""

from __future__ import annotations

import json
import os
import re
import time

from ragvid.errors import ProviderNotConfigured, RateLimited
from ragvid.spec import GradeSpec

BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
ATTEMPTS_PER_MODEL = 2
MAX_SLEEP = 30.0

# ms before m: alternation is first-match, so "500ms" must not parse as 500 minutes.
_DURATION = re.compile(r"([\d.]+)\s*(ms|h|m|s)")


def parse_reset(value: str | None) -> float | None:
    """Groq's x-ratelimit-reset-tokens: '7.66s', '2m59.56s', '1h2m3s', '500ms'."""
    if not value:
        return None
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    parts = _DURATION.findall(value)
    if not parts:
        # bare seconds: RFC-style `retry-after: 22`, which carries no unit.
        try:
            return float(value)
        except ValueError:
            return None
    return sum(float(n) * units[u] for n, u in parts)


class GroqProvider:
    name = "groq"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or DEFAULT_MODEL
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import openai

            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise ProviderNotConfigured("groq", "GROQ_API_KEY")
            # max_retries=0: the retry/backoff policy below is the only one, so the
            # SDK's own retries don't compound our sleeps.
            self._client = openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=0)
        return self._client

    def plan(self, system: str, user: str) -> GradeSpec:
        import openai

        models = [self.model] + ([FALLBACK_MODEL] if self.model != FALLBACK_MODEL else [])
        last: Exception | None = None
        last_wait: float | None = None
        for model in models:
            for attempt in range(ATTEMPTS_PER_MODEL):
                try:
                    return self._call(model, system, user)
                except openai.RateLimitError as exc:
                    last = exc
                    last_wait = _retry_after(exc)
                    if attempt + 1 >= ATTEMPTS_PER_MODEL:
                        break
                    wait = last_wait or 2.0 ** attempt
                    if wait > MAX_SLEEP:
                        # The bucket resets further out than we are willing to
                        # block (a real 8000-TPM 429 reports ~54s). Sleeping
                        # MAX_SLEEP and retrying anyway just burns the wait and
                        # 429s again -- the fallback model has its own bucket,
                        # so go there now.
                        break
                    time.sleep(wait)
        # Every model exhausted. retry_after carries the provider's own reset
        # time when it gave one, so a UI can show a countdown rather than a
        # dead end.
        raise RateLimited("groq", retry_after=last_wait) from last

    def _call(self, model: str, system: str, user: str) -> GradeSpec:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "grade_spec",
                    "strict": True,
                    "schema": GradeSpec.llm_json_schema(),
                },
            },
        )
        return GradeSpec(**json.loads(response.choices[0].message.content)).sanitize()


def _retry_after(exc) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    return parse_reset(headers.get("x-ratelimit-reset-tokens") or headers.get("retry-after"))
