"""Anthropic provider. Separate from the OpenAI-compatible ones on purpose: a
different SDK, and a different structured-output mechanism (`output_config`,
not `response_format`).

It sits at the top rung of the reliability ladder — the schema is enforced, so
every field comes back — which is why it needs none of openai_compat's
step-down logic.
"""

from __future__ import annotations

import json

from ragvid.errors import ProviderError, ProviderNotConfigured
from ragvid.spec import GradeSpec

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 4096


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or DEFAULT_MODEL
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            from ragvid import settings

            key = settings.key("anthropic", "ANTHROPIC_API_KEY")
            if not key:
                raise ProviderNotConfigured("anthropic", "ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def plan(self, system: str, user: str) -> GradeSpec:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            # effort=low: this is ~14 numbers, not a reasoning problem. Thinking stays
            # on (the default on opus-5) so we don't hit the disabled-thinking quirks.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": GradeSpec.llm_json_schema()},
            },
        )
        if response.stop_reason == "refusal":
            raise ProviderError("anthropic", f"declined the request ({response.stop_details})")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ProviderError("anthropic", f"returned no text block (stop_reason={response.stop_reason})")
        return GradeSpec(**json.loads(text)).sanitize()
