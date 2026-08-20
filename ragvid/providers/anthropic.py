"""Anthropic provider. Same contract as Groq, via structured outputs on the same schema."""

from __future__ import annotations

import json
import os

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

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set (put it in .env or the environment)"
                )
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
            raise RuntimeError(f"Anthropic declined the request ({response.stop_details})")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"Anthropic returned no text block (stop_reason={response.stop_reason})")
        return GradeSpec(**json.loads(text)).sanitize()
