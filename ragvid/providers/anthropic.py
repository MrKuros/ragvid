"""Anthropic provider. Separate from the OpenAI-compatible ones on purpose: a
different SDK, and a different structured-output mechanism (`output_config`,
not `response_format`).

It sits at the top rung of the reliability ladder — the schema is enforced, so
every field comes back — which is why it needs none of openai_compat's
step-down logic.

THE MECHANISM, because it is easy to get wrong from memory: constrained
decoding on the Messages API is `output_config={"format": {"type":
"json_schema", "schema": ...}}` on messages.create(). The older top-level
`output_format` parameter is deprecated API-wide, and `response_format` is the
OpenAI dialect and is not accepted here at all.
    https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md
"""

from __future__ import annotations

from ragvid.errors import ProviderError, ProviderNotConfigured
from ragvid.providers.openai_compat import parse_json_reply
from ragvid.spec import GradeSpec

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 4096


class AnthropicProvider:
    name = "anthropic"

    # Provider.schema_enforced. Constant, unlike openai_compat's: there is one
    # endpoint behind this SDK and it honours the schema, so there is no rung to
    # step down to and nothing to discover at runtime.
    schema_enforced = True

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
        raw = self._ask(system, user, GradeSpec.llm_json_schema())
        try:
            return GradeSpec(**raw).sanitize()
        except ProviderError:
            raise
        except Exception as exc:  # pydantic ValidationError, TypeError, ...
            raise ProviderError(self.name, f"returned JSON that is not a grade spec: {exc}") from exc

    def plan_json(self, system: str, user: str, schema: dict) -> dict:
        """Provider.plan_json — the intent path (roadmap A1), reachable here at last.

        Nothing about this endpoint made it unfit for typed verbs; it was simply
        that vibe.ask_intent spoke chat-completions and this speaks Messages.
        Same request as plan(), a caller's schema instead of GradeSpec's.
        """
        return self._ask(system, user, schema)

    def _ask(self, system: str, user: str, schema: dict) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            # effort=low: this is a form to fill in, not a reasoning problem. Thinking stays
            # on (the default on opus-5) so we don't hit the disabled-thinking quirks.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if response.stop_reason == "refusal":
            raise ProviderError(self.name, f"declined the request ({response.stop_details})")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ProviderError(
                self.name, f"returned no text block (stop_reason={response.stop_reason})"
            )
        # A max_tokens stop truncates the JSON mid-object even under constrained
        # decoding, so the reply still gets judged -- and judged by the same
        # function every other provider uses, so it fails the same way.
        return parse_json_reply(self.name, text, schema)
