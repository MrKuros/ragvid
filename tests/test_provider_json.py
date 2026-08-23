"""Provider.plan_json: one schema-constrained object, or a clear failure.

No live API calls anywhere in here — both SDKs are faked at their own boundary,
because the whole point of this method is that the two boundaries differ
(`output_config` on Messages, `response_format` on chat-completions) and the
caller should not have to know which it is talking to.

tests/test_providers.py owns the direct path; this file owns the second method
and the capability flag that routes to it.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragvid.errors import ProviderError, RateLimited
from ragvid.intent import Intent
from ragvid.providers.anthropic import AnthropicProvider
from ragvid.providers.base import CATALOG, custom_info, get_provider
from ragvid.providers.groq import GroqProvider
from ragvid.providers.openai_compat import OpenAICompatProvider, missing_fields

SCHEMA = Intent.llm_json_schema()

# What a good answer looks like: typed verbs, not numbers.
GOOD = {"ops": [{"op": "warmth", "dir": "up", "amount": "moderate", "target": ""}],
        "strength": "moderate"}


# ---- fakes, one per SDK dialect -------------------------------------------


class FakeCompletions:
    """client.chat.completions — scripted, and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("R", (), {"choices": [
            type("C", (), {"message": type("M", (), {"content": outcome})()})()
        ]})()


class FakeChatClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": FakeCompletions(script)})()

    @property
    def calls(self):
        return self.chat.completions.calls


class FakeMessages:
    """client.messages — the Anthropic surface, which returns content blocks."""

    def __init__(self, text, stop_reason="end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = [type("T", (), {"type": "thinking", "thinking": ""})()]
        if self.text is not None:
            blocks.append(type("T", (), {"type": "text", "text": self.text})())
        return type("R", (), {"content": blocks, "stop_reason": self.stop_reason,
                              "stop_details": None})()


class FakeAnthropicClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.messages = FakeMessages(text, stop_reason)

    @property
    def calls(self):
        return self.messages.calls


def _chat(script) -> tuple[OpenAICompatProvider, FakeChatClient]:
    client = FakeChatClient(script)
    return GroqProvider(client=client), client


def _claude(text, stop_reason="end_turn") -> tuple[AnthropicProvider, FakeAnthropicClient]:
    client = FakeAnthropicClient(text, stop_reason)
    return AnthropicProvider(client=client), client


def _rate_limited(reset="0.01s"):
    import openai

    response = httpx.Response(
        429, headers={"x-ratelimit-reset-tokens": reset},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def _bad_request(msg="response_format json_schema is not supported"):
    import openai

    response = httpx.Response(
        400, json={"error": {"message": msg}},
        request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
    )
    return openai.BadRequestError(msg, response=response, body=None)


# ---- the round trip, once per dialect -------------------------------------


def test_an_openai_compatible_endpoint_returns_a_schema_constrained_object():
    p, client = _chat([json.dumps(GOOD)])

    raw = p.plan_json("sys", "usr", SCHEMA)

    assert client.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": SCHEMA},
    }
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    # The contract is the schema, not GradeSpec: what comes back builds an Intent.
    assert missing_fields(SCHEMA, raw) == []
    assert Intent(**raw).ops[0].op == "warmth"


def test_anthropic_returns_a_schema_constrained_object_through_output_config():
    """The mechanism, and the reason this provider was stuck on the worse path:
    the schema rides in `output_config`, not `response_format`."""
    p, client = _claude(json.dumps(GOOD))

    raw = p.plan_json("sys", "usr", SCHEMA)

    sent = client.calls[0]
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "usr"}]
    assert "response_format" not in sent   # the OpenAI dialect is not accepted here
    assert missing_fields(SCHEMA, raw) == []
    assert Intent(**raw).strength == "moderate"


def test_both_dialects_return_the_same_dict_for_the_same_answer():
    """The point of the second protocol method: the caller stops caring which
    SDK is underneath."""
    chat, _ = _chat([json.dumps(GOOD)])
    claude, _ = _claude(json.dumps(GOOD))
    assert chat.plan_json("sys", "usr", SCHEMA) == claude.plan_json("sys", "usr", SCHEMA) == GOOD


# ---- a bad reply fails the way every other bad reply does -----------------


@pytest.mark.parametrize("body, match", [
    ("I'd rather not.", "no JSON object"),
    ("", "empty response"),
    ('{"ops": [{"op": "warmth", "dir": "up"', "no JSON object"),  # truncated
    ('{"strength": "full"}', "required field"),                          # missing ops
])
def test_a_bad_reply_raises_the_same_provider_error_in_both_dialects(body, match):
    """Not a new failure mode: openai_compat._parse's two, reached through the
    one helper both providers now share, so the message a user sees does not
    depend on which SDK carried the answer."""
    for provider, name in ((_chat([body])[0], "groq"), (_claude(body)[0], "anthropic")):
        with pytest.raises(ProviderError, match=match) as info:
            provider.plan_json("sys", "usr", SCHEMA)
        assert info.value.provider == name


def test_a_missing_field_names_the_field_rather_than_defaulting_it():
    """Every field of Intent has a default, so pydantic would turn a reply with
    no `ops` into the identity grade — a model that looks like it did nothing."""
    p, _ = _chat(['{"strength": "full"}'])
    with pytest.raises(ProviderError, match=r"answered without 1 required field\(s\) \(ops\)"):
        p.plan_json("sys", "usr", SCHEMA)


def test_anthropic_surfaces_a_refusal_rather_than_parsing_it():
    p, _ = _claude(None, stop_reason="refusal")
    with pytest.raises(ProviderError, match="declined"):
        p.plan_json("sys", "usr", SCHEMA)


def test_anthropic_surfaces_a_missing_text_block():
    p, _ = _claude(None, stop_reason="max_tokens")
    with pytest.raises(ProviderError, match="no text block"):
        p.plan_json("sys", "usr", SCHEMA)


# ---- the capability flag, which is what the router actually asks ----------


def test_every_catalog_row_agrees_with_the_provider_it_builds(monkeypatch):
    """The routing test in vibe.plan_vibe is `schema_enforced`, so a new catalog
    row whose provider disagrees with its own note is a silently wrong route:
    either a good endpoint stuck on the 43-float path, or a weak one handed an
    Intent it cannot be trusted to fill in."""
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    monkeypatch.delenv("RAGVID_PROVIDER", raising=False)
    monkeypatch.delenv("RAGVID_MODEL", raising=False)

    for name, info in CATALOG.items():
        assert get_provider(name).schema_enforced == (info.structured == "json_schema"), name

    # ...and Anthropic is on the good side of that line, which is the whole
    # reason plan_json exists: it reads mood language best and used to fall
    # through to the worse path only because it speaks a different SDK.
    assert get_provider("anthropic").schema_enforced is True


def test_the_escape_hatch_starts_at_the_top_rung_and_can_come_down(monkeypatch):
    monkeypatch.setenv("RAGVID_BASE_URL", "https://my-endpoint/v1")
    monkeypatch.setenv("RAGVID_MODEL", "some-model")
    assert custom_info().structured == "json_schema"

    p = OpenAICompatProvider("custom", "https://my-endpoint/v1", "some-model",
                             structured="json_schema", client=FakeChatClient([]))
    assert p.schema_enforced is True
    p.structured = "json_object"
    assert p.schema_enforced is False


def test_a_weak_endpoint_refuses_instead_of_coaxing_json_from_the_prompt():
    """The rule that makes the flag worth having: no prompt-only fallback here.
    A schema pasted into the prompt comes back parseable and possibly wrong, and
    a wrong Intent compiles to a plausible grade nobody asked for."""
    client = FakeChatClient([json.dumps(GOOD)])
    p = OpenAICompatProvider("deepseek", "https://api.deepseek.com/v1", "deepseek-chat",
                             structured="json_object", client=client)

    with pytest.raises(ProviderError, match="cannot constrain output to a schema"):
        p.plan_json("sys", "usr", SCHEMA)
    assert client.calls == []   # it never even asked


def test_an_endpoint_that_rejects_the_schema_steps_off_the_intent_path():
    """An optimistic catalogue row (OpenRouter, the escape hatch) meets reality.
    One wasted request, then `schema_enforced` is False and the router sends the
    next grade down the direct path, where the ladder finds its real level."""
    client = FakeChatClient([_bad_request()])
    p = OpenAICompatProvider("custom", "https://my-endpoint/v1", "some-model",
                             structured="json_schema", client=client)

    with pytest.raises(ProviderError, match="refused a strict JSON schema"):
        p.plan_json("sys", "usr", SCHEMA)
    assert p.schema_enforced is False


# ---- shared machinery still applies --------------------------------------


def test_plan_json_gets_the_same_rate_limit_retry_and_fallback_as_plan(monkeypatch):
    """plan_json runs through the same retry loop rather than a second copy of
    it, so Groq's 8000-token-per-minute bucket behaves identically either way."""
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", lambda _: None)
    p, client = _chat([_rate_limited(), _rate_limited(), json.dumps(GOOD)])

    assert p.plan_json("sys", "usr", SCHEMA) == GOOD
    assert [c["model"] for c in client.calls] == [
        p.model, p.model, p.fallback_model,
    ]


def test_plan_json_reports_an_exhausted_rate_limit_as_rate_limited(monkeypatch):
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", lambda _: None)
    p, client = _chat([_rate_limited() for _ in range(4)])

    with pytest.raises(RateLimited) as info:
        p.plan_json("sys", "usr", SCHEMA)
    assert len(client.calls) == 4
    assert info.value.provider == "groq" and info.value.retry_after == 0.01


def test_the_direct_path_still_works_in_both_dialects():
    """plan() shares _ask/_retrying with plan_json now; it must be unchanged."""
    from ragvid.spec import GradeSpec

    spec_json = GradeSpec(saturation=0.8, rationale="Cool.").model_dump_json()
    chat, _ = _chat([spec_json])
    claude, _ = _claude(spec_json)
    assert chat.plan("sys", "usr").saturation == 0.8
    assert claude.plan("sys", "usr").saturation == 0.8


def test_a_key_never_reaches_a_plan_json_error_message():
    p = OpenAICompatProvider("custom", "https://h/v1", "m", "RAGVID_API_KEY",
                             structured="json_object", client=FakeChatClient([]))
    with pytest.raises(ProviderError) as info:
        p.plan_json("sys", "usr", SCHEMA)
    assert "sk-" not in str(info.value) and "RAGVID_API_KEY" not in str(info.value)
