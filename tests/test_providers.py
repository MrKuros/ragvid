"""LLM layer: prompt construction, sanitizing, .env loading, Groq retry/fallback.

No live API calls anywhere in here — every provider is faked.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from ragvid.probe import ClipStats
from ragvid.providers.base import DEFAULTS, get_provider, load_env
from ragvid.providers.groq import FALLBACK_MODEL, GroqProvider, parse_reset
from ragvid.refine import refine_spec
from ragvid.spec import RGB, GradeSpec
from ragvid.vibe import plan_vibe

STATS = ClipStats(
    mean=RGB(r=0.2137, g=0.1904, b=0.2510),
    std=RGB(r=0.0821, g=0.0764, b=0.0902),
    saturation=0.1735,
    frames_sampled=10,
    width=640,
    height=360,
    duration=4.0,
)

CANNED = GradeSpec(
    slope=RGB(r=0.95, g=1.0, b=1.05),
    saturation=0.8,
    temperature=-800.0,
    contrast=0.25,
    rationale="Cool and moody.",
)


class FakeProvider:
    """Records the prompts it was handed and returns a canned spec."""

    name = "fake"

    def __init__(self, spec: GradeSpec = CANNED):
        self.spec = spec
        self.system = None
        self.user = None

    def plan(self, system: str, user: str) -> GradeSpec:
        self.system, self.user = system, user
        return self.spec


# ---- system prompt --------------------------------------------------------


def test_system_prompt_states_identity_and_sign_conventions():
    p = FakeProvider()
    plan_vibe("gloomy", STATS, provider=p)
    s = p.system
    # identity values, so the model knows what "unchanged" is
    assert "0.435" in s and "saturation 1.0" in s
    # sign conventions for the two axes that are easy to get backwards
    assert "NEGATIVE = cooler" in s and "POSITIVE = warmer" in s
    assert "NEGATIVE = greener" in s and "POSITIVE = more magenta" in s
    # gamma inversion is the other classic footgun
    assert ">1 DARKENS midtones" in s
    # every parameter is described
    for field in GradeSpec.model_fields:
        assert field in s, f"system prompt never mentions {field}"


# ---- plan_vibe ------------------------------------------------------------


def test_plan_vibe_prompt_carries_the_vibe_and_the_measured_stats():
    p = FakeProvider()
    plan_vibe("rainy tokyo night", STATS, provider=p)
    assert "rainy tokyo night" in p.user
    # calibrated to THIS footage: the measured numbers must actually be present
    assert "0.2137" in p.user and "0.2510" in p.user  # mean r / mean b
    assert "0.0821" in p.user  # std r
    assert "0.1735" in p.user  # saturation
    assert "640x360" in p.user and "10 frames sampled" in p.user


def test_plan_vibe_returns_the_provider_spec():
    assert plan_vibe("gloomy", STATS, provider=FakeProvider()) == CANNED


# ---- refine_spec ----------------------------------------------------------


def test_refine_prompt_includes_current_spec_and_demands_a_full_spec():
    p = FakeProvider()
    refine_spec(CANNED, "less blue, more contrast", STATS, provider=p)

    assert "less blue, more contrast" in p.user
    # the whole current spec, as JSON the model can copy fields out of
    assert json.loads(_json_block(p.user)) == CANNED.model_dump()
    # ...and stats, so refinement stays calibrated to the footage
    assert "0.2137" in p.user

    # the instruction that makes refinement work at all
    assert "FULL spec" in p.system
    assert "not a diff" in p.system
    assert "unchanged" in p.system


def test_refine_system_prompt_extends_the_vibe_one():
    from ragvid.vibe import SYSTEM

    p = FakeProvider()
    refine_spec(CANNED, "warmer", STATS, provider=p)
    assert p.system.startswith(SYSTEM)  # parameter semantics are still in scope


def _json_block(text: str) -> str:
    start = text.index("{")
    return text[start : text.rindex("}") + 1]


# ---- sanitizing insane model output ---------------------------------------

INSANE = GradeSpec.model_construct(
    slope=RGB(r=99.0, g=-5.0, b=1.0),
    offset=RGB(r=12.0, g=-9.0, b=0.0),
    power=RGB(r=0.0, g=1e6, b=1.0),
    saturation=-4.0,
    temperature=99999.0,
    tint=17.0,
    contrast=8.0,
    pivot=-3.0,
    rationale="nonsense",
)


@pytest.mark.parametrize("call", [
    lambda p: plan_vibe("gloomy", STATS, provider=p),
    lambda p: refine_spec(CANNED, "less blue", STATS, provider=p),
])
def test_insane_model_output_is_clamped(call):
    out = call(FakeProvider(INSANE))
    assert out.slope.r == 8.0 and out.slope.g == 0.0
    assert out.offset.r == 1.0 and out.offset.g == -1.0
    assert out.power.r == 0.05 and out.power.g == 8.0  # power=0 would blow up the LUT
    assert out.saturation == 0.0
    assert out.temperature == 6000.0
    assert out.tint == 2.0
    assert out.contrast == 1.0
    assert out.pivot == 0.05
    assert out.rationale == "nonsense"  # prose is passed through, not clamped


def test_a_valid_spec_survives_sanitizing_unchanged():
    assert plan_vibe("gloomy", STATS, provider=FakeProvider()) == CANNED


# ---- .env loading ---------------------------------------------------------


def test_load_env_fills_missing_vars_and_never_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "RAGVID_TEST_NEW=from_file\n"
        'RAGVID_TEST_QUOTED="quoted value"\n'
        "RAGVID_TEST_EXISTING=from_file\n"
        "not_a_pair\n"
    )
    monkeypatch.delenv("RAGVID_TEST_NEW", raising=False)
    monkeypatch.delenv("RAGVID_TEST_QUOTED", raising=False)
    monkeypatch.setenv("RAGVID_TEST_EXISTING", "from_environment")

    load_env(env)

    assert os.environ["RAGVID_TEST_NEW"] == "from_file"
    assert os.environ["RAGVID_TEST_QUOTED"] == "quoted value"
    assert os.environ["RAGVID_TEST_EXISTING"] == "from_environment"


def test_load_env_is_a_noop_when_there_is_no_file(tmp_path):
    load_env(tmp_path / "nope.env")  # must not raise


# ---- get_provider ---------------------------------------------------------


def test_get_provider_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("RAGVID_PROVIDER", raising=False)
    monkeypatch.delenv("RAGVID_MODEL", raising=False)
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    p = get_provider()
    assert p.name == "groq" and p.model == DEFAULTS["groq"]


def test_get_provider_reads_env(monkeypatch):
    monkeypatch.setenv("RAGVID_PROVIDER", "anthropic")
    monkeypatch.delenv("RAGVID_MODEL", raising=False)
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    assert get_provider().model == DEFAULTS["anthropic"] == "claude-opus-5"

    monkeypatch.setenv("RAGVID_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("RAGVID_PROVIDER", "groq")
    assert get_provider().model == "openai/gpt-oss-20b"


def test_get_provider_arguments_win_over_env(monkeypatch):
    monkeypatch.setenv("RAGVID_PROVIDER", "groq")
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    assert get_provider("anthropic", "claude-opus-5").name == "anthropic"


def test_get_provider_rejects_unknown(monkeypatch):
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("gpt5000")


def test_providers_satisfy_the_protocol():
    from ragvid.providers import Provider

    assert isinstance(GroqProvider(), Provider)
    assert isinstance(FakeProvider(), Provider)


# ---- Groq: rate-limit header parsing --------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("7.66s", 7.66),
    ("2m59.56s", 179.56),
    ("1h2m3s", 3723.0),
    ("500ms", 0.5),
    ("", None),
    (None, None),
    ("soon", None),
])
def test_parse_reset(value, expected):
    assert parse_reset(value) == expected


# ---- Groq: request shape, retry, fallback ---------------------------------


class FakeCompletions:
    """Stands in for client.chat.completions; scripted per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": FakeCompletions(script)})()

    @property
    def calls(self):
        return self.chat.completions.calls


def _ok(spec: GradeSpec):
    message = type("M", (), {"content": spec.model_dump_json()})()
    return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def _rate_limited(reset="0.01s"):
    import openai

    response = httpx.Response(
        429,
        headers={"x-ratelimit-reset-tokens": reset},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_groq_sends_the_frozen_strict_schema_and_sanitizes():
    client = FakeClient([_ok(INSANE)])
    out = GroqProvider(client=client).plan("sys", "usr")

    sent = client.calls[0]
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "grade_spec",
            "strict": True,
            "schema": GradeSpec.llm_json_schema(),
        },
    }
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert out.saturation == 0.0 and out.power.r == 0.05  # sanitized on the way out


def test_groq_retries_then_falls_back_to_the_smaller_model(monkeypatch):
    slept = []
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", slept.append)

    client = FakeClient([_rate_limited(), _rate_limited(), _ok(CANNED)])
    out = GroqProvider(model="openai/gpt-oss-120b", client=client).plan("sys", "usr")

    assert out == CANNED
    assert [c["model"] for c in client.calls] == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-120b",
        FALLBACK_MODEL,
    ]
    assert slept == [0.01]  # backoff came from the reset header, once


def test_groq_raises_a_clear_error_when_everything_is_rate_limited(monkeypatch):
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", lambda _: None)
    client = FakeClient([_rate_limited() for _ in range(4)])

    with pytest.raises(RuntimeError, match="8000 tokens/min"):
        GroqProvider(client=client).plan("sys", "usr")
    assert len(client.calls) == 4


def test_groq_does_not_fall_back_to_itself(monkeypatch):
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", lambda _: None)
    client = FakeClient([_rate_limited(), _rate_limited()])
    with pytest.raises(RuntimeError):
        GroqProvider(model=FALLBACK_MODEL, client=client).plan("sys", "usr")
    assert len(client.calls) == 2


def test_groq_client_property_needs_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _ = GroqProvider().client


# ---- Anthropic: request shape --------------------------------------------


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _anthropic_response(spec: GradeSpec, stop_reason="end_turn"):
    blocks = [
        type("T", (), {"type": "thinking", "thinking": ""})(),
        type("T", (), {"type": "text", "text": spec.model_dump_json()})(),
    ]
    return type("R", (), {"content": blocks, "stop_reason": stop_reason, "stop_details": None})()


def test_anthropic_uses_the_same_schema_and_sanitizes():
    from ragvid.providers.anthropic import DEFAULT_MODEL, AnthropicProvider

    client = type("C", (), {"messages": FakeMessages(_anthropic_response(INSANE))})()
    out = AnthropicProvider(client=client).plan("sys", "usr")

    sent = client.messages.calls[0]
    assert sent["model"] == DEFAULT_MODEL == "claude-opus-5"
    assert sent["system"] == "sys"
    assert sent["output_config"]["format"] == {
        "type": "json_schema",
        "schema": GradeSpec.llm_json_schema(),
    }
    assert out.saturation == 0.0  # skipped the empty thinking block, then clamped


def test_anthropic_surfaces_a_refusal():
    from ragvid.providers.anthropic import AnthropicProvider

    response = type("R", (), {"content": [], "stop_reason": "refusal", "stop_details": None})()
    client = type("C", (), {"messages": FakeMessages(response)})()
    with pytest.raises(RuntimeError, match="declined"):
        AnthropicProvider(client=client).plan("sys", "usr")
