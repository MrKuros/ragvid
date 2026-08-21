"""LLM layer: prompt construction, sanitizing, .env loading, Groq retry/fallback.

No live API calls anywhere in here — every provider is faked.
"""

from __future__ import annotations

import json
import os
import re

import httpx
import pytest

from ragvid.errors import ProviderError, ProviderNotConfigured, RateLimited
from ragvid.probe import ClipStats
from ragvid.providers.base import CATALOG, DEFAULTS, describe, get_provider, load_env
from ragvid.providers.groq import FALLBACK_MODEL, GroqProvider, parse_reset
from ragvid.refine import refine_spec
from ragvid.spec import RGB, GradeSpec
from ragvid.vibe import SYSTEM, format_stats, plan_vibe

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


def test_system_prompt_couples_a_slope_push_to_highlight_rolloff():
    # The measured defect this exists to fix: slope 1.3 pins 23% of a ramp at pure
    # white. The model can only avoid it if the prompt links the two fields.
    assert "IF YOU RAISE slope OR exposure" in SYSTEM
    assert "highlight_rolloff" in SYSTEM


# Ranges in the prompt are written `field [low..high]`, one uniform form, so they can be
# parsed back out and checked against the clamps that actually run.
RANGE_RE = re.compile(r"\b([a-z_]+(?:\.[a-z_]+)?) \[(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)\]")


def _sanitized(path: str, value: float) -> float:
    """`value` put into `path`, run through sanitize(), read back."""
    d = GradeSpec.identity().model_dump()
    head, _, sub = path.partition(".")
    node = d[head]
    if isinstance(node, dict):  # RGB, HueBand or EffectSpec
        keys = [sub] if sub else list(node)
        for k in keys:
            node[k] = value
        return GradeSpec(**d).sanitize().model_dump()[head][keys[0]]
    d[head] = value
    return GradeSpec(**d).sanitize().model_dump()[head]


def test_every_range_quoted_in_the_prompt_survives_sanitize():
    # The prompt used to promise slope 0.7-1.4 while sanitize() allowed 0-8. A range the
    # clamp does not honour is a lie to the model; a range outside it is unreachable.
    found = RANGE_RE.findall(SYSTEM)
    assert len(found) >= 20, "the parameter list stopped quoting ranges"
    for path, lo, hi in found:
        lo, hi = float(lo), float(hi)
        assert lo < hi, path
        assert _sanitized(path, lo) == pytest.approx(lo), f"{path} low end is clamped away"
        assert _sanitized(path, hi) == pytest.approx(hi), f"{path} high end is clamped away"


def test_every_numeric_field_has_a_quoted_range():
    from ragvid.spec import HUE_FIELDS, EffectSpec

    found = RANGE_RE.findall(SYSTEM)
    paths = {p for p, _, _ in found}
    # The other five bands share hue_red's two ranges, and the prompt says so.
    covered = {p.split(".")[0] for p in paths} | set(HUE_FIELDS[1:]) | {"rationale"}
    assert not set(GradeSpec.model_fields) - covered
    assert {f"effects.{k}" for k in EffectSpec.model_fields} <= paths


# ---- format_stats ---------------------------------------------------------


def test_format_stats_carries_the_new_measurements():
    text = format_stats(STATS.model_copy(update={
        "p1": RGB.of(0.0142), "p99": RGB.of(0.8120),
        "dominant_hue": 214.0, "clipped_high": 0.031, "crushed_low": 0.004,
        "frame_variance": 0.0413,
    }))
    assert "0.0142" in text and "0.8120" in text
    assert "214 deg" in text
    assert "3.1%" in text and "0.4%" in text
    assert "0.0413" in text


def test_stats_notes_turn_a_measurement_into_an_instruction():
    blown = format_stats(STATS.model_copy(update={
        "clipped_high": 0.06, "p99": RGB.of(0.99), "frame_variance": 0.04,
    }))
    assert "no headroom" in blown and "highlight_rolloff 0.2-0.4" in blown

    crushed = format_stats(STATS.model_copy(update={
        "crushed_low": 0.09, "p99": RGB.of(0.99), "frame_variance": 0.04,
    }))
    assert "negative offset" in crushed

    # A clip with nothing wrong pays no tokens for advice it does not need.
    clean = format_stats(STATS.model_copy(update={
        "p99": RGB.of(0.94), "frame_variance": 0.04,
    }))
    assert "What these measurements mean" not in clean


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
    with pytest.raises(ProviderError, match="unknown provider"):
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

    with pytest.raises(RateLimited) as info:
        GroqProvider(client=client).plan("sys", "usr")
    assert len(client.calls) == 4
    # Typed, and carrying what a UI needs: whose limit, and when it lifts.
    assert info.value.provider == "groq" and info.value.retry_after == 0.01
    assert "rate limit" in str(info.value)


def test_groq_does_not_fall_back_to_itself(monkeypatch):
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", lambda _: None)
    client = FakeClient([_rate_limited(), _rate_limited()])
    with pytest.raises(RateLimited):
        GroqProvider(model=FALLBACK_MODEL, client=client).plan("sys", "usr")
    assert len(client.calls) == 2


def test_groq_client_property_needs_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured, match="GROQ_API_KEY") as info:
        _ = GroqProvider().client
    assert (info.value.provider, info.value.env_var) == ("groq", "GROQ_API_KEY")


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
    with pytest.raises(ProviderError, match="declined"):
        AnthropicProvider(client=client).plan("sys", "usr")


# ---- every provider, not two ----------------------------------------------
# The catalog, the capability ladder, and the escape hatch. Still no network:
# each test drives a fake client or stops before one is built.


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """settings.json must come from tmp_path, never the developer's real one."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    # ...and never from the repo's .env, which would make "is groq configured?"
    # depend on whose machine the suite is running on.
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    for info in CATALOG.values():
        if info.env_var:
            monkeypatch.delenv(info.env_var, raising=False)


def _fake(script):
    return FakeClient(script)


def _text(body: str):
    """A completion whose content is exactly `body` (prose, fences and all)."""
    message = type("M", (), {"content": body})()
    return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def _bad_request(msg="response_format not supported"):
    import openai

    response = httpx.Response(
        400, json={"error": {"message": msg}},
        request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
    )
    return openai.BadRequestError(msg, response=response, body=None)


def test_the_catalog_covers_the_providers_people_actually_have():
    expected = {"groq", "anthropic", "openai", "moonshot", "deepseek",
                "openrouter", "together", "xai", "mistral", "ollama"}
    assert expected <= set(CATALOG)


def test_every_catalog_row_declares_a_usable_structured_output_level():
    from ragvid.providers.openai_compat import MODES

    for name, info in CATALOG.items():
        assert info.structured in MODES, name
        assert info.model, name
        # An OpenAI-compatible row needs somewhere to send the request; the
        # Anthropic SDK brings its own.
        assert bool(info.base_url) == (info.kind == "openai"), name
        # Anything weaker than strict decoding must SAY so, in its own text.
        if info.structured != "json_schema":
            assert "incomplete" in info.note or "miss fields" in info.note, name


def test_only_ollama_runs_without_a_key():
    assert [n for n, i in CATALOG.items() if i.env_var is None] == ["ollama"]
    assert CATALOG["ollama"].base_url.startswith("http://127.0.0.1")


@pytest.mark.parametrize("name", ["openai", "deepseek", "moonshot", "together",
                                  "xai", "mistral", "openrouter", "ollama"])
def test_each_provider_builds_with_its_own_url_model_and_capability(name, monkeypatch):
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    monkeypatch.delenv("RAGVID_PROVIDER", raising=False)
    monkeypatch.delenv("RAGVID_MODEL", raising=False)
    info = CATALOG[name]
    p = get_provider(name)
    assert (p.name, p.base_url, p.model) == (name, info.base_url, info.model)
    assert (p.env_var, p.structured) == (info.env_var, info.structured)


def test_an_unknown_endpoint_works_through_the_escape_hatch(monkeypatch):
    """RAGVID_BASE_URL, not the catalog, is the real answer to 'all of them'."""
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    monkeypatch.delenv("RAGVID_PROVIDER", raising=False)
    monkeypatch.setenv("RAGVID_BASE_URL", "https://llm.internal/v1")
    monkeypatch.setenv("RAGVID_MODEL", "house-model-1")
    monkeypatch.setenv("RAGVID_API_KEY", "k")

    p = get_provider()                      # no provider named: the hatch wins
    assert (p.name, p.base_url, p.model) == ("custom", "https://llm.internal/v1", "house-model-1")
    assert p.env_var == "RAGVID_API_KEY"
    assert "custom" in [row["name"] for row in describe()]


def test_the_escape_hatch_is_invisible_until_it_is_configured(monkeypatch):
    monkeypatch.delenv("RAGVID_BASE_URL", raising=False)
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    assert "custom" not in [row["name"] for row in describe()]
    with pytest.raises(ProviderError, match="RAGVID_BASE_URL"):
        get_provider("custom")


# ---- the structured-output ladder -----------------------------------------


def test_a_json_object_endpoint_gets_the_schema_in_the_prompt_instead():
    """No constrained decoding, so the schema has to travel as an instruction."""
    from ragvid.providers.openai_compat import OpenAICompatProvider

    client = _fake([_ok(INSANE)])
    p = OpenAICompatProvider("deepseek", "https://x/v1", "m", "K",
                             structured="json_object", client=client)
    out = p.plan("sys", "usr")

    sent = client.calls[0]
    assert sent["response_format"] == {"type": "json_object"}
    system = sent["messages"][0]["content"]
    assert system.startswith("sys")
    assert json.dumps(GradeSpec.llm_json_schema()) in system   # the whole schema
    assert "json" in system.lower()          # several endpoints require the word
    assert out.saturation == 0.0             # and it still goes through sanitize()


def test_a_text_only_endpoint_gets_json_dug_out_of_its_prose():
    from ragvid.providers.openai_compat import OpenAICompatProvider

    fenced = "Sure!\n```json\n" + CANNED.model_dump_json() + "\n```\nHope that helps."
    client = _fake([_text(fenced)])
    p = OpenAICompatProvider("weird", "https://x/v1", "m", "K",
                             structured="text", client=client)
    out = p.plan("sys", "usr")

    assert "response_format" not in client.calls[0]   # asking would 400
    assert out == CANNED


def test_a_short_answer_is_an_error_not_a_half_filled_spec():
    """The failure this whole ladder exists to make visible.

    Pydantic would default the missing fields and hand back a spec that looks
    fine and grades wrong. Naming the fields is far cheaper to diagnose.
    """
    from ragvid.providers.openai_compat import OpenAICompatProvider

    half = json.loads(CANNED.model_dump_json())
    half.pop("saturation")
    half["slope"].pop("b")
    client = _fake([_text(json.dumps(half))])
    p = OpenAICompatProvider("deepseek", "https://x/v1", "m", "K",
                             structured="json_object", client=client)

    with pytest.raises(ProviderError) as info:
        p.plan("sys", "usr")
    assert "saturation" in str(info.value) and "slope.b" in str(info.value)
    assert info.value.provider == "deepseek"


def test_prose_with_no_json_at_all_is_a_typed_error():
    from ragvid.providers.openai_compat import OpenAICompatProvider

    client = _fake([_text("I'd rather not.")])
    p = OpenAICompatProvider("weird", "https://x/v1", "m", "K",
                             structured="text", client=client)
    with pytest.raises(ProviderError, match="no JSON object"):
        p.plan("sys", "usr")


def test_an_endpoint_that_rejects_strict_mode_steps_down_by_itself():
    """The escape hatch points at an unknown endpoint, so the flag can be wrong.
    A 400 on the response_format must degrade, not fail the grade."""
    from ragvid.providers.openai_compat import OpenAICompatProvider

    client = _fake([_bad_request(), _bad_request(), _text(CANNED.model_dump_json())])
    p = OpenAICompatProvider("custom", "https://x/v1", "m", "K",
                             structured="json_schema", client=client)

    assert p.plan("sys", "usr") == CANNED
    formats = [c.get("response_format", {}).get("type") for c in client.calls]
    assert formats == ["json_schema", "json_object", None]
    assert p.structured == "text"   # remembered, so the next grade costs one call


def test_a_provider_never_climbs_above_its_declared_capability():
    from ragvid.providers.openai_compat import OpenAICompatProvider

    client = _fake([_ok(CANNED)])
    OpenAICompatProvider("deepseek", "https://x/v1", "m", "K",
                         structured="json_object", client=client).plan("sys", "usr")
    assert client.calls[0]["response_format"]["type"] == "json_object"


def test_a_bad_request_below_the_last_rung_still_surfaces():
    import openai

    from ragvid.providers.openai_compat import OpenAICompatProvider

    client = _fake([_bad_request("model not found")])
    p = OpenAICompatProvider("custom", "https://x/v1", "m", "K",
                             structured="text", client=client)
    with pytest.raises(openai.BadRequestError):
        p.plan("sys", "usr")


# ---- keys: precedence, and never leaking one ------------------------------

SENTINEL = "gsk_SENTINEL_never_leak_me_0123456789"


def test_a_key_typed_into_the_app_beats_the_environment(monkeypatch):
    from ragvid import settings

    monkeypatch.setenv("GROQ_API_KEY", "from_environment")
    settings.set_key("groq", SENTINEL)
    assert settings.key("groq", "GROQ_API_KEY") == SENTINEL


def test_the_environment_still_configures_a_provider_with_nothing_stored(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from_environment")
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    client = get_provider("groq").client        # must not raise
    assert client.api_key == "from_environment"


def test_a_stored_key_configures_the_provider_without_touching_the_environment(monkeypatch):
    from ragvid import settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    settings.set_key("groq", SENTINEL)
    assert get_provider("groq").client.api_key == SENTINEL
    assert "GROQ_API_KEY" not in os.environ     # nothing was smuggled into the env


def test_the_saved_provider_and_model_are_what_a_grade_uses(monkeypatch):
    from ragvid import settings

    monkeypatch.setenv("RAGVID_PROVIDER", "groq")   # settings win over this
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    settings.select(provider="openai", model="gpt-4.1")
    p = get_provider()
    assert (p.name, p.model) == ("openai", "gpt-4.1")

    # ...but a model saved for openai must not follow the user to anthropic
    assert get_provider("anthropic").model == DEFAULTS["anthropic"]


def test_a_provider_never_carries_its_key_in_str_or_repr(monkeypatch):
    from ragvid import settings

    settings.set_key("groq", SENTINEL)
    p = get_provider("groq")
    p.client                                     # force the key to be read
    assert SENTINEL not in str(p) + repr(p) + str(vars(p))
    assert "groq" in repr(p)                     # still says something useful


def test_ollama_needs_no_key_at_all(monkeypatch):
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    p = get_provider("ollama")
    assert p.client.base_url.host == "127.0.0.1"   # built without a key, no raise


def test_describe_reports_configured_state_and_never_a_key():
    from ragvid import settings

    settings.set_key("groq", SENTINEL)
    rows = {row["name"]: row for row in describe()}
    assert rows["groq"]["configured"] is True
    assert rows["groq"]["hint"] == "…6789"
    assert rows["groq"]["source"] == "settings"
    assert rows["anthropic"]["configured"] is False
    assert rows["ollama"]["configured"] is True and rows["ollama"]["needs_key"] is False
    assert SENTINEL not in json.dumps(rows)
