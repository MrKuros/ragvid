"""One client for every OpenAI-compatible endpoint. Groq is just the first row.

The chat-completions wire format is the same everywhere; what differs is the
base URL, the model name, the env var holding the key, and — the part that
actually matters here — how much of a JSON schema the endpoint will enforce.

STRUCTURED OUTPUT IS NOT UNIFORM, AND THE DIFFERENCE IS REAL.

    "json_schema"  the endpoint constrains decoding to GradeSpec's schema. All
                   43 numbers come back, every time, by construction. Groq,
                   OpenAI, xAI, Mistral.
    "json_object"  the endpoint guarantees syntactically valid JSON and nothing
                   more. The schema goes in the prompt as a request, so the
                   model can still omit fields or invent them. DeepSeek,
                   Moonshot/Kimi, Together, Ollama.
    "text"         no guarantee at all: the JSON has to be dug out of prose.
                   The fallback rung, and where an unknown endpoint may land.

Below `json_schema` this provider IS less reliable than Groq, and pretending
otherwise would be a lie told to whoever picks a provider from a dropdown. What
it does instead is refuse to paper over the gap: every rung parses through
`GradeSpec(**...)`, checks that the schema's required fields are actually
present, and raises ProviderError if they are not. A missing field is a visible
failure, never a silently defaulted one — a half-filled spec looks like a bad
grade, which is much harder to diagnose than an error.

The ladder also steps down on its own: an endpoint that rejects a
`response_format` (400) is retried one rung lower, so an unknown endpoint
configured through RAGVID_BASE_URL finds its own level at the cost of one
wasted request, and the working rung is remembered for the rest of the process.
"""

from __future__ import annotations

import json
import re
import time

from ragvid.errors import ProviderError, ProviderNotConfigured, RateLimited
from ragvid.spec import GradeSpec

ATTEMPTS_PER_MODEL = 2
MAX_SLEEP = 30.0

# Strongest first. A provider starts at its catalogued capability and can only
# move down this list, never up.
MODES = ("json_schema", "json_object", "text")

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


def schema_prompt() -> str:
    """The schema as an instruction, for endpoints that will not enforce one."""
    return (
        "\n\nReply with a single JSON object and nothing else — no prose, no code "
        "fence. It must match this JSON schema exactly: every property listed in "
        '"required" must be present, with a number (not a string) for every '
        "numeric field.\n" + json.dumps(GradeSpec.llm_json_schema())
    )


class OpenAICompatProvider:
    """A chat-completions endpoint that speaks OpenAI's dialect.

    `structured` is the strongest response_format the endpoint supports; see the
    module docstring for what each level actually guarantees.
    """

    def __init__(self, name: str, base_url: str, model: str, env_var: str | None = None,
                 fallback_model: str | None = None, structured: str = "json_schema",
                 client=None):
        self.name = name
        self.base_url = base_url
        self.model = model
        self.env_var = env_var
        self.fallback_model = fallback_model
        self.structured = structured if structured in MODES else "text"
        self._client = client

    def __repr__(self) -> str:
        # Explicit, so that adding a `self.key` attribute later cannot leak one
        # into a log line or a traceback.
        return f"<{type(self).__name__} {self.name} model={self.model} mode={self.structured}>"

    @property
    def client(self):
        if self._client is None:
            import openai

            from ragvid import settings

            key = settings.key(self.name, self.env_var)
            if not key and self.env_var:
                raise ProviderNotConfigured(self.name, self.env_var)
            # max_retries=0: the retry/backoff policy below is the only one, so the
            # SDK's own retries don't compound our sleeps.
            self._client = openai.OpenAI(
                api_key=key or "not-needed", base_url=self.base_url, max_retries=0
            )
        return self._client

    # ---- the call ---------------------------------------------------------

    def plan(self, system: str, user: str) -> GradeSpec:
        import openai

        models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models.append(self.fallback_model)
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
        raise RateLimited(self.name, retry_after=last_wait) from last

    def _call(self, model: str, system: str, user: str) -> GradeSpec:
        import openai

        rungs = MODES[MODES.index(self.structured):]
        for i, mode in enumerate(rungs):
            try:
                text = self._request(model, system, user, mode)
            except openai.BadRequestError:
                # This endpoint does not accept that response_format. Step down
                # rather than fail: it is the whole point of the ladder.
                if i + 1 >= len(rungs):
                    raise
                continue
            self.structured = mode  # remember what this endpoint actually took
            return self._parse(text, mode)
        raise ProviderError(self.name, "no usable response format")  # unreachable

    def _request(self, model: str, system: str, user: str, mode: str) -> str:
        kwargs = {}
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "grade_spec",
                    "strict": True,
                    "schema": GradeSpec.llm_json_schema(),
                },
            }
        else:
            # No constrained decoding: the schema has to travel in the prompt.
            # (json_object mode on several endpoints also *requires* the word
            # "json" to appear in the messages, which this satisfies.)
            system = system + schema_prompt()
            if mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return response.choices[0].message.content

    def _parse(self, text: str | None, mode: str) -> GradeSpec:
        """Every rung ends here: real JSON, every required field, then sanitize."""
        raw = extract_json(text)
        if raw is None:
            raise ProviderError(self.name, f"returned no JSON object ({_snippet(text)})")
        missing = sorted(missing_fields(GradeSpec.llm_json_schema(), raw))
        if missing:
            # Pydantic would happily default these. That would hand back a spec
            # that looks fine and grades wrong, which is worse than an error.
            raise ProviderError(
                self.name,
                f"answered without {len(missing)} required field(s) "
                f"({', '.join(missing[:6])}) — this endpoint does not enforce a "
                f"schema (mode={mode}); try a provider that does",
            )
        try:
            return GradeSpec(**raw).sanitize()
        except ProviderError:
            raise
        except Exception as exc:  # pydantic ValidationError, TypeError, ...
            raise ProviderError(self.name, f"returned JSON that is not a grade spec: {exc}") from exc


# ---- helpers --------------------------------------------------------------


def extract_json(text: str | None) -> dict | None:
    """The first JSON object in `text`, whatever it is wrapped in.

    Weak endpoints answer with fences, a preamble, or both. Outermost braces,
    because a nested object would parse but be the wrong thing.
    """
    if not text:
        return None
    for candidate in (text, _between_braces(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _between_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None


def missing_fields(schema: dict, data, prefix: str = "") -> list[str]:
    """Required properties absent from `data`, nested objects included."""
    out = []
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if not isinstance(data, dict) or name not in data:
            out.append(prefix + name)
        elif props.get(name, {}).get("type") == "object":
            out += missing_fields(props[name], data[name], f"{prefix}{name}.")
    return out


def _snippet(text: str | None) -> str:
    return (text or "").strip()[:120] or "empty response"


def _retry_after(exc) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    return parse_reset(headers.get("x-ratelimit-reset-tokens") or headers.get("retry-after"))


def _selfcheck() -> None:
    """`python -m ragvid.providers.openai_compat` — the parsing edges."""
    good = json.loads(GradeSpec().model_dump_json())
    assert extract_json("```json\n" + json.dumps(good) + "\n```") == good
    assert extract_json("Sure! " + json.dumps(good) + " Hope that helps.") == good
    assert extract_json("no json here") is None
    assert extract_json(None) is None
    assert missing_fields(GradeSpec.llm_json_schema(), good) == []
    partial = dict(good)
    partial.pop("saturation")
    partial["slope"] = {"r": 1.0, "g": 1.0}
    assert sorted(missing_fields(GradeSpec.llm_json_schema(), partial)) == ["saturation", "slope.b"]
    assert "sk-" not in repr(OpenAICompatProvider("x", "http://h", "m", "K"))
    print("openai_compat selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
