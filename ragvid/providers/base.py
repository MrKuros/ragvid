"""Provider protocol, catalog and selection. The only thing a provider ever
returns is a GradeSpec.

WHERE A KEY COMES FROM, in order:

    settings.json (typed into the app)  >  os.environ  >  .env

`.env` is folded into os.environ by load_env() below, which never overwrites a
variable that is already set — so a real shell variable still wins over the
file. What beats both is a key the user deliberately typed into this app; see
ragvid/settings.py for why that way round.

WHICH PROVIDERS EXIST: the CATALOG below, plus one escape hatch. Any
OpenAI-compatible endpoint that is not in the catalog works with no code change
at all:

    RAGVID_BASE_URL=https://my-endpoint/v1  RAGVID_API_KEY=...  RAGVID_MODEL=...

That is the real answer to "does it work with everything"; the catalog only
saves the common cases from having to know their own URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragvid import settings
from ragvid.errors import ProviderError
from ragvid.spec import GradeSpec

# Repo root .env — ragvid/providers/base.py -> parents[2].
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


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


# ---- the catalog ----------------------------------------------------------


@dataclass(frozen=True)
class ProviderInfo:
    """One row of the catalog. Data only — never holds a key.

    `structured` is the strongest response_format the endpoint honours:

        "json_schema"  constrained decoding against GradeSpec's schema. Every
                       one of the 43 numbers comes back by construction.
        "json_object"  valid JSON guaranteed, the schema itself only requested
                       in the prompt. Fields can go missing; ragvid checks and
                       raises rather than defaulting them.
        "text"         no guarantee; the JSON gets extracted from prose.

    It is the STARTING rung, not a promise: OpenAICompatProvider steps down a
    rung whenever an endpoint rejects the format outright.
    """

    name: str
    label: str
    model: str
    kind: str = "openai"           # "openai" (chat completions) or "anthropic"
    base_url: str = ""
    env_var: str | None = None     # None = no key needed (Ollama)
    fallback: str | None = None
    structured: str = "json_schema"
    keys_url: str = ""             # where a human gets a key
    note: str = ""


CATALOG: dict[str, ProviderInfo] = {
    p.name: p for p in [
        ProviderInfo(
            "groq", "Groq", "openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1", env_var="GROQ_API_KEY",
            fallback="openai/gpt-oss-20b", structured="json_schema",
            keys_url="https://console.groq.com/keys",
            note="Fast and free to start. Enforces the schema, so grades come back complete.",
        ),
        ProviderInfo(
            "anthropic", "Anthropic (Claude)", "claude-opus-5", kind="anthropic",
            env_var="ANTHROPIC_API_KEY", structured="json_schema",
            keys_url="https://console.anthropic.com/settings/keys",
            note="Enforces the schema. The most careful at reading a mood.",
        ),
        ProviderInfo(
            "openai", "OpenAI", "gpt-4.1-mini",
            base_url="https://api.openai.com/v1", env_var="OPENAI_API_KEY",
            fallback="gpt-4o-mini", structured="json_schema",
            keys_url="https://platform.openai.com/api-keys",
            note="Enforces the schema (Structured Outputs).",
        ),
        ProviderInfo(
            "xai", "xAI (Grok)", "grok-4",
            base_url="https://api.x.ai/v1", env_var="XAI_API_KEY",
            fallback="grok-3-mini", structured="json_schema",
            keys_url="https://console.x.ai/",
            note="Enforces the schema.",
        ),
        ProviderInfo(
            "mistral", "Mistral", "mistral-large-latest",
            base_url="https://api.mistral.ai/v1", env_var="MISTRAL_API_KEY",
            fallback="mistral-small-latest", structured="json_schema",
            keys_url="https://console.mistral.ai/api-keys/",
            note="Enforces the schema (custom structured outputs).",
        ),
        ProviderInfo(
            "openrouter", "OpenRouter", "openai/gpt-4.1-mini",
            base_url="https://openrouter.ai/api/v1", env_var="OPENROUTER_API_KEY",
            structured="json_schema",
            keys_url="https://openrouter.ai/keys",
            note="One key, many models. Schema enforcement depends on the model you "
                 "pick; ragvid drops to plain JSON if the model refuses it.",
        ),
        ProviderInfo(
            "deepseek", "DeepSeek", "deepseek-chat",
            base_url="https://api.deepseek.com/v1", env_var="DEEPSEEK_API_KEY",
            structured="json_object",
            keys_url="https://platform.deepseek.com/api_keys",
            note="Cheap. Guarantees valid JSON but not the full schema, so a grade "
                 "can come back incomplete — ragvid says so instead of guessing.",
        ),
        ProviderInfo(
            "moonshot", "Moonshot (Kimi)", "kimi-k2-0711-preview",
            base_url="https://api.moonshot.ai/v1", env_var="MOONSHOT_API_KEY",
            fallback="moonshot-v1-32k", structured="json_object",
            keys_url="https://platform.moonshot.ai/console/api-keys",
            note="Guarantees valid JSON but not the full schema; see DeepSeek.",
        ),
        ProviderInfo(
            "together", "Together", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            base_url="https://api.together.xyz/v1", env_var="TOGETHER_API_KEY",
            structured="json_object",
            keys_url="https://api.together.xyz/settings/api-keys",
            note="Open models. JSON mode only, so a grade can come back incomplete.",
        ),
        ProviderInfo(
            "ollama", "Ollama (on this machine)", "llama3.1",
            base_url="http://127.0.0.1:11434/v1", env_var=None,
            structured="json_object",
            note="Runs locally, no key and no bill. Small models often miss fields; "
                 "if grades keep failing, use a bigger model or a hosted provider.",
        ),
    ]
}

# Back-compat: {name: default model}. The catalog is the source of truth.
DEFAULTS = {name: info.model for name, info in CATALOG.items()}

CUSTOM = "custom"


def custom_info() -> ProviderInfo | None:
    """The RAGVID_BASE_URL escape hatch, if it is configured."""
    base = os.environ.get("RAGVID_BASE_URL")
    if not base:
        return None
    return ProviderInfo(
        CUSTOM, "Custom endpoint", os.environ.get("RAGVID_MODEL") or "",
        base_url=base, env_var="RAGVID_API_KEY", structured="json_schema",
        note=f"Whatever is at {base}. ragvid tries strict JSON first and steps "
             "down if the endpoint refuses it.",
    )


def catalog() -> list[ProviderInfo]:
    """Every provider a UI can offer, escape hatch included when it is set up."""
    extra = custom_info()
    return list(CATALOG.values()) + ([extra] if extra else [])


def info_for(name: str) -> ProviderInfo:
    if name == CUSTOM:
        custom = custom_info()
        if custom is None:
            raise ProviderError(name, "set RAGVID_BASE_URL (and RAGVID_API_KEY) first")
        return custom
    try:
        return CATALOG[name]
    except KeyError:
        raise ProviderError(
            name, "unknown provider; expected one of " + ", ".join(CATALOG)
            + " (or set RAGVID_BASE_URL for any other OpenAI-compatible endpoint)"
        ) from None


def describe() -> list[dict]:
    """The catalog as JSON a UI can render — never including a key.

    `hint` is the last four characters and nothing else; `source` says whether
    the key came from this app or from the environment, which is the difference
    between "I can change it here" and "someone set a variable".
    """
    load_env()
    active, active_model = active_choice()
    out = []
    for info in catalog():
        out.append({
            "name": info.name,
            "label": info.label,
            "model": info.model,
            "needs_key": info.env_var is not None,
            "env_var": info.env_var,
            "configured": info.env_var is None or settings.key(info.name, info.env_var) is not None,
            "hint": settings.hint(info.name, info.env_var),
            "source": settings.source(info.name, info.env_var),
            "structured": info.structured,
            "enforces_schema": info.structured == "json_schema",
            "keys_url": info.keys_url,
            "note": info.note,
            "active": info.name == active,
        })
    return out


def active_choice() -> tuple[str, str | None]:
    """(provider, model override) after settings, environment and defaults."""
    chosen, model = settings.selected()
    name = (chosen or os.environ.get("RAGVID_PROVIDER") or _default_name()).strip().lower()
    if chosen != name:
        model = None  # a model saved for another provider means nothing here
    return name, model or os.environ.get("RAGVID_MODEL") or None


def _default_name() -> str:
    # A configured escape hatch is a deliberate act; honour it as the default.
    return CUSTOM if os.environ.get("RAGVID_BASE_URL") else "groq"


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    """Build the configured provider.

    Explicit arguments win; then settings.json; then RAGVID_PROVIDER / RAGVID_MODEL
    (and .env); then Groq.
    """
    load_env()
    chosen, chosen_model = active_choice()
    name = (name or chosen).strip().lower()
    if not model:
        model = chosen_model if name == chosen else None
    info = info_for(name)
    if not (model or info.model):
        raise ProviderError(name, "no model set — put one in RAGVID_MODEL")

    if info.kind == "anthropic":
        from ragvid.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model or info.model)

    from ragvid.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(
        name=info.name,
        base_url=info.base_url,
        model=model or info.model,
        env_var=info.env_var,
        fallback_model=info.fallback,
        structured=info.structured,
    )
