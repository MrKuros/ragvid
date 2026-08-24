"""The settings store: precedence, and the file that holds a credential.

Nothing here talks to a network. SENTINEL is a fake key; the point of most of
these tests is that it never turns up anywhere it should not.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from ragvid import settings
from ragvid.platform import data_dir
from tests.conftest import posix_modes

SENTINEL = "gsk_SENTINEL_never_leak_me_0123456789"


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Every test gets its own settings dir — never the real one under $HOME."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))       # macOS branch of data_dir()
    monkeypatch.setenv("APPDATA", str(tmp_path))    # Windows branch
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return tmp_path


# ---- where it lives -------------------------------------------------------


def test_settings_live_in_the_platform_data_dir():
    assert settings.path() == data_dir() / "settings.json"
    assert settings.load() == {}          # nothing there yet, and that is not an error


def test_a_damaged_file_reads_as_no_settings():
    settings.path().parent.mkdir(parents=True)
    settings.path().write_text("{not json")
    assert settings.load() == {}
    settings.set_key("groq", SENTINEL)    # and the next write repairs it
    assert settings.key("groq", "GROQ_API_KEY") == SENTINEL


# ---- the security properties ----------------------------------------------


@posix_modes
def test_the_key_file_is_owner_only_and_so_is_its_directory():
    settings.set_key("groq", SENTINEL)
    assert stat.S_IMODE(settings.path().stat().st_mode) == 0o600
    assert stat.S_IMODE(settings.path().parent.stat().st_mode) == 0o700


@posix_modes
def test_the_mode_is_right_even_under_a_permissive_umask():
    """0600 must come from the open() flags, not from luck with the umask.

    An open()-then-chmod would leave the key world-readable for the moment in
    between; this is the regression test for that ordering.
    """
    old = os.umask(0o000)
    try:
        settings.set_key("groq", SENTINEL)
    finally:
        os.umask(old)
    assert stat.S_IMODE(settings.path().stat().st_mode) == 0o600


@posix_modes
def test_a_rewrite_over_a_loose_file_tightens_it(store):
    """A settings.json left behind at 0644 (an older ragvid, a stray editor) must
    not stay that way once a key goes into it: O_CREAT does not change the mode
    of an existing file, which is why save() fchmods."""
    settings.path().parent.mkdir(parents=True)
    settings.path().write_text("{}")
    os.chmod(settings.path(), 0o644)
    settings.set_key("groq", SENTINEL)
    assert stat.S_IMODE(settings.path().stat().st_mode) == 0o600


def test_clearing_removes_the_key_from_the_file(store):
    settings.set_key("groq", SENTINEL)
    assert SENTINEL in settings.path().read_text()

    settings.clear_key("groq")

    body = settings.path().read_text()
    assert SENTINEL not in body                       # gone, not blanked
    assert "groq" not in json.loads(body).get("keys", {})
    assert settings.key("groq", "GROQ_API_KEY") is None
    # and clearing something that was never there is a no-op, not a crash
    settings.clear_key("openai")


def test_writes_are_atomic_and_leave_no_temp_file_behind(store):
    settings.set_key("groq", SENTINEL)
    settings.select(provider="openai")
    leftovers = [p.name for p in settings.path().parent.iterdir() if p.name != "settings.json"]
    assert leftovers == []


def test_the_hint_is_the_only_shape_of_a_key_that_leaves_the_module():
    settings.set_key("groq", SENTINEL)
    hint = settings.hint("groq", "GROQ_API_KEY")
    assert hint == "…6789"
    assert SENTINEL not in hint and len(hint) == 5


# ---- precedence -----------------------------------------------------------


def test_settings_beat_the_environment(monkeypatch):
    """The deliberate act (typing it in) wins over an inherited variable.

    'I pasted my key in the box and nothing happened' is the worse failure.
    """
    monkeypatch.setenv("GROQ_API_KEY", "from_environment")
    settings.set_key("groq", SENTINEL)
    assert settings.key("groq", "GROQ_API_KEY") == SENTINEL
    assert settings.source("groq", "GROQ_API_KEY") == "settings"


def test_the_environment_still_works_when_nothing_is_stored(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from_environment")
    assert settings.key("groq", "GROQ_API_KEY") == "from_environment"
    assert settings.source("groq", "GROQ_API_KEY") == "environment"
    assert settings.hint("groq", "GROQ_API_KEY") == "…ment"


def test_clearing_hands_the_environment_back(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from_environment")
    settings.set_key("groq", SENTINEL)
    settings.clear_key("groq")
    assert settings.key("groq", "GROQ_API_KEY") == "from_environment"


def test_an_empty_variable_counts_as_no_key(monkeypatch):
    """`GROQ_API_KEY=` is how a shell says unset, and the providers already read
    it that way -- they raise ProviderNotConfigured on an empty string. Before
    this, the settings layer disagreed: the service showed as ready in the UI and
    then failed on the first grade."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert settings.key("groq", "GROQ_API_KEY") is None
    assert settings.source("groq", "GROQ_API_KEY") is None
    assert settings.hint("groq", "GROQ_API_KEY") is None


def test_a_provider_with_no_key_and_no_variable_is_simply_unset():
    assert settings.key("ollama", None) is None
    assert settings.source("ollama", None) is None
    assert settings.hint("ollama", None) is None


# ---- the active choice ----------------------------------------------------


def test_provider_and_model_round_trip():
    assert settings.selected() == (None, None)
    settings.select(provider="openai", model="gpt-4.1-mini")
    assert settings.selected() == ("openai", "gpt-4.1-mini")
    settings.select(model="")                 # empty model = back to the default
    assert settings.selected() == ("openai", None)


def test_choosing_a_provider_keeps_the_stored_keys():
    settings.set_key("groq", SENTINEL)
    settings.select(provider="anthropic")
    assert settings.key("groq", "GROQ_API_KEY") == SENTINEL


def test_an_empty_key_is_refused_rather_than_stored():
    with pytest.raises(ValueError):
        settings.set_key("groq", "   ")
