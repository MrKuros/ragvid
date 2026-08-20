"""Security / failure-mode regressions. No network: the provider is always faked."""

from __future__ import annotations

import traceback

import numpy as np
import pytest
from PIL import Image

from ragvid import cli, render
from ragvid.lut import bake_cube
from ragvid.providers.groq import GroqProvider, parse_reset
from ragvid.session import SESSION_DIR
from ragvid.spec import GradeSpec

SAMPLE = "assets/sample.mp4"


# ---- ffmpeg injection ------------------------------------------------------

HOSTILE = [
    "a:b=c.cube",                        # '=' used to be parsed as an option name
    "interp=nearest:x,y'z [1].cube",     # option-injection attempt
    "; touch pwned ; $(id) `id`.cube",   # shell metacharacters
    "back\\slash 'q' [b],s;.cube",
]


@pytest.mark.parametrize("name", HOSTILE)
def test_hostile_cube_paths_reach_lut3d_intact(tmp_path, name):
    """The LUT must actually apply -- not be dropped, mis-parsed, or shell-expanded."""
    cube = tmp_path / name
    bake_cube(GradeSpec(saturation=0.0), str(cube))  # greyscale: visible in the output
    out = tmp_path / "sheet.png"

    render.render_preview(SAMPLE, str(cube), str(out), n_frames=2)

    px = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    assert np.abs(px[..., 0] - px[..., 1]).mean() < 1.0  # r == g => desaturated
    assert not (tmp_path / "pwned").exists()


def test_lut_filter_names_the_file_option():
    assert render._lut_filter("/a=b/x.cube").startswith("lut3d=file=")


# ---- Groq 429 --------------------------------------------------------------


def test_parse_reset_accepts_bare_seconds():
    """Groq's real 429 also carries `retry-after: 22`, which has no unit suffix."""
    assert parse_reset("22") == 22.0
    assert parse_reset("soon") is None


def test_groq_skips_the_fallback_sleep_when_the_reset_is_far_out(monkeypatch):
    """A real 8000-TPM 429 resets in ~54s. Sleeping the 30s cap only wastes it."""
    import openai
    import httpx

    def limited():
        resp = httpx.Response(
            429, headers={"x-ratelimit-reset-tokens": "54.067s"},
            request=httpx.Request("POST", "https://api.groq.com/"),
        )
        return openai.RateLimitError("rate limited", response=resp, body=None)

    slept = []
    monkeypatch.setattr("ragvid.providers.groq.time.sleep", slept.append)

    class Client:
        def __init__(self):
            self.models = []
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            self.models.append(kw["model"])
            raise limited()

    client = Client()
    with pytest.raises(RuntimeError, match="8000 tokens/min"):
        GroqProvider(client=client).plan("sys", "usr")

    assert slept == []                    # never blocked on a wait it can't honour
    assert len(set(client.models)) == 2   # went straight to the fallback model


# ---- CLI failure modes -----------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["refine", "warmer"], ["export", "out.mp4"], ["spec"], ["reset"],
    ["grade", "no-such-file.mp4", "--ref", "assets/ref_warm.png"],
    ["grade", SAMPLE, "--ref", "no-such-file.png"],
])
def test_cli_failures_are_messages_not_tracebacks(argv, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .ragvid session here
    monkeypatch.setattr(cli, "CUBE", str(tmp_path / "c.cube"))
    argv = [f"{_repo()}/{a}" if a.startswith("assets/") else a for a in argv]

    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("ragvid: ") and "Traceback" not in err


def _repo() -> str:
    from pathlib import Path
    return str(Path(__file__).resolve().parents[1])


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    from ragvid.providers.base import get_provider

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        get_provider("groq").client


def test_provider_errors_never_echo_the_key():
    """A 401 must not surface the credential. The SDK sends it; the error must not."""
    import httpx
    import openai

    sentinel = "gsk_SENTINEL_never_log_me_0123456789"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    client = openai.OpenAI(
        api_key=sentinel, base_url="https://api.groq.com/openai/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(openai.AuthenticationError) as info:
        GroqProvider(client=client).plan("sys", "usr")

    assert sentinel in seen["auth"]  # it really was sent, so the check is meaningful
    exc = info.value
    blob = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert sentinel not in blob + str(exc) + repr(exc)


# ---- corrupt session -------------------------------------------------------

CORRUPT = [
    "not json at all",
    "[1, 2, 3]",                                        # right type, wrong shape
    '{"source": "a.mp4", "stats": {}, "specs": [{}]}',  # stats fail validation
    '{"source": "a.mp4", "stats": {}, "specs": []}',    # .spec would IndexError
]


@pytest.mark.parametrize("body", CORRUPT)
def test_corrupt_session_is_a_message_not_a_traceback(body, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / SESSION_DIR).mkdir()
    (tmp_path / SESSION_DIR / "session.json").write_text(body)

    assert cli.main(["spec"]) == 1
    assert capsys.readouterr().err.strip() == "ragvid: no session here — run 'ragvid grade' first"
