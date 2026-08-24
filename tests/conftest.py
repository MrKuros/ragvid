"""Generate test fixtures on demand instead of committing binaries.

sample.mp4 and the reference stills are ~600KB of regenerable binary. Keeping
them out of git history costs one ffmpeg call each, and the suite already
requires ffmpeg for everything else it does.

Tests refer to fixtures by repo-relative path ("assets/sample.mp4"), so this
also chdir's to the repo root — the suite then passes regardless of where
pytest was invoked from.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"

# name -> ffmpeg args producing it (input side only; output path appended)
FIXTURES = {
    "sample.mp4": [
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
    ],
    "ref_warm.png": [
        "-f", "lavfi", "-i", "color=c=0x8B4513:size=320x240,noise=alls=20:allf=t",
        "-frames:v", "1",
    ],
    "ref_bars.png": [
        "-f", "lavfi", "-i", "smptebars=size=320x240", "-frames:v", "1",
    ],
    # An animated, audio-less, palette-based input. Guards the GIF export path
    # and the no-audio case, neither of which sample.mp4 exercises.
    "sample.gif": [
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=1",
    ],
}


def _build(name: str, args: list[str]) -> None:
    out = ASSETS / name
    if out.exists():
        return
    ASSETS.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args, str(out)],
        check=True, capture_output=True,
    )


@pytest.fixture(scope="session", autouse=True)
def _fixtures_exist():
    """Ensure the generated media fixtures are present before any test runs."""
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg not available", allow_module_level=True)
    for name, args in FIXTURES.items():
        _build(name, args)
    yield


@pytest.fixture(scope="session", autouse=True)
def _run_from_repo_root():
    """Tests use repo-relative asset paths; make that true wherever pytest ran."""
    prev = os.getcwd()
    os.chdir(REPO)
    yield
    os.chdir(prev)


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    """Point the settings store at a throwaway dir for every test.

    Without this the suite reads the developer's REAL settings file, so anyone
    who has ever pasted a key into the Settings panel of `ragvid serve` sees
    test_robustness.py::test_missing_api_key_is_a_clear_error fail with
    "DID NOT RAISE ProviderNotConfigured" -- a green suite on a clean machine
    and a red one on a configured machine, which is the worst way to learn
    about a test. Reproduced before this fixture existed, not theorised.

    All three branches of platform.data_dir() are pinned, not just the Linux
    one, so the isolation holds on macOS and Windows too. The provider key
    variables are cleared for the same reason: a real key in the environment
    must not decide whether a test passes.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))   # Linux/XDG
    monkeypatch.setenv("HOME", str(tmp_path / "home"))            # macOS
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))      # Windows
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "XAI_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY",
                "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "TOGETHER_API_KEY",
                "RAGVID_API_KEY", "RAGVID_BASE_URL", "RAGVID_PROVIDER",
                "RAGVID_MODEL"):
        monkeypatch.delenv(var, raising=False)


# A file mode assertion is a POSIX assertion. Windows has no mode bits at all:
# stat() there synthesises S_IWRITE from the read-only attribute, so S_IMODE
# reports 0o666 for every writable file and 0o444 for every read-only one, and
# `== 0o600` can never hold. settings.save() already documents the platform's
# own answer to the same question -- the per-user ACL %APPDATA% inherits -- and
# skips the fchmod there, so the code under test is behaving correctly and only
# the assertion is unportable. Skipped rather than relaxed: an assertion that
# passes on 0o666 is not an assertion about a private file.
posix_modes = pytest.mark.skipif(
    os.name == "nt", reason="Windows has no POSIX mode bits; see settings.save()"
)
