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
