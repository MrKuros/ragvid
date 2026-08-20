#!/usr/bin/env python
"""ONE real provider call, so a human can eyeball the wiring end to end.

    uv run scripts/smoke_llm.py                 # default vibe, provider from .env
    uv run scripts/smoke_llm.py "sun-bleached western"
    RAGVID_PROVIDER=anthropic uv run scripts/smoke_llm.py

Not a test: this hits the live API. Costs one small request.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragvid.probe import ClipStats  # noqa: E402
from ragvid.providers import get_provider  # noqa: E402
from ragvid.refine import refine_spec  # noqa: E402
from ragvid.spec import RGB  # noqa: E402
from ragvid.vibe import plan_vibe  # noqa: E402

# Stand-in for probe_video() so the smoke test needs no ffmpeg and no asset.
# Display space (sRGB 0-1), matching what probe.py reports.
STATS = ClipStats(
    mean=RGB(r=0.4996, g=0.4736, b=0.5381),
    std=RGB(r=0.0888, g=0.0884, b=0.0888),
    saturation=0.1735,
    frames_sampled=10,
    width=640,
    height=360,
    duration=4.0,
)


def main() -> int:
    vibe = sys.argv[1] if len(sys.argv) > 1 else "gloomy rainy night"
    provider = get_provider()
    print(f"provider={provider.name} model={provider.model}\n")

    print(f'--- plan_vibe("{vibe}") ---')
    spec = plan_vibe(vibe, STATS, provider=provider)
    print(spec.model_dump_json(indent=2))
    print(f"\nidentity? {spec.is_identity()}   rationale: {spec.rationale}")

    if "--refine" in sys.argv:
        instruction = "less blue, more contrast"
        print(f'\n--- refine_spec("{instruction}") ---')
        after = refine_spec(spec, instruction, STATS, provider=provider)
        print(after.model_dump_json(indent=2))
        print(
            f"\ntemperature {spec.temperature} -> {after.temperature}"
            f"   contrast {spec.contrast} -> {after.contrast}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
