# ragvid

Describe the vibe, get the grade.

```bash
ragvid grade clip.mp4 --vibe "gloomy, teal shadows, grainy"
ragvid refine "less blue, half strength"
ragvid export out.mp4
```

Point it at a video and either a mood word or a reference image. It measures the
footage, produces a colour grade, and lets you talk your way to the look you
want. There is a local web UI too — `ragvid serve`.

## How it works

An LLM never sees your frames. It produces **43 numbers**, and everything after
that is numpy and ffmpeg:

```
reference image ──▶ closed-form match ──┐
                                        │   37 colour numbers
                                        ├──▶ GradeSpec ──┬──▶ .cube LUT ──┐
vibe word ────────▶ LLM ────────────────┘     (JSON)     │                ├──▶ ffmpeg
                                            ▲            └──▶ 6 effects ──┘
"less blue" ────────────────────────────────┘                (filters)
```

Three consequences worth knowing up front:

- **Cheap.** One small JSON object per request, so grading a two-hour film costs
  the same as grading a two-second clip.
- **Refinable.** The `.cube` is a derived artifact, not the source of truth. You
  cannot meaningfully adjust a baked LUT; you can adjust numbers.
- **Offline for reference images.** That path is closed-form linear colour
  transfer — no model, no network.

Analysis samples ~10 frames and takes the median: clips drift, so trusting frame
0 is a mistake. Application touches every frame, but it is a fast filter chain
with no model in the loop.

## Install

```bash
git clone https://github.com/MrKuros/ragvid && cd ragvid
uv venv && uv pip install -e ".[dev]"
```

Requires `ffmpeg` (any recent build has the `lut3d` filter).

Then set a key — either in the web UI's Settings panel, or:

```bash
ragvid config --set-key groq     # prompts; never pass a key as an argument
ragvid config                    # list providers, show which are ready
```

Keys live in `~/.local/share/ragvid/settings.json`, created `0600` before it
holds anything. A key is never printed, logged, or returned by the API — only
its last four characters.

Then run it:

```bash
uv run ragvid serve            # http://127.0.0.1:8765, opens a browser
```

`uv run` is what makes `ragvid` resolve without activating the venv; inside an
activated venv the bare `ragvid ...` in this README works as written. `--port`
picks a different port, and a taken port falls through to the next free one.

## Providers

`groq · anthropic · openai · xai · mistral · openrouter · deepseek · moonshot ·
together · ollama`, plus any OpenAI-compatible endpoint via `RAGVID_BASE_URL`.

They differ in one way that matters. Filling 43 required numbers reliably needs
strict JSON-schema decoding, and not every provider enforces it. `ragvid config`
marks each one **enforced** or **best effort**; a best-effort provider that
returns an incomplete grade raises an error naming the missing fields instead of
quietly filling them with defaults.

Groq is the default. Its free tier is 8000 tokens/min — fine interactively,
roughly two calls a minute at this spec size.

> **Rotate any key that has been shared.** `.env` is gitignored from the first
> commit, but nothing protects a key that already leaked.

## Commands

| Command | Does |
|---|---|
| `ragvid grade IN --vibe "..."` | Measure, plan a grade, write a preview |
| `ragvid grade IN --ref img.jpg` | Same, matched to a reference image (offline) |
| `ragvid refine "..."` | Adjust the current grade in words |
| `ragvid spec` | Print the current grade as JSON |
| `ragvid reset` | Step back one edit |
| `ragvid export OUT` | Render the full video |
| `ragvid config` | Providers and API keys |
| `ragvid serve` | Open the local web UI |

`grade` and `refine` render a 3-frame contact sheet, not the whole file, so the
refinement loop stays sub-second. Only `export` does the full render.

## As a library

The CLI is a thin shell over `Project`, which is the whole API — probe, plan,
refine, bake, preview, export. Build a UI against this instead of
reimplementing the flow.

```python
from ragvid import Project

p = Project.create("clip.mp4", root="~/grades/clip")
p.plan_from_vibe("gloomy")                              # or plan_from_reference(img)
p.refine("less blue")
p.set_spec(p.spec.model_copy(update={"contrast": 0.4})) # a slider moved
p.undo()
p.export("out.mp4", progress=lambda f: bar.set(f))      # 0.0 -> 1.0
```

Nothing in the core prints, reads argv, calls `sys.exit`, or assumes a working
directory. Failures are typed (`InputError`, `RateLimited`, `FFmpegError`, …)
and carry the fields a UI needs, so nothing is recovered by parsing a message.

`python examples/api_tour.py` runs every call in order, no API key needed.

## What a grade is

**37 colour numbers**, baked into the LUT: gain, lift, gamma ([ASC CDL][cdl]),
saturation, temperature/tint, contrast, exposure in stops, a shadow/highlight
split with independent tint and lift, six hue qualifiers, a soft highlight
shoulder, and a mix of the whole look back toward the source.

**6 spatial effects**, applied by ffmpeg around the LUT: denoise, glow, softness
(blur or sharpen), grain, vignette, chromatic fringing.

That split is where the `.cube` stops. A 3D LUT maps one pixel to one pixel and
cannot see its neighbours, so no effect can live in one — take a `.cube` into
Resolve and you get the colour and nothing else.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) covers that seam.

Effects are not free, and the bill lands on export. Measured at 1080p30:

| Export | Speed |
|---|---|
| colour only | **1.38×** realtime |
| grain + vignette | **0.83×** |
| all six effects | **0.42×** |

`--gpu` does not rescue that: it swaps the encoder, and the encoder is not the
bottleneck. Leave at zero the effects you do not actually want.

One clip gets one look; for several looks, split into clips. Out of scope:
pacing, cuts, and audio (stream-copied on export, never re-encoded).

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layering, the load-bearing
  evaluation order, and the gotchas that cost real debugging time
- [`docs/WEB_API.md`](docs/WEB_API.md) — the local HTTP API behind `serve`

[cdl]: https://en.wikipedia.org/wiki/ASC_CDL

## License

MIT
