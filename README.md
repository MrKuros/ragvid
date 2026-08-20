# ragvid

Describe the vibe, get the grade.

```bash
ragvid grade clip.mp4 --vibe "gloomy"
ragvid refine "less blue, more contrast"
ragvid export out.mp4
```

Point it at a video and either a mood word or a reference image. It analyses the
footage, produces a color grade, and lets you talk your way to the look you want.

## Why it works this way

An LLM never sees your frames. It produces **~14 numbers** — an [ASC CDL][cdl]
grade spec — and everything after that is numpy and ffmpeg:

```
reference image ──▶ closed-form match ──┐
                                        ├──▶ GradeSpec (JSON) ──▶ .cube LUT ──▶ ffmpeg
vibe word ────────▶ LLM ────────────────┘           ▲
                                                    │
"less blue" ────────────────────────────────────────┘
```

Two consequences:

- **It's cheap.** One small JSON object per request, so grading a two-hour film
  costs the same as grading a two-second clip.
- **It's refinable.** The `.cube` LUT is a derived artifact, not the source of
  truth. You can't meaningfully adjust a baked LUT; you can adjust numbers.

Analysis samples ~10 frames (not every frame — clips drift, so the median across
samples beats trusting frame 0). Application touches every frame, but that's a
single fast ffmpeg filter with no model in the loop.

The reference-image path uses **no LLM at all** — it's closed-form linear color
transfer — so it works entirely offline.

## Install

```bash
git clone <this repo> && cd rag-video
uv venv && uv pip install -e ".[dev]"
cp .env.example .env    # then add your key
```

Requires `ffmpeg` (with the `lut3d` filter — any recent build has it).

### Providers

| Provider | Env | Notes |
|---|---|---|
| Groq (default) | `GROQ_API_KEY` | Free tier is **8000 tokens/min** — fine interactively, rate-limits under load. Default model `openai/gpt-oss-120b`. |
| Anthropic | `ANTHROPIC_API_KEY` | `RAGVID_PROVIDER=anthropic`. Higher quality baseline. |

> **Rotate your key on first use.** If you were handed a key in a chat or shared
> a `.env`, treat it as compromised — regenerate at the provider console. `.env`
> is gitignored here from the very first commit, but nothing protects a key that
> already leaked.

## Commands

| Command | Does |
|---|---|
| `ragvid grade IN --vibe "..."` | Analyse, plan a grade, write a preview |
| `ragvid grade IN --ref img.jpg` | Same, matched to a reference image (offline) |
| `ragvid refine "..."` | Adjust the current grade conversationally |
| `ragvid spec` | Print the current grade spec as JSON |
| `ragvid reset` | Step back to the previous spec |
| `ragvid export OUT` | Render the full video |

`grade` and `refine` render a 3-frame contact sheet, not the whole file — the
refinement loop stays sub-second. Only `export` does the full render.

## Scope

v1 is **color only**: gain, lift, gamma, saturation, temperature/tint, contrast.
One clip gets one look; for multiple looks, split into clips.

Not yet: grain, vignette, halation (needs a real filtergraph rather than a LUT),
pacing, cuts, audio.

[cdl]: https://en.wikipedia.org/wiki/ASC_CDL

## License

MIT
