# ragvid

**Tell it the mood you want. It colours your video.**

Type "gloomy, teal shadows, grainy" and ragvid works out the colour grade,
shows you a preview, and lets you talk your way to the look you want — "less
blue", "half as strong", "warmer highlights". When you like it, it renders the
whole video.

You can also hand it a photo instead of a description, and it will match your
footage to that photo's colours.

**Your video never leaves your computer.** Only your few words of description
are sent away. No frames, no clips, no uploads.

## Getting started

You need [ffmpeg](https://ffmpeg.org/download.html) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) installed first.
Then:

```bash
git clone https://github.com/MrKuros/ragvid && cd ragvid
uv venv && uv pip install -e ".[dev]"
uv run ragvid serve
```

That last line opens ragvid in your browser. Leave the terminal window open
while you use it.

**One more step the first time.** ragvid uses an AI service to turn your words
into a grade, and those services need an account. Click **Settings**, pick a
service, paste your key, Save. Groq is free to start and is a good first
choice — the Settings panel links straight to the page where you get a key.

Then: open a video, type a mood, and adjust from there.

## What you can ask for

Anything a colourist would understand:

> `warm golden hour` · `cold and clinical` · `faded 70s film` ·
> `teal shadows, orange skin` · `moody but keep faces natural` ·
> `grainy and soft` · `bleach bypass` · `like an old VHS tape`

Then nudge it in plain words: *less blue* · *brighter* · *half as strong* ·
*more contrast, but don't blow out the sky* · *grainier*.

Every adjustment is a small change to the look you already have, so you can
keep going until it's right. **Ctrl-Z** steps back.

## Which AI service?

Free to start: **Groq** (fastest way in), or **Ollama** if you want to run a
model on your own machine with no account and no bill at all.

Also supported: OpenAI, Anthropic, xAI, Mistral, DeepSeek, Moonshot,
OpenRouter, Together — and anything OpenAI-compatible.

The Settings panel tells you which services reliably return a complete grade
and which occasionally leave gaps. When one leaves a gap, ragvid says so
instead of quietly making something up.

Your key is stored on your own computer, in a file only you can read. ragvid
never shows it again — just the last four characters, so you can tell which key
is which.

> **If you have ever pasted a key into a chat, a screenshot, or a shared file,
> replace it.** Anyone who has seen it can use it, and it is billed to you.

## Good to know

**It costs very little.** ragvid sends a few words and gets back a small set of
numbers — it never sends the video. Grading a two-hour film costs the same as
grading a two-second clip.

**Previews are instant, exports are not.** While you're adjusting, ragvid only
draws three frames. The full render happens once, when you export, and effects
like grain and glow make it slower — a heavy look can take about two and a half
times the length of the clip. Turn off the effects you don't actually want.

**Matching a photo needs no account at all.** That path is pure maths, done on
your own computer, with no AI service involved.

**One clip, one look.** If different parts of your video need different looks,
split it into separate clips first. ragvid doesn't do cuts, pacing, or audio —
your audio is copied across untouched.

## For developers

There's a terminal interface, and everything the app does is available as a
Python API:

```bash
ragvid grade clip.mp4 --vibe "gloomy, teal shadows, grainy"
ragvid refine "less blue, half strength"
ragvid export out.mp4
```

| Command | Does |
|---|---|
| `ragvid grade IN --vibe "..."` | Measure, plan a grade, write a preview |
| `ragvid grade IN --ref img.jpg` | Same, matched to a reference image (offline) |
| `ragvid refine "..."` | Adjust the current grade in words |
| `ragvid spec` | Print the current grade as JSON |
| `ragvid reset` | Step back one edit |
| `ragvid export OUT` | Render the full video |
| `ragvid serve` | Open the local web UI |

API keys are the one thing the terminal cannot set — they are entered in the
Settings panel and nowhere else, so a key can never reach your shell history or
be read out of the process list by another user on the machine.

A grade is **43 numbers**: 37 colour values baked into a `.cube` LUT, plus 6
spatial effects that ffmpeg applies around it. A LUT maps one pixel at a time
and cannot see its neighbours, so a `.cube` taken into Resolve carries the
colour and none of the effects.

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the `Project` API to build
  against, the load-bearing evaluation order, and the gotchas that cost real
  debugging time
- [`docs/WEB_API.md`](docs/WEB_API.md) — the local HTTP API behind `serve`
- `python examples/api_tour.py` — every call in order, no API key needed

## License

MIT
