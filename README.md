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

### 1. Install ffmpeg and uv

ragvid needs two things on your machine first: **ffmpeg**, which does the video
work, and **uv**, which runs ragvid.

| Your computer | ffmpeg | uv |
|---|---|---|
| **Windows** | `winget install Gyan.FFmpeg` | `winget install astral-sh.uv` |
| **macOS** | `brew install ffmpeg` | `brew install uv` |
| **Linux** (Debian/Ubuntu) | `sudo apt install ffmpeg` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Linux** (Fedora) | `sudo dnf install ffmpeg` | same as above |
| **Linux** (Arch) | `sudo pacman -S ffmpeg` | same as above |

No package manager? [ffmpeg downloads](https://ffmpeg.org/download.html) ·
[uv install guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Install ragvid

Open a terminal — **PowerShell** on Windows, **Terminal** on macOS and Linux —
and run:

```bash
uv tool install ragvid
```

That is the whole install. There is no folder to keep, nothing to download by
hand, and no second step. It takes a minute the first time and never again.

To update later: `uv tool upgrade ragvid`.

### 3. Start it

```bash
ragvid serve
```

ragvid opens in your browser at **http://127.0.0.1:8765**. Leave the terminal
window open while you use it — closing it stops ragvid. To stop it deliberately,
press **Ctrl-C** in that window.

Next time, `ragvid serve` is all you need.

> **"ragvid: command not found"?** uv installed it somewhere your terminal has
> not been told about. Run `uv tool update-shell`, close the terminal, open a
> new one. If you would rather not touch anything, `uvx ragvid serve` works
> without it.

### 4. Connect an AI service

ragvid asks the first time you open it. Click **Connect a service**, pick one,
paste your key, Save. **Groq** is free to start and the panel links straight to
the page where you get a key.

You only do this once — ragvid remembers.

## What you can ask for

Anything a colourist would understand:

> `warm golden hour` · `cold and clinical` · `faded 70s film` ·
> `teal shadows, orange skin` · `moody but keep faces natural` ·
> `grainy and soft` · `bleach bypass` · `like an old VHS tape`

Then nudge it in plain words: *less blue* · *brighter* · *half as strong* ·
*more contrast, but don't blow out the sky* · *grainier*.

Every adjustment is a small change to the look you already have, so you can
keep going until it's right. **Ctrl-Z** steps back.

You can also say **which part of the picture**: by colour — *"deepen the blues,
leave skin alone"* — or by place — *"darken the top of the frame"*, *"lift the
edges"*. Everything else in the shot stays as it was.

### Naming a thing in the picture

*"make the sky moody"* · *"warm the foliage"* · *"cool down the water"* — ragvid
can find the thing itself, but only with one extra piece installed:

```bash
uv tool install --force "ragvid[masks]"
```

It costs a **13 MB download** — about 70 MB once unpacked — plus a **15 MB**
model that arrives the first time you actually name a thing, and only after
ragvid has asked you. After that it runs on your own machine like everything
else, with nothing sent anywhere.

Without it, colours and places still work; naming a thing tells you what to
install rather than quietly ignoring the word.

Five things are recognised — **sky, foliage, person, water, buildings**. They
are found once, at the start of the clip, not frame by frame: if your subject
walks out of shot halfway through, their grade stays where they were.

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

**Shot in log?** Pick your camera from the **"Shot in log?"** list under the
prompt box — S-Log3, V-Log, Canon Log 3, ARRI LogC, N-Log — and ragvid builds
the conversion itself and applies it before measuring and before grading. No
hunting for the vendor's LUT file; if you do have it, the same list takes it.
Without this the picture is the flat grey one straight off the sensor, and the
result is a guess at un-flattening it.

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
- [`CHANGELOG.md`](CHANGELOG.md) — what changed, in each release

### Working on ragvid

The install above is for *using* ragvid. To change it, take a checkout and the
dev extra instead — one line at a time, because Windows PowerShell has no `&&`:

```bash
git clone https://github.com/MrKuros/ragvid
cd ragvid
uv sync --extra dev
uv run ragvid serve
```

`uv run pytest -q` runs the suite (~2 min, no API key — a test that reaches a
live LLM is a bug). `./scripts/check.sh` is the gate: tests plus the invariants
a passing test run does not cover, and it exits non-zero with the count.

Read [`CLAUDE.md`](CLAUDE.md) before the first change. It is short, and it is
the list of things that silently corrupt output if you break them.

## License

MIT
