# Architecture

Read this before building a front end.

## The shape

```
                    ┌──────────────┐   ┌──────────────┐
  front ends        │  ragvid.cli  │   │   your UI    │
                    └──────┬───────┘   └──────┬───────┘
                           └────────┬─────────┘
                                    ▼
  facade                    ragvid.project.Project
                            (orchestration, state)
                                    │
        ┌───────────┬───────────┬───┴───┬───────────┬───────────┐
        ▼           ▼           ▼       ▼           ▼           ▼
  core  probe     match       vibe   refine       lut        render
        stats   reference     LLM      LLM       .cube      ffmpeg
                  (no LLM)
                            └────┬────┘
                                 ▼
                            providers/
                          groq · anthropic
                                 │
                                 ▼
                          spec.GradeSpec
                     (14 numbers — the only currency)
```

Three rules hold the whole thing together:

1. **`GradeSpec` is the only currency.** Every planner produces one, every
   consumer takes one. Nothing else crosses layers.
2. **No pixels reach a model, ever.** Planners see statistics and words. The
   LUT and ffmpeg see pixels. They never meet.
3. **Core modules have no I/O policy.** They don't print, don't read argv, don't
   call `sys.exit`, don't assume a working directory. Those are front-end
   concerns and live in the front end.

## Build against `Project`, not the modules

`Project` is the whole API. It owns probe → plan → bake → preview → export, so
a GUI and the CLI run identical code instead of each reimplementing the flow.
`ragvid/cli.py` is ~110 lines and is the reference consumer — read it as the
shortest possible example of driving the facade.

```python
from ragvid import Project

p = Project.create("clip.mp4", root="~/grades/clip")   # probes, no grade yet
p.plan_from_vibe("gloomy")                             # or plan_from_reference(img)
p.refine("less blue")
p.set_spec(p.spec.model_copy(update={"contrast": 0.4}))  # a slider moved
p.undo()
p.preview()                                            # contact sheet, sub-second
p.export("out.mp4", progress=lambda f: bar.set(f))     # the slow one
```

### What each piece gives a UI

| Need | Use |
|---|---|
| Open / create a project | `Project.open(root)` · `Project.create(video, root)` · `Project.exists(root)` |
| Render current state | `project.to_dict()` — JSON-serializable, everything a view needs |
| Undo stack | `project.history` (oldest first) · `project.can_undo` · `project.undo()` |
| Sliders and numeric fields | `project.set_spec(spec)` — pushes to history, so undo covers slider drags too |
| Live preview | `project.preview()` — one ffmpeg call, independent of clip length |
| Export with a bar | `project.export(path, progress=fn)` — `fn` receives 0.0–1.0 |
| Provider dropdown | `available_providers()` |
| Where files landed | `project.cube_path` · `project.preview_path` · `project.state_dir` |

### Threading

Nothing is thread-safe and nothing needs to be — the objects are plain data.
Run `export` off the UI thread (it is the only call that takes real time) and
marshal the `progress` callback back yourself. `preview` is fast enough to call
synchronously on a keystroke; `plan_from_vibe` and `refine` block on a network
round trip, so those want a worker too.

## Errors are typed because a UI branches on them

Catch `RagvidError` to be sure you are handling a known failure rather than
swallowing a bug. Then branch:

| Class | What a UI should do | Carries |
|---|---|---|
| `InputError` | reopen the file picker | `path`, `reason` |
| `SessionNotFound` | offer "new project" | `root` |
| `NoGrade` | disable Export until something is planned | — |
| `SessionCorrupt` | offer "reset project" | `path`, `reason` |
| `ProviderNotConfigured` | open settings | `env_var` |
| `RateLimited` | show a countdown, retry | `retry_after` (seconds or None) |
| `ProviderError` | show the message | `provider` |
| `FFmpegError` | show the log, offer a bug report | `returncode`, `stderr` |
| `FFmpegNotFound` | tell the user to install it, or point `RAGVID_FFMPEG` at it | `binary`, `env_var`, `hint` |

Every field is populated at raise time so nothing has to be recovered by parsing
a message string.

## Where state lives

`<root>/.ragvid/` — `session.json` (source path, cached `ClipStats`, spec
history), `current.cube`, `preview.png`. One folder per project; delete it to
reset. `root` is explicit everywhere and defaults to the working directory,
which is what the CLI wants and what a GUI must override.

`ragvid serve` has no cwd to fall back on, so its default root is the per-user
data directory: `~/.local/share/ragvid/work` on Linux (XDG), `~/Library/
Application Support/ragvid/work` on macOS, `%APPDATA%\ragvid\work` on Windows.
That branch, and every other place the three platforms disagree, lives in
`ragvid/platform.py` — see it before adding a `sys.platform` check anywhere else.

The cached `ClipStats` is load-bearing: `refine` never re-probes the video, and
that is the entire reason the refine loop is sub-second rather than seconds.

## Gotchas worth knowing before you touch the internals

- **`spec.py` documents an evaluation order and the order is load-bearing:**
  CDL → white balance → saturation → contrast. `lut.py` bakes by calling
  `spec.apply()` rather than reimplementing it, and a test asserts every other
  permutation differs materially.
- **`.cube` varies RED fastest.** Reversing it produces a plausible-looking file
  that grades channels wrongly.
- **Statistics are display-space, not linear.** `spec.apply()`, the LUT and
  ffmpeg's `lut3d` all operate on display values, so `probe` must match them.
  An earlier version computed linear-light moments and every reference match
  came out roughly 4× too dark with the hue destroyed.
- **GIF output takes its own path.** The encoder wants `pal8`, so the
  `-pix_fmt yuv420p` pin that H.264 needs is an invalid argument there.
- **Groq's `strict` JSON schema does not enforce array `minItems`.** That is why
  RGB triples are objects with required `r`/`g`/`b` rather than 3-element
  arrays — models reliably emit 1-element arrays and the request then fails
  validation.

## Testing

`tests/conftest.py` generates its media fixtures with ffmpeg, so no binaries are
committed and a fresh clone runs the suite with no setup. Tests never call a
live LLM — the free Groq tier is 8000 tokens/min and parallel tests would
exhaust it. Mock the provider.
