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
            groq · openai · anthropic · 8 more · custom
                                 │
                                 ▼
                          spec.GradeSpec
             (43 numbers + a rationale — the only currency)
                                 │
          37 colour numbers ─────┴───── 6 spatial effects
                    │                          │
                    ▼                          ▼
              lut.bake_cube               render.py filtergraph
              33³ .cube (65³ if a         denoise · glow · softness ·
              hue band is in use)         grain · vignette · fringe
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

The two front ends are equals on grading only. API keys are entered, changed and
cleared solely in the GUI's Settings panel (`ragvid serve`); the CLI reads keys
from the settings file and the environment but never writes them.

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

## The `.cube` is colour-only — a seam you can see from outside

`GradeSpec.apply()`, and therefore the baked LUT, implements 37 of the 43
numbers. The six `effects` fields are spatial. A 3D LUT maps one RGB triple to
one RGB triple with no knowledge of the neighbouring pixel, so blur, grain,
vignette, glow and chromatic fringing cannot be expressed in one at all — not
approximately, structurally. They live in `render.py` as ffmpeg filters wrapped
around `lut3d`, and `apply()` ignores `spec.effects` completely.

That matters the moment a `.cube` leaves this tool. Take one into Resolve,
Premiere or OBS and you get the entire colour transform, exactly: exposure,
CDL, white balance, saturation, the six hue qualifiers, the shadow/highlight
split, the highlight shoulder, the contrast S-curve and `look_mix`, in that
order. You get **none** of `effects`. So a spec with `grain` or `vignette` set
renders differently in Resolve than ragvid's own export of the same spec, and
nothing in the file says so — a `.cube` has no place to record what it left
out. Read `effects` out of `session.json` before assuming a `.cube` round-trips
the look, and rebuild those six with the host's own nodes.

Second surprise for a host application: the LUT is 33³ normally but **65³
whenever any hue band is off identity** (see the gotcha below), which takes the
file from 0.97 MB to 7.4 MB, measured. Anything that caps LUT size or uploads
LUTs will notice.

## Gotchas worth knowing before you touch the internals

- **`spec.py` documents an evaluation order and the order is load-bearing:**

  ```
  0 src captured → 1 exposure → 2 CDL → 3 white balance → 4 saturation
  → 5 hue qualifiers → 6 tonal split → 7 highlight rolloff → 8 contrast
  → 9 look_mix → 10 clip to [0,1]
  ```

  `lut.py` bakes by calling `spec.apply()` rather than reimplementing it, and a
  test asserts every other permutation differs materially. Three of those
  placements are counter-intuitive and each has a reason:

  - **Exposure before the CDL**, so `offset` stays an *absolute* lift instead
    of being scaled by the exposure move.
  - **Rolloff at 7, not at the end.** `_smoothstep(u) = u²(3-2u)` has
    derivative `6u(1-u)`, which is negative for `u > 1` — feed the contrast
    S-curve an out-of-range value and it is non-monotonic, i.e. brighter input
    maps to *darker* output, baked into the `.cube` forever. That is why
    `_s_curve` clips its input and why the clip is not removable. Everything
    that can exceed 1 must therefore sit before the shoulder, and the shoulder
    immediately before contrast. A shoulder at the final return would leave the
    clip that actually destroys highlights — the one inside step 8 — untouched.
  - **Hue qualifiers before the tonal split.** A qualifier reads a hue angle, so
    if the split injected its tint first, "teal shadows" would make the cyan
    qualifier fire on every shadow in the frame. A feedback loop, avoided by
    ordering rather than by a special case.

- **Highlight clipping was silently eating every "brighter" prompt.** Measured
  on a 4096-step ramp: `slope=1.3` — inside the 0.7–1.4 range `vibe.py` tells
  the model is sane — pins **23.1%** of the ramp at pure white; `slope=1.6`
  pins 37.5%. `highlight_rolloff` replaces that hard clip with an
  extended-Reinhard shoulder at step 7; the same ramp at `rolloff=0.3` pins
  **0.0%**. The price is not avoidable and is not a bug: no monotone `f` with
  `f(x) = x` on `[0,1]` and `f ≤ 1` can also roll off, so a real shoulder must
  pull legal white down. Measured `f(1)` = 0.976 / 0.928 / 0.856 / 0.760 at
  rolloff 0.1 / 0.3 / 0.6 / 1.0. Hence the default is 0, and 0 inserts *no
  code at all* rather than an "identity" shoulder.

- **Smoothstepped hue bands are necessary but not sufficient for 33³.** The six
  band weights are an exact partition of unity (measured within 2e-15 of 1 over
  1e4 hues), so adjacent band settings interpolate C1 and no hue is unweighted
  or double-counted. That buys smoothness in *hue*; it buys nothing in *RGB*.
  `hue()` and `chroma()` have gradient kinks on the planes r=g, g=b, r=b, so a
  qualifier's output is C0-but-not-C1 inside the cube and trilinear
  reconstruction is only first-order accurate there — error ~1/n, not 1/n².
  Measured max error against exact `apply()` over 3×10⁵ random points, in 8-bit
  code values:

  | Grade | 33³ | 65³ |
  |---|---|---|
  | full stack, no qualifiers | 1.68 | 0.82 |
  | mild qualifiers (sat .82–1.15) | 1.66 | 0.77 |
  | strong (sat .65–1.30) | 1.80 | 0.88 |
  | extreme (sat .30–1.60) | 3.52 | 1.19 |

  So `bake_cube` escalates to 65³ when — and only when — a band is off identity.
  Widening the bands does not help; the kinks are intrinsic to hue selection.
  Raising the default instead would 8× every `.cube` for grades that never touch
  a qualifier.

- **The identity LUT is a bit-for-bit gate, not an `atol` one.**
  `GradeSpec.identity().apply(_grid(33))` must hash to
  `517467be3ba6b7a8afe71a05c847061dc597f0ea92e41b422164b579fbc74291`. Every
  step added since is wrapped in an explicit identity check that skips it
  entirely, because `L + (x-L)*1.0` and `src + (x-src)*1.0` are not bitwise
  identities in floating point (~1 ulp). An unguarded step would still pass
  `test_lut`'s `atol=1e-6` assertion while shifting every saved grade. Add a
  field, add its guard — and add it to `is_identity()`, or
  `GradeSpec(exposure=3).is_identity()` returns True and the tests assert
  nothing. `python -m ragvid.spec` runs that gate plus the order's invariants.
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
