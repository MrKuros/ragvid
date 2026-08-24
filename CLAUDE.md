# ragvid — working notes

Prompt-driven video colour grading. Type a sentence, get a graded clip.
Open source, MIT, https://github.com/MrKuros/ragvid.

This file is the orientation for anyone picking the project up cold. The three
docs it points at are the detail:

- `docs/ARCHITECTURE.md` — the rules the code is built on, and why
- `docs/ROADMAP.md` — every planned capability, marked `[have]`/`[partial]`/`[none]`, sorted by what it costs the interface
- `docs/WEB_API.md` — the HTTP surface

---

## The one constraint that decides everything

**The interface stays simple. The backend gets arbitrarily clever.**

A person opens a clip, types a sentence, sees a frame. No node graph, no
timeline, no scopes panel, no window-drawing tools, no 40-slider inspector.
Every advanced capability has to land somewhere the user never sees.

A change that makes the screen busier has failed, however well it works. When
a feature seems to need a control, the first question is whether the sentence
can carry it instead.

## How a sentence becomes a grade

```
prompt → LLM → Intent (typed verbs, no numbers) → compiler → GradeStack → .cube + ffmpeg filters
                                        ↑
                            measured ClipStats from the clip
```

The model **never emits a number**. It picks from a closed vocabulary of 17
verbs with a direction and a coarse amount (`subtle`/`moderate`/`strong`), and
deterministic code in `compiler.py` decides magnitudes by consulting what was
actually measured off the footage. `intent.py`'s schema is 834 bytes against
`GradeSpec`'s 2692.

This is the load-bearing decision in the project. Measured against the older
path where the model authored all 43 numbers: **10/10 prompts vs 8/10, 2.8×
cheaper** (`scripts/bakeoff_intent.py`, judged by applying each spec to real
pixels and re-measuring — not by reading spec fields).

The reason it matters is growth. Adding regions cost six vocabulary words. On
the old path it would have cost a schema the model could not fill — at ~200
fields, direct authoring does not degrade, it collapses.

`plan_vibe` routes on `provider.schema_enforced` — a capability test, not a
preference. Endpoints that cannot constrain decoding to a schema keep
`plan_direct`, which still works. Never coax JSON out of a weak endpoint and
hope: a malformed `Intent` compiles to something plausible and wrong, which is
worse than falling back.

## Module map

| File | Job |
|---|---|
| `spec.py` | `GradeSpec` — 44 numbers, and `apply()`. The evaluation order is load-bearing; the docstring explains every position. |
| `region.py` | `Region`, `Layer`, `GradeStack` — the container that lets a grade apply to part of a frame. |
| `intent.py` | The typed verb vocabulary the model emits, plus `describe()` which renders it as English. |
| `compiler.py` | `compile_stack` / `compile_intent` — verbs + measurements → numbers. Also auto-balance. |
| `probe.py` | Frame sampling and `ClipStats`. Display space, 16-bit, median across frames. |
| `match.py` | Closed-form reference matching. No LLM. The pattern the compiler extends. |
| `lut.py` | Bakes a spec to a `.cube`. 33³, escalating to 65³ when hue qualifiers are active. |
| `logspace.py` | S-Log3 / V-Log / C-Log3 / LogC3 / N-Log transfer functions and generated conversion LUTs. |
| `render.py` | The ffmpeg filter chain: technical LUT → grade → region layers → spatial effects. |
| `segment.py` | The local segmentation model behind semantic masks. Optional extra; `onnxruntime` is imported lazily. |
| `refine.py` | "Less blue" — edits the verb list on the intent path, the 44 numbers on the direct one. |
| `sidecar.py` | `look.json` (lossless) and `.cdl` (universal) written beside every export. |
| `vibe.py` | The system prompts and the planning entry points. |
| `looks.py` | The retrieval corpus, used by the direct path only. |
| `project.py` | Orchestrates probe → plan → bake → preview → export. |
| `session.py` | Persistence, history, undo. |
| `server.py` | The local HTTP server. `API_VERSION` currently **11**. |
| `web/index.html` | The whole UI, one file. WebGL preview + the "what it did" list. |

## Invariants — break these and something silently corrupts

**Identity must be bit-for-bit.** `GradeSpec.identity().apply(grid)` must equal
the grid exactly, and the baked identity table must keep hashing to
`517467be…`. Every operation is explicitly guarded to skip at identity, and
that is correctness, not optimisation: `L + (x-L)*1.0` is ~1 ulp off in
floating point, so an unguarded step breaks the hash gate while still passing
an `atol=1e-6` test.

**The evaluation order in `spec.py` is pinned by a real constraint.**
`_smoothstep`'s derivative goes negative above 1, so feeding the S-curve an
out-of-range value makes it non-monotonic — brighter input mapping to darker
output, baked permanently into the `.cube`. Every op that can exceed 1 sits
before the soft clip; the soft clip sits immediately before contrast. Change
the docstring in the same commit as the code, or not at all.

**`GradeSpec` is the currency of one correction; `GradeStack` is the frame.**
This rule was revised deliberately when regions landed — read the note in
`docs/ARCHITECTURE.md`. A flat grade is a stack with no layers, so a consumer
that only speaks `GradeSpec` still gets exactly one and its `.cube` is
unchanged byte for byte.

**Spatial things cannot bake into a `.cube`.** `EffectSpec` (grain, vignette,
glow, softness, denoise, fringe) and region masks are ffmpeg filters. That is
why `look.json` exists and why the WebGL preview *declines the job* — falls
back to the server-rendered frame — rather than showing a look the export will
not produce.

**The preview must match the export.** Every serious bug in this project's
history was a preview that disagreed with the file. `render_preview` grades
each tile before stacking for exactly this reason; read that comment before
touching the still path.

**`_lut_filter` returns the bare `lut3d` node.** `tests/test_platform.py`
asserts its exact string. New filters compose *around* it.

**The API version gate.** `server.API_VERSION` and `index.html`'s
`EXPECTED_API` must agree, and a test asserts it. Bump both when the shape
changes, so a stale page announces itself instead of silently dropping fields.

**No pixels leave the machine.** Planners — the things that cost money and
travel over a network — see statistics and words, never an image. Exactly one
model sees pixels and it runs locally: the segmentation model in `segment.py`.
The rule used to read "pixels never reach a model"; `docs/ARCHITECTURE.md`
rule 2 records what it was, why it stopped fitting, and why the new wording is
stricter in the direction that matters.

## Working rules

- **Tests must NEVER call a live LLM.** The Groq free tier is 8000 tokens/min. Mock the provider; autouse fixtures already make provider construction raise. The one sanctioned exception is `scripts/bakeoff_intent.py`, run by hand, sequential, ~25s between calls.
- **Never read, echo, log or commit `.env`.** Keys must not reach a log line, an error message or a report.
- **Commits use `mrkuros`, no email, no trailers**: `git -c user.name=mrkuros -c user.email= commit`.
- **The server binds 127.0.0.1 only.** No new listeners, no wider bind.
- **No broad process kills.** Target specific PIDs or ports — a wide `pkill` killed a running server here once.
- **Isolate live experiments**: set `XDG_DATA_HOME` and `--root` so the real session and settings are untouched. Forgetting this cost a real session once.
- Comments explain *why*. An empirical claim carries its measured number. Module docstrings state the load-bearing constraint up front — `spec.py` and `probe.py` are the house style.
- Fixes belong in the library, not the caller. A bug surfacing in `server.py` usually wants fixing in `project.py`.

## Verify by measuring, not by running

Green tests have hidden every serious bug in this project — the 4×-too-dark
colour space, the export rendering the wrong grade, the 0-byte GIF, the dead
undo. "No exception raised" and "the tests pass" are not evidence.

- Assert on **measured pixels**, not on spec field values. A field moving is not proof the image moved.
- When a claim is empirical, put the number in the comment and in the commit message.
- For a new check, **run it against the pre-change code and confirm it fails.** A harness that passes on both versions is testing nothing.
- For JS, the technique is: extract the `<script>` block, stub the DOM and `fetch`, assert under `node`. It has caught real bugs here.

## Running it

```
uv run ragvid serve              # the app, opens a browser
uv run ragvid serve --no-browser --port 8765
uv run pytest -q                 # 792 tests, ~2 min
./scripts/check.sh               # the gate: tests + invariants (--ci, --live)
```

Keys are entered in the GUI only. There is no CLI path for a key and no
`--key` flag — argv is visible in `ps` and shell history, so this is enforced
by construction.

## Where it stands

Tier A is complete. Tier C is complete. Tier B is complete except B3b
hue-vs-hue, B3c lum-vs-sat, B4 a real HSL qualifier and B5 keyframes. (B3 was
five curve types on paper and three of them turned out to be already shipped —
see the split rows in the roadmap.) Packaged for PyPI as 0.2.0
(`uv tool install ragvid`) — the wheel is verified from a clean venv, but the
tag has not been pushed and nothing is published yet.

A sentence can name **what** to do (18 verbs), **how much** (three amounts plus
a whole-look strength), **which pixels** — by colour (ten hue families,
including skin), by place (top/bottom/left/right/center/edges) or by thing
(sky/foliage/person/water/buildings, via the local segmentation model) — and
**what to spare** (`protect`). Refine edits that list rather than the numbers,
so a second sentence no longer destroys the first one's regions.

A sentence can also name **the shape of the tone curve**: `shadows down` bends
the toe instead of translating it, and `shoulder` is the top end. On rail-black
footage the old flat lift welded 7.37% of samples onto pure black at
"moderate"; the toe adds 0.00%. Only `strong` is allowed to spend the black
point — the one verb whose `amount` switches mechanism rather than magnitude.

Semantic masks are sampled **once**, not per frame: inference is 244 ms, so a
10-minute clip would cost 58 minutes of segmentation against a 14-minute
export. The honest consequence — a subject leaving frame keeps its grade — is
documented in `segment.py`.

Next: **B3b hue-vs-hue** and **B3c lum-vs-sat**, one compiler pass each. After
that the honest 1.0 blocker is not a feature: nobody has yet graded real footage with this,
and the CI matrix has only just gained Windows and macOS — whose first run
already turned up an `os.fchmod` call that made the app unconfigurable there.

## Things that bit us, so they don't again

- **swscale widens 8-bit by left shift, not replication** — 255 arrives as `65280`, not 65535. Dividing by 65535 put a 0.4% gain error on every statistic.
- **8-bit measurement of log footage lies**: `crushed_low` over-reported 2×, `p1` snapped to exactly 1/255. Those are the fields a shadow verb reads.
- **`maskedmerge` is per-plane** — a grey mask in yuv420p carries the ramp in Y and a flat 128 in U/V, so every chroma pixel blends at 50%.
- **`scale2ref` before `alphamerge` is not optional**: it refuses a size mismatch, and the mask is written at `ffprobe`'s dimensions while the frame is what the decoder produced. Those differ on any clip with rotation side data.
- **None of nvenc/qsv/amf encodes 10-bit H.264.** The bit-depth gate is `enc == "libx264"` specifically, not "not a hardware encoder".
- **Default args bind at def time** — patching a module-level path after import does not change a function that defaulted to it.
- **Windows PowerShell 5.1 has no `&&`.** One command per line in the README.
- **A `<dialog>` opened with `showModal()` sits in the browser top layer**, so a normal-flow error banner is invisible behind its backdrop.
- **`fetch()` cannot report upload progress**; `XMLHttpRequest.upload.onprogress` can. That is the one place XHR earns its keep.
- **FFV1 belongs in `.mkv`, not `.mp4`.** MP4 only learned to carry it recently, so a lossless test fixture written to `.mp4` passes locally and fails CI with `Could not find tag for codec ffv1`. More generally: a green local suite does not prove a green CI, because the runner's ffmpeg is older than yours.
