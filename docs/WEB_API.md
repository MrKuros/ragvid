# Local web API

The contract between `ragvid/server.py` and `ragvid/web/index.html`. Both are
written against this; neither may change it unilaterally.

Single user, loopback only, no auth. `ragvid serve` starts it and opens a browser.

## Principles

- **The server holds one `Project`.** The browser holds no grading state of its
  own — it renders whatever `/api/state` returns. No client-side spec math.
- **Every mutating call returns the new full state**, so the client never has to
  re-fetch or reconcile. One request, one render.
- **`api_version` stamps the contract itself.** `index.html` carries the same
  constant and warns when they differ. The HTML is re-read on every request but
  the Python is imported once at startup, so an old `ragvid serve` will happily
  serve a new page and then answer with a contract it does not implement —
  fields silently missing, routes 404ing, and nothing on screen to explain it.
  Bump it whenever the JSON shape changes.
- **`version` increments on every mutation.** Append it to media URLs as a cache
  buster; without it the browser will happily show a stale frame after a refine,
  which is the single most confusing bug this UI can have.

## State

`GET /api/state` → `200`

```json
{
  "open": true,
  "source": "/abs/path/clip.mp4",
  "name": "clip.mp4",
  "duration": 4.0,
  "planned": true,
  "can_undo": true,
  "history_depth": 3,
  "steps": [{"index":0,"label":"warm and nostalgic","rationale":"...","current":false}],
  "intent": {"strength": "full",
             "ops": [{"op":"warmth","dir":"up","amount":"moderate","target":"",
                      "text":"warmed it up"}]},
  "auto_balance": true,
  "balance": "neutralised a green cast, set the black point",
  "api_version": 13,
  "version": 8,
  "spec": { "slope": {"r":1,"g":1,"b":1}, "offset": {...}, "power": {...},
            "saturation": 1.0, "saturation_balance": 0,
            "temperature": 0, "tint": 0,
            "contrast": 0, "pivot": 0.435, "contrast_balance": 0,
            "exposure": 0, "look_mix": 1.0, "highlight_rolloff": 0,
            "shadow_tint": {"r":0,"g":0,"b":0}, "highlight_tint": {...},
            "shadow_lift": 0, "highlight_lift": 0,
            "hue_red": {"sat":1.0,"lum":0,"rot":0}, "hue_yellow": {...},
            "hue_green": {...}, "hue_cyan": {...}, "hue_blue": {...},
            "hue_magenta": {...},
            "effects": {"denoise":0,"glow":0,"softness":0,
                        "grain":0,"vignette":0,"fringe":0},
            "rationale": "..." },
  "layers": [{"region": {"shape": "linear", "edge": "top", "extent": 0.4,
                         "cx": 0.5, "cy": 0.5, "rx": 0.5, "ry": 0.5,
                         "softness": 0.4, "invert": false},
              "spec": { "...one whole GradeSpec..." }}],
  "input_lut": null,
  "input_format": null,
  "stats": { "mean": {"r":0.2,"g":0.15,"b":0.15}, "saturation": 0.48,
             "width": 720, "height": 405, "frames_sampled": 10 },
  "providers": ["groq", "anthropic", "openai", "..."],
  "provider": "groq",
  "model": "openai/gpt-oss-120b",
  "configured": true
}
```

When nothing is open: `{"open": false, "providers": [...], "provider": "groq",
"configured": false}`.

`configured` says whether the active provider has a key. It is repeated here,
rather than left to `/api/providers`, so an opening screen can prompt for a key
without a second round trip — a first-time user does not know one is needed, and
finds out by typing a mood and getting an error.
When open but not yet graded: `"planned": false`, `"spec": null`.

### Regional layers

`layers` is the spatial half of the grade (roadmap B1) and is `[]` for the flat
grade almost every look is. Each entry is one whole `GradeSpec` plus the `Region`
saying where it lands, and they compose **in list order**, each grading the
accumulated result: `out = out + (layer.spec(out) - out) * layer.region.mask()`.
That is `region.GradeStack.apply` and `render._region_filters`, and a client
reproducing the picture has to keep the order.

`region` is closed-form geometry in frame-relative coordinates, so the same
fields describe a 4K export and a 480p preview. `shape` selects which fields are
read — `edge`/`extent` for `linear`, `cx`/`cy`/`rx`/`ry` for `radial` — and
`softness` (falloff width, 0 = hard) and `invert` apply to both. The falloff is
`smoothstep`, sampled at pixel centres. **Treat `shape` as an allow-list**: more
shapes will land, and one a client cannot evaluate must send it back to
`/media/frame` rather than be approximated.

A grade with layers is colour-only in `/media/cube`, which serves the **base**
alone. `/media/cube?layer=<n>` is the rest of it.

### The intent object

`intent` is the typed verbs the current grade was compiled from — the whole
point of the intent path (`ragvid/intent.py`, roadmap A1). It is **`null`
whenever there are none**, which is the common case, not an edge one: a photo
match, a refine, a moved slider and any provider on the direct path all produce
a spec no verb list describes. A client must fall back to `spec.rationale` and
must never render an empty list.

| Key | Meaning |
|---|---|
| `strength` | `"subtle"` / `"moderate"` / `"strong"` / `"full"` — how much of the whole look survives. It is what `spec.look_mix` was compiled from (0.4 / 0.65 / 0.85 / 1.0). |
| `ops[]` | One requested move each: `op` (one of sixteen verbs), `dir` (`"up"`/`"down"`), `amount` (`"subtle"`/`"moderate"`/`"strong"`), `target` (`""` or a colour). |
| `ops[].text` | The move as English, e.g. `"cooled it down"` — server-rendered, so a client never has to know the vocabulary. It carries **no** magnitude; the magnitude is `amount`, and printing both means they disagree for as long as a drag lasts. |

`text` is the one key `POST /api/intent` does not read. It is ignored on the way
back in, so a client posts the object it was given with one `amount` changed
rather than reconstructing anything.

**There are no floats anywhere in an intent, deliberately.** Every number in the
grade is computed from the clip's measured statistics by `compiler.py`. A client
that wants to change how far a move goes changes `amount`, not a spec field —
that is what makes the change land correctly on *this* footage.

### Auto-balance

`auto_balance` is one boolean per project, on by default, persisted in the
session. When on, the clip is neutralised from its own measurements — cast and
black point — *before* the creative look, so the same sentence lands the same
way on two differently-lit shots.

`balance` is what that pass does to **this** clip, in its own words, or `""`
when there is nothing to correct. It is computed from the measurements alone,
so it is present whether the switch is on or off — a client shows it as the
report when on, and as what would happen when off.

Two things a client must not get wrong:

- It applies to the **intent path only**. Auto-balance is a compiler pass and
  the direct path has no compiler, so on a `json_object` endpoint the flag is
  stored and does nothing. Do not offer it as a control that appears to work
  there.
- **It has to be visible.** A correction nobody asked for that silently fights
  the user's own grade is worse than no correction. `balance` is the sentence
  that prevents that; show it, and put the off switch on it.

### The spec object

25 keys: 51 numbers and a `rationale` string. Every value shown above **is** the
identity value, so a client can render "is this field moved?" by comparing
against the literal defaults rather than tracking a baseline.

| Group | Keys | Notes for a UI |
|---|---|---|
| CDL core | `slope` `offset` `power` `saturation` `saturation_balance` `temperature` `tint` `contrast` `pivot` `contrast_balance` | the two `_balance` keys aim `saturation` and `contrast` at one end of the tone scale; each does nothing on its own, so a UI showing one has to show what it scales |
| Tone | `exposure` `look_mix` `highlight_rolloff` | `exposure` in stops; `look_mix` 0–1 fades the whole look back to source; `highlight_rolloff` 0 = hard clip |
| Tonal split | `shadow_tint` `highlight_tint` `shadow_lift` `highlight_lift` | tints are luma-stripped, so a tint slider never changes brightness and a lift slider never changes colour — the two axes are independent by construction, and a UI can show them as such |
| Hue qualifiers | `hue_red` `hue_yellow` `hue_green` `hue_cyan` `hue_blue` `hue_magenta` | each `{"sat": 1.0, "lum": 0.0, "rot": 0.0}`, centred at 0/60/120/180/240/300°. `rot` turns that hue, in degrees; it is the one field the planning schema does not expose, so only the intent path writes it |
| Effects | `effects` | six spatial filters — **not** part of the colour transform |

`slope`/`offset`/`power` and both tints are `{"r","g","b"}` objects, and hue
bands are `{"sat","lum","rot"}` objects, never arrays. That is not style: Groq's
strict `json_schema` does not enforce array `minItems` during constrained
generation, models reliably emit 1-element arrays, and the request then fails
validation. Do not "simplify" them client-side either — `POST /api/spec` feeds
straight into the same pydantic model.

**`effects` never reaches the LUT.** `GradeSpec.apply()` ignores it and
`bake_cube` never sees it; ffmpeg applies those six as filters at render time.
So `GET /media/cube` is a *colour-only* file — a "download LUT" button should
say so when any `effects` value is non-zero, or the user takes a `.cube` into
Resolve and quietly loses their grain and vignette.

Two more consequences of the wider spec, both easy to get wrong:

- **`POST /api/spec` takes the whole object, not a patch.** Missing keys fall
  back to pydantic defaults, which are the *identity* values — not the current
  ones. Posting `{"spec": {"saturation": 1.1}}` therefore resets exposure,
  contrast, every hue band and every effect to identity. Send the `spec` you
  got from `/api/state` with your one field changed. Unknown keys are ignored
  rather than rejected, so a typo'd field name silently does nothing.
- **`steps[]` entries carry `index`, `label`, `rationale`, `current`, plus **`intent`**
(that step's own verbs, so a UI can show what any entry did without restoring it
first) and **`tweak`** (true when the entry is an adjustment OF the one before it
— a slider drag, an item switched off, the balance toggle — rather than
something that was asked for).

**B3 added `contrast_balance`, `saturation_balance` and a `rot` on each hue band, and did NOT move `api_version` for any of them, deliberately.**
The gate exists so a stale page announces itself instead of silently dropping
fields — and nothing is dropped here: the page enumerates no spec fields, and
its one spec write (`POST /api/spec` from the Strength slider) spreads the
server's own object, so a new key round-trips through an old page byte for
byte. Bumping would have blanked every open tab for a change no client can
observe. Bump it when a client could be wrong, not when the object grew.

`api_version` moves whenever this shape moves.** The spec grew from 8 keys
  to 23, so a client written against the older contract will render a grade it
  cannot fully represent and will round-trip identity into every new field the
  moment it posts. Check `api_version` before trusting `spec`.

## Mutations

All return the same state object as `GET /api/state`.

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/project` | multipart `file`, or `{"path": "/abs/path"}` | Opens a clip, **reopening its session when there is one**. Each clip has its own state directory under the work dir, so opening a second clip does not touch the first one's history, and reopening a clip (or restarting the server) brings its grades back. A damaged session answers **409 `SessionCorrupt`** with `path` and `reason`. Uploads land in the work dir. |
| `POST` | `/api/vibe` | `{"vibe": "gloomy"}` | Needs a provider key. Slow (network). |
| `POST` | `/api/reference` | multipart `file`, or `{"path": "..."}` | Offline, no key, fast. |
| `POST` | `/api/look` | multipart `file`, or `{"path": "/abs/x.look.json"}` | Applies a `look.json` written beside somebody else's export. Offline, no key, fast. It re-compiles the look's **intent** against THIS clip's statistics rather than copying its numbers: those were measured off the clip that was exported, auto-balance included, so pasting them onto differently lit footage is a LUT copy wearing a better name. A version-2 file (or any look with no intent behind it — a photo match, a hand-edited spec) has only numbers to give, so it falls back to `/api/spec` behaviour and **flattens**: the regional layers go, exactly as `refine` does on a provider that cannot constrain decoding. A file that is not a readable look is a `400`. |
| `POST` | `/api/refine` | `{"instruction": "less blue"}` | Requires `planned`. Slow (network). |
| `POST` | `/api/intent` | `{"intent": {"ops":[...], "strength":"full"}}` | The per-item strength path. Re-**compiles** the grade from the verbs against the cached stats — fast, no network. Send back the `intent` from `/api/state` with one `amount` changed, or with an op dropped to remove that move. A verb outside the vocabulary is a `400`. |
| `POST` | `/api/spec` | `{"spec": {...}}` | The raw-spec path: the only way to reach a field no verb covers (`pivot`, per-band `lum`). Fast, no network. Full spec object — omitted keys reset to identity, this is not a patch. **Drops the intent**, since 44 numbers are not described by any verb list; `/api/state` then returns `"intent": null`. |
| `POST` | `/api/balance` | `{"on": true}` | Turns auto-balance on or off. Re-compiles the current grade when there is an `intent` behind it, so the switch is something you see; a spec from anywhere else is left alone, because a balance already baked into 44 numbers cannot be decomposed back out. |
| `POST` | `/api/undo` | — | Steps back one, *including* undoing the first grade back to the ungraded clip. `409` only when there is nothing left. |
| `POST` | `/api/close` | — | Drops the project; back to the empty state. |
| `POST` | `/api/restore` | `{"index": 2}` or `{"index": 2, "intent": {...}}` | Puts an earlier step back at the END of the history. **Deletes nothing** — this is what clicking a history entry does now. The optional `intent` restores that step with an edit already applied, which is one call and one history row for one gesture (dragging a slider on a step that is not the current one). |
| `POST` | `/api/step/delete` | `{"index": 2}` or `{"index": 2, "count": 3}` | Removes `count` entries starting at `index`, leaving the rest in order. The built-in UI shows one row per PROMPT with that prompt's tweaks folded into it, so deleting a row deletes the run. |
| `POST` | `/api/revert` | `{"index": 0}` | Jump back to any step. `-1` is the ungraded clip. Undo is `revert` to the previous index. |
| `POST` | `/api/reset` | — | Discards every grade, keeps the clip open. The "start over" button — distinct from undo (one step) and close (drops the clip). |

## Browsing — a visual picker

`GET /api/browse?path=<dir>` → `200`

```json
{"path": "/home/me/clips", "parent": "/home/me", "home": "/home/me",
 "trail": [{"name":"/","path":"/"}, {"name":"home","path":"/home"}, ...],
 "dirs":  [{"name":"raw","path":"/home/me/clips/raw"}],
 "files": [{"name":"a.mp4","path":"/home/me/clips/a.mp4","kind":"video","size":812634}]}
```

`kind` is `video` or `image`; other files and dotfiles are omitted. A path that
points at a file lists its containing folder instead. Defaults to `$HOME`.

This exists because a browser file input cannot report a real path and cannot
pick a folder at all — so for a local tool it is the only way to open a large
clip in place rather than uploading a copy of it.

## Frames — the check-before-you-render path

`GET /media/frame?t=<seconds>&graded=<0|1>&v=<version>` → `200 image/png`

One frame at `t`, graded or not. ~190ms regardless of clip length, so it is safe
to call on every scrubber drag. `graded=0` works even before anything is planned.

Frames and previews carry the six `effects` as well as the colour — `project.py`
passes `spec.effects` into `render_frame`, `render_preview` and `render_video`
alike, so grain and vignette are in the check-before-you-render frame and not
just in the export. `graded=0` deliberately passes no effects at all: the
"before" half of a compare is the untouched source.

This exists so a full export is never how someone finds out a grade is wrong.

`GET /media/cube?v=<version>` → the `.cube` LUT, `text/plain`, for download.
Colour only — see above — and 33³ or 65³ depending on whether the grade uses a
hue qualifier, so the response is ~1 MB or ~7.4 MB. Do not assume a fixed size.

`GET /media/cube?input=1` → the camera log conversion in force instead, same
media type. `404 NotFound` when there is none.

`GET /media/cube?layer=<n>&v=<version>` → regional layer `n`'s own `.cube`, same
media type, indexed into `state.layers` in evaluation order. `404 NotFound` for
an index the grade does not have (a stale page, not a crash) and `400 InputError`
for one that is not an integer. It is an **index**, never a filename: nothing a
caller sends reaches the filesystem, the same rule `/media/source` keeps by
taking no parameter at all.

All three exist because the ffmpeg chain is a stack of `lut3d` nodes — the
technical conversion, then the grade, then one masked node per layer — and
anything reproducing that picture needs every one of them.

No mask image is served, and none is needed: a `Region` is closed-form geometry
and `state.layers[n].region` carries the six fields it takes to evaluate. The
PNG `render.py` blends with exists because ffmpeg needs a file.

## The source clip — the in-browser preview path

`GET /media/source` → the open clip's own bytes, `Accept-Ranges: bytes`.

**No parameter.** The path served is the `Project`'s, the same one `/media/frame`
already renders from, so there is nothing caller-supplied to sanitise and this
cannot be aimed at another file on the machine — a deliberately narrower rule
than `POST /api/project`, where a path *is* something the user chose.

Byte ranges are answered (`206` with `Content-Range`, `416` when unsatisfiable)
because a `<video>` cannot seek without them. An open-ended range (`bytes=0-`,
which is what a video element sends) is capped at 4 MB: serving fewer bytes than
were asked for is what the mechanism is for, and the element simply asks again.

This exists so the display path can skip ffmpeg entirely: `index.html` decodes
the clip in a `<video>`, uploads `/media/cube` (and `?input=1`) as 3D textures
and samples them in a fragment shader, tetrahedrally, the way `lut3d` does.
Scrubbing then costs nothing and playback is possible at all.

Regional layers ride along in the same shader: one more 3D LUT per layer, one
more `mix(c, lut(c), mask(uv))`, in `state.layers` order — `region.py`'s
evaluation order and `render._region_filters`' composite, which are the same
thing. The mask is rebuilt from `region` analytically, at pixel centres,
`smoothstep` falloff.

It is used **only** when the browser can produce the same picture the export
will:

- WebGL2 present and the shader compiles;
- the codec is one this browser decodes;
- `spec.effects` are all zero. The six are an ffmpeg filter chain — `hqdn3d`,
  `gblur`, `cas`, ffmpeg's own noise PRNG — and are not in the cube;
- every `layers[n].region.shape` is one the shader can evaluate in closed form
  (`linear`, `radial`), and there are at most six layers. An **allow-list**: a
  shape a client does not recognise — roadmap B2's semantic masks come out of a
  segmentation model, not out of `cx`/`cy`/`rx`/`ry` — must fall back, not be
  drawn as something it is not.

Anything failing falls back to `/media/frame` rather than showing a different
picture from the one that will be written. A client that skips that check is
lying to its user.

## Export

`POST /api/export` `{"out": "/abs/out.mp4", "gpu": false}` → `202 {"job": "j3"}`

Runs on a worker thread. Poll:

`GET /api/export/<job>` → `200`

```json
{"state": "running", "progress": 0.42, "path": null, "error": null}
```

`state` is `running` | `done` | `error`. `progress` is 0.0–1.0. On `done`,
`path` is the written file. On `error`, `error` is the object below.

Poll every ~300ms. Only one export runs at a time; a second `POST` while one is
running returns `409`.

`gpu` swaps in a hardware H.264 encoder and falls back to libx264 with a warning
when none works. It is worth far less than a UI checkbox implies once `effects`
are in play: measured at 1080p30, the full effect stack runs 0.42× realtime on
libx264 and 0.48× on NVENC, because the filters — not the encoder — are the
ceiling. Colour-only exports run 1.38×. Size the progress bar's ETA accordingly.

## The segmentation model

A sentence that names a **thing** in the picture ("the sky", "the person") is
resolved by a local model that is not installed by default. Every mutation that
would compile such a region is refused **before** it lands, with `428
SegmentUnavailable` — the grade never reaches the history, so the session keeps
rendering whatever it had.

The error's `needs_install` says which of the two preconditions is missing, and
the fixes are not the same:

| `needs_install` | What is missing | Client should |
|---|---|---|
| `true` | the optional extra | show `pip install 'ragvid[masks]'`. **Offer no button** — nothing over HTTP can install a Python package. |
| `false` | the 15 MB weights | offer the download below |

`POST /api/segment/download` → `202`

```json
{"state": "running", "progress": 0.0, "error": null}
```

The consent gate: posting here *is* the consent, and it is the only download in
the app. Runs on a worker thread and is polled exactly like an export. One at a
time — a second `POST` while one runs is `409 ExportBusy`. It returns `428` with
`needs_install: true` when the extra is missing, because the weights are useless
without it.

`GET /api/segment/download` → `200`

```json
{"state": "idle", "progress": 0.0, "error": null, "ready": false}
```

`state` is `idle` | `running` | `done` | `error`; `error` is the object under
[Errors](#errors). `ready` answers the question a client actually has — can a
semantic region be resolved right now — and is `true` with no job at all on a
machine that already has the weights.

Once it is `done`, re-send the request that was refused. Nothing is retried
automatically: the user asked for a grade, not for a download that then grades.

## Providers and keys

`GET /api/providers` → `200`

```json
{"providers": [
  {"name": "groq", "label": "Groq", "model": "openai/gpt-oss-120b",
   "needs_key": true, "env_var": "GROQ_API_KEY",
   "configured": true, "hint": "…7f2a", "source": "environment",
   "structured": "json_schema", "enforces_schema": true,
   "keys_url": "https://console.groq.com/keys",
   "note": "Fast and free to start. Enforces the schema, so grades come back complete.",
   "active": true}
]}
```

`enforces_schema` is the field a UI should surface. Filling 43 required numbers
reliably needs strict JSON-schema decoding; providers without it are labelled
best effort and may return an incomplete grade, which the server rejects with a
`ProviderError` naming the missing fields rather than filling them with
defaults.

`source` is where the key came from — `"settings"`, `"environment"`, or absent.
`hint` is the last four characters and is the **only** key-shaped value any
route ever returns. There is no endpoint that reads a key back.

`POST /api/provider` `{"provider": "ollama", "model": "llama3.1"}` → the new
state. `model` is optional; omit it for the provider's default, or pass `""` to
restore it.

`POST /api/key` `{"provider": "openai", "key": "sk-…"}` stores a key;
`{"provider": "openai", "key": null}` forgets it. The response is the provider
list again, so a UI re-renders from one round trip. The stored file is created
`0600` before it holds anything.

Keys set here take precedence over the environment and over `.env`, because a
key typed into the app should win over ambient configuration.

Note that `POST /api/key` names its own provider and `POST /api/provider` is a
separate call: storing a key does not switch to that provider. Keep those two
apart in the UI as well. ragvid's own settings panel once aimed its key box at
whichever provider was *active*, so adding a key for a second service meant
selecting it first — which switched grading to a service that had no key yet
and left the app unusable until the key was typed.

## Log footage

`POST /api/input_lut` `{"format": "slog3"}` names the camera's recording curve
and ragvid bakes the conversion itself — one of `slog3`, `vlog`, `clog3`,
`logc3`, `nlog` (`ragvid.logspace.NAMES`; display names such as "Sony S-Log3"
are the UI's business, not this contract's). `{"path": "/luts/vendor.cube"}`
uses a vendor file instead, for the minority who have one. Either key, null or
absent, clears. Returns the new state.

State reports both: `input_lut` is always the `.cube` actually in force, and
`input_format` is the format name when ragvid generated that file, `null` when
it is the user's own. A generated cube lands in the session dir beside
`current.cube` and is re-baked (67 ms, measured) on every change, so it is
derived data and nothing has to invalidate it.

Opening a clip asks `logspace.detect()` first. It answers `null` for almost
every file on purpose — a camera *make* is not evidence of a picture profile,
and H.273 has no code point for any log curve — so on the rare clip that carries
an explicit tag the format simply arrives already set, and on every other clip
nothing changes.

It is applied before the grade — a creative look sits on top of the conversion,
never mixed into it — and before the clip is measured. Setting one **re-probes**,
which is the slow part of this route: every statistic describes the image the
grade will land on, and a conversion changes all of them (measured on simulated
S-Log3: mean 0.50 → 0.75, std 0.11 → 0.25, p99 0.65 → 1.00). Without it the model
is told the clip is flat and grey and answers with an invented contrast push.

`/api/browse` lists `.cube` files with `"kind": "lut"` so a picker can offer them.

The grade LUT at `/media/cube` remains display-referred and does **not** include
the conversion. A `.cube` taken into Resolve still expects converted input, and
`/media/cube?input=1` is how a client gets the conversion itself.

## Errors

Any non-2xx carries:

```json
{"error": {"type": "RateLimited", "message": "groq: rate limit reached — retry in 54s",
           "retry_after": 54.0}}
```

`type` is the exception class name. Extra keys are the fields that class carries
(`retry_after`, `env_var`, `path`, `reason`, `root`, `returncode`,
`needs_install`, `hint`). The client
branches on `type` and shows `message`; it must never parse `message`.

| `type` | Status | Client should |
|---|---|---|
| `InputError` | 400 | reopen the picker |
| `NoGrade` | 409 | prompt to grade first |
| `SessionNotFound` | 404 | show the empty state |
| `ProviderNotConfigured` | 428 | explain which env var is missing |
| `SegmentUnavailable` | 428 | branch on `needs_install`; see [the segmentation model](#the-segmentation-model) |
| `RateLimited` | 429 | show a countdown from `retry_after` |
| `ProviderError` | 502 | show the message |
| `FFmpegError` | 500 | show the message, offer the log |
| anything else | 500 | generic message |
