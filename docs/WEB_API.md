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
  "api_version": 2,
  "version": 7,
  "spec": { "slope": {"r":1,"g":1,"b":1}, "offset": {...}, "power": {...},
            "saturation": 1.0, "temperature": 0, "tint": 0,
            "contrast": 0, "pivot": 0.435,
            "exposure": 0, "look_mix": 1.0, "highlight_rolloff": 0,
            "shadow_tint": {"r":0,"g":0,"b":0}, "highlight_tint": {...},
            "shadow_lift": 0, "highlight_lift": 0,
            "hue_red": {"sat":1.0,"lum":0}, "hue_yellow": {...},
            "hue_green": {...}, "hue_cyan": {...}, "hue_blue": {...},
            "hue_magenta": {...},
            "effects": {"denoise":0,"glow":0,"softness":0,
                        "grain":0,"vignette":0,"fringe":0},
            "rationale": "..." },
  "stats": { "mean": {"r":0.2,"g":0.15,"b":0.15}, "saturation": 0.48,
             "width": 720, "height": 405, "frames_sampled": 10 },
  "providers": ["groq", "anthropic"],
  "provider": "groq"
}
```

When nothing is open: `{"open": false, "providers": [...], "provider": "groq"}`.
When open but not yet graded: `"planned": false`, `"spec": null`.

### The spec object

23 keys: 43 numbers and a `rationale` string. Every value shown above **is** the
identity value, so a client can render "is this field moved?" by comparing
against the literal defaults rather than tracking a baseline.

| Group | Keys | Notes for a UI |
|---|---|---|
| CDL core | `slope` `offset` `power` `saturation` `temperature` `tint` `contrast` `pivot` | unchanged since api_version 1 |
| Tone | `exposure` `look_mix` `highlight_rolloff` | `exposure` in stops; `look_mix` 0–1 fades the whole look back to source; `highlight_rolloff` 0 = hard clip |
| Tonal split | `shadow_tint` `highlight_tint` `shadow_lift` `highlight_lift` | tints are luma-stripped, so a tint slider never changes brightness and a lift slider never changes colour — the two axes are independent by construction, and a UI can show them as such |
| Hue qualifiers | `hue_red` `hue_yellow` `hue_green` `hue_cyan` `hue_blue` `hue_magenta` | each `{"sat": 1.0, "lum": 0.0}`, centred at 0/60/120/180/240/300° |
| Effects | `effects` | six spatial filters — **not** part of the colour transform |

`slope`/`offset`/`power` and both tints are `{"r","g","b"}` objects, and hue
bands are `{"sat","lum"}` objects, never arrays. That is not style: Groq's
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
- **`api_version` moves whenever this shape moves.** The spec grew from 8 keys
  to 23, so a client written against the older contract will render a grade it
  cannot fully represent and will round-trip identity into every new field the
  moment it posts. Check `api_version` before trusting `spec`.

## Mutations

All return the same state object as `GET /api/state`.

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/project` | multipart `file`, or `{"path": "/abs/path"}` | Opens a clip. Uploads land in the work dir. Replaces any open project. |
| `POST` | `/api/vibe` | `{"vibe": "gloomy"}` | Needs a provider key. Slow (network). |
| `POST` | `/api/reference` | multipart `file`, or `{"path": "..."}` | Offline, no key, fast. |
| `POST` | `/api/refine` | `{"instruction": "less blue"}` | Requires `planned`. Slow (network). |
| `POST` | `/api/spec` | `{"spec": {...}}` | The slider path. Fast, no network. Full spec object — omitted keys reset to identity, this is not a patch. |
| `POST` | `/api/undo` | — | Steps back one, *including* undoing the first grade back to the ungraded clip. `409` only when there is nothing left. |
| `POST` | `/api/close` | — | Drops the project; back to the empty state. |
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

This exists so a full export is never how someone finds out a grade is wrong.

`GET /media/cube?v=<version>` → the `.cube` LUT, `text/plain`, for download.
Colour only — see above — and 33³ or 65³ depending on whether the grade uses a
hue qualifier, so the response is ~1 MB or ~7.4 MB. Do not assume a fixed size.

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

## Errors

Any non-2xx carries:

```json
{"error": {"type": "RateLimited", "message": "groq: rate limit reached — retry in 54s",
           "retry_after": 54.0}}
```

`type` is the exception class name. Extra keys are the fields that class carries
(`retry_after`, `env_var`, `path`, `reason`, `root`, `returncode`). The client
branches on `type` and shows `message`; it must never parse `message`.

| `type` | Status | Client should |
|---|---|---|
| `InputError` | 400 | reopen the picker |
| `NoGrade` | 409 | prompt to grade first |
| `SessionNotFound` | 404 | show the empty state |
| `ProviderNotConfigured` | 428 | explain which env var is missing |
| `RateLimited` | 429 | show a countdown from `retry_after` |
| `ProviderError` | 502 | show the message |
| `FFmpegError` | 500 | show the message, offer the log |
| anything else | 500 | generic message |
