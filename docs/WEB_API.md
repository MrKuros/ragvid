# Local web API

The contract between `ragvid/server.py` and `ragvid/web/index.html`. Both are
written against this; neither may change it unilaterally.

Single user, loopback only, no auth. `ragvid serve` starts it and opens a browser.

## Principles

- **The server holds one `Project`.** The browser holds no grading state of its
  own — it renders whatever `/api/state` returns. No client-side spec math.
- **Every mutating call returns the new full state**, so the client never has to
  re-fetch or reconcile. One request, one render.
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
  "version": 7,
  "spec": { "slope": {"r":1,"g":1,"b":1}, "offset": {...}, "power": {...},
            "saturation": 1.0, "temperature": 0, "tint": 0,
            "contrast": 0, "pivot": 0.435, "rationale": "..." },
  "stats": { "mean": {"r":0.2,"g":0.15,"b":0.15}, "saturation": 0.48,
             "width": 720, "height": 405, "frames_sampled": 10 },
  "providers": ["groq", "anthropic"],
  "provider": "groq"
}
```

When nothing is open: `{"open": false, "providers": [...], "provider": "groq"}`.
When open but not yet graded: `"planned": false`, `"spec": null`.

## Mutations

All return the same state object as `GET /api/state`.

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/project` | multipart `file`, or `{"path": "/abs/path"}` | Opens a clip. Uploads land in the work dir. Replaces any open project. |
| `POST` | `/api/vibe` | `{"vibe": "gloomy"}` | Needs a provider key. Slow (network). |
| `POST` | `/api/reference` | multipart `file`, or `{"path": "..."}` | Offline, no key, fast. |
| `POST` | `/api/refine` | `{"instruction": "less blue"}` | Requires `planned`. Slow (network). |
| `POST` | `/api/spec` | `{"spec": {...}}` | The slider path. Fast, no network. Full spec object. |
| `POST` | `/api/undo` | — | `409` if already at the first grade. |
| `POST` | `/api/close` | — | Drops the project; back to the empty state. |

## Frames — the check-before-you-render path

`GET /media/frame?t=<seconds>&graded=<0|1>&v=<version>` → `200 image/png`

One frame at `t`, graded or not. ~190ms regardless of clip length, so it is safe
to call on every scrubber drag. `graded=0` works even before anything is planned.

This exists so a full export is never how someone finds out a grade is wrong.

`GET /media/cube?v=<version>` → the `.cube` LUT, `text/plain`, for download.

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
