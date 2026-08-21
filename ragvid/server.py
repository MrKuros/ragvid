"""The local web API — stdlib http.server, one Project, loopback only.

Implements docs/WEB_API.md. The browser holds no grading state: every mutation
returns the whole state object and bumps `version`, which the client appends to
media URLs so a stale frame can never survive a refine.

Deliberately small: a route table, one error handler that introspects the
exception, and a worker thread for the only slow call (export). No framework —
this is a single-user tool on 127.0.0.1.
"""

from __future__ import annotations

import errno
import json
import re
import threading
import traceback
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .errors import InputError, ProviderError, RagvidError, SessionNotFound
from .platform import data_dir, is_windows
from .project import Project
from .session import Session
from .spec import GradeSpec

HOST = "127.0.0.1"  # never 0.0.0.0: this API opens local files by path.
DEFAULT_PORT = 8765
MAX_UPLOAD = 512 * 1024 * 1024  # ponytail: uploads are read into memory; the
# {"path": ...} body is the route for anything big, so 512MB is a cliff not a limit.

# Bumped whenever the JSON contract changes shape. index.html carries the same
# number; a mismatch means the browser has fresh HTML (re-read per request) but
# the server is still running the Python it was started with -- which otherwise
# shows up as fields silently missing and routes 404ing, with nothing on screen
# to explain it.
API_VERSION = 4

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".mpg", ".mpeg", ".wmv", ".m2ts"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

INDEX = Path(__file__).parent / "web" / "index.html"
PLACEHOLDER = b"""<!doctype html><meta charset=utf-8><title>ragvid</title>
<body style="font:16px system-ui;padding:3rem;max-width:40rem">
<h1>ragvid</h1><p>The API is up. <code>ragvid/web/index.html</code> is not
written yet, so there is no UI to serve. Try <code>/api/state</code>.</p>"""


# ---- errors ---------------------------------------------------------------
# Three failures the web layer has that the library does not. Everything else
# comes straight from ragvid.errors.


class NotFound(RagvidError):
    """No such route or job."""


class NothingToUndo(RagvidError):
    """Already at the first grade."""


class ExportBusy(RagvidError):
    """One export at a time."""


# Looked up along the exception's MRO, so a new subclass of ProviderError (or of
# anything else here) gets the right status with no change to this file.
STATUS = {
    "InputError": 400,
    "NotFound": 404,
    "SessionNotFound": 404,
    "NoGrade": 409,
    "NothingToUndo": 409,
    "ExportBusy": 409,
    "ProviderNotConfigured": 428,
    "RateLimited": 429,
    "FFmpegError": 500,
    "ProviderError": 502,
}


def _jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


def _error_response(exc: BaseException) -> tuple[int, str, bytes]:
    """One handler for every failure: status from the MRO, body from the fields
    the exception happens to carry (retry_after, env_var, path, reason, ...)."""
    status = next((STATUS[c.__name__] for c in type(exc).__mro__ if c.__name__ in STATUS), 500)
    fields = {k: v for k, v in vars(exc).items() if not k.startswith("_") and _jsonable(v)}
    if not isinstance(exc, RagvidError):
        traceback.print_exception(exc)  # unknown failure: it is a bug, leave a trace
    body = {"error": {"type": type(exc).__name__, "message": str(exc), **fields}}
    return status, "application/json", json.dumps(body).encode()


# ---- server state ---------------------------------------------------------


class _State:
    """The whole server. One project, one lock, one export at a time."""

    project: Project | None = None
    version: int = 0
    root: Path = Path.cwd()
    exports: dict[str, dict] = {}
    n_exports: int = 0


S = _State()
# ponytail: one global lock. Mutations and frame renders serialize (~190ms each),
# which is fine for one browser; split it per-operation only if scrubbing stutters.
LOCK = threading.RLock()


def _work_dir() -> Path:
    # XDG on Linux, ~/Library/Application Support on macOS, %APPDATA% on Windows
    # -- see ragvid.platform.data_dir.
    return data_dir() / "work"


def _active() -> dict:
    """Which provider a grade would use right now, and with what model."""
    from .providers.base import active_choice, info_for

    name, model = active_choice()
    try:
        model = model or info_for(name).model
    except ProviderError:
        model = model or ""
    return {"provider": name, "model": model}


def _state_json() -> dict:
    from .providers.base import catalog

    base = {
        "api_version": API_VERSION,
        "version": S.version,
        "providers": [p.name for p in catalog()],
        **_active(),
    }
    p = S.project
    if p is None:
        return {"open": False, **base}
    stats = p.stats  # cached; never re-probes, and carries the duration
    return {
        "open": True,
        "source": p.source,
        "name": Path(p.source).name,
        "duration": stats.duration,
        "planned": p.is_planned,
        "can_undo": p.can_undo,
        "history_depth": len(p.history),
        "steps": p.steps,
        "spec": p.spec.model_dump() if p.is_planned else None,
        "stats": stats.model_dump(),
        **base,
    }


def _project() -> Project:
    if S.project is None:
        raise SessionNotFound(str(S.root))
    return S.project


def _ok(payload: dict, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(payload).encode()


def _mutated() -> tuple[int, str, bytes]:
    S.version += 1
    return _ok(_state_json())


# ---- request bodies -------------------------------------------------------


def _json_body(req) -> dict:
    if not req.body:
        return {}
    try:
        parsed = json.loads(req.body)
    except ValueError as exc:
        raise InputError("<body>", f"invalid JSON ({exc})") from exc
    if not isinstance(parsed, dict):
        raise InputError("<body>", "expected a JSON object")
    return parsed


def _multipart_file(ctype: str, body: bytes) -> tuple[str, bytes]:
    """Minimal multipart: first part that has a filename wins."""
    match = re.search(r'boundary="?([^";]+)"?', ctype)
    if not match:
        raise InputError("<upload>", "multipart body with no boundary")
    for part in body.split(b"--" + match.group(1).encode()):
        head, sep, data = part.partition(b"\r\n\r\n")
        name = re.search(rb'filename="([^"]*)"', head)
        if not sep or not name or not name.group(1):
            continue
        return name.group(1).decode("utf-8", "replace"), data[:-2] if data.endswith(b"\r\n") else data
    raise InputError("<upload>", "no file part in multipart body")


def _safe_name(name: str) -> str:
    """Basename only, and nothing that can climb out of the work dir."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name if name.strip("._") else "upload"


def _incoming_file(req, exts: set[str]) -> Path:
    """A multipart upload (saved into the work dir) or {"path": "/abs/path"}."""
    ctype = req.headers.get("Content-Type", "")
    if ctype.startswith("multipart/form-data"):
        raw_name, data = _multipart_file(ctype, req.body)
        name = _safe_name(raw_name)
        _check_ext(name, raw_name, exts)
        path = _work_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    raw = _json_body(req).get("path")
    if not raw or not isinstance(raw, str):
        raise InputError("<body>", 'expected a multipart file or {"path": "/abs/path"}')
    path = Path(raw).expanduser()
    _check_ext(path.name, str(path), exts)
    return path


def _check_ext(name: str, shown: str, exts: set[str]) -> None:
    if Path(name).suffix.lower() not in exts:
        raise InputError(shown, "not a supported media file (" + " ".join(sorted(exts)) + ")")


# ---- routes ---------------------------------------------------------------


def r_state(req, q):
    return _ok(_state_json())


def r_open(req, q):
    path = _incoming_file(req, VIDEO_EXT)
    with LOCK:
        project = Project.create(path, root=S.root)
        project.save()
        S.project = project
        return _mutated()


def r_vibe(req, q):
    vibe = str(_json_body(req).get("vibe") or "").strip()
    if not vibe:
        raise InputError("<body>", "missing 'vibe'")
    with LOCK:
        _project().plan_from_vibe(vibe)
        return _mutated()


def r_reference(req, q):
    path = _incoming_file(req, IMAGE_EXT)
    with LOCK:
        _project().plan_from_reference(path)
        return _mutated()


def r_refine(req, q):
    instruction = str(_json_body(req).get("instruction") or "").strip()
    if not instruction:
        raise InputError("<body>", "missing 'instruction'")
    with LOCK:
        _project().refine(instruction)
        return _mutated()


def r_spec(req, q):
    raw = _json_body(req).get("spec")
    if not isinstance(raw, dict):
        raise InputError("<body>", "missing 'spec' object")
    try:
        spec = GradeSpec(**raw)
    except Exception as exc:  # pydantic ValidationError -> the picker, not a 500
        raise InputError("<spec>", str(exc)) from exc
    with LOCK:
        _project().set_spec(spec)
        return _mutated()


def r_undo(req, q):
    with LOCK:
        if not _project().undo():
            raise NothingToUndo("already at the first grade — nothing to step back to")
        return _mutated()


def r_close(req, q):
    with LOCK:
        S.project = None
        return _mutated()


def r_revert(req, q):
    """Jump back to any point in the history. -1 is the ungraded clip.

    Clicking an entry in the history list is just a multi-step undo.
    """
    body = _json_body(req)
    try:
        index = int(body.get("index"))
    except (TypeError, ValueError) as exc:
        raise InputError("<index>", "expected an integer index") from exc
    with LOCK:
        if not _project().revert_to(index):
            raise InputError("<index>", f"no step {index} to go back to")
        return _mutated()


def r_reset(req, q):
    """Discard every grade, keep the clip. The 'start over' button.

    Distinct from undo (one step) and from close (drops the clip entirely).
    """
    with LOCK:
        _project().reset()
        return _mutated()


def _listing(path: Path) -> dict:
    """One directory, split into folders and openable media."""
    dirs, files = [], []
    try:
        entries = sorted(path.iterdir(), key=lambda e: e.name.lower())
    except (PermissionError, OSError) as exc:
        raise InputError(str(path), f"cannot list ({exc.__class__.__name__})") from exc

    for e in entries:
        if e.name.startswith("."):
            continue                      # dotfiles are noise in a picker
        try:
            if e.is_dir():
                dirs.append({"name": e.name, "path": str(e)})
            elif e.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
                files.append({
                    "name": e.name,
                    "path": str(e),
                    "kind": "video" if e.suffix.lower() in VIDEO_EXT else "image",
                    "size": e.stat().st_size,
                })
        except OSError:
            continue                      # a broken symlink shouldn't kill the listing
    return {"dirs": dirs, "files": files}


def r_browse(req, q):
    """Directory listing, so the UI can offer a visual picker instead of
    asking someone to type a path.

    A browser file input cannot give us a real path and cannot pick a folder at
    all, so for a local tool this is the only way to open a large clip in place
    rather than uploading a copy of it.
    """
    raw = _param(q, "path", "") or str(Path.home())
    path = Path(raw).expanduser()
    if not path.is_dir():
        path = path.parent if path.parent.is_dir() else Path.home()
    path = path.resolve()

    # Breadcrumb trail, so the UI can offer one click per ancestor.
    trail = [{"name": p.name or str(p), "path": str(p)}
             for p in reversed([path, *path.parents])]
    body = {
        "path": str(path),
        "parent": str(path.parent) if path.parent != path else None,
        "home": str(Path.home()),
        "trail": trail,
        **_listing(path),
    }
    return 200, "application/json", json.dumps(body).encode()


def _param(q, key, default):
    return q.get(key, [default])[0]


def r_frame(req, q):
    project = _project()
    try:
        at = float(_param(q, "t", "0") or 0)
    except ValueError as exc:
        raise InputError("t", f"not a number: {_param(q, 't', '')!r}") from exc
    graded = _param(q, "graded", "1") not in ("0", "false", "no", "")
    # Clamp: a scrubber pinned to the very end otherwise seeks past the last
    # frame and ffmpeg writes nothing.
    at = max(0.0, min(at, max(project.stats.duration - 0.05, 0.0)))
    with LOCK:
        return 200, "image/png", project.frame(at=at, graded=graded).read_bytes()


def r_cube(req, q):
    with LOCK:
        data = _project().bake().read_bytes()
    return 200, "text/plain; charset=utf-8", data


def _export_snapshot(project: Project) -> Project:
    """The Project the export thread renders from: its own root, so its own
    .cube, and a spec list frozen at the moment Export was pressed.

    Not paranoia — the live project re-bakes `current.cube` on every graded
    frame, and the UI invites scrubbing (and sliders) while an export runs. With
    one shared cube a slider nudge in the first second of an encode makes ffmpeg
    pick up the *new* grade and silently write a file that matches neither what
    the user checked nor what they asked for. Measured, not theorised.
    """
    # One dir, reused: exports are serialised, so only one snapshot exists.
    return Project(replace(project.session, specs=[project.spec]), S.root / ".export")


def r_export(req, q):
    body = _json_body(req)
    out = body.get("out")
    if not out or not isinstance(out, str):
        raise InputError("<body>", "missing 'out' path")
    project = _project()
    project.spec  # raises NoGrade -> 409 before a thread is spawned
    with LOCK:
        if any(job["state"] == "running" for job in S.exports.values()):
            raise ExportBusy("an export is already running")
        snapshot = _export_snapshot(project)
        S.n_exports += 1
        name = f"j{S.n_exports}"
        S.exports[name] = {"state": "running", "progress": 0.0, "path": None, "error": None}
    threading.Thread(
        target=_run_export, args=(snapshot, name, out, bool(body.get("gpu"))), daemon=True
    ).start()
    return _ok({"job": name}, 202)


def _run_export(project: Project, name: str, out: str, gpu: bool) -> None:
    # Renders outside LOCK so the scrubber stays live during an export; safe
    # because `project` here is a snapshot with a private cube (see above).
    job = S.exports[name]
    try:
        path = project.export(out, gpu=gpu, progress=lambda f: job.update(progress=round(float(f), 4)))
        job.update(state="done", progress=1.0, path=str(path))
    except Exception as exc:
        _, _, payload = _error_response(exc)
        job.update(state="error", error=json.loads(payload)["error"])


def r_export_status(req, q, name: str):
    job = S.exports.get(name)
    if job is None:
        raise NotFound(f"no export job {name!r}")
    return _ok(job)


# ---- settings -------------------------------------------------------------
# These live on the same 127.0.0.1 listener as everything else -- no new socket,
# no wider bind. A key is only ever ACCEPTED here; nothing below ever returns
# one, logs one, or puts one in an error. The most a caller learns about a
# stored key is `hint`: an ellipsis and its last four characters.


def _provider_arg(body: dict) -> str:
    from .providers.base import info_for

    name = str(body.get("provider") or "").strip().lower()
    try:
        info_for(name)
    except ProviderError as exc:
        # A name typed by a UI is bad input, not an upstream failure.
        raise InputError("provider", str(exc)) from exc
    return name


def r_providers(req, q):
    """Every provider, whether it is ready to use, and which one is active."""
    from .providers.base import describe

    return _ok({"providers": describe(), **_active()})


def r_set_provider(req, q):
    """Choose the provider (and optionally the model) for the next grade."""
    from ragvid import settings

    body = _json_body(req)
    name = _provider_arg(body)
    model = body.get("model")
    settings.select(provider=name, model=None if model is None else str(model).strip())
    return _ok(_state_json())


def r_set_key(req, q):
    """Store an API key, or clear it when `key` is empty or null.

    Clearing removes the entry from settings.json outright, so the old bytes do
    not linger in the file.
    """
    from ragvid import settings

    body = _json_body(req)
    name = _provider_arg(body)
    key = body.get("key")
    if key is None or not str(key).strip():
        settings.clear_key(name)
    else:
        settings.set_key(name, str(key))
    return r_providers(req, q)


def r_index(req, q):
    body = INDEX.read_bytes() if INDEX.is_file() else PLACEHOLDER
    return 200, "text/html; charset=utf-8", body


ROUTES = {
    ("GET", "/"): r_index,
    ("GET", "/api/state"): r_state,
    ("GET", "/media/frame"): r_frame,
    ("GET", "/media/cube"): r_cube,
    ("POST", "/api/project"): r_open,
    ("POST", "/api/vibe"): r_vibe,
    ("POST", "/api/reference"): r_reference,
    ("POST", "/api/refine"): r_refine,
    ("POST", "/api/spec"): r_spec,
    ("POST", "/api/undo"): r_undo,
    ("POST", "/api/close"): r_close,
    ("POST", "/api/reset"): r_reset,
    ("POST", "/api/revert"): r_revert,
    ("GET", "/api/browse"): r_browse,
    ("GET", "/api/providers"): r_providers,
    ("POST", "/api/provider"): r_set_provider,
    ("POST", "/api/key"): r_set_key,
    ("POST", "/api/export"): r_export,
}


# ---- plumbing -------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "ragvid"
    protocol_version = "HTTP/1.1"  # every response below sets Content-Length
    body = b""

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            self._send(*_error_response(InputError("<upload>", f"body over {MAX_UPLOAD} bytes")))
            self.close_connection = True
            return
        self.body = self.rfile.read(length) if length > 0 else b""
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        route = ROUTES.get((method, url.path))
        job = url.path[len("/api/export/"):] if url.path.startswith("/api/export/") else None
        try:
            if job and method == "GET":
                result = r_export_status(self, parse_qs(url.query), job)
            elif route:
                result = route(self, parse_qs(url.query))
            else:
                raise NotFound(f"no route for {method} {url.path}")
        except Exception as exc:
            result = _error_response(exc)
        self._send(*result)

    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # belt and braces with ?v=
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # one short line, not apache's
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer with a platform-correct SO_REUSEADDR."""

    def server_bind(self):
        # On POSIX, SO_REUSEADDR only waives TIME_WAIT -- exactly what we want,
        # so a restart does not have to wait out the previous socket. On Windows
        # it means something else entirely: it lets bind() SUCCEED on a port
        # another process is actively LISTENING on, and the two then split
        # incoming connections at random. That silently defeats the port walk
        # below -- a second `ragvid serve` would report 8765 and steal half the
        # first one's requests instead of moving to 8766. Windows already
        # releases TIME_WAIT ports to a new listener, so it needs nothing here.
        self.allow_reuse_address = not is_windows()
        super().server_bind()


def _bind(port: int) -> tuple[ThreadingHTTPServer, int]:
    """First free port at or after `port` — a second `ragvid serve` should work."""
    for candidate in range(port, port + 21):
        try:
            return _Server((HOST, candidate), Handler), candidate
        except OSError as exc:
            # Windows raises WSAEADDRINUSE (10048); CPython maps it to
            # EADDRINUSE on most builds but exposes the raw code as .winerror,
            # so check both rather than re-raising a busy port as a hard failure.
            if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) != 10048:
                raise
    raise OSError(f"no free port in {port}..{port + 20}")


def serve(port: int = DEFAULT_PORT, root: str | Path | None = None,
          open_browser: bool = True) -> None:
    from .providers.base import load_env  # so the reported provider matches .env

    load_env()
    S.root = Path(root).expanduser() if root else _work_dir()
    S.root.mkdir(parents=True, exist_ok=True)
    httpd, port = _bind(port)
    url = f"http://{HOST}:{port}/"
    print(f"ragvid serving {url}\n  state: {S.root}\n  Ctrl-C to stop", flush=True)
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def _selfcheck() -> None:
    """`python -m ragvid.server` — the three bits with edges: multipart, filenames,
    status mapping. Everything else is one call into Project."""
    from .errors import NoGrade, ProviderNotConfigured, RateLimited

    body = (b'--X\r\nContent-Disposition: form-data; name="file"; filename="a b.mp4"\r\n'
            b"Content-Type: video/mp4\r\n\r\n\x00\x01PAYLOAD\r\n--X--\r\n")
    assert _multipart_file("multipart/form-data; boundary=X", body) == ("a b.mp4", b"\x00\x01PAYLOAD")
    assert _safe_name("../../etc/passwd") == "passwd"
    assert _safe_name("..") == "upload"
    assert _safe_name("a/b\\c.mp4") == "c.mp4"
    for exc, code in [(InputError("f", "bad"), 400), (NoGrade(), 409), (NothingToUndo("x"), 409),
                      (SessionNotFound("/r"), 404), (ProviderNotConfigured("groq", "K"), 428),
                      (RateLimited("groq", 54.0), 429), (ValueError("boom"), 500)]:
        status, _, raw = _error_response(exc)
        assert status == code, (exc, status, code)
        assert json.loads(raw)["error"]["type"] == type(exc).__name__
    assert json.loads(_error_response(RateLimited("groq", 54.0))[2])["error"]["retry_after"] == 54.0

    # An export must render the grade that was current when it started, from a
    # cube nothing else writes. Both halves of that, without touching ffmpeg:
    live = Project(Session(source="c.mp4", stats=None, specs=[GradeSpec()]), S.root)
    snap = _export_snapshot(live)
    assert snap.cube_path != live.cube_path, snap.cube_path
    live.session.push(GradeSpec(saturation=0.0))
    assert snap.spec.saturation == 1.0 and live.spec.saturation == 0.0
    print("server selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
