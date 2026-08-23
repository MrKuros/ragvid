"""The local web API, driven over a real socket.

A real ThreadingHTTPServer on an ephemeral port, real HTTP with urllib, real
ffmpeg for frames and one real export. What is faked is exactly one thing: the
provider. No test here may reach the network beyond loopback.

The server keeps its state in module globals (`ragvid.server.S`), so every test
resets them and points the work dir at tmp_path — a test must never write into
~/.local/share.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

import pytest

from ragvid import server
from ragvid.errors import FFmpegError, ProviderError, ProviderNotConfigured, RateLimited
from ragvid.project import Project
from ragvid.spec import GradeSpec

SAMPLE = Path("assets/sample.mp4")
REF = Path("assets/ref_warm.png")

FAKE_SPEC = GradeSpec(saturation=0.4, temperature=800.0, contrast=0.3, rationale="fake grade")


# ---- transport -------------------------------------------------------------


class Resp(NamedTuple):
    status: int
    ctype: str
    body: bytes

    @property
    def json(self) -> dict:
        return json.loads(self.body)

    @property
    def error(self) -> dict:
        return json.loads(self.body)["error"]


def _multipart(filename: str, data: bytes, field: str = "file") -> tuple[bytes, str]:
    b = "----ragvidtest"
    head = (
        f"--{b}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


class Api:
    """Just enough client. Non-2xx comes back as a Resp, not an exception."""

    def __init__(self, url: str) -> None:
        self.url = url

    def call(self, method: str, path: str, data: bytes | None = b"", ctype: str | None = None) -> Resp:
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=60) as f:
                return Resp(f.status, f.headers.get("Content-Type", ""), f.read())
        except urllib.error.HTTPError as exc:
            return Resp(exc.code, exc.headers.get("Content-Type", ""), exc.read())

    def get(self, path: str) -> Resp:
        return self.call("GET", path, data=None)

    def post(self, path: str, obj: dict | None = None) -> Resp:
        body = json.dumps(obj).encode() if obj is not None else b""
        return self.call("POST", path, body, "application/json")

    def upload(self, path: str, filename: str, data: bytes) -> Resp:
        body, ctype = _multipart(filename, data)
        return self.call("POST", path, body, ctype)

    # -- shorthands used all over the file --
    def state(self) -> dict:
        return self.get("/api/state").json

    def open_clip(self) -> dict:
        r = self.post("/api/project", {"path": str(SAMPLE.resolve())})
        assert r.status == 200, r.body
        return r.json

    def plan(self) -> dict:
        """Plan a grade the offline way — no provider, no network."""
        r = self.post("/api/reference", {"path": str(REF.resolve())})
        assert r.status == 200, r.body
        return r.json


# ---- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def httpd():
    """One server for the file; port 0 so nothing collides with a real ragvid."""
    srv = ThreadingHTTPServer((server.HOST, 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(5)
    assert not thread.is_alive()


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Keys and the provider choice come from tmp_path, never from the developer's
    real settings.json or the repo's .env."""
    from ragvid.providers.base import CATALOG

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr("ragvid.providers.base.load_env", lambda *a, **k: None)
    for info in CATALOG.values():
        if info.env_var:
            monkeypatch.delenv(info.env_var, raising=False)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Nothing reaches a provider unless a test explicitly installs a fake one."""

    def boom(*a, **kw):
        raise AssertionError("a provider was constructed; this path must stay offline")

    monkeypatch.setattr("ragvid.providers.get_provider", boom)
    monkeypatch.setattr("ragvid.providers.base.get_provider", boom)


@pytest.fixture
def api(httpd, tmp_path, monkeypatch) -> Api:
    """Fresh server state per test, with all writes confined to tmp_path."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(server.S, "project", None)
    monkeypatch.setattr(server.S, "version", 0)
    monkeypatch.setattr(server.S, "exports", {})
    monkeypatch.setattr(server.S, "n_exports", 0)
    monkeypatch.setattr(server.S, "root", tmp_path / "state")
    host, port = httpd.server_address[:2]
    return Api(f"http://{host}:{port}")


@pytest.fixture
def fake_llm(monkeypatch) -> list:
    """A provider that answers instantly with FAKE_SPEC. Records its prompts."""
    calls: list[tuple[str, str]] = []

    class FakeProvider:
        name = "fake"

        def plan(self, system: str, user: str) -> GradeSpec:
            calls.append((system, user))
            return FAKE_SPEC

    monkeypatch.setattr("ragvid.providers.get_provider", lambda *a, **kw: FakeProvider())
    return calls


@pytest.fixture
def llm_raises(monkeypatch):
    """Install a provider that fails the way a real one would."""

    def install(exc: Exception):
        def boom(*a, **kw):
            raise exc

        monkeypatch.setattr("ragvid.providers.get_provider", boom)

    return install


def poll_export(api: Api, job: str, deadline: float = 60.0) -> dict:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        body = api.get(f"/api/export/{job}").json
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    pytest.fail(f"export {job} still running after {deadline}s")


# ---- happy paths -----------------------------------------------------------


def test_index_is_served(api):
    r = api.get("/")
    assert r.status == 200
    assert r.ctype.startswith("text/html")
    assert len(r.body) > 200


def test_empty_state(api):
    body = api.state()
    assert body["open"] is False
    assert body["version"] == 0
    assert {"groq", "anthropic", "ollama"} <= set(body["providers"])
    assert body["provider"] and body["model"]


def test_open_by_path(api):
    body = api.open_clip()
    assert body["open"] is True
    assert body["source"] == str(SAMPLE.resolve())
    assert body["name"] == "sample.mp4"
    assert body["duration"] == pytest.approx(4.0, abs=0.2)
    assert body["planned"] is False
    assert body["spec"] is None
    assert body["can_undo"] is False
    assert body["history_depth"] == 0
    assert body["stats"]["width"] == 640 and body["stats"]["height"] == 360
    assert api.state() == body  # a read returns exactly what the mutation did


def test_upload_lands_in_the_work_dir_byte_identical(api, tmp_path):
    data = SAMPLE.read_bytes()
    body = api.upload("/api/project", "clip.mp4", data).json
    landed = Path(body["source"])
    assert landed == tmp_path / "xdg" / "ragvid" / "work" / "clip.mp4"
    assert landed.read_bytes() == data


def test_upload_filename_cannot_escape_the_work_dir(api, tmp_path):
    work = tmp_path / "xdg" / "ragvid" / "work"
    body = api.upload("/api/project", "../../../../etc/passwd.mp4", SAMPLE.read_bytes()).json
    assert Path(body["source"]) == work / "passwd.mp4"
    # Nothing was created anywhere above the work dir.
    assert sorted(p.name for p in work.iterdir()) == ["passwd.mp4"]
    assert not (tmp_path / "etc").exists() and not (tmp_path / "xdg" / "etc").exists()


def test_reference_plans_a_grade_offline(api):
    api.open_clip()
    body = api.plan()
    assert body["planned"] is True
    assert body["history_depth"] == 1
    assert body["can_undo"] is True          # one grade is undoable
    assert body["spec"]["rationale"]
    assert [s["label"] for s in body["steps"]] == ["photo: ref_warm.png"]


def test_reference_upload(api):
    api.open_clip()
    body = api.upload("/api/reference", "ref.png", REF.read_bytes()).json
    assert body["planned"] is True


def test_vibe_uses_the_provider(api, fake_llm):
    api.open_clip()
    body = api.post("/api/vibe", {"vibe": "gloomy"}).json
    assert body["planned"] is True
    assert body["spec"]["saturation"] == FAKE_SPEC.saturation
    assert len(fake_llm) == 1 and "gloomy" in fake_llm[0][1]


def test_refine_uses_the_provider(api, fake_llm):
    api.open_clip()
    api.plan()
    body = api.post("/api/refine", {"instruction": "less blue"}).json
    assert body["spec"]["saturation"] == FAKE_SPEC.saturation
    assert body["history_depth"] == 2 and body["can_undo"] is True
    assert "less blue" in fake_llm[0][1]


def test_spec_is_the_slider_path(api):
    api.open_clip()
    api.plan()
    spec = api.state()["spec"] | {"contrast": 0.6}
    body = api.post("/api/spec", {"spec": spec}).json
    assert body["spec"]["contrast"] == 0.6
    assert body["history_depth"] == 2 and body["can_undo"] is True


def test_undo_steps_back(api):
    api.open_clip()
    first = api.plan()["spec"]
    api.post("/api/spec", {"spec": first | {"contrast": 0.6}})
    body = api.post("/api/undo").json
    assert body["spec"] == first
    # a single remaining grade is still undoable -- the old floor made undo a
    # dead button exactly when someone most wants it
    assert body["history_depth"] == 1 and body["can_undo"] is True


def test_close_returns_to_the_empty_state(api):
    api.open_clip()
    body = api.post("/api/close").json
    assert body["open"] is False
    assert "source" not in body
    assert api.get("/media/frame?t=0&graded=0").status == 404


def test_cube_downloads(api):
    api.open_clip()
    api.plan()
    r = api.get("/media/cube?v=2")
    assert r.status == 200
    assert r.ctype.startswith("text/plain")
    assert b"LUT_3D_SIZE" in r.body


# ---- frames ----------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n"


def test_frame_returns_real_png_bytes(api):
    api.open_clip()
    r = api.get("/media/frame?t=1&graded=0")
    assert r.status == 200
    assert r.ctype == "image/png"
    assert r.body.startswith(PNG) and r.body.endswith(b"IEND\xaeB`\x82")


def test_different_timestamps_give_different_frames(api):
    api.open_clip()
    frames = [api.get(f"/media/frame?t={t}&graded=0").body for t in (0, 2, 3.9)]
    assert all(f.startswith(PNG) for f in frames)
    assert len(set(frames)) == 3


def test_frame_past_the_end_is_clamped_not_empty(api):
    api.open_clip()
    r = api.get("/media/frame?t=999&graded=0")
    assert r.status == 200 and r.body.startswith(PNG) and len(r.body) > 1000


def test_graded_frame_differs_from_ungraded(api):
    api.open_clip()
    ungraded = api.get("/media/frame?t=1&graded=0").body
    api.plan()
    graded = api.get("/media/frame?t=1&graded=1&v=2").body
    assert graded.startswith(PNG)
    assert graded != ungraded


def test_ungraded_frame_works_before_planning_graded_does_not(api):
    api.open_clip()
    assert api.get("/media/frame?t=0&graded=0").status == 200
    r = api.get("/media/frame?t=0&graded=1")
    assert r.status == 409 and r.error["type"] == "NoGrade"


# ---- version ---------------------------------------------------------------


def test_version_increments_on_every_mutation_and_never_on_a_read(api):
    assert api.state()["version"] == 0
    mutations = [
        lambda: api.post("/api/project", {"path": str(SAMPLE.resolve())}),
        lambda: api.post("/api/reference", {"path": str(REF.resolve())}),
        lambda: api.post("/api/spec", {"spec": api.state()["spec"] | {"contrast": 0.2}}),
        lambda: api.post("/api/undo"),
        lambda: api.post("/api/close"),
    ]
    for i, mutate in enumerate(mutations, start=1):
        body = mutate().json
        assert body["version"] == i, body
        assert api.state()["version"] == i


def test_reads_do_not_bump_the_version(api):
    api.open_clip()
    api.plan()
    before = api.state()["version"]
    api.get("/")
    api.get("/media/frame?t=0.5&graded=1")
    api.get("/media/frame?t=0.5&graded=0")
    api.get("/media/cube")
    api.get("/api/state")
    assert api.state()["version"] == before


def test_a_failed_mutation_does_not_bump_the_version(api):
    api.open_clip()
    before = api.state()["version"]
    assert api.post("/api/spec", {"spec": {"saturation": "purple"}}).status == 400
    assert api.post("/api/undo").status == 409
    assert api.state()["version"] == before


# ---- export ----------------------------------------------------------------


def test_export_runs_and_reports_progress(api, tmp_path):
    api.open_clip()
    api.plan()
    out = tmp_path / "out.mp4"
    r = api.post("/api/export", {"out": str(out), "gpu": False})
    assert r.status == 202
    job = r.json["job"]

    body = poll_export(api, job)
    assert body["state"] == "done", body
    assert body["error"] is None
    assert body["progress"] == 1.0
    assert Path(body["path"]) == out
    assert out.stat().st_size > 1000


def test_only_one_export_at_a_time(api, tmp_path, monkeypatch):
    """The second POST is refused while the first is still encoding."""
    release = threading.Event()

    def slow_export(self, out_path, gpu=False, progress=None):
        if progress:
            progress(0.5)
        release.wait(30)
        Path(out_path).write_bytes(b"encoded")
        return Path(out_path)

    monkeypatch.setattr(Project, "export", slow_export)
    api.open_clip()
    api.plan()
    try:
        first = api.post("/api/export", {"out": str(tmp_path / "a.mp4")})
        assert first.status == 202
        assert api.get(f"/api/export/{first.json['job']}").json["state"] == "running"

        second = api.post("/api/export", {"out": str(tmp_path / "b.mp4")})
        assert second.status == 409
        assert second.error["type"] == "ExportBusy"
    finally:
        release.set()
    assert poll_export(api, first.json["job"])["state"] == "done"

    # ...and once it is finished, a second export is allowed again.
    third = api.post("/api/export", {"out": str(tmp_path / "c.mp4")})
    assert third.status == 202
    assert poll_export(api, third.json["job"])["state"] == "done"


def test_editing_the_grade_mid_export_cannot_change_what_is_being_written(api, tmp_path, monkeypatch):
    """The regression that shipped a greyscale file after a slider nudge.

    The export thread must render from a snapshot: its own .cube (the live
    project re-bakes that file on every graded frame) and a spec frozen when
    Export was pressed.
    """
    release, rendered = threading.Event(), []

    def slow_export(self, out_path, gpu=False, progress=None):
        rendered.append(self)
        release.wait(30)
        Path(out_path).write_bytes(b"encoded")
        return Path(out_path)

    monkeypatch.setattr(Project, "export", slow_export)
    api.open_clip()
    planned = api.plan()["spec"]
    job = api.post("/api/export", {"out": str(tmp_path / "a.mp4")}).json["job"]
    try:
        while not rendered:
            time.sleep(0.01)
        # The user keeps working: slider to greyscale, and a graded frame render
        # (which re-bakes the live cube) on top.
        api.post("/api/spec", {"spec": planned | {"saturation": 0.0}})
        assert api.get("/media/frame?t=1&graded=1").status == 200
        snapshot = rendered[0]
        assert snapshot.spec.saturation == planned["saturation"]
        assert snapshot.cube_path != server.S.project.cube_path
    finally:
        release.set()
    assert poll_export(api, job)["state"] == "done"


def test_export_failure_lands_in_the_job_not_the_response(api, tmp_path, monkeypatch):
    def boom(self, out_path, gpu=False, progress=None):
        raise FFmpegError(1, ["ffmpeg", "-i", "x"], "No such file or directory")

    monkeypatch.setattr(Project, "export", boom)
    api.open_clip()
    api.plan()
    job = api.post("/api/export", {"out": str(tmp_path / "nope.mp4")}).json["job"]
    body = poll_export(api, job)
    assert body["state"] == "error"
    assert body["error"]["type"] == "FFmpegError"
    assert body["error"]["returncode"] == 1


def test_export_before_a_grade_is_refused_before_a_thread_starts(api, tmp_path):
    api.open_clip()
    r = api.post("/api/export", {"out": str(tmp_path / "out.mp4")})
    assert r.status == 409 and r.error["type"] == "NoGrade"
    assert server.S.exports == {}


# ---- error mapping ---------------------------------------------------------


def test_input_error_is_400_and_carries_its_fields(api):
    r = api.post("/api/project", {"path": "README.md"})
    assert r.status == 400
    assert r.ctype.startswith("application/json")
    assert r.error["type"] == "InputError"
    assert r.error["path"] == "README.md" and r.error["reason"]


def test_missing_file_is_400(api):
    r = api.post("/api/project", {"path": "/nope/missing.mp4"})
    assert r.status == 400 and r.error["type"] == "InputError"


@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/project", None),  # neither multipart nor {"path": ...}
        ("/api/vibe", {}),  # missing 'vibe'
        ("/api/refine", {"instruction": "  "}),  # blank instruction
        ("/api/spec", {"spec": "not an object"}),
        ("/api/spec", {"spec": {"saturation": "purple"}}),  # pydantic, not a 500
        ("/api/export", {}),  # missing 'out'
    ],
)
def test_bad_bodies_are_400_input_errors(api, path, body):
    api.open_clip()
    r = api.post(path, body)
    assert r.status == 400, r.body
    assert r.error["type"] == "InputError"


def test_malformed_json_is_400(api):
    r = api.call("POST", "/api/spec", b"{not json", "application/json")
    assert r.status == 400 and r.error["type"] == "InputError"


def test_bad_timestamp_is_400(api):
    api.open_clip()
    r = api.get("/media/frame?t=abc")
    assert r.status == 400 and r.error["type"] == "InputError"


def test_unknown_route_is_404(api):
    for path in ("/api/nope", "/nope.html"):
        r = api.get(path)
        assert r.status == 404 and r.error["type"] == "NotFound"


def test_unknown_export_job_is_404(api):
    r = api.get("/api/export/j99")
    assert r.status == 404 and r.error["type"] == "NotFound"


@pytest.mark.parametrize("path", ["/media/frame?t=0&graded=0", "/media/cube", "/api/undo"])
def test_no_project_is_404_session_not_found(api, path):
    method = "GET" if path.startswith("/media") else "POST"
    r = api.call(method, path, data=None if method == "GET" else b"")
    assert r.status == 404 and r.error["type"] == "SessionNotFound"


def test_refine_without_a_grade_is_409_and_never_calls_the_provider(api):
    api.open_clip()
    r = api.post("/api/refine", {"instruction": "less blue"})
    # The autouse guard would turn any provider construction into a 500.
    assert r.status == 409 and r.error["type"] == "NoGrade"


def test_undo_walks_back_off_the_first_grade_then_409s(api):
    api.open_clip()
    api.plan()

    # undoing the only grade lands on the ungraded clip, it does not refuse
    body = api.post("/api/undo").json
    assert body["planned"] is False
    assert body["history_depth"] == 0 and body["can_undo"] is False
    assert body["spec"] is None

    # the clip stays open, and an ungraded frame still renders
    assert body["open"] is True

    # only now is there nothing left to undo
    r = api.post("/api/undo")
    assert r.status == 409 and r.error["type"] == "NothingToUndo"


def test_provider_not_configured_is_428_with_the_env_var(api, llm_raises):
    llm_raises(ProviderNotConfigured("groq", "GROQ_API_KEY"))
    api.open_clip()
    r = api.post("/api/vibe", {"vibe": "gloomy"})
    assert r.status == 428
    assert r.error["type"] == "ProviderNotConfigured"
    assert r.error["env_var"] == "GROQ_API_KEY"
    assert api.state()["planned"] is False


def test_rate_limited_is_429_with_retry_after(api, llm_raises):
    llm_raises(RateLimited("groq", 54.0))
    api.open_clip()
    r = api.post("/api/vibe", {"vibe": "gloomy"})
    assert r.status == 429
    assert r.error["type"] == "RateLimited" and r.error["retry_after"] == 54.0


def test_provider_error_is_502(api, llm_raises):
    llm_raises(ProviderError("groq", "the model returned nonsense"))
    api.open_clip()
    api.plan()
    r = api.post("/api/refine", {"instruction": "warmer"})
    assert r.status == 502 and r.error["type"] == "ProviderError"
    assert "nonsense" in r.error["message"]


def test_ffmpeg_error_is_500(api, monkeypatch):
    def boom(self, at=0.0, graded=True, path=None):
        raise FFmpegError(69, ["ffmpeg", "-i", "clip.mp4"], "Invalid data found")

    monkeypatch.setattr(Project, "frame", boom)
    api.open_clip()
    r = api.get("/media/frame?t=0&graded=0")
    assert r.status == 500
    assert r.error["type"] == "FFmpegError" and r.error["returncode"] == 69


def test_an_unexpected_exception_is_a_500_not_a_hang(api, monkeypatch):
    monkeypatch.setattr(Project, "bake", lambda self, *a, **kw: 1 / 0)
    api.open_clip()
    api.plan()
    r = api.get("/media/cube")
    assert r.status == 500 and r.error["type"] == "ZeroDivisionError"
    assert api.state()["open"] is True  # the server is still alive and coherent


# ---- binding ---------------------------------------------------------------


def test_binds_loopback_only(httpd):
    assert server.HOST == "127.0.0.1"
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    # And it is genuinely unreachable off loopback: same port, this machine's
    # routable address, refused. (Skipped on hosts that only resolve to 127.x.)
    try:
        outside = socket.gethostbyname(socket.gethostname())
    except OSError:
        pytest.skip("no resolvable hostname")
    if outside.startswith("127."):
        pytest.skip("host resolves to loopback; nothing else to try")
    with socket.socket() as sock:
        sock.settimeout(2)
        assert sock.connect_ex((outside, port)) != 0


# ---- settings: providers and keys ------------------------------------------
# The key never comes back out. Every assertion here is about that, or about
# the panel having enough to render without it.

SENTINEL = "gsk_SENTINEL_never_leak_me_0123456789"


def test_providers_route_lists_everything_with_its_state(api):
    rows = {p["name"]: p for p in api.get("/api/providers").json["providers"]}
    assert {"groq", "anthropic", "openai", "deepseek", "ollama"} <= set(rows)
    assert rows["groq"]["configured"] is False and rows["groq"]["hint"] is None
    assert rows["ollama"]["needs_key"] is False and rows["ollama"]["configured"] is True
    # The UI needs to be able to warn about a weaker endpoint.
    assert rows["groq"]["enforces_schema"] is True
    assert rows["deepseek"]["enforces_schema"] is False
    assert sum(p["active"] for p in rows.values()) == 1


def test_choosing_a_provider_sticks(api):
    body = api.post("/api/provider", {"provider": "anthropic", "model": "claude-opus-5"}).json
    assert body["provider"] == "anthropic" and body["model"] == "claude-opus-5"
    assert api.state()["provider"] == "anthropic"

    # An empty model falls back to the catalog default rather than sending "".
    assert api.post("/api/provider", {"provider": "openai", "model": ""}).json["model"]


def test_an_unknown_provider_is_a_400_not_a_502(api):
    r = api.post("/api/provider", {"provider": "gpt5000"})
    assert r.status == 400
    assert r.error["type"] == "InputError"
    assert "unknown provider" in r.error["message"]


def test_a_key_can_be_set_and_cleared_from_the_ui(api, tmp_path):
    import stat

    from ragvid import settings

    r = api.post("/api/key", {"provider": "groq", "key": SENTINEL})
    assert r.status == 200
    groq = [p for p in r.json["providers"] if p["name"] == "groq"][0]
    assert groq["configured"] is True and groq["hint"] == "…6789"

    # It really landed on disk, and only its owner can read it.
    assert settings.key("groq", "GROQ_API_KEY") == SENTINEL
    assert stat.S_IMODE(settings.path().stat().st_mode) == 0o600

    # Clearing removes it from the file rather than blanking it.
    api.post("/api/key", {"provider": "groq", "key": None})
    assert SENTINEL not in settings.path().read_text()
    assert settings.key("groq", "GROQ_API_KEY") is None


def test_state_says_whether_the_active_provider_has_a_key(api):
    """The opening screen prompts for a key off this flag.

    Nobody arrives at a colour grader expecting to bring an API key, so the
    first-run prompt has to appear before the user types anything. It reads
    `configured` straight off /api/state -- if that field stopped being sent,
    the prompt would silently never show and the first grade would fail with an
    error instead.
    """
    assert api.get("/api/state").json["configured"] is False

    api.post("/api/key", {"provider": "groq", "key": SENTINEL})
    assert api.get("/api/state").json["configured"] is True

    api.post("/api/key", {"provider": "groq", "key": None})
    assert api.get("/api/state").json["configured"] is False


def test_a_log_conversion_lut_can_be_set_and_cleared(api, tmp_path):
    """Log footage is unusable without this, and it is not a look.

    Setting one has to re-probe: the stats describe the image the grade lands
    on, and a conversion changes every one of them. If this route ever stopped
    re-probing, the model would keep being told the clip is flat and grey.
    """
    from ragvid.lut import bake_cube
    from ragvid.spec import GradeSpec

    cube = tmp_path / "log_to_709.cube"
    bake_cube(GradeSpec(contrast=0.6, saturation=1.4), str(cube), size=17)

    api.open_clip()
    assert api.state()["input_lut"] is None
    flat = api.state()["stats"]["std"]

    r = api.post("/api/input_lut", {"path": str(cube)})
    assert r.status == 200
    assert r.json["input_lut"] == str(cube.resolve())
    assert r.json["stats"]["std"] != flat, "the clip was not re-probed through the LUT"

    assert api.post("/api/input_lut", {"path": None}).json["input_lut"] is None
    assert api.state()["stats"]["std"] == flat


def test_a_lut_that_is_not_there_is_a_400_not_a_broken_render(api, tmp_path):
    """Caught on the way in, or ffmpeg reports it from inside a filter graph
    minutes into an export."""
    api.open_clip()
    r = api.post("/api/input_lut", {"path": str(tmp_path / "nope.cube")})
    assert r.status == 400
    assert r.error["type"] == "InputError"

    (tmp_path / "notes.txt").write_text("x")
    assert api.post("/api/input_lut", {"path": str(tmp_path / "notes.txt")}).status == 400


def test_the_browser_offers_cube_files(api, tmp_path):
    """The picker has to show LUTs or the feature is unreachable from the UI."""
    (tmp_path / "look.cube").write_text("LUT_3D_SIZE 2\n")
    (tmp_path / "notes.txt").write_text("x")
    listing = api.get(f"/api/browse?path={tmp_path}").json
    kinds = {f["name"]: f["kind"] for f in listing["files"]}
    assert kinds == {"look.cube": "lut"}


def test_no_route_ever_answers_with_the_key(api):
    """The sentinel goes in through the one route that accepts it, and must not
    come back out of any of them -- not in a body, not in an error."""
    from ragvid import settings

    api.post("/api/key", {"provider": "groq", "key": SENTINEL})
    api.post("/api/provider", {"provider": "groq"})
    api.open_clip()

    bodies = [
        api.get("/api/providers").body,
        api.get("/api/state").body,
        api.post("/api/provider", {"provider": "groq"}).body,
        api.post("/api/key", {"provider": "groq", "key": SENTINEL}).body,
        api.get("/api/browse").body,
        api.post("/api/vibe", {"vibe": ""}).body,        # an error body too
        api.get("/api/export/nope").body,
        api.get("/").body,
    ]
    for body in bodies:
        assert SENTINEL.encode() not in body
    assert settings.key("groq", "GROQ_API_KEY") == SENTINEL  # it really was stored


def test_a_missing_key_is_still_a_428_pointing_at_settings(api, llm_raises):
    from ragvid.errors import ProviderNotConfigured

    api.open_clip()
    llm_raises(ProviderNotConfigured("groq", "GROQ_API_KEY"))
    r = api.post("/api/vibe", {"vibe": "gloomy"})
    assert r.status == 428
    assert r.error["env_var"] == "GROQ_API_KEY"


def test_the_page_and_the_server_agree_on_the_api_version():
    """Bump one without the other and a stale page silently drops fields; this
    is the check that makes the two move together."""
    import re

    page = server.INDEX.read_text()
    found = re.search(r"const EXPECTED_API = (\d+);", page)
    assert found, "index.html no longer declares EXPECTED_API"
    assert int(found.group(1)) == server.API_VERSION


def test_the_settings_panel_never_puts_a_key_into_its_input():
    """The one UI rule that cannot be checked from the Python side at runtime."""
    page = server.INDEX.read_text()
    assert 'id="keyInput"' in page and 'type="password"' in page
    assert '$("keyInput").value = "";' in page


def _log_clip(path: Path, fmt: str = "slog3") -> Path:
    """A clip whose pixel values really are `fmt` code values.

    A scene-linear ramp from 0 to 1.5 encoded through the curve, so the
    conversion has something true to invert rather than a picture that merely
    looks flat. testsrc2 cannot stand in here: it is full-range bars, already
    sitting at p99 = 1.0 and std = 0.48, so a conversion cannot move it.
    """
    import subprocess

    import numpy as np

    from ragvid import logspace

    w, h, n = 160, 120, 25
    lin = np.linspace(0.0, 1.5, w)[None, :].repeat(h, 0)
    code = np.clip(logspace.lin_to_log(fmt, lin), 0.0, 1.0)
    frame = np.repeat((code * 255 + 0.5).astype(np.uint8)[:, :, None], 3, axis=2).tobytes()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
         "-frames:v", str(n), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
         str(path)],
        input=frame * n, check=True, capture_output=True,
    )
    return path


def test_a_camera_format_can_be_chosen_by_name(api, tmp_path):
    """The point of the whole feature: someone who shot log names what they
    shot, instead of hunting for a vendor .cube they probably do not have.

    The assertions are on measured statistics, not on the string coming back --
    a route that stored the name and generated nothing would pass that.
    Decoding log expands the range the camera compressed, so contrast rises and
    the highlights reach the top: measured on this ramp, std 0.104 -> 0.253 and
    p99 0.637 -> 0.988.
    """
    clip = _log_clip(tmp_path / "slog3.mp4")
    assert api.post("/api/project", {"path": str(clip)}).status == 200
    before = api.state()["stats"]
    assert before["p99"]["g"] < 0.8, "the fixture is not log-like; nothing to convert"

    r = api.post("/api/input_lut", {"format": "slog3"})
    assert r.status == 200
    assert r.json["input_format"] == "slog3"
    assert r.json["input_lut"].endswith("log_slog3.cube")
    after = r.json["stats"]
    assert after["std"]["g"] > before["std"]["g"] * 1.8
    assert after["p99"]["g"] > 0.95

    cleared = api.post("/api/input_lut", {"format": None}).json
    assert cleared["input_format"] is None and cleared["input_lut"] is None
    assert cleared["stats"]["std"]["g"] == before["std"]["g"]


def test_an_unknown_format_name_is_a_400(api):
    """Only the five names, and the error arrives at the moment it is picked --
    a name ragvid cannot bake is the same failure as a .cube that is not there."""
    api.open_clip()
    r = api.post("/api/input_lut", {"format": "slog9"})
    assert r.status == 400
    assert r.error["type"] == "InputError"
    assert api.state()["input_format"] is None


def test_the_page_offers_every_format_the_module_implements(api):
    """The display names live in the HTML because they are display text; this is
    the check that stops the list drifting from logspace.NAMES."""
    from ragvid import logspace

    page = server.INDEX.read_text()
    for name in logspace.NAMES:
        assert f'<option value="{name}">' in page, f"{name} is not offered in the UI"
