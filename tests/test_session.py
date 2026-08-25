import json
from pathlib import Path

import pytest

from ragvid.errors import SessionCorrupt
from ragvid.probe import ClipStats
from ragvid.session import NoSession, Session
from ragvid.spec import RGB, GradeSpec


def _stats():
    return ClipStats(
        mean=RGB(r=0.2, g=0.3, b=0.4),
        std=RGB.of(0.1),
        saturation=0.25,
        frames_sampled=10,
        width=640,
        height=360,
        duration=4.0,
    )


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_round_trip():
    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec(saturation=1.4, rationale="punchy"))
    s.save()

    got = Session.load()
    assert got.source == "clip.mp4"
    assert got.stats == _stats()
    assert got.spec.saturation == 1.4
    assert got.spec.rationale == "punchy"


def test_push_pop_history():
    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec.identity())
    s.push(GradeSpec(contrast=0.5))

    assert s.spec.contrast == 0.5
    assert s.pop() is True
    assert s.spec.is_identity()
    # popping the FIRST grade is allowed and lands on the ungraded clip -- the
    # old floor of one spec made undo a no-op right after a first grade
    assert s.pop() is True
    assert s.specs == [] and s.labels == []
    # only now is there nothing left
    assert s.pop() is False


def test_pop_survives_save_load():
    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec.identity())
    s.push(GradeSpec(temperature=1200))
    s.save()

    s2 = Session.load()
    assert len(s2.specs) == 2
    assert s2.pop() is True
    s2.save()
    assert len(Session.load().specs) == 1


def test_load_without_session_is_friendly():
    with pytest.raises(NoSession, match="run 'ragvid grade' first"):
        Session.load()


def test_an_interrupted_write_leaves_the_previous_session_intact(tmp_path, monkeypatch):
    """`Path.write_text` opens "w", which TRUNCATES before the new bytes land.
    An interrupted write -- a crash, a full disk -- therefore left a truncated
    session.json, which `load` can only report as SessionCorrupt. There is no
    backup and no repair path, so that is somebody's whole grade.

    Simulated faithfully rather than by patching json.dumps: dumps is evaluated
    BEFORE write_text is called, so a failure there never reaches the file and
    proves nothing. The failure that matters is a partial write, and the only
    thing that separates the two versions is WHICH file it lands on -- the
    session itself, or a temp beside it that os.replace never promotes.
    """
    real = Path.write_text

    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec(saturation=1.4), "first")
    s.save(tmp_path)
    good = Session.path(tmp_path).read_text(encoding="utf-8")

    def half_a_write(self, text, *a, **kw):
        real(self, text[: len(text) // 2])                 # the bytes that fit
        raise OSError(28, "No space left on device")        # ...and then the disk

    s.push(GradeSpec(contrast=0.5), "second")
    monkeypatch.setattr(Path, "write_text", half_a_write)
    with pytest.raises(OSError):
        s.save(tmp_path)
    monkeypatch.setattr(Path, "write_text", real)

    assert Session.path(tmp_path).read_text(encoding="utf-8") == good, \
        "a half-finished save destroyed the session that was already there"
    assert len(Session.load(tmp_path).specs) == 1, "and it still loads"
    left = [f.name for f in Session.dir(tmp_path).iterdir() if ".tmp" in f.name]
    assert left == [], f"a temp file was left behind: {left}"


def test_load_corrupt_session_is_friendly(tmp_path):
    p = Session.path()
    p.parent.mkdir()
    p.write_text("{not json")
    # Distinct from NoSession on purpose: "this file is damaged" is different
    # advice from "grade something first".
    with pytest.raises(SessionCorrupt, match="unreadable") as info:
        Session.load()
    assert not isinstance(info.value, NoSession)
    assert info.value.path == str(p)


def test_session_file_is_json_in_dot_ragvid():
    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec.identity())
    s.save()
    raw = json.loads(Session.path().read_text())
    assert Session.path().parts[-2:] == (".ragvid", "session.json")
    assert Session.path().is_absolute()
    assert raw["source"] == "clip.mp4"
    assert len(raw["specs"]) == 1


def test_input_format_round_trips_and_an_older_session_still_loads():
    """The generated-LUT case needs both halves back: the file in force, and
    which format ragvid baked it from."""
    s = Session.create("clip.mp4", _stats(), input_lut="/w/.ragvid/log_slog3.cube",
                       input_format="slog3")
    s.save()

    got = Session.load()
    assert got.input_lut == "/w/.ragvid/log_slog3.cube"
    assert got.input_format == "slog3"

    # Every session written before format names existed. It must open, and a
    # vendor .cube with no format is exactly what None means.
    raw = json.loads(Session.path().read_text())
    del raw["input_format"]
    Session.path().write_text(json.dumps(raw))
    older = Session.load()
    assert older.input_format is None
    assert older.input_lut == "/w/.ragvid/log_slog3.cube"


def test_an_intent_round_trips_and_an_older_session_still_loads():
    """The verbs behind a grade are stored per spec, exactly like `labels` --
    and, exactly like labels, every session already on disk predates the key."""
    from ragvid.intent import Intent, Op

    intent = Intent(ops=[Op(op="warmth", dir="up", amount="subtle")], strength="moderate")
    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec(temperature=400.0), "warmer", intent)
    s.push(GradeSpec(contrast=0.3), "punchier")      # no intent: a photo match
    s.save()

    got = Session.load()
    assert got.intents == [intent, None]
    assert got.intent is None                        # the CURRENT grade has none
    assert got.pop() is True
    assert got.intent == intent                      # ... and undo brings it back

    # A session written before the intent path. It must open, and every grade in
    # it must read as "no verbs", not as a missing key.
    raw = json.loads(Session.path().read_text())
    del raw["intents"]
    Session.path().write_text(json.dumps(raw))
    older = Session.load()
    assert older.intents == [None, None]
    assert older.intent is None
    assert len(older.specs) == 2


def test_auto_balance_round_trips_and_defaults_on_for_an_older_session():
    s = Session.create("clip.mp4", _stats())
    assert s.auto_balance is True
    s.auto_balance = False
    s.save()
    assert Session.load().auto_balance is False

    # A session written before auto-balance existed opens the way a new one
    # would: on. A missing key is not "off".
    raw = json.loads(Session.path().read_text())
    del raw["auto_balance"]
    Session.path().write_text(json.dumps(raw))
    assert Session.load().auto_balance is True
