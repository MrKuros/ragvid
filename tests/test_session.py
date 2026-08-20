import json

import pytest

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
    # at the floor: one spec left, nothing to step back to
    assert s.pop() is False
    assert len(s.specs) == 1


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


def test_load_corrupt_session_is_friendly(tmp_path):
    p = Session.path()
    p.parent.mkdir()
    p.write_text("{not json")
    with pytest.raises(NoSession):
        Session.load()


def test_session_file_is_json_in_dot_ragvid():
    s = Session.create("clip.mp4", _stats())
    s.push(GradeSpec.identity())
    s.save()
    raw = json.loads(Session.path().read_text())
    assert str(Session.path()) == ".ragvid/session.json"
    assert raw["source"] == "clip.mp4"
    assert len(raw["specs"]) == 1
