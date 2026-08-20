

def test_stat_notes_do_not_fire_from_unmeasured_defaults():
    """A session written before the percentile fields existed loads with
    ClipStats defaults and session.py does not migrate, so an unguarded
    threshold would feed the model instructions about footage nobody measured."""
    from ragvid.probe import ClipStats
    from ragvid.vibe import _stat_notes

    old = ClipStats(mean=dict(r=0.5, g=0.5, b=0.5), std=dict(r=0.2, g=0.2, b=0.2),
                    saturation=0.2, dominant_hue=30.0, clipped_high=0.0,
                    crushed_low=0.0, width=1920, height=1080, duration=4.0,
                    frames_sampled=10)
    assert old.p99.r == 0.0 and old.frame_variance == 0.0
    assert [n for fires, n in _stat_notes(old) if fires] == []
