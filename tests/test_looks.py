"""The looks corpus is a build artifact; this is the rot detector.

ragvid/looks.py already carries the real assertions (every entry round-trips
through GradeSpec with all 43 fields, retrieval is deterministic, an unmatched
query grounds on nothing, and the measured warm/cold gaps hold). Duplicating
them here would mean maintaining them twice, so CI just runs them.
"""

from ragvid import looks


def test_corpus_self_check():
    assert looks.load_corpus(), "corpus is empty -- did looks/ move again?"
    looks._self_check()
