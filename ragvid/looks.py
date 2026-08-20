"""The looks corpus — the retrieval half of "RAG".

A small model asked for "make it feel like a rainy night" has to invent 43
numbers from nothing, and it invents badly: it moves fields it does not
understand and leaves the ones that matter at identity. Retrieval fixes the
shape of that problem rather than the model. We hand it one or two COMPLETE
neighbouring grades that were MEASURED off real pixels, and the job collapses
from "author 43 numbers" to "edit these 43 numbers a bit". That is what makes a
spec this large tractable at 20B parameters.

WHERE THE NUMBERS COME FROM. Not from a model, and not from a human: every
corpus spec is the output of `match.match_reference(probe(base), probe(ref))`,
the existing closed-form matcher, run over a reference still by
scripts/build_looks.py. Hand-authored numbers would reproduce exactly the
hallucination the corpus exists to prevent. Each entry carries the ffmpeg chain
that produced its reference still, so every claim is re-derivable from pixels.
`ragvid/look_corpus/*.json` is a build artifact; scripts/build_looks.py is the source.

RETRIEVAL IS DELIBERATELY DUMB: token overlap over the name and mood words. No
embedding model, no vector DB, no new dependency. This is a few dozen JSON
files, and the query is a mood word — a cosine similarity over a 384-dim
sentence embedding would rank the same handful of entries at a hundred times
the cost and a new install. Measured on the shipped corpus (`_self_check`,
corpus mean warmth +0.013): the top-2 hits for "warm" render mid grey +0.211
warmer in (r - b), for "cold night" -0.146. The gap is an order of magnitude
wider than the noise, which is all a corpus this size needs to carry.

A query that overlaps nothing returns nothing. Silence beats a random look:
grounding on an unrelated grade is worse than no grounding at all.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

# Inside the package, not at the repo root: hatch ships `packages = ["ragvid"]`,
# so a top-level looks/ would be absent from an installed wheel and load_corpus()
# would silently return () -- un-grounded prompts, no error, nobody notices.
CORPUS_DIR = Path(__file__).resolve().parent / "look_corpus"

# Number of examples handed to the model. 2, not 5: each entry is a full 43-field
# spec (~350 tokens), and Groq's free tier is 8000 tokens/min.
DEFAULT_K = 2

# A query token matches a corpus token if either is a prefix of the other and the
# prefix is at least this long. Covers warm/warmer/warmth and moody/mood without
# a stemmer. Below 4 characters prefixes are noise ("col" hits "cold" and "color").
_PREFIX_MIN = 4
_PREFIX_SCORE = 0.6

_WORD = re.compile(r"[a-z0-9]+")


@functools.lru_cache(maxsize=1)
def load_corpus() -> tuple[dict, ...]:
    """Every entry in the corpus dir, sorted by name. Cached; empty if the dir is absent."""
    if not CORPUS_DIR.is_dir():
        return ()
    entries = [json.loads(p.read_text()) for p in sorted(CORPUS_DIR.glob("*.json"))]
    return tuple(entries)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _entry_tokens(entry: dict) -> set[str]:
    return _tokens(entry["name"]) | {t.lower() for t in entry["tags"]}


def score(query: str, entry: dict) -> float:
    """Token overlap between the vibe words and the entry's name + mood words."""
    have = _entry_tokens(entry)
    total = 0.0
    for q in _tokens(query):
        if q in have:
            total += 1.0
            continue
        # ponytail: prefix match instead of a stemmer. Swap in a real stemmer
        # only if the corpus grows past a few hundred entries.
        best = 0.0
        for t in have:
            n = min(len(q), len(t))
            if n >= _PREFIX_MIN and (q.startswith(t) or t.startswith(q)):
                best = _PREFIX_SCORE
                break
        total += best
    return total


def retrieve(vibe: str, k: int = DEFAULT_K) -> list[dict]:
    """The k best-matching corpus entries, best first. Empty if nothing overlaps."""
    hits = [(score(vibe, e), e["name"], e) for e in load_corpus()]
    # Sort by name as the tiebreak so the same query always grounds on the same
    # examples -- an LLM prompt that changes between runs is unreproducible.
    hits = [h for h in hits if h[0] > 0]
    hits.sort(key=lambda h: (-h[0], h[1]))
    return [h[2] for h in hits[:k]]


def _round(obj):
    """Round every float in a nested structure. The corpus stores full float64
    repr; 4 decimals is far below what a grade can resolve and saves ~40% of the
    prompt tokens these examples cost."""
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def format_examples(entries: list[dict]) -> str:
    """Corpus hits as few-shot grounding text.

    The FULL spec goes in, identity fields included. That is the point: a
    complete nearby look is something to edit, whereas a partial one is still 43
    numbers to invent, and the fields left at identity are themselves the signal
    that a real grade moves only a few of them.
    """
    if not entries:
        return ""
    out = [
        "REFERENCE LOOKS from the corpus. These specs were MEASURED from real "
        "reference stills by a closed-form matcher, not written by a model, so "
        "the numbers are known-good. Use the closest one as your starting point "
        "and adjust it to this footage and this request -- do not copy it "
        "verbatim, and do not re-derive every field from scratch.",
    ]
    for e in entries:
        out.append(
            f'\n{e["name"]} [{" ".join(e["tags"])}]\n'
            f'  measured from: {e["provenance"]["reference"]}\n'
            f'  {json.dumps(_round(e["spec"]), separators=(",", ":"))}'
        )
    return "\n".join(out)


def ground(vibe: str, k: int = DEFAULT_K) -> str:
    """The hook vibe.plan_vibe calls: grounding text for `vibe`, or "" if none.

    Append the result to the USER message. It is deliberately not part of
    vibe.SYSTEM: the examples are query-dependent, so putting them in the system
    prompt would defeat prompt caching and mix retrieved data into instructions.
    """
    return format_examples(retrieve(vibe, k))


# ---- self-check -----------------------------------------------------------


def _warmth(entry: dict) -> float:
    """How much warmer this look renders the footage it was measured from:
    the change in (r - b) when the spec is applied to the base still's mean.

    Slope alone does not answer this -- the matcher splits the colour shift
    across slope AND offset, so a look can be warm with a blue-heavy slope
    (`underwater` is). Nor does (r - b) of the output alone: the base footage is
    already red-heavy (0.577 / 0.500 / 0.388), so every grade scores positive.
    The DELTA against the base is the part the look is responsible for.
    """
    import numpy as np

    from .spec import GradeSpec

    m = entry["provenance"]["measured_base"]["mean"]
    src = np.array([[m["r"], m["g"], m["b"]]])
    out = GradeSpec(**entry["spec"]).apply(src)
    return float((out[0, 0] - out[0, 2]) - (src[0, 0] - src[0, 2]))


def _self_check() -> None:
    """`python -m ragvid.looks` -- proves retrieval is signal, not noise."""
    corpus = load_corpus()
    assert corpus, "no corpus: run scripts/build_looks.py"

    from .spec import GradeSpec

    # Every entry must round-trip through the real spec, or the grounding text
    # is teaching the model a schema the code rejects.
    for e in corpus:
        assert set(e["spec"]) == set(GradeSpec.model_fields), e["name"]
        GradeSpec(**e["spec"])

    warm = {e["name"]: _warmth(e) for e in corpus}
    base = sum(warm.values()) / len(warm)
    print(f"corpus: {len(corpus)} entries, mean warmth {base:+.4f}")
    for name, w in sorted(warm.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {w:+.4f}")

    for query, sign in (("warm", +1), ("cool", -1), ("golden sunset", +1), ("cold night", -1)):
        hits = retrieve(query)
        assert hits, query
        got = sum(warm[h["name"]] for h in hits) / len(hits)
        print(f"  {query!r:16s} -> {[h['name'] for h in hits]}  warmth {got:+.4f} vs {base:+.4f}")
        assert (got - base) * sign > 0.02, f"{query}: retrieval is noise ({got:+.4f})"

    assert retrieve("xyzzy") == [], "an unmatched query must ground on nothing"
    assert ground("xyzzy") == ""
    assert "REFERENCE LOOKS" in ground("warm")
    assert retrieve("warm") == retrieve("warm"), "retrieval must be deterministic"
    print("looks self-check OK")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
