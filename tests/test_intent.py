"""The schema a 20B model has to fill in, and the English it renders back to.

Two claims are load-bearing here and neither is "no exception raised":

  * the schema is strict in the exact sense Groq's constrained decoding
    enforces (properties AND required, matching, at every object level), and
  * it is measurably SMALLER than the 43-float schema it replaces. If it were
    not, the whole architecture change would be paying tokens for nothing, so
    the size comparison is asserted, not assumed.
"""

from __future__ import annotations

import json

import pytest

from ragvid.intent import (
    AMOUNTS,
    DIRECTIONS,
    OPS,
    STRENGTH_MIX,
    STRENGTHS,
    TARGETS,
    Intent,
    Op,
    describe,
)
from ragvid.providers.openai_compat import missing_fields
from ragvid.spec import GradeSpec


def every_op(**kw) -> Intent:
    return Intent(ops=[Op(op=o, **kw) for o in OPS])


# ---- schema ---------------------------------------------------------------


def walk_objects(node, path="$"):
    """Every object-typed subschema, including the ones inside arrays."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        if node.get("type") == "array":
            yield from walk_objects(node["items"], path + "[]")
        for name, sub in node.get("properties", {}).items():
            yield from walk_objects(sub, f"{path}.{name}")


def test_schema_is_strict_at_every_object_level():
    """Groq rejects a property that is present but not required, so the two
    lists must be identical everywhere -- not just at the top level."""
    schema = Intent.llm_json_schema()
    levels = list(walk_objects(schema))
    assert len(levels) == 2, [p for p, _ in levels]  # the intent, and one op
    for path, node in levels:
        assert node["properties"], path
        assert sorted(node["required"]) == sorted(node["properties"]), path
        assert node["additionalProperties"] is False, path


def test_schema_enums_cannot_drift_from_the_model():
    op = Intent.llm_json_schema()["properties"]["ops"]["items"]["properties"]
    assert op["op"]["enum"] == list(OPS)
    assert op["dir"]["enum"] == list(DIRECTIONS)
    assert op["amount"]["enum"] == list(AMOUNTS)
    assert op["target"]["enum"] == list(TARGETS)
    assert Intent.llm_json_schema()["properties"]["strength"]["enum"] == list(STRENGTHS)


def test_a_real_intent_satisfies_its_own_schema():
    """The same validator the providers run over a model's reply."""
    data = json.loads(every_op(target="").model_dump_json())
    assert missing_fields(Intent.llm_json_schema(), data) == []
    assert Intent(**data) == every_op(target="")


def test_the_model_cannot_emit_a_number():
    """The entire point: there is no float anywhere for a model to invent."""
    for _, node in walk_objects(Intent.llm_json_schema()):
        for name, prop in node["properties"].items():
            assert prop.get("type") in ("string", "array"), (name, prop)


@pytest.mark.parametrize("bad", [
    {"ops": [{"op": "sharpen"}]},              # not in the vocabulary
    {"ops": [{"op": "warmth", "dir": "hot"}]},
    {"ops": [{"op": "warmth", "amount": 0.5}]},  # a number, which is the failure mode
    {"strength": "half"},
])
def test_closed_vocabularies_are_actually_closed(bad):
    with pytest.raises(Exception):
        Intent(**bad)


def test_the_schema_is_smaller_than_the_one_it_replaces(capsys):
    intent = len(json.dumps(Intent.llm_json_schema(), separators=(",", ":")))
    spec = len(json.dumps(GradeSpec.llm_json_schema(), separators=(",", ":")))
    with capsys.disabled():
        print(f"\n  schema bytes: intent {intent}  vs  GradeSpec {spec}"
              f"  ({spec / intent:.1f}x smaller)")
    # Not just "smaller": the roadmap's gate is that token cost measurably goes
    # DOWN, and a 5% saving would not be worth an architecture change.
    assert intent < spec / 2


def test_an_intent_is_smaller_on_the_wire_than_the_spec_it_compiles_to(capsys):
    """The reply is the other half of the token bill.

    Measured against a REALISTIC intent, not the worst case: a real grade is
    three to eight moves, which is what the model will actually emit. The
    worst case -- every verb at once, which no sentence produces -- is printed
    alongside because it is LARGER than the spec, and pretending otherwise
    would be exactly the kind of buried number this project keeps getting
    caught by."""
    real = len(Intent(ops=[
        Op(op="warmth"), Op(op="exposure", dir="down"), Op(op="contrast"),
        Op(op="saturation", dir="down"), Op(op="grain", amount="subtle"),
    ], strength="strong").model_dump_json())
    worst = len(every_op().model_dump_json())
    spec = len(GradeSpec.identity().model_dump_json())
    with capsys.disabled():
        print(f"  reply bytes:  intent {real} (5 verbs, typical)  "
              f"vs GradeSpec {spec} ({spec / real:.1f}x)   "
              f"[worst case, all {len(OPS)} verbs: {worst}]")
    assert real < spec * 0.55


# ---- English --------------------------------------------------------------


def test_every_verb_renders_a_sentence_in_both_directions():
    for direction in DIRECTIONS:
        lines = describe(every_op(dir=direction))
        assert len(lines) == len(OPS)
        for line in lines:
            assert line and "{" not in line       # no unfilled template slot
            assert line == line.strip().lower() or line[0].islower()


def test_an_empty_intent_says_nothing():
    assert describe(Intent()) == []
    assert describe(Intent(strength="full")) == []


def test_the_sentences_read_like_a_person_wrote_them():
    lines = describe(Intent(ops=[
        Op(op="warmth", amount="subtle"),
        Op(op="highlights", dir="down"),
        Op(op="saturation", dir="down", target="green", amount="strong"),
        Op(op="shadow_tint", target="teal"),
        Op(op="exposure", dir="down", target="skin"),
    ]))
    assert lines == [
        "warmed it up a little",
        "pulled the highlights down",
        "drained the colour in the greens a lot",
        "pushed teal into the shadows",
        "darkened it in the skin tones",
    ]


def test_a_reduced_strength_quotes_the_number_the_compiler_will_use():
    """A sentence that disagrees with the grade is worse than no sentence."""
    for s in STRENGTHS:
        line = describe(Intent(ops=[Op(op="warmth")], strength=s))[-1]
        if s == "full":
            assert "strength" not in line
        else:
            assert f"{STRENGTH_MIX[s]:.0%}" in line


# ---- regions (roadmap B1) -------------------------------------------------


def test_a_region_reads_as_a_sentence_a_person_would_say():
    """The list of phrases IS the UI, so "darkened the top" is the requirement,
    not "exposure -0.35 in region linear/top"."""
    lines = describe(Intent(ops=[
        Op(op="exposure", dir="down", target="top"),
        Op(op="exposure", target="center", amount="subtle"),
        Op(op="warmth", target="left"),
        Op(op="tint", dir="down", target="right"),
        Op(op="contrast", target="center"),
        Op(op="saturation", dir="down", target="edges"),
        Op(op="shadows", target="bottom"),
    ]))
    assert lines == [
        "darkened the top",
        "brightened the middle a little",
        "warmed the left side up",
        "pushed the right side green",
        "added contrast in the middle",
        "drained the colour at the edges",
        "lifted the shadows in the bottom",
    ]


def test_a_tint_verb_in_a_region_still_names_a_colour():
    """`target` answers "which colour to add" for the two tint verbs and "which
    pixels" for everything else. A region takes the second job and leaves the
    first, so the sentence has to fall back to the same default colour the
    compiler does -- it used to print a literal "{c}"."""
    from ragvid.intent import DEFAULT_TINT

    assert describe(Intent(ops=[Op(op="shadow_tint", target="top")])) == \
        [f"pushed {DEFAULT_TINT['shadow_tint']} into the shadows in the top"]
    assert describe(Intent(ops=[Op(op="highlight_tint", target="center")])) == \
        [f"pushed {DEFAULT_TINT['highlight_tint']} into the highlights in the middle"]


def test_a_colour_target_still_reads_as_a_colour():
    """The two kinds of target share one field, so the regions must not have
    changed what a colour name does."""
    assert describe(Intent(ops=[Op(op="saturation", dir="down", target="green")])) == \
        ["drained the colour in the greens"]
    assert describe(Intent(ops=[Op(op="shadow_tint", target="teal")])) == \
        ["pushed teal into the shadows"]


def test_a_region_on_a_texture_verb_never_reaches_the_sentence():
    """grain lives in the ffmpeg chain and cannot be masked, so promising it in
    the list would describe a move that did not happen."""
    op = Op(op="grain", target="top")
    assert op.target == ""
    assert describe(Intent(ops=[op])) == ["added grain"]


@pytest.mark.parametrize("region", ["top", "bottom", "left", "right", "center", "edges"])
def test_every_region_word_is_in_the_schema_and_has_a_geometry(region):
    """A word the model can emit that the compiler cannot resolve would compile
    to a silent no-op with a sentence attached."""
    from ragvid.region import for_target

    assert region in TARGETS
    assert region in json.dumps(Intent.llm_json_schema())
    assert for_target(region) is not None


def test_a_colour_name_is_not_a_region():
    from ragvid.region import for_target

    assert for_target("green") is None and for_target("") is None


# ---- semantic targets (roadmap B2) ----------------------------------------
#
# The second mask source costs the model FIVE more words and nothing else. That
# is the whole claim of B2 being a mask source rather than an architecture: the
# schema grows by five enum entries, `Op` is unchanged, and describe() renders
# them through the machinery the geometric regions already installed.


@pytest.mark.parametrize("word", ["sky", "foliage", "person", "water", "buildings"])
def test_every_semantic_word_is_in_the_schema_and_has_a_region(word):
    """Same guard as the geometric words: a word the model can emit that the
    compiler cannot resolve is a silent no-op with a sentence attached."""
    from ragvid.region import for_target

    assert word in TARGETS
    assert word in json.dumps(Intent.llm_json_schema())
    r = for_target(word)
    assert r is not None and r.shape == "semantic"


def test_the_word_list_and_the_class_map_cannot_drift():
    """`Target` has to spell the words (a Literal cannot be built from a runtime
    tuple) while segment.CLASSES holds the model side. This is the seam where
    adding a class and forgetting the vocabulary — or the reverse — would show
    up as a KeyError at grade time."""
    from ragvid.intent import GEOMETRIC, REGIONS, SEMANTIC
    from ragvid.region import _FOR_TARGET
    from ragvid.segment import CLASSES

    assert SEMANTIC == tuple(CLASSES)
    assert set(SEMANTIC) <= set(TARGETS)
    assert set(SEMANTIC).isdisjoint(GEOMETRIC), "a word cannot be both a place and a thing"
    assert REGIONS == GEOMETRIC + SEMANTIC
    assert set(REGIONS) <= set(_FOR_TARGET), "every region word needs a Region"


def test_a_semantic_target_reads_as_the_thing_it_names():
    """"made the sky moodier" is the sentence the roadmap promised. It comes out
    of the SAME pronoun substitution the geometric regions use -- "darkened it"
    has exactly one slot and the subject fills it."""
    assert describe(Intent(ops=[
        Op(op="exposure", dir="down", target="sky"),
        Op(op="saturation", dir="down", target="sky", amount="subtle"),
        Op(op="contrast", target="foliage"),
        Op(op="warmth", target="person"),
        Op(op="exposure", dir="down", target="buildings"),
        Op(op="saturation", target="water", amount="strong"),
    ])) == [
        "darkened the sky",
        "drained the colour in the sky a little",
        "added contrast in the foliage",
        "warmed the person up",
        "darkened the buildings",
        "richened the colour in the water a lot",
    ]


def test_a_tint_verb_on_a_subject_still_names_a_colour():
    """`target` answers "which colour to add" for the two tint verbs. A semantic
    target takes the "which pixels" job and leaves the colour, exactly as a
    geometric one does -- it used to print a literal "{c}"."""
    from ragvid.intent import DEFAULT_TINT

    assert describe(Intent(ops=[Op(op="shadow_tint", target="water")])) == \
        [f"pushed {DEFAULT_TINT['shadow_tint']} into the shadows in the water"]


def test_a_subject_on_a_texture_verb_never_reaches_the_sentence():
    """grain lives in the ffmpeg chain and cannot be masked. The Op validator
    already dropped geometric regions there; semantic ones join them for free by
    being in REGIONS, and this is the test that they did."""
    op = Op(op="grain", target="sky")
    assert op.target == ""
    assert describe(Intent(ops=[op])) == ["added grain"]


def test_skin_is_still_a_colour_and_not_a_mask():
    """The one word the segmentation model does NOT take over. ADE20K has no
    skin class, and a hue qualifier follows a face through movement and a cut
    where a per-frame matte does not."""
    from ragvid.intent import SEMANTIC
    from ragvid.region import for_target

    assert "skin" not in SEMANTIC
    assert for_target("skin") is None
    assert describe(Intent(ops=[Op(op="saturation", dir="down", target="skin")])) == \
        ["drained the colour in the skin tones"]
