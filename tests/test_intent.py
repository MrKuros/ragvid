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
