"""Tool argument models and the OpenAI tools JSON."""

import json

import pytest

from pointdraw.agent.schema import (
    PRESET_MM,
    TOOL_ARGS,
    DeicticPlace,
    EmptySpace,
    LabelObject,
    LastObject,
    NamedPlace,
    PresetSize,
    RelationPlace,
    ToolCallError,
    check_against_transcript,
    openai_tools,
    parse_call,
)

BRIEF_TOOLS = {
    "place_drawing", "draw_connector", "write_label", "adjust", "confirm", "cancel", "stop",
    "new_sheet", "ask_user", "look",
}  # fmt: skip
COORD_KEYS = {"x", "y", "x_mm", "y_mm", "mm", "coords", "coordinates", "point", "gcode", "box"}


def _keys(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "properties":
                out.update(v.keys())
            _keys(v, out)
    elif isinstance(node, list):
        for x in node:
            _keys(x, out)
    return out


def test_tools_json_has_every_tool_and_no_coordinates():
    tools = openai_tools()
    assert {t["function"]["name"] for t in tools} == BRIEF_TOOLS == set(TOOL_ARGS)
    for t in tools:
        assert t["type"] == "function"
        params = t["function"]["parameters"]
        assert params["type"] == "object"
        assert t["function"]["description"]
        text = json.dumps(params)
        assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    props = _keys(tools, set())
    assert not props & COORD_KEYS, props & COORD_KEYS
    json.dumps(tools)  # serialisable


def test_place_ref_shapes():
    cases = {
        '{"deixis": "here", "word_index": 3}': DeicticPlace,
        '{"relation": "left_of", "target": {"label": "house"}}': RelationPlace,
        '{"named": "top left corner"}': NamedPlace,
        '{"empty_space": true}': EmptySpace,
    }
    for place, cls in cases.items():
        call = parse_call("place_drawing", f'{{"object": "tree", "place": {place}}}')
        assert isinstance(call.args.place, cls)


def test_between_needs_two_targets():
    with pytest.raises(ToolCallError, match="target2"):
        parse_call(
            "place_drawing",
            {"object": "bird", "place": {"relation": "between", "target": {"label": "a"}}},
        )
    call = parse_call(
        "place_drawing",
        {
            "object": "bird",
            "place": {"relation": "between", "target": {"label": "a"}, "target2": {"last": True}},
        },
    )
    assert isinstance(call.args.place.target2, LastObject)


def test_repairs_common_slips():
    call = parse_call(
        "place_drawing",
        {"object": "sun", "place": "empty_space", "size": "large"},
    )
    assert isinstance(call.args.place, EmptySpace)
    assert call.args.size == PresetSize(preset="big")
    call = parse_call(
        "draw_connector",
        {"from": {"deixis": "sun", "word_index": 3}, "to": {"deixis": "it"}},
    )
    assert call.args.from_ == LabelObject(label="sun")
    assert isinstance(call.args.to, LastObject)
    call = parse_call(
        "place_drawing",
        {"object": "cat", "place": {"relation": "next to", "target": "house", "target2": None}},
    )
    assert call.args.place.relation == "next_to"
    call = parse_call("ask_user", {"question": "Which?", "options": list("abcdef")})
    assert len(call.args.options) == 4


@pytest.mark.parametrize(
    "name,args,match",
    [
        ("fly", {}, "unknown tool"),
        ("place_drawing", "{not json", "not valid JSON"),
        ("place_drawing", {"object": "tree", "place": {"foo": 1}}, "PlaceRef"),
        ("place_drawing", {"object": "tree", "x": 3}, "Extra inputs"),
        ("adjust", {"dx": 20}, "less than or equal"),
        ("confirm", {"now": True}, "Extra inputs"),
        ("write_label", {"text": ""}, "at least 1"),
    ],
)
def test_bad_calls_give_readable_errors(name, args, match):
    with pytest.raises(ToolCallError, match=match):
        parse_call(name, args)


def test_no_arg_tools_accept_empty_arguments():
    for name in ("confirm", "cancel", "stop", "new_sheet"):
        assert parse_call(name, "").args_dict() == {}
        assert parse_call(name, None).name == name


def test_to_dict_uses_from_alias_and_drops_nones():
    call = parse_call(
        "draw_connector",
        {"kind": "arrow", "from": {"deixis": "this", "word_index": 2}, "to": {"label": "fish"}},
    )
    d = call.to_dict()
    assert d == {
        "name": "draw_connector",
        "args": {
            "kind": "arrow",
            "from": {"deixis": "this", "word_index": 2},
            "to": {"label": "fish"},
        },
    }
    assert parse_call(d["name"], d["args"]).to_dict() == d
    assert "place" not in parse_call("place_drawing", {"object": "cat"}).args_dict()


def test_check_against_transcript():
    words = "put a tree here this big".split()
    ok = parse_call(
        "place_drawing",
        {"object": "tree", "place": {"deixis": "here", "word_index": 3}, "size": {"word_index": 4}},
    )
    assert check_against_transcript(ok, words, ["tree"]) == []
    bad = parse_call(
        "place_drawing", {"object": "tree", "place": {"deixis": "here", "word_index": 2}}
    )
    (msg,) = check_against_transcript(bad, words)
    assert "word 2 is 'tree'" in msg and "'here' is word 3" in msg
    far = parse_call(
        "place_drawing", {"object": "tree", "place": {"deixis": "here", "word_index": 9}}
    )
    assert "past the last word" in check_against_transcript(far, words)[0]
    unicorn = parse_call("place_drawing", {"object": "unicorn"})
    assert "not drawable" in check_against_transcript(unicorn, words, ["tree"])[0]


def test_presets_match_plan():
    assert PRESET_MM == {"tiny": 20, "small": 35, "medium": 55, "big": 80, "whole_page": None}


def test_article_as_label_is_sent_back():
    """Live 30 Sep: Lightning sent {"label": "the"} for "an arrow from the brain to the hand"."""
    from pointdraw.agent.schema import check_against_transcript, parse_call

    words = "an arrow from the brain to the hand".split()
    bad = parse_call("draw_connector", {"from": {"label": "the"}, "to": {"label": "hand"}})
    assert any("not a name" in p for p in check_against_transcript(bad, words))
    good = parse_call("draw_connector", {"from": {"label": "brain"}, "to": {"label": "hand"}})
    assert check_against_transcript(good, words) == []


def test_a_bare_relation_is_not_a_named_place():
    """Live Nemotron sent {"named": "inside"} for "put a hand inside that" (1 Oct)."""
    for word in ("inside", "next to", "in", "between", "left_of", "inside that", "next to it"):
        with pytest.raises(ToolCallError, match="relation, not a place name"):
            parse_call("place_drawing", {"object": "hand", "place": {"named": word}})
    for ok in ("top left corner", "middle", "on the roof", "inside the house"):
        call = parse_call("place_drawing", {"object": "hand", "place": {"named": ok}})
        assert isinstance(call.args.place, NamedPlace)
    with pytest.raises(ToolCallError, match='"deixis": "that"'):
        parse_call("place_drawing", {"object": "hand", "place": {"named": "inside that one"}})
