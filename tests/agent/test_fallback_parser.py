"""The offline rule-based parser."""

from pathlib import Path

import pytest

from pointdraw.agent.fallback_parser import parse
from pointdraw.agent.prompts import BUILTIN_CATEGORIES, AgentContext
from pointdraw.agent.scoring import item_matches, load_items

CATS = list(BUILTIN_CATEGORIES)
PREVIEW = AgentContext(previewing="place_drawing(sun)")


def calls(text, ctx=None):
    return [c.to_dict() for c in parse(text, ctx, CATS)]


def names(text, ctx=None):
    return [c["name"] for c in calls(text, ctx)]


# Phrasings deliberately different from the R3 set, to catch over-fitting.
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "a cow here",
            [("place_drawing", {"object": "cow", "place": {"deixis": "here", "word_index": 2}})],
        ),
        (
            "Put an apple there.",
            [("place_drawing", {"object": "apple", "place": {"deixis": "there", "word_index": 3}})],
        ),
        (
            "a kitty here",
            [("place_drawing", {"object": "cat", "place": {"deixis": "here", "word_index": 2}})],
        ),
        (
            "a bus there that long",
            [
                (
                    "place_drawing",
                    {
                        "object": "bus",
                        "place": {"deixis": "there", "word_index": 2},
                        "size": {"word_index": 3},
                    },
                )
            ],
        ),
        (
            "line from this to that",
            [
                (
                    "draw_connector",
                    {
                        "kind": "line",
                        "from": {"deixis": "this", "word_index": 2},
                        "to": {"deixis": "that", "word_index": 4},
                    },
                )
            ],
        ),
        (
            "write hello world here",
            [
                (
                    "write_label",
                    {"text": "hello world", "place": {"deixis": "here", "word_index": 3}},
                )
            ],
        ),
        ("write cancel", [("write_label", {"text": "cancel"})]),
        (
            "a huge whale in the centre",
            [
                (
                    "place_drawing",
                    {"object": "whale", "place": {"named": "middle"}, "size": {"preset": "big"}},
                )
            ],
        ),
        (
            "a spider anywhere",
            [("place_drawing", {"object": "spider", "place": {"empty_space": True}})],
        ),
    ],
)
def test_held_out_phrasings(text, expected):
    assert calls(text) == [{"name": n, "args": a} for n, a in expected]


@pytest.mark.parametrize(
    "text,name",
    [
        ("yeah", "confirm"),
        ("go ahead", "confirm"),
        ("nope", "cancel"),
        ("never mind", "cancel"),
        ("halt", "stop"),
        ("please stop now", "stop"),
        ("clear the page", "new_sheet"),
        ("another sheet", "new_sheet"),
        ("smaller", "adjust"),
        ("a bit higher", "adjust"),
        ("where is the sun", "look"),
    ],
)
def test_short_commands(text, name):
    assert names(text, PREVIEW) == [name]


def test_adjust_directions():
    (a,) = calls("much bigger and to the right", PREVIEW)
    assert a["args"] == {"dx": 3, "dy": 0, "scale": 1.5}
    (a,) = calls("down a bit", PREVIEW)
    assert a["args"] == {"dx": 0, "dy": -1, "scale": 1.0}


def test_stop_wins_unless_it_is_label_text():
    assert names("put a tree here stop") == ["stop"]
    assert names("write stop there") == ["write_label"]


def test_unknown_input_gives_nothing_or_a_question():
    assert parse("", None, CATS) == []
    assert parse("hmm", None, CATS) == []
    assert names("draw me a dragon") == ["ask_user"]


def test_categories_from_argument():
    out = [c.to_dict() for c in parse("a robot here", None, ["robot"])]
    assert out[0]["args"]["object"] == "robot"


def test_r3_accuracy():
    items = load_items(Path(__file__).parent / "data" / "r3_utterances.jsonl")
    ok = sum(item_matches(it, calls(it.text, AgentContext.from_dict(it.context))) for it in items)
    assert ok / len(items) >= 0.9, f"{ok}/{len(items)}"
