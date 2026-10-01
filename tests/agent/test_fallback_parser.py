"""The offline rule-based parser."""

import difflib
from pathlib import Path

import pytest

from pointdraw.agent.fallback_parser import parse, previewed
from pointdraw.agent.prompts import BUILTIN_CATEGORIES, AgentContext
from pointdraw.agent.scoring import item_matches, load_items
from pointdraw.drawing import Library

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


# ---------------------------------------------------------------- the virtual world's list
# What the runtime passes: the 41 library categories plus the four primitive shapes.

LIB = Library.load().categories + ["cross", "circle", "square", "triangle"]


def lib_calls(text, ctx=None):
    return [c.to_dict() for c in parse(text, ctx, LIB)]


def call(name, **args):
    return {"name": name, "args": args}


def here(i, word="here"):
    return {"deixis": word, "word_index": i}


def below(target):
    return {"relation": "below", "target": target}


TREE = AgentContext(previewing="place_drawing(tree)")
EAR = AgentContext(previewing="place_drawing(ear)")
EMPTY = {"empty_space": True}


def test_library_list_is_what_the_runtime_sends():
    # 41 Quick, Draw! categories + 30 kids' favourites (SANA, 1 Oct 2026) + 4 primitives
    assert len(LIB) == 75 and {"flower", "brain", "ear", "house", "tree", "rocket"} <= set(LIB)


@pytest.mark.parametrize(
    "text",
    [
        "a flower in an empty spot",
        "a flower in the empty space",
        "a flower somewhere empty",
        "a flower in a free spot",
        "a flower where there's room",
        "a flower where there is some room",
        "draw a flower wherever it fits",
    ],
)
def test_empty_space_phrases(text):
    assert lib_calls(text) == [call("place_drawing", object="flower", place=EMPTY)]


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "label that one nemotron",
            [call("write_label", text="nemotron", place=below(here(1, "that")))],
        ),
        ("label this cosmos", [call("write_label", text="cosmos", place=below(here(1, "this")))]),
        ("label it magpie", [call("write_label", text="magpie", place=below({"last": True}))]),
        (
            "label this cosmos and this asr",
            [
                call("write_label", text="cosmos", place=below(here(1, "this"))),
                call("write_label", text="asr", place=below(here(4, "this"))),
            ],
        ),
        (
            "Label that thing Hello World.",
            [call("write_label", text="Hello World", place=below(here(1, "that")))],
        ),
        # "and" followed by a pointing word with no text after it stays part of the text
        (
            "label it this and that",
            [call("write_label", text="this and that", place=below({"last": True}))],
        ),
        ("label that says hello", [call("write_label", text="hello")]),
        ("label this", [call("ask_user", question="What should I write?", options=[])]),
        # unchanged: "write X <place>"
        ("write hello there", [call("write_label", text="hello", place=here(2, "there"))]),
        ("write nemo under it", [call("write_label", text="nemo", place=below({"last": True}))]),
        ("write not here", [call("write_label", text="not", place=here(2))]),
    ],
)
def test_labels_on_objects(text, expected):
    assert lib_calls(text) == expected


def test_label_on_a_named_scene_object_and_after_a_drawing():
    ctx = AgentContext(scene_labels=["house"], last_label="house")
    assert lib_calls("label the house home", ctx) == [
        call("write_label", text="home", place=below({"label": "house"}))
    ]
    assert lib_calls("draw a fish here then label it nemo") == [
        call("place_drawing", object="fish", place=here(3)),
        call("write_label", text="nemo", place=below({"last": True})),
    ]


@pytest.mark.parametrize(
    "text,place",
    [
        ("no there", here(1, "there")),
        ("no put it here", here(3)),
        ("move it there", here(2, "there")),
        ("not there here", here(2)),  # the last demonstrative wins
        ("put it here instead", here(2)),
        ("there no here", here(2)),
        ("right there", here(1, "there")),
        ("in the middle", {"named": "middle"}),
        ("somewhere empty", EMPTY),
    ],
)
def test_repointing_a_preview(text, place):
    assert lib_calls(text, TREE) == [call("place_drawing", object="tree", place=place)]


@pytest.mark.parametrize("text", ["no", "nope", "not there", "no not there"])
def test_rejecting_a_preview_cancels(text):
    assert lib_calls(text, TREE) == [call("cancel")]


def test_repointing_reads_the_preview_string():
    json_preview = 'place_drawing({"object":"floor lamp","place":{"deixis":"here","word_index":3}})'
    for prev, obj in [
        ("place_drawing(tree)", "tree"),
        (json_preview, "floor lamp"),
        ("place_drawing(object=tree, size=big)", "tree"),
        ("tree", "tree"),
    ]:
        assert previewed(prev) == ("place_drawing", obj)
        out = lib_calls("no there", AgentContext(previewing=prev))
        assert out == [call("place_drawing", object=obj, place=here(1, "there"))]
    label = AgentContext(previewing="write_label(hello world)")
    assert lib_calls("no there", label) == [
        call("write_label", text="hello world", place=here(1, "there"))
    ]
    assert previewed("place_drawing()") is None
    # without a preview, a bare place is not a command
    assert lib_calls("no there") == []
    assert [c["name"] for c in lib_calls("put it there")] == ["ask_user"]


@pytest.mark.parametrize(
    "text,index",
    [("about this big", 1), ("this big", 0), ("make it this big", 2), ("that size", 0)],
)
def test_size_gesture_during_a_preview(text, index):
    # no place: the runtime keeps the preview's place
    assert lib_calls(text, EAR) == [call("place_drawing", object="ear", size={"word_index": index})]


def test_amounts_are_not_sizes_during_a_preview():
    for text in ("a little bit to the left", "a tiny bit to the left", "move it a little left"):
        assert lib_calls(text, EAR) == [call("adjust", dx=-1, dy=0, scale=1.0)], text
    assert lib_calls("a little bird there") == [
        call("place_drawing", object="bird", place=here(3, "there"), size={"preset": "small"})
    ]


@pytest.mark.parametrize(
    "tail,adjust",
    [
        ("a bit bigger", {"dx": 0, "dy": 0, "scale": 1.25}),
        ("a bit smaller", {"dx": 0, "dy": 0, "scale": 0.8}),
        ("a bit to the left", {"dx": -1, "dy": 0, "scale": 1.0}),
        ("a bit lower", {"dx": 0, "dy": -1, "scale": 1.0}),
        ("and make it much bigger", {"dx": 0, "dy": 0, "scale": 1.5}),
    ],
)
def test_place_then_trailing_correction(tail, adjust):
    assert lib_calls(f"a brain up here {tail}") == [
        call("place_drawing", object="brain", place=here(3)),
        call("adjust", **adjust),
    ]


def test_trailing_correction_after_repointing_a_preview():
    assert lib_calls("no there and a bit bigger", TREE) == [
        call("place_drawing", object="tree", place=here(1, "there")),
        call("adjust", dx=0, dy=0, scale=1.25),
    ]


@pytest.mark.parametrize(
    "text",
    ["a tree here this big", "a tree here a bit", "a sun there and a cloud next to it"],
)
def test_no_trailing_correction_without_a_change(text):
    assert "adjust" not in [c["name"] for c in lib_calls(text)]


@pytest.mark.parametrize(
    "text,thing,probe",
    [
        ("draw a dragon", "a dragon", "dragon"),
        ("a unicorn here", "a unicorn", "unicorn"),
        ("draw me a big red dragon there", "a dragon", "dragon"),
        ("a big penguin there", "a penguin", "penguin"),  # owl and rocket joined the library
        ("draw a volcano", "a volcano", "volcano"),  # on 1 Oct (SANA kids' favourites)
        ("draw dragons", "dragons", "dragons"),
    ],
)
def test_unknown_objects_ask_with_close_categories(text, thing, probe):
    (out,) = lib_calls(text)
    assert out["name"] == "ask_user"
    assert out["args"]["question"].startswith(f"I can't draw {thing} yet.")
    close = difflib.get_close_matches(probe, LIB, n=3, cutoff=0.5)
    if probe.endswith("s"):
        close += [c for c in difflib.get_close_matches(probe[:-1], LIB, 3, 0.5) if c not in close]
    assert out["args"]["options"] == close[:3]
    assert len(out["args"]["options"]) <= 3


def test_unknown_object_examples():
    assert lib_calls("draw a dragon")[0]["args"] == {
        "question": "I can't draw a dragon yet. Which one instead?",
        "options": ["train", "crown", "brain"],  # "train" since the 1 Oct library
    }
    assert lib_calls("draw a volcano")[0]["args"] == {
        "question": "I can't draw a volcano yet. What should I draw instead?",
        "options": [],
    }
    # the same thing while something is previewed: still a question, not a re-point
    assert [c["name"] for c in lib_calls("a unicorn here", TREE)] == ["ask_user"]


def test_general_kinds_ask_which_one():
    assert lib_calls("draw an animal over there")[0]["args"] == {
        "question": "Which animal should I draw?",
        "options": ["cat", "dog", "bird", "fish"],
    }


def test_unknown_object_check_leaves_other_readings_alone():
    # the two-way category question still wins ("bat" is the animal or a baseball bat)
    (bat,) = calls("put a bat there")
    assert bat["args"]["options"] == ["bat", "baseball bat"]
    assert lib_calls("draw something") == [
        call("ask_user", question="What should I draw?", options=[])
    ]
    ctx = AgentContext(previewing="place_drawing(tree)", scene_labels=["house"])
    assert lib_calls("next to the house", ctx) == [
        call(
            "place_drawing",
            object="tree",
            place={"relation": "next_to", "target": {"label": "house"}},
        )
    ]
    assert lib_calls("hmm") == []


def test_library_synonyms():
    assert lib_calls("a boat here")[0]["args"]["object"] == "sailboat"
    assert lib_calls("a ship here")[0]["args"]["object"] == "sailboat"
    assert lib_calls("a tv there")[0]["args"]["object"] == "television"
    assert lib_calls("a lamp there")[0]["args"]["object"] == "floor lamp"
    assert calls("a boat here")[0]["args"]["object"] == "boat"  # a real category wins


def test_answer_to_a_pending_question_is_read_as_a_plain_request():
    """The runtime matches answers to `options` itself; the parser must just not crash or
    misfire (no cancel/confirm/stop/new_sheet/ask) when the answer names an option."""
    ctx = AgentContext(
        scene_labels=["house", "tree"],
        pending_question="The house or the tree?",
        options=["house", "tree"],
    )
    for text in ("the house", "house please", "The house."):
        assert lib_calls(text, ctx) == [call("place_drawing", object="house")], text
    assert lib_calls("the tree please", ctx) == [call("place_drawing", object="tree")]
    assert lib_calls("the second one", ctx) == []
    asked = AgentContext(pending_question="Which bat do you mean?", options=["bat", "baseball bat"])
    assert calls("the baseball bat", asked) == [call("place_drawing", object="baseball bat")]
    q = AgentContext(
        pending_question="I can't draw a dragon yet. Which one instead?", options=["crown"]
    )
    assert lib_calls("crown please", q) == [call("place_drawing", object="crown")]


@pytest.mark.parametrize(
    "text,ctx,name",
    [
        ("what did i draw", None, "look"),
        ("is there room for a sun", None, "look"),
        ("stop", None, "stop"),
        ("halt", TREE, "stop"),
        ("new page", None, "new_sheet"),
        ("clear the page", None, "new_sheet"),
        ("start over", None, "new_sheet"),
        ("a fresh sheet please", None, "new_sheet"),
        ("yes", TREE, "confirm"),
        ("do it", TREE, "confirm"),
    ],
)
def test_short_commands_on_the_library_list(text, ctx, name):
    assert [c["name"] for c in lib_calls(text, ctx)] == [name]


def test_simple_and_detailed_choose_the_drawing_style():
    """rahul, 1 Oct 2026: "a simple X" draws the icon style, "a detailed X" the colouring page."""
    (c,) = lib_calls("draw a detailed tree here")
    assert (c["name"], c["args"]["object"], c["args"]["style"]) == (
        "place_drawing",
        "tree",
        "detailed",
    )
    (c,) = lib_calls("a simple house in the top left")
    assert c["args"]["style"] == "simple"
    assert lib_calls("a tree here")[0]["args"].get("style") is None
    (c,) = lib_calls("draw a fancy dragon")  # the style word is not the object
    assert c["name"] == "ask_user" and "a dragon" in c["args"]["question"]
