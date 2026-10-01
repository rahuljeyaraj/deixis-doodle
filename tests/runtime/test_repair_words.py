"""Known ASR slips on drawable words are undone before the agent sees them."""

import pytest

from deixis.types import Word
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import repair_words

CATS = Planner().categories


def fix(s: str) -> str:
    ws = [Word(x, i * 0.5, i * 0.5 + 0.4) for i, x in enumerate(s.split())]
    return " ".join(w.text for w in repair_words(ws, CATS))


@pytest.mark.parametrize(
    "heard,fixed",
    [
        ("across here", "a cross here"),
        ("put across", "put a cross"),
        ("a near here", "an ear here"),
        ("a son here", "a sun here"),
        ("an i next to it", "an eye next to it"),
        ("draw a line across the page", "draw a line across the page"),
        ("a nail", "a nail"),
        ("near the house", "near the house"),
        # rahul's voice, live 1 Oct: "this big" heard as "this week" and "best bit"
        ("this week", "this big"),
        ("draw a cross here this week", "draw a cross here this big"),
        ("about that weak please", "about that big please"),
        ("best bit", "this big"),
        ("draw it this week then a tree", "draw it this week then a tree"),
        ("the best bit of the drawing", "the best bit of the drawing"),
        ("next week", "next week"),
    ],
)
def test_repair(heard, fixed):
    assert fix(heard) == fixed


def test_times_stay_ordered():
    ws = repair_words([Word("across", 1.0, 1.6), Word("here", 1.7, 2.0)], CATS)
    assert [w.text for w in ws] == ["a", "cross", "here"]
    assert ws[0].t0 == 1.0 and ws[0].t1 <= ws[1].t0 and ws[1].t1 == 1.6


def test_shared_label_text_is_split():
    """Live 30 Sep: two labels both came back as "Cosmos and ASR"."""
    from pointdraw.agent.schema import parse_call
    from pointdraw.runtime.runtime import split_shared_labels

    two = [
        parse_call("write_label", {"text": "Cosmos and ASR"}),
        parse_call("write_label", {"text": "Cosmos and ASR"}),
    ]
    assert [c.args.text for c in split_shared_labels(two)] == ["Cosmos", "ASR"]
    one = [parse_call("write_label", {"text": "salt and pepper"})]
    assert split_shared_labels(one) == one
