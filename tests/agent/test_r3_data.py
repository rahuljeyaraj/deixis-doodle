"""The R3 utterance set is well formed and covers what the brief asks for."""

from collections import Counter
from pathlib import Path

from pointdraw.agent.prompts import BUILTIN_CATEGORIES
from pointdraw.agent.schema import check_against_transcript
from pointdraw.agent.scoring import (
    calls_match,
    canon_call,
    expected_calls,
    load_items,
    percentile,
)

ITEMS = load_items(Path(__file__).parent / "data" / "r3_utterances.jsonl")


def test_size_and_ids():
    assert len(ITEMS) >= 40
    assert len({it.id for it in ITEMS}) == len(ITEMS)
    for it in ITEMS:
        assert it.words == it.text.split()


def test_expected_calls_validate_and_point_at_demonstratives():
    for it in ITEMS:
        for exp in it.expectations:
            probe = type(it)(**{**it.__dict__, "expected": exp, "alt": []})
            for call in expected_calls(probe):
                assert check_against_transcript(call, it.words, list(BUILTIN_CATEGORIES)) == [], (
                    it.id
                )


def test_coverage():
    tags = Counter(t for it in ITEMS for t in it.tags)
    for needed in (
        "deictic_place", "deictic_size", "deictic_object", "relation", "named", "empty_space",
        "preset_size", "relative_size", "implicit", "multi", "correction", "ambiguous",
        "tricky", "preview", "stop", "new_sheet", "connector", "label", "look",
    ):  # fmt: skip
        assert tags[needed] >= 1, needed
    assert tags["ambiguous"] >= 4 and tags["multi"] >= 3 and tags["correction"] >= 3
    ask = [it for it in ITEMS if it.expected[0]["name"] == "ask_user"]
    assert len(ask) == tags["ambiguous"]


def test_scoring_rules():
    exp = [{"name": "place_drawing", "args": {"object": "Tree", "place": {"named": "centre"}}}]
    got = [
        {
            "name": "place_drawing",
            "args": {
                "object": "tree",
                "place": {"named": "the middle"},
                "size": {"preset": "medium"},
            },
        }
    ]
    assert calls_match(got, exp)
    assert not calls_match(got[:0], exp)
    assert canon_call({"name": "adjust", "args": {"dx": -2, "scale": 1.3}}) == (
        "adjust",
        (-1, 0, 1),
    )
    assert calls_match(
        [{"name": "ask_user", "args": {"question": "which?"}}], [{"name": "ask_user", "args": {}}]
    )
    deictic = {"deixis": "here", "word_index": 3}
    assert calls_match(
        [{"name": "place_drawing", "args": {"object": "a", "place": deictic}}],
        [
            {
                "name": "place_drawing",
                "args": {"object": "a", "place": {**deictic, "deixis": "there"}},
            }
        ],
    )


def test_percentile():
    assert percentile([], 50) == 0.0
    xs = [float(i) for i in range(1, 11)]
    assert percentile(xs, 50) == 5.0
    assert percentile(xs, 95) == 10.0
