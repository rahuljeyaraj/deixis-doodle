"""System prompt, few-shot examples, categories and the user message."""

import json
import re

import pytest

from pointdraw.agent.nemotron import calls_from_text
from pointdraw.agent.prompts import (
    BUILTIN_CATEGORIES,
    PROMPTS,
    AgentContext,
    load_categories,
    system_prompt,
    user_message,
)
from pointdraw.agent.schema import check_against_transcript, parse_call


@pytest.mark.parametrize("version", sorted(PROMPTS))
def test_few_shot_examples_are_valid_calls(version):
    _, examples = PROMPTS[version]
    for text, _ctx, calls_text in examples:
        raw = calls_from_text(calls_text)
        assert len(raw) == len(calls_text.splitlines()), calls_text
        for rc in raw:
            call = parse_call(rc.function.name, rc.function.arguments)
            assert check_against_transcript(call, text.split(), list(BUILTIN_CATEGORIES)) == []


def test_few_shot_examples_are_not_test_items():
    from pathlib import Path

    data = Path(__file__).parent / "data" / "r3_utterances.jsonl"
    texts = {json.loads(line)["text"] for line in data.read_text().splitlines() if line}
    for _, examples in PROMPTS.values():
        assert not {e[0] for e in examples} & texts


@pytest.mark.parametrize("version", sorted(PROMPTS))
def test_system_prompt_mentions_rules_and_categories(version):
    p = system_prompt(["cat", "dog"], version)
    assert "Drawable categories: cat, dog" in p
    assert "Never output coordinates" in p
    assert "{" in p and "{{" not in p  # format braces resolved
    assert "0=draw 1=a 2=cat 3=here" in p


def test_user_message_format():
    ctx = AgentContext(scene_labels=["house", "tree"], last_label="tree", previewing="x(y)")
    msg = user_message(["Put", "a", "tree", "here."], ctx)
    assert msg.splitlines() == [
        "Scene: house, tree (last)",
        "Previewing: x(y)",
        'Utterance: "Put a tree here."',
        "Words: 0=put 1=a 2=tree 3=here",
    ]
    assert user_message(["yes"]) == 'Utterance: "yes"\nWords: 0=yes'


@pytest.mark.parametrize(
    "content",
    [
        ["Cat", "dog"],
        [{"name": "cat"}, {"category": "dog"}],
        {"categories": ["cat", "dog"]},
        {"items": [{"label": "cat"}, {"name": "dog"}]},
        {"cat": {"n": 3}, "dog": {"n": 1}},
    ],
)
def test_load_categories_shapes(tmp_path, content):
    p = tmp_path / "library" / "index.json"
    p.parent.mkdir()
    p.write_text(json.dumps(content))
    assert load_categories(p) == ["cat", "dog"]


def test_load_categories_falls_back(tmp_path):
    assert load_categories(tmp_path / "missing.json") == list(BUILTIN_CATEGORIES)
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert load_categories(bad) == list(BUILTIN_CATEGORIES)


def test_builtin_categories_are_plain_names():
    assert len(BUILTIN_CATEGORIES) == len(set(BUILTIN_CATEGORIES)) >= 60
    assert all(re.fullmatch(r"[a-z ]+", c) for c in BUILTIN_CATEGORIES)
