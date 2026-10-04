"""The vacuum's agent: the rule parser, and Nemotron's replies checked against the profile."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from deixis.actions import tools_for_agent
from pointdraw.vacuum.agent import Context, RuleAgent, VacuumAgent, parse, user_message
from pointdraw.vacuum.profile import vacuum_profile

PROFILE = vacuum_profile()
HERE1 = {"place": {"deixis": "here", "word_index": 1}}


@pytest.mark.parametrize(
    "said,calls",
    [
        ("clean here", [("clean", HERE1)]),
        ("Clean here.", [("clean", HERE1)]),
        ("please clean from here to there", [("clean", {"place": {
            "from": {"deixis": "here", "word_index": 3},
            "to": {"deixis": "there", "word_index": 5}}})]),
        ("clean", [("clean", {"place": {"pointed": True}})]),
        ("here", [("clean", {"place": {"deixis": "here", "word_index": 0}})]),
        ("no here", [("clean", HERE1)]),
        ("yes", [("confirm", {})]),
        ("yes go ahead", [("confirm", {})]),
        ("that's right", [("confirm", {})]),
        ("no", [("cancel", {})]),
        ("never mind", [("cancel", {})]),
        ("a bit left", [("adjust", {"dx": -1})]),
        ("a bit more to the right", [("adjust", {"dx": 1})]),
        ("much further away", [("adjust", {"dy": 3})]),
        ("a bit closer", [("adjust", {"dy": -1})]),
        ("bigger", [("adjust", {"scale": 1.25})]),
        ("stop", [("stop", {})]),
        ("wait", [("stop", {})]),
        ("go back to the dock", [("dock", {})]),
        ("go home", [("dock", {})]),
        ("what is that", []),
    ],
)  # fmt: skip
def test_rules(said, calls):
    got = parse(said.split(), Context(previewing=True))
    assert [(c["name"], c["args"]) for c in got] == calls


def test_right_alone_is_a_yes_only_to_a_preview():
    assert parse(["right"], Context(previewing=True)) == [{"name": "confirm", "args": {}}]
    assert parse(["right"], Context()) == [{"name": "adjust", "args": {"dx": 1}}]


def test_the_rule_agent_checks_calls_against_the_profile():
    res = RuleAgent(PROFILE).understand(["clean", "here"])
    assert [str(c) for c in res.calls] == ['clean({"place":{"deixis":"here","word_index":1}})']
    assert res.source == "fallback" and res.errors == []


def test_the_prompt_numbers_the_words_and_says_what_is_going_on():
    msg = user_message(["yes", "please"], Context(previewing=True, question="Clean here?"))
    assert "waits for the person's yes or no" in msg and 'just asked: "Clean here?"' in msg
    assert msg.endswith("Words: 0:yes 1:please")


class FakeOpenAI:
    """Replies with the given tool calls, one reply per request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        calls = [
            SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
            for i, (n, a) in enumerate(self.replies.pop(0))
        ]
        msg = SimpleNamespace(tool_calls=calls, content="")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def agent(client):
    return VacuumAgent(PROFILE, tools_for_agent(PROFILE), client=client, model="test-model")


def test_nemotron_gets_the_vacuums_tools_and_its_calls_are_checked():
    client = FakeOpenAI([("clean", HERE1)])
    res = agent(client).understand(["clean", "here"])
    assert res.source == "nemotron" and [c.name for c in res.calls] == ["clean"]
    sent = client.requests[0]
    names = [t["function"]["name"] for t in sent["tools"]]
    assert names[:2] == ["clean", "dock"] and "locate" not in names and "measure" not in names
    assert sent["messages"][1]["content"].endswith("Words: 0:clean 1:here")


def test_a_wrong_word_index_is_sent_back_once():
    wrong = ("clean", {"place": {"deixis": "here", "word_index": 0}})
    client = FakeOpenAI([wrong], [("clean", HERE1)])
    res = agent(client).understand(["clean", "here"])
    assert res.attempts == 2 and [c.name for c in res.calls] == ["clean"]
    feedback = client.requests[1]["messages"][-1]["content"]
    assert "word 0 is 'clean', not a demonstrative; 'here' is word 1" in feedback


def test_unusable_replies_fall_back_to_the_rules():
    bad = ("clean", {"place": {"x": 100, "y": 200}})  # coordinates: never allowed
    res = agent(FakeOpenAI([bad], [bad])).understand(["clean", "here"])
    assert res.source == "fallback" and [c.name for c in res.calls] == ["clean"]
    assert res.errors and "clean" in res.errors[0]


def test_no_key_falls_back_to_the_rules(monkeypatch):
    from pydantic import SecretStr

    a = agent(None)
    a.settings = a.settings.model_copy(update={"nebius_api_key": SecretStr("")})
    res = a.understand(["go", "home"])
    assert res.source == "fallback" and [c.name for c in res.calls] == ["dock"]
    assert "NEBIUS_API_KEY" in res.errors[0]
