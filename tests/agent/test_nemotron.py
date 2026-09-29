"""The Nemotron client, with a fake OpenAI client (no network, no key)."""

import json
from types import SimpleNamespace as NS

import pytest
from pydantic import SecretStr

from pointdraw.agent.nemotron import (
    AgentUnavailable,
    NemotronAgent,
    calls_from_text,
    understand,
)
from pointdraw.agent.prompts import BUILTIN_CATEGORIES
from pointdraw.config import Settings

WORDS = "put a tree here".split()
GOOD = {"object": "tree", "place": {"deixis": "here", "word_index": 3}}


def reply(*calls, content=None):
    tcs = [
        NS(id=f"call-{i}", type="function", function=NS(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    msg = NS(content=content, tool_calls=tcs or None)
    usage = NS(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    return NS(choices=[NS(message=msg, finish_reason="tool_calls")], usage=usage)


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def agent(client, **kw):
    s = Settings(nebius_api_key=SecretStr("test"), nemotron_model="")
    return NemotronAgent(s, client=client, categories=list(BUILTIN_CATEGORIES), **kw)


def test_first_try_success_and_request_shape():
    fake = FakeClient(reply(("place_drawing", GOOD)))
    res = agent(fake).run(WORDS)
    assert res.ok and res.attempts == 1 and not res.errors
    assert [c.to_dict() for c in res.calls] == [{"name": "place_drawing", "args": GOOD}]
    req = fake.requests[0]
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert req["tool_choice"] == "auto"
    assert req["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert len(req["tools"]) == 10
    assert req["messages"][1]["content"].endswith("Words: 0=put 1=a 2=tree 3=here")
    assert res.usage == {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


def test_retry_feeds_validation_error_back():
    bad = {"object": "tree", "place": {"deixis": "here", "word_index": 1}}
    fake = FakeClient(reply(("place_drawing", bad)), reply(("place_drawing", GOOD)))
    res = agent(fake).run(WORDS)
    assert res.ok and res.attempts == 2
    assert "word 1 is 'a'" in res.errors[0]
    msgs = fake.requests[1]["messages"]
    assert msgs[2]["role"] == "assistant" and msgs[2]["tool_calls"][0]["id"] == "call-0"
    assert msgs[3] == {
        "role": "tool",
        "tool_call_id": "call-0",
        "content": "not executed: the reply had errors",
    }
    assert msgs[4]["role"] == "user" and "word 1 is 'a'" in msgs[4]["content"]


def test_gives_up_after_one_retry_keeping_valid_calls():
    bad = ("place_drawing", {"object": "tree", "place": {"nope": 1}})
    fake = FakeClient(reply(bad), reply(("stop", {}), bad))
    res = agent(fake).run(WORDS)
    assert res.attempts == 2 and [c.name for c in res.calls] == ["stop"]
    assert len(fake.requests) == 2


def test_plain_text_is_retried_then_returned_as_speech():
    fake = FakeClient(reply(content="Where?"), reply(content="Where should it go?"))
    res = agent(fake).run(WORDS)
    assert not res.ok and res.speech == "Where should it go?" and res.attempts == 2
    assert fake.requests[1]["messages"][-1]["role"] == "user"


def test_calls_written_as_text_are_salvaged():
    fake = FakeClient(reply(content="cancel()"))
    res = agent(fake).run(["no"])
    assert [c.name for c in res.calls] == ["cancel"] and res.salvaged and res.attempts == 1
    raw = calls_from_text('place_drawing({"object": "cat"})\nhello\n`stop()`')
    assert [(r.function.name, r.function.arguments) for r in raw] == [
        ("place_drawing", '{"object": "cat"}'),
        ("stop", "{}"),
    ]
    assert calls_from_text("fly()") == []


def test_guard_replaces_commands_after_write():
    fake = FakeClient(reply(("new_sheet", {})))
    res = agent(fake).run("write new sheet there".split())
    assert res.guarded
    assert [c.to_dict() for c in res.calls] == [
        {
            "name": "write_label",
            "args": {"text": "new sheet", "place": {"deixis": "there", "word_index": 3}},
        }
    ]
    fake = FakeClient(reply(("stop", {})))
    res = agent(fake).run("write stop here".split())
    assert not res.guarded and [c.name for c in res.calls] == ["stop"]  # never undo a stop


def test_unknown_category_is_sent_back():
    fake = FakeClient(
        reply(("place_drawing", {"object": "unicorn"})),
        reply(("ask_user", {"question": "A horse?", "options": ["horse"]})),
    )
    res = agent(fake).run("draw a unicorn".split())
    assert "not drawable" in res.errors[0] and res.calls[0].name == "ask_user"


def test_api_errors_become_agent_unavailable():
    fake = FakeClient(TimeoutError("slow"))
    with pytest.raises(AgentUnavailable, match="TimeoutError"):
        agent(fake).run(WORDS)


def test_no_key_is_unavailable():
    a = NemotronAgent(Settings(nebius_api_key=SecretStr("")), categories=["tree"])
    with pytest.raises(AgentUnavailable, match="NEBIUS_API_KEY"):
        a.run(WORDS)


def test_understand_falls_back_to_rules():
    res = understand(WORDS, agent=agent(FakeClient(ConnectionError("offline"))))
    assert res.source == "fallback" and "offline" in res.errors[0]
    assert [c.to_dict() for c in res.calls] == [{"name": "place_drawing", "args": GOOD}]
    res = understand(WORDS, agent=agent(FakeClient(reply(("place_drawing", GOOD)))))
    assert res.source == "nemotron"
    assert understand(["stop"]).calls[0].name == "stop"  # no agent at all
