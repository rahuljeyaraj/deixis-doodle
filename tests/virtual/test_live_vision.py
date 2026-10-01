"""LiveVision (the real Cosmos path) with a fake endpoint: requests, replies and whole scenes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tests.cosmos.fakes import FakeOpenAI

from pointdraw.config import Settings
from pointdraw.cosmos.client import CosmosClient
from pointdraw.virtual import scenefile, vision, world
from pointdraw.virtual.world import RunConfig


def _spec(scene_id: str) -> scenefile.SceneSpec:
    return next(s for s in scenefile.all_scenes() if s.id == scene_id)


def _reply(kwargs) -> str:
    text = FakeOpenAI.prompt_text(kwargs)
    if "hand-drawn mark" in text:  # label_crop
        return '{"label": "square", "alternatives": ["box", "frame"]}'
    return "<think>A circle on the left, a star on the right.</think>You drew a circle and a star."


@pytest.fixture
def fake_live(monkeypatch):
    """world.run(..., cosmos='live') with the fake endpoint behind LiveVision."""
    fakes: list[FakeOpenAI] = []

    class FakeLiveVision(vision.LiveVision):
        def __init__(self) -> None:
            fake = FakeOpenAI(_reply)
            fakes.append(fake)
            client = CosmosClient(Settings(_env_file=None), openai_client=fake, think=False)
            super().__init__(client)

    monkeypatch.setattr(vision, "LiveVision", FakeLiveVision)
    return fakes


@pytest.mark.virtual
def test_s16_look_live_path(fake_live):
    res = world.run(_spec("S16"), RunConfig(cosmos="live"))
    fails = [f"{c['name']}: {c['detail']}" for c in res.report["checks"] if not c["ok"]]
    assert not fails, fails
    looks = res.log.of("look")
    assert looks and looks[0]["answer"] == "You drew a circle and a star."  # no <think> spoken
    calls = fake_live[0].calls
    prompt = FakeOpenAI.prompt_text(calls[-1])
    assert '"What did I draw?"' in prompt and "robot drew" not in prompt
    assert calls[-1]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/")
    assert res.report["metrics"]["cosmos_calls"] >= 1
    assert res.report["cosmos"]["replies"][-1]["task"] == "look"


@pytest.mark.virtual
def test_s15_live_label_differs_but_the_hand_goes_inside(fake_live):
    """Cosmos calls the user's box a 'square'; the hand must still land inside it."""
    res = world.run(_spec("S15"), RunConfig(cosmos="live"))
    fails = [f"{c['name']}: {c['detail']}" for c in res.report["checks"] if not c["ok"]]
    assert not fails, fails
    labels = [x["label"] for x in res.report["cosmos"]["labels"]]
    assert "square" in labels
    # a crop is sent, not the whole sheet (plan 4.5)
    first = fake_live[0].calls[0]
    assert "hand-drawn mark" in FakeOpenAI.prompt_text(first)


def test_look_prompt_names_the_robots_drawings():
    from pointdraw.cosmos.prompts import look_prompt

    p = look_prompt("What's on the page?", ["house", "tree"])
    assert "The robot drew the house and the tree; the person drew everything else." in p
    assert look_prompt("What did I draw?").count("robot") == 0
    assert '"What did I draw?"' in look_prompt("What did I draw")  # the agent drops the "?"


def test_look_falls_back_when_the_reply_is_only_thinking():
    fake = FakeOpenAI("<think>hmm</think>")
    lv = vision.LiveVision(CosmosClient(Settings(_env_file=None), openai_client=fake))
    scene = SimpleNamespace(robot=[])
    out = lv.look("What did I draw?", np.full((80, 80, 3), 255, np.uint8), 0.5, scene)
    assert out == "Sorry, I couldn't make that out."
