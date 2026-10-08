"""Sending the head back to the corner (rahul, 7 Oct 2026).

Two ways the head used to be left standing over the paper, in the camera's view of it and
with no way to shift it:

- a stop halts mid-drawing and nothing moves again, by design, so "park" and the window's
  P key have to work while the plotter sits stopped;
- a "no" to a guess left the pen where it had traced the preview.

"Park" is spotted here, before the agent, like a stop, so it works with no network and
whatever the model would have made of the word.
"""

import pytest

from deixis.types import Word
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.schema import parse_call
from pointdraw.controller.config import MachineConfig
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import Runtime
from pointdraw.virtual.plotter import VirtualPlotter

CROSS = parse_call("place_drawing", {"object": "cross", "place": {"empty_space": True}})


class Draws:
    """Answers every utterance with place_drawing(cross)."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def understand(self, words, context):
        self.asked.append(" ".join(words))
        return AgentResult(calls=[CROSS], source="cache")


def make() -> tuple[Runtime, VirtualPlotter, Draws]:
    agent = Draws()
    plotter = VirtualPlotter()
    rt = Runtime(agent, plotter, MachineConfig(), Planner())
    rt.start(0.0)
    return rt, plotter, agent


def say(rt: Runtime, text: str, t: float = 3.0) -> None:
    words = [Word(w, t - 2.0 + 0.3 * i, t - 1.8 + 0.3 * i) for i, w in enumerate(text.split())]
    rt.on_utterance(words, t)


def parks(plotter: VirtualPlotter) -> int:
    """Park jobs queued or run so far."""
    queued = [j for j in plotter.jobs if j.label == "park"]
    done = [e for e in plotter.events if e.kind == "job_start" and e.label == "park"]
    return len(queued) + len(done)


@pytest.mark.parametrize("text", ["Park", "park it", "Go home", "Okay park please"])
def test_park_is_spotted_before_the_agent(text):
    rt, plotter, agent = make()
    before = parks(plotter)
    say(rt, text)
    assert agent.asked == []  # never asked the model
    assert parks(plotter) == before + 1
    calls = rt.log.of("tool_calls")
    assert calls and calls[-1]["calls"] == [{"name": "park", "args": {}}]
    assert rt.log.of("park")[-1]["why"] == "park word"


@pytest.mark.parametrize(
    "text", ["Draw a cross in the park", "A cross over home plate", "park a cross here"]
)
def test_a_longer_sentence_holding_the_word_still_goes_to_the_agent(text):
    rt, _, agent = make()
    say(rt, text)
    rt.tick(5.0)
    assert agent.asked == [text] and not rt.log.of("park")
    assert rt.proposal is not None


def test_park_works_while_the_plotter_sits_stopped():
    rt, plotter, _ = make()
    say(rt, "A cross here")
    rt.tick(5.0)
    rt.confirm(6.0)
    rt.tick(7.0)
    rt.do_stop(8.0, "test")
    assert plotter.status().state == "stopped"
    before = parks(plotter)
    say(rt, "Park", t=9.0)
    assert parks(plotter) == before + 1


def test_a_no_to_a_guess_parks_the_pen():
    rt, plotter, _ = make()
    say(rt, "A cross here")
    rt.tick(5.0)
    assert rt.proposal is not None
    before = parks(plotter)
    rt.cancel(6.0)
    assert rt.proposal is None
    assert parks(plotter) == before + 1
    assert rt.log.of("park")[-1]["why"] == "cancelled"
    # "Cancelled." is the only thing said: the park is not worth a sentence of its own.
    assert [s["text"] for s in rt.log.of("say")][-1] == "Cancelled."


def test_a_stop_does_not_move_the_head_by_itself():
    """A stop is a stop: the park queued by the confirm is dropped and no new one is sent."""
    rt, plotter, _ = make()
    say(rt, "A cross here")
    rt.tick(5.0)
    rt.confirm(6.0)
    rt.tick(7.0)
    rt.do_stop(8.0, "test")
    started = len([e for e in plotter.events if e.kind == "job_start" and e.label == "park"])
    for t in range(9, 20):
        rt.tick(float(t))
    assert not [j for j in plotter.jobs if j.label == "park"]
    after = len([e for e in plotter.events if e.kind == "job_start" and e.label == "park"])
    assert after == started


def test_park_says_so_when_the_plotter_is_not_answering():
    rt, plotter, _ = make()
    rt.plotter_problem = "link failed"
    before = parks(plotter)
    rt.do_park(4.0, "p key")
    assert parks(plotter) == before
    assert "cable" in [s["text"] for s in rt.log.of("say")][-1]
