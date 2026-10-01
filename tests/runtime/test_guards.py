"""Heard on the Ender (1 Oct 2026): fillers and chatter must not draw anything.

A lone "hello", "hmm" or "okay" came back from the model as place_drawing of the last
object, which was then drawn in the middle of the page.
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


class Repeats:
    """Answers every utterance with place_drawing(cross), as the model did."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def understand(self, words, context):
        self.asked.append(" ".join(words))
        return AgentResult(calls=[CROSS], source="cache")


def run(text: str) -> tuple[Runtime, Repeats]:
    agent = Repeats()
    rt = Runtime(agent, VirtualPlotter(), MachineConfig(), Planner())
    rt.start(0.0)
    words = [Word(w, 1.0 + 0.3 * i, 1.2 + 0.3 * i) for i, w in enumerate(text.split())]
    rt.on_utterance(words, 3.0)
    rt.tick(5.0)
    return rt, agent


@pytest.mark.parametrize("text", ["Hello", "Hmm", "Okay", "Yeah so", "Sure", "I mean", "Um"])
def test_fillers_alone_are_ignored(text):
    rt, agent = run(text)
    assert agent.asked == [] and rt.log.of("ignored") and rt.proposal is None


@pytest.mark.parametrize(
    ("text", "heard"),
    [
        ("Maybe I m not sure", "Maybe I m not sure"),
        ("You know what", "You know what"),
        ("Best bit", "this big"),  # repaired since 2 Oct (rahul's "this big", live 1 Oct)
    ],
)
def test_an_object_not_said_is_not_drawn(text, heard):
    rt, agent = run(text)
    assert agent.asked == [heard]
    assert rt.proposal is None and not rt.scene.objects
    assert any(
        g["why"].startswith("place_drawing of an object not said") for g in rt.log.of("guard")
    )


@pytest.mark.parametrize("text", ["A cross", "Draw a cross", "draw two crosses", "another one"])
def test_an_object_said_is_drawn(text):
    rt, _ = run(text)
    assert rt.proposal is not None and rt.proposal.items[0].label == "cross"


def test_an_answer_to_how_big_keeps_the_asked_object():
    from pointdraw.runtime.references import Question

    agent = Repeats()
    rt = Runtime(agent, VirtualPlotter(), MachineConfig(), Planner())
    rt.start(0.0)
    rt._ask(Question("How big? Show me with your fingers or hands.", "no_size"), None, CROSS, 0.5)
    rt.on_utterance([Word("This", 1.0, 1.2), Word("big", 1.3, 1.5)], 2.0)
    rt.tick(4.0)
    assert not rt.log.of("guard")
    assert rt.proposal is not None and rt.proposal.items[0].label == "cross"
