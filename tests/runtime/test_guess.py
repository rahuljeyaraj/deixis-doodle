"""The plotter says its guess when the preview starts, and draws only on "yes" (6 Oct 2026).

Claim 16: "A kite, about four centimetres?" / "A tree, next to the house?". Claim 7: every
robot acts only on "yes", so `auto_confirm` is off by default (the virtual world turns it on).
"""

import pytest

from deixis.types import Word
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.schema import parse_call
from pointdraw.controller.config import MachineConfig
from pointdraw.runtime.guess import guess_line, item_guess, size_words
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import PREVIEW, Runtime, RuntimeParams
from pointdraw.virtual.plotter import VirtualPlotter
from pointdraw.vision.hands import HandsFrame

CFG = MachineConfig()
CROSS = parse_call("place_drawing", {"object": "circle", "place": {"empty_space": True}})
YES = parse_call("confirm", {})


@pytest.mark.parametrize(
    ("mm", "words"),
    [
        (40, "about four centimetres"),
        (4, "about one centimetre"),
        (152, "about fifteen centimetres"),
    ],
)
def test_size_words(mm, words):
    assert size_words(mm) == words


def test_item_guesses():
    assert guess_line([item_guess("object", "kite", None, 41)]) == "A kite, about four centimetres?"
    assert (
        guess_line([item_guess("object", "tree", None, 60, "next_to", "house")])
        == "A tree, next to the house?"
    )
    assert item_guess("object", "eye", None, 10) == "An eye, about one centimetre"
    assert item_guess("label", None, "Nemotron", 30) == "The words 'Nemotron'"
    assert guess_line([]) == ""


class Script:
    """Answers each utterance with the next calls in the list."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def understand(self, words, context):
        return AgentResult(calls=self.answers.pop(0), source="cache")


def run_until(rt: Runtime, plotter: VirtualPlotter, t0: float, t1: float) -> None:
    t = t0
    while t < t1:
        rt.on_frame(HandsFrame(t, [], (1920, 1080)))  # no hand over the paper
        plotter.tick(t)
        rt.tick(t)
        t += 0.05


def say(rt: Runtime, text: str, t: float) -> None:
    rt.on_utterance(
        [Word(w, t + 0.3 * i, t + 0.3 * i + 0.2) for i, w in enumerate(text.split())], t + 1
    )


def test_the_guess_is_said_and_nothing_is_drawn_without_yes():
    plotter = VirtualPlotter(CFG)
    rt = Runtime(Script([CROSS], [YES]), plotter, CFG, Planner())
    rt.start(0.0)
    say(rt, "draw a circle", 0.0)
    run_until(rt, plotter, 1.0, 40.0)  # long after the trace: still waiting
    assert rt.proposal is not None and rt.proposal.preview_done is not None
    assert rt.state == PREVIEW and not rt.log.of("confirm")
    said = [e["text"] for e in rt.log.of("say")]
    assert "A circle, about four centimetres?" in said
    say(rt, "yes", 40.0)
    run_until(rt, plotter, 41.0, 42.0)
    assert [e["reason"] for e in rt.log.of("confirm")] == ["confirm"]


def test_auto_confirm_still_works_when_turned_on():
    plotter = VirtualPlotter(CFG)
    rt = Runtime(Script([CROSS]), plotter, CFG, Planner(), params=RuntimeParams(auto_confirm=True))
    rt.start(0.0)
    say(rt, "draw a circle", 0.0)
    run_until(rt, plotter, 1.0, 40.0)
    assert [e["reason"] for e in rt.log.of("confirm")] == ["quiet"]
