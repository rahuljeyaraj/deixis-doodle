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


class Says:
    """Answers with one given call (as recorded from live Nemotron)."""

    def __init__(self, call) -> None:
        self.call = call

    def understand(self, words, context):
        return AgentResult(calls=[self.call], source="cache")


def grounded(text: str, args: dict):
    """The guard's version of the model's call for `text` (and the runtime, for its log)."""
    from pointdraw.runtime.references import Utterance

    rt = Runtime(Says(parse_call("place_drawing", args)), VirtualPlotter(), MachineConfig())
    rt.start(0.0)
    words = [Word(w, 1.0 + 0.3 * i, 1.2 + 0.3 * i) for i, w in enumerate(text.split())]
    return rt, rt._grounded(rt.agent.call, Utterance(0, words, 3.0), 3.0)


SUN_ABOVE = {
    "object": "sun",
    "place": {"relation": "above", "target": {"label": "house"}},
    "size": {"preset": "small"},
}


def test_a_place_and_size_not_said_become_the_pointed_place():
    """Live Nemotron, 2 Oct (garden scene): "A sun up here" came back as above the house,
    small. The pointing was lost; nobody said small."""
    rt, call = grounded("A sun up here", SUN_ABOVE)
    assert call.args.place.deixis == "here" and call.args.place.word_index == 3
    assert call.args.size is None
    assert rt.log.of("guard")[-1]["why"].startswith("a place or size not said")


@pytest.mark.parametrize(
    "text",
    [
        "A sun above the house",  # the relation was said
        "A sun up here over it",  # a relation word
        "A sun here, a small one",  # the size was said (place grounded, size kept)
    ],
)
def test_said_places_and_sizes_are_kept(text):
    _, call = grounded(text, SUN_ABOVE)
    if "small" in text:
        assert call.args.size is not None
    else:
        assert call.args.place.relation == "above"


def test_a_relation_to_the_pointed_object_is_kept():
    args = {
        "object": "hand",
        "place": {"relation": "inside", "target": {"deixis": "that", "word_index": 3}},
    }
    _, call = grounded("Put a hand that here", args)
    assert call.args.place.relation == "inside"


def test_an_arrow_whose_ends_are_the_same_place_asks():
    """Stress test, 2 Oct: S12 seed 7 lost the second point, both words took the first, and
    the arrow code raised ValueError (it would have stopped the live app)."""
    call = parse_call(
        "draw_connector",
        {
            "kind": "arrow",
            "from": {"deixis": "this", "word_index": 3},
            "to": {"deixis": "that", "word_index": 5},
        },
    )
    rt = Runtime(Says(call), VirtualPlotter(), MachineConfig())
    rt.start(0.0)
    words = "An arrow from this to that".split()
    rt.on_utterance([Word(w, 1.0 + 0.2 * i, 1.15 + 0.2 * i) for i, w in enumerate(words)], 3.0)
    rt.tick(5.0)
    assert not rt.log.of("error")
    asks = [e["question"] for e in rt.log.of("ask")]
    assert asks  # no crash; it asks (no point at all, or the same point twice)


def test_a_call_that_fails_to_plan_is_reported_not_raised(monkeypatch):
    rt = Runtime(Says(CROSS), VirtualPlotter(), MachineConfig(), Planner())
    rt.start(0.0)

    def boom(*a, **k):
        raise ValueError("planner broke")

    monkeypatch.setattr(rt, "_place_drawing", boom)
    rt.on_utterance([Word("cross", 1.0, 1.3)], 3.0)
    rt.tick(5.0)
    assert rt.log.of("error")[-1]["why"] == "ValueError: planner broke"
    assert rt.log.of("say")[-1]["text"] == "Sorry, I couldn't draw that."


def test_a_lone_one_is_not_a_request_for_another():
    """Live 8 Oct 2026: a mis-heard "One", just after an elephant was drawn, came back as
    place_drawing of another elephant at a corner nobody named, and it previewed there.

    "one" counts as "another one" only beside one of the other repeat words.
    """
    from pointdraw.runtime.references import Utterance
    from pointdraw.runtime.runtime import WEAK_REPEAT_WORDS

    assert WEAK_REPEAT_WORDS == {"one", "ones"}
    call = parse_call("place_drawing", {"object": "elephant", "place": {"empty_space": True}})

    def utt(text):
        words = [Word(w, 0.1 * i, 0.1 * i + 0.05) for i, w in enumerate(text.split())]
        return Utterance(0, words, 1.0)

    rt = Runtime(Repeats(), VirtualPlotter(), MachineConfig(), Planner())
    assert rt._invented(call, utt("One")), "a lone 'One' was taken as 'another one'"
    assert rt._invented(call, utt("Ones"))
    assert not rt._invented(call, utt("another one"))  # a real repeat still works
    assert not rt._invented(call, utt("one more"))
    assert not rt._invented(call, utt("the same one"))
    assert not rt._invented(call, utt("draw an elephant"))  # named outright
