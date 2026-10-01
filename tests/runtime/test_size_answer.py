"""'How big?' ... 'this big' is answered by the runtime itself (live run, 1 Oct 2026).

Sent to the agent alone, "this big" came back as a new object (a moon, a cat, a circle):
the agent had no context for the question. Now the size comes from the answer's gesture
and the rest from the request.
"""

import numpy as np

from deixis.types import Word
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.schema import parse_call
from pointdraw.controller.config import MachineConfig
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import Runtime
from pointdraw.virtual.plotter import VirtualPlotter
from pointdraw.vision.hands import HandObservation, HandsFrame, PoseResult

CFG = MachineConfig()
PAPER = CFG.paper_box
REQUEST = parse_call(
    "place_drawing",
    {"object": "cross", "place": {"deixis": "here", "word_index": 3}, "size": {"word_index": 4}},
)


class OnlyTheRequest:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def understand(self, words, context):
        self.asked.append(" ".join(words))
        return AgentResult(calls=[REQUEST], source="cache")


def pointing(x, y) -> HandObservation:
    mm = np.tile([PAPER.x + x, PAPER.y + y], (21, 1)).astype(float)
    return HandObservation("Right", 0.9, mm * 4, mm, None, PoseResult(True))


def pinch(span: float) -> HandObservation:
    """Thumb and index `span` mm apart, over the middle of the paper."""
    cx, cy = PAPER.x + 90, PAPER.y + 60
    mm = np.tile([cx, cy], (21, 1)).astype(float)
    mm[9] = (cx, cy + 10)  # middle MCP: the hand's scale is 10 mm from the wrist
    mm[5] = (cx + 2, cy + 10)  # index MCP
    mm[4] = (cx + 30, cy + 10)  # thumb tip, out to the side
    mm[8] = mm[4] + (0, span)  # index tip, `span` from the thumb tip
    return HandObservation("Right", 0.9, mm * 4, mm, None, PoseResult(False))


def frames(rt: Runtime, t0: float, t1: float, hand) -> None:
    k = 0
    while t0 + k / 30 < t1:
        t = t0 + k / 30
        rt.on_frame(HandsFrame(t, [hand] if hand else [], (1920, 1080)))
        rt.tick(t)
        k += 1


def test_this_big_answers_how_big():
    agent = OnlyTheRequest()
    rt = Runtime(agent, VirtualPlotter(CFG), CFG, Planner())
    rt.start(0.0)
    frames(rt, 0.0, 2.0, pointing(60, 50))  # point while saying "draw a cross here"
    said = ["Draw", "a", "cross", "here", "this", "big"]
    rt.on_utterance([Word(w, 0.4 + 0.25 * i, 0.6 + 0.25 * i) for i, w in enumerate(said)], 2.0)
    frames(rt, 2.0, 3.0, None)
    assert rt.question is not None and rt.question.q.reason == "no_size"

    frames(rt, 3.0, 4.0, pinch(50.0))  # then show the size while saying "this big"
    rt.on_utterance([Word("This", 3.4, 3.6), Word("big", 3.7, 3.9)], 4.1)
    frames(rt, 4.1, 5.0, None)
    assert agent.asked == [" ".join(said)]  # the answer never went to the agent
    assert rt.log.of("answer") and rt.log.of("answer")[0]["answer"] == "size"
    item = rt.proposal.items[0]
    assert item.label == "cross"
    assert abs(max(item.box.w, item.box.h) - 50.0) < 5.0
    assert abs(item.center[0] - 60) < 3 and abs(item.center[1] - 50) < 3


def test_fingers_opened_a_few_seconds_after_this_big():
    # Live run, 1 Oct 2026: pointing while saying "this big", the size shown 3-4 s later.
    agent = OnlyTheRequest()
    rt = Runtime(agent, VirtualPlotter(CFG), CFG, Planner())
    rt.start(0.0)
    frames(rt, 0.0, 2.0, pointing(60, 50))
    said = ["Draw", "a", "cross", "here", "this", "big"]
    rt.on_utterance([Word(w, 0.4 + 0.25 * i, 0.6 + 0.25 * i) for i, w in enumerate(said)], 2.0)
    frames(rt, 2.0, 3.0, None)
    frames(rt, 3.0, 4.0, pointing(60, 50))
    rt.on_utterance([Word("This", 3.4, 3.6), Word("big", 3.7, 3.9)], 4.1)
    frames(rt, 4.1, 7.0, pointing(60, 50))
    assert rt.proposal is None and rt.question is not None
    frames(rt, 7.0, 8.0, pinch(30.0))
    frames(rt, 8.0, 9.0, None)
    assert len(agent.asked) == 1
    item = rt.proposal.items[0]
    assert abs(max(item.box.w, item.box.h) - 30.0) < 4.0


def test_this_big_without_a_gesture_asks_again():
    agent = OnlyTheRequest()
    rt = Runtime(agent, VirtualPlotter(CFG), CFG, Planner())
    rt.start(0.0)
    frames(rt, 0.0, 2.0, pointing(60, 50))
    said = ["Draw", "a", "cross", "here", "this", "big"]
    rt.on_utterance([Word(w, 0.4 + 0.25 * i, 0.6 + 0.25 * i) for i, w in enumerate(said)], 2.0)
    frames(rt, 2.0, 3.0, None)
    rt.on_utterance([Word("This", 3.4, 3.6), Word("big", 3.7, 3.9)], 4.1)
    frames(rt, 4.1, 11.0, None)  # no fingers within size_wait_s: ask again
    assert len(agent.asked) == 1
    assert rt.question is not None and rt.question.q.reason == "no_size"
    asks = [a["question"] for a in rt.log.of("ask")]
    assert len(asks) == 2 and asks[-1].startswith("How big?")
