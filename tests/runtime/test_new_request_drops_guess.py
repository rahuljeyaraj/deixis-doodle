"""A new request during a preview drops that guess; it is never drawn (rahul, 6 Oct 2026).

Until 6 Oct a different request *confirmed* the guess that was waiting, so one misheard word
turned into ink: in rahul's run that day "Beaker" drew the car he was still being asked
about, and the next request drew the label. "Yes" and "cancel" are the only ways a guess
becomes ink (claim 16). With `auto_confirm` on (the virtual world) nothing waits on a person
and the guess would have been drawn after the quiet anyway, so there it still confirms.
"""

import math
import re

import pytest

from deixis.types import Word
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.schema import parse_call
from pointdraw.controller import gcode
from pointdraw.controller.config import MachineConfig
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import NOT_UNDERSTOOD, Runtime, RuntimeParams
from pointdraw.virtual.plotter import VirtualPlotter
from pointdraw.vision.hands import HandsFrame

CFG = MachineConfig()
CAT = parse_call("place_drawing", {"object": "cat", "place": {"empty_space": True}})
LABEL = parse_call("write_label", {"text": "Beaker", "place": {"empty_space": True}})


class Script:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def understand(self, words, context):
        return AgentResult(calls=self.answers.pop(0), source="cache")


def run(rt: Runtime, plotter: VirtualPlotter, t0: float, t1: float, hand=None) -> None:
    t = t0
    while t < t1:
        rt.on_frame(HandsFrame(t, [hand] if hand else [], (1920, 1080)))
        plotter.tick(t)
        rt.tick(t)
        t += 0.05


def pointing(x: float, y: float):
    """A hand pointing at this spot on the paper."""
    import numpy as np

    from pointdraw.vision.hands import HandObservation, PoseResult

    mm = np.tile([CFG.paper_box.x + x, CFG.paper_box.y + y], (21, 1)).astype(float)
    return HandObservation("Right", 0.9, mm * 4, mm, None, PoseResult(True))


def say(rt: Runtime, text: str, t: float) -> None:
    rt.on_utterance(
        [Word(w, t + 0.3 * i, t + 0.3 * i + 0.2) for i, w in enumerate(text.split())], t + 1
    )


def _runtime(**params):
    plotter = VirtualPlotter(CFG)
    kw = {"params": RuntimeParams(**params)} if params else {}
    rt = Runtime(Script([CAT], [LABEL]), plotter, CFG, Planner(), **kw)
    rt.start(0.0)
    return rt, plotter


def test_a_new_request_drops_the_waiting_guess_instead_of_drawing_it():
    rt, plotter = _runtime()
    assert rt.params.auto_confirm is False
    say(rt, "draw a cat here", 0.0)
    run(rt, plotter, 1.0, 4.0)
    assert [o.label for o in rt.proposal.items] == ["cat"]

    say(rt, "Beaker", 5.0)  # misheard: a different request, not a yes
    run(rt, plotter, 6.0, 20.0)

    assert rt.log.of("confirm") == []  # nothing was ever said yes to
    assert [e["items"] for e in rt.log.of("dropped")] == [["cat"]]
    assert "cat" not in {e["id"] for e in rt.log.of("drawn")}


def test_with_auto_confirm_a_new_request_still_confirms_the_guess():
    rt, plotter = _runtime(auto_confirm=True)
    say(rt, "draw a cat here", 0.0)
    run(rt, plotter, 1.0, 4.0)
    say(rt, "Beaker", 5.0)
    run(rt, plotter, 6.0, 20.0)

    assert [e["reason"] for e in rt.log.of("confirm")] == ["new request"]
    assert rt.log.of("dropped") == []


def test_a_mangled_correction_is_not_turned_into_a_place_nobody_said():
    """rahul, 6 Oct: "a bit bigger" reached the app as "Well we went". The model answered
    place_drawing of the crown already showing, at the "top left corner" — neither the object
    nor the place was said — and the crown jumped there while the guess read the same size."""
    plotter = VirtualPlotter(CFG)
    moved = parse_call("place_drawing", {"object": "cat", "place": {"named": "top left corner"}})
    rt = Runtime(Script([CAT], [moved]), plotter, CFG, Planner())
    rt.start(0.0)
    say(rt, "draw a cat here", 0.0)
    run(rt, plotter, 1.0, 4.0)
    where = rt.proposal.items[0].center
    said = len(rt.log.of("say"))

    say(rt, "Well we went", 5.0)
    run(rt, plotter, 6.0, 12.0)

    assert rt.proposal.items[0].center == where  # it did not jump
    assert [e["text"] for e in rt.log.of("say")[said:]] == [NOT_UNDERSTOOD]


def test_a_correction_that_was_said_still_moves_the_preview():
    """The guard must not block the real thing: "no, there" names no object either.

    The model answers with a corner nobody said. "there" was said, so the correction lands on
    the pointed spot rather than on the corner (rahul, 8 Oct 2026).
    """
    plotter = VirtualPlotter(CFG)
    moved = parse_call("place_drawing", {"object": "cat", "place": {"named": "top left corner"}})
    rt = Runtime(Script([CAT], [moved]), plotter, CFG, Planner())
    rt.start(0.0)
    say(rt, "draw a cat here", 0.0)
    run(rt, plotter, 1.0, 4.0)
    where = rt.proposal.items[0].center

    run(rt, plotter, 4.0, 5.0, hand=pointing(70, 60))  # point while saying it
    say(rt, "no, over there", 5.0)
    run(rt, plotter, 6.0, 12.0)

    now = rt.proposal.items[0].center
    assert now != where  # the move was asked for, so it happened
    assert abs(now[0] - 70) < 3 and abs(now[1] - 60) < 3  # where the finger was, not a corner


def test_a_corner_nobody_said_is_not_used_as_a_place():
    """Live 8 Oct 2026: a mis-heard "One" came back as place_drawing of the elephant just
    drawn at the "bottom right corner", and it previewed there. Nothing was pointed at either,
    so the right answer is to ask."""
    plotter = VirtualPlotter(CFG)
    corner = parse_call(
        "place_drawing", {"object": "cat", "place": {"named": "bottom right corner"}}
    )
    rt = Runtime(Script([corner]), plotter, CFG, Planner())
    rt.start(0.0)
    say(rt, "draw a cat", 0.0)
    run(rt, plotter, 1.0, 4.0)
    assert rt.proposal is None, "it previewed at a place nobody named"
    assert any("not said" in e["why"] for e in rt.log.of("guard")), rt.log.of("guard")
    assert rt.question is not None and rt.question.q.reason in ("no_gesture", "unsteady")


def test_the_preview_traces_at_the_preview_height_not_over_the_clips():
    """Starting from park, outside the drawable area, the trace used to stay at clear_z: 15 mm
    over the paper, which read as drawn in the air."""
    cfg = CFG.model_copy(update={"preview_z": (CFG.pen_up_z + CFG.pen_down_z * 2) / 3})
    assert cfg.pen_down_z < cfg.preview_height < cfg.pen_up_z
    centre = cfg.drawable.center
    lines = gcode.preview_circle_gcode(centre, 40, cfg, start=(cfg.park_x, cfg.park_y))

    drop = f"G0 Z{gcode.fmt(cfg.preview_height)} F{gcode.fmt(cfg.feed.z)}"
    i = lines.index(drop)  # the whole line: "G0 Z1" would also match the "G0 Z15" above
    assert lines[0] == f"G0 Z{gcode.fmt(cfg.clear_z)} F{gcode.fmt(cfg.feed.z)}"  # over the clips
    trace = lines[i + 1 : -1]
    assert trace and all("Z" not in ln for ln in trace)  # the whole circle at that one height
    assert all(math.dist(_xy(ln), centre) == pytest.approx(20, abs=0.25) for ln in trace)
    assert lines[-1].startswith(f"G0 Z{gcode.fmt(cfg.pen_up_z)}")  # lifts again at the end


def _xy(line: str) -> tuple[float, float]:
    x = re.search(r"X(-?[\d.]+)", line)
    y = re.search(r"Y(-?[\d.]+)", line)
    return (float(x.group(1)) if x else math.nan, float(y.group(1)) if y else math.nan)


@pytest.mark.parametrize("bad", [0.0, 20.0])
def test_a_preview_height_at_or_below_the_pen_is_refused(bad):
    with pytest.raises(ValueError, match="preview_z"):
        CFG.model_copy(update={"preview_z": bad}).model_validate(
            CFG.model_copy(update={"preview_z": bad}).model_dump()
        )
