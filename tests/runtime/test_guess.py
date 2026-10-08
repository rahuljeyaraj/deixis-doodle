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
    assert "A circle, about eight centimetres?" in said
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


def test_a_tree_next_to_the_house_cosmos_called_a_building():
    """Command 2 of the film: the guess repeats the person's word, and the core finds the
    sketch through Cosmos's other labels (claims 16, 26c)."""
    from deixis.geometry import Box
    from pointdraw.scene.ink import UserMark

    tree = parse_call(
        "place_drawing",
        {"object": "tree", "place": {"relation": "next_to", "target": {"label": "house"}}},
    )
    plotter = VirtualPlotter(CFG)
    rt = Runtime(Script([tree]), plotter, CFG, Planner())
    rt.start(0.0)
    m = UserMark(1, Box(40, 60, 50, 45), 400.0, (65, 82), (0, 0, 1, 1))
    rt.scene.set_user_marks([m], lambda _: ("building", ["house", "home"]))
    say(rt, "draw a tree next to the house", 0.0)
    run_until(rt, plotter, 1.0, 3.0)
    assert rt.proposal is not None and rt.proposal.items[0].label == "tree"
    assert rt.proposal.items[0].box.x >= 90  # beside the sketch, not on it
    assert "A tree, next to the house?" in [e["text"] for e in rt.log.of("say")]


class CountingCamera:
    """A PaperCamera stand-in: a blank sheet, counting every look."""

    def __init__(self) -> None:
        self.looks = 0

    def paper_image(self, t):
        import numpy as np

        self.looks += 1
        return np.full((200, 220, 3), 255, np.uint8), 1.0


def test_yes_does_not_make_it_look_at_the_paper_again():
    """Live on 8 Oct 2026 rahul had to say "yes" twice.

    The first "yes" was heard, but answering it re-read the paper, and naming the new marks
    through Cosmos took 8 s, so the yes waited behind it. While a guess is on the table the
    scene has already been read to make it, and the answer is about the guess.
    """
    plotter = VirtualPlotter(CFG)
    camera = CountingCamera()
    rt = Runtime(Script([CROSS], [YES]), plotter, CFG, Planner(), camera=camera)
    rt.start(0.0)
    say(rt, "draw a circle", 0.0)
    run_until(rt, plotter, 1.0, 40.0)
    assert rt.state == PREVIEW
    looks = camera.looks
    assert looks  # the request itself did look
    say(rt, "yes", 40.0)
    run_until(rt, plotter, 41.0, 42.0)
    assert [e["reason"] for e in rt.log.of("confirm")] == ["confirm"]
    assert camera.looks == looks  # answering the guess did not look again


class SlowVision:
    """A Cosmos stand-in that takes `delay_s` to name one mark, as the real one did."""

    def __init__(self, delay_s: float = 0.4, name: str = "house") -> None:
        self.delay_s, self.name, self.calls = delay_s, name, 0

    def label(self, img, mark, mmpp):
        import time

        self.calls += 1
        time.sleep(self.delay_s)
        return self.name


class InkCamera:
    """One pen outline on a white sheet, so segmenting finds exactly one mark.

    An outline, not a filled block: the local paper level is a median over 12 mm, so a big
    solid patch reads as the paper itself and no ink is found."""

    def paper_image(self, t):
        import cv2
        import numpy as np

        img = np.full((200, 220, 3), 255, np.uint8)
        cv2.rectangle(img, (60, 60), (120, 120), (20, 20, 20), 3)
        return img, 1.0


def test_the_at_rest_look_does_not_block_the_loop():
    """Live on 8 Oct 2026 the window froze for 4.7 s after each drawing while Cosmos named
    the new marks. Nothing waits on that look, so it runs in the background now.
    """
    import time

    plotter = VirtualPlotter(CFG)
    vision = SlowVision(delay_s=0.4)
    rt = Runtime(Script([CROSS], [YES]), plotter, CFG, Planner(), camera=InkCamera(), vision=vision)
    rt.start(0.0)
    try:
        t0 = time.monotonic()
        rt.refresh_scene_soon(1.0)
        assert time.monotonic() - t0 < 0.2, "the call itself blocked"
        assert not rt.log.of("scene")[1:], "the marks were applied before the look finished"
        # The scene is only changed on the main thread, in tick.
        deadline = time.monotonic() + 5.0
        while not rt.log.of("scene")[1:] and time.monotonic() < deadline:
            rt.tick(2.0)
            time.sleep(0.02)
        names = [o.label for o in rt.scene.marks]
        assert names == ["house"], names
        assert vision.calls == 1
    finally:
        rt.close()


def test_a_background_look_never_undoes_a_fresh_one():
    """If the paper is read again while a background look is still running, the late answer
    is dropped: it was taken before the newer one and would put the scene back."""
    plotter = VirtualPlotter(CFG)
    rt = Runtime(
        Script([CROSS]), plotter, CFG, Planner(), camera=InkCamera(), vision=SlowVision(0.3)
    )
    rt.start(0.0)
    try:
        rt.refresh_scene_soon(1.0)
        rt.refresh_scene(1.5)  # a blocking look, while the background one is still going
        job = rt._scene_job
        assert job is not None
        job.result()  # let it finish
        rt.tick(2.0)
        assert [e["why"] for e in rt.log.of("scene_dropped")] == [
            "the paper was read again while this one ran"
        ]
    finally:
        rt.close()


class DeadVision:
    """Cosmos after its endpoint was preempted: every call raises."""

    def __init__(self) -> None:
        self.calls = 0

    def label(self, img, mark, mmpp):
        self.calls += 1
        raise RuntimeError("Not Found")


def test_a_dead_cosmos_endpoint_does_not_stop_the_app():
    """Live on 8 Oct 2026 a preempted endpoint's 404 came out of the blocking look and
    killed the app at start-up. The marks are kept, just unnamed."""
    plotter = VirtualPlotter(CFG)
    vision = DeadVision()
    rt = Runtime(Script([CROSS]), plotter, CFG, Planner(), camera=InkCamera(), vision=vision)
    rt.start(0.0)  # start() looks at the paper: this used to raise
    try:
        rt.refresh_scene(1.0)
        assert vision.calls >= 1
        assert [o.label for o in rt.scene.marks] == [None]  # kept, with no name
        assert [e["why"] for e in rt.log.of("naming_failed")][-1] == "RuntimeError: Not Found"
    finally:
        rt.close()


def test_one_failure_gives_up_on_the_rest_of_that_look():
    """Asking for every mark when the endpoint is down costs a timeout each."""
    import cv2
    import numpy as np

    class TwoMarks:
        def paper_image(self, t):
            img = np.full((200, 220, 3), 255, np.uint8)
            cv2.rectangle(img, (20, 20), (80, 80), (20, 20, 20), 3)
            cv2.rectangle(img, (120, 120), (180, 180), (20, 20, 20), 3)
            return img, 1.0

    vision = DeadVision()
    rt = Runtime(
        Script([CROSS]), VirtualPlotter(CFG), CFG, Planner(), camera=TwoMarks(), vision=vision
    )
    try:
        rt.refresh_scene(1.0)
        assert len(rt.scene.marks) == 2
        assert vision.calls == 1, "it kept asking after the first failure"
    finally:
        rt.close()
