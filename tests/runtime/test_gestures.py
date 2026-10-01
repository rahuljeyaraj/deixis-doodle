"""Holds with a second hand in view (live run on the Ender, 1 Oct 2026)."""

import numpy as np

from deixis.geometry import Box
from pointdraw.runtime.gestures import GestureTracker
from pointdraw.vision.hands import HandObservation, HandsFrame, PoseResult

PAPER = Box(33.0, 38.0, 187.0, 177.0)  # bed mm


def hand(x: float, y: float, pointing: bool) -> HandObservation:
    """A hand drawn as one spot (bed mm): enough for the tracker, with the thumb tucked."""
    mm = np.tile([x, y], (21, 1)).astype(float)
    return HandObservation("Right", 0.9, mm * 4, mm, None, PoseResult(pointing))


def holds(*hands_per_frame) -> list:
    g = GestureTracker(PAPER)
    for k in range(60):  # 2 s at 30 fps, then hands away
        g.update(HandsFrame(k / 30, list(hands_per_frame), (1920, 1080)))
    g.update(HandsFrame(2.0, [], (1920, 1080)))
    return g.recent_holds()


def test_one_pointing_hand_holds():
    assert len(holds(hand(100, 100, True))) == 1


def test_a_resting_hand_off_the_paper_does_not_block_the_point():
    hs = holds(hand(100, 100, True), hand(90, -60, False))  # 6 cm in front of the paper
    assert len(hs) == 1
    assert np.allclose(hs[0].mean, (100 - PAPER.x, 100 - PAPER.y), atol=0.5)


def test_two_hands_over_the_paper_make_no_place():
    assert holds(hand(80, 100, True), hand(150, 100, True)) == []
