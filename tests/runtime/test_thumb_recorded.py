"""Thumb out or tucked, on landmarks recorded from rahul's hand on the Ender (1 Oct 2026).

`data/hands_1oct.json`: six frames of a held point ("here") and six of a held size gesture
("this big": thumb and index apart, both along the paper). Every frame of both was
classified as pointing; only the thumb tells them apart.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pointdraw.runtime.gestures import thumb_out
from pointdraw.vision.hands import HandObservation, PoseResult, classify_pointing

DATA = json.loads((Path(__file__).parent / "data" / "hands_1oct.json").read_text())


def obs(h: dict) -> HandObservation:
    w, mm = np.array(h["world"]), np.array(h["mm"])
    return HandObservation("Right", 0.9, mm * 4, mm, w, classify_pointing(w))


@pytest.mark.parametrize("kind,out", [("point", False), ("size", True)])
def test_thumb_out_on_recorded_hands(kind, out):
    hands = [obs(h) for h in DATA[kind]]
    assert all(h.pointing for h in hands)  # the index alone can't tell them apart
    assert [thumb_out(h) for h in hands] == [out] * len(hands)


def test_without_world_landmarks_bed_mm_also_works():
    for kind, out in (("point", False), ("size", True)):
        for h in DATA[kind]:
            mm = np.array(h["mm"])
            o = HandObservation("Right", 0.9, mm * 4, mm, None, PoseResult(True))
            assert thumb_out(o) == out
