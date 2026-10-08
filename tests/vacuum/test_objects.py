"""A place given by name: Cosmos finds the thing, our geometry places it on the floor.

No endpoint and no camera: the Cosmos client is a stand-in that replies with the JSON the
real one replied with on 8 Oct, and the floor fit is a plain 4 mm-a-pixel view.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointdraw.cosmos.client import CosmosReply
from pointdraw.vacuum.floor import FloorView
from pointdraw.vacuum.objects import Eyes, footprint_mm, room_find_prompt

VIEW = FloorView(np.diag([4.0, 4.0, 1.0]), (10, 11, 12, 13), True)  # 4 mm a pixel


class FakeCosmos:
    """Replies with `raw`, or raises `error`. Records what it was asked."""

    def __init__(self, raw: str = "[]", error: Exception | None = None) -> None:
        self.raw, self.error, self.prompts = raw, error, []
        self.profile = type("P", (), {"coord_mode": "norm1000", "xy_order": "xy"})()

    def ask(self, image, prompt, **kw):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return CosmosReply(raw=self.raw, latency_s=0.5, model="fake"), 1920, 1080


def reply_with(box_norm):
    x1, y1, x2, y2 = box_norm
    return f'[{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}], "label": "chair"}}]'


@pytest.mark.parametrize(
    "said, want",
    [
        ("around the chair", "the chair"),
        ("the chair", "the chair"),
        ("chair", "the chair"),
        ("next to the bin", "the bin"),
        ("under my desk", "my desk"),
    ],
)
def test_the_prompt_asks_for_the_thing_not_the_preposition(said: str, want: str) -> None:
    assert f"of {want} as a whole" in room_find_prompt(said)


def test_the_footprint_is_as_wide_as_the_bottom_edge():
    """200 px across at 4 mm a pixel = 800 mm; this view has no perspective, so the square is
    pushed its half-side straight up the floor (-y), onto the thing rather than in front."""
    box = footprint_mm(VIEW, (400.0, 200.0, 600.0, 800.0), margin_mm=0.0)
    assert box == pytest.approx((1600.0, 2400.0, 800.0, 800.0))


def test_how_tall_the_thing_looks_barely_moves_it():
    """A chair and a bookcase standing on the same spot must be cleaned in the same place:
    the height of the picture box says nothing about where it stands."""
    short = footprint_mm(VIEW, (400.0, 700.0, 600.0, 800.0), margin_mm=0.0)
    tall = footprint_mm(VIEW, (400.0, 100.0, 600.0, 800.0), margin_mm=0.0)
    assert short == pytest.approx(tall, abs=100.0)  # under 10 cm apart


def test_the_margin_grows_it_on_every_side():
    box = footprint_mm(VIEW, (400.0, 200.0, 600.0, 800.0), margin_mm=350.0)
    assert box == pytest.approx((1250.0, 2050.0, 1500.0, 1500.0))


def test_a_found_thing_comes_back_as_a_floor_box():
    eyes = Eyes(FakeCosmos(reply_with((562, 606, 710, 974))), margin_mm=350.0)
    where = eyes.look("around the chair", np.zeros((1080, 1920, 3), np.uint8), VIEW)
    assert where is not None
    assert where.frame == "floor" and where.kind == "box" and where.source == "vlm"
    assert where.label == "chair"
    assert where.evidence["asked"] == "around the chair"
    # 562..710 of 1000 across a 1920-wide frame = 1079..1363 px = 1136 mm, plus 2 x 350
    assert where.box_mm[2] == pytest.approx(1836, abs=8)
    assert where.box_mm[2] == where.box_mm[3]


def test_nothing_seen_is_not_a_guess():
    assert Eyes(FakeCosmos("[]")).look("the chair", np.zeros((4, 4, 3), np.uint8), VIEW) is None


def test_the_endpoint_falling_over_is_not_a_guess():
    eyes = Eyes(FakeCosmos(error=RuntimeError("endpoint is down")))
    assert eyes.look("the chair", np.zeros((4, 4, 3), np.uint8), VIEW) is None


def test_the_biggest_of_several_is_taken():
    raw = ('[{"bbox_2d": [100, 100, 150, 150], "label": "chair"},'
           ' {"bbox_2d": [500, 500, 700, 900], "label": "chair"}]')  # fmt: skip
    eyes = Eyes(FakeCosmos(raw), margin_mm=0.0)
    where = eyes.look("the chairs", np.zeros((1080, 1920, 3), np.uint8), VIEW)
    assert where is not None and where.evidence["box_px"][0] == 960
