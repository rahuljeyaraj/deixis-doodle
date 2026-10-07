"""The box sent to the robot is smaller than the box the person asked for.

7 Oct, live: rahul watched the robot's own reported position cross the line it was given.
Measured over 225 positions: 2 mm outside at the 75th percentile, 40 mm at the 95th. The
robot hugs the line and rides a little past it, so it is handed a line 40 mm in.
"""

from __future__ import annotations

from deixis.where import Where
from pointdraw.vacuum.app import area_text, inset


def box(x=1000.0, y=1000.0, w=1000.0, h=1000.0) -> Where:
    return Where(frame="map", kind="box", box_mm=(x, y, w, h),
                 mean_mm=(x + w / 2, y + h / 2), source="gesture")


def test_it_shrinks_every_side_about_the_same_middle():
    got = inset(box(), 40.0)
    assert got.box_mm == (1040.0, 1040.0, 920.0, 920.0)
    assert got.mean_mm == (1500.0, 1500.0), "the middle must not move"


def test_the_robot_now_overshoots_onto_the_line_that_was_asked_for():
    """40 mm in, 40 mm of overshoot: the 95th percentile lands on the asked-for edge."""
    asked = box()
    sent = inset(asked, 40.0)
    assert sent.box_mm[0] - 40.0 == asked.box_mm[0]
    assert sent.box_mm[0] + sent.box_mm[2] + 40.0 == asked.box_mm[0] + asked.box_mm[2]


def test_a_narrow_strip_stays_a_strip():
    """Half a metre wide, shrunk by 40 mm a side, must not collapse."""
    got = inset(box(w=500.0, h=2000.0), 40.0)
    assert got.box_mm[2] == 420.0
    got = inset(box(w=90.0, h=2000.0), 40.0)  # absurdly thin: never more than a third
    assert got.box_mm[2] == 30.0


def test_nothing_happens_with_no_inset():
    assert inset(box(), 0.0).box_mm == (1000.0, 1000.0, 1000.0, 1000.0)


def test_what_it_says_out_loud_is_still_what_was_asked_for():
    """The person is told the size they asked for; the shrinking is for the robot only."""
    assert area_text(box()) == "about one metre square"
