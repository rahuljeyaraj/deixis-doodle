"""Hershey text: cap height, alignment, bounds."""

import pytest

from pointdraw.drawing.place import bbox
from pointdraw.drawing.text import text, text_bounds


def test_cap_height_is_the_requested_size():
    b = bbox(text("H", (0, 0), cap_height=7, valign="baseline"))
    assert b.h == pytest.approx(7)
    assert b.y == pytest.approx(0)  # sits on the baseline (y up)


def test_middle_valign_centres_caps_on_anchor():
    b = bbox(text("HELLO", (50, 40), cap_height=6, valign="middle"))
    assert b.y == pytest.approx(37)
    assert b.y2 == pytest.approx(43)


def test_top_valign():
    b = bbox(text("H", (0, 100), cap_height=5, valign="top"))
    assert b.y2 == pytest.approx(100)


def test_center_align_is_centred_on_anchor():
    b = text_bounds("Nemotron", (105, 74), cap_height=5)
    assert b.center[0] == pytest.approx(105)


def test_left_and_right_align():
    assert text_bounds("Cosmos", (20, 0), 5, align="left").x == pytest.approx(20)
    assert text_bounds("Cosmos", (20, 0), 5, align="right").x2 == pytest.approx(20)


def test_width_scales_with_size():
    w5 = text_bounds("Nemotron", (0, 0), 5).w
    w10 = text_bounds("Nemotron", (0, 0), 10).w
    assert w10 == pytest.approx(2 * w5)
    assert 4 * 5 < w5 < 8 * 5  # plausible for 8 letters


def test_descender_goes_below_baseline():
    b = bbox(text("g", (0, 0), 5, valign="baseline"))
    assert b.y < -1


def test_multiline_stacks_down():
    one = text_bounds("ASR", (0, 0), 5, valign="baseline")
    two = text_bounds("ASR\nASR", (0, 0), 5, valign="baseline")
    assert two.y == pytest.approx(one.y - 1.7 * 5)
    assert two.y2 == pytest.approx(one.y2)


def test_single_stroke_polylines():
    pls = text("A", (0, 0), 5)
    assert 1 <= len(pls) <= 3
    assert all(len(pl) >= 2 for pl in pls)


def test_bad_inputs():
    with pytest.raises(ValueError):
        text("x", (0, 0), 0)
    with pytest.raises(ValueError):
        text("x", (0, 0), 5, font="comic-sans")


def test_blank_text_is_empty():
    assert text("   ", (0, 0), 5) == []
