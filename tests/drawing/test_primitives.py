"""Arrow geometry, trimming at object boxes, circles and boxes."""

import math

import pytest

from deixis.geometry import Box
from pointdraw.drawing.primitives import (
    arrow,
    arrow_head_length,
    box,
    circle,
    line,
    trim_to_boxes,
)


def test_arrow_tip_and_symmetric_head():
    shaft, head = arrow((0, 0), (100, 0))
    assert shaft == [(0, 0), (100, 0)]
    left, tip, right = head
    assert tip == (100, 0)
    assert math.dist(left, tip) == pytest.approx(math.dist(right, tip))
    assert left[1] == pytest.approx(-right[1])  # mirrored about the shaft
    assert left[0] < 100 and right[0] < 100  # barbs point back along the shaft
    half_angle = math.degrees(math.atan2(abs(left[1]), 100 - left[0]))
    assert half_angle == pytest.approx(28)


@pytest.mark.parametrize(
    ("length", "expected"), [(10, 3.0), (50, 9.0), (200, 10.0)]
)  # clamped low, proportional, clamped high
def test_head_is_proportional_and_clamped(length, expected):
    assert arrow_head_length(length) == pytest.approx(expected)
    _, head = arrow((0, 0), (length, 0))
    assert math.dist(head[0], head[1]) == pytest.approx(expected)


def test_arrow_any_direction():
    _, head = arrow((50, 50), (50, 10))  # pointing down the page
    assert head[1] == (50, 10)
    assert head[0][1] > 10 and head[2][1] > 10


def test_trim_to_boxes_gaps():
    a_box = Box.from_center(0, 0, 20, 20)
    b_box = Box.from_center(100, 0, 40, 40)
    a, b = trim_to_boxes((0, 0), (100, 0), a_box, b_box, gap=2)
    assert a == pytest.approx((12, 0))  # exits at x=10, plus 2 mm
    assert b == pytest.approx((78, 0))  # enters at x=80, minus 2 mm


def test_trim_diagonal_exits_through_nearest_side():
    a, _ = trim_to_boxes((0, 0), (100, 50), Box.from_center(0, 0, 20, 20), None, gap=0)
    assert a[0] == pytest.approx(10)
    assert a[1] == pytest.approx(5)


def test_trim_ignores_ends_outside_their_box():
    a, b = trim_to_boxes((0, 0), (100, 0), Box(50, 50, 10, 10), Box(-50, -50, 5, 5))
    assert (a, b) == ((0, 0), (100, 0))


def test_trim_falls_back_when_boxes_overlap():
    a, b = trim_to_boxes((0, 0), (10, 0), Box(-20, -20, 40, 40), Box(-20, -20, 40, 40))
    assert (a, b) == ((0, 0), (10, 0))


def test_arrow_trimmed_between_objects():
    shaft, head = arrow(
        (0, 0), (100, 0), Box.from_center(0, 0, 20, 20), Box.from_center(100, 0, 20, 20)
    )
    assert shaft[0] == pytest.approx((12, 0))
    assert head[1] == pytest.approx((88, 0))


def test_arrow_rejects_zero_length():
    with pytest.raises(ValueError):
        arrow((5, 5), (5, 5))


def test_circle_closed_and_accurate():
    (pts,) = circle((10, 20), 30, tol=0.05)
    assert pts[0] == pts[-1]
    for x, y in pts:
        assert math.hypot(x - 10, y - 20) == pytest.approx(30)
    # chord midpoints stay within tolerance of the true circle
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        assert 30 - math.hypot(mx - 10, my - 20) <= 0.05 + 1e-9


def test_circle_rejects_bad_radius():
    with pytest.raises(ValueError):
        circle((0, 0), 0)


def test_box_and_line():
    (pts,) = box(Box(1, 2, 3, 4))
    assert pts == [(1, 2), (4, 2), (4, 6), (1, 6), (1, 2)]
    assert line((0, 0), (1, 1)) == [[(0.0, 0.0), (1.0, 1.0)]]
