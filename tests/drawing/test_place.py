"""Placement maths: size sets the larger dimension, centring, smoothing, keeping inside."""

import math

import pytest

from deixis.geometry import Box
from pointdraw.drawing.place import bbox, chaikin, fit, fit_in_box, keep_inside, smooth

WIDE = [[(0, 0), (100, 0), (100, 40)], [(10, 10), (50, 30)]]  # 100 x 40
TALL = [[(0, 0), (0, 200), (50, 200)]]  # 50 x 200


@pytest.mark.parametrize("strokes", [WIDE, TALL])
@pytest.mark.parametrize("smooth_iterations", [0, 2])
def test_fit_sets_larger_dimension_and_centre(strokes, smooth_iterations):
    out = fit(strokes, center=(80, 60), size=40, smooth_iterations=smooth_iterations)
    b = bbox(out)
    assert max(b.w, b.h) == pytest.approx(40)
    assert b.center[0] == pytest.approx(80)
    assert b.center[1] == pytest.approx(60)


def test_fit_keeps_aspect_ratio():
    b = bbox(fit(WIDE, (0, 0), 50, smooth_iterations=0))
    assert b.w / b.h == pytest.approx(100 / 40)


def test_fit_rejects_bad_size():
    with pytest.raises(ValueError):
        fit(WIDE, (0, 0), 0)


def test_fit_in_box_fits_and_centres():
    box = Box(10, 10, 30, 30)
    b = bbox(fit_in_box(WIDE, box, smooth_iterations=0))
    assert b.w == pytest.approx(30)
    assert b.h == pytest.approx(12)
    assert b.center == pytest.approx(box.center)


def test_chaikin_keeps_endpoints_and_sharp_corners():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]  # open, three 90 degree corners
    out = chaikin(square, iterations=3)
    assert out[0] == (0, 0) and out[-1] == (0, 10)
    for corner in [(10, 0), (10, 10)]:
        assert corner in out


def test_chaikin_smooths_gentle_turns():
    zig = [(0, 0), (10, 2), (20, 0), (30, 2)]  # turns of about 22 degrees
    out = chaikin(zig, iterations=2)
    assert len(out) > len(zig)
    assert (10, 2) not in out  # gentle vertices are cut


def test_chaikin_closed_ring_stays_closed():
    ring = [(math.cos(t) * 10, math.sin(t) * 10) for t in [i * math.pi / 4 for i in range(8)]]
    ring.append(ring[0])
    out = chaikin(ring, iterations=2)
    assert out[0] == pytest.approx(out[-1])
    assert len(out) > len(ring)


def test_smooth_short_strokes_untouched():
    assert smooth([[(0, 0), (1, 1)]]) == [[(0.0, 0.0), (1.0, 1.0)]]


def test_keep_inside_shifts_minimum():
    paper = Box(0, 0, 210, 148)
    pls = [[(200, 70), (220, 90)]]  # pokes 10 mm past the right edge
    out, changed = keep_inside(pls, paper, margin=5)
    b = bbox(out)
    assert changed
    assert b.x2 == pytest.approx(205)
    assert b.w == pytest.approx(20)  # not scaled
    assert b.y == pytest.approx(70)  # not moved vertically


def test_keep_inside_shrinks_when_too_big():
    paper = Box(0, 0, 210, 148)
    out, changed = keep_inside([[(0, 0), (300, 100)]], paper)
    b = bbox(out)
    assert changed
    assert b.w == pytest.approx(210)
    assert paper.contains((b.x, b.y)) and paper.contains((b.x2, b.y2))


def test_keep_inside_no_change():
    pls = [[(10, 10), (20, 20)]]
    out, changed = keep_inside(pls, Box(0, 0, 210, 148))
    assert not changed and out == [[(10, 10), (20, 20)]]


def test_clip_inside_cuts_not_moves():
    from pointdraw.drawing.place import clip_inside

    paper = Box(0, 0, 100, 100)
    # a cross centred near the east edge: the arm past x 100 is cut, the centre stays
    pls = [[(80, 50), (120, 50)], [(100, 30), (100, 70)], [(90, 40), (90, 60)]]
    out, cut = clip_inside(pls, paper)
    assert cut
    assert out[0] == [(80, 50), (100.0, 50.0)]
    assert out[2] == [(90, 40), (90, 60)]
    assert bbox(out).x2 <= 100


def test_clip_inside_splits_a_line_that_leaves_and_returns():
    from pointdraw.drawing.place import clip_inside

    out, cut = clip_inside([[(10, 90), (10, 110), (20, 110), (20, 90)]], Box(0, 0, 100, 100))
    assert cut
    assert out == [[(10, 90), (10.0, 100.0)], [(20.0, 100.0), (20, 90)]]


def test_clip_inside_no_change_and_all_outside():
    from pointdraw.drawing.place import clip_inside

    pls = [[(10, 10), (20, 20), (30, 10)]]
    assert clip_inside(pls, Box(0, 0, 100, 100)) == (pls, False)
    assert clip_inside([[(200, 200), (210, 210)]], Box(0, 0, 100, 100)) == ([], True)
