"""Centre-line tracing: simple shapes come back as single centre lines, never as outlines, and
real library drawings survive a render -> trace round trip within the plan's targets."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pointdraw.drawing.optimize import path_length
from pointdraw.drawing.trace import (
    LIBRARY_SIDE,
    compare,
    degrade,
    double_line_score,
    load_gray,
    render_strokes,
    to_library,
    trace,
    trace_to_library,
)

LIB = Path(__file__).resolve().parents[2] / "library"
PX_PER_MM = 10.0  # for the shape tests: 1 mm = 10 px


def traced(strokes, size=400, width=10.0, margin=60.0, noisy=True, seed=0):
    img, orig, _ = render_strokes(strokes, size, width, margin)
    if noisy:
        img = degrade(img, seed=seed)
    res = trace(img)
    return res, [o.tolist() for o in orig]


def test_straight_thick_line_is_one_stroke_along_its_centre():
    res, orig = traced([[(0, 0), (100, 30)]], width=12)
    assert len(res.strokes) == 1
    m = compare(res.strokes, orig, PX_PER_MM)
    assert m["mean_mm"] < 0.1
    assert m["max_mm"] < 0.5
    # the ends land on the original ends (either direction)
    a, b = res.strokes[0][0], res.strokes[0][-1]
    p, q = orig[0][0], orig[0][-1]
    assert min(math.dist(a, p) + math.dist(b, q), math.dist(a, q) + math.dist(b, p)) < 6
    assert double_line_score(res.strokes, 1.5 * 12, min_gap_px=0.3 * 12) == 0.0


def test_width_is_measured():
    for w in (4, 8, 14):
        res, _ = traced([[(0, 0), (100, 0)], [(0, 50), (100, 50)]], width=w, noisy=False)
        assert res.line_width_px == pytest.approx(w, abs=1.5)


def test_l_shape_is_one_stroke_with_a_sharp_corner():
    res, orig = traced([[(0, 100), (0, 0), (80, 0)]], width=10)
    assert len(res.strokes) == 1
    m = compare(res.strokes, orig, PX_PER_MM)
    assert m["mean_mm"] < 0.1 and m["p95_mm"] < 0.3


def test_t_shape_keeps_the_bar_whole_and_ends_the_stem_on_it():
    bar, stem = [(0, 100), (100, 100)], [(50, 100), (50, 0)]
    res, orig = traced([bar, stem], width=10)
    assert len(res.strokes) == 2
    lengths = sorted(path_length(s) for s in res.strokes)
    o_bar, o_stem = path_length(orig[0]), path_length(orig[1])
    assert lengths[1] == pytest.approx(o_bar, rel=0.05)  # the bar goes straight through
    assert lengths[0] == pytest.approx(o_stem, rel=0.08)
    assert compare(res.strokes, orig, PX_PER_MM)["p95_mm"] < 0.3


def test_x_crossing_is_two_straight_strokes():
    res, orig = traced([[(0, 0), (100, 100)], [(0, 100), (100, 0)]], width=10)
    assert len(res.strokes) == 2
    for s in res.strokes:  # each goes corner to corner, not round a bend
        assert math.dist(s[0], s[-1]) == pytest.approx(path_length(s), rel=0.02)


def test_shallow_crossing_is_still_two_strokes():
    # thinning turns a 20 degree X into two Ys joined by a bridge; they must be merged
    res, orig = traced([[(0, 0), (200, 35)], [(0, 35), (200, 0)]], size=600, width=10)
    assert len(res.strokes) == 2
    assert compare(res.strokes, orig, PX_PER_MM)["p95_mm"] < 0.5


def test_sharp_v_stays_one_stroke_and_reaches_its_tip():
    tip = (50, 0)
    res, orig = traced([[(0, 100), tip, (30, 100)]], width=12)
    assert len(res.strokes) == 1
    m = compare(res.strokes, orig, PX_PER_MM)
    assert m["p95_mm"] < 0.5 and m["max_mm"] < 1.0


def test_ring_is_one_closed_stroke_with_no_double_line():
    t = np.linspace(0, 2 * math.pi, 73)
    ring = [(50 + 40 * math.cos(a), 50 + 40 * math.sin(a)) for a in t]
    res, orig = traced([ring], width=14)
    assert len(res.strokes) == 1
    s = res.strokes[0]
    assert s[0] == s[-1]  # closed
    assert double_line_score(res.strokes, 1.5 * 14, min_gap_px=0.3 * 14) == 0.0
    m = compare(res.strokes, orig, PX_PER_MM)
    assert m["mean_mm"] < 0.1 and m["max_mm"] < 0.3


def test_double_line_score_catches_outlines():
    # an outline tracer draws both edges of a line: two parallel strokes one width apart
    w = 10.0
    edges = [[(0.0, 0.0), (300.0, 0.0)], [(0.0, w), (300.0, w)]]
    assert double_line_score(edges, 1.5 * w, min_gap_px=0.3 * w) > 0.95
    assert double_line_score(edges[:1], 1.5 * w, min_gap_px=0.3 * w) == 0.0


def test_blank_image_gives_nothing_and_specks_are_dropped():
    assert trace(np.full((200, 200), 255, np.uint8)).strokes == []
    img, _, _ = render_strokes([[(0, 0), (100, 40)]], 400, 10, 60)
    res = trace(degrade(img, specks=80, seed=3))  # 80 dark specks 1 to 3 px across
    assert len(res.strokes) == 1


def test_load_gray_accepts_rgba_float_and_files(tmp_path):
    img, _, _ = render_strokes([[(0, 0), (100, 0)]], 200, 8, 40)
    rgba = np.dstack([img, img, img, np.full_like(img, 255)])
    assert np.array_equal(load_gray(rgba), img)
    assert np.abs(load_gray(img / 255.0).astype(int) - img).max() <= 1
    import cv2

    path = tmp_path / "line.png"
    cv2.imwrite(str(path), img)
    assert len(trace(path).strokes) == 1


def test_to_library_normalises_like_the_library():
    strokes = [[(100.0, 300.0), (500.0, 300.0)], [(100.0, 100.0), (100.0, 300.0)]]
    lib = to_library(strokes)
    pts = [p for s in lib for p in s]
    assert min(p[0] for p in pts) == pytest.approx(0)
    assert min(p[1] for p in pts) == pytest.approx(0)
    assert max(max(p[0] for p in pts), max(p[1] for p in pts)) == pytest.approx(LIBRARY_SIDE)
    # y is flipped: the pixel row 100 (top of the image) is the top of the drawing
    assert lib[1][0][1] > lib[1][1][1]


@pytest.mark.parametrize("category,variant", [("ear", 0), ("house", 1), ("bicycle", 1)])
def test_round_trip_of_library_drawings_meets_the_targets(category, variant):
    data = json.loads((LIB / "quickdraw" / f"{category}.json").read_text(encoding="utf-8"))
    strokes = data["variants"][variant]["strokes"]
    width = 8.0
    img, orig_px, _ = render_strokes(strokes, 1024, width, 64)
    res = trace(degrade(img, seed=variant))
    orig = [o.tolist() for o in orig_px]
    px_per_mm = (1024 - 128) / 40  # the drawing at 40 mm
    m = compare(res.strokes, orig, px_per_mm)
    assert m["mean_mm"] <= 0.3
    assert m["p95_mm"] <= 1.0
    dbl = double_line_score(res.strokes, 1.5 * width, min_gap_px=0.3 * width)
    dbl_orig = double_line_score(orig, 1.5 * width, min_gap_px=0.3 * width)
    assert dbl - dbl_orig < 0.02
    # ink is kept and the pen doesn't lift much more than in the original
    ink = sum(path_length(s) for s in res.strokes) / sum(path_length(s) for s in orig)
    assert 0.9 < ink < 1.1
    assert len(res.strokes) <= len(orig) + 2


def test_trace_to_library_output_loads_as_library_strokes():
    img, _, _ = render_strokes([[(0, 0), (60, 100), (120, 0)], [(30, 50), (90, 50)]], 512, 8)
    lib = trace_to_library(img)
    assert 1 <= len(lib) <= 3
    assert all(len(s) >= 2 and all(len(p) == 2 for p in s) for s in lib)
