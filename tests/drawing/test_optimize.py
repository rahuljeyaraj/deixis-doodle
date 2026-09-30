"""The optimiser keeps all ink, reduces travel and lifts, and simplifies within tolerance."""

import math
import random

import pytest

from pointdraw.drawing.optimize import (
    PlotParams,
    merge,
    optimize,
    order,
    path_length,
    simplify,
    stats,
)


def ink(pls):
    return sum(path_length(pl) for pl in pls)


def test_merge_joins_touching_lines_in_any_direction():
    pls = [[(0, 0), (10, 0)], [(20, 0), (10, 0)], [(20, 0.05), (30, 0)]]
    out = merge(pls, tol=0.1)
    assert len(out) == 1
    assert ink(out) == pytest.approx(ink(pls), abs=0.1)


def test_merge_closes_loops_and_leaves_them():
    square = [[(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)]]
    out = merge(square)
    assert len(out) == 1
    assert out[0][0] == out[0][-1]
    assert ink(out) == pytest.approx(40)


def test_merge_keeps_separate_lines():
    pls = [[(0, 0), (10, 0)], [(0, 5), (10, 5)]]
    assert len(merge(pls)) == 2


def test_order_reduces_travel_on_shuffled_strokes():
    rng = random.Random(1)
    pls = [[(x * 10.0, y * 10.0), (x * 10.0 + 5, y * 10.0)] for x in range(8) for y in range(6)]
    rng.shuffle(pls)
    before = stats(pls, start=(0, 0)).travel_mm
    after = stats(order(pls, (0, 0)), start=(0, 0)).travel_mm
    assert after < before / 3


def test_order_reverses_open_lines():
    out = order([[(100, 0), (10, 0)]], start=(0, 0))
    assert out[0][0] == (10, 0)


def test_order_restarts_closed_loop_at_nearest_vertex():
    loop = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    out = order([loop], start=(11, 11))
    assert out[0][0] == (10, 10) and out[0][-1] == (10, 10)
    assert path_length(out[0]) == pytest.approx(40)


def test_simplify_drops_collinear_points_and_respects_tolerance():
    line = [(x / 10, 0.05 * math.sin(x)) for x in range(101)]  # wobble 0.05 mm < 0.2
    (out,) = simplify([line], tol=0.2)
    assert out == [line[0], line[-1]]
    corner = [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10)]
    (out,) = simplify([corner], tol=0.2)
    assert out == [(0, 0), (10, 0), (10, 10)]


def test_simplify_keeps_small_loops():
    loop = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    (out,) = simplify([loop], tol=0.2)
    assert len(out) == 5 and out[0] == out[-1]


def test_optimize_keeps_all_ink_and_cuts_lifts_and_travel():
    rng = random.Random(7)
    # a polyline chopped into pieces, some reversed, shuffled: merge should rebuild it
    pts = [(i * 3.0, 20 + 10 * math.sin(i / 3)) for i in range(40)]
    pieces = [pts[i : i + 5] for i in range(0, 39, 4)]
    pieces = [p[::-1] if rng.random() < 0.5 else p for p in pieces]
    extra = [[(x, 100.0), (x + 4, 104.0)] for x in range(0, 120, 12)]
    pls = pieces + extra
    rng.shuffle(pls)
    before = stats(pls, start=(0, 0))
    out, after = optimize(pls, start=(0, 0))
    assert after.pen_down_mm == pytest.approx(before.pen_down_mm, rel=0.01)
    assert after.lifts < before.lifts
    assert after.travel_mm < before.travel_mm
    assert after.est_time_s < before.est_time_s
    assert after.points <= before.points


def test_stats_numbers():
    p = PlotParams(draw_mm_s=10, travel_mm_s=20, lift_s=1)
    s = stats([[(0, 0), (10, 0)], [(10, 20), (10, 30)]], start=(0, 0), params=p)
    assert s.pen_down_mm == pytest.approx(20)
    assert s.travel_mm == pytest.approx(20)
    assert s.lifts == 2
    assert s.est_time_s == pytest.approx(20 / 10 + 20 / 20 + 2)
    assert "2 lifts" in s.summary()


def test_single_points_survive():
    out, s = optimize([[(5, 5)], [(0, 0), (1, 0)]])
    assert [(5.0, 5.0)] in out
    assert s.lifts == 2
