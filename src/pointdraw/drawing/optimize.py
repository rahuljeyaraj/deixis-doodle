"""Pen-plotter path optimisation: merge, order, simplify, and job statistics.

Pure Python and numpy, so it is fast to test and has no heavy dependencies. It does what
vpype's `linemerge`, `linesort` and `linesimplify` do, tuned for this robot: Z lifts on an
Ender 3 are slow (leadscrew), so fewer lifts matter more than a little extra travel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from deixis.geometry import Point, Polyline


@dataclass(frozen=True)
class PlotParams:
    """Numbers for the time estimate. Defaults match the plan's conservative plotter speeds."""

    draw_mm_s: float = 2000 / 60  # G1 feedrate while the pen is down
    travel_mm_s: float = 6000 / 60  # G0 feedrate with the pen up
    lift_s: float = 1.0  # one pen up plus pen down (Ender 3 Z is about 5 mm/s)


DEFAULT_PLOT = PlotParams()


@dataclass(frozen=True)
class Stats:
    pen_down_mm: float
    travel_mm: float
    lifts: int  # number of pen-down strokes (each ends with one lift)
    points: int
    est_time_s: float

    def summary(self) -> str:
        return (
            f"{self.pen_down_mm:.0f} mm ink, {self.travel_mm:.0f} mm travel, "
            f"{self.lifts} lifts, {self.points} points, ~{self.est_time_s:.0f} s"
        )


def path_length(pl: Sequence[Point]) -> float:
    if len(pl) < 2:
        return 0.0
    a = np.asarray(pl, dtype=float)
    return float(np.sum(np.hypot(*np.diff(a, axis=0).T)))


def _is_closed(pl: Sequence[Point], tol: float) -> bool:
    return len(pl) > 2 and math.dist(pl[0], pl[-1]) <= tol


def _join(a: Polyline, b: Polyline, tol: float) -> Polyline | None:
    if math.dist(a[-1], b[0]) <= tol:
        return a + b[1:]
    if math.dist(a[-1], b[-1]) <= tol:
        return a + b[-2::-1]
    if math.dist(a[0], b[-1]) <= tol:
        return b + a[1:]
    if math.dist(a[0], b[0]) <= tol:
        return b[::-1] + a[1:]
    return None


def merge(polylines: Sequence[Sequence[Point]], tol: float = 0.1) -> list[Polyline]:
    """Join polylines whose endpoints are within `tol` mm, reversing where needed.

    Closed loops and single points are left alone. Each join saves one pen lift.
    """
    lines = [[(float(x), float(y)) for x, y in pl] for pl in polylines if len(pl) > 0]
    fixed = [pl for pl in lines if len(pl) < 2 or _is_closed(pl, tol)]
    open_ = [pl for pl in lines if len(pl) >= 2 and not _is_closed(pl, tol)]
    i = 0
    while i < len(open_):
        for j in range(i + 1, len(open_)):
            joined = _join(open_[i], open_[j], tol)
            if joined is None:
                continue
            del open_[j]
            if _is_closed(joined, tol):
                joined[-1] = joined[0]
                del open_[i]
                fixed.append(joined)
            else:
                open_[i] = joined
            break
        else:
            i += 1  # nothing joins open_[i] any more
    return fixed + open_


def order(
    polylines: Sequence[Sequence[Point]], start: Point = (0.0, 0.0), tol: float = 0.1
) -> list[Polyline]:
    """Greedy nearest-neighbour order from `start`. Open lines may be reversed; closed
    loops are restarted at their vertex nearest the pen."""
    todo = [list(pl) for pl in polylines if len(pl) > 0]
    out: list[Polyline] = []
    pen = start
    while todo:
        best: tuple[float, int, bool, int | None] | None = None
        for i, pl in enumerate(todo):
            if _is_closed(pl, tol):
                ring = np.asarray(pl[:-1], dtype=float)
                d = np.hypot(ring[:, 0] - pen[0], ring[:, 1] - pen[1])
                k = int(np.argmin(d))
                cand = (float(d[k]), i, False, k)
            else:
                d0, d1 = math.dist(pen, pl[0]), math.dist(pen, pl[-1])
                cand = (d0, i, False, None) if d0 <= d1 else (d1, i, True, None)
            if best is None or cand[0] < best[0]:
                best = cand
        assert best is not None
        _, i, rev, k = best
        pl = todo.pop(i)
        if k:
            ring = pl[:-1]
            pl = ring[k:] + ring[:k] + [ring[k]]
        elif rev:
            pl = pl[::-1]
        out.append(pl)
        pen = pl[-1]
    return out


def _rdp(pts: np.ndarray, tol: float) -> list[int]:
    """Indices kept by Ramer-Douglas-Peucker (iterative, so long strokes can't overflow)."""
    keep = {0, len(pts) - 1}
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = b - a
        n = math.hypot(*seg)
        mid = pts[i + 1 : j]
        if n == 0:
            d = np.hypot(*(mid - a).T)
        else:
            d = np.abs(seg[0] * (mid[:, 1] - a[1]) - seg[1] * (mid[:, 0] - a[0])) / n
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep.add(m)
            stack += [(i, m), (m, j)]
    return sorted(keep)


def simplify(polylines: Sequence[Sequence[Point]], tol: float = 0.2) -> list[Polyline]:
    """Drop points that move the line less than `tol` mm. Endpoints are always kept."""
    out: list[Polyline] = []
    for pl in polylines:
        if len(pl) < 3:
            out.append([(float(x), float(y)) for x, y in pl])
            continue
        a = np.asarray(pl, dtype=float)
        if _is_closed(pl, 1e-9):
            # split the loop at its farthest point so it never collapses to a line
            far = int(np.argmax(np.hypot(*(a - a[0]).T)))
            first = _rdp(a[: far + 1], tol)
            second = [far + i for i in _rdp(a[far:], tol)]
            idx = first + second[1:]
        else:
            idx = _rdp(a, tol)
        out.append([(float(a[i, 0]), float(a[i, 1])) for i in idx])
    return out


def stats(
    polylines: Sequence[Sequence[Point]],
    start: Point | None = None,
    params: PlotParams = DEFAULT_PLOT,
) -> Stats:
    """Job numbers in the given order. Travel counts from `start` if given; the trip back
    afterwards is not counted."""
    down = sum(path_length(pl) for pl in polylines)
    travel = 0.0
    pen = start
    for pl in polylines:
        if not pl:
            continue
        if pen is not None:
            travel += math.dist(pen, pl[0])
        pen = pl[-1]
    lifts = sum(1 for pl in polylines if pl)
    t = down / params.draw_mm_s + travel / params.travel_mm_s + lifts * params.lift_s
    return Stats(down, travel, lifts, sum(len(pl) for pl in polylines), t)


def optimize(
    polylines: Sequence[Sequence[Point]],
    start: Point = (0.0, 0.0),
    merge_tol: float = 0.1,
    simplify_tol: float = 0.2,
    params: PlotParams = DEFAULT_PLOT,
) -> tuple[list[Polyline], Stats]:
    """merge, simplify, order. Returns the polylines to draw and their stats."""
    pls = merge(polylines, merge_tol)
    pls = simplify(pls, simplify_tol)
    pls = order(pls, start, merge_tol)
    return pls, stats(pls, start, params)
