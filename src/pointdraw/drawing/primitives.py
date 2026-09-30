"""Geometric primitives in millimetres: line, arrow, box and circle.

Every function returns a list of polylines, ready for `optimize.optimize`.
"""

from __future__ import annotations

import math

from deixis.geometry import Box, Point, Polyline


def line(a: Point, b: Point) -> list[Polyline]:
    return [[(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))]]


def box(b: Box) -> list[Polyline]:
    """Closed rectangle outline, one stroke."""
    return [[(b.x, b.y), (b.x2, b.y), (b.x2, b.y2), (b.x, b.y2), (b.x, b.y)]]


def circle(center: Point, r: float, tol: float = 0.05) -> list[Polyline]:
    """Closed circle, one stroke, with chord error at most `tol` mm."""
    if r <= 0:
        raise ValueError("radius must be positive")
    # sagitta r(1 - cos(pi/n)) <= tol
    n = max(12, math.ceil(math.pi / math.acos(max(-1.0, 1 - tol / r))))
    cx, cy = center
    pts = [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return [pts + [pts[0]]]


def _exit_t(p: Point, d: Point, b: Box) -> float | None:
    """Ray p + t*d starts inside b: return t where it leaves b (None if p is outside b)."""
    if not b.contains(p):
        return None
    ts = []
    for pc, dc, lo, hi in ((p[0], d[0], b.x, b.x2), (p[1], d[1], b.y, b.y2)):
        if dc > 0:
            ts.append((hi - pc) / dc)
        elif dc < 0:
            ts.append((lo - pc) / dc)
    return min(ts) if ts else None


def trim_to_boxes(
    a: Point,
    b: Point,
    from_box: Box | None = None,
    to_box: Box | None = None,
    gap: float = 2.0,
    min_length: float = 5.0,
) -> tuple[Point, Point]:
    """Shorten segment a->b so it starts `gap` mm outside from_box and ends `gap` mm
    before to_box. Only trims ends that lie inside their box. If trimming would leave less
    than `min_length` (the boxes touch or overlap), the untrimmed segment is returned.
    """
    length = math.dist(a, b)
    if length == 0:
        return a, b
    d = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    t0, t1 = 0.0, length
    if from_box is not None:
        t = _exit_t(a, d, from_box)
        if t is not None:
            t0 = t + gap
    if to_box is not None:
        t = _exit_t(b, (-d[0], -d[1]), to_box)
        if t is not None:
            t1 = length - t - gap
    if t1 - t0 < min_length:
        return a, b
    return (a[0] + d[0] * t0, a[1] + d[1] * t0), (a[0] + d[0] * t1, a[1] + d[1] * t1)


def arrow_head_length(
    length: float, ratio: float = 0.18, lo: float = 3.0, hi: float = 10.0
) -> float:
    """Head size proportional to the shaft, clamped so short arrows still read as arrows."""
    return max(lo, min(hi, ratio * length))


def arrow(
    a: Point,
    b: Point,
    from_box: Box | None = None,
    to_box: Box | None = None,
    gap: float = 2.0,
    head: float | None = None,
    head_angle_deg: float = 28.0,
) -> list[Polyline]:
    """Arrow from a to b: the shaft, then a V head at b as a second stroke.

    With boxes (the objects being connected), the arrow is trimmed back so it starts and
    ends `gap` mm outside them rather than on top of the drawings.
    """
    a, b = trim_to_boxes(a, b, from_box, to_box, gap)
    length = math.dist(a, b)
    if length == 0:
        raise ValueError("arrow needs two distinct points")
    h = head if head is not None else arrow_head_length(length)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    spread = math.radians(head_angle_deg)
    left = (b[0] - h * math.cos(ang - spread), b[1] - h * math.sin(ang - spread))
    right = (b[0] - h * math.cos(ang + spread), b[1] - h * math.sin(ang + spread))
    return [[a, b], [left, b, right]]
