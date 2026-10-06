"""Small test shapes in bed mm, shared by `scripts/plot.py`, calibration and the M1 app."""

from __future__ import annotations

import math
from typing import Any

from deixis.geometry import Box, Point, Polyline


def cross(center: Point, size: float) -> list[Polyline]:
    """A plus sign `size` mm across: one horizontal and one vertical stroke."""
    cx, cy, h = center[0], center[1], size / 2
    return [[(cx - h, cy), (cx + h, cy)], [(cx, cy - h), (cx, cy + h)]]


def cross_grid(center: Point, cols: int, rows: int, spacing: float) -> list[dict[str, Any]]:
    """Cross centres, row by row from the front (low y), left to right."""
    x0 = center[0] - spacing * (cols - 1) / 2
    y0 = center[1] - spacing * (rows - 1) / 2
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(
                {"id": len(out), "col": c, "row": r, "x": x0 + c * spacing, "y": y0 + r * spacing}
            )
    return out


def frame(box: Box) -> Polyline:
    """The box's outline, closed, from its south-west corner. The live app draws the drawing
    area's frame at the start of a session: everything inside it is seen by the parked camera
    and reached by the pen (rahul, 2 Oct)."""
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def circle_points(
    center: Point,
    size: float,
    area: Box | None = None,
    tol: float = 0.2,
    start: Point | None = None,
) -> Polyline:
    """The circle of diameter `size` round `center`, closed, with chord error at most `tol` mm.
    It begins at the point nearest `start` (where the head is), else due east. With `area`
    every point is pulled onto it, so a circle partly off the page runs along the edge there:
    the pen-up preview shows the part that will be cut."""
    r = size / 2
    if r <= 0:
        return [center]
    n = max(24, math.ceil(math.pi / math.acos(max(-1.0, 1 - tol / r))))
    a0 = math.atan2(start[1] - center[1], start[0] - center[0]) if start else 0.0
    pts = [
        (center[0] + r * math.cos(a0 + 2 * math.pi * i / n),
         center[1] + r * math.sin(a0 + 2 * math.pi * i / n))
        for i in range(n + 1)
    ]  # fmt: skip
    if area is not None:
        pts = [(min(max(x, area.x), area.x2), min(max(y, area.y), area.y2)) for x, y in pts]
    out: Polyline = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) > 1e-9:
            out.append(p)
    return out
