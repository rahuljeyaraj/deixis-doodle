"""Small test shapes in bed mm, shared by `scripts/plot.py`, calibration and the M1 app."""

from __future__ import annotations

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
