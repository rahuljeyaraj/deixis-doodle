"""What the vacuum app shows: the camera's picture with the floor drawn on it.

A faint 50 cm grid over the markers' patch (so a bad floor fit shows at once), the person's
spot, the box being asked about (yellow; green once "yes" is said; blue once sent) and a
banner in big words saying what the app is doing.
"""

from __future__ import annotations

import cv2
import numpy as np

from pointdraw.vacuum.app import CLEANING, PREVIEW, WAIT_CLEAR, VacuumApp
from pointdraw.vacuum.feet import FeetSeen
from pointdraw.vacuum.floor import FloorLayout, FloorView

YELLOW, GREEN, BLUE, RED, WHITE, GREY = (
    (0, 220, 255),
    (60, 220, 60),
    (255, 140, 40),
    (60, 60, 255),
    (255, 255, 255),
    (170, 170, 170),
)


def _poly(view: FloorView, pts_mm: list[tuple[float, float]], steps: int = 20) -> np.ndarray:
    """A floor polygon as image points, with its edges cut into steps (a lens may bend them)."""
    out = []
    for a, b in zip(pts_mm, pts_mm[1:] + pts_mm[:1], strict=True):
        for k in range(steps):
            f = k / steps
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return np.round(view.to_px(np.array(out))).astype(np.int32)


def _line(
    view: FloorView, a: tuple[float, float], b: tuple[float, float], steps: int = 10
) -> np.ndarray:
    pts = [
        (a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
        for k in range(steps + 1)
    ]
    return np.round(view.to_px(np.array(pts))).astype(np.int32)


def box_corners(box: tuple[float, float, float, float] | list[float]) -> list[tuple[float, float]]:
    x, y, w, h = box
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def draw(
    image: np.ndarray,
    view: FloorView | None,
    layout: FloorLayout,
    app: VacuumApp,
    feet: FeetSeen | None,
    t: float,
    sent_box: list[float] | None = None,
) -> np.ndarray:
    vis = image.copy()
    if view is None:
        _banner(vis, "Can't see the 4 floor markers", RED)
        return vis
    x, y, w, h = layout.area_mm
    for gx in np.arange(x, x + w + 1, 500.0):
        cv2.polylines(vis, [_line(view, (gx, y), (gx, y + h))], False, GREY, 1)
    for gy in np.arange(y, y + h + 1, 500.0):
        cv2.polylines(vis, [_line(view, (x, gy), (x + w, gy))], False, GREY, 1)
    cv2.polylines(vis, [_poly(view, box_corners((x, y, w, h)))], True, WHITE, 2)
    if not view.fresh:
        cv2.putText(vis, "floor fit kept (markers hidden)", (20, vis.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)  # fmt: skip
    if sent_box is not None:
        cv2.polylines(vis, [_poly(view, box_corners(sent_box))], True, BLUE, 3)
    prev = app.preview
    if prev is not None and prev.where.box_mm is not None:
        colour = GREEN if app.state == WAIT_CLEAR else YELLOW
        pts = _poly(view, box_corners(prev.where.box_mm))
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        cv2.polylines(vis, [pts], True, colour, 3)
    if feet is not None:
        for f in feet.feet_px:
            cv2.circle(vis, (int(f[0]), int(f[1])), 6, WHITE, -1)
        if feet.spot_px is not None:
            colour = GREEN if feet.standing else YELLOW
            cv2.circle(vis, (int(feet.spot_px[0]), int(feet.spot_px[1])), 14, colour, 3)
    colour = {PREVIEW: YELLOW, WAIT_CLEAR: GREEN, CLEANING: BLUE}.get(app.state, WHITE)
    _banner(vis, app.banner(), colour)
    return vis


def _banner(vis: np.ndarray, text: str, colour: tuple[int, int, int]) -> None:
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 70), (30, 30, 30), -1)
    cv2.putText(vis, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
