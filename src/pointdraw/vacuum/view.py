"""What the vacuum app shows: the camera's picture with the floor drawn on it.

A faint 50 cm grid over the markers' patch (so a bad floor fit shows at once), and the one
look of every robot (claim 39q, `live/look.py`): where you stand is a filled amber dot, the
robot's guess a dashed blue box (a thin solid blue line once sent), and the status in large
plain words with the robot's last words under it.
"""

from __future__ import annotations

import cv2
import numpy as np

from pointdraw.live import look
from pointdraw.vacuum.app import CLEANING, PREVIEW, WAIT_CLEAR, VacuumApp
from pointdraw.vacuum.feet import FeetSeen
from pointdraw.vacuum.floor import FloorView

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
    area_mm: tuple[float, float, float, float],
    app: VacuumApp,
    feet: FeetSeen | None,
    t: float,
    sent_box: list[float] | None = None,
) -> np.ndarray:
    vis = image.copy()
    if view is None:
        look.status(vis, "Can't see the floor", RED)
        return vis
    # Nothing is drawn for the floor itself (rahul, 7 Oct): the 50 cm grid and the patch
    # outline were both correctly placed, but over a tiled floor they read as crooked and
    # pull the eye off what matters. What is on the floor is the dot and the robot's box.
    if not view.fresh:
        cv2.putText(vis, "floor fit kept (markers hidden)", (20, vis.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)  # fmt: skip
    if sent_box is not None:  # sent to the robot: a thin solid line
        cv2.polylines(vis, [_poly(view, box_corners(sent_box))], True, look.GUESS, 2, cv2.LINE_AA)
    prev = app.preview
    if prev is not None and prev.where.box_mm is not None:  # the robot's guess: dashed blue
        look.dashed(vis, _poly(view, box_corners(prev.where.box_mm)), thick=4)
    if feet is not None:  # you: filled amber
        for f in feet.feet_px:
            look.dot(vis, f, 5, WHITE)
        if feet.spot_px is not None:
            look.dot(vis, feet.spot_px, 14 if feet.standing else 9)
    said = _last_said(app) if app.state in (PREVIEW, WAIT_CLEAR) or app.question else None
    accent = {PREVIEW: look.GUESS, WAIT_CLEAR: look.AMBER, CLEANING: GREEN}.get(app.state)
    look.status(vis, app.banner(), accent, said)
    return vis


def _last_said(app: VacuumApp) -> str | None:
    says = app.log.of("say")
    return says[-1]["text"] if says else None
