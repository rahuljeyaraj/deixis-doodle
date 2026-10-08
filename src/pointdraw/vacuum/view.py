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

EDGE = (26, 26, 26)
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


TRAIL_ALPHA = 0.5  # rahul, 8 Oct: the path is where it has been, not what it is doing now


def trail(
    vis: np.ndarray,
    points_px: list[tuple[float, float]],
    k: float = 1.0,
    alpha: float = TRAIL_ALPHA,
) -> None:
    """Where the robot's lid marker has been, in the picture's own pixels: a green line with
    a dark edge (a pale floor washes out a plain green one) and a dot where it is now, all
    laid over the picture at `alpha` so it never fights with the box or the amber dot."""
    if len(points_px) < 2:
        return
    pts = np.round(np.array(points_px, dtype=float)).astype(np.int32)
    layer = vis.copy()
    cv2.polylines(layer, [pts], False, EDGE, max(3, round(6 * k)), cv2.LINE_AA)
    cv2.polylines(layer, [pts], False, GREEN, max(1, round(3 * k)), cv2.LINE_AA)
    look.dot(layer, pts[-1], max(5, round(9 * k)), GREEN)
    cv2.addWeighted(layer, alpha, vis, 1 - alpha, 0, dst=vis)


def box_corners(box: tuple[float, float, float, float] | list[float]) -> list[tuple[float, float]]:
    x, y, w, h = box
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def chat_lines(vis: np.ndarray, lines: list[tuple[str, str]]) -> None:
    """The thread of the take at the bottom left: both sides, oldest first.

    Sizes are relative to the frame's width, 1280 being 1: the marks go on the camera's own
    frame, so a fixed font would come out half the size on a 1920-wide camera and twice it on
    a 640 one. "heard"/"deixis" rather than "you", which must not appear in anything judges
    see.
    """
    if not lines:
        return
    k = _k(vis)
    scale, thick = 0.5 * k, max(1, round(2 * k))
    step, pad = max(12, round(30 * k)), max(6, round(14 * k))
    texts = [(who, f"{'heard' if who == 'you' else 'deixis'}: {_cut(text)}") for who, text in lines]
    wide = max(cv2.getTextSize(t, look.FONT, scale, thick)[0][0] for _, t in texts)
    w = min(vis.shape[1], wide + 2 * pad)
    top = vis.shape[0] - pad - step * len(texts)
    panel = vis[max(0, top - pad) : vis.shape[0], 0:w]
    panel[:] = cv2.addWeighted(panel, 0.35, np.full_like(panel, look.BAR[0]), 0.65, 0)
    for i, (who, text) in enumerate(texts):
        colour = WHITE if who == "you" else look.AMBER
        y = top + step * i + round(step * 0.75)
        cv2.putText(vis, text, (pad, y), look.FONT, scale, colour, thick, cv2.LINE_AA)


def _k(vis: np.ndarray) -> float:
    """How much bigger than a 1280-wide picture this frame is, for text and line sizes."""
    return vis.shape[1] / 1280


def _cut(text: str, most: int = 52) -> str:
    return text if len(text) <= most else text[: most - 1] + "..."


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
    k = _k(vis)
    if view is None:
        look.status(vis, "Can't see the floor", RED, scale=1.4 * k)
        return vis
    # Nothing is drawn for the floor itself (rahul, 7 Oct): the 50 cm grid and the patch
    # outline were both correctly placed, but over a tiled floor they read as crooked and
    # pull the eye off what matters. What is on the floor is the dot and the robot's box.
    if not view.fresh:
        note, font = "floor fit kept (markers hidden)", cv2.FONT_HERSHEY_SIMPLEX
        (nw, _), _ = cv2.getTextSize(note, font, 0.7 * k, max(1, round(2 * k)))
        # bottom right: the chat thread has the bottom left, the banner the top
        cv2.putText(vis, note, (vis.shape[1] - nw - round(20 * k), vis.shape[0] - round(20 * k)),
                    font, 0.7 * k, GREY, max(1, round(2 * k)))  # fmt: skip
    trail(vis, list(app.trail), k)  # where the robot has driven, as the camera saw it
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
    look.status(vis, app.banner(), accent, said, scale=1.4 * k)
    chat_lines(vis, app.chat())
    return vis


def _last_said(app: VacuumApp) -> str | None:
    says = app.log.of("say")
    return says[-1]["text"] if says else None
