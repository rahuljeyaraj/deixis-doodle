"""One look for every robot's window (claim 39q; the same as the web demo and the film).

- What **you** did is **filled amber**: your fingertip dot, your holds, your standing spot.
- The **robot's guess** is a **dashed blue outline**, with a blue cross on a circle's centre.
- Both have a thin dark edge, so they read on white paper, a wood floor or a yellow arm.
- The status is **large plain text** on a dark bar; the robot's last words sit under it.

Colours are OpenCV BGR. Amber #FFB000, guess blue #1E88E5, edge #1A1A1A.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

AMBER = (0, 176, 255)
GUESS = (229, 136, 30)
EDGE = (26, 26, 26)
WHITE = (255, 255, 255)
BAR = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_DUPLEX


def dot(vis: np.ndarray, pt: Sequence[float], r: int = 9, colour=AMBER) -> None:
    """A filled dot with a dark edge (your fingertip, a hold, your feet's spot)."""
    c = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(vis, c, r + 2, EDGE, -1, cv2.LINE_AA)
    cv2.circle(vis, c, r, colour, -1, cv2.LINE_AA)


def dashed(
    vis: np.ndarray,
    pts: np.ndarray,
    closed: bool = True,
    colour=GUESS,
    thick: int = 2,
    dash: float = 14.0,
    gap: float = 9.0,
) -> None:
    """A dashed outline through image points `pts` (N x 2), with a dark edge."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) < 2:
        return
    if closed:
        p = np.vstack([p, p[:1]])
    segs = []
    period, phase = dash + gap, 0.0  # phase: how far into the dash-gap pattern we are
    for a, b in zip(p[:-1], p[1:], strict=True):
        d = float(np.hypot(*(b - a)))
        s = 0.0
        while s < d - 1e-9:
            if phase < dash:  # in a dash
                e = min(d, s + dash - phase)
                segs.append((a + (b - a) * (s / d), a + (b - a) * (e / d)))
            else:  # in a gap
                e = min(d, s + period - phase)
            phase = (phase + e - s) % period
            s = e
    for a, b in segs:
        cv2.line(vis, _i(a), _i(b), EDGE, thick + 2, cv2.LINE_AA)
    for a, b in segs:
        cv2.line(vis, _i(a), _i(b), colour, thick, cv2.LINE_AA)


def cross(vis: np.ndarray, pt: Sequence[float], r: int = 16, colour=GUESS, thick: int = 2) -> None:
    """A cross with a dark edge: the centre of a guess. It is open in the middle and wider than
    the fingertip dot, so the amber dot under it still shows (rahul 6 Oct: a filled mark would
    hide where he pointed)."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    arms = (((x - r, y), (x + r, y)), ((x, y - r), (x, y + r)))
    for a, b in arms:
        cv2.line(vis, a, b, EDGE, thick + 2, cv2.LINE_AA)
    for a, b in arms:
        cv2.line(vis, a, b, colour, thick, cv2.LINE_AA)


def fill(vis: np.ndarray, pts: np.ndarray, colour=AMBER, alpha: float = 0.35) -> np.ndarray:
    """A see-through filled polygon with a dark edge (your hold bars, your box)."""
    poly = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 2)
    over = vis.copy()
    cv2.fillPoly(over, [poly], colour, cv2.LINE_AA)
    out = cv2.addWeighted(over, alpha, vis, 1 - alpha, 0)
    cv2.polylines(out, [poly], True, EDGE, 2, cv2.LINE_AA)
    return out


def status(
    vis: np.ndarray, text: str, accent=None, said: str | None = None, scale: float = 1.4
) -> int:
    """The status in large plain words on a dark bar at the top, a colour stripe at its left
    (red for a problem or a stop), and the robot's last words under it. Returns the y under it."""
    th = max(2, round(2 * scale))
    (_, h), _ = cv2.getTextSize(text, FONT, scale, th)
    bar = h + 32
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar), BAR, -1)
    if accent is not None:
        cv2.rectangle(vis, (0, 0), (14, bar), accent, -1)
    cv2.putText(vis, text, (26, bar - 16), FONT, scale, WHITE, th, cv2.LINE_AA)
    y = bar
    if said:
        s2 = scale * 0.7
        line = f"deixis: {said}"
        (_, h2), _ = cv2.getTextSize(line, FONT, s2, th)
        cv2.rectangle(vis, (0, y), (vis.shape[1], y + h2 + 22), (55, 55, 55), -1)
        cv2.putText(vis, line, (26, y + h2 + 10), FONT, s2, WHITE, th, cv2.LINE_AA)
        y += h2 + 22
    return y


def _i(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))
