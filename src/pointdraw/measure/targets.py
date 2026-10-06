"""Printed target sheets (M01, M02, M03) and finding them again in a parked camera photo.

A sheet is a 200 x 185 mm piece cut from an A4 print and taped onto a fresh 235 mm square.
Its targets are printed in mid grey (the pen is black, so a drawn line that lands exactly
on a printed one still shows in the before/after difference). Positions are in **sheet mm**:
origin at the piece's bottom-left corner as you see it (the "SOUTH" edge toward you), +x to
the right (east), +y away from you (north), the same way round as the plotter's paper mm.

- **Crosses** (M01, M03): 20 grey "x" marks, 12 mm across, at least 30 mm apart. The drawn
  cross is a "+", so the two never lie on top of each other along a whole arm.
- **Circles** (M02): 10 grey circles of known diameter, a small grey "x" at each centre.

Where the sheet lies on the bed doesn't matter: `find_targets` finds the marks in a top-down
picture of the paper (template match), fits the sheet's layout to them (a turn, a shift and
the printer's scale), and keeps the found position of each mark (the fit only numbers them).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import cv2
import numpy as np

PIECE_W, PIECE_H = 200.0, 185.0  # mm, the cut piece
MARGIN = 15.0  # no cross centre nearer an edge than this
X_HALF = 6.0  # the printed "x": arms 6 mm from the centre along the diagonals
CENTRE_X_HALF = 3.0  # the small "x" at a circle's centre
LINE_MM = 0.5
GREY = 0.55  # printed line grey (0 black, 1 white)
CHI2_95 = 5.991  # 2 degrees of freedom: inside the 95 % ellipse when d' S^-1 d <= this

SHEETS: dict[str, dict] = {
    "crosses-A": {"kind": "crosses", "seed": 20261022},  # Thu 22 Oct
    "crosses-B": {"kind": "crosses", "seed": 20261023},  # Fri 23 Oct: 20 new points (M03)
    "circles-1": {"kind": "circles", "rows": [[60, 46, 36], [50, 32, 26, 22], [40, 30, 20]]},
    "circles-2": {"kind": "circles", "rows": [[56, 44, 34], [48, 38, 28, 24], [42, 26, 18]]},
}


@dataclass
class Target:
    id: int
    x: float  # sheet mm (layout) or paper mm (found)
    y: float
    d: float | None = None  # a circle's diameter, mm
    found: bool = True


def layout(name: str) -> list[Target]:
    spec = SHEETS[name]
    if spec["kind"] == "crosses":
        return _crosses(spec["seed"])
    return _circles(spec["rows"])


def _crosses(seed: int, n_cols: int = 5, n_rows: int = 4, jitter: float = 5.0) -> list[Target]:
    rng = random.Random(seed)
    xs = np.linspace(MARGIN + 5, PIECE_W - MARGIN - 5, n_cols)
    ys = np.linspace(PIECE_H - MARGIN - 5, MARGIN + 5, n_rows)  # row 1 is the far row
    out = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            jx = round(rng.uniform(-jitter, jitter) * 2) / 2
            jy = round(rng.uniform(-jitter, jitter) * 2) / 2
            out.append(Target(r * n_cols + c + 1, float(x) + jx, float(y) + jy))
    return out


def _circles(rows: list[list[float]], m: float = 8.0) -> list[Target]:
    heights = [max(r) for r in rows]
    gy = (PIECE_H - 2 * m - sum(heights)) / (len(rows) - 1)
    out: list[Target] = []
    y = PIECE_H - m  # from the far edge down
    for r, h in zip(rows, heights, strict=True):
        gx = (PIECE_W - 2 * m - sum(r)) / (len(r) - 1)
        x = m
        for d in r:
            out.append(Target(len(out) + 1, round(x + d / 2, 2), round(y - h / 2, 2), float(d)))
            x += d + gx
        y -= h + gy
    return out


def min_spacing(ts: list[Target]) -> float:
    return min(math.dist((a.x, a.y), (b.x, b.y)) for i, a in enumerate(ts) for b in ts[i + 1 :])


# ---------------------------------------------------------------- finding them


def x_template(half_mm: float, s: float, line_mm: float = LINE_MM) -> np.ndarray:
    """A dark "x" on white, `s` mm a pixel, blurred a little like a photo."""
    r = int(round((half_mm + 2) / s))
    t = np.full((2 * r + 1, 2 * r + 1), 255, np.uint8)
    a = half_mm / s
    w = max(1, int(round(line_mm / s)))
    for sx in (-1, 1):
        p0 = (int(round(r - a)), int(round(r - sx * a)))
        p1 = (int(round(r + a)), int(round(r + sx * a)))
        cv2.line(t, p0, p1, 0, w, cv2.LINE_AA)
    return cv2.GaussianBlur(t, (0, 0), max(0.5, 0.3 / s))


def candidates(
    top: np.ndarray, s: float, half_mm: float, n: int, thresh: float = 0.35
) -> list[tuple[float, float, float]]:
    """The best `n` "x" marks in a top-down picture (row 0 the far edge): (x, y, score),
    paper mm from the picture's bottom-left corner."""
    g = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top
    g = cv2.GaussianBlur(g, (0, 0), max(0.5, 0.3 / s))
    tpl = x_template(half_mm, s)
    res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
    c = tpl.shape[0] // 2
    sep = int(round(max(8.0, 1.5 * half_mm) / s))
    found = []
    H = g.shape[0]
    for _ in range(n):
        _, v, _, (x, y) = cv2.minMaxLoc(res)
        if v < thresh:
            break
        # sub-pixel: the weighted centre of the score peak
        y0, y1, x0, x1 = max(0, y - 2), y + 3, max(0, x - 2), x + 3
        patch = np.clip(res[y0:y1, x0:x1], 0, None)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = float(patch.sum()) or 1.0
        px, py = float((xx * patch).sum() / w), float((yy * patch).sum() / w)
        found.append(((px + c + 0.5) * s, (H - (py + c + 0.5)) * s, float(v)))
        cv2.circle(res, (x, y), sep, -1.0, -1)
    return found


def fit_layout(
    lay: list[Target], pts: np.ndarray, tol: float = 4.0
) -> tuple[np.ndarray, list[int | None]]:
    """A similarity (turn, shift, scale) taking sheet mm to paper mm, from unnumbered points.
    Tries turns of -20..20 and 160..200 degrees; each layout-point/found-point pair votes for
    a shift. Returns the 2 x 3 matrix and, per layout target, the index of its found point."""
    L = np.array([(t.x, t.y) for t in lay], float)
    P = np.asarray(pts, float).reshape(-1, 2)
    best: tuple[int, np.ndarray | None] = (-1, None)
    for deg in list(range(-20, 21)) + list(range(160, 201)):
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        RL = L @ R.T
        for sh in (P[None, :, :] - RL[:, None, :]).reshape(-1, 2):
            n = _inliers(RL + sh, P, tol)
            if n > best[0]:
                best = (n, np.hstack([R, sh[:, None]]))
    M = best[1]
    assert M is not None
    for _ in range(3):  # refine on the matched pairs
        match = _match(L @ M[:, :2].T + M[:, 2], P, tol)
        src = np.array([L[i] for i, j in enumerate(match) if j is not None])
        dst = np.array([P[j] for j in match if j is not None])
        if len(src) < 3:
            break
        M2, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M2 is None:
            break
        M = M2
    return M, _match(L @ M[:, :2].T + M[:, 2], P, tol)


def _inliers(A: np.ndarray, P: np.ndarray, tol: float) -> int:
    d = np.linalg.norm(A[:, None, :] - P[None, :, :], axis=2)
    return int((d.min(axis=1) < tol).sum())


def _match(A: np.ndarray, P: np.ndarray, tol: float) -> list[int | None]:
    out: list[int | None] = []
    for a in A:
        if not len(P):
            out.append(None)
            continue
        d = np.linalg.norm(P - a, axis=1)
        j = int(np.argmin(d))
        out.append(j if d[j] < tol else None)
    return out


def find_targets(top: np.ndarray, s: float, name: str) -> tuple[list[Target], dict]:
    """The sheet's targets in paper mm, found in `top` (a top-down picture of the paper, `s`
    mm a pixel). Unfound marks get the fitted position and `found=False`."""
    lay = layout(name)
    circles = SHEETS[name]["kind"] == "circles"
    half = CENTRE_X_HALF if circles else X_HALF
    cands = candidates(top, s, half, n=len(lay) * 3)
    P = np.array([(x, y) for x, y, _ in cands], float).reshape(-1, 2)
    if len(P) < 3:
        raise ValueError(f"found only {len(P)} marks: is the sheet in view and lit?")
    M, match = fit_layout(lay, P)
    scale = float(math.hypot(M[0, 0], M[1, 0]))
    turn = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    out = []
    for t, j in zip(lay, match, strict=True):
        fx, fy = M[:, :2] @ (t.x, t.y) + M[:, 2]
        if j is not None:
            fx, fy = P[j]
        d = round(t.d * scale, 2) if t.d else None
        out.append(Target(t.id, round(float(fx), 2), round(float(fy), 2), d, j is not None))
    info = {
        "sheet": name,
        "found": sum(t.found for t in out),
        "of": len(out),
        "scale": round(scale, 4),
        "turn_deg": round(turn, 2),
    }
    return out, info


def inside_95(target, mean, cov) -> bool:
    """Is the target inside the 95 % ellipse of the core's `Where` (mean_mm, cov_mm2)?"""
    d = np.subtract(target, mean)
    try:
        m = float(d @ np.linalg.solve(np.asarray(cov, float), d))
    except np.linalg.LinAlgError:
        return False
    return m <= CHI2_95


def render(name: str, s: float = 0.1) -> np.ndarray:
    """The piece as a top-down picture (white, grey marks), `s` mm a pixel: for tests and the
    preview PNG. Row 0 is the far edge."""
    W, H = int(round(PIECE_W / s)), int(round(PIECE_H / s))
    img = np.full((H, W), 255, np.uint8)
    g = int(round(GREY * 255))
    w = max(1, int(round(LINE_MM / s)))

    def P(x, y):
        return int(round(x / s)), int(round(H - y / s))

    for t in layout(name):
        h = CENTRE_X_HALF if t.d else X_HALF
        for sx in (-1, 1):
            cv2.line(img, P(t.x - h, t.y - sx * h), P(t.x + h, t.y + sx * h), g, w, cv2.LINE_AA)
        if t.d:
            cv2.circle(img, P(t.x, t.y), int(round(t.d / 2 / s)), g, w, cv2.LINE_AA)
    return img


def as_dicts(ts: list[Target]) -> list[dict]:
    return [asdict(t) for t in ts]
