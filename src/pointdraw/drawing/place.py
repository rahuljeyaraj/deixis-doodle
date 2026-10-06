"""Fit drawings into millimetre circles and boxes: scale, centre, smooth, keep inside the paper.

A requested size is the **diameter of a circle** round the requested point (rahul, 6 Oct 2026:
"this big" is shown across, from one side to the other, and the plotter traces that circle
in the air before it draws). The drawing's box is centred on the point, the aspect ratio is
kept, and the drawing grows until its farthest line touches the circle. A square's corners
touch it; a circle fills it. (Until 6 Oct the size set the larger side of the box.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from deixis.geometry import Box, Point, Polyline

CLOSED_TOL = 1e-6  # relative to the drawing's size: first == last means a closed stroke


def bbox(polylines: Sequence[Sequence[Point]]) -> Box:
    """Bounding box of all points. Raises ValueError if there are none."""
    xs = [p[0] for pl in polylines for p in pl]
    ys = [p[1] for pl in polylines for p in pl]
    if not xs:
        raise ValueError("no points")
    return Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def transform(
    polylines: Sequence[Sequence[Point]], scale: float = 1.0, dx: float = 0.0, dy: float = 0.0
) -> list[Polyline]:
    """Scale about the origin, then translate."""
    return [[(x * scale + dx, y * scale + dy) for x, y in pl] for pl in polylines]


def _turn_deg(a: Point, b: Point, c: Point) -> float:
    """How far the path turns at b, in degrees (0 = straight on)."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu == 0 or nv == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(cos))


def _chaikin_open(pts: list[Point], iterations: int) -> list[Point]:
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        out.append(pts[-1])
        pts = out
    return pts


def _chaikin_ring(pts: list[Point], iterations: int) -> list[Point]:
    """pts is closed (first == last)."""
    ring = pts[:-1]
    for _ in range(iterations):
        out = []
        for i, (x0, y0) in enumerate(ring):
            x1, y1 = ring[(i + 1) % len(ring)]
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        ring = out
    return ring + [ring[0]]


def chaikin(pl: Sequence[Point], iterations: int = 2, corner_deg: float = 70.0) -> Polyline:
    """Chaikin corner cutting that keeps endpoints and **sharp corners** (turns > corner_deg).

    Doodles are simplified to a few points, so plain Chaikin rounds a star or a roof into a
    blob. Splitting at sharp corners smooths the wobbly curves and keeps the deliberate ones.
    """
    pts = [(float(x), float(y)) for x, y in pl]
    if iterations <= 0 or len(pts) < 3:
        return pts
    size = max(bbox([pts]).w, bbox([pts]).h, 1e-9)
    closed = math.dist(pts[0], pts[-1]) <= CLOSED_TOL * size and len(pts) > 3
    if closed:
        ring = pts[:-1]
        n = len(ring)
        sharp = [
            i for i in range(n) if _turn_deg(ring[i - 1], ring[i], ring[(i + 1) % n]) > corner_deg
        ]
        if not sharp:
            return _chaikin_ring(pts, iterations)
        k = sharp[0]  # start the loop at a sharp corner, then treat it as open
        pts = ring[k:] + ring[:k] + [ring[k]]
    corners = [0] + [
        i for i in range(1, len(pts) - 1) if _turn_deg(pts[i - 1], pts[i], pts[i + 1]) > corner_deg
    ] + [len(pts) - 1]  # fmt: skip
    out: Polyline = [pts[0]]
    for a, b in zip(corners, corners[1:], strict=False):
        out.extend(_chaikin_open(pts[a : b + 1], iterations)[1:])
    return out


def smooth(
    polylines: Sequence[Sequence[Point]], iterations: int = 2, corner_deg: float = 70.0
) -> list[Polyline]:
    return [chaikin(pl, iterations, corner_deg) for pl in polylines]


def reach(polylines: Sequence[Sequence[Point]], center: Point) -> float:
    """How far the farthest point is from `center` (0 if there are none). Lines are straight
    between points, so the farthest point of the drawing is always one of them."""
    return max((math.dist(center, p) for pl in polylines for p in pl), default=0.0)


def circle_size(polylines: Sequence[Sequence[Point]]) -> float:
    """The diameter of the circle round the drawing's box centre that its farthest line touches:
    the size `fit` gave it."""
    return 2 * reach(polylines, bbox(polylines).center)


def fit(
    polylines: Sequence[Sequence[Point]],
    center: Point,
    size: float,
    smooth_iterations: int = 2,
) -> list[Polyline]:
    """Centre the bbox on `center` and scale until the farthest line touches the circle of
    diameter `size` (mm) round it.

    Smoothing runs before scaling, so the result's size is exact.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    src = smooth(polylines, smooth_iterations)
    cx, cy = bbox(src).center
    r = reach(src, (cx, cy))
    s = size / 2 / r if r > 0 else 1.0
    return transform(src, s, center[0] - cx * s, center[1] - cy * s)


def fit_in_box(
    polylines: Sequence[Sequence[Point]], box: Box, smooth_iterations: int = 2
) -> list[Polyline]:
    """The largest copy that fits inside `box`, aspect kept, centred in it."""
    src = smooth(polylines, smooth_iterations)
    b = bbox(src)
    s = min(box.w / b.w if b.w > 0 else math.inf, box.h / b.h if b.h > 0 else math.inf)
    if not math.isfinite(s):
        s = 1.0
    cx, cy = b.center
    bx, by = box.center
    return transform(src, s, bx - cx * s, by - cy * s)


def keep_inside(
    polylines: Sequence[Sequence[Point]], area: Box, margin: float = 0.0
) -> tuple[list[Polyline], bool]:
    """Shift the drawing the least distance that keeps it inside `area` minus `margin`.

    If it is bigger than the area, it is also shrunk about its centre. Returns the new
    polylines and whether anything changed, so the caller can say "I moved it a little".
    """
    inner = area.inset(margin)
    b = bbox(polylines)
    out = [list(pl) for pl in polylines]
    changed = False
    s = min(1.0, inner.w / b.w if b.w > 0 else 1.0, inner.h / b.h if b.h > 0 else 1.0)
    if s < 1.0:
        cx, cy = b.center
        out = transform(out, s, cx - cx * s, cy - cy * s)
        b = bbox(out)
        changed = True
    dx = max(0.0, inner.x - b.x) - max(0.0, b.x2 - inner.x2)
    dy = max(0.0, inner.y - b.y) - max(0.0, b.y2 - inner.y2)
    if dx or dy:
        out = transform(out, 1.0, dx, dy)
        changed = True
    return out, changed


def _clip_segment(a: Point, b: Point, box: Box) -> tuple[Point, Point] | None:
    """The part of segment a-b inside `box` (Liang-Barsky), or None if none is."""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in (
        (-dx, a[0] - box.x),
        (dx, box.x2 - a[0]),
        (-dy, a[1] - box.y),
        (dy, box.y2 - a[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


def clip_inside(polylines: Sequence[Sequence[Point]], area: Box) -> tuple[list[Polyline], bool]:
    """Cut away every part of the drawing outside `area`; the rest stays where it is.

    rahul, 2 Oct 2026: never move or shrink a drawing to make it fit ("if you move it, what
    was the point in asking to draw here"); a part that would go off the paper is left out.
    A polyline that leaves the area and comes back is split in two. Returns the polylines
    left (maybe none) and whether anything was cut.
    """
    out: list[Polyline] = []
    cut = False
    for pl in polylines:
        if len(pl) == 1:
            if area.contains(pl[0]):
                out.append([pl[0]])
            else:
                cut = True
            continue
        cur: Polyline = []
        for a, b in zip(pl, pl[1:], strict=False):
            seg = _clip_segment(a, b, area)
            if seg is None:
                cut = True
                if cur:
                    out.append(cur)
                    cur = []
                continue
            p, q = seg
            if p != a:
                cut = True
            if not cur or cur[-1] != p:
                if cur:
                    out.append(cur)
                cur = [p]
            cur.append(q)
            if q != b:
                cut = True
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
    return out, cut
