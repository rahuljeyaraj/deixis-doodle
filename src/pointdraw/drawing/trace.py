"""Centre-line tracing: black line art on white -> pen strokes in library units.

Ordinary tracers follow the *edges* of the ink, so every thick line comes out as two parallel
pen lines. This one follows the *centre* of each line:

1. threshold (Otsu, after a light blur), drop specks, fill pinholes in the ink
2. thin the ink to a one-pixel skeleton (OpenCV contrib, Zhang-Suen)
3. turn the skeleton into a graph: end points, junctions, and the pixel paths between them
4. prune short dangling spurs (thinning noise), merge junctions joined by a tiny bridge
5. at each junction, pair the branches that continue most straight, so a line that crosses
   another stays one stroke; walk the pairs into strokes
6. smooth along each stroke, simplify (RDP), join stroke ends that nearly touch

`trace` returns strokes in image pixels (x right, y down); `to_library` turns them into the
library format (y up, bbox min corner at (0, 0), larger side 255). The second half of the file
has the ground-truth helpers used by `scripts/trace_library.py --eval` and the tests: render
strokes as thick lines, add blur and noise, and measure what came back.

Needs numpy, OpenCV (opencv-contrib-python, for `cv2.ximgproc.thinning`) and, for the
evaluation helpers only, scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from deixis.geometry import Point, Polyline
from pointdraw.drawing.optimize import merge, path_length, simplify

LIBRARY_SIDE = 255.0  # larger bbox side of a library drawing, in library units

if not hasattr(cv2, "ximgproc"):  # pragma: no cover - depends on the installed wheel
    raise ImportError(
        "pointdraw.drawing.trace needs cv2.ximgproc: install opencv-contrib-python "
        "(uv sync --extra drawing), not opencv-python"
    )


@dataclass(frozen=True)
class TraceParams:
    """Tracing knobs. Most are multiples of the measured line width w (pixels)."""

    speck_area_w2: float = 1.5  # drop ink blobs smaller than this times w squared
    speck_side_frac: float = 0.02  # ... or under 2 % of the largest blob's size
    hole_area_w2: float = 0.3  # fill paper holes inside the ink smaller than this times w^2
    spur_w: float = 1.0  # prune dangling branches shorter than this times w (plus 2 px)
    bridge_w: float = 0.75  # merge two junctions joined by a path shorter than this times w
    cross_w: float = 8.0  # ... or two 3-way junctions joined by a straight path this short
    cusp_w: float = 8.0  # retrace the stem of a sharp V if it is shorter than this times w
    min_stroke_w: float = 0.5  # drop strokes shorter than this times w (single dots are kept)
    smooth_px: float = 1.5  # Gaussian sigma along the stroke, pixels
    simplify_px: float = 0.5  # RDP tolerance, pixels
    join_w: float = 1.0  # join stroke ends closer than this times w
    thinning: str = "zhangsuen"  # or "guohall"


DEFAULT_PARAMS = TraceParams()


@dataclass
class TraceResult:
    """Strokes in image pixels (x right, y down, pixel centres at integers) plus numbers
    that help judge the input: `max_width_px` far above `line_width_px` means filled areas."""

    strokes: list[Polyline]
    line_width_px: float
    image_size: tuple[int, int]  # (width, height)
    ink_px: int
    max_width_px: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Image to mask
# ---------------------------------------------------------------------------------------------


def load_gray(src: str | Path | np.ndarray) -> np.ndarray:
    """A uint8 grey image from a path or an array (grey, RGB or RGBA; uint8 or 0..1 floats).
    Transparent pixels count as white paper."""
    if isinstance(src, (str, Path)):
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image {src}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(src)
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    return img


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Dark ink as True. Otsu after a light blur; an image with no contrast has no ink.
    Light lines on a dark page are inverted first (the page is whatever covers most of it)."""
    if int(gray.max()) - int(gray.min()) < 16:
        return np.zeros(gray.shape, bool)
    if np.median(gray) < 128:
        gray = 255 - gray
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


def _drop_small(mask: np.ndarray, min_area: float, min_side_frac: float) -> np.ndarray:
    """Keep ink blobs with at least `min_area` pixels whose larger bbox side is at least
    `min_side_frac` of the largest blob's."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    area = st[:, cv2.CC_STAT_AREA]
    side = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    main = 1 + int(np.argmax(area[1:]))
    keep = (area >= min_area) & (side >= min_side_frac * side[main])
    keep[0] = False
    return keep[lab]


def clean_mask(mask: np.ndarray, width_px: float, params: TraceParams = DEFAULT_PARAMS):
    """Drop ink specks and fill pinholes. A speck is small against the line width (area) or
    against the drawing (size): thin lines make the area test alone too weak."""
    w2 = width_px * width_px
    out = _drop_small(mask, max(4.0, params.speck_area_w2 * w2), params.speck_side_frac)
    holes = ~out
    # background components touching the border are paper, never holes
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    small = st[:, cv2.CC_STAT_AREA] < max(3.0, params.hole_area_w2 * w2)
    small[0] = False
    fill = small[lab] & holes
    return out | fill


def skeleton(mask: np.ndarray, method: str = "zhangsuen") -> np.ndarray:
    """One-pixel-wide centre line of the mask (True = skeleton)."""
    kind = {
        "zhangsuen": cv2.ximgproc.THINNING_ZHANGSUEN,
        "guohall": cv2.ximgproc.THINNING_GUOHALL,
    }[method]
    return cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=kind) > 0


def _distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def line_width(dist: np.ndarray, skel: np.ndarray) -> float:
    """Typical width of the thresholded lines in pixels, from the median distance-to-paper
    on the centre line (the centre pixel of a w-pixel line is (w + 1) / 2 from the paper).

    Anti-aliased edges make the thresholded line up to a pixel wider than the drawn one."""
    d = dist[skel]
    if d.size == 0:
        return 1.0
    return max(1.0, 2.0 * float(np.median(d)) - 1.0)


# ---------------------------------------------------------------------------------------------
# Skeleton to graph
# ---------------------------------------------------------------------------------------------

_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _pixel_graph(skel: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Skeleton pixel positions (N x 2, as x, y) and each pixel's neighbour list.

    A diagonal link is dropped when a shared 4-neighbour already connects the two pixels, so
    staircase corners in the skeleton don't look like junctions."""
    h, w = skel.shape
    pad = np.zeros((h + 2, w + 2), bool)
    pad[1:-1, 1:-1] = skel
    ys, xs = np.nonzero(pad)
    idx = np.full(pad.shape, -1, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    cols = []
    for dy, dx in _N8:
        j = idx[ys + dy, xs + dx]
        if dy and dx:
            j = np.where(pad[ys + dy, xs] | pad[ys, xs + dx], -1, j)
        cols.append(j)
    nb = np.stack(cols, axis=1) if cols else np.zeros((0, 8), np.int64)
    adj = [row[row >= 0].tolist() for row in nb]
    pts = np.stack([xs - 1, ys - 1], axis=1).astype(float)
    return pts, adj


@dataclass
class _Edge:
    u: int  # node ids (after merging, look them up through _Graph.find)
    v: int
    pts: np.ndarray  # path from u to v, pixel coordinates

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.pts, axis=0).T))) if len(self.pts) > 1 else 0


@dataclass
class _Graph:
    pos: list[np.ndarray]  # node positions
    edges: dict[int, _Edge]
    cycles: list[np.ndarray]  # closed pixel loops with no nodes on them
    dots: list[np.ndarray]  # isolated skeleton pixels
    parent: list[int]  # union-find over nodes (junctions merged by a short bridge)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def ends(self) -> dict[int, list[tuple[int, int]]]:
        """For each live node, the (edge id, end) pairs touching it; end 0 = u, 1 = v."""
        out: dict[int, list[tuple[int, int]]] = {}
        for k, e in self.edges.items():
            out.setdefault(self.find(e.u), []).append((k, 0))
            out.setdefault(self.find(e.v), []).append((k, 1))
        return out


def _build_graph(skel: np.ndarray) -> _Graph:
    pts, adj = _pixel_graph(skel)
    n = len(pts)
    deg = np.array([len(a) for a in adj], int)
    node_of = np.full(n, -1, int)
    pos: list[np.ndarray] = []
    # junction pixels that touch each other form one node; end points are nodes of their own
    for p in range(n):
        if deg[p] == 2 or node_of[p] >= 0 or deg[p] == 0:
            continue
        nid = len(pos)
        if deg[p] == 1:
            node_of[p] = nid
            pos.append(pts[p].copy())
            continue
        stack, members = [p], []
        node_of[p] = nid
        while stack:
            q = stack.pop()
            members.append(q)
            for r in adj[q]:
                if deg[r] >= 3 and node_of[r] < 0:
                    node_of[r] = nid
                    stack.append(r)
        pos.append(pts[members].mean(axis=0))
    edges: dict[int, _Edge] = {}
    seen: set[tuple[int, int]] = set()
    for p in range(n):
        if node_of[p] < 0:
            continue
        for q in adj[p]:
            if node_of[q] >= 0 and node_of[q] == node_of[p]:
                continue  # inside one junction
            if (p, q) in seen:
                continue
            path, prev, cur = [p, q], p, q
            seen.add((p, q))
            while node_of[cur] < 0:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                seen.add((cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            seen.add((cur, prev))
            u, v = int(node_of[p]), int(node_of[cur])
            xy = pts[path].copy()
            xy[0], xy[-1] = pos[u], pos[v]
            edges[len(edges)] = _Edge(u, v, xy)
    # loops of plain pixels (a ring, a closed outline) have no node on them
    on_edge = np.zeros(n, bool)
    for a, b in seen:
        on_edge[a] = on_edge[b] = True
    cycles = []
    for p in range(n):
        if deg[p] != 2 or on_edge[p]:
            continue
        path, prev, cur = [p], -1, p
        while True:
            on_edge[cur] = True
            nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
            if nxt == p:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        cycles.append(np.vstack([pts[path], pts[path[:1]]]))
    dots = [pts[p] for p in range(n) if deg[p] == 0]
    return _Graph(pos, edges, cycles, dots, list(range(len(pos))))


def _prune_spurs(g: _Graph, max_len: float, passes: int = 3) -> None:
    """Remove short branches that end in nothing (thinning noise at corners and bumps).

    At a junction, every short dangling branch goes as long as one other branch is left; if all
    of them are short (a tiny star), the longest stays."""
    for _ in range(passes):
        ends = g.ends()
        degree = {k: len(v) for k, v in ends.items()}
        drop: set[int] = set()
        for node, items in ends.items():
            if degree[node] < 3:
                continue
            spurs, keep = [], []
            for k, end in items:
                e = g.edges[k]
                other = g.find(e.v if end == 0 else e.u)
                if other != node and degree.get(other, 0) == 1 and e.length < max_len:
                    spurs.append(k)
                else:
                    keep.append(k)
            if not spurs:
                continue
            if not keep:
                spurs.remove(max(spurs, key=lambda k: g.edges[k].length))
            drop.update(spurs)
        if not drop:
            return
        for k in drop:
            del g.edges[k]


def _bend(pts: np.ndarray) -> float:
    """Largest distance of a path from the straight line between its ends."""
    a, b = pts[0], pts[-1]
    d = b - a
    n = math.hypot(*d)
    if n == 0:
        return float(np.max(np.hypot(*(pts - a).T)))
    return float(np.max(np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / n))


def _merge_bridges(g: _Graph, max_len: float, max_len_x: float, max_bend: float) -> None:
    """Merge two junctions into one crossing when the path between them is short.

    Any two junctions closer than `max_len`. Two three-way junctions up to `max_len_x` apart
    along a nearly straight path: that is how thinning draws two lines crossing at a shallow
    angle (an X becomes two Ys joined by a bridge). Merging makes the X a four-way node, where
    each line goes straight on. If the pair was really two T-joins, the pen just retraces the
    short bridge, which is drawn anyway, and saves a lift."""
    ends = g.ends()
    degree = {k: len(v) for k, v in ends.items()}
    for k in sorted(g.edges, key=lambda k: g.edges[k].length):
        e = g.edges[k]
        a, b = g.find(e.u), g.find(e.v)
        if a == b or degree.get(a, 0) < 3 or degree.get(b, 0) < 3:
            continue
        short = e.length < max_len
        crossing = (
            e.length < max_len_x and degree[a] == 3 and degree[b] == 3 and _bend(e.pts) <= max_bend
        )
        if not (short or crossing):
            continue
        g.pos[a] = (g.pos[a] + g.pos[b]) / 2
        g.parent[b] = a
        degree[a] = degree[a] + degree[b] - 2
        del g.edges[k]
    for e in g.edges.values():  # paths now start and end at the merged positions
        e.pts[0] = g.pos[g.find(e.u)]
        e.pts[-1] = g.pos[g.find(e.v)]


def _fold_cusps(g: _Graph, max_len: float, reach: float) -> None:
    """Turn the stem of a sharp V into an out-and-back loop, so the V stays one stroke.

    Where a thick line folds at an acute angle, the two arms merge into one stem before the
    tip, and the skeleton becomes a Y: two arms, a junction, and a short stem to the tip.
    Pairing the straightest branches would break the stroke there. Instead the pen goes down
    the stem to the tip and back up it (retracing its own line), then on along the other arm."""
    ends = g.ends()
    for node, items in list(ends.items()):
        if len(items) != 3:
            continue
        stems = []
        for k, end in items:
            e = g.edges[k]
            other = g.find(e.v if end == 0 else e.u)
            if other != node and len(ends.get(other, [])) == 1 and e.length <= max_len:
                stems.append((k, end))
        if len(stems) != 1:
            continue
        k, end = stems[0]
        arms = [it for it in items if it != (k, end)]
        d = []
        for kk, ee in [(k, end), *arms]:
            p = g.edges[kk].pts
            d.append(_direction(p if ee == 0 else p[::-1], reach))
        acute = float(np.dot(d[1], d[2])) > 0.0  # the arms leave less than 90 degrees apart
        away = float(np.dot(d[0], d[1])) < 0.0 and float(np.dot(d[0], d[2])) < 0.0
        if acute and away:
            e = g.edges[k]
            p = e.pts if end == 0 else e.pts[::-1]
            g.edges[k] = _Edge(node, node, np.vstack([p, p[-2::-1]]))


def _direction(pts: np.ndarray, reach: float) -> np.ndarray:
    """Unit vector from pts[0] towards the path point `reach` pixels along it.

    After two junctions were merged, a path starts with a jump from the merged node to its
    old junction; the branch's own direction is measured from the old junction instead."""
    if len(pts) > 2 and math.dist(pts[0], pts[1]) > 2.0:
        pts = pts[1:]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = min(reach, s[-1])
    i = int(np.searchsorted(s, target))
    p = pts[min(i, len(pts) - 1)]
    d = p - pts[0]
    n = math.hypot(*d)
    return d / n if n > 0 else np.zeros(2)


def _walk_strokes(g: _Graph, reach: float) -> list[np.ndarray]:
    """Pair branches at each node (straightest first) and walk the pairs into strokes."""
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for items in g.ends().values():
        if len(items) < 2:
            continue
        dirs = []
        for k, end in items:
            p = g.edges[k].pts
            dirs.append(_direction(p if end == 0 else p[::-1], reach))
        pairs = sorted(
            (
                (float(np.dot(dirs[i], dirs[j])), i, j)
                for i in range(len(items))
                for j in range(i + 1, len(items))
            ),
        )  # most negative dot first = most nearly straight on
        used: set[int] = set()
        for _, i, j in pairs:
            if i in used or j in used:
                continue
            used.update((i, j))
            partner[items[i]] = items[j]
            partner[items[j]] = items[i]

    done: set[int] = set()

    def walk(k: int, start: int) -> np.ndarray:
        parts = []
        while True:
            done.add(k)
            p = g.edges[k].pts
            parts.append(p if start == 0 else p[::-1])
            nxt = partner.get((k, 1 - start))
            if nxt is None or nxt[0] in done:
                break
            k, start = nxt
        return np.vstack([parts[0]] + [q[1:] for q in parts[1:]])

    strokes = []
    for k in list(g.edges):  # start at free ends first
        for end in (0, 1):
            if k not in done and (k, end) not in partner:
                strokes.append(walk(k, end))
    for k in list(g.edges):  # what is left is made of closed loops
        if k not in done:
            strokes.append(walk(k, 0))
    return strokes


# ---------------------------------------------------------------------------------------------
# Stroke clean-up
# ---------------------------------------------------------------------------------------------


def resample(pts: np.ndarray | Sequence[Point], step: float) -> np.ndarray:
    """Points every `step` along the polyline (both ends kept)."""
    a = np.asarray(pts, float)
    if len(a) < 2:
        return a.copy()
    seg = np.hypot(*np.diff(a, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return a[:1].copy()
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, a[:, 0]), np.interp(t, s, a[:, 1])], axis=1)


def _smooth(a: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    """Gaussian smoothing along a 1-px-resampled path. Open paths keep their end points."""
    if sigma <= 0 or len(a) < 5:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if closed:
        ring = a[:-1]
        padded = np.concatenate([ring[-r:], ring, ring[:r]]) if len(ring) > r else None
        if padded is None:
            return a
        out = np.stack([np.convolve(padded[:, i], k, "valid") for i in (0, 1)], axis=1)
        return np.vstack([out, out[:1]])
    padded = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    out = np.stack([np.convolve(padded[:, i], k, "valid") for i in (0, 1)], axis=1)
    out[0], out[-1] = a[0], a[-1]
    return out


def _as_polyline(a: np.ndarray) -> Polyline:
    return [(float(x), float(y)) for x, y in a]


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------


def _crop(mask: np.ndarray, pad: int = 4) -> tuple[np.ndarray, int, int]:
    """The mask cut to its ink's bounding box plus `pad`, and the cut's top-left corner."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    return mask[y0:y1, x0:x1], x0, y0


def trace(src: str | Path | np.ndarray, params: TraceParams = DEFAULT_PARAMS) -> TraceResult:
    """Centre-line strokes of black line art on white, in image pixels (y down)."""
    gray = load_gray(src)
    h, w = gray.shape
    mask = ink_mask(gray)
    notes: list[str] = []
    if not mask.any():
        return TraceResult([], 0.0, (w, h), 0, 0.0, ["no ink found"])
    # work on the ink's bounding box only (thinning a full 1024 x 1024 page is slow)
    crop, x0, y0 = _crop(mask)
    # first pass only to measure the line width, which scales everything else
    width = line_width(_distance(crop), skeleton(crop, params.thinning))
    crop = clean_mask(crop, width, params)
    if not crop.any():
        return TraceResult([], width, (w, h), 0, 0.0, ["only specks found"])
    crop, dx, dy = _crop(crop)  # smaller now that stray specks are gone
    x0, y0 = x0 + dx, y0 + dy
    dist = _distance(crop)
    skel = skeleton(crop, params.thinning)
    width = line_width(dist, skel)
    max_width = 2.0 * float(dist.max()) - 1.0
    if max_width > 3.0 * width:
        notes.append(f"filled areas: widest ink {max_width:.0f} px vs lines {width:.0f} px")

    g = _build_graph(skel)
    _prune_spurs(g, params.spur_w * width + 2.0)
    _merge_bridges(g, params.bridge_w * width, params.cross_w * width, 0.5 * width)
    reach = max(4.0, 2.0 * width)
    _fold_cusps(g, params.cusp_w * width, reach)
    raw = _walk_strokes(g, reach)
    raw += g.cycles

    strokes: list[Polyline] = []
    offset = np.array([x0, y0], float)
    for a in raw:
        closed = len(a) > 3 and np.allclose(a[0], a[-1])
        r = resample(a, 1.0)
        if closed and len(r) > 3:
            r[-1] = r[0]
        s = _smooth(r, params.smooth_px, closed)
        strokes.append(_as_polyline(s + offset))
    strokes = simplify(strokes, params.simplify_px)
    strokes = merge(strokes, tol=params.join_w * width)
    strokes = [s for s in strokes if path_length(s) >= params.min_stroke_w * width]
    strokes += [[(float(p[0] + x0), float(p[1] + y0))] for p in g.dots]
    return TraceResult(strokes, width, (w, h), int(crop.sum()), max_width, notes)


def to_library(strokes: Sequence[Sequence[Point]], side: float = LIBRARY_SIDE) -> list[Polyline]:
    """Image-pixel strokes (y down) to library units: y up, bbox min corner at (0, 0), the
    larger side `side` long."""
    pts = [p for s in strokes for p in s]
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [-p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0)
    k = side / span if span > 0 else 1.0
    return [[((x - x0) * k, (-y - y0) * k) for x, y in s] for s in strokes]


def trace_to_library(
    src: str | Path | np.ndarray, params: TraceParams = DEFAULT_PARAMS
) -> list[Polyline]:
    """`trace`, then `to_library`."""
    return to_library(trace(src, params).strokes)


# ---------------------------------------------------------------------------------------------
# Ground truth: render known strokes, trace them back, measure (scripts/trace_library.py, tests)
# ---------------------------------------------------------------------------------------------


def render_strokes(
    strokes: Sequence[Sequence[Point]],
    size_px: int = 1024,
    line_px: float = 8.0,
    margin_px: float = 64.0,
    y_up: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Thick black lines (round caps and joins) on white, `line_px` wide to about 0.25 px.

    Returns the grey image, the strokes in image pixels (x right, y down) and the scale in
    pixels per stroke unit. The drawing's larger side spans size_px - 2 * margin_px.
    Drawn 4x larger and averaged down: OpenCV only draws odd widths (thickness t comes out
    t + 1 or t + 2 px wide), which would make every requested width wrong by a pixel."""
    ss = 4
    pts = np.asarray([p for s in strokes for p in s], float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    k = (size_px - 2 * margin_px) / max(x1 - x0, y1 - y0, 1e-9)
    ox = (size_px - (x1 - x0) * k) / 2
    oy = (size_px - (y1 - y0) * k) / 2
    big = np.full((size_px * ss, size_px * ss), 255, np.uint8)
    want = max(1, int(round(line_px * ss)))
    drawn = {t: t + 1 if t % 2 == 0 else t + 2 for t in range(max(1, want - 2), want + 1)}
    thick = min(drawn, key=lambda t: abs(drawn[t] - want))  # cv2 thickness t is drawn[t] wide
    out = []
    shift = 4
    for s in strokes:
        a = np.asarray(s, float)
        px = np.empty_like(a)
        px[:, 0] = ox + (a[:, 0] - x0) * k
        px[:, 1] = (size_px - oy - (a[:, 1] - y0) * k) if y_up else (oy + (a[:, 1] - y0) * k)
        out.append(px)
        # output pixel i averages big pixels ss*i .. ss*i + ss - 1, centred on ss*i + 1.5
        fixed = np.round((px * ss + (ss - 1) / 2) * (1 << shift)).astype(np.int32)
        fixed = fixed.reshape(-1, 1, 2)
        if len(px) == 1:
            c = (int(fixed[0, 0, 0]), int(fixed[0, 0, 1]))
            cv2.circle(big, c, (want << shift) // 2, 0, -1, cv2.LINE_AA, shift)
        else:
            cv2.polylines(big, [fixed], False, 0, thick, cv2.LINE_AA, shift)
    img = cv2.resize(big, (size_px, size_px), interpolation=cv2.INTER_AREA)
    return img, out, k


def degrade(
    img: np.ndarray,
    blur_sigma: float = 0.8,
    noise_sigma: float = 6.0,
    specks: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Mild blur, Gaussian grain and a few random dark specks (1 to 3 px), like a scan or a
    generated image."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    out += rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    h, w = out.shape
    for _ in range(specks):
        r = int(rng.integers(1, 3))
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), r, 40.0, -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _samples(strokes: Sequence[Sequence[Point]], step: float) -> np.ndarray:
    parts = [resample(s, step) for s in strokes if len(s) > 0]
    return np.vstack(parts) if parts else np.zeros((0, 2))


def compare(
    traced: Sequence[Sequence[Point]],
    original: Sequence[Sequence[Point]],
    px_per_mm: float,
    step_px: float = 0.5,
) -> dict[str, float]:
    """Distances between two stroke sets in the same pixel frame, reported in mm.

    mean_mm: the average of the two directed mean distances (traced -> original and back).
    p95_mm: the larger of the two directed 95th percentiles. max_mm likewise."""
    from scipy.spatial import cKDTree

    a, b = _samples(traced, step_px), _samples(original, step_px)
    if len(a) == 0 or len(b) == 0:
        return {"mean_mm": math.inf, "p95_mm": math.inf, "max_mm": math.inf}
    dab = cKDTree(b).query(a)[0] / px_per_mm
    dba = cKDTree(a).query(b)[0] / px_per_mm
    return {
        "mean_mm": float((dab.mean() + dba.mean()) / 2),
        "p95_mm": float(max(np.percentile(dab, 95), np.percentile(dba, 95))),
        "max_mm": float(max(dab.max(), dba.max())),
    }


def double_line_score(
    strokes: Sequence[Sequence[Point]],
    window_px: float,
    step_px: float = 1.0,
    max_angle_deg: float = 30.0,
    min_gap_px: float = 0.0,
) -> float:
    """Fraction of the ink that has *other* ink running parallel to it within `window_px`.

    Edge tracing gives about 1.0 (each line drawn twice, one line width apart); clean centre
    lines give about 0. "Other" means another stroke, or the same stroke more than three
    windows away along it, so a line's own neighbouring points never count. Ink closer than
    min_gap_px lies on top (a deliberate retrace) and doesn't count either."""
    from scipy.spatial import cKDTree

    pts, tan, sid, arc, loop_len = [], [], [], [], []
    for i, s in enumerate(strokes):
        r = resample(s, step_px)
        if len(r) < 2:
            continue
        t = np.gradient(r, axis=0)
        t /= np.maximum(np.hypot(*t.T), 1e-12)[:, None]
        closed = len(s) > 2 and math.dist(s[0], s[-1]) < 1e-6
        length = path_length(r)
        pts.append(r)
        tan.append(t)
        sid.append(np.full(len(r), i))
        arc.append(np.linspace(0, length, len(r)))
        loop_len.append(np.full(len(r), length if closed else np.inf))
    if not pts:
        return 0.0
    p, t = np.vstack(pts), np.vstack(tan)
    s_id, s_arc, s_loop = np.concatenate(sid), np.concatenate(arc), np.concatenate(loop_len)
    pairs = cKDTree(p).query_pairs(window_px, output_type="ndarray")
    if len(pairs) == 0:
        return 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    along = np.abs(s_arc[i] - s_arc[j])
    along = np.minimum(along, s_loop[i] - along)
    other = (s_id[i] != s_id[j]) | (along > 3 * window_px)
    other &= np.hypot(*(p[i] - p[j]).T) >= min_gap_px
    parallel = np.abs(np.sum(t[i] * t[j], axis=1)) > math.cos(math.radians(max_angle_deg))
    hit = other & parallel
    doubled = np.zeros(len(p), bool)
    doubled[i[hit]] = True
    doubled[j[hit]] = True
    return float(doubled.mean())
