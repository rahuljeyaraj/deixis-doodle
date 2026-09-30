"""The scene model (plan T1.9): what is on the sheet, and places relative to it.

Everything is in the **paper frame** (mm; +x to the user's right, +y away from the user).

- **Robot objects** are known exactly: the planner's polylines and ink box.
- **User marks** come from ink segmentation of the camera's paper image
  (`scene.ink.segment_ink`, with the robot's own strokes subtracted). A vision model
  (Cosmos, or a mock) gives them labels.
- **Relations** (`next_to`, `left_of`, `right_of`, `above`, `below`, `between`,
  `inside`, `on`), **named regions** ("top left corner") and **empty space** are computed
  by geometry here, never by a language model (LR§3.2). Empty space is the centre of the
  largest empty square in the occupancy map of every object's box.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from deixis.geometry import Box, Point, Polyline
from pointdraw.scene.ink import UserMark, distance_to_box

Source = Literal["robot", "user"]


@dataclass
class SceneObject:
    id: str
    source: Source
    label: str | None  # category for robot drawings, a vision label for user marks
    box: Box
    kind: str = "object"  # object | label | arrow | mark
    polylines: list[Polyline] = field(default_factory=list)
    text: str | None = None
    t: float = 0.0
    mark_id: int | None = None
    pending: bool = False  # proposed (being previewed), not yet confirmed
    anchor: str | None = None  # for labels and arrows: the object(s) they belong to

    @property
    def center(self) -> Point:
        return self.box.center


@dataclass(frozen=True)
class SceneParams:
    gap_mm: float = 4.0  # space between an object and one placed next to it
    near_mm: float = 20.0  # "that one": objects within this distance of the point
    ambiguous_mm: float = 6.0  # two objects this close in distance -> ask
    empty_margin_mm: float = 3.0  # keep this far from other ink in empty space
    inside_fill: float = 0.7  # "inside": the new object's larger side / target's smaller side
    mark_match_iou: float = 0.3  # a new mark with this IoU against an old one keeps its id


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: Sequence[Box]) -> Box:
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x2 for b in boxes)
    y1 = max(b.y2 for b in boxes)
    return Box(x0, y0, x1 - x0, y1 - y0)


class SceneModel:
    def __init__(self, drawable: Box, params: SceneParams | None = None) -> None:
        self.drawable = drawable  # paper frame
        self.params = params or SceneParams()
        self.objects: list[SceneObject] = []
        self._n: dict[str, int] = {}

    # ------------------------------------------------------------------ contents

    def new_id(self, base: str) -> str:
        base = base.replace(" ", "_") or "item"
        self._n[base] = self._n.get(base, 0) + 1
        return base if self._n[base] == 1 else f"{base}{self._n[base]}"

    def add(self, obj: SceneObject) -> SceneObject:
        self.objects.append(obj)
        return obj

    def remove(self, obj: SceneObject) -> None:
        if obj in self.objects:
            self.objects.remove(obj)

    def clear(self) -> None:
        self.objects.clear()
        self._n.clear()

    @property
    def robot(self) -> list[SceneObject]:
        return [o for o in self.objects if o.source == "robot"]

    @property
    def marks(self) -> list[SceneObject]:
        return [o for o in self.objects if o.source == "user"]

    def set_user_marks(
        self,
        marks: Sequence[UserMark],
        labeler: Callable[[UserMark], str | None] | None = None,
        t: float = 0.0,
    ) -> list[SceneObject]:
        """Replace the user marks with a fresh segmentation. A mark that overlaps an old one
        keeps its id and label (so Cosmos is only asked about new marks). Returns new ones."""
        old = self.marks
        keep = [o for o in self.objects if o.source != "user"]
        fresh: list[SceneObject] = []
        added: list[SceneObject] = []
        for m in marks:
            match = max(old, key=lambda o: iou(o.box, m.box_mm), default=None)
            if match is not None and iou(match.box, m.box_mm) >= self.params.mark_match_iou:
                match.box, match.mark_id = m.box_mm, m.id
                fresh.append(match)
                continue
            label = labeler(m) if labeler else None
            obj = SceneObject(
                self.new_id(f"mark_{label or 'ink'}"),
                "user",
                label,
                m.box_mm,
                "mark",
                t=t,
                mark_id=m.id,
            )
            fresh.append(obj)
            added.append(obj)
        self.objects = keep + fresh
        return added

    def last(self) -> SceneObject | None:
        """ "It": the most recent drawing (robot object or user mark), not labels or arrows."""
        cands = [o for o in self.objects if o.kind in ("object", "mark")]
        return max(cands, key=lambda o: o.t, default=None)

    def labels(self) -> list[str]:
        """Labels for the agent's context, oldest first, without repeats."""
        out: list[str] = []
        for o in sorted(self.objects, key=lambda o: o.t):
            if o.kind in ("object", "mark") and o.label and o.label not in out:
                out.append(o.label)
        return out

    def find(self, label: str) -> list[SceneObject]:
        """Objects with this label (or text), most recent first."""
        want = _norm(label)
        hits = [
            o
            for o in self.objects
            if o.label
            and (_norm(o.label) == want or _norm(o.label).rstrip("s") == want.rstrip("s"))
            or (o.text and _norm(o.text) == want)
        ]
        return sorted(hits, key=lambda o: o.t, reverse=True)

    def near(self, p: Point, max_mm: float | None = None) -> list[tuple[SceneObject, float]]:
        """Objects near a point (0 = inside the box), nearest first. Labels and arrows skipped."""
        lim = self.params.near_mm if max_mm is None else max_mm
        out = [(o, distance_to_box(p, o.box)) for o in self.objects if o.kind in ("object", "mark")]
        out = [(o, d) for o, d in out if d <= lim]
        return sorted(out, key=lambda od: (od[1], od[0].box.w * od[0].box.h))

    def object_at(self, p: Point) -> tuple[SceneObject | None, list[SceneObject]]:
        """The object pointed at, or (None, candidates) when it is ambiguous or empty."""
        near = self.near(p)
        if not near:
            return None, []
        best, d1 = near[0]
        if len(near) >= 2:
            other, d2 = near[1]
            nested = d1 == 0 and d2 == 0  # inside both: the smaller (sorted first) wins
            if not nested and d2 - d1 < self.params.ambiguous_mm:
                return None, [best, other]
        return best, [best]

    # ------------------------------------------------------------------ places

    def fits(self, c: Point, size: float) -> bool:
        b = Box.from_center(c[0], c[1], size, size)
        d = self.drawable
        return d.x - 0.5 <= b.x and b.x2 <= d.x2 + 0.5 and d.y - 0.5 <= b.y and b.y2 <= d.y2 + 0.5

    def free(self, c: Point, size: float, ignore: Sequence[SceneObject] = ()) -> bool:
        b = Box.from_center(c[0], c[1], size, size)
        for o in self.objects:
            if o in ignore:
                continue
            if iou(o.box, b) > 0 or _overlap(o.box, b):
                return False
        return True

    def relation(
        self, rel: str, target: SceneObject, size: float, target2: SceneObject | None = None
    ) -> Point:
        """Centre for a new object of `size` (its larger side) placed `rel` the target."""
        t = target.box
        g = self.params.gap_mm
        h = size / 2
        cx, cy = t.center
        options = {
            "left_of": (t.x - g - h, cy),
            "right_of": (t.x2 + g + h, cy),
            "above": (cx, t.y2 + g + h),
            "below": (cx, t.y - g - h),
        }
        if rel in options:
            return options[rel]
        if rel == "next_to":
            order = ["right_of", "left_of", "above", "below"]
            for r in order:
                c = options[r]
                if self.fits(c, size) and self.free(c, size, [target]):
                    return c
            for r in order:
                if self.fits(options[r], size):
                    return options[r]
            return options["right_of"]
        if rel == "between":
            assert target2 is not None
            a, b = target.center, target2.center
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if rel == "inside":
            return (cx, cy)
        if rel == "on":
            return (cx, t.y2 + h * 0.7)
        raise ValueError(f"unknown relation {rel!r}")

    def inside_size(self, target: SceneObject) -> float:
        return self.params.inside_fill * min(target.box.w, target.box.h)

    def named(self, name: str, size: float) -> Point | None:
        """A page region in words: middle, top, bottom left corner, left side, ..."""
        words = set(_norm(name).replace("-", " ").split())
        d = self.drawable
        h = size / 2
        x = {"left": d.x + h, "right": d.x2 - h}
        y = {"bottom": d.y + h, "top": d.y2 - h}
        words |= {"top" for w in list(words) if w == "upper"}
        words |= {"bottom" for w in list(words) if w == "lower"}
        cx, cy = d.center
        hx = next((x[w] for w in ("left", "right") if w in words), None)
        vy = next((y[w] for w in ("top", "bottom") if w in words), None)
        if hx is None and vy is None:
            if words & {"middle", "centre", "center"}:
                return (cx, cy)
            return None
        return (hx if hx is not None else cx, vy if vy is not None else cy)

    def empty_space(self, size: float, res_mm: float = 1.0) -> tuple[Point, float] | None:
        """Centre and side of the largest empty square (every object's box kept clear by a
        margin). The side returned is at most `size`; None if nothing is free at all."""
        d = self.drawable
        w, h = int(d.w / res_mm), int(d.h / res_mm)
        free = np.ones((h, w), np.uint8)
        m = self.params.empty_margin_mm
        for o in self.objects:
            b = o.box
            x0 = int(math.floor((b.x - m - d.x) / res_mm))
            x1 = int(math.ceil((b.x2 + m - d.x) / res_mm))
            y0 = int(math.floor((b.y - m - d.y) / res_mm))
            y1 = int(math.ceil((b.y2 + m - d.y) / res_mm))
            free[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0
        padded = np.pad(free, 1, constant_values=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_C, 3)[1:-1, 1:-1]
        best = float(dist.max())
        if best <= 0:
            return None
        ys, xs = np.nonzero(dist >= best - 0.5)
        # Several equally good spots: the one nearest the middle of the sheet.
        cx0, cy0 = w / 2, h / 2
        k = int(np.argmin((xs - cx0) ** 2 + (ys - cy0) ** 2))
        side = (2 * best - 1) * res_mm
        center = (d.x + (xs[k] + 0.5) * res_mm, d.y + (ys[k] + 0.5) * res_mm)
        return center, min(size, side)


def _overlap(a: Box, b: Box) -> bool:
    return a.x < b.x2 and b.x < a.x2 and a.y < b.y2 and b.y < a.y2


def _norm(s: str) -> str:
    s = s.strip().lower()
    for art in ("the ", "a ", "an ", "my ", "your "):
        if s.startswith(art):
            s = s[len(art) :]
    return s.strip(" '\"")
