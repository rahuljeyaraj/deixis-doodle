"""What to draw, in paper mm: library drawings, simple shapes, labels and arrows.

The resolver supplies a centre and a size (never the language model); this module turns
them into polylines with `pointdraw.drawing`, and keeps every drawing inside the drawable
area (moving or shrinking it a little if needed, so the robot can say so).

Shapes that aren't in the Quick, Draw! library but that people ask for ("a cross here"
for M1, "a circle") are drawn as clean primitives.
"""

from __future__ import annotations

import math

from deixis.geometry import Box, Point
from pointdraw.controller.shapes import cross
from pointdraw.drawing import Library, Placed, place_label, place_object
from pointdraw.drawing import primitives as prim
from pointdraw.drawing.place import bbox, keep_inside, transform


def _triangle(c: Point, size: float) -> list[list[Point]]:
    """Equilateral, point up; its width (the larger side) is `size`."""
    h = size * math.sqrt(3) / 2
    pts = [(c[0] - size / 2, c[1] - h / 2), (c[0] + size / 2, c[1] - h / 2), (c[0], c[1] + h / 2)]
    return [[*pts, pts[0]]]


PRIMITIVES = {
    "cross": lambda c, s: cross(c, s),
    "circle": lambda c, s: prim.circle(c, s / 2),
    "square": lambda c, s: prim.box(Box.from_center(c[0], c[1], s, s)),
    "triangle": _triangle,
}
PRIMITIVE_SYNONYMS = {
    "plus": "cross",
    "x": "cross",
    "ring": "circle",
    "box": "square",
    "rectangle": "square",
}


class Planner:
    def __init__(self, library: Library | None = None, drawable: Box | None = None) -> None:
        self.library = library or Library.load()
        self.drawable = drawable  # paper frame

    @property
    def categories(self) -> list[str]:
        """Everything that can be drawn by name: the library plus the primitives."""
        return sorted(set(self.library.categories) | set(PRIMITIVES))

    def resolve(self, name: str) -> str | None:
        n = name.strip().lower()
        for art in ("a ", "an ", "the "):
            if n.startswith(art):
                n = n[len(art) :]
        n = PRIMITIVE_SYNONYMS.get(n, n)
        if n in PRIMITIVES:
            return n
        return self.library.resolve(n)

    def suggestions(self, name: str, n: int = 3) -> list[str]:
        import difflib

        pool = self.categories + list(PRIMITIVE_SYNONYMS)
        out = []
        for s in difflib.get_close_matches(name.lower(), pool, n=n, cutoff=0.5):
            s = PRIMITIVE_SYNONYMS.get(s, s)
            s = self.library.resolve(s) or s
            if s not in out:
                out.append(s)
        return out

    def object(self, category: str, center: Point, size: float, id: str | None = None) -> Placed:
        if category in PRIMITIVES:
            pls = PRIMITIVES[category](center, size)
            return Placed(id or category, "object", pls, bbox(pls), category)
        return place_object(category, center, size, self.library, id=id)

    def label(self, text: str, center: Point, cap_mm: float, id: str | None = None) -> Placed:
        return place_label(text, center, cap_mm, id=id)

    def arrow(
        self,
        a: Point,
        b: Point,
        a_box: Box | None,
        b_box: Box | None,
        kind: str = "arrow",
        id: str | None = None,
    ) -> Placed:
        if kind == "line":
            p, q = prim.trim_to_boxes(a, b, a_box, b_box, gap=2.0)
            pls = prim.line(p, q)
        elif kind == "double_arrow":
            pls = prim.arrow(a, b, a_box, b_box, gap=2.0)
            back = prim.arrow(b, a, b_box, a_box, gap=2.0)
            pls = pls + back[1:]
        else:
            pls = prim.arrow(a, b, a_box, b_box, gap=2.0)
        return Placed(id or kind, "arrow", pls, bbox(pls))

    def keep_inside(self, p: Placed) -> tuple[Placed, bool]:
        """Move (or shrink) a drawing so it lies inside the drawable area."""
        if self.drawable is None:
            return p, False
        pls, moved = keep_inside(p.polylines, self.drawable)
        if not moved:
            return p, False
        return Placed(p.id, p.kind, pls, bbox(pls), p.category, p.key_id), True

    @staticmethod
    def shift(p: Placed, dx: float, dy: float, scale: float = 1.0) -> Placed:
        """Move by (dx, dy) mm and scale about the drawing's own centre."""
        cx, cy = p.box.center
        pls = transform(p.polylines, scale, cx - cx * scale + dx, cy - cy * scale + dy)
        return Placed(p.id, p.kind, pls, bbox(pls), p.category, p.key_id)
