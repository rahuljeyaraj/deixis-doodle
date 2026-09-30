"""Turn "object, place, size" requests and scene specs into millimetre polylines.

This is the planner's front door: the resolver supplies a centre and size in mm (never
the LLM), and these functions return polylines plus the ink's bounding box, which the
scene model keeps for later references ("an arrow from this to that").

Scene spec (JSON), used by scripts/render_drawing.py for storyboards:

    {"paper": [0, 0, 210, 148],
     "items": [
       {"type": "object", "id": "ear", "name": "ear", "at": [40, 90], "size": 35},
       {"type": "label", "text": "ASR", "at": [40, 62], "size": 5},
       {"type": "label", "text": "ASR", "below": "ear", "size": 5, "gap": 3},
       {"type": "arrow", "from": "ear", "to": "brain"},
       {"type": "line", "from": [10, 10], "to": [50, 10]},
       {"type": "box", "box": [x, y, w, h]},
       {"type": "circle", "at": [100, 70], "r": 10}]}

Arrow and line ends can be an item id (the item's box centre, trimmed at its box) or a
point [x, y] in mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deixis.geometry import Box, Point, Polyline
from pointdraw.drawing import primitives
from pointdraw.drawing.library import Library
from pointdraw.drawing.optimize import Stats, optimize
from pointdraw.drawing.place import bbox, fit
from pointdraw.drawing.render import A5
from pointdraw.drawing.text import text


@dataclass(frozen=True)
class Placed:
    """One thing on the paper: its polylines (mm) and the box of its ink."""

    id: str
    kind: str  # object | label | arrow | line | box | circle
    polylines: list[Polyline]
    box: Box
    category: str | None = None
    key_id: str | None = None


def place_object(
    name: str,
    center: Point,
    size: float,
    library: Library | None = None,
    variant: int | str | None = None,
    smooth_iterations: int = 2,
    id: str | None = None,
) -> Placed:
    """A library drawing whose larger bbox side is `size` mm, centred on `center`."""
    lib = library or Library.load()
    d = lib.get(name, variant)
    pls = fit(d.strokes, center, size, smooth_iterations)
    return Placed(id or d.category, "object", pls, bbox(pls), d.category, d.key_id)


def place_label(s: str, center: Point, cap_height: float = 5.0, id: str | None = None) -> Placed:
    pls = text(s, center, cap_height)
    return Placed(id or s, "label", pls, bbox(pls))


def compose(spec: dict[str, Any], library: Library | None = None) -> tuple[list[Placed], Box]:
    """Build every item of a scene spec, in order. Returns the items and the paper box."""
    paper = Box(*spec["paper"]) if "paper" in spec else A5
    lib = library
    items: dict[str, Placed] = {}
    out: list[Placed] = []

    def end(ref: Any) -> tuple[Point, Box | None]:
        if isinstance(ref, str):
            if ref not in items:
                raise KeyError(f"unknown item id {ref!r}; define it before referring to it")
            return items[ref].box.center, items[ref].box
        return (float(ref[0]), float(ref[1])), None

    for n, it in enumerate(spec["items"]):
        kind = it["type"]
        iid = it.get("id", f"{kind}{n}")
        if kind == "object":
            lib = lib or Library.load()
            p = place_object(
                it["name"], tuple(it["at"]), float(it["size"]), lib, it.get("variant"), id=iid
            )
        elif kind == "label":
            cap = float(it.get("size", 5.0))
            gap = float(it.get("gap", 3.0))
            if "below" in it or "above" in it:
                _, ref = end(it.get("below", it.get("above")))
                if ref is None:
                    raise ValueError("a label's below/above must name an item id")
                y = ref.y - gap - cap / 2 if "below" in it else ref.y2 + gap + cap / 2
                at = (ref.center[0], y)
            else:
                at = (float(it["at"][0]), float(it["at"][1]))
            p = place_label(it["text"], at, cap, id=iid)
        elif kind in ("arrow", "line"):
            (a, abox), (b, bbox_) = end(it["from"]), end(it["to"])
            if kind == "arrow":
                pls = primitives.arrow(a, b, abox, bbox_, gap=float(it.get("gap", 2.0)))
            else:
                a, b = primitives.trim_to_boxes(a, b, abox, bbox_, gap=float(it.get("gap", 2.0)))
                pls = primitives.line(a, b)
            p = Placed(iid, kind, pls, bbox(pls))
        elif kind == "box":
            b = Box(*map(float, it["box"]))
            p = Placed(iid, kind, primitives.box(b), b)
        elif kind == "circle":
            pls = primitives.circle(tuple(it["at"]), float(it["r"]))
            p = Placed(iid, kind, pls, bbox(pls))
        else:
            raise ValueError(f"unknown item type {kind!r}")
        items[iid] = p
        out.append(p)
    return out, paper


def plan(placed: list[Placed], start: Point = (0.0, 0.0)) -> tuple[list[Polyline], Stats]:
    """All items' polylines, optimised as one job."""
    return optimize([pl for p in placed for pl in p.polylines], start)
