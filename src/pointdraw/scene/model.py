"""The scene model (plan T1.9): what is on the sheet, and places relative to it.

Everything is in the **paper frame** (mm; +x to the user's right, +y away from the user).

- **Robot objects** are known exactly: the planner's polylines and ink box.
- **User marks** come from ink segmentation of the camera's paper image
  (`scene.ink.segment_ink`, with the robot's own strokes subtracted). A vision model
  (Cosmos, or a mock) gives them labels.
- **Relations** (`next_to`, `left_of`, `right_of`, `above`, `below`, `behind`,
  `in_front_of`, `between`, `inside`, `on`), **named regions** ("top left corner") and
  **empty space** are computed by geometry in the deixis core (`deixis.scene`), never by a
  language model (LR§3.2). This model keeps the plotter's objects (polylines, ink marks,
  ids) and hands the core a `Thing` for each.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from deixis import scene as core
from deixis.geometry import Box, Point, Polyline
from deixis.profile import Profile, SceneSettings
from deixis.scene import Thing, iou, union_box
from pointdraw.profile import plotter_profile
from pointdraw.scene.ink import UserMark

__all__ = ["SceneModel", "SceneObject", "SceneParams", "iou", "union_box"]

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
    key_id: str | None = None  # robot objects: which library drawing (to re-place the same one)

    @property
    def center(self) -> Point:
        return self.box.center


@dataclass(frozen=True)
class SceneParams:
    gap_mm: float = 10.0  # space between an object and one placed next to it (room for a label)
    near_mm: float = 20.0  # "that one": objects within this distance of the point
    ambiguous_mm: float = 6.0  # two objects this close in distance -> ask
    empty_margin_mm: float = 3.0  # keep this far from other ink in empty space
    inside_fill: float = 0.7  # "inside": the new object's larger side / target's smaller side
    mark_match_iou: float = 0.3  # a new mark with this IoU against an old one keeps its id


class SceneModel:
    def __init__(self, drawable: Box, params: SceneParams | None = None) -> None:
        self.drawable = drawable  # paper frame
        self.params = params or SceneParams()
        self.objects: list[SceneObject] = []
        self._n: dict[str, int] = {}
        self.profile = scene_profile(drawable, self.params)

    # ------------------------------------------------------------------ the core's view

    def things(self) -> list[Thing]:
        """Every object as a core `Thing` (labels and arrows are not selectable)."""
        return [thing(o) for o in self.objects]

    def _obj(self, t: Thing | None) -> SceneObject | None:
        if t is None:
            return None
        return next(o for o in self.objects if o.id == t.id)

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
        return self._obj(core.last(self.things()))

    def labels(self) -> list[str]:
        """Labels for the agent's context, oldest first, without repeats."""
        return core.labels(self.things())

    def find(self, label: str) -> list[SceneObject]:
        """Objects with this label (or text), most recent first."""
        return [self._obj(t) for t in core.find(self.things(), label)]  # type: ignore[misc]

    def near(self, p: Point, max_mm: float | None = None) -> list[tuple[SceneObject, float]]:
        """Objects near a point (0 = inside the box), nearest first. Labels and arrows skipped."""
        return [(self._obj(t), d) for t, d in core.near(self.things(), p, self.profile, max_mm)]  # type: ignore[misc]

    def object_at(self, p: Point) -> tuple[SceneObject | None, list[SceneObject]]:
        """The object pointed at, or (None, candidates) when it is ambiguous or empty."""
        best, cands = core.object_at(self.things(), p, self.profile)
        return self._obj(best), [self._obj(c) for c in cands]  # type: ignore[misc]

    # ------------------------------------------------------------------ places

    def fits(self, c: Point, size: float) -> bool:
        return core.fits(c, size, self.profile)

    def free(self, c: Point, size: float, ignore: Sequence[SceneObject] = ()) -> bool:
        return core.free(self.things(), c, size, [o.id for o in ignore])

    def relation(
        self,
        rel: str,
        target: SceneObject,
        size: float,
        target2: SceneObject | None = None,
        extent: tuple[float, float] | None = None,
    ) -> Point:
        """Centre for a new object of `size` (its larger side) placed `rel` the target.

        With `extent` (the drawing's real width and height) it sits `gap_mm` from the target;
        without, it is treated as a square of side `size`."""
        spot = core.relation(
            rel,
            thing(target),
            size,
            self.profile,
            self.things(),
            thing(target2) if target2 is not None else None,
            extent,
        )
        return spot.center

    def inside_size(self, target: SceneObject) -> float:
        return core.inside_size(thing(target), self.profile)

    def named(self, name: str, size: float) -> Point | None:
        """A page region in words: middle, top, bottom left corner, left side, ..."""
        return core.named(name, size, self.profile)

    def empty_space(self, size: float, res_mm: float = 1.0) -> tuple[Point, float] | None:
        """Centre and side of the largest empty square (every object's box kept clear by a
        margin). The side returned is at most `size`; None if nothing is free at all."""
        return core.empty_space(self.things(), size, self.profile, res_mm)


def thing(o: SceneObject) -> Thing:
    """The core's view of an object: written text is an attribute, so "the Cosmos label"
    finds it; labels and arrows are not selectable (pointing and "it" skip them)."""
    b = o.box
    return Thing(
        id=o.id,
        label=o.label,
        box_mm=(b.x, b.y, b.w, b.h),
        kind=o.kind,
        attrs={"text": o.text} if o.text else {},
        t=o.t,
        selectable=o.kind in ("object", "mark"),
    )


def scene_profile(drawable: Box, params: SceneParams) -> Profile:
    """The plotter's profile with this drawable area and these scene settings."""
    base = plotter_profile(drawable)
    scene = SceneSettings(
        near_mm=params.near_mm,
        ambiguous_mm=params.ambiguous_mm,
        gap_mm=params.gap_mm,
        inside_fill=params.inside_fill,
        empty_margin_mm=params.empty_margin_mm,
        grid_mm=base.scene.grid_mm,
        label_ties=base.scene.label_ties,
    )
    return base.model_copy(update={"scene": scene})
