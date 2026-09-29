"""Plain geometry types shared by every package. Units are millimetres unless noted."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
Polyline = list[Point]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box: (x, y) is the minimum corner, w and h are the size."""

    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float) -> Box:
        return cls(cx - w / 2, cy - h / 2, w, h)

    @property
    def center(self) -> Point:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        """True if p lies inside the box shrunk by `margin` on every side."""
        return (
            self.x + margin <= p[0] <= self.x2 - margin
            and self.y + margin <= p[1] <= self.y2 - margin
        )

    def inset(self, margin: float) -> Box:
        return Box(self.x + margin, self.y + margin, self.w - 2 * margin, self.h - 2 * margin)
