"""The robot-agnostic answer to "where did the person mean?".

A `Where` is what the resolver hands to any robot: a point, box or path in a named frame,
with an uncertainty and a record of the evidence. See plan section 4.3.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Vec2 = tuple[float, float]
Cov2 = tuple[tuple[float, float], tuple[float, float]]

Source = Literal["gesture", "gesture+vlm", "vlm", "verbal_preset", "object_ref", "geometry"]


class Where(BaseModel):
    frame: str = Field(description="Frame of all coordinates, e.g. 'paper', 'floor', 'robot_base'.")
    kind: Literal["point", "box", "path"]
    mean_mm: Vec2 | None = Field(
        default=None, description="Point location, or the centre of a box. Filled in for boxes."
    )
    box_mm: tuple[float, float, float, float] | None = Field(
        default=None, description="x, y, w, h with (x, y) the minimum corner."
    )
    path_mm: list[Vec2] | None = Field(default=None, description="At least two points.")
    cov_mm2: Cov2 = ((0.0, 0.0), (0.0, 0.0))
    source: Source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description='E.g. {"word": "here", "word_t": 12.3, "hold_t": [11.9, 12.2]}.',
    )

    @model_validator(mode="after")
    def _check_kind(self) -> Where:
        if self.kind == "point" and self.mean_mm is None:
            raise ValueError("a point needs mean_mm")
        if self.kind == "box":
            if self.box_mm is None:
                raise ValueError("a box needs box_mm")
            x, y, w, h = self.box_mm
            if w < 0 or h < 0:
                raise ValueError("box width and height must be non-negative")
            if self.mean_mm is None:
                self.mean_mm = (x + w / 2, y + h / 2)
        if self.kind == "path" and (self.path_mm is None or len(self.path_mm) < 2):
            raise ValueError("a path needs at least two points")
        return self


def where_json_schema() -> dict[str, Any]:
    """JSON Schema for `Where`, for other languages and robots."""
    return Where.model_json_schema()
