"""Timed inputs to the resolver. All times are seconds on the shared monotonic clock."""

from __future__ import annotations

from dataclasses import dataclass

from deixis.geometry import Point


@dataclass(frozen=True)
class Word:
    """One recognised word with its start and end time."""

    text: str
    t0: float
    t1: float
    confidence: float | None = None


@dataclass(frozen=True)
class Hold:
    """A period when a tracked point (fingertip, foot) stayed still in a pointing pose."""

    t0: float
    t1: float
    mean: Point  # mm, in the frame named by `frame`
    spread: Point  # standard deviation in x and y, mm
    frame: str = "paper"
    source: str = "index_tip"  # e.g. "index_tip", "foot"

    @property
    def mid(self) -> float:
        return (self.t0 + self.t1) / 2
