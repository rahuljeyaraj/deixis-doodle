"""The contract the rest of the system uses to move the plotter.

Coordinates are millimetres in the bed frame (= machine XY; see pointdraw.layout).
The controller is the only code allowed to talk to the printer's serial port, and it
enforces every safety rule itself (plan section 4.5), whatever it is asked to do.

The controller session owns this file and may refine it. Keep changes backwards compatible
or list them in the session report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from deixis.geometry import Box, Polyline

Speed = Literal["slow", "normal", "fast"]
State = Literal["disconnected", "idle", "previewing", "drawing", "paused", "stopped", "error"]


@dataclass(frozen=True)
class ControllerStatus:
    state: State
    progress: float  # 0..1 for the current job
    hands_clear: bool  # last value received from the interlock
    message: str = ""


class PlotterController(Protocol):
    def preview_box(self, box: Box) -> None:
        """Trace the box outline with the pen up, so the user sees the intended place."""

    def draw_polylines(self, polylines: list[Polyline], speed: Speed = "normal") -> None:
        """Validate and draw. Rejects any point outside the drawable area. Non-blocking."""

    def park(self) -> None:
        """Pen up, bed toward the user, carriage to the side."""

    def stop(self) -> None:
        """Immediate halt, then pen up if possible."""

    def set_hands_clear(self, clear: bool) -> None:
        """Interlock input from vision. Motion pauses while False and never starts."""

    def status(self) -> ControllerStatus: ...
