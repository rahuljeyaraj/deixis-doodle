"""The contract the rest of the system uses to move the plotter.

Coordinates are millimetres in the bed frame (= machine XY; see pointdraw.layout).
The controller is the only code allowed to talk to the printer's serial port, and it
enforces every safety rule itself (plan section 4.5), whatever it is asked to do.

The controller session owns this file and may refine it. Keep changes backwards compatible
or list them in the session report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from deixis.geometry import Box, Point, Polyline

Speed = Literal["slow", "normal", "fast"]
State = Literal[
    "disconnected", "idle", "previewing", "drawing", "moving", "paused", "stopped", "error"
]
# "moving": park, home and bring-up moves (added by session CTRL).
# "paused": the hands-clear interlock is holding motion (also before a job starts).
# "stopped": after stop(), until the next command starts.


@dataclass(frozen=True)
class ControllerStatus:
    state: State
    progress: float  # 0..1 for the current job
    hands_clear: bool  # last value received from the interlock
    message: str = ""


@dataclass
class PlotEvent:
    """Something that happened to a job, stamped on the controller's clock."""

    t: float
    # job_start, job_end, pause_request, paused, resumed, stop_request, halted, error,
    # reconnected
    kind: str
    label: str = ""  # the job kind: preview, draw, park, home
    tag: Any = None  # the caller's handle for the job (e.g. a drawing id)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class InkSegment:
    t: float  # when the pen finished laying this piece
    a: Point  # bed mm
    b: Point
    tag: Any = None


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
