"""`PlotterPort` over the real `controller.Controller`, so the runtime can drive the machine.

The runtime follows jobs by tag: it reads `events` (job_start, job_end, paused, resumed,
halted, ...) by index, checks `busy`, and asks `ink_polylines(tag)` where a drawing went.
`VirtualPlotter` does all that in simulated time; here the controller's worker thread
reports it through `on_event` and `on_line`.

Ink is worked out from the lines sent (a move with the pen down before and after it), so
it is where the machine was told to go. A line counts as drawn once it has been sent,
which on the real machine is a little before the pen gets there (Marlin queues moves), and
a stop can cut the last piece short.

`events`, `ink`, `track` and `sent` match `VirtualPlotter`'s, so the virtual world's checks
and replay work on either.

A command the controller refuses because the link has failed (a pulled USB cable) does not
raise into the runtime: it becomes an `error` event with the job's tag, as a job that fails
on the worker does, and the runtime drops that drawing. `reconnect()` opens the link again.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from deixis.geometry import Box, Point, Polyline
from pointdraw.controller.api import ControllerStatus, InkSegment, PlotEvent, Speed
from pointdraw.controller.controller import Controller
from pointdraw.controller.safety import parse_line
from pointdraw.controller.transport import LinkError


class LivePlotter:
    """Implements the runtime's `PlotterPort` with a `Controller`.

    `stamp` dates each sent line for `ink` and `track` (default: the controller's clock).
    """

    def __init__(self, controller: Controller, *, stamp: Callable[[], float] | None = None):
        self.ctl = controller
        self.config = controller.config
        self.stamp = stamp or controller.clock
        self.events: list[PlotEvent] = []
        self.ink: list[InkSegment] = []
        self.track: list[tuple[float, float, float, bool]] = []  # t, x, y, pen down
        self.sent: list[str] = []
        self._lock = threading.Lock()
        self._pos: tuple[float | None, float | None, float | None] = (None, None, None)
        controller.on_event = self._on_event
        controller.on_line = self._on_line

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.ctl.start()

    def close(self) -> None:
        self.ctl.close()

    def assume_at(self, x: float, y: float, z: float) -> None:
        """The head is known to be here (for example parked by an earlier run)."""
        with self._lock:
            self._pos = (x, y, z)

    # ------------------------------------------------------------------ PlotterPort

    @property
    def busy(self) -> bool:
        return self.ctl.busy

    def preview_box(self, box: Box, tag: Any = None) -> None:
        self._submit("preview", tag, self.ctl.preview_box, box, tag=tag)

    def preview_circle(self, center: Point, size: float, tag: Any = None) -> None:
        self._submit("preview", tag, self.ctl.preview_circle, center, size, tag=tag)

    def draw_polylines(self, polylines: list[Polyline], speed: Speed = "normal", tag: Any = None):
        self._submit("draw", tag, self.ctl.draw_polylines, polylines, speed, tag=tag)

    def park(self, tag: Any = None) -> None:
        self._submit("park", tag, self.ctl.park, tag=tag)

    def home(self, tag: Any = None) -> None:
        self._submit("home", tag, self.ctl.home, tag=tag)

    def reconnect(self) -> None:
        """Open the link again after a failure (raises LinkError if it still fails)."""
        self.ctl.reconnect()
        with self._lock:
            self._pos = (None, None, None)  # unknown until the next home

    def _submit(self, label: str, job: Any, fn: Callable[..., None], *a: Any, **k: Any) -> None:
        try:
            fn(*a, **k)
        except LinkError as e:
            self._on_event(PlotEvent(self.ctl.clock(), "error", label, job, {"message": str(e)}))

    def stop(self) -> None:
        self.ctl.stop()

    def set_hands_clear(self, clear: bool) -> None:
        self.ctl.set_hands_clear(clear)

    def status(self) -> ControllerStatus:
        return self.ctl.status()

    def wait(self, timeout: float | None = None) -> bool:
        return self.ctl.wait(timeout)

    def ink_polylines(self, tag: Any = None) -> list[Polyline]:
        with self._lock:
            return [[s.a, s.b] for s in self.ink if tag is None or s.tag == tag]

    def position(self) -> tuple[float | None, float | None, float | None]:
        """Where the head was last sent (None for an axis not known since start)."""
        with self._lock:
            return self._pos

    # ------------------------------------------------------------------ controller hooks

    def _on_event(self, e: PlotEvent) -> None:
        x, y, z = self.position()
        if x is not None and y is not None:
            e.detail = {"x": round(x, 2), "y": round(y, 2), **e.detail}
            if z is not None:
                e.detail["z"] = round(z, 2)
        self.events.append(e)

    def _on_line(self, line: str, tag: Any) -> None:
        cmd, p = parse_line(line)
        t = self.stamp()
        with self._lock:
            self.sent.append(line)
            if cmd not in ("G0", "G1", "G28"):
                return
            before = self._pos
            if cmd == "G28":
                axes = [a for a in "XYZ" if a in p] or ["X", "Y", "Z"]
                new = {a: 0.0 for a in axes}
            else:
                new = {a: p[a] for a in "XYZ" if p.get(a) is not None}
            x, y, z = (new.get(a, v) for a, v in zip("XYZ", before, strict=True))
            self._pos = (x, y, z)
            x0, y0, z0 = before
            if x is None or y is None or z is None:
                return
            down = self.config.pen_threshold_z
            if x0 is not None and y0 is not None and z0 is not None:
                if z0 < down and z < down and (x0, y0) != (x, y):
                    self.ink.append(InkSegment(t, (x0, y0), (x, y), tag))
            self.track.append((t, x, y, z < down))
