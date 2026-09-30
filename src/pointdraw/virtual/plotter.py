"""A `PlotterController` that runs in simulated time, for the virtual world.

It reuses the real controller's pure parts, so jobs are checked and built exactly as on
the machine:

- `safety.validate_polylines` / `validate_points` (the drawable area, job limits)
- `gcode.*` (pen up and down, splitting into 10 mm pieces, preview trace, park, home)
- `safety.check_line` on every line (the whitelist: no heater, fan or extrusion commands)
- `gcode.insert_syncs` (an `M400` every 30 mm)
- `SimulatedPlotter` to interpret the G-code and record the strokes

The streaming policy of `controller.Controller` is modelled in simulated time instead of
threads, so runs are fast and repeatable:

- Motion never starts, and a job never resumes, until hands have been clear for
  `hands_clear_hold_s`.
- **Pause latency:** Marlin acknowledges moves when they enter its planner, so when the
  interlock trips, the moves already sent still run: up to the next `M400` plus the one line
  queued behind it (CTRL report, decision 4). Then the pen lifts.
- **Stop:** with `emergency_parser`, `M410` halts mid-move at once. Without it, `M410` waits
  behind the queued lines, so motion runs to the next `M400` first. Either way the queue is
  dropped, the pen lifts and the next job homes first.

Durations are distance / feedrate (no acceleration), as in `SimulatedPlotter`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from deixis.geometry import Box, Point, Polyline
from pointdraw.controller import gcode
from pointdraw.controller.api import ControllerStatus, Speed, State
from pointdraw.controller.config import MachineConfig
from pointdraw.controller.safety import check_line, parse_line, validate_points, validate_polylines
from pointdraw.controller.simulator import SimulatedPlotter

FORBIDDEN_PREFIXES = ("M104", "M109", "M140", "M190", "M106", "M107", "G92 E")


@dataclass
class _Job:
    label: str
    lines: list[str]
    state: State
    tag: Any = None  # the caller's handle for this job (e.g. a drawing id)


@dataclass
class PlotEvent:
    t: float
    kind: str  # job_start, job_end, pause_request, paused, resumed, stop_request, halted
    label: str = ""
    tag: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class InkSegment:
    t: float  # when the pen finished laying this piece
    a: Point  # bed mm
    b: Point
    tag: Any = None


class VirtualPlotter:
    """Implements `api.PlotterController`, plus `tick(t)` to advance simulated time."""

    def __init__(
        self,
        config: MachineConfig | None = None,
        *,
        emergency_parser: bool = False,
        home_s: float = 4.0,
    ) -> None:
        self.config = config or MachineConfig()
        self.emergency_parser = emergency_parser
        self.home_s = home_s
        self.sim = SimulatedPlotter(self.config)
        self.sim.connect()
        for line in gcode.startup_block(self.config):
            self.sim.send(check_line(line, phase="startup", config=self.config))
        self.sim.x, self.sim.y = self.config.park_x, self.config.park_y  # starts parked
        self.t = 0.0
        self.jobs: deque[_Job] = deque()
        self.job: _Job | None = None
        self.i = 0  # next line of the current job
        self.move_left = 0.0  # seconds left on the line in progress
        self.move_total = 0.0
        self.move_from: tuple[float, float, float] | None = None
        self.hands_clear = False
        self.clear_since = math.inf
        self.state: State = "idle"
        self.drain_to: int | None = None  # committed lines when the gate closed
        self.lifted = False  # pen lifted because of a pause
        self.stopping = False
        self.needs_home = False
        self.events: list[PlotEvent] = []
        self.ink: list[InkSegment] = []
        self.track: list[tuple[float, float, float, bool]] = []  # t, x, y, pen down
        self.sent: list[str] = []
        self._job_started = False

    # ------------------------------------------------------------------ PlotterController

    def preview_box(self, box: Box, tag: Any = None) -> None:
        corners = [(box.x, box.y), (box.x2, box.y), (box.x2, box.y2), (box.x, box.y2)]
        d = self.config.drawable
        validate_points([corners], d, self.config.clamp_tolerance_mm)
        x0, y0 = max(box.x, d.x), max(box.y, d.y)
        x1, y1 = min(box.x2, d.x2), min(box.y2, d.y2)
        lines = gcode.preview_box_gcode(Box(x0, y0, x1 - x0, y1 - y0), self.config)
        self._submit("preview", lines, "previewing", tag)

    def draw_polylines(self, polylines: list[Polyline], speed: Speed = "normal", tag: Any = None):
        clean = validate_polylines(polylines, self.config)
        feed = self.config.feed_for(speed)
        self._submit("draw", gcode.polylines_to_gcode(clean, self.config, feed), "drawing", tag)

    def park(self, tag: Any = None) -> None:
        self._submit("park", gcode.park_gcode(self.config), "moving", tag)

    def stop(self) -> None:
        self.events.append(PlotEvent(self.t, "stop_request", self.job.label if self.job else ""))
        dropped = len(self.jobs)
        self.jobs.clear()
        if not self._moving():
            self._halted(dropped)
            return
        if self.emergency_parser:
            self._cut_move()
            self._halted(dropped)
        else:
            self.stopping = True
            self.drain_to = self._committed()

    def set_hands_clear(self, clear: bool) -> None:
        if clear and not self.hands_clear:
            self.clear_since = self.t
        if not clear and self.hands_clear and self._moving() and self.drain_to is None:
            self.drain_to = self._committed()
            self.events.append(PlotEvent(self.t, "pause_request", self.job.label, self.job.tag))
        self.hands_clear = clear

    def status(self) -> ControllerStatus:
        progress = self.i / len(self.job.lines) if self.job and self.job.lines else 0.0
        return ControllerStatus(self.state, progress, self.hands_clear, "")

    # ------------------------------------------------------------------ queries

    @property
    def busy(self) -> bool:
        return self.job is not None or bool(self.jobs)

    def position(self) -> tuple[float, float, float]:
        """Where the head is now, interpolated inside the move in progress."""
        if self.move_from is None or self.move_total <= 0 or self.job is None:
            return (self.sim.x, self.sim.y, self.sim.z)
        tgt = self._target(self.job.lines[self.i])
        u = 1.0 - self.move_left / self.move_total
        return tuple(a + (b - a) * u for a, b in zip(self.move_from, tgt, strict=True))  # type: ignore[return-value]

    def ink_polylines(self, tag: Any = None) -> list[Polyline]:
        return [[s.a, s.b] for s in self.ink if tag is None or s.tag == tag]

    # ------------------------------------------------------------------ time

    def tick(self, t: float) -> None:
        """Advance simulated time to `t`, running whatever motion is allowed."""
        guard = 0
        while self.t < t - 1e-9:
            guard += 1
            if guard > 100_000:
                raise RuntimeError("virtual plotter did not advance")
            if self.job is None:
                if not self.jobs:
                    self._set_state("idle" if self.state != "stopped" else "stopped")
                    self.t = t
                    break
                self.job, self.i, self._job_started = self.jobs.popleft(), 0, False
            job = self.job
            if self.i >= len(job.lines):
                self._finish_job()
                continue
            if self.move_left <= 0:  # about to start line i: may it go?
                if not self._gate():
                    self.t = t
                    break
                if not self._begin_line(job.lines[self.i]):
                    continue
            dt = min(self.move_left, t - self.t)
            self.t += dt
            self.move_left -= dt
            if self.move_left <= 1e-12:
                self._complete_line(job.lines[self.i])
        self._record_track()

    # ------------------------------------------------------------------ internals

    def _submit(self, label: str, lines: list[str], state: State, tag: Any) -> None:
        for line in lines:
            check_line(line, phase="job", config=self.config)
            if line.startswith(FORBIDDEN_PREFIXES):
                raise AssertionError(f"forbidden line {line!r}")
        lines = gcode.insert_syncs(lines, self.config.sync_every_mm)
        if self.needs_home and lines[:1] != ["G28"]:
            lines = gcode.home_gcode(self.config) + lines
            self.needs_home = False
        self.jobs.append(_Job(label, lines, state, tag))

    def _moving(self) -> bool:
        """A job has sent lines to Marlin and isn't held by the interlock."""
        return self.job is not None and self._job_started and self.state != "paused"

    def _may_move(self) -> bool:
        return self.hands_clear and self.t - self.clear_since >= self.config.hands_clear_hold_s

    def _gate(self) -> bool:
        """Whether line `i` may start now. Handles draining, pausing, lifting and resuming."""
        job = self.job
        assert job is not None
        if self.drain_to is not None:
            if self.i <= self.drain_to:
                return True
            if self.stopping:
                self._halted(0)
                return False
            self.drain_to = None
            self.events.append(PlotEvent(self.t, "paused", job.label, job.tag, self._pos()))
        if self._may_move():
            if not self._job_started:
                self._job_started = True
                self.events.append(PlotEvent(self.t, "job_start", job.label, job.tag, self._pos()))
            elif self.state == "paused":
                self.events.append(PlotEvent(self.t, "resumed", job.label, job.tag, self._pos()))
            self._set_state(job.state)
            if self.lifted:
                self.lifted = False
                if _is_draw_move(job.lines[self.i]):
                    job.lines.insert(self.i, gcode.pen_down(self.config))
            return True
        self._set_state("paused")
        if self.sim.pen_is_down and not self.lifted:
            job.lines.insert(self.i, gcode.pen_up(self.config))
            self.lifted = True
            return True  # the lift itself may run
        return False

    def _begin_line(self, line: str) -> bool:
        """Start timing line i. Returns False for zero-time lines (already completed)."""
        cmd, _ = parse_line(line)
        if cmd == "G28":
            self.move_from = (self.sim.x, self.sim.y, self.sim.z)
            self.move_total = self.move_left = self.home_s
            return True
        if cmd not in ("G0", "G1"):
            self._complete_line(line)
            return False
        start = (self.sim.x, self.sim.y, self.sim.z)
        tgt = self._target(line)
        _, p = parse_line(line)
        feed = p.get("F") or 1000.0
        dur = math.dist(start, tgt) / (feed / 60.0)
        if dur <= 0:
            self._complete_line(line)
            return False
        self.move_from = start
        self.move_total = self.move_left = dur
        return True

    def _target(self, line: str) -> tuple[float, float, float]:
        cmd, p = parse_line(line)
        if cmd == "G28":
            return (0.0, 0.0, 0.0)
        s = self.sim
        return (
            p["X"] if p.get("X") is not None else s.x,
            p["Y"] if p.get("Y") is not None else s.y,
            p["Z"] if p.get("Z") is not None else s.z,
        )

    def _complete_line(self, line: str) -> None:
        a = (self.sim.x, self.sim.y)
        down_before = self.sim.pen_is_down
        self.sim.send(line)
        self.sent.append(line)
        b = (self.sim.x, self.sim.y)
        if down_before and self.sim.pen_is_down and a != b:
            self.ink.append(InkSegment(self.t, a, b, self.job.tag if self.job else None))
        self.move_left = 0.0
        self.move_from = None
        self.i += 1
        self._record_track()

    def _cut_move(self) -> None:
        """Emergency stop: end the move in progress where the head is now."""
        if self.job is None or self.move_left <= 0 or self.move_from is None:
            return
        x, y, z = self.position()
        line = self.job.lines[self.i]
        cmd, p = parse_line(line)
        if cmd in ("G0", "G1"):
            f = p.get("F") or 1000.0
            cut = f"{cmd} X{gcode.fmt(x)} Y{gcode.fmt(y)} Z{gcode.fmt(z)} F{gcode.fmt(f)}"
            self.job.lines[self.i] = cut
            self._complete_line(cut)

    def _committed(self) -> int:
        """Index of the last line already in Marlin's queue: up to the next M400, plus one."""
        assert self.job is not None
        lines = self.job.lines
        j = self.i
        while j < len(lines) and lines[j] != "M400":
            j += 1
        return min(j + 1, len(lines) - 1)

    def _halted(self, dropped: int) -> None:
        label = self.job.label if self.job else ""
        tag = self.job.tag if self.job else None
        self.events.append(
            PlotEvent(self.t, "halted", label, tag, {**self._pos(), "dropped": dropped})
        )
        self.job = None
        self.jobs.clear()
        self.move_left = 0.0
        self.move_from = None
        self.drain_to = None
        self.stopping = False
        self.lifted = False
        if self.sim.pen_is_down:
            self.sim.send(gcode.pen_up(self.config))
        self.needs_home = self.config.rehome_after_stop
        self._set_state("stopped")

    def _finish_job(self) -> None:
        job = self.job
        assert job is not None
        self.events.append(PlotEvent(self.t, "job_end", job.label, job.tag, self._pos()))
        self.job = None
        self.drain_to = None
        self.stopping = False
        self._set_state("idle")

    def _set_state(self, s: State) -> None:
        self.state = s

    def _pos(self) -> dict[str, Any]:
        x, y, z = self.position()
        return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}

    def _record_track(self) -> None:
        x, y, z = self.position()
        down = z < self.config.pen_threshold_z
        if not self.track or self.track[-1][1:] != (x, y, down):
            self.track.append((self.t, x, y, down))


def _is_draw_move(line: str) -> bool:
    cmd, p = parse_line(line)
    return cmd == "G1" and (p.get("X") is not None or p.get("Y") is not None)
