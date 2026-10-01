"""`PlotterController` on top of a transport (the serial link or the simulator).

Jobs are validated and turned into G-code on the caller's thread, so bad input fails at once
with a `SafetyError`. A worker thread then streams each job one line at a time:

- Before every line it checks the interlock. While hands are not clear it sends nothing, lifts
  the pen if it was down, and reports `paused`. Motion resumes (pen lowered again if needed)
  once hands have been clear for `hands_clear_hold_s`. Motion never starts while not clear.
- Each job can carry a `tag`. `on_event` hears job_start, pause_request, paused, resumed,
  job_end, stop_request, halted and error (`api.PlotEvent`, stamped with `clock`), and
  `on_line` hears every line sent with its job's tag, so a caller can follow the ink.
- `stop()` sends M410 at once from the caller's thread, drops every queued job, then the
  worker renumbers the link and lifts the pen. M410 stops without deceleration and Marlin
  warns that steps may be lost, so (with `rehome_after_stop`) the next command homes first.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from deixis.geometry import Box, Point, Polyline
from pointdraw.clock import now
from pointdraw.controller import gcode
from pointdraw.controller.api import ControllerStatus, PlotEvent, Speed, State
from pointdraw.controller.config import MachineConfig
from pointdraw.controller.safety import (
    check_line,
    parse_line,
    validate_points,
    validate_polylines,
)
from pointdraw.controller.transport import LinkError, LinkStopped, Transport

log = logging.getLogger(__name__)


@dataclass
class _Job:
    label: str
    lines: list[str]
    state: State
    gen: int
    recover: bool = False
    tag: Any = None
    started: bool = False  # the interlock let it move at least once
    stopped: _Job | None = None  # recover: the job that was running at the stop
    dropped: int = 0  # recover: queued jobs dropped by the stop


class Controller:
    """The only code that moves the plotter. Implements `api.PlotterController`."""

    def __init__(
        self,
        transport: Transport,
        config: MachineConfig | None = None,
        *,
        hands_clear: bool = False,
        clock: Callable[[], float] = now,
        on_event: Callable[[PlotEvent], None] | None = None,
        on_line: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.config = config or MachineConfig()
        self._t = transport
        self.clock = clock  # stamps events and times the hands-clear hold
        self.on_event = on_event  # called from the worker thread (and from stop())
        self.on_line = on_line
        self._cond = threading.Condition()
        self._jobs: deque[_Job] = deque()
        self._gen = 0
        self._state: State = "disconnected"
        self._progress = 0.0
        self._message = ""
        self._hands_clear = hands_clear
        self._clear_since = -math.inf if hands_clear else math.inf
        self._busy = False
        self._closing = False
        self._pen_down = False
        self._needs_home = False
        self._job: _Job | None = None  # the job the worker is running
        # Where the head will be when the queued jobs are done (None = unknown). Each job's
        # G-code starts from here, so its first move is split into short pieces too, and a
        # pause during it still takes effect within ~sync_every_mm (session VIRTUAL).
        self._end: Point | None = None
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Connect, send the startup block, and start the worker."""
        self._t.connect()
        for line in gcode.startup_block(self.config):
            self._t.send(check_line(line, phase="startup", config=self.config))
        self._t.wait_idle(timeout=self.config.ack_timeout_s)
        with self._cond:
            self._state = "idle"
            self._closing = False
        self._worker = threading.Thread(target=self._run, name="plotter-worker", daemon=True)
        self._worker.start()

    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        self._t.close()
        with self._cond:
            self._state = "disconnected"

    def __enter__(self) -> Controller:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ PlotterController

    def preview_box(self, box: Box, tag: Any = None) -> None:
        corners = [(box.x, box.y), (box.x2, box.y), (box.x2, box.y2), (box.x, box.y2)]
        d = self.config.drawable
        validate_points([corners], d, self.config.clamp_tolerance_mm)
        x0, y0 = max(box.x, d.x), max(box.y, d.y)  # clamps only sub-tolerance rounding
        x1, y1 = min(box.x2, d.x2), min(box.y2, d.y2)
        clamped = Box(x0, y0, x1 - x0, y1 - y0)
        lines = gcode.preview_box_gcode(clamped, self.config, start=self._end)
        self._submit("preview", lines, "previewing", tag)

    def draw_polylines(
        self, polylines: list[Polyline], speed: Speed = "normal", tag: Any = None
    ) -> None:
        self.draw_polylines_at(polylines, self.config.feed_for(speed), tag)

    def draw_polylines_at(
        self, polylines: Sequence[Sequence[Point]], feed: float, tag: Any = None
    ) -> None:
        """Like draw_polylines, with an explicit drawing feedrate in mm/min (for feed tests)."""
        clean = validate_polylines(polylines, self.config)
        lines = gcode.polylines_to_gcode(clean, self.config, feed, start=self._end)
        self._submit("draw", lines, "drawing", tag)

    def park(self, tag: Any = None) -> None:
        self._submit("park", gcode.park_gcode(self.config, start=self._end), "moving", tag)

    def home(self, tag: Any = None) -> None:
        self._submit("home", gcode.home_gcode(self.config), "moving", tag)

    def run_lines(self, lines: list[str], label: str = "lines", tag: Any = None) -> None:
        """Run raw G-code lines (whitelist and travel limits still apply). For bring-up tools."""
        self._submit(label, list(lines), "moving", tag)

    def stop(self) -> None:
        if self._state == "disconnected":
            return
        job = self._job
        self._emit("stop_request", job)
        try:
            self._t.quickstop()
        except Exception:  # noqa: BLE001 - keep going: the queue must still be dropped
            log.exception("quickstop failed")
        with self._cond:
            dropped = len([j for j in self._jobs if not j.recover])
            self._gen += 1
            self._jobs.clear()
            stopped = job.stopped if job is not None and job.recover else job
            self._jobs.append(
                _Job("recover", [], "stopped", self._gen, True, stopped=stopped, dropped=dropped)
            )
            self._state = "stopped"
            self._message = "stopped"
            self._needs_home = self.config.rehome_after_stop
            self._end = (0.0, 0.0) if self._needs_home else None
            self._cond.notify_all()

    def set_hands_clear(self, clear: bool) -> None:
        request = None
        with self._cond:
            if clear and not self._hands_clear:
                self._clear_since = self.clock()
            job = self._job
            if not clear and self._hands_clear and job and job.started and self._state != "paused":
                request = job
            self._hands_clear = clear
            self._cond.notify_all()
        if request is not None:
            self._emit("pause_request", request)

    def status(self) -> ControllerStatus:
        with self._cond:
            return ControllerStatus(self._state, self._progress, self._hands_clear, self._message)

    # ------------------------------------------------------------------ extras

    @property
    def busy(self) -> bool:
        """A job is queued or running."""
        with self._cond:
            return bool(self._jobs) or self._busy

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every queued job has finished. False on timeout."""
        deadline = None if timeout is None else now() + timeout
        with self._cond:
            while self._jobs or self._busy:
                remaining = None if deadline is None else deadline - now()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(0.05 if remaining is None else min(0.05, remaining))
        return True

    def query(self, line: str) -> list[str]:
        """Send a read-only or diagnostic line (M114, M115, M105) while idle; return the reply."""
        check_line(line, config=self.config)
        with self._cond:
            if self._jobs or self._busy:
                raise RuntimeError("the controller is busy")
        return self._t.request(line, timeout=self.config.ack_timeout_s)

    # ------------------------------------------------------------------ internals

    def _submit(self, label: str, lines: list[str], state: State, tag: Any = None) -> None:
        for line in lines:
            check_line(line, phase="job", config=self.config)
        lines = gcode.insert_syncs(lines, self.config.sync_every_mm, self.config.pen_threshold_z)
        with self._cond:
            if self._state == "disconnected":
                raise RuntimeError("not started; call start() first")
            if self._state == "error":
                raise LinkError(f"controller in error state: {self._message}")
            if self._needs_home:
                if lines[:1] != ["G28"]:
                    lines = gcode.home_gcode(self.config) + lines
                self._needs_home = False
            self._end = gcode.end_position(lines, self._end)
            self._jobs.append(_Job(label, lines, state, self._gen, tag=tag))
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._closing:
                    self._cond.wait()
                if self._closing:
                    return
                job = self._jobs.popleft()
                self._busy = True
                self._job = job
            try:
                if job.recover:
                    self._recover(job)
                else:
                    self._execute(job)
            except Exception as e:  # noqa: BLE001 - the worker must survive and report
                log.exception("job %s failed", job.label)
                with self._cond:
                    self._state = "error"
                    self._message = f"{job.label}: {e}"
                    self._jobs.clear()
                self._emit("error", job, message=str(e))
            finally:
                with self._cond:
                    self._busy = False
                    self._job = None
                    self._cond.notify_all()

    def _execute(self, job: _Job) -> None:
        with self._cond:
            if job.gen != self._gen:
                return
            self._state = job.state
            self._progress = 0.0
            self._message = job.label
        n = len(job.lines)
        try:
            for i, line in enumerate(job.lines):
                if not self._gate(job, line):
                    return
                self._send(line)
                with self._cond:
                    self._progress = (i + 1) / n
            # No overall deadline: a G28 or a long M400 can run past ack_timeout_s while the
            # firmware prints `busy: processing`. The link still fails after that much silence.
            self._t.wait_idle()
        except LinkStopped:
            return
        with self._cond:
            if job.gen != self._gen:
                return
            self._state = "idle"
            self._progress = 1.0
            self._message = f"{job.label} done"
        self._emit("job_end", job)

    def _may_move(self) -> bool:
        hold = self.config.hands_clear_hold_s
        return self._hands_clear and self.clock() - self._clear_since >= hold

    def _gate(self, job: _Job, next_line: str) -> bool:
        """Wait until motion is allowed. Lifts the pen while paused. False if stopped."""
        lifted = paused = False
        while True:
            with self._cond:
                if job.gen != self._gen or self._closing:
                    return False
                if self._may_move():
                    self._state = job.state
                    self._message = job.label
                    break
                if self._state != "paused":
                    log.info("paused: waiting for hands to clear")
                self._state = "paused"
                self._message = "waiting for hands to clear"
                announce = job.started and not paused
                paused = True
                need_lift = self._pen_down
                if not need_lift and not announce:
                    self._cond.wait(0.02)
                    continue
            if need_lift:
                self._send(gcode.pen_up(self.config))
                lifted = True
            if announce:  # report the pause once the moves already queued have run
                self._send("M400")
                self._t.wait_idle()
                self._emit("paused", job)
        if not job.started:
            job.started = True
            self._emit("job_start", job)
        elif paused:
            self._emit("resumed", job)
        if lifted and _is_draw_move(next_line):
            self._send(gcode.pen_down(self.config))
        return True

    def _send(self, line: str) -> None:
        self._t.send(line)  # the link times out on silence, not on a long busy command
        if self.on_line is not None:
            job = self._job
            self.on_line(line, job.tag if job is not None else None)
        cmd, p = parse_line(line)
        if cmd == "G28":
            self._pen_down = True  # Z homes to 0, which is at or below pen-down
        elif cmd in ("G0", "G1") and p.get("Z") is not None:
            self._pen_down = p["Z"] < self.config.pen_threshold_z

    def _recover(self, job: _Job) -> None:
        self._t.resync(timeout=self.config.ack_timeout_s)  # motion has ended once M410 is done
        self._emit("halted", job.stopped, dropped=job.dropped)
        self._send(gcode.pen_up(self.config))
        self._t.wait_idle(timeout=self.config.ack_timeout_s)
        with self._cond:
            if job.gen == self._gen:
                self._state = "stopped"
                self._message = "stopped; pen up"

    def _emit(self, kind: str, job: _Job | None, **detail: Any) -> None:
        if self.on_event is None:
            return
        label, tag = ("", None) if job is None or job.recover else (job.label, job.tag)
        try:
            self.on_event(PlotEvent(self.clock(), kind, label, tag, detail))
        except Exception:  # noqa: BLE001 - a listener must never stop the plotter
            log.exception("on_event failed for %s", kind)


def _is_draw_move(line: str) -> bool:
    cmd, p = parse_line(line)
    return cmd == "G1" and (p.get("X") is not None or p.get("Y") is not None)
