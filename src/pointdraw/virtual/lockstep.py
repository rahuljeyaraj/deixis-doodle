"""The real `controller.Controller`, run in simulated time, for the virtual world.

`VirtualPlotter` models the controller; this runs the controller itself (its worker thread,
interlock gate, stop and recovery) behind `LivePlotter`, the adapter the live app uses. So
the scenes check the code that drives the machine, not a copy of it.

Time is a shared clock that only `tick(t)` moves. The transport, `SteppedTransport`, lets a
line go once the head has finished the one before at the current time, and says how long
each move takes (distance / feedrate, as in `SimulatedPlotter`). Moves that end inside a
frame are followed straight on, so drawing is continuous in simulated time.

To keep runs repeatable, `tick(t)` waits until the worker thread has nothing left to do at
the old time, moves the clock, and waits again. "Nothing left to do" means the worker is
parked: waiting for a job, held by the interlock, or waiting for the head. The controller's
condition is swapped for one that reports when the worker parks (test harness only).

Differences from the machine (and from `VirtualPlotter`): the planner holds one line, not
16, so a pause takes effect within one 2.5 mm piece. With `emergency_parser` a stop cuts the
move in progress at once; without it the move ends first. Either way the ink list keeps the
whole of the last piece.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from pointdraw.controller.config import MachineConfig
from pointdraw.controller.controller import Controller
from pointdraw.controller.safety import parse_line
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkStopped
from pointdraw.runtime.live_plotter import LivePlotter

WORKER = "plotter-worker"  # the controller's worker thread
SETTLE_TIMEOUT_S = 10.0  # wall time; a worker that never parks is a bug


class Lockstep:
    """The simulated clock, and whether the worker thread is parked."""

    def __init__(self) -> None:
        self.t = 0.0
        self.t_prev = -math.inf  # the previous tick
        self._cv = threading.Condition()
        self._parked = False
        self._seq = 0  # times the worker has parked

    def clock(self) -> float:
        return self.t

    def advance(self, t: float) -> None:
        self.t_prev, self.t = self.t, t

    def park(self) -> None:
        with self._cv:
            self._seq += 1
            self._parked = True
            self._cv.notify_all()

    def unpark(self) -> None:
        with self._cv:
            self._parked = False

    def seq(self) -> int:
        with self._cv:
            return self._seq

    def wait_parked(self, after: int) -> None:
        """Block until the worker has parked again since `seq() == after`."""
        with self._cv:
            ok = self._cv.wait_for(lambda: self._parked and self._seq > after, SETTLE_TIMEOUT_S)
        if not ok:
            raise RuntimeError("the controller's worker did not settle")


class ParkingCondition(threading.Condition):
    """A condition that tells the `Lockstep` when the worker thread waits on it."""

    def __init__(self, ls: Lockstep) -> None:
        super().__init__()
        self.ls = ls

    def wait(self, timeout: float | None = None) -> bool:
        if threading.current_thread().name != WORKER:
            return super().wait(timeout)
        self.ls.park()
        try:
            return super().wait(timeout)
        finally:
            self.ls.unpark()


class SteppedTransport:
    """A `Transport` whose moves take simulated time on the `Lockstep` clock."""

    def __init__(
        self,
        config: MachineConfig,
        ls: Lockstep,
        *,
        emergency_parser: bool = False,
        home_s: float = 4.0,
    ) -> None:
        self.config = config
        self.ls = ls
        self.emergency_parser = emergency_parser
        self.home_s = home_s
        self.sim = SimulatedPlotter(config)
        self.cond = ParkingCondition(ls)
        self.head_t = 0.0  # when the head finishes the lines sent so far
        self.move: tuple[float, float, tuple, tuple] | None = None  # t0, t1, from, to
        self.stopped = False

    def connect(self) -> None:
        self.sim.connect()

    def close(self) -> None:
        self.sim.close()

    def send(self, line: str, timeout: float | None = None) -> None:
        with self.cond:
            self._wait_head()
            a = self.sim.position
            self.sim.send(line)
            b = self.sim.position
            cmd, p = parse_line(line)
            if cmd == "G28":
                dur = self.home_s
            elif cmd in ("G0", "G1"):
                dur = math.dist(a, b) / ((p.get("F") or 1000.0) / 60.0)
            else:
                dur = 0.0
            # Straight on from the last move if it ended in this frame, else from now.
            t0 = self.head_t if self.head_t > self.ls.t_prev else self.ls.t
            self.head_t = t0 + dur
            self.move = (t0, self.head_t, a, b)

    def request(self, line: str, timeout: float | None = None) -> list[str]:
        with self.cond:
            return self.sim.request(line)

    def wait_idle(self, timeout: float | None = None) -> None:
        with self.cond:
            self._wait_head()

    def quickstop(self) -> None:
        with self.cond:
            self.stopped = True
            m = self.move
            if self.emergency_parser and m is not None and m[1] > self.ls.t:
                t0, t1, a, b = m
                u = (self.ls.t - t0) / (t1 - t0) if t1 > t0 else 1.0
                u = min(max(u, 0.0), 1.0)
                self.sim.x, self.sim.y, self.sim.z = (
                    p + (q - p) * u for p, q in zip(a, b, strict=True)
                )
                self.head_t = self.ls.t
                self.move = (t0, self.ls.t, a, self.sim.position)
            self.cond.notify_all()

    def resync(self, timeout: float | None = None) -> None:
        with self.cond:
            self._wait_head(stoppable=False)
            self.stopped = False

    def _wait_head(self, stoppable: bool = True) -> None:
        while self.head_t > self.ls.t + 1e-9 and not (stoppable and self.stopped):
            self.cond.wait()
        if stoppable and self.stopped:
            raise LinkStopped("stopped; call resync() first")


class LockstepPlotter(LivePlotter):
    """`LivePlotter` over a real `Controller` and a `SteppedTransport`, with `tick(t)`."""

    def __init__(
        self,
        config: MachineConfig | None = None,
        *,
        emergency_parser: bool = False,
        start_parked: bool = True,
    ) -> None:
        config = config or MachineConfig()
        self.ls = Lockstep()
        self.transport = SteppedTransport(config, self.ls, emergency_parser=emergency_parser)
        ctl = Controller(self.transport, config, clock=self.ls.clock)
        ctl._cond = ParkingCondition(self.ls)  # before start(): the worker waits on it
        super().__init__(ctl, stamp=lambda: self.transport.head_t)
        if start_parked:  # like VirtualPlotter
            x, y, z = config.park_x, config.park_y, config.pen_up_z
            sim = self.transport.sim
            sim.x, sim.y, sim.z = x, y, z
            ctl._end = (x, y)
            self.assume_at(x, y, z)

    @property
    def t(self) -> float:
        return self.ls.t

    def start(self) -> None:
        super().start()
        self._settle()

    def tick(self, t: float) -> None:
        """Let the worker finish at the old time, then move the clock to `t`."""
        self._settle()
        if t > self.ls.t:
            self.ls.advance(t)
            self._settle()

    def stop(self) -> None:
        super().stop()
        self._settle()

    def _settle(self) -> None:
        if self.ctl._worker is None or not self.ctl._worker.is_alive():
            return
        seq = self.ls.seq()
        for cond in (self.ctl._cond, self.transport.cond):
            with cond:
                cond.notify_all()
        self.ls.wait_parked(seq)

    def __enter__(self) -> LockstepPlotter:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
