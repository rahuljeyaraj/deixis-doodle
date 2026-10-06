"""The robot's own reported positions, logged while the app runs (measure list M24).

`RobotWatch` asks the vacuum's server for `status` every `every_s` in a background thread
(the server answers from what the robot last reported, so this costs nothing on the robot),
and keeps an event each time the robot reports a new position or a new state:
`{"kind": "robot", "state", "x", "y"}` in the robot's map mm. The camera loop moves them
into the run's timeline with `drain()`, so `scripts/measure_vacuum.py` can compare them with
the box that was sent. The robot reports a position only every few seconds while it moves
(about 5 in 17 s on 3 Oct).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pointdraw import clock


class RobotWatch:
    def __init__(
        self, link: Any, every_s: float = 0.5, now: Callable[[], float] = clock.now
    ) -> None:
        self.link = link
        self.every_s = every_s
        self.now = now
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last: tuple[Any, Any] | None = None  # (state, position) last kept
        self._thread = threading.Thread(target=self._run, name="robot-watch", daemon=True)

    def start(self) -> RobotWatch:
        self._thread.start()
        return self

    def poll(self) -> None:
        """One status call; keeps an event if the state or the position changed."""
        try:
            st = self.link.status()
        except Exception as e:  # noqa: BLE001 - a lost link is logged by the app
            self._keep({"t": self.now(), "kind": "robot_error", "error": str(e)})
            return
        r = st.get("robot") or {}
        pos = (r.get("x"), r.get("y")) if r else None
        key = (st.get("state"), pos)
        if key == self._last:
            return
        self._last = key
        t = self.now() - float(r.get("age_s") or 0.0) if r else self.now()
        ev = {"t": round(t, 3), "kind": "robot", "state": st.get("state")}
        if pos is not None:
            ev |= {"x": pos[0], "y": pos[1]}
        self._keep(ev)

    def _keep(self, ev: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(ev)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self._events = self._events, []
        return out

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            self.poll()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(5)
