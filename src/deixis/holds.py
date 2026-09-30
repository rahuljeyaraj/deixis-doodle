"""Streaming hold detector (plan section 4.3).

A hold is a stretch of time when a tracked point (normally the index fingertip) is in a
pointing pose and moving slower than about 20 mm/s for at least about 150 ms. Holds are what
the resolver pairs with demonstrative words ("here", "this", "that").

Feed it one sample per camera frame with `update()`. It returns a `Hold` when one *ends*;
`current()` gives the hold still in progress, so a word spoken mid-hold can use it.
Positions should already be smoothed (see `deixis.filters`).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from deixis.geometry import Point
from deixis.types import Hold


@dataclass(frozen=True)
class HoldParams:
    """Hold detection parameters. Units: mm and seconds. All are starting points to tune."""

    speed_enter_mm_s: float = 20.0  # below this a hold can start
    speed_exit_mm_s: float = 30.0  # above this an ongoing hold ends (hysteresis)
    min_duration_s: float = 0.15  # shorter still periods are not holds
    speed_window_s: float = 0.10  # speed = displacement over this window / time
    max_gap_s: float = 0.12  # dropouts or lost pose longer than this end a hold
    require_pointing: bool = True
    history_s: float = 30.0  # how long finished holds are kept for queries
    frame: str = "paper"
    source: str = "index_tip"


class HoldDetector:
    """Streaming hold detector. Not thread-safe; call from one thread."""

    def __init__(self, params: HoldParams | None = None) -> None:
        self.params = params or HoldParams()
        self._track: deque[tuple[float, Point]] = deque()
        self._seg: list[tuple[float, Point]] = []
        self._last_ok_t: float | None = None
        self._history: deque[Hold] = deque()
        self.speed: float | None = None  # last speed estimate, mm/s

    # --- streaming -------------------------------------------------------------------------

    def update(self, t: float, p: Point | None, pointing: bool = True) -> Hold | None:
        """Add one sample. `p` is None when the point was not tracked in this frame.

        Returns the hold that ended with this sample, if any.
        """
        prm = self.params
        ok = p is not None and (pointing or not prm.require_pointing)
        if not ok:
            self.speed = None
            if self._last_ok_t is None or t - self._last_ok_t > prm.max_gap_s:
                self._track.clear()
                return self._end_segment()
            return None

        assert p is not None
        p = (float(p[0]), float(p[1]))
        if self._last_ok_t is not None and t - self._last_ok_t > prm.max_gap_s:
            # Came back after a long dropout: the old track and segment are stale.
            self._track.clear()
            ended = self._end_segment()
        else:
            ended = None
        self._last_ok_t = t
        self._track.append((t, p))
        while len(self._track) > 2 and self._track[1][0] <= t - prm.speed_window_s:
            self._track.popleft()

        self.speed = self._speed(t, p)
        threshold = prm.speed_exit_mm_s if self._seg else prm.speed_enter_mm_s
        still = self.speed is not None and self.speed < threshold
        if still:
            self._seg.append((t, p))
            return ended
        return self._end_segment() or ended

    def flush(self) -> Hold | None:
        """End any hold in progress (for example at the end of an utterance)."""
        return self._end_segment()

    def reset(self) -> None:
        self._track.clear()
        self._seg.clear()
        self._last_ok_t = None
        self.speed = None

    # --- queries ---------------------------------------------------------------------------

    def current(self) -> Hold | None:
        """The hold in progress, if it is already long enough to count."""
        if not self._seg:
            return None
        return self._make_hold(self._seg)

    def holds(
        self, t_from: float = -math.inf, t_to: float = math.inf, include_current: bool = True
    ) -> list[Hold]:
        """Finished holds (and optionally the current one) overlapping [t_from, t_to]."""
        out = [h for h in self._history if h.t1 >= t_from and h.t0 <= t_to]
        cur = self.current() if include_current else None
        if cur is not None and cur.t1 >= t_from and cur.t0 <= t_to:
            out.append(cur)
        return out

    # --- internals -------------------------------------------------------------------------

    def _speed(self, t: float, p: Point) -> float | None:
        if len(self._track) < 2:
            return None
        t_ref, p_ref = self._track[0]
        dt = t - t_ref
        if dt < 0.5 * self.params.speed_window_s or dt <= 0:
            return None
        return math.hypot(p[0] - p_ref[0], p[1] - p_ref[1]) / dt

    def _make_hold(self, seg: list[tuple[float, Point]]) -> Hold | None:
        t0, t1 = seg[0][0], seg[-1][0]
        if t1 - t0 < self.params.min_duration_s:
            return None
        n = len(seg)
        mx = sum(q[0] for _, q in seg) / n
        my = sum(q[1] for _, q in seg) / n
        sx = math.sqrt(sum((q[0] - mx) ** 2 for _, q in seg) / n)
        sy = math.sqrt(sum((q[1] - my) ** 2 for _, q in seg) / n)
        return Hold(
            t0=t0,
            t1=t1,
            mean=(mx, my),
            spread=(sx, sy),
            frame=self.params.frame,
            source=self.params.source,
        )

    def _end_segment(self) -> Hold | None:
        if not self._seg:
            return None
        hold = self._make_hold(self._seg)
        self._seg = []
        if hold is not None:
            self._history.append(hold)
            while self._history and self._history[0].t1 < hold.t1 - self.params.history_s:
                self._history.popleft()
        return hold
