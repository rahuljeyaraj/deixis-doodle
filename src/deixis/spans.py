"""Steady size gestures ("about this big"): a pinch span or a two-hand span held still.

Feed one span sample per camera frame. A `SpanHold` is a stretch where the span changes
slower than `max_rate_mm_s` for at least `min_duration_s`. Changing kind (pinch to two
hands) or losing the gesture ends it. Mirrors `deixis.holds.HoldDetector`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from deixis.geometry import Point
from deixis.resolver import SpanHold, SpanKind


@dataclass(frozen=True)
class SpanParams:
    max_rate_mm_s: float = 25.0  # span change rate below which the gesture counts as held
    window_s: float = 0.10  # rate = change over this window / time
    min_duration_s: float = 0.20
    min_span_mm: float = 8.0  # smaller spans are a closed pinch, not a size
    max_gap_s: float = 0.12  # dropouts shorter than this are bridged
    history_s: float = 30.0


class SpanHoldDetector:
    """Streaming detector for steady spans. Call from one thread."""

    def __init__(self, params: SpanParams | None = None) -> None:
        self.params = params or SpanParams()
        self._track: deque[tuple[float, float]] = deque()
        self._seg: list[tuple[float, float, Point]] = []
        self._kind: SpanKind | None = None
        self._last_t: float | None = None
        self._history: deque[SpanHold] = deque()

    def update(
        self, t: float, span: float | None, kind: SpanKind | None = None, center: Point = (0, 0)
    ) -> SpanHold | None:
        """One sample; `span` None when no size gesture is shown. Returns a hold that ended."""
        p = self.params
        if span is None or span < p.min_span_mm or kind is None:
            if self._last_t is None or t - self._last_t > p.max_gap_s:
                self._track.clear()
                return self._end()
            return None
        ended = None
        if kind != self._kind or (self._last_t is not None and t - self._last_t > p.max_gap_s):
            self._track.clear()
            ended = self._end()
            self._kind = kind
        self._last_t = t
        self._track.append((t, span))
        while len(self._track) > 2 and self._track[1][0] <= t - p.window_s:
            self._track.popleft()
        t_ref, s_ref = self._track[0]
        dt = t - t_ref
        if dt < 0.5 * p.window_s:
            return ended
        if abs(span - s_ref) / dt < p.max_rate_mm_s:
            self._seg.append((t, span, center))
            return ended
        return self._end() or ended

    def flush(self) -> SpanHold | None:
        return self._end()

    def holds(self, t_from: float = -math.inf, t_to: float = math.inf) -> list[SpanHold]:
        out = [h for h in self._history if h.t1 >= t_from and h.t0 <= t_to]
        cur = self._make(self._seg) if self._seg else None
        if cur is not None and cur.t1 >= t_from and cur.t0 <= t_to:
            out.append(cur)
        return out

    def _make(self, seg: list[tuple[float, float, Point]]) -> SpanHold | None:
        if self._kind is None or seg[-1][0] - seg[0][0] < self.params.min_duration_s:
            return None
        spans = sorted(s for _, s, _ in seg)
        n = len(seg)
        cx = sum(c[0] for _, _, c in seg) / n
        cy = sum(c[1] for _, _, c in seg) / n
        return SpanHold(seg[0][0], seg[-1][0], spans[n // 2], self._kind, (cx, cy))

    def _end(self) -> SpanHold | None:
        if not self._seg:
            return None
        h = self._make(self._seg)
        self._seg = []
        if h is not None:
            self._history.append(h)
            while self._history and self._history[0].t1 < h.t1 - self.params.history_s:
                self._history.popleft()
        return h
