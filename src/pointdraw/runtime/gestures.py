"""From hands to gestures: holds, size gestures and the hands-clear interlock.

Input is one `HandsFrame` per camera frame, with landmarks in bed mm (the vision layer's
output). Everything here works in the **paper frame** (bed mm minus the paper's corner,
`pointdraw.layout`), which is what the resolver, the scene and the planner use.

- **Holds** (plan 4.3): the index fingertip of the one pointing hand, smoothed with a
  One-Euro filter, into `deixis.holds.HoldDetector`. With two hands up, no place holds are
  made (two hands show a size, not a place). A second hand away from the paper (resting on
  the table) doesn't count: on the Ender (1 Oct 2026) one did, and "here" found no point.
- **Size gestures:** one hand with the thumb out gives a pinch span (thumb tip to index
  tip); two hands give the distance between the index tips. Both go into
  `deixis.spans.SpanHoldDetector`.
- **Interlock** (plan 4.5): a hand counts as over the paper if any landmark is inside the
  paper plus `margin_mm`. Not clear after `enter_frames` such frames in a row; clear again
  only after `exit_frames` frames without one (so a one-frame tracking dropout never
  releases the plotter). The controller adds its own hold time before it moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from deixis.filters import OneEuroFilter2D
from deixis.geometry import Box, Point
from deixis.holds import HoldDetector, HoldParams
from deixis.resolver import SpanHold
from deixis.spans import SpanHoldDetector, SpanParams
from deixis.types import Hold
from pointdraw.vision.hands import INDEX_TIP, THUMB_TIP, HandObservation, HandsFrame

WRIST, INDEX_MCP, MIDDLE_MCP = 0, 5, 9


@dataclass(frozen=True)
class GestureParams:
    margin_mm: float = 10.0  # interlock: paper plus this margin counts as "over the paper"
    enter_frames: int = 2  # frames over the paper before the interlock trips
    exit_frames: int = 3  # frames clear before it releases
    thumb_out_ratio: float = 0.6  # |thumb tip - index MCP| / |wrist - middle MCP|
    # Kept 2 min, not 30 s: a "How big?" answered 40 s later still needs the request's point
    # (live run, 1 Oct 2026). Old holds never match new words: assignment goes by word time.
    holds: HoldParams = field(default_factory=lambda: HoldParams(history_s=120.0))
    spans: SpanParams = field(default_factory=SpanParams)


def thumb_out(obs: HandObservation, ratio: float = 0.6) -> bool:
    """True if the thumb is held away from the hand, as when showing a size."""
    lm = obs.world if obs.world is not None else obs.landmarks_mm
    if lm is None:
        return False
    lm = np.asarray(lm, dtype=np.float64)
    scale = float(np.linalg.norm(lm[WRIST] - lm[MIDDLE_MCP])) + 1e-9
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP])) / scale > ratio


@dataclass
class GestureSample:
    """What the tracker saw in one frame (for timelines and replays)."""

    t: float
    tip: Point | None  # smoothed pointing fingertip, paper mm
    pointing: bool
    n_hands: int
    over_paper: bool
    span: float | None
    span_kind: str | None


class GestureTracker:
    """Feed `update()` every frame; read `holds`, `spans` and `hands_clear`."""

    def __init__(self, paper: Box, params: GestureParams | None = None) -> None:
        self.paper = paper  # the paper in the bed frame; its corner is the paper origin
        self.params = params or GestureParams()
        self.holds = HoldDetector(self.params.holds)
        self.spans = SpanHoldDetector(self.params.spans)
        self._filter = OneEuroFilter2D()
        self._over_run = 0
        self._clear_run = 0
        self.hands_clear = True
        self.last: GestureSample | None = None
        self.ended_hold: Hold | None = None  # a hold that ended on the last frame
        self.ended_span: SpanHold | None = None

    # ------------------------------------------------------------------ frames

    def to_paper(self, p_bed: np.ndarray) -> np.ndarray:
        return np.asarray(p_bed, dtype=np.float64) - np.array([self.paper.x, self.paper.y])

    def update(self, frame: HandsFrame) -> GestureSample:
        t = frame.t
        hands = [h for h in frame.hands if h.landmarks_mm is not None]
        on = [h for h in hands if self._over_paper(h)]
        over = bool(on)
        self._interlock(over)
        if len(hands) > 1:  # of several, only hands over the paper point or pinch
            hands = on

        tip: Point | None = None
        pointing = False
        pointers = [h for h in hands if h.pointing]
        if len(hands) == 1 and pointers:
            raw = self.to_paper(pointers[0].landmarks_mm[INDEX_TIP])  # type: ignore[index]
            tip = self._filter(t, (float(raw[0]), float(raw[1])))
            pointing = True
        else:
            self._filter.reset()
        self.ended_hold = self.holds.update(t, tip, pointing)

        span, kind, center = self._span(hands)
        self.ended_span = self.spans.update(t, span, kind, center)  # type: ignore[arg-type]
        self.last = GestureSample(t, tip, pointing, len(hands), over, span, kind)
        return self.last

    def _span(self, hands: list[HandObservation]) -> tuple[float | None, str | None, Point]:
        if len(hands) >= 2 and all(h.pointing for h in hands[:2]):
            a = self.to_paper(hands[0].landmarks_mm[INDEX_TIP])  # type: ignore[index]
            b = self.to_paper(hands[1].landmarks_mm[INDEX_TIP])  # type: ignore[index]
            c = (a + b) / 2
            return float(np.linalg.norm(a - b)), "two_hands", (float(c[0]), float(c[1]))
        if len(hands) == 1 and thumb_out(hands[0], self.params.thumb_out_ratio):
            lm = self.to_paper(hands[0].landmarks_mm)  # type: ignore[arg-type]
            c = (lm[THUMB_TIP] + lm[INDEX_TIP]) / 2
            span = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
            return span, "pinch", (float(c[0]), float(c[1]))
        return None, None, (0.0, 0.0)

    def _over_paper(self, h: HandObservation) -> bool:
        m = self.params.margin_mm
        lm = self.to_paper(h.landmarks_mm)  # type: ignore[arg-type]
        inside = (
            (lm[:, 0] >= -m)
            & (lm[:, 0] <= self.paper.w + m)
            & (lm[:, 1] >= -m)
            & (lm[:, 1] <= self.paper.h + m)
        )
        return bool(inside.any())

    def _interlock(self, over: bool) -> None:
        p = self.params
        if over:
            self._over_run += 1
            self._clear_run = 0
            if self._over_run >= p.enter_frames:
                self.hands_clear = False
        else:
            self._clear_run += 1
            self._over_run = 0
            if self._clear_run >= p.exit_frames:
                self.hands_clear = True

    # ------------------------------------------------------------------ queries

    def recent_holds(self, t_from: float = -math.inf, t_to: float = math.inf) -> list[Hold]:
        return self.holds.holds(t_from, t_to)
