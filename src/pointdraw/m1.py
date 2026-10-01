"""Milestone M1: "a cross under the finger" (plan section 5.1).

The user points at the parked paper and holds still; the plotter draws a small cross at the
fingertip. This module is the camera-free core: feed it one fingertip sample per frame and it
detects holds, drives the controller's interlock, queues the crosses and scores them against
printed targets. `scripts/cross_under_finger.py` adds the camera, MediaPipe and a window.

Positions are bed-frame mm (= machine XY), which is what the bed calibration produces.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from deixis.filters import OneEuroFilter2D
from deixis.geometry import Box, Point
from deixis.holds import HoldDetector, HoldParams
from deixis.types import Hold
from pointdraw.controller.api import PlotterController
from pointdraw.controller.safety import SafetyError
from pointdraw.controller.shapes import cross

# M1 exit test (plan 5.0): 10 trials, median error <= 3 mm, max <= 5 mm.
PASS_MEDIAN_MM = 3.0
PASS_MAX_MM = 5.0

# A deliberate hold: longer than the resolver's 150 ms, so passing hands don't draw crosses.
M1_HOLD = HoldParams(min_duration_s=0.6, frame="bed")

Trigger = Literal["hold", "key"]


@dataclass(frozen=True)
class Target:
    """A printed target, in bed mm."""

    id: int
    x: float
    y: float


def load_targets(path: str | Path, paper: Box) -> list[Target]:
    """Read `hardware/print/m1-targets.json` (paper-frame mm) and move it onto the bed.

    The paper frame is the bed frame shifted to the sheet's bottom-left corner (see
    `pointdraw.layout`), so this assumes the sheet is taped square at `paper` (the controller
    config's paper rectangle, measured at bring-up).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Target(int(t["id"]), paper.x + float(t["x_mm"]), paper.y + float(t["y_mm"]))
        for t in data["targets"]
    ]


def over_bed(landmarks_mm, bed: Box, margin_mm: float = 20.0) -> bool:
    """True if any hand landmark (bed mm) is over the bed, or within `margin_mm` of it.

    A hand or foot elsewhere in the picture must not pause the plotter; one reaching in must.
    `None` (no camera-to-bed mapping) counts as over the bed, to stay safe.
    """
    if landmarks_mm is None:
        return True
    lm = np.asarray(landmarks_mm, dtype=np.float64).reshape(-1, 2)
    inside = (
        (lm[:, 0] >= bed.x - margin_mm)
        & (lm[:, 0] <= bed.x2 + margin_mm)
        & (lm[:, 1] >= bed.y - margin_mm)
        & (lm[:, 1] <= bed.y2 + margin_mm)
    )
    return bool(inside.any())


def nearest_target(p: Point, targets: list[Target]) -> tuple[Target, float] | None:
    if not targets:
        return None
    best = min(targets, key=lambda t: math.hypot(t.x - p[0], t.y - p[1]))
    return best, math.hypot(best.x - p[0], best.y - p[1])


@dataclass
class Trial:
    """One hold that asked for a cross."""

    t: float
    x: float
    y: float
    hold_s: float
    spread_mm: float
    drawn: bool
    reason: str = ""
    target_id: int | None = None
    error_mm: float | None = None  # commanded cross to nearest target: pointing + vision error


@dataclass
class Summary:
    trials: int
    drawn: int
    scored: int
    median_mm: float | None
    max_mm: float | None

    @property
    def passed(self) -> bool | None:
        if not self.scored or self.median_mm is None or self.max_mm is None:
            return None
        return self.median_mm <= PASS_MEDIAN_MM and self.max_mm <= PASS_MAX_MM

    def text(self) -> str:
        if not self.scored:
            return f"{self.drawn}/{self.trials} crosses drawn (no targets to score against)"
        verdict = "PASS" if self.passed else "not yet"
        return (
            f"{self.drawn}/{self.trials} crosses drawn; error to nearest target over "
            f"{self.scored}: median {self.median_mm:.2f} mm, max {self.max_mm:.2f} mm -> "
            f"{verdict} (M1 needs median <= {PASS_MEDIAN_MM:g}, max <= {PASS_MAX_MM:g})"
        )


@dataclass
class CrossUnderFinger:
    """Holds in, crosses out.

    Call `update()` once per camera frame. With `trigger="hold"` a cross is queued as soon as
    a hold ends (the finger moves away or leaves the view); with `trigger="key"` the latest
    hold is kept until `fire()` is called. The controller only moves once hands have been out
    of view for its `hands_clear_hold_s`, so the user points, withdraws the hand, and the
    plotter draws, then parks again.
    """

    ctl: PlotterController
    cross_mm: float = 6.0
    trigger: Trigger = "hold"
    targets: list[Target] = field(default_factory=list)
    hold_params: HoldParams = M1_HOLD
    park_after: bool = True
    trials: list[Trial] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.smoother = OneEuroFilter2D()
        self.holds = HoldDetector(self.hold_params)
        self.pending: Hold | None = None  # key mode: the hold waiting for fire()
        self._clear: bool | None = None

    def update(
        self, t: float, tip_mm: Point | None, pointing: bool, hand_in_view: bool
    ) -> Trial | None:
        """One frame. `tip_mm` is the index fingertip in bed mm (None if not tracked)."""
        clear = not hand_in_view
        if clear != self._clear:
            self.ctl.set_hands_clear(clear)
            self._clear = clear
        if tip_mm is None:
            self.smoother.reset()
            ended = self.holds.update(t, None)
        else:
            ended = self.holds.update(t, self.smoother(t, tip_mm), pointing)
        if ended is None:
            return None
        if self.trigger == "key":
            self.pending = ended
            return None
        return self._draw(ended)

    def fire(self) -> Trial | None:
        """Key mode: draw at the latest hold (the one in progress, else the last one ended)."""
        hold = self.holds.current() or self.pending
        self.pending = None
        return self._draw(hold) if hold is not None else None

    def current(self) -> Hold | None:
        return self.holds.current()

    def _draw(self, hold: Hold) -> Trial:
        x, y = hold.mean
        trial = Trial(
            t=hold.t1,
            x=round(x, 2),
            y=round(y, 2),
            hold_s=round(hold.t1 - hold.t0, 3),
            spread_mm=round(max(hold.spread), 2),
            drawn=False,
        )
        near = nearest_target((x, y), self.targets)
        if near is not None:
            trial.target_id, err = near[0].id, near[1]
            trial.error_mm = round(err, 2)
        try:
            self.ctl.draw_polylines(cross((x, y), self.cross_mm), "normal")
        except SafetyError as e:
            trial.reason = f"refused: {e}"
        else:
            trial.drawn = True
            if self.park_after:
                self.ctl.park()
        self.trials.append(trial)
        return trial

    def summary(self) -> Summary:
        errs = [t.error_mm for t in self.trials if t.drawn and t.error_mm is not None]
        return Summary(
            trials=len(self.trials),
            drawn=sum(t.drawn for t in self.trials),
            scored=len(errs),
            median_mm=statistics.median(errs) if errs else None,
            max_mm=max(errs) if errs else None,
        )

    def write_log(self, path: str | Path) -> Path:
        """Append the trials as JSON lines."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for tr in self.trials:
                f.write(json.dumps(asdict(tr)) + "\n")
        return p
