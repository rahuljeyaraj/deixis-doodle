"""Floor to map: where the robot's own map lies on the camera's floor.

The robot reports its position in its map (mm). The camera sees a marker on the robot's lid
(ID 14, `layout.ROBOT_MARKER_ID`, printed top toward the robot's front) and maps it onto the
floor. Pairs of the two fit a similarity (turn, shift, scale, and a mirror if the map is
mirrored) with the deixis core's `fit_transform`. Redo it each session, or whenever the
robot rebuilds its map (`scripts/vacuum_calibrate.py`).

Two ways to get pairs:
- `PathMatcher` (the usual one): let the robot clean across the floor patch. The camera's
  track of the lid is matched to the robot's reports, which arrive a little late; the lag
  that gives the tightest fit is kept.
- `PairCollector`: the robot stands still (docked, paused) in a few places 30 cm apart.

The lid's middle carries the LiDAR turret, so the marker sits ahead of the robot's centre:
`RobotMarker.offset_mm` (measure it). The marker's own direction is the robot's front, so the
centre is found on the floor before the map is known.

Known error: the lid is about 10 cm above the floor, and the floor fit is for the floor
itself, so a camera that does not look straight down sees the marker a little too far from
the point under the camera. Fitting a similarity absorbs most of it (it is a scaling about
that point) but leaves the scale about 4 % off at a camera height of 2.5 m (10 cm / 2.5 m).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deixis.frames import Transform
from pointdraw.layout import ROBOT_MARKER_ID

DEFAULT_MAP_PATH = Path(".cache/vacuum/floor_to_map.json")

Pair = tuple[tuple[float, float], tuple[float, float]]  # (floor mm, map mm)


@dataclass(frozen=True)
class RobotMarker:
    id: int = ROBOT_MARKER_ID
    offset_mm: float = 0.0  # the marker's centre is this far ahead of the robot's centre

    def centre(self, corners_mm: np.ndarray) -> tuple[float, float]:
        """The robot's centre on the floor, from the marker's 4 corners (TL, TR, BR, BL)."""
        c = np.asarray(corners_mm, np.float64)
        mid = c.mean(axis=0)
        front = (c[0] + c[1]) / 2 - (c[2] + c[3]) / 2  # printed bottom -> top
        n = float(np.linalg.norm(front))
        if n > 0:
            mid = mid - front / n * self.offset_mm
        return (float(mid[0]), float(mid[1]))


@dataclass
class PairCollector:
    """Takes a (floor, map) pair when the camera has seen the robot still for `still_s` and
    the robot's report is fresh (asked for with `locate`)."""

    still_s: float = 1.0
    still_mm: float = 20.0  # the robot's marker moved less than this
    min_apart_mm: float = 300.0  # a new pair must be this far from the others
    max_age_s: float = 1.5  # the robot's report must be this fresh
    pairs: list[Pair] = field(default_factory=list)
    _seen: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=600))

    def see(self, t: float, floor_xy: tuple[float, float] | None) -> None:
        if floor_xy is not None:
            self._seen.append((t, floor_xy[0], floor_xy[1]))

    def still_at(self, t: float) -> tuple[float, float] | None:
        """Where the robot stands, if the camera saw it still for the last `still_s`."""
        recent = [(x, y) for ts, x, y in self._seen if ts >= t - self.still_s]
        span = [ts for ts, _, _ in self._seen if ts >= t - self.still_s]
        if len(recent) < 5 or span[0] > t - self.still_s * 0.8:
            return None
        pts = np.array(recent)
        mid = pts.mean(axis=0)
        if float(np.max(np.linalg.norm(pts - mid, axis=1))) > self.still_mm:
            return None
        return (float(mid[0]), float(mid[1]))

    def offer(self, t: float, report: dict[str, Any]) -> Pair | None:
        """The robot's status (`locate`): a new pair, or None (moving, stale, too close)."""
        floor = self.still_at(t)
        robot = report.get("robot")
        if floor is None or not robot or robot.get("age_s", math.inf) > self.max_age_s:
            return None
        if any(math.dist(floor, f) < self.min_apart_mm for f, _ in self.pairs):
            return None
        pair: Pair = (floor, (float(robot["x"]), float(robot["y"])))
        self.pairs.append(pair)
        return pair

    def spread_mm(self) -> float:
        """How far apart the pairs are (the larger side of their bounding box, floor)."""
        if len(self.pairs) < 2:
            return 0.0
        pts = np.array([f for f, _ in self.pairs])
        return float(np.max(pts.max(axis=0) - pts.min(axis=0)))

    def enough(self, n: int = 4, spread_mm: float = 1000.0) -> bool:
        return len(self.pairs) >= n and self.spread_mm() >= spread_mm


@dataclass
class PathMatcher:
    """Pairs from a robot on the move: the camera's track of its lid against the robot's own
    position reports.

    A report says where the robot was a moment before it arrived; that lag is not known, so
    `best` tries lags from 0 to 3 s and keeps the one whose fit (by the core) is tightest. A
    report is used only if the camera saw the robot within `max_gap_s` either side of it."""

    max_gap_s: float = 0.25
    track: list[tuple[float, float, float]] = field(default_factory=list)  # t, floor x, y
    reports: list[tuple[float, float, float]] = field(default_factory=list)  # t, map x, y

    def see(self, t: float, floor_xy: tuple[float, float] | None) -> None:
        if floor_xy is not None:
            self.track.append((t, floor_xy[0], floor_xy[1]))

    def report(self, t_now: float, status: dict[str, Any]) -> bool:
        """A `status()` reply: keep its robot position if it is a new report."""
        robot = status.get("robot")
        if not robot:
            return False
        t = t_now - float(robot.get("age_s", 0.0))
        rec = (t, float(robot["x"]), float(robot["y"]))
        if self.reports and abs(self.reports[-1][0] - t) < 0.05 and self.reports[-1][1:] == rec[1:]:
            return False  # the same report, polled again
        self.reports.append(rec)
        return True

    def floor_at(self, t: float) -> tuple[float, float] | None:
        """The camera's robot position at time t (between two frames close to it)."""
        ts = [p[0] for p in self.track]
        i = int(np.searchsorted(ts, t))
        if i == 0 or i == len(ts):
            return None
        (t0, x0, y0), (t1, x1, y1) = self.track[i - 1], self.track[i]
        if t - t0 > self.max_gap_s or t1 - t > self.max_gap_s:
            return None
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)

    def pairs(self, lag_s: float) -> list[Pair]:
        out: list[Pair] = []
        for t, mx, my in self.reports:
            f = self.floor_at(t - lag_s)
            if f is not None:
                out.append((f, (mx, my)))
        return out

    def spread_mm(self) -> float:
        if not self.track:
            return 0.0
        pts = np.array([(x, y) for _, x, y in self.track])
        return float(np.max(pts.max(axis=0) - pts.min(axis=0)))

    def best(
        self, client: Any, lags: Sequence[float] | None = None, min_pairs: int = 8
    ) -> tuple[float, list[Pair], float] | None:
        """(lag, its pairs, rms mm) of the tightest fit, or None if too few pairs."""
        best: tuple[float, list[Pair], float] | None = None
        for lag in lags if lags is not None else np.arange(0.0, 3.01, 0.1):
            ps = self.pairs(float(lag))
            if len(ps) < min_pairs:
                continue
            tr = client.fit_transform(
                [f for f, _ in ps], [m for _, m in ps], "similarity", allow_mirror=True
            )
            rms = float(tr.rms_mm or 0.0)
            if best is None or rms < best[2]:
                best = (float(lag), ps, rms)
        return best


@dataclass
class FloorToMap:
    transform: Transform
    pairs: list[Pair]
    created: str = ""

    @classmethod
    def fit(cls, pairs: list[Pair], client: Any) -> FloorToMap:
        """Fit with the core (a `DeixisClient`, over MCP in the app)."""
        if len(pairs) < 3:
            raise ValueError(f"need at least 3 pairs, got {len(pairs)}")
        tr = client.fit_transform(
            [f for f, _ in pairs],
            [m for _, m in pairs],
            "similarity",
            "floor",
            "map",
            allow_mirror=True,
        )
        return cls(tr, list(pairs), dt.datetime.now().isoformat(timespec="seconds"))

    def save(self, path: str | Path = DEFAULT_MAP_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "transform": self.transform.model_dump(mode="json"),
            "pairs": [{"floor_mm": list(f), "map_mm": list(m)} for f, m in self.pairs],
            "created": self.created,
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MAP_PATH) -> FloorToMap:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pairs = [(tuple(p["floor_mm"]), tuple(p["map_mm"])) for p in d.get("pairs", [])]
        return cls(Transform.model_validate(d["transform"]), pairs, d.get("created", ""))  # type: ignore[arg-type]

    @classmethod
    def load_or_none(cls, path: str | Path = DEFAULT_MAP_PATH) -> FloorToMap | None:
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None
