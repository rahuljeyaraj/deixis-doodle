"""The floor, as the vacuum app's camera sees it: four big ArUco markers (IDs 10 to 13, one
per A4 sheet, `scripts/make_markers.py`) give every pixel a place on the floor in mm.

The floor frame (the vacuum profile's "floor"): stand on the near side of the patch, facing
it. Marker 10 is near left (the origin), 11 near right, 12 far right, 13 far left; +x runs
to your right, +y away from you, in mm. Where the sheets lie is measured once with a tape
(each sheet's centre) and saved as a `FloorLayout` (`scripts/vacuum_floor.py`).

`FloorCamera` fits pixels -> floor mm from the markers' centres whenever all four are seen
(exact, and it doesn't matter which way each sheet is turned). It then remembers where each
marker's corners lie, so later frames with only two or three markers in view (a foot or the
robot on a sheet) still fit. The camera stands still, so with fewer markers the last fit is
kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from deixis.frames import apply_matrix
from pointdraw.layout import FLOOR_MARKER_IDS
from pointdraw.vision.aruco import BedCalibration, MarkerDetection, solve_bed_homography

DEFAULT_LAYOUT_PATH = Path(".cache/vacuum/floor.json")


@dataclass
class FloorLayout:
    """Each floor marker's centre in floor mm, measured with a tape."""

    centres_mm: dict[int, tuple[float, float]]

    @classmethod
    def rectangle(cls, width_mm: float, depth_mm: float) -> FloorLayout:
        """The four sheets' centres on the corners of a width x depth rectangle."""
        ids = FLOOR_MARKER_IDS
        pts = [(0.0, 0.0), (width_mm, 0.0), (width_mm, depth_mm), (0.0, depth_mm)]
        return cls(dict(zip(ids, pts, strict=True)))

    @property
    def area_mm(self) -> tuple[float, float, float, float]:
        """The rectangle the centres span: (x, y, w, h)."""
        xs = [p[0] for p in self.centres_mm.values()]
        ys = [p[1] for p in self.centres_mm.values()]
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def save(self, path: str | Path = DEFAULT_LAYOUT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "frame": "floor",
            "units": "mm",
            "marker_centres_mm": {str(k): list(v) for k, v in sorted(self.centres_mm.items())},
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LAYOUT_PATH) -> FloorLayout:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({int(k): (float(v[0]), float(v[1])) for k, v in d["marker_centres_mm"].items()})


@dataclass(frozen=True)
class FloorView:
    """Pixels <-> floor mm for one frame."""

    H: np.ndarray  # 3x3, pixels -> floor mm
    marker_ids: tuple[int, ...]  # the markers this fit used
    fresh: bool  # False: kept from an earlier frame
    t: float | None = None

    def to_mm(self, px: np.ndarray) -> np.ndarray:
        return apply_matrix(self.H, np.asarray(px, np.float64).reshape(-1, 2))

    def to_px(self, mm: np.ndarray) -> np.ndarray:
        return apply_matrix(np.linalg.inv(self.H), np.asarray(mm, np.float64).reshape(-1, 2))


@dataclass
class FloorCamera:
    layout: FloorLayout
    max_rms_mm: float = 25.0  # a corner fit worse than this is not used
    learned: dict[int, np.ndarray] = field(default_factory=dict)  # id -> corners (4, 2), mm
    last: FloorView | None = None

    def update(self, markers: list[MarkerDetection], t: float | None = None) -> FloorView | None:
        """The fit for this frame, or the last good one, or None before the first."""
        seen = {m.id: m for m in markers if m.id in self.layout.centres_mm}
        if len(seen) == len(self.layout.centres_mm) >= 4:
            ids = sorted(seen)
            px = np.array([seen[i].center_px for i in ids], np.float64)
            mm = np.array([self.layout.centres_mm[i] for i in ids], np.float64)
            H, _ = cv2.findHomography(px, mm, 0)
            if H is not None:
                for i in ids:
                    self.learned[i] = apply_matrix(H, seen[i].corners_px)
                self.last = FloorView(H, tuple(ids), True, t)
                return self.last
        if len(seen) >= 2 and all(i in self.learned for i in seen):
            calib = BedCalibration(marker_corners_mm={i: self.learned[i] for i in seen})
            h = solve_bed_homography(list(seen.values()), calib, min_markers=2, ransac_mm=20.0)
            if h is not None and h.rms_mm <= self.max_rms_mm:
                self.last = FloorView(h.H, h.marker_ids, True, t)
                return self.last
        if self.last is not None:
            return FloorView(self.last.H, self.last.marker_ids, False, self.last.t)
        return None
