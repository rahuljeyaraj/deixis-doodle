"""Bed markers: detect ArUco stickers and solve the pixel-to-bed-mm homography every frame.

The markers' bed coordinates are not fixed in advance. Calibration (`vision.calibration`)
learns each marker's four corner positions in bed mm from crosses the plotter draws, and
saves them as a `BedCalibration`. At run time any visible markers with known corners give
point pairs (pixel corner, bed-mm corner) and a homography is fitted to them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pointdraw import layout
from pointdraw.vision.lens import LensModel

DEFAULT_CALIBRATION_PATH = Path(".cache/calibration/bed.json")


@dataclass(frozen=True)
class MarkerDetection:
    id: int
    corners_px: np.ndarray  # (4, 2) float, ArUco order: TL, TR, BR, BL as printed

    @property
    def center_px(self) -> np.ndarray:
        return self.corners_px.mean(axis=0)


def make_detector(dictionary: str = layout.ARUCO_DICTIONARY) -> cv2.aruco.ArucoDetector:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(d, p)


_DETECTOR: cv2.aruco.ArucoDetector | None = None


def detect_markers(
    image: np.ndarray,
    ids: tuple[int, ...] | None = layout.BED_MARKER_IDS,
    detector: cv2.aruco.ArucoDetector | None = None,
) -> list[MarkerDetection]:
    """Detect markers in a BGR or grey image. `ids=None` keeps every ID in the dictionary."""
    global _DETECTOR
    if detector is None:
        _DETECTOR = _DETECTOR or make_detector()
        detector = _DETECTOR
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, found, _ = detector.detectMarkers(gray)
    out: list[MarkerDetection] = []
    if found is None:
        return out
    for c, i in zip(corners, found.ravel(), strict=True):
        if ids is None or int(i) in ids:
            out.append(MarkerDetection(int(i), c.reshape(4, 2).astype(np.float64)))
    return sorted(out, key=lambda m: m.id)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


# --- calibration file ------------------------------------------------------------------------


@dataclass
class BedCalibration:
    """What calibration learned: marker corners in bed mm, plus the fit it came from."""

    marker_corners_mm: dict[int, np.ndarray] = field(default_factory=dict)  # id -> (4, 2)
    H_px_to_mm: np.ndarray | None = None  # from crosses, for the calibration image only
    image_size: tuple[int, int] | None = None  # (width, height) of the calibration image
    cross_rms_mm: float | None = None
    cross_max_mm: float | None = None
    created: str = ""
    notes: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "version": 1,
            "frame": "bed",
            "units": "mm",
            "marker_size_mm": layout.BED_MARKER_SIZE_MM,
            "marker_corners_mm": {
                str(k): np.round(v, 4).tolist() for k, v in sorted(self.marker_corners_mm.items())
            },
            "H_px_to_mm": None if self.H_px_to_mm is None else self.H_px_to_mm.tolist(),
            "image_size": self.image_size,
            "cross_rms_mm": self.cross_rms_mm,
            "cross_max_mm": self.cross_max_mm,
            "created": self.created,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict[str, object]) -> BedCalibration:
        corners = {
            int(k): np.asarray(v, dtype=np.float64).reshape(4, 2)
            for k, v in dict(d.get("marker_corners_mm") or {}).items()  # type: ignore[arg-type]
        }
        H = d.get("H_px_to_mm")
        size = d.get("image_size")
        return cls(
            marker_corners_mm=corners,
            H_px_to_mm=None if H is None else np.asarray(H, dtype=np.float64),
            image_size=None if size is None else tuple(size),  # type: ignore[arg-type]
            cross_rms_mm=d.get("cross_rms_mm"),  # type: ignore[arg-type]
            cross_max_mm=d.get("cross_max_mm"),  # type: ignore[arg-type]
            created=str(d.get("created", "")),
            notes=dict(d.get("notes") or {}),  # type: ignore[arg-type]
        )

    def save(self, path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION_PATH) -> BedCalibration:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_none(cls, path: str | Path = DEFAULT_CALIBRATION_PATH) -> BedCalibration | None:
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None


# --- per-frame homography --------------------------------------------------------------------


@dataclass(frozen=True)
class BedHomography:
    """Pixel-to-bed-mm mapping for one frame. `to_mm` and `to_px` use *raw* image pixels;
    with a lens model, H itself maps undistorted pixels."""

    H: np.ndarray  # 3x3, (undistorted) pixels -> bed mm
    marker_ids: tuple[int, ...]
    rms_mm: float  # reprojection error of the marker corners, in mm
    max_mm: float
    rms_px: float  # the same, measured back in the image
    t: float | None = None
    lens: LensModel | None = None

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if self.lens is not None:
            pts = self.lens.undistort(pts)
        return apply_homography(self.H, pts)

    def to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        pts = apply_homography(self.H_inv, pts_mm)
        return self.lens.distort(pts) if self.lens is not None else pts


def solve_bed_homography(
    detections: list[MarkerDetection],
    calib: BedCalibration,
    min_markers: int = 2,
    ransac_mm: float = 1.0,
    t: float | None = None,
    lens: LensModel | None = None,
) -> BedHomography | None:
    """Fit pixels -> bed mm from the visible markers whose corners calibration knows.

    One marker (4 corners) is enough in principle but is poorly conditioned across the sheet,
    so the default needs two. With three or more markers RANSAC drops a bad detection.
    """
    known = [d for d in detections if d.id in calib.marker_corners_mm]
    if len(known) < max(1, min_markers):
        return None
    px = np.concatenate([d.corners_px for d in known])
    if lens is not None:
        px = lens.undistort(px)
    mm = np.concatenate([calib.marker_corners_mm[d.id] for d in known])
    if len(known) >= 3:
        H, mask = cv2.findHomography(px, mm, cv2.RANSAC, ransac_mm)
        if H is not None and mask is not None and mask.sum() >= 8:
            inl = mask.ravel().astype(bool)
            H, _ = cv2.findHomography(px[inl], mm[inl], 0)
    else:
        H, _ = cv2.findHomography(px, mm, 0)
    if H is None:
        return None
    err_mm = np.linalg.norm(apply_homography(H, px) - mm, axis=1)
    err_px = np.linalg.norm(apply_homography(np.linalg.inv(H), mm) - px, axis=1)
    return BedHomography(
        H=H,
        marker_ids=tuple(d.id for d in known),
        rms_mm=float(np.sqrt(np.mean(err_mm**2))),
        max_mm=float(err_mm.max()),
        rms_px=float(np.sqrt(np.mean(err_px**2))),
        t=t,
        lens=lens,
    )


def fixed_bed_homography(
    calib: BedCalibration, t: float | None = None, lens: LensModel | None = None
) -> BedHomography | None:
    """The calibration photo's cross fit, used as is: right only while the camera stays put.

    The lens is applied only if the fit was made on undistorted pixels.
    """
    if calib.H_px_to_mm is None:
        return None
    rms = calib.cross_rms_mm or 0.0
    return BedHomography(
        H=calib.H_px_to_mm,
        marker_ids=(),
        rms_mm=rms,
        max_mm=calib.cross_max_mm if calib.cross_max_mm is not None else rms,
        rms_px=float("nan"),
        t=t,
        lens=lens if calib.notes.get("undistorted") else None,
    )


class BedTracker:
    """Keeps the latest good bed homography, so a hand briefly covering markers is harmless.

    A new solution is accepted when its corner RMS is under `max_rms_mm`. The last good one is
    reused for up to `max_age_s` seconds.

    A calibration that learned fewer than `min_markers` markers (a bed with no room for them)
    has nothing to track: the cross fit is used for every frame, so the camera must not move
    and frames must be taken with the bed parked, as at calibration.
    """

    def __init__(
        self,
        calib: BedCalibration,
        min_markers: int = 2,
        max_rms_mm: float = 1.0,
        max_age_s: float = 2.0,
        lens: LensModel | None = None,
    ) -> None:
        self.calib = calib
        self.lens = lens
        self.min_markers = min_markers
        self.max_rms_mm = max_rms_mm
        self.max_age_s = max_age_s
        self.last: BedHomography | None = None

    def update(self, detections: list[MarkerDetection], t: float) -> BedHomography | None:
        if len(self.calib.marker_corners_mm) < max(1, self.min_markers):
            self.last = fixed_bed_homography(self.calib, t, self.lens)
            return self.last
        h = solve_bed_homography(detections, self.calib, self.min_markers, t=t, lens=self.lens)
        if h is not None and h.rms_mm <= self.max_rms_mm:
            self.last = h
            return h
        if self.last is not None and self.last.t is not None and t - self.last.t <= self.max_age_s:
            return self.last
        return None
