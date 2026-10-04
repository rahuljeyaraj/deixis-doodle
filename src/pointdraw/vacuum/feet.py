"""Where the person stands: MediaPipe Pose's heels and toes, mapped onto the floor.

Each foot is the middle of its heel and toe (landmarks 29 to 32, when MediaPipe can see
them). The person's spot is the middle of both feet. Both feet seen counts as standing
(`Sample.pointing`, which the core's hold rule asks for); one foot only (mid-step, or one
hidden) still gives a point, not a standing one. Heels and toes touch the floor, so the
floor fit maps them without parallax.

The model file (Apache-2.0) is downloaded once to `.cache/models/pose_landmarker_full.task`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deixis.types import Sample
from pointdraw.vacuum.floor import FloorView
from pointdraw.vision.hands import ensure_model

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path(".cache/models/pose_landmarker_full.task")

LEFT_HEEL, RIGHT_HEEL, LEFT_TOE, RIGHT_TOE = 29, 30, 31, 32
FEET = ((LEFT_HEEL, LEFT_TOE), (RIGHT_HEEL, RIGHT_TOE))


@dataclass(frozen=True)
class FeetSeen:
    """One frame's feet: pixels, and the spot on the floor if the floor fit is known."""

    t: float
    feet_px: tuple[np.ndarray, ...]  # one (2,) point per foot seen
    spot_px: np.ndarray | None
    spot_mm: tuple[float, float] | None
    standing: bool

    def sample(self) -> Sample:
        return Sample(self.t, self.spot_mm, self.standing)


def feet_from_landmarks(
    px: np.ndarray, visibility: np.ndarray, min_visibility: float = 0.5
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]:
    """The feet seen (heel and toe midpoints, pixels) and their middle, from 33 landmarks."""
    feet = []
    for heel, toe in FEET:
        pts = [px[i] for i in (heel, toe) if visibility[i] >= min_visibility]
        if pts:
            feet.append(np.mean(pts, axis=0))
    if not feet:
        return (), None
    return tuple(feet), np.mean(feet, axis=0)


def feet_seen(
    t: float,
    px: np.ndarray | None,
    visibility: np.ndarray | None,
    floor: FloorView | None,
    min_visibility: float = 0.5,
) -> FeetSeen:
    if px is None or visibility is None:
        return FeetSeen(t, (), None, None, False)
    feet, spot = feet_from_landmarks(px, visibility, min_visibility)
    spot_mm = None
    if spot is not None and floor is not None:
        x, y = floor.to_mm(spot)[0]
        spot_mm = (round(float(x), 1), round(float(y), 1))
    return FeetSeen(t, feet, spot, spot_mm, len(feet) == 2)


class FeetTracker:
    """MediaPipe Pose Landmarker in VIDEO mode, one person. `process()` once per frame."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        min_detection: float = 0.5,
        min_presence: float = 0.5,
        min_tracking: float = 0.5,
        min_visibility: float = 0.5,
    ) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        path = ensure_model(model_path, MODEL_URL)
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.min_visibility = min_visibility
        self._last_ms = -1

    def process(self, image_bgr: np.ndarray, t: float, floor: FloorView | None) -> FeetSeen:
        import cv2
        import mediapipe as mp

        ms = max(int(round(t * 1000)), self._last_ms + 1)  # VIDEO mode needs increasing ms
        self._last_ms = ms
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms
        )
        if not result.pose_landmarks:
            return feet_seen(t, None, None, floor)
        h, w = image_bgr.shape[:2]
        lms = result.pose_landmarks[0]
        px = np.array([[p.x * w, p.y * h] for p in lms], np.float64)
        vis = np.array([p.visibility if p.visibility is not None else 1.0 for p in lms])
        return feet_seen(t, px, vis, floor, self.min_visibility)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FeetTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
