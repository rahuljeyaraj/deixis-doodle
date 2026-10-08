"""Where the person stands: MediaPipe Pose's heels and toes, mapped onto the floor.

Each foot is the middle of its heel and toe (landmarks 29 to 32, when MediaPipe can see
them). The person's spot is the middle of both feet. Both feet seen counts as standing
(`Sample.pointing`, which the core's hold rule asks for); one foot only (mid-step, or one
hidden) still gives a point, not a standing one. Heels and toes touch the floor, so the
floor fit maps them without parallax.

The model file (Apache-2.0) is downloaded once to `.cache/models/pose_landmarker_full.task`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


MAX_WALK_MM_S = 2500.0  # nobody crosses a room faster than this; a jump beyond it is the
# tracker losing the person and finding a chair leg, not a person moving
RECOVER_S = 0.8  # with nothing believed for this long, take whatever comes next
SMOOTH = 5  # frames in the median: about 0.17 s at 30 fps


@dataclass
class SteadySpot:
    """Hold the dot still: drop impossible jumps, and smooth what is left.

    MediaPipe is given one frame at a time and will happily put a foot on a chair leg for a
    few frames. 7 Oct, live: the dot "jumps away most of the time". Each new spot is checked
    against the last believed one -- faster than a person can walk means the tracker lost the
    person, so the spot is dropped rather than drawn somewhere wrong. After `recover_s` with
    nothing believed, the next spot is taken on trust, so a person who really did walk out
    and back is picked up again.
    """

    max_speed_mm_s: float = MAX_WALK_MM_S
    recover_s: float = RECOVER_S
    smooth: int = SMOOTH
    _kept: list[tuple[float, np.ndarray, tuple[float, float]]] = field(default_factory=list)
    dropped: int = 0
    offered: int = 0

    def offer(
        self, t: float, spot_px: np.ndarray | None, spot_mm: tuple[float, float] | None
    ) -> tuple[np.ndarray | None, tuple[float, float] | None, bool]:
        """The spot to believe this frame, and whether this frame's own spot was taken."""
        if spot_px is None or spot_mm is None:
            return (*self._best(), False)
        self.offered += 1
        if self._kept:
            t0, _, last_mm = self._kept[-1]
            gap = max(t - t0, 1e-3)
            if gap < self.recover_s and math.dist(spot_mm, last_mm) / gap > self.max_speed_mm_s:
                self.dropped += 1
                return (*self._best(), False)
        self._kept.append((t, np.asarray(spot_px, np.float64), spot_mm))
        del self._kept[: -self.smooth]
        return (*self._best(), True)

    def _best(self) -> tuple[np.ndarray | None, tuple[float, float] | None]:
        if not self._kept:
            return None, None
        px = np.median(np.array([p for _, p, _ in self._kept]), axis=0)
        mm = np.median(np.array([m for _, _, m in self._kept]), axis=0)
        return px, (round(float(mm[0]), 1), round(float(mm[1]), 1))


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
        min_visibility: float = 0.6,
        steady: bool = True,
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
        self.steady = SteadySpot() if steady else None
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
        seen = feet_seen(t, px, vis, floor, self.min_visibility)
        return self._steadied(seen)

    def _steadied(self, seen: FeetSeen) -> FeetSeen:
        """The same frame with an impossible jump dropped and the spot smoothed."""
        if self.steady is None:
            return seen
        spot_px, spot_mm, took = self.steady.offer(seen.t, seen.spot_px, seen.spot_mm)
        if spot_px is None:
            return FeetSeen(seen.t, seen.feet_px, None, None, False)
        # Standing means both feet seen *this* frame. A held or dropped spot is not standing:
        # the core's hold rule must never be told the person is still when we have lost them.
        return FeetSeen(seen.t, seen.feet_px, spot_px, spot_mm, seen.standing and took)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FeetTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
