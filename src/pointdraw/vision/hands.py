"""Hands: MediaPipe Hand Landmarker -> fingertips, pointing pose and pinch spans in bed mm.

MediaPipe Tasks API, VIDEO running mode, up to 2 hands. The model file (Apache-2.0) is
downloaded once to `.cache/models/hand_landmarker.task`.

Pose classification uses MediaPipe's 3D *world* landmarks (metres, hand-centred), so it
does not depend on how the overhead camera sees the hand. Positions (fingertip, spans) use
the 2D image landmarks mapped through the bed homography.

Not done yet (see the VIS report): height and parallax correction for hands held above the
paper. Spans and fingertips are measured as if the hand touched the paper.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(".cache/models/hand_landmarker.task")

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]  # fmt: skip

PixelsToMm = Callable[[np.ndarray], np.ndarray]


# --- pose ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseParams:
    """Thresholds for finger states. A finger is scored on two cues:

    - reach: |wrist -> tip| / |wrist -> PIP|, above ~1.2 when straight, below ~1.0 curled
    - bend: total bend at the PIP and DIP joints, near 0 degrees when straight
    """

    extended_reach: float = 1.12
    extended_bend_deg: float = 60.0
    curled_reach: float = 1.02
    curled_bend_deg: float = 90.0
    min_curled_of_ring_pinky: int = 1  # the middle finger must curl; ring or pinky may lag
    # A fingertip pressed on the paper bends the index a little, and seen from straight above
    # it is foreshortened: neither straight nor curled. On the Ender (1 Oct 2026) that was most
    # frames of a held point. Count it, as long as the index still reaches past the middle.
    half_index_ok: bool = True
    # Pressed harder, the index hooks (PIP + DIP bend ~115 degrees) and reads as curled, but it
    # still reaches well past the curled middle finger: on the Ender (1 Oct 2026, live run) a
    # hooked point was 0.15..0.40 further in reach, a closed or resting hand about 0.05. Count
    # a hook as pointing when the index reaches this much further (0 turns it off).
    hooked_index_margin: float = 0.15


DEFAULT_POSE = PoseParams()


@dataclass(frozen=True)
class FingerState:
    reach: float
    bend_deg: float
    extended: bool
    curled: bool


@dataclass(frozen=True)
class PoseResult:
    pointing: bool
    fingers: dict[str, FingerState] = field(default_factory=dict)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def finger_state(lm: np.ndarray, finger: str, p: PoseParams = DEFAULT_POSE) -> FingerState:
    mcp, pip, dip, tip = (lm[i] for i in FINGERS[finger])
    wrist = lm[WRIST]
    reach = float(np.linalg.norm(tip - wrist) / (np.linalg.norm(pip - wrist) + 1e-12))
    bend = _angle(pip - mcp, dip - pip) + _angle(dip - pip, tip - dip)
    return FingerState(
        reach=reach,
        bend_deg=bend,
        extended=reach >= p.extended_reach and bend <= p.extended_bend_deg,
        curled=reach <= p.curled_reach or bend >= p.curled_bend_deg,
    )


def classify_pointing(lm: np.ndarray, p: PoseParams = DEFAULT_POSE) -> PoseResult:
    """Pointing = index extended (or, with `half_index_ok`, not curled and reaching further
    than the middle finger; or, with `hooked_index_margin`, hooked but reaching that much
    further), middle curled, and ring or pinky curled.

    `lm` is (21, 3) world landmarks (preferred) or (21, 2) image landmarks.
    The thumb is ignored: people point with the thumb in or out.
    """
    lm = np.asarray(lm, dtype=np.float64)
    fingers = {f: finger_state(lm, f, p) for f in FINGERS}
    others = sum(fingers[f].curled for f in ("ring", "pinky"))
    index, middle = fingers["index"], fingers["middle"]
    half = p.half_index_ok and not index.curled and index.reach > middle.reach
    hook = 0 < p.hooked_index_margin <= index.reach - middle.reach
    pointing = (
        (index.extended or half or hook) and middle.curled and others >= p.min_curled_of_ring_pinky
    )
    return PoseResult(pointing=pointing, fingers=fingers)


# --- per-frame results -----------------------------------------------------------------------


@dataclass(frozen=True)
class HandObservation:
    handedness: str  # MediaPipe's label ("Left"/"Right"), assuming an unmirrored image
    score: float
    landmarks_px: np.ndarray  # (21, 2) raw image pixels
    landmarks_mm: np.ndarray | None  # (21, 2) bed mm, if a homography was given
    world: np.ndarray | None  # (21, 3) metres, hand-centred
    pose: PoseResult

    @property
    def pointing(self) -> bool:
        return self.pose.pointing

    @property
    def index_tip_px(self) -> tuple[float, float]:
        return tuple(self.landmarks_px[INDEX_TIP])  # type: ignore[return-value]

    @property
    def index_tip_mm(self) -> tuple[float, float] | None:
        if self.landmarks_mm is None:
            return None
        return tuple(self.landmarks_mm[INDEX_TIP])  # type: ignore[return-value]

    @property
    def pinch_px(self) -> float:
        return float(np.linalg.norm(self.landmarks_px[THUMB_TIP] - self.landmarks_px[INDEX_TIP]))

    @property
    def pinch_mm(self) -> float | None:
        if self.landmarks_mm is None:
            return None
        return float(np.linalg.norm(self.landmarks_mm[THUMB_TIP] - self.landmarks_mm[INDEX_TIP]))


@dataclass(frozen=True)
class HandsFrame:
    t: float
    hands: list[HandObservation]
    image_size: tuple[int, int]  # (width, height)

    @property
    def has_mm(self) -> bool:
        return bool(self.hands) and self.hands[0].landmarks_mm is not None

    def pointing_hand(self) -> HandObservation | None:
        """The most confident hand in a pointing pose, if any."""
        cands = [h for h in self.hands if h.pointing]
        return max(cands, key=lambda h: h.score) if cands else None

    def primary(self) -> HandObservation | None:
        """The pointing hand, else the most confident hand."""
        return self.pointing_hand() or (
            max(self.hands, key=lambda h: h.score) if self.hands else None
        )

    @property
    def two_hand_span_px(self) -> float | None:
        if len(self.hands) < 2:
            return None
        a, b = self.hands[0].landmarks_px[INDEX_TIP], self.hands[1].landmarks_px[INDEX_TIP]
        return float(np.linalg.norm(a - b))

    @property
    def two_hand_span_mm(self) -> float | None:
        if len(self.hands) < 2 or not self.has_mm:
            return None
        a, b = self.hands[0].landmarks_mm, self.hands[1].landmarks_mm
        assert a is not None and b is not None
        return float(np.linalg.norm(a[INDEX_TIP] - b[INDEX_TIP]))


def observations_from_result(
    result: Any,
    image_size: tuple[int, int],
    to_mm: PixelsToMm | None = None,
    pose: PoseParams = DEFAULT_POSE,
) -> list[HandObservation]:
    """Convert a MediaPipe HandLandmarkerResult (or anything shaped like one)."""
    w, h = image_size
    out: list[HandObservation] = []
    worlds = list(getattr(result, "hand_world_landmarks", None) or [])
    handed = list(getattr(result, "handedness", None) or [])
    for i, lms in enumerate(result.hand_landmarks or []):
        px = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float64)
        world = (
            np.array([(lm.x, lm.y, lm.z) for lm in worlds[i]], dtype=np.float64)
            if i < len(worlds) and worlds[i]
            else None
        )
        cat = handed[i][0] if i < len(handed) and handed[i] else None
        label = getattr(cat, "category_name", None) or "Unknown"
        score = float(getattr(cat, "score", 1.0) or 1.0)
        mm = to_mm(px) if to_mm is not None else None
        pose_input = world if world is not None else np.array([(lm.x, lm.y, lm.z) for lm in lms])
        out.append(
            HandObservation(label, score, px, mm, world, classify_pointing(pose_input, pose))
        )
    return out


# --- tracker ---------------------------------------------------------------------------------


def ensure_model(path: str | Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the hand landmarker model if it is not already there (about 7.8 MB)."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https URL)
    tmp.replace(path)
    return path


class HandTracker:
    """MediaPipe Hand Landmarker in VIDEO mode. Call `process()` once per frame, in order."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_detection: float = 0.5,
        min_presence: float = 0.5,
        min_tracking: float = 0.5,
        pose: PoseParams = DEFAULT_POSE,
    ) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        path = ensure_model(model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self.pose = pose
        self._last_ms = -1

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def process(
        self, image_bgr: np.ndarray, t: float, to_mm: PixelsToMm | None = None
    ) -> HandsFrame:
        """Track hands in one BGR frame taken at time `t` (seconds on pointdraw.clock)."""
        import cv2
        import mediapipe as mp

        ms = max(int(round(t * 1000)), self._last_ms + 1)  # VIDEO mode needs increasing ms
        self._last_ms = ms
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms
        )
        size = (image_bgr.shape[1], image_bgr.shape[0])
        return HandsFrame(t, observations_from_result(result, size, to_mm, self.pose), size)

    def close(self) -> None:
        self._landmarker.close()
