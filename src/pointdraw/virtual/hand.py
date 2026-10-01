"""The virtual hand: fingertip keyframes -> `HandsFrame`s, as MediaPipe would give them.

Scene files describe the hand as keyframes in **paper mm** (see `tests/scenes/README.md`).
They are expanded here to one `HandsFrame` per camera frame (23 fps), with landmarks in
**bed mm** so the runtime's own paper/bed conversion is exercised. Everything after
MediaPipe is real code: the pose classifier reads these world landmarks, and the One-Euro
filter, hold detector, span detector and interlock read the 2D landmarks.

Realism added: per-frame jitter, occasional one-frame dropouts, and (for a hand held above
the paper) the parallax an overhead camera sees: points shift away from the camera's nadir
and spans look bigger by Z / (Z - h).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from deixis.geometry import Box, Point
from pointdraw.vision.hands import (
    INDEX_TIP,
    THUMB_TIP,
    HandObservation,
    HandsFrame,
    classify_pointing,
)

FPS = 23.0

# Hand geometry in metres (as in tests/vision/test_hands.py): MCP positions and the
# proximal, middle and distal segment lengths. Fingers point along +y.
_MCP = {"index": (0.030, 0.090), "middle": (0.010, 0.095), "ring": (-0.010, 0.090),
        "pinky": (-0.030, 0.080)}  # fmt: skip
_SEG = {"index": (0.045, 0.025, 0.020), "middle": (0.050, 0.030, 0.022),
        "ring": (0.045, 0.028, 0.020), "pinky": (0.035, 0.020, 0.018)}  # fmt: skip
_ORDER = ("index", "middle", "ring", "pinky")
STRAIGHT = (0.0, 0.0, 0.0)
CURLED = (80.0, 100.0, 60.0)

POSES = ("point", "hooked", "pinch", "open", "fist")
HOOKED = (20.0, 70.0, 40.0)  # the index pressed on the paper, as rahul points (1 Oct 2026)


def hand_model(pose: str) -> np.ndarray:
    """21 world landmarks (metres) for a pose: point, hooked (a point pressed on the paper,
    the index bent), pinch (thumb out), open or fist."""
    flex = {
        "point": {"index": STRAIGHT},
        "hooked": {"index": HOOKED},
        "pinch": {"index": STRAIGHT},
        "open": {f: STRAIGHT for f in _ORDER},
        "fist": {},
    }[pose]
    lm = np.zeros((21, 3))
    lm[1] = (0.025, 0.020, 0.0)
    lm[2] = (0.045, 0.040, 0.0)
    if pose in ("pinch", "open"):
        lm[3], lm[4] = (0.070, 0.055, 0.0), (0.095, 0.065, 0.0)
    else:
        lm[3], lm[4] = (0.050, 0.060, -0.015), (0.040, 0.075, -0.030)
    for k, finger in enumerate(_ORDER):
        base = 5 + 4 * k
        p = np.array([*_MCP[finger], 0.0])
        lm[base] = p
        angle = 0.0
        for seg, bend in zip(_SEG[finger], flex.get(finger, CURLED), strict=True):
            angle += math.radians(bend)
            p = p + seg * np.array([0.0, math.cos(angle), -math.sin(angle)])
            base += 1
            lm[base] = p
    if pose in ("point", "hooked", "fist"):
        # The thumb tucked onto the curled middle finger, as recorded on the Ender (its tip
        # 0.02..0.07 of the palm length from the middle fingertip while pointing).
        lm[4] = lm[12] + (0.004, 0.0, 0.0)
        lm[3] = (lm[2] + lm[4]) / 2
    return lm


_MODELS = {p: hand_model(p) for p in POSES}
_POSE_OK = {p: classify_pointing(m) for p, m in _MODELS.items()}


@dataclass(frozen=True)
class HandState:
    present: bool = False
    tip: Point = (0.0, 0.0)  # index fingertip, paper mm
    pose: str = "point"
    pinch_mm: float | None = None  # thumb-to-index span when pose == "pinch"
    two_mm: float | None = None  # second hand's index tip is this far to the left
    height_mm: float = 0.0  # fingertip height above the paper
    rest: Point | None = None  # a second hand resting here (a fist), paper mm; stays on leave


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    a: HandState
    b: HandState  # tip, pinch and two-hand span are interpolated from a to b


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))


def expand(keys: list[dict[str, Any]], default_height: float = 0.0) -> list[Segment]:
    """Turn scene-file hand keyframes into timed segments."""
    segs: list[Segment] = []
    tc = 0.0
    st = HandState(height_mm=default_height)

    def still(until: float) -> None:
        nonlocal tc
        if until > tc:
            segs.append(Segment(tc, until, st, st))
            tc = until

    for k in keys:
        if "at" in k:
            still(float(k["at"]))
        if "height" in k:
            st = replace(st, height_mm=float(k["height"]))
        if "pose" in k:
            if k["pose"] not in POSES:
                raise ValueError(f"unknown pose {k['pose']!r}; use one of {POSES}")
            st = replace(st, pose=k["pose"], pinch_mm=None if k["pose"] != "pinch" else st.pinch_mm)
        if "enter" in k:
            x, y = k["enter"]
            st = replace(st, present=True, tip=(float(x), float(y)))
        if "pinch" in k:
            st = replace(st, pose="pinch", pinch_mm=float(k["pinch"]))
        if "two_hands" in k:
            st = replace(st, two_mm=float(k["two_hands"]))
        if k.get("one_hand"):
            st = replace(st, two_mm=None)
        if "rest" in k:
            r = k["rest"]
            st = replace(st, rest=None if r is False else (float(r[0]), float(r[1])))
        if "move" in k:
            x, y = map(float, k["move"])
            dist = math.dist(st.tip, (x, y))
            dur = float(k.get("dur", max(0.3, dist / 200.0)))
            nxt = replace(st, present=True, tip=(x, y))
            segs.append(Segment(tc, tc + dur, st, nxt))
            tc += dur
            st = nxt
        if k.get("leave"):
            dur = float(k.get("dur", 0.4))
            nxt = replace(st, tip=(st.tip[0], -80.0), two_mm=None)
            segs.append(Segment(tc, tc + dur, st, nxt))
            tc += dur
            st = HandState(height_mm=default_height, rest=st.rest)
        for key in ("hold", "wait"):
            if key in k:
                still(tc + float(k[key]))
    segs.append(Segment(tc, math.inf, st, st))
    return segs


def state_at(segs: list[Segment], t: float) -> HandState:
    for s in segs:
        if s.t0 <= t < s.t1:
            if s.a == s.b or not s.a.present:
                return s.a
            u = _ease((t - s.t0) / (s.t1 - s.t0))
            tip = (s.a.tip[0] + (s.b.tip[0] - s.a.tip[0]) * u,
                   s.a.tip[1] + (s.b.tip[1] - s.a.tip[1]) * u)  # fmt: skip
            return replace(
                s.a,
                tip=tip,
                pinch_mm=_lerp(s.a.pinch_mm, s.b.pinch_mm, u),
                two_mm=_lerp(s.a.two_mm, s.b.two_mm, u),
            )
    return segs[-1].a


def _lerp(a: float | None, b: float | None, u: float) -> float | None:
    if a is None or b is None:
        return b if u >= 0.5 else a
    return a + (b - a) * u


@dataclass(frozen=True)
class CameraGeometry:
    """Where the overhead camera is, for parallax (VIS report: ~370 mm up, the paper
    centre ~135 mm in front of the point below the lens)."""

    nadir: Point = (105.0, 209.0)  # paper mm
    height_mm: float = 370.0


@dataclass(frozen=True)
class NoiseParams:
    jitter_mm: float = 0.35  # shared per-frame jitter of the whole hand
    landmark_mm: float = 0.2  # extra per-landmark jitter
    dropout: float = 0.02  # chance a frame loses the hand


class VirtualHand:
    """Samples `HandsFrame`s from keyframes. Landmarks are returned in bed mm."""

    def __init__(
        self,
        keys: list[dict[str, Any]],
        paper: Box,
        seed: int = 0,
        noise: NoiseParams | None = None,
        camera: CameraGeometry | None = None,
        default_height: float = 0.0,
    ) -> None:
        self.segs = expand(keys, default_height)
        self.paper = paper
        self.rng = np.random.default_rng(seed)
        self.noise = noise or NoiseParams()
        self.camera = camera or CameraGeometry()

    def state(self, t: float) -> HandState:
        return state_at(self.segs, t)

    def end_time(self) -> float:
        finite = [s.t1 for s in self.segs if math.isfinite(s.t1)]
        return max(finite, default=0.0)

    def frame(self, t: float) -> HandsFrame:
        st = self.state(t)
        hands: list[HandObservation] = []
        if st.rest is not None:  # listed first: the tracker must not depend on the order
            rest = HandState(present=True, tip=st.rest, pose="fist")
            hands.append(self._observe(rest, mirror=True))
        if st.present and self.rng.random() >= self.noise.dropout:
            hands.append(self._observe(st, mirror=False))
            if st.two_mm is not None:
                other = replace(st, tip=(st.tip[0] - st.two_mm, st.tip[1]), pose="point")
                hands.append(self._observe(other, mirror=True))
        return HandsFrame(t, hands, (1920, 1080))

    def _observe(self, st: HandState, mirror: bool) -> HandObservation:
        world = _MODELS[st.pose].copy()
        if mirror:
            world[:, 0] *= -1
        mm = world[:, :2] * 1000.0  # hand-centred mm, fingers along +y
        mm = mm - mm[INDEX_TIP] + np.asarray(st.tip)
        if st.pose == "pinch" and st.pinch_mm is not None:
            d = mm[THUMB_TIP] - mm[INDEX_TIP]
            d = d / (np.linalg.norm(d) + 1e-9)
            mm[THUMB_TIP] = mm[INDEX_TIP] + d * st.pinch_mm
            mm[3] = mm[INDEX_TIP] + d * st.pinch_mm * 0.8 - np.array([0.0, 12.0])
        if st.height_mm > 0:  # parallax: the camera sees raised points pushed outwards
            c = np.asarray(self.camera.nadir)
            z = self.camera.height_mm
            mm = c + (mm - c) * z / (z - st.height_mm)
        n = self.noise
        mm = mm + self.rng.normal(0, n.jitter_mm, 2) + self.rng.normal(0, n.landmark_mm, mm.shape)
        bed = mm + np.array([self.paper.x, self.paper.y])
        px = bed * 4.0 + 100.0
        return HandObservation(
            handedness="Left" if mirror else "Right",
            score=0.95 if not mirror else 0.9,
            landmarks_px=px,
            landmarks_mm=bed,
            world=world,
            pose=_POSE_OK[st.pose],
        )
