"""Pointing-pose classifier on made-up landmarks, and conversion of landmarker results."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pointdraw.vision.hands import (
    INDEX_TIP,
    THUMB_TIP,
    HandsFrame,
    PoseParams,
    classify_pointing,
    observations_from_result,
)

# Finger geometry in metres: MCP position and segment lengths (proximal, middle, distal).
_MCP = {
    "index": (0.030, 0.090),
    "middle": (0.010, 0.095),
    "ring": (-0.010, 0.090),
    "pinky": (-0.030, 0.080),
}
_SEG = {
    "index": (0.045, 0.025, 0.020),
    "middle": (0.050, 0.030, 0.022),
    "ring": (0.045, 0.028, 0.020),
    "pinky": (0.035, 0.020, 0.018),
}
_ORDER = ["index", "middle", "ring", "pinky"]
STRAIGHT = (0, 0, 0)
CURLED = (80, 100, 60)


def make_hand(flex, thumb_out=False, rotation=None, noise=0.0, seed=0):
    """21 world landmarks. flex: finger -> (MCP, PIP, DIP) flexion in degrees."""
    lm = np.zeros((21, 3))
    lm[1] = (0.025, 0.020, 0.0)
    lm[2] = (0.045, 0.040, 0.0)
    if thumb_out:
        lm[3], lm[4] = (0.070, 0.055, 0.0), (0.095, 0.065, 0.0)
    else:
        lm[3], lm[4] = (0.050, 0.060, -0.015), (0.040, 0.075, -0.030)
    for k, finger in enumerate(_ORDER):
        base = 5 + 4 * k
        p = np.array([*_MCP[finger], 0.0])
        lm[base] = p
        angle = 0.0
        for seg, bend in zip(_SEG[finger], flex.get(finger, STRAIGHT), strict=True):
            angle += math.radians(bend)
            p = p + seg * np.array([0.0, math.cos(angle), -math.sin(angle)])
            base += 1
            lm[base] = p
    if rotation is not None:
        lm = lm @ rotation.T
    if noise:
        lm = lm + np.random.default_rng(seed).normal(0, noise, lm.shape)
    return lm


def _random_rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


POINT = {"index": STRAIGHT, "middle": CURLED, "ring": CURLED, "pinky": CURLED}


@pytest.mark.parametrize("seed", range(5))
def test_pointing_detected_in_any_orientation(seed):
    lm = make_hand(POINT, rotation=_random_rotation(seed), noise=0.001, seed=seed)
    res = classify_pointing(lm)
    assert res.pointing
    assert res.fingers["index"].extended and res.fingers["middle"].curled


def test_pointing_with_thumb_out_and_lazy_pinky():
    lazy = dict(POINT, pinky=(30, 30, 10))
    assert classify_pointing(make_hand(lazy, thumb_out=True)).pointing


def test_pointing_down_at_the_paper():
    # Fingertip pressed on the sheet: the index bends a little and is neither straight nor
    # curled. It counts by default, not with half_index_ok off.
    hand = make_hand(dict(POINT, index=(0, 40, 30)))
    fs = classify_pointing(hand).fingers["index"]
    assert not fs.extended and not fs.curled
    assert classify_pointing(hand).pointing
    assert not classify_pointing(hand, PoseParams(half_index_ok=False)).pointing


@pytest.mark.parametrize(
    "flex",
    [
        {f: STRAIGHT for f in _ORDER},  # open hand
        {f: CURLED for f in _ORDER},  # fist
        dict(POINT, middle=STRAIGHT),  # peace sign
        dict(POINT, index=(20, 70, 40)),  # hooked index
        dict(POINT, ring=STRAIGHT, pinky=STRAIGHT),  # middle curled only
    ],
)
def test_not_pointing(flex):
    assert not classify_pointing(make_hand(flex)).pointing


def _fake_result(hands):
    """Shape of a MediaPipe HandLandmarkerResult: normalised image landmarks per hand."""

    def lms(arr):
        return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in arr]

    return SimpleNamespace(
        hand_landmarks=[lms(img) for img, _, _ in hands],
        hand_world_landmarks=[lms(world) for _, world, _ in hands],
        handedness=[[SimpleNamespace(category_name=name, score=0.9)] for _, _, name in hands],
    )


def _image_landmarks(world, center, scale=4.0):
    """Project world landmarks straight down into normalised image coordinates."""
    xy = world[:, :2] * scale + np.asarray(center)
    return np.column_stack([xy, world[:, 2]])


def test_observations_pixels_mm_and_spans():
    w1 = make_hand(POINT)
    w2 = make_hand({f: STRAIGHT for f in _ORDER}, thumb_out=True)
    result = _fake_result(
        [
            (_image_landmarks(w1, (0.3, 0.3)), w1, "Right"),
            (_image_landmarks(w2, (0.6, 0.5)), w2, "Left"),
        ]
    )
    size = (1920, 1080)

    def to_mm(px):  # a simple scale: 4 px per mm, origin at the image corner
        return px / 4.0

    obs = observations_from_result(result, size, to_mm)
    assert len(obs) == 2
    a, b = obs
    assert a.handedness == "Right" and a.pointing and not b.pointing
    exp_px = (_image_landmarks(w1, (0.3, 0.3))[INDEX_TIP, :2]) * size
    assert a.index_tip_px == pytest.approx(tuple(exp_px))
    assert a.index_tip_mm == pytest.approx(tuple(exp_px / 4.0))
    d_px = np.linalg.norm(a.landmarks_px[THUMB_TIP] - a.landmarks_px[INDEX_TIP])
    assert a.pinch_px == pytest.approx(d_px)
    assert a.pinch_mm == pytest.approx(d_px / 4.0)

    frame = HandsFrame(1.0, obs, size)
    assert frame.has_mm and frame.pointing_hand() is a and frame.primary() is a
    span = np.linalg.norm(a.landmarks_px[INDEX_TIP] - b.landmarks_px[INDEX_TIP])
    assert frame.two_hand_span_px == pytest.approx(span)
    assert frame.two_hand_span_mm == pytest.approx(span / 4.0)


def test_observations_without_homography():
    w = make_hand(POINT)
    obs = observations_from_result(
        _fake_result([(_image_landmarks(w, (0.5, 0.5)), w, "Left")]), (640, 480)
    )
    frame = HandsFrame(0.0, obs, (640, 480))
    assert not frame.has_mm and obs[0].index_tip_mm is None and obs[0].pinch_mm is None
    assert frame.two_hand_span_px is None
    assert HandsFrame(0.0, [], (640, 480)).primary() is None
