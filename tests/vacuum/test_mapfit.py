"""The floor from the robot's drive, checked against a camera we know the answer for.

A made-up room: the floor is z = 0, the camera hangs on a wall looking down across it, and
the robot drives a curve with the lid marker 100 mm up. We know where everything really is,
so we can ask the only question that matters: feed in the pixel of a **foot on the floor**,
and how many millimetres out is the answer?
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pointdraw.layout import ROBOT_MARKER_SIZE_MM
from pointdraw.vacuum.mapfit import (
    Sighting,
    camera_pose,
    fit_px_to_map,
    floor_frame,
    lift_to_floor,
)

IMAGE = (1920, 1080)
FOCAL = 1.05 * IMAGE[0]
CAMERA = np.array([1500.0, -1600.0, 2200.0])  # on the wall, 2.2 m up
LOOK_AT = np.array([1200.0, 900.0, 0.0])
LID_MM = 100.0
OFFSET_MM = 90.0  # the marker sits this far in front of the robot's centre


def _camera() -> tuple[np.ndarray, np.ndarray]:
    """(rotation world->camera, the camera's place), looking at LOOK_AT."""
    fwd = LOOK_AT - CAMERA
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.vstack([right, down, fwd]), CAMERA


def project(world: np.ndarray) -> np.ndarray:
    """World mm (n, 3) -> pixels (n, 2)."""
    R, C = _camera()
    K = np.array([[FOCAL, 0, IMAGE[0] / 2], [0, FOCAL, IMAGE[1] / 2], [0, 0, 1]])
    cam = (R @ (np.asarray(world, np.float64).reshape(-1, 3) - C).T).T
    px = (K @ cam.T).T
    return px[:, :2] / px[:, 2:3]


def _path(n: int = 60) -> list[tuple[np.ndarray, float]]:
    """Where the robot is and which way it faces, along a curve across the patch."""
    out = []
    for i in range(n):
        u = i / (n - 1)
        x = 400.0 + 2000.0 * u
        y = 300.0 + 1500.0 * u + 400.0 * math.sin(u * 6.0)
        heading = math.atan2(1500.0 + 2400.0 * math.cos(u * 6.0), 2000.0)
        out.append((np.array([x, y]), heading))
    return out


def _sightings(noise_px: float = 0.0, seed: int = 0) -> list[Sighting]:
    rng = np.random.default_rng(seed)
    half = ROBOT_MARKER_SIZE_MM / 2
    out = []
    for centre, heading in _path():
        fwd = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])
        mid = centre + fwd * OFFSET_MM
        # printed TL, TR, BR, BL, with the printed top toward the robot's front
        corners = np.array(
            [
                [*(mid + fwd * half + left * half), LID_MM],
                [*(mid + fwd * half - left * half), LID_MM],
                [*(mid - fwd * half - left * half), LID_MM],
                [*(mid - fwd * half + left * half), LID_MM],
            ]
        )
        px = project(corners)
        if noise_px:
            px = px + rng.normal(0.0, noise_px, px.shape)
        out.append(Sighting(px, (float(centre[0]), float(centre[1]))))
    return out


def _floor_mapper(sightings: list[Sighting], camera_height: float | None = None) -> np.ndarray:
    fit = fit_px_to_map(sightings, offset_mm=OFFSET_MM)
    assert fit is not None
    H_lid, _ = fit
    pose = camera_pose(H_lid, IMAGE, focal_px=FOCAL)
    assert pose is not None
    above_lid, nadir = pose
    height = camera_height if camera_height is not None else above_lid + LID_MM
    return lift_to_floor(H_lid, nadir, height, LID_MM)


def test_the_drive_finds_the_camera_and_the_spot_under_it():
    fit = fit_px_to_map(_sightings(), offset_mm=OFFSET_MM)
    assert fit is not None
    H_lid, rms = fit
    assert rms < 1.0
    pose = camera_pose(H_lid, IMAGE, focal_px=FOCAL)
    assert pose is not None
    above_lid, nadir = pose
    assert above_lid == pytest.approx(CAMERA[2] - LID_MM, abs=60.0)  # seen: 34 mm out
    assert nadir[0] == pytest.approx(CAMERA[0], abs=60.0)
    assert nadir[1] == pytest.approx(CAMERA[1], abs=60.0)


def test_a_foot_on_the_floor_lands_within_a_few_mm():
    H_floor = _floor_mapper(_sightings())
    feet = np.array([[600.0, 400.0], [1800.0, 900.0], [2300.0, 1900.0], [900.0, 1700.0]])
    px = project(np.hstack([feet, np.zeros((len(feet), 1))]))
    got = np.array([(H_floor @ [x, y, 1.0])[:2] / (H_floor @ [x, y, 1.0])[2] for x, y in px])
    assert np.max(np.linalg.norm(got - feet, axis=1)) < 8.0  # seen: 5.1 mm worst


def test_without_the_lift_a_foot_is_out_by_centimetres():
    """What the lid-height fit alone would do: the reason `lift_to_floor` exists."""
    fit = fit_px_to_map(_sightings(), offset_mm=OFFSET_MM)
    assert fit is not None
    H_lid = fit[0]
    foot = np.array([[2300.0, 1900.0]])
    px = project(np.hstack([foot, np.zeros((1, 1))]))[0]
    got = (H_lid @ [px[0], px[1], 1.0])[:2] / (H_lid @ [px[0], px[1], 1.0])[2]
    assert np.linalg.norm(got - foot[0]) > 30.0


def test_a_camera_height_by_eye_is_good_enough():
    """20 cm out on a 2.2 m camera still puts a foot within 2 cm (seen: 17 mm)."""
    sightings = _sightings()
    feet = np.array([[2300.0, 1900.0]])
    px = project(np.hstack([feet, np.zeros((1, 1))]))[0]
    for height in (2000.0, 2200.0, 2400.0):
        H_floor = _floor_mapper(sightings, camera_height=height)
        got = (H_floor @ [px[0], px[1], 1.0])[:2] / (H_floor @ [px[0], px[1], 1.0])[2]
        assert np.linalg.norm(got - feet[0]) < 25.0


def test_the_marker_offset_is_taken_out():
    """Ignoring the 90 mm the marker sits ahead costs tens of millimetres."""
    sightings = _sightings()
    with_offset = fit_px_to_map(sightings, offset_mm=OFFSET_MM)
    without = fit_px_to_map(sightings, offset_mm=0.0)
    assert with_offset is not None and without is not None
    assert with_offset[1] < 1.0
    assert without[1] > 20.0


def test_camera_shake_in_the_pixels_does_not_wreck_it():
    H_floor = _floor_mapper(_sightings(noise_px=0.5, seed=3))
    feet = np.array([[600.0, 400.0], [2300.0, 1900.0]])
    px = project(np.hstack([feet, np.zeros((len(feet), 1))]))
    got = np.array([(H_floor @ [x, y, 1.0])[:2] / (H_floor @ [x, y, 1.0])[2] for x, y in px])
    assert np.max(np.linalg.norm(got - feet, axis=1)) < 40.0


def test_the_floor_frame_runs_right_and_away():
    H_floor = _floor_mapper(_sightings())
    covered = np.array([(H_floor @ [*s.centre_px, 1.0])[:2] / (H_floor @ [*s.centre_px, 1.0])[2]
                        for s in _sightings()])  # fmt: skip
    R, area = floor_frame(H_floor, IMAGE, covered)
    assert area[0] == 0.0 and area[1] == 0.0
    assert area[2] > 1000.0 and area[3] > 1000.0

    def to_floor(px: np.ndarray) -> np.ndarray:
        v = H_floor @ [px[0], px[1], 1.0]
        w = R @ [v[0] / v[2], v[1] / v[2], 1.0]
        return w[:2] / w[2]

    mid = np.array([IMAGE[0] / 2, IMAGE[1] / 2])
    assert to_floor(mid + [200, 0])[0] > to_floor(mid)[0]  # right in the picture is +x
    assert to_floor(mid + [0, -200])[1] > to_floor(mid)[1]  # up in the picture is +y


def test_a_fitted_floor_saves_and_comes_back(tmp_path):
    """The app loads this instead of the markers, so it must survive a round trip."""
    from pointdraw.vacuum.floor import FixedFloor

    H = _floor_mapper(_sightings())
    floor = FixedFloor(H, (0.0, 0.0, 2400.0, 1800.0), "2026-10-07T12:00:00", {"pairs": 42})
    path = floor.save(tmp_path / "floor_fixed.json")
    back = FixedFloor.load(path)
    assert np.allclose(back.H, H)
    assert back.area_mm == (0.0, 0.0, 2400.0, 1800.0)
    assert back.note["pairs"] == 42
    view = back.update([], t=1.5)  # markers ignored, and never stale
    assert view.fresh and view.t == 1.5
    assert np.allclose(view.H, H)
    assert FixedFloor.load_or_none(tmp_path / "nothing.json") is None
