"""Synthetic ArUco markers and plotter crosses: detection, matching, fitting, marker learning."""

import json

import cv2
import numpy as np
import pytest

from pointdraw.vision import synthetic as syn
from pointdraw.vision.aruco import (
    BedCalibration,
    BedTracker,
    apply_homography,
    detect_markers,
    solve_bed_homography,
)
from pointdraw.vision.calibration import (
    CalibrationError,
    OrientationPrior,
    calibrate_from_image,
    detect_crosses,
    fit_cross_homography,
    load_cross_points,
    match_crosses,
)

PAPER = np.array([(x, y) for x in np.linspace(0, 210, 15) for y in np.linspace(0, 148, 11)])


def _max_err(H_px_to_mm, H_true_mm_to_px, pts=PAPER):
    est = apply_homography(H_px_to_mm, apply_homography(H_true_mm_to_px, pts))
    return float(np.linalg.norm(est - pts, axis=1).max())


def _rotate_camera(H, deg, size=(1920, 1080)):
    """Same camera, rotated about the image centre (as if mounted another way round)."""
    R = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), deg, 1.0)
    return np.vstack([R, [0, 0, 1]]) @ H


@pytest.fixture(scope="module")
def scene_and_image():
    scene, H = syn.default_scene()
    return scene, H, syn.render(scene, H, noise=3.0, seed=1)


def test_markers_detected_with_ids(scene_and_image):
    scene, H, img = scene_and_image
    found = detect_markers(img)
    assert [m.id for m in found] == sorted(scene.markers)
    truth = scene.marker_corners()
    for m in found:
        err = np.linalg.norm(m.corners_px - apply_homography(H, truth[m.id]), axis=1)
        assert err.max() < 0.8  # about 0.2 mm at 4 px/mm


def test_crosses_detected_subpixel(scene_and_image):
    scene, H, img = scene_and_image
    crosses = detect_crosses(img, markers=detect_markers(img))
    assert len(crosses) == len(scene.crosses)
    truth = apply_homography(H, scene.crosses)
    for c in crosses:
        assert np.linalg.norm(truth - c.center_px, axis=1).min() < 0.4


def test_calibration_recovers_homography_within_0_3_mm(scene_and_image):
    scene, H, img = scene_and_image
    res = calibrate_from_image(img, scene.crosses)
    assert res.fit.inliers.sum() == 9
    assert res.fit.rms_mm < 0.1
    assert _max_err(res.fit.H, H) < 0.3
    truth = scene.marker_corners()
    assert sorted(res.calibration.marker_corners_mm) == sorted(scene.markers)
    for i, corners in res.calibration.marker_corners_mm.items():
        assert np.linalg.norm(corners - truth[i], axis=1).max() < 0.3
    assert "rms" in res.summary()


def test_markers_alone_recover_bed_frame_in_a_new_view(scene_and_image, tmp_path):
    """Learn marker corners once, then find the bed from markers in a different camera pose."""
    scene, H, img = scene_and_image
    calib = calibrate_from_image(img, scene.crosses).calibration
    path = calib.save(tmp_path / "bed.json")
    calib = BedCalibration.load(path)

    # The bed moved 30 mm and the camera tilted a little: a new true homography.
    shift = np.array([[1, 0, 0], [0, 1, -30], [0, 0, 1]], float)  # new bed mm -> old bed mm
    persp = np.array([[1, 0.01, 5], [0.005, 1, -8], [2e-6, 1e-5, 1]], float)
    H2 = persp @ H @ shift
    img2 = syn.render(syn.Scene(scene.markers, np.empty((0, 2))), H2, noise=3.0, seed=2)
    bh = solve_bed_homography(detect_markers(img2), calib)
    assert bh is not None and len(bh.marker_ids) >= 4  # the shift pushed some out of view
    assert bh.rms_mm < 0.3
    assert _max_err(bh.H, H2) < 0.3
    # Two markers are enough.
    two = [m for m in detect_markers(img2) if m.id in (2, 5)]
    bh2 = solve_bed_homography(two, calib)
    assert bh2 is not None and _max_err(bh2.H, H2) < 0.5
    # to_px inverts to_mm.
    p = np.array([[100.0, 50.0]])
    assert np.allclose(bh.to_mm(bh.to_px(p)), p, atol=1e-6)


def test_bed_tracker_holds_last_good_solution(scene_and_image):
    scene, H, img = scene_and_image
    calib = calibrate_from_image(img, scene.crosses).calibration
    tr = BedTracker(calib, max_age_s=1.0)
    dets = detect_markers(img)
    assert tr.update(dets, t=10.0) is not None
    assert tr.update(dets[:1], t=10.5) is tr.last  # one marker is not enough: reuse
    assert tr.update([], t=12.0) is None  # too old


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_match_asymmetric_pattern_any_rotation_without_prior(deg):
    scene, H = syn.default_scene()
    known = np.vstack([scene.crosses, [[92.5, 36.5]]])  # one extra cross inside a corner cell
    Hr = _rotate_camera(H, deg)
    px = apply_homography(Hr, known)
    rng = np.random.default_rng(deg)
    order = rng.permutation(len(px))
    pairs = match_crosses(px[order], known, prior=None)
    assert len(pairs) == len(known)
    assert all(order[i] == j for i, j in pairs)


def test_symmetric_grid_needs_prior():
    scene, H = syn.default_scene()
    px = apply_homography(H, scene.crosses)
    with pytest.raises(CalibrationError, match="symmetric"):
        match_crosses(px, scene.crosses, prior=None)
    pairs = match_crosses(px, scene.crosses)  # default prior: +y is image-up, not mirrored
    assert all(i == j for i, j in pairs)
    # Image content turned 90 degrees clockwise: +y now points right and -x points up.
    Hr = _rotate_camera(H, -90)
    px_r = apply_homography(Hr, scene.crosses)
    pairs = match_crosses(px_r, scene.crosses, prior=OrientationPrior(up="-x"))
    assert all(i == j for i, j in pairs)


def test_match_with_guess_and_spurious_detection():
    scene, H = syn.default_scene()
    px = np.vstack([apply_homography(H, scene.crosses), [[1800.0, 60.0]]])
    guess = np.linalg.inv(H) @ np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], float)
    pairs = match_crosses(px, scene.crosses, H_guess_px_to_mm=guess)
    assert len(pairs) == 9 and all(i == j for i, j in pairs)
    pairs = match_crosses(px, scene.crosses)  # no guess: the outlier sits on the hull
    assert len(pairs) == 9 and all(i == j for i, j in pairs)


def test_fit_rejects_one_bad_cross():
    scene, H = syn.default_scene()
    px = apply_homography(H, scene.crosses)
    px[4] += (12.0, -9.0)  # a smudge moved the centre cross by several mm
    fit = fit_cross_homography(px, scene.crosses)
    assert not fit.inliers[4] and fit.inliers.sum() == 8
    assert fit.rms_mm < 0.05 and fit.residuals_mm[4] > 2.0


def test_distractors_are_not_crosses():
    scene, H = syn.default_scene()
    scene.extra_lines = [((5, 140), (40, 140)), ((197, 50), (197, 90))]  # straight strokes
    scene.extra_lines += [((170, 142), (200, 142)), ((170, 142), (170, 115))]  # an L
    scene.extra_lines += [((8, 70), (38, 70)), ((23, 70), (23, 95))]  # a T
    img = syn.render(scene, H, seed=3)
    crosses = detect_crosses(img, markers=detect_markers(img))
    assert len(crosses) == 9


def test_too_few_crosses_raises():
    scene, H = syn.default_scene()
    img = syn.render(syn.Scene(scene.markers, scene.crosses[:3]), H)
    with pytest.raises(CalibrationError):
        calibrate_from_image(img, scene.crosses)


def test_load_cross_points_formats(tmp_path):
    for payload in (
        [[0, 0], [50, 0]],
        [{"x": 0, "y": 0}, {"x": 50, "y": 0}],
        {"crosses": [{"x": 0, "y": 0, "label": "a"}, {"x": 50, "y": 0}]},
        {"points": [[0, 0], [50, 0]], "grid": "3x3"},
    ):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(payload))
        assert load_cross_points(p).tolist() == [[0, 0], [50, 0]]
