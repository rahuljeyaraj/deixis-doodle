"""Lens model: ChArUco calibration on synthetic distorted views, and undistorted homographies."""

import math

import cv2
import numpy as np
import pytest

from pointdraw.vision import synthetic as syn
from pointdraw.vision.aruco import (
    BedCalibration,
    MarkerDetection,
    apply_homography,
    solve_bed_homography,
)
from pointdraw.vision.lens import (
    BOARD_SQUARE_MM,
    BOARD_SQUARES,
    LensModel,
    board_image,
    calibrate,
    detect_board,
)

SIZE = (960, 540)
TRUE = syn.Lens(k1=-0.30, k2=0.10, f_px=505.0)  # a 95-degree-diagonal webcam at half size


def _view(rx, ry, rz, tz, dx=0.0, dy=0.0, ppm=4.0):
    """Render the board under a pose (degrees, mm) through the true lens."""
    board = board_image(px_per_mm=ppm)
    R, _ = cv2.Rodrigues(np.radians([rx, ry, rz]))
    w_mm, h_mm = BOARD_SQUARES[0] * BOARD_SQUARE_MM, BOARD_SQUARES[1] * BOARD_SQUARE_MM
    t = np.array([dx - w_mm / 2, dy - h_mm / 2, tz])
    K = TRUE.K(SIZE)
    # Board plane (x, y, 0) mm -> ideal pixels is K [r1 r2 t]; the canvas is scaled by ppm.
    H_mm = K @ np.column_stack([R[:, 0], R[:, 1], t])
    S = np.diag([1 / ppm, 1 / ppm, 1.0])
    img = syn.warp_plane(board, H_mm @ S, SIZE, TRUE, border=128)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def model():
    # Cover the whole image, corners included: distortion is only known where the board was.
    poses = [
        (0, 0, 0, 300, 0, 0),
        (15, 0, 5, 300, -170, -90),
        (-15, 0, -5, 300, 170, -90),
        (0, 20, 0, 300, -170, 90),
        (0, -20, 10, 300, 170, 90),
        (20, 15, 0, 330, 0, -110),
        (-20, -15, 0, 330, 0, 110),
        (10, -20, 20, 280, -120, 0),
        (-10, 20, -20, 280, 120, 0),
    ]
    views = [_view(*p) for p in poses]
    return calibrate(views)


def test_board_detected_in_distorted_view():
    found = detect_board(_view(0, 0, 0, 420))
    assert found is not None and len(found[0]) >= 20


def test_calibration_recovers_distortion(model):
    assert model.n_views >= 7
    assert model.rms_px < 0.5
    assert model.dist[0] == pytest.approx(TRUE.k1, abs=0.05)
    assert model.K[0, 0] == pytest.approx(TRUE.f_px, rel=0.03)


def test_undistorted_points_fit_a_homography(model):
    """Points on a plane seen through the lens: a homography fits once undistorted."""
    H = np.array([[1.8, 0.05, 150], [-0.03, -1.8, 460], [0, 0, 1]])  # bed mm -> ideal px
    grid = np.array([(x, y) for x in np.linspace(-40, 250, 12) for y in np.linspace(-40, 188, 9)])
    raw = TRUE.distort(apply_homography(H, grid), SIZE)
    Hr, _ = cv2.findHomography(raw, grid, 0)
    raw_err = np.linalg.norm(apply_homography(Hr, raw) - grid, axis=1).max()
    und = model.undistort(raw)
    Hu, _ = cv2.findHomography(und, grid, 0)
    und_err = np.linalg.norm(apply_homography(Hu, und) - grid, axis=1).max()
    assert raw_err > 3.0, raw_err  # the distortion matters (mm) ...
    assert und_err < 0.4, und_err  # ... and the lens model removes it


def test_distort_inverts_undistort_and_json_round_trip(model, tmp_path):
    pts = np.array([[100.0, 80.0], [480.0, 270.0], [900.0, 500.0]])
    assert np.allclose(model.distort(model.undistort(pts)), pts, atol=0.05)
    again = LensModel.load_or_none(model.save(tmp_path / "lens.json"))
    assert again is not None and np.allclose(again.K, model.K) and again.image_size == SIZE
    assert LensModel.load_or_none(tmp_path / "missing.json") is None
    with pytest.raises(ValueError):
        model.check_size((1920, 1080))


def test_bed_homography_with_lens_uses_raw_pixels():
    lens = LensModel(TRUE.K(SIZE), TRUE.dist, SIZE)
    H = np.array([[1.8, 0.0, 150], [0.0, -1.8, 460], [0, 0, 1]])
    corners = {
        i: syn.marker_corners_mm(c, 20) for i, c in {0: (0, 0), 1: (200, 0), 2: (200, 140)}.items()
    }
    calib = BedCalibration(marker_corners_mm=corners)
    dets = [
        MarkerDetection(i, TRUE.distort(apply_homography(H, c), SIZE)) for i, c in corners.items()
    ]
    bh = solve_bed_homography(dets, calib, lens=lens)
    assert bh is not None and bh.rms_mm < 1e-3
    p_mm = np.array([[105.0, 70.0], [10.0, 130.0]])
    raw = TRUE.distort(apply_homography(H, p_mm), SIZE)
    assert np.allclose(bh.to_mm(raw), p_mm, atol=0.02)
    assert np.allclose(bh.to_px(p_mm), raw, atol=0.05)
    assert math.isfinite(bh.rms_px)
