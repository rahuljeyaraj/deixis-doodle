"""The floor camera on a drawn floor picture, the feet from pose landmarks, and the
floor-to-map calibration (pairs, the robot's centre from its lid marker, the fit)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from deixis.frames import apply_matrix
from deixis.mcp import DeixisClient
from pointdraw.vacuum.calibrate import FloorToMap, PairCollector, RobotMarker
from pointdraw.vacuum.feet import LEFT_HEEL, LEFT_TOE, RIGHT_HEEL, RIGHT_TOE, feet_seen
from pointdraw.vacuum.floor import FloorCamera, FloorLayout, FloorView
from pointdraw.vacuum.profile import vacuum_profile
from pointdraw.vision.aruco import detect_markers
from pointdraw.vision.synthetic import Scene, camera_homography, marker_corners_mm, render

LAYOUT = FloorLayout.rectangle(2400, 1800)
# A camera high up at the near side, looking down and across the floor (1920 x 1080).
FLOOR_QUAD = np.array([[-300, -300], [2700, -300], [2700, 2100], [-300, 2100]], np.float64)
IMAGE_QUAD = np.array([[100, 1040], [1820, 1040], [1470, 120], [450, 120]], np.float64)
H_MM_TO_PX = camera_homography(FLOOR_QUAD, IMAGE_QUAD)


def floor_image(hide=()):
    markers = {i: c for i, c in LAYOUT.centres_mm.items() if i not in hide}
    scene = Scene(markers=markers, crosses=np.empty((0, 2)), marker_size=150.0,
                  paper=(-290, -290, 1, 1), bed=(-300, -300, 3000, 2400))  # fmt: skip
    return render(scene, H_MM_TO_PX, canvas_px_per_mm=1.0, noise=2.0, blur_px=0.5)


def test_layout_saves_and_loads(tmp_path):
    path = LAYOUT.save(tmp_path / "floor.json")
    back = FloorLayout.load(path)
    assert back.centres_mm == {10: (0, 0), 11: (2400, 0), 12: (2400, 1800), 13: (0, 1800)}
    assert back.area_mm == (0, 0, 2400, 1800)


def test_the_floor_camera_maps_pixels_to_floor_mm():
    cam = FloorCamera(LAYOUT)
    view = cam.update(detect_markers(floor_image(), ids=None), t=0.0)
    assert view is not None and view.fresh and view.marker_ids == (10, 11, 12, 13)
    pts = np.array([[300, 200], [1200, 900], [2100, 1600], [600, 1500]], np.float64)
    got = view.to_mm(apply_matrix(H_MM_TO_PX, pts))
    assert np.max(np.linalg.norm(got - pts, axis=1)) < 15  # mm, over a 2.4 m patch
    assert np.allclose(view.to_px(got), apply_matrix(H_MM_TO_PX, pts), atol=0.5)

    # A foot on marker 12: the other three still fit, from the corners learned above.
    v2 = cam.update(detect_markers(floor_image(hide=(12,)), ids=None), t=1.0)
    assert v2 is not None and v2.fresh and 12 not in v2.marker_ids
    got2 = v2.to_mm(apply_matrix(H_MM_TO_PX, pts))
    assert np.max(np.linalg.norm(got2 - pts, axis=1)) < 20

    # Only one marker in view: the camera stands still, so the last fit is kept.
    v3 = cam.update(detect_markers(floor_image(hide=(11, 12, 13)), ids=None), t=2.0)
    assert v3 is not None and not v3.fresh and np.allclose(v3.H, v2.H)


def test_no_fit_before_all_four_are_seen():
    cam = FloorCamera(LAYOUT)
    assert cam.update(detect_markers(floor_image(hide=(13,)), ids=None)) is None


def landmarks(feet_px, vis=1.0):
    px = np.zeros((33, 2))
    v = np.zeros(33)
    for (heel, toe), (p, q) in zip(((LEFT_HEEL, LEFT_TOE), (RIGHT_HEEL, RIGHT_TOE)), feet_px,
                                   strict=False):  # fmt: skip
        px[heel], px[toe] = p, q
        v[heel] = v[toe] = vis
    return px, v


def test_feet_are_the_middle_of_heels_and_toes():
    view = FloorView(np.eye(3), (10, 11, 12, 13), True)  # 1 px = 1 mm
    px, v = landmarks([((100, 200), (100, 260)), ((180, 200), (180, 260))])
    f = feet_seen(5.0, px, v, view)
    assert f.standing and f.spot_mm == (140.0, 230.0)
    assert f.sample().point == (140.0, 230.0) and f.sample().pointing
    px, v = landmarks([((100, 200), (100, 260))])  # one foot: a point, not standing
    f = feet_seen(5.0, px, v, view)
    assert f.spot_mm == (100.0, 230.0) and not f.standing
    f = feet_seen(5.0, None, None, view)  # nobody
    assert f.spot_mm is None and f.sample().point is None
    px, v = landmarks([((100, 200), (100, 260)), ((180, 200), (180, 260))], vis=0.2)
    assert feet_seen(5.0, px, v, view).spot_mm is None  # MediaPipe isn't sure: not used
    assert feet_seen(5.0, *landmarks([((1, 2), (3, 4))]), None).spot_mm is None  # no floor yet


def test_the_robots_centre_is_behind_its_lid_marker():
    m = RobotMarker(offset_mm=100)
    front_up = marker_corners_mm((500, 500), 100)  # printed top toward +y
    assert m.centre(front_up) == pytest.approx((500, 400))
    turned = front_up[[1, 2, 3, 0]]  # turned a quarter: the marker's top edge on the right
    assert m.centre(turned) == pytest.approx((400, 500))  # the robot faces +x


def test_pairs_are_taken_only_when_still_fresh_and_apart():
    pc = PairCollector()
    for k in range(40):  # 1.33 s still at (1000, 1000), a few mm of noise
        pc.see(k / 30, (1000 + (k % 3), 1000 - (k % 2)))
    t = 39 / 30
    assert pc.offer(t, {"robot": {"x": 20, "y": 30, "age_s": 3.0}}) is None  # stale report
    pair = pc.offer(t, {"robot": {"x": 20, "y": 30, "age_s": 0.4}})
    assert pair is not None and math.dist(pair[0], (1001, 999.5)) < 2 and pair[1] == (20, 30)
    assert pc.offer(t, {"robot": {"x": 25, "y": 30, "age_s": 0.1}}) is None  # too close
    for k in range(40):  # moving: no pair
        pc.see(2 + k / 30, (1000 + k * 20, 1000))
    assert pc.offer(2 + 39 / 30, {"robot": {"x": 900, "y": 30, "age_s": 0.1}}) is None
    assert len(pc.pairs) == 1 and not pc.enough()


def test_floor_to_map_fits_saves_and_loads(tmp_path):
    def to_map(p):  # turned, mirrored, shifted
        c, s = math.cos(0.4), math.sin(0.4)
        return (500 + c * p[0] + s * p[1], -200 + s * p[0] - c * p[1])

    pairs = [((x, y), to_map((x, y))) for x, y in [(0, 0), (2000, 100), (1800, 1500), (200, 1700)]]
    core = DeixisClient(vacuum_profile(), mode="direct")
    f2m = FloorToMap.fit(pairs, core)
    assert f2m.transform.rms_mm < 1e-6 and f2m.transform.to_frame == "map"
    back = FloorToMap.load(f2m.save(tmp_path / "f2m.json"))
    assert back.transform == f2m.transform and back.pairs == pairs
    assert FloorToMap.load_or_none(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="at least 3"):
        FloorToMap.fit(pairs[:2], core)


def test_a_moving_robots_path_finds_the_lag_and_the_map():
    """The robot drives a loop; its reports arrive 0.7 s late and 3 times a second."""
    from pointdraw.vacuum.calibrate import PathMatcher

    def to_map(p):
        c, s = math.cos(1.1), math.sin(1.1)
        return (300 + c * p[0] - s * p[1], 900 + s * p[0] + c * p[1])

    def path(t):  # 250 mm/s round a 1.6 x 1.0 m loop
        d = (t * 250.0) % 5200.0
        legs = [((300, 300), (1900, 300)), ((1900, 300), (1900, 1300)),
                ((1900, 1300), (300, 1300)), ((300, 1300), (300, 300))]  # fmt: skip
        for a, b in legs:
            n = math.dist(a, b)
            if d <= n:
                return (a[0] + (b[0] - a[0]) * d / n, a[1] + (b[1] - a[1]) * d / n)
            d -= n
        return legs[-1][1]

    rng = np.random.default_rng(3)
    pm = PathMatcher()
    for k in range(30 * 30):  # 30 s of camera frames
        t = k / 30
        x, y = path(t)
        pm.see(t, (x + rng.normal(0, 8), y + rng.normal(0, 8)))
    for k in range(90):  # reports: measured at tm, polled 0.7 s later (age 0.2 s at the poll)
        tm = 1.0 + k / 3
        if tm > 29:
            break
        mx, my = to_map(path(tm))
        assert pm.report(tm + 0.7 + 0.2, {"robot": {"x": mx, "y": my, "age_s": 0.2}})
        assert not pm.report(tm + 0.7 + 0.25, {"robot": {"x": mx, "y": my, "age_s": 0.25}})
    core = DeixisClient(vacuum_profile(), mode="direct")
    lag, pairs, rms = pm.best(core)
    assert lag == pytest.approx(0.7, abs=0.11) and rms < 20 and len(pairs) > 60
    f2m = FloorToMap.fit(pairs, core)
    got = apply_matrix(np.array(f2m.transform.matrix), np.array([[1000.0, 800.0]]))[0]
    assert math.dist(got, to_map((1000, 800))) < 15
