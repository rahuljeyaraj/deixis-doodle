"""Paper detection and warping, on a synthetic photo and on rahul's 22 Sep photos if present."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pointdraw.cosmos.scoring import iou
from pointdraw.scene.ink import segment_ink
from pointdraw.scene.paper import find_paper_quad, order_corners, warp_paper
from pointdraw.scene.synth import make_sheet

TEST_DATA = Path(r"C:\work\test_nebius_nvidia\test_data")
COSMOS_22SEP = Path(__file__).parent / "data" / "cosmos_grounding_22sep.json"


def _fake_photo(seed=0):
    """A synthetic sheet pasted with perspective onto a textured brown desk, with glare."""
    rng = np.random.default_rng(seed)
    sheet = make_sheet(rng, n_marks=3, n_robot=0)
    h, w = 1080, 1920
    desk = np.zeros((h, w, 3), np.float32)
    grain = cv2.resize(rng.normal(0, 18, (h // 8, w // 2)).astype(np.float32), (w, h))
    desk[:] = (40, 75, 120)
    desk += grain[..., None]
    yy, xx = np.mgrid[0:h, 0:w]
    glare = np.exp(-(((xx - 1700) / 250.0) ** 2 + ((yy - 950) / 150.0) ** 2)) * 110
    desk += glare[..., None] * np.array([1.0, 0.9, 0.8], np.float32)
    desk = np.clip(desk, 0, 255).astype(np.uint8)
    sh, sw = sheet.image.shape[:2]
    corners = np.array([[480, 100], [1450, 70], [1520, 900], [470, 940]], np.float32)
    hom = cv2.getPerspectiveTransform(
        np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32), corners
    )
    warped = cv2.warpPerspective(sheet.image, hom, (w, h))
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), hom, (w, h))
    photo = np.where(mask[..., None] > 0, warped, desk)
    return photo, corners, sheet


def test_order_corners():
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], float)
    assert order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_finds_and_warps_a_synthetic_photo():
    photo, corners, sheet = _fake_photo()
    quad = find_paper_quad(photo)
    assert quad is not None
    assert np.abs(quad - corners).max() < 12  # px, at 1920x1080
    wp = warp_paper(photo, quad, size_mm=(210.0, 148.0), mm_per_px=0.25)
    assert wp.image.shape[:2] == (592, 840)
    found = segment_ink(wp.image, wp.mm_per_px)
    for m in sheet.marks:
        c = m.box_mm.center
        assert any(
            abs(f.box_mm.center[0] - c[0]) < 4 and abs(f.box_mm.center[1] - c[1]) < 4 for f in found
        ), m.label
    back = wp.to_photo(np.array([[0, 0], [840, 592]]))
    assert np.abs(back - corners[[0, 2]]).max() < 12


def test_nominal_size_from_aspect():
    photo, corners, _ = _fake_photo(1)
    wp = warp_paper(photo, corners, long_side_mm=280.0)
    assert max(wp.size_mm) == pytest.approx(280.0)


def test_no_paper_returns_none():
    rng = np.random.default_rng(0)
    desk = (rng.normal(90, 25, (600, 800, 3))).clip(0, 255).astype(np.uint8)
    assert find_paper_quad(desk) is None


@pytest.mark.skipif(not TEST_DATA.is_dir(), reason="rahul's 22 Sep test photos are not here")
def test_real_photos_match_cosmos_boxes():
    """Sanity check on the 10 real overhead photos (never copied into the repo).

    The union of our ink marks, mapped back to the photo, should agree with the object box
    Cosmos3-Nano returned on 22 Sep. Page 10 is lower because Cosmos boxed only the flower
    and the pot and missed the stem and leaves, which segmentation keeps.
    """
    ref = json.loads(COSMOS_22SEP.read_text())
    ious = {}
    for page, items in ref["pages"].items():
        path = TEST_DATA / f"page{page}.jpg"
        if not path.is_file():
            continue
        img = cv2.imread(str(path))
        quad = find_paper_quad(img)
        assert quad is not None, page
        wp = warp_paper(img, quad)
        marks = segment_ink(wp.image, wp.mm_per_px)
        total = sum(m.area_mm2 for m in marks)
        big = [m for m in marks if m.area_mm2 >= 0.03 * total]
        assert big, page
        pts = []
        for m in big:
            x1, y1, x2, y2 = m.box_px
            pts += [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ph = wp.to_photo(np.array(pts, float))
        ours = (ph[:, 0].min(), ph[:, 1].min(), ph[:, 0].max(), ph[:, 1].max())
        objs = [b for label, b in items if label != "paper"]
        sx, sy = img.shape[1] / 1000, img.shape[0] / 1000
        theirs = (
            min(b[0] for b in objs) * sx,
            min(b[1] for b in objs) * sy,
            max(b[2] for b in objs) * sx,
            max(b[3] for b in objs) * sy,
        )
        ious[page] = iou(ours, theirs)
    assert len(ious) == 10
    good = [p for p, v in ious.items() if v >= 0.85]
    assert len(good) >= 9, ious
    assert min(ious.values()) >= 0.5, ious
