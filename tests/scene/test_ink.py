"""Ink segmentation on synthetic top-down paper images."""

import numpy as np
import pytest

from deixis.geometry import Box
from pointdraw.cosmos.scoring import iou
from pointdraw.scene.ink import (
    InkParams,
    box_mm_to_px,
    box_px_to_mm,
    crop_around,
    crop_for_mark,
    ink_mask,
    mark_at,
    mm_to_px,
    px_to_mm,
    robot_stroke_mask,
    segment_ink,
)
from pointdraw.scene.synth import INKS, make_mark, make_sheet, render_sheet

MPP = 0.25


def _ink(name):
    return next(i for i in INKS if i[0] == name)


def test_frame_conversions_round_trip():
    h = 592
    assert px_to_mm(0, h - 1, h, MPP) == pytest.approx((0.125, 0.125))  # bottom-left pixel
    for p in [(10.0, 20.0), (200.0, 140.0)]:
        assert px_to_mm(*mm_to_px(p, h, MPP), h, MPP) == pytest.approx(p)
    b = box_px_to_mm(box_mm_to_px(Box(10, 20, 30, 40), h, MPP), h, MPP)
    assert (b.x, b.y, b.w, b.h) == pytest.approx((10, 20, 30, 40))
    # +y is away from the user, so a box high on the sheet is near the top of the image.
    x1, y1, x2, y2 = box_mm_to_px(Box(0, 138, 10, 10), h, MPP)
    assert y1 == pytest.approx(0) and y2 == pytest.approx(40)


@pytest.mark.parametrize("seed", range(6))
def test_finds_every_mark_on_random_sheets(seed):
    sheet = make_sheet(np.random.default_rng(seed), n_marks=4, n_robot=1)
    found = segment_ink(sheet.image, MPP, sheet.robot_polylines_mm)
    h = sheet.image.shape[0]
    for m in sheet.marks:
        truth = box_mm_to_px(m.box_mm, h, MPP)
        best = max((iou(f.box_px, truth) for f in found), default=0.0)
        assert best >= 0.6, (m.label, m.ink, best)
    # Nothing big that isn't a user mark (robot strokes removed, no paper noise).
    big = [f for f in found if f.area_mm2 > 20]
    assert len(big) <= len(sheet.marks) + 1


def test_robot_strokes_are_subtracted():
    rng = np.random.default_rng(7)
    robot = [
        [(40.0, 40.0), (120.0, 40.0), (120.0, 100.0), (40.0, 100.0), (40.0, 40.0)],
        [(140.0, 70.0), (190.0, 70.0)],
    ]
    sheet = render_sheet([], robot_polylines_mm=robot, rng=rng)
    assert len(segment_ink(sheet.image, MPP)) >= 2  # visible without subtraction
    assert segment_ink(sheet.image, MPP, robot) == []
    # Calibration error of 0.8 mm is still inside the default 1.5 mm margin.
    shifted = [[(x + 0.8, y - 0.5) for x, y in line] for line in robot]
    assert segment_ink(sheet.image, MPP, shifted) == []


def test_user_mark_touching_robot_stroke_survives():
    rng = np.random.default_rng(8)
    robot = [[(20.0, 74.0), (190.0, 74.0)]]
    mark = make_mark("circle", (105.0, 74.0), 40.0, rng, ink=_ink("blue crayon"))
    sheet = render_sheet([mark], robot_polylines_mm=robot, rng=rng)
    found = segment_ink(sheet.image, MPP, robot)
    assert len(found) == 1  # the circle is cut by the subtraction but bridged back together
    assert found[0].box_mm.w > 30 and found[0].box_mm.h > 30


def test_marks_touching_one_robot_line_stay_separate():
    rng = np.random.default_rng(9)
    robot = [[(10.0, 74.0), (200.0, 74.0)]]
    a = make_mark("circle", (50.0, 74.0), 30.0, rng, ink=_ink("pencil"))
    b = make_mark("square", (150.0, 74.0), 30.0, rng, ink=_ink("pencil"))
    sheet = render_sheet([a, b], robot_polylines_mm=robot, rng=rng)
    found = segment_ink(sheet.image, MPP, robot)
    assert len(found) == 2
    assert sorted(round(f.box_mm.center[0] / 50) for f in found) == [1, 3]


@pytest.mark.parametrize("ink", ["pencil", "orange crayon", "green crayon", "black pen"])
def test_pale_and_dark_inks(ink):
    rng = np.random.default_rng(11)
    mark = make_mark("star", (100.0, 70.0), 40.0, rng, ink=_ink(ink))
    sheet = render_sheet([mark], rng=rng)
    found = segment_ink(sheet.image, MPP)
    assert len(found) == 1
    truth = box_mm_to_px(mark.box_mm, sheet.image.shape[0], MPP)
    assert iou(found[0].box_px, truth) > 0.8
    assert found[0].box_mm.center == pytest.approx(mark.box_mm.center, abs=2.0)


def test_parts_inside_a_mark_are_merged():
    """The house's door lies inside its outline box, so it is one mark, not two."""
    rng = np.random.default_rng(5)
    house = make_mark("house", (100.0, 70.0), 50.0, rng, ink=_ink("pencil"), rotate_deg=0)
    sheet = render_sheet([house], rng=rng)
    assert len(segment_ink(sheet.image, MPP)) == 1
    apart = segment_ink(sheet.image, MPP, params=InkParams(merge_contained=False, merge_gap_mm=0))
    assert len(apart) >= 1


def test_separate_marks_stay_separate_and_are_sorted_by_area():
    rng = np.random.default_rng(2)
    big = make_mark("square", (60.0, 70.0), 50.0, rng, ink=_ink("black pen"))
    small = make_mark("circle", (160.0, 70.0), 20.0, rng, ink=_ink("black pen"))
    sheet = render_sheet([big, small], rng=rng)
    found = segment_ink(sheet.image, MPP)
    assert [m.id for m in found] == [0, 1]
    assert found[0].area_mm2 > found[1].area_mm2
    assert found[0].centroid_mm[0] < 100 < found[1].centroid_mm[0]


def test_blank_paper_has_no_marks():
    sheet = render_sheet([], rng=np.random.default_rng(1))
    assert segment_ink(sheet.image, MPP) == []
    assert ink_mask(sheet.image, MPP).max() == 0


def test_robot_mask_width():
    m = robot_stroke_mask((400, 400), [[(10.0, 50.0), (90.0, 50.0)]], MPP, 0.6, 1.5)
    col = m[:, 200]
    width_px = int((col > 0).sum())
    assert 3.4 / MPP <= width_px <= 4.5 / MPP  # 0.6 + 2 * 1.5 = 3.6 mm, rounded up, antialiased


def test_mark_at_and_crops():
    rng = np.random.default_rng(4)
    a = make_mark("circle", (50.0, 70.0), 30.0, rng, ink=_ink("black pen"))
    b = make_mark("triangle", (150.0, 70.0), 30.0, rng, ink=_ink("black pen"))
    sheet = render_sheet([a, b], rng=rng)
    marks = segment_ink(sheet.image, MPP)
    hit = mark_at(marks, (52.0, 72.0))
    assert hit is not None and hit.box_mm.center[0] < 100
    near = mark_at(marks, (170.0, 70.0), max_dist_mm=10)
    assert near is not None and near.box_mm.center[0] > 100
    assert mark_at(marks, (100.0, 140.0), max_dist_mm=5) is None

    crop = crop_around(sheet.image, (50.0, 70.0), 40.0, MPP)
    assert crop.image.shape[:2] == (160, 160)
    assert crop.box_mm.center == pytest.approx((50.0, 70.0), abs=MPP)
    edge = crop_around(sheet.image, (2.0, 2.0), 20.0, MPP)  # past the corner: padded white
    assert edge.image.shape[:2] == (80, 80)
    assert (edge.image[:5, :5] == 255).all()  # top-left of the crop is off the paper
    fc = crop_for_mark(sheet.image, hit, MPP)
    assert fc.box_mm.w >= max(hit.box_mm.w, hit.box_mm.h)
