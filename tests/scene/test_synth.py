"""The synthetic sheet generator and its ground truth."""

import json

import numpy as np

from pointdraw.scene.synth import SHAPES, SYNONYMS, make_mark, make_sheet


def test_every_shape_has_synonyms_and_sensible_size():
    rng = np.random.default_rng(0)
    assert set(SHAPES) == set(SYNONYMS)
    for label in SHAPES:
        m = make_mark(label, (100.0, 70.0), 40.0, rng, rotate_deg=0)
        b = m.box_mm
        assert 25 < max(b.w, b.h) < 55, label
        assert abs(b.center[0] - 100) < 12 and abs(b.center[1] - 70) < 12, label


def test_sheet_ground_truth_is_consistent():
    sheet = make_sheet(np.random.default_rng(1), n_marks=4, n_robot=1)
    gt = json.loads(json.dumps(sheet.ground_truth("x.png")))
    h, w = sheet.image.shape[:2]
    assert (w, h) == (840, 592)
    assert len(gt["marks"]) == 4 and gt["robot_polylines_mm"]
    for m in gt["marks"]:
        x1, y1, x2, y2 = m["box_px"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert m["synonyms"] == SYNONYMS[m["label"]]
    assert 1 <= len(gt["point_queries"]) <= 3
    for q in gt["point_queries"]:
        x1, y1, x2, y2 = q["accept_box_px"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert q["query"].startswith("empty space ")


def test_generation_is_deterministic():
    a = make_sheet(np.random.default_rng(9))
    b = make_sheet(np.random.default_rng(9))
    assert np.array_equal(a.image, b.image)
