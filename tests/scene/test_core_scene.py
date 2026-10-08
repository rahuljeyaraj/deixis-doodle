"""The scene geometry moved to the deixis core (core split C7): same answers as before.

The old OpenCV empty space and the old relation formulas are copied here as references.
One answer has changed since, on purpose: beside something on a page now stands on its
ground rather than on its middle (rahul, 8 Oct 2026). The test below spells that out, so the
rest of each formula is still pinned to the pre-split code.
"""

import math

import cv2
import numpy as np
import pytest

from deixis.geometry import Box
from deixis.scene import chessboard_distance
from pointdraw.scene.model import SceneModel, SceneObject, SceneParams

DRAWABLE = Box(0, 0, 220.5, 196)


def old_empty_space(objects, drawable, size, margin=3.0, res_mm=1.0):
    """pointdraw.scene.model.SceneModel.empty_space before the split (OpenCV)."""
    d = drawable
    w, h = int(d.w / res_mm), int(d.h / res_mm)
    free = np.ones((h, w), np.uint8)
    for b in objects:
        x0 = int(math.floor((b.x - margin - d.x) / res_mm))
        x1 = int(math.ceil((b.x2 + margin - d.x) / res_mm))
        y0 = int(math.floor((b.y - margin - d.y) / res_mm))
        y1 = int(math.ceil((b.y2 + margin - d.y) / res_mm))
        free[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0
    padded = np.pad(free, 1, constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_C, 3)[1:-1, 1:-1]
    best = float(dist.max())
    if best <= 0:
        return None
    ys, xs = np.nonzero(dist >= best - 0.5)
    cx0, cy0 = w / 2, h / 2
    k = int(np.argmin((xs - cx0) ** 2 + (ys - cy0) ** 2))
    side = (2 * best - 1) * res_mm
    center = (d.x + (xs[k] + 0.5) * res_mm, d.y + (ys[k] + 0.5) * res_mm)
    return center, min(size, side)


def test_chessboard_distance_equals_opencv_on_200_random_grids():
    rng = np.random.default_rng(7)
    for _ in range(200):
        h, w = rng.integers(1, 80, size=2)
        grid = (rng.random((h, w)) > rng.uniform(0.02, 0.6)).astype(np.uint8)
        ref = cv2.distanceTransform(np.pad(grid, 1), cv2.DIST_C, 3)[1:-1, 1:-1]
        assert np.array_equal(chessboard_distance(grid), ref.astype(np.int32))


def _random_scene(rng, n):
    model = SceneModel(DRAWABLE)
    for i in range(n):
        w, h = rng.uniform(5, 70, size=2)
        x, y = rng.uniform(-20, DRAWABLE.w, size=1)[0], rng.uniform(-20, DRAWABLE.h, size=1)[0]
        model.add(SceneObject(f"o{i}", "robot", "thing", Box(x, y, w, h), t=float(i)))
    return model


def test_empty_space_equals_the_old_opencv_version():
    rng = np.random.default_rng(11)
    for trial in range(200):
        model = _random_scene(rng, int(rng.integers(0, 12)))
        size = float(rng.uniform(10, 120))
        res = 1.0 if trial % 4 else 2.0
        boxes = [o.box for o in model.objects]
        assert model.empty_space(size, res) == old_empty_space(boxes, DRAWABLE, size, 3.0, res)


def old_relation(rel, t, size, gap, extent=None):
    """The page-view formulas before the split (next_to and between left out)."""
    hx, hy = (extent[0] / 2, extent[1] / 2) if extent else (size / 2, size / 2)
    cx, cy = t.center
    return {
        "left_of": (t.x - gap - hx, cy),
        "right_of": (t.x2 + gap + hx, cy),
        "above": (cx, t.y2 + gap + hy),
        "below": (cx, t.y - gap - hy),
        "inside": (cx, cy),
        "on": (cx, t.y2 + hy * 0.7),
    }[rel]


@pytest.mark.parametrize("rel", ["left_of", "right_of", "above", "below", "inside", "on"])
def test_relations_are_exactly_the_old_ones(rel):
    rng = np.random.default_rng(3)
    model = SceneModel(DRAWABLE, SceneParams())
    for i in range(50):
        x, y, w, h = rng.uniform(0, 150, size=4)
        target = model.add(SceneObject(f"t{i}", "robot", "house", Box(x, y, w + 1, h + 1)))
        size = float(rng.uniform(5, 80))
        ext = (float(rng.uniform(5, 80)), float(rng.uniform(5, 80))) if i % 2 else None
        got = model.relation(rel, target, size, extent=ext)
        want = old_relation(rel, target.box, size, 10.0, ext)
        if rel in ("left_of", "right_of"):
            # The one deliberate change: the base lines up with the target's base, so the
            # sideways offset is as before but the height is not.
            hy = ext[1] / 2 if ext else size / 2
            want = (want[0], target.box.y + hy)
        assert got == want
        model.remove(target)


def test_a_written_label_is_found_by_its_text_and_skipped_by_pointing():
    model = SceneModel(DRAWABLE)
    house = model.add(SceneObject("house", "robot", "house", Box(10, 10, 40, 40), t=1.0))
    text = model.add(
        SceneObject("label_Cosmos", "robot", None, Box(10, 0, 30, 6), "label", text="Cosmos", t=2)
    )
    assert model.find("Cosmos") == [text]
    assert model.find("the houses") == [house]
    assert model.object_at((20, 3)) == (house, [house])  # the label is not selectable
    assert model.last() is house and model.labels() == ["house"]
