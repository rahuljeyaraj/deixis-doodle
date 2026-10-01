"""`segment_ink` on a real photo: the 1 Oct sheet with 16 robot crosses (night 2 Oct 2026).

The picture is a saved 640x360 live frame, turned back, scaled to the camera's size and
warped top-down with the fixed calibration (scripts/paper_check.py), at the runtime's
0.5 mm per pixel. The sheet has no hand-drawn marks; two small dots are not the robot's.
"""

import json
from pathlib import Path

import cv2

from pointdraw.controller.shapes import cross
from pointdraw.scene.ink import segment_ink

DATA = Path(__file__).parent / "data"


def test_robot_crosses_disappear_on_the_real_sheet():
    img = cv2.imread(str(DATA / "sheet_1oct_warped.png"))
    boxes = json.loads((DATA / "sheet_1oct_crosses.json").read_text())["boxes"]
    robot = [pl for x, y, w, h in boxes for pl in cross((x + w / 2, y + h / 2), max(w, h))]
    assert len(segment_ink(img, 0.5)) >= 12  # the crosses are seen as ink
    left = segment_ink(img, 0.5, robot)
    assert len(left) <= 2
    for m in left:  # only small dots remain, none on a cross's stroke
        assert m.area_mm2 < 10 and max(m.box_mm.w, m.box_mm.h) < 5
