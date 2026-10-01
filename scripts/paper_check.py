"""Does the app see the paper properly on a real photo? (night 2 Oct 2026, item 8)

    uv run python scripts/paper_check.py
    uv run python scripts/paper_check.py --photo out/live/20261001-225450/frames/0123026.jpg
    uv run python scripts/paper_check.py --photo out/live/<run>/000123-camera.png  # S in live_run

`PaperCamera` and `segment_ink` had only run on virtual sheets. This takes a saved picture
of the sheet from a live run (640x360, turned 180 degrees: turned back and scaled to
1920x1080, bed parked), warps the paper top-down as the runtime does (0.5 mm per pixel),
and runs `segment_ink` twice: without the robot's strokes (every cross should be a mark)
and with them subtracted, as the runtime does (the crosses should disappear; anything left
is a mark the user drew, or a miss).

The robot's strokes are the crosses drawn in the live runs (`drawn` events), as the
planner draws them. Writes the warped image, both masks and an annotated picture to
`<out>/` and a summary to `<out>/paper_check.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from deixis.geometry import Box
from pointdraw.controller.config import DEFAULT_CONFIG_PATH, load_config
from pointdraw.controller.shapes import cross
from pointdraw.live.frames import camera_image
from pointdraw.live.view import BedView, warp_to_paper
from pointdraw.scene.ink import box_mm_to_px, ink_mask, segment_ink
from pointdraw.vision.aruco import DEFAULT_CALIBRATION_PATH, BedCalibration, fixed_bed_homography
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel


def robot_crosses(live: Path) -> list[dict]:
    out, seen = [], set()
    for tl in sorted(live.glob("*/timeline.jsonl")):
        for line in tl.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["kind"] == "drawn" and e.get("label") == "cross" and e.get("ink_box"):
                key = tuple(round(v, 1) for v in e["ink_box"])
                if key not in seen:
                    seen.add(key)
                    out.append({"run": tl.parent.name, "id": e["id"], "box": e["ink_box"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--live", type=Path, default=Path("out/live"))
    ap.add_argument("--photo", type=Path, default=None, help="default: the last saved picture")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--out", type=Path, default=Path("out/night/paper"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    paper = cfg.paper_box
    calib = BedCalibration.load_or_none(DEFAULT_CALIBRATION_PATH)
    if calib is None:
        print("no bed calibration")
        return 1
    bed = fixed_bed_homography(calib, lens=LensModel.load_or_none(DEFAULT_LENS_PATH))
    assert bed is not None
    view = BedView(bed, cfg.park_y, lambda: None)  # parked
    photo = a.photo or sorted(a.live.glob("*/frames/*.jpg"))[-1]
    s = 0.5  # the runtime's PaperCamera scale
    top = warp_to_paper(camera_image(photo), view, paper, s)

    drawn = robot_crosses(a.live)
    robot = []
    for d in drawn:
        x, y, w, h = d["box"]
        robot += cross(((x + w / 2), (y + h / 2)), max(w, h))
    seen_all = segment_ink(top, s)
    left = segment_ink(top, s, robot)

    def near_robot(b: Box) -> bool:
        c = b.center
        return any(
            abs(c[0] - (d["box"][0] + d["box"][2] / 2)) < d["box"][2] / 2 + 3
            and abs(c[1] - (d["box"][1] + d["box"][3] / 2)) < d["box"][3] / 2 + 3
            for d in drawn
        )

    rows = [
        {
            "box_mm": [round(v, 1) for v in (m.box_mm.x, m.box_mm.y, m.box_mm.w, m.box_mm.h)],
            "area_mm2": round(m.area_mm2, 1),
            "inside_a_robot_cross": near_robot(m.box_mm),
        }
        for m in left
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out / "paper-warped.png"), top)
    cv2.imwrite(str(a.out / "mask-all-ink.png"), 255 - ink_mask(top, s))
    cv2.imwrite(str(a.out / "mask-minus-robot.png"), 255 - ink_mask(top, s, robot))
    pic = top.copy()
    h = pic.shape[0]
    for pl in robot:
        pts = np.array([(p[0] / s, h - p[1] / s) for p in pl], np.int32)
        cv2.polylines(pic, [pts], False, (60, 170, 60), 1)
    for m, r in zip(left, rows, strict=True):
        x1, y1, x2, y2 = (int(round(v)) for v in box_mm_to_px(m.box_mm, h, s))
        colour = (0, 140, 255) if r["inside_a_robot_cross"] else (0, 0, 255)
        cv2.rectangle(pic, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), colour, 1)
    cv2.imwrite(str(a.out / "paper-marks.png"), pic)
    summary = {
        "photo": str(photo),
        "robot_crosses": len(drawn),
        "marks_without_subtraction": len(seen_all),
        "marks_after_subtraction": len(left),
        "left": rows,
    }
    (a.out / "paper_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(drawn)} robot crosses; marks seen: {len(seen_all)} without subtraction, ")
    print(f"{len(left)} after subtracting the robot's strokes:")
    for r in rows:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
