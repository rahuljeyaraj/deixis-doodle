"""Make synthetic paper images with hand-drawn-looking marks, plus ground truth, for R1.

Writes PNGs and a labels.json (the format in data/r1/README.md) to .cache/r1/synthetic/.
Each sheet also carries the robot's own strokes and their polylines, so ink segmentation can be
checked against the truth too.

    uv run python scripts/r1_make_synthetic.py --n 20 --seed 1
    uv run python scripts/r1_make_synthetic.py --n 5 --check      # also score ink segmentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pointdraw.cosmos.scoring import iou
from pointdraw.scene.ink import segment_ink
from pointdraw.scene.synth import make_sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of sheets")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--marks", type=int, default=4, help="user marks per sheet")
    ap.add_argument("--robot", type=int, default=1, help="robot drawings per sheet")
    ap.add_argument("--mm-per-px", type=float, default=0.25)
    ap.add_argument("--out", default=".cache/r1/synthetic")
    ap.add_argument("--check", action="store_true", help="score ink segmentation on the sheets")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    images = []
    ious: list[float] = []
    for i in range(args.n):
        sheet = make_sheet(rng, n_marks=args.marks, n_robot=args.robot, mm_per_px=args.mm_per_px)
        name = f"sheet{i:03d}.png"
        cv2.imwrite(str(out / name), sheet.image)
        gt = sheet.ground_truth(name)
        images.append(gt)
        if args.check:
            found = segment_ink(sheet.image, sheet.mm_per_px, sheet.robot_polylines_mm)
            for m in gt["marks"]:
                best = max((iou(f.box_px, m["box_px"]) for f in found), default=0.0)
                ious.append(best)
            print(f"{name}: {len(gt['marks'])} marks, segmentation found {len(found)}")
    labels = {
        "version": 1,
        "source": f"scripts/r1_make_synthetic.py --n {args.n} --seed {args.seed}",
        "images": images,
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    n_marks = sum(len(im["marks"]) for im in images)
    n_q = sum(len(im["point_queries"]) for im in images)
    print(f"wrote {args.n} sheets, {n_marks} marks, {n_q} point queries to {out}")
    if ious:
        hit = sum(v >= 0.5 for v in ious)
        print(
            f"ink segmentation: {hit}/{len(ious)} marks with IoU >= 0.5, median IoU "
            f"{float(np.median(ious)):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
