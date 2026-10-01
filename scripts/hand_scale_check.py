"""Does running MediaPipe on a smaller image move the fingertip? (night 2 Oct 2026, item 5)

    uv run python scripts/hand_scale_check.py
    uv run python scripts/hand_scale_check.py --scales 0.5 0.75 --limit 0.5

For each folder of recorded frames, one `HandTracker` per scale is fed the same frames in
order (VIDEO mode, as live), and the index fingertip is mapped to bed mm with the fixed
calibration (bed parked). Reports how far each scale's fingertip is from the full-size one:
median, 95th percentile and worst, in mm, over frames where both found a hand.

Frames: `out/pose_probe/*/` (1920x1080 photos of hands over the paper, 1 Oct) and the live
runs' saved pictures (640x360, scaled up: they say less about fine detail).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pointdraw.live.frames import camera_image, frame_paths
from pointdraw.vision.aruco import DEFAULT_CALIBRATION_PATH, BedCalibration, fixed_bed_homography
from pointdraw.vision.hands import INDEX_TIP, HandTracker
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel


def tips(paths: list[Path], scale: float, to_mm, loader) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    with HandTracker(scale=scale) as tr:
        for i, p in enumerate(paths):
            hf = tr.process(loader(p), i / 30.0, to_mm)
            if not hf.hands:
                out.append(None)
                continue
            h = max(hf.hands, key=lambda h: h.score)
            out.append(h.landmarks_mm[INDEX_TIP])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--limit", type=float, default=0.5, help="mm")
    ap.add_argument("--live", nargs="*", default=["out/night/frames-hands"])
    ap.add_argument("--out", type=Path, default=Path("out/night/hand_scale.json"))
    a = ap.parse_args(argv)
    calib = BedCalibration.load_or_none(DEFAULT_CALIBRATION_PATH)
    if calib is None:
        print("no bed calibration")
        return 1
    lens = LensModel.load_or_none(DEFAULT_LENS_PATH)
    bed = fixed_bed_homography(calib, lens=lens)
    assert bed is not None
    sets: list[tuple[str, list[Path], object]] = []
    for d in sorted(Path("out/pose_probe").glob("*/")):
        ps = sorted(d.glob("*.jpg"))
        if ps:
            sets.append((f"probe/{d.name}", ps, lambda p: cv2.imread(str(p))))
    for d in a.live:
        ps = frame_paths(d)
        if ps:
            sets.append((f"live/{Path(d).name}", ps, camera_image))
    result: dict[str, dict] = {}
    for name, paths, loader in sets:
        full = tips(paths, 1.0, bed.to_mm, loader)
        result[name] = {"frames": len(paths), "with_hand": sum(t is not None for t in full)}
        for s in a.scales:
            other = tips(paths, s, bed.to_mm, loader)
            d = [
                float(np.linalg.norm(f - o))
                for f, o in zip(full, other, strict=True)
                if f is not None and o is not None
            ]
            missed = sum(1 for f, o in zip(full, other, strict=True) if (f is None) != (o is None))
            result[name][str(s)] = {
                "n": len(d),
                "median_mm": round(float(np.median(d)), 3) if d else None,
                "p95_mm": round(float(np.percentile(d, 95)), 3) if d else None,
                "max_mm": round(max(d), 3) if d else None,
                "found_differs": missed,
            }
            print(f"{name:22} scale {s}: {result[name][str(s)]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
