"""Accuracy map from the 1 Oct crosses: where the finger was, where the pen drew, what the
camera sees (night 2 Oct 2026, item 7).

    uv run python scripts/accuracy_map.py
    uv run python scripts/accuracy_map.py --photo out/live/20261001-225450/frames/0123026.jpg

1. **Finger to ink.** Every cross drawn live (`drawn` events in `out/live/*/timeline.jsonl`)
   has the place the deixis resolver chose from the fingertip hold (`item` -> `where`). The
   ink centre is where the plotter was told to draw. The difference is what the planner
   changed (kept inside the drawable area, moved off another drawing).
2. **Ink to camera.** On a saved picture of the sheet (640x360, turned 180 degrees; turned
   back and scaled to 1920x1080, bed parked), the paper is warped top-down with the fixed
   calibration and each cross is found near where it was drawn: the centre of its two
   strokes, from the dark-pixel profiles. Camera minus plotter is the calibration's real
   error across the sheet (if the plotter puts ink where it is told, about 2 mm on 1 Oct).

Writes `<out>/accuracy.json` and a picture of the sheet with arrows (`accuracy-map.png`):
red from where the finger pointed to where the pen drew, blue from where the pen drew to
where the camera sees the cross (both drawn 5x longer than they are).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pointdraw.controller.config import DEFAULT_CONFIG_PATH, load_config
from pointdraw.live.frames import camera_image
from pointdraw.live.view import BedView, warp_to_paper
from pointdraw.vision.aruco import DEFAULT_CALIBRATION_PATH, BedCalibration, fixed_bed_homography
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel

HOOK_FIX = "20261001-224600"  # commit 557d0c4 (hooked index points) at 22:46
MIN_STRENGTH = 3.0  # mean darkness along the arms; paper alone is about 0


def crosses(live: Path) -> list[dict[str, Any]]:
    out = []
    for tl in sorted(live.glob("*/timeline.jsonl")):
        evs = [json.loads(x) for x in tl.read_text(encoding="utf-8").splitlines()]
        items = {e["id"]: e for e in evs if e["kind"] == "item"}
        for e in evs:
            if e["kind"] != "drawn" or e.get("label") != "cross" or not e.get("ink_box"):
                continue
            it = items.get(e["id"], {})
            where = it.get("where") or {}
            x, y, w, h = e["ink_box"]
            out.append(
                {
                    "run": tl.parent.name,
                    "id": e["id"],
                    "t": e["t"],
                    "size_mm": max(w, h),
                    "ink": [x + w / 2, y + h / 2],
                    "finger": where.get("mean_mm") if where.get("source") == "gesture" else None,
                    "rule": (where.get("evidence") or {}).get("rule"),
                    "before_hook_fix": tl.parent.name < HOOK_FIX,
                }
            )
    return out


def _refine(f: np.ndarray, i: int) -> float:
    """Sub-sample peak position from a parabola through f[i-1], f[i], f[i+1]."""
    if 0 < i < len(f) - 1:
        a, b, c = f[i - 1], f[i], f[i + 1]
        d = a - 2 * b + c
        if d < 0:
            return i + 0.5 * (a - c) / d
    return float(i)


def find_cross(dark: np.ndarray, c_mm, size: float, s: float, search: float) -> dict | None:
    """The "+" near paper point `c_mm` in the dark-pixel map (row 0 = the far edge).

    Each candidate centre within `search` mm is scored by the mean darkness along both arms
    of a cross that size (at most 15 mm of arm, so a big neighbour's long stroke alone can't
    win). Returns the best centre in paper mm and its score (mean darkness along the arms).
    """
    h, w = dark.shape
    cx, cy = c_mm[0] / s - 0.5, h - c_mm[1] / s - 0.5
    arm = int(round(min(size / 2, 15.0) / s))
    r = int(round(search / s))
    colsum = np.vstack([np.zeros((1, w)), np.cumsum(dark, axis=0)])  # down the columns
    rowsum = np.hstack([np.zeros((h, 1)), np.cumsum(dark, axis=1)])  # along the rows
    best, score = None, np.full((2 * r + 1, 2 * r + 1), -np.inf)
    for j, dy in enumerate(range(-r, r + 1)):
        py = int(round(cy)) + dy
        if py - arm < 0 or py + arm + 1 > h:
            continue
        for i, dx in enumerate(range(-r, r + 1)):
            px = int(round(cx)) + dx
            if px - arm < 0 or px + arm + 1 > w:
                continue
            vert = colsum[py + arm + 1, px] - colsum[py - arm, px]
            horiz = rowsum[py, px + arm + 1] - rowsum[py, px - arm]
            # Both arms must carry ink: the weaker one counts double.
            score[j, i] = (vert + horiz + 2 * min(vert, horiz)) / (4 * (2 * arm + 1))
    if not np.isfinite(score).any():
        return None
    j, i = np.unravel_index(int(np.argmax(score)), score.shape)
    fy = _refine(score[:, i], j) - r
    fx = _refine(score[j, :], i) - r
    px, py = round(cx) + fx, round(cy) + fy
    mm = [float((px + 0.5) * s), float((h - py - 0.5) * s)]
    best = {"mm": mm, "strength": round(float(score[j, i]), 2)}
    return best


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    u = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + u * ab)))


def crowded(r: dict, rows: list[dict], clear_mm: float = 6.0) -> bool:
    """Another cross's stroke passes within `clear_mm` of this one's centre (ambiguous)."""
    c = r["ink"]
    for o in rows:
        oc, half = o["ink"], o["size_mm"] / 2
        if np.hypot(*np.subtract(oc, c)) < 1.0:
            continue  # the same mark
        strokes = [
            ((oc[0] - half, oc[1]), (oc[0] + half, oc[1])),
            ((oc[0], oc[1] - half), (oc[0], oc[1] + half)),
        ]
        if min(_seg_dist(c, a, b) for a, b in strokes) < clear_mm:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--live", type=Path, default=Path("out/live"))
    ap.add_argument("--photo", type=Path, default=None, help="default: the last saved picture")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--mm-per-px", type=float, default=0.25)
    ap.add_argument("--search", type=float, default=8.0, help="mm around the drawn place")
    ap.add_argument("--out", type=Path, default=Path("out/night"))
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
    img = camera_image(photo)
    s = a.mm_per_px
    top = warp_to_paper(img, view, paper, s)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), 4.0 / s)
    dark = np.clip(bg - gray - 6.0, 0, None)  # ink: darker than the paper around it

    rows = crosses(a.live)
    for r in rows:
        r["finger_to_ink_mm"] = (
            None
            if r["finger"] is None
            else round(float(np.hypot(*np.subtract(r["ink"], r["finger"]))), 2)
        )
        f = find_cross(dark, r["ink"], r["size_mm"], s, a.search)
        r["camera"] = None if f is None else [round(v, 2) for v in f["mm"]]
        r["strength"] = None if f is None else f["strength"]
        r["camera_minus_ink_mm"] = (
            None
            if f is None
            else [round(f["mm"][0] - r["ink"][0], 2), round(f["mm"][1] - r["ink"][1], 2)]
        )
    for r in rows:
        r["crowded"] = crowded(r, rows)
        # The same place drawn again (the "cross in the middle" bug) is one mark on the sheet.
        r["repeat"] = any(
            o is not r
            and rows.index(o) < rows.index(r)
            and np.hypot(*np.subtract(o["ink"], r["ink"])) < 1.0
            for o in rows
        )
    found = [
        r
        for r in rows
        if r["camera"] is not None
        and r["strength"] >= MIN_STRENGTH
        and not r["crowded"]
        and not r["repeat"]
    ]
    cam_err = [float(np.hypot(*r["camera_minus_ink_mm"])) for r in found]
    fin_err = [r["finger_to_ink_mm"] for r in rows if r["finger_to_ink_mm"] is not None]

    def stats(v: list[float]) -> dict[str, Any]:
        return (
            {"n": len(v), "median": round(statistics.median(v), 2), "max": round(max(v), 2)}
            if v
            else {"n": 0}
        )

    mean_off = (
        [round(statistics.fmean(r["camera_minus_ink_mm"][i] for r in found), 2) for i in (0, 1)]
        if found
        else None
    )
    summary = {
        "photo": str(photo),
        "crosses": len(rows),
        "finger_to_ink_mm": stats(fin_err),
        "camera_minus_ink_mm": stats(cam_err),
        "camera_minus_ink_mean_xy_mm": mean_off,
        "measured": len(found),
        "left_out": {
            "crowded": sum(r["crowded"] for r in rows),
            "repeats": sum(r["repeat"] for r in rows),
        },
        "rows": rows,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "accuracy.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # The picture: the sheet top-down at 4 px per mm, contrast stretched; each drawn cross in
    # thin green where the plotter drew it; arrows 5x their length.
    g = cv2.normalize(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX)
    pic = cv2.cvtColor(cv2.resize(g, None, fx=s * 4, fy=s * 4), cv2.COLOR_GRAY2BGR)
    pic = cv2.copyMakeBorder(pic, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ppm = 4.0

    def to_pic(p) -> tuple[int, int]:
        return int(round(p[0] * ppm)), int(round(pic.shape[0] - p[1] * ppm))

    def arrow(a_mm, b_mm, colour) -> None:
        a_px = to_pic(a_mm)
        b_px = to_pic([a_mm[0] + 5 * (b_mm[0] - a_mm[0]), a_mm[1] + 5 * (b_mm[1] - a_mm[1])])
        if a_px != b_px:
            cv2.arrowedLine(pic, a_px, b_px, colour, 2, cv2.LINE_AA, tipLength=0.2)

    for r in rows:
        (cx, cy), half = r["ink"], r["size_mm"] / 2
        cv2.line(pic, to_pic((cx - half, cy)), to_pic((cx + half, cy)), (60, 170, 60), 1)
        cv2.line(pic, to_pic((cx, cy - half)), to_pic((cx, cy + half)), (60, 170, 60), 1)
    for r in rows:
        if r["finger"] is not None and r["finger_to_ink_mm"]:
            cv2.arrowedLine(
                pic, to_pic(r["finger"]), to_pic(r["ink"]), (40, 40, 220), 2, cv2.LINE_AA, 0, 0.1
            )
        if r["camera"] is not None and r["strength"] >= MIN_STRENGTH and not r["repeat"]:
            colour = (150, 150, 150) if r["crowded"] else (220, 110, 0)
            arrow(r["ink"], r["camera"], colour)
    for i, text in enumerate(
        (
            "green: crosses as drawn   blue: drawn -> seen by the camera (x5; grey: crowded)",
            "red: fingertip hold -> drawn (the planner moved it)",
        )
    ):
        cv2.putText(
            pic, text, (6, 14 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
        )
    cv2.imwrite(str(a.out / "accuracy-map.png"), pic)
    print(f"{len(rows)} crosses; finger->ink {summary['finger_to_ink_mm']}")
    print(f"camera-ink {summary['camera_minus_ink_mm']}, mean offset {mean_off} mm")
    for r in rows:
        print(
            f"  {r['run']} {r['id']:7} ink {r['ink']} finger {r['finger']} "
            f"cam {r['camera']} d={r['camera_minus_ink_mm']} str={r['strength']}"
            f"{' crowded' if r['crowded'] else ''}{' repeat' if r['repeat'] else ''}"
            f"{' (before hook fix)' if r['before_hook_fix'] else ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
