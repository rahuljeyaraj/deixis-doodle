"""Label R1 photos: draw a box around each mark, then type its name.

For each image in a folder, a window opens with the detected paper outline drawn in green (only
a hint). Drag a box around one mark, press SPACE or ENTER, repeat, and press ESC when that
image is done. The terminal then asks for each box's label and synonyms. The result is merged
into <folder>/labels.json (existing entries are kept unless --redo).

    uv run python scripts/r1_label.py data/r1/photos

Point queries are not labelled here; add them to labels.json by hand (see data/r1/README.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pointdraw.scene.paper import find_paper_quad

EXTS = {".jpg", ".jpeg", ".png"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--redo", action="store_true", help="relabel images already in labels.json")
    ap.add_argument("--max-side", type=int, default=1400, help="display size")
    args = ap.parse_args()

    folder = Path(args.folder)
    out = folder / "labels.json"
    data = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"version": 1}
    images = {im["file"]: im for im in data.get("images", [])}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"no images in {folder}")
        return 1
    for path in files:
        if path.name in images and not args.redo:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable {path.name}")
            continue
        scale = min(1.0, args.max_side / max(img.shape[:2]))
        view = cv2.resize(img, None, fx=scale, fy=scale)
        quad = find_paper_quad(img)
        if quad is not None:
            pts = (quad * scale).astype(int).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], True, (0, 200, 0), 2)
        rois = cv2.selectROIs(f"{path.name}: box each mark, ESC when done", view, False)
        cv2.destroyAllWindows()
        marks = []
        for x, y, w, h in rois:
            box = [round(v / scale, 1) for v in (x, y, x + w, y + h)]
            label = input(f"{path.name} box {box}: label? ").strip()
            if not label:
                continue
            syn = input("  synonyms (comma separated, optional)? ").strip()
            marks.append(
                {
                    "label": label,
                    "synonyms": [s.strip() for s in syn.split(",") if s.strip()],
                    "box_px": box,
                }
            )
        images[path.name] = {
            "file": path.name,
            "marks": marks,
            "point_queries": images.get(path.name, {}).get("point_queries", []),
        }
        data["images"] = list(images.values())
        out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"saved {len(marks)} marks for {path.name}")
    print(f"labels: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
