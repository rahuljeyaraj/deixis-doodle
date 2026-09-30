"""Render an object or a scene spec to a PNG on the A5 sheet, for storyboards.

    uv run python scripts/render_drawing.py ear --at 105 74 --size 60
    uv run python scripts/render_drawing.py tree --variant 2 --size 40 -o out/renders/tree.png
    uv run python scripts/render_drawing.py --scene library/scenes/hero-senses.json --boxes
    uv run python scripts/render_drawing.py --label "Nemotron" --size 6
    uv run python scripts/render_drawing.py --list

Coordinates are mm in the paper frame: origin at the sheet's bottom-left as the user sees
it, +x right, +y away from the user. Prints the job stats (ink, travel, lifts, time).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pointdraw.drawing import Library, UnknownCategory, compose, plan
from pointdraw.drawing.render import A5, render
from pointdraw.drawing.scene import place_label, place_object

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", help="object name or synonym, e.g. 'ear' or 'kitty'")
    ap.add_argument("--scene", type=Path, help="scene spec JSON (see pointdraw.drawing.scene)")
    ap.add_argument("--label", help="render this text instead of an object")
    ap.add_argument("--at", nargs=2, type=float, default=[A5.w / 2, A5.h / 2], metavar=("X", "Y"))
    ap.add_argument(
        "--size", type=float, default=50.0, help="larger side in mm (cap height for labels)"
    )
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--boxes", action="store_true", help="show each item's box, dashed")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--px-per-mm", type=float, default=5.0)
    ap.add_argument("--list", action="store_true", help="list library categories")
    ap.add_argument("-o", "--out", type=Path, help="PNG path (default out/renders/<name>.png)")
    args = ap.parse_args()

    lib = Library.load()
    if args.list:
        for c in lib.categories:
            syn = lib.index["categories"][c].get("synonyms", [])
            print(f"{c:14s} {lib.index['categories'][c]['variants']} variants  {', '.join(syn)}")
        return 0

    try:
        if args.scene:
            spec = json.loads(args.scene.read_text(encoding="utf-8"))
            placed, paper = compose(spec, lib)
            stem = args.scene.stem
        elif args.label:
            placed, paper, stem = [place_label(args.label, tuple(args.at), args.size)], A5, "label"
        elif args.name:
            p = place_object(args.name, tuple(args.at), args.size, lib, args.variant)
            placed, paper, stem = [p], A5, f"{p.category.replace(' ', '_')}_{args.variant}"
        else:
            ap.error("give an object name, --label or --scene")
    except UnknownCategory as e:
        print(e, file=sys.stderr)
        return 2

    polylines, stats = plan(placed)
    out = args.out or ROOT / "out" / "renders" / f"{stem}.png"
    render(
        polylines,
        out,
        paper=paper,
        px_per_mm=args.px_per_mm,
        grid_mm=None if args.no_grid else 10.0,
        boxes=[p.box for p in placed] if args.boxes else (),
    )
    off = [p.id for p in placed if not paper.contains(p.box.center) or p.box.x < paper.x
           or p.box.y < paper.y or p.box.x2 > paper.x2 or p.box.y2 > paper.y2]  # fmt: skip
    if off:
        print(f"warning: these items leave the paper: {off}", file=sys.stderr)
    print(f"{out}: {stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
