"""Save where the four floor markers lie (the vacuum app's floor frame), from a tape measure.

    uv run python scripts/vacuum_floor.py --width 2400 --depth 1800
    uv run python scripts/vacuum_floor.py --centre 10=0,0 --centre 11=2410,15 \\
                                          --centre 12=2395,1790 --centre 13=-10,1805

Lay the four A4 floor sheets (IDs 10 to 13, hardware/print/floor-markers.pdf) on the corners
of a rectangle on the floor the camera sees. Stand on the near side, facing the patch:
10 near left, 11 near right, 12 far right, 13 far left. Which way each sheet is turned
doesn't matter. Measure from sheet centre to sheet centre, in millimetres.

With --width and --depth the four centres are the rectangle's corners. If the rectangle came
out a little skewed, give each centre with --centre ID=X,Y instead (10 is 0,0; x to your
right, y away from you). Writes .cache/vacuum/floor.json (or --out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pointdraw.layout import FLOOR_MARKER_IDS
from pointdraw.vacuum.floor import DEFAULT_LAYOUT_PATH, FloorLayout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--width", type=float, help="10 to 11 (and 13 to 12), mm")
    ap.add_argument("--depth", type=float, help="10 to 13 (and 11 to 12), mm")
    ap.add_argument("--centre", action="append", default=[], help="ID=X,Y in mm (all four)")
    ap.add_argument("--out", type=Path, default=DEFAULT_LAYOUT_PATH)
    a = ap.parse_args(argv)
    if a.centre:
        centres = {}
        for item in a.centre:
            mid, _, xy = item.partition("=")
            x, _, y = xy.partition(",")
            centres[int(mid)] = (float(x), float(y))
        if sorted(centres) != sorted(FLOOR_MARKER_IDS):
            print(f"give all four markers {list(FLOOR_MARKER_IDS)}, got {sorted(centres)}")
            return 1
        layout = FloorLayout(centres)
    elif a.width and a.depth:
        layout = FloorLayout.rectangle(a.width, a.depth)
    else:
        print("give --width and --depth, or --centre four times (see --help)")
        return 1
    path = layout.save(a.out)
    x, y, w, h = layout.area_mm
    print(f"saved {path}: the floor patch is {w / 1000:.2f} m wide and {h / 1000:.2f} m deep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
