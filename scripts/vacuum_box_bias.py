"""Does the robot clean the box you asked for, or one beside it? (measure list M24)

    uv run python scripts/vacuum_box_bias.py --trials 8

Sends the robot a box, watches where it says it goes, and compares. The box is placed in a
different direction from the robot each time, so it arrives from a different side: a bias
that is the same every time can be cancelled by shifting the box we send, while one that
turns with the robot cannot, and must be reported instead.

The drive *to* the box is not counted -- that is travel, not cleaning -- so counting starts
the first time the robot is inside the box.

**Every box is placed around where the robot stood when the run began**, not around where it
is now. The first version of this chained them -- each box 900 mm from the robot's current
spot -- and the robot walked itself out of the room and out of the camera's view, one box at
a time (7 Oct). The patch is fixed at the start and never moves.

Nothing here uses the camera. Both the box and the positions are in the robot's own map, so
the result says nothing about our floor fit; it is only about the robot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from pointdraw.vacuum.link import LinkError, VacuumLink

# where the box goes relative to the robot, so it arrives from a different side each time
AROUND = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
AWAY_MM = 900.0  # how far the box's centre sits from the robot
SETTLE_S = 6.0  # no new position for this long: it has finished or given up


def robot_at(link: VacuumLink, wait_s: float = 1.5) -> tuple[float, float] | None:
    st = link.locate(wait_s=wait_s)
    r = st.get("robot")
    return (float(r["x"]), float(r["y"])) if r else None


def watch_clean(link: VacuumLink, limit_s: float) -> list[tuple[float, float]]:
    """Where the robot says it goes, until it stops reporting anything new."""
    seen: list[tuple[float, float]] = []
    t0 = last_new = time.monotonic()
    while time.monotonic() - t0 < limit_s:
        at = robot_at(link, wait_s=0.0)
        if at is not None and (not seen or at != seen[-1]):
            seen.append(at)
            last_new = time.monotonic()
        if seen and time.monotonic() - last_new > SETTLE_S:
            break
        time.sleep(0.3)
    return seen


def score(box: list[float], seen: list[tuple[float, float]]) -> dict | None:
    """How much of the cleaning happened inside the box, and how far off its middle it was."""
    x, y, w, h = box
    first = next((i for i, (px, py) in enumerate(seen)
                  if x <= px <= x + w and y <= py <= y + h), None)  # fmt: skip
    if first is None:
        return None
    pts = np.array(seen[first:], np.float64)
    inside = (pts[:, 0] >= x) & (pts[:, 0] <= x + w) & (pts[:, 1] >= y) & (pts[:, 1] <= y + h)
    out = np.maximum(
        np.maximum.reduce([x - pts[:, 0], pts[:, 0] - (x + w), y - pts[:, 1], pts[:, 1] - (y + h)]),
        0.0,
    )
    off = pts.mean(axis=0) - np.array([x + w / 2, y + h / 2])
    return {
        "n": int(len(pts)),
        "travel": int(first),
        "inside": int(inside.sum()),
        "share": round(float(inside.mean()), 3),
        "worst_out_mm": round(float(out.max())),
        "bias_mm": [round(float(off[0])), round(float(off[1]))],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--side", type=float, default=1000.0, help="the box's side, mm")
    ap.add_argument("--limit", type=float, default=150.0, help="seconds to watch each clean")
    ap.add_argument("--leash", type=float, default=3000.0,
                    help="stop if the robot gets this far from where it started, mm")  # fmt: skip
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    try:
        link = VacuumLink()
    except (LinkError, OSError) as e:
        print(f"could not start the robot's server: {e}")
        return 1
    trials = []
    home = robot_at(link, wait_s=2.0)
    if home is None:
        print("the robot will not say where it is")
        link.close()
        return 1
    print(f"the patch is fixed around ({home[0]:.0f}, {home[1]:.0f}), "
          f"{AWAY_MM:.0f} mm out at most")  # fmt: skip
    try:
        for i in range(a.trials):
            here = robot_at(link, wait_s=2.0)
            if here is None:
                print(f"trial {i + 1}: the robot will not say where it is")
                continue
            gone = math.dist(here, home)
            if gone > a.leash:
                print(f"trial {i + 1}: it has wandered {gone / 1000:.1f} m away -- stopping")
                break
            ux, uy = AROUND[i % len(AROUND)]
            n = math.hypot(ux, uy)
            cx = home[0] + ux / n * AWAY_MM  # around HOME, never around where it is now
            cy = home[1] + uy / n * AWAY_MM
            box = [cx - a.side / 2, cy - a.side / 2, a.side, a.side]
            where = {"frame": "map", "kind": "box", "box_mm": box}
            reply = link.clean_area(where)
            if not reply.get("ok"):
                print(f"trial {i + 1}: refused: {reply}")
                continue
            print(f"trial {i + 1}: box {a.side:.0f} mm at ({cx:.0f}, {cy:.0f}), "
                  f"the robot {AWAY_MM:.0f} mm to the {('N','E','S','W')[i % 4]}...",
                  end="", flush=True)  # fmt: skip
            seen = watch_clean(link, a.limit)
            got = score(box, seen)
            if got is None:
                print(f" never reached it ({len(seen)} positions)")
                continue
            got |= {"box": [round(v) for v in box], "from": [ux, uy]}
            trials.append(got)
            print(
                f" {got['inside']}/{got['n']} inside ({100 * got['share']:.0f} %), "
                f"off by {got['bias_mm']} mm, worst {got['worst_out_mm']} mm out",
                flush=True,
            )
            link.stop()
            time.sleep(2.0)
    finally:
        try:
            link.stop()
        finally:
            link.close()

    if not trials:
        print("no trials completed")
        return 1
    bias = np.array([t["bias_mm"] for t in trials], np.float64)
    shares = [t["share"] for t in trials]
    mean = bias.mean(axis=0)
    spread = bias.std(axis=0)
    steady = bool(np.all(np.abs(mean) > spread))
    print()
    print(f"M24 over {len(trials)} cleans: median {100 * statistics.median(shares):.0f} % inside, "
          f"worst {max(t['worst_out_mm'] for t in trials)} mm out")  # fmt: skip
    print(
        f"bias: mean ({mean[0]:+.0f}, {mean[1]:+.0f}) mm, "
        f"spread ({spread[0]:.0f}, {spread[1]:.0f}) mm"
    )
    print("the bias is steady enough to cancel" if steady else
          "the bias moves about more than it points one way: do not cancel it")  # fmt: skip
    out = a.out or Path("out/vacuum_bias") / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bias.json").write_text(
        json.dumps({"trials": trials, "mean_mm": mean.tolist(), "spread_mm": spread.tolist(),
                    "steady": steady}, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
