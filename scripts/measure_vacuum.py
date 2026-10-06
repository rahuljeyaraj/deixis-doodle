"""The vacuum's numbers from live runs (measure list M24 and M25).

    uv run python scripts/measure_vacuum.py out/vacuum/<run> [<run> ...] --out <folder>

M24: the robot's own reported positions while it cleaned each box, against that box (the
share inside, and the furthest outside, in cm). M25: last word to the echo starting to play.
Writes `summary.json` and `summary.txt` to `--out`. Nothing here talks to the robot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from pointdraw.measure import vacuum


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("out/measure/vacuum"))
    a = ap.parse_args(argv)
    res = vacuum.summary(a.runs)
    text = vacuum.lines(res)
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"made": dt.datetime.now().isoformat(timespec="seconds"), **res}
    (a.out / "summary.json").write_text(json.dumps(res, indent=1), "utf-8")
    (a.out / "summary.txt").write_text("\n".join(text) + "\n", "utf-8")
    print("\n".join(text))
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
