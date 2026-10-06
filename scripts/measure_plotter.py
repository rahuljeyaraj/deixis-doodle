"""The plotter's numbers for the submission, from live runs (measure list M01 to M09).

    uv run python scripts/measure_plotter.py logs out/live/<run> [<run> ...] --out <folder>
    uv run python scripts/measure_plotter.py targets out/live/<run> --sheet crosses-A --out <folder>

`logs` (no photos needed): M04 last word to the guess, M05 hand before word, M06 draw time
per object, M07 the take's length from the log, M09 "a bit bigger" during the guess.

`targets` (a run on a printed sheet from `scripts/target_sheets.py`): finds the printed marks
in the run's first parked photo, finds each drawing's ink (`scripts/m2_score.py`), and
scores each drawing against its target: M01 and M03 with a crosses sheet, M02 with a circles
sheet. Writes `targets-found.png` (the marks it found, numbered) and `crops.png` (each
drawing); look at both before trusting the numbers.

Both write `summary.json` and `summary.txt` (the numbers in plain words) to `--out`.
Nothing here moves the plotter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pointdraw.measure import accuracy, logs
from pointdraw.measure import targets as T

ROOT = Path(__file__).resolve().parents[1]
S = 0.1  # mm a pixel of the top-down picture the marks are found in


def _m2_score():
    spec = importlib.util.spec_from_file_location("m2_score", ROOT / "scripts" / "m2_score.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m2_score"] = mod
    spec.loader.exec_module(mod)
    return mod


def write(out: Path, res: dict, text: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    res = {"made": dt.datetime.now().isoformat(timespec="seconds"), **res}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    (out / "summary.txt").write_text("\n".join(text) + "\n", "utf-8")
    print("\n".join(text))
    print(f"\nwritten to {out}")


def cmd_logs(a) -> int:
    res = logs.summary(a.runs)
    write(a.out, res, logs.lines(res))
    return 0


def cmd_targets(a) -> int:
    m2 = _m2_score()
    run: Path = a.run
    events = m2.load_timeline(run)
    photos = [e for e in events if e["kind"] == "photo"]
    if not photos:
        print("no parked photo in this run: was it run with live_run.py after 2 Oct?")
        return 1
    first = run / photos[0]["file"]
    img = cv2.imread(str(first))
    size = (img.shape[1], img.shape[0])
    view, cfg_paper = m2.bed_view(size)
    paper = m2.run_paper(run) or cfg_paper
    from pointdraw.live.view import warp_to_paper

    top = warp_to_paper(img, view, paper, S, 0.0)
    found, info = T.find_targets(top, S, a.sheet)
    kind = T.SHEETS[a.sheet]["kind"]
    a.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out / "targets-found.png"), _marks_picture(top, found))

    cmds, pics = m2.score(run, size, view, paper)
    wheres = _wheres(events)
    rows = [
        {
            "id": c.id,
            "label": c.label,
            "ink_xy": c.ink_xy,
            "ink_size": c.ink_size,
            "where": wheres.get(c.id),
        }
        for c in cmds
        if c.label == ("cross" if kind == "crosses" else "circle")
    ]
    res, summ = accuracy.match(rows, found, kind)
    if pics:
        h = max(p.shape[0] for p in pics)
        strip = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 4, cv2.BORDER_CONSTANT) for p in pics]
        cv2.imwrite(str(a.out / "crops.png"), np.hstack(strip))
    text = [
        f"sheet {a.sheet}: found {info['found']} of {info['of']} printed marks "
        f"(printer scale {info['scale']:.3f}, turned {info['turn_deg']:.1f} deg)",
        *accuracy.lines(summ),
    ]
    write(
        a.out,
        {"run": str(run), "photo": str(first), "sheet": info, "summary": summ, "drawings": res,
         "targets": T.as_dicts(found)},
        text,
    )  # fmt: skip
    return 0


def _wheres(events: list[dict]) -> dict:
    """The last `where` logged for each item (paper mm, with cov_mm2)."""
    out = {}
    for e in events:
        if e["kind"] == "item" and e.get("where"):
            out[e["id"]] = e["where"]
    return out


def _marks_picture(top: np.ndarray, found: list[T.Target]) -> np.ndarray:
    vis = top.copy()
    H = vis.shape[0]
    for t in found:
        p = (int(t.x / S), int(H - t.y / S))
        col = (0, 160, 0) if t.found else (0, 0, 255)
        cv2.circle(vis, p, int(20 / 2 / S) if not t.d else int(t.d / 2 / S), col, 3)
        cv2.putText(vis, str(t.id), (p[0] + 40, p[1] - 40), 0, 2.0, col, 4)
    return cv2.resize(vis, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("logs", help="M04 to M09 from timelines")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("out/measure/logs"))
    p = sub.add_parser("targets", help="M01 to M03 on a printed sheet")
    p.add_argument("run", type=Path)
    p.add_argument("--sheet", required=True, choices=list(T.SHEETS))
    p.add_argument("--out", type=Path, default=Path("out/measure/targets"))
    a = ap.parse_args(argv)
    return cmd_logs(a) if a.cmd == "logs" else cmd_targets(a)


if __name__ == "__main__":
    raise SystemExit(main())
