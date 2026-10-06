"""The vacuum's numbers from `scripts/vacuum_run.py` logs (measure list M24, M25).

- **M24** box cleaned vs box asked: for each box sent to the robot (`clean_area`, map mm),
  the positions the robot itself reported (`robot` events) while it cleaned it: from the
  first reported position inside the box to the last one, so the trip from and back to the
  dock isn't counted. The share inside the box, and the furthest any was outside, in cm.
  It measures how well the robot cleans the box it was sent, not where the person stood.
- **M25** last word to the guess: the end of the last word of the request to the echo
  starting to play ("Clean from there to here?"), from the `spoken` events. Runs logged
  before the `spoken` events existed fall back to when the line was handed to the voice
  (a little early), and say so.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ECHO_STARTS = ("Clean ", "That's a narrow strip", "That's past the edge")


def load(run: Path) -> list[dict]:
    lines = (Path(run) / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    return sorted((json.loads(x) for x in lines if x.strip()), key=lambda e: e["t"])


def outside_mm(p: tuple[float, float], box: list[float]) -> float:
    """How far the point is outside the box (0 inside)."""
    x, y, w, h = box
    dx = max(x - p[0], 0.0, p[0] - (x + w))
    dy = max(y - p[1], 0.0, p[1] - (y + h))
    return math.hypot(dx, dy)


def box_cleaned(ev: list[dict]) -> list[dict]:
    out = []
    sends = [e for e in ev if e["kind"] == "clean_area" and e.get("map_box")]
    for k, s in enumerate(sends):
        if not (s.get("reply") or {}).get("ok"):
            continue
        end = sends[k + 1]["t"] if k + 1 < len(sends) else math.inf
        stops = [e["t"] for e in ev if e["kind"] in ("stop", "dock") and e["t"] > s["t"]]
        end = min([end, *stops])
        pts, seen = [], set()
        for e in ev:
            if e["kind"] == "robot" and "x" in e and s["t"] <= e["t"] < end:
                key = (e["x"], e["y"])
                if key not in seen:  # the same report polled twice
                    seen.add(key)
                    pts.append((e["t"], e["x"], e["y"]))
        box = s["map_box"]
        inside = [i for i, (_, x, y) in enumerate(pts) if outside_mm((x, y), box) == 0]
        r = {"t": s["t"], "map_box": box, "reported": len(pts)}
        if inside:
            during = pts[inside[0] : inside[-1] + 1]
            outs = [outside_mm((x, y), box) for _, x, y in during]
            r |= {
                "during": len(during),
                "inside": sum(o == 0 for o in outs),
                "share_inside": round(sum(o == 0 for o in outs) / len(during), 3),
                "furthest_out_cm": round(max(outs) / 10, 1),
            }
        out.append(r)
    return out


def echo_latency(ev: list[dict]) -> list[dict]:
    out = []
    utts = [e for e in ev if e["kind"] == "utterance" and e.get("words")]
    spoken = [e for e in ev if e["kind"] == "spoken"]
    for p in (e for e in ev if e["kind"] == "preview"):
        said = next(
            (e for e in ev if e["kind"] == "say" and e["t"] >= p["t"] - 1e-6
             and e["text"].startswith(ECHO_STARTS)),
            None,
        )  # fmt: skip
        before = [u for u in utts if u["t"] <= p["t"] + 1e-6]
        if said is None or not before:
            continue
        end = before[-1]["words"][-1][2]
        play = next((e for e in spoken if e["text"] == said["text"] and e["t"] >= said["t"]), None)
        t = play["t"] if play else said["t"]
        out.append({"said": said["text"], "s": round(t - end, 3), "played": play is not None})
    return out


def summary(runs: list[Path]) -> dict:
    boxes, lat = [], []
    for r in runs:
        ev = load(r)
        boxes += [{**b, "run": str(r)} for b in box_cleaned(ev)]
        lat += echo_latency(ev)
    m24 = [b for b in boxes if "share_inside" in b]
    xs = sorted(x["s"] for x in lat)
    return {
        "runs": [str(r) for r in runs],
        "M24_box": {
            "runs": len(m24),
            "positions": sum(b["during"] for b in m24),
            "inside": sum(b["inside"] for b in m24),
            "share_inside": round(sum(b["inside"] for b in m24) / sum(b["during"] for b in m24), 3)
            if m24
            else None,
            "furthest_out_cm": max((b["furthest_out_cm"] for b in m24), default=None),
            "per_run": boxes,
        },
        "M25_last_word_to_echo_s": {
            "n": len(xs),
            "median": round(statistics.median(xs), 3) if xs else None,
            "min": xs[0] if xs else None,
            "max": xs[-1] if xs else None,
            "all_played": all(x["played"] for x in lat),
            "detail": lat,
        },
    }


def lines(res: dict) -> list[str]:
    out = []
    b = res["M24_box"]
    if b["runs"]:
        out.append(
            f"M24 box cleaned vs box asked: {100 * b['share_inside']:.0f} % of the robot's "
            f"{b['positions']} reported positions inside the box; at most "
            f"{b['furthest_out_cm']:.0f} cm outside ({b['runs']} runs)"
        )
    m = res["M25_last_word_to_echo_s"]
    if m["n"]:
        note = "" if m["all_played"] else " (some from the log line, not the sound: a bit early)"
        out.append(
            f"M25 last word to the echo: median {m['median']:.2f} s, range {m['min']:.2f} to "
            f"{m['max']:.2f} s, {m['n']} commands{note}"
        )
    return out or ["nothing to measure in these runs"]
