"""How fast the plotter pauses for a hand: from the live logs and from the scenes.

    uv run python scripts/pause_latency.py                       # both parts
    uv run python scripts/pause_latency.py --live-only

1. **Live logs** (`out/live/*/`): the camera loop's frame times (`hands.jsonl`), and for each
   time the interlock tripped, how long after a hand first came over the paper (+10 mm). Runs
   logged after 1 Oct 23:30 have every landmark; older ones only the index fingertip, so a
   hand that came in wrist first is seen late there (the trip then looks early). Runs logged
   from 2 Oct also have the plotter's events: trip to `pause_request`, to `paused`.
2. **Scenes on the real controller code** (`RunConfig(plotter="controller")`, mock AI): the
   virtual hand's true time over the paper -> interlock trip -> pause_request -> paused, the
   travel in between, and the scene checks' limits (1.7 s, 50 mm). Run with the default
   machine and with the live machine's feeds (`.cache/controller.json`).

3. **The real machine's worst case**, worked out: the scenes' firmware model holds one line
   ahead, but Marlin acknowledges a move once it is planned, and its planner holds about 16
   (`BLOCK_BUFFER_SIZE`), plus the 2 lines in flight. A pause request can therefore leave
   that many 2.5 mm pieces to run at the drawing feed, then the pen lift.

Writes `<out>/pause_latency.json`. Nothing moves; no network.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pointdraw.controller.config import (
    DEFAULT_CONFIG_PATH,
    MachineConfig,
    load_config,
    with_changes,
)

MARGIN = 10.0
ENTER_FRAMES = 2


def _over(h: dict[str, Any], paper: Any) -> bool:
    """A hand over the paper (+margin), in the gesture tracker's sense where possible."""
    if "mm" in h:
        lm = np.asarray(h["mm"], dtype=float) - [paper.x, paper.y]
    else:
        lm = np.asarray([h["tip"]], dtype=float)  # paper frame already
    inside = (
        (lm[:, 0] >= -MARGIN)
        & (lm[:, 0] <= paper.w + MARGIN)
        & (lm[:, 1] >= -MARGIN)
        & (lm[:, 1] <= paper.h + MARGIN)
    )
    return bool(inside.any())


def pct(v: list[float], q: float) -> float:
    return float(np.percentile(v, q)) if v else math.nan


def live_part(root: Path, machine: MachineConfig) -> dict[str, Any]:
    paper = machine.paper_box
    runs = []
    all_dt: list[float] = []
    all_trip: list[float] = []
    for d in sorted(p for p in root.iterdir() if (p / "hands.jsonl").exists()):
        lines = (d / "hands.jsonl").read_text(encoding="utf-8").splitlines()
        recs = [json.loads(x) for x in lines]
        recs = [r for r in recs if "hands" in r]
        if len(recs) < 20:
            continue
        ts = [r["t"] for r in recs]
        dt = np.diff(ts)
        dt = dt[dt < 1.0]  # skip the start-up gap
        all_dt += dt.tolist()
        landmarks = any("mm" in h for r in recs for h in r["hands"])
        trips = []
        first_over: float | None = None
        prev_clear = True
        for r in recs:
            over = any(_over(h, paper) for h in r["hands"])
            if over and first_over is None:
                first_over = r["t"]
            if not over and r["clear"]:
                first_over = None
            if prev_clear and not r["clear"]:
                trips.append(None if first_over is None else round(r["t"] - first_over, 3))
            prev_clear = r["clear"]
        seen = [x for x in trips if x is not None]
        all_trip += seen
        plotter = []
        tl = d / "timeline.jsonl"
        if tl.exists():
            evs = [json.loads(x) for x in tl.read_text(encoding="utf-8").splitlines()]
            pe = [e for e in evs if e["kind"] == "plotter"]
            trip_t = [e["t"] for e in evs if e["kind"] == "interlock" and not e["clear"]]
            for t in trip_t:
                req = next((e for e in pe if e["event"] == "pause_request" and e["t"] >= t), None)
                if req is None or req["t"] - t > 2:
                    continue
                done = next(
                    (
                        e
                        for e in pe
                        if e["event"] in ("paused", "job_end", "halted") and e["t"] >= req["t"]
                    ),
                    None,
                )
                plotter.append(
                    {
                        "trip_to_request_s": round(req["t"] - t, 3),
                        "request_to_paused_s": None
                        if done is None
                        else round(done["t"] - req["t"], 3),
                    }
                )
        runs.append(
            {
                "run": d.name,
                "frames": len(recs),
                "landmarks": landmarks,
                "fps_median": round(1 / statistics.median(dt), 1) if len(dt) else None,
                "frame_dt_p95_s": round(pct(dt.tolist(), 95), 3) if len(dt) else None,
                "trips": len(trips),
                "trip_after_hand_s": trips,
                "plotter": plotter,
            }
        )
    return {
        "runs": runs,
        "frame_dt_median_s": round(statistics.median(all_dt), 4) if all_dt else None,
        "fps_median": round(1 / statistics.median(all_dt), 1) if all_dt else None,
        "frame_dt_p95_s": round(pct(all_dt, 95), 3),
        "trip_after_hand_s": {
            "n": len(all_trip),
            "median": round(statistics.median(all_trip), 3) if all_trip else None,
            "max": round(max(all_trip), 3) if all_trip else None,
        },
    }


def scene_part(machine: MachineConfig, name: str) -> dict[str, Any]:
    from pointdraw.virtual import scenefile, world
    from pointdraw.virtual.world import RunConfig, over_paper

    rows = []
    for spec in scenefile.all_scenes():
        if spec.voice == "tts":
            continue
        res = world.run(spec, RunConfig(plotter="controller"), machine=machine)
        p = res.plotter.config.paper
        evs = res.plotter.events
        trips = [e["t"] for e in res.log.of("interlock") if not e["clear"]]
        for t in trips:
            # The true moment the hand came over the paper: step back while it was.
            t0 = t
            while t0 > 0 and over_paper(res.hand.state(t0 - 0.01), p.w, p.h):
                t0 -= 0.01
            req = next((e for e in evs if e.kind == "pause_request" and e.t >= t - 1e-6), None)
            if req is None or req.t - t > 0.5:
                continue  # nothing was moving: no pause to measure
            done = next(
                (e for e in evs if e.kind in ("paused", "job_end", "halted") and e.t >= req.t),
                None,
            )
            pts = [
                (x, y)
                for tt, x, y, _ in res.plotter.track
                if req.t <= tt <= (done.t if done else req.t)
            ]
            travel = sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))
            rows.append(
                {
                    "scene": spec.id,
                    "hand_to_trip_s": round(t - t0, 3),
                    "trip_to_request_s": round(req.t - t, 3),
                    "request_to_paused_s": None if done is None else round(done.t - req.t, 3),
                    "ended_by": None if done is None else done.kind,
                    "travel_mm": round(travel, 1),
                }
            )

    def stats(key: str) -> dict[str, Any]:
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return {"n": 0}
        return {
            "n": len(v),
            "median": round(statistics.median(v), 3),
            "max": round(max(v), 3),
        }

    return {
        "machine": name,
        "pauses": rows,
        "hand_to_trip_s": stats("hand_to_trip_s"),
        "trip_to_request_s": stats("trip_to_request_s"),
        "request_to_paused_s": stats("request_to_paused_s"),
        "travel_mm": stats("travel_mm"),
    }


def marlin_bound(cfg: MachineConfig, blocks: int = 16) -> dict[str, float]:
    """Worst case after a pause request on the real machine (no emergency parser needed)."""
    mm = (blocks + cfg.window) * cfg.max_segment_mm
    v = min(cfg.feed.normal / 60.0, cfg.limits.max_xy_mm_s)
    a = cfg.limits.accel_xy
    move_s = mm / v + v / (2 * a)  # at speed, then the stop
    dz = cfg.pen_up_z - cfg.pen_down_z
    vz = min(cfg.feed.z / 60.0, cfg.limits.max_z_mm_s)
    lift_s = dz / vz + vz / cfg.limits.accel_z
    return {
        "planner_blocks": blocks,
        "pen_down_mm": round(mm, 1),
        "pen_down_s": round(move_s, 2),
        "lift_s": round(lift_s, 2),
        "to_paused_s": round(move_s + lift_s, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--live", type=Path, default=Path("out/live"))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="the live machine config")
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out/night"))
    a = ap.parse_args(argv)
    live_cfg = load_config(a.config)
    out: dict[str, Any] = {"live": live_part(a.live, live_cfg), "marlin": marlin_bound(live_cfg)}
    if not a.live_only:
        feeds = {
            f"feed.{k}": getattr(live_cfg.feed, k) for k in ("normal", "fast", "preview", "travel")
        }
        feeds["hands_clear_hold_s"] = live_cfg.hands_clear_hold_s
        out["scenes"] = [
            scene_part(MachineConfig(), "default"),
            scene_part(with_changes(MachineConfig(), feeds), "live feeds"),
        ]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pause_latency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lv = out["live"]
    print(f"live loop: {lv['fps_median']} fps median, frame gap p95 {lv['frame_dt_p95_s']} s")
    print(f"live trips after a hand came over the paper: {lv['trip_after_hand_s']}")
    for r in lv["runs"]:
        print(f"  {r['run']}: {r['fps_median']} fps, trips {r['trip_after_hand_s']} {r['plotter']}")
    print(f"real machine worst case (live config): {out['marlin']}")
    for s in out.get("scenes", []):
        print(f"scenes ({s['machine']}): hand->trip {s['hand_to_trip_s']}")
        print(
            f"   trip->request {s['trip_to_request_s']}, request->paused {s['request_to_paused_s']}"
        )
        print(f"   travel {s['travel_mm']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
