"""Stress test: every scene, many seeds, hover heights and speech shifts, on both plotters
(night 2 Oct 2026, item 14).

    uv run python scripts/stress.py            # 20 seeds x hover 0, 8 x shift -0.3, 0, +0.3
    uv run python scripts/stress.py --seeds 5 --plotters controller --scenes S21 S22

Each run is `world.run` with the mock agent (recorded replies, else the rule parser) and the
text voice. Runs go to a process pool. For each scene and plotter it reports the pass rate,
and for each kind of failure (a failed check, a crash, the lockstep harness not settling)
how often it happened and an example. Writes `<out>/stress.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def one(task: tuple[str, str, int, float, float]) -> dict[str, Any]:
    sid, plotter, seed, hover, shift = task
    from pointdraw.virtual import scenefile, world
    from pointdraw.virtual.world import RunConfig

    cfg = RunConfig(plotter=plotter, seed=seed, hover_mm=hover, shift_s=shift)
    t0 = time.perf_counter()
    try:
        res = world.run(scenefile.find(sid), cfg)
    except Exception as e:  # noqa: BLE001 - every failure is data here
        kind = "not settled" if "did not settle" in str(e) else f"crash {type(e).__name__}"
        return {
            "scene": sid,
            "plotter": plotter,
            "variant": cfg.variant,
            "pass": False,
            "fails": [kind],
            "details": [traceback.format_exc(limit=3)[-600:]],
            "wall_s": round(time.perf_counter() - t0, 2),
        }
    rep = res.report
    bad = [c for c in rep["checks"] if not c["ok"]]
    return {
        "scene": sid,
        "plotter": plotter,
        "variant": cfg.variant,
        "pass": rep["pass"],
        "fails": [c["name"] for c in bad],
        "details": [f"{c['name']}: {c['detail']}" for c in bad],
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--hovers", type=float, nargs="+", default=[0.0, 8.0])
    ap.add_argument("--shifts", type=float, nargs="+", default=[-0.3, 0.0, 0.3])
    ap.add_argument("--plotters", nargs="+", default=["virtual", "controller"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("out/night"))
    a = ap.parse_args(argv)

    from pointdraw.virtual import scenefile

    specs = [s for s in scenefile.all_scenes() if s.voice != "tts"]
    if a.scenes:
        specs = [s for s in specs if s.id in a.scenes]
    tasks = [
        (s.id, p, seed, h, sh)
        for s in specs
        for p in a.plotters
        for seed in range(a.seeds)
        for h in a.hovers
        for sh in a.shifts
    ]
    print(f"{len(tasks)} runs on {a.workers} workers", flush=True)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)} in {time.perf_counter() - t0:.0f} s", flush=True)

    per: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        per[(r["scene"], r["plotter"])].append(r)
    kinds: Counter[str] = Counter()
    examples: dict[str, dict] = {}
    by_cond: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        for k, d in zip(r["fails"], r["details"], strict=False):
            key = f"{r['scene']} {r['plotter']}: {k}"
            kinds[key] += 1
            examples.setdefault(key, {"variant": r["variant"], "detail": d})
            by_cond[key][r["variant"].split("-", 1)[1] if "-" in r["variant"] else "base"] += 1
    table = {
        f"{sid} {p}": {
            "runs": len(rs),
            "pass": sum(r["pass"] for r in rs),
            "rate": round(sum(r["pass"] for r in rs) / len(rs), 3),
        }
        for (sid, p), rs in sorted(per.items())
    }
    out = {
        "runs": len(rows),
        "pass": sum(r["pass"] for r in rows),
        "wall_s": round(time.perf_counter() - t0, 1),
        "settings": {"seeds": a.seeds, "hovers": a.hovers, "shifts": a.shifts},
        "per_scene": table,
        "failure_kinds": [
            {"kind": k, "times": n, "example": examples[k], "where": dict(by_cond[k])}
            for k, n in kinds.most_common()
        ],
        "not_settled": sum(1 for r in rows if "not settled" in r["fails"]),
        "crashes": sum(1 for r in rows if any(f.startswith("crash") for f in r["fails"])),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stress.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{out['pass']}/{out['runs']} pass in {out['wall_s']} s; not settled "
          f"{out['not_settled']}, crashes {out['crashes']}")  # fmt: skip
    for k, v in table.items():
        if v["rate"] < 1:
            print(f"  {k:28} {v['pass']}/{v['runs']}")
    print("failure kinds:")
    for fk in out["failure_kinds"][:40]:
        print(f"  {fk['times']:4}  {fk['kind']}  e.g. {fk['example']['variant']}: "
              f"{fk['example']['detail'][:150]}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
