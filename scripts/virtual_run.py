"""Run scenes in the virtual world: no hardware, and by default no keys and no cost.

    uv run python scripts/virtual_run.py --scene S1                 # one scene, mock AI, text voice
    uv run python scripts/virtual_run.py --scene S1 --open          # ... and open the replay page
    uv run python scripts/virtual_run.py --all                      # every scene, summary table
    uv run python scripts/virtual_run.py --all --variations         # plus seeds, hover and timing
    uv run python scripts/virtual_run.py --all --ai live --record    # real Nemotron (Token Factory)
    uv run python scripts/virtual_run.py --scene S16 --cosmos live  # real Cosmos (endpoint up)
    uv run python scripts/virtual_run.py --scene S26 --voice tts     # Windows voice + Parakeet
    uv run python scripts/virtual_run.py --all --plotter controller  # the real controller code

Output per run goes to `out/virtual/<scene>/<variant>/`: `index.html` (open this),
`replay.gif`, `final.png`, `report.json` and `timeline.jsonl`. With `--all`, also
`out/virtual/index.html` and `summary.json`.

The plotter is always simulated: `VirtualPlotter`, or with `--plotter controller` the real
controller code in simulated time (`virtual.lockstep`). This script never opens a serial port.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import webbrowser
from pathlib import Path

from pointdraw.virtual import replay, scenefile, world
from pointdraw.virtual.world import RunConfig


def variants(spec: scenefile.SceneSpec, base: RunConfig, all_variations: bool) -> list[RunConfig]:
    out = [base]
    if not all_variations:
        return out
    v = spec.variations
    for s in v.seeds:
        out.append(RunConfig(**{**base.__dict__, "seed": s}))
    if v.hover_mm:
        out.append(RunConfig(**{**base.__dict__, "hover_mm": v.hover_mm}))
    for sh in v.shifts:
        out.append(RunConfig(**{**base.__dict__, "shift_s": sh}))
    return out


def run_one(spec, cfg: RunConfig, out_root: Path, gif: bool, speed: float = 1.0) -> dict:
    res = world.run(spec, cfg)
    d = out_root / spec.id / cfg.variant
    replay.write_report(res, d / "report.json")
    replay.write_timeline(res, d / "timeline.jsonl")
    replay.render_final(res, d / "final.png")
    if gif:
        replay.render_gif(res, d / "replay.gif", speed=speed)
    replay.write_page(res, d / "index.html", "replay.gif" if gif else None, "final.png")
    rep = res.report
    rep["_dir"] = str(d)
    agent = res.agent
    if hasattr(agent, "calls"):
        rep["token_factory_calls"] = agent.calls
        rep["token_factory_usage"] = getattr(agent, "usage", {})
    if hasattr(agent, "hits"):
        rep["cache_hits"], rep["cache_misses"] = agent.hits, agent.misses
    return rep


def summary_page(reports: list[dict], out_root: Path) -> Path:
    rows = []
    for r in reports:
        rel = Path(r["_dir"]).relative_to(out_root).as_posix()
        fails = [c["name"] for c in r["checks"] if not c["ok"]]
        m = r["metrics"]
        err = m.get("center_error_mm", {}).get("max")
        rows.append(
            f"<tr><td><a href='{rel}/index.html'>{r['scene']}</a></td><td>{r['title']}</td>"
            f"<td>{r['variant']}</td><td class='{'ok' if r['pass'] else 'bad'}'>"
            f"{'pass' if r['pass'] else 'FAIL'}</td><td>{'' if err is None else err}</td>"
            f"<td>{', '.join(fails)}</td></tr>"
        )
    n_pass = sum(r["pass"] for r in reports)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Virtual scenes</title><style>{replay.PAGE_CSS}</style></head><body>
<h1>Virtual scenes: {n_pass}/{len(reports)} pass</h1>
<div class="wrap"><table><tr><th>Scene</th><th>Title</th><th>Variant</th><th>Result</th>
<th>Max centre error (mm)</th><th>Failed checks</th></tr>{"".join(rows)}</table></div>
</body></html>"""
    p = out_root / "index.html"
    p.write_text(body, encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", action="append", help="scene id (S1) or a .yaml path; repeatable")
    g.add_argument("--all", action="store_true", help="every scene in tests/scenes")
    ap.add_argument("--ai", choices=["mock", "live"], default="mock")
    ap.add_argument("--cosmos", choices=["mock", "live"], default="mock")
    ap.add_argument("--voice", choices=["text", "tts"], default="text")
    ap.add_argument("--model", default=None, help="live agent model (default: the code default)")
    ap.add_argument("--record", action="store_true", help="live: save replies to the mock cache")
    ap.add_argument("--no-cache", action="store_true", help="mock: rule parser only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hover", type=float, default=0.0, help="finger height above the paper, mm")
    ap.add_argument("--shift", type=float, default=0.0, help="move speech later by this many s")
    ap.add_argument(
        "--ep",
        choices=["scene", "on", "off"],
        default="scene",
        help="Marlin EMERGENCY_PARSER in the simulated firmware",
    )
    ap.add_argument(
        "--plotter",
        choices=["virtual", "controller"],
        default="virtual",
        help="the simulated plotter, or the real controller code in simulated time",
    )
    ap.add_argument("--variations", action="store_true", help="also the scene's variations")
    ap.add_argument("--out", type=Path, default=Path("out/virtual"))
    ap.add_argument("--no-gif", action="store_true", help="skip the replay GIF (faster)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, e.g. 4 for the hero")
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    os.environ.pop("SERIAL_PORT", None)  # belt and braces: nothing here may use it

    specs = scenefile.all_scenes() if a.all else [scenefile.find(s) for s in a.scene]
    if a.voice == "text" and a.all:
        specs = [s for s in specs if s.voice != "tts"]  # tts-only scenes need real audio
    ep = None if a.ep == "scene" else a.ep == "on"
    base = RunConfig(
        ai=a.ai,
        cosmos=a.cosmos,
        voice=a.voice,
        seed=a.seed,
        hover_mm=a.hover,
        shift_s=a.shift,
        model=a.model,
        record=a.record,
        emergency_parser=ep,
        use_cache=not a.no_cache,
        plotter=a.plotter,
    )
    reports = []
    t0 = time.perf_counter()
    for spec in specs:
        for cfg in variants(spec, base, a.variations):
            try:
                rep = run_one(spec, cfg, a.out, gif=not a.no_gif, speed=a.speed)
            except Exception as e:  # noqa: BLE001 - one broken scene must not stop the rest
                logging.exception("%s %s crashed", spec.id, cfg.variant)
                rep = {
                    "scene": spec.id,
                    "title": spec.title,
                    "variant": cfg.variant,
                    "pass": False,
                    "checks": [
                        {"name": "crash", "ok": False, "detail": f"{type(e).__name__}: {e}"}
                    ],
                    "metrics": {},
                    "_dir": str(a.out / spec.id / cfg.variant),
                }
            reports.append(rep)
            fails = [f"{c['name']}: {c['detail']}" for c in rep["checks"] if not c["ok"]]
            m = rep.get("metrics", {})
            err = m.get("center_error_mm", {})
            print(
                f"{'PASS' if rep['pass'] else 'FAIL'}  {rep['scene']:<4} {cfg.variant:<16} "
                f"{rep['title'][:44]:<44} err={err.get('max', '-')}  "
                f"sim={m.get('sim_s', '-')}s wall={m.get('wall_s', '-')}s"
            )
            for f in fails:
                print(f"        - {f}")
    wall = time.perf_counter() - t0
    n = sum(r["pass"] for r in reports)
    print(f"\n{n}/{len(reports)} runs pass in {wall:.1f} s")
    if a.ai == "live":
        calls = sum(r.get("token_factory_calls", 0) for r in reports)
        print(f"Token Factory calls: {calls}")
    a.out.mkdir(parents=True, exist_ok=True)
    errs = [
        r["metrics"]["center_error_mm"]["max"]
        for r in reports
        if r.get("metrics", {}).get("center_error_mm")
    ]
    summary = {
        "runs": len(reports),
        "pass": n,
        "wall_s": round(wall, 1),
        "config": {"ai": a.ai, "cosmos": a.cosmos, "voice": a.voice, "model": a.model},
        "median_max_center_error_mm": round(statistics.median(errs), 3) if errs else None,
        "token_factory_calls": sum(r.get("token_factory_calls", 0) for r in reports),
        "reports": reports,
    }
    (a.out / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8"
    )
    page = (
        summary_page(reports, a.out)
        if len(reports) > 1
        else Path(reports[0]["_dir"]) / "index.html"
    )
    print(f"Open: {page.resolve()}")
    if a.open:
        webbrowser.open(page.resolve().as_uri())
    return 0 if n == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
