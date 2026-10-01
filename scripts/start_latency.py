"""Where the time goes from the end of speech to the pen going down (night 2 Oct, item 10).

    uv run python scripts/start_latency.py                      # every variant, all scenes
    uv run python scripts/start_latency.py --variants base fast_preview

Runs every text-voice scene with the mock agent on the real controller code
(`RunConfig(plotter="controller")`), once per variant, and reports per variant:

- the scenes' medians of `to_preview_s` (end of speech -> preview trace starts) and
  `to_pen_s` (end of speech -> drawing job starts), as in the scene report;
- for each drawing request, the stages: agent reply, preview job start, the trace itself,
  the quiet wait after it, and confirm -> drawing starts (travel and the interlock);
- how many scenes pass.

Variants: `base`; `fast_preview` (trace at 3,500 mm/min, within the 60 mm/s cap);
`quiet_1_5` (auto-confirm after 1.5 s of quiet); `diagonal` (trace corner to opposite
corner); `all` (the three together). Writes `<out>/start_latency.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from pointdraw.controller.config import MachineConfig, with_changes
from pointdraw.runtime.runtime import RuntimeParams

VARIANTS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "base": ({}, {}),
    "fast_preview": ({"feed.preview": 3500.0}, {}),
    "quiet_1_5": ({}, {"confirm_quiet_s": 1.5}),
    "diagonal": ({"preview_path": "diagonal"}, {}),
    "all": ({"feed.preview": 3500.0, "preview_path": "diagonal"}, {"confirm_quiet_s": 1.5}),
}


def stages(res: Any) -> list[dict[str, float]]:
    """Per drawing request: the time spent in each stage, in seconds."""
    log, evs = res.log, res.plotter.events
    out = []
    for c in log.of("confirm"):
        item = c["items"][0]
        prev = next(
            (e for e in reversed(log.of("preview")) if item in e["items"] and e["t"] <= c["t"]),
            None,
        )
        if prev is None:
            continue
        utt = next(
            (u for u in reversed(log.of("utterance")) if u["t"] <= prev["t"] and u["words"]), None
        )
        if utt is None:
            continue
        end = utt["words"][-1][2]
        tc = next((e for e in log.of("tool_calls") if e["utt"] == utt["id"]), None)
        p_start = next(
            (
                e
                for e in evs
                if e.kind == "job_start" and str(e.tag).startswith("preview") and e.t >= prev["t"]
            ),
            None,
        )
        done = next(
            (e for e in log.of("preview_done") if item in e["items"] and e["t"] >= prev["t"]), None
        )
        d_start = next((e for e in evs if e.kind == "job_start" and e.tag == item), None)
        if None in (tc, p_start, done, d_start):
            continue
        out.append(
            {
                "asr_s": utt["t"] - end,
                "agent_s": tc["latency_s"],
                "preview_start_s": p_start.t - (utt["t"] + tc["latency_s"]),
                "trace_s": done["t"] - p_start.t,
                "quiet_s": c["t"] - done["t"],
                "to_draw_s": d_start.t - c["t"],
                "to_pen_s": d_start.t - end,
            }
        )
    return out


def run_variant(name: str) -> dict[str, Any]:
    from pointdraw.virtual import scenefile, world
    from pointdraw.virtual.checks import score
    from pointdraw.virtual.world import RunConfig

    mch, prm = VARIANTS[name]
    machine = with_changes(MachineConfig(), mch) if mch else MachineConfig()
    params = RuntimeParams(**prm)
    rows, st, passed, fails = [], [], 0, []
    for spec in scenefile.all_scenes():
        if spec.voice == "tts":
            continue
        res = world.run(spec, RunConfig(plotter="controller"), machine=machine, params=params)
        rep = res.report if hasattr(res, "report") and res.report else score(res)
        m = rep["metrics"]
        passed += bool(rep["pass"])
        if not rep["pass"]:
            fails.append(spec.id)
        rows.append(
            {"scene": spec.id, "to_preview_s": m.get("to_preview_s"), "to_pen_s": m.get("to_pen_s")}
        )
        st += stages(res)

    def med(key: str, src: list[dict]) -> float | None:
        v = [r[key] for r in src if r.get(key) is not None]
        return round(statistics.median(v), 2) if v else None

    return {
        "variant": name,
        "machine": mch,
        "params": prm,
        "scenes": len(rows),
        "pass": passed,
        "failed": fails,
        "to_preview_s": med("to_preview_s", rows),
        "to_pen_s": med("to_pen_s", rows),
        "stages_median_s": {k: med(k, st) for k in (st[0] if st else {})},
        "requests": len(st),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--out", type=Path, default=Path("out/night"))
    a = ap.parse_args(argv)
    results = []
    for v in a.variants:
        r = run_variant(v)
        results.append(r)
        print(
            f"{v:13} pass {r['pass']}/{r['scenes']}  to_preview {r['to_preview_s']} s  "
            f"to_pen {r['to_pen_s']} s  stages {r['stages_median_s']}  failed {r['failed']}",
            flush=True,
        )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "start_latency.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
