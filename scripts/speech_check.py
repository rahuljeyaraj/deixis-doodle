"""The whole speech path offline: every scene spoken by Windows' voice, heard by the local ASR,
compared word by word with what was said (night 2 Oct 2026, item 12).

    uv run python scripts/speech_check.py
    uv run python scripts/speech_check.py --scenes S01 S28 --variations

Each scene runs with `RunConfig(voice="tts")` (System.Speech -> mixed stream -> Segmenter
with Silero -> Parakeet), mock agent, simulated plotter. The words said (the scene's lines,
in order) are aligned with the words heard (raw, before `repair_words`) and every
difference is listed: replaced, dropped or added words, with how often each happened. The
runtime's own repairs (`asr_repair` events) are listed too. Also whether each scene passed.

Nothing is played aloud: the voice is written to WAV files and mixed in memory.
Writes `<out>/speech_check.json`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower()).strip("'")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--variations", action="store_true", help="also the scenes' variations")
    ap.add_argument("--out", type=Path, default=Path("out/night"))
    a = ap.parse_args(argv)

    from pointdraw.virtual import scenefile, world
    from pointdraw.virtual.world import RunConfig

    specs = scenefile.all_scenes() if not a.scenes else [scenefile.find(s) for s in a.scenes]
    slips: Counter[tuple[str, str]] = Counter()
    repairs: Counter[tuple[str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        cfgs = [RunConfig(voice="tts")]
        if a.variations:
            v = spec.variations
            cfgs += [RunConfig(voice="tts", seed=s) for s in v.seeds]
            cfgs += [RunConfig(voice="tts", shift_s=sh) for sh in v.shifts]
        for cfg in cfgs:
            try:
                res = world.run(spec, cfg)
            except Exception as e:  # noqa: BLE001 - report and go on
                rows.append({"scene": spec.id, "variant": cfg.variant, "error": str(e)})
                print(f"{spec.id} {cfg.variant}: crashed: {e}", flush=True)
                continue
            said = [norm(w) for ln in spec.lines for w in ln.text.split() if norm(w)]
            heard = [norm(w[0]) for u in res.log.of("utterance") for w in u["words"] if norm(w[0])]
            diffs = []
            sm = difflib.SequenceMatcher(a=said, b=heard, autojunk=False)
            for op, i0, i1, j0, j1 in sm.get_opcodes():
                if op == "equal":
                    continue
                pair = (" ".join(said[i0:i1]), " ".join(heard[j0:j1]))
                slips[pair] += 1
                diffs.append({"said": pair[0], "heard": pair[1]})
            for e in res.log.of("asr_repair"):
                repairs[(e["heard"], e["fixed"])] += 1
            rep = res.report
            fails = [c["name"] for c in rep["checks"] if not c["ok"]]
            rows.append(
                {
                    "scene": spec.id,
                    "variant": cfg.variant,
                    "pass": rep["pass"],
                    "failed": fails,
                    "said": " ".join(said),
                    "heard": " ".join(heard),
                    "slips": diffs,
                }
            )
            mark = "PASS" if rep["pass"] else "FAIL"
            print(f"{mark} {spec.id:4} {cfg.variant:12} slips {diffs} {fails}", flush=True)
    summary = {
        "runs": len(rows),
        "pass": sum(1 for r in rows if r.get("pass")),
        "slips": [{"said": s, "heard": h, "times": n} for (s, h), n in slips.most_common()],
        "repairs": [{"heard": h, "fixed": f, "times": n} for (h, f), n in repairs.most_common()],
        "rows": rows,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "speech_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{summary['pass']}/{summary['runs']} pass")
    print("slips (said -> heard, times):")
    for s in summary["slips"]:
        print(f"  {s['said']!r:30} -> {s['heard']!r:30} x{s['times']}")
    print("runtime repairs:", summary["repairs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
