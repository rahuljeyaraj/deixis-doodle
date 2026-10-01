"""Run a logged live session again, offline, with today's code (no Ender, no camera).

    uv run python scripts/replay_live.py out/live/20261002-101500
    uv run python scripts/replay_live.py out/live/20261002-101500 --all    # every event

Prints what the runtime does now with the same hands and words, then compares the counts
(questions asked, drawings made) with what happened live. See `pointdraw.live.replay`.
Needs a run logged by `live_run.py` after 1 Oct 2026, 23:30 (hands.jsonl with landmarks).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pointdraw.controller.config import DEFAULT_CONFIG_PATH, load_config
from pointdraw.live.replay import SHOWN, load, replay, summary
from pointdraw.runtime.events import EventLog


def line(e: dict, t0: float) -> str:
    rest = {k: v for k, v in e.items() if k not in ("t", "kind", "words", "context")}
    return f"[{e['t'] - t0:7.2f}] {e['kind']:10} {rest}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path, help="a folder in out/live/")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="machine config JSON")
    ap.add_argument("--all", action="store_true", help="print every event, not just the main")
    a = ap.parse_args(argv)
    try:
        session = load(a.run)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return 1
    rt, agent = replay(session, load_config(a.config))
    for e in rt.log.events:
        if a.all or e["kind"] in SHOWN:
            print(line(e, session.t0))
    now = summary(rt.log)
    then = summary(EventLog(session.events))
    print()
    print(f"agent replies: {agent.recorded} recorded, {agent.parsed} from the rule parser")
    for k in now:
        print(f"{k:7} live {then[k]:3}   replay {now[k]:3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
