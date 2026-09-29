"""R1: run the three Cosmos tasks against a live endpoint and report accuracy and latency.

Tasks (plan 5.1): (a) label a pointed crop, (b) point to a semantic place, (c) find a named mark.
Needs a running endpoint (scripts/nebius/cosmos_endpoint.sh) and COSMOS_BASE_URL plus
COSMOS_API_KEY in .env. Costs endpoint time only (about $1.80/h while it is up).

    uv run python scripts/r1_eval.py --labels .cache/r1/synthetic/labels.json --limit 5
    uv run python scripts/r1_eval.py --labels data/r1/labels.json
    uv run python scripts/r1_eval.py --labels data/r1/labels.json --tasks label --think
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from pointdraw.config import get_settings
from pointdraw.cosmos.client import CosmosClient
from pointdraw.cosmos.prompts import PROMPT_VERSION
from pointdraw.cosmos.r1 import TASKS, evaluate, load_dataset
from pointdraw.cosmos.scoring import format_table, summarise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels", default=".cache/r1/synthetic/labels.json")
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma list of label,point,find")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--model", default=None, help="default: COSMOS_MODEL or nvidia/Cosmos3-Nano")
    ap.add_argument("--base-url", default=None, help="default: COSMOS_BASE_URL")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--think", action="store_true", help="ask for reasoning (slower)")
    ap.add_argument("--out", default=None, help="results JSON (default .cache/r1/results-*.json)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    labels = Path(args.labels)
    if not labels.is_file():
        print(
            f"No labels file at {labels}. Make one with scripts/r1_make_synthetic.py, "
            "or see data/r1/README.md."
        )
        return 1
    s = get_settings()
    if not (args.base_url or s.cosmos_base_url):
        print(
            "COSMOS_BASE_URL is not set. Start an endpoint (scripts/nebius/README.md), then put "
            "its https URL in .env as COSMOS_BASE_URL."
        )
        return 1
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    client = CosmosClient(
        s,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        think=True if args.think else False,
    )
    cases = load_dataset(labels)
    print(f"R1 on {labels} ({len(cases)} images), model {client.model}, prompts {PROMPT_VERSION}")

    def show(t) -> None:
        mark = "ok " if t.correct else "BAD"
        print(f"  {mark} {t.task:5s} {t.latency_s:5.2f}s {t.image}: {t.query!r} {t.detail}")
        sys.stdout.flush()

    trials = evaluate(client, cases, tasks, args.limit, on_trial=show)
    summary = summarise(trials)
    print()
    print(format_table(summary))
    out = Path(args.out or f".cache/r1/results-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "labels": str(labels),
                "model": client.model,
                "prompt_version": PROMPT_VERSION,
                "think": bool(args.think),
                "summary": summary,
                "trials": [asdict(t) for t in trials],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
