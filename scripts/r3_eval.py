"""Spike R3: score a Token Factory model, or the offline fallback parser, on the R3 set.

    uv run python scripts/r3_eval.py --fallback                     # offline, no key
    uv run python scripts/r3_eval.py --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
    uv run python scripts/r3_eval.py --model nvidia/Nemotron-3_5-Lightning --tool-choice auto

Needs NEBIUS_API_KEY (in .env) for --model. Writes per-item results to out/r3/.
The built-in category list is used unless --library is given, so runs are comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointdraw.agent.fallback_parser import parse as fallback_parse  # noqa: E402
from pointdraw.agent.prompts import (  # noqa: E402
    BUILTIN_CATEGORIES,
    PROMPT_VERSION,
    AgentContext,
    load_categories,
)
from pointdraw.agent.scoring import (  # noqa: E402
    item_matches,
    load_items,
    names_match,
    percentile,
)

DATA = ROOT / "tests" / "agent" / "data" / "r3_utterances.jsonl"


def run(args: argparse.Namespace) -> int:
    items = load_items(Path(args.data))
    if args.only:
        items = [it for it in items if any(t in it.tags or t == it.id for t in args.only)]
    cats = load_categories() if args.library else list(BUILTIN_CATEGORIES)

    agent = None
    if not args.fallback:
        from pointdraw.agent.nemotron import AgentUnavailable, NemotronAgent
        from pointdraw.config import get_settings

        settings = get_settings()
        if not settings.nebius_api_key.get_secret_value():
            print(
                "NEBIUS_API_KEY is not set, so the model can't be scored.\n"
                "  1. Copy C:\\work\\deixis-doodle\\.env into this folder (it is gitignored),\n"
                "     or set NEBIUS_API_KEY in your shell.\n"
                "  2. Re-run this command. Or score the offline parser: --fallback"
            )
            return 0
        agent = NemotronAgent(
            settings,
            model=args.model,
            thinking=args.thinking,
            tool_choice=args.tool_choice,
            timeout_s=args.timeout,
            max_retries=args.retries,
            categories=cats,
        )
    name = "fallback" if agent is None else agent.model.split("/")[-1]
    if agent is not None:
        name += f"_{args.tool_choice}" + ("_think" if args.thinking else "")

    rows = []
    for it in items:
        ctx = AgentContext.from_dict(it.context)
        row: dict = {"id": it.id, "tags": it.tags, "text": it.text}
        if agent is None:
            calls = fallback_parse(it.words, ctx, cats)
            row.update(latency_s=0.0, attempts=1, errors=[], speech=None)
        else:
            try:
                res = agent.run(it.words, ctx)
            except AgentUnavailable as e:
                print(f"{it.id}: unavailable: {e}")
                row.update(error=str(e), calls=[], ok=False, names_ok=False)
                rows.append(row)
                continue
            calls = res.calls
            row.update(
                latency_s=round(res.latency_s, 3),
                attempts=res.attempts,
                errors=res.errors,
                speech=res.speech,
                usage=res.usage,
            )
        got = [c.to_dict() for c in calls]
        row.update(calls=got, ok=item_matches(it, got), names_ok=names_match(it, got))
        rows.append(row)
        mark = "ok " if row["ok"] else ("~  " if row["names_ok"] else "XX ")
        extra = f" {row.get('latency_s', 0):.2f}s" if agent else ""
        if row.get("attempts", 1) > 1:
            extra += " (retried)"
        print(f"{mark}{it.id:5} {it.text!r:50}{extra}")
        if not row["ok"] and args.verbose:
            print(f"      got:      {json.dumps(got)}")
            print(f"      expected: {json.dumps(it.expected)}")
            if row.get("errors"):
                print(f"      errors:   {row['errors']}")

    summary = summarise(rows, items, name)
    print_summary(summary)
    out_dir = ROOT / "out" / "r3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


def summarise(rows: list[dict], items: list, name: str) -> dict:
    n = len(rows)
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for t in r["tags"]:
            by_tag[t].append(r["ok"])
    exp_ask = {it.id for it in items if any(c["name"] == "ask_user" for c in it.expected)}
    got_ask = {r["id"] for r in rows if any(c["name"] == "ask_user" for c in r["calls"])}
    return {
        "name": name,
        "prompt_version": PROMPT_VERSION,
        "items": n,
        "exact": sum(r["ok"] for r in rows),
        "tool_names": sum(r["names_ok"] for r in rows),
        "no_call": sum(1 for r in rows if not r["calls"]),
        "retried": sum(1 for r in rows if r.get("attempts", 1) > 1),
        "unavailable": sum(1 for r in rows if r.get("error")),
        "ask_expected": len(exp_ask),
        "ask_correct": len(exp_ask & got_ask),
        "ask_unneeded": len(got_ask - exp_ask),
        "latency_p50_s": round(percentile(lat, 50), 3),
        "latency_p95_s": round(percentile(lat, 95), 3),
        "prompt_tokens_mean": _mean_usage(rows, "prompt_tokens"),
        "completion_tokens_mean": _mean_usage(rows, "completion_tokens"),
        "by_tag": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_tag.items())},
    }


def _mean_usage(rows: list[dict], key: str) -> float:
    xs = [r["usage"][key] for r in rows if r.get("usage", {}).get(key)]
    return round(sum(xs) / len(xs), 1) if xs else 0.0


def print_summary(s: dict) -> None:
    n = s["items"] or 1
    print(f"\n== {s['name']} (prompt {s['prompt_version']}) ==")
    print(f"exact match      {s['exact']}/{s['items']} ({100 * s['exact'] / n:.0f}%)")
    print(f"tool names right {s['tool_names']}/{s['items']}")
    print(f"no tool call     {s['no_call']}   retried {s['retried']}")
    print(
        f"ask_user         {s['ask_correct']}/{s['ask_expected']} when needed, "
        f"{s['ask_unneeded']} unneeded"
    )
    if s["latency_p50_s"]:
        print(f"latency          p50 {s['latency_p50_s']:.2f} s, p95 {s['latency_p95_s']:.2f} s")
        print(
            f"tokens/item      prompt {s['prompt_tokens_mean']:.0f}, "
            f"completion {s['completion_tokens_mean']:.0f}"
        )
    print("by tag:", ", ".join(f"{k} {v}" for k, v in s["by_tag"].items()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fallback", action="store_true", help="score the offline rule parser")
    g.add_argument("--model", default=None, help="Token Factory model ID")
    p.add_argument("--thinking", action="store_true", help="leave reasoning on (slow)")
    p.add_argument("--tool-choice", default="required", choices=["required", "auto"])
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--library", action="store_true", help="use library/index.json categories")
    p.add_argument("--only", nargs="*", help="item IDs or tags to run")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
