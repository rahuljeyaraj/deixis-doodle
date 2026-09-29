"""Token Factory smoke test: list models, one tool-call test, latency per candidate model.

    uv run python scripts/tf_smoke.py                    # list + tool test + 10 calls/model
    uv run python scripts/tf_smoke.py --list-only
    uv run python scripts/tf_smoke.py --models nvidia/Nemotron-3_5-Lightning --n 5

Needs NEBIUS_API_KEY (from .env). Without it, prints what to do and exits 0.
Reasoning is off (`chat_template_kwargs.enable_thinking=false`) unless --thinking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointdraw.agent.nemotron import (  # noqa: E402
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    AgentUnavailable,
    NemotronAgent,
)
from pointdraw.agent.scoring import percentile  # noqa: E402
from pointdraw.config import get_settings  # noqa: E402

TEST_WORDS = "put a small tree here and write hello under it".split()


def no_key() -> int:
    print(
        "NEBIUS_API_KEY is not set, so nothing was called.\n"
        "To run this smoke test:\n"
        "  1. Get a key at https://tokenfactory.nebius.com (API keys page), or reuse\n"
        "     C:\\work\\deixis-doodle\\.env by copying it into this folder (it is gitignored).\n"
        "  2. Put NEBIUS_API_KEY=<key> in .env (see .env.example). Never commit it.\n"
        "  3. uv run python scripts/tf_smoke.py"
    )
    return 0


def _limit(x: object) -> str:
    """Per-model limit; Token Factory reports 1e10 for 'no per-model limit'."""
    if not isinstance(x, (int, float)):
        return "?"
    return "none" if x >= 1e9 else f"{x:g}"


def list_models(base_url: str, key: str, show_all: bool) -> list[dict]:
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/models?verbose=true",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        models = json.loads(r.read().decode("utf-8")).get("data", [])
    print(f"{len(models)} models on Token Factory (verbose list)")
    rows = []
    for m in models:
        mid = m.get("id", "")
        modality = (m.get("architecture") or {}).get("modality", "?")
        nvidia = "nemotron" in mid.lower() or mid.lower().startswith("nvidia/")
        vision = "image" in modality.split("->")[0]
        if not (show_all or nvidia or vision):
            continue
        price = m.get("pricing") or {}
        lim = m.get("per_request_limits") or {}

        def per_m(x: object) -> str:
            try:
                return f"{float(x) * 1e6:.2f}"
            except (TypeError, ValueError):
                return "?"

        rows.append(
            {
                "id": mid,
                "modality": modality,
                "context": m.get("context_length", "?"),
                "price": f"{per_m(price.get('prompt'))}/{per_m(price.get('completion'))}",
                "rpm": _limit(lim.get("requests_per_minute")),
                "tpm": _limit(lim.get("tokens_per_minute")),
                "tag": ("NVIDIA " if nvidia else "") + ("vision" if vision else ""),
            }
        )
    rows.sort(key=lambda r: (not r["tag"].startswith("NVIDIA"), r["id"]))
    print(f"{'id':48} {'modality':16} {'ctx':>7} {'$/M in/out':>12} {'rpm':>6} {'tpm':>9}  tag")
    for r in rows:
        print(
            f"{r['id']:48} {r['modality']:16} {r['context']!s:>7} {r['price']:>12} "
            f"{r['rpm']!s:>6} {r['tpm']!s:>9}  {r['tag']}"
        )
    return models


def bench(model: str, n: int, thinking: bool, tool_choice: str) -> dict:
    agent = NemotronAgent(model=model, thinking=thinking, tool_choice=tool_choice, timeout_s=30)
    lat, ok, retried, no_call = [], 0, 0, 0
    sample = ""
    for i in range(n):
        try:
            res = agent.run(TEST_WORDS)
        except AgentUnavailable as e:
            print(f"  {model}: call {i + 1} failed: {e}")
            return {"model": model, "error": str(e)}
        lat.append(res.latency_s)
        retried += res.attempts > 1
        no_call += not res.calls
        names = [c.name for c in res.calls]
        ok += names == ["place_drawing", "write_label"]
        if i == 0:
            sample = "; ".join(str(c) for c in res.calls) or f"(text) {res.speech!r}"
            print(f"  first reply: {sample}")
    return {
        "model": model,
        "n": n,
        "thinking": thinking,
        "tool_choice": tool_choice,
        "p50_s": round(percentile(lat, 50), 3),
        "p95_s": round(percentile(lat, 95), 3),
        "min_s": round(min(lat), 3),
        "max_s": round(max(lat), 3),
        "right_tools": ok,
        "retried": retried,
        "no_call": no_call,
        "usage_last": res.usage,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="*", default=[DEFAULT_MODEL, FALLBACK_MODEL])
    p.add_argument("--n", type=int, default=10, help="calls per model for latency")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--tool-choice", default="auto", choices=["required", "auto"])
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--all", action="store_true", help="list every model, not just NVIDIA/vision")
    args = p.parse_args()

    s = get_settings()
    key = s.nebius_api_key.get_secret_value()
    if not key:
        return no_key()

    try:
        models = list_models(s.nebius_base_url, key, args.all)
    except Exception as e:  # noqa: BLE001
        print(f"Listing models failed: {type(e).__name__}: {e}")
        print("Check the key in .env and https://tokenfactory.nebius.com status.")
        return 1
    if args.list_only:
        return 0

    ids = {m.get("id") for m in models}
    print(f"\nTool-call test: {' '.join(TEST_WORDS)!r}")
    print(f"(reasoning {'on' if args.thinking else 'off'}, tool_choice={args.tool_choice})")
    results = []
    for model in args.models:
        if model not in ids:
            print(f"  {model}: not in the model list, skipped")
            continue
        print(f"\n{model}: {args.n} calls")
        results.append(bench(model, args.n, args.thinking, args.tool_choice))
    print("\nSummary")
    for r in results:
        if "error" in r:
            print(f"  {r['model']}: ERROR {r['error']}")
            continue
        print(
            f"  {r['model']:44} p50 {r['p50_s']:.2f} s  p95 {r['p95_s']:.2f} s  "
            f"right tools {r['right_tools']}/{r['n']}  retried {r['retried']}  "
            f"no call {r['no_call']}"
        )
    out = ROOT / "out" / "tf_smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
