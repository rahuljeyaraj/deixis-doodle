"""The probe's numbers (measure list M35): p50, p95, p99, errors, cached tokens.

    uv run python scripts/probe_summary.py out/probe/job-log.txt [--out <folder>]

Reads the lines `scripts/nebius/probe.py` printed (a Job log fetched with
`probe_job.sh fetch-logs`, or a laptop run's output): anything before the first "{" on a
line, such as a log timestamp, is skipped. Errors are listed by code, never dropped.

It also checks that the key never reached the log: it reads NEBIUS_API_KEY from .env and
says only "the key is not in the log" (or that it is, and then the log must be deleted and
the key rotated before any number is quoted). The key itself is never printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path


def lines(path: Path) -> list[dict]:
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        i = raw.find("{")
        if i < 0:
            continue
        try:
            d = json.loads(raw[i:])
        except ValueError:
            continue
        if "i" in d and "status" in d:
            out.append(d)
    return out


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summary(rows: list[dict]) -> dict:
    rows = sorted({r["i"]: r for r in rows}.values(), key=lambda r: r["i"])  # repeats once
    ok = [r for r in rows if r["status"] == 200]
    secs = [r["s"] for r in ok]
    codes: dict[str, int] = {}
    for r in rows:
        if r["status"] != 200:
            codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1
    prompt = sum(r.get("prompt") or 0 for r in ok)
    cached = sum(r.get("cached") or 0 for r in ok)
    res = {
        "calls": len(rows),
        "ok": len(ok),
        "errors": codes,
        "no_tool_call": sum(1 for r in ok if r.get("calls") == 0),
        "first_utc": rows[0]["utc"] if rows else None,
        "last_utc": rows[-1]["utc"] if rows else None,
    }
    if secs:
        res |= {
            "p50_s": round(statistics.median(secs), 3),
            "p95_s": round(pct(secs, 0.95), 3),
            "p99_s": round(pct(secs, 0.99), 3),
            "max_s": round(max(secs), 3),
            "cached_share": round(cached / prompt, 3) if prompt else None,
            "mean_prompt_tokens": round(prompt / len(ok)),
            "mean_completion_tokens": round(sum(r.get("completion") or 0 for r in ok) / len(ok)),
        }
    return res


def key_in(path: Path, env: Path = Path(".env")) -> bool | None:
    """True if the key from .env appears in the file; None if there is no key to check."""
    key = ""
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "NEBIUS_API_KEY":
                key = v.strip().strip("\"'")
    if not key:
        return None
    return key in path.read_text(encoding="utf-8", errors="replace")


def text(res: dict, key: bool | None) -> list[str]:
    out = []
    if res.get("p50_s") is not None:
        err = ", ".join(f"{n} x {c}" for c, n in res["errors"].items()) or "none"
        out.append(
            f"M35 probe: {res['ok']} of {res['calls']} calls answered; p50 {res['p50_s']:.2f} s, "
            f"p95 {res['p95_s']:.2f} s, p99 {res['p99_s']:.2f} s; errors: {err}"
        )
        if res["cached_share"] is not None:
            out.append(f"    cached prompt tokens: {100 * res['cached_share']:.0f} %")
        out.append(f"    from {res['first_utc']} to {res['last_utc']} (UTC)")
    else:
        out.append(f"M35 probe: no answered calls in {res['calls']} lines")
    if key is None:
        out.append("key check: no NEBIUS_API_KEY in .env here, so not checked")
    elif key:
        out.append("KEY CHECK FAILED: the key is in this log. Delete the log, rotate the key, "
                   "and quote no number from it.")  # fmt: skip
    else:
        out.append("key check: the key is not in the log")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write summary.json/.txt here")
    a = ap.parse_args(argv)
    res = summary(lines(a.log))
    k = key_in(a.log)
    t = text(res, k)
    print("\n".join(t))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        res |= {"made": dt.datetime.now().isoformat(timespec="seconds"), "log": str(a.log),
                "key_in_log": k}  # fmt: skip
        (a.out / "summary.json").write_text(json.dumps(res, indent=1), "utf-8")
        (a.out / "summary.txt").write_text("\n".join(t) + "\n", "utf-8")
    return 1 if k else 0


if __name__ == "__main__":
    raise SystemExit(main())
