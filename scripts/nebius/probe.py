"""The 24-hour Token Factory probe: one Nemotron call a minute, timings only (measure list M35).

    python3 probe.py --request probe_request.json --calls 1440 --every 60      # in the Job
    uv run python scripts/nebius/probe.py --request out/probe/probe_request.json --calls 60
                                                                               # from a laptop

Standard library only, so the Job needs no pip install. The key comes from the environment
variable NEBIUS_API_KEY (in the Job: `--env-secret`, from Nebius SecretStash). It is never
printed, and neither is any reply text: each call prints ONE line of JSON with only

    {"i": 17, "utc": "...", "status": 200, "s": 0.84, "prompt": 3021, "cached": 2944,
     "completion": 41, "calls": 1}

(`status` is the HTTP code, or "timeout" / "neterror"; `calls` is how many tool calls came
back, so a reply that isn't a tool call shows as 0). Calls start on a fixed one-minute grid,
so a slow call doesn't push the rest later. The last line is {"done": ...} with the counts.
`scripts/probe_summary.py` turns the lines into p50, p95, p99, errors and cached share.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

DEFAULT_URL = "https://api.tokenfactory.nebius.com/v1/"


def call_once(url: str, key: str, body: bytes, timeout: float) -> dict:
    """One request. Returns only the status, the time and the token counts."""
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    t = time.perf_counter()
    out: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            out["status"] = r.status
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        out |= {
            "prompt": usage.get("prompt_tokens"),
            "cached": details.get("cached_tokens"),
            "completion": usage.get("completion_tokens"),
            "calls": len(msg.get("tool_calls") or []),
        }
    except urllib.error.HTTPError as e:
        out["status"] = e.code  # the body is not read: it is never printed
    except TimeoutError:  # socket.timeout too, since Python 3.10
        out["status"] = "timeout"
    except (urllib.error.URLError, OSError) as e:
        out["status"] = "neterror"
        out["why"] = type(getattr(e, "reason", e)).__name__
    out["s"] = round(time.perf_counter() - t, 3)
    return out


def run(
    body: bytes,
    calls: int,
    every: float,
    url: str,
    key: str,
    timeout: float = 30.0,
    once: Callable[[str, str, bytes, float], dict] = call_once,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    emit: Callable[[str], None] = print,
) -> dict:
    start = now()
    counts: dict[str, int] = {}
    for i in range(calls):
        wait = start + i * every - now()
        if wait > 0:
            sleep(wait)
        r = once(url, key, body, timeout)
        line = {"i": i, "utc": datetime.now(UTC).isoformat(timespec="seconds"), **r}
        text = json.dumps(line)
        if key and key in text:  # can't happen; never print a key, whatever comes back
            text = json.dumps({"i": i, "status": "redacted"})
        emit(text)
        counts[str(r["status"])] = counts.get(str(r["status"]), 0) + 1
    done = {"done": calls, "status_counts": counts, "seconds": round(now() - start, 1)}
    emit(json.dumps(done))
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--request", required=True, help="the request body (probe_request.json)")
    ap.add_argument("--calls", type=int, default=1440)
    ap.add_argument("--every", type=float, default=60.0, help="seconds between call starts")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--url", default=os.environ.get("NEBIUS_BASE_URL", DEFAULT_URL))
    a = ap.parse_args(argv)
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key:
        print(json.dumps({"error": "NEBIUS_API_KEY is not set"}), flush=True)
        return 2
    with open(a.request, "rb") as f:
        body = f.read()
    json.loads(body)  # fail early on a broken file
    print(json.dumps({"start": datetime.now(UTC).isoformat(timespec="seconds"),
                      "calls": a.calls, "every_s": a.every, "url": a.url,
                      "request_bytes": len(body)}), flush=True)  # fmt: skip
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    run(body, a.calls, a.every, a.url, key, a.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
