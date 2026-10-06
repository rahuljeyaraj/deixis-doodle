"""The 24-hour probe (M35): one line per call with timings and counts only, never the key;
calls on a fixed grid; the summary's percentiles, error counts and the key check."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = "tf-secret-key-123"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = load("probe", ROOT / "scripts" / "nebius" / "probe.py")
summ = load("probe_summary", ROOT / "scripts" / "probe_summary.py")


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_one_line_per_call_on_a_grid_and_no_key():
    clock, out, starts = Clock(), [], []
    replies = iter(
        [
            {"status": 200, "s": 1.2, "prompt": 7000, "cached": 4000, "completion": 60, "calls": 1},
            {"status": 429, "s": 0.1},
            {"status": "timeout", "s": 30.0, "why": f"echo {KEY}"},  # can't happen; still safe
        ]
    )

    def once(url, key, body, timeout):
        starts.append(clock.t)
        r = next(replies)
        clock.t += r["s"]
        return r

    done = probe.run(b"{}", 3, 60.0, "https://x/v1/", KEY, once=once, sleep=clock.sleep,
                     now=clock.now, emit=out.append)  # fmt: skip
    assert starts == [0.0, 60.0, 120.0]  # a slow call doesn't push the rest
    assert all(KEY not in line for line in out)
    rows = [json.loads(x) for x in out]
    assert rows[0]["status"] == 200 and rows[2] == {"i": 2, "status": "redacted"}
    assert done["status_counts"] == {"200": 1, "429": 1, "timeout": 1}


def test_summary_and_key_check(tmp_path):
    rows = [
        {"i": i, "utc": f"t{i}", "status": 200, "s": 1.0 + i / 100, "prompt": 7000,
         "cached": 3500, "completion": 60, "calls": 1}
        for i in range(100)
    ]  # fmt: skip
    rows[5] = {"i": 5, "utc": "t5", "status": 503, "s": 0.2}
    log = tmp_path / "job-log.txt"
    log.write_text("\n".join(f"[2026-10-10T10:00:00Z] {json.dumps(r)}" for r in rows) + "\n")
    res = summ.summary(summ.lines(log))
    assert res["calls"] == 100 and res["ok"] == 99 and res["errors"] == {"503": 1}
    assert res["p50_s"] == 1.5 and res["cached_share"] == 0.5
    env = tmp_path / ".env"
    env.write_text(f"NEBIUS_API_KEY={KEY}\n")
    assert summ.key_in(log, env) is False
    log.write_text(log.read_text() + f"oops {KEY}\n")
    assert summ.key_in(log, env) is True
    assert "KEY CHECK FAILED" in "\n".join(summ.text(res, True))
