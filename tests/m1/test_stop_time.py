"""scripts/stop_time.py scoring (M08), offline: a few noisy frames are not a late move, and a
trial whose holder wasn't moving at the stop is not counted (2 Oct tracks, rescored 6 Oct)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("stop_time", ROOT / "scripts" / "stop_time.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stop_time"] = mod
    spec.loader.exec_module(mod)
    return mod


def track(stop_after_s: float, speed_px_s: float, noise_at=()):
    ts = np.arange(-1.0, 3.0, 1 / 30)
    xs = 1400 - speed_px_s * np.clip(ts + 1.0, 0, 1.0 + stop_after_s)
    for t0 in noise_at:  # three frames in a row read 5 px off
        xs[(ts >= t0) & (ts < t0 + 0.1)] -= 5
    return [[round(float(t), 3), round(float(x), 1)] for t, x in zip(ts, xs, strict=True)]


def test_noise_is_not_a_late_move(tmp_path):
    st = load()
    runs = [{"track": track(0.05, 110, noise_at=(1.5, 2.2))} for _ in range(3)]
    p = tmp_path / "stop_time.json"
    p.write_text(json.dumps(runs))
    s = st.rescore(p)
    assert s["counted"] == 3 and s["max_s"] < 0.12 and s["pass"]


def test_a_holder_that_never_moved_is_not_counted(tmp_path):
    st = load()
    runs = [{"track": track(0.05, 110)}, {"track": track(0.0, 0)}]
    p = tmp_path / "stop_time.json"
    p.write_text(json.dumps(runs))
    s = st.rescore(p)
    assert s["counted"] == 1 and s["trials"] == 2
    assert s["line"].startswith("M08 stop:")


def test_a_slow_stop_fails():
    st = load()
    s = st.summarise([{"stop_s": 0.04}, {"stop_s": 1.4}])
    assert not s["pass"] and "OVER" in s["line"]
