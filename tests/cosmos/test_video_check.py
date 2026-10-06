"""scripts/cosmos_video_check.py clip: a real-time 4 fps clip from a run's saved pictures."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("cvc", ROOT / "scripts" / "cosmos_video_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cvc"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_clip_from_saved_pictures(tmp_path):
    run = tmp_path / "run"
    (run / "frames").mkdir(parents=True)
    (run / "hands.jsonl").write_text(json.dumps({"t0": 100.0}) + "\n")
    for k in range(10):  # twice a second, as live_run.py saves them
        img = np.full((360, 640, 3), 25 * k, np.uint8)
        cv2.imwrite(str(run / "frames" / f"{k * 500:07d}.jpg"), img)
    out = tmp_path / "clip.mp4"
    n = load().make_clip(run, 101.0, 103.0, out)
    assert n == 10  # pictures at 1.0, 1.5, ..., 3.0 s, each shown twice
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10 and round(cap.get(cv2.CAP_PROP_FPS)) == 4
    cap.release()
