"""`scripts/live_run.py` pieces that run without a camera, plotter or microphone."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from pointdraw.live.logs import TimelineWriter
from pointdraw.vision.camera import CameraUnavailable

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Plotter:
    def __init__(self) -> None:
        self.clear: list[bool] = []

    def set_hands_clear(self, clear: bool) -> None:
        self.clear.append(clear)


class _Cam:
    def __init__(self, *results) -> None:
        self.results = list(results)

    def read(self, timeout=None):
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_no_frame_holds_the_plotter_and_a_dead_camera_ends_the_run():
    live_run = load_script("live_run")
    p = _Plotter()
    assert live_run.next_frame(_Cam(None), p, headless=True) is None
    assert p.clear == [False]
    assert live_run.next_frame(_Cam(CameraUnavailable("gone")), p, headless=True) == "quit"
    assert p.clear == [False, False]
    assert live_run.next_frame(_Cam("frame"), p, headless=True) == "frame"


def test_timeline_is_written_as_it_happens(tmp_path):
    path = tmp_path / "timeline.jsonl"
    events = [{"t": 1.0, "kind": "state", "state": "LISTEN"}]
    w = TimelineWriter(path)
    w.write_new(events)
    # Readable before the writer is closed: a crash keeps what was logged.
    assert [json.loads(x) for x in path.read_text().splitlines()] == events
    events.append({"t": 2.0, "kind": "say", "text": "Stopped."})
    w.write_new(events)
    w.write_new(events)  # nothing new: nothing written twice
    w.close()
    assert [json.loads(x)["kind"] for x in path.read_text().splitlines()] == ["state", "say"]
