"""`scripts/live_run.py` pieces that run without a camera, plotter or microphone."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_startup_check_counts_native_crashes_only():
    check = load_script("startup_check")
    assert check.crashed(-11) and check.crashed(0xC0000005)
    assert not check.crashed(0) and not check.crashed(1)  # a Python error is a failure


def test_plotter_events_go_in_the_timeline_and_replay_skips_them(tmp_path):
    from pointdraw.live.replay import load
    from pointdraw.virtual.lockstep import LockstepPlotter

    path = tmp_path / "timeline.jsonl"
    w = TimelineWriter(path)
    w.write_new([{"t": 0.0, "kind": "state", "state": "LISTEN"}])
    with LockstepPlotter() as p:
        p.set_hands_clear(True)
        p.draw_polylines([[(60.0, 100.0), (180.0, 100.0)]], tag="cross#1")
        t, t_ink = 0.0, None
        while t < 25.0:
            t += 1 / 23
            p.tick(t)
            if t_ink is None and p.ink_polylines():
                t_ink = t  # drawing (after the lift to clear_z and the drop)
            if t_ink is not None and t_ink + 1.0 < t < t_ink + 1.1:
                p.set_hands_clear(False)  # a hand over the paper mid-line
            if t_ink is not None and t > t_ink + 2.0:
                p.set_hands_clear(True)
            w.write_plotter(p.events)
    w.close()
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    plot = [e for e in lines if e["kind"] == "plotter"]
    assert [e["event"] for e in plot] == [
        "job_start",
        "pause_request",
        "paused",
        "resumed",
        "job_end",
    ]
    for e in plot:
        assert set(e) >= {"t", "kind", "event", "label", "tag", "x", "y"}
        assert e["label"] == "draw" and e["tag"] == "cross#1"
    assert [e["t"] for e in plot] == sorted(e["t"] for e in plot)

    (tmp_path / "hands.jsonl").write_text('{"t0": 0.0}\n{"t": 0.1, "hands": []}\n')
    s = load(tmp_path)
    assert [e["kind"] for e in s.events] == ["state"]
    assert len(s.plotter) == len(plot)


def test_hands_log_is_written_off_the_loop(tmp_path):
    from pointdraw.live.logs import JsonlWriter

    w = JsonlWriter(tmp_path / "hands.jsonl")
    for i in range(100):
        w.write({"t": i / 30, "hands": []})
    w.close()
    rows = [json.loads(x) for x in (tmp_path / "hands.jsonl").read_text().splitlines()]
    assert [r["t"] for r in rows] == [i / 30 for i in range(100)]


def test_window_picture_is_window_sized_and_turned(tmp_path):
    import numpy as np

    from deixis.geometry import Box
    from pointdraw.live.view import BedView
    from pointdraw.vision.aruco import BedHomography
    from pointdraw.vision.hands import HandsFrame

    live_run = load_script("live_run")

    class _P:
        def ink_polylines(self):
            return [[(50.0, 50.0), (60.0, 60.0)]]

    h = np.array([[0.2, 0.0, 0.0], [0.0, -0.2, 220.0], [0.0, 0.0, 1.0]])
    view = BedView(BedHomography(h, (), 0.0, 0.0, float("nan")), 220.0, lambda: None)
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:10, :10] = 255  # camera top left: bottom right once turned
    vis = live_run.render_window(
        img, HandsFrame(0.0, [], (1920, 1080)), view, Box(30, 40, 150, 120), _P(), [], []
    )
    assert vis.shape == (720, 1280, 3)
    assert vis[-2, -2].min() > 200


def test_parked_photos_one_per_rest_with_the_paper_clear(tmp_path):
    """A photo once the plotter rests at park and no hand is near for 1 s; one per rest."""
    live_run = load_script("live_run")
    ph = live_run.ParkedPhotos(tmp_path / "photos", (0.0, 220.0), clear_s=1.0)
    img = np.zeros((4, 4, 3), np.uint8)
    park, away = (0.0, 220.0, 15.0), (80.0, 100.0, 4.5)
    assert ph.offer(img, 0.0, False, True, park) is None  # still moving
    assert ph.offer(img, 0.5, False, False, (None, None, None)) is None  # not known yet
    assert ph.offer(img, 1.0, False, False, park) is None  # parked, clear since 1.0
    assert ph.offer(img, 1.5, True, False, park) is None  # a hand comes in
    assert ph.offer(img, 2.0, False, False, park) is None  # clear again since 2.0
    assert ph.offer(img, 3.0, False, False, park) == "photos/3000.png"
    assert (tmp_path / "photos" / "3000.png").exists()
    assert ph.offer(img, 5.0, False, False, park) is None  # one per rest
    assert ph.offer(img, 6.0, False, False, away) is None  # previewing over the paper
    assert ph.offer(img, 9.0, False, True, park) is None  # drawing, then parking
    assert ph.offer(img, 10.0, False, False, park) is None
    assert ph.offer(img, 11.0, False, False, park) == "photos/11000.png"


@pytest.mark.parametrize("guess", ["box", "circle"])
def test_window_shows_the_guess_dashed_blue_and_holds_amber(guess):
    """Claim 39q: the robot's guess is dashed blue (a drawing's circle since 6 Oct, else the
    box of words or an arrow); your holds are filled amber."""
    import numpy as np

    from deixis.geometry import Box
    from pointdraw.controller.shapes import circle_points
    from pointdraw.live import look
    from pointdraw.live.view import BedView
    from pointdraw.vision.aruco import BedHomography
    from pointdraw.vision.hands import HandsFrame

    live_run = load_script("live_run")

    class _P:
        def ink_polylines(self):
            return []

    h = np.array([[0.2, 0.0, 0.0], [0.0, -0.2, 220.0], [0.0, 0.0, 1.0]])
    view = BedView(BedHomography(h, (), 0.0, 0.0, float("nan")), 220.0, lambda: None)
    img = np.full((1080, 1920, 3), 255, np.uint8)
    vis = live_run.render_window(
        img,
        HandsFrame(0.0, [], (1920, 1080)),
        view,
        Box(30, 40, 150, 120),
        _P(),
        [],
        [],
        ("Say yes to draw it, or change it", None),
        guesses=[Box(80, 80, 40, 40) if guess == "box" else circle_points((100, 100), 40)],
        holds=[(100.0, 100.0)],
        said="A kite, about four centimetres?",
    )
    colours = {tuple(c) for c in vis.reshape(-1, 3)[::7]}
    assert look.GUESS in colours and look.AMBER in colours
