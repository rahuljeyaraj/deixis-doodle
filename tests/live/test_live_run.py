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


def test_the_window_keeps_the_camera_view_s_shape(monkeypatch):
    """Maximised, the picture is scaled to fit and centred on black bars, never stretched.

    OpenCV's WINDOW_KEEPRATIO stretched it to the window on Windows, so the bed came out
    wide and squat (rahul, 7 Oct 2026).
    """
    live = load_script("live_run")
    vis = np.full((720, 1280, 3), 200, np.uint8)

    # a 16:9 picture in a tall 4:3 window: full width, bars top and bottom
    monkeypatch.setattr(live.cv2, "getWindowImageRect", lambda name: (0, 0, 1600, 1200))
    out = live.fit_window("w", vis)
    assert out.shape == (1200, 1600, 3)
    assert out[600, 800].tolist() == [200, 200, 200]  # the picture, centred
    assert out[5, 800].tolist() == [0, 0, 0] and out[1195, 800].tolist() == [0, 0, 0]
    rows = [y for y in range(1200) if out[y, 800].any()]
    assert round((max(rows) - min(rows) + 1) / 1600, 2) == round(720 / 1280, 2)  # same shape

    # a very wide window: full height, bars left and right
    monkeypatch.setattr(live.cv2, "getWindowImageRect", lambda name: (0, 0, 2000, 500))
    out = live.fit_window("w", vis)
    assert out.shape == (500, 2000, 3)
    assert out[250, 1000].any() and not out[250, 5].any() and not out[250, 1995].any()

    # a window the same shape as the picture: no bars anywhere
    monkeypatch.setattr(live.cv2, "getWindowImageRect", lambda name: (0, 0, 640, 360))
    out = live.fit_window("w", vis)
    assert out.shape == (360, 640, 3) and out.min() == 200


def test_the_window_fit_copes_with_no_window(monkeypatch):
    """Before the window exists (and while a thumbnail is taken) the rect is empty."""
    live = load_script("live_run")
    vis = np.full((720, 1280, 3), 200, np.uint8)
    monkeypatch.setattr(live.cv2, "getWindowImageRect", lambda name: (-1, -1, 0, 0))
    assert live.fit_window("w", vis) is vis

    def boom(name):
        raise live.cv2.error("no window named w")

    monkeypatch.setattr(live.cv2, "getWindowImageRect", boom)
    assert live.fit_window("w", vis) is vis


class _View:
    """A BedView stand-in: the paper sits at a known rectangle of pixels."""

    def __init__(self, rect):
        self.rect = rect

    def to_px(self, corners, offset_y=None):
        import numpy as _np

        x0, y0, x1, y1 = self.rect
        return _np.array([(x1, y0), (x0, y0), (x0, y1), (x1, y1)], float)


class _Paper:
    x, y, x2, y2 = 21.0, 26.0, 241.5, 222.0


def test_the_window_shows_the_paper_not_the_table():
    """The camera sees far more table than paper, so the view is cropped to it.

    On the real bed the paper is 680 px of 1920 (rahul, 7 Oct 2026: "so much waste space").
    """
    live = load_script("live_run")
    view = _View((597, 244, 1277, 796))  # the measured paper on this bed
    x0, y0, x1, y1 = live.view_crop(view, _Paper(), (1920, 1080), 0.18)
    assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
    assert x1 - x0 < 1920 and y1 - y0 < 1080  # something was trimmed
    assert x0 < 597 and x1 > 1277 and y0 < 244 and y1 > 796  # the whole paper is still in
    assert abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.02  # the window's shape, so no bars
    assert (x1 - x0) / 1920 < 0.8  # a real saving


def test_a_tight_margin_still_holds_the_whole_paper():
    live = load_script("live_run")
    view = _View((597, 244, 1277, 796))
    x0, y0, x1, y1 = live.view_crop(view, _Paper(), (1920, 1080), 0.0)
    assert x0 <= 597 and x1 >= 1277 and y0 <= 244 and y1 >= 796


def test_a_crop_never_leaves_the_frame():
    """A paper near an edge (a knocked camera) must not ask for pixels that do not exist."""
    live = load_script("live_run")
    for rect in ((10, 10, 300, 250), (1700, 900, 1910, 1070), (0, 0, 1920, 1080)):
        x0, y0, x1, y1 = live.view_crop(_View(rect), _Paper(), (1920, 1080), 0.3)
        assert 0 <= x0 < x1 <= 1920, rect
        assert 0 <= y0 < y1 <= 1080, rect


def test_the_banner_does_not_cover_the_paper_north_edge():
    """`top` keeps a band free along the window's top, where the banner sits.

    Live on 8 Oct 2026 the banner hid the paper's north edge. The 180 degree turn puts the
    window's top at the camera frame's high-y side, so the band is taken from there and the
    whole paper must sit clear of it.
    """
    live = load_script("live_run")
    view = _View((597, 244, 1277, 796))  # the measured paper on this bed
    band = 0.12
    x0, y0, x1, y1 = live.view_crop(view, _Paper(), (1920, 1080), 0.18, top=band)
    assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
    assert x0 < 597 and x1 > 1277 and y0 < 244 and y1 > 796  # the whole paper is still in
    free_from = y0 + (1 - band) * (y1 - y0)
    assert 796 <= free_from  # the paper's high-y edge clears the banner's band
    assert abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.02  # still the window's shape


def test_the_top_band_works_beside_the_right_band():
    live = load_script("live_run")
    view = _View((597, 244, 1277, 796))
    x0, y0, x1, y1 = live.view_crop(view, _Paper(), (1920, 1080), 0.18, right=0.3, top=0.12)
    assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
    assert x0 < 597 and x1 > 1277 and y0 < 244 and y1 > 796
    assert 796 <= y0 + 0.88 * (y1 - y0)  # clear of the banner
    assert 597 >= x0 + 0.3 * (x1 - x0)  # clear of the inset's band (low-x side)


def test_the_guess_stays_until_the_drawing_is_back_and_settled():
    """The circle used to go the instant "yes" was said, so the drawing could not be seen
    landing inside it (rahul, 8 Oct 2026). It now waits for the plotter to park and settle.
    """
    live = load_script("live_run")
    linger = live.GuessLinger(linger_s=4.0)
    circle = [(10.0, 10.0), (20.0, 10.0), (10.0, 10.0)]
    live_marks = {"guesses": [circle], "centres": [(15.0, 10.0)], "holds": [], "said": None}
    gone = {"guesses": [], "centres": [], "holds": [], "said": None}

    assert linger.apply(live_marks, 0.0, busy=False)["guesses"] == [circle]  # the proposal
    # "yes": the proposal is gone, the plotter has not started yet
    assert linger.apply(gone, 1.0, busy=False)["guesses"] == [circle]
    assert linger.apply(gone, 2.0, busy=True)["guesses"] == [circle]  # drawing
    assert linger.apply(gone, 20.0, busy=True)["guesses"] == [circle]  # still drawing
    out = linger.apply(gone, 21.0, busy=False)  # parked: the wait starts now
    assert out["guesses"] == [circle] and out["centres"] == [(15.0, 10.0)]
    assert linger.apply(gone, 24.0, busy=False)["guesses"] == [circle]  # 3 s: still there
    assert linger.apply(gone, 25.1, busy=False)["guesses"] == []  # past 4 s: gone


def test_a_guess_that_is_never_drawn_does_not_hang_about():
    live = load_script("live_run")
    linger = live.GuessLinger(linger_s=4.0, cap_s=40.0)
    circle = [(10.0, 10.0), (20.0, 10.0), (10.0, 10.0)]
    gone = {"guesses": [], "centres": [], "holds": [], "said": None}
    linger.apply({"guesses": [circle], "centres": [], "holds": [], "said": None}, 0.0, False)
    assert linger.apply(gone, 1.0, busy=False)["guesses"] == [circle]  # cancelled, never drawn
    assert linger.apply(gone, 39.0, busy=False)["guesses"] == [circle]
    assert linger.apply(gone, 42.0, busy=False)["guesses"] == []  # the backstop


def test_the_paper_marks_are_drawn_where_the_paper_parks():
    """The bed slides in Y to draw. The circle must stay where the parked paper is, so the
    drawing can be seen coming back inside it (rahul, 8 Oct 2026).
    """
    live = load_script("live_run")
    assert live.PARKED == 0.0

    class Spy:
        """A BedView stand-in that records the offset each call asked for."""

        def __init__(self):
            self.offsets = []

        def to_px(self, pts_mm, offset_y=None):
            import numpy as _np

            self.offsets.append(offset_y)
            return _np.array(pts_mm, dtype=float).reshape(-1, 2)

    class NoInk:
        def ink_polylines(self):
            return []

    import numpy as _np

    vis = _np.zeros((400, 600, 3), _np.uint8)
    view = Spy()

    class NoHands:
        hands: list = []

    live.draw_overlay(
        vis,
        NoHands(),
        view,
        _Paper(),
        NoInk(),
        1.0,
        guesses=[live.Box(10, 10, 20, 20)],
        holds=[(5.0, 5.0)],
        centres=[(20.0, 20.0)],
        area=live.Box(0, 0, 100, 100),
    )
    assert view.offsets, "nothing was drawn"
    assert all(o == live.PARKED for o in view.offsets), view.offsets


def test_clearing_the_window_drops_a_guess_still_showing():
    """C gives a clean window for the next take, without restarting the app."""
    live = load_script("live_run")
    linger = live.GuessLinger(linger_s=4.0)
    circle = [(10.0, 10.0), (20.0, 10.0), (10.0, 10.0)]
    gone = {"guesses": [], "centres": [], "holds": [], "said": None}
    linger.apply(
        {"guesses": [circle], "centres": [(15.0, 10.0)], "holds": [], "said": None}, 0.0, False
    )
    assert linger.apply(gone, 1.0, busy=False)["guesses"] == [circle]  # lingering
    linger.clear()
    assert linger.apply(gone, 2.0, busy=False)["guesses"] == []
    # And it still works for the next guess.
    out = linger.apply({"guesses": [circle], "centres": [], "holds": [], "said": None}, 3.0, False)
    assert out["guesses"] == [circle]


def test_the_border_key_draws_the_frame_and_parks():
    """B redraws the drawing area's outline for a fresh sheet, so changing the paper no
    longer means restarting the app (rahul, 8 Oct 2026)."""
    live = load_script("live_run")
    from pointdraw.controller.config import MachineConfig
    from pointdraw.runtime.events import EventLog

    class FakePlotter:
        def __init__(self, state="idle", busy=False):
            self.state, self.busy = state, busy
            self.drawn: list = []
            self.parked: list = []

        def status(self):
            return type("S", (), {"state": self.state})()

        def draw_polylines(self, polys, tag=None):
            self.drawn.append((polys, tag))

        def park(self, tag=None):
            self.parked.append(tag)

    cfg, log = MachineConfig(), EventLog()
    plotter = FakePlotter()
    assert live.draw_border(plotter, cfg, log) is True
    (polys, tag) = plotter.drawn[0]
    assert tag == "frame" and plotter.parked == ["border"]
    assert polys[0][0] == polys[0][-1]  # a closed outline
    assert [e["why"] for e in log.of("border")] == ["b key"]

    # It must not cut into a drawing, or talk to a dead link.
    busy = FakePlotter(busy=True)
    assert live.draw_border(busy, cfg, log) is False and not busy.drawn
    dead = FakePlotter(state="error")
    assert live.draw_border(dead, cfg, log) is False and not dead.drawn
