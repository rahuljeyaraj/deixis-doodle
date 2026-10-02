"""The live adapter: the real controller's jobs, followed by tag, as the runtime needs them.

Most tests run the controller in simulated time (`LockstepPlotter`); one runs it on the real
clock with the plain simulator transport, as the live app will (but with no serial port).
"""

import math

from deixis.geometry import Box
from pointdraw.controller import Controller, MachineConfig, SimulatedPlotter
from pointdraw.runtime.live_plotter import LivePlotter
from pointdraw.virtual.lockstep import LockstepPlotter

LINE = [[(60.0, 100.0), (180.0, 100.0)]]  # 120 mm, inside the drawable area (bed frame)


def run_until(p: LockstepPlotter, t_end: float, dt: float = 1 / 23, on=None) -> None:
    t = p.t
    while t < t_end:
        t += dt
        p.tick(t)
        if on:
            on(t)


def kinds(p, tag=None):
    return [e.kind for e in p.events if tag is None or e.tag == tag]


def ink_mm(p, tag=None):
    return sum(math.dist(*pl) for pl in p.ink_polylines(tag))


def test_jobs_are_followed_by_tag():
    with LockstepPlotter() as p:
        p.set_hands_clear(True)
        p.preview_box(Box(80, 80, 40, 40), tag="preview_1")
        p.draw_polylines(LINE, tag="tree_1")
        p.park(tag="park_1")
        assert p.busy
        run_until(p, 60.0)
        assert not p.busy
        for tag in ("preview_1", "tree_1", "park_1"):
            assert kinds(p, tag) == ["job_start", "job_end"]
        starts = [e.tag for e in p.events if e.kind == "job_start"]
        assert starts == ["preview_1", "tree_1", "park_1"]
        assert ink_mm(p, "preview_1") == 0  # the pen stays up for a preview
        assert math.isclose(ink_mm(p, "tree_1"), 120.0, abs_tol=0.01)
        assert math.isclose(ink_mm(p), 120.0, abs_tol=0.01)
        end = next(e for e in p.events if e.kind == "job_end" and e.tag == "tree_1")
        assert end.label == "draw" and "x" in end.detail


def test_never_starts_without_hands_clear():
    with LockstepPlotter() as p:
        p.draw_polylines(LINE, tag="a")
        run_until(p, 3.0)
        assert not p.events and not p.ink and p.status().state == "paused"
        p.set_hands_clear(True)
        clear_at = p.t
        run_until(p, 30.0)
        start = next(e for e in p.events if e.kind == "job_start")
        assert start.t >= clear_at + p.config.hands_clear_hold_s - 1e-9
        assert math.isclose(ink_mm(p, "a"), 120.0, abs_tol=0.01)


def test_pause_lifts_the_pen_and_resumes():
    with LockstepPlotter() as p:
        p.set_hands_clear(True)
        p.draw_polylines(LINE, tag="a")
        drawing = {}

        def hand(t):
            if "in" not in drawing and ink_mm(p, "a") > 30:
                drawing["in"] = t
                p.set_hands_clear(False)
            elif "in" in drawing and "out" not in drawing and t - drawing["in"] > 3:
                drawing["out"] = t
                p.set_hands_clear(True)

        run_until(p, 60.0, on=hand)
        assert kinds(p, "a") == ["job_start", "pause_request", "paused", "resumed", "job_end"]
        req = next(e for e in p.events if e.kind == "pause_request")
        paused = next(e for e in p.events if e.kind == "paused")
        assert paused.t - req.t < 1.0  # one 2.5 mm piece, then the pen lift (3 mm at 5 mm/s)
        assert paused.detail["z"] >= p.config.pen_threshold_z  # pen up while paused
        laid = [s for s in p.ink if req.t < s.t <= paused.t]
        # The piece in progress, and the one already sent (the transport holds one line).
        assert sum(math.dist(s.a, s.b) for s in laid) <= 2 * p.config.max_segment_mm + 1e-6
        assert math.isclose(ink_mm(p, "a"), 120.0, abs_tol=0.01)


def test_stop_halts_drops_the_queue_and_rehomes():
    with LockstepPlotter() as p:
        p.set_hands_clear(True)
        p.draw_polylines(LINE, tag="a")
        p.park(tag="park")
        t = 0.0
        while ink_mm(p, "a") < 10:  # the job lifts to clear_z and comes down first
            t += 1 / 23
            assert t < 30
            p.tick(t)
        assert 0 < ink_mm(p, "a") < 120
        p.stop()
        run_until(p, t + 4.0)
        assert kinds(p)[-2:] == ["stop_request", "halted"]
        halt = p.events[-1]
        assert halt.tag == "a" and halt.label == "draw" and halt.detail["dropped"] == 1
        assert not p.busy and p.status().state == "stopped"
        assert "park" not in [e.tag for e in p.events]
        drawn = ink_mm(p, "a")
        p.draw_polylines([[(60.0, 150.0), (100.0, 150.0)]], tag="b")
        run_until(p, t + 40.0)
        assert "G28" in p.sent  # the next job homes first: M410 may lose steps
        assert kinds(p, "b") == ["job_start", "job_end"]
        assert math.isclose(ink_mm(p, "a"), drawn)  # nothing more of the stopped job
        assert math.isclose(ink_mm(p, "b"), 40.0, abs_tol=0.01)


def test_real_clock_with_the_simulator():
    cfg = MachineConfig(hands_clear_hold_s=0.0)
    p = LivePlotter(Controller(SimulatedPlotter(cfg), cfg))
    p.start()
    try:
        p.set_hands_clear(True)
        p.draw_polylines(LINE, tag="a")
        p.park(tag="park")
        assert p.wait(10.0)
        assert kinds(p) == ["job_start", "job_end", "job_start", "job_end"]
        assert [e.tag for e in p.events] == ["a", "a", "park", "park"]
        assert all(a.t <= b.t for a, b in zip(p.events, p.events[1:], strict=False))
        assert math.isclose(ink_mm(p, "a"), 120.0, abs_tol=0.01)
        assert not p.busy
    finally:
        p.close()
