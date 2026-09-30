"""The sim-time plotter: interlock, pause latency, stop with and without the emergency parser."""

import math

import pytest

from pointdraw.controller.safety import SafetyError
from pointdraw.virtual.plotter import VirtualPlotter

LINE = [[(40.0, 60.0), (190.0, 60.0)]]  # 150 mm, inside the drawable area (bed frame)


def run_until(p: VirtualPlotter, t_end: float, dt: float = 0.02, on=None):
    t = p.t
    while t < t_end:
        t += dt
        p.tick(t)
        if on:
            on(t)


def test_never_starts_without_hands_clear():
    p = VirtualPlotter()
    p.draw_polylines(LINE, tag="a")
    run_until(p, 3.0)
    assert not p.ink and p.state == "paused"
    p.set_hands_clear(True)
    run_until(p, 4.0)
    assert not any(e.kind == "job_start" and e.t < 3.5 for e in p.events)
    run_until(p, 20.0)
    assert math.isclose(sum(math.dist(s.a, s.b) for s in p.ink), 150.0, abs_tol=0.01)


def test_pause_drains_to_the_next_sync_then_resumes():
    p = VirtualPlotter()
    p.set_hands_clear(True)
    p.draw_polylines(LINE, tag="a")
    run_until(p, 6.0)  # travel from park, then drawing
    p.set_hands_clear(False)
    req = p.t
    run_until(p, 12.0)
    paused = next(e for e in p.events if e.kind == "paused")
    assert paused.t - req <= 1.7
    assert not p.sim.pen_is_down
    p.set_hands_clear(True)
    run_until(p, 40.0)
    assert any(e.kind == "resumed" for e in p.events)
    assert math.isclose(sum(math.dist(s.a, s.b) for s in p.ink), 150.0, abs_tol=0.01)


@pytest.mark.parametrize("ep", [True, False])
def test_stop_halts_and_next_job_homes(ep):
    p = VirtualPlotter(emergency_parser=ep)
    p.set_hands_clear(True)
    p.draw_polylines(LINE, tag="a")
    run_until(p, 6.0)
    p.stop()
    run_until(p, 10.0)
    halt = next(e for e in p.events if e.kind == "halted")
    req = next(e for e in p.events if e.kind == "stop_request")
    assert halt.t - req.t <= (0.0 if ep else 1.7) + 1e-9
    assert p.state == "stopped" and not p.sim.pen_is_down
    p.park()
    run_until(p, 20.0)
    assert "G28" in p.sent[-5:] or any(line == "G28" for line in p.sent)


def test_outside_the_drawable_area_is_refused():
    p = VirtualPlotter()
    with pytest.raises(SafetyError):
        p.draw_polylines([[(0.0, 0.0), (300.0, 0.0)]])
