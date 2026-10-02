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


def run_until_drawing(p: VirtualPlotter, mm: float = 20.0, t_max: float = 30.0) -> None:
    """Run until `mm` of ink is down: the job first lifts to clear_z and comes down, which
    takes a few seconds, so a fixed time isn't a fixed place in the line."""
    t = p.t
    while sum(math.dist(s.a, s.b) for s in p.ink) < mm:
        assert t < t_max, "never started drawing"
        t += 0.02
        p.tick(t)


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
    run_until_drawing(p)  # lift, travel from park, then drawing
    p.set_hands_clear(False)
    req = p.t
    run_until(p, req + 6.0)
    paused = next(e for e in p.events if e.kind == "paused")
    assert paused.t - req <= 1.7
    assert not p.sim.pen_is_down
    p.set_hands_clear(True)
    run_until(p, p.t + 40.0)
    assert any(e.kind == "resumed" for e in p.events)
    assert math.isclose(sum(math.dist(s.a, s.b) for s in p.ink), 150.0, abs_tol=0.01)


@pytest.mark.parametrize("ep", [True, False])
def test_stop_halts_and_next_job_homes(ep):
    p = VirtualPlotter(emergency_parser=ep)
    p.set_hands_clear(True)
    p.draw_polylines(LINE, tag="a")
    run_until_drawing(p)
    p.stop()
    run_until(p, p.t + 4.0)
    halt = next(e for e in p.events if e.kind == "halted")
    req = next(e for e in p.events if e.kind == "stop_request")
    assert halt.t - req.t <= (0.0 if ep else 1.7) + 1e-9
    assert p.state == "stopped" and not p.sim.pen_is_down
    p.park()
    run_until(p, p.t + 20.0)
    assert "G28" in p.sent[-5:] or any(line == "G28" for line in p.sent)


def test_stop_without_the_emergency_parser_worst_case():
    """Without EMERGENCY_PARSER, M410 waits behind an M400 after a pen lift until Marlin's
    planner drains (up to 16 moves). Swept over stop moments in S22 (night 2 Oct): up to
    2.2 s and 45.6 mm, so the 1.7 s CTRL estimate doesn't hold in the worst case; the Ender's
    firmware has the emergency parser. This pins the worst case so it can't grow unseen."""
    from pointdraw.virtual import scenefile, world
    from pointdraw.virtual.world import RunConfig

    base = scenefile.find("S22")
    worst_s = worst_mm = 0.0
    for k in range(0, 25, 2):
        spec = base.model_copy(deep=True)
        spec.lines[1].at = 17.0 + 0.25 * k
        m = world.run(spec, RunConfig(emergency_parser=False)).report["metrics"]
        worst_s = max(worst_s, m["stop_latency_s"])
        worst_mm = max(worst_mm, m["stop_travel_mm"])
    assert worst_s <= 2.3 and worst_mm <= 50.0  # the distance limit holds; the time does not


def test_outside_the_drawable_area_is_refused():
    p = VirtualPlotter()
    with pytest.raises(SafetyError):
        p.draw_polylines([[(0.0, 0.0), (300.0, 0.0)]])
