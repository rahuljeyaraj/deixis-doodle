"""Controller behaviour: interlock, pause/resume, stop, status, and the serial path end to end."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from deixis.geometry import Box
from pointdraw.controller import gcode
from pointdraw.controller.config import MachineConfig, with_changes
from pointdraw.controller.controller import Controller
from pointdraw.controller.safety import OutsideDrawableArea, validate_polylines
from pointdraw.controller.serial_link import SerialLink
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkError, LinkStopped

CFG = with_changes(MachineConfig(), {"hands_clear_hold_s": 0.05})
D = CFG.drawable
Y = D.center[1]
X0 = D.x + 5
LONG = [[(X0, Y), (X0 + 150, Y)]]  # 15 segments; 0.5 s at 10x real time


def wait_for(cond: Callable[[], bool], timeout: float = 3.0) -> None:
    t_end = time.perf_counter() + timeout
    while not cond():
        if time.perf_counter() > t_end:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


@pytest.fixture
def slow():
    """A controller on a real-time simulator running 10x faster than the machine."""
    sim = SimulatedPlotter(CFG, realtime=True, speed_factor=10)
    ctl = Controller(sim, CFG, hands_clear=True)
    ctl.start()
    yield ctl, sim
    ctl.close()


@pytest.fixture
def fast():
    sim = SimulatedPlotter(CFG)
    ctl = Controller(sim, CFG, hands_clear=True)
    ctl.start()
    yield ctl, sim
    ctl.close()


def test_start_sends_startup_block_first(fast):
    ctl, sim = fast
    assert sim.lines == gcode.startup_block(CFG)
    assert ctl.status().state == "idle"


def test_draw_matches_the_request(fast):
    ctl, sim = fast
    square = [[(60, 50), (110, 50), (110, 90), (60, 90), (60, 50)]]
    ctl.draw_polylines(square, "fast")
    assert ctl.wait(3)
    st = ctl.status()
    assert (st.state, st.progress) == ("idle", 1.0)
    assert len(sim.strokes) == 1
    assert sim.strokes[0][0] == (60, 50) and sim.strokes[0][-1] == (60, 50)
    assert not sim.pen_is_down
    assert any(ln.endswith(f"F{gcode.fmt(CFG.feed.fast)}") for ln in sim.lines)


def test_rejects_outside_area_without_moving(fast):
    ctl, sim = fast
    n = len(sim.lines)
    with pytest.raises(OutsideDrawableArea):
        ctl.draw_polylines([[D.center, (D.x2 + 1, D.y)]])
    with pytest.raises(OutsideDrawableArea):
        ctl.preview_box(Box(D.x - 5, D.y, 20, 20))
    assert ctl.wait(1)
    assert len(sim.lines) == n


def test_preview_and_park(fast):
    ctl, sim = fast
    ctl.preview_box(Box(60, 50, 40, 30))
    ctl.park()
    assert ctl.wait(3)
    assert sim.strokes == []
    assert sim.position == (CFG.park_x, CFG.park_y, CFG.pen_up_z)


def test_not_started_raises():
    ctl = Controller(SimulatedPlotter(CFG), CFG, hands_clear=True)
    with pytest.raises(RuntimeError):
        ctl.draw_polylines(LONG)


def test_never_starts_while_hands_not_clear():
    cfg = with_changes(CFG, {"hands_clear_hold_s": 0.3})
    sim = SimulatedPlotter(cfg)
    ctl = Controller(sim, cfg)  # hands_clear defaults to False
    ctl.start()
    try:
        ctl.draw_polylines(LONG)
        time.sleep(0.2)
        assert sim.lines == gcode.startup_block(cfg)
        st = ctl.status()
        assert (st.state, st.progress, st.hands_clear) == ("paused", 0.0, False)
        ctl.set_hands_clear(True)
        time.sleep(0.1)  # clear, but not yet for hands_clear_hold_s
        assert sim.lines == gcode.startup_block(cfg)
        assert ctl.wait(3)
        assert len(sim.strokes) == 1
    finally:
        ctl.close()


def test_pause_lifts_pen_and_resume_finishes_the_line(slow):
    ctl, sim = slow
    ctl.draw_polylines(LONG)
    wait_for(lambda: sim.position[0] > X0 + 30 and sim.pen_is_down)
    ctl.set_hands_clear(False)
    wait_for(lambda: ctl.status().state == "paused" and not sim.pen_is_down, 1.0)
    x_paused = sim.position[0]
    n = len(sim.lines)
    time.sleep(0.25)
    assert len(sim.lines) == n  # nothing is sent while paused
    assert X0 + 30 < x_paused < X0 + 150
    assert 0 < ctl.status().progress < 1
    ctl.set_hands_clear(True)
    assert ctl.wait(5)
    assert ctl.status().state == "idle"
    assert len(sim.strokes) == 2
    first, second = sim.strokes
    assert first[0] == (X0, Y) and second[-1] == (X0 + 150, Y)
    assert second[0] == pytest.approx(first[-1])  # resumes exactly where it lifted


def test_stop_is_immediate_and_lifts_the_pen(slow):
    ctl, sim = slow
    ctl.draw_polylines(LONG)
    ctl.draw_polylines([[(X0, Y + 20), (X0 + 50, Y + 20)]])  # queued: must be dropped
    wait_for(lambda: sim.position[0] > X0 + 30 and sim.pen_is_down)
    t0 = time.perf_counter()
    ctl.stop()
    assert sim.quickstop_count == 1
    assert time.perf_counter() - t0 < 0.05
    assert ctl.wait(3)
    st = ctl.status()
    assert st.state == "stopped"
    assert not sim.pen_is_down
    assert sim.position[0] < X0 + 150
    assert len(sim.strokes) == 1  # the queued job never ran
    assert sim.lines[-1] == gcode.pen_up(CFG)
    # A new command after a stop works, and homes first (M410 may lose steps).
    n = len(sim.lines)
    ctl.draw_polylines([[(X0, Y + 20), (X0 + 20, Y + 20)]])
    assert ctl.wait(3)
    assert ctl.status().state == "idle" and len(sim.strokes) == 2
    assert sim.lines[n] == "G28"
    ctl.park()
    assert ctl.wait(3)
    assert sim.lines.count("G28") == 1  # only once


def test_no_rehome_when_disabled():
    cfg = with_changes(CFG, {"rehome_after_stop": False})
    sim = SimulatedPlotter(cfg)
    ctl = Controller(sim, cfg, hands_clear=True)
    ctl.start()
    try:
        ctl.stop()
        ctl.park()
        assert ctl.wait(3)
        assert "G28" not in sim.lines
    finally:
        ctl.close()


def test_stop_while_paused(slow):
    ctl, sim = slow
    ctl.draw_polylines(LONG)
    wait_for(lambda: sim.position[0] > X0 + 20)
    ctl.set_hands_clear(False)
    wait_for(lambda: ctl.status().state == "paused", 1.0)
    ctl.stop()
    ctl.set_hands_clear(True)
    assert ctl.wait(3)
    assert ctl.status().state == "stopped"
    assert len(sim.strokes) == 1 and not sim.pen_is_down


def test_progress_rises(slow):
    ctl, _ = slow
    ctl.draw_polylines(LONG)
    seen = []
    t_end = time.perf_counter() + 3
    while time.perf_counter() < t_end:
        st = ctl.status()
        seen.append(st.progress)
        if st.state == "idle" and st.progress == 1.0:
            break
        time.sleep(0.01)
    assert seen == sorted(seen)
    assert any(0 < p < 1 for p in seen) and seen[-1] == 1.0


# ---------------------------------------------------------------------------- serial path


def serial_controller(fake) -> Controller:
    link = SerialLink("FAKE", serial_factory=lambda: fake, boot_timeout=3, ack_timeout=3)
    ctl = Controller(link, CFG, hands_clear=True)
    ctl.start()
    return ctl


def test_serial_end_to_end(fake_marlin):
    fake = fake_marlin(corrupt={9})
    ctl = serial_controller(fake)
    try:
        square = [[(60, 50), (110, 50), (110, 90), (60, 90), (60, 50)]]
        ctl.draw_polylines(square)
        assert ctl.wait(5)
        expected = gcode.startup_block(CFG) + gcode.insert_syncs(
            gcode.polylines_to_gcode(validate_polylines(square, CFG), CFG, CFG.feed.normal),
            CFG.sync_every_mm,
            CFG.pen_threshold_z,
        )
        assert fake.commands == expected
        assert fake.max_unacked <= 2
        assert ctl.query("M114")[0].startswith("X:")
    finally:
        ctl.close()


def test_serial_stop_sends_m410_then_renumbers_and_lifts(fake_marlin):
    fake = fake_marlin()
    ctl = serial_controller(fake)
    try:
        fake.auto_ok = False  # a full planner: nothing is acknowledged
        ctl.draw_polylines(LONG)
        wait_for(lambda: len(fake.held) == 2)
        ctl.stop()
        assert fake.received[-1] == "M410"
        fake.auto_ok = True
        fake.release()
        assert ctl.wait(5)
        assert ctl.status().state == "stopped"
        i = fake.received.index("M410")
        after = fake.received[i + 1 :]
        assert after[0] == "N0 M110 N0*125"
        assert fake.commands[-1] == gcode.pen_up(CFG)
        assert not any(" G1 X" in r for r in after)  # no drawing after the stop
    finally:
        fake.auto_ok = True
        fake.release()
        ctl.close()


def test_firmware_halt_puts_controller_in_error(fake_marlin):
    fake = fake_marlin()
    ctl = serial_controller(fake)
    try:
        fake.hooks["G1"] = lambda f: ["Error:Printer halted. kill() called!", "ok"]
        ctl.draw_polylines(LONG)
        assert ctl.wait(5)
        st = ctl.status()
        assert st.state == "error" and "halted" in st.message
        with pytest.raises(LinkError):
            ctl.draw_polylines(LONG)
    finally:
        ctl.close()


class _GatedSim(SimulatedPlotter):
    """A simulator whose stop lands while the worker moves from one job to the next.

    Once `armed`, `wait_idle` holds the worker at the end of a job until the stop arrives,
    and `quickstop` returns only after the worker has met the stop in the following job.
    """

    def __init__(self, cfg: MachineConfig) -> None:
        super().__init__(cfg)
        self.armed = False
        self.at_end = threading.Event()
        self.release = threading.Event()
        self.cut = threading.Event()

    def wait_idle(self, timeout: float | None = None) -> None:
        if self.armed and not self.release.is_set():
            self.at_end.set()
            self.release.wait(3)
        super().wait_idle(timeout)

    def send(self, line: str, timeout: float | None = None) -> None:
        try:
            super().send(line, timeout)
        except LinkStopped:
            self.cut.set()
            raise

    def quickstop(self) -> None:
        super().quickstop()
        self.release.set()
        self.cut.wait(3)


def test_halted_names_the_job_the_stop_cut():
    """A stop that lands as one job ends and the next begins reports the next one as halted.

    Review 2 Oct: `stop()` read the running job before sending M410, so the halt named a job
    that had already finished, and the job really cut was never reported.
    """
    sim = _GatedSim(CFG)
    events: list = []
    ctl = Controller(sim, CFG, hands_clear=True, on_event=events.append)
    ctl.start()
    try:
        sim.armed = True
        ctl.draw_polylines([[(X0, Y), (X0 + 20, Y)]], tag="first")
        ctl.park(tag="second")
        assert sim.at_end.wait(3)
        ctl.stop()
        assert ctl.wait(3)
        got = [(e.kind, e.tag) for e in events]
        assert ("job_end", "first") in got
        assert ("halted", "second") in got
        assert ("halted", "first") not in got
        assert ("job_end", "second") not in got
    finally:
        ctl.close()
