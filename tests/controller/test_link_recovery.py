"""A pulled USB cable: reported, motion held, and the link opened again (night 2 Oct 2026)."""

from __future__ import annotations

import time

from pointdraw.controller.config import MachineConfig, with_changes
from pointdraw.controller.controller import Controller
from pointdraw.controller.serial_link import SerialLink
from pointdraw.runtime.live_plotter import LivePlotter

CFG = with_changes(MachineConfig(), {"hands_clear_hold_s": 0.0})
LINE = [[(60.0, 100.0), (100.0, 100.0)]]


def wait_for(cond, timeout: float = 5.0) -> None:
    t_end = time.perf_counter() + timeout
    while not cond():
        assert time.perf_counter() < t_end, "condition not met in time"
        time.sleep(0.01)


def unpluggable(base):
    class _Unpluggable(base):
        """A fake Marlin whose cable can be pulled: reads and writes then fail."""

        pulled = False

        def readline(self) -> bytes:
            if self.pulled:
                raise OSError("ClearCommError failed (the device is not connected)")
            return super().readline()

        def write(self, data: bytes) -> int:
            if self.pulled:
                raise OSError("WriteFile failed (the device is not connected)")
            return super().write(data)

    return _Unpluggable


def test_pulled_cable_is_reported_motion_held_and_reconnect_homes_first(fake_marlin):
    board = unpluggable(fake_marlin)
    boards = [board(), board()]
    link = SerialLink("FAKE", serial_factory=lambda: boards[0], boot_timeout=3, ack_timeout=3)
    p = LivePlotter(Controller(link, CFG, hands_clear=True))
    events = p.events
    p.start()
    try:
        p.draw_polylines(LINE, tag="a")
        assert p.wait(5) and p.status().state == "idle"
        boards[0].pulled = True
        # Idle: the reader notices; status() says so without a command being sent.
        wait_for(lambda: p.status().state == "error")
        assert "link" in p.status().message
        assert [e.kind for e in events][-1] == "error"
        # A command now is refused as an event, not an exception (the runtime keeps going).
        p.draw_polylines(LINE, tag="b")
        assert (events[-1].kind, events[-1].tag) == ("error", "b")
        # Plugged in again: reconnect, and the next job homes before it moves.
        boards.pop(0)
        p.reconnect()
        assert p.status().state == "idle" and events[-1].kind == "reconnected"
        assert p.position() == (None, None, None)
        n = len(boards[0].commands)
        p.draw_polylines(LINE, tag="c")
        assert p.wait(5) and p.status().state == "idle"
        cmds = boards[0].commands[n:]
        assert cmds[0] == "G28"
        assert any(c.startswith("G1 X") for c in cmds)
    finally:
        p.close()
