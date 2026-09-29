"""Serial protocol: numbering, checksums, window, resends, busy, errors and quickstop."""

from __future__ import annotations

import threading
import time

import pytest

from pointdraw.controller.safety import UnsafeCommand
from pointdraw.controller.serial_link import SerialLink, checksum, numbered
from pointdraw.controller.transport import LinkError, LinkStopped, LinkTimeout


def make_link(fake, **kw) -> SerialLink:
    kw.setdefault("boot_timeout", 3.0)
    kw.setdefault("ack_timeout", 2.0)
    link = SerialLink("FAKE", serial_factory=lambda: fake, **kw)
    link.connect()
    return link


def moves(n: int) -> list[str]:
    return [f"G1 X{10 + i} Y20 F1800" for i in range(n)]


def test_checksum_matches_marlin_reference():
    # The handshake line hosts such as OctoPrint send; its checksum is well known.
    assert numbered(0, "M110 N0") == "N0 M110 N0*125"
    assert checksum("N1 G28") == (ord("N") ^ ord("1") ^ ord(" ") ^ ord("G") ^ ord("2") ^ ord("8"))


def test_connect_handshake_and_ordered_sends(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    try:
        assert fake.executed[0] == "M110 N0"
        for line in moves(8):
            link.send(line)
        link.wait_idle(timeout=2)
        assert fake.commands == moves(8)
        assert fake.received[1].startswith("N1 G1 X10 ")
        assert fake.max_unacked <= 2
        assert "start" in link.boot_lines
    finally:
        link.close()


def test_handshake_retries_while_board_boots(fake_marlin):
    fake = fake_marlin(ignore_first_m110=True)
    link = make_link(fake)
    try:
        assert [r for r in fake.received if "M110" in r].__len__() == 2
        link.send("G90")
        link.wait_idle(timeout=2)
        assert fake.commands == ["G90"]
    finally:
        link.close()


def test_boot_timeout_when_firmware_silent(fake_marlin):
    fake = fake_marlin(silent=True, boot_lines=())
    link = SerialLink("FAKE", serial_factory=lambda: fake, boot_timeout=0.5)
    t0 = time.perf_counter()
    with pytest.raises(LinkTimeout):
        link.connect()
    assert time.perf_counter() - t0 < 2.5
    assert fake.closed


def test_window_holds_at_most_two_unacknowledged(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    fake.auto_ok = False
    sent_before = len(fake.received)
    done = threading.Event()

    def sender():
        for line in moves(5):
            link.send(line)
        done.set()

    t = threading.Thread(target=sender)
    t.start()
    try:
        time.sleep(0.2)
        assert len(fake.received) - sent_before == 2
        assert not done.is_set()
        fake.release(1)
        time.sleep(0.15)
        assert len(fake.received) - sent_before == 3
        fake.auto_ok = True
        fake.release()
        assert done.wait(2)
        link.wait_idle(timeout=2)
        assert fake.commands == moves(5)
        assert fake.max_unacked <= 2
    finally:
        fake.auto_ok = True
        fake.release()
        t.join(2)
        link.close()


@pytest.mark.parametrize("corrupt", [{1}, {3}, {3, 4}, {5, 9}, {2, 3, 4, 5}])
def test_resend_recovers_every_line_in_order(fake_marlin, corrupt):
    fake = fake_marlin(corrupt=corrupt)
    link = make_link(fake)
    try:
        for line in moves(12):
            link.send(line)
        link.wait_idle(timeout=3)
        assert fake.commands == moves(12)
        assert link.resend_count >= 1
        assert fake.errors_sent >= len(corrupt)
    finally:
        link.close()


def test_ok_payload_and_replies(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    try:
        assert link.request("M105", timeout=2) == ["T:25.00 /0.00 B:25.00 /0.00 @:0 B@:0"]
        assert link.request("M114", timeout=2)[0].startswith("X:0.00 Y:0.00 Z:3.00")
        assert "Marlin" in link.request("M115", timeout=2)[0]
    finally:
        link.close()


def test_busy_messages_extend_the_ack_timeout(fake_marlin):
    fake = fake_marlin()

    def slow_m400(f):
        def later():
            for _ in range(8):
                time.sleep(0.1)
                f.emit("echo:busy: processing")
            f.emit("ok")

        threading.Thread(target=later, daemon=True).start()
        return []  # the ok comes later

    fake.hooks["M400"] = slow_m400
    link = make_link(fake, ack_timeout=0.4)
    try:
        link.send("M400")
        link.wait_idle(timeout=3)  # 0.8 s total, but never 0.4 s of silence
    finally:
        link.close()


def test_ack_timeout_raises(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake, ack_timeout=0.3)
    fake.auto_ok = False
    try:
        link.send("G90")
        with pytest.raises(LinkTimeout):
            link.wait_idle(timeout=3)
    finally:
        link.close()


def test_fatal_firmware_error_stops_the_link(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    try:
        fake.emit("Error:Printer halted. kill() called!")
        time.sleep(0.15)
        with pytest.raises(LinkError, match="halted"):
            link.send("G90")
    finally:
        link.close()


def test_board_restart_is_an_error(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    try:
        fake.emit("start")
        time.sleep(0.15)
        with pytest.raises(LinkError, match="restarted"):
            link.send("G90")
    finally:
        link.close()


def test_whitelist_enforced_before_writing(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    try:
        before = len(fake.received)
        for bad in ("M104 S200", "M140 S60", "G1 X10 E5", "M106 S255", "M211 S0"):
            with pytest.raises(UnsafeCommand):
                link.send(bad)
        assert len(fake.received) == before
    finally:
        link.close()


def test_quickstop_jumps_the_queue_then_resync(fake_marlin):
    fake = fake_marlin()
    link = make_link(fake)
    fake.auto_ok = False
    errors: list[Exception] = []

    def sender():
        try:
            for line in moves(6):
                link.send(line)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=sender)
    t.start()
    try:
        time.sleep(0.2)  # window full, sender blocked on line 3
        n_before = len(fake.received)
        link.quickstop()
        assert fake.received[-1] == "M410"  # written at once, no line number
        assert len(fake.received) == n_before + 1
        assert fake.emergency_stops == 1
        t.join(2)
        assert errors and isinstance(errors[0], LinkStopped)
        with pytest.raises(LinkStopped):
            link.send("G90")
        fake.auto_ok = True
        fake.release()  # the in-flight lines and the M410 get their oks
        link.resync(timeout=2)
        assert fake.received[-1] == numbered(0, "M110 N0")
        link.send("G0 Z3 F300")
        link.wait_idle(timeout=2)
        assert fake.received[-1].startswith("N1 G0 Z3")
        assert fake.commands[-1] == "G0 Z3 F300"
        assert not any(c.startswith("G1 X12") for c in fake.commands)  # line 3 never sent
    finally:
        fake.auto_ok = True
        fake.release()
        t.join(2)
        link.close()


def test_open_failure_is_a_link_error():
    def boom():
        raise OSError("no such port")

    link = SerialLink("COM99", serial_factory=boom)
    with pytest.raises(LinkError, match="could not open"):
        link.connect()
