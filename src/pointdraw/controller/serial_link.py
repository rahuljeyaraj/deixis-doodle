"""Serial transport for Marlin (115200 baud) with line numbers, checksums and a small window.

Protocol notes (Marlin 2.x):

- Each line goes out as `N<n> <command>*<checksum>`; the checksum is the XOR of every byte
  before `*`. `N0 M110 N0` resets the numbering.
- Marlin answers every received line with exactly one `ok` (M105 puts its temperatures on the
  `ok` line itself). A corrupted or out-of-order line gets `Error:...`, `Resend: <n>` and an
  `ok`; each later line already in flight gets the same trio, which we ignore once the first
  resend is scheduled. Blocking commands print `echo:busy: processing` every ~2 s, which resets
  the acknowledgement timeout.
- We keep at most `window` (2) lines unacknowledged. The reader runs on a background thread;
  `send` blocks the caller while the window is full.
- `ok` means "accepted into the planner", not "finished". The planner holds many moves, so
  the controller also inserts `M400` every `sync_every_mm` (see `gcode.insert_syncs`).

`quickstop()` writes a bare `M410` straight to the port, ahead of anything we have not sent.
With `EMERGENCY_PARSER` in the firmware, Marlin acts on it the moment the bytes arrive. Without
it, M410 waits in Marlin's command queue behind the (at most `window`) lines already sent, so
motion halts only once those are planned: within about `sync_every_mm` plus two segments of
travel. Either way the hardware E-stop on the 24 V line is the real emergency stop.

M410 empties the planner but not the lines still waiting to get into it (Marlin issue
#19774). The quickstop then holds Marlin for its 1 s "drop new moves" time, and those lines
are planned just after it and would run (seen on the Ender, 2 Oct 2026: the carriage halted
in 0.05 s, then moved on about 20 mm a second later). So `resync()` waits until every line
has its `ok`, which means planned but not started (Marlin holds a first move 100 ms), and
sends a second M410 that empties them. A stop with no line waiting does not need it.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pointdraw.clock import now
from pointdraw.controller.safety import check_line, link_whitelist
from pointdraw.controller.transport import LinkError, LinkStopped, LinkTimeout

log = logging.getLogger(__name__)

_RESEND = re.compile(r"^(?:Resend|rs)[: ]\s*N?(\d+)", re.IGNORECASE)
_RECOVERABLE = ("checksum", "line number", "no line number", "format error")
_FATAL = ("halted", "kill", "stopped", "mintemp", "maxtemp", "thermal", "heating failed")


def checksum(body: str) -> int:
    cs = 0
    for ch in body.encode("ascii"):
        cs ^= ch
    return cs & 0xFF


def numbered(n: int, cmd: str) -> str:
    body = f"N{n} {cmd}"
    return f"{body}*{checksum(body)}"


def list_ports() -> list[tuple[str, str, str]]:
    """(device, description, hardware id) for every serial port."""
    from serial.tools import list_ports as lp

    return [(p.device, p.description, p.hwid) for p in lp.comports()]


@dataclass
class _Entry:
    n: int | None  # None for the out-of-band M410
    lines: list[str] = field(default_factory=list)
    done: bool = False


class SerialLink:
    """A `Transport` over pyserial (or any object with readline/write/close)."""

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        *,
        window: int = 2,
        boot_timeout: float = 10.0,
        ack_timeout: float = 30.0,
        servo: bool = False,
        serial_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.window = window
        self.boot_timeout = boot_timeout
        self.ack_timeout = ack_timeout
        self._allowed = link_whitelist(servo)
        self._factory = serial_factory
        self._ser: Any = None
        self._cond = threading.Condition()
        self._reader: threading.Thread | None = None
        self._closed = False
        self._pending: deque[_Entry] = deque()
        self._history: dict[int, str] = {}
        self._next_n = 1
        self._resend_from: int | None = None
        self._last_resend: int | None = None
        self._ignore_resends = 0
        self._stopped = False
        self._waiting_at_stop = False  # numbered lines were unacknowledged when M410 went out
        self._error: LinkError | None = None
        self._last_rx = now()
        self._connected = False
        self.boot_lines: list[str] = []
        self.resend_count = 0

    # ------------------------------------------------------------------ lifecycle

    def connect(self) -> None:
        """Open the port and handshake. Also reopens after a failure (a pulled cable)."""
        with self._cond:
            self._error = None
            self._pending.clear()
            self._resend_from = None
            self._last_resend = None
            self._ignore_resends = 0
            self._stopped = False
            self._waiting_at_stop = False
            self._next_n = 1
            self._last_rx = now()
        try:
            if self._factory is not None:
                self._ser = self._factory()
            else:
                import serial

                self._ser = serial.Serial(self.port, self.baud, timeout=0.1, write_timeout=2)
        except Exception as e:  # noqa: BLE001 - pyserial raises several types
            raise LinkError(f"could not open {self.port}: {e}") from e
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, name="serial-reader", daemon=True)
        self._reader.start()
        try:
            self._wait_for_boot()
        except LinkError:
            self.close()
            raise
        self._connected = True
        log.info("connected to %s at %d baud", self.port, self.baud)

    def _wait_for_boot(self) -> None:
        """Let any boot chatter finish, then handshake with M110 until an `ok` arrives.

        8-bit boards reset when the port opens and print `start`; the SKR Mini E3 V3 (native
        USB) does not, so the handshake is what proves the firmware is listening.
        """
        deadline = now() + self.boot_timeout
        quiet_until = min(deadline, now() + 3.0)
        while now() < quiet_until and now() - self._last_rx < 0.3:
            with self._cond:
                self._cond.wait(0.05)
        while now() < deadline:
            with self._cond:
                self._pending.clear()
                self._resend_from = None
                entry = self._write(0, "M110 N0")
                self._next_n = 1
                t_end = min(deadline, now() + 1.5)
                while not entry.done and now() < t_end and self._error is None:
                    self._cond.wait(0.05)
                if self._error is not None:
                    raise self._error
                if entry.done:
                    return
            log.info("no answer from %s yet, retrying handshake", self.port)
        raise LinkTimeout(
            f"no answer from the firmware on {self.port} within {self.boot_timeout} s"
        )

    @property
    def failed(self) -> str | None:
        """Why the link failed (a read or write error, a firmware halt), or None."""
        e = self._error
        return None if e is None or isinstance(e, LinkStopped) else str(e)

    def close(self) -> None:
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                log.exception("closing the port failed")
        self._connected = False

    # ------------------------------------------------------------------ sending

    def send(self, line: str, timeout: float | None = None) -> None:
        self._submit(line, timeout)

    def request(self, line: str, timeout: float | None = None) -> list[str]:
        entry = self._submit(line, timeout)
        with self._cond:
            self._wait(lambda: entry.done, timeout)
            return list(entry.lines)

    def wait_idle(self, timeout: float | None = None) -> None:
        with self._cond:
            self._wait(lambda: not self._pending and self._resend_from is None, timeout)

    def _submit(self, line: str, timeout: float | None) -> _Entry:
        check_line(line, allowed=self._allowed)
        if not self._connected:
            raise LinkError("not connected")
        with self._cond:
            self._wait(
                lambda: len(self._pending) < self.window and self._resend_from is None,
                timeout,
                stoppable=True,
            )
            entry = self._write(self._next_n, line)
            self._next_n += 1
            return entry

    def _wait(
        self, ready: Callable[[], bool], timeout: float | None, stoppable: bool = False
    ) -> None:
        """Wait on the condition (lock held) until ready(), an error, a stop or a timeout."""
        deadline = None if timeout is None else now() + timeout
        while True:
            if self._error is not None:
                raise self._error
            if stoppable and self._stopped:
                raise LinkStopped("stopped; call resync() first")
            if ready():
                return
            if self._closed:
                raise LinkError("link closed")
            t = now()
            if deadline is not None and t > deadline:
                raise LinkTimeout("timed out waiting for the firmware")
            if self._pending and t - self._last_rx > self.ack_timeout:
                self._error = LinkTimeout(
                    f"no reply from the firmware for {self.ack_timeout:.0f} s"
                )
                raise self._error
            self._cond.wait(0.05)

    def _write(self, n: int, cmd: str) -> _Entry:
        """Write a numbered line (lock held) and track it."""
        text = numbered(n, cmd)
        self._history[n] = text
        self._history.pop(n - 256, None)
        entry = _Entry(n)
        if not self._pending:
            self._last_rx = now()  # the ack clock starts when the queue becomes non-empty
        self._pending.append(entry)
        self._raw_write(text)
        return entry

    def _raw_write(self, text: str) -> None:
        log.debug(">> %s", text)
        try:
            self._ser.write((text + "\n").encode("ascii"))
        except Exception as e:  # noqa: BLE001 - any port failure is a link failure
            self._error = LinkError(f"write failed: {e}")
            raise self._error from e

    # ------------------------------------------------------------------ stop

    def quickstop(self) -> None:
        with self._cond:
            self._stopped = True
            self._resend_from = None
            if any(e.n is not None for e in self._pending):
                self._waiting_at_stop = True
            try:
                self._raw_write("M410")
            except LinkError:
                log.exception("could not send M410")
                return
            self._pending.append(_Entry(None))  # Marlin also answers the bare M410 with an ok
            self._cond.notify_all()

    def resync(self, timeout: float | None = 5.0) -> None:
        with self._cond:
            self._drain(timeout)
            if self._waiting_at_stop:
                # Every line now has its ok: the ones that waited at the stop are planned
                # and about to run. Stop again before they move (see the module notes).
                self._waiting_at_stop = False
                log.info("second M410: lines that waited at the stop were planned after it")
                try:
                    self._raw_write("M410")
                except LinkError:
                    log.exception("could not send the second M410")
                else:
                    self._pending.append(_Entry(None))
                    self._drain(timeout)
            self._pending.clear()
            self._ignore_resends = 0
            self._last_resend = None
            entry = self._write(0, "M110 N0")
            self._next_n = 1
            self._wait(lambda: entry.done, timeout)
            self._stopped = False

    def _drain(self, timeout: float | None) -> None:
        """Wait (lock held) until every line has its ok; a silent firmware is not fatal here."""
        try:
            self._wait(lambda: not self._pending, timeout)
        except LinkTimeout:
            log.warning("pending lines not acknowledged after stop; renumbering anyway")
        if isinstance(self._error, LinkTimeout):
            self._error = None
        if self._error is not None:
            raise self._error

    # ------------------------------------------------------------------ reading

    def _read_loop(self) -> None:
        while not self._closed:
            try:
                raw = self._ser.readline()
            except Exception as e:  # noqa: BLE001 - any port failure is a link failure
                if not self._closed:
                    with self._cond:
                        self._error = LinkError(f"read failed: {e}")
                        self._cond.notify_all()
                return
            if not raw:
                continue
            text = raw.decode("ascii", errors="replace").strip()
            if not text:
                continue
            log.debug("<< %s", text)
            with self._cond:
                self._last_rx = now()
                self._handle(text)
                self._cond.notify_all()

    def _handle(self, text: str) -> None:
        low = text.lower()
        if low.startswith("ok"):
            payload = text[2:].strip()
            if self._pending:
                entry = self._pending.popleft()
                if payload:
                    entry.lines.append(payload)
                entry.done = True
            else:
                log.warning("unexpected ok: %s", text)
            self._pump_resends()
            return
        m = _RESEND.match(text)
        if m:
            n = int(m.group(1))
            if self._ignore_resends > 0 and n == self._last_resend:
                self._ignore_resends -= 1
                return
            self.resend_count += 1
            log.info("firmware asked to resend from line %d", n)
            self._resend_from = n
            self._last_resend = n
            self._ignore_resends = max(0, len(self._pending) - 1)
            return
        if low.startswith("error"):
            if any(k in low for k in _RECOVERABLE):
                log.info("firmware: %s", text)
            elif any(k in low for k in _FATAL):
                log.error("firmware: %s", text)
                self._error = LinkError(f"firmware error: {text}")
            else:
                log.warning("firmware: %s", text)
                if self._pending:
                    self._pending[0].lines.append(text)
            return
        if low.startswith("echo:busy"):
            return
        if low == "start":
            if self._connected:
                self._error = LinkError("the board restarted; position and settings are lost")
            self.boot_lines.append(text)
            return
        if not self._connected:
            self.boot_lines.append(text)
        if self._pending:
            self._pending[0].lines.append(text)
        else:
            log.info("firmware: %s", text)

    def _pump_resends(self) -> None:
        while self._resend_from is not None and len(self._pending) < self.window:
            n = self._resend_from
            if n >= self._next_n:
                self._resend_from = None
                break
            text = self._history.get(n)
            if text is None:
                self._error = LinkError(f"firmware asked for line {n}, which is no longer kept")
                self._resend_from = None
                return
            self._pending.append(_Entry(n))
            self._raw_write(text)
            self._resend_from = n + 1
        if self._resend_from is not None and self._resend_from >= self._next_n:
            self._resend_from = None
