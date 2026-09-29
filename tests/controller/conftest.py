"""A scripted fake serial port that answers like Marlin 2.x."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable

import pytest


def _checksum(body: str) -> int:
    cs = 0
    for ch in body.encode("ascii"):
        cs ^= ch
    return cs


class FakeMarlin:
    """Checks line numbers and checksums, answers `Resend`/`ok` like Marlin, records commands.

    - `auto_ok=False` holds each line's replies until `release()`, like a full planner.
    - `corrupt={n}` makes the first copy of line n arrive damaged (checksum mismatch).
    - `ignore_first_m110` drops the first handshake line, like a board still booting.
    - `hooks[cmd] = fn(fake) -> list[str] | None` customises replies (None = send nothing).
    """

    def __init__(
        self,
        *,
        auto_ok: bool = True,
        corrupt: set[int] | None = None,
        boot_lines: tuple[str, ...] = ("start", "echo:Marlin 2.1.2.5"),
        ignore_first_m110: bool = False,
        silent: bool = False,
    ) -> None:
        self.out: queue.Queue[str] = queue.Queue()
        self.received: list[str] = []
        self.executed: list[str] = []
        self.last_n = 0
        self.corrupt = set(corrupt or ())
        self.auto_ok = auto_ok
        self.silent = silent
        self.ignore_m110 = 1 if ignore_first_m110 else 0
        self.held: deque[list[str]] = deque()
        self.unacked = 0
        self.max_unacked = 0
        self.emergency_stops = 0
        self.errors_sent = 0
        self.hooks: dict[str, Callable[[FakeMarlin], list[str] | None]] = {}
        self.closed = False
        self._lock = threading.Lock()
        for line in boot_lines:
            self.out.put(line)

    # serial-like interface
    def readline(self) -> bytes:
        try:
            return (self.out.get(timeout=0.02) + "\n").encode("ascii")
        except queue.Empty:
            return b""

    def write(self, data: bytes) -> int:
        for raw in data.decode("ascii").splitlines():
            self._receive(raw)
        return len(data)

    def close(self) -> None:
        self.closed = True

    # control
    def release(self, groups: int = 1_000_000) -> None:
        with self._lock:
            for _ in range(min(groups, len(self.held))):
                self._emit_now(self.held.popleft())

    def emit(self, *lines: str) -> None:
        for line in lines:
            self.out.put(line)

    @property
    def commands(self) -> list[str]:
        """Executed commands without the M110 handshakes."""
        return [c for c in self.executed if not c.startswith("M110")]

    # internals
    def _receive(self, raw: str) -> None:
        with self._lock:
            self.received.append(raw)
            if self.silent:
                return
            if "M410" in raw:
                self.emergency_stops += 1  # the emergency parser acts on arrival
            self.unacked += 1
            self.max_unacked = max(self.max_unacked, self.unacked)
            replies = self._process(raw)
            if replies is None:
                self.unacked -= 1
                return
            if self.auto_ok:
                self._emit_now(replies)
            else:
                self.held.append(replies)

    def _emit_now(self, lines: list[str]) -> None:
        for line in lines:
            self.out.put(line)
            if line.startswith("ok"):
                self.unacked -= 1

    def _resend(self, why: str) -> list[str]:
        self.errors_sent += 1
        return [f"Error:{why}, Last Line: {self.last_n}", f"Resend: {self.last_n + 1}", "ok"]

    def _process(self, raw: str) -> list[str] | None:
        cmd = raw
        if raw.startswith("N"):
            body, star, cs = raw.partition("*")
            n = int(body.split()[0][1:])
            if n in self.corrupt:
                self.corrupt.discard(n)
                return self._resend("checksum mismatch")
            if not star or int(cs) != _checksum(body):
                return self._resend("checksum mismatch")
            cmd = body.split(" ", 1)[1]
            if cmd.startswith("M110"):
                if self.ignore_m110:
                    self.ignore_m110 -= 1
                    return None
                self.last_n = int(cmd.split("N")[1])
            elif n != self.last_n + 1:
                return self._resend("Line Number is not Last Line Number+1")
            else:
                self.last_n = n
        self.executed.append(cmd)
        name = cmd.split()[0]
        if name in self.hooks:
            return self.hooks[name](self)
        if name == "M105":
            return ["ok T:25.00 /0.00 B:25.00 /0.00 @:0 B@:0"]
        if name == "M114":
            return ["X:0.00 Y:0.00 Z:3.00 E:0.00 Count X:0 Y:0 Z:1200", "ok"]
        if name == "M115":
            return ["FIRMWARE_NAME:Marlin 2.1.2.5 (fake)", "ok"]
        return ["ok"]


@pytest.fixture
def fake_marlin() -> type[FakeMarlin]:
    return FakeMarlin
