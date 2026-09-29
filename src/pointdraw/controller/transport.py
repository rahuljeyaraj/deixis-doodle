"""The interface shared by the real serial link and the simulator."""

from __future__ import annotations

from typing import Protocol


class LinkError(RuntimeError):
    """The transport failed: no connection, a firmware error, or a timeout."""


class LinkTimeout(LinkError):
    """No acknowledgement arrived in time."""


class LinkStopped(LinkError):
    """`quickstop()` was called; nothing more is sent until `resync()`."""


class Transport(Protocol):
    """Sends bare G-code lines to a Marlin-like machine, one at a time.

    `send` returns once the line is on its way, blocking while the window of unacknowledged
    lines is full. Pausing is therefore just not calling `send`.
    """

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def send(self, line: str, timeout: float | None = None) -> None: ...

    def request(self, line: str, timeout: float | None = None) -> list[str]:
        """Send a line and return the reply lines printed before its `ok`."""
        ...

    def wait_idle(self, timeout: float | None = None) -> None:
        """Block until every sent line has been acknowledged."""
        ...

    def quickstop(self) -> None:
        """Halt motion now (M410). Safe to call from any thread. Later sends raise LinkStopped."""
        ...

    def resync(self, timeout: float | None = None) -> None:
        """After a quickstop: drain, renumber lines, and accept sends again."""
        ...
