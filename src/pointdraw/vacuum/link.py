"""The app's line to the robot vacuum: the `deixis-vacuum-mcp` server, over MCP (stdio).

The server lives in the separate GPL-3.0 repo `deixis-vacuum` (it uses deebot-client), on
Python 3.14, so this MIT app never imports it: it starts the server as a child process and
calls its tools. `clean_area` takes a `Where` box in the robot's map frame; the server
cleans exactly that box or says why not.

The command comes from `DEIXIS_VACUUM_MCP` (a full command line), or else the server in
`../deixis-vacuum/.venv`. `fake=True` adds `--fake`: a pretend robot, nothing moves.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from deixis.where import Where

ROOT = Path(__file__).resolve().parents[3]


class LinkError(RuntimeError):
    """The vacuum's server could not be started or did not answer."""


def default_command(fake: bool = False) -> list[str]:
    env = os.environ.get("DEIXIS_VACUUM_MCP")
    if env:
        cmd = shlex.split(env, posix=False)
    else:
        venv = ROOT.parent / "deixis-vacuum" / ".venv"
        exe = venv / (
            "Scripts/deixis-vacuum-mcp.exe" if sys.platform == "win32" else "bin/deixis-vacuum-mcp"
        )
        cmd = [str(exe)]
    return cmd + (["--fake"] if fake else [])


class VacuumLink:
    """Synchronous calls to the vacuum's MCP server, from any thread."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        fake: bool = False,
        timeout_s: float = 10.0,
        start_timeout_s: float = 90.0,  # the cloud login and MQTT take a few seconds
    ) -> None:
        self.command = list(command) if command is not None else default_command(fake)
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._session: Any = None
        self._closing: asyncio.Event | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="vacuum-link", daemon=True)
        self._thread.start()
        if not self._ready.wait(start_timeout_s):
            raise LinkError(f"the vacuum server did not start within {start_timeout_s} s")
        if self._error is not None or self._session is None:
            raise LinkError(f"the vacuum server did not start: {self._error!r}")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._closing = asyncio.Event()
        params = StdioServerParameters(command=self.command[0], args=self.command[1:])
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._closing.wait()
        except BaseException as e:  # noqa: BLE001 - reported to the caller
            self._error = e
        finally:
            self._session = None
            self._ready.set()

    @property
    def alive(self) -> bool:
        return self._session is not None and self._thread.is_alive()

    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.alive:
            raise LinkError(f"the vacuum server is not running ({self._error!r})")
        with self._lock:
            fut = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(tool, args or {}), self._loop
            )
            try:
                result = fut.result(self.timeout_s)
            except FutureTimeout:
                fut.cancel()
                raise LinkError(f"the vacuum server did not answer {tool}") from None
            except Exception as e:  # noqa: BLE001
                raise LinkError(f"{tool}: {e}") from e
        if getattr(result, "is_error", False):
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise LinkError(f"{tool}: {text or 'tool error'}")
        data = result.structured_content or {}
        return data["result"] if set(data) == {"result"} else data

    # ---------------------------------------------------------------- the robot's tools

    def clean_area(self, where: Where | dict[str, Any]) -> dict[str, Any]:
        w = where.model_dump(mode="json") if isinstance(where, Where) else where
        return self.call("clean_area", {"where": w})

    def stop(self) -> dict[str, Any]:
        return self.call("stop")

    def dock(self) -> dict[str, Any]:
        return self.call("dock")

    def status(self) -> dict[str, Any]:
        return self.call("status")

    def locate(self, wait_s: float = 2.0) -> dict[str, Any]:
        return self.call("locate", {"wait_s": wait_s})

    def close(self) -> None:
        if self._closing is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._closing.set)
        self._thread.join(10)

    def __enter__(self) -> VacuumLink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
