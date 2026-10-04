"""The plotter's link to the deixis core: one `DeixisClient` per profile, shared.

`DEIXIS_MODE=mcp` (the default) starts one `deixis-mcp` child process per profile and sends
every resolve over MCP; `DEIXIS_MODE=direct` calls the core in-process. In MCP mode a failed
call falls back to in-process calls, logged loudly (filming-day safety, plan 4.9).

Runtimes with the same profile (the same drawable area) share one client, so the virtual
scenes start one server for the whole test session.
"""

from __future__ import annotations

import atexit
import os
import threading

from deixis.mcp import DeixisClient
from deixis.profile import Profile

_lock = threading.Lock()
_clients: dict[tuple[str, str], DeixisClient] = {}


def mode() -> str:
    return os.environ.get("DEIXIS_MODE", "mcp")


def core_client(profile: Profile) -> DeixisClient:
    """The shared client for this profile (started on first use)."""
    key = (mode(), profile.model_dump_json())
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = DeixisClient(profile, mode=key[0], fallback="direct")  # type: ignore[arg-type]
            _clients[key] = client
        return client


def close_all() -> None:
    with _lock:
        for c in _clients.values():
            c.close()
        _clients.clear()


atexit.register(close_all)
