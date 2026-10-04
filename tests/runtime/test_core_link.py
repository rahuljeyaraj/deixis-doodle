"""The plotter really asks the deixis core over MCP (core split C15): places, the "a bit
left" step and the "this big" answer all go to `deixis-mcp`, with no quiet fallback."""

from __future__ import annotations

from typing import Any

import pytest

from deixis.mcp import DeixisClient
from pointdraw import core
from pointdraw.virtual import scenefile, world
from pointdraw.virtual.world import RunConfig

SCENES = {s.id: s for s in scenefile.all_scenes()}


def _calls(tool: str, args: dict[str, Any]) -> list[str]:
    if tool != "resolve":
        return [tool]
    return [f"resolve:{c['name']}" for c in args["request"]["calls"]]


@pytest.mark.virtual
@pytest.mark.parametrize(
    "scene,expect",
    [
        ("S1", "resolve:place_drawing"),  # a cross here
        ("S17", "adjust"),  # bigger, a bit left, yes
        ("S28", "resolve:measure"),  # How big? ... this big
    ],
)
def test_the_plotter_asks_the_core_over_mcp(monkeypatch, scene, expect):
    monkeypatch.setenv("DEIXIS_MODE", "mcp")
    monkeypatch.setenv("DEIXIS_FALLBACK", "none")
    sent: list[str] = []
    real = DeixisClient._mcp

    def spy(self: DeixisClient, tool: str, args: dict[str, Any]) -> Any:
        sent.extend(_calls(tool, args))
        return real(self, tool, args)

    monkeypatch.setattr(DeixisClient, "_mcp", spy)
    res = world.run(SCENES[scene], RunConfig())
    fails = [f"{c['name']}: {c['detail']}" for c in res.report["checks"] if not c["ok"]]
    assert not fails, fails
    assert expect in sent, sent
    used = [c for c in core.clients() if c.mode == "mcp"]
    assert used and all(c.over_mcp for c in used)
