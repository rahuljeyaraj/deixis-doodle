"""'A detailed house' draws the colouring-book drawing; plain 'a house' the default (icon)."""

from __future__ import annotations

import pytest

from pointdraw.virtual import scenefile, world
from pointdraw.virtual.world import RunConfig


def _s4(text: str) -> scenefile.SceneSpec:
    spec = next(s for s in scenefile.all_scenes() if s.id == "S4").model_copy(deep=True)
    spec.lines[0].text = text
    spec.max_time = 240.0  # a big colouring-page house takes longer than S4's 90 s
    return spec


@pytest.mark.virtual
@pytest.mark.parametrize(
    "text,style",
    [
        ("A big house in the top left.", "icon"),
        ("A big simple house in the top left.", "icon"),
        ("A big detailed house in the top left.", "colouring"),
    ],
)
def test_the_spoken_style_picks_the_drawing(text, style):
    res = world.run(_s4(text), RunConfig(use_cache=False))
    houses = [o for o in res.runtime.scene.robot if o.label == "house"]
    assert len(houses) == 1 and f"_{style}_" in (houses[0].key_id or "")
    centre = [c for c in res.report["checks"] if c["name"].endswith(".center")]
    assert centre and all(c["ok"] for c in centre)
