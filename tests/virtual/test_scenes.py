"""Every scene in tests/scenes, end to end in the virtual world (mock AI, text voice).

Run just these with `uv run pytest -m virtual`. They also run in the normal test run.
No hardware, no network, no keys. Each scene runs as scored by `pointdraw.virtual.checks`,
plus its variations (other seeds for the jitter, a hovering finger, speech shifted
earlier and later against the gesture), so a pass isn't luck. Each scene also runs once on
the real controller behind the live adapter (`plotter="controller"`, in simulated time).
"""

from __future__ import annotations

import pytest

from pointdraw.virtual import scenefile, world
from pointdraw.virtual.world import RunConfig

SPECS = scenefile.all_scenes()


def _cases():
    for spec in SPECS:
        if spec.voice == "tts":
            continue  # tts-only scenes need the Windows voice and the ASR model
        v = spec.variations
        cfgs = [RunConfig(), RunConfig(use_cache=False)]  # recorded replies, then parser only
        cfgs += [RunConfig(seed=s) for s in v.seeds]
        if v.hover_mm:
            cfgs.append(RunConfig(hover_mm=v.hover_mm))
        cfgs += [RunConfig(shift_s=s) for s in v.shifts]
        cfgs.append(RunConfig(plotter="controller"))
        for cfg in cfgs:
            marks = [pytest.mark.xfail(reason=spec.known_issue)] if spec.known_issue else []
            yield pytest.param(spec, cfg, id=f"{spec.id}-{cfg.variant}", marks=marks)


@pytest.mark.virtual
@pytest.mark.parametrize("spec,cfg", list(_cases()))
def test_scene(spec: scenefile.SceneSpec, cfg: RunConfig) -> None:
    res = world.run(spec, cfg)
    fails = [f"{c['name']}: {c['detail']}" for c in res.report["checks"] if not c["ok"]]
    assert not fails, f"{spec.id} {spec.title} [{cfg.variant}]\n  " + "\n  ".join(fails)


def test_every_scene_file_loads() -> None:
    ids = [s.id for s in SPECS]
    assert len(ids) == len(set(ids)), "duplicate scene ids"
    assert "S1" in ids
