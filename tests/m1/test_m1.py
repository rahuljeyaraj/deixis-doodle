"""pointdraw.m1: holds in, crosses out, against the simulated plotter (no camera)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pointdraw.controller.config import MachineConfig
from pointdraw.controller.controller import Controller
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.m1 import CrossUnderFinger, Target, load_targets, nearest_target

ROOT = Path(__file__).resolve().parents[2]
DT = 1 / 30


@pytest.fixture
def rig():
    cfg = MachineConfig(hands_clear_hold_s=0.0)
    sim = SimulatedPlotter(cfg)
    ctl = Controller(sim, cfg, hands_clear=False)
    ctl.start()
    yield ctl, sim
    ctl.close()


def point_and_withdraw(app: CrossUnderFinger, at: tuple[float, float], hold_s: float = 1.0):
    """Move in, hold still with a little jitter, then take the hand out of view."""
    rng = np.random.default_rng(1)
    t = 0.0
    start = np.array([at[0] - 40.0, at[1] - 20.0])
    for i in range(15):  # approach at ~90 mm/s
        p = start + (np.array(at) - start) * (i / 14)
        app.update(t, (float(p[0]), float(p[1])), True, True)
        t += DT
    for _ in range(int(hold_s / DT)):
        j = rng.normal(0, 0.2, 2)
        app.update(t, (at[0] + j[0], at[1] + j[1]), True, True)
        t += DT
    trials = []
    for _ in range(10):  # hand gone
        tr = app.update(t, None, False, False)
        if tr is not None:
            trials.append(tr)
        t += DT
    return trials


def cross_centre(strokes) -> tuple[float, float]:
    pts = np.array([p for s in strokes for p in s])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)


def test_hold_draws_a_cross_under_the_finger_then_parks(rig):
    ctl, sim = rig
    app = CrossUnderFinger(ctl, cross_mm=6.0)
    trials = point_and_withdraw(app, (100.0, 70.0))
    assert len(trials) == 1 and trials[0].drawn
    assert ctl.wait(10)
    cx, cy = cross_centre(sim.strokes)
    assert abs(cx - 100.0) < 0.5 and abs(cy - 70.0) < 0.5
    pts = np.array([p for s in sim.strokes for p in s])
    assert np.ptp(pts[:, 0]) == pytest.approx(6.0, abs=0.01)
    assert sim.position[:2] == pytest.approx((ctl.config.park_x, ctl.config.park_y))


def test_no_motion_while_a_hand_is_in_view(rig):
    ctl, sim = rig
    app = CrossUnderFinger(ctl)
    app.update(0.0, (100.0, 70.0), True, True)
    assert ctl.status().hands_clear is False
    app.update(DT, None, False, False)
    assert ctl.status().hands_clear is True


def test_passing_hand_does_not_draw(rig):
    ctl, sim = rig
    app = CrossUnderFinger(ctl)
    assert point_and_withdraw(app, (100.0, 70.0), hold_s=0.3) == []  # shorter than 0.6 s
    assert ctl.wait(5) and sim.strokes == []


def test_outside_the_drawable_area_is_refused(rig):
    ctl, sim = rig
    app = CrossUnderFinger(ctl)
    trials = point_and_withdraw(app, (4.0, 4.0))
    assert len(trials) == 1 and not trials[0].drawn and "refused" in trials[0].reason
    assert ctl.wait(5) and sim.strokes == []


def test_key_trigger_waits_for_fire(rig):
    ctl, sim = rig
    app = CrossUnderFinger(ctl, trigger="key")
    assert point_and_withdraw(app, (120.0, 90.0)) == []
    trial = app.fire()
    assert trial is not None and trial.drawn
    assert ctl.wait(10)
    cx, cy = cross_centre(sim.strokes)
    assert abs(cx - 120.0) < 0.5 and abs(cy - 90.0) < 0.5
    assert app.fire() is None  # nothing pending any more


def test_targets_scoring_and_log(rig, tmp_path):
    ctl, _ = rig
    paper = ctl.config.paper_box
    targets = load_targets(ROOT / "hardware" / "print" / "m1-targets.json", paper)
    raw = json.loads((ROOT / "hardware" / "print" / "m1-targets.json").read_text())
    first = raw["targets"][0]
    assert targets[0].x == pytest.approx(paper.x + first["x_mm"])
    assert targets[0].y == pytest.approx(paper.y + first["y_mm"])

    app = CrossUnderFinger(ctl, targets=targets)
    aim = (targets[0].x + 1.0, targets[0].y)  # 1 mm to the right of target 1
    (trial,) = point_and_withdraw(app, aim)
    assert trial.target_id == targets[0].id
    assert trial.error_mm == pytest.approx(1.0, abs=0.3)
    s = app.summary()
    assert s.scored == 1 and s.passed is True and "PASS" in s.text()
    log = app.write_log(tmp_path / "m1" / "trials.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["target_id"] == targets[0].id and rows[0]["drawn"] is True


def test_nearest_target():
    ts = [Target(1, 0.0, 0.0), Target(2, 10.0, 0.0)]
    t, d = nearest_target((7.0, 0.0), ts)
    assert t.id == 2 and d == pytest.approx(3.0)
    assert nearest_target((0.0, 0.0), []) is None
