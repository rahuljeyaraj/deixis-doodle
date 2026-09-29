"""Machine config, whitelist, drawable-area checks and job limits."""

from __future__ import annotations

import json
import math

import pytest

from deixis.geometry import Box
from pointdraw.controller.config import MachineConfig, load_config, save_config, with_changes
from pointdraw.controller.safety import (
    JobLimitExceeded,
    OutsideDrawableArea,
    UnsafeCommand,
    check_line,
    job_stats,
    link_whitelist,
    validate_points,
    validate_polylines,
)

CFG = MachineConfig()


# ---------------------------------------------------------------------------- config


def test_defaults_are_consistent():
    d = CFG.drawable
    assert d == CFG.paper_box.inset(CFG.margin_mm)
    assert CFG.travel_box.contains((d.x, d.y)) and CFG.travel_box.contains((d.x2, d.y2))
    assert CFG.park_y == CFG.y_max  # bed toward the user on a stock Ender 3
    assert CFG.pen_up_z > CFG.pen_down_z
    assert CFG.feed.slow < CFG.feed.normal < CFG.feed.fast
    assert CFG.max_segment_mm == 10


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "controller.json"
    changed = with_changes(CFG, {"pen_down_z": 0.4, "feed.normal": 2000, "paper.y": 20})
    save_config(changed, path)
    assert load_config(path) == changed


def test_partial_override_keeps_defaults(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(json.dumps({"park_x": 200, "feed": {"travel": 3500}}))
    cfg = load_config(path)
    assert cfg.park_x == 200 and cfg.feed.travel == 3500
    assert cfg.feed.normal == CFG.feed.normal and cfg.pen_up_z == CFG.pen_up_z


def test_missing_override_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.json") == CFG
    assert load_config(None) == CFG


@pytest.mark.parametrize(
    "changes",
    [
        {"pen_up_z": -1},
        {"pen_down_z": 5, "pen_up_z": 3},
        {"feed.normal": 99_999},
        {"feed.z": 1000},
        {"paper.x": 100},  # paper runs off the bed
        {"park_y": 500},
        {"margin_mm": 80},
        {"unknown_key": 1},
    ],
)
def test_invalid_configs_rejected(changes):
    with pytest.raises(ValueError):
        with_changes(CFG, changes)


# ---------------------------------------------------------------------------- whitelist


@pytest.mark.parametrize(
    "line",
    [
        "G0 X10 Y20 F3000",
        "G1 X10.5 Y20.25 F1800",
        "G0 Z3 F300",
        "G28",
        "G28 X Y",
        "G90",
        "G21",
        "M400",
        "M114",
        "M410",
        "M115",
        "M105",
    ],
)
def test_job_whitelist_accepts(line):
    assert check_line(line, config=CFG) == line


@pytest.mark.parametrize(
    "line",
    [
        "M104 S200",  # hot end
        "M109 S200",
        "M140 S60",  # bed
        "M190 S60",
        "M303 E0 S200 C8",  # autotune
        "M106 S255",  # fan
        "M107",
        "G1 X10 E5 F1800",  # extrusion
        "G10",
        "G92 X0",
        "M112",
        "M500",
        "M201 X500",  # startup only
        "M211 S1",
        "M110 N0",  # transport only
        "M280 P0 S90",  # servo disabled
        "g1 x10",  # strict: upper case
        "G1X10Y10",  # strict: spaces
        "N1 G1 X10*23",  # no line numbers from callers
        "G1 X10 ; comment",
        "G1 X10 X20",
        "G1 X",
        "G28 X10",
        "T0",
        "",
        "G1 X10\nM104 S200",
    ],
)
def test_job_whitelist_rejects(line):
    with pytest.raises(UnsafeCommand):
        check_line(line, config=CFG)


def test_startup_phase_allows_limits_but_not_disabling_endstops():
    for line in ("M201 X500 Y500 Z100", "M203 X60 Y60 Z5", "M204 P500 T500", "M211 S1"):
        check_line(line, phase="startup", config=CFG)
    for bad in ("M211 S0", "M211", "M203 X0", "M204 P-5", "M104 S200"):
        with pytest.raises(UnsafeCommand):
            check_line(bad, phase="startup", config=CFG)


def test_servo_only_when_enabled():
    servo = with_changes(CFG, {"servo_enabled": True})
    check_line("M280 P0 S90", config=servo)
    with pytest.raises(UnsafeCommand):
        check_line("M280 P0 S270", config=servo)


def test_moves_checked_against_travel_limits_and_feed_caps():
    for bad in (
        "G0 X-1 Y10 F3000",
        "G0 X10 Y221 F3000",
        "G0 Z25 F300",
        "G1 X10 F99999",
        "G0 Z3 F600",
    ):
        with pytest.raises(UnsafeCommand):
            check_line(bad, config=CFG)


def test_link_whitelist_is_phase_agnostic_but_still_blocks_heat():
    allowed = link_whitelist()
    check_line("M110 N0", allowed=allowed)
    check_line("M201 X500", allowed=allowed)
    with pytest.raises(UnsafeCommand):
        check_line("M104 S200", allowed=allowed)


# ---------------------------------------------------------------------------- area and limits


def test_inside_points_pass_and_empties_drop():
    d = CFG.drawable
    pls = [[(d.x, d.y), (d.x2, d.y2)], [], [d.center]]
    assert validate_points(pls, d) == [[(d.x, d.y), (d.x2, d.y2)], [d.center]]


def test_rounding_errors_clamped():
    d = CFG.drawable
    out = validate_points([[(d.x - 0.04, d.y2 + 0.049)]], d)
    assert out == [[(d.x, d.y2)]]


@pytest.mark.parametrize("dx", [0.06, 1.0, 500.0])
def test_points_outside_rejected(dx):
    d = CFG.drawable
    with pytest.raises(OutsideDrawableArea, match="outside the drawable area"):
        validate_points([[d.center, (d.x2 + dx, d.center[1])]], d)


@pytest.mark.parametrize("bad", [(math.nan, 50), (50, math.inf), (1, 2, 3), ("a", 1), None])
def test_malformed_points_rejected(bad):
    with pytest.raises(OutsideDrawableArea):
        validate_points([[bad]], CFG.drawable)


def test_job_limits():
    cx, cy = CFG.drawable.center
    zigzag = [[(cx - 90 + (i % 2) * 180, cy) for i in range(80)]]  # 79 x 180 mm = 14 km
    with pytest.raises(JobLimitExceeded, match="pen-down"):
        validate_polylines(zigzag, CFG)
    dots = [[(cx, cy)] for _ in range(CFG.max_pen_lifts + 1)]
    with pytest.raises(JobLimitExceeded, match="lifts"):
        validate_polylines(dots, CFG)
    many = [[(cx, cy)] * (CFG.max_points + 1)]
    with pytest.raises(JobLimitExceeded, match="points"):
        validate_polylines(many, CFG)


def test_job_stats():
    s = job_stats([[(0, 0), (3, 4)], [(0, 0)], []])
    assert s.pen_down_mm == pytest.approx(5)
    assert (s.pen_lifts, s.points) == (2, 3)


def test_box_type_used_for_areas():
    assert isinstance(CFG.drawable, Box)
