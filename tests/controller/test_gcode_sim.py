"""G-code generation and the simulator."""

from __future__ import annotations

import math

import pytest

from deixis.geometry import Box
from pointdraw.controller import gcode
from pointdraw.controller.config import MachineConfig, with_changes
from pointdraw.controller.safety import check_line, parse_line
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkStopped

CFG = MachineConfig()


def run(lines: list[str], cfg: MachineConfig = CFG) -> SimulatedPlotter:
    sim = SimulatedPlotter(cfg)
    sim.connect()
    for line in lines:
        sim.send(line)
    return sim


def xy_moves(lines: list[str], start=(0.0, 0.0)):
    """(code, from, to) for every XY move."""
    pos = start
    out = []
    for line in lines:
        cmd, p = parse_line(line)
        if cmd in ("G0", "G1") and (p.get("X") is not None or p.get("Y") is not None):
            new = (p.get("X", pos[0]), p.get("Y", pos[1]))
            out.append((cmd, pos, new))
            pos = new
    return out


def test_fmt():
    assert gcode.fmt(10.0) == "10"
    assert gcode.fmt(-0.0001) == "0"
    assert gcode.fmt(1.23456) == "1.235"
    assert gcode.fmt(0.5) == "0.5"


@pytest.mark.parametrize("length", [0.0, 3.0, 10.0, 10.0001, 37.5, 100.0])
def test_split_segment(length):
    pts = gcode.split_segment((0, 0), (length, 0), 10)
    assert pts[-1] == (length, 0)
    prev = (0, 0)
    for p in pts:
        assert math.dist(prev, p) <= 10 + 1e-9
        prev = p


def test_startup_block():
    lines = gcode.startup_block(CFG)
    assert lines[:3] == ["G21", "G90", "M211 S1"]
    assert [ln.split()[0] for ln in lines[3:]] == ["M201", "M203", "M204", "M84"]
    assert lines[-1] == "M84 S0"  # motors stay on, so a bumped bed cannot slide
    for line in lines:
        check_line(line, phase="startup", config=CFG)


def test_polylines_to_gcode_structure():
    pls = [[(50, 50), (80, 50), (80, 90)], [(100, 60), (140, 60)]]
    lines = gcode.polylines_to_gcode(pls, CFG, CFG.feed.normal)
    for line in lines:
        check_line(line, config=CFG)
    assert lines[0] == gcode.pen_up(CFG)
    assert lines[-1] == gcode.pen_up(CFG)
    assert lines.count(gcode.pen_down(CFG)) == 2
    for _, a, b in xy_moves(lines, start=(50, 50)):
        assert math.dist(a, b) <= CFG.max_segment_mm + 1e-6
    for line in lines:
        cmd, p = parse_line(line)
        assert p["F"] is not None  # every move carries its own feedrate
        if p.get("X") is not None:
            assert p["F"] == (CFG.feed.normal if cmd == "G1" else CFG.feed.travel)


def test_joined_polylines_do_not_lift_and_dots_work():
    pls = [[(50, 50), (60, 50)], [(60, 50), (60, 60)], [(90, 90)]]
    lines = gcode.polylines_to_gcode(pls, CFG, 1800)
    assert lines.count(gcode.pen_down(CFG)) == 2  # the join shares one pen-down
    sim = run(lines)
    stroke = sim.strokes[0]  # split into max_segment_mm pieces
    assert stroke[0] == (50, 50) and (60, 50) in stroke and stroke[-1] == (60, 60)
    assert sim.strokes[1] == [(90, 90)]
    assert not sim.pen_is_down


def test_preview_is_pen_up_only():
    box = Box(60, 50, 40, 30)
    outline = with_changes(CFG, {"preview_path": "box"})
    lines = gcode.preview_box_gcode(box, outline)
    assert all(not ln.startswith("G1") for ln in lines)
    sim = run(lines)
    assert sim.strokes == []
    assert sim.position[:2] == (60, 50)  # round the box and back to its corner
    assert all(ln.endswith(f"F{gcode.fmt(CFG.feed.preview)}") for ln in lines[2:])


def test_diagonal_preview_goes_corner_to_corner_pen_up():
    box = Box(60, 50, 40, 30)
    lines = gcode.preview_box_gcode(box, CFG)  # the default since 2 Oct
    assert CFG.preview_path == "diagonal"
    assert all(not ln.startswith("G1") for ln in lines)
    sim = run(lines)
    assert sim.strokes == [] and sim.position[:2] == (100, 80)
    outline = gcode.preview_box_gcode(box, with_changes(CFG, {"preview_path": "box"}))
    assert len(lines) < len(outline) / 2  # 50 mm of path, not 140


def test_park_and_home():
    sim = run(gcode.home_gcode(CFG) + gcode.park_gcode(CFG))
    assert sim.homed
    assert sim.position == (CFG.park_x, CFG.park_y, CFG.pen_up_z)
    assert sim.strokes == []


def test_insert_syncs():
    pls = [[(20, 60), (200, 60)]]  # 180 mm
    lines = gcode.insert_syncs(gcode.polylines_to_gcode(pls, CFG, 1800), 30)
    n = lines.count("M400")
    assert 5 <= n <= 7
    assert gcode.insert_syncs(lines, 0) == lines


def test_insert_syncs_waits_for_pen_up():
    # A sync stops the head dead, which blots the paper if the pen is down.
    pls = [[(20, 60), (200, 60)], [(20, 80), (60, 80)]]
    lines = gcode.polylines_to_gcode(pls, CFG, 1800)
    synced = gcode.insert_syncs(lines, 30, CFG.pen_threshold_z)
    z = None
    for line in synced:
        cmd, p = parse_line(line)
        if "Z" in p:
            z = p["Z"]
        if line == "M400":
            assert z is None or z > CFG.pen_threshold_z
    assert synced.count("M400") >= 1
    assert [x for x in synced if x != "M400"] == lines
    # ...and only straight after a lift: a long travel move runs without stops
    for i, line in enumerate(synced):
        if line == "M400":
            assert synced[i - 1] == gcode.pen_up(CFG)


def test_end_position():
    lines = gcode.polylines_to_gcode([[(50, 50), (80, 70)]], CFG, 1800)
    assert gcode.end_position(lines) == (80, 70)


def test_simulator_replies_and_soft_endstop_record():
    sim = SimulatedPlotter(CFG)
    sim.connect()
    assert sim.request("M114")[0].startswith("X:0.00 Y:0.00 Z:3.00")
    assert "25.00" in sim.request("M105")[0]
    sim.config = CFG.model_copy(update={"x_max": 100.0})  # pretend a smaller machine
    sim.send("G0 X150 Y10 F3000")
    assert sim.violations and sim.position[0] == 100


def test_simulator_quickstop_interrupts_a_move():
    sim = SimulatedPlotter(CFG, realtime=True, speed_factor=10)
    sim.connect()
    sim.send("G1 Z0 F300")  # 0.06 s at 10x

    import threading

    t = threading.Timer(0.1, sim.quickstop)
    t.start()
    with pytest.raises(LinkStopped):
        sim.send("G1 X200 Y10 F600")  # 2 s at 10x; stopped partway
        sim.send("G1 X10 Y10 F600")
    t.join()
    assert 0 < sim.position[0] < 200
    sim.resync()
    sim.send("G0 Z3 F300")
    assert not sim.pen_is_down


def test_render_png(tmp_path):
    from PIL import Image

    sim = run(gcode.polylines_to_gcode([[(50, 50), (100, 100)]], CFG, 1800))
    out = sim.render_png(tmp_path / "sub" / "plot.png", px_per_mm=2)
    img = Image.open(out)
    assert img.size == (int(220 * 2) + 80, int(220 * 2) + 80)
    # The stroke passes through (75, 75) mm: a pixel there is dark.
    x, y = 40 + 75 * 2, 40 + (220 - 75) * 2
    near = [img.getpixel((x + dx, y + dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    assert min(sum(p[:3]) for p in near) < 200
