"""Pure functions that turn polylines (mm, bed frame) into Marlin G-code lines.

Lines are bare (no line numbers, checksums or comments); the transport adds those. Every
G0/G1 carries its own F, so inserting a line (a pause's pen lift, say) can never leave a
slow or fast feedrate behind for the next move. Every XY move is split into pieces of at most
`max_segment_mm`, so a pause or stop between lines takes effect quickly (Marlin's planner
holds a fixed number of moves, so short pieces mean little motion queued).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from deixis.geometry import Box, Point
from pointdraw.controller.config import MachineConfig
from pointdraw.controller.safety import parse_line

_SAME_POINT_MM = 0.01


def fmt(v: float) -> str:
    """Shortest fixed-point form with at most 3 decimals: 10.0 -> '10', -0.0 -> '0'."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def split_segment(p0: Point, p1: Point, max_len: float) -> list[Point]:
    """Points after p0 up to and including p1, each step at most `max_len` long."""
    d = math.dist(p0, p1)
    n = max(1, math.ceil(d / max_len - 1e-9))
    return [
        (p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n) for i in range(1, n + 1)
    ]


def pen_up(c: MachineConfig) -> str:
    return f"G0 Z{fmt(c.pen_up_z)} F{fmt(c.feed.z)}"


def pen_down(c: MachineConfig, z: float | None = None) -> str:
    return f"G1 Z{fmt(c.pen_down_z if z is None else z)} F{fmt(c.feed.z)}"


def startup_block(c: MachineConfig) -> list[str]:
    """Units, absolute mode, soft endstops on, conservative motion limits, motors held."""
    lim = c.limits
    return [
        "G21",
        "G90",
        "M211 S1",
        f"M201 X{fmt(lim.accel_xy)} Y{fmt(lim.accel_xy)} Z{fmt(lim.accel_z)}",
        f"M203 X{fmt(lim.max_xy_mm_s)} Y{fmt(lim.max_xy_mm_s)} Z{fmt(lim.max_z_mm_s)}",
        f"M204 P{fmt(lim.accel_xy)} T{fmt(lim.accel_travel)}",
        # Marlin powers the motors down after a few idle minutes; the bed then slides if
        # bumped and every later move is off by that much. Keep them on.
        "M84 S0",
    ]


class _Emitter:
    """Tracks where the head is while emitting lines, so moves can be split."""

    def __init__(self, c: MachineConfig, start: Point | None = None) -> None:
        self.c = c
        self.lines: list[str] = []
        self.pos = start
        self.pen_is_down: bool | None = None

    def up(self) -> None:
        if self.pen_is_down is not False:
            self.lines.append(pen_up(self.c))
            self.pen_is_down = False

    def down(self) -> None:
        if self.pen_is_down is not True:
            self.lines.append(pen_down(self.c))
            self.pen_is_down = True

    def move(self, p: Point, feed: float, draw: bool) -> None:
        code = "G1" if draw else "G0"
        pts = [p] if self.pos is None else split_segment(self.pos, p, self.c.max_segment_mm)
        for x, y in pts:
            self.lines.append(f"{code} X{fmt(x)} Y{fmt(y)} F{fmt(feed)}")
        self.pos = p


def polylines_to_gcode(
    polylines: Iterable[Sequence[Point]],
    c: MachineConfig,
    feed: float,
    *,
    travel_feed: float | None = None,
    start: Point | None = None,
) -> list[str]:
    """Draw each polyline pen-down at `feed`, travelling between them pen-up.

    Starts and ends with the pen up. A polyline that starts where the previous one ended is
    joined without a lift. A one-point polyline makes a dot.
    """
    e = _Emitter(c, start)
    travel = c.feed.travel if travel_feed is None else travel_feed
    e.up()
    for pl in polylines:
        if not pl:
            continue
        first = (float(pl[0][0]), float(pl[0][1]))
        joined = e.pen_is_down and e.pos is not None and math.dist(e.pos, first) <= _SAME_POINT_MM
        if not joined:
            e.up()
            e.move(first, travel, draw=False)
            e.down()
        for p in pl[1:]:
            e.move((float(p[0]), float(p[1])), feed, draw=True)
    e.up()
    return e.lines


def preview_box_gcode(
    box: Box, c: MachineConfig, *, loops: int = 1, start: Point | None = None
) -> list[str]:
    """Trace the box pen-up at the preview feedrate: round the outline, ending back at its
    corner, or (`c.preview_path == "diagonal"`) from one corner to the opposite one."""
    e = _Emitter(c, start)
    e.up()
    corners = [(box.x, box.y), (box.x2, box.y), (box.x2, box.y2), (box.x, box.y2), (box.x, box.y)]
    if c.preview_path == "diagonal":
        corners = [(box.x, box.y), (box.x2, box.y2)]
    e.move(corners[0], c.feed.travel, draw=False)
    for _ in range(loops):
        for p in corners[1:]:
            e.move(p, c.feed.preview, draw=False)
    return e.lines


def travel_gcode(p: Point, c: MachineConfig, *, start: Point | None = None) -> list[str]:
    """Pen up, then move to p at the travel feedrate."""
    e = _Emitter(c, start)
    e.up()
    e.move(p, c.feed.travel, draw=False)
    return e.lines


def park_gcode(c: MachineConfig, *, start: Point | None = None) -> list[str]:
    """Pen up, bed toward the user, carriage to the side."""
    return travel_gcode((c.park_x, c.park_y), c, start=start)


def home_gcode(c: MachineConfig) -> list[str]:
    """Home all axes, then lift the pen clear of the paper."""
    return ["G28", pen_up(c)]


def insert_syncs(
    lines: Iterable[str], every_mm: float, pen_up_above: float | None = None
) -> list[str]:
    """Insert `M400` after every `every_mm` of XY motion (0 = never).

    Marlin acknowledges a move once it is in its planner, which holds many moves. Waiting for
    the planner to empty every so often bounds how much motion is queued, so a pause or a stop
    without the emergency parser still takes effect within about `every_mm` of travel.

    `M400` brings the head to a full stop, which leaves a blot if the pen is down and a
    visible stutter on a travel move. With `pen_up_above` (the Z above which the pen is clear
    of the paper), a due sync waits for the next pen lift and goes right after it, where the
    head is stopped for the Z move anyway. Elsewhere the planner's own depth (about 16
    `max_segment_mm` pieces) bounds the queued motion.
    """
    out: list[str] = []
    pos: Point | None = None
    z: float | None = None  # unknown counts as up
    acc = 0.0
    for line in lines:
        out.append(line)
        if every_mm <= 0:
            continue
        cmd, params = parse_line(line)
        if cmd == "G28":
            pos = None
            z = 0.0
        was_up = pen_up_above is None or z is None or z > pen_up_above
        if cmd in ("G0", "G1") and params.get("Z") is not None:
            z = params["Z"]
        if cmd not in ("G0", "G1"):
            continue
        x, y = params.get("X"), params.get("Y")
        if x is not None or y is not None:
            new = (
                x if x is not None else (pos or (0.0, 0.0))[0],
                y if y is not None else (pos or (0.0, 0.0))[1],
            )
            if pos is not None:
                acc += math.dist(pos, new)
            pos = new
        if pen_up_above is None:
            due = True
        else:  # only on the line that lifts the pen
            due = not was_up and z is not None and z > pen_up_above
        if acc >= every_mm and due:
            out.append("M400")
            acc = 0.0
    return out


def end_position(lines: Iterable[str], start: Point | None = None) -> Point | None:
    """Where the head ends up after `lines` (None if unknown, e.g. after homing without XY)."""
    pos = start
    for line in lines:
        cmd, params = parse_line(line)
        if cmd == "G28":
            pos = (0.0, 0.0)
        elif cmd in ("G0", "G1"):
            x, y = params.get("X"), params.get("Y")
            if x is not None or y is not None:
                base = pos or (0.0, 0.0)
                pos = (x if x is not None else base[0], y if y is not None else base[1])
    return pos
