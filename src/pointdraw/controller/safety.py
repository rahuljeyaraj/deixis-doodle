"""Safety rules the controller enforces whatever it is asked to do (plan section 4.5).

Three layers:

1. **Whitelist** (`check_line`): every line that reaches the printer is parsed and checked.
   Only known commands with known parameters pass. Heater, fan and extrusion commands can never
   be sent, and `M211` may only switch soft endstops *on*.
2. **Drawable area** (`validate_points`): every point of a job must lie inside the drawable
   area (the paper minus a margin). Only rounding errors under 0.05 mm are clamped; anything
   further out rejects the whole job.
3. **Job limits** (`validate_polylines`): maximum pen-down length, pen lifts and points.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from deixis.geometry import Box, Point, Polyline
from pointdraw.controller.config import MachineConfig

Phase = Literal["startup", "job"]


class SafetyError(ValueError):
    """Base class: the controller refused something."""


class UnsafeCommand(SafetyError):
    """A G-code line failed the whitelist."""


class OutsideDrawableArea(SafetyError):
    """A job has points outside the drawable area."""


class JobLimitExceeded(SafetyError):
    """A job is too long or has too many pen lifts or points."""


# Allowed in jobs, with the parameter letters each may carry.
JOB_COMMANDS: dict[str, frozenset[str]] = {
    "G0": frozenset("XYZF"),
    "G1": frozenset("XYZF"),
    "G28": frozenset("XYZ"),  # axis flags only
    "G90": frozenset(),
    "G21": frozenset(),
    "M400": frozenset(),
    "M114": frozenset(),
    "M410": frozenset(),
}
# Sent only in the startup block.
STARTUP_COMMANDS: dict[str, frozenset[str]] = {
    "M201": frozenset("XYZ"),
    "M203": frozenset("XYZ"),
    "M204": frozenset("PT"),
    "M211": frozenset("S"),
}
# Read-only diagnostics: firmware info and temperature report. Neither moves nor heats anything.
# M105 is how bring-up confirms the 100 kOhm resistor trick reads about 25 C.
DIAGNOSTIC_COMMANDS: dict[str, frozenset[str]] = {
    "M115": frozenset(),
    "M105": frozenset(),
}
# Line-number reset, used only by the serial transport itself.
PROTOCOL_COMMANDS: dict[str, frozenset[str]] = {"M110": frozenset("N")}
# Pen-lift servo, only when `servo_enabled`.
SERVO_COMMANDS: dict[str, frozenset[str]] = {"M280": frozenset("PS")}

# Never sent. Named here only to give a clearer error message.
FORBIDDEN_HINTS: dict[str, str] = {
    "M104": "hot-end heater",
    "M109": "hot-end heater",
    "M140": "bed heater",
    "M190": "bed heater",
    "M141": "chamber heater",
    "M303": "PID autotune heats",
    "M106": "fan",
    "M107": "fan",
    "G10": "retract",
    "G11": "retract",
    "M82": "extruder mode",
    "M83": "extruder mode",
    "G92": "position reset",
    "M112": "emergency kill needs a board reset; stop() uses M410",
    "M500": "EEPROM write",
    "M502": "factory reset",
}

FLAG_ONLY = {"G28"}
_TOKEN = re.compile(r"^([A-Z])(-?\d+(?:\.\d+)?|-?\.\d+)?$")
_COMMAND = re.compile(r"^[GM]\d+$")


def link_whitelist(servo: bool = False) -> dict[str, frozenset[str]]:
    """Everything a transport may put on the wire, whatever the phase."""
    allowed = {**JOB_COMMANDS, **STARTUP_COMMANDS, **DIAGNOSTIC_COMMANDS, **PROTOCOL_COMMANDS}
    if servo:
        allowed.update(SERVO_COMMANDS)
    return allowed


def parse_line(line: str) -> tuple[str, dict[str, float | None]]:
    """Split a bare G-code line into its command and parameters.

    Strict on purpose: upper case, one command, space-separated `<letter><number>` words, no
    line numbers, checksums or comments (the transport adds `N` and `*`).
    """
    if not isinstance(line, str) or "\n" in line or "\r" in line:
        raise UnsafeCommand(f"not a single line: {line!r}")
    words = line.strip().split()
    if not words:
        raise UnsafeCommand("empty line")
    cmd = words[0]
    if not _COMMAND.match(cmd):
        raise UnsafeCommand(f"not a G or M command: {line!r}")
    cmd = f"{cmd[0]}{int(cmd[1:])}"  # G01 -> G1
    params: dict[str, float | None] = {}
    for w in words[1:]:
        m = _TOKEN.match(w)
        if not m:
            raise UnsafeCommand(f"bad parameter {w!r} in {line!r}")
        letter, value = m.group(1), m.group(2)
        if letter in params:
            raise UnsafeCommand(f"repeated parameter {letter} in {line!r}")
        params[letter] = None if value is None else float(value)
    return cmd, params


def check_line(
    line: str,
    *,
    phase: Phase = "job",
    config: MachineConfig | None = None,
    allowed: dict[str, frozenset[str]] | None = None,
) -> str:
    """Return the line unchanged if it is safe to send, else raise UnsafeCommand.

    `allowed` overrides the command table (transports use `link_whitelist()`). With a
    `config`, G0/G1 targets must lie within the travel limits and feedrates within the caps.
    """
    cmd, params = parse_line(line)
    if allowed is None:
        allowed = {**JOB_COMMANDS, **DIAGNOSTIC_COMMANDS}
        if phase == "startup":
            allowed.update(STARTUP_COMMANDS)
        if config is not None and config.servo_enabled:
            allowed.update(SERVO_COMMANDS)
    if cmd not in allowed:
        hint = FORBIDDEN_HINTS.get(cmd)
        if cmd in STARTUP_COMMANDS:
            hint = "allowed in the startup block only"
        raise UnsafeCommand(f"{cmd} is not allowed" + (f" ({hint})" if hint else ""))
    bad = set(params) - allowed[cmd]
    if bad:
        raise UnsafeCommand(f"{cmd} may not carry {''.join(sorted(bad))}: {line!r}")
    for letter, value in params.items():
        if cmd in FLAG_ONLY:
            if value is not None:
                raise UnsafeCommand(f"{cmd} takes axis flags only: {line!r}")
        elif value is None or not math.isfinite(value):
            raise UnsafeCommand(f"{cmd} parameter {letter} needs a value: {line!r}")

    if cmd == "M211" and params.get("S") != 1:
        raise UnsafeCommand("M211 may only enable soft endstops (M211 S1)")
    if cmd in ("M201", "M203", "M204") and any(v is not None and v <= 0 for v in params.values()):
        raise UnsafeCommand(f"{cmd} values must be positive: {line!r}")
    if cmd == "M280" and not (0 <= (params.get("S") or 0) <= 180):
        raise UnsafeCommand("M280 angle must be 0..180")
    if cmd in ("G0", "G1") and config is not None:
        _check_move(params, config, line)
    return line


def _check_move(params: dict[str, float | None], c: MachineConfig, line: str) -> None:
    eps = 1e-6
    for axis, lo, hi in (("X", c.x_min, c.x_max), ("Y", c.y_min, c.y_max), ("Z", c.z_min, c.z_max)):
        v = params.get(axis)
        if v is not None and not lo - eps <= v <= hi + eps:
            raise UnsafeCommand(f"{axis}{v:g} is outside the travel limits: {line!r}")
    f = params.get("F")
    if f is not None:
        if f <= 0:
            raise UnsafeCommand(f"feedrate must be positive: {line!r}")
        z_only = params.get("X") is None and params.get("Y") is None
        cap = (c.limits.max_z_mm_s if z_only else c.limits.max_xy_mm_s) * 60
        if f > cap + eps:
            raise UnsafeCommand(f"feedrate F{f:g} exceeds the cap F{cap:g}: {line!r}")


# ---------------------------------------------------------------------------- job checks


@dataclass(frozen=True)
class JobStats:
    pen_down_mm: float
    pen_lifts: int
    points: int


def job_stats(polylines: Iterable[Sequence[Point]]) -> JobStats:
    down = 0.0
    lifts = 0
    points = 0
    for pl in polylines:
        if not pl:
            continue
        lifts += 1
        points += len(pl)
        down += sum(math.dist(a, b) for a, b in zip(pl, pl[1:], strict=False))
    return JobStats(down, lifts, points)


def _clamp_point(p: Point, area: Box, tol: float) -> Point | None:
    if area.contains(p):
        return p
    x, y = p
    cx = min(max(x, area.x), area.x2)
    cy = min(max(y, area.y), area.y2)
    if abs(cx - x) < tol and abs(cy - y) < tol:
        return (cx, cy)
    return None


def validate_points(
    polylines: Iterable[Sequence[Point]], area: Box, tol: float = 0.05
) -> list[Polyline]:
    """Clean polylines: finite 2D points, rounding errors under `tol` clamped, empties dropped.

    Raises OutsideDrawableArea (naming the first bad point and the count) if any point is
    further outside `area` than `tol`.
    """
    out: list[Polyline] = []
    bad: list[Point] = []
    for i, pl in enumerate(polylines):
        clean: Polyline = []
        for p in pl:
            try:
                if len(p) != 2:
                    raise ValueError
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError) as e:
                raise OutsideDrawableArea(f"polyline {i}: not a 2D point: {p!r}") from e
            if not (math.isfinite(x) and math.isfinite(y)):
                raise OutsideDrawableArea(f"polyline {i}: not a finite point: {p!r}")
            q = _clamp_point((x, y), area, tol)
            if q is None:
                bad.append((x, y))
            else:
                clean.append(q)
        if clean:
            out.append(clean)
    if bad:
        x, y = bad[0]
        raise OutsideDrawableArea(
            f"{len(bad)} point(s) outside the drawable area "
            f"x {area.x:g}..{area.x2:g}, y {area.y:g}..{area.y2:g} mm; first ({x:.2f}, {y:.2f})"
        )
    return out


def validate_polylines(
    polylines: Iterable[Sequence[Point]], config: MachineConfig
) -> list[Polyline]:
    """Area check plus job limits. Returns the cleaned polylines, or raises SafetyError."""
    clean = validate_points(polylines, config.drawable, config.clamp_tolerance_mm)
    stats = job_stats(clean)
    if stats.pen_down_mm > config.max_pen_down_mm:
        raise JobLimitExceeded(
            f"pen-down length {stats.pen_down_mm:.0f} mm exceeds {config.max_pen_down_mm:.0f} mm"
        )
    if stats.pen_lifts > config.max_pen_lifts:
        raise JobLimitExceeded(f"{stats.pen_lifts} pen lifts exceed {config.max_pen_lifts}")
    if stats.points > config.max_points:
        raise JobLimitExceeded(f"{stats.points} points exceed {config.max_points}")
    return clean
