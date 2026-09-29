"""Bring-up tool for the pen plotter: drive it from the command line, or simulate it.

Examples (from the repo root):

    uv run python scripts/plot.py ports
    uv run python scripts/plot.py --port COM5 info
    uv run python scripts/plot.py --port COM5 home
    uv run python scripts/plot.py --port COM5 square --size 50
    uv run python scripts/plot.py --sim out/square.png square --size 50
    uv run python scripts/plot.py --dry-run circle --diameter 40
    uv run python scripts/plot.py --port COM5 crosses --grid 3x3 --spacing 50
    uv run python scripts/plot.py --port COM5 pen-test --z 0.5
    uv run python scripts/plot.py --port COM5 feed-test --feeds 1000,2000,3000
    uv run python scripts/plot.py config --set pen_down_z=0.4 --set feed.normal=2000

There is no camera interlock here: you are the interlock. Keep hands off the bed while it
moves. Ctrl+C stops at once (M410) and lifts the pen.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deixis.geometry import Point, Polyline
from pointdraw.controller import gcode
from pointdraw.controller.config import (
    DEFAULT_CONFIG_PATH,
    MachineConfig,
    load_config,
    save_config,
    with_changes,
)
from pointdraw.controller.controller import Controller
from pointdraw.controller.safety import SafetyError, validate_points
from pointdraw.controller.serial_link import SerialLink, list_ports
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkError, Transport

DEFAULT_CROSSES_JSON = Path(".cache") / "crosses.json"
LIKELY_BOARD_IDS = ("0483:5740", "1A86:7523", "0403:6001", "2341:")  # STM32 CDC, CH340, FTDI

# ---------------------------------------------------------------------------- shapes


def square(center: Point, size: float) -> list[Polyline]:
    cx, cy, h = center[0], center[1], size / 2
    return [
        [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]
    ]


def circle(center: Point, diameter: float, max_chord_mm: float = 1.0) -> list[Polyline]:
    r = diameter / 2
    n = max(24, math.ceil(math.pi * diameter / max_chord_mm))
    pts = [
        (
            center[0] + r * math.cos(2 * math.pi * i / n),
            center[1] + r * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return [pts + [pts[0]]]


def cross(center: Point, size: float) -> list[Polyline]:
    cx, cy, h = center[0], center[1], size / 2
    return [[(cx - h, cy), (cx + h, cy)], [(cx, cy - h), (cx, cy + h)]]


def parse_grid(text: str) -> tuple[int, int]:
    """'3x3' -> (columns, rows)."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"grid must look like 3x3, not {text!r}") from e
    if cols < 1 or rows < 1:
        raise argparse.ArgumentTypeError("grid needs at least 1x1")
    return cols, rows


def cross_grid(center: Point, cols: int, rows: int, spacing: float) -> list[dict[str, Any]]:
    """Cross centres, row by row from the front (low y), left to right."""
    x0 = center[0] - spacing * (cols - 1) / 2
    y0 = center[1] - spacing * (rows - 1) / 2
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(
                {"id": len(out), "col": c, "row": r, "x": x0 + c * spacing, "y": y0 + r * spacing}
            )
    return out


def square_and_circle(center: Point, size: float) -> list[Polyline]:
    return square(center, size) + circle(center, size * 0.8)


# ---------------------------------------------------------------------------- plumbing


def make_transport(args: argparse.Namespace, cfg: MachineConfig) -> Transport:
    if args.dry_run or args.sim:
        return SimulatedPlotter(cfg, echo=print if args.dry_run else None)
    port = args.port
    if not port:
        from pointdraw.config import get_settings

        port = get_settings().serial_port
    if not port:
        raise SystemExit(
            "no port: pass --port COMx (see `plot.py ports`), --sim out.png or --dry-run"
        )
    return SerialLink(
        port,
        cfg.baud,
        window=cfg.window,
        boot_timeout=cfg.boot_timeout_s,
        ack_timeout=cfg.ack_timeout_s,
        servo=cfg.servo_enabled,
    )


def centre(args: argparse.Namespace, cfg: MachineConfig) -> Point:
    return tuple(args.center) if args.center else cfg.drawable.center  # type: ignore[return-value]


def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_ports(_: argparse.Namespace) -> int:
    ports = list_ports()
    if not ports:
        print("no serial ports found (is the board powered and the USB cable a data cable?)")
    for dev, desc, hwid in ports:
        likely = any(i in hwid.upper() for i in LIKELY_BOARD_IDS)
        print(f"{dev:10} {desc}  [{hwid}]" + ("  <- likely the printer board" if likely else ""))
    return 0


def cmd_config(args: argparse.Namespace, cfg: MachineConfig) -> int:
    if args.set:
        changes = {}
        for item in args.set:
            key, sep, value = item.partition("=")
            if not sep:
                raise SystemExit(f"--set needs key=value, not {item!r}")
            changes[key.strip()] = parse_value(value.strip())
        try:
            cfg = with_changes(cfg, changes)
        except ValueError as e:
            raise SystemExit(f"not saved: {e}") from e
        path = save_config(cfg, args.config)
        print(f"saved {path}")
    print(json.dumps(cfg.model_dump(), indent=2))
    d = cfg.drawable
    print(f"drawable area: x {d.x:g}..{d.x2:g}, y {d.y:g}..{d.y2:g} mm")
    return 0


def run_motion(args: argparse.Namespace, cfg: MachineConfig) -> int:
    transport = make_transport(args, cfg)
    ctl = Controller(transport, cfg, hands_clear=True)
    simulated = isinstance(transport, SimulatedPlotter)
    if not simulated:
        print("No camera interlock: keep hands off the bed. Ctrl+C stops.", file=sys.stderr)
    try:
        ctl.start()
    except LinkError as e:
        transport.close()
        print(f"could not connect: {e}", file=sys.stderr)
        return 1
    try:
        rc = _run_command(args, cfg, ctl, simulated)
        st = ctl.status()
        if st.state == "error":
            print(f"error: {st.message}", file=sys.stderr)
            rc = 1
    finally:
        ctl.close()
    if args.sim and simulated:
        print(f"rendered {transport.render_png(args.sim)}")
        if transport.violations:
            print(f"warning: {len(transport.violations)} soft-endstop clamps", file=sys.stderr)
    return rc


def _run_command(
    args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, simulated: bool
) -> int:
    try:
        if args.home:
            ctl.home()
        rc = COMMANDS[args.command](args, cfg, ctl, simulated)
        ctl.wait()
        return rc
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
        ctl.stop()
        ctl.wait(10)
        return 130
    except SafetyError as e:
        print(f"refused: {e}", file=sys.stderr)
        ctl.wait()  # anything queued before the refusal (e.g. --home) still finishes
        return 2
    except LinkError as e:
        print(f"link error: {e}", file=sys.stderr)
        return 1


def do_info(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    for line, what in (("M115", "firmware"), ("M114", "position"), ("M105", "temperatures")):
        reply = ctl.query(line)
        print(f"{what:13} {' | '.join(reply) or '(no reply text)'}")
        if line == "M115":
            caps = " ".join(reply)
            if "EMERGENCY_PARSER:1" in caps:
                print("emergency parser: yes, stop (M410) acts the moment it arrives")
            elif "Cap:" in caps:
                print("emergency parser: NO. Stops wait behind queued lines; use the E-stop")
    print("temperatures should read about 25 C with the 100 kOhm resistors fitted (Path A)")
    return 0


def do_home(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    if not args.home:
        ctl.home()
    return 0


def do_park(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    ctl.park()
    return 0


def do_square(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    ctl.draw_polylines(square(centre(args, cfg), args.size), args.speed)
    return 0


def do_circle(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    ctl.draw_polylines(circle(centre(args, cfg), args.diameter), args.speed)
    return 0


def do_crosses(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    cols, rows = args.grid
    points = cross_grid(centre(args, cfg), cols, rows, args.spacing)
    polylines = [pl for p in points for pl in cross((p["x"], p["y"]), args.size)]
    ctl.draw_polylines(polylines, args.speed)  # validates first: nothing is written if refused
    record = {
        "frame": "bed",
        "units": "mm",
        "grid": [cols, rows],
        "spacing_mm": args.spacing,
        "size_mm": args.size,
        "simulated": sim,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "paper": cfg.paper.model_dump(),
        "crosses": points,
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(points)} cross positions to {path}")
    return 0


def do_pen_test(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    at = centre(args, cfg)
    validate_points([[at]], cfg.drawable, cfg.clamp_tolerance_mm)
    z = cfg.pen_down_z if args.z is None else args.z
    ctl.run_lines(gcode.travel_gcode(at, cfg) + [gcode.pen_down(cfg, z)], "pen down")
    ctl.wait()
    if not (sim or args.no_wait):
        input(f"pen at Z={z:g} at ({at[0]:g}, {at[1]:g}). Check it, then press Enter to lift ")
    ctl.run_lines([gcode.pen_up(cfg)], "pen up")
    print(f"to keep this height: plot.py config --set pen_down_z={z:g}")
    return 0


def do_feed_test(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller, sim: bool) -> int:
    feeds = [float(f) for f in args.feeds.split(",") if f.strip()]
    gap = 6.0
    total = len(feeds) * args.size + (len(feeds) - 1) * gap
    cx, cy = centre(args, cfg)
    x = cx - total / 2 + args.size / 2
    for f in feeds:
        ctl.draw_polylines_at(square_and_circle((x, cy), args.size), f)
        print(f"F{f:g} (mm/min) at x={x:.0f} mm")
        x += args.size + gap
    return 0


COMMANDS = {
    "info": do_info,
    "home": do_home,
    "park": do_park,
    "square": do_square,
    "circle": do_circle,
    "crosses": do_crosses,
    "pen-test": do_pen_test,
    "feed-test": do_feed_test,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", help="serial port, e.g. COM5 (default: SERIAL_PORT from .env)")
    p.add_argument("--sim", metavar="OUT.PNG", help="use the simulator and render a PNG")
    p.add_argument("--dry-run", action="store_true", help="print the G-code; move nothing")
    p.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal")
    p.add_argument("--home", action="store_true", help="home (G28) before the command")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config override JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="log serial traffic")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ports", help="list serial ports")
    c = sub.add_parser("config", help="show the machine config, or change and save it")
    c.add_argument("--set", action="append", metavar="KEY=VALUE", help="e.g. feed.normal=2000")
    sub.add_parser("info", help="firmware (M115), position (M114), temperatures (M105)")
    sub.add_parser("home", help="home all axes, then lift the pen")
    sub.add_parser("park", help="pen up, bed toward you, carriage to the side")

    def with_center(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"), help="mm, bed frame")
        return sp

    with_center(sub.add_parser("square", help="draw a square")).add_argument(
        "--size", type=float, default=50
    )
    with_center(sub.add_parser("circle", help="draw a circle")).add_argument(
        "--diameter", type=float, default=40
    )
    cr = with_center(sub.add_parser("crosses", help="draw a grid of crosses for calibration"))
    cr.add_argument("--grid", type=parse_grid, default=(3, 3), help="COLSxROWS, e.g. 3x3")
    cr.add_argument("--spacing", type=float, default=50)
    cr.add_argument("--size", type=float, default=6, help="cross arm length, mm")
    cr.add_argument("--json", default=str(DEFAULT_CROSSES_JSON), help="where to write positions")
    pt = with_center(sub.add_parser("pen-test", help="lower the pen, wait for Enter, lift"))
    pt.add_argument("--z", type=float, help="pen-down Z to try (default: config pen_down_z)")
    pt.add_argument("--no-wait", action="store_true", help="don't wait for Enter")
    ft = with_center(sub.add_parser("feed-test", help="the same shape at several feedrates"))
    ft.add_argument("--feeds", default="1000,1500,2000,2500,3000", help="mm/min, comma-separated")
    ft.add_argument("--size", type=float, default=25)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.command == "ports":
        return cmd_ports(args)
    try:
        cfg = load_config(args.config)
    except ValueError as e:
        raise SystemExit(f"bad config in {args.config}: {e}") from e
    if args.command == "config":
        return cmd_config(args, cfg)
    return run_motion(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
