"""Draw the spinner's shapes at one size and time each one (M06).

    uv run python scripts/spinner_times.py --sim out/spinner.png
    uv run python scripts/spinner_times.py --port COM9 --size 60

Every shape is drawn in the "detailed" (colouring) style, laid out in a grid on the sheet.
The table at the end gives each shape's real wall-clock time, which is what the film's
uncut take is budgeted from. Keep hands off the bed; Ctrl+C stops at once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointdraw.controller.config import load_config
from pointdraw.controller.controller import Controller
from pointdraw.controller.serial_link import SerialLink
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.drawing import Library, plan
from pointdraw.drawing.scene import place_object

SHAPES = ["balloon", "star", "heart", "cup", "banana", "flower", "kite", "apple"]


def layout(cfg, size: float, n: int, cols: int = 3):
    """Centres for n shapes of `size` mm, in a grid inside the drawable area."""
    d = cfg.drawable
    rows = (n + cols - 1) // cols
    stepx = (d.x2 - d.x - size) / max(cols - 1, 1)
    stepy = (d.y2 - d.y - size) / max(rows - 1, 1)
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append((d.x + size / 2 + c * stepx, d.y2 - size / 2 - r * stepy))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port, e.g. COM9")
    ap.add_argument("--sim", type=Path, help="simulate and render this PNG instead")
    ap.add_argument("--size", type=float, default=60.0, help="shape size in mm (default 60)")
    ap.add_argument(
        "--no-home",
        action="store_true",
        help="skip homing: only safe when the machine has not been powered off or moved by hand",
    )
    ap.add_argument("--variant", type=int, default=0, help="which variant, 0 = fastest")
    ap.add_argument("--shapes", help="comma-separated names (default: the eight)")
    ap.add_argument("--numbers", help="comma-separated catalogue numbers, e.g. 81,97")
    ap.add_argument(
        "--slot",
        type=int,
        default=1,
        help="first grid cell to use (1-9), so a repeat lands on clean paper",
    )
    ap.add_argument(
        "--catalogue",
        type=Path,
        default=Path("out/detailed-catalogue.json"),
        help="the numbered list --numbers refers to",
    )
    args = ap.parse_args()

    lib = Library.load()
    cfg = load_config()
    jobs = []

    if args.numbers:
        import json

        wanted = [int(n) for n in args.numbers.split(",")]
        rows = {r["n"]: r for r in json.loads(args.catalogue.read_text(encoding="utf-8"))}
        missing = [n for n in wanted if n not in rows]
        if missing:
            raise SystemExit(f"not in {args.catalogue}: {missing}")
        spots = layout(cfg, args.size, 9)[args.slot - 1 : args.slot - 1 + len(wanted)]
        for n, centre in zip(wanted, spots, strict=True):
            r = rows[n]
            p = place_object(
                r["object"], centre, args.size, lib, variant=r["variant"], style="detailed"
            )
            _, st = plan([p])
            jobs.append((f"{n}. {r['object']}", p, st.est_time_s))
    else:
        names = [s.strip() for s in args.shapes.split(",")] if args.shapes else SHAPES
        # The fastest "detailed" variant of each.
        for name, centre in zip(names, layout(cfg, args.size, len(names)), strict=True):
            best, est = None, None
            for d in lib.variants(name, style="detailed"):
                p = place_object(name, centre, args.size, lib, variant=d.key_id, style="detailed")
                _, st = plan([p])
                if est is None or st.est_time_s < est:
                    best, est = p, st.est_time_s
            jobs.append((name, best, est))

    transport = (
        SimulatedPlotter(cfg)
        if args.sim
        else SerialLink(
            args.port,
            cfg.baud,
            window=cfg.window,
            boot_timeout=cfg.boot_timeout_s,
            ack_timeout=cfg.ack_timeout_s,
            servo=cfg.servo_enabled,
        )
    )
    ctl = Controller(transport, cfg, hands_clear=True)
    if not args.sim:
        print("No camera interlock: keep hands off the bed. Ctrl+C stops.", file=sys.stderr)
    ctl.start()
    results = []
    try:
        if not args.no_home:
            ctl.home()
            ctl.wait()
        for name, placed, est in jobs:
            print(f"drawing {name} ...", flush=True)
            t0 = time.monotonic()
            ctl.draw_polylines(list(placed.polylines))
            ctl.wait()
            results.append((name, time.monotonic() - t0, est))
        ctl.park()
        ctl.wait()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
        ctl.stop()
        ctl.wait(10)
    finally:
        if args.sim:
            transport.render_png(args.sim)
            print(f"wrote {args.sim}")
        ctl.close()
        transport.close()

    print(f"\n{'shape':12s} {'measured':>10s} {'estimate':>10s}")
    for name, real, est in results:
        print(f"{name:16s} {real:9.1f}s {est:9.1f}s")
    if results:
        print(f"\ntotal {sum(r for _, r, _ in results):.1f}s for {len(results)} shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
