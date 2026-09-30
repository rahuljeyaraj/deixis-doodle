"""Calibrate camera to plotter: draw crosses, park, photograph them, fit, save.

    uv run python scripts/calibrate.py --port COM5 --backend msmf     # the real thing
    uv run python scripts/calibrate.py --port COM5 --no-draw          # crosses already on paper
    uv run python scripts/calibrate.py --image photo.png --no-draw    # from a saved photo
    uv run python scripts/calibrate.py --sim out/calibrate            # simulator + synthetic photo

Steps: (1) the plotter draws a grid of crosses at known machine mm (like `plot.py crosses`)
and writes their positions to `.cache/crosses.json`; (2) it parks, so the paper is in front
of the gantry and in view; (3) the camera takes a photo; (4) vision finds the crosses and the
bed markers, fits pixels -> machine mm and learns where each marker sits on the bed; (5) the
result goes to `.cache/calibration/bed.json`, which `vision_demo.py` and
`cross_under_finger.py` load.

Before the first run: stick the bed markers on upright as you read them from the front, and
do the one-off lens calibration (`python -m pointdraw.vision.lens`), because the Lenovo's wide
lens bends straight lines by a few mm near the edges. There is no camera interlock while the
crosses are drawn: keep hands off the bed. Ctrl+C stops the plotter at once.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from pointdraw.config import get_settings
from pointdraw.controller.config import DEFAULT_CONFIG_PATH, MachineConfig, load_config
from pointdraw.controller.controller import Controller
from pointdraw.controller.safety import SafetyError
from pointdraw.controller.serial_link import SerialLink
from pointdraw.controller.shapes import cross, cross_grid
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkError, Transport
from pointdraw.vision import synthetic as syn
from pointdraw.vision.aruco import DEFAULT_CALIBRATION_PATH
from pointdraw.vision.calibration import (
    CalibrationError,
    OrientationPrior,
    calibrate_from_image,
    load_cross_points,
)
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel

DEFAULT_CROSSES_JSON = Path(".cache") / "crosses.json"
WARN_RMS_MM = 0.5  # cross residual above this: check focus, lighting, lens model


def parse_grid(text: str) -> tuple[int, int]:
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"grid must look like 3x3, not {text!r}") from e
    if cols < 2 or rows < 2:
        raise argparse.ArgumentTypeError("calibration needs at least a 2x2 grid")
    return cols, rows


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    plot = ap.add_mutually_exclusive_group()
    plot.add_argument("--port", help="plotter serial port (default: Settings.serial_port)")
    plot.add_argument("--sim", type=Path, metavar="OUT_DIR", help="simulate plotter and camera")
    ap.add_argument("--no-draw", action="store_true", help="crosses already drawn: use --crosses")
    ap.add_argument("--home", action="store_true", help="home before drawing")
    ap.add_argument("--grid", type=parse_grid, default=(3, 3))
    ap.add_argument("--spacing", type=float, default=50.0, help="mm between crosses")
    ap.add_argument("--size", type=float, default=10.0, help="cross size, mm")
    ap.add_argument("--crosses", type=Path, default=DEFAULT_CROSSES_JSON)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="machine config JSON")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None, help="camera index (default: Settings)")
    src.add_argument("--image", type=Path, help="use this photo instead of the camera")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--exposure", type=float, default=None, help="e.g. -6 (MSMF)")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds for exposure to settle")

    ap.add_argument("--up", default="auto", help="machine axis at the image top, e.g. -y")
    ap.add_argument("--lens", type=Path, default=DEFAULT_LENS_PATH)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIBRATION_PATH)
    ap.add_argument("--out", type=Path, default=Path("out/calibration"), help="photo and overlay")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------- plotter


def make_transport(args: argparse.Namespace, cfg: MachineConfig) -> Transport:
    if args.sim:
        return SimulatedPlotter(cfg)
    port = args.port or get_settings().serial_port
    if not port:
        raise SystemExit("no port: pass --port COMx (see `plot.py ports`), or --sim OUT_DIR")
    return SerialLink(
        port,
        cfg.baud,
        window=cfg.window,
        boot_timeout=cfg.boot_timeout_s,
        ack_timeout=cfg.ack_timeout_s,
        servo=cfg.servo_enabled,
    )


def draw_crosses(args: argparse.Namespace, cfg: MachineConfig, ctl: Controller) -> np.ndarray:
    cols, rows = args.grid
    points = cross_grid(cfg.drawable.center, cols, rows, args.spacing)
    polylines = [pl for p in points for pl in cross((p["x"], p["y"]), args.size)]
    if args.home:
        ctl.home()
    ctl.draw_polylines(polylines, "slow")  # validates first: refused jobs raise here
    ctl.park()
    ctl.wait()
    record = {
        "frame": "bed",
        "units": "mm",
        "grid": [cols, rows],
        "spacing_mm": args.spacing,
        "size_mm": args.size,
        "simulated": bool(args.sim),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "paper": cfg.paper.model_dump(),
        "crosses": points,
    }
    args.crosses.parent.mkdir(parents=True, exist_ok=True)
    args.crosses.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"drew {len(points)} crosses; positions in {args.crosses}")
    return np.array([(p["x"], p["y"]) for p in points])


# ---------------------------------------------------------------------------- photo


def sim_photo(sim: SimulatedPlotter, cfg: MachineConfig) -> np.ndarray:
    """What a camera over the parked bed would see: the simulated strokes plus six markers."""
    ox, oy = cfg.paper.x, cfg.paper.y
    base, _ = syn.default_scene()  # markers around an A5 sheet at the origin
    markers = {i: (x + ox, y + oy) for i, (x, y) in base.markers.items()}
    lines = [(p, q) for s in sim.strokes for p, q in zip(s[:-1], s[1:], strict=False)]
    scene = syn.Scene(
        markers=markers,
        crosses=np.zeros((0, 2)),
        paper=(ox, oy, cfg.paper.w, cfg.paper.h),
        bed=(ox - 40, oy - 40, 290.0, 228.0),
        extra_lines=lines,
    )
    bed_quad = np.array([[-40, -40], [250, -40], [250, 188], [-40, 188]], float) + (ox, oy)
    img_quad = np.array([[360, 1000], [1575, 985], [1545, 70], [395, 95]], float)
    return syn.render(scene, syn.camera_homography(bed_quad, img_quad))


def camera_photo(args: argparse.Namespace) -> np.ndarray:
    index = args.camera if args.camera is not None else get_settings().camera_index
    settings = CameraSettings(
        index=index,
        width=args.width,
        height=args.height,
        backend=args.backend,
        exposure=args.exposure,
    )
    with Camera(settings) as cam:  # __enter__ starts it
        print(f"camera {index}: {cam.backend}; letting exposure settle for {args.settle:g} s")
        time.sleep(args.settle)
        frame = None
        for _ in range(5):  # the newest of a few frames
            frame = cam.read(timeout=2.0) or frame
        if frame is None:
            raise CameraUnavailable("the camera gave no frame")
        return frame.image


def load_lens(path: Path, size: tuple[int, int]) -> LensModel | None:
    lens = LensModel.load_or_none(path)
    if lens is None:
        print(f"no lens model at {path}: the wide lens may cost a few mm near the edges")
        return None
    if tuple(lens.image_size) != tuple(size):
        print(f"lens model {path} is for {lens.image_size}, photo is {size}: not used")
        return None
    return lens


def save_overlay(out: Path, img: np.ndarray, res) -> Path:
    vis = img.copy()
    for c in res.crosses:
        x, y = (int(v) for v in c.center_px)
        cv2.drawMarker(vis, (x, y), (60, 60, 255), cv2.MARKER_CROSS, 30, 2)
    for m in res.markers:
        cv2.polylines(vis, [m.corners_px.astype(np.int32)], True, (60, 220, 60), 2)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(str(out / f"{stamp}-photo.png"), img)
    path = out / f"{stamp}-overlay.png"
    cv2.imwrite(str(path), vis)
    return path


# ---------------------------------------------------------------------------- main


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    sim: SimulatedPlotter | None = None
    if args.no_draw:
        if not args.crosses.exists():
            print(f"--no-draw needs the cross positions in {args.crosses}")
            return 1
        crosses = load_cross_points(args.crosses)
    else:
        transport = make_transport(args, cfg)
        sim = transport if isinstance(transport, SimulatedPlotter) else None
        if sim is None:
            print("No camera interlock while drawing: keep hands off the bed.", file=sys.stderr)
        ctl = Controller(transport, cfg, hands_clear=True)
        try:
            ctl.start()
        except LinkError as e:
            transport.close()
            print(f"could not connect: {e}")
            return 1
        try:
            crosses = draw_crosses(args, cfg, ctl)
        except SafetyError as e:
            print(f"refused: {e}. Smaller --spacing, or measure the paper (plot.py config)")
            return 2
        except KeyboardInterrupt:
            ctl.stop()
            ctl.wait(10)
            print("stopped. Run `plot.py home` before drawing again")
            return 130
        finally:
            ctl.close()
        if ctl.status().state == "error":
            print(f"plotter error: {ctl.status().message}")
            return 1

    try:
        if args.image:
            img = cv2.imread(str(args.image))
            if img is None:
                print(f"cannot read {args.image}")
                return 1
        elif sim is not None:
            img = sim_photo(sim, cfg)
            sim.render_png(args.sim / "plot.png")
        else:
            img = camera_photo(args)
    except CameraUnavailable as e:
        print(f"No camera: {e}. Try --camera 1, --backend msmf, or --image PHOTO --no-draw.")
        return 1

    lens = None if sim is not None else load_lens(args.lens, (img.shape[1], img.shape[0]))
    prior = "auto" if args.up == "auto" else OrientationPrior(up=args.up)
    out = args.sim if sim is not None else args.out
    try:
        res = calibrate_from_image(img, crosses, lens=lens, prior=prior)
    except CalibrationError as e:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "failed-photo.png"), img)
        print(f"calibration failed: {e}. Photo saved to {out / 'failed-photo.png'}")
        return 1

    path = res.calibration.save(args.calib)
    print(f"calibrated: {res.summary()}")
    print(f"saved {path}; overlay {save_overlay(out, img, res)}")
    if len(res.calibration.marker_corners_mm) < 2:
        print("warning: fewer than 2 markers learned; the live bed tracking needs at least 2")
    if res.fit.rms_mm > WARN_RMS_MM:
        print(f"warning: residual rms {res.fit.rms_mm:.2f} mm is high: check focus, light, lens")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cv2.setNumThreads(4)
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
