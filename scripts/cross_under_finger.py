"""M1 app: point at the paper, hold still, take your hand away, and the plotter draws a cross.

    uv run python scripts/cross_under_finger.py --port COM5 --backend msmf
    uv run python scripts/cross_under_finger.py --port COM5 --targets hardware/print/m1-targets.json
    uv run python scripts/cross_under_finger.py --sim out/m1.png          # no plotter
    uv run python scripts/cross_under_finger.py --sim out/m1.png --trigger key   # Space draws

Needs `.cache/calibration/bed.json` from `scripts/calibrate.py`, and at least two bed markers
in view. The fingertip is found by MediaPipe, turned into bed mm by the markers, smoothed,
and a still, pointing fingertip held for 0.6 s becomes a hold. With `--trigger hold` (the
default) the cross is queued when the hold ends; with `--trigger key` press Space.

Camera interlock: the plotter only moves after no hand has been in view for half a second,
and pauses the moment a hand comes back. After each cross it parks again.

With `--targets` (the printed M1 sheet, taped square at the config's paper position) each
cross is scored against the nearest target and the trials are appended to `--log`. M1 passes
at a median error of 3 mm or less and a maximum of 5 mm or less over 10 trials.

Keys: Q or Esc quit · Space draw at the current hold (key mode) · S save a snapshot
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from pointdraw.config import get_settings
from pointdraw.controller import gcode
from pointdraw.controller.config import DEFAULT_CONFIG_PATH, MachineConfig, load_config
from pointdraw.controller.controller import Controller
from pointdraw.controller.serial_link import SerialLink
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkError, Transport
from pointdraw.m1 import CrossUnderFinger, load_targets, over_bed
from pointdraw.vision.aruco import (
    DEFAULT_CALIBRATION_PATH,
    BedCalibration,
    BedTracker,
    detect_markers,
)
from pointdraw.vision.camera import (
    Camera,
    CameraSettings,
    CameraUnavailable,
    FpsCounter,
    ImageSource,
    VideoFileSource,
)
from pointdraw.vision.hands import HAND_CONNECTIONS, INDEX_TIP, HandTracker
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel

GREEN, YELLOW, RED, CYAN, WHITE, BLUE = (
    (60, 220, 60),
    (0, 220, 255),
    (60, 60, 255),
    (255, 200, 0),
    (255, 255, 255),
    (255, 120, 40),
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None, help="camera index (default: Settings)")
    src.add_argument("--video", type=Path, help="replay a video file")
    src.add_argument("--image", type=Path, help="loop over one still image")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--exposure", type=float, default=None, help="e.g. -6 (MSMF)")

    plot = ap.add_mutually_exclusive_group()
    plot.add_argument("--port", help="plotter serial port (default: Settings.serial_port)")
    plot.add_argument("--sim", type=Path, metavar="OUT.PNG", help="simulated plotter, rendered")
    plot.add_argument("--dry-run", action="store_true", help="simulated plotter, print G-code")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="machine config JSON")
    ap.add_argument(
        "--no-home",
        action="store_true",
        help="skip homing X, Y and Z at start (not advised: the head's place is then a guess)",
    )

    ap.add_argument("--trigger", choices=["hold", "key"], default="hold")
    ap.add_argument("--cross-size", type=float, default=6.0, help="mm")
    ap.add_argument("--targets", type=Path, help="e.g. hardware/print/m1-targets.json")
    ap.add_argument("--log", type=Path, default=Path("out/m1/trials.jsonl"))
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIBRATION_PATH)
    ap.add_argument("--lens", type=Path, default=DEFAULT_LENS_PATH)
    ap.add_argument("--headless", action="store_true", help="no window (for tests)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0: never)")
    ap.add_argument("--out", type=Path, default=Path("out/m1"), help="snapshots")
    return ap.parse_args(argv)


def open_source(args: argparse.Namespace):
    if args.image:
        return ImageSource([args.image], loop=True), "image"
    if args.video:
        return VideoFileSource(args.video, realtime=not args.headless), "video"
    index = args.camera if args.camera is not None else get_settings().camera_index
    cam = Camera(
        CameraSettings(
            index=index,
            width=args.width,
            height=args.height,
            backend=args.backend,
            exposure=args.exposure,
        )
    ).start()
    print(f"camera {index}: {cam.backend}")
    return cam, "camera"


def make_transport(args: argparse.Namespace, cfg: MachineConfig) -> Transport:
    if args.sim or args.dry_run:
        return SimulatedPlotter(cfg, echo=print if args.dry_run else None)
    port = args.port or get_settings().serial_port
    if not port:
        raise SystemExit(
            "no port: pass --port COMx (see `plot.py ports`), --sim OUT.PNG or --dry-run"
        )
    return SerialLink(
        port,
        cfg.baud,
        window=cfg.window,
        boot_timeout=cfg.boot_timeout_s,
        ack_timeout=cfg.ack_timeout_s,
        servo=cfg.servo_enabled,
    )


def load_lens(path: Path, size: tuple[int, int]) -> LensModel | None:
    lens = LensModel.load_or_none(path)
    if lens is not None and tuple(lens.image_size) != tuple(size):
        print(f"lens model {path} is for {lens.image_size}, camera gives {size}: not used")
        return None
    return lens


def put(vis: np.ndarray, text: str, org: tuple[int, int], colour=WHITE, scale: float = 0.8) -> None:
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def mm_to_px(bed, p: tuple[float, float]) -> tuple[int, int]:
    q = bed.to_px(np.array([p]))[0]
    return int(q[0]), int(q[1])


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    calib = BedCalibration.load_or_none(args.calib)
    if calib is None:
        print(f"No bed calibration at {args.calib}: run scripts/calibrate.py first.")
        return 1
    try:
        source, kind = open_source(args)
    except CameraUnavailable as e:
        print(f"No camera: {e}. Try --camera 1, --backend msmf, or --video FILE.")
        return 1
    except FileNotFoundError as e:
        print(f"Cannot open {e}")
        return 1
    try:
        hands_tracker = HandTracker()
    except Exception as e:  # model download or MediaPipe start-up failed
        print(f"Hand tracking unavailable: {e}")
        source.close()
        return 1

    transport = make_transport(args, cfg)
    ctl = Controller(transport, cfg, hands_clear=False)
    try:
        ctl.start()
    except LinkError as e:
        transport.close()
        source.close()
        hands_tracker.close()
        print(f"could not connect to the plotter: {e}")
        return 1
    if not args.no_home:
        # Never trust the head's place after power-up (6 Oct: X was off and the head hit the east
        # end; Z was off and the pen drew in the air). The bed is also pushed by hand easily, and
        # the camera calibration assumes it parked exactly where it was then.
        ctl.run_lines(gcode.home_gcode(cfg), label="home")
    ctl.park()  # runs once hands are clear; pointing happens with the bed parked

    targets = load_targets(args.targets, cfg.paper_box) if args.targets else []
    app = CrossUnderFinger(ctl, cross_mm=args.cross_size, trigger=args.trigger, targets=targets)
    tracker: BedTracker | None = None
    fps = FpsCounter()
    frames = 0
    window = "cross under the finger (Q quit, Space draw, S save)"
    print(f"ready: {args.trigger} trigger, {len(targets)} targets; point, hold 0.6 s, withdraw")
    try:
        while True:
            frame = source.read(timeout=2.0)
            if frame is None:
                if kind == "camera":
                    print("camera gave no frame for 2 s")
                    continue
                break
            t, img = frame.t, frame.image
            fps.tick(t)
            frames += 1
            if tracker is None:
                tracker = BedTracker(calib, lens=load_lens(args.lens, frame.size))

            markers = detect_markers(img)
            bed = tracker.update(markers, t)
            hf = hands_tracker.process(img, t, bed.to_mm if bed is not None else None)
            hand = hf.primary()
            tip = hand.index_tip_mm if (hand is not None and bed is not None) else None
            # The paper plus 10 mm, as in the runtime: the camera looks from the front, so a
            # hand raised in front of the bed appears over its front edge (parallax).
            near = any(over_bed(h.landmarks_mm, cfg.paper_box, 10.0) for h in hf.hands)
            trial = app.update(t, tip, bool(hand and hand.pointing), near)
            if trial is not None:
                print(f"trial {len(app.trials)}: {trial}")

            vis = img.copy()
            for h in hf.hands:
                col = GREEN if h.pointing else CYAN
                for a, b in HAND_CONNECTIONS:
                    pa, pb = np.int32(h.landmarks_px[a]), np.int32(h.landmarks_px[b])
                    cv2.line(vis, tuple(pa), tuple(pb), col, 2, cv2.LINE_AA)
                cv2.circle(vis, tuple(np.int32(h.landmarks_px[INDEX_TIP])), 10, col, 2)
            if bed is not None:
                for tg in targets:
                    cv2.circle(vis, mm_to_px(bed, (tg.x, tg.y)), 8, BLUE, 1, cv2.LINE_AA)
                for tr in app.trials:
                    colour = GREEN if tr.drawn else RED
                    cv2.drawMarker(
                        vis, mm_to_px(bed, (tr.x, tr.y)), colour, cv2.MARKER_CROSS, 24, 2
                    )
                cur = app.current()
                if cur is not None:
                    cv2.circle(vis, mm_to_px(bed, cur.mean), 20, YELLOW, 2, cv2.LINE_AA)
            st = ctl.status()
            y = 40
            if tip is not None:
                put(vis, f"tip {tip[0]:6.1f}, {tip[1]:6.1f} mm", (20, y))
                y += 36
            put(
                vis,
                f"plotter {st.state}{'' if st.hands_clear else ' (waiting: hands in view)'}",
                (20, y),
                YELLOW,
            )
            status = f"{fps.fps:4.1f} fps  markers {[m.id for m in markers]}  "
            status += f"bed rms {bed.rms_mm:.2f} mm" if bed else "bed not found (need 2 markers)"
            status += f"  trials {len(app.trials)}"
            put(vis, status, (20, vis.shape[0] - 20), WHITE, 0.7)

            key = -1
            if not args.headless:
                scale = min(1.0, 1280 / vis.shape[1])
                cv2.imshow(window, cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                trial = app.fire()
                print(f"trial {len(app.trials)}: {trial}" if trial else "no hold to draw at yet")
            if key == ord("s"):
                args.out.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(str(args.out / f"{stamp}-overlay.png"), vis)
                print(f"saved {args.out / stamp}-overlay.png")
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopping")
        ctl.stop()
    finally:
        source.close()
        hands_tracker.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if isinstance(transport, SimulatedPlotter):
            ctl.set_hands_clear(True)  # simulated: let queued crosses finish for the render
            ctl.wait(120)
        elif not ctl.wait(0.2):
            # The camera is closed, so nothing watches for hands any more: don't move.
            print("stopping queued plotter moves (no camera interlock after quitting)")
            ctl.stop()
            ctl.wait(10)
        ctl.close()
    if args.sim and isinstance(transport, SimulatedPlotter):
        print(f"rendered {transport.render_png(args.sim)}")
    if app.trials:
        print(f"log: {app.write_log(args.log)}")
    print(app.summary().text())
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cv2.setNumThreads(4)
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
