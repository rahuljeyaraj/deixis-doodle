"""Live vision demo: bed markers, hand skeleton, fingertip, holds and pinch span.

    uv run python scripts/vision_demo.py                 # webcam from Settings (CAMERA_INDEX)
    uv run python scripts/vision_demo.py --camera 1 --backend msmf
    uv run python scripts/vision_demo.py --video clip.mp4
    uv run python scripts/vision_demo.py --crosses out/crosses.json   # enables the C key

Works with any webcam and no markers: without a bed calibration it shows pixels. With
`.cache/calibration/bed.json` and at least two known markers in view, it shows bed mm.
A lens model in `.cache/calibration/lens.json` (see `python -m pointdraw.vision.lens`) is
used automatically when its resolution matches the camera.

Keys: Q or Esc quit · S save a snapshot to out/vision/ · C calibrate from the crosses in
view (needs --crosses) · P open the camera driver's settings dialog (DSHOW only)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from deixis.filters import OneEuroFilter2D
from deixis.holds import HoldDetector, HoldParams
from pointdraw import clock
from pointdraw.config import get_settings
from pointdraw.vision.aruco import (
    DEFAULT_CALIBRATION_PATH,
    BedCalibration,
    BedTracker,
    detect_markers,
)
from pointdraw.vision.calibration import (
    CalibrationError,
    OrientationPrior,
    calibrate_from_image,
    load_cross_points,
)
from pointdraw.vision.camera import (
    Camera,
    CameraSettings,
    CameraUnavailable,
    FpsCounter,
    ImageSource,
    VideoFileSource,
)
from pointdraw.vision.hands import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, HandTracker
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel

log = logging.getLogger("vision_demo")

GREEN, YELLOW, RED, CYAN, WHITE = (
    (60, 220, 60),
    (0, 220, 255),
    (60, 60, 255),
    (255, 200, 0),
    (255, 255, 255),
)
FLASH_S = 0.7


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
    ap.add_argument("--exposure", type=float, default=None, help="e.g. -6 (log2 seconds)")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIBRATION_PATH)
    ap.add_argument("--lens", type=Path, default=DEFAULT_LENS_PATH)
    ap.add_argument("--crosses", type=Path, help="JSON of the crosses' machine mm (for C)")
    ap.add_argument(
        "--up",
        default="auto",
        help="machine axis at the image top, e.g. -y (for C); auto reads it off the markers",
    )
    ap.add_argument("--px-per-mm", type=float, default=2.0, help="rough scale when uncalibrated")
    ap.add_argument("--headless", action="store_true", help="no window (for tests)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0: never)")
    ap.add_argument("--out", type=Path, default=Path("out/vision"))
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
    print(f"camera {index}: {cam.backend}; controls: {cam.controls or 'untouched'}")
    return cam, "camera"


def load_lens(path: Path, size: tuple[int, int]) -> LensModel | None:
    lens = LensModel.load_or_none(path)
    if lens is None:
        return None
    if tuple(lens.image_size) != tuple(size):
        print(f"lens model {path} is for {lens.image_size}, camera gives {size}: not used")
        return None
    print(f"lens model: {path} (rms {lens.rms_px:.2f} px)")
    return lens


def draw_hand(vis: np.ndarray, pts: np.ndarray, colour: tuple[int, int, int]) -> None:
    for a, b in HAND_CONNECTIONS:
        cv2.line(vis, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(vis, tuple(np.int32(p)), 3, WHITE, -1, cv2.LINE_AA)


def to_px(p: tuple[float, float], mm: bool, bed, px_per_mm: float) -> tuple[int, int]:
    """A hold position (bed mm, or pixels / px_per_mm) back to image pixels."""
    q = bed.to_px(np.array([p]))[0] if mm and bed is not None else np.array(p) * px_per_mm
    return int(q[0]), int(q[1])


def put(vis: np.ndarray, text: str, org: tuple[int, int], colour=WHITE, scale: float = 0.8) -> None:
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> int:
    try:
        source, kind = open_source(args)
    except CameraUnavailable as e:
        print(f"No camera: {e}. Plug one in, or try --camera 1, --backend msmf, or --video FILE.")
        return 1
    except FileNotFoundError as e:
        print(f"Cannot open {e}")
        return 1

    try:
        tracker_hands = HandTracker()
    except Exception as e:  # model download or MediaPipe start-up failed
        print(f"Hand tracking unavailable: {e}")
        print("Download the model by hand into .cache/models/hand_landmarker.task (see hands.py).")
        source.close()
        return 1

    calib = BedCalibration.load_or_none(args.calib)
    print(f"bed calibration: {args.calib if calib else 'none (showing pixels)'}")
    crosses = load_cross_points(args.crosses) if args.crosses else None
    lens: LensModel | None = None
    lens_checked = False
    bed_tracker: BedTracker | None = None

    smoother = OneEuroFilter2D()
    holds = HoldDetector(HoldParams(frame="bed"))
    flashes: list[tuple[float, tuple[float, float], bool]] = []  # (until, point, in mm)
    fps = FpsCounter()
    was_mm = False
    frames = 0
    started = clock.now()
    window = "vision demo (Q quit, S save, C calibrate, P camera settings)"
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
            if not lens_checked:
                lens = load_lens(args.lens, frame.size)
                if calib is not None and bool(calib.notes.get("undistorted")) != (lens is not None):
                    print("warning: calibration and lens model disagree; recalibrate (C)")
                lens_checked = True
            if calib is not None and bed_tracker is None:
                bed_tracker = BedTracker(calib, lens=lens)

            markers = detect_markers(img)
            bed = bed_tracker.update(markers, t) if bed_tracker is not None else None
            hands = tracker_hands.process(img, t, bed.to_mm if bed is not None else None)
            hand = hands.primary()

            in_mm = bed is not None
            if in_mm != was_mm:  # units changed: old filter state is meaningless
                smoother.reset()
                holds.reset()
                was_mm = in_mm
            if hand is None:
                smoother.reset()
                ended = holds.update(t, None)
            else:
                raw = (
                    hand.index_tip_mm
                    if in_mm
                    else tuple(np.array(hand.index_tip_px) / args.px_per_mm)
                )
                ended = holds.update(t, smoother(t, raw), hand.pointing)
            if ended is not None:
                flashes.append((t + FLASH_S, ended.mean, in_mm))
                unit = "mm" if in_mm else "px/scale"
                print(
                    f"HOLD {ended.t0:.2f}-{ended.t1:.2f} s ({ended.t1 - ended.t0:.2f} s) at "
                    f"({ended.mean[0]:.1f}, {ended.mean[1]:.1f}) {unit}, spread "
                    f"{max(ended.spread):.2f}"
                )

            vis = img.copy()
            if markers:
                cv2.aruco.drawDetectedMarkers(
                    vis,
                    [m.corners_px.reshape(1, 4, 2).astype(np.float32) for m in markers],
                    np.array([[m.id] for m in markers], dtype=np.int32),
                )

            for h in hands.hands:
                draw_hand(vis, h.landmarks_px, GREEN if h.pointing else CYAN)
            y = 40
            if hand is not None:
                tip = tuple(np.int32(hand.landmarks_px[INDEX_TIP]))
                cv2.circle(vis, tip, 10, GREEN if hand.pointing else CYAN, 2, cv2.LINE_AA)
                if in_mm:
                    x_mm, y_mm = hand.index_tip_mm
                    tip_text = f"tip {x_mm:7.1f}, {y_mm:7.1f} mm"
                    pinch = f"pinch {hand.pinch_mm:5.1f} mm"
                else:
                    tip_text = f"tip {tip[0]:5d}, {tip[1]:5d} px"
                    pinch = f"pinch {hand.pinch_px:5.0f} px"
                put(vis, f"{tip_text}  {'POINTING' if hand.pointing else ''}", (20, y))
                y += 36
                thumb = tuple(np.int32(hand.landmarks_px[THUMB_TIP]))
                cv2.line(vis, thumb, tip, YELLOW, 1, cv2.LINE_AA)
                span = hands.two_hand_span_mm if in_mm else hands.two_hand_span_px
                two = f"   two hands {span:5.0f} {'mm' if in_mm else 'px'}" if span else ""
                put(vis, pinch + two, (20, y), YELLOW)
                y += 36
            cur = holds.current()
            if cur is not None:
                cv2.circle(
                    vis, to_px(cur.mean, in_mm, bed, args.px_per_mm), 18, YELLOW, 2, cv2.LINE_AA
                )
            flashes = [f for f in flashes if f[0] > t]
            for until, p, mm in flashes:
                r = int(20 + 30 * (until - t) / FLASH_S)
                cv2.circle(vis, to_px(p, mm, bed, args.px_per_mm), r, RED, 4, cv2.LINE_AA)
            status = f"{fps.fps:4.1f} fps  markers {[m.id for m in markers]}"
            if bed is not None:
                status += f"  bed rms {bed.rms_mm:.2f} mm"
            elif calib is None:
                status += "  uncalibrated"
            else:
                status += "  bed not found (need 2 known markers)"
            put(vis, status, (20, vis.shape[0] - 20), WHITE, 0.7)

            key = -1
            if not args.headless:
                scale = min(1.0, 1280 / vis.shape[1])
                cv2.imshow(window, cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                args.out.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(str(args.out / f"{stamp}-raw.png"), img)
                cv2.imwrite(str(args.out / f"{stamp}-overlay.png"), vis)
                print(f"saved {args.out / stamp}-*.png")
            if key == ord("p") and isinstance(source, Camera):
                print(
                    "settings dialog:",
                    "opened" if source.show_settings_dialog() else "not supported",
                )
            if key == ord("c"):
                if crosses is None:
                    print("C needs --crosses FILE (the machine mm of the crosses the plotter drew)")
                else:
                    try:
                        prior = "auto" if args.up == "auto" else OrientationPrior(up=args.up)
                        res = calibrate_from_image(img, crosses, lens=lens, prior=prior)
                    except CalibrationError as e:
                        print(f"calibration failed: {e}")
                    else:
                        calib = res.calibration
                        print(f"calibrated: {res.summary()}; saved {calib.save(args.calib)}")
                        bed_tracker = BedTracker(calib, lens=lens)
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        wall = clock.now() - started
        if frames:
            print(f"{frames} frames in {wall:.1f} s ({frames / max(wall, 1e-9):.1f} fps processed)")
        source.close()
        tracker_hands.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cv2.setNumThreads(4)
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
