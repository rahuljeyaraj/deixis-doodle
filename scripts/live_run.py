"""The whole app, live: point at the paper, talk, and the plotter draws.

    uv run python scripts/live_run.py --port COM9 --backend msmf             # the real thing
    uv run python scripts/live_run.py --dry-run --backend msmf               # print G-code only
    uv run python scripts/live_run.py --dry-run --backend msmf --type        # type, don't talk
    uv run python scripts/live_run.py --port COM9 --backend msmf --ai mock   # no network

Camera (fixed bed homography from `.cache/calibration/bed.json`, no bed markers), MediaPipe
hands, `GestureTracker` (holds, sizes, the hands-clear interlock), the microphone with local
ASR (or `--type`), the agent (Nemotron, or `--ai mock`: recorded replies and the rule
parser), and the real controller behind `LivePlotter`. Cosmos is off unless `--cosmos live`.

At start it homes Y (the bed slips if pushed) and parks, once no hand is over the paper.
The plotter moves only when no hand has been over the paper (plus 10 mm) for half a second,
and pauses when one comes back. If the camera stops, so does the plotter.

The window shows the camera turned 180 degrees, as the user sees the bed. The run is logged
in `out/live/<time>/`: `timeline.jsonl` (the runtime's events), `hands.jsonl` (per frame:
each hand's pose, fingertip and what the gesture tracker made of it) and `frames/` (a small
picture twice a second, named by milliseconds since start).

Keys: Q or Esc quit, Space stop the plotter, S save a snapshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pointdraw import clock  # noqa: E402
from pointdraw.config import get_settings  # noqa: E402
from pointdraw.controller.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MachineConfig,
    load_config,
)
from pointdraw.controller.controller import Controller  # noqa: E402
from pointdraw.controller.serial_link import SerialLink  # noqa: E402
from pointdraw.controller.simulator import SimulatedPlotter  # noqa: E402
from pointdraw.controller.transport import LinkError, Transport  # noqa: E402
from pointdraw.drawing import Library  # noqa: E402
from pointdraw.live.agent import BackgroundAgent  # noqa: E402
from pointdraw.live.view import BedView, PaperCamera  # noqa: E402
from pointdraw.m1 import over_bed  # noqa: E402
from pointdraw.runtime.events import EventLog  # noqa: E402
from pointdraw.runtime.gestures import thumb_out  # noqa: E402
from pointdraw.runtime.live_plotter import LivePlotter  # noqa: E402
from pointdraw.runtime.planner import Planner  # noqa: E402
from pointdraw.runtime.runtime import Runtime  # noqa: E402
from pointdraw.vision.aruco import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    BedCalibration,
    fixed_bed_homography,
)
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable  # noqa: E402
from pointdraw.vision.hands import HAND_CONNECTIONS, INDEX_TIP, HandTracker  # noqa: E402
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel  # noqa: E402

GREEN, YELLOW, RED, CYAN, WHITE, BLUE = (
    (60, 220, 60),
    (0, 220, 255),
    (60, 60, 255),
    (255, 200, 0),
    (255, 255, 255),
    (255, 120, 40),
)
SHOWN = ("utterance", "tool_calls", "say", "state", "preview", "confirm", "drawn", "stop")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: Settings)")
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--exposure", type=float, default=None, help="e.g. -6 (MSMF)")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIBRATION_PATH)
    ap.add_argument("--lens", type=Path, default=DEFAULT_LENS_PATH)

    plot = ap.add_mutually_exclusive_group()
    plot.add_argument("--port", help="plotter serial port (default: Settings.serial_port)")
    plot.add_argument("--dry-run", action="store_true", help="simulated plotter, print G-code")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="machine config JSON")
    ap.add_argument("--no-home-y", action="store_true", help="skip homing the bed (Y) at start")

    ap.add_argument("--ai", choices=["live", "mock"], default="live")
    ap.add_argument("--model", default=None, help="live agent model (default: the code default)")
    ap.add_argument("--cosmos", choices=["off", "live"], default="off")
    ap.add_argument("--type", action="store_true", help="type lines instead of speaking")
    ap.add_argument("--mic", default=None, help="input device index or name")
    ap.add_argument("--asr", default=None, help="ASR model (default: the code default)")
    ap.add_argument("--vad", default="silero", choices=["silero", "energy"])

    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0: never)")
    ap.add_argument("--out", type=Path, default=Path("out/live"))
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def make_transport(args: argparse.Namespace, cfg: MachineConfig) -> Transport:
    if args.dry_run:
        return SimulatedPlotter(cfg, realtime=True, echo=lambda line: print(f"  gcode {line}"))
    port = args.port or get_settings().serial_port
    if not port:
        raise SystemExit("no port: pass --port COMx (see `plot.py ports`) or --dry-run")
    return SerialLink(
        port,
        cfg.baud,
        window=cfg.window,
        boot_timeout=cfg.boot_timeout_s,
        ack_timeout=cfg.ack_timeout_s,
        servo=cfg.servo_enabled,
    )


def make_agent(args: argparse.Namespace, categories: list[str]) -> BackgroundAgent:
    from pointdraw.virtual.agents import LiveAgent, MockAgent

    if args.ai == "live":
        inner = LiveAgent(categories, model=args.model, record=False)
    else:
        inner = MockAgent(categories)
    return BackgroundAgent(inner, categories)


def make_words(args: argparse.Namespace):
    from pointdraw.live.speech import Listener, TypedInput

    if args.type:
        print("type what you would say, then Enter (point first)")
        return TypedInput().start()
    from pointdraw.audio.asr_local import AsrConfig, LocalAsr
    from pointdraw.audio.models import DEFAULT_ASR
    from pointdraw.audio.vad import make_vad

    asr = LocalAsr(AsrConfig(model=args.asr or DEFAULT_ASR))
    mic = int(args.mic) if args.mic and args.mic.isdigit() else args.mic
    print(f"speech: {args.asr or DEFAULT_ASR} loaded in {asr.load_s:.1f} s")
    return Listener(asr, make_vad(args.vad), device=mic).start()


def load_lens(path: Path, size: tuple[int, int]) -> LensModel | None:
    lens = LensModel.load_or_none(path)
    if lens is not None and tuple(lens.image_size) != tuple(size):
        print(f"lens model {path} is for {lens.image_size}, camera gives {size}: not used")
        return None
    return lens


def put(vis: np.ndarray, text: str, org: tuple[int, int], colour=WHITE, scale: float = 0.9) -> None:
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def describe(e: dict) -> str | None:
    k = e["kind"]
    if k == "utterance":
        return f'heard: "{e["text"]}"'
    if k == "tool_calls":
        calls = ", ".join(c["name"] for c in e["calls"]) or "nothing"
        return f"agent ({e['source']}, {e['latency_s']:.1f} s): {calls}"
    if k == "say":
        return f"SAYS: {e['text']}"
    if k == "state":
        return f"state {e['state']}"
    if k in ("ignored", "guard"):
        return f"{k}: {e['why']}"
    if k in ("preview", "confirm", "drawn", "stop"):
        rest = {a: b for a, b in e.items() if a not in ("t", "kind")}
        return f"{k} {rest}"
    return None


def hand_record(t: float, hf, rt: Runtime, paper) -> dict:
    """One line of hands.jsonl: what the camera saw and what the gesture tracker made of it."""
    s = rt.tracker.last
    hands = []
    for h in hf.hands:
        tip = h.landmarks_mm[INDEX_TIP] - np.array([paper.x, paper.y])
        hands.append(
            {
                "pointing": h.pointing,
                "thumb_out": thumb_out(h),
                "score": round(h.score, 2),
                "tip": [round(float(tip[0]), 1), round(float(tip[1]), 1)],
                "fingers": {
                    n: [round(f.reach, 2), round(f.bend_deg), f.extended, f.curled]
                    for n, f in h.pose.fingers.items()
                },
            }
        )
    return {
        "t": round(t, 3),
        "hands": hands,
        "tip": [round(v, 1) for v in s.tip] if s and s.tip else None,
        "span": round(s.span, 1) if s and s.span else None,
        "clear": rt.tracker.hands_clear,
        "state": rt.state,
    }


def draw_overlay(vis, hf, view: BedView, paper, plotter: LivePlotter) -> None:
    corners = np.array(
        [(paper.x, paper.y), (paper.x2, paper.y), (paper.x2, paper.y2), (paper.x, paper.y2)]
    )
    quad = view.to_px(corners).astype(np.int32)
    cv2.polylines(vis, [quad], True, BLUE, 2, cv2.LINE_AA)
    ink = plotter.ink_polylines()
    for a, b in ink[-2000:]:
        pa, pb = view.to_px(np.array([a, b])).astype(np.int32)
        cv2.line(vis, tuple(pa), tuple(pb), RED, 1, cv2.LINE_AA)
    for h in hf.hands:
        col = GREEN if h.pointing else CYAN
        for a, b in HAND_CONNECTIONS:
            pa, pb = np.int32(h.landmarks_px[a]), np.int32(h.landmarks_px[b])
            cv2.line(vis, tuple(pa), tuple(pb), col, 2, cv2.LINE_AA)
        cv2.circle(vis, tuple(np.int32(h.landmarks_px[INDEX_TIP])), 10, col, 2)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    calib = BedCalibration.load_or_none(args.calib)
    if calib is None:
        print(f"No bed calibration at {args.calib}: run scripts/calibrate.py first.")
        return 1
    index = args.camera if args.camera is not None else get_settings().camera_index
    try:
        cam = Camera(
            CameraSettings(
                index=index,
                width=args.width,
                height=args.height,
                backend=args.backend,
                exposure=args.exposure,
            )
        ).start()
    except CameraUnavailable as e:
        print(f"No camera: {e}. Try --camera 1 or --backend dshow.")
        return 1
    first = cam.read(timeout=5.0)
    if first is None:
        print("the camera gave no picture")
        cam.close()
        return 1
    bed = fixed_bed_homography(calib, lens=load_lens(args.lens, first.size))
    if bed is None:
        print("the calibration has no cross fit: run scripts/calibrate.py again")
        cam.close()
        return 1
    hands = HandTracker()

    transport = make_transport(args, cfg)
    plotter = LivePlotter(Controller(transport, cfg, hands_clear=False))
    try:
        plotter.start()
    except LinkError as e:
        transport.close()
        cam.close()
        hands.close()
        print(f"could not connect to the plotter: {e}")
        return 1
    if not args.no_home_y:
        # The bed slips if pushed, and the camera fit assumes it parked where it was then.
        plotter.ctl.run_lines(["G28 Y"], label="home Y", tag="start")
    plotter.park(tag="start")
    paper = cfg.paper_box
    view = BedView(bed, cfg.park_y, lambda: plotter.position()[1])
    camera = PaperCamera(view, paper)

    planner = Planner(Library.load())
    agent = make_agent(args, planner.categories)
    vision = None
    if args.cosmos == "live":
        from pointdraw.virtual.vision import LiveVision

        vision = LiveVision()
    log = EventLog()
    rt = Runtime(agent, plotter, cfg, planner, camera=camera, vision=vision, log=log)
    words = make_words(args)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out / stamp
    (out / "frames").mkdir(parents=True, exist_ok=True)
    hands_log = (out / "hands.jsonl").open("w", encoding="utf-8")
    next_pic = 0.0
    window = "pointdraw live (Q quit, Space stop, S save)"
    shown = 0
    frames = 0
    lines: list[str] = []
    print("ready: homing Y and parking once no hand is over the paper")
    rt.start(clock.now())
    try:
        frame = first
        while True:
            if frame is None:
                frame = cam.read(timeout=0.5)
            if frame is None:
                plotter.set_hands_clear(False)  # no picture: nothing watches for hands
                print("camera gave no frame for 0.5 s: plotter held")
                continue
            t, img = frame.t, frame.image
            frame = None
            frames += 1
            hf = hands.process(img, t, view.to_mm)
            near = any(over_bed(h.landmarks_mm, paper, 10.0) for h in hf.hands)
            camera.offer(img, t, near)
            while not words.events.empty():
                ev = words.events.get_nowait()
                if ev[0] == "speech_start":
                    rt.on_speech_start(ev[1])
                elif ev[0] == "utterance":
                    rt.on_utterance(ev[1], ev[2])
                else:
                    print(f"speech input stopped: {ev[1]}")
            rt.on_frame(hf)
            print(json.dumps(hand_record(t - first.t, hf, rt, paper)), file=hands_log)
            if t >= next_pic:
                next_pic = t + 0.5
                pic = cv2.rotate(cv2.resize(img, (640, 360)), cv2.ROTATE_180)
                cv2.imwrite(str(out / "frames" / f"{round((t - first.t) * 1000):07d}.jpg"), pic)
            rt.tick(t)
            for e in log.events[shown:]:
                text = describe(e)
                if text:
                    print(f"[{e['t'] - first.t:7.2f}] {text}")
                    if e["kind"] in ("utterance", "say", "state"):
                        lines = (lines + [text])[-4:]
            shown = len(log.events)

            key = -1
            if not args.headless:
                vis = img.copy()
                draw_overlay(vis, hf, view, paper, plotter)
                vis = cv2.rotate(vis, cv2.ROTATE_180)  # as the user sees the bed
                st = plotter.status()
                held = "" if rt.tracker.hands_clear else "  (hand over the paper)"
                put(vis, f"plotter {st.state}{held}", (20, 45), YELLOW)
                put(vis, f"app {rt.state}", (20, 90), YELLOW)
                for i, s in enumerate(lines):
                    put(vis, s, (20, vis.shape[0] - 30 - 45 * (len(lines) - 1 - i)))
                scale = min(1.0, 1280 / vis.shape[1])
                small = cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis
                cv2.imshow(window, small)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                rt.do_stop(clock.now(), "space key")
            if key == ord("s"):
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"{frames:06d}-camera.png"), img)
                print(f"saved {out / f'{frames:06d}-camera.png'}")
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if plotter.busy:
            # The camera is closing, so nothing watches for hands any more: don't move.
            print("stopping the plotter (no camera interlock after quitting)")
            plotter.stop()
            plotter.wait(10)
        plotter.close()
        words.close()
        agent.close()
        cam.close()
        hands.close()
        if not args.headless:
            cv2.destroyAllWindows()
        out.mkdir(parents=True, exist_ok=True)
        hands_log.close()
        with (out / "timeline.jsonl").open("w", encoding="utf-8") as f:
            for e in log.events:
                f.write(json.dumps(e, default=str) + "\n")
        print(f"log: {out / 'timeline.jsonl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cv2.setNumThreads(4)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
