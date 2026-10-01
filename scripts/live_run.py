"""The whole app, live: point at the paper, talk, and the plotter draws.

    uv run python scripts/live_run.py --port COM9 --backend msmf             # the real thing
    uv run python scripts/live_run.py --dry-run --backend msmf               # print G-code only
    uv run python scripts/live_run.py --dry-run --backend msmf --type        # type, don't talk
    uv run python scripts/live_run.py --port COM9 --backend msmf --ai mock   # no network
    uv run python scripts/live_run.py --dry-run --frames out/live/20261001-225450 --type
                                       # no camera: replay a run's saved pictures

Camera (fixed bed homography from `.cache/calibration/bed.json`, no bed markers), MediaPipe
hands, `GestureTracker` (holds, sizes, the hands-clear interlock), the microphone with local
ASR (or `--type`), the agent (Nemotron, or `--ai mock`: recorded replies and the rule
parser), and the real controller behind `LivePlotter`. Cosmos is off unless `--cosmos live`.

At start it homes Y (the bed slips if pushed) and parks, once no hand is over the paper.
The plotter moves only when no hand has been over the paper (plus 10 mm) for half a second,
and pauses when one comes back. If the camera stops, so does the plotter.

The window shows the camera turned 180 degrees, as the user sees the bed, with a banner in
big words saying what the plotter is doing and why it waits ("Waiting: hand over the
paper", "Drawing", "Paused: hand over the paper", "Stopped", "Plotter problem"); the
banner's changes go in the timeline (kind "banner"). The run is logged
in `out/live/<time>/`: `timeline.jsonl` (the runtime's events), `hands.jsonl` (a header
with the start time, then per frame: each hand's landmarks, pose and fingertip, and what the
gesture tracker made of it) and `frames/` (a small picture twice a second, named by
milliseconds since start). The timeline also has the plotter's own events (kind "plotter":
job start and end, pause request, paused, resumed, halted), written as they happen.
`scripts/replay_live.py` runs a logged session again offline.

If the USB cable to the plotter is pulled, the window says so, nothing moves, and R
reconnects (the plotter then homes and parks). If the microphone is unplugged, the window
says so and it is tried again every 2 s.

Keys: Q or Esc quit, Space stop the plotter, S save a snapshot, R reconnect the plotter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import faulthandler
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
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
from pointdraw.live.frames import SavedFrames, frame_paths  # noqa: E402
from pointdraw.live.logs import JsonlWriter, TimelineWriter  # noqa: E402
from pointdraw.live.status import Banner  # noqa: E402
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
    ap.add_argument(
        "--frames", type=Path, default=None, help="play a run's saved pictures, not the camera"
    )
    ap.add_argument("--frames-fps", type=float, default=30.0, help="with --frames: pictures/s")
    ap.add_argument(
        "--hand-scale",
        type=float,
        default=1.0,
        help="run MediaPipe on the frame scaled by this (0.5 moved the fingertip up to 1 mm)",
    )
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
    ap.add_argument("--profile", action="store_true", help="time each stage of the loop")
    ap.add_argument("--shots", type=int, default=0, help="save the window every N frames")
    ap.add_argument(
        "--hands-log",
        choices=["thread", "sync"],
        default="thread",
        help="write hands.jsonl from a background thread (or on the loop, as on 1 Oct)",
    )
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


WINDOW_W = 1280  # the window's width in pixels


def put(vis: np.ndarray, text: str, org: tuple[int, int], colour=WHITE, scale: float = 0.6) -> None:
    thick = max(2, round(4 * scale / 0.6))
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick - 1, cv2.LINE_AA)


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
                "handedness": h.handedness,
                "mm": np.round(h.landmarks_mm, 2).tolist(),  # bed mm, for replay_live.py
                "world": None if h.world is None else np.round(h.world, 4).tolist(),
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


class Laps:
    """With --profile: the time each stage of the camera loop takes, per frame."""

    def __init__(self, on: bool) -> None:
        self.on = on
        self.laps: dict[str, list[float]] = defaultdict(list)
        self._t = time.perf_counter()
        self._start = self._t
        self.frames = 0

    def lap(self, name: str) -> None:
        if self.on:
            now = time.perf_counter()
            self.laps[name].append(now - self._t)
            self._t = now

    def report(self) -> dict:
        wall = time.perf_counter() - self._start
        out = {
            "frames": self.frames,
            "wall_s": round(wall, 2),
            "fps": round(self.frames / wall, 1) if wall > 0 else 0.0,
            "stages_ms": {},
        }
        for k, v in self.laps.items():
            v = sorted(v)
            out["stages_ms"][k] = {
                "median": round(1000 * statistics.median(v), 2),
                "p95": round(1000 * v[int(0.95 * (len(v) - 1))], 2),
                "mean": round(1000 * statistics.fmean(v), 2),
            }
        return out


def draw_overlay(vis, hf, view: BedView, paper, plotter: LivePlotter, scale: float = 1.0) -> None:
    """The paper, the robot's ink and the hands, on `vis` (the camera image times `scale`)."""
    corners = np.array(
        [(paper.x, paper.y), (paper.x2, paper.y), (paper.x2, paper.y2), (paper.x, paper.y2)]
    )
    quad = (view.to_px(corners) * scale).astype(np.int32)
    cv2.polylines(vis, [quad], True, BLUE, 2, cv2.LINE_AA)
    ink = plotter.ink_polylines()[-2000:]
    if ink:
        px = (view.to_px(np.array(ink).reshape(-1, 2)) * scale).astype(np.int32).reshape(-1, 2, 2)
        cv2.polylines(vis, list(px), False, RED, 1, cv2.LINE_AA)
    for h in hf.hands:
        col = GREEN if h.pointing else CYAN
        lm = (h.landmarks_px * scale).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(vis, tuple(lm[a]), tuple(lm[b]), col, 2, cv2.LINE_AA)
        cv2.circle(vis, tuple(lm[INDEX_TIP]), 7, col, 2)


def render_window(
    img, hf, view: BedView, paper, plotter: LivePlotter, top, lines, banner=None
) -> np.ndarray:
    """The window's picture: shrunk first (drawing on the full 1080p frame took ~7 ms), the
    overlay, turned 180 degrees as the user sees the bed, then the `banner` (text, colour) in
    big letters on a bar, `top` [(text, colour)] under it and the last few `lines` at the
    bottom."""
    scale = min(1.0, WINDOW_W / img.shape[1])
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    draw_overlay(vis, hf, view, paper, plotter, scale)
    vis = cv2.rotate(vis, cv2.ROTATE_180)
    y0 = 32
    if banner is not None:
        text, colour = banner
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 64), colour, -1)
        cv2.putText(vis, text, (16, 46), cv2.FONT_HERSHEY_DUPLEX, 1.25, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(vis, text, (16, 46), cv2.FONT_HERSHEY_DUPLEX, 1.25, WHITE, 2, cv2.LINE_AA)
        y0 = 96
    for i, (text, colour) in enumerate(top):
        put(vis, text, (14, y0 + 30 * i), colour)
    for i, s in enumerate(lines):
        put(vis, s, (14, vis.shape[0] - 20 - 30 * (len(lines) - 1 - i)))
    return vis


def problems(plotter_state: str, plotter_problem: str | None, mic_problem: str | None):
    """What is wrong, in plain words for the window."""
    out = []
    if plotter_state == "error" or plotter_problem:
        out.append("Plotter problem: it isn't answering. Check its USB cable, then press R.")
    if mic_problem:
        out.append("Microphone lost: plug it back in. Trying again every 2 seconds.")
    return out


def reconnect(plotter: LivePlotter, rt: Runtime, log: EventLog) -> bool:
    """R: open the plotter's link again after a failure, then home and park."""
    t = clock.now()
    if plotter.status().state != "error":
        print("R: the plotter link is fine")
        return False
    print("reconnecting the plotter ...", flush=True)
    try:
        plotter.reconnect()
    except (LinkError, RuntimeError) as e:
        log.add(t, "reconnect_failed", why=str(e))
        print(f"reconnect failed: {e}")
        return False
    rt.plotter_back(clock.now())
    plotter.park(tag="reconnect")  # homes first, then parks (once no hand is over the paper)
    print("plotter reconnected: it homes and parks once no hand is over the paper")
    return True


def next_frame(cam, plotter, headless: bool):
    """The next camera frame; None if none came for 0.5 s (motion held); "quit" to end.

    With no frame nothing watches for hands, so the plotter is held. The window keeps
    answering keys meanwhile, so Q still quits. A camera that has stopped for good ends
    the run (review 2 Oct 2026: the loop used to spin without the window, and a camera
    error was not caught).
    """
    try:
        frame = cam.read(timeout=0.5)
    except CameraUnavailable as e:
        plotter.set_hands_clear(False)
        print(f"the camera stopped ({e}): plotter held, ending the run")
        return "quit"
    if frame is not None:
        return frame
    plotter.set_hands_clear(False)  # no picture: nothing watches for hands
    print("camera gave no frame for 0.5 s: plotter held")
    if not headless and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        return "quit"
    return None


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    calib = BedCalibration.load_or_none(args.calib)
    if calib is None:
        print(f"No bed calibration at {args.calib}: run scripts/calibrate.py first.")
        return 1
    # The hand tracker first, before the camera's grab thread runs: on 1 Oct one start in six
    # crashed in this stretch (see scripts/startup_check.py and the 2 Oct night report).
    print("start: hand tracker", flush=True)
    hands = HandTracker(scale=args.hand_scale)
    index = args.camera if args.camera is not None else get_settings().camera_index
    print(f"start: camera {index} ({args.backend})", flush=True)
    try:
        if args.frames is not None:
            cam = SavedFrames(
                frame_paths(args.frames), fps=args.frames_fps, size=(args.width, args.height)
            )
        else:
            cam = Camera(
                CameraSettings(
                    index=index,
                    width=args.width,
                    height=args.height,
                    backend=args.backend,
                    exposure=args.exposure,
                )
            ).start()
    except (CameraUnavailable, ValueError) as e:
        print(f"No camera: {e}. Try --camera 1 or --backend dshow.")
        hands.close()
        return 1
    first = cam.read(timeout=5.0)
    if first is None:
        print("the camera gave no picture")
        cam.close()
        hands.close()
        return 1
    bed = fixed_bed_homography(calib, lens=load_lens(args.lens, first.size))
    if bed is None:
        print("the calibration has no cross fit: run scripts/calibrate.py again")
        cam.close()
        hands.close()
        return 1
    print("start: plotter", flush=True)

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
    hands_log = JsonlWriter(out / "hands.jsonl")
    if args.hands_log == "sync":  # as on 1 Oct, for --profile comparisons
        hands_log.close()
        sync_log = (out / "hands.jsonl").open("w", encoding="utf-8")
        hands_log.write = lambda r: print(json.dumps(r), file=sync_log)  # type: ignore[method-assign]
        hands_log.close = sync_log.close  # type: ignore[method-assign]
    timeline = TimelineWriter(out / "timeline.jsonl")
    hands_log.write({"t0": first.t, "config": args.config, "hand_scale": args.hand_scale})
    laps = Laps(args.profile)
    next_pic = 0.0
    window = "pointdraw live (Q quit, Space stop, S save, R reconnect)"
    mic_problem: str | None = None
    banner = Banner()
    said_banner = ""
    if args.shots:
        (out / "window").mkdir(parents=True, exist_ok=True)
    shown = 0
    frames = 0
    lines: list[str] = []
    print("ready: homing Y and parking once no hand is over the paper")
    rt.start(clock.now())
    try:
        frame = first
        while True:
            if frame is None:
                frame = next_frame(cam, plotter, args.headless)
            if isinstance(frame, str):  # "quit"
                break
            if frame is None:
                continue
            laps.lap("read")
            t, img = frame.t, frame.image
            frame = None
            frames += 1
            laps.frames = frames
            hf = hands.process(img, t, view.to_mm)
            laps.lap("hands")
            near = any(over_bed(h.landmarks_mm, paper, 10.0) for h in hf.hands)
            camera.offer(img, t, near)
            while not words.events.empty():
                ev = words.events.get_nowait()
                if ev[0] == "speech_start":
                    rt.on_speech_start(ev[1])
                elif ev[0] == "utterance":
                    rt.on_utterance(ev[1], ev[2])
                elif ev[0] == "mic_back":
                    mic_problem = None
                    log.add(ev[1], "mic_back")
                    print("microphone back")
                else:
                    mic_problem = ev[1]
                    log.add(ev[2], "mic_lost", why=ev[1])
                    print(f"microphone lost ({ev[1]}): trying again every 2 s")
            rt.on_frame(hf)
            laps.lap("gestures")
            hands_log.write(hand_record(t - first.t, hf, rt, paper))
            laps.lap("hands_log")
            if t >= next_pic:
                next_pic = t + 0.5
                pic = cv2.rotate(cv2.resize(img, (640, 360)), cv2.ROTATE_180)
                cv2.imwrite(str(out / "frames" / f"{round((t - first.t) * 1000):07d}.jpg"), pic)
            laps.lap("picture")
            rt.tick(t)
            for e in log.events[shown:]:
                text = describe(e)
                if text:
                    print(f"[{e['t'] - first.t:7.2f}] {text}")
                    if e["kind"] in ("utterance", "say", "state"):
                        lines = (lines + [text])[-4:]
            shown = len(log.events)
            timeline.write_new(log.events)
            timeline.write_plotter(plotter.events)
            laps.lap("tick")

            key = -1
            st = plotter.status()  # also notices a link that died while idle
            banner.update(plotter.events)
            words_now = banner.text(
                st.state, plotter.busy, rt.tracker.hands_clear, rt.state, rt.plotter_problem
            )
            if words_now[0] != said_banner:
                said_banner = words_now[0]
                log.add(t, "banner", text=said_banner)
            shoot = args.shots and frames % args.shots == 0
            if not args.headless or shoot:
                top = [(f"plotter {st.state}, app {rt.state}", YELLOW)]
                top += [(text, RED) for text in problems(st.state, rt.plotter_problem, mic_problem)]
                vis = render_window(img, hf, view, paper, plotter, top, lines, words_now)
                if shoot:
                    slug = "".join(c if c.isalnum() else "-" for c in words_now[0].lower())
                    cv2.imwrite(str(out / "window" / f"{frames:06d}-{slug[:24]}.png"), vis)
            if not args.headless:
                cv2.imshow(window, vis)
                key = cv2.waitKey(1) & 0xFF
            laps.lap("window")
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                rt.do_stop(clock.now(), "space key")
            if key == ord("r"):
                reconnect(plotter, rt, log)
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
        if args.profile:
            rep = laps.report()
            (out / "timing.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
            print(f"loop: {rep['fps']} fps over {rep['frames']} frames")
            for k, v in rep["stages_ms"].items():
                print(f"  {k:10} median {v['median']:6.2f} ms  p95 {v['p95']:6.2f} ms")
        timeline.write_new(log.events)
        timeline.write_plotter(plotter.events)
        timeline.close()
        print(f"log: {out / 'timeline.jsonl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # A crash in native code (camera driver, MediaPipe, ONNX) prints its Python stack.
    faulthandler.enable()
    a = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cv2.setNumThreads(4)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
