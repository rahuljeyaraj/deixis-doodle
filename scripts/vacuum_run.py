"""The vacuum app, live: stand on a spot, say "clean here", say "yes", step out.

    uv run python scripts/vacuum_run.py --camera 1                    # the real robot
    uv run python scripts/vacuum_run.py --camera 1 --robot fake       # a pretend robot
    uv run python scripts/vacuum_run.py --camera 1 --robot none       # send nothing at all
    uv run python scripts/vacuum_run.py --camera 1 --type             # type, don't talk
    uv run python scripts/vacuum_run.py --camera 1 --speak            # replies aloud
    uv run python scripts/vacuum_run.py --camera 1 --ai rules         # no network for the agent

Before the first run: lay the four floor markers and save where they are
(`scripts/vacuum_floor.py`), then line the camera's floor up with the robot's map
(`scripts/vacuum_calibrate.py`). Without that calibration the app still shows the boxes, but
won't send one.

A camera high up looks at the floor patch. Each frame: the floor markers give the floor fit,
MediaPipe Pose gives the feet, and the spot between them goes to the app. Speech (local ASR,
or `--type`) gives timed words; the agent (Nemotron, or `--ai rules`) turns them into the
vacuum's tool calls; the deixis core (`deixis-mcp`, over MCP; `DEIXIS_MODE=direct` to run it
in-process) finds the box. On "yes", once you've stepped out of the box, the box goes onto the
robot's map and to the robot's server (`deixis-vacuum-mcp`, in ../deixis-vacuum).

The window shows the camera with the floor's 50 cm grid, your spot (filled amber), the
robot's guess (a dashed blue box; a thin solid line once sent) and, in large words, what the
app is doing, with its last words under it. The run is logged in `out/vacuum/<time>/`:
`timeline.jsonl` (what was heard, the calls, the boxes, when each reply was spoken, and the
robot's own reported positions) and `feet.jsonl` (the spot in every frame).
`scripts/measure_vacuum.py` reads it for M24 and M25.

Keys: Q or Esc quit, Space stop the robot, S save a snapshot, C clear the chat thread,
R reset for a fresh take (back to listening, the guess and the sent box gone, chat cleared;
the robot is left alone, so if it is cleaning it carries on and "stop" still reaches it).
"""

from __future__ import annotations

import argparse
import datetime as dt
import faulthandler
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from deixis.mcp import DeixisClient  # noqa: E402
from pointdraw.layout import (
    FLOOR_MARKER_IDS,  # noqa: E402
    ROBOT_MARKER_ID,  # noqa: E402
)
from pointdraw.live.logs import JsonlWriter, TimelineWriter  # noqa: E402
from pointdraw.live.look import fit_window  # noqa: E402
from pointdraw.runtime.events import EventLog  # noqa: E402
from pointdraw.vacuum.agent import Background, RuleAgent, VacuumAgent  # noqa: E402
from pointdraw.vacuum.app import VacuumApp  # noqa: E402
from pointdraw.vacuum.calibrate import DEFAULT_MAP_PATH, FloorToMap  # noqa: E402
from pointdraw.vacuum.feet import FeetTracker  # noqa: E402
from pointdraw.vacuum.floor import (  # noqa: E402
    DEFAULT_FIXED_PATH,
    DEFAULT_LAYOUT_PATH,
    FixedFloor,
    FloorCamera,
    FloorLayout,
)
from pointdraw.vacuum.link import LinkError, VacuumLink  # noqa: E402
from pointdraw.vacuum.profile import vacuum_profile  # noqa: E402
from pointdraw.vacuum.view import draw  # noqa: E402
from pointdraw.vacuum.watch import RobotWatch  # noqa: E402
from pointdraw.vision.aruco import detect_markers  # noqa: E402
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0, help="camera index (the floor camera)")
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--floor", type=Path, default=DEFAULT_LAYOUT_PATH, help="floor layout JSON")
    ap.add_argument("--fixed", type=Path, default=DEFAULT_FIXED_PATH,
                    help="the floor fitted from the robot's drive (vacuum_map.py)")  # fmt: skip
    ap.add_argument("--markers", action="store_true",
                    help="use the 4 floor markers even if a fitted floor is saved")  # fmt: skip
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH, help="floor-to-map JSON")
    ap.add_argument("--robot", choices=["real", "fake", "none"], default="real")
    ap.add_argument(
        "--cosmos",
        choices=["auto", "live", "off"],
        default="auto",
        help="find a thing named by name ('clean around the chair'); auto = on if "
        "COSMOS_BASE_URL is set",
    )
    ap.add_argument("--ai", choices=["live", "rules"], default="live")
    ap.add_argument("--model", default=None, help="Nemotron model (default: the code default)")
    ap.add_argument("--type", action="store_true", help="type lines instead of speaking")
    ap.add_argument("--mic", default=None, help="input device index or name")
    ap.add_argument("--list-mics", action="store_true", help="print the microphones and stop")
    ap.add_argument("--asr", default=None, help="ASR model (default: the code default)")
    ap.add_argument("--vad", default="silero", choices=["silero", "energy"])
    ap.add_argument("--speak", action="store_true", help="say the replies aloud")
    ap.add_argument("--out", type=Path, default=Path("out/vacuum"))
    return ap.parse_args(argv)


def make_words(args: argparse.Namespace, muted=None):
    from pointdraw.live.speech import Listener, TypedInput

    if args.type:
        print("type what you would say, then Enter (stand on the spot first)")
        return TypedInput().start()
    from pointdraw.audio.asr_local import AsrConfig, LocalAsr
    from pointdraw.audio.models import DEFAULT_ASR
    from pointdraw.audio.vad import make_vad

    asr = LocalAsr(AsrConfig(model=args.asr or DEFAULT_ASR))
    mic = int(args.mic) if args.mic and args.mic.isdigit() else args.mic
    print(f"speech: {args.asr or DEFAULT_ASR} loaded in {asr.load_s:.1f} s")
    return Listener(asr, make_vad(args.vad), device=mic, muted=muted).start()


def make_eyes(args: argparse.Namespace):
    """Cosmos, for a place given by name. None when it is off or not configured."""
    if args.cosmos == "off":
        return None
    from pointdraw.config import get_settings
    from pointdraw.cosmos.client import CosmosClient
    from pointdraw.vacuum.objects import Eyes

    if not get_settings().cosmos_base_url:
        if args.cosmos == "live":
            print("no COSMOS_BASE_URL: a place given by name will not be found")
        return None
    print("start: cosmos (a place given by name)", flush=True)
    return Eyes(CosmosClient())


def run(args: argparse.Namespace) -> int:
    if args.list_mics:
        from pointdraw.audio.capture import default_input_device, list_input_devices

        default = default_input_device()
        for d in list_input_devices():
            mark = " <- the default" if d["index"] == default else ""
            print(f"  {d['index']:>2}  {d['name']}  ({d['hostapi']}){mark}")
        return 0
    fixed = None if args.markers else FixedFloor.load_or_none(args.fixed)
    if fixed is not None:
        floor_source: FixedFloor | FloorCamera = fixed
        area_mm = fixed.area_mm
        print(f"floor: fitted from the robot's drive, {args.fixed} ({fixed.created}). "
              f"Nothing on the floor to look at, so the camera must not have moved.")  # fmt: skip
    else:
        try:
            layout = FloorLayout.load(args.floor)
        except FileNotFoundError:
            print(f"no floor at {args.fixed} or {args.floor}.")
            print("run scripts/vacuum_map.py (the robot drives it), or")
            print("scripts/vacuum_floor.py (the 4 markers and a tape measure)")
            return 1
        floor_source = FloorCamera(layout)
        area_mm = layout.area_mm
        print(f"floor: the 4 markers, {args.floor}")
    profile = vacuum_profile(area_mm)
    f2m = FloorToMap.load_or_none(args.map)
    if f2m is None:
        print(f"no floor-to-map calibration at {args.map}: boxes are shown, never sent")
    else:
        print(f"floor to map: {len(f2m.pairs)} pairs, {f2m.transform.rms_mm:.0f} mm rms")

    print("start: deixis core", flush=True)
    core = DeixisClient(profile, fallback="direct")
    print(f"  {'over MCP' if core.over_mcp else 'in-process'}")
    link = None
    if args.robot != "none":
        print(f"start: the robot's server ({args.robot})", flush=True)
        try:
            link = VacuumLink(fake=args.robot == "fake")
        except (LinkError, OSError) as e:
            print(f"could not start the robot's server: {e}")
            print("(log in first: see ../deixis-vacuum/README.md; or use --robot fake)")
            core.close()
            return 1
        print(f"  {link.status()}")
    if args.ai == "live":
        agent = Background(VacuumAgent(profile, core.tools_for_agent(), model=args.model))
    else:
        agent = Background(RuleAgent(profile))

    speaker = None
    if args.speak:
        from pointdraw.live.speaker import Speaker

        speaker = Speaker()
    run_dir = args.out / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    timeline = TimelineWriter(run_dir / "timeline.jsonl")
    feet_log = JsonlWriter(run_dir / "feet.jsonl")
    robot_log = JsonlWriter(run_dir / "robot_seen.jsonl")  # the lid marker, every frame it is found

    def say(text: str) -> None:
        print(f"robot: {text}", flush=True)
        if speaker is not None:
            speaker.say(text)

    log = EventLog()
    # The eyes look at whatever the camera last gave the loop, so a name ("the chair") is
    # found in the room as it is now. The call blocks the window for a second or two; it
    # says "Let me look for ..." first, so the wait is explained.
    latest: dict[str, Any] = {"image": None, "view": None}
    eyes = make_eyes(args)

    def look(name: str):
        if eyes is None or latest["image"] is None or latest["view"] is None:
            return None
        return eyes.look(name, latest["image"], latest["view"])

    app = VacuumApp(agent, core, link, f2m, profile, log=log, say=say, eyes=look)
    words = make_words(args, muted=speaker.muted if speaker else None)
    print("start: pose tracker", flush=True)
    tracker = FeetTracker()
    floor = floor_source
    try:
        cam = Camera(
            CameraSettings(
                index=args.camera,
                width=args.width,
                height=args.height,
                backend=args.backend,
                exposure=args.exposure,
            )
        ).start()
    except (CameraUnavailable, ValueError) as e:
        print(f"No camera: {e}. Try --camera 1 or --backend dshow.")
        return 1
    watch = RobotWatch(link).start() if link is not None else None
    n_spoken = 0
    window = "deixis vacuum"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"logging to {run_dir}. Q quits, Space stops the robot, C clears the chat, R resets.")
    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                print("the camera stopped")
                break
            t = frame.t
            # One pass finds both: the floor's markers (ignored once the floor is fitted)
            # and the one on the robot's lid, whose track is the green path in the window.
            seen = detect_markers(frame.image, ids=None)
            view = floor.update([m for m in seen if m.id in FLOOR_MARKER_IDS], t)
            latest["image"], latest["view"] = frame.image, view or latest["view"]
            lid = next((m for m in seen if m.id == ROBOT_MARKER_ID), None)
            if lid is not None:
                app.on_robot_seen(t, lid.center_px)
                # Kept for the report: the camera sees the robot 30 times a second, where its
                # own reports come about once and late (M24).
                px = [round(float(c), 1) for c in lid.center_px]
                mm = view.to_mm(np.array([lid.center_px]))[0] if view is not None else None
                on_floor = None if mm is None else [round(float(q), 1) for q in mm]
                robot_log.write({"t": round(t, 4), "px": px, "mm": on_floor})
            feet = tracker.process(frame.image, t, view)
            app.on_sample(feet.sample())
            feet_log.write({"t": round(t, 4), "spot": feet.spot_mm, "standing": feet.standing,
                            "floor": view is not None and view.fresh})  # fmt: skip
            while True:
                try:
                    ev = words.events.get_nowait()
                except Exception:  # noqa: BLE001 - queue.Empty
                    break
                if ev[0] == "utterance" and ev[1]:
                    print(f"heard: {' '.join(w.text for w in ev[1])}", flush=True)
                    app.on_utterance(ev[1], ev[2])
                elif ev[0] == "error":
                    print(f"microphone: {ev[1]}")
            app.tick(t)
            for ev in watch.drain() if watch is not None else []:  # the robot's positions (M24)
                if ev.get("kind") == "robot":
                    app.on_robot_state(t, ev.get("state"))
                log.add(ev.pop("t"), ev.pop("kind"), **ev)
            if speaker is not None and len(speaker.spoken) > n_spoken:  # when it spoke (M25)
                for text, t0, t1 in speaker.spoken[n_spoken:]:
                    log.add(t0, "spoken", text=text, t1=round(t1, 3))
                n_spoken = len(speaker.spoken)
            timeline.write_new(log.events)
            vis = draw(frame.image, view, area_mm, app, feet, t, app.last_floor_box)
            # The marks are drawn on the camera's own full-size frame and the picture is then
            # fitted to the window (rahul, 8 Oct: shrinking to a fixed 1280 and letting OpenCV
            # stretch that back up to a maximised window made a full-HD camera look pixelated).
            cv2.imshow(window, fit_window(window, vis))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                app.stop(t)
            if key == ord("c"):
                app.clear_chat(t)
                print("chat cleared")
            if key == ord("r"):
                app.reset(t, "r key")
                print("reset: listening, nothing showing (the robot was left as it was)")
            if key == ord("s"):
                path = run_dir / f"snapshot-{frame.index}.png"
                cv2.imwrite(str(path), vis)
                print(f"saved {path}")
    finally:
        cv2.destroyAllWindows()
        cam.close()
        tracker.close()
        words.close()
        agent.close()
        if watch is not None:
            watch.close()
            for ev in watch.drain():
                log.add(ev.pop("t"), ev.pop("kind"), **ev)
        timeline.write_new(log.events)
        timeline.close()
        feet_log.close()
        robot_log.close()
        if speaker is not None:
            speaker.close()
        if link is not None:
            link.close()
        core.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
