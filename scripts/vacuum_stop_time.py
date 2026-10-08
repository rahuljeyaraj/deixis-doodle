"""How long does the vacuum take to stop? Timed by the camera, not by the robot.

    uv run python scripts/vacuum_stop_time.py --camera 0 --trials 3

Start a clean in the ECOVACS app and leave it running. The camera follows the marker on the
robot's lid. Once it has been moving for a moment, the script sends `stop` and keeps
watching: the stop has acted when the marker stops moving.

The robot's own position reports arrive about once a second and late, so they can only say
"somewhere under a second or two". The camera sees about 20 frames a second, so it can say
what actually happened. This is the vacuum's half of the plotter's `stop_time.py` (M08), and
it backs the same claim: Stop stops it.

Between trials it waits for you to start cleaning again. Writes <out>/stop_time.json and
prints the median and the worst.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

from pointdraw import clock
from pointdraw.layout import ROBOT_MARKER_ID
from pointdraw.vacuum.link import LinkError, VacuumLink
from pointdraw.vision.aruco import detect_markers
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable

MOVING_PX_S = 40.0  # the marker must be going at least this fast before a stop counts
# (7 Oct: a crawling robot read 7 px/s and a real drive peaks well over 100)
STILL_PX = 2.5  # within this of where it ends counts as still (camera jitter)
SETTLE_S = 3.0  # watch this long after it looks still, in case it creeps on
LIMIT_S = 2.0  # the bar: a stop over this long is a fail


def marker_at(image: np.ndarray) -> tuple[float, float] | None:
    """The lid marker's centre in pixels, or None if it is not in view."""
    m = next((x for x in detect_markers(image, ids=None) if x.id == ROBOT_MARKER_ID), None)
    if m is None:
        return None
    c = np.asarray(m.corners_px, np.float64).reshape(4, 2).mean(axis=0)
    return (float(c[0]), float(c[1]))


def speed_px_s(track: list[tuple[float, tuple[float, float]]], over_s: float = 1.0) -> float:
    """How fast the marker has been going over the last `over_s` seconds."""
    if len(track) < 2:
        return 0.0
    t_end = track[-1][0]
    recent = [(t, p) for t, p in track if t >= t_end - over_s]
    if len(recent) < 2 or recent[-1][0] <= recent[0][0]:
        return 0.0
    moved = math.dist(recent[0][1], recent[-1][1])
    return moved / (recent[-1][0] - recent[0][0])


def send_drive(link: VacuumLink, side_mm: float) -> dict | None:
    """Set the robot cleaning a box around where it stands, so it drives for a while."""
    st = link.locate(wait_s=1.5)
    robot = st.get("robot")
    if not robot:
        print("  the robot will not say where it is")
        return None
    x, y = float(robot["x"]), float(robot["y"])
    box = [x - side_mm / 2, y - side_mm / 2, side_mm, side_mm]
    where = {"frame": "map", "kind": "box", "box_mm": box}
    reply = link.clean_area(where)
    print(f"  sent it to clean {side_mm / 1000:.1f} m around ({x:.0f}, {y:.0f}): "
          f"{'ok' if reply.get('ok') else reply}")  # fmt: skip
    return reply


def one_trial(cam: Camera, link: VacuumLink, wait_s: float) -> dict | None:
    """Wait for the robot to be driving, stop it, and time how long it keeps going.

    The stop goes out on its own thread: it takes about half a second to come back, and that
    is the half second the robot is still moving in. Reading frames must not pause for it.
    """
    track: list[tuple[float, tuple[float, float]]] = []
    t0 = clock.now()
    while clock.now() - t0 < wait_s:
        frame = cam.read(timeout=2.0)
        if frame is None:
            return None
        at = marker_at(frame.image)
        if at is not None:
            track.append((frame.t, at))
        if speed_px_s(track) >= MOVING_PX_S:
            break
    else:
        print(f"  the robot never got going in {wait_s:.0f} s -- start a clean")
        return None

    before = speed_px_s(track)
    done: dict = {}

    def send() -> None:
        t = clock.now()
        try:
            done["reply"] = link.stop()
        except (LinkError, OSError) as e:  # pragma: no cover - the robot's server
            done["reply"] = {"error": str(e)}
        done["sent"] = t
        done["replied"] = clock.now() - t

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    while "sent" not in done:  # the thread stamps it the moment it goes out
        time.sleep(0.001)
    t_stop = done["sent"]

    after: list[tuple[float, tuple[float, float]]] = []
    while clock.now() - t_stop < SETTLE_S + 8.0:
        frame = cam.read(timeout=2.0)
        if frame is None:
            break
        at = marker_at(frame.image)
        if at is not None and frame.t >= t_stop:
            after.append((frame.t, at))
        if after and after[-1][0] - t_stop > SETTLE_S and not worker.is_alive():
            break
    worker.join(5.0)
    if len(after) < 5:
        print("  too few frames after the stop to time it")
        return None

    rest = np.median(np.array([p for _, p in after[-10:]], np.float64), axis=0)
    last_moving = t_stop
    for t, p in after:
        if math.dist(p, (float(rest[0]), float(rest[1]))) > STILL_PX:
            last_moving = t
    return {
        "stop_s": round(last_moving - t_stop, 3),
        "reply_s": round(done.get("replied", 0.0), 3),
        "speed_px_s": round(before, 1),
        "moved_px": round(math.dist(after[0][1], (float(rest[0]), float(rest[1]))), 1),
        "frames_after": len(after),
        "reply": done.get("reply"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--wait", type=float, default=90.0, help="seconds to wait for it to drive")
    ap.add_argument("--drive", type=float, default=0.0,
                    help="clean a box this many mm across before each trial")  # fmt: skip
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    try:
        link = VacuumLink()
    except (LinkError, OSError) as e:
        print(f"could not start the robot's server: {e}")
        return 1
    try:
        cam = Camera(
            CameraSettings(index=a.camera, width=a.width, height=a.height, backend=a.backend)
        ).start()
    except (CameraUnavailable, ValueError) as e:
        print(f"No camera: {e}")
        link.close()
        return 1

    trials = []
    try:
        for i in range(a.trials):
            print(f"trial {i + 1} of {a.trials}: waiting for the robot to drive...", flush=True)
            if a.drive:
                if send_drive(link, a.drive) is None:
                    continue
            got = one_trial(cam, link, a.wait)
            if got is None:
                continue
            trials.append(got)
            print(f"  moving at {got['speed_px_s']:.0f} px/s; stopped {got['stop_s']:.2f} s "
                  f"after the order, {got['moved_px']:.0f} px further on "
                  f"(the robot answered in {got['reply_s']:.2f} s)", flush=True)  # fmt: skip
            if i + 1 < a.trials and not a.drive:
                print("  start another clean when you are ready", flush=True)
    finally:
        cam.close()
        link.close()

    if not trials:
        print("no trials completed")
        return 1
    times = [t["stop_s"] for t in trials]
    worst = max(times)
    ok = worst <= LIMIT_S
    verdict = "PASS" if ok else "FAIL"
    line = (f"vacuum stop: median {statistics.median(times):.2f} s, worst {worst:.2f} s "
            f"over {len(trials)} trials ({verdict}: limit {LIMIT_S:.0f} s each)")  # fmt: skip
    print(line)
    out = a.out or Path("out/vacuum_stop") / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stop_time.json").write_text(
        json.dumps({"trials": trials, "summary": line, "limit_s": LIMIT_S}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
