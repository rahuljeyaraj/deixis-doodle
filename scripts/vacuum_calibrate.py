"""Line the floor camera up with the robot's own map (floor to map). Once per session.

    uv run python scripts/vacuum_calibrate.py --camera 1 --offset 90

1. Tape the robot marker (ID 14, `hardware/print/robot-marker.pdf`) on the robot's lid with
   its printed top toward the robot's front, in front of the LiDAR turret (never on it).
   Measure how far the marker's centre is ahead of the robot's centre: --offset, in mm.
2. Start this script, then start the robot cleaning in the ECOVACS app (or
   `vacuum.py zone ...`) so that it drives across the floor patch the camera sees.
3. The window draws the robot's track. Four times a second it asks the robot where it is
   on its map. When the track is over a metre across and the fit is tight, it says so:
   press F to save (.cache/vacuum/floor_to_map.json, or --out).

The robot's reports arrive a little late; the script tries lags of 0 to 3 s and keeps the
one that fits best (`PathMatcher`). The floor markers must be in view at the start (the
floor fit is then kept while the robot covers one).

Keys: F fit and save, C clear and start again, Q or Esc quit.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from deixis.mcp import DeixisClient
from pointdraw import clock
from pointdraw.layout import FLOOR_MARKER_IDS, ROBOT_MARKER_ID
from pointdraw.vacuum.calibrate import DEFAULT_MAP_PATH, FloorToMap, PathMatcher, RobotMarker
from pointdraw.vacuum.floor import DEFAULT_LAYOUT_PATH, FloorCamera, FloorLayout
from pointdraw.vacuum.link import LinkError, VacuumLink
from pointdraw.vacuum.profile import vacuum_profile
from pointdraw.vision.aruco import detect_markers
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable

WINDOW_W = 1280


def poll(link: VacuumLink, out: queue.Queue, stop: threading.Event, hz: float = 4.0) -> None:
    while not stop.is_set():
        try:
            out.put((clock.now(), link.status()))
        except LinkError as e:
            out.put((clock.now(), {"error": str(e)}))
        stop.wait(1 / hz)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--offset", type=float, default=0.0, help="marker ahead of centre, mm")
    ap.add_argument("--floor", type=Path, default=DEFAULT_LAYOUT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_MAP_PATH)
    ap.add_argument("--fake", action="store_true", help="a pretend robot (to try the window)")
    a = ap.parse_args(argv)
    try:
        layout = FloorLayout.load(a.floor)
    except FileNotFoundError:
        print(f"no floor layout at {a.floor}: run scripts/vacuum_floor.py first")
        return 1
    core = DeixisClient(vacuum_profile(layout.area_mm), fallback="direct")
    try:
        link = VacuumLink(fake=a.fake)
    except (LinkError, OSError) as e:
        print(f"could not start the robot's server: {e}")
        core.close()
        return 1
    try:
        cam = Camera(
            CameraSettings(index=a.camera, width=a.width, height=a.height, backend=a.backend)
        ).start()
    except (CameraUnavailable, ValueError) as e:
        print(f"No camera: {e}")
        link.close()
        core.close()
        return 1
    marker = RobotMarker(offset_mm=a.offset)
    floor = FloorCamera(layout)
    pm = PathMatcher()
    reports: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=poll, args=(link, reports, stop), daemon=True).start()
    result = None
    last_fit = 0.0
    window = "vacuum calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                print("the camera stopped")
                break
            t = frame.t
            found = detect_markers(frame.image, ids=None)
            view = floor.update([m for m in found if m.id in FLOOR_MARKER_IDS], t)
            lid = next((m for m in found if m.id == ROBOT_MARKER_ID), None)
            if view is not None and lid is not None:
                pm.see(t, marker.centre(view.to_mm(lid.corners_px)))
            while not reports.empty():
                tr, st = reports.get_nowait()
                if "error" in st:
                    print(f"robot: {st['error']}")
                else:
                    pm.report(tr, st)
            if time.monotonic() - last_fit > 3.0 and pm.spread_mm() > 500:
                last_fit = time.monotonic()
                result = pm.best(core)
                if result is not None:
                    lag, ps, rms = result
                    print(f"lag {lag:.1f} s, {len(ps)} pairs, {rms:.0f} mm rms, "
                          f"track {pm.spread_mm() / 1000:.1f} m across")  # fmt: skip
            vis = frame.image.copy()
            if view is not None and pm.track:
                pts = view.to_px(np.array([(x, y) for _, x, y in pm.track[-3000:]]))
                cv2.polylines(vis, [np.round(pts).astype(np.int32)], False, (255, 140, 40), 2)
            ready = result is not None and result[2] < 80 and pm.spread_mm() >= 1000
            text = ("Ready: press F to save" if ready else
                    "Floor markers not seen" if view is None else
                    "Robot marker not seen" if lid is None else
                    f"Following the robot: {len(pm.reports)} reports")  # fmt: skip
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 70), (30, 30, 30), -1)
            cv2.putText(vis, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (60, 220, 60) if ready else (255, 255, 255), 3)  # fmt: skip
            scale = WINDOW_W / vis.shape[1]
            cv2.imshow(window, cv2.resize(vis, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                pm, result = PathMatcher(), None
                print("cleared")
            if key == ord("f"):
                if result is None:
                    print("not enough yet: let the robot drive further across the patch")
                    continue
                lag, ps, rms = result
                f2m = FloorToMap.fit(ps[:: max(1, len(ps) // 60)], core)
                path = f2m.save(a.out)
                print(f"saved {path}: {len(f2m.pairs)} pairs, lag {lag:.1f} s, "
                      f"{f2m.transform.rms_mm:.0f} mm rms")  # fmt: skip
    finally:
        stop.set()
        cv2.destroyAllWindows()
        cam.close()
        link.close()
        core.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
