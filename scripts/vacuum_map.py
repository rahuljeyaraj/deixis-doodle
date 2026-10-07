"""Map the floor from the robot's own drive. No paper markers, no tape measure.

    uv run python scripts/vacuum_map.py --camera 0

1. Tape the robot marker (ID 14, `hardware/print/robot-marker.pdf`) on the robot's lid with
   its printed top toward the robot's front, in front of the LiDAR turret (never on it).
   How far the marker's centre sits ahead of the robot's centre is worked out from the
   drive itself; give it with --offset <mm> only if you would rather say.
2. Start this script, then start the robot cleaning in the ECOVACS app (or
   `vacuum.py zone ...`) so that it drives around the floor the camera sees. The more of
   the patch it covers, and the more it turns, the better the fit.
3. The window draws where the camera has seen the robot. Four times a second it asks the
   robot where it is on its own map. When the track is over a metre across and the fit is
   tight, it says so: press F to save, then Q.

It writes both files the app needs, so `vacuum_floor.py` and `vacuum_calibrate.py` are not
run at all:
  .cache/vacuum/floor_fixed.json   pixels -> floor mm, and the patch the robot covered
  .cache/vacuum/floor_to_map.json  floor mm -> the robot's map

**The camera must not move after this.** With markers on the floor the app re-finds them in
every frame, so a knock fixes itself; with this it cannot. If the camera is ever moved,
run this again.

The lid marker rides about 100 mm above the floor, so the fit reads the lid's height, not
the floor's. The script works out how high the camera is and corrects for it (--lid-height,
--camera-height to override). It prints the camera height it worked out: if that is far from
what you can see with your eyes, pass --camera-height and run again.

Keys: F fit and save, C clear and start again, Q or Esc quit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from deixis.frames import apply_matrix
from deixis.mcp import DeixisClient
from pointdraw.layout import ROBOT_MARKER_ID
from pointdraw.vacuum.calibrate import DEFAULT_MAP_PATH, FloorToMap
from pointdraw.vacuum.floor import DEFAULT_FIXED_PATH, FixedFloor
from pointdraw.vacuum.link import LinkError, VacuumLink
from pointdraw.vacuum.mapfit import (
    LID_HEIGHT_MM,
    best_lag,
    best_offset,
    camera_pose,
    floor_frame,
    lift_to_floor,
    readiness,
)
from pointdraw.vacuum.profile import vacuum_profile
from pointdraw.vision.aruco import detect_markers
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable

WINDOW_W = 1280
ENOUGH_SPREAD_MM = 1000.0


def poll(link: VacuumLink, out: queue.Queue, stop: threading.Event) -> None:
    """Ask the robot where it is, four times a second.

    A parked robot does not stream its position, so `status` alone can sit at nothing for
    ever. Every fourth turn asks for a fresh one (`locate`, which pokes the robot and then
    reads the status back) -- that is what makes a slow or stop-start drive work.
    """
    n = 0
    while not stop.is_set():
        try:
            st = link.locate(wait_s=0.0) if n % 4 == 0 else link.status()
            out.put((time.monotonic(), st))
        except (LinkError, OSError) as e:  # pragma: no cover - the robot's server
            out.put((time.monotonic(), {"error": str(e)}))
        n += 1
        stop.wait(0.25)


def _spread_mm(track: list[tuple[float, np.ndarray]]) -> float:
    """How far apart the camera's sightings are, in pixels' worth of map -- here, pixels."""
    if len(track) < 2:
        return 0.0
    pts = np.array([c.reshape(4, 2).mean(axis=0) for _, c in track])
    return float(np.max(pts.max(axis=0) - pts.min(axis=0)))




def send_drive(link: VacuumLink, side_mm: float) -> bool:
    """Set the robot cleaning a box around where it stands, so that it drives about."""
    st = link.locate(wait_s=1.5)
    robot = st.get("robot")
    if not robot:
        print("  the robot will not say where it is")
        return False
    x, y = float(robot["x"]), float(robot["y"])
    where = {"frame": "map", "kind": "box",
             "box_mm": [x - side_mm / 2, y - side_mm / 2, side_mm, side_mm]}  # fmt: skip
    reply = link.clean_area(where)
    ok = bool(reply.get("ok"))
    print(f"sent it to clean {side_mm / 1000:.1f} m around ({x:.0f}, {y:.0f}): "
          f"{'ok' if ok else reply}", flush=True)  # fmt: skip
    return ok

def _fit(track, reports, offset: float | None):
    """(lag, sightings, pixels -> map, rms, the marker's offset). The offset is scanned for
    unless it was given: a wrong one pushes the marker's map point out by that much."""
    if offset is not None:
        got = best_lag(track, reports, offset)
        return None if got is None else (*got, offset)
    got = best_offset(track, reports)
    return None if got is None else (got[1], got[2], got[3], got[4], got[0])

def _refit(track, reports, offset: float | None, out: list) -> None:
    """Fit in the background. On the window's thread it stalls the picture every few
    seconds, which reads as a stuttering camera."""
    got = _fit(track, reports, offset)
    if got is not None:
        out[0] = got
        lag, seen, _, rms, off = got
        print(f"lag {lag:.1f} s, {len(seen)} pairs, {rms:.0f} mm rms, "
              f"marker {off:.0f} mm ahead")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--offset", default="auto",
                    help="marker ahead of the robot's centre, mm, or 'auto'")  # fmt: skip
    ap.add_argument("--lid-height", type=float, default=LID_HEIGHT_MM, help="marker height, mm")
    ap.add_argument("--camera-height", type=float, default=None, help="above the floor, mm")
    ap.add_argument("--focal", type=float, default=None, help="the camera's focal length, px")
    ap.add_argument("--out", type=Path, default=DEFAULT_FIXED_PATH)
    ap.add_argument("--map-out", type=Path, default=DEFAULT_MAP_PATH)
    ap.add_argument("--drive", type=float, default=0.0,
                    help="clean a box this many mm across, to make it drive")  # fmt: skip
    ap.add_argument("--auto", action="store_true", help="save and quit without a keypress")
    ap.add_argument("--want", type=int, default=150,
                    help="--auto: keep driving until this many positions")  # fmt: skip
    ap.add_argument("--fake", action="store_true", help="a pretend robot (to try the window)")
    a = ap.parse_args(argv)

    core = DeixisClient(vacuum_profile(), fallback="direct")
    try:
        link = VacuumLink(fake=a.fake)
    except (LinkError, OSError) as e:
        print(f"could not start the robot's server: {e}")
        print("(log in first: see ../deixis-vacuum/README.md; or use --fake)")
        core.close()
        return 1
    try:
        cam = Camera(
            CameraSettings(index=a.camera, width=a.width, height=a.height, backend=a.backend)
        ).start()
    except (CameraUnavailable, ValueError) as e:
        print(f"No camera: {e}. Try --camera 1 or --backend dshow.")
        link.close()
        core.close()
        return 1

    track: list[tuple[float, np.ndarray]] = []  # t, the lid marker's 4 corners in pixels
    reports: list[tuple[float, tuple[float, float]]] = []  # t, the robot's map mm
    inbox: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=poll, args=(link, inbox, stop), daemon=True).start()
    size = (a.width, a.height)
    offset = None if str(a.offset).lower() == "auto" else float(a.offset)
    found: list = [None]  # the newest fit, handed over by the worker thread
    fitter: threading.Thread | None = None
    result = None
    last_fit = 0.0
    window = "floor from the robot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if a.drive and not send_drive(link, a.drive):
        cv2.destroyAllWindows()
        cam.close()
        link.close()
        core.close()
        return 1
    last_drive = time.monotonic()
    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                print("the camera stopped")
                break
            lid = next((m for m in detect_markers(frame.image, ids=None)
                        if m.id == ROBOT_MARKER_ID), None)  # fmt: skip
            if lid is not None:
                track.append((frame.t, np.asarray(lid.corners_px, np.float64).reshape(4, 2)))
            while not inbox.empty():
                tr, st = inbox.get_nowait()
                if "error" in st:
                    print(f"robot: {st['error']}")
                    continue
                robot = st.get("robot")
                if robot:
                    t = tr - float(robot.get("age_s", 0.0))
                    rec = (t, (float(robot["x"]), float(robot["y"])))
                    if not reports or reports[-1][1] != rec[1]:
                        reports.append(rec)
            busy = fitter is not None and fitter.is_alive()
            if not busy and time.monotonic() - last_fit > 3.0 and _spread_mm(track) > 100:
                last_fit = time.monotonic()
                fitter = threading.Thread(
                    target=_refit, args=(list(track), list(reports), offset, found), daemon=True
                )
                fitter.start()
            result = found[0]

            vis = frame.image.copy()
            if len(track) > 1:
                pts = np.array([c.mean(axis=0) for _, c in track[-3000:]])
                cv2.polylines(vis, [np.round(pts).astype(np.int32)], False, (255, 140, 40), 2)
            ready, why = (readiness(result[1], result[3]) if result else (False, ""))
            if a.auto and ready and result is not None and len(result[1]) < a.want:
                why = f"Collecting: {len(result[1])} of {a.want} positions, {result[3]:.0f} mm"
            text = (why if ready else
                    "Robot marker not seen" if lid is None else
                    "Start a clean: the robot is not reporting" if not reports else
                    why or f"Following the robot: {len(reports)} reports")  # fmt: skip
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 70), (30, 30, 30), -1)
            cv2.putText(vis, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (60, 220, 60) if ready else (255, 255, 255), 3)  # fmt: skip
            scale = WINDOW_W / vis.shape[1]
            cv2.imshow(window, cv2.resize(vis, None, fx=scale, fy=scale))
            if a.drive and not ready and time.monotonic() - last_drive > 45.0:
                last_drive = time.monotonic()  # it will have finished the box by now
                send_drive(link, a.drive)
            if a.auto and ready and len(result[1]) >= a.want:
                print(why, flush=True)
                if save(result, size, a, core) == 0:
                    link.stop()
                    break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                track, reports, result, found[0] = [], [], None, None
                print("cleared")
            if key == ord("f"):
                if result is None or not ready:
                    print(why or "not enough yet: let the robot drive across the patch")
                    continue
                if save(result, size, a, core) == 0:
                    print("done: you can press Q")
    finally:
        stop.set()
        cv2.destroyAllWindows()
        cam.close()
        link.close()
        core.close()
    return 0


def save(result, size: tuple[int, int], a: argparse.Namespace, core: DeixisClient) -> int:
    """Work the fit up into the two files the app loads, and write them."""
    lag, seen, H_lid, rms, found_offset = result
    pose = camera_pose(H_lid, size, focal_px=a.focal)
    if pose is None:
        print("the fit will not give a camera position: let the robot drive further")
        return 1
    above_lid, nadir = pose
    height = a.camera_height if a.camera_height else above_lid + a.lid_height
    print(f"camera about {height / 1000:.2f} m above the floor"
          f"{'' if a.camera_height else ' (worked out from the fit)'}; "
          f"the spot under it is at map ({nadir[0]:.0f}, {nadir[1]:.0f}) mm")  # fmt: skip
    if height <= a.lid_height:
        print("that cannot be right: pass --camera-height (mm above the floor)")
        return 1
    H_floor_map = lift_to_floor(H_lid, nadir, height, a.lid_height)
    covered = np.array([s.map_mm for s in seen], np.float64)
    R, area = floor_frame(H_floor_map, size, covered)
    fixed = FixedFloor(
        R @ H_floor_map,
        area,
        dt.datetime.now().isoformat(timespec="seconds"),
        {
            "from": "robot drive",
            "pairs": len(seen),
            "lag_s": round(lag, 2),
            "rms_mm": round(rms, 1),
            "camera_height_mm": round(height),
            "camera_height_measured": a.camera_height is not None,
            "lid_height_mm": a.lid_height,
            "marker_offset_mm": round(found_offset, 1),
            "nadir_map_mm": [round(nadir[0]), round(nadir[1])],
        },
        R @ H_lid,  # the same frame, but for things at the lid's height: the robot itself
    )
    path = fixed.save(a.out)
    floor_pts = apply_matrix(R, covered)
    step = max(1, len(covered) // 60)
    f2m = FloorToMap.fit(
        [
            ((float(f[0]), float(f[1])), (float(m[0]), float(m[1])))
            for f, m in zip(floor_pts[::step], covered[::step], strict=True)
        ],  # fmt: skip
        core,
    )
    map_path = f2m.save(a.map_out)
    print(f"saved {path}: the patch is {area[2] / 1000:.2f} m wide and {area[3] / 1000:.2f} m "
          f"deep, from {len(seen)} pairs at {rms:.0f} mm rms (lag {lag:.1f} s, the marker "
          f"{found_offset:.0f} mm ahead of the robot's centre)")  # fmt: skip
    print(f"saved {map_path}: floor to map, {f2m.transform.rms_mm:.0f} mm rms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
