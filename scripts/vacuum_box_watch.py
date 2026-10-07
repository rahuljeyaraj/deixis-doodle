"""Watch the robot clean a box, with the box and the robot's own position drawn on it.

    uv run python scripts/vacuum_box_watch.py --camera 0 --side 1000

Sends the robot one box in the middle of the floor the camera knows, then draws:

- the **box** it was asked for, in white, on the floor;
- where the robot **says** it is, as a circle the robot's size, in green;
- the **lid marker**, in orange, which is what your eye follows.

The green circle and the orange marker will not sit on top of each other, and that is not an
error: the marker is 100 mm above the floor, so a camera looking across the room sees it
pushed away from the spot underneath itself. The green circle is the honest one -- it comes
from the robot's own map through the same fit the app uses.

Keys: Q or Esc quit (and stop the robot), S stop the robot, C clean the box again.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from deixis.frames import apply_matrix
from pointdraw.layout import ROBOT_MARKER_ID
from pointdraw.vacuum.calibrate import DEFAULT_MAP_PATH, FloorToMap
from pointdraw.vacuum.feet import FeetTracker
from pointdraw.vacuum.floor import DEFAULT_FIXED_PATH, FixedFloor, FloorView
from pointdraw.vacuum.link import LinkError, VacuumLink
from pointdraw.vision.aruco import detect_markers
from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable

WINDOW_W = 1280
WHITE, GREEN, ORANGE, GREY = (255, 255, 255), (60, 220, 60), (40, 140, 255), (150, 150, 150)
BLUE = (255, 180, 60)  # the same box, drawn up at the lid's height
AMBER = (40, 190, 255)  # where the app thinks you are standing
ROBOT_R_MM = 175.0  # the N50 is about 350 mm across


def poll(link: VacuumLink, out: queue.Queue, stop: threading.Event) -> None:
    n = 0
    while not stop.is_set():
        try:
            out.put(link.locate(wait_s=0.0) if n % 4 == 0 else link.status())
        except (LinkError, OSError) as e:  # pragma: no cover - the robot's server
            out.put({"error": str(e)})
        n += 1
        stop.wait(0.25)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--side", type=float, default=1000.0, help="the box's side, mm")
    ap.add_argument("--inset", type=float, default=0.0,
                    help="shrink the box SENT by this; the drawn one is as asked")  # fmt: skip
    ap.add_argument("--floor", type=Path, default=DEFAULT_FIXED_PATH)
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    a = ap.parse_args(argv)

    floor = FixedFloor.load_or_none(a.floor)
    if floor is None:
        print(f"no floor at {a.floor}: run scripts/vacuum_map.py first")
        return 1
    f2m = FloorToMap.load_or_none(a.map)
    if f2m is None:
        print(f"no floor-to-map at {a.map}")
        return 1
    to_map = np.array(f2m.transform.matrix, np.float64)
    to_floor = np.linalg.inv(to_map)
    to_px = np.linalg.inv(floor.H)  # for things on the floor: the box
    lid_H = floor.H if floor.H_lid is None else floor.H_lid
    to_px_lid = np.linalg.inv(lid_H)  # for the robot, which is seen by its lid
    if floor.H_lid is None:
        print('this floor fit has no lid plane: remap to put the circle on the robot')

    x, y, w, h = floor.area_mm
    cx, cy = x + w / 2, y + h / 2
    s = a.side
    corners_floor = np.array(
        [[cx - s / 2, cy - s / 2], [cx + s / 2, cy - s / 2],
         [cx + s / 2, cy + s / 2], [cx - s / 2, cy + s / 2]], np.float64)  # fmt: skip
    corners_map = apply_matrix(to_map, corners_floor)
    box_map = [
        float(corners_map[:, 0].min()), float(corners_map[:, 1].min()),
        float(corners_map[:, 0].max() - corners_map[:, 0].min()),
        float(corners_map[:, 1].max() - corners_map[:, 1].min()),
    ]  # fmt: skip
    # draw the box the ROBOT was given, not the one we started from: they differ by the
    # turn between the two frames, and the robot cleans an upright rectangle on its own map
    drawn_map = np.array(
        [[box_map[0], box_map[1]], [box_map[0] + box_map[2], box_map[1]],
         [box_map[0] + box_map[2], box_map[1] + box_map[3]], [box_map[0], box_map[1] + box_map[3]]],
        np.float64)  # fmt: skip
    drawn_floor = apply_matrix(to_floor, drawn_map)
    drawn_px = apply_matrix(to_px, drawn_floor)  # on the floor: what actually gets cleaned
    drawn_lid_px = apply_matrix(to_px_lid, drawn_floor)  # the same box, up at the lid
    # The robot is seen by its lid, so its path can only be compared with the blue box. The
    # white one is where the cleaning happens. They are the same box, 100 mm apart.

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

    def send() -> None:
        k = min(a.inset, box_map[2] / 3, box_map[3] / 3)
        sent = [box_map[0] + k, box_map[1] + k, box_map[2] - 2 * k, box_map[3] - 2 * k]
        reply = link.clean_area({"frame": "map", "kind": "box", "box_mm": sent})
        print(f"asked {box_map[2]:.0f} x {box_map[3]:.0f} mm, sent {sent[2]:.0f} x "
              f"{sent[3]:.0f} (in by {k:.0f}): {reply}", flush=True)  # fmt: skip

    tracker = FeetTracker()
    feet_view = FloorView(floor.H, (), True, None)  # feet are on the floor, like the box
    inbox: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=poll, args=(link, inbox, stop), daemon=True).start()
    send()
    state, robot_px, trail = None, None, []
    window = "the robot and the box"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                break
            while not inbox.empty():
                st = inbox.get_nowait()
                if "error" in st:
                    continue
                state = st.get("state")
                r = st.get("robot")
                if r:
                    p = apply_matrix(to_floor, np.array([[float(r["x"]), float(r["y"])]]))
                    robot_px = apply_matrix(to_px_lid, p)[0]  # where its lid will be seen
                    if not trail or np.linalg.norm(robot_px - trail[-1]) > 2:
                        trail.append(robot_px)
                        del trail[:-400]
            vis = frame.image.copy()
            cv2.polylines(vis, [np.round(drawn_px).astype(np.int32)], True, WHITE, 2)
            cv2.polylines(vis, [np.round(drawn_lid_px).astype(np.int32)], True, BLUE, 2)
            if len(trail) > 1:
                cv2.polylines(vis, [np.round(np.array(trail)).astype(np.int32)], False, GREY, 1)
            if robot_px is not None:
                # a circle the robot's size, measured on the floor where the robot is
                here = apply_matrix(lid_H, robot_px.reshape(1, 2))[0]
                rim = apply_matrix(
                    to_px_lid, np.array([[here[0] + ROBOT_R_MM, here[1]]])
                )[0]
                r_px = int(max(4, np.linalg.norm(rim - robot_px)))
                cv2.circle(vis, tuple(np.round(robot_px).astype(int)), r_px, GREEN, 2)
            feet = tracker.process(frame.image, frame.t, feet_view)
            for f in feet.feet_px:
                cv2.circle(vis, tuple(np.round(f).astype(int)), 5, WHITE, -1)
            if feet.spot_px is not None:
                cv2.circle(vis, tuple(np.round(feet.spot_px).astype(int)),
                           14 if feet.standing else 9, AMBER, -1)  # fmt: skip
            lid = next((m for m in detect_markers(frame.image, ids=None)
                        if m.id == ROBOT_MARKER_ID), None)  # fmt: skip
            if lid is not None:
                c = np.asarray(lid.corners_px, np.float64).reshape(4, 2).mean(axis=0)
                cv2.circle(vis, tuple(np.round(c).astype(int)), 8, ORANGE, -1)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 92), (30, 30, 30), -1)
            cv2.putText(vis, "white = the box on the floor (same mapping as your feet)   "
                        f"blue = it at lid height   green = the robot ({state})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            cv2.putText(vis, "amber = your feet.  Stand on a white corner: the dot should "
                        "land on it.", (20, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2)  # fmt: skip
            scale = WINDOW_W / vis.shape[1]
            cv2.imshow(window, cv2.resize(vis, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print("stop:", link.stop(), flush=True)
            if key == ord("c"):
                trail.clear()
                send()
    finally:
        stop.set()
        try:
            link.stop()
        finally:
            cv2.destroyAllWindows()
            cam.close()
            link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
