"""How fast does a stop act on the real machine? (M1 exit test: 0.5 s or less, 2 Oct 2026)

    uv run python scripts/stop_time.py --port COM9 --trials 5

The pen carriage (the orange holder) travels pen up along X, from one side of the bed to the
other. Once the camera sees it moving, the script waits a random 0.1 to 0.5 s and calls
`Controller.stop()` (M410, the emergency parser acts on it at once). The overhead camera
follows the orange holder; the stop has acted when it stops moving. Both clocks are
`pointdraw.clock`, so the time is camera frame minus the moment `stop()` was called.

A frame is stamped when the camera thread gets it, a little after the light hit the sensor,
so the time measured is an upper bound (the real stop is a little sooner). The frames come
about 33 ms apart, so each number is good to about one frame.

After a stop the controller homes before its next move (Marlin may have lost steps), so each
trial takes about 20 s. Keep hands off the bed. Writes `<out>/stop_time.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from pointdraw.clock import now
from pointdraw.controller import gcode
from pointdraw.controller.config import DEFAULT_CONFIG_PATH, load_config
from pointdraw.controller.controller import Controller
from pointdraw.controller.serial_link import SerialLink
from pointdraw.vision.camera import Camera, CameraSettings

ROWS = (700, 860)  # image rows the holder moves along at park Y (1920x1080, 2 Oct camera)
SETTLED_PX = 3.0  # within this of where it ends counts as still (about 1 mm; camera jitter 2 px)


def holder_x(img: np.ndarray) -> float | None:
    """Image x of the orange pen holder's centre, or None if it is not seen."""
    band = img[ROWS[0] : ROWS[1]]
    m = cv2.inRange(cv2.cvtColor(band, cv2.COLOR_BGR2HSV), (5, 120, 120), (22, 255, 255))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return float(cen[i][0]) if st[i, cv2.CC_STAT_AREA] > 2000 else None


def settled(ts: np.ndarray, xs: np.ndarray, t_stop: float) -> float:
    """When the holder came to rest for good: after it, every frame is within SETTLED_PX of
    where it ended. Also catches a restart: the end is then where the late move stopped."""
    end = float(np.median(xs[-5:]))
    off = np.abs(xs - end) > SETTLED_PX
    last = int(np.nonzero(off)[0][-1]) + 1 if off.any() else 0
    return float(max(ts[min(last, len(ts) - 1)], t_stop))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--x", type=float, nargs=2, default=(10.0, 210.0), help="travel from, to")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--out", type=Path, default=Path("out/stop_time"))
    ap.add_argument("-v", action="store_true", help="log every serial line, with the clock")
    a = ap.parse_args(argv)
    if a.v:
        logging.basicConfig(
            level=logging.DEBUG, format="%(t).3f %(name)s %(message)s", stream=sys.stdout
        )
        old = logging.getLogRecordFactory()

        def factory(*args, **kw):  # stamp each record on pointdraw.clock
            r = old(*args, **kw)
            r.t = now()
            return r

        logging.setLogRecordFactory(factory)
    cfg = load_config(a.config)
    y = cfg.park_y
    link = SerialLink(
        a.port,
        cfg.baud,
        window=cfg.window,
        boot_timeout=cfg.boot_timeout_s,
        ack_timeout=cfg.ack_timeout_s,
        servo=cfg.servo_enabled,
    )
    ctl = Controller(link, cfg, hands_clear=True)
    track: list[tuple[float, float | None]] = []
    lock = threading.Lock()
    results = []
    with Camera(CameraSettings(backend="msmf", width=1920, height=1080)) as cam:
        running = True

        def follow() -> None:
            last = -1
            while running:
                f = cam.read(timeout=1.0)
                if f is None or f.index == last:
                    continue
                last = f.index
                with lock:
                    track.append((f.t, holder_x(f.image)))

        th = threading.Thread(target=follow, daemon=True)
        th.start()
        ctl.start()
        try:
            for k in range(a.trials):
                start, end = (a.x[0], a.x[1]) if k % 2 == 0 else (a.x[1], a.x[0])
                ctl.run_lines(gcode.travel_gcode((start, y), cfg), "to start")
                ctl.wait()
                time.sleep(1.0)
                with lock:
                    track.clear()
                ctl.run_lines(gcode.travel_gcode((end, y), cfg, start=(start, y)), "trial")
                # wait until the camera sees the holder moving
                t_seen = None
                deadline = now() + 10
                while t_seen is None and now() < deadline:
                    time.sleep(0.01)
                    with lock:
                        xs = [x for _, x in track if x is not None]
                    if len(xs) >= 2 and abs(xs[-1] - xs[0]) > 20:
                        t_seen = now()
                if t_seen is None:
                    print(f"trial {k + 1}: the holder never moved on camera", file=sys.stderr)
                    ctl.stop()
                    ctl.wait()
                    continue
                time.sleep(random.uniform(0.1, 0.5))
                t_stop = now()
                print(f"{t_stop:.3f} STOP", flush=True)
                ctl.stop()
                t_call = now() - t_stop
                time.sleep(3.0)  # a line that waited at the stop would move about 1.2 s on
                ctl.wait()
                with lock:
                    tr = [(t, x) for t, x in track if x is not None]
                ts = np.array([t for t, _ in tr])
                xs = np.array([x for _, x in tr])
                v = np.abs(np.diff(xs))
                t_end = settled(ts, xs, t_stop)
                before = (ts < t_stop) & (ts > t_stop - 0.3)
                speed = (
                    float(np.median(v[before[1:]]) / np.median(np.diff(ts)))
                    if before[1:].sum() > 1
                    else float("nan")
                )
                after = xs[ts >= t_stop]
                travel_px = float(abs(after[-1] - after[0])) if len(after) else 0.0
                r = {
                    "trial": k + 1,
                    "stop_s": round(float(t_end - t_stop), 3),
                    "stop_call_s": round(t_call, 4),
                    "speed_px_s": round(speed, 1),
                    "after_stop_px": round(travel_px, 1),
                    "frames": len(tr),
                    "track": [[round(float(t - t_stop), 3), round(float(x), 1)] for t, x in tr],
                }
                results.append(r)
                print(
                    f"trial {k + 1}: moving until {r['stop_s']:.3f} s after stop(); "
                    f"{travel_px:.0f} px after the stop at {speed:.0f} px/s"
                )
        finally:
            running = False
            ctl.park()
            ctl.wait()
            ctl.close()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stop_time.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if results:
        s = [r["stop_s"] for r in results]
        print(f"stop: median {np.median(s):.3f} s, max {max(s):.3f} s (M1 limit 0.5 s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
