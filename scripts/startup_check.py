"""Repeat each start-up step of `live_run.py` in fresh processes, to find a rare crash.

    uv run python scripts/startup_check.py --runs 30                  # every step, 30 times each
    uv run python scripts/startup_check.py --runs 30 --steps camera hands
    uv run python scripts/startup_check.py --step camera              # one step, once, here
    uv run python scripts/startup_check.py --runs 30 --steps camera reopen full

To try a camera setting in the child processes, add for example
`--env OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` (MSMF without hardware transforms).

On 1 Oct 2026 `live_run.py` crashed once in six starts with a segmentation fault, before it
printed anything: while opening the camera or creating the hand tracker. Each run here is a
new `python -X faulthandler` process (a crash kills the process), started right after the
last one ended, as when the app is quit and started again. A crash is a negative or
0xC0000005 exit code; faulthandler's stack goes to `<out>/<step>-<n>.txt`.

Steps (the live app's order is `full`):

- `camera`: open the camera (MSMF 1920x1080) and read a frame, then close it.
- `reopen`: open, read, close, and open again at once (a camera reopened too soon).
- `hands`: create `HandTracker()` and track one blank frame.
- `hands-busy`: the hand tracker created and run while another thread does OpenCV work on
  1080p frames, as the camera thread did while the live app created it.
- `asr`: load the local speech recogniser.
- `full`: as `live_run.py` started on 1 Oct: thread settings, camera and a frame, hand
  tracker on that frame, then the speech recogniser. (Since 2 Oct it makes the tracker first.)
- `full-nocam`: `full` with a saved frame (`--frame`) and a busy OpenCV thread in place of
  the camera.

Opens the camera only; nothing moves. Results: `<out>/summary.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = ("camera", "reopen", "hands", "hands-busy", "asr", "full", "full-nocam")
NO_CAMERA = ("hands", "hands-busy", "asr", "full-nocam")
ROOT = Path(__file__).resolve().parents[1]


def _camera(index: int, backend: str):
    from pointdraw.vision.camera import Camera, CameraSettings

    cam = Camera(CameraSettings(index=index, width=1920, height=1080, backend=backend)).start()
    frame = cam.read(timeout=5.0)
    if frame is None:
        cam.close()
        raise RuntimeError("the camera gave no picture")
    return cam, frame


class _Busy:
    """A thread doing OpenCV work on 1080p frames, like the camera's grab thread."""

    def __init__(self) -> None:
        import threading

        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import cv2
        import numpy as np

        img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), np.uint8)
        while not self.stop.is_set():
            cv2.cvtColor(cv2.resize(img, (960, 540)), cv2.COLOR_BGR2RGB)

    def __enter__(self) -> _Busy:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join(2)


def run_step(step: str, index: int, backend: str, frame_path: Path | None = None) -> None:
    import numpy as np

    if step in ("camera", "reopen"):
        cam, _ = _camera(index, backend)
        cam.close()
        if step == "reopen":
            cam, _ = _camera(index, backend)
            cam.close()
    elif step == "hands":
        from pointdraw.vision.hands import HandTracker

        hands = HandTracker()
        hands.process(np.full((1080, 1920, 3), 255, np.uint8), 0.0)
        hands.close()
    elif step == "hands-busy":
        from pointdraw.vision.hands import HandTracker

        with _Busy():
            hands = HandTracker()
            for i in range(5):
                hands.process(np.full((1080, 1920, 3), 255, np.uint8), i / 30)
            hands.close()
    elif step == "full-nocam":
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        import cv2

        cv2.setNumThreads(4)
        img = np.full((1080, 1920, 3), 255, np.uint8)
        if frame_path is not None:
            img = cv2.resize(cv2.rotate(cv2.imread(str(frame_path)), cv2.ROTATE_180), (1920, 1080))
        with _Busy():
            from pointdraw.vision.hands import HandTracker

            hands = HandTracker()
            hands.process(img, 0.0)
            from pointdraw.audio.asr_local import AsrConfig, LocalAsr
            from pointdraw.audio.models import DEFAULT_ASR

            LocalAsr(AsrConfig(model=DEFAULT_ASR))
            hands.close()
    elif step == "asr":
        from pointdraw.audio.asr_local import AsrConfig, LocalAsr
        from pointdraw.audio.models import DEFAULT_ASR

        LocalAsr(AsrConfig(model=DEFAULT_ASR))
    elif step == "full":
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        import cv2

        cv2.setNumThreads(4)
        cam, frame = _camera(index, backend)
        from pointdraw.vision.hands import HandTracker

        hands = HandTracker()
        hands.process(frame.image, frame.t)
        from pointdraw.audio.asr_local import AsrConfig, LocalAsr
        from pointdraw.audio.models import DEFAULT_ASR

        LocalAsr(AsrConfig(model=DEFAULT_ASR))
        hands.close()
        cam.close()
    else:
        raise SystemExit(f"unknown step {step}")


def crashed(code: int) -> bool:
    return code < 0 or code in (0xC0000005, 0xC0000409, 0xC0000374, 3)


def drive(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for step in args.steps:
        rows = []
        for n in range(args.runs):
            cmd = [sys.executable, "-X", "faulthandler", __file__, "--step", step]
            cmd += ["--camera", str(args.camera), "--backend", args.backend]
            if args.frame is not None:
                cmd += ["--frame", str(args.frame)]
            env = dict(os.environ, **dict(kv.split("=", 1) for kv in args.env))
            t0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=180, env=env)
            dt = time.perf_counter() - t0
            code = p.returncode & 0xFFFFFFFF if p.returncode < 0 else p.returncode
            bad = crashed(p.returncode) or crashed(code)
            rows.append({"n": n, "code": p.returncode, "s": round(dt, 2), "crash": bad})
            if p.returncode != 0:
                (out / f"{step}-{n:02d}.txt").write_text(
                    f"exit {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}",
                    encoding="utf-8",
                )
            mark = "CRASH" if bad else ("fail" if p.returncode else "ok")
            print(f"{step:7} {n:2d} {mark:5} exit {p.returncode} in {dt:5.1f} s", flush=True)
        summary[step] = {
            "env": args.env,
            "runs": len(rows),
            "crashes": sum(r["crash"] for r in rows),
            "failures": sum(1 for r in rows if r["code"] != 0),
            "median_s": sorted(r["s"] for r in rows)[len(rows) // 2],
            "rows": rows,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for step, s in summary.items():
        print(f"{step:7} {s['crashes']} crashes, {s['failures']} failures in {s['runs']} runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--step", choices=STEPS, help="run one step here, once")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="msmf", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE for the child processes")
    ap.add_argument("--frame", type=Path, default=None, help="a saved picture for full-nocam")
    ap.add_argument("--out", type=Path, default=Path("out/night/startup"))
    a = ap.parse_args(argv)
    if a.step:
        run_step(a.step, a.camera, a.backend, a.frame)
        print("done", flush=True)
        return 0
    return drive(a)


if __name__ == "__main__":
    sys.exit(main())
