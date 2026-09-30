"""Camera capture with timestamps on the shared clock, plus image and video sources for tests.

Every frame is stamped with `pointdraw.clock.now()` right after `grab()` returns, so it is
comparable with audio and plotter events. The fixed capture-to-grab latency is not removed
here; the clap test (plan W19) measures the audio-video offset as a whole.

Backend notes (Windows 11, OpenCV 5.0, measured 30 Sep 2026 on the Lenovo 300 FHD webcam,
USB 17EF:4831; full table in docs/sessions/vis-report.md):

- **DSHOW** (`CAP_DSHOW`): opens in ~3 s. The Lenovo never grants MJPG (FOURCC reads YUY2
  whether it is set before or after the size), so over USB 2.0 it runs 1080p at 4 to 5 fps
  and 720p at 10 fps: unusable for hands. `CAP_PROP_EXPOSURE` (log2 seconds, -6 = 1/64 s) is
  accepted and read back, but made no visible difference. `CAP_PROP_AUTO_EXPOSURE` reads -1
  (not implemented). `CAP_PROP_AUTO_WB` 0/1 works; `CAP_PROP_WB_TEMPERATURE`, gain, focus
  and autofocus are refused (fixed-focus lens). `CAP_PROP_SETTINGS` opens the driver's
  property dialog, the manual fallback for locking settings.
- **MSMF** (`CAP_MSMF`): opens in 5 to 9 s but delivers 1080p at 30 fps. `CAP_PROP_EXPOSURE`
  writes return True and clearly change the image, but read-back stays stale (-4).
  White balance, gain, focus and autofocus writes are refused.

`backend="auto"` tries DSHOW with MJPG first (other webcams, such as a C920, do grant it); if
MJPG is not granted above 720p it falls back to MSMF, which is what happens on the Lenovo.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pointdraw import clock

log = logging.getLogger(__name__)


class CameraUnavailable(RuntimeError):
    """No camera could be opened (or it stopped delivering frames)."""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8, H x W x 3
    t: float  # seconds on pointdraw.clock
    index: int  # frame counter from this source

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self.image.shape[1], self.image.shape[0]


class FrameSource(Protocol):
    """Anything that yields timestamped frames: a live camera, an image list, a video file."""

    def read(self, timeout: float | None = None) -> Frame | None: ...

    def close(self) -> None: ...


class FpsCounter:
    """Frames per second over a sliding window of timestamps."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._ts: deque[float] = deque()

    def tick(self, t: float) -> None:
        self._ts.append(t)
        while self._ts and self._ts[0] < t - self.window_s:
            self._ts.popleft()

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


# --- live camera -------------------------------------------------------------------------------


@dataclass
class CameraSettings:
    index: int = 0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    backend: str = "auto"  # "auto", "dshow", "msmf" or "any"
    exposure: float | None = None  # backend units; DSHOW/MSMF: log2 seconds, e.g. -6
    white_balance_k: int | None = None  # e.g. 4500; also turns auto white balance off
    focus: int | None = None  # e.g. 0 to 255; also turns autofocus off (not on fixed focus)
    ring_size: int = 8


_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


def fourcc_str(value: float) -> str:
    v = int(value)
    if v <= 0:
        return ""
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def open_capture(s: CameraSettings) -> tuple[cv2.VideoCapture, str]:
    """Open the camera with the requested backend and size. Raises CameraUnavailable."""
    order = ["dshow", "msmf"] if s.backend == "auto" else [s.backend]
    for name in order:
        if name not in _BACKENDS:
            raise ValueError(f"unknown backend {name!r}")
        cap = cv2.VideoCapture(s.index, _BACKENDS[name])
        if not cap.isOpened():
            cap.release()
            log.info("camera %d: %s would not open", s.index, name)
            continue
        if name == "dshow":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        cap.set(cv2.CAP_PROP_FPS, s.fps)
        fourcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        big = s.width * s.height > 1280 * 720
        if s.backend == "auto" and name == "dshow" and big and fourcc != "MJPG":
            log.warning(
                "camera %d: DSHOW gave %s, not MJPG, at %dx%d (too slow); trying MSMF",
                s.index,
                fourcc or "?",
                s.width,
                s.height,
            )
            cap.release()
            continue
        ok, _ = cap.read()
        if not ok:
            cap.release()
            log.info("camera %d: %s opened but gave no frame", s.index, name)
            continue
        return cap, name
    raise CameraUnavailable(f"no camera at index {s.index} (tried {', '.join(order)})")


def apply_controls(cap: cv2.VideoCapture, s: CameraSettings) -> dict[str, dict[str, object]]:
    """Try to lock exposure, white balance and focus. Returns what each write did.

    Each entry is {"requested", "accepted", "readback"}; see the module docstring for what
    the values mean on each backend. Nothing here is guaranteed: read the report.
    """
    writes: list[tuple[str, int, float]] = []
    if s.exposure is not None:
        writes += [
            ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),  # V4L-style "manual"
            ("exposure", cv2.CAP_PROP_EXPOSURE, float(s.exposure)),
        ]
    if s.white_balance_k is not None:
        writes += [
            ("auto_wb", cv2.CAP_PROP_AUTO_WB, 0.0),
            ("wb_temperature", cv2.CAP_PROP_WB_TEMPERATURE, float(s.white_balance_k)),
        ]
    if s.focus is not None:
        writes += [
            ("autofocus", cv2.CAP_PROP_AUTOFOCUS, 0.0),
            ("focus", cv2.CAP_PROP_FOCUS, float(s.focus)),
        ]
    report: dict[str, dict[str, object]] = {}
    for name, prop, value in writes:
        accepted = bool(cap.set(prop, value))
        report[name] = {"requested": value, "accepted": accepted, "readback": cap.get(prop)}
    return report


class Camera:
    """Threaded live capture. Use as a context manager, or call start() and close().

    A background thread grabs frames as fast as the camera delivers them and keeps the last
    `ring_size` in a ring buffer, so a slow consumer always gets the newest frame.
    """

    def __init__(self, settings: CameraSettings | None = None) -> None:
        self.settings = settings or CameraSettings()
        self.backend: str | None = None
        self.controls: dict[str, dict[str, object]] = {}
        self._cap: cv2.VideoCapture | None = None
        self._ring: deque[Frame] = deque(maxlen=self.settings.ring_size)
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self._count = 0
        self._last_read = -1
        self._error: str | None = None
        self._fps = FpsCounter()

    def __enter__(self) -> Camera:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> Camera:
        self._cap, self.backend = open_capture(self.settings)
        self.controls = apply_controls(self._cap, self.settings)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("camera %d open: %s %dx%d", self.settings.index, self.backend, w, h)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        failures = 0
        while self._running:
            if not self._cap.grab():
                failures += 1
                if failures > 50:
                    self._error = "camera stopped delivering frames"
                    log.error(self._error)
                    break
                time.sleep(0.01)
                continue
            t = clock.now()
            ok, img = self._cap.retrieve()
            if not ok or img is None:
                failures += 1
                continue
            failures = 0
            with self._cond:
                self._ring.append(Frame(img, t, self._count))
                self._count += 1
                self._fps.tick(t)
                self._cond.notify_all()
        with self._cond:
            self._running = False
            self._cond.notify_all()

    @property
    def fps(self) -> float:
        with self._cond:
            return self._fps.fps

    def latest(self) -> Frame | None:
        with self._cond:
            return self._ring[-1] if self._ring else None

    def recent(self, n: int | None = None) -> list[Frame]:
        """Up to n most recent frames, oldest first."""
        with self._cond:
            frames = list(self._ring)
        return frames if n is None else frames[-n:]

    def read(self, timeout: float | None = 2.0) -> Frame | None:
        """Wait for a frame newer than the last one returned. None on timeout or stop."""
        with self._cond:
            self._cond.wait_for(
                lambda: (
                    (bool(self._ring) and self._ring[-1].index > self._last_read)
                    or not self._running
                ),
                timeout=timeout,
            )
            if not self._ring or self._ring[-1].index <= self._last_read:
                if self._error:
                    raise CameraUnavailable(self._error)
                return None
            frame = self._ring[-1]
            self._last_read = frame.index
            return frame

    def show_settings_dialog(self) -> bool:
        """Open the driver's property dialog (DSHOW only). Useful to lock settings by hand."""
        return bool(self._cap is not None and self._cap.set(cv2.CAP_PROP_SETTINGS, 1))

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# --- replay sources ----------------------------------------------------------------------------


def _load(img: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if out is None:
        raise FileNotFoundError(img)
    return out


class ImageSource:
    """Frames from a list of images or paths, with synthetic timestamps t0 + i / fps.

    `loop=True` repeats forever (handy for a demo on one still image).
    """

    def __init__(
        self,
        images: Iterable[np.ndarray | str | Path],
        fps: float = 30.0,
        t0: float | None = None,
        loop: bool = False,
    ) -> None:
        self._images = [_load(i) for i in images]
        self.fps_nominal = fps
        self.t0 = clock.now() if t0 is None else t0
        self.loop = loop
        self._i = 0

    def read(self, timeout: float | None = None) -> Frame | None:
        if not self._images or (not self.loop and self._i >= len(self._images)):
            return None
        img = self._images[self._i % len(self._images)]
        frame = Frame(img, self.t0 + self._i / self.fps_nominal, self._i)
        self._i += 1
        return frame

    def close(self) -> None:
        pass


class VideoFileSource:
    """Frames from a video file. Timestamps are t0 plus the file's own frame time.

    With `realtime=True`, `read()` sleeps so frames come out at the recorded pace.
    """

    def __init__(
        self, path: str | Path, realtime: bool = False, t0: float | None = None, loop: bool = False
    ) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(self.path)
        self.fps_nominal = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.loop = loop
        self.t0 = clock.now() if t0 is None else t0
        self._i = 0
        self._offset = 0.0  # added on each loop so time keeps increasing
        self._last_media_t = 0.0

    def read(self, timeout: float | None = None) -> Frame | None:
        ok, img = self._cap.read()
        if not ok and self.loop and self._i > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._offset += self._last_media_t + 1.0 / self.fps_nominal
            ok, img = self._cap.read()
        if not ok:
            return None
        # Frame time from the index: POS_MSEC is unreliable across containers and backends.
        media_t = (int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1) / self.fps_nominal
        self._last_media_t = media_t
        t = self.t0 + self._offset + media_t
        if self.realtime:
            delay = t - clock.now()
            if delay > 0:
                time.sleep(delay)
        frame = Frame(img, t, self._i)
        self._i += 1
        return frame

    def close(self) -> None:
        self._cap.release()
