"""The small pictures a live run saves (`out/live/<time>/frames/`), used again offline.

`live_run.py` saves a 640x360 copy of the camera twice a second, turned 180 degrees (as the
user sees the bed), named by milliseconds since the run's first frame. `camera_image` turns
one back into the camera's own orientation and size, so the calibration (`BedView`) applies.
`SavedFrames` plays a folder of them as if it were the camera, for dry runs without one.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from pointdraw import clock
from pointdraw.vision.camera import Frame

CAMERA_SIZE = (1920, 1080)


def camera_image(path: str | Path, size: tuple[int, int] = CAMERA_SIZE) -> np.ndarray:
    """A saved picture, turned back and scaled to the camera's size."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(cv2.rotate(img, cv2.ROTATE_180), size, interpolation=cv2.INTER_LINEAR)


def frame_paths(run_dir: str | Path) -> list[Path]:
    """The run's saved pictures, in time order."""
    d = Path(run_dir)
    d = d / "frames" if (d / "frames").is_dir() else d
    return sorted(d.glob("*.jpg"))


def frame_ms(path: Path) -> int:
    """Milliseconds since the run's first frame (the file name)."""
    return int(path.stem)


class SavedFrames:
    """A `FrameSource` over saved pictures: each shown for its own gap, at `fps`, in real time.

    Frames are stamped with `clock.now()`, like the camera's. With `loop`, it starts again
    at the end; otherwise `read` returns None from then on.
    """

    def __init__(
        self,
        paths: list[Path],
        fps: float = 30.0,
        loop: bool = True,
        size: tuple[int, int] = CAMERA_SIZE,
    ) -> None:
        if not paths:
            raise ValueError("no saved pictures")
        self.paths = paths
        self.fps = fps
        self.loop = loop
        self.size = size
        ms = [frame_ms(p) for p in paths]
        self._at = [(m - ms[0]) / 1000.0 for m in ms]
        self._length = self._at[-1] + 0.5
        self._cache: dict[int, np.ndarray] = {}
        self._t0: float | None = None
        self._next = 0.0
        self._count = 0

    def _image(self, i: int) -> np.ndarray:
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[i] = camera_image(self.paths[i], self.size)
        return self._cache[i]

    def read(self, timeout: float | None = None) -> Frame | None:
        now = clock.now()
        if self._t0 is None:
            self._t0 = self._next = now
        if self._next > now:
            time.sleep(self._next - now)
            now = clock.now()
        self._next = max(self._next + 1.0 / self.fps, now)
        el = now - self._t0
        if el >= self._length:
            if not self.loop:
                return None
            el %= self._length
        i = max(0, int(np.searchsorted(self._at, el, side="right")) - 1)
        frame = Frame(self._image(i), now, self._count)
        self._count += 1
        return frame

    def close(self) -> None:
        self._cache.clear()
