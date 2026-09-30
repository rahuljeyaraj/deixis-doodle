"""Camera helpers and replay sources, without a camera."""

import time

import cv2
import numpy as np
import pytest

from pointdraw.vision import camera as cam_mod
from pointdraw.vision.camera import (
    Camera,
    CameraSettings,
    CameraUnavailable,
    FpsCounter,
    ImageSource,
    VideoFileSource,
    apply_controls,
    fourcc_str,
    open_capture,
)


def _img(v):
    return np.full((48, 64, 3), v, np.uint8)


def test_image_source_timestamps_and_loop(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), _img(9))
    src = ImageSource([_img(1), path], fps=10.0, t0=100.0)
    f0, f1 = src.read(), src.read()
    assert (f0.t, f1.t) == (100.0, pytest.approx(100.1))
    assert f1.image[0, 0, 0] == 9 and f1.size == (64, 48)
    assert src.read() is None
    looped = ImageSource([np.zeros((4, 4), np.uint8)], loop=True, t0=0.0)
    frames = [looped.read() for _ in range(3)]
    assert [f.index for f in frames] == [0, 1, 2] and frames[0].image.shape == (4, 4, 3)


def test_video_file_source(tmp_path):
    path = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 20.0, (64, 48))
    for v in range(5):
        vw.write(_img(v * 40))
    vw.release()
    src = VideoFileSource(path, t0=5.0, loop=True)
    ts = [src.read().t for _ in range(7)]
    src.close()
    assert ts[:5] == pytest.approx([5.0, 5.05, 5.1, 5.15, 5.2])
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))  # keeps increasing on loop
    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "missing.avi")


def test_fps_counter_and_fourcc():
    c = FpsCounter(window_s=1.0)
    for i in range(31):
        c.tick(i / 30)
    assert c.fps == pytest.approx(30.0)
    assert fourcc_str(cv2.VideoWriter.fourcc(*"MJPG")) == "MJPG"
    assert fourcc_str(-1) == ""


class _FakeCap:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, opened=True, fourcc="MJPG", accept=()):
        self.opened, self.fourcc, self.accept, self.props = opened, fourcc, set(accept), {}
        self.n = 0

    def isOpened(self):  # noqa: N802 (OpenCV name)
        return self.opened

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOURCC:
            return True
        if prop in self.accept:
            self.props[prop] = value
            return True
        return False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FOURCC:
            return cv2.VideoWriter.fourcc(*self.fourcc) if self.fourcc else 0
        return self.props.get(prop, -1.0)

    def read(self):
        return True, _img(0)

    def grab(self):
        time.sleep(0.005)
        self.n += 1
        return True

    def retrieve(self):
        return True, _img(self.n % 255)

    def release(self):
        self.opened = False


def test_open_capture_falls_back_to_msmf_when_dshow_refuses_mjpg(monkeypatch):
    made = []

    def fake(index, backend):
        cap = _FakeCap(fourcc="YUY2" if backend == cv2.CAP_DSHOW else "")
        made.append(backend)
        return cap

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", fake)
    _, name = open_capture(CameraSettings())
    assert name == "msmf" and made == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    _, name = open_capture(CameraSettings(width=1280, height=720))  # 720p YUY2 is acceptable
    assert name == "dshow"


def test_no_camera_raises(monkeypatch):
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", lambda i, b: _FakeCap(opened=False))
    with pytest.raises(CameraUnavailable):
        Camera(CameraSettings(index=3)).start()


def test_apply_controls_reports_each_write():
    cap = _FakeCap(accept={cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_AUTO_WB})
    rep = apply_controls(cap, CameraSettings(exposure=-6, white_balance_k=4500, focus=0))
    assert rep["exposure"] == {"requested": -6.0, "accepted": True, "readback": -6.0}
    assert rep["wb_temperature"]["accepted"] is False and rep["focus"]["accepted"] is False
    assert apply_controls(cap, CameraSettings()) == {}


def test_threaded_camera_with_fake_capture(monkeypatch):
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", lambda i, b: _FakeCap())
    with Camera(CameraSettings(ring_size=4)) as cam:
        a = cam.read(timeout=1.0)
        b = cam.read(timeout=1.0)
        assert a is not None and b is not None and b.index > a.index and b.t >= a.t
        time.sleep(0.1)
        assert len(cam.recent()) == 4 and cam.latest() is not None and cam.fps > 0
