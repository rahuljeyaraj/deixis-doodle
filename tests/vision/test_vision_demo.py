"""scripts/vision_demo.py: exits cleanly without a camera; runs headless on a still image."""

import importlib.util
from pathlib import Path

import cv2
import pytest

from pointdraw.vision import synthetic as syn
from pointdraw.vision.camera import CameraUnavailable
from pointdraw.vision.hands import DEFAULT_MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location(
        "vision_demo", ROOT / "scripts" / "vision_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_camera_exits_cleanly(demo, monkeypatch, capsys):
    class NoCamera:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise CameraUnavailable("no camera at index 7")

    monkeypatch.setattr(demo, "Camera", NoCamera)
    assert demo.main(["--camera", "7", "--headless"]) == 1
    assert "No camera" in capsys.readouterr().out


@pytest.mark.skipif(
    not (ROOT / DEFAULT_MODEL_PATH).exists() and not DEFAULT_MODEL_PATH.exists(),
    reason="hand model not downloaded (uv run python scripts/vision_demo.py fetches it)",
)
def test_headless_on_synthetic_image(demo, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)  # the model path is relative to the repo
    scene, H = syn.default_scene()
    img_path = tmp_path / "bed.png"
    cv2.imwrite(str(img_path), syn.render(scene, H))
    crosses = tmp_path / "crosses.json"
    crosses.write_text(str(scene.crosses.tolist()))
    args = ["--image", str(img_path), "--headless", "--max-frames", "3"]
    args += ["--calib", str(tmp_path / "bed.json"), "--lens", str(tmp_path / "none.json")]
    assert demo.main(args) == 0
    out = capsys.readouterr().out
    assert "bed calibration: none" in out and "3 frames" in out
