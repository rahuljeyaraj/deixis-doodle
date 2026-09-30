"""scripts/calibrate.py and scripts/cross_under_finger.py with the simulator (no hardware)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pointdraw.vision.camera import CameraUnavailable
from pointdraw.vision.hands import DEFAULT_MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def calibrate():
    return load_script("calibrate")


@pytest.fixture(scope="module")
def cuf():
    return load_script("cross_under_finger")


@pytest.fixture
def simulated_calibration(calibrate, tmp_path):
    """Run calibrate.py end to end: simulated plot, synthetic photo, fit, save."""
    paths = {
        "config": tmp_path / "controller.json",
        "crosses": tmp_path / "crosses.json",
        "calib": tmp_path / "bed.json",
        "sim": tmp_path / "sim",
    }
    args = ["--sim", str(paths["sim"]), "--config", str(paths["config"])]
    args += ["--crosses", str(paths["crosses"]), "--calib", str(paths["calib"])]
    assert calibrate.main(args) == 0
    return paths


def test_calibrate_in_simulation(simulated_calibration, capsys):
    p = simulated_calibration
    calib = json.loads(p["calib"].read_text())
    assert len(calib["marker_corners_mm"]) == 6
    assert calib["cross_rms_mm"] < 0.3
    crosses = json.loads(p["crosses"].read_text())
    assert len(crosses["crosses"]) == 9 and crosses["simulated"] is True
    assert (p["sim"] / "plot.png").exists()
    assert list(p["sim"].glob("*-overlay.png"))


def test_calibrate_no_draw_needs_crosses(calibrate, tmp_path, capsys):
    args = ["--no-draw", "--crosses", str(tmp_path / "missing.json"), "--image", "x.png"]
    assert calibrate.main(args) == 1
    assert "--no-draw needs" in capsys.readouterr().out


def test_calibrate_refuses_a_grid_off_the_paper(calibrate, tmp_path, capsys):
    args = ["--sim", str(tmp_path / "sim"), "--config", str(tmp_path / "c.json")]
    args += ["--crosses", str(tmp_path / "x.json"), "--spacing", "120"]
    assert calibrate.main(args) == 2
    assert "refused" in capsys.readouterr().out
    assert not (tmp_path / "x.json").exists()


def test_cross_under_finger_needs_calibration(cuf, tmp_path, capsys):
    assert cuf.main(["--calib", str(tmp_path / "none.json"), "--sim", "x.png"]) == 1
    assert "run scripts/calibrate.py first" in capsys.readouterr().out


def test_cross_under_finger_without_camera(cuf, simulated_calibration, monkeypatch, capsys):
    class NoCamera:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise CameraUnavailable("no camera at index 7")

    monkeypatch.setattr(cuf, "Camera", NoCamera)
    args = ["--camera", "7", "--calib", str(simulated_calibration["calib"]), "--sim", "x.png"]
    assert cuf.main(args + ["--headless"]) == 1
    assert "No camera" in capsys.readouterr().out


@pytest.mark.skipif(
    not (ROOT / DEFAULT_MODEL_PATH).exists(),
    reason="hand model not downloaded (uv run python scripts/vision_demo.py fetches it)",
)
def test_cross_under_finger_headless_on_the_calibration_photo(
    cuf, simulated_calibration, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(ROOT)  # the hand model path is relative to the repo
    p = simulated_calibration
    photo = next(p["sim"].glob("*-photo.png"))
    png = tmp_path / "m1.png"
    args = ["--image", str(photo), "--calib", str(p["calib"]), "--config", str(p["config"])]
    args += ["--sim", str(png), "--headless", "--max-frames", "5"]
    args += ["--targets", str(ROOT / "hardware" / "print" / "m1-targets.json")]
    args += ["--log", str(tmp_path / "trials.jsonl")]
    assert cuf.main(args) == 0
    out = capsys.readouterr().out
    assert "ready: hold trigger, 15 targets" in out
    assert png.exists()  # the simulated plotter parked (no hands in view) and rendered
