"""scripts/plot.py against the simulator (no hardware)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def plot():
    spec = importlib.util.spec_from_file_location("plot_cli", ROOT / "scripts" / "plot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg_path(tmp_path):
    return str(tmp_path / "controller.json")  # no override: defaults


def test_dry_run_prints_gcode(plot, cfg_path, capsys):
    assert plot.main(["--config", cfg_path, "--dry-run", "square", "--size", "20"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["G21", "G90", "M211 S1"]
    assert "G1 Z0 F300" in out
    assert not any(ln.startswith(("M104", "M140", "M106")) for ln in out)


def test_sim_renders_png(plot, cfg_path, tmp_path):
    png = tmp_path / "circle.png"
    assert plot.main(["--config", cfg_path, "--sim", str(png), "--home", "circle"]) == 0
    assert png.stat().st_size > 1000


def test_crosses_write_json(plot, cfg_path, tmp_path):
    out = tmp_path / "crosses.json"
    rc = plot.main(
        [
            "--config",
            cfg_path,
            "--sim",
            str(tmp_path / "c.png"),
            "crosses",
            "--grid",
            "3x2",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["frame"] == "bed" and data["grid"] == [3, 2] and data["simulated"] is True
    xs = sorted({c["x"] for c in data["crosses"]})
    ys = sorted({c["y"] for c in data["crosses"]})
    assert len(data["crosses"]) == 6
    assert xs[1] - xs[0] == pytest.approx(50) and ys[1] - ys[0] == pytest.approx(50)


def test_refused_job_returns_2_and_writes_nothing(plot, cfg_path, tmp_path, capsys):
    out = tmp_path / "crosses.json"
    rc = plot.main(
        ["--config", cfg_path, "--dry-run", "crosses", "--grid", "9x9", "--json", str(out)]
    )
    assert rc == 2
    assert "outside the drawable area" in capsys.readouterr().err
    assert not out.exists()


def test_pen_test_and_feed_test(plot, cfg_path, capsys):
    assert plot.main(["--config", cfg_path, "--dry-run", "pen-test", "--z", "0.5"]) == 0
    assert "G1 Z0.5 F300" in capsys.readouterr().out
    assert plot.main(["--config", cfg_path, "--dry-run", "feed-test", "--feeds", "1000,3000"]) == 0
    out = capsys.readouterr().out
    assert "F1000" in out and "F3000" in out
    assert plot.main(["--config", cfg_path, "--dry-run", "pen-test", "--z", "-2"]) == 2


def test_info_and_park(plot, cfg_path, capsys):
    assert plot.main(["--config", cfg_path, "--dry-run", "info"]) == 0
    assert "25.00" in capsys.readouterr().out
    assert plot.main(["--config", cfg_path, "--dry-run", "park"]) == 0


def test_config_set_saves_override(plot, cfg_path, capsys):
    assert (
        plot.main(
            ["--config", cfg_path, "config", "--set", "pen_down_z=0.4", "--set", "feed.normal=2000"]
        )
        == 0
    )
    saved = json.loads(Path(cfg_path).read_text())
    assert saved["pen_down_z"] == 0.4 and saved["feed"]["normal"] == 2000
    with pytest.raises(SystemExit, match="not saved"):
        plot.main(["--config", cfg_path, "config", "--set", "pen_up_z=-5"])
    assert json.loads(Path(cfg_path).read_text())["pen_up_z"] == 3.0  # unchanged


def test_parse_grid(plot):
    assert plot.parse_grid("3x3") == (3, 3)
    assert plot.parse_grid("4X2") == (4, 2)
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        plot.parse_grid("three")
