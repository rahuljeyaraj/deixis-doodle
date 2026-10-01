"""Getting a library Job's images out through its log (library_job.py Sink + logs_to_files.py).

On 1 Oct 2026 every write to the Job's bucket mount failed, so files also travel in the log.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "scripts" / "nebius" / "library_job.py"


def _logs_to_files():
    spec = importlib.util.spec_from_file_location(
        "logs_to_files", ROOT / "scripts" / "nebius" / "logs_to_files.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_job(tmp_path: Path, out: Path) -> str:
    pytest.importorskip("yaml")
    cmd = [sys.executable, str(JOB), "--model", "fake", "--set", "pilot", "--objects", "ear"]
    cmd += ["tree", "--per-object", "2", "--out", str(out), "--work", str(tmp_path / "work")]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return run.stdout


def test_files_come_back_from_the_log_when_the_bucket_fails(tmp_path):
    blocker = tmp_path / "bucket"
    blocker.write_text("not a folder")  # every write under it fails, like the broken mount
    stdout = _run_job(tmp_path, blocker / "pilot")
    assert "files go to the log only" in stdout
    log = tmp_path / "job.log"
    # the CLI's --timestamps prefix must not matter
    log.write_text("".join(f"[2026-10-01T02:49:16Z] {line}\n" for line in stdout.splitlines()))
    out = tmp_path / "rebuilt"
    assert _logs_to_files().main([str(log), str(out)]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["bucket_ok"] is False
    assert [e["file"] for e in man["images"]] == [
        "ear/ear_s1000.png",
        "ear/ear_s1001.png",
        "tree/tree_s1000.png",
        "tree/tree_s1001.png",
    ]
    for e in man["images"]:
        assert (out / e["file"]).stat().st_size > 0
        # the log copy is the compact one: grey, 16 levels, same as the local copy's content
        from PIL import Image

        with Image.open(out / e["file"]) as im:
            assert im.mode == "L" and im.size == (1024, 1024)
            assert len(im.getcolors(256) or []) <= 16


def test_a_missing_chunk_is_caught_and_the_file_skipped(tmp_path):
    stdout = _run_job(tmp_path, tmp_path / "bucket" / "pilot")
    lines = [ln for ln in stdout.splitlines() if not ln.startswith("@@C 2 1 ")]
    log = tmp_path / "job.log"
    log.write_text("\n".join(lines) + "\n")
    mod = _logs_to_files()
    files, problems = mod.parse(log.read_text().splitlines())
    assert problems == [f"ear/ear_s1001.png: 1 of {files[2]['n']} chunks missing"]
    assert mod.main([str(log), str(tmp_path / "rebuilt")]) == 1
    assert not (tmp_path / "rebuilt" / "ear" / "ear_s1001.png").exists()
    assert (tmp_path / "rebuilt" / "ear" / "ear_s1000.png").exists()


def test_with_a_working_bucket_the_full_images_land_there_too(tmp_path):
    out = tmp_path / "bucket" / "pilot"
    _run_job(tmp_path, out)
    man = json.loads((out / "manifest.json").read_text())
    assert man["bucket_ok"] is True and len(man["images"]) == 4
    assert all((out / e["file"]).exists() for e in man["images"])


def test_fetch_reads_a_big_log_in_slices_under_the_api_limit(tmp_path):
    """The logs API returns at most 1000 lines per request (the newest ones, like --tail)."""
    from datetime import UTC, datetime

    mod = _logs_to_files()
    t0 = 1_790_000_000
    # 3000 lines over 100 s, bursts of 300 in some seconds (a file's chunks land at once)
    log = [(t0 + (k // 300) * 10 + (k % 7) * 0.01, f"line {k}") for k in range(3000)]
    asked = []

    def query(a, b):
        asked.append((a, b))
        got = [(t, s) for t, s in log if a <= t <= b]
        stamp = lambda t: datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
        return [f"[{stamp(t)}] {s}" for t, s in got][-mod.LIMIT :]

    lines = mod.fetch_lines(query, t0, t0 + 100, step=30)
    got = {ln.split("] ", 1)[1] for ln in lines}
    assert got == {s for _, s in log}  # nothing lost; duplicates from the overlap are fine
    assert len(asked) > 4  # it had to split


def test_a_rebuilt_manifest_reads_object_and_style_from_the_path(tmp_path):
    """No manifest in the log (job killed early): object from the folder, style from the name."""
    import base64
    import hashlib

    mod = _logs_to_files()
    lines = []
    for no, rel in enumerate(["ice_cream/ice_cream_icon_s1002.png", "ear/ear_s1000.png"], 1):
        data = f"png {rel}".encode()
        b64 = base64.b64encode(data).decode()
        lines.append(f"@@FILE {no} 1 {len(data)} {hashlib.sha256(data).hexdigest()} {rel}")
        lines.append(f"@@C {no} 0 {b64}")
    log = tmp_path / "job.log"
    log.write_text("\n".join(lines) + "\n")
    assert mod.main([str(log), str(tmp_path / "out")]) == 0
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["images"] == [
        {"file": "ear/ear_s1000.png", "object": "ear", "seed": 1000},
        {
            "file": "ice_cream/ice_cream_icon_s1002.png",
            "object": "ice cream",
            "seed": 1002,
            "style": "icon",
        },
    ]
