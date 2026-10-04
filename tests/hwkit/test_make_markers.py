"""Tests for scripts/make_markers.py: page sizes, marker detection and M1 target positions."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("reportlab")
pdfium = pytest.importorskip("pypdfium2")

from pointdraw.layout import (  # noqa: E402
    BED_MARKER_IDS,
    BED_MARKER_SIZE_MM,
    FLOOR_MARKER_IDS,
    FLOOR_MARKER_SIZE_MM,
    PAPER_HEIGHT_MM,
    PAPER_WIDTH_MM,
)

ROOT = Path(__file__).resolve().parents[2]
PT_PER_MM = 72 / 25.4
A4_MM = (210.0, 297.0)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_markers", ROOT / "scripts" / "make_markers.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses look the module up while decorating
    spec.loader.exec_module(module)
    return module


mm_script = _load_script()


@pytest.fixture(scope="module")
def out(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("print")
    mm_script.make_all(folder)
    return folder


def _page_sizes_mm(path: Path) -> list[tuple[float, float]]:
    pdf = pdfium.PdfDocument(str(path))
    sizes = [tuple(v / PT_PER_MM for v in pdf[i].get_size()) for i in range(len(pdf))]
    pdf.close()
    return sizes


def _render(path: Path, dpi: int) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(str(path))
    pages = []
    for i in range(len(pdf)):
        image = pdf[i].render(scale=dpi / 72, grayscale=True).to_numpy()
        pages.append(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    pdf.close()
    return pages


def _detect(gray: np.ndarray) -> tuple[list[int], list[float]]:
    """Detected IDs and the mean edge length (px) of each marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [], []
    edges = []
    for quad in corners:
        q = quad.reshape(4, 2)
        edges.append(float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])))
    return [int(i) for i in ids.ravel()], edges


def _close(a: tuple[float, float], b: tuple[float, float], tol: float = 0.5) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def test_page_sizes(out: Path) -> None:
    bed = _page_sizes_mm(out / "bed-markers.pdf")
    floor = _page_sizes_mm(out / "floor-markers.pdf")
    m1 = _page_sizes_mm(out / "m1-targets.pdf")
    assert len(bed) == 1 and _close(bed[0], A4_MM)
    assert len(floor) == len(FLOOR_MARKER_IDS)
    assert all(_close(s, A4_MM) for s in floor)
    assert len(m1) == 1 and _close(m1[0], (PAPER_WIDTH_MM, PAPER_HEIGHT_MM))


def test_bed_markers_detected_at_size(out: Path) -> None:
    dpi = 300
    (page,) = _render(out / "bed-markers.pdf", dpi)
    ids, edges = _detect(page)
    # Main set plus one spare of each.
    assert sorted(ids) == sorted(list(BED_MARKER_IDS) * 2)
    expected = BED_MARKER_SIZE_MM / 25.4 * dpi
    for e in edges:
        assert e == pytest.approx(expected, rel=0.02)


def test_floor_markers_one_per_page(out: Path) -> None:
    dpi = 100
    pages = _render(out / "floor-markers.pdf", dpi)
    expected = FLOOR_MARKER_SIZE_MM / 25.4 * dpi
    for page, marker_id in zip(pages, FLOOR_MARKER_IDS, strict=True):
        ids, edges = _detect(page)
        assert ids == [marker_id]
        assert edges[0] == pytest.approx(expected, rel=0.02)


def test_robot_marker(out: Path) -> None:
    from pointdraw.layout import ROBOT_MARKER_ID, ROBOT_MARKER_SIZE_MM

    dpi = 150
    (page,) = _render(out / "robot-marker.pdf", dpi)
    ids, edges = _detect(page)
    assert ids == [ROBOT_MARKER_ID]
    assert edges[0] == pytest.approx(ROBOT_MARKER_SIZE_MM / 25.4 * dpi, rel=0.02)


def test_m1_targets_json(out: Path) -> None:
    data = json.loads((out / "m1-targets.json").read_text(encoding="utf-8"))
    targets = data["targets"]
    assert data["width_mm"] == PAPER_WIDTH_MM and data["height_mm"] == PAPER_HEIGHT_MM
    assert 12 <= len(targets) <= 18
    assert [t["id"] for t in targets] == list(range(1, len(targets) + 1))
    for t in targets:
        assert 15.0 <= t["x_mm"] <= PAPER_WIDTH_MM - 15.0
        assert 15.0 <= t["y_mm"] <= PAPER_HEIGHT_MM - 15.0
    for a, b in itertools.combinations(targets, 2):
        assert math.dist((a["x_mm"], a["y_mm"]), (b["x_mm"], b["y_mm"])) >= 20.0


def test_m1_crosshairs_at_json_positions(out: Path) -> None:
    """Each crosshair centre is dark in the render; the ring interior next to it is white."""
    dpi = 300
    (page,) = _render(out / "m1-targets.pdf", dpi)
    height = page.shape[0]
    px = dpi / 25.4
    data = json.loads((out / "m1-targets.json").read_text(encoding="utf-8"))
    for t in data["targets"]:
        cx = round(t["x_mm"] * px)
        cy = round(height - t["y_mm"] * px)  # the paper frame's y points up the page
        assert page[cy - 1 : cy + 2, cx - 1 : cx + 2].min() < 128, t
        # 1.5 mm diagonally from the centre lies between the arms and inside the 3 mm ring.
        d = round(1.5 * px)
        assert page[cy - d, cx + d] > 200, t


def test_committed_json_is_current(out: Path) -> None:
    committed = ROOT / "hardware" / "print" / "m1-targets.json"
    assert committed.read_text(encoding="utf-8") == (out / "m1-targets.json").read_text(
        encoding="utf-8"
    )
    for name in ("bed-markers.pdf", "floor-markers.pdf", "m1-targets.pdf"):
        assert (ROOT / "hardware" / "print" / name).stat().st_size < 200_000
