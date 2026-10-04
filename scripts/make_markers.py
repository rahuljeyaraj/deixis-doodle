"""Make the printable sheets for hardware bring-up (docs/hardware-bringup.md, section 2).

Writes, into hardware/print/ by default:

- bed-markers.pdf: ArUco stickers for the plotter bed (IDs 0-7 plus a spare set), A4.
- floor-markers.pdf: one large ArUco marker per A4 page for the room-scale floor view.
- m1-targets.pdf: an A5 landscape sheet of numbered crosshair targets for the M1 test.
- m1-targets.json: those targets in mm, paper frame (origin at the sheet's bottom-left
  corner as the user sees it, +x to the right, +y away from the user).
- robot-marker.pdf: one 100 mm marker (ID 14) for the vacuum's lid (floor-to-map
  calibration, `scripts/vacuum_calibrate.py`).

Sizes, IDs and the dictionary come from pointdraw.layout. Markers are drawn as vector
rectangles, so they stay sharp at any printer resolution. Output is byte-for-byte
reproducible (reportlab invariant mode), so re-running it does not churn git.

Usage: uv run python scripts/make_markers.py [--out hardware/print]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, A5, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from pointdraw.layout import (
    ARUCO_DICTIONARY,
    BED_MARKER_IDS,
    BED_MARKER_SIZE_MM,
    FLOOR_MARKER_IDS,
    FLOOR_MARKER_SIZE_MM,
    PAPER_HEIGHT_MM,
    PAPER_WIDTH_MM,
    ROBOT_MARKER_ID,
    ROBOT_MARKER_SIZE_MM,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "hardware" / "print"

BED_QUIET_ZONE_MM = 6.0  # white margin around each bed marker, inside the cut line
FLOOR_QUIET_ZONE_MM = 20.0
M1_EDGE_MARGIN_MM = 15.0  # no target centre closer than this to any edge
M1_RING_RADII_MM = (3.0, 5.0)  # M1 pass limits: median <= 3 mm, max <= 5 mm (plan 5.0)
M1_SEED = 20261004  # M1 exit test day; fixes the jitter so the sheet never changes

GREY = (0.55, 0.55, 0.55)
ACTUAL_SIZE = "Print at ACTUAL SIZE / 100 % (turn off 'fit to page' and scaling)."


@dataclass(frozen=True)
class Target:
    id: int
    x_mm: float
    y_mm: float


def aruco_bits(marker_id: int) -> np.ndarray:
    """The marker as a 6x6 array (4x4 data plus the 1-cell black border); True = white."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY))
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
    return image > 127


def _canvas(path: Path, pagesize: tuple[float, float], title: str) -> Canvas:
    c = Canvas(str(path), pagesize=pagesize, invariant=1)
    c.setTitle(title)
    c.setAuthor("deixis-doodle scripts/make_markers.py")
    return c


def draw_marker(c: Canvas, marker_id: int, x_mm: float, y_mm: float, size_mm: float) -> None:
    """Draw a marker with its outer black edge at (x, y)..(x + size, y + size), in mm."""
    bits = aruco_bits(marker_id)
    n = bits.shape[0]
    cell = size_mm / n
    c.setFillColorRGB(0, 0, 0)
    c.rect(x_mm * mm, y_mm * mm, size_mm * mm, size_mm * mm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    for row in range(n):
        top = y_mm + size_mm - row * cell  # image row 0 is the top of the marker
        col = 0
        while col < n:
            if not bits[row, col]:
                col += 1
                continue
            start = col
            while col < n and bits[row, col]:
                col += 1
            c.rect(
                (x_mm + start * cell) * mm,
                (top - cell) * mm,
                (col - start) * cell * mm,
                cell * mm,
                stroke=0,
                fill=1,
            )


def draw_scale_bar(c: Canvas, x_mm: float, y_mm: float, length_mm: float = 100.0) -> None:
    """A ruler-style bar with ticks every 10 mm and a check instruction above it."""
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.3 * mm)
    c.line(x_mm * mm, y_mm * mm, (x_mm + length_mm) * mm, y_mm * mm)
    for i in range(int(length_mm // 10) + 1):
        h = 4.0 if i % 5 == 0 else 2.0
        tx = (x_mm + i * 10) * mm
        c.line(tx, y_mm * mm, tx, (y_mm + h) * mm)
    c.setFont("Helvetica", 7)
    c.drawCentredString(x_mm * mm, (y_mm - 3.5) * mm, "0")
    c.drawCentredString((x_mm + length_mm) * mm, (y_mm - 3.5) * mm, f"{length_mm:.0f} mm")
    c.drawString(
        x_mm * mm, (y_mm + 5.5) * mm, f"This bar must measure {length_mm:.0f} mm on paper."
    )


def make_bed_markers(path: Path) -> None:
    """A4: bed marker stickers, one main set and one spare set, with cut guides."""
    page_w, page_h = A4[0] / mm, A4[1] / mm
    size, quiet = BED_MARKER_SIZE_MM, BED_QUIET_ZONE_MM
    cell_w = size + 2 * quiet
    cell_h = cell_w + 8.0  # room for the ID label under the quiet zone
    cols, gutter = 4, 10.0
    left = (page_w - (cols * cell_w + (cols - 1) * gutter)) / 2

    c = _canvas(path, A4, "Bed markers")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left * mm, (page_h - 20) * mm, "Bed markers")
    c.setFont("Helvetica", 8.5)
    lines = [
        f"ArUco {ARUCO_DICTIONARY}, IDs {BED_MARKER_IDS[0]}-{BED_MARKER_IDS[-1]}, "
        f"{size:.0f} mm black square. Matte sticker paper.",
        ACTUAL_SIZE + " Check the 100 mm bar before cutting.",
        f"Cut on the dashed lines: keep the white border (>= {quiet:.0f} mm) round every marker.",
        "Stick 4-6 flat on the bed around the paper; any two in view are enough.",
    ]
    for i, text in enumerate(lines):
        c.drawString(left * mm, (page_h - 27 - i * 4.2) * mm, text)

    blocks = [("Main set", ""), ("Spares: only to replace a damaged sticker", " spare")]
    top = page_h - 52
    for title, suffix in blocks:
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left * mm, top * mm, title)
        top -= 4
        for k, marker_id in enumerate(BED_MARKER_IDS):
            r, col = divmod(k, cols)
            x0 = left + col * (cell_w + gutter)
            y0 = top - (r + 1) * cell_h - r * gutter
            c.setStrokeColorRGB(*GREY)
            c.setLineWidth(0.2 * mm)
            c.setDash(2 * mm, 1.5 * mm)
            c.rect(x0 * mm, y0 * mm, cell_w * mm, cell_h * mm, stroke=1, fill=0)
            c.setDash()
            draw_marker(c, marker_id, x0 + quiet, y0 + cell_h - quiet - size, size)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 8)
            c.drawCentredString((x0 + cell_w / 2) * mm, (y0 + 2.5) * mm, f"ID {marker_id}{suffix}")
        rows = -(-len(BED_MARKER_IDS) // cols)
        top -= rows * cell_h + (rows - 1) * gutter + 10

    draw_scale_bar(c, left, 22)
    c.showPage()
    c.save()


def make_floor_markers(path: Path) -> None:
    """A4 pages, one large floor marker each (room-scale view, plan 4.8)."""
    page_w, page_h = A4[0] / mm, A4[1] / mm
    size = FLOOR_MARKER_SIZE_MM
    x0 = (page_w - size) / 2
    y0 = (page_h - size) / 2 + 8
    c = _canvas(path, A4, "Floor markers")
    for marker_id in FLOOR_MARKER_IDS:
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 8.5)
        c.drawString(20 * mm, (page_h - 18) * mm, f"Floor marker, ArUco {ARUCO_DICTIONARY}.")
        c.drawString(20 * mm, (page_h - 22.5) * mm, ACTUAL_SIZE)
        c.drawString(
            20 * mm,
            (page_h - 27) * mm,
            f"Black square = {size:.0f} mm. Keep the white margin; tape flat, matte side up.",
        )
        draw_marker(c, marker_id, x0, y0, size)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 28)
        c.drawCentredString(
            page_w / 2 * mm, (y0 - FLOOR_QUIET_ZONE_MM - 10) * mm, f"ID {marker_id}"
        )
        draw_scale_bar(c, (page_w - 100) / 2, 22)
        c.showPage()
    c.save()


def make_robot_marker(path: Path) -> None:
    """One A4 page: the marker for the vacuum's lid (floor-to-map calibration)."""
    page_w, page_h = A4[0] / mm, A4[1] / mm
    size = ROBOT_MARKER_SIZE_MM
    x0 = (page_w - size) / 2
    y0 = (page_h - size) / 2
    c = _canvas(path, A4, "Robot marker")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 8.5)
    lines = [
        f"Robot marker, ArUco {ARUCO_DICTIONARY}, for the vacuum's lid.",
        ACTUAL_SIZE,
        f"Black square = {size:.0f} mm. Cut along the dashed line; keep the white margin.",
        "Tape it flat with this side up and FRONT toward the robot's front.",
        "Never cover the LiDAR turret or the front bumper's sensors.",
    ]
    for k, line in enumerate(lines):
        c.drawString(20 * mm, (page_h - 18 - 4.5 * k) * mm, line)
    draw_marker(c, ROBOT_MARKER_ID, x0, y0, size)
    q = FLOOR_QUIET_ZONE_MM / 2
    c.setDash(3, 3)
    c.rect((x0 - q) * mm, (y0 - q - 12) * mm, (size + 2 * q) * mm, (size + 2 * q + 24) * mm)
    c.setDash()
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w / 2 * mm, (y0 + size + 4) * mm, "FRONT")
    c.drawCentredString(page_w / 2 * mm, (y0 - 9) * mm, f"ID {ROBOT_MARKER_ID}")
    draw_scale_bar(c, (page_w - 100) / 2, 22)
    c.showPage()
    c.save()


def m1_target_positions() -> list[Target]:
    """15 targets: a 5 x 3 grid with fixed jitter, in the paper frame (mm)."""
    rng = random.Random(M1_SEED)
    xs = np.linspace(28.0, PAPER_WIDTH_MM - 28.0, 5)
    ys = np.linspace(PAPER_HEIGHT_MM - 30.0, 30.0, 3)  # row 1 is the far row
    targets = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            jx = round(rng.uniform(-6.0, 6.0) * 2) / 2
            jy = round(rng.uniform(-6.0, 6.0) * 2) / 2
            targets.append(Target(row * len(xs) + col + 1, float(x) + jx, float(y) + jy))
    return targets


def draw_crosshair(c: Canvas, t: Target) -> None:
    x, y = t.x_mm, t.y_mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.2 * mm)
    arm = 8.0
    c.line((x - arm) * mm, y * mm, (x + arm) * mm, y * mm)
    c.line(x * mm, (y - arm) * mm, x * mm, (y + arm) * mm)
    inner, outer = M1_RING_RADII_MM
    c.setStrokeColorRGB(*GREY)
    c.setDash(0.8 * mm, 0.8 * mm)
    c.circle(x * mm, y * mm, inner * mm, stroke=1, fill=0)
    c.setDash()
    c.circle(x * mm, y * mm, outer * mm, stroke=1, fill=0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString((x + 5.5) * mm, (y + 4.0) * mm, str(t.id))


def make_m1_targets(pdf_path: Path, json_path: Path) -> list[Target]:
    """A5 landscape target sheet for the M1 test, plus its JSON."""
    pagesize = landscape(A5)
    assert abs(pagesize[0] / mm - PAPER_WIDTH_MM) < 0.5
    assert abs(pagesize[1] / mm - PAPER_HEIGHT_MM) < 0.5
    targets = m1_target_positions()
    c = _canvas(pdf_path, pagesize, "M1 targets")
    for t in targets:
        draw_crosshair(c, t)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 6.5)
    c.drawString(8 * mm, 9 * mm, "M1 targets. Print at 100 %. This edge toward you.")
    c.drawString(8 * mm, 5.5 * mm, "Rings: 3 mm (dashed) and 5 mm radius.")
    draw_scale_bar(c, 100, 5)
    c.showPage()
    c.save()

    data = {
        "sheet": "m1-targets.pdf",
        "width_mm": PAPER_WIDTH_MM,
        "height_mm": PAPER_HEIGHT_MM,
        "frame": (
            "paper frame: origin at the sheet's bottom-left corner as the user sees it "
            "(printed text upright, bottom edge toward the user); +x right, +y away"
        ),
        "ring_radii_mm": list(M1_RING_RADII_MM),
        "targets": [{"id": t.id, "x_mm": t.x_mm, "y_mm": t.y_mm} for t in targets],
    }
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return targets


def make_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    names = ("bed-markers.pdf", "floor-markers.pdf", "m1-targets.pdf", "robot-marker.pdf")
    paths = [out / n for n in names]
    make_bed_markers(paths[0])
    make_floor_markers(paths[1])
    make_m1_targets(paths[2], out / "m1-targets.json")
    make_robot_marker(paths[3])
    return [*paths, out / "m1-targets.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    args = parser.parse_args()
    for p in make_all(args.out):
        print(p)


if __name__ == "__main__":
    main()
