"""Render polylines (mm, paper frame, y up) to a PNG with the paper outline and a mm grid.

For storyboards, contact sheets and debugging. The controller's simulator has its own
renderer for what the machine actually did; this one shows what we plan to draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from deixis.geometry import Box, Point
from pointdraw import layout

A5 = Box(0.0, 0.0, layout.PAPER_WIDTH_MM, layout.PAPER_HEIGHT_MM)

INK = (20, 20, 30)
PAPER = (255, 255, 255)
BACKGROUND = (232, 232, 228)
GRID_MINOR = (236, 240, 246)
GRID_MAJOR = (214, 222, 234)
OUTLINE = (120, 120, 120)
BOX = (215, 90, 60)


def render(
    polylines: Sequence[Sequence[Point]],
    out: str | Path | None = None,
    paper: Box = A5,
    px_per_mm: float = 5.0,
    margin_mm: float = 8.0,
    grid_mm: float | None = 10.0,
    boxes: Sequence[Box] = (),
    pen_mm: float = 0.6,
) -> Image.Image:
    """Draw the paper, an optional grid (minor lines every grid_mm, major every 5), dashed
    boxes (e.g. preview boxes) and the ink. Saves to `out` if given."""
    s = px_per_mm
    w = round((paper.w + 2 * margin_mm) * s)
    h = round((paper.h + 2 * margin_mm) * s)
    img = Image.new("RGB", (w, h), BACKGROUND)
    d = ImageDraw.Draw(img)

    def px(p: Point) -> tuple[float, float]:
        return ((p[0] - paper.x + margin_mm) * s, (paper.y2 - p[1] + margin_mm) * s)

    d.rectangle([px((paper.x, paper.y2)), px((paper.x2, paper.y))], fill=PAPER)
    if grid_mm:
        n = 0
        x = paper.x
        while x <= paper.x2 + 1e-9:
            d.line(
                [px((x, paper.y)), px((x, paper.y2))], fill=GRID_MAJOR if n % 5 == 0 else GRID_MINOR
            )
            x += grid_mm
            n += 1
        n = 0
        y = paper.y
        while y <= paper.y2 + 1e-9:
            d.line(
                [px((paper.x, y)), px((paper.x2, y))], fill=GRID_MAJOR if n % 5 == 0 else GRID_MINOR
            )
            y += grid_mm
            n += 1
    d.rectangle([px((paper.x, paper.y2)), px((paper.x2, paper.y))], outline=OUTLINE, width=2)
    for b in boxes:
        corners = [(b.x, b.y), (b.x2, b.y), (b.x2, b.y2), (b.x, b.y2), (b.x, b.y)]
        for a, c in zip(corners, corners[1:], strict=False):
            _dashed(d, px(a), px(c), BOX, dash=3 * s)
    width = max(1, round(pen_mm * s))
    for pl in polylines:
        pts = [px(p) for p in pl]
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=INK)
        elif pts:
            d.line(pts, fill=INK, width=width, joint="curve")
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
    return img


def _dashed(d: ImageDraw.ImageDraw, a, b, fill, dash: float) -> None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    n = max(1, int(length // dash))
    for i in range(0, n, 2):
        t0, t1 = i / n, min(1.0, (i + 1) / n)
        d.line(
            [(a[0] + dx * t0, a[1] + dy * t0), (a[0] + dx * t1, a[1] + dy * t1)], fill=fill, width=2
        )


def render_cell(
    strokes: Sequence[Sequence[Point]], size_px: int = 160, pad_px: int = 12, y_up: bool = False
) -> Image.Image:
    """A square thumbnail of strokes in any units, scaled to fit (for contact sheets)."""
    img = Image.new("RGB", (size_px, size_px), PAPER)
    pts = [p for s in strokes for p in s]
    if not pts:
        return img
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    k = (size_px - 2 * pad_px) / max(x1 - x0, y1 - y0, 1e-9)
    ox = (size_px - (x1 - x0) * k) / 2
    oy = (size_px - (y1 - y0) * k) / 2
    d = ImageDraw.Draw(img)
    for s in strokes:
        if y_up:
            line = [(ox + (x - x0) * k, size_px - oy - (y - y0) * k) for x, y in s]
        else:
            line = [(ox + (x - x0) * k, oy + (y - y0) * k) for x, y in s]
        if len(line) >= 2:
            d.line(line, fill=INK, width=2, joint="curve")
    return img
