"""Printable target sheets for the plotter's accuracy tests (measure list M01, M02, M03).

    uv run python scripts/target_sheets.py --out <folder>

Two sheets, each an A4 PDF to print at 100 % (rahul, 6 Oct: keep it simple; print a fresh
copy for each run, and the copy from the counted run is the one the judges see):
- `crosses.pdf`: 20 grey "x" marks, for "draw a cross here" (M01, M03), layout crosses-A;
- `circles.pdf`: 10 grey circles, a small "x" at each centre, for "draw a circle here, this
  big" (M02), layout circles-1.
Each says in plain words, inside the strip, what is tested. Plus a PNG preview per layout and
`sheets.json`, the layouts (sheet mm). `scripts/measure_plotter.py targets` finds the marks
again in the run's first parked photo, so the strip needn't lie in an exact place.

How to use one: print at ACTUAL SIZE (100 %), check the 100 mm bar with a ruler, cut along
the two dashed lines (top and bottom; the full A4 width, 210 mm, fits inside the frame), and
tape the strip onto a fresh 235 mm square, roughly in the middle, the word SOUTH nearest.
All marks must be inside the frame the app draws. Nothing printed says "your": photos of
these sheets go to the judges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from pointdraw.measure import targets as T

PAGE_W, PAGE_H = A4[0] / mm, A4[1] / mm  # 210 x 297
OX, OY = (PAGE_W - T.PIECE_W) / 2, PAGE_H - T.PIECE_H - 12  # the piece's corner on the page

#: print file -> (layout, the test label printed inside the piece, the title above the steps)
PRINTS = {
    "crosses": (
        "crosses-A",
        'Pointing test: point at a grey cross and say "draw a cross here". '
        "The gap to the black cross is the error.",
        "Crosses: the pointing test (M01, M03). Print a fresh copy for each run",
    ),
    "circles": (
        "circles-1",
        'Size test: show a grey circle\'s size and say "draw a circle here, this big". '
        "The black circle should match it.",
        "Circles: the size test (M02). Print a fresh copy for each run",
    ),
}
SAY = {
    "crosses": 'Point at the centre of an "x" and say "draw a cross here". Then "yes".',
    "circles": 'Point at a circle\'s centre, say "draw a circle here, this big", show its '
    'size with thumb and finger. Then "yes".',
}


def make_pdf(print_name: str, path: Path) -> None:
    name, label, title = PRINTS[print_name]
    c = Canvas(str(path), pagesize=A4)
    c.setTitle(f"deixis target sheet {print_name}")
    g = T.GREY

    def P(x, y):  # sheet mm -> page points
        return (OX + x) * mm, (OY + y) * mm

    # the two cut lines, top and bottom, kept 10 mm from the page's sides (printers leave the
    # outer few mm blank); cut them on to the paper's edge
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.2 * mm)
    c.setDash(2 * mm, 1.5 * mm)
    for y in (OY, OY + T.PIECE_H):
        c.line(10 * mm, y * mm, (PAGE_W - 10) * mm, y * mm)
    c.setDash()
    # the marks
    c.setStrokeColorRGB(g, g, g)
    c.setLineWidth(T.LINE_MM * mm)
    for t in T.layout(name):
        h = T.CENTRE_X_HALF if t.d else T.X_HALF
        c.line(*P(t.x - h, t.y - h), *P(t.x + h, t.y + h))
        c.line(*P(t.x - h, t.y + h), *P(t.x + h, t.y - h))
        if t.d:
            c.circle(*P(t.x, t.y), t.d / 2 * mm, stroke=1, fill=0)
    # small numbers, away from the marks
    c.setFillColorRGB(g, g, g)
    c.setFont("Helvetica", 7)
    for t in T.layout(name):
        dx = (t.d / 2 * 0.72 + 1.5) if t.d else T.X_HALF + 1.5
        c.drawString(*P(t.x + dx, t.y + dx), str(t.id))
    # which way round, inside the piece
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(*P(T.PIECE_W / 2, 3.0), "SOUTH (near side)")
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(*P(T.PIECE_W / 2, T.PIECE_H - 5.5), label)
    # outside the piece: the instructions and the 100 mm check bar
    kind = T.SHEETS[name]["kind"]
    y = OY - 12
    c.setFont("Helvetica-Bold", 11)
    c.drawString(OX * mm, y * mm, title)
    c.setFont("Helvetica", 9.5)
    lines = [
        "1. Printed at ACTUAL SIZE / 100 %? The bar below must measure exactly 100 mm.",
        "2. Cut along the two dashed lines (top and bottom), right across the page.",
        "3. Tape the strip flat onto a fresh 235 mm square, in the middle, SOUTH nearest.",
        "4. Clip the square on the plotter as usual. All marks must be inside the drawn frame.",
        f"5. {SAY[kind]}",
    ]
    for i, s in enumerate(lines):
        c.drawString(OX * mm, (y - 7 - 5.5 * i) * mm, s)
    by = y - 7 - 5.5 * len(lines) - 8
    c.setLineWidth(0.4 * mm)
    c.line(OX * mm, by * mm, (OX + 100) * mm, by * mm)
    for k in range(0, 101, 10):
        tick = 3 if k % 50 == 0 else 1.5
        c.line((OX + k) * mm, by * mm, (OX + k) * mm, (by + tick) * mm)
    c.setFont("Helvetica", 8)
    c.drawString((OX + 102) * mm, (by - 1) * mm, "100 mm")
    c.showPage()
    c.save()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out/targets"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    data = {
        "frame": "sheet mm: origin at the piece's bottom-left corner, SOUTH toward you; "
        "+x right (east), +y away (north)",
        "piece_mm": [T.PIECE_W, T.PIECE_H],
        "sheets": {},
    }
    for print_name in PRINTS:
        make_pdf(print_name, a.out / f"{print_name}.pdf")
        print(f"wrote {a.out / (print_name + '.pdf')}")
    for name in T.SHEETS:
        cv2.imwrite(str(a.out / f"{name}.png"), T.render(name, 0.25))
        data["sheets"][name] = T.as_dicts(T.layout(name))
    (a.out / "sheets.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
