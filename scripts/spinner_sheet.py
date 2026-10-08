"""Printable spinner dials for the plotter demo (the film's shape pick and side pick).

    uv run python scripts/spinner_sheet.py --out <folder>

Two A4 PDFs, printed at 100 %, with the plastic pointer on its ball-bearing base stood on
the middle. The pointer is 70 mm from the axis to the tip, so the tip sweeps the dashed ring
and comes to rest just outside a word.
- `spinner-shapes.pdf`: eight shapes, one per segment, the shape to ask the robot for;
- `spinner-side.pdf`: LEFT and RIGHT, two segments each, the side the tree goes on.
The dial alone is on the page, nothing printed round it. It is 190 mm across, which is the
measurement to check if the printer was asked for 100 %.
Nothing printed says "your": the sheet is on camera in the film.

Pass `--words` to try a different set (`--words CAT,DUCK,...`).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = A4[0] / mm, A4[1] / mm  # 210 x 297
CX, CY = PAGE_W / 2, PAGE_H / 2  # the axis, in the middle of the page
R_OUT = 95.0  # the dial's edge
R_IN = 16.0  # the ball-bearing base is 32 mm across: the circle to stand it in
R_TIP = 70.0  # how far the pointer's tip reaches, so the words stay outside it
WORD_H = 14.0  # the biggest cap height; every word is drawn at one size
PAD = 2.5  # white space left round a word
FAINT = 0.93  # the lighter segments

#: the eight on the wheel, rahul's pick on 6 Oct: the nine timed on the machine that day
#: (M06), without the tree. Catalogue numbers 103, 89, 130, 98, 101, 116, 120, 122.
SHAPES = ["DUCK", "MUSHROOM", "DOG", "BED", "GIFT", "ELEPHANT", "FROG", "CAT"]
SIDES = ["LEFT", "RIGHT"] * 2


def clears(word: str, h: float, mid: float, r: float, half: float) -> bool:
    """Does this word, lying flat and centred at radius `r`, sit inside its own segment?

    Every word is written the same way up, so it is a plain upright box; its four corners
    are what can foul. Each must stay inside the dial's edge, clear of the middle circle,
    and between the segment's two dividing lines.
    """
    a = math.radians(mid)
    cx, cy = r * math.cos(a), r * math.sin(a)
    half_w = stringWidth(word, "Helvetica-Bold", h / 0.717) / 2 + PAD
    half_h = h / 2 + PAD
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * half_w, cy + sy * half_h
            if not R_IN + 2 <= math.hypot(x, y) <= R_OUT - 2:
                return False
            off = (math.atan2(y, x) - a + math.pi) % (2 * math.pi) - math.pi
            if abs(off) > half:
                return False
    return True


def radius(word: str, h: float, mid: float, half: float) -> float | None:
    """How far out this word goes, or None if it does not fit anywhere in the segment.

    A segment is a wedge, so a long word has to sit far enough out for the wedge to be wide
    enough, but not so far that a corner crosses the rim. The middle of whatever range works
    is the roomiest place for it.
    """
    ok = [r for r in range(int(R_IN) + 2, int(R_OUT)) if clears(word, h, mid, r, half)]
    return (ok[0] + ok[-1]) / 2 if ok else None


def fit(words: list[str], n: int) -> float:
    """One cap height for every word: the word that fits worst sets it, so all eight match."""
    half = math.radians(180 / n)

    def fits(h: float) -> bool:
        return all(radius(w, h, a0 + 180 / n, half) is not None for _i, a0, w in segments(words))

    lo, hi = 1.0, WORD_H
    if fits(hi):
        return hi
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if fits(m) else (lo, m)
    return lo


def segments(words: list[str]):
    """(index, start angle, word) per segment, the first centred straight up."""
    step = 360 / len(words)
    for i, w in enumerate(words):
        yield i, 90 + step / 2 + i * step, w


def make_pdf(path: Path, words: list[str]) -> None:
    c = Canvas(str(path), pagesize=A4)
    c.setTitle(path.stem)
    step = 360 / len(words)
    # every other segment in a very light grey, so the pointer's rest place reads on camera
    for i, a0, _ in segments(words):
        if i % 2:
            continue
        c.setFillColorRGB(FAINT, FAINT, FAINT)
        p = c.beginPath()
        p.moveTo(CX * mm, CY * mm)
        p.arcTo(
            (CX - R_OUT) * mm,
            (CY - R_OUT) * mm,
            (CX + R_OUT) * mm,
            (CY + R_OUT) * mm,
            a0,
            step,
        )
        p.close()
        c.drawPath(p, stroke=0, fill=1)
    # the circle the base stands in, the size of the bearing itself
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.4 * mm)
    c.circle(CX * mm, CY * mm, R_IN * mm, stroke=1, fill=1)
    # the edge and the dividing lines
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.8 * mm)
    c.circle(CX * mm, CY * mm, R_OUT * mm, stroke=1, fill=0)
    c.setLineWidth(0.4 * mm)
    for _i, a0, _ in segments(words):
        a = math.radians(a0)
        c.line(
            (CX + R_IN * math.cos(a)) * mm,
            (CY + R_IN * math.sin(a)) * mm,
            (CX + R_OUT * math.cos(a)) * mm,
            (CY + R_OUT * math.sin(a)) * mm,
        )
    # the words, all lying flat, so every one reads the same way up
    h = fit(words, len(words))
    half = math.radians(step / 2)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", h / 0.717 * mm)
    for _i, a0, w in segments(words):
        mid = a0 + step / 2
        a, r = math.radians(mid), radius(w, h, mid, half)
        c.drawCentredString((CX + r * math.cos(a)) * mm, (CY + r * math.sin(a) - h / 2) * mm, w)
    c.showPage()
    c.save()


def face(px: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arialbd.ttf", px)
    except OSError:
        return ImageFont.load_default(px)


def make_png(path: Path, words: list[str], dpmm: int = 4) -> None:
    """The same dial, to look at on screen; the PDF is the one to print."""
    img = Image.new("RGB", (int(PAGE_W * dpmm), int(PAGE_H * dpmm)), "white")
    d = ImageDraw.Draw(img)
    step = 360 / len(words)

    def P(x, y):  # page mm -> pixels, y down
        return x * dpmm, (PAGE_H - y) * dpmm

    def ring(r):
        return [*P(CX - r, CY + r), *P(CX + r, CY - r)]

    for i, a0, _ in segments(words):
        if i % 2 == 0:
            d.pieslice(ring(R_OUT), -a0 - step, -a0, fill=(237, 237, 237))
    d.ellipse(ring(R_OUT), outline="black", width=max(1, int(0.8 * dpmm)))
    d.ellipse(ring(R_IN), fill="white", outline="black", width=max(1, int(0.4 * dpmm)))
    for _i, a0, _ in segments(words):
        a = math.radians(a0)
        d.line(
            [
                *P(CX + R_IN * math.cos(a), CY + R_IN * math.sin(a)),
                *P(CX + R_OUT * math.cos(a), CY + R_OUT * math.sin(a)),
            ],
            fill="black",
            width=max(1, int(0.4 * dpmm)),
        )
    h = fit(words, len(words))
    half = math.radians(step / 2)
    font = face(int(h / 0.717 * dpmm))
    for _i, a0, w in segments(words):
        mid = a0 + step / 2
        a, r = math.radians(mid), radius(w, h, mid, half)
        d.text(
            P(CX + r * math.cos(a), CY + r * math.sin(a)), w, fill="black", font=font, anchor="mm"
        )
    img.save(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out/spinner"))
    ap.add_argument("--words", help="comma separated, for the shapes dial")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    shapes = [w.strip().upper() for w in a.words.split(",")] if a.words else SHAPES
    for name, words in (("spinner-shapes", shapes), ("spinner-side", SIDES)):
        make_pdf(a.out / f"{name}.pdf", words)
        make_png(a.out / f"{name}.png", words)
        print(f"wrote {a.out / name}.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
