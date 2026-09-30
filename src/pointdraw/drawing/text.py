"""Single-stroke Hershey text in millimetres, for labels.

Glyphs come from vpype (MIT), which ships the Hershey vector fonts (Dr A. V. Hershey,
US National Bureau of Standards, 1967; free to use, with this credit). Single-stroke
fonts draw each line once, which is what a pen wants; outline fonts would draw it twice.

Size is the **cap height** in mm (the height of "H"), which is what a reader judges a
label's size by. The anchor is in the paper frame (y up).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import vpype as vp

from deixis.geometry import Box, Point, Polyline
from pointdraw.drawing.place import bbox

DEFAULT_FONT = "futural"  # Hershey Simplex: plain, single stroke, very legible
LINE_SPACING = 1.7  # baseline-to-baseline distance, in cap heights

HAlign = Literal["left", "center", "right"]
VAlign = Literal["baseline", "middle", "top"]


@lru_cache(maxsize=16)
def _metrics(font: str) -> tuple[float, float]:
    """(cap height, baseline y) of the font at vpype size 1, in vpype's frame (y down)."""
    if font not in vp.FONT_NAMES:
        raise ValueError(f"unknown Hershey font {font!r}; choose from {sorted(vp.FONT_NAMES)}")
    _, top, _, bottom = vp.text_line("H", font, size=100.0).bounds()
    return (bottom - top) / 100.0, bottom / 100.0


def text(
    s: str,
    anchor: Point,
    cap_height: float = 5.0,
    align: HAlign = "center",
    valign: VAlign = "middle",
    font: str = DEFAULT_FONT,
) -> list[Polyline]:
    """Polylines (mm, y up) for `s`. Lines split on newlines.

    valign: "baseline" puts the first line's baseline on the anchor; "middle" centres the
    block's cap-height span on it; "top" puts the first line's cap tops on it.
    """
    if cap_height <= 0:
        raise ValueError("cap_height must be positive")
    cap1, base1 = _metrics(font)
    size = cap_height / cap1
    lines = s.split("\n")
    pitch = LINE_SPACING * cap_height
    block = cap_height + pitch * (len(lines) - 1)  # first cap top to last baseline
    if valign == "baseline":
        first_base = 0.0
    elif valign == "top":
        first_base = -cap_height
    else:
        first_base = block / 2 - cap_height
    out: list[Polyline] = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        # vpype aligns on the advance width, which includes the first glyph's side
        # bearing; align on the ink instead so a centred label is centred on the point.
        lc = vp.text_line(ln, font, size=size, align="left")
        x0, _, x1, _ = lc.bounds()
        dx = {"left": -x0, "center": -(x0 + x1) / 2, "right": -x1}[align]
        base_y = anchor[1] + first_base - i * pitch
        for path in lc:
            pts = [(anchor[0] + dx + z.real, base_y - (z.imag - base1 * size)) for z in path]
            if len(pts) >= 2:
                out.append(pts)
    return out


def text_bounds(
    s: str,
    anchor: Point,
    cap_height: float = 5.0,
    align: HAlign = "center",
    valign: VAlign = "middle",
    font: str = DEFAULT_FONT,
) -> Box:
    """Box of the ink the label will put down (descenders included)."""
    return bbox(text(s, anchor, cap_height, align, valign, font))
