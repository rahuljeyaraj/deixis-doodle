"""Synthetic top-down paper images with hand-drawn-looking marks, plus exact ground truth.

Used by the ink-segmentation tests and by scripts/r1_make_synthetic.py (R1 harness). Shapes are
defined in a unit box, wobbled like a hand-drawn line, scaled to a size in mm, rotated a little,
and rendered with a pencil, crayon or pen colour on slightly uneven paper. The robot's own strokes
(a thin black pen) can be added so the subtraction step can be tested.

Frames follow scene/ink.py: paper mm with +y away from the user; pixel rows grow downwards.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from deixis.geometry import Box, Point, Polyline
from pointdraw.scene.ink import box_mm_to_px, mm_to_px

Unit = list[list[tuple[float, float]]]  # polylines in a unit box, x and y in [-0.5, 0.5]


def _circle(n: int = 48, rx: float = 0.5, ry: float = 0.5, cx: float = 0.0, cy: float = 0.0):
    return [
        (cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
        for i in range(n + 1)
    ]


def _star() -> Unit:
    pts = []
    for i in range(11):
        r = 0.5 if i % 2 == 0 else 0.2
        a = math.pi / 2 + i * math.pi / 5
        pts.append((r * math.cos(a), r * math.sin(a)))
    return [pts]


def _heart() -> Unit:
    pts = []
    for i in range(61):
        t = 2 * math.pi * i / 60
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((x / 34, (y + 2.5) / 34))
    return [pts]


def _spiral() -> Unit:
    return [
        [
            (0.5 * (i / 80) * math.cos(i * 0.3), 0.5 * (i / 80) * math.sin(i * 0.3))
            for i in range(81)
        ]
    ]


def _sun() -> Unit:
    lines: Unit = [_circle(32, 0.22, 0.22)]
    for k in range(8):
        a = k * math.pi / 4
        lines.append(
            [(0.3 * math.cos(a), 0.3 * math.sin(a)), (0.5 * math.cos(a), 0.5 * math.sin(a))]
        )
    return lines


def _smiley() -> Unit:
    mouth = [(0.25 * math.cos(a), -0.05 + 0.25 * math.sin(a)) for a in np.linspace(3.6, 5.8, 12)]
    return [
        _circle(40),
        _circle(10, 0.05, 0.05, -0.17, 0.15),
        _circle(10, 0.05, 0.05, 0.17, 0.15),
        mouth,
    ]


def _cloud() -> Unit:
    pts = []
    bumps = [(-0.3, -0.05, 0.2), (-0.1, 0.12, 0.25), (0.18, 0.1, 0.22), (0.32, -0.08, 0.17)]
    for cx, cy, r in bumps:
        pts += [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(3.3, 6.1, 8)]
    pts = [(x, max(y, -0.25)) for x, y in pts]
    pts.append(pts[0])
    return [pts]


SHAPES: dict[str, Callable[[], Unit]] = {
    "circle": lambda: [_circle()],
    "square": lambda: [[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]],
    "triangle": lambda: [[(-0.5, -0.43), (0.5, -0.43), (0.0, 0.43), (-0.5, -0.43)]],
    "star": _star,
    "heart": _heart,
    "arrow": lambda: [[(-0.5, 0.0), (0.5, 0.0)], [(0.25, 0.2), (0.5, 0.0), (0.25, -0.2)]],
    "house": lambda: [
        [(-0.4, -0.5), (0.4, -0.5), (0.4, 0.1), (-0.4, 0.1), (-0.4, -0.5)],
        [(-0.5, 0.1), (0.0, 0.5), (0.5, 0.1)],
        [(-0.1, -0.5), (-0.1, -0.2), (0.1, -0.2), (0.1, -0.5)],
    ],
    "tree": lambda: [
        [(-0.07, -0.5), (-0.07, -0.05)],
        [(0.07, -0.5), (0.07, -0.05)],
        _circle(36, 0.33, 0.28, 0.0, 0.2),
    ],
    "sun": _sun,
    "smiley face": _smiley,
    "cloud": _cloud,
    "spiral": _spiral,
    "zigzag": lambda: [[(-0.5 + i * 0.125, 0.25 if i % 2 else -0.25) for i in range(9)]],
    "cross": lambda: [[(-0.5, -0.5), (0.5, 0.5)], [(-0.5, 0.5), (0.5, -0.5)]],
    "fish": lambda: [
        _circle(30, 0.32, 0.2, -0.1, 0.0),
        [(0.2, 0.0), (0.5, 0.2), (0.5, -0.2), (0.2, 0.0)],
    ],
}

SYNONYMS: dict[str, list[str]] = {
    "circle": ["ring", "oval", "round", "o", "zero"],
    "square": ["box", "rectangle"],
    "triangle": ["pyramid"],
    "star": [],
    "heart": ["love"],
    "arrow": ["pointer"],
    "house": ["home", "hut", "building"],
    "tree": ["plant"],
    "sun": ["star burst", "sunshine"],
    "smiley face": ["smiley", "face", "happy face", "emoji", "smile"],
    "cloud": [],
    "spiral": ["swirl", "coil", "snail"],
    "zigzag": ["zig zag", "wave", "lightning", "line"],
    "cross": ["x", "plus", "cross mark"],
    "fish": [],
}

# BGR ink colours: (name, colour, typical width mm, alpha)
INKS = [
    ("pencil", (95, 95, 100), 0.6, 0.75),
    ("black pen", (30, 30, 30), 0.7, 1.0),
    ("blue crayon", (200, 140, 60), 1.3, 0.8),
    ("red crayon", (60, 60, 200), 1.3, 0.8),
    ("green crayon", (90, 170, 80), 1.3, 0.8),
    ("orange crayon", (70, 150, 235), 1.4, 0.7),
]


@dataclass
class SynthMark:
    label: str
    polylines_mm: list[Polyline]
    width_mm: float
    ink: str

    @property
    def box_mm(self) -> Box:
        """Box of the rendered stroke, including half the line width."""
        xs = [p[0] for line in self.polylines_mm for p in line]
        ys = [p[1] for line in self.polylines_mm for p in line]
        r = self.width_mm / 2
        return Box(min(xs) - r, min(ys) - r, max(xs) - min(xs) + 2 * r, max(ys) - min(ys) + 2 * r)


@dataclass
class SynthSheet:
    image: np.ndarray
    mm_per_px: float
    size_mm: tuple[float, float]
    marks: list[SynthMark]
    robot_polylines_mm: list[Polyline] = field(default_factory=list)

    def ground_truth(self, file: str = "") -> dict:
        h = self.image.shape[0]
        return {
            "file": file,
            "mm_per_px": self.mm_per_px,
            "size_mm": list(self.size_mm),
            "marks": [
                {
                    "label": m.label,
                    "synonyms": SYNONYMS.get(m.label, []),
                    "ink": m.ink,
                    "box_mm": [
                        round(v, 2) for v in (m.box_mm.x, m.box_mm.y, m.box_mm.w, m.box_mm.h)
                    ],
                    "box_px": [round(v, 1) for v in box_mm_to_px(m.box_mm, h, self.mm_per_px)],
                }
                for m in self.marks
            ],
            "robot_polylines_mm": [
                [[round(x, 2), round(y, 2)] for x, y in line] for line in self.robot_polylines_mm
            ],
            "point_queries": point_queries(self),
        }


def _resample(line: list[tuple[float, float]], step: float) -> np.ndarray:
    pts = np.asarray(line, np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def wobble(line: list[tuple[float, float]], rng: np.random.Generator, amp_mm: float) -> Polyline:
    """Resample at 1 mm and add smooth sideways noise, like an unsteady hand."""
    pts = _resample(line, 1.0)
    if len(pts) < 3 or amp_mm <= 0:
        return [(float(x), float(y)) for x, y in pts]
    noise = rng.normal(0.0, 1.0, (len(pts), 2))
    k = np.ones(9) / 9
    for d in range(2):
        noise[:, d] = np.convolve(np.pad(noise[:, d], 4, mode="edge"), k, mode="valid")
    noise *= amp_mm / max(1e-6, noise.std())
    pts = pts + noise
    return [(float(x), float(y)) for x, y in pts]


def make_mark(
    label: str,
    center_mm: Point,
    size_mm: float,
    rng: np.random.Generator,
    ink: tuple[str, tuple[int, int, int], float, float] | None = None,
    rotate_deg: float = 10.0,
    wobble_frac: float = 0.012,
) -> SynthMark:
    """A shape whose larger dimension is about `size_mm`, centred on `center_mm`."""
    unit = SHAPES[label]()
    a = math.radians(rng.uniform(-rotate_deg, rotate_deg))
    ca, sa = math.cos(a), math.sin(a)
    aspect = rng.uniform(0.85, 1.15)
    lines: list[Polyline] = []
    for ln in unit:
        pts = [
            (
                center_mm[0] + size_mm * (x * aspect * ca - y * sa),
                center_mm[1] + size_mm * (x * aspect * sa + y * ca),
            )
            for x, y in ln
        ]
        lines.append(wobble(pts, rng, wobble_frac * size_mm))
    name, _, width, _ = ink or INKS[int(rng.integers(len(INKS)))]
    return SynthMark(label, lines, width, name)


def _ink(name: str):
    return next(i for i in INKS if i[0] == name)


def render_sheet(
    marks: list[SynthMark],
    size_mm: tuple[float, float] = (210.0, 148.0),
    mm_per_px: float = 0.25,
    robot_polylines_mm: list[Polyline] | None = None,
    rng: np.random.Generator | None = None,
    robot_width_mm: float = 0.6,
) -> SynthSheet:
    rng = rng or np.random.default_rng(0)
    w = int(round(size_mm[0] / mm_per_px))
    h = int(round(size_mm[1] / mm_per_px))
    # Slightly warm paper with a lighting gradient and sensor noise.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 12.0 * (xx / w) - 8.0 * (yy / h) + rng.uniform(-4, 4)
    base = np.array([228, 234, 238], np.float32) + rng.uniform(-10, 6)
    img = base[None, None, :] + grad[..., None]

    shift = 3
    for m in marks:
        _, colour, _, alpha = _ink(m.ink)
        layer = np.zeros((h, w), np.uint8)
        t = max(1, int(round(m.width_mm / mm_per_px)))
        for line in m.polylines_mm:
            pts = np.array([mm_to_px(p, h, mm_per_px) for p in line]) * (1 << shift)
            cv2.polylines(
                layer, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 255, t, cv2.LINE_AA, shift
            )
        # Crayon and pencil are grainy: modulate coverage with noise.
        grain = rng.uniform(0.75, 1.0, (h, w)).astype(np.float32)
        a = (layer.astype(np.float32) / 255.0) * alpha * grain
        img = img * (1 - a[..., None]) + np.array(colour, np.float32)[None, None, :] * a[..., None]

    robot = robot_polylines_mm or []
    if robot:
        layer = np.zeros((h, w), np.uint8)
        t = max(1, int(round(robot_width_mm / mm_per_px)))
        for line in robot:
            pts = np.array([mm_to_px(p, h, mm_per_px) for p in line]) * (1 << shift)
            cv2.polylines(
                layer, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 255, t, cv2.LINE_AA, shift
            )
        a = layer.astype(np.float32) / 255.0 * 0.95
        img = img * (1 - a[..., None]) + np.float32(25) * a[..., None]

    img += rng.normal(0.0, 2.5, img.shape).astype(np.float32)
    image = np.clip(img, 0, 255).astype(np.uint8)
    return SynthSheet(image, mm_per_px, size_mm, marks, list(robot))


def _overlaps(b: Box, others: list[Box], gap: float) -> bool:
    return any(
        b.x < o.x2 + gap and o.x < b.x2 + gap and b.y < o.y2 + gap and o.y < b.y2 + gap
        for o in others
    )


def robot_drawing(rng: np.random.Generator, center: Point, size: float) -> list[Polyline]:
    """A few clean robot strokes: a box with a label-like underline, or an arrow."""
    cx, cy = center
    s = size / 2
    if rng.random() < 0.5:
        return [
            [
                (cx - s, cy - s),
                (cx + s, cy - s),
                (cx + s, cy + s),
                (cx - s, cy + s),
                (cx - s, cy - s),
            ],
            [(cx - s, cy - s - 4), (cx + s, cy - s - 4)],
        ]
    return [
        [(cx - s, cy), (cx + s, cy)],
        [(cx + s - 5, cy + 3), (cx + s, cy), (cx + s - 5, cy - 3)],
    ]


def make_sheet(
    rng: np.random.Generator,
    n_marks: int = 4,
    n_robot: int = 1,
    size_mm: tuple[float, float] = (210.0, 148.0),
    mm_per_px: float = 0.25,
    labels: list[str] | None = None,
    size_range: tuple[float, float] = (18.0, 45.0),
    margin_mm: float = 8.0,
    gap_mm: float = 8.0,
) -> SynthSheet:
    """A random sheet with non-overlapping user marks and robot strokes."""
    names = labels or list(SHAPES)
    boxes: list[Box] = []
    marks: list[SynthMark] = []
    robot: list[Polyline] = []
    n_robot_done = 0
    for _ in range(600):
        if len(marks) >= n_marks and n_robot_done >= n_robot:
            break
        size = float(rng.uniform(*size_range))
        half = size * 0.75
        c = (
            float(rng.uniform(margin_mm + half, size_mm[0] - margin_mm - half)),
            float(rng.uniform(margin_mm + half, size_mm[1] - margin_mm - half)),
        )
        if len(marks) >= n_marks:
            lines = robot_drawing(rng, c, size)
            xs = [p[0] for ln in lines for p in ln]
            ys = [p[1] for ln in lines for p in ln]
            b = Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            if not _overlaps(b, boxes, gap_mm):
                boxes.append(b)
                robot += lines
                n_robot_done += 1
            continue
        label = names[int(rng.integers(len(names)))]
        m = make_mark(label, c, size, rng)
        b = m.box_mm
        if b.x < margin_mm / 2 or b.y < margin_mm / 2:
            continue
        if b.x2 > size_mm[0] - margin_mm / 2 or b.y2 > size_mm[1] - margin_mm / 2:
            continue
        if _overlaps(b, boxes, gap_mm):
            continue
        boxes.append(b)
        marks.append(m)
    return render_sheet(marks, size_mm, mm_per_px, robot, rng)


RELATIONS = {
    "right": "to the right of",
    "left": "to the left of",
    "above": "above",
    "below": "below",
}


def point_queries(
    sheet: SynthSheet, min_region_mm: float = 15.0, edge_mm: float = 3.0, max_queries: int = 3
) -> list[dict]:
    """Semantic-place queries with an accepted region (pixels) that geometry can check.

    "Empty space to the right of the X": the band beside X's box, as tall as the box, out to
    the paper edge (the scorer also rejects points on any mark's box). At most one query per
    mark and `max_queries` per sheet; regions shallower than `min_region_mm` are skipped.
    "Above" means away from the user, which is up in the image.
    """
    w_mm, h_mm = sheet.size_mm
    h_px = sheet.image.shape[0]
    out = []
    order = list(RELATIONS)
    for k, m in enumerate(sheet.marks):
        if len(out) >= max_queries:
            break
        b = m.box_mm
        regions = {
            "right": Box(b.x2 + 2, b.y, w_mm - edge_mm - b.x2 - 2, b.h),
            "left": Box(edge_mm, b.y, b.x - 2 - edge_mm, b.h),
            "above": Box(b.x, b.y2 + 2, b.w, h_mm - edge_mm - b.y2 - 2),
            "below": Box(b.x, edge_mm, b.w, b.y - 2 - edge_mm),
        }
        # One query per mark, rotating through the relations so every kind gets tested.
        for rel in order[k % 4 :] + order[: k % 4]:
            region = regions[rel]
            depth = region.w if rel in ("right", "left") else region.h
            if depth < min_region_mm or region.w <= 0 or region.h <= 0:
                continue
            out.append(
                {
                    "query": f"empty space {RELATIONS[rel]} the {m.label}",
                    "target": m.label,
                    "relation": rel,
                    "accept_box_px": [
                        round(v, 1) for v in box_mm_to_px(region, h_px, sheet.mm_per_px)
                    ],
                }
            )
            break
    return out
