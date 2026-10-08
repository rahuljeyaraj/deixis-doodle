"""Segment the user's own marks on a warped, top-down image of the paper.

Input is an image of the paper alone, warped so that it is seen from directly above, at a known
scale `mm_per_px`, and oriented as the user sees the sheet (the user at the bottom edge). Pixel
(col, row) maps to the paper frame (layout.py: +x to the user's right, +y away from the user) as

    x_mm = (col + 0.5) * mm_per_px
    y_mm = (height_px - row - 0.5) * mm_per_px

Steps (plan 4.3 "object references"):
1. Adaptive threshold on how much darker each pixel is than the local paper in its worst colour
   channel, so pale crayon (yellow, light blue, orange) shows up as well as pencil and pen.
2. Remove the robot's own strokes: render its known polylines as a dilated mask and subtract it.
3. Clean up: open (remove specks), close (join broken strokes), blank a thin border band.
4. Group connected components, merging pieces closer than `merge_gap_mm` and pieces whose box
   lies inside another mark's box (the windows of a house belong to the house).

Geometry gives precise boxes; Cosmos then labels a crop around each mark (`crop_around`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from deixis.geometry import Box, Point, Polyline


@dataclass(frozen=True)
class UserMark:
    """One mark the user drew, in paper millimetres."""

    id: int
    box_mm: Box
    area_mm2: float  # inked area, not the box area
    centroid_mm: Point  # centroid of the inked pixels
    box_px: tuple[int, int, int, int]  # x1, y1, x2, y2 in the warped image, x2/y2 exclusive


@dataclass
class InkParams:
    block_mm: float = 12.0  # neighbourhood for the local paper level (median filter)
    offset: float = 14.0  # grey levels darker than the local paper, in any channel, to be ink
    open_mm: float = 0.3
    close_mm: float = 1.0
    border_mm: float = 2.0  # ignore a band this wide at the paper edge (shadows, cut edges)
    robot_pen_width_mm: float = 0.7
    robot_dilate_mm: float = 1.5  # extra margin for calibration error around robot strokes
    min_area_mm2: float = 2.0
    # The larger side of a mark's box. 7 Oct 2026: 2.5 mm let dust, paper texture and pen
    # dots through as "marks". In one live run six specks of 1 to 8 mm each cost a blocking
    # Cosmos call, and the app froze. The 80 ground-truth marks in .cache/r1/synthetic are
    # 18 mm or bigger (median 32), so 10 mm drops the specks with room to spare.
    min_extent_mm: float = 10.0
    merge_gap_mm: float = 2.0
    merge_contained: bool = True


def _odd(n: float, minimum: int = 3) -> int:
    k = max(minimum, int(round(n)))
    return k if k % 2 == 1 else k + 1


def _kernel(mm: float, mm_per_px: float) -> np.ndarray | None:
    k = int(round(mm / mm_per_px))
    if k < 2:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def mm_to_px(p: Point, height_px: int, mm_per_px: float) -> tuple[float, float]:
    return (p[0] / mm_per_px - 0.5, height_px - p[1] / mm_per_px - 0.5)


def px_to_mm(col: float, row: float, height_px: int, mm_per_px: float) -> Point:
    return ((col + 0.5) * mm_per_px, (height_px - row - 0.5) * mm_per_px)


def box_px_to_mm(box_px: Sequence[float], height_px: int, mm_per_px: float) -> Box:
    """(x1, y1, x2, y2) pixel box with exclusive x2/y2 -> paper-frame Box."""
    x1, y1, x2, y2 = box_px
    return Box(
        x1 * mm_per_px, (height_px - y2) * mm_per_px, (x2 - x1) * mm_per_px, (y2 - y1) * mm_per_px
    )


def box_mm_to_px(box: Box, height_px: int, mm_per_px: float) -> tuple[float, float, float, float]:
    return (
        box.x / mm_per_px,
        height_px - box.y2 / mm_per_px,
        box.x2 / mm_per_px,
        height_px - box.y / mm_per_px,
    )


def robot_stroke_mask(
    shape: tuple[int, int],
    polylines_mm: Sequence[Polyline],
    mm_per_px: float,
    pen_width_mm: float = 0.7,
    dilate_mm: float = 1.5,
) -> np.ndarray:
    """uint8 mask (255 = robot ink, dilated) of the robot's known strokes."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    thickness = max(1, int(math.ceil((pen_width_mm + 2 * dilate_mm) / mm_per_px)))
    shift = 4  # sub-pixel precision for cv2.polylines
    for line in polylines_mm:
        if len(line) < 1:
            continue
        pts = np.array([mm_to_px(p, h, mm_per_px) for p in line], np.float64)
        pts = np.round(pts * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        if len(line) == 1:
            c = tuple(int(v) for v in pts[0, 0])
            cv2.circle(mask, c, (thickness // 2) << shift, 255, -1, cv2.LINE_8, shift)
        else:
            cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_8, shift)
    return mask


def darkening(image: np.ndarray, mm_per_px: float, block_mm: float = 12.0) -> np.ndarray:
    """uint8 image: how much darker each pixel is than the local paper, in its worst channel.

    The paper background is a per-channel median over `block_mm`, so uneven light and tinted
    paper cancel out. Taking the largest drop over the three channels makes pale crayon count:
    pale orange barely changes brightness but drops the blue channel a lot.
    """
    block = _odd(block_mm / mm_per_px)
    img = cv2.GaussianBlur(image, (3, 3), 0)
    bg = cv2.medianBlur(img, block)
    diff = bg.astype(np.int16) - img.astype(np.int16)
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    return np.clip(diff, 0, 255).astype(np.uint8)


def ink_mask(
    image: np.ndarray,
    mm_per_px: float,
    robot_polylines_mm: Sequence[Polyline] = (),
    params: InkParams | None = None,
) -> np.ndarray:
    """uint8 mask (255 = the user's ink) after thresholding, robot removal and cleanup."""
    return _ink_and_robot(image, mm_per_px, robot_polylines_mm, params or InkParams())[0]


def _ink_and_robot(
    image: np.ndarray, mm_per_px: float, robot_polylines_mm: Sequence[Polyline], p: InkParams
) -> tuple[np.ndarray, np.ndarray | None]:
    dark = darkening(image, mm_per_px, p.block_mm)
    mask = np.where(dark >= p.offset, 255, 0).astype(np.uint8)

    robot = None
    if robot_polylines_mm:
        robot = robot_stroke_mask(
            dark.shape, robot_polylines_mm, mm_per_px, p.robot_pen_width_mm, p.robot_dilate_mm
        )
        mask[robot > 0] = 0

    k_open = _kernel(p.open_mm, mm_per_px)
    if k_open is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    k_close = _kernel(p.close_mm, mm_per_px)
    if k_close is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    b = int(round(p.border_mm / mm_per_px))
    if b > 0:
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
    return mask, robot


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], tol: int) -> bool:
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def marks_from_mask(
    mask: np.ndarray,
    mm_per_px: float,
    params: InkParams | None = None,
    bridge: np.ndarray | None = None,
) -> list[UserMark]:
    """Group an ink mask into `UserMark`s, largest first.

    `bridge` is the removed robot-stroke mask. Where a user's line crosses a robot stroke the
    subtraction cuts it; robot pixels close to user ink are treated as connecting, so a circle
    crossed by a robot line stays one mark. The bridge is local (about one stroke width around
    the crossing), so two marks that merely touch the same long robot line stay separate.
    """
    p = params or InkParams()
    h, w = mask.shape
    gap = int(round(p.merge_gap_mm / mm_per_px))
    grouping = mask
    if gap >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap + 1, gap + 1))
        grouping = cv2.dilate(mask, k)
    if bridge is not None:
        # Radius of one full stroke width, so ink on either side reaches across the band.
        band = int(math.ceil((p.robot_pen_width_mm + 2 * p.robot_dilate_mm) / mm_per_px))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 3, 2 * band + 3))
        near_ink = cv2.dilate(mask, k)
        grouping = cv2.bitwise_or(grouping, cv2.bitwise_and(bridge, near_ink))
    n, labels = cv2.connectedComponents(grouping, connectivity=8)
    ys, xs = np.nonzero(mask)
    groups: dict[int, list[np.ndarray]] = {}
    if len(xs):
        lab = labels[ys, xs]
        order = np.argsort(lab, kind="stable")
        lab, xs, ys = lab[order], xs[order], ys[order]
        cuts = np.flatnonzero(np.diff(lab)) + 1
        for seg_x, seg_y, seg_l in zip(
            np.split(xs, cuts), np.split(ys, cuts), np.split(lab, cuts), strict=True
        ):
            groups[int(seg_l[0])] = [seg_x, seg_y]

    # Merge groups whose box sits inside another group's box.
    items = []
    for gx, gy in groups.values():
        items.append([gx, gy, (int(gx.min()), int(gy.min()), int(gx.max()) + 1, int(gy.max()) + 1)])
    if p.merge_contained and len(items) > 1:
        tol = max(1, gap)
        items.sort(key=lambda it: (it[2][2] - it[2][0]) * (it[2][3] - it[2][1]), reverse=True)
        merged: list[list] = []
        for it in items:
            for m in merged:
                if _contains(m[2], it[2], tol):
                    m[0] = np.concatenate([m[0], it[0]])
                    m[1] = np.concatenate([m[1], it[1]])
                    break
            else:
                merged.append(it)
        items = merged

    px_area = mm_per_px * mm_per_px
    marks: list[UserMark] = []
    for gx, gy, _ in items:
        x1, y1, x2, y2 = int(gx.min()), int(gy.min()), int(gx.max()) + 1, int(gy.max()) + 1
        area = len(gx) * px_area
        extent = max(x2 - x1, y2 - y1) * mm_per_px
        if area < p.min_area_mm2 or extent < p.min_extent_mm:
            continue
        cx, cy = px_to_mm(float(gx.mean()), float(gy.mean()), h, mm_per_px)
        marks.append(
            UserMark(
                id=-1,
                box_mm=box_px_to_mm((x1, y1, x2, y2), h, mm_per_px),
                area_mm2=round(area, 2),
                centroid_mm=(round(cx, 2), round(cy, 2)),
                box_px=(x1, y1, x2, y2),
            )
        )
    marks.sort(key=lambda m: m.area_mm2, reverse=True)
    return [UserMark(i, m.box_mm, m.area_mm2, m.centroid_mm, m.box_px) for i, m in enumerate(marks)]


def segment_ink(
    image: np.ndarray,
    mm_per_px: float,
    robot_polylines_mm: Sequence[Polyline] = (),
    params: InkParams | None = None,
) -> list[UserMark]:
    """The user's marks on a warped paper image, minus the robot's own strokes."""
    p = params or InkParams()
    mask, robot = _ink_and_robot(image, mm_per_px, robot_polylines_mm, p)
    return marks_from_mask(mask, mm_per_px, p, bridge=robot)


def distance_to_box(point: Point, box: Box) -> float:
    dx = max(box.x - point[0], 0.0, point[0] - box.x2)
    dy = max(box.y - point[1], 0.0, point[1] - box.y2)
    return math.hypot(dx, dy)


def mark_at(
    marks: Sequence[UserMark], point_mm: Point, max_dist_mm: float = 10.0
) -> UserMark | None:
    """The mark nearest a pointed spot (0 inside its box), if within `max_dist_mm`.

    Ties (nested or overlapping boxes) go to the mark whose centroid is closer.
    """
    best = None
    best_key = (math.inf, math.inf)
    for m in marks:
        d = distance_to_box(point_mm, m.box_mm)
        if d > max_dist_mm:
            continue
        key = (d, math.dist(point_mm, m.centroid_mm))
        if key < best_key:
            best, best_key = m, key
    return best


@dataclass(frozen=True)
class Crop:
    image: np.ndarray
    box_mm: Box  # the paper-frame area the crop covers (may extend past the paper)
    origin_px: tuple[int, int]  # warped-image pixel of the crop's top-left corner (may be < 0)


def crop_around(
    image: np.ndarray,
    point_mm: Point,
    size_mm: float,
    mm_per_px: float,
    pad_value: int = 255,
) -> Crop:
    """A square crop of side `size_mm` centred on `point_mm`, padded with white past the edges.

    Used to send only the pointed region to Cosmos `label_crop` (plan 4.5 privacy rule too).
    """
    h, w = image.shape[:2]
    side = max(1, int(round(size_mm / mm_per_px)))
    cx, cy = mm_to_px(point_mm, h, mm_per_px)
    x0 = int(round(cx + 0.5 - side / 2))
    y0 = int(round(cy + 0.5 - side / 2))
    shape = (side, side) + image.shape[2:]
    out = np.full(shape, pad_value, dtype=image.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    box = box_px_to_mm((x0, y0, x0 + side, y0 + side), h, mm_per_px)
    return Crop(image=out, box_mm=box, origin_px=(x0, y0))


def crop_for_mark(
    image: np.ndarray, mark: UserMark, mm_per_px: float, pad: float = 1.4, min_mm: float = 25.0
) -> Crop:
    """A square crop around a mark, `pad` times its larger side (at least `min_mm`)."""
    side = max(min_mm, pad * max(mark.box_mm.w, mark.box_mm.h))
    return crop_around(image, mark.box_mm.center, side, mm_per_px)
