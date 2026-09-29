"""Find a sheet of paper in a photo and warp it to a top-down image at a known scale.

In the app the warp comes from the ArUco bed homography (vision/aruco.py). This module is for
photos without markers: rahul's 22 Sep test pages and the R1 phone photos. It assumes a light
sheet on a darker background, which is true for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """4x2 corners -> top-left, top-right, bottom-right, bottom-left (image coordinates)."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]], np.float64
    )


def find_paper_quad(image: np.ndarray, min_area_frac: float = 0.05) -> np.ndarray | None:
    """Corners (TL, TR, BR, BL) of the largest bright, smooth quadrilateral, or None.

    Paper is (1) bright in its darkest colour channel, unlike wood, which is dark in blue, and
    (2) smooth: a reflection on a desk can be bright but keeps its grain texture. On the 22 Sep
    photos, brightness alone leaked into a glare patch on 5 of 10 pages; adding the smoothness
    test (local std < 4 grey levels at 800 px) fixed all of them.
    """
    gray = image if image.ndim == 2 else image.min(axis=2)
    small_scale = 800.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA)
    f = small.astype(np.float32)
    mean = cv2.blur(f, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (7, 7)) - mean * mean, 0.0))
    _, bright = cv2.threshold(
        cv2.GaussianBlur(small, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bw = cv2.bitwise_and(bright, (std < 4.0).astype(np.uint8) * 255)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))  # drop glare specks
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # A line drawn edge to edge (a horizon) splits the sheet into pieces. Add further pieces
    # only while the merged convex hull stays solid: pieces of one sheet form a near-rectangle,
    # a glare patch next to the sheet does not.
    pts = contours[0].reshape(-1, 2)
    area = cv2.contourArea(contours[0])
    for c in contours[1:]:
        a = cv2.contourArea(c)
        if a < 0.005 * small.size:
            break
        cand = np.vstack([pts, c.reshape(-1, 2)])
        hull_area = cv2.contourArea(cv2.convexHull(cand))
        if hull_area > 0 and (area + a) / hull_area >= 0.9:
            pts, area = cand, area + a
    c = cv2.convexHull(pts.reshape(-1, 1, 2))
    if cv2.contourArea(c) < min_area_frac * small.shape[0] * small.shape[1]:
        return None
    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c))
    # The 7x7 smoothness window and the blur mark a band about 3 px wide inside the paper edge
    # as rough, so push each edge back out by that much.
    quad = offset_quad(order_corners(np.asarray(quad, np.float64)), 3.0)
    return quad / small_scale


def offset_quad(quad: np.ndarray, d: float) -> np.ndarray:
    """Move each edge of an ordered convex quad outward by `d` (negative moves inward)."""
    center = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([t[1], -t[0]])
        if np.dot(n, a - center) < 0:
            n = -n
        lines.append((a + d * n, t))
    out = []
    for i in range(4):
        (p1, t1), (p2, t2) = lines[i - 1], lines[i]
        m = np.array([t1, -t2]).T
        if abs(np.linalg.det(m)) < 1e-9:
            out.append(quad[i])
            continue
        s = np.linalg.solve(m, p2 - p1)[0]
        out.append(p1 + s * t1)
    return np.array(out)


def quad_size_px(quad: np.ndarray) -> tuple[float, float]:
    """Mean width and height of an ordered quad, in pixels."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(w), float(h)


@dataclass(frozen=True)
class WarpedPaper:
    image: np.ndarray
    mm_per_px: float
    size_mm: tuple[float, float]
    homography: np.ndarray  # photo pixels -> warped pixels

    def to_photo(self, pts_px: np.ndarray) -> np.ndarray:
        """Warped-image pixels (N x 2) -> photo pixels."""
        pts = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.homography)).reshape(-1, 2)


def warp_paper(
    image: np.ndarray,
    quad: np.ndarray,
    size_mm: tuple[float, float] | None = None,
    mm_per_px: float = 0.25,
    long_side_mm: float = 280.0,
) -> WarpedPaper:
    """Warp the paper to a top-down image.

    If `size_mm` (width, height as the user sees the sheet) is unknown, it is estimated from the
    quad's aspect ratio with the longer side set to `long_side_mm`, so the scale is nominal.
    """
    quad = order_corners(quad)
    if size_mm is None:
        wq, hq = quad_size_px(quad)
        k = long_side_mm / max(wq, hq)
        size_mm = (wq * k, hq * k)
    w_px = int(round(size_mm[0] / mm_per_px))
    h_px = int(round(size_mm[1] / mm_per_px))
    dst = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], np.float64)
    hom = cv2.getPerspectiveTransform(quad.astype(np.float32), dst.astype(np.float32))
    warped = cv2.warpPerspective(
        image, hom, (w_px, h_px), flags=cv2.INTER_AREA, borderValue=(255, 255, 255)
    )
    return WarpedPaper(warped, mm_per_px, (float(size_mm[0]), float(size_mm[1])), hom)
