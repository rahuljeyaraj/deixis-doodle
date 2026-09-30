"""Calibration from crosses the plotter draws: pixels -> machine mm, and marker corners in mm.

Procedure (plan section 4.3, T1.2):
1. The controller draws a grid of crosses (normally 3x3) at known machine XY and parks.
2. The camera takes one image. `detect_crosses` finds the cross centres in pixels.
3. `match_crosses` pairs detections with the known machine points. It needs no prior guess:
   it tries every rotation and mirror image of the outer hull, so the camera can be mounted
   any way round.
4. `fit_cross_homography` fits pixels -> machine mm with RANSAC and reports residuals in mm.
5. `learn_marker_corners` maps each visible bed marker's pixel corners through that fit, so
   later frames can recover the bed frame from the markers alone (`vision.aruco`).

The bed frame is machine XY because the paper is fixed to the bed. Points should be
undistorted first when a lens model is available (`vision.lens`).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pointdraw.vision.aruco import (
    BedCalibration,
    MarkerDetection,
    apply_homography,
    detect_markers,
)
from pointdraw.vision.lens import LensModel


class CalibrationError(RuntimeError):
    pass


# --- cross detection ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossParams:
    """Cross detector settings, in pixels. Defaults suit 1.5 to 6 px/mm and 6 to 20 mm crosses."""

    min_arm_px: float = 5.0  # half-length of each stroke, minimum
    max_arm_px: float = 120.0
    block_size: int = 31  # adaptive-threshold neighbourhood (odd)
    c: float = 8.0  # adaptive-threshold offset; higher ignores fainter ink
    max_fill: float = 0.5  # ink pixels / bounding-box area; crosses are sparse
    min_angle_deg: float = 50.0  # between the two strokes
    min_inlier_frac: float = 0.85  # of the component's pixels explained by the two strokes
    paper_only: bool = True  # search only inside the largest bright region
    marker_margin_px: float = 8.0  # ignore ink this close to detected markers


DEFAULT_CROSS_PARAMS = CrossParams()


@dataclass(frozen=True)
class Cross:
    center_px: tuple[float, float]
    arm_px: float  # mean half-length of the two strokes
    angle_deg: float  # angle between the strokes


def paper_mask(gray: np.ndarray, erode_px: int = 6) -> np.ndarray | None:
    """Mask of the largest bright region (the paper), or None if nothing plausible is found."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.03 * gray.size:
        return None
    region = (labels == i).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(region)
    cv2.drawContours(mask, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 1, -1)
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1,) * 2, np.uint8))
    return mask.astype(bool)


def _fit_line_pca(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line: (point on line, unit direction)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _line_dist(pts: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    v = pts - c
    return np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])


def _ransac_line(
    pts: np.ndarray, tol: float, min_sep: float, rng: np.random.Generator, iters: int = 150
) -> np.ndarray | None:
    """Boolean inlier mask of the best line through pts, refined by PCA."""
    n = len(pts)
    if n < 4:
        return None
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        d = pts[j] - pts[i]
        norm = math.hypot(d[0], d[1])
        if norm < min_sep:
            continue
        inl = _line_dist(pts, pts[i], d / norm) <= tol
        count = int(inl.sum())
        if count > best_count:
            best, best_count = inl, count
    if best is None:
        return None
    for _ in range(2):  # refine: refit on inliers, recompute inliers
        c, d = _fit_line_pca(pts[best])
        best = _line_dist(pts, c, d) <= tol
    return best


def _intersect(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    a = np.array([d1, -d2]).T
    s = np.linalg.solve(a, c2 - c1)
    return c1 + s[0] * d1


def _analyse_component(
    pts: np.ndarray, stroke_px: float, p: CrossParams, rng: np.random.Generator
) -> Cross | None:
    tol = max(1.2, 0.6 * stroke_px + 0.5)
    inl1 = _ransac_line(pts, tol, p.min_arm_px, rng)
    if inl1 is None:
        return None
    rest = pts[~inl1]
    # Pixels near the first line's centre overlap both strokes; line 2 is fitted to the rest.
    inl2_rest = _ransac_line(rest, tol, p.min_arm_px, rng)
    if inl2_rest is None:
        return None
    c1, d1 = _fit_line_pca(pts[inl1])
    c2, d2 = _fit_line_pca(rest[inl2_rest])
    cosang = abs(float(np.dot(d1, d2)))
    angle = math.degrees(math.acos(min(1.0, cosang)))
    if angle < p.min_angle_deg:
        return None
    explained = (_line_dist(pts, c1, d1) <= tol) | (_line_dist(pts, c2, d2) <= tol)
    if explained.mean() < p.min_inlier_frac:
        return None
    x = _intersect(c1, d1, c2, d2)
    arms = []
    for d, sel in ((d1, pts[inl1]), (d2, rest[inl2_rest])):
        s = (sel - x) @ d
        lo, hi = float(s.min()), float(s.max())
        # A cross has ink on both sides of the centre (rules out L and T shapes).
        if -lo < p.min_arm_px or hi < p.min_arm_px:
            return None
        if min(-lo, hi) < 0.4 * max(-lo, hi):
            return None
        arms.append((hi - lo) / 2)
    arm = float(np.mean(arms))
    if not (p.min_arm_px <= arm <= p.max_arm_px):
        return None
    return Cross((float(x[0]), float(x[1])), arm, angle)


def ink_mask(
    gray: np.ndarray,
    params: CrossParams = DEFAULT_CROSS_PARAMS,
    markers: list[MarkerDetection] | None = None,
) -> np.ndarray:
    """Binary uint8 mask (0/1) of dark marks, limited to the paper and away from markers."""
    ink = cv2.adaptiveThreshold(
        gray,
        1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params.block_size,
        params.c,
    )
    if params.paper_only:
        m = paper_mask(gray)
        if m is not None:
            ink[~m] = 0
    for mk in markers or []:
        quad = mk.corners_px.astype(np.float32)
        centre = quad.mean(axis=0)
        grow = quad + (quad - centre) / np.linalg.norm(quad - centre, axis=1, keepdims=True) * (
            params.marker_margin_px * 1.5
        )
        cv2.fillConvexPoly(ink, np.round(grow).astype(np.int32), 0)
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def detect_crosses(
    image: np.ndarray,
    params: CrossParams = DEFAULT_CROSS_PARAMS,
    markers: list[MarkerDetection] | None = None,
    seed: int = 0,
) -> list[Cross]:
    """Find plotter-drawn crosses ("+" or "x") and return their centres, sub-pixel."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray, params, markers)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    rng = np.random.default_rng(seed)
    out: list[Cross] = []
    lo, hi = 2 * params.min_arm_px, 2 * params.max_arm_px * 1.5
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        size = max(w, h)
        if not (lo <= size <= hi) or min(w, h) < 0.3 * size:
            continue  # wrong size, or a straight line
        if area / float(w * h) > params.max_fill:
            continue
        comp = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        stroke = max(1.0, 2.0 * float(np.percentile(dist[comp > 0], 90)))
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs + x, ys + y], axis=1).astype(np.float64)
        cross = _analyse_component(pts, stroke, params, rng)
        if cross is not None:
            out.append(cross)
    return out


# --- matching and fitting ----------------------------------------------------------------------


def _extremes(mm: np.ndarray) -> np.ndarray:
    """Indices of four extreme points (corners of a grid), in counter-clockwise order."""
    c = mm - mm.mean(axis=0)
    idx = [
        int(np.argmax(c[:, 0] + c[:, 1])),
        int(np.argmax(-c[:, 0] + c[:, 1])),
        int(np.argmax(-c[:, 0] - c[:, 1])),
        int(np.argmax(c[:, 0] - c[:, 1])),
    ]
    return np.array(idx)


def _corner_candidates(px: np.ndarray, layers: int = 2, cap: int = 12) -> list[int]:
    """Points that could be the grid's outer corners: the outer two convex-hull layers.

    One hull layer is not enough: a stray detection outside the grid can hide a true corner
    from the hull, but then that corner is on the second layer.
    """
    remaining = list(range(len(px)))
    out: list[int] = []
    for _ in range(layers):
        if len(remaining) < 3:
            out += remaining
            break
        hull = cv2.convexHull(px[remaining].astype(np.float32), returnPoints=False).ravel()
        layer = [remaining[k] for k in hull.tolist()]
        out += layer
        remaining = [k for k in remaining if k not in layer]
    if len(out) > cap:  # keep the ones farthest from the centroid
        c = px.mean(axis=0)
        out = sorted(out, key=lambda k: -float(np.linalg.norm(px[k] - c)))[:cap]
    return out


def _nearest_pairs(
    proj: np.ndarray, det: np.ndarray, tol: float
) -> tuple[list[tuple[int, int]], float]:
    """Greedy one-to-one nearest matching of projected known points to detections."""
    d = np.linalg.norm(proj[:, None, :] - det[None, :, :], axis=2)
    pairs: list[tuple[int, int]] = []
    used_i: set[int] = set()
    used_j: set[int] = set()
    err = 0.0
    for flat in np.argsort(d, axis=None):
        j_known, i_det = np.unravel_index(flat, d.shape)
        if d[j_known, i_det] > tol:
            break
        if i_det in used_i or j_known in used_j:
            continue
        used_i.add(int(i_det))
        used_j.add(int(j_known))
        pairs.append((int(i_det), int(j_known)))
        err += float(d[j_known, i_det])
    return pairs, err / max(1, len(pairs))


@dataclass(frozen=True)
class OrientationPrior:
    """How the camera is expected to see the bed, used only to break ties.

    A square grid of crosses looks the same rotated by 90 degrees, so the image alone cannot
    say which way round it is. `up` is the machine axis that points to the top of the image
    ("+y" when the user sits at the bottom of the image). `mirrored=False` means an ordinary
    camera looking down at a right-handed bed frame (x right, y away from the user), which
    always maps with a negative determinant; set True for a mirrored (selfie) feed, or None
    to allow both.
    """

    up: str = "+y"
    mirrored: bool | None = False


DEFAULT_PRIOR = OrientationPrior()


def _orientation(H_mm_to_px: np.ndarray, centre: np.ndarray) -> tuple[float, dict[str, float]]:
    """(det of the local Jacobian, cosine of each machine axis with image-up)."""
    p0, px_, py_ = apply_homography(
        H_mm_to_px, np.array([centre, centre + (1.0, 0.0), centre + (0.0, 1.0)])
    )
    dx, dy = px_ - p0, py_ - p0
    det = float(dx[0] * dy[1] - dx[1] * dy[0])

    def up(v: np.ndarray) -> float:
        return float(-v[1] / (np.linalg.norm(v) + 1e-12))

    return det, {"+x": up(dx), "-x": -up(dx), "+y": up(dy), "-y": -up(dy)}


def match_crosses(
    px: np.ndarray,
    mm: np.ndarray,
    H_guess_px_to_mm: np.ndarray | None = None,
    tol_frac: float = 0.3,
    prior: OrientationPrior | None = DEFAULT_PRIOR,
) -> list[tuple[int, int]]:
    """Pair detected pixel points with known mm points. Returns (index in px, index in mm).

    Without a guess, every choice of 4 detected hull vertices, in all 4 rotations and both
    mirror orders, is tried against the 4 extreme known points; the hypothesis that explains
    the most detections wins. Symmetric patterns give ties, which `prior` breaks (handedness
    first, then which axis points up). With `prior=None` a tie raises CalibrationError.
    An asymmetric pattern (for example a 3x3 grid plus one extra cross inside a corner cell)
    needs no prior. `tol_frac` is the match tolerance as a fraction of the point spacing.
    """
    px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    mm = np.asarray(mm, dtype=np.float64).reshape(-1, 2)
    if len(px) < 4 or len(mm) < 4:
        raise CalibrationError(f"need at least 4 crosses, found {len(px)} (expected {len(mm)})")
    dmm = np.linalg.norm(mm[:, None] - mm[None], axis=2)
    np.fill_diagonal(dmm, np.inf)
    spacing_mm = float(np.median(dmm.min(axis=1)))

    if H_guess_px_to_mm is not None:
        pairs, _ = _nearest_pairs(mm, apply_homography(H_guess_px_to_mm, px), tol_frac * spacing_mm)
        return pairs

    dpx = np.linalg.norm(px[:, None] - px[None], axis=2)
    np.fill_diagonal(dpx, np.inf)
    tol_px = tol_frac * float(np.median(dpx.min(axis=1)))
    ext = _extremes(mm)
    if len(set(ext.tolist())) < 4:
        raise CalibrationError("known points need four distinct extreme corners (use a grid)")
    hyps: dict[frozenset[tuple[int, int]], tuple[int, float, list[tuple[int, int]]]] = {}
    for combo in itertools.combinations(_corner_candidates(px), 4):
        # Order the four points around their centroid so they form a quadrilateral.
        c = px[list(combo)].mean(axis=0)
        ordered = sorted(combo, key=lambda k: math.atan2(px[k, 1] - c[1], px[k, 0] - c[0]))
        for mirror in (False, True):
            seq = ordered[::-1] if mirror else ordered
            for r in range(4):
                quad = seq[r:] + seq[:r]
                try:
                    H = cv2.getPerspectiveTransform(
                        mm[ext].astype(np.float32), px[quad].astype(np.float32)
                    )
                except cv2.error:
                    continue
                if H is None or not np.all(np.isfinite(H)):
                    continue
                pairs, err = _nearest_pairs(apply_homography(H, mm), px, tol_px)
                if len(pairs) >= 4:
                    key = frozenset(pairs)
                    if key not in hyps or err < hyps[key][1]:
                        hyps[key] = (len(pairs), err, pairs)
    if not hyps:
        raise CalibrationError("could not match the detected crosses to the known grid")
    top = max(h[0] for h in hyps.values())
    min_err = min(h[1] for h in hyps.values() if h[0] == top)
    tied = [h for h in hyps.values() if h[0] == top and h[1] <= 1.5 * min_err + 0.5]
    if len(tied) == 1:
        return tied[0][2]
    if prior is None:
        raise CalibrationError(
            f"the cross pattern is symmetric ({len(tied)} equally good matches): give an "
            "orientation prior or a guess, or add an asymmetric extra cross"
        )
    centre = mm.mean(axis=0)
    scored = []
    for _, err, pairs in tied:
        i_px = [i for i, _ in pairs]
        i_mm = [j for _, j in pairs]
        H, _ = cv2.findHomography(mm[i_mm], px[i_px], 0)
        if H is None:
            continue
        det, ups = _orientation(H, centre)
        hand_ok = prior.mirrored is None or ((det > 0) == prior.mirrored)
        scored.append((hand_ok, ups.get(prior.up, 0.0), -err, pairs))
    if not scored:
        raise CalibrationError("could not match the detected crosses to the known grid")
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
    return scored[0][3]


@dataclass
class CrossFit:
    H: np.ndarray  # pixels -> mm
    px: np.ndarray  # (K, 2) matched pixel points
    mm: np.ndarray  # (K, 2) matched known points
    residuals_mm: np.ndarray  # (K,)
    inliers: np.ndarray  # (K,) bool
    affine_rms_mm: float  # an affine fit's RMS on the same inliers, for comparison

    @property
    def rms_mm(self) -> float:
        r = self.residuals_mm[self.inliers]
        return float(np.sqrt(np.mean(r**2)))

    @property
    def max_mm(self) -> float:
        return float(self.residuals_mm[self.inliers].max())

    @property
    def median_mm(self) -> float:
        return float(np.median(self.residuals_mm[self.inliers]))


def fit_cross_homography(px: np.ndarray, mm: np.ndarray, ransac_mm: float = 1.5) -> CrossFit:
    """Fit pixels -> mm. RANSAC (threshold in mm) when there are 6 or more points."""
    px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    mm = np.asarray(mm, dtype=np.float64).reshape(-1, 2)
    if len(px) < 4:
        raise CalibrationError(f"need at least 4 matched crosses, have {len(px)}")
    inl = np.ones(len(px), bool)
    if len(px) >= 6:
        H, mask = cv2.findHomography(px, mm, cv2.RANSAC, ransac_mm)
        if H is not None and mask is not None and mask.sum() >= 4:
            inl = mask.ravel().astype(bool)
    H, _ = cv2.findHomography(px[inl], mm[inl], 0)
    if H is None:
        raise CalibrationError("homography fit failed (points collinear?)")
    res = np.linalg.norm(apply_homography(H, px) - mm, axis=1)
    A, _ = cv2.estimateAffine2D(px[inl], mm[inl], method=cv2.LMEDS)
    if A is not None:
        aff = px[inl] @ A[:, :2].T + A[:, 2]
        affine_rms = float(np.sqrt(np.mean(np.sum((aff - mm[inl]) ** 2, axis=1))))
    else:
        affine_rms = math.nan
    return CrossFit(H, px, mm, res, inl, affine_rms)


def learn_marker_corners(
    detections: list[MarkerDetection], H_px_to_mm: np.ndarray
) -> dict[int, np.ndarray]:
    """Each visible marker's four corners in bed mm, via the cross fit."""
    return {d.id: apply_homography(H_px_to_mm, d.corners_px) for d in detections}


# --- one-shot calibration ----------------------------------------------------------------------


@dataclass
class CalibrationResult:
    calibration: BedCalibration
    fit: CrossFit
    crosses: list[Cross]
    markers: list[MarkerDetection]
    pairs: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> str:
        f = self.fit
        return (
            f"{int(f.inliers.sum())}/{len(f.px)} crosses used; residual rms {f.rms_mm:.2f} mm, "
            f"median {f.median_mm:.2f}, max {f.max_mm:.2f} (affine rms {f.affine_rms_mm:.2f}); "
            f"markers learned: {sorted(self.calibration.marker_corners_mm)}"
        )


def calibrate_from_image(
    image: np.ndarray,
    machine_mm: np.ndarray,
    params: CrossParams = DEFAULT_CROSS_PARAMS,
    H_guess_px_to_mm: np.ndarray | None = None,
    lens: LensModel | None = None,
    prior: OrientationPrior | None = DEFAULT_PRIOR,
) -> CalibrationResult:
    """Detect crosses and markers in one image and build a `BedCalibration`.

    With a `lens` model, cross centres and marker corners are undistorted before fitting, and
    the saved homography and marker corners then refer to undistorted pixels.
    """
    markers = detect_markers(image)
    crosses = detect_crosses(image, params, markers)
    if len(crosses) < 4:
        raise CalibrationError(
            f"found {len(crosses)} crosses, need at least 4: check lighting, focus and that the "
            "whole grid is in view"
        )
    px = np.array([c.center_px for c in crosses])

    def fix(p: np.ndarray) -> np.ndarray:
        return p if lens is None else lens.undistort(p)

    px_u = fix(px)
    machine_mm = np.asarray(machine_mm, dtype=np.float64).reshape(-1, 2)
    pairs = match_crosses(px_u, machine_mm, H_guess_px_to_mm, prior=prior)
    ip = [i for i, _ in pairs]
    im = [j for _, j in pairs]
    fit = fit_cross_homography(px_u[ip], machine_mm[im])
    mk_u = [MarkerDetection(m.id, fix(m.corners_px)) for m in markers]
    calib = BedCalibration(
        marker_corners_mm=learn_marker_corners(mk_u, fit.H),
        H_px_to_mm=fit.H,
        image_size=(image.shape[1], image.shape[0]),
        cross_rms_mm=fit.rms_mm,
        cross_max_mm=fit.max_mm,
        created=_dt.datetime.now().isoformat(timespec="seconds"),
        notes={"n_crosses": len(pairs), "undistorted": lens is not None},
    )
    return CalibrationResult(calib, fit, crosses, markers, pairs)


def load_cross_points(path: str | Path) -> np.ndarray:
    """Read known cross positions (machine mm) from JSON.

    Accepts a list of [x, y], a list of {"x", "y"} objects, or an object holding such a list
    under "crosses", "points" or "targets" (so the controller's output format can vary).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("crosses", "points", "targets", "crosses_mm"):
            if key in data:
                data = data[key]
                break
        else:
            raise CalibrationError(f"{path}: no 'crosses' or 'points' list")
    pts = []
    for item in data:
        if isinstance(item, dict):
            pts.append((float(item["x"]), float(item["y"])))
        else:
            pts.append((float(item[0]), float(item[1])))
    return np.array(pts, dtype=np.float64)
