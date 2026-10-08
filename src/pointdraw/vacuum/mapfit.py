"""The floor from the robot alone: no paper markers, one calibration drive.

The camera watches the marker on the robot's lid while the robot drives, and four times a
second the robot says where it is on its own map, in mm. Those pairs fit a homography
straight from pixels to the robot's map (`fit_px_to_map`), which is what the app needs:
somewhere on the floor in the picture -> somewhere the robot can be sent.

Two corrections make it honest:

- **The marker is not the robot.** It sits `offset_mm` ahead of the robot's centre, in front
  of the LiDAR turret, so it swings as the robot turns. The fit is repeated two or three
  times: each round uses the current homography to read the marker's heading off its own
  corners, works out where the marker's centre must be on the map, and fits again.

- **The lid is not the floor.** The marker rides about 100 mm up, so a homography fitted from
  it reads the *lid* plane. A foot is on the floor, one ray further out: at a camera height H
  and a lid height h, a floor point sits H / (H - h) further from the spot under the camera
  than the lid point on the same ray. `lift_to_floor` scales about that spot, which puts feet
  where they really are. The spot and H come from the fit itself (`camera_pose`), or H can be
  given by eye -- at h = 100 mm it is a 4 % correction, so H to the nearest 20 cm is plenty.

The floor frame is then a turn and a shift of the robot's map (`floor_frame`), so that x runs
to the viewer's right and y away from them, with the origin at a corner of the patch the
robot covered. Distances in it are the robot's own millimetres.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from deixis.frames import apply_matrix

Pair = tuple[tuple[float, float], tuple[float, float]]  # (pixels, map mm)

LID_HEIGHT_MM = 100.0  # the lid marker's height above the floor (N50 PRO OMNI, with turret)
FOCAL_PER_WIDTH = 1.05  # a plain webcam's focal length in pixels, as a share of image width


def _homography(pairs: Sequence[Pair]) -> tuple[np.ndarray, float] | None:
    """Pixels -> map mm, and the rms of the fit in mm. None if it will not solve."""
    if len(pairs) < 4:
        return None
    px = np.array([p for p, _ in pairs], np.float64)
    mm = np.array([m for _, m in pairs], np.float64)
    H, _ = cv2.findHomography(px, mm, cv2.RANSAC, 50.0)
    if H is None:
        return None
    err = np.linalg.norm(apply_matrix(H, px) - mm, axis=1)
    return H, float(np.sqrt(float(np.mean(err**2))))


def marker_front(corners_px: np.ndarray, H: np.ndarray) -> tuple[float, float] | None:
    """The way the lid marker's printed top points, as a unit vector in map mm."""
    c = apply_matrix(H, np.asarray(corners_px, np.float64).reshape(4, 2))
    front = (c[0] + c[1]) / 2 - (c[2] + c[3]) / 2  # printed bottom -> top
    n = float(np.linalg.norm(front))
    return None if n <= 0 else (float(front[0] / n), float(front[1] / n))


@dataclass
class Sighting:
    """One frame: where the lid marker's corners were, and where the robot said it was."""

    corners_px: np.ndarray  # (4, 2), TL TR BR BL
    map_mm: tuple[float, float]  # the robot's centre, on its own map

    @property
    def centre_px(self) -> tuple[float, float]:
        c = np.asarray(self.corners_px, np.float64).reshape(4, 2).mean(axis=0)
        return (float(c[0]), float(c[1]))


def fit_px_to_map(
    sightings: Sequence[Sighting], offset_mm: float = 0.0, rounds: int = 3
) -> tuple[np.ndarray, float] | None:
    """Pixels -> the robot's map, with the lid marker's forward offset taken out.

    The first round pairs the marker's centre with the robot's centre, which is wrong by
    `offset_mm` in whichever way the robot was facing. Each round after it reads the heading
    off the marker's own corners and moves the map point to where the marker must have been.
    """
    pairs: list[Pair] = [(s.centre_px, s.map_mm) for s in sightings]
    out = _homography(pairs)
    if out is None or offset_mm == 0.0:
        return out
    for _ in range(max(0, rounds - 1)):
        H = out[0]
        pairs = []
        for s in sightings:
            f = marker_front(s.corners_px, H)
            if f is None:
                continue
            mx, my = s.map_mm
            pairs.append((s.centre_px, (mx + f[0] * offset_mm, my + f[1] * offset_mm)))
        nxt = _homography(pairs)
        if nxt is None:
            break
        out = nxt
    return out


def camera_pose(
    H_px_to_plane: np.ndarray, image_size: tuple[int, int], focal_px: float | None = None
) -> tuple[float, tuple[float, float]] | None:
    """How high the camera is above the plane it was fitted on, and the spot under it.

    Returns (height mm, (x, y) in plane mm). The focal length is guessed from the image width
    unless given; it only moves the height, and the height only moves a 4 % correction, so a
    rough one is fine.
    """
    w, h = image_size
    f = float(focal_px) if focal_px else FOCAL_PER_WIDTH * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    try:
        G = np.linalg.inv(np.asarray(H_px_to_plane, np.float64))  # plane mm -> pixels
    except np.linalg.LinAlgError:
        return None
    A = np.linalg.inv(K) @ G
    n1 = float(np.linalg.norm(A[:, 0]))
    n2 = float(np.linalg.norm(A[:, 1]))
    if n1 <= 0 or n2 <= 0:
        return None
    lam = 2.0 / (n1 + n2)
    r1, r2, t = A[:, 0] * lam, A[:, 1] * lam, A[:, 2] * lam
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    u, _, vt = np.linalg.svd(R)  # the nearest true rotation
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    centre = -R.T @ t  # the camera, in plane mm
    height = abs(float(centre[2]))
    if not math.isfinite(height) or height <= 0:
        return None
    return height, (float(centre[0]), float(centre[1]))


def lift_to_floor(
    H_px_to_lid: np.ndarray,
    nadir_mm: tuple[float, float],
    camera_height_mm: float,
    lid_height_mm: float = LID_HEIGHT_MM,
) -> np.ndarray:
    """Pixels -> the floor, from a fit made on the lid plane.

    `camera_height_mm` is above the **floor**. A floor point is H / (H - h) further from the
    spot under the camera than the lid point the same ray passes through.
    """
    big, small = float(camera_height_mm), float(lid_height_mm)
    if big <= small:
        raise ValueError(f"the camera ({big:.0f} mm) must be above the lid ({small:.0f} mm)")
    k = big / (big - small)
    cx, cy = nadir_mm
    S = np.array([[k, 0, cx * (1 - k)], [0, k, cy * (1 - k)], [0, 0, 1]], np.float64)
    return S @ np.asarray(H_px_to_lid, np.float64)


def floor_frame(
    H_px_to_map: np.ndarray, image_size: tuple[int, int], covered_mm: np.ndarray
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Turn and shift the robot's map so x runs right and y away, as the camera sees it.

    Returns (map mm -> floor mm, the patch as (x, y, w, h) in floor mm). "Right" and "away"
    are taken from the picture: the image's own axes at the middle of the frame.
    """
    w, h = image_size
    probe = np.array([[w / 2, h / 2], [w / 2 + 100, h / 2], [w / 2, h / 2 - 100]], np.float64)
    p = apply_matrix(np.asarray(H_px_to_map, np.float64), probe)
    right = p[1] - p[0]
    n = float(np.linalg.norm(right))
    if n <= 0:
        raise ValueError("the fit gives no direction on the floor")
    ang = math.atan2(float(right[1] / n), float(right[0] / n))
    c, s = math.cos(-ang), math.sin(-ang)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)
    away = apply_matrix(R, (p[2] - p[0]).reshape(1, 2))[0]
    if away[1] < 0:  # make y run away from the viewer, not toward them
        R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]], np.float64) @ R
    pts = apply_matrix(R, np.asarray(covered_mm, np.float64).reshape(-1, 2))
    lo = pts.min(axis=0)
    R[0, 2] -= float(lo[0])
    R[1, 2] -= float(lo[1])
    hi = (pts - lo).max(axis=0)
    return R, (0.0, 0.0, float(hi[0]), float(hi[1]))


def pair_with_lag(
    track: Sequence[tuple[float, np.ndarray]],
    reports: Sequence[tuple[float, tuple[float, float]]],
    lag_s: float,
    max_gap_s: float = 0.25,
) -> list[Sighting]:
    """Match each of the robot's reports to the frame the camera took at the same moment.

    The robot's reports arrive late by an amount nobody knows, so the caller tries several
    lags and keeps the tightest (`best_lag`). A report is used only if a frame was taken
    within `max_gap_s` of it.
    """
    if not track:
        return []
    ts = np.array([t for t, _ in track])
    out: list[Sighting] = []
    for t, map_mm in reports:
        i = int(np.argmin(np.abs(ts - (t - lag_s))))
        if abs(float(ts[i]) - (t - lag_s)) <= max_gap_s:
            out.append(Sighting(track[i][1], map_mm))
    return out


def best_lag(
    track: Sequence[tuple[float, np.ndarray]],
    reports: Sequence[tuple[float, tuple[float, float]]],
    offset_mm: float = 0.0,
    lags: Sequence[float] | None = None,
    min_pairs: int = 8,
    max_gap_s: float = 0.25,
) -> tuple[float, list[Sighting], np.ndarray, float] | None:
    """(lag, its sightings, pixels -> map, rms mm) for the lag that fits tightest."""
    best: tuple[float, list[Sighting], np.ndarray, float] | None = None
    for lag in lags if lags is not None else np.arange(0.0, 3.01, 0.1):
        seen = pair_with_lag(track, reports, float(lag), max_gap_s)
        if len(seen) < min_pairs:
            continue
        fit = fit_px_to_map(seen, offset_mm)
        if fit is None:
            continue
        H, rms = fit
        if best is None or rms < best[3]:
            best = (float(lag), seen, H, rms)
    return best


def best_offset(
    track: Sequence[tuple[float, np.ndarray]],
    reports: Sequence[tuple[float, tuple[float, float]]],
    offsets: Sequence[float] | None = None,
    lags: Sequence[float] | None = None,
    min_pairs: int = 8,
    max_gap_s: float = 0.25,
) -> tuple[float, float, list[Sighting], np.ndarray, float] | None:
    """Let the drive say how far ahead the marker sits: the offset that fits tightest.

    A wrong offset puts the marker's map point out by that much, in whichever way the robot
    was facing, so the rms rises on both sides of the right answer. Returns
    (offset, lag, sightings, pixels -> map, rms mm).
    """

    def scan(
        try_offsets: Sequence[float], try_lags: Sequence[float] | None
    ) -> tuple[float, float, list[Sighting], np.ndarray, float] | None:
        found: tuple[float, float, list[Sighting], np.ndarray, float] | None = None
        for off in try_offsets:
            got = best_lag(track, reports, float(off), try_lags, min_pairs, max_gap_s)
            if got is None:
                continue
            lag, seen, H, rms = got
            if found is None or rms < found[4]:
                found = (float(off), lag, seen, H, rms)
        return found

    if offsets is not None:
        return scan(offsets, lags)
    # Coarse, then close in: the full grid of offsets against the full grid of lags is a few
    # hundred fits and stalls the window it is drawn in.
    coarse = scan(np.arange(0.0, 171.0, 40.0), lags)
    if coarse is None:
        return None
    near = np.arange(max(0.0, coarse[0] - 40.0), coarse[0] + 41.0, 10.0)
    return scan(near, [coarse[1]]) or coarse


MIN_PAIRS = 40
MIN_LONG_MM = 1500.0  # how far the robot drove, along its longest line
MIN_WIDE_MM = 600.0  # and across it: a straight line cannot fix a homography at all
MAX_RMS_MM = 80.0


def readiness(
    seen: Sequence[Sighting],
    rms: float,
    min_pairs: int = MIN_PAIRS,
    min_long_mm: float = MIN_LONG_MM,
    min_wide_mm: float = MIN_WIDE_MM,
    max_rms_mm: float = MAX_RMS_MM,
) -> tuple[bool, str]:
    """Is the drive enough to fit on, and if not, what should the robot do about it?

    A low rms is not enough. A homography has 8 free numbers, so a handful of positions fit
    almost perfectly and say nothing; and a robot that drove in a straight line cannot fix
    one at all, however many positions it sent. So this asks for a count, for distance
    along the drive's longest line, and for distance **across** it.
    """
    n = len(seen)
    if n < min_pairs:
        return False, f"Keep driving: {n} of {min_pairs} positions"
    pts = np.array([s.map_mm for s in seen], np.float64)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vt[0]
    across = centred @ vt[1]
    long_mm = float(along.max() - along.min())
    wide_mm = float(across.max() - across.min())
    if long_mm < min_long_mm:
        return False, f"Drive further: {long_mm / 1000:.1f} m of {min_long_mm / 1000:.1f} m"
    if wide_mm < min_wide_mm:
        return False, f"Too straight a line: drive across it too ({wide_mm / 1000:.1f} m wide)"
    if rms > max_rms_mm:
        return False, f"The fit is loose: {rms:.0f} mm (want under {max_rms_mm:.0f})"
    return True, f"Ready: press F to save ({n} positions, {rms:.0f} mm)"
