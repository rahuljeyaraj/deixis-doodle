"""Synthetic overhead images of the bed: markers and crosses under a known camera mapping.

Used by tests and for trying the calibration pipeline without hardware. The scene is drawn
in bed mm on a fine canvas, then warped into the camera image with a known homography
(bed mm -> pixels), optionally with radial lens distortion, blur and noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pointdraw import layout


def marker_corners_mm(center: tuple[float, float], size: float) -> np.ndarray:
    """Corners of an upright marker in ArUco order (TL, TR, BR, BL as printed).

    The bed frame is right-handed seen from above (x right, y away from the user) and the
    marker's printed top faces +y, so its top-left corner is at (-h, +h) from the centre.
    """
    cx, cy = center
    h = size / 2
    return np.array([[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]])


def grid_points(
    center: tuple[float, float], nx: int = 3, ny: int = 3, spacing: float = 50.0
) -> np.ndarray:
    xs = (np.arange(nx) - (nx - 1) / 2) * spacing + center[0]
    ys = (np.arange(ny) - (ny - 1) / 2) * spacing + center[1]
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


def camera_homography(bed_quad_mm: np.ndarray, image_quad_px: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        np.asarray(bed_quad_mm, np.float32), np.asarray(image_quad_px, np.float32)
    )


@dataclass
class Lens:
    """Simple radial distortion (OpenCV model, k1 and k2 only) about the image centre."""

    k1: float = 0.0
    k2: float = 0.0
    f_px: float = 1000.0  # focal length in pixels

    def K(self, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return np.array([[self.f_px, 0, w / 2], [0, self.f_px, h / 2], [0, 0, 1]], np.float64)

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, 0.0, 0.0, 0.0])

    def distort(self, pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Ideal pixels -> distorted pixels (what a real lens would show)."""
        K = self.K(size)
        n = (np.asarray(pts, np.float64) - K[:2, 2]) / self.f_px
        r2 = np.sum(n**2, axis=1, keepdims=True)
        n = n * (1 + self.k1 * r2 + self.k2 * r2**2)
        return n * self.f_px + K[:2, 2]


@dataclass
class Scene:
    """What to draw, in bed mm (right-handed: x right, y away from the user)."""

    markers: dict[int, tuple[float, float]]  # id -> centre
    crosses: np.ndarray  # (N, 2) centres
    marker_size: float = layout.BED_MARKER_SIZE_MM
    cross_arm: float = 5.0  # half-length of each stroke
    line_width: float = 0.5  # pen width
    paper: tuple[float, float, float, float] = (0.0, 0.0, 210.0, 148.0)  # x, y, w, h
    bed: tuple[float, float, float, float] = (-40.0, -40.0, 290.0, 228.0)
    extra_lines: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )  # distractor strokes

    def marker_corners(self) -> dict[int, np.ndarray]:
        return {i: marker_corners_mm(c, self.marker_size) for i, c in self.markers.items()}


def render(
    scene: Scene,
    H_mm_to_px: np.ndarray,
    size: tuple[int, int] = (1920, 1080),
    canvas_px_per_mm: float = 12.0,
    lens: Lens | None = None,
    noise: float = 3.0,
    blur_px: float = 0.7,
    seed: int = 0,
) -> np.ndarray:
    """Render the scene as a BGR camera image."""
    s = canvas_px_per_mm
    bx, by, bw, bh = scene.bed
    cw, ch = int(round(bw * s)), int(round(bh * s))
    canvas = np.full((ch, cw), 70, np.uint8)  # dark bed

    def c(p: tuple[float, float] | np.ndarray) -> tuple[int, int]:
        return int(round((p[0] - bx) * s)), int(round((p[1] - by) * s))

    px_, py_, pw, ph = scene.paper
    cv2.rectangle(canvas, c((px_, py_)), c((px_ + pw, py_ + ph)), 235, -1)

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.ARUCO_DICTIONARY))
    m_px = int(round(scene.marker_size * s))
    quiet = int(round(4 * s))
    for mid, (mx, my) in scene.markers.items():
        x0, y0 = c((mx - scene.marker_size / 2, my - scene.marker_size / 2))
        cv2.rectangle(
            canvas, (x0 - quiet, y0 - quiet), (x0 + m_px + quiet, y0 + m_px + quiet), 245, -1
        )
        # The canvas rows run along +y, so draw the marker flipped: seen from above
        # (through a camera that flips y back) it then reads the right way round.
        img = np.flipud(cv2.aruco.generateImageMarker(d, mid, m_px))
        img = np.where(img > 127, 245, 25).astype(np.uint8)
        canvas[y0 : y0 + m_px, x0 : x0 + m_px] = img

    lw = max(1, int(round(scene.line_width * s)))
    a = scene.cross_arm
    for cx, cy in np.asarray(scene.crosses):
        cv2.line(canvas, c((cx - a, cy)), c((cx + a, cy)), 40, lw, cv2.LINE_AA)
        cv2.line(canvas, c((cx, cy - a)), c((cx, cy + a)), 40, lw, cv2.LINE_AA)
    for p, q in scene.extra_lines:
        cv2.line(canvas, c(p), c(q), 40, lw, cv2.LINE_AA)

    # Anti-alias before shrinking: blur by about half a camera pixel in canvas pixels.
    scale = np.sqrt(abs(np.linalg.det(H_mm_to_px[:2, :2]))) / s  # rough camera px per canvas px
    sigma = max(0.5, 0.5 / max(scale, 1e-6))
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)

    S = np.array([[1 / s, 0, bx], [0, 1 / s, by], [0, 0, 1]], np.float64)  # canvas -> mm
    Hc = H_mm_to_px @ S  # canvas px -> ideal camera px
    if lens is None or (lens.k1 == 0 and lens.k2 == 0):
        img = cv2.warpPerspective(canvas, Hc, size, flags=cv2.INTER_LINEAR, borderValue=60)
    else:
        # For every distorted output pixel, find the ideal pixel, then the canvas pixel.
        w, h = size
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
        K = lens.K(size)
        ideal = cv2.undistortPoints(pts, K, lens.dist, P=K).reshape(-1, 2)
        src = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), np.linalg.inv(Hc)).reshape(-1, 2)
        mapx = src[:, 0].reshape(h, w).astype(np.float32)
        mapy = src[:, 1].reshape(h, w).astype(np.float32)
        img = cv2.remap(canvas, mapx, mapy, cv2.INTER_LINEAR, borderValue=60)

    rng = np.random.default_rng(seed)
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    if noise > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def default_scene() -> tuple[Scene, np.ndarray]:
    """A5 paper, 6 markers around it, a 3x3 grid of crosses 50 mm apart, a typical camera.

    The camera is 1920x1080 at about 4 px/mm with mild keystone and a slight rotation, with
    the user at the bottom of the image (+y points up in the image).
    Returns the scene and the true bed-mm -> pixel homography.
    """
    scene = Scene(
        markers={
            0: (-18.0, -18.0),
            1: (105.0, -20.0),
            2: (228.0, -18.0),
            3: (228.0, 166.0),
            4: (105.0, 168.0),
            5: (-18.0, 166.0),
        },
        crosses=grid_points((105.0, 74.0), 3, 3, 50.0),
    )
    bed_quad = np.array([[-40, -40], [250, -40], [250, 188], [-40, 188]], np.float64)
    img_quad = np.array([[360, 1000], [1575, 985], [1545, 70], [395, 95]], np.float64)
    return scene, camera_homography(bed_quad, img_quad)
