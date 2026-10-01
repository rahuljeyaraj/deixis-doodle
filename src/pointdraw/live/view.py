"""What the fixed camera sees, in bed mm, while the bed moves.

The calibration (`vision.aruco.fixed_bed_homography`) was fitted with the bed parked. On the
Ender the bed is the Y axis: the nozzle stays put in Y and the bed slides under it, so with
the head at machine Y = y the bed has moved by `y - park_y` since then. A camera pixel that
saw bed point (x, b) at calibration now sees (x, b + y - park_y). `BedView` applies that
shift, so a fingertip lands on the right spot of the paper during a preview (the head waits
over the box, not at park) as well as when parked.

The head's Y comes from the lines sent to Marlin, which run a little ahead of the machine
while it moves (its planner holds about 16 short pieces). It is exact once the head stops.
Heights above the paper (a hovering finger) are not corrected.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from deixis.geometry import Box
from pointdraw.vision.aruco import BedHomography


class BedView:
    """Camera pixels <-> bed mm, for wherever the bed is now."""

    def __init__(
        self, bed: BedHomography, park_y: float, bed_y: Callable[[], float | None]
    ) -> None:
        self.bed = bed
        self.park_y = park_y
        self.bed_y = bed_y  # machine Y now; None while unknown (taken as parked)

    def offset_y(self) -> float:
        y = self.bed_y()
        return 0.0 if y is None else y - self.park_y

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        mm = self.bed.to_mm(pts_px)
        mm[:, 1] += self.offset_y()
        return mm

    def to_px(self, pts_mm: np.ndarray, offset_y: float | None = None) -> np.ndarray:
        """Where bed points are in the picture; `offset_y` for a picture taken earlier."""
        mm = np.array(pts_mm, dtype=np.float64).reshape(-1, 2)
        mm[:, 1] -= self.offset_y() if offset_y is None else offset_y
        return self.bed.to_px(mm)


class PaperCamera:
    """The runtime's `CameraPort`: a top-down image of the paper, the user at the bottom.

    The main loop hands it frames with `offer(image, hands_over_paper)`. Frames with a hand
    over the paper are skipped (a hand would look like ink), so the image is the latest
    clear view, normally from a moment before the user spoke.

    The bed may have moved since that frame (review, 2 Oct 2026: it was warped with the bed's
    position at the time of asking, so the robot's own ink no longer lined up and showed as
    user marks). So each frame keeps the bed position it was taken at, and frames are taken
    only once the bed has been still for `settle_s`: the position known here is where the
    head was sent, which runs up to Marlin's planner depth (about 16 pieces) ahead of it.
    """

    def __init__(
        self, view: BedView, paper: Box, mm_per_px: float = 0.5, settle_s: float = 1.0
    ) -> None:
        self.view = view
        self.paper = paper
        self.mm_per_px = mm_per_px
        self.settle_s = settle_s
        self.image: np.ndarray | None = None
        self.t: float | None = None  # when `image` was taken
        self.offset_y = 0.0  # the bed's Y offset when `image` was taken
        self._last_y: float | None = None
        self._still_since = -np.inf

    def offer(self, image: np.ndarray, t: float, hands_over_paper: bool) -> None:
        off = self.view.offset_y()
        if self._last_y is not None and off != self._last_y:
            self._still_since = t
        self._last_y = off
        if not hands_over_paper and t - self._still_since >= self.settle_s:
            self.image, self.t, self.offset_y = image, t, off

    def paper_image(self, t: float) -> tuple[np.ndarray, float]:
        s = self.mm_per_px
        w, h = round(self.paper.w / s), round(self.paper.h / s)
        if self.image is None:
            return np.full((h, w, 3), 255, np.uint8), s
        return warp_to_paper(self.image, self.view, self.paper, s, self.offset_y), s


def warp_to_paper(
    image: np.ndarray,
    view: BedView,
    paper: Box,
    mm_per_px: float,
    offset_y: float | None = None,
) -> np.ndarray:
    """Resample the camera image onto the paper frame (`scene.ink`'s convention).

    `offset_y`: the bed's offset when the image was taken (default: where it is now)."""
    s = mm_per_px
    w, h = round(paper.w / s), round(paper.h / s)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    x_mm = paper.x + (cols + 0.5) * s
    y_mm = paper.y + (h - rows - 0.5) * s  # row 0 is the far edge
    px = view.to_px(np.stack([x_mm, y_mm], axis=-1), offset_y).reshape(h, w, 2)
    px = px.astype(np.float32)
    return cv2.remap(
        image,
        px[..., 0],
        px[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
