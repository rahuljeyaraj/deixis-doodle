"""Lens model: calibrate the webcam's intrinsics and distortion once, then undistort points.

Why: the Lenovo 300 FHD has a 95-degree lens. A homography assumes a pinhole camera, and a
simulation with typical wide-angle distortion (k1 about -0.3) gave bed-frame errors of 1 to
4 mm median and up to 10 mm at the sheet edges, the whole M1 budget. Undistorting the
*points* (marker corners, cross centres, fingertips) before any homography removes this.
Whole frames are never remapped; that would cost ~10 ms per frame for nothing.

Calibrate with a ChArUco board (DICT_5X5_100, so it cannot be confused with the 4x4 bed
markers): print it, hold it flat in 15 to 25 views across the whole image, then run
`uv run python -m pointdraw.vision.lens capture`. See `main()` for the commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_LENS_PATH = Path(".cache/calibration/lens.json")

BOARD_DICT = "DICT_5X5_100"
BOARD_SQUARES = (9, 6)  # columns, rows: 270 x 180 mm, fits A4 landscape
BOARD_SQUARE_MM = 30.0
BOARD_MARKER_MM = 22.0


def charuco_board() -> cv2.aruco.CharucoBoard:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE_MM, BOARD_MARKER_MM, d)


def board_image(px_per_mm: float = 11.811) -> np.ndarray:
    """The board as a greyscale image (default 300 dpi). Print at 100% on A4 landscape."""
    w = int(round(BOARD_SQUARES[0] * BOARD_SQUARE_MM * px_per_mm))
    h = int(round(BOARD_SQUARES[1] * BOARD_SQUARE_MM * px_per_mm))
    return charuco_board().generateImage((w, h), 0, 1)


_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-6)


def undistort_points(pts_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Raw -> undistorted pixels (same K). Iterates to convergence: OpenCV's default five
    iterations are not enough near the corners of a strongly distorted wide-angle image."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, R=np.eye(3), P=K, criteria=_UNDISTORT_CRITERIA)
    return out.reshape(-1, 2)


@dataclass
class LensModel:
    K: np.ndarray  # 3x3 camera matrix
    dist: np.ndarray  # OpenCV distortion coefficients (k1, k2, p1, p2, k3)
    image_size: tuple[int, int]  # (width, height) the model was calibrated at
    rms_px: float = float("nan")
    n_views: int = 0

    def undistort(self, pts_px: np.ndarray) -> np.ndarray:
        """Raw pixels -> undistorted pixels in the same camera matrix (N, 2)."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return pts.reshape(0, 2)
        return undistort_points(pts, self.K, self.dist)

    def distort(self, pts_px: np.ndarray) -> np.ndarray:
        """Undistorted pixels -> raw pixels (to draw overlays on the live image)."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return pts
        n = (pts - self.K[:2, 2]) / np.diag(self.K)[:2]
        obj = np.concatenate([n, np.ones((len(n), 1))], axis=1).reshape(-1, 1, 3)
        out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)

    def check_size(self, size: tuple[int, int]) -> None:
        if tuple(size) != tuple(self.image_size):
            raise ValueError(
                f"lens model is for {self.image_size[0]}x{self.image_size[1]}, "
                f"frame is {size[0]}x{size[1]}: recalibrate or change the resolution"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "K": self.K.tolist(),
            "dist": np.asarray(self.dist).ravel().tolist(),
            "image_size": list(self.image_size),
            "rms_px": self.rms_px,
            "n_views": self.n_views,
        }

    @classmethod
    def from_json(cls, d: dict[str, object]) -> LensModel:
        return cls(
            K=np.asarray(d["K"], dtype=np.float64),
            dist=np.asarray(d["dist"], dtype=np.float64),
            image_size=tuple(d["image_size"]),  # type: ignore[arg-type]
            rms_px=float(d.get("rms_px", float("nan"))),  # type: ignore[arg-type]
            n_views=int(d.get("n_views", 0)),  # type: ignore[arg-type]
        )

    def save(self, path: str | Path = DEFAULT_LENS_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_or_none(cls, path: str | Path = DEFAULT_LENS_PATH) -> LensModel | None:
        try:
            return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None


def detect_board(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(object points (N, 3) mm, image points (N, 2) px) of the ChArUco corners, or None."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board = charuco_board()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 8:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3), img.reshape(-1, 2)


def calibrate(images: list[np.ndarray], min_views: int = 5) -> LensModel:
    """Fit K and (k1, k2, p1, p2) from ChArUco views. k3 is fixed at 0 (more stable)."""
    objs, imgs = [], []
    size: tuple[int, int] | None = None
    for im in images:
        size = (im.shape[1], im.shape[0])
        found = detect_board(im)
        if found is not None:
            objs.append(found[0].astype(np.float32))
            imgs.append(found[1].astype(np.float32))
    if len(objs) < min_views or size is None:
        raise RuntimeError(f"board found in {len(objs)} views, need at least {min_views}")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None, flags=cv2.CALIB_FIX_K3)
    return LensModel(K=K, dist=dist.ravel(), image_size=size, rms_px=float(rms), n_views=len(objs))


# --- command line ------------------------------------------------------------------------------


def _capture(args: argparse.Namespace) -> int:
    from pointdraw.vision.camera import Camera, CameraSettings, CameraUnavailable

    try:
        cam = Camera(CameraSettings(index=args.camera, backend=args.backend)).start()
    except CameraUnavailable as e:
        print(f"No camera: {e}")
        return 1
    views: list[np.ndarray] = []
    print("SPACE: keep this view (board must be detected)  C: calibrate and save  Q: quit")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            found = detect_board(frame.image)
            vis = frame.image.copy()
            if found is not None:
                for p in found[1]:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
            msg = f"views {len(views)}  board {'yes' if found is not None else 'no'}"
            cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.imshow("lens calibration", cv2.resize(vis, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found is not None:
                views.append(frame.image.copy())
            elif key == ord("c"):
                model = calibrate(views)
                path = model.save(args.out)
                print(f"saved {path}: rms {model.rms_px:.3f} px, dist {np.round(model.dist, 4)}")
                return 0
            elif key == ord("q"):
                return 0
    finally:
        cam.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pointdraw.vision.lens", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="write the ChArUco board PNG (print at 100%, A4 landscape)")
    b.add_argument("out", nargs="?", default="out/charuco-board.png")
    c = sub.add_parser("capture", help="collect views from the camera, calibrate, save")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--backend", default="auto")
    c.add_argument("--out", default=str(DEFAULT_LENS_PATH))
    f = sub.add_parser("files", help="calibrate from saved images")
    f.add_argument("images", nargs="+")
    f.add_argument("--out", default=str(DEFAULT_LENS_PATH))
    args = ap.parse_args(argv)

    if args.cmd == "board":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        img = board_image()
        try:  # tag 300 dpi so "actual size" printing comes out right
            from PIL import Image

            Image.fromarray(img).save(out, dpi=(300, 300))
        except ImportError:
            cv2.imwrite(str(out), img)
        w, h = BOARD_SQUARES[0] * BOARD_SQUARE_MM, BOARD_SQUARES[1] * BOARD_SQUARE_MM
        print(f"wrote {out} ({w:.0f} x {h:.0f} mm at 300 dpi); print at 100%, check a square")
        return 0
    if args.cmd == "capture":
        return _capture(args)
    images = [cv2.imread(p) for p in args.images]
    model = calibrate([im for im in images if im is not None])
    path = model.save(args.out)
    print(f"saved {path}: rms {model.rms_px:.3f} px over {model.n_views} views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
