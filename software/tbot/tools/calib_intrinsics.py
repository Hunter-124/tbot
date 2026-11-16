"""`tbot calib-intrinsics` — chessboard intrinsic calibration (contract: docs/ARCHITECTURE.md [calib]).

Sources: a live camera (views are auto-accepted when they are sharp and add a new pose), a directory of
images (--from-dir) or a recorded session (--session). The calibration core, `calibrate_views`, is a pure
function so it can be driven from tests and from the live/offline paths alike.

The default model is the equidistant fisheye (cv2.fisheye): the M12 f = 2.8 mm lens on the OV9281 is
specified at "FOV H118 / D148" with TV distortion < -17 %, far outside what the pinhole model handles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from tbot.config import Calibration, Config, add_common_args, load_from_args, now_iso, save_json

DEFAULT_COLS = 9
DEFAULT_ROWS = 6
DEFAULT_SQUARE_MM = 25.0
DEFAULT_VIEWS = 25

# View acceptance (live/session). [INFERENCE] bench-tuned defaults, all overridable on the command line.
SHARPNESS_MIN = 60.0  # variance of the 4-neighbour Laplacian inside the board bbox (DN^2)
MIN_POSE_DIST = 0.12  # distance in the pose descriptor below which a view adds nothing
COVERAGE_GRID = 3  # the image is split COVERAGE_GRID^2 ways; the corners must be reached

# Write guards: a calibration worse than this is a measurement failure, not a result.
MIN_VIEWS_OK = 8
MAX_RMS_PX = 1.0
REJECT_RMS_FACTOR = 3.0  # drop views whose RMS exceeds this times the median, then re-calibrate
MAX_REJECT_PASSES = 4

# innomaker CAM-MIPIOV9281 V2 manual V1.4 §3.1 lens claim (for the report only).
VENDOR_FOV_H_DEG = 118.0
VENDOR_FOV_D_DEG = 148.0

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")
SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)


def _calib_flag(name: str) -> int:
    """CALIB_* flag lookup: OpenCV 4.x exposes them on cv2.fisheye, OpenCV 5.0 only on cv2."""
    for ns in (cv2.fisheye, cv2):
        v = getattr(ns, name, None)
        if v is not None:
            return int(v)
    raise AttributeError(f"this OpenCV build has no {name}")


# --------------------------------------------------------------------------------- corner detection
def to_gray(image: np.ndarray) -> np.ndarray:
    """uint8 single-channel view of a mono or BGR frame."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def find_corners(image: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Inner chessboard corners as (N, 2) float64 in raster order, or None.

    findChessboardCornersSB first (sub-pixel accurate by construction, robust to blur and perspective);
    the classic detector + cornerSubPix is the fallback for images SB refuses.
    """
    gray = to_gray(image)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=SB_FLAGS)
    if ok and corners is not None:
        return np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    ok, corners = cv2.findChessboardCorners(
        gray, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS)
    if not ok or corners is None:
        return None
    refined = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, refined, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
    return refined.reshape(-1, 2).astype(np.float64)


class Chessboard:
    """Plain chessboard target: every view must show all cols x rows inner corners."""

    def __init__(self, pattern: tuple[int, int], square_m: float) -> None:
        self.pattern = (int(pattern[0]), int(pattern[1]))
        self.square_m = float(square_m)

    @property
    def grid(self) -> tuple[int, int]:
        """(cols, rows) of the full inner-corner grid, for the pose descriptor."""
        return self.pattern

    def describe(self) -> str:
        return f"{self.pattern[0]}x{self.pattern[1]} chessboard, {self.square_m * 1000.0:g} mm squares"

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(image points (N,2), board points (N,3)) or None."""
        c = find_corners(image, self.pattern)
        return None if c is None else (c, board_object_points(self.pattern, self.square_m))

    def full_grid(self, img: np.ndarray, obj: np.ndarray) -> np.ndarray:
        return img


ARUCO_DICTS = ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000", "DICT_5X5_50", "DICT_5X5_100",
               "DICT_5X5_250", "DICT_5X5_1000", "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
               "DICT_ARUCO_ORIGINAL")
CHARUCO_MIN_CORNERS = 12  # fewer corners than this in a view constrain little and fit the homography badly


class CharucoBoard:
    """ChArUco target. Views may show any part of the board (>= CHARUCO_MIN_CORNERS corners), so the board can
    be pushed into the image corners, where a fisheye model is decided.

    "7x10" is ambiguous (which side is which) and OpenCV >= 4.6 lays out boards with an even row count
    differently from older generators (legacy pattern). A wrong layout detects nothing, so every candidate is
    tried until one view shows at least half the board with exactly one layout; that layout is then locked.
    """

    def __init__(self, squares: tuple[int, int], square_m: float, marker_m: float,
                 dictionary: str = "DICT_4X4_50", legacy: bool | None = None,
                 min_corners: int = CHARUCO_MIN_CORNERS) -> None:
        if dictionary not in ARUCO_DICTS:
            raise ValueError(f"unknown ArUco dictionary {dictionary!r} ({', '.join(ARUCO_DICTS)})")
        if not 0 < marker_m < square_m:
            raise ValueError(f"marker size {marker_m * 1000:g} mm must be smaller than the square "
                             f"{square_m * 1000:g} mm")
        sx, sy = int(squares[0]), int(squares[1])
        if min(sx, sy) < 3:
            raise ValueError(f"a {sx}x{sy} ChArUco board is too small")
        self.squares = (sx, sy)
        self.square_m = float(square_m)
        self.marker_m = float(marker_m)
        self.dictionary = dictionary
        self.min_corners = int(min_corners)
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        layouts = [(sx, sy), (sy, sx)] if sx != sy else [(sx, sy)]
        legacies = [False, True] if legacy is None else [bool(legacy)]
        self._candidates = []
        for lay in layouts:
            for leg in legacies:
                b = cv2.aruco.CharucoBoard(lay, self.square_m, self.marker_m, d)
                b.setLegacyPattern(leg)
                obj = np.asarray(b.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
                self._candidates.append((lay, leg, cv2.aruco.CharucoDetector(b), obj))
        self.locked = None if len(self._candidates) > 1 else self._candidates[0]

    @property
    def layout(self) -> tuple[tuple[int, int], bool] | None:
        """((squares_x, squares_y), legacy) once known."""
        return None if self.locked is None else (self.locked[0], self.locked[1])

    @property
    def grid(self) -> tuple[int, int]:
        lay = self.locked[0] if self.locked is not None else self.squares
        return (lay[0] - 1, lay[1] - 1)

    def describe(self) -> str:
        if self.locked is None:
            lay = f"{self.squares[0]}x{self.squares[1]} (layout not yet seen)"
        else:
            lay = f"{self.locked[0][0]}x{self.locked[0][1]}{' legacy' if self.locked[1] else ''}"
        return (f"{lay} ChArUco, {self.square_m * 1000.0:g} mm squares, {self.marker_m * 1000.0:g} mm markers, "
                f"{self.dictionary}")

    @staticmethod
    def _run(cand, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        corners, ids, _, _ = cand[2].detectBoard(gray)
        if corners is None or ids is None or len(ids) == 0:
            return None
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        return np.asarray(corners, dtype=np.float64).reshape(-1, 2), cand[3][ids]

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        gray = to_gray(image)
        if self.locked is not None:
            hit = self._run(self.locked, gray)
        else:
            found = [(c, r) for c in self._candidates if (r := self._run(c, gray)) is not None]
            if not found:
                return None
            best = max(found, key=lambda cr: cr[1][0].shape[0])
            n_inner = best[1][1].shape[0]
            total = (best[0][0][0] - 1) * (best[0][0][1] - 1)
            others = [r[0].shape[0] for c, r in found if c is not best[0]]
            if n_inner * 2 >= total and all(k < self.min_corners for k in others):
                self.locked = best[0]
            if self.locked is None:
                return None  # a partial view cannot tell the layouts apart: wait for a clear one
            hit = best[1]
        if hit is None or hit[0].shape[0] < self.min_corners:
            return None
        return hit

    def full_grid(self, img: np.ndarray, obj: np.ndarray) -> np.ndarray:
        """The whole inner-corner grid projected through a board->image homography fitted to the seen
        corners (approximate under fisheye distortion; only used to describe the pose)."""
        all_obj = self.locked[3] if self.locked is not None else obj
        H, _ = cv2.findHomography(obj[:, :2].astype(np.float32), img.astype(np.float32), 0)
        if H is None:
            return img
        pts = cv2.perspectiveTransform(all_obj[:, :2].reshape(-1, 1, 2).astype(np.float64), H)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def parse_squares(text: str) -> tuple[int, int]:
    try:
        a, b = text.lower().replace("*", "x").split("x")
        return int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected SQUARES_XxSQUARES_Y, e.g. 7x10, got {text!r}") from None


def board_object_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    """(N, 3) float64 board-frame corner coordinates matching find_corners' raster order (Z = 0)."""
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, 0] = grid[:, 0] * square_m
    obj[:, 1] = grid[:, 1] * square_m
    return obj


def sharpness(image: np.ndarray, corners: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian inside the board bounding box (motion blur / defocus test)."""
    gray = to_gray(image)
    h, w = gray.shape[:2]
    x0 = max(0, int(math.floor(corners[:, 0].min())))
    x1 = min(w, int(math.ceil(corners[:, 0].max())) + 1)
    y0 = max(0, int(math.floor(corners[:, 1].min())))
    y1 = min(h, int(math.ceil(corners[:, 1].max())) + 1)
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        return 0.0
    lap = (4.0 * roi[1:-1, 1:-1] - roi[:-2, 1:-1] - roi[2:, 1:-1] - roi[1:-1, :-2] - roi[1:-1, 2:])
    return float(lap.var())


def pose_descriptor(corners: np.ndarray, pattern: tuple[int, int], image_size: tuple[int, int]) -> np.ndarray:
    """[cx, cy, scale, tilt_h, tilt_v] — where the board is, how big it is and how it is tilted.

    tilt_* are the relative length differences of opposite board edges, i.e. the perspective foreshortening
    that distinguishes a tilted view from a frontal one. All components are dimensionless and O(1).
    """
    cols, rows = pattern
    w, h = image_size
    grid = corners.reshape(rows, cols, 2)
    quad = np.stack([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
    area = 0.5 * abs(float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(np.roll(quad[:, 0], -1), quad[:, 1])))
    diag = math.hypot(w, h)
    top = float(np.linalg.norm(grid[0, -1] - grid[0, 0]))
    bottom = float(np.linalg.norm(grid[-1, -1] - grid[-1, 0]))
    left = float(np.linalg.norm(grid[-1, 0] - grid[0, 0]))
    right = float(np.linalg.norm(grid[-1, -1] - grid[0, -1]))
    centre = corners.mean(axis=0)
    return np.array([
        centre[0] / w,
        centre[1] / h,
        math.sqrt(area) / diag,
        (left - right) / max(left + right, 1e-9),
        (top - bottom) / max(top + bottom, 1e-9),
    ], dtype=np.float64)


_POSE_WEIGHTS = np.array([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)


class ViewSelector:
    """Accepts board views that are sharp and bring a new pose, and tracks image coverage.

    `board` is a Chessboard / CharucoBoard, or a (cols, rows) inner-corner tuple (chessboard; object points
    then use a unit square, which nothing here depends on). `accepted` holds the image points of each accepted
    view and `objects` the matching board points.
    """

    def __init__(self, board, image_size: tuple[int, int], *,
                 sharpness_min: float = SHARPNESS_MIN, min_pose_dist: float = MIN_POSE_DIST,
                 grid: int = COVERAGE_GRID) -> None:
        self.board = Chessboard(board, 1.0) if isinstance(board, tuple) else board
        self.image_size = image_size
        self.sharpness_min = float(sharpness_min)
        self.min_pose_dist = float(min_pose_dist)
        self.accepted: list[np.ndarray] = []
        self.objects: list[np.ndarray] = []
        self.descriptors: list[np.ndarray] = []
        self.coverage = np.zeros((grid, grid), dtype=bool)

    @property
    def pattern(self) -> tuple[int, int]:
        return self.board.grid

    def offer(self, image: np.ndarray) -> tuple[np.ndarray | None, str]:
        """(corners, reason): corners is not None only when the view was accepted."""
        hit = self.board.detect(image)
        if hit is None:
            if getattr(self.board, "locked", True) is None:
                return None, "no board yet (show the WHOLE board once, flat-on, so its layout can be identified)"
            return None, "no board"
        corners, obj = hit
        sharp = sharpness(image, corners)
        if sharp < self.sharpness_min:
            return None, f"blurred ({sharp:.0f} < {self.sharpness_min:.0f})"
        desc = pose_descriptor(self.board.full_grid(corners, obj), self.board.grid, self.image_size)
        if self.descriptors:
            d = float(np.min(np.linalg.norm((np.stack(self.descriptors) - desc) * _POSE_WEIGHTS, axis=1)))
            if d < self.min_pose_dist:
                return None, f"pose already covered ({d:.3f} < {self.min_pose_dist:.3f})"
        self.accepted.append(corners)
        self.objects.append(obj)
        self.descriptors.append(desc)
        self._mark(corners)
        return corners, "accepted"

    def _mark(self, corners: np.ndarray) -> None:
        n = self.coverage.shape[0]
        w, h = self.image_size
        gx = np.clip((corners[:, 0] / max(w, 1) * n).astype(np.int32), 0, n - 1)
        gy = np.clip((corners[:, 1] / max(h, 1) * n).astype(np.int32), 0, n - 1)
        self.coverage[gy, gx] = True

    @property
    def coverage_frac(self) -> float:
        return float(self.coverage.mean())

    def missing_cells(self) -> list[tuple[int, int]]:
        """(col, row) grid cells no accepted view has reached — where to hold the board next."""
        ys, xs = np.nonzero(~self.coverage)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def coverage_hint(self, limit: int = 3) -> str:
        """Where to hold the board next, in words ("top-left, bottom-right")."""
        missing = self.missing_cells()
        if not missing:
            return "coverage complete"
        n = self.coverage.shape[0]
        rows = ("top", "middle", "bottom") if n == 3 else tuple(f"row {i}" for i in range(n))
        cols = ("left", "centre", "right") if n == 3 else tuple(f"col {i}" for i in range(n))
        names = [f"{rows[y]}-{cols[x]}" for x, y in missing[:limit]]
        more = "" if len(missing) <= limit else f" (+{len(missing) - limit} more)"
        return "still need " + ", ".join(names) + more


# --------------------------------------------------------------------------------- calibration core
def _fisheye_calibrate(obj: list[np.ndarray], img: list[np.ndarray], image_size: tuple[int, int],
                       flags: int) -> tuple[float, np.ndarray, np.ndarray, list, list]:
    """cv2.fisheye.calibrate across OpenCV versions (4.x wants (N,1,3)/(N,1,2), 5.0 wants (1,N,3)/(1,N,2)).
    Views may have different point counts (ChArUco)."""
    shapes = (((1, -1, 3), (1, -1, 2)), ((-1, 1, 3), (-1, 1, 2)))
    last: Exception | None = None
    for os_, is_ in shapes:
        o = [np.ascontiguousarray(a.reshape(os_), dtype=np.float64) for a in obj]
        i = [np.ascontiguousarray(a.reshape(is_), dtype=np.float64) for a in img]
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(o, i, image_size, K, D, flags=flags,
                                                            criteria=_CALIB_CRITERIA)
        except cv2.error as e:  # shape convention rejected by this build
            last = e
            continue
        return float(rms), np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64).reshape(-1), \
            list(rvecs), list(tvecs)
    raise RuntimeError(f"cv2.fisheye.calibrate rejected both point layouts: {last}")


def _per_view_rms(obj: np.ndarray, img: np.ndarray, rvec, tvec, K: np.ndarray, D: np.ndarray,
                  model: str) -> float:
    rv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    n = obj.shape[0]
    if model == "fisheye":
        last: Exception | None = None
        for shape in ((n, 1, 3), (1, n, 3)):  # same layout split as cv2.fisheye.calibrate
            try:
                proj, _ = cv2.fisheye.projectPoints(np.ascontiguousarray(obj.reshape(shape)), rv, tv,
                                                    K, D.reshape(1, 4))
                break
            except cv2.error as e:
                last = e
        else:
            raise RuntimeError(f"cv2.fisheye.projectPoints rejected both point layouts: {last}")
    else:
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D.reshape(1, -1))
    err = np.asarray(proj, dtype=np.float64).reshape(-1, 2) - img.reshape(-1, 2)
    return float(math.sqrt(float(np.mean(np.sum(err * err, axis=1)))))


def _calibrate_once(objs: list[np.ndarray], corners: list[np.ndarray], image_size: tuple[int, int],
                    model: str) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    if model == "fisheye":
        flags = _calib_flag("CALIB_RECOMPUTE_EXTRINSIC") | _calib_flag("CALIB_FIX_SKEW")
        rms, K, D, rvecs, tvecs = _fisheye_calibrate(objs, corners, image_size, flags)
    else:
        o32 = [a.reshape(-1, 1, 3).astype(np.float32) for a in objs]
        i32 = [a.reshape(-1, 1, 2).astype(np.float32) for a in corners]
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(o32, i32, image_size, None, None,
                                                      criteria=_CALIB_CRITERIA)
        K = np.asarray(K, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        rms = float(rms)
    per = [_per_view_rms(o, c, rv, tv, K, D, model) for o, c, rv, tv in zip(objs, corners, rvecs, tvecs)]
    return rms, K, D, per


def calibrate_views(corners_list: list[np.ndarray], pattern: tuple[int, int], square_m: float,
                    image_size: tuple[int, int], model: str = "fisheye",
                    object_points: list[np.ndarray] | None = None) -> dict:
    """Calibrate from detected corners. Pure: no I/O, no config.

    corners_list: per view, the (N, 2) inner corners from find_corners (N = cols*rows, raster order).
    object_points: per view, the matching (N, 3) board points instead (ChArUco: N varies per view);
    pattern/square_m are then not used for the geometry.
    image_size: (width, height). model: "fisheye" (cv2.fisheye, D of 4) or "pinhole" (D of 5).
    Views whose reprojection RMS exceeds REJECT_RMS_FACTOR x the median are dropped and the calibration
    is repeated, because one mis-detected board biases every parameter.
    Returns the intrinsics.json payload (without "created"/"notes") plus per_view_rms and the view indices.
    """
    if model not in ("fisheye", "pinhole"):
        raise ValueError(f"unknown model {model!r} (fisheye | pinhole)")
    cols, rows = pattern
    n_pts = cols * rows
    if object_points is not None and len(object_points) != len(corners_list):
        raise ValueError(f"{len(object_points)} object point sets for {len(corners_list)} views")
    views, objs = [], []
    for i, c in enumerate(corners_list):
        a = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        if object_points is None:
            if a.shape[0] != n_pts:
                raise ValueError(f"view {i}: {a.shape[0]} corners, expected {n_pts} for a {cols}x{rows} pattern")
            o = board_object_points(pattern, square_m)
        else:
            o = np.asarray(object_points[i], dtype=np.float64).reshape(-1, 3)
            if o.shape[0] != a.shape[0] or a.shape[0] < 6:
                raise ValueError(f"view {i}: {a.shape[0]} image points for {o.shape[0]} board points")
        views.append(a)
        objs.append(o)
    if len(views) < 4:
        raise ValueError(f"need at least 4 views, got {len(views)}")

    keep = list(range(len(views)))
    rms, K, D, per = _calibrate_once([objs[i] for i in keep], [views[i] for i in keep], image_size, model)
    rejected: list[int] = []
    for _ in range(MAX_REJECT_PASSES):
        median = float(np.median(per))
        limit = REJECT_RMS_FACTOR * median
        good = [i for i, e in zip(keep, per) if e <= limit]
        if len(good) == len(keep) or len(good) < max(6, MIN_VIEWS_OK // 2):
            break
        rejected += [i for i in keep if i not in good]
        keep = good
        rms, K, D, per = _calibrate_once([objs[i] for i in keep], [views[i] for i in keep], image_size, model)

    return {
        "model": model,
        "width": int(image_size[0]),
        "height": int(image_size[1]),
        "K": [[float(v) for v in row] for row in K],
        "D": [float(v) for v in np.asarray(D).reshape(-1)],
        "rms_px": float(rms),
        "n_views": len(keep),
        "per_view_rms": [float(e) for e in per],
        "used_views": [int(i) for i in keep],
        "rejected_views": sorted(int(i) for i in rejected),
    }


# --------------------------------------------------------------------------------- reporting
def model_rays(K: np.ndarray, D: np.ndarray, model: str, uv: np.ndarray) -> np.ndarray:
    """(N,2) distorted pixels -> (N,3) unit rays in the camera frame, using the calibrated model only."""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        und = cv2.fisheye.undistortPoints(pts, np.asarray(K, dtype=np.float64),
                                          np.asarray(D, dtype=np.float64).reshape(1, 4))
    else:
        und = cv2.undistortPoints(pts, np.asarray(K, dtype=np.float64),
                                  np.asarray(D, dtype=np.float64).reshape(1, -1))
    xy = np.asarray(und, dtype=np.float64).reshape(-1, 2)
    rays = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float64)], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = np.cross(a, b)
    return float(math.degrees(math.atan2(float(np.linalg.norm(c)), float(np.dot(a, b)))))


def fov_deg(K: np.ndarray, D: np.ndarray, model: str, image_size: tuple[int, int]) -> dict:
    """Full field of view across the image: {"h","v","d"} degrees (pixel centres at integers,
    so the sensor edges are at -0.5 and W-0.5)."""
    w, h = image_size
    x0, x1 = -0.5, w - 0.5
    y0, y1 = -0.5, h - 0.5
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    uv = np.array([[x0, cy], [x1, cy], [cx, y0], [cx, y1], [x0, y0], [x1, y1], [x1, y0], [x0, y1]],
                  dtype=np.float64)
    r = model_rays(K, D, model, uv)
    if not np.all(np.isfinite(r)):
        nan = float("nan")
        return {"h": nan, "v": nan, "d": nan}
    d1 = _angle_deg(r[4], r[5])
    d2 = _angle_deg(r[6], r[7])
    return {"h": _angle_deg(r[0], r[1]), "v": _angle_deg(r[2], r[3]), "d": 0.5 * (d1 + d2)}


def nominal_fx(cfg: Config, width: int) -> float:
    """Prior focal length in pixels at `width`: f / pixel pitch, scaled from the full sensor width."""
    fx_full = cfg.optics.focal_length_mm * 1000.0 / cfg.optics.pixel_pitch_um
    return float(fx_full * width / cfg.optics.sensor_width_px)


def coverage_image(corners_list: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    """Where the board was seen: uint8 (H, W), brighter where more views contributed corners."""
    w, h = image_size
    acc = np.zeros((h, w), dtype=np.float32)
    layer = np.zeros((h, w), dtype=np.uint8)
    radius = max(3, int(round(min(w, h) / 60.0)))
    for corners in corners_list:
        layer[:] = 0
        for u, v in np.asarray(corners, dtype=np.float64).reshape(-1, 2):
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(layer, (int(round(u)), int(round(v))), radius, 255, -1)
        acc += layer.astype(np.float32) * (1.0 / 255.0)
    if acc.max() > 0:
        acc *= 255.0 / acc.max()
    return acc.astype(np.uint8)


# --------------------------------------------------------------------------------- view sources
def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


Views = tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]  # image points, board points, (w, h)


def _corners_from_dir(d: Path, board, verbose: bool) -> Views:
    files = _list_images(d)
    if not files:
        raise ValueError(f"no images in {d} (looked for {', '.join(IMAGE_EXTS)})")
    if getattr(board, "locked", True) is None:  # find the ChArUco layout on the clearest image first
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and board.detect(img) is not None:
                break
    corners: list[np.ndarray] = []
    objects: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  {p.name}: unreadable, skipped")
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            raise ValueError(f"{p.name} is {w}x{h}, earlier images are {size[0]}x{size[1]}")
        hit = board.detect(img)
        if verbose:
            print(f"  {p.name}: {f'{hit[0].shape[0]} corners' if hit is not None else 'no board'}")
        if hit is not None:
            corners.append(hit[0])
            objects.append(hit[1])
    if size is None:
        raise ValueError(f"no readable images in {d}")
    return corners, objects, size


def _open_session(d: Path) -> tuple[np.ndarray, dict]:
    meta_p = d / "meta.json"
    frames_p = d / "frames.bin"
    if not meta_p.exists() or not frames_p.exists():
        raise ValueError(f"{d} is not a session directory (needs meta.json and frames.bin)")
    meta = json.loads(meta_p.read_text())
    w, h = int(meta["width"]), int(meta["height"])
    ch = int(meta.get("channels", 1))
    dtype = np.dtype(meta.get("dtype", "uint8"))
    stride = w * h * ch * dtype.itemsize
    n = frames_p.stat().st_size // stride
    if n < 1:
        raise ValueError(f"{frames_p} holds no complete frame")
    shape = (n, h, w) if ch == 1 else (n, h, w, ch)
    return np.memmap(str(frames_p), dtype=dtype, mode="r", shape=shape), meta


def _corners_from_session(d: Path, board, every: int, views: int, sel_kwargs: dict,
                          save_dir: Path | None) -> Views:
    frames, meta = _open_session(d)
    size = (int(meta["width"]), int(meta["height"]))
    sel = ViewSelector(board, size, **sel_kwargs)
    step = max(1, int(every))
    for i in range(0, frames.shape[0], step):
        img = np.ascontiguousarray(frames[i])
        corners, reason = sel.offer(img)
        if corners is not None:
            print(f"  frame {i}: accepted ({len(sel.accepted)}/{views}, "
                  f"coverage {int(sel.coverage.sum())}/{sel.coverage.size})")
            if save_dir is not None:
                _save_view(save_dir, len(sel.accepted) - 1, img)
        if len(sel.accepted) >= views and sel.coverage_frac >= 1.0:
            break
    return sel.accepted, sel.objects, size


def _save_view(save_dir: Path, index: int, image: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f"view_{index:03d}.png"), to_gray(image))


def _corners_live(cfg: Config, calib: Calibration, board, views: int, gui: bool,
                  timeout_s: float, sel_kwargs: dict, save_dir: Path | None, server=None) -> Views:
    from tbot.camera import open_source

    source = open_source(cfg.camera.source, cfg, calib)
    source.start()
    size = (int(source.width), int(source.height))
    sel = ViewSelector(board, size, **sel_kwargs)
    layout_known = getattr(board, "locked", True) is not None
    window = "tbot calib-intrinsics"
    t0 = source.clock.now_ns()
    last_report = ""
    try:
        if gui:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame = source.read(timeout_s=1.0)
            if frame is None:
                if timeout_s > 0 and (source.clock.now_ns() - t0) > timeout_s * 1e9:
                    break
                continue
            corners, reason = sel.offer(frame.image)
            if not layout_known and getattr(board, "locked", True) is not None:
                layout_known = True
                print(f"  board identified: {board.describe()}")
            if corners is not None:
                print(f"  accepted {len(sel.accepted)}/{views} ({corners.shape[0]} corners)  coverage "
                      f"{int(sel.coverage.sum())}/{sel.coverage.size}  {sel.coverage_hint()}")
                if save_dir is not None:
                    _save_view(save_dir, len(sel.accepted) - 1, frame.image)
            elif not gui and reason != last_report:
                print(f"  waiting: {reason}")
                last_report = reason
            if server is not None and server.due():
                server.publish(_overlay(frame.image, corners, sel, views, reason),
                               {"views": len(sel.accepted), "target": views,
                                "coverage": f"{int(sel.coverage.sum())}/{sel.coverage.size}",
                                "hint": sel.coverage_hint(), "last": reason, "board": board.describe()})
            if gui:
                cv2.imshow(window, _overlay(frame.image, corners, sel, views, reason))
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            if len(sel.accepted) >= views and sel.coverage_frac >= 1.0:
                break
            if len(sel.accepted) >= 2 * views:
                break
            if timeout_s > 0 and (source.clock.now_ns() - t0) > timeout_s * 1e9:
                print("  timeout reached")
                break
    except KeyboardInterrupt:
        print("  interrupted: calibrating with the views accepted so far")
    finally:
        source.stop()
        if gui:
            cv2.destroyAllWindows()
    return sel.accepted, sel.objects, size


def _overlay(image: np.ndarray, corners: np.ndarray | None, sel: ViewSelector, views: int,
             reason: str = "") -> np.ndarray:
    """Live coverage view: grid cells green once reached, earlier corners orange, this view's corners."""
    vis = cv2.cvtColor(to_gray(image), cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    n = sel.coverage.shape[0]
    for gy in range(n):
        for gx in range(n):
            x0, y0 = int(gx * w / n), int(gy * h / n)
            x1, y1 = int((gx + 1) * w / n) - 1, int((gy + 1) * h / n) - 1
            colour = (0, 180, 0) if sel.coverage[gy, gx] else (40, 40, 160)
            cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 1)
    for acc in sel.accepted:
        for u, v in acc:
            cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 140, 255), -1)
    if corners is not None:
        for u, v in corners:
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"views {len(sel.accepted)}/{views}  coverage {int(sel.coverage.sum())}"
                f"/{sel.coverage.size}  {sel.coverage_hint(limit=2)}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if reason:
        cv2.putText(vis, reason[:90], (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0) if reason == "accepted" else (0, 200, 255), 1, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot calib-intrinsics",
                                description="chessboard intrinsic calibration -> calib/intrinsics.json")
    add_common_args(p)
    g = p.add_argument_group("views")
    g.add_argument("--from-dir", default=None, metavar="DIR", help="calibrate from a directory of images")
    g.add_argument("--session", default=None, metavar="DIR", help="calibrate from a recorded session")
    g.add_argument("--every", type=int, default=10, help="use every Nth session frame (default 10)")
    g.add_argument("--views", type=int, default=DEFAULT_VIEWS, help=f"target view count (default {DEFAULT_VIEWS})")
    g.add_argument("--timeout-s", type=float, default=0.0, help="live capture time limit (0 = until done)")
    g.add_argument("--sharp-min", type=float, default=SHARPNESS_MIN, help="minimum Laplacian variance")
    g.add_argument("--min-pose-dist", type=float, default=MIN_POSE_DIST, help="minimum new-pose distance")
    g.add_argument("--no-save-views", action="store_true", help="do not write accepted views to disk")
    b = p.add_argument_group("board")
    b.add_argument("--cols", type=int, default=DEFAULT_COLS, help="inner corners across (default 9)")
    b.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="inner corners down (default 6)")
    b.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM, help="square size, mm (default 25)")
    b.add_argument("--charuco", type=parse_squares, default=None, metavar="SXxSY",
                   help="ChArUco board with this many SQUARES (e.g. 7x10; orientation is detected); "
                        "--cols/--rows are then ignored")
    b.add_argument("--marker-mm", type=float, default=None, help="ChArUco marker size, mm (required with --charuco)")
    b.add_argument("--dict", default="DICT_4X4_50", choices=ARUCO_DICTS, help="ChArUco ArUco dictionary")
    b.add_argument("--legacy", choices=("auto", "yes", "no"), default="auto",
                   help="ChArUco pre-OpenCV-4.6 layout (only differs for an even row count; default: detect)")
    o = p.add_argument_group("output")
    o.add_argument("--model", choices=("fisheye", "pinhole"), default=None, help="default: optics.model")
    o.add_argument("--out", default=None, help="output JSON (default <calib-dir>/intrinsics.json)")
    o.add_argument("--gui", action="store_true", help="show the live coverage window (needs a display)")
    o.add_argument("--stream", action="store_true",
                   help="live: serve the coverage view on the debug server (http://<pi>:<stream.port>/)")
    o.add_argument("--force", action="store_true", help="write even when the quality guards fail")
    args = p.parse_args(argv)

    cfg, calib = load_from_args(args)
    square_m = float(args.square_mm) / 1000.0
    try:
        if args.charuco is not None:
            if args.marker_mm is None:
                p.error("--charuco needs --marker-mm")
            legacy = {"auto": None, "yes": True, "no": False}[args.legacy]
            board = CharucoBoard(args.charuco, square_m, float(args.marker_mm) / 1000.0, args.dict, legacy)
        else:
            board = Chessboard((int(args.cols), int(args.rows)), square_m)
    except ValueError as e:
        print(f"tbot calib-intrinsics: {e}", file=sys.stderr)
        return 2
    model = args.model or cfg.optics.model
    out = Path(args.out) if args.out else calib.path("intrinsics.json")
    views_dir = None if args.no_save_views else calib.dir / "intrinsics_views"
    sel_kwargs = {"sharpness_min": float(args.sharp_min), "min_pose_dist": float(args.min_pose_dist)}

    server = None
    try:
        if args.from_dir:
            src_name = f"images in {args.from_dir}"
            print(f"scanning {args.from_dir} for a {board.describe()}")
            corners, objects, size = _corners_from_dir(Path(args.from_dir), board, verbose=True)
        elif args.session:
            src_name = f"session {args.session} (every {args.every})"
            print(f"scanning {src_name} for a {board.describe()}")
            corners, objects, size = _corners_from_session(Path(args.session), board, args.every, args.views,
                                                           sel_kwargs, views_dir)
        else:
            src_name = f"live {cfg.camera.source}"
            if args.stream:
                from tbot.app.stream import DebugServer
                server = DebugServer(cfg).start()
                print(f"coverage view: http://{server.host}:{server.port}/")
            if views_dir is not None and views_dir.exists() and any(views_dir.iterdir()):
                old = views_dir.with_name(f"intrinsics_views.prev")
                if old.exists():
                    import shutil
                    shutil.rmtree(old)
                views_dir.rename(old)
                print(f"previous views moved to {old}")
            print(f"live capture from {cfg.camera.source} ({board.describe()}): hold the board at varied "
                  f"distances and tilts, cover the image corners (Ctrl-C or q to stop)")
            corners, objects, size = _corners_live(cfg, calib, board, args.views, args.gui, args.timeout_s,
                                                   sel_kwargs, views_dir, server)
    except (ValueError, OSError) as e:
        print(f"tbot calib-intrinsics: {e}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.stop()

    print(f"{len(corners)} usable view(s) at {size[0]}x{size[1]}")
    if len(corners) < 4:
        print("tbot calib-intrinsics: fewer than 4 views with a detected board — nothing to calibrate",
              file=sys.stderr)
        return 1

    try:
        result = calibrate_views(corners, board.grid, square_m, size, model, object_points=objects)
    except (ValueError, RuntimeError, cv2.error) as e:
        print(f"tbot calib-intrinsics: calibration failed: {e}", file=sys.stderr)
        return 1

    K = np.asarray(result["K"], dtype=np.float64)
    D = np.asarray(result["D"], dtype=np.float64)
    fov = fov_deg(K, D, model, size)
    fx_prior = nominal_fx(cfg, size[0])
    dev = 100.0 * (K[0, 0] - fx_prior) / fx_prior
    used = result["used_views"]
    per = result["per_view_rms"]

    print(f"model : {model}   views {result['n_views']}"
          + (f" (rejected {result['rejected_views']})" if result["rejected_views"] else ""))
    print(f"K     : fx {K[0, 0]:.2f}  fy {K[1, 1]:.2f}  cx {K[0, 2]:.2f}  cy {K[1, 2]:.2f}")
    print(f"D     : [{', '.join(f'{v:.5f}' for v in D)}]")
    print(f"rms   : {result['rms_px']:.3f} px   per view {min(per):.3f}..{max(per):.3f} px")
    print(f"prior : nominal fx = {cfg.optics.focal_length_mm:g} mm / {cfg.optics.pixel_pitch_um:g} um "
          f"= {fx_prior:.2f} px at {size[0]}x{size[1]}  ->  measured is {dev:+.1f} %")
    print(f"fov   : H {fov['h']:.1f}  V {fov['v']:.1f}  D {fov['d']:.1f} deg   "
          f"(vendor claim H {VENDOR_FOV_H_DEG:g} / D {VENDOR_FOV_D_DEG:g}: measured "
          f"{100.0 * fov['h'] / VENDOR_FOV_H_DEG:.0f} % / {100.0 * fov['d'] / VENDOR_FOV_D_DEG:.0f} % of it)")

    bad = []
    if result["n_views"] < MIN_VIEWS_OK:
        bad.append(f"only {result['n_views']} views survived (need {MIN_VIEWS_OK})")
    if not math.isfinite(result["rms_px"]) or result["rms_px"] > MAX_RMS_PX:
        bad.append(f"rms {result['rms_px']:.3f} px > {MAX_RMS_PX:.1f} px")
    if bad and not args.force:
        print("tbot calib-intrinsics: refusing to write " + str(out) + ": " + "; ".join(bad)
              + " — capture more/sharper views (or --force)", file=sys.stderr)
        return 1

    cov = coverage_image([corners[i] for i in used], size)
    cov_path = calib.dir / "intrinsics_coverage.png"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cov_path), cov)

    doc = dict(result)
    doc["created"] = now_iso()
    doc["notes"] = (f"{board.describe()}, {src_name}; "
                    f"fov_deg H{fov['h']:.1f}/V{fov['v']:.1f}/D{fov['d']:.1f}; "
                    f"nominal fx {fx_prior:.1f} px ({dev:+.1f} %)"
                    + ("; WRITTEN WITH --force: " + "; ".join(bad) if bad else ""))
    doc["fov_deg"] = fov
    doc["pattern"] = [board.grid[0], board.grid[1]]
    doc["square_mm"] = float(args.square_mm)
    if isinstance(board, CharucoBoard):
        doc["charuco"] = {"squares": list(board.layout[0]), "legacy": bool(board.layout[1]),
                          "marker_mm": float(args.marker_mm), "dict": board.dictionary}
    doc["source"] = src_name
    save_json(out, doc)
    print(f"wrote : {out}  (coverage map {cov_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
