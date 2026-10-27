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
    """Accepts board views that are sharp and bring a new pose, and tracks image coverage."""

    def __init__(self, pattern: tuple[int, int], image_size: tuple[int, int], *,
                 sharpness_min: float = SHARPNESS_MIN, min_pose_dist: float = MIN_POSE_DIST,
                 grid: int = COVERAGE_GRID) -> None:
        self.pattern = pattern
        self.image_size = image_size
        self.sharpness_min = float(sharpness_min)
        self.min_pose_dist = float(min_pose_dist)
        self.accepted: list[np.ndarray] = []
        self.descriptors: list[np.ndarray] = []
        self.coverage = np.zeros((grid, grid), dtype=bool)

    def offer(self, image: np.ndarray) -> tuple[np.ndarray | None, str]:
        """(corners, reason): corners is not None only when the view was accepted."""
        corners = find_corners(image, self.pattern)
        if corners is None:
            return None, "no board"
        sharp = sharpness(image, corners)
        if sharp < self.sharpness_min:
            return None, f"blurred ({sharp:.0f} < {self.sharpness_min:.0f})"
        desc = pose_descriptor(corners, self.pattern, self.image_size)
        if self.descriptors:
            d = float(np.min(np.linalg.norm((np.stack(self.descriptors) - desc) * _POSE_WEIGHTS, axis=1)))
            if d < self.min_pose_dist:
                return None, f"pose already covered ({d:.3f} < {self.min_pose_dist:.3f})"
        self.accepted.append(corners)
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
    """cv2.fisheye.calibrate across OpenCV versions (4.x wants (N,1,3)/(N,1,2), 5.0 wants (1,N,3)/(1,N,2))."""
    n = obj[0].shape[0]
    shapes = (((1, n, 3), (1, n, 2)), ((n, 1, 3), (n, 1, 2)))
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


def _calibrate_once(obj: np.ndarray, corners: list[np.ndarray], image_size: tuple[int, int],
                    model: str) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    objs = [obj.copy() for _ in corners]
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
    per = [_per_view_rms(obj, c, rv, tv, K, D, model) for c, rv, tv in zip(corners, rvecs, tvecs)]
    return rms, K, D, per


def calibrate_views(corners_list: list[np.ndarray], pattern: tuple[int, int], square_m: float,
                    image_size: tuple[int, int], model: str = "fisheye") -> dict:
    """Calibrate from detected corners. Pure: no I/O, no config.

    corners_list: per view, the (N, 2) inner corners from find_corners (N = cols*rows, raster order).
    image_size: (width, height). model: "fisheye" (cv2.fisheye, D of 4) or "pinhole" (D of 5).
    Views whose reprojection RMS exceeds REJECT_RMS_FACTOR x the median are dropped and the calibration
    is repeated, because one mis-detected board biases every parameter.
    Returns the intrinsics.json payload (without "created"/"notes") plus per_view_rms and the view indices.
    """
    if model not in ("fisheye", "pinhole"):
        raise ValueError(f"unknown model {model!r} (fisheye | pinhole)")
    cols, rows = pattern
    n_pts = cols * rows
    views = []
    for i, c in enumerate(corners_list):
        a = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        if a.shape[0] != n_pts:
            raise ValueError(f"view {i}: {a.shape[0]} corners, expected {n_pts} for a {cols}x{rows} pattern")
        views.append(a)
    if len(views) < 4:
        raise ValueError(f"need at least 4 views, got {len(views)}")

    obj = board_object_points(pattern, square_m)
    keep = list(range(len(views)))
    rms, K, D, per = _calibrate_once(obj, [views[i] for i in keep], image_size, model)
    rejected: list[int] = []
    for _ in range(MAX_REJECT_PASSES):
        median = float(np.median(per))
        limit = REJECT_RMS_FACTOR * median
        good = [i for i, e in zip(keep, per) if e <= limit]
        if len(good) == len(keep) or len(good) < max(6, MIN_VIEWS_OK // 2):
            break
        rejected += [i for i in keep if i not in good]
        keep = good
        rms, K, D, per = _calibrate_once(obj, [views[i] for i in keep], image_size, model)

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


def _corners_from_dir(d: Path, pattern: tuple[int, int], verbose: bool) -> tuple[list[np.ndarray], tuple[int, int]]:
    files = _list_images(d)
    if not files:
        raise ValueError(f"no images in {d} (looked for {', '.join(IMAGE_EXTS)})")
    corners: list[np.ndarray] = []
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
        c = find_corners(img, pattern)
        if verbose:
            print(f"  {p.name}: {'board found' if c is not None else 'no board'}")
        if c is not None:
            corners.append(c)
    if size is None:
        raise ValueError(f"no readable images in {d}")
    return corners, size


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


def _corners_from_session(d: Path, pattern: tuple[int, int], every: int, views: int, sel_kwargs: dict,
                          save_dir: Path | None) -> tuple[list[np.ndarray], tuple[int, int]]:
    frames, meta = _open_session(d)
    size = (int(meta["width"]), int(meta["height"]))
    sel = ViewSelector(pattern, size, **sel_kwargs)
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
    return sel.accepted, size


def _save_view(save_dir: Path, index: int, image: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f"view_{index:03d}.png"), to_gray(image))


def _corners_live(cfg: Config, calib: Calibration, pattern: tuple[int, int], views: int, gui: bool,
                  timeout_s: float, sel_kwargs: dict,
                  save_dir: Path | None) -> tuple[list[np.ndarray], tuple[int, int]]:
    from tbot.camera import open_source

    source = open_source(cfg.camera.source, cfg, calib)
    source.start()
    size = (int(source.width), int(source.height))
    sel = ViewSelector(pattern, size, **sel_kwargs)
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
            if corners is not None:
                print(f"  accepted {len(sel.accepted)}/{views}  coverage "
                      f"{int(sel.coverage.sum())}/{sel.coverage.size}  {sel.coverage_hint()}")
                if save_dir is not None:
                    _save_view(save_dir, len(sel.accepted) - 1, frame.image)
            elif not gui and reason != last_report:
                print(f"  waiting: {reason}")
                last_report = reason
            if gui:
                if _gui_tick(window, frame.image, pattern, corners, sel, views):
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
    return sel.accepted, size


def _gui_tick(window: str, image: np.ndarray, pattern: tuple[int, int], corners: np.ndarray | None,
              sel: ViewSelector, views: int) -> bool:
    """Draw the live coverage view; True when the operator asked to stop (q/ESC)."""
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
        cv2.drawChessboardCorners(vis, pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
    cv2.putText(vis, f"views {len(sel.accepted)}/{views}  coverage {int(sel.coverage.sum())}"
                f"/{sel.coverage.size}  {sel.coverage_hint(limit=2)}  [q] done",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(window, vis)
    return (cv2.waitKey(1) & 0xFF) in (27, ord("q"))


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
    o = p.add_argument_group("output")
    o.add_argument("--model", choices=("fisheye", "pinhole"), default=None, help="default: optics.model")
    o.add_argument("--out", default=None, help="output JSON (default <calib-dir>/intrinsics.json)")
    o.add_argument("--gui", action="store_true", help="show the live coverage window (needs a display)")
    o.add_argument("--force", action="store_true", help="write even when the quality guards fail")
    args = p.parse_args(argv)

    cfg, calib = load_from_args(args)
    pattern = (int(args.cols), int(args.rows))
    square_m = float(args.square_mm) / 1000.0
    model = args.model or cfg.optics.model
    out = Path(args.out) if args.out else calib.path("intrinsics.json")
    views_dir = None if args.no_save_views else calib.dir / "intrinsics_views"
    sel_kwargs = {"sharpness_min": float(args.sharp_min), "min_pose_dist": float(args.min_pose_dist)}

    try:
        if args.from_dir:
            src_name = f"images in {args.from_dir}"
            print(f"scanning {args.from_dir} for a {pattern[0]}x{pattern[1]} board")
            corners, size = _corners_from_dir(Path(args.from_dir), pattern, verbose=True)
        elif args.session:
            src_name = f"session {args.session} (every {args.every})"
            print(f"scanning {src_name} for a {pattern[0]}x{pattern[1]} board")
            corners, size = _corners_from_session(Path(args.session), pattern, args.every, args.views,
                                                  sel_kwargs, views_dir)
        else:
            src_name = f"live {cfg.camera.source}"
            print(f"live capture from {cfg.camera.source}: hold the board at varied distances and tilts, "
                  f"cover the image corners (Ctrl-C or q to stop)")
            corners, size = _corners_live(cfg, calib, pattern, args.views, args.gui, args.timeout_s,
                                          sel_kwargs, views_dir)
    except (ValueError, OSError) as e:
        print(f"tbot calib-intrinsics: {e}", file=sys.stderr)
        return 1

    print(f"{len(corners)} usable view(s) at {size[0]}x{size[1]}")
    if len(corners) < 4:
        print("tbot calib-intrinsics: fewer than 4 views with a detected board — nothing to calibrate",
              file=sys.stderr)
        return 1

    try:
        result = calibrate_views(corners, pattern, square_m, size, model)
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
    doc["notes"] = (f"{pattern[0]}x{pattern[1]} board, {args.square_mm:g} mm squares, {src_name}; "
                    f"fov_deg H{fov['h']:.1f}/V{fov['v']:.1f}/D{fov['d']:.1f}; "
                    f"nominal fx {fx_prior:.1f} px ({dev:+.1f} %)"
                    + ("; WRITTEN WITH --force: " + "; ".join(bad) if bad else ""))
    doc["fov_deg"] = fov
    doc["pattern"] = [pattern[0], pattern[1]]
    doc["square_mm"] = float(args.square_mm)
    doc["source"] = src_name
    save_json(out, doc)
    print(f"wrote : {out}  (coverage map {cov_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
