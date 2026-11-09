"""Screen -> camera mapping for the HITL bench.

A flat screen maps to the camera's *undistorted normalised* image plane (x/z, y/z) by a plane homography,
so the fit runs on rays from `CameraModel.pixel_to_ray` (lens distortion removed) rather than on distorted
pixels. Residuals are reported in distorted pixels: a pattern of large radial residuals means the
intrinsics are wrong, not the screen.
"""

from __future__ import annotations

import math

import numpy as np

from tbot.optics import CameraModel


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to 0, mean distance sqrt(2). Returns (points, T)."""
    c = pts.mean(axis=0)
    d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = math.sqrt(2.0) / d if d > 0 else 1.0
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (pts - c) * s, T


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalised DLT: H with dst ~ H @ src (homogeneous). src, dst (N,2), N >= 4."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError(f"need >= 4 point pairs of equal count, got {src.shape} and {dst.shape}")
    a, Ta = _normalise(src)
    b, Tb = _normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2]


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    return q[:, :2] / q[:, 2:3]


class ScreenMap:
    """Screen device pixels <-> camera rays / distorted camera pixels."""

    def __init__(self, H: np.ndarray, screen_w: int, screen_h: int, cam: CameraModel):
        self.H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        self.Hinv = np.linalg.inv(self.H)
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.cam = cam
        self.rms_px = float("nan")
        self.max_px = float("nan")
        self.n_points = 0

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, screen_xy, uv, cam: CameraModel, screen_w: int, screen_h: int) -> "ScreenMap":
        screen_xy = np.asarray(screen_xy, dtype=np.float64).reshape(-1, 2)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        rays = np.asarray(cam.pixel_to_ray(uv), dtype=np.float64).reshape(-1, 3)
        norm = rays[:, :2] / rays[:, 2:3]
        smap = cls(fit_homography(screen_xy, norm), screen_w, screen_h, cam)
        err = np.linalg.norm(smap.to_pixel(screen_xy) - uv, axis=1)
        smap.rms_px = float(np.sqrt(np.mean(err**2)))
        smap.max_px = float(err.max())
        smap.n_points = int(len(uv))
        return smap

    def residuals(self, screen_xy, uv) -> np.ndarray:
        return self.to_pixel(screen_xy) - np.asarray(uv, dtype=np.float64).reshape(-1, 2)

    # ---------------------------------------------------------------- mapping
    def to_norm(self, xy) -> np.ndarray:
        return _apply(self.H, xy)

    def to_ray(self, xy) -> np.ndarray:
        n = self.to_norm(xy)
        r = np.column_stack([n, np.ones(len(n))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def to_pixel(self, xy) -> np.ndarray:
        return np.asarray(self.cam.ray_to_pixel(self.to_ray(xy)), dtype=np.float64).reshape(-1, 2)

    def from_ray(self, rays) -> np.ndarray:
        r = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        return _apply(self.Hinv, r[:, :2] / r[:, 2:3])

    def from_pixel(self, uv) -> np.ndarray:
        return self.from_ray(self.cam.pixel_to_ray(np.asarray(uv, dtype=np.float64).reshape(-1, 2)))

    def px_angle(self, xy) -> np.ndarray:
        """Local angle (rad) subtended by one screen pixel at screen point(s), mean of x and y."""
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.empty(len(p))
        for i, (x, y) in enumerate(p):
            r = self.to_ray([[x - 0.5, y], [x + 0.5, y], [x, y - 0.5], [x, y + 0.5]])
            ax = math.acos(min(1.0, float(r[0] @ r[1])))
            ay = math.acos(min(1.0, float(r[2] @ r[3])))
            out[i] = 0.5 * (ax + ay)
        return out

    def radius_for(self, xy, angular_radius_rad: float) -> float:
        """Screen radius (px) that subtends `angular_radius_rad` at screen point xy."""
        return float(angular_radius_rad / self.px_angle(xy)[0])

    def visible(self, xy, margin_px: float = 0.0) -> bool:
        x, y = (float(v) for v in np.asarray(xy, dtype=np.float64).reshape(2))
        return margin_px <= x <= self.screen_w - margin_px and margin_px <= y <= self.screen_h - margin_px

    def camera_mask(self, width: int, height: int, inset_px: float = 0.0) -> np.ndarray:
        """uint8 (H,W): 255 where the camera pixel sees the screen at least `inset_px` screen px inside its edge."""
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        xy = self.from_pixel(np.column_stack([uu.ravel(), vv.ravel()]))
        ok = ((xy[:, 0] >= inset_px) & (xy[:, 0] <= self.screen_w - inset_px)
              & (xy[:, 1] >= inset_px) & (xy[:, 1] <= self.screen_h - inset_px))
        rays = np.asarray(self.cam.pixel_to_ray(np.column_stack([uu.ravel(), vv.ravel()])))
        ok &= rays[:, 2] > 0
        return (ok.reshape(height, width) * 255).astype(np.uint8)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "H_screen_to_norm": self.H.tolist(),
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "cam_w": int(self.cam.width),
            "cam_h": int(self.cam.height),
            "rms_px": self.rms_px,
            "max_px": self.max_px,
            "n_points": self.n_points,
        }

    @classmethod
    def from_dict(cls, d: dict, cam: CameraModel) -> "ScreenMap":
        if (int(d["cam_w"]), int(d["cam_h"])) != (int(cam.width), int(cam.height)):
            raise ValueError(f"screen map was fitted at {d['cam_w']}x{d['cam_h']}, camera is {cam.width}x{cam.height}")
        m = cls(np.asarray(d["H_screen_to_norm"], dtype=np.float64), int(d["screen_w"]), int(d["screen_h"]), cam)
        m.rms_px = float(d.get("rms_px", float("nan")) or float("nan"))
        m.max_px = float(d.get("max_px", float("nan")) or float("nan"))
        m.n_points = int(d.get("n_points", 0))
        return m


def screen_pose_homography(screen_w: int, screen_h: int, *, distance_m: float, width_m: float,
                           roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                           offset_m: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Exact screen -> normalised-camera homography of a physical screen pose (simulation and tests).

    The screen centre sits at (offset_x, offset_y, distance) in the camera frame (+x right, +y down,
    +z forward), screen +x/+y along camera +x/+y before the rotation Rz(roll) Ry(yaw) Rx(pitch).
    """
    pitch_m = float(width_m) / float(screen_w)
    S = np.array([[pitch_m, 0.0, -0.5 * screen_w * pitch_m],
                  [0.0, pitch_m, -0.5 * screen_h * pitch_m],
                  [0.0, 0.0, 1.0]])
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    R = Rz @ Ry @ Rx
    t = np.array([offset_m[0], offset_m[1], distance_m], dtype=np.float64)
    H = np.column_stack([R[:, 0], R[:, 1], t]) @ S
    return H / H[2, 2]
