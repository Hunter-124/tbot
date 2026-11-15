"""Camera projection, mount geometry, barrel masking and geometric helpers.

Contract: docs/ARCHITECTURE.md, section "tbot/optics.py [optics]".

Frames: CAMERA C is the OpenCV frame (+x image right, +y image down, +z along the optical
axis), GUN G is the layout frame (+X toward the muzzle, +Y gun-left, +Z up). Pixels are
(u, v) = (column, row) of the UPRIGHT image at the runtime resolution, distorted, with pixel
centres at integer coordinates. All internal lengths are metres and all angles radians; the
config carries mm/deg and is converted here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Calibration, Config

__all__ = [
    "CameraModel",
    "MountModel",
    "barrel_mask",
    "combined_mask",
    "range_from_size",
    "rot_rpy",
    "dir_to_azel",
    "azel_to_dir",
    "NOMINAL_R_GUN_CAM",
]

# Columns are the camera axes expressed in G: camera +x = gun -Y, +y = gun -Z, +z = gun +X.
NOMINAL_R_GUN_CAM = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

_Z_EPS = 1e-9  # rays with z <= this are behind/along the image plane -> nan pixels
_NEWTON_ITERS = 32  # fisheye theta_d -> theta inversion
_NEWTON_TOL = 1e-14
_IFOV_SCALAR_MAX = 8  # ifov of up to this many points goes point by point (scalar ray path)

# barrel_mask sampling of the barrel surface
_BARREL_SLICE_STEP_M = 0.002  # axial spacing of the sampled rings
_BARREL_MIN_SLICES = 24
_BARREL_ANGLES = 72  # samples around the circumference (chord error < 0.1 % of the radius)
_BARREL_MAX_INCIDENCE_RAD = math.radians(89.0)
_DRAW_SHIFT = 4  # cv2 fixed-point sub-pixel bits for fillConvexPoly
_DRAW_LIMIT_PX = 8.0  # polygon vertices are clipped to this multiple of the image size


# --------------------------------------------------------------------------------- small helpers
def _points(value, dim: int, name: str) -> tuple[np.ndarray, bool]:
    """Return (N, dim) float64 points and whether the caller passed a single (dim,) vector."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"{name}: expected ({dim},) or (N,{dim}), got {arr.shape}")
        return arr.reshape(1, dim), True
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"{name}: expected ({dim},) or (N,{dim}), got {arr.shape}")
    return arr, False


def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in radians, about the axes of the frame."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def dir_to_azel(d: np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Direction(s) -> (azimuth, elevation) in radians: az = atan2(y, x), el = asin(z)."""
    arr = np.asarray(d, dtype=np.float64)
    if arr.shape == (3,) and np.isfinite(arr).all():
        x, y, z = map(float, arr)
        n = math.sqrt(x * x + y * y + z * z)
        if n == 0.0:
            return float("nan"), float("nan")
        return math.atan2(y / n, x / n), math.asin(max(-1.0, min(1.0, z / n)))
    pts, single = _points(d, 3, "dir_to_azel")
    n = np.linalg.norm(pts, axis=1)
    safe = np.where(n > 0.0, n, 1.0)
    unit = pts / safe[:, None]
    unit = np.where((n > 0.0)[:, None], unit, np.nan)
    az = np.arctan2(unit[:, 1], unit[:, 0])
    el = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    if single:
        return float(az[0]), float(el[0])
    return az, el


def azel_to_dir(az, el) -> np.ndarray:
    """(az, el) in radians -> unit direction(s); inverse of dir_to_azel."""
    a = np.asarray(az, dtype=np.float64)
    e = np.asarray(el, dtype=np.float64)
    a, e = np.broadcast_arrays(a, e)
    ce = np.cos(e)
    out = np.stack([ce * np.cos(a), ce * np.sin(a), np.sin(e)], axis=-1)
    return out.astype(np.float64, copy=False)


def range_from_size(
    diameter_px: float, ifov_rad: float, clay_diameter_m: float, psf_px: float
) -> float:
    """Range (m) of a clay of known size from its apparent diameter, blur-corrected."""
    d = float(diameter_px)
    ifov = float(ifov_rad)
    if not (ifov > 0.0):
        raise ValueError(f"range_from_size: ifov_rad must be > 0, got {ifov_rad}")
    d_eff = math.sqrt(max(d * d - float(psf_px) ** 2, 0.25))
    return float(clay_diameter_m) / (d_eff * ifov)


# --------------------------------------------------------------------------------- camera model
class CameraModel:
    """Intrinsic projection model: "fisheye" (cv2.fisheye, D len 4) or "pinhole" (D len 5)."""

    __slots__ = ("width", "height", "K", "D", "model")

    def __init__(self, width: int, height: int, K, D, model: str = "fisheye") -> None:
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"CameraModel: bad size {width}x{height}")
        self.K = np.array(K, dtype=np.float64).reshape(3, 3)
        self.D = np.array(D, dtype=np.float64).reshape(-1)
        m = str(model).lower()
        if m not in ("fisheye", "pinhole"):
            raise ValueError(f"CameraModel: unknown model {model!r} (fisheye|pinhole)")
        self.model = m
        if m == "fisheye" and self.D.size != 4:
            raise ValueError(f"fisheye needs 4 distortion coefficients, got {self.D.size}")
        if m == "pinhole" and self.D.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"pinhole needs 4/5/8/12/14 distortion coefficients, got {self.D.size}")
        if not (self.K[0, 0] > 0.0 and self.K[1, 1] > 0.0):
            raise ValueError("CameraModel: fx and fy must be positive")

    # ---- construction ---------------------------------------------------------------------
    @classmethod
    def nominal(cls, cfg: "Config", width: int, height: int) -> "CameraModel":
        """Prior model from cfg.optics (no calibration): f = focal/pitch, centred, D = 0."""
        o = cfg.optics
        if o.pixel_pitch_um <= 0.0 or o.sensor_width_px <= 0:
            raise ValueError("optics: pixel_pitch_um and sensor_width_px must be positive")
        f_full_px = float(o.focal_length_mm) / (float(o.pixel_pitch_um) * 1e-3)
        f = f_full_px * (float(width) / float(o.sensor_width_px))
        K = np.array(
            [[f, 0.0, (width - 1) * 0.5], [0.0, f, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        model = str(o.model).lower()
        n_d = 4 if model == "fisheye" else 5
        return cls(width, height, K, np.zeros(n_d, dtype=np.float64), model)

    @classmethod
    def from_calibration(
        cls, intr: dict | None, cfg: "Config", width: int, height: int
    ) -> "CameraModel":
        """Model from an intrinsics.json dict (None => nominal), rescaled to the runtime size.

        Same aspect ratio (binning/scaling) is rescaled about pixel centres
        (c' = (c + 0.5) * s - 0.5); an aspect mismatch > 1 % is a calibration error.
        """
        if not intr:
            return cls.nominal(cfg, width, height)
        if "K" not in intr or "D" not in intr:
            raise ValueError("intrinsics: missing 'K' or 'D'")
        K = np.array(intr["K"], dtype=np.float64).reshape(3, 3)
        D = np.array(intr["D"], dtype=np.float64).reshape(-1)
        model = str(intr.get("model", cfg.optics.model))
        cal_w = int(intr.get("width", width))
        cal_h = int(intr.get("height", height))
        if cal_w <= 0 or cal_h <= 0:
            raise ValueError(f"intrinsics: bad calibration size {cal_w}x{cal_h}")
        ar_cal = cal_w / cal_h
        ar_run = float(width) / float(height)
        if abs(ar_cal - ar_run) > 0.01 * ar_run:
            raise ValueError(
                f"intrinsics calibrated at {cal_w}x{cal_h} (aspect {ar_cal:.4f}) cannot be used at "
                f"{width}x{height} (aspect {ar_run:.4f}): aspect differs by more than 1 %"
            )
        sx = float(width) / float(cal_w)
        sy = float(height) / float(cal_h)
        K[0, 0] *= sx
        K[0, 1] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return cls(width, height, K, D, model)

    # ---- intrinsics accessors -------------------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def skew(self) -> float:
        """OpenCV fisheye 'alpha': u = fx * (x_d + alpha * y_d) + cx."""
        return float(self.K[0, 1]) / float(self.K[0, 0])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CameraModel({self.model}, {self.width}x{self.height}, f=({self.fx:.2f},{self.fy:.2f}), "
            f"c=({self.cx:.2f},{self.cy:.2f}), D={np.array2string(self.D, precision=4)})"
        )

    # ---- projection -----------------------------------------------------------------------
    def _theta_d(self, theta: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4 = self.D
        t2 = theta * theta
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _theta_from_theta_d_scalar(self, theta_d: float) -> float:
        k1, k2, k3, k4 = map(float, self.D)
        theta = min(max(theta_d, 0.0), math.pi)
        for _ in range(_NEWTON_ITERS):
            t2 = theta * theta
            f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
            df = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
            if abs(df) < 1e-12:
                df = 1e-12
            step = f / df
            theta = min(max(theta - step, 0.0), math.pi)
            if abs(step) < _NEWTON_TOL:
                break
        return theta

    def _theta_from_theta_d(self, theta_d: np.ndarray) -> np.ndarray:
        """Newton inversion of theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)."""
        if theta_d.shape == (1,) and math.isfinite(float(theta_d[0])):
            return np.array([self._theta_from_theta_d_scalar(float(theta_d[0]))], dtype=np.float64)
        k1, k2, k3, k4 = self.D
        theta = np.clip(theta_d.astype(np.float64, copy=True), 0.0, math.pi)
        for _ in range(_NEWTON_ITERS):
            t2 = theta * theta
            f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
            df = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
            df = np.where(np.abs(df) < 1e-12, 1e-12, df)
            step = f / df
            theta = np.clip(theta - step, 0.0, math.pi)
            if np.all(np.abs(step) < _NEWTON_TOL):
                break
        return theta

    def pixel_to_ray(self, uv: np.ndarray) -> np.ndarray:
        """Distorted pixel(s) -> unit ray(s) in the camera frame. (N,2)->(N,3), (2,)->(3,)."""
        arr = np.asarray(uv, dtype=np.float64)
        if arr.shape == (2,) and np.isfinite(arr).all():
            if self.model == "fisheye":
                y = (float(arr[1]) - self.cy) / self.fy
                x = (float(arr[0]) - self.cx) / self.fx - self.skew * y
                theta_d = math.hypot(x, y)
                theta = self._theta_from_theta_d_scalar(theta_d)
                scale = math.sin(theta) / theta_d if theta_d > 0.0 else 0.0
                return np.array([x * scale, y * scale, math.cos(theta)], dtype=np.float64)
            xy = cv2.undistortPoints(arr.reshape(1, 1, 2), self.K, self.D).reshape(2)
            x, y = float(xy[0]), float(xy[1])
            norm = math.sqrt(x * x + y * y + 1.0)
            return np.array([x / norm, y / norm, 1.0 / norm], dtype=np.float64)
        pts, single = _points(uv, 2, "pixel_to_ray")
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if self.model == "fisheye":
            y_d = (pts[:, 1] - self.cy) / self.fy
            x_d = (pts[:, 0] - self.cx) / self.fx - self.skew * y_d
            theta_d = np.hypot(x_d, y_d)
            theta = self._theta_from_theta_d(theta_d)
            scale = np.where(theta_d > 0.0, np.sin(theta) / np.where(theta_d > 0.0, theta_d, 1.0), 0.0)
            rays = np.stack([x_d * scale, y_d * scale, np.cos(theta)], axis=1)
        else:
            und = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D)
            xy = np.asarray(und, dtype=np.float64).reshape(-1, 2)
            rays = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float64)], axis=1)
            rays /= np.linalg.norm(rays, axis=1)[:, None]
        rays = rays.astype(np.float64, copy=False)
        return rays[0] if single else rays

    def ray_to_pixel(self, rays: np.ndarray) -> np.ndarray:
        """Camera-frame direction(s) -> distorted pixel(s); rows with z <= 1e-9 give nan."""
        arr = np.asarray(rays, dtype=np.float64)
        if arr.shape == (3,):
            x, y, z = map(float, arr)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and z > _Z_EPS):
                return np.full(2, np.nan, dtype=np.float64)
            if self.model == "fisheye":
                a, b = x / z, y / z
                r = math.hypot(a, b)
                theta = math.atan(r)
                k1, k2, k3, k4 = map(float, self.D)
                t2 = theta * theta
                theta_d = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
                scale = theta_d / r if r > 0.0 else 1.0
                return np.array(
                    [self.fx * (a * scale + self.skew * b * scale) + self.cx,
                     self.fy * b * scale + self.cy], dtype=np.float64
                )
            zeros = np.zeros(3, dtype=np.float64)
            img, _ = cv2.projectPoints(arr.reshape(1, 1, 3), zeros, zeros, self.K, self.D)
            return np.asarray(img, dtype=np.float64).reshape(2)
        pts, single = _points(rays, 3, "ray_to_pixel")
        out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        ok = np.isfinite(pts).all(axis=1) & (pts[:, 2] > _Z_EPS)
        if np.any(ok):
            p = pts[ok]
            if self.model == "fisheye":
                a = p[:, 0] / p[:, 2]
                b = p[:, 1] / p[:, 2]
                r = np.hypot(a, b)
                theta = np.arctan(r)
                theta_d = self._theta_d(theta)
                scale = np.where(r > 0.0, theta_d / np.where(r > 0.0, r, 1.0), 1.0)
                x_d = a * scale
                y_d = b * scale
                out[ok, 0] = self.fx * (x_d + self.skew * y_d) + self.cx
                out[ok, 1] = self.fy * y_d + self.cy
            else:
                zeros = np.zeros(3, dtype=np.float64)
                img, _ = cv2.projectPoints(
                    np.ascontiguousarray(p).reshape(-1, 1, 3), zeros, zeros, self.K, self.D
                )
                out[ok] = np.asarray(img, dtype=np.float64).reshape(-1, 2)
        return out[0] if single else out

    def ifov(self, uv: np.ndarray) -> np.ndarray:
        """Local angular size of one pixel (rad), mean of the u and v directions."""
        pts, single = _points(uv, 2, "ifov")
        if pts.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        h = 0.5
        if pts.shape[0] <= _IFOV_SCALAR_MAX and np.isfinite(pts).all():
            # a few points (the tracker's detections): the scalar ray path and plain math, no small-array NumPy
            # calls (four vectorised pixel_to_ray calls cost ~1.5 ms on a Pi 3B)
            out = np.empty(pts.shape[0], dtype=np.float64)
            for k in range(pts.shape[0]):
                u, v = float(pts[k, 0]), float(pts[k, 1])
                ang_u = _angle3(self.pixel_to_ray(np.array([u - h, v])), self.pixel_to_ray(np.array([u + h, v])))
                ang_v = _angle3(self.pixel_to_ray(np.array([u, v - h])), self.pixel_to_ray(np.array([u, v + h])))
                out[k] = 0.5 * (ang_u + ang_v) / (2.0 * h)
            return float(out[0]) if single else out
        du = np.array([h, 0.0], dtype=np.float64)
        dv = np.array([0.0, h], dtype=np.float64)
        ang_u = _ray_angle(self.pixel_to_ray(pts - du), self.pixel_to_ray(pts + du))
        ang_v = _ray_angle(self.pixel_to_ray(pts - dv), self.pixel_to_ray(pts + dv))
        out = 0.5 * (ang_u + ang_v) / (2.0 * h)
        return float(out[0]) if single else out


def _angle3(a: np.ndarray, b: np.ndarray) -> float:
    """_ray_angle for one pair of 3-vectors, in plain floats (same formula, same rounding order)."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), ax * bx + ay * by + az * bz)


def _ray_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (rad) between corresponding rows of two (N,3) arrays of unit-ish vectors."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.sum(a * b, axis=1)
    return np.arctan2(cross, dot)


# ---------------------------------------------------------------------------------- mount model
class MountModel:
    """Rigid camera/barrel geometry in the gun frame (cfg.mount + calib/boresight.json)."""

    __slots__ = ("R_gun_cam", "cam_pos_g", "muzzle_pos_g", "bore_dir_g")

    def __init__(self, R_gun_cam, cam_pos_g, muzzle_pos_g, bore_dir_g) -> None:
        self.R_gun_cam = np.array(R_gun_cam, dtype=np.float64).reshape(3, 3)
        self.cam_pos_g = np.array(cam_pos_g, dtype=np.float64).reshape(3)
        self.muzzle_pos_g = np.array(muzzle_pos_g, dtype=np.float64).reshape(3)
        d = np.array(bore_dir_g, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n <= 0.0:
            raise ValueError("MountModel: bore_dir_g must be non-zero")
        self.bore_dir_g = d / n

    @classmethod
    def from_config(cls, cfg: "Config", boresight: dict | None = None) -> "MountModel":
        """Build from cfg.mount, overridden by boresight.json ('rpy_deg', 'poi_offset_rad')."""
        m = cfg.mount
        rpy_deg = m.cam_rpy_deg
        if boresight and boresight.get("rpy_deg") is not None:
            rpy_deg = boresight["rpy_deg"]
        rpy = np.asarray(rpy_deg, dtype=np.float64).reshape(3) * (math.pi / 180.0)
        R = rot_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])) @ NOMINAL_R_GUN_CAM

        bore = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if boresight and boresight.get("poi_offset_rad") is not None:
            poi = np.asarray(boresight["poi_offset_rad"], dtype=np.float64).reshape(2)
            bore = azel_to_dir(float(poi[0]), float(poi[1]))  # yaw -> +Y, pitch -> +Z

        return cls(
            R,
            np.asarray(m.cam_pos_mm, dtype=np.float64).reshape(3) * 1e-3,
            np.asarray(m.muzzle_pos_mm, dtype=np.float64).reshape(3) * 1e-3,
            bore,
        )

    def cam_to_gun(self, v_c: np.ndarray) -> np.ndarray:
        pts, single = _points(v_c, 3, "cam_to_gun")
        out = pts @ self.R_gun_cam.T
        return out[0] if single else out

    def gun_to_cam(self, v_g: np.ndarray) -> np.ndarray:
        pts, single = _points(v_g, 3, "gun_to_cam")
        out = pts @ self.R_gun_cam
        return out[0] if single else out

    def bore_point_g(self, range_m: float) -> np.ndarray:
        """Point on the shot line at range_m from the muzzle (gun frame, m)."""
        return self.muzzle_pos_g + float(range_m) * self.bore_dir_g

    def bore_pixel(self, cam: CameraModel, range_m: float) -> tuple[float, float]:
        """Pixel where the shot line at range_m appears (camera/muzzle parallax included)."""
        v_c = self.gun_to_cam(self.bore_point_g(range_m) - self.cam_pos_g)
        uv = cam.ray_to_pixel(v_c)
        return float(uv[0]), float(uv[1])


# ----------------------------------------------------------------------------------- barrel mask
def _barrel_radius_m(profile, x_mm: np.ndarray) -> np.ndarray:
    """Barrel radius (m) at axial station(s) x_mm from the (X, OD) table, linearly extrapolated."""
    tab = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
    if tab.shape[0] == 0:
        raise ValueError("mount.barrel_od_mm is empty")
    order = np.argsort(tab[:, 0])
    xs = tab[order, 0]
    od = tab[order, 1]
    x = np.asarray(x_mm, dtype=np.float64)
    if xs.shape[0] == 1:
        out = np.full(x.shape, od[0], dtype=np.float64)
    else:
        out = np.interp(x, xs, od)
        lo = x < xs[0]
        if np.any(lo):
            slope = (od[1] - od[0]) / (xs[1] - xs[0])
            out = np.where(lo, od[0] + slope * (x - xs[0]), out)
        hi = x > xs[-1]
        if np.any(hi):
            slope = (od[-1] - od[-2]) / (xs[-1] - xs[-2])
            out = np.where(hi, od[-1] + slope * (x - xs[-1]), out)
    return np.maximum(out, 0.0) * 0.5e-3


def barrel_mask(cam: CameraModel, mount: MountModel, cfg: "Config") -> np.ndarray:
    """uint8 (H,W) mask of the sky: 255 usable, 0 where the barrel occludes the view.

    The barrel is the cylinder about (Y 0, Z cfg.mount.barrel_axis_z_mm) from the lens tip to
    the muzzle, with the OD profile of cfg.mount.barrel_od_mm. Its surface is sampled, projected
    with `cam`, filled slice pair by slice pair and grown by cfg.mount.barrel_mask_margin_px.
    """
    h, w = cam.height, cam.width
    barrel = np.zeros((h, w), dtype=np.uint8)
    m = cfg.mount

    x0 = float(m.lens_tip_x_mm) * 1e-3
    x1 = float(np.asarray(m.muzzle_pos_mm, dtype=np.float64).reshape(3)[0]) * 1e-3
    if x1 > x0:
        n_slices = max(_BARREL_MIN_SLICES, int(math.ceil((x1 - x0) / _BARREL_SLICE_STEP_M)) + 1)
        xs = np.linspace(x0, x1, n_slices)
        radii = _barrel_radius_m(m.barrel_od_mm, xs * 1e3)
        # Degenerate rings at both ends close the end caps, so the filled silhouette is solid.
        xs = np.concatenate([[x0], xs, [x1]])
        radii = np.concatenate([[0.0], radii, [0.0]])

        phi = np.linspace(0.0, 2.0 * math.pi, _BARREL_ANGLES, endpoint=False)
        z_axis = float(m.barrel_axis_z_mm) * 1e-3
        pts_g = np.empty((xs.shape[0], _BARREL_ANGLES, 3), dtype=np.float64)
        pts_g[:, :, 0] = xs[:, None]
        pts_g[:, :, 1] = radii[:, None] * np.sin(phi)[None, :]
        pts_g[:, :, 2] = z_axis + radii[:, None] * np.cos(phi)[None, :]

        d_g = pts_g - mount.cam_pos_g
        d_c = d_g.reshape(-1, 3) @ mount.R_gun_cam
        norm = np.linalg.norm(d_c, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_inc = np.where(norm > 0.0, d_c[:, 2] / np.where(norm > 0.0, norm, 1.0), -1.0)
        uv = cam.ray_to_pixel(d_c)
        valid = (cos_inc > math.cos(_BARREL_MAX_INCIDENCE_RAD)) & np.isfinite(uv).all(axis=1)
        uv = uv.reshape(xs.shape[0], _BARREL_ANGLES, 2)
        valid = valid.reshape(xs.shape[0], _BARREL_ANGLES)

        nxt = np.roll(np.arange(_BARREL_ANGLES), -1)
        quads = np.stack(
            [uv[:-1, :, :], uv[:-1, nxt, :], uv[1:, nxt, :], uv[1:, :, :]], axis=2
        )  # (slices-1, angles, 4, 2)
        keep = valid[:-1, :] & valid[:-1, nxt] & valid[1:, nxt] & valid[1:, :]
        lim = _DRAW_LIMIT_PX * max(w, h)
        q_min = quads.min(axis=2)
        q_max = quads.max(axis=2)
        keep &= (
            (q_max[:, :, 0] >= 0.0)
            & (q_min[:, :, 0] <= w - 1)
            & (q_max[:, :, 1] >= 0.0)
            & (q_min[:, :, 1] <= h - 1)
        )
        if np.any(keep):
            sel = np.clip(quads[keep], -lim, lim)
            polys = np.rint(sel * (1 << _DRAW_SHIFT)).astype(np.int32)
            for poly in polys:
                cv2.fillConvexPoly(barrel, poly, 255, cv2.LINE_8, _DRAW_SHIFT)

    margin = int(m.barrel_mask_margin_px)
    if margin > 0 and np.any(barrel):
        k = 2 * margin + 1
        acc = cv2.boxFilter(
            (barrel > 0).astype(np.float32),
            -1,
            (k, k),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        barrel = (np.asarray(acc) > 0.5).astype(np.uint8) * 255

    return np.where(barrel > 0, np.uint8(0), np.uint8(255)).astype(np.uint8)


def combined_mask(
    cam: CameraModel, mount: MountModel, cfg: "Config", calib: "Calibration"
) -> np.ndarray:
    """barrel_mask AND the static calibration mask (resized NEAREST) when one exists."""
    mask = barrel_mask(cam, mount, cfg)
    user = getattr(calib, "mask", None)
    if user is None:
        return mask
    u = np.asarray(user)
    if u.ndim == 3:
        u = u[:, :, 0]
    if u.ndim != 2:
        raise ValueError(f"calibration mask must be 2-D, got shape {np.asarray(user).shape}")
    if u.shape != mask.shape:
        u = cv2.resize(
            np.ascontiguousarray(u, dtype=np.uint8),
            (cam.width, cam.height),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.where((mask > 0) & (np.asarray(u) > 0), np.uint8(255), np.uint8(0)).astype(np.uint8)
