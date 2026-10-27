"""Gun attitude from the IMU: gravity init + gyro integration + complementary accel correction.

Contract: docs/ARCHITECTURE.md [imu]. WORLD frame W is gravity aligned (+Z up) with X = the horizontal
projection of the gun's +X axis at initialisation (heading 0). R_wg maps gun vectors into W.
"""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, Iterable

import numpy as np

from tbot.imu.mpu6050 import G_MPS2
from tbot.types import ImuSample

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tbot.config import Config

__all__ = ["AttitudeEstimator", "FixedAttitude", "exp_so3", "log_so3", "orthonormalize"]

MAX_GAP_S = 0.2  # a gap longer than this is a dropout: do not integrate blindly through it
MIN_ACCEL_NORM = 1e-3


def exp_so3(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues, exact; Taylor series near zero)."""
    w = np.asarray(v, dtype=np.float64).reshape(3)
    theta2 = float(w @ w)
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=np.float64)
    if theta2 < 1e-16:
        a, b = 1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0
    else:
        theta = math.sqrt(theta2)
        a, b = math.sin(theta) / theta, (1.0 - math.cos(theta)) / theta2
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (inverse of exp_so3)."""
    M = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_t = min(1.0, max(-1.0, (float(np.trace(M)) - 1.0) * 0.5))
    theta = math.acos(cos_t)
    w = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]], dtype=np.float64) * 0.5
    if theta < 1e-8:
        return w  # sin(theta) ~ theta: w already is theta*axis to first order
    if theta < math.pi - 1e-6:
        return w * (theta / math.sin(theta))
    # Near pi the off-diagonal part vanishes: take the axis from (R + I)/2 = axis axis^T.
    A = (M + np.eye(3)) * 0.5
    k = int(np.argmax(np.diag(A)))
    axis = A[:, k] / math.sqrt(max(A[k, k], 1e-300))
    axis = axis / np.linalg.norm(axis)
    if float(w @ axis) < 0.0:
        axis = -axis
    return axis * theta


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (Gram-Schmidt on the rows), undoing float drift."""
    M = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
    x = M[0] / np.linalg.norm(M[0])
    y = M[1] - (x @ M[1]) * x
    y = y / np.linalg.norm(y)
    M[0], M[1], M[2] = x, y, np.cross(x, y)
    return M


def _world_from_up(up_g: np.ndarray) -> np.ndarray:
    """R_wg from the world-up direction expressed in the gun frame (heading 0 = gun +X forward)."""
    z = np.asarray(up_g, dtype=np.float64).reshape(3)
    z = z / np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0])  # gun +X (bore)
    x = ref - (ref @ z) * z
    if float(x @ x) < 1e-4:  # muzzle within ~0.6 deg of vertical: use gun -Z (the gun's underside)
        ref = np.array([0.0, 0.0, -1.0])
        x = ref - (ref @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z], dtype=np.float64)  # rows map gun vectors onto the world axes


class AttitudeEstimator:
    """Gun orientation in the world frame from gun-frame IMU samples.

    Gravity initialises the tilt from the mean specific force over the first cfg.attitude.init_still_s
    (the gun must be still); after that the gyro integrates and the accelerometer pulls the estimated
    up-vector toward the measured one (tilt only, never heading) whenever the specific force magnitude
    is within cfg.attitude.accel_gate_g of g.
    """

    def __init__(self, cfg: "Config"):
        att = cfg.attitude
        self.accel_gain = float(att.accel_gain)
        self.accel_gate_m_s2 = float(att.accel_gate_g) * G_MPS2
        self.history_ns = int(float(att.history_s) * 1e9)
        self.init_still_ns = int(float(att.init_still_s) * 1e9)
        rate = float(max(1.0, cfg.imu.sample_rate_hz))
        self.max_history = max(64, int(float(att.history_s) * rate) + 8)
        self.ready = False
        self.samples_used = 0
        self.accel_updates = 0
        self.gaps = 0
        self._R = np.eye(3)
        self._last_t_ns: int | None = None
        self._init_t: list[int] = []
        self._init_accel: list[np.ndarray] = []
        self._ts: list[int] = []
        self._Rs: list[np.ndarray] = []
        self._ws: list[np.ndarray] = []

    # ------------------------------------------------------------------ input
    def update(self, samples: Iterable[ImuSample]) -> None:
        """Fold new (time ordered) gun-frame samples into the estimate."""
        for s in samples:
            t = int(s.t_ns)
            accel = np.asarray(s.accel, dtype=np.float64).reshape(3)
            gyro = np.asarray(s.gyro, dtype=np.float64).reshape(3)
            if not self.ready:
                self._init_step(t, accel, gyro)
                continue
            if self._last_t_ns is not None and t <= self._last_t_ns:
                continue  # out-of-order sample: ignore
            dt = (t - self._last_t_ns) * 1e-9
            if dt > MAX_GAP_S:
                # Dropout: integrating minutes of a single stale rate would be fiction, so bound it.
                self.gaps += 1
                dt = MAX_GAP_S
            self._R = self._R @ exp_so3(gyro * dt)
            self._accel_correct(accel)
            self._last_t_ns = t
            self.samples_used += 1
            self._push(t, gyro)
        if self.ready:
            self._R = orthonormalize(self._R)
            if self._Rs:
                self._Rs[-1] = self._R.copy()

    def _init_step(self, t_ns: int, accel: np.ndarray, gyro: np.ndarray) -> None:
        self._init_t.append(t_ns)
        self._init_accel.append(accel)
        if t_ns - self._init_t[0] < self.init_still_ns:
            return
        mean = np.mean(np.asarray(self._init_accel, dtype=np.float64), axis=0)
        if float(np.linalg.norm(mean)) < MIN_ACCEL_NORM:
            self._init_t.clear()
            self._init_accel.clear()
            return  # free fall / dead accelerometer: cannot find "up" yet
        self._R = _world_from_up(mean)
        self._last_t_ns = t_ns
        self.ready = True
        self.samples_used += 1
        self._init_t.clear()
        self._init_accel.clear()
        self._push(t_ns, gyro)

    def _accel_correct(self, accel: np.ndarray) -> None:
        norm = float(np.linalg.norm(accel))
        if norm < MIN_ACCEL_NORM or abs(norm - G_MPS2) > self.accel_gate_m_s2:
            return
        up_meas = accel / norm
        up_pred = self._R[2]  # world +Z expressed in the gun frame
        # Rotating the body by gain * (up_meas x up_pred) tilts the estimate toward the measurement;
        # the axis is perpendicular to the up-vector, so heading is untouched.
        self._R = self._R @ exp_so3(self.accel_gain * np.cross(up_meas, up_pred))
        self.accel_updates += 1

    def _push(self, t_ns: int, gyro: np.ndarray) -> None:
        self._ts.append(t_ns)
        self._Rs.append(self._R.copy())
        self._ws.append(gyro.copy())
        cutoff = t_ns - self.history_ns
        drop = 0
        while drop + 2 < len(self._ts) and self._ts[drop] < cutoff:
            drop += 1
        over = len(self._ts) - self.max_history
        if over > drop:
            drop = over
        if drop > 0:
            del self._ts[:drop]
            del self._Rs[:drop]
            del self._ws[:drop]

    # ------------------------------------------------------------------ output
    def R_wg(self, t_ns: int) -> np.ndarray:
        """Gun -> world rotation at t_ns (interpolated in history, extrapolated past the last sample)."""
        if not self.ready or not self._ts:
            return np.eye(3)
        t = int(t_ns)
        if t >= self._ts[-1]:
            dt = (t - self._ts[-1]) * 1e-9
            if dt <= 0.0:
                return self._Rs[-1].copy()
            return self._Rs[-1] @ exp_so3(self._ws[-1] * dt)
        if t <= self._ts[0]:
            return self._Rs[0].copy()
        i = bisect.bisect_right(self._ts, t)
        t0, t1 = self._ts[i - 1], self._ts[i]
        R0, R1 = self._Rs[i - 1], self._Rs[i]
        f = (t - t0) / float(t1 - t0)
        return R0 @ exp_so3(f * log_so3(R0.T @ R1))

    def omega_g(self, t_ns: int) -> np.ndarray:
        """Gun-frame angular rate (rad/s) of the sample nearest t_ns."""
        if not self._ts:
            return np.zeros(3)
        t = int(t_ns)
        i = bisect.bisect_left(self._ts, t)
        if i <= 0:
            return self._ws[0].copy()
        if i >= len(self._ts):
            return self._ws[-1].copy()
        before, after = self._ts[i - 1], self._ts[i]
        return self._ws[i - 1].copy() if (t - before) <= (after - t) else self._ws[i].copy()

    @property
    def history_len(self) -> int:
        return len(self._ts)


class FixedAttitude:
    """No IMU: the gun is treated as fixed, so the world frame is the gun frame."""

    __slots__ = ("R", "ready")

    def __init__(self, R: np.ndarray | None = None):
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.ready = True

    def update(self, samples: Iterable[ImuSample]) -> None:
        """Accepts samples so it can stand in for AttitudeEstimator; they carry no attitude here."""

    def R_wg(self, t_ns: int) -> np.ndarray:
        return self.R.copy()

    def omega_g(self, t_ns: int) -> np.ndarray:
        return np.zeros(3)
