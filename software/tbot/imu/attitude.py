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
ALPHA_WINDOW_S = 0.03  # gyro history used for the angular-acceleration slope (15 samples at 500 Hz)
ALPHA_MIN_SAMPLES = 5
MIN_ACCEL_NORM = 1e-3


def exp_so3(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues, exact; Taylor series near zero)."""
    w = np.asarray(v, dtype=np.float64).reshape(3)
    x, y, z = map(float, w)
    theta2 = x * x + y * y + z * z
    if theta2 < 1e-16:
        a, b = 1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0
    else:
        theta = math.sqrt(theta2)
        a, b = math.sin(theta) / theta, (1.0 - math.cos(theta)) / theta2
    return np.array(
        [[1.0 - b * (y * y + z * z), b * x * y - a * z, b * x * z + a * y],
         [b * y * x + a * z, 1.0 - b * (x * x + z * z), b * y * z - a * x],
         [b * z * x - a * y, b * z * y + a * x, 1.0 - b * (x * x + y * y)]],
        dtype=np.float64,
    )


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


# --- pure-float 3x3 helpers for the per-sample loop ------------------------------------------------------
# At 500 Hz the filter runs ~12 samples per frame; with numpy each 3x3 op costs microseconds of call overhead
# (7-11 ms per frame on a Pi 3B). Plain floats in row-major 9-tuples do the same maths several times faster.
def _exp9(x: float, y: float, z: float) -> tuple:
    theta2 = x * x + y * y + z * z
    if theta2 < 1e-16:
        a, b = 1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0
    else:
        theta = math.sqrt(theta2)
        a, b = math.sin(theta) / theta, (1.0 - math.cos(theta)) / theta2
    return (1.0 - b * (y * y + z * z), b * x * y - a * z, b * x * z + a * y,
            b * y * x + a * z, 1.0 - b * (x * x + z * z), b * y * z - a * x,
            b * z * x - a * y, b * z * y + a * x, 1.0 - b * (x * x + y * y))


def _mul9(A: tuple, B: tuple) -> tuple:
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = A
    b0, b1, b2, b3, b4, b5, b6, b7, b8 = B
    return (a0 * b0 + a1 * b3 + a2 * b6, a0 * b1 + a1 * b4 + a2 * b7, a0 * b2 + a1 * b5 + a2 * b8,
            a3 * b0 + a4 * b3 + a5 * b6, a3 * b1 + a4 * b4 + a5 * b7, a3 * b2 + a4 * b5 + a5 * b8,
            a6 * b0 + a7 * b3 + a8 * b6, a6 * b1 + a7 * b4 + a8 * b7, a6 * b2 + a7 * b5 + a8 * b8)


def _ortho9(R: tuple) -> tuple:
    x0, x1, x2, y0, y1, y2 = R[0], R[1], R[2], R[3], R[4], R[5]
    n = math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    x0, x1, x2 = x0 / n, x1 / n, x2 / n
    d = x0 * y0 + x1 * y1 + x2 * y2
    y0, y1, y2 = y0 - d * x0, y1 - d * x1, y2 - d * x2
    n = math.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
    y0, y1, y2 = y0 / n, y1 / n, y2 / n
    return (x0, x1, x2, y0, y1, y2, x1 * y2 - x2 * y1, x2 * y0 - x0 * y2, x0 * y1 - x1 * y0)


def _np9(R: tuple) -> np.ndarray:
    return np.array(R, dtype=np.float64).reshape(3, 3)


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
    is within cfg.attitude.accel_gate_g of g and the gun turns slower than cfg.attitude.accel_max_rate_dps.
    The correction has to be slow (seconds): the gyro is good to a fraction of a degree over a swing, while a
    hand-held gun's accelerometer reads its own motion, and a magnitude gate alone lets half a g sideways
    through (28 degrees of false tilt).

    Gyro bias (cfg.attitude.auto_bias): the stored bias is from one temperature and an MPU6050 drifts by
    ~1 deg/s over a warm-up. Whenever every gyro axis stays within bias_still_dps (peak to peak) for
    bias_still_s - a gun on a rack or a table; a held gun never is that quiet - the mean rate of that
    window is the remaining bias and is removed from then on (never more than bias_max_dps in total, so
    a slow steady rotation on a turntable cannot be mistaken for bias beyond that).
    """

    def __init__(self, cfg: "Config"):
        att = cfg.attitude
        self.accel_gain = float(att.accel_gain)
        self.accel_gate_m_s2 = float(att.accel_gate_g) * G_MPS2
        # the accelerometer only says where "up" is while the gun is not being turned (see update)
        rate = float(getattr(att, "accel_max_rate_dps", 0.0) or 0.0)
        self.accel_max_rate2 = math.radians(rate) ** 2 if rate > 0.0 else float("inf")
        self.history_ns = int(float(att.history_s) * 1e9)
        self.init_still_ns = int(float(att.init_still_s) * 1e9)
        rate = float(max(1.0, cfg.imu.sample_rate_hz))
        self.max_history = max(64, int(float(att.history_s) * rate) + 8)
        self.ready = False
        self.samples_used = 0
        self.accel_updates = 0
        self.gaps = 0
        self._R9: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self._last_t_ns: int | None = None
        self._init_t: list[int] = []
        self._init_accel: list[np.ndarray] = []
        self._ts: list[int] = []
        self._Rs: list[tuple] = []  # row-major 9-tuples
        self._ws: list[tuple] = []  # (wx, wy, wz) rad/s
        # zero-rate gyro bias update (see the class docstring)
        self.auto_bias = bool(getattr(att, "auto_bias", False))
        self._zb_window_ns = int(float(getattr(att, "bias_still_s", 1.0)) * 1e9)
        self._zb_range = math.radians(float(getattr(att, "bias_still_dps", 0.5)))
        self._zb_max = math.radians(float(getattr(att, "bias_max_dps", 2.0)))
        self.gyro_bias_resid = (0.0, 0.0, 0.0)  # rad/s, gun frame: subtracted from every sample
        self.bias_updates = 0
        self._zb_t0: int | None = None
        self._zb_n = 0
        self._zb_sum = [0.0, 0.0, 0.0]
        self._zb_lo = [0.0, 0.0, 0.0]
        self._zb_hi = [0.0, 0.0, 0.0]

    @property
    def _R(self) -> np.ndarray:
        return _np9(self._R9)

    @_R.setter
    def _R(self, value: np.ndarray) -> None:
        self._R9 = tuple(float(v) for v in np.asarray(value, dtype=np.float64).reshape(9))

    # ------------------------------------------------------------------ input
    def update(self, samples: Iterable[ImuSample]) -> None:
        """Fold new (time ordered) gun-frame samples into the estimate."""
        R = self._R9
        gain = self.accel_gain
        gate = self.accel_gate_m_s2
        rate2 = self.accel_max_rate2
        for s in samples:
            t = int(s.t_ns)
            if not self.ready:
                self._init_step(t, np.asarray(s.accel, dtype=np.float64).reshape(3),
                                np.asarray(s.gyro, dtype=np.float64).reshape(3))
                R = self._R9
                continue
            if self._last_t_ns is not None and t <= self._last_t_ns:
                continue  # out-of-order sample: ignore
            dt = (t - self._last_t_ns) * 1e-9
            if dt > MAX_GAP_S:
                # Dropout: integrating minutes of a single stale rate would be fiction, so bound it.
                self.gaps += 1
                dt = MAX_GAP_S
            g = s.gyro
            gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
            a = s.accel
            ax, ay, az = float(a[0]), float(a[1]), float(a[2])
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if self.auto_bias:
                self._zero_rate(t, gx, gy, gz, abs(norm - G_MPS2) <= gate)
                b = self.gyro_bias_resid
                gx, gy, gz = gx - b[0], gy - b[1], gz - b[2]
            R = _mul9(R, _exp9(gx * dt, gy * dt, gz * dt))
            if norm >= MIN_ACCEL_NORM and abs(norm - G_MPS2) <= gate and gx * gx + gy * gy + gz * gz <= rate2:
                # Rotating the body by gain * (up_meas x up_pred) tilts the estimate toward the measurement;
                # the axis is perpendicular to the up-vector, so heading is untouched.
                mx, my, mz = ax / norm, ay / norm, az / norm
                px, py, pz = R[6], R[7], R[8]  # world +Z expressed in the gun frame
                R = _mul9(R, _exp9(gain * (my * pz - mz * py), gain * (mz * px - mx * pz),
                                   gain * (mx * py - my * px)))
                self.accel_updates += 1
            self._last_t_ns = t
            self.samples_used += 1
            self._R9 = R
            self._push(t, (gx, gy, gz))
        if self.ready:
            R = _ortho9(self._R9)
            self._R9 = R
            if self._Rs:
                self._Rs[-1] = R

    def _zero_rate(self, t_ns: int, gx: float, gy: float, gz: float, accel_ok: bool) -> None:
        """Track a window of RAW rates; when it is long and quiet enough its mean becomes the bias."""
        lo, hi, acc = self._zb_lo, self._zb_hi, self._zb_sum
        if self._zb_t0 is not None and accel_ok:
            if gx < lo[0]: lo[0] = gx
            elif gx > hi[0]: hi[0] = gx
            if gy < lo[1]: lo[1] = gy
            elif gy > hi[1]: hi[1] = gy
            if gz < lo[2]: lo[2] = gz
            elif gz > hi[2]: hi[2] = gz
            rng = self._zb_range
            if hi[0] - lo[0] <= rng and hi[1] - lo[1] <= rng and hi[2] - lo[2] <= rng:
                acc[0] += gx
                acc[1] += gy
                acc[2] += gz
                self._zb_n += 1
                if t_ns - self._zb_t0 < self._zb_window_ns or self._zb_n < 20:
                    return
                n = self._zb_n
                mx, my, mz = acc[0] / n, acc[1] / n, acc[2] / n
                if math.sqrt(mx * mx + my * my + mz * mz) <= self._zb_max:
                    self.gyro_bias_resid = (mx, my, mz)
                    self.bias_updates += 1
        # (re)start the window at this sample
        self._zb_t0 = t_ns if accel_ok else None
        self._zb_n = 1
        acc[0], acc[1], acc[2] = gx, gy, gz
        lo[0], lo[1], lo[2] = gx, gy, gz
        hi[0], hi[1], hi[2] = gx, gy, gz

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
        self._push(t_ns, (float(gyro[0]), float(gyro[1]), float(gyro[2])))

    def _push(self, t_ns: int, gyro: tuple) -> None:
        self._ts.append(t_ns)
        self._Rs.append(self._R9)
        self._ws.append(gyro)
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
                return _np9(self._Rs[-1])
            w = self._ws[-1]
            return _np9(_mul9(self._Rs[-1], _exp9(w[0] * dt, w[1] * dt, w[2] * dt)))
        if t <= self._ts[0]:
            return _np9(self._Rs[0])
        i = bisect.bisect_right(self._ts, t)
        t0, t1 = self._ts[i - 1], self._ts[i]
        if t == t0:
            return _np9(self._Rs[i - 1])
        R0, R1 = _np9(self._Rs[i - 1]), _np9(self._Rs[i])
        f = (t - t0) / float(t1 - t0)
        return R0 @ exp_so3(f * log_so3(R0.T @ R1))

    def omega_g(self, t_ns: int) -> np.ndarray:
        """Gun-frame angular rate (rad/s) of the sample nearest t_ns."""
        if not self._ts:
            return np.zeros(3)
        t = int(t_ns)
        if t >= self._ts[-1]:
            return np.array(self._ws[-1], dtype=np.float64)
        if t <= self._ts[0]:
            return np.array(self._ws[0], dtype=np.float64)
        i = bisect.bisect_left(self._ts, t)
        # t is strictly between the endpoints.
        before, after = self._ts[i - 1], self._ts[i]
        return np.array(self._ws[i - 1] if (t - before) <= (after - t) else self._ws[i], dtype=np.float64)

    def alpha_g(self, t_ns: int, window_s: float = ALPHA_WINDOW_S) -> np.ndarray:
        """Gun-frame angular acceleration (rad/s^2): least-squares slope of the gyro over the window ending at
        t_ns. Zero until the window holds enough samples. The solver uses it to extrapolate the swing over the
        trigger delay; a constant-rate extrapolation misses by alpha * dt^2 / 2 (~2 deg for a brisk swing)."""
        if len(self._ts) < 3:
            return np.zeros(3)
        t_end = min(int(t_ns), self._ts[-1])
        lo = bisect.bisect_left(self._ts, t_end - int(window_s * 1e9))
        hi = bisect.bisect_right(self._ts, t_end)
        n = hi - lo
        if n < ALPHA_MIN_SAMPLES:
            return np.zeros(3)
        t0 = self._ts[hi - 1]
        st = stt = sx = sy = sz = stx = sty = stz = 0.0
        for k in range(lo, hi):
            tk = (self._ts[k] - t0) * 1e-9
            w = self._ws[k]
            st += tk
            stt += tk * tk
            sx += w[0]
            sy += w[1]
            sz += w[2]
            stx += tk * w[0]
            sty += tk * w[1]
            stz += tk * w[2]
        den = n * stt - st * st
        if den <= 0.0:
            return np.zeros(3)
        return np.array([(n * stx - st * sx) / den, (n * sty - st * sy) / den, (n * stz - st * sz) / den],
                        dtype=np.float64)

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
