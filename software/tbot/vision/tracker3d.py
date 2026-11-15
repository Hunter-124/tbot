"""Clay tracking in 3-D world coordinates (`tracker.model = "cartesian"`).

Same interface as `ClayTracker` (angular), different state. Each track carries an extended Kalman filter on
the clay's world position, velocity and acceleration in metres, origin at the camera's optical centre:

    s = [p(3), v(3), a(3)]   with white-jerk process noise (tracker.jerk_psd_mps3, m^2/s^5)

Why: a clay's 3-D acceleration (gravity + drag + lift) changes slowly over a flight, so constant acceleration
in 3-D fits it well, while its ANGULAR motion is strongly non-polynomial whenever the range changes (an
outgoing clay's angular rate decays like 1/r^2). The angular filter then needs a short memory to keep up,
which inflates its uncertainty; here the long memory is unbiased.

Range: the measurement is a direction (az, el) and, when the detector reports a size, a noisy range from the
apparent diameter. Scaling the whole state by a constant leaves every predicted DIRECTION unchanged, so range
errors do not bias the lead angle; they only matter for time of flight and the pattern (as before).

Compatibility: `Track.x/P` keep the angular layout [az, az_rate, az_acc, el, el_rate, el_acc] as a derived
view (for overlays, telemetry and older code); the filter state lives in `Track.s/S`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..config import Config
from ..optics import CameraModel, MountModel, range_from_size
from ..types import Detection, Track
from .tracker import ClayTracker, _MIN_COS_EL, _TENTATIVE_MISSES, _inv2, _wrap_pi

# Initial uncertainty of a new track (the first detection fixes only a direction and maybe a range).
_NEW_SPEED_SIGMA_MPS = 25.0  # skeet/trap clays fly at 18-30 m/s, any direction
_NEW_ACC_SIGMA_MPS2 = 12.0  # gravity, drag (~5 m/s^2) and lift (~g at launch speed) roughly cancel vertically
_RANGE_MIN_M = 1.0
_RANGE_MAX_M = 200.0


def _norm3(v: np.ndarray) -> float:
    """np.linalg.norm of a 1-D vector (NumPy computes it as sqrt(v . v) too) without its ~65 us overhead."""
    return math.sqrt(float(v @ v))


@lru_cache(maxsize=64)
def _matrices(dt_us: int, q: float) -> tuple[np.ndarray, np.ndarray]:
    """(F, Q) for a step of dt_us microseconds; frames repeat the same step, so these are cached.
    The arrays are shared: callers must not modify them in place."""
    dt = dt_us * 1e-6
    f, qn = _transition9(dt), _noise9(dt, q)
    f.setflags(write=False)
    qn.setflags(write=False)
    return f, qn


def _transition9(dt: float) -> np.ndarray:
    f = np.eye(9, dtype=np.float64)
    for i in range(3):
        f[i, 3 + i] = dt
        f[i, 6 + i] = 0.5 * dt * dt
        f[3 + i, 6 + i] = dt
    return f


def _noise9(dt: float, q: float) -> np.ndarray:
    """White-jerk covariance, isotropic, for [p, v, a] interleaved by block."""
    dt2, dt3 = dt * dt, dt * dt * dt
    blk = q * np.array([[dt3 * dt2 / 20.0, dt2 * dt2 / 8.0, dt3 / 6.0],
                        [dt2 * dt2 / 8.0, dt3 / 3.0, dt2 / 2.0],
                        [dt3 / 6.0, dt2 / 2.0, dt]], dtype=np.float64)
    out = np.zeros((9, 9), dtype=np.float64)
    for i in range(3):
        idx = [i, 3 + i, 6 + i]
        out[np.ix_(idx, idx)] = blk
    return out


def _azel(p: np.ndarray) -> tuple[float, float]:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return math.atan2(y, x), math.atan2(z, math.hypot(x, y))


def _azel_jacobian(p: np.ndarray) -> np.ndarray:
    """d(az, el)/d(p) at p (2x3)."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    rho2 = x * x + y * y
    rho = math.sqrt(max(rho2, 1e-18))
    r2 = rho2 + z * z
    return np.array([[-y / max(rho2, 1e-18), x / max(rho2, 1e-18), 0.0],
                     [-x * z / (r2 * rho), -y * z / (r2 * rho), rho / r2]], dtype=np.float64)


def _angular_view(s: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[az, az_rate, 0, el, el_rate, 0] and the matching position covariance, in closed form.

    For overlays and telemetry only (the solver uses predict/predict_dir/angular_rate); the angular
    accelerations are not filled in."""
    p, v = s[0:3], s[3:6]
    az, el = _azel(p)
    j = _azel_jacobian(p)  # rows: d az/dp, d el/dp; the angular rates are the same rows applied to v
    rate = j @ v
    c = j @ S[0:3, 0:3] @ j.T
    jv = j @ S[3:6, 3:6] @ j.T
    x = np.array([az, rate[0], 0.0, el, rate[1], 0.0], dtype=np.float64)
    P = np.zeros((6, 6), dtype=np.float64)
    P[0, 0], P[3, 3], P[0, 3], P[3, 0] = c[0, 0], c[1, 1], c[0, 1], c[0, 1]
    P[1, 1], P[4, 4] = jv[0, 0], jv[1, 1]
    return x, P


class CartesianClayTracker(ClayTracker):
    """Multi-target tracker with a 3-D constant-acceleration EKF per clay (world frame, metres)."""

    def __init__(self, cfg: Config, cam: CameraModel, mount: MountModel):
        super().__init__(cfg, cam, mount)
        self._q3 = float(cfg.tracker.jerk_psd_mps3)

    # ----------------------------------------------------------------- public API
    def update(self, t_ns: int, detections: list[Detection], R_wg: np.ndarray) -> list[Track]:
        t_ns = int(t_ns)
        rot = np.asarray(R_wg, dtype=np.float64).reshape(3, 3)
        preds = []
        for tr in self.tracks:
            f, qn = _matrices(max(0, int(round((t_ns - tr.t_ns) * 1e-3))), self._q3)
            preds.append((f @ tr.s, f @ tr.S @ f.T + qn))

        n_det = len(detections)
        if n_det:
            az_m, el_m, sig_az, sig_el, ifov = self._measure(detections, rot)
            usable = np.isfinite(az_m) & np.isfinite(el_m) & np.isfinite(sig_az) & (sig_az > 0.0)
        else:
            az_m = el_m = sig_az = sig_el = ifov = np.zeros(0)
            usable = np.zeros(0, dtype=bool)

        pairs = []
        for j, (sp, Sp) in enumerate(preds):
            az_p, el_p = _azel(sp[0:3])
            H = _azel_jacobian(sp[0:3])
            C = H @ Sp[0:3, 0:3] @ H.T
            for i in range(n_det):
                if not usable[i]:
                    continue
                d_az = _wrap_pi(float(az_m[i]) - az_p)
                d_el = float(el_m[i]) - el_p
                floor_az = self._floor_var / max(math.cos(float(el_m[i])), _MIN_COS_EL) ** 2
                s00 = C[0, 0] + sig_az[i] ** 2 + floor_az
                s11 = C[1, 1] + sig_el[i] ** 2 + self._floor_var
                s01 = C[0, 1]
                det = s00 * s11 - s01 * s01
                if det <= 0.0 or not math.isfinite(det):
                    continue
                d2 = (s11 * d_az * d_az - 2.0 * s01 * d_az * d_el + s00 * d_el * d_el) / det
                if self._gate >= 0.0 and 0.0 <= d2 <= self._gate * self._gate:
                    pairs.append((d2, j, i))

        pairs.sort()
        det_of_track: dict[int, int] = {}
        used_det: set[int] = set()
        for _, j, i in pairs:
            if j in det_of_track or i in used_det:
                continue
            det_of_track[j] = i
            used_det.add(i)

        survivors: list[Track] = []
        for j, tr in enumerate(self.tracks):
            sp, Sp = preds[j]
            tr.age += 1
            tr.t_ns = t_ns
            i = det_of_track.get(j)
            if i is None:
                tr.s, tr.S = sp, Sp
                tr.misses += 1
            else:
                det = detections[i]
                tr.s, tr.S = self._ekf_update(sp, Sp, float(az_m[i]), float(el_m[i]), float(sig_az[i]),
                                              float(sig_el[i]), det, float(ifov[i]))
                tr.hits += 1
                tr.misses = 0
                tr.last_detection = det
                if tr.hits >= self.cfg.tracker.confirm_hits:
                    tr.confirmed = True
            self._sync_views(tr)
            limit = self.cfg.tracker.max_misses if tr.confirmed else _TENTATIVE_MISSES - 1
            if tr.misses <= limit:
                survivors.append(tr)
        self.tracks = survivors

        for i in range(n_det):
            if i in used_det or not usable[i]:
                continue
            if not self._make_room():
                break
            self.tracks.append(self._new_track(t_ns, detections[i], float(az_m[i]), float(el_m[i]),
                                               float(sig_az[i]), float(sig_el[i]), float(ifov[i])))
        return self.tracks

    def _propagate(self, track: Track, t_ns: int) -> tuple[np.ndarray, np.ndarray]:
        dt_us = max(0, int(round((int(t_ns) - track.t_ns) * 1e-3)))
        if dt_us <= 0:
            return track.s, track.S
        f, qn = _matrices(dt_us, self._q3)
        return f @ track.s, f @ track.S @ f.T + qn

    def _position(self, track: Track, t_ns: int) -> np.ndarray:
        dt = max(0.0, (int(t_ns) - track.t_ns) * 1e-9)
        s = track.s
        return s[0:3] + dt * s[3:6] + (0.5 * dt * dt) * s[6:9]

    def predict(self, track: Track, t_ns: int) -> tuple[float, float, float]:
        """(az, el, sigma_rad) at t_ns; sigma is the on-sphere 1-sigma of the predicted direction.

        Only the 3x3 position block of the propagated covariance is needed, so it is formed from the 3x3
        sub-blocks directly (the solver calls this every frame for each candidate fire time)."""
        dt = max(0.0, (int(t_ns) - track.t_ns) * 1e-9)
        h = 0.5 * dt * dt
        S = track.S
        pp, pv, pa = S[0:3, 0:3], S[0:3, 3:6], S[0:3, 6:9]
        vv, va, aa = S[3:6, 3:6], S[3:6, 6:9], S[6:9, 6:9]
        cross = dt * pv + h * pa + dt * h * va
        cov = pp + cross + cross.T + dt * dt * vv + h * h * aa
        cov = cov + np.eye(3) * (self._q3 * dt ** 5 / 20.0)
        p = self._position(track, t_ns)
        az, el = _azel(p)
        H = _azel_jacobian(p)
        C = H @ cov @ H.T
        var = math.cos(el) ** 2 * C[0, 0] + C[1, 1]
        return az, el, math.sqrt(max(float(var), 0.0))

    def predict_dir(self, track: Track, t_ns: int) -> np.ndarray:
        """Unit world direction at t_ns: the mean position only (no covariance work)."""
        p = self._position(track, t_ns)
        return p / max(_norm3(p), 1e-12)

    def angular_rate(self, track: Track) -> float:
        """Clay angular speed (rad/s) on the sphere: transverse velocity over range."""
        p, v = track.s[0:3], track.s[3:6]
        r = max(_norm3(p), 1e-6)
        u = p / r
        vt = v - float(v @ u) * u
        return _norm3(vt) / r

    # ----------------------------------------------------------------- internals
    def _ekf_update(self, sp, Sp, az_m, el_m, sig_az, sig_el, det: Detection, ifov: float):
        az_p, el_p = _azel(sp[0:3])
        H = np.zeros((2, 9), dtype=np.float64)
        H[:, 0:3] = _azel_jacobian(sp[0:3])
        z = np.array([_wrap_pi(az_m - az_p), el_m - el_p], dtype=np.float64)
        R = np.diag([sig_az ** 2, sig_el ** 2])
        r_size = self._size_range(det, ifov)
        if r_size is not None:
            r_p = _norm3(sp[0:3])
            Hr = np.zeros((1, 9), dtype=np.float64)
            Hr[0, 0:3] = sp[0:3] / max(r_p, 1e-9)
            H = np.vstack([H, Hr])
            z = np.append(z, r_size - r_p)
            R = np.diag([sig_az ** 2, sig_el ** 2, (float(self.cfg.tracker.size_range_sigma_frac) * r_size) ** 2])
        Sm = H @ Sp @ H.T + R
        K = Sp @ H.T @ np.linalg.inv(Sm)
        s = sp + K @ z
        A = np.eye(9) - K @ H
        S = A @ Sp @ A.T + K @ R @ K.T
        # keep the clay in front of the camera and at a sane range
        r = _norm3(s[0:3])
        if not (_RANGE_MIN_M <= r <= _RANGE_MAX_M) and r > 0.0:
            s[0:3] *= min(max(r, _RANGE_MIN_M), _RANGE_MAX_M) / r
        return s, 0.5 * (S + S.T)

    def _size_range(self, det: Detection, ifov: float) -> float | None:
        if not (det.diameter_px > 0.0 and ifov > 0.0):
            return None
        r = float(range_from_size(float(det.diameter_px), ifov, float(self.cfg.mount.clay_diameter_m),
                                  float(self.cfg.detector.psf_px)))
        return r if math.isfinite(r) and r > 0.0 else None

    def _new_track(self, t_ns, det, az, el, sig_az, sig_el, ifov) -> Track:
        tc = self.cfg.tracker
        r0 = self._size_range(det, ifov)
        r0, sig_r = ((r0, float(tc.size_range_sigma_frac) * r0) if r0 is not None
                     else (float(tc.range_prior_m), float(tc.range_prior_sigma_m)))
        r0 = min(max(r0, _RANGE_MIN_M), _RANGE_MAX_M)
        u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        e_az = np.array([-math.sin(az), math.cos(az), 0.0])
        e_el = np.cross(u, e_az)
        Sp = (sig_r ** 2) * np.outer(u, u) + (r0 * sig_az * math.cos(el)) ** 2 * np.outer(e_az, e_az) \
            + (r0 * sig_el) ** 2 * np.outer(e_el, e_el)
        S = np.zeros((9, 9), dtype=np.float64)
        S[0:3, 0:3] = Sp
        S[3:6, 3:6] = np.eye(3) * _NEW_SPEED_SIGMA_MPS ** 2
        S[6:9, 6:9] = np.eye(3) * _NEW_ACC_SIGMA_MPS2 ** 2
        s = np.zeros(9, dtype=np.float64)
        s[0:3] = r0 * u
        tr = Track(track_id=self._next_id, t_ns=t_ns, x=np.zeros(6), P=np.eye(6), range_m=r0, range_sigma_m=sig_r,
                   hits=1, misses=0, age=1, confirmed=tc.confirm_hits <= 1, last_detection=det, first_t_ns=t_ns)
        tr.s, tr.S = s, S
        self._next_id += 1
        self._sync_views(tr)
        return tr

    def _sync_views(self, tr: Track) -> None:
        """Refresh the angular view (x, P) and the range estimate from the 3-D state."""
        tr.x, tr.P = _angular_view(tr.s, tr.S)
        p = tr.s[0:3]
        r = _norm3(p)
        u = p / max(r, 1e-12)
        sig = math.sqrt(max(float(u @ tr.S[0:3, 0:3] @ u), 0.0))
        tr.range_m = min(max(r, _RANGE_MIN_M), _RANGE_MAX_M)
        # size samples share a systematic error: never claim better than the correlated floor
        tr.range_sigma_m = max(sig, float(self.cfg.tracker.size_range_bias_frac) * tr.range_m)


def make_tracker(cfg: Config, cam: CameraModel, mount: MountModel) -> ClayTracker:
    """The tracker named by cfg.tracker.model ("angular" | "cartesian")."""
    model = str(getattr(cfg.tracker, "model", "angular")).lower()
    if model == "cartesian":
        return CartesianClayTracker(cfg, cam, mount)
    if model == "angular":
        return ClayTracker(cfg, cam, mount)
    raise ValueError(f"tracker.model must be 'angular' or 'cartesian', got {model!r}")
