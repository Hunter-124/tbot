"""Clay tracking in the WORLD frame (contract: docs/ARCHITECTURE.md [tracker]).

Each track carries a constant-acceleration Kalman filter on (azimuth, elevation) with white-jerk
process noise, plus an independent 1-D filter for range driven by the apparent clay size.
Measurements are detection pixels turned into world directions:
    pixel -> CameraModel.pixel_to_ray -> MountModel.cam_to_gun -> R_wg -> (az, el).
Azimuth is kept unwrapped (it may leave [-pi, pi]) so a clay crossing the +/-pi seam stays on one track.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..optics import CameraModel, MountModel, azel_to_dir, dir_to_azel, range_from_size
from ..types import Detection, Track

_TAU = 2.0 * math.pi

# Initial 1-sigma of the unobserved states of a brand new track: a clay crosses at up to ~1.5 rad/s
# at skeet ranges (20 m/s at 20 m) and its angular acceleration peaks around 10 rad/s^2. [INFERENCE]
_NEW_VEL_SIGMA = 1.5  # rad/s
_NEW_ACC_SIGMA = 10.0  # rad/s^2

# Range filter: random walk with the clay's own line-of-sight speed scale (launch speed class). [INFERENCE]
_RANGE_WALK_MPS = 15.0
_RANGE_MIN_M = 1.0
_RANGE_MAX_M = 200.0

# A tentative (unconfirmed) track dies on its second consecutive miss; confirmed tracks get
# cfg.tracker.max_misses.
_TENTATIVE_MISSES = 2

# Azimuth noise blows up near the zenith (d az = d angle / cos el); clamp the divisor.
_MIN_COS_EL = 0.2


def _wrap_pi(a):
    """Wrap an angle (scalar or array) into [-pi, pi)."""
    return (a + math.pi) % _TAU - math.pi


def _transition(dt: float) -> np.ndarray:
    f = np.eye(6, dtype=np.float64)
    f[0, 1] = f[1, 2] = f[3, 4] = f[4, 5] = dt
    f[0, 2] = f[3, 5] = 0.5 * dt * dt
    return f


def _process_noise(dt: float, q: float) -> np.ndarray:
    """White-jerk covariance for one [p, v, a] axis, replicated on both axes."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    blk = q * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    qm = np.zeros((6, 6), dtype=np.float64)
    qm[:3, :3] = blk
    qm[3:, 3:] = blk
    return qm


def _propagate(x: np.ndarray, p: np.ndarray, dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        return x.copy(), p.copy()
    f = _transition(dt)
    return f @ x, f @ p @ f.T + _process_noise(dt, q)


def _inv2(m: np.ndarray) -> np.ndarray:
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if not det or not math.isfinite(det):
        return np.full((2, 2), np.inf)
    return np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]], dtype=np.float64) / det


class ClayTracker:
    """Multi-target tracker over clay detections; all state is in the world frame."""

    def __init__(self, cfg: Config, cam: CameraModel, mount: MountModel):
        self.cfg = cfg
        self.cam = cam
        self.mount = mount
        tc = cfg.tracker
        self._q = float(tc.jerk_psd_deg) * (math.pi / 180.0) ** 2  # deg^2/s^5 -> rad^2/s^5
        self._gate = float(tc.gate_sigma)
        self._gate_min = math.radians(float(tc.gate_min_deg))
        # Angular floor expressed as an extra innovation variance, so a gate of gate_sigma sigmas is
        # never tighter than gate_min_deg.
        self._floor_var = (self._gate_min / self._gate) ** 2 if self._gate > 0 else 0.0
        self.tracks: list[Track] = []
        self._next_id = 1

    # ----------------------------------------------------------------- public API
    def reset(self) -> None:
        """Drop every track and restart track numbering."""
        self.tracks = []
        self._next_id = 1

    def update(self, t_ns: int, detections: list[Detection], R_wg: np.ndarray) -> list[Track]:
        """Advance every track to t_ns, associate the detections and return the live tracks."""
        t_ns = int(t_ns)
        rot = np.asarray(R_wg, dtype=np.float64).reshape(3, 3)
        preds = [_propagate(tr.x, tr.P, max(0.0, (t_ns - tr.t_ns) * 1e-9), self._q) for tr in self.tracks]

        n_det = len(detections)
        if n_det:
            az_m, el_m, sig_az, sig_el, ifov = self._measure(detections, rot)
            usable = np.isfinite(az_m) & np.isfinite(el_m) & np.isfinite(sig_az) & (sig_az > 0.0)
        else:
            az_m = el_m = sig_az = sig_el = ifov = np.zeros(0)
            usable = np.zeros(0, dtype=bool)

        pairs = []
        for j, (xp, pp) in enumerate(preds):
            for i in range(n_det):
                if not usable[i]:
                    continue
                d_az = _wrap_pi(float(az_m[i]) - float(xp[0]))
                d_el = float(el_m[i]) - float(xp[3])
                floor_az = self._floor_var / max(math.cos(float(el_m[i])), _MIN_COS_EL) ** 2
                s = np.array(
                    [
                        [pp[0, 0] + sig_az[i] ** 2 + floor_az, pp[0, 3]],
                        [pp[3, 0], pp[3, 3] + sig_el[i] ** 2 + self._floor_var],
                    ],
                    dtype=np.float64,
                )
                inv = _inv2(s)
                d2 = (
                    inv[0, 0] * d_az * d_az
                    + (inv[0, 1] + inv[1, 0]) * d_az * d_el
                    + inv[1, 1] * d_el * d_el
                )
                if d2 >= 0.0 and math.sqrt(d2) <= self._gate:
                    pairs.append((math.sqrt(d2), j, i))

        # Greedy globally-nearest assignment (tracks and detections are few; ties broken by index).
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
            xp, pp = preds[j]
            dt = max(0.0, (t_ns - tr.t_ns) * 1e-9)
            tr.age += 1
            tr.t_ns = t_ns
            self._range_predict(tr, dt)
            i = det_of_track.get(j)
            if i is None:
                tr.x, tr.P = xp, pp
                tr.misses += 1
            else:
                det = detections[i]
                tr.x, tr.P = _kf_update(xp, pp, float(az_m[i]), float(el_m[i]), float(sig_az[i]), float(sig_el[i]))
                tr.hits += 1
                tr.misses = 0
                tr.last_detection = det
                if tr.hits >= self.cfg.tracker.confirm_hits:
                    tr.confirmed = True
                self._range_update(tr, det, float(ifov[i]))
            limit = self.cfg.tracker.max_misses if tr.confirmed else _TENTATIVE_MISSES - 1
            if tr.misses <= limit:
                survivors.append(tr)
        self.tracks = survivors

        for i in range(n_det):
            if i in used_det or not usable[i]:
                continue
            if not self._make_room():
                break
            self.tracks.append(
                self._new_track(
                    t_ns, detections[i], float(az_m[i]), float(el_m[i]),
                    float(sig_az[i]), float(sig_el[i]), float(ifov[i]),
                )
            )
        return self.tracks

    def predict(self, track: Track, t_ns: int) -> tuple[float, float, float]:
        """(az, el, sigma_rad) of the track at t_ns; sigma is the on-sphere 1-sigma of the mean."""
        x, p = _propagate(track.x, track.P, max(0.0, (int(t_ns) - track.t_ns) * 1e-9), self._q)
        el = float(x[3])
        var = math.cos(el) ** 2 * float(p[0, 0]) + float(p[3, 3])
        return float(x[0]), el, math.sqrt(max(var, 0.0))

    def predict_dir(self, track: Track, t_ns: int) -> np.ndarray:
        """Unit world direction to the track at t_ns."""
        az, el, _ = self.predict(track, t_ns)
        return np.asarray(azel_to_dir(az, el), dtype=np.float64).reshape(3)

    def best(self) -> Track | None:
        """Confirmed track with the most hits (ties: the most recently created one)."""
        best: Track | None = None
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            if best is None or (tr.hits, tr.first_t_ns, tr.track_id) > (best.hits, best.first_t_ns, best.track_id):
                best = tr
        return best

    # ----------------------------------------------------------------- internals
    def _measure(self, detections: list[Detection], rot: np.ndarray):
        uv = np.array([[d.u, d.v] for d in detections], dtype=np.float64)
        rays_c = np.asarray(self.cam.pixel_to_ray(uv), dtype=np.float64).reshape(-1, 3)
        rays_g = np.asarray(self.mount.cam_to_gun(rays_c), dtype=np.float64).reshape(-1, 3)
        rays_w = rays_g @ rot.T
        az, el = dir_to_azel(rays_w)
        az = np.atleast_1d(np.asarray(az, dtype=np.float64))
        el = np.atleast_1d(np.asarray(el, dtype=np.float64))
        ifov = np.atleast_1d(np.asarray(self.cam.ifov(uv), dtype=np.float64))
        sig = float(self.cfg.tracker.meas_sigma_px) * ifov
        sig_az = sig / np.maximum(np.cos(el), _MIN_COS_EL)
        return az, el, sig_az, sig, ifov

    def _new_track(self, t_ns, det, az, el, sig_az, sig_el, ifov) -> Track:
        x = np.array([az, 0.0, 0.0, el, 0.0, 0.0], dtype=np.float64)
        p = np.diag(
            np.array(
                [sig_az**2, _NEW_VEL_SIGMA**2, _NEW_ACC_SIGMA**2,
                 sig_el**2, _NEW_VEL_SIGMA**2, _NEW_ACC_SIGMA**2],
                dtype=np.float64,
            )
        )
        tr = Track(
            track_id=self._next_id,
            t_ns=t_ns,
            x=x,
            P=p,
            range_m=float(self.cfg.tracker.range_prior_m),
            range_sigma_m=float(self.cfg.tracker.range_prior_sigma_m),
            hits=1,
            misses=0,
            age=1,
            confirmed=self.cfg.tracker.confirm_hits <= 1,
            last_detection=det,
            first_t_ns=t_ns,
        )
        self._next_id += 1
        self._range_update(tr, det, ifov)
        return tr

    def _make_room(self) -> bool:
        """Ensure a new track fits under max_tracks by dropping the weakest tentative one."""
        if len(self.tracks) < self.cfg.tracker.max_tracks:
            return True
        weakest = None
        for tr in self.tracks:
            if tr.confirmed:
                continue
            if weakest is None or (tr.hits, -tr.misses, tr.age) < (weakest.hits, -weakest.misses, weakest.age):
                weakest = tr
        if weakest is None:
            return False
        self.tracks.remove(weakest)
        return True

    def _range_predict(self, tr: Track, dt: float) -> None:
        if dt > 0.0:
            tr.range_sigma_m = math.sqrt(tr.range_sigma_m**2 + _RANGE_WALK_MPS**2 * dt)

    def _range_update(self, tr: Track, det: Detection, ifov: float) -> None:
        if not (det.diameter_px > 0.0 and ifov > 0.0):
            return
        r_i = float(
            range_from_size(
                float(det.diameter_px), ifov, float(self.cfg.mount.clay_diameter_m),
                float(self.cfg.detector.psf_px),
            )
        )
        if not math.isfinite(r_i) or r_i <= 0.0:
            return
        var_m = (float(self.cfg.tracker.size_range_sigma_frac) * r_i) ** 2
        var = tr.range_sigma_m**2
        k = var / (var + var_m)
        r = tr.range_m + k * (r_i - tr.range_m)
        tr.range_m = min(max(r, _RANGE_MIN_M), _RANGE_MAX_M)
        # Size samples of one track share a systematic error (psf, detector bias, clay presentation),
        # so the fused sigma must not shrink like 1/sqrt(N): floor it at the correlated part.
        floor = float(self.cfg.tracker.size_range_bias_frac) * tr.range_m
        tr.range_sigma_m = max(math.sqrt((1.0 - k) * var), floor)


def _kf_update(xp, pp, az_m, el_m, sig_az, sig_el):
    """Joseph-form update of the (az, el) measurement, azimuth unwrapped onto the prediction."""
    z = np.array([float(xp[0]) + _wrap_pi(az_m - float(xp[0])), el_m], dtype=np.float64)
    h = np.zeros((2, 6), dtype=np.float64)
    h[0, 0] = 1.0
    h[1, 3] = 1.0
    r = np.diag(np.array([sig_az**2, sig_el**2], dtype=np.float64))
    s = h @ pp @ h.T + r
    k = pp @ h.T @ _inv2(s)
    x = xp + k @ (z - h @ xp)
    a = np.eye(6, dtype=np.float64) - k @ h
    p = a @ pp @ a.T + k @ r @ k.T
    return x, 0.5 * (p + p.T)
