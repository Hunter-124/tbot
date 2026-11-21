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

import ctypes
import math
from functools import lru_cache

import numpy as np

from ..config import Config
from ..optics import CameraModel, MountModel, range_from_size
from ..types import Detection, Track
from .tracker import ClayTracker, _MIN_COS_EL, _TENTATIVE_MISSES, _inv2, _wrap_pi

# Initial uncertainty of a new track (the first detection fixes only a direction and maybe a range), 1 sigma per
# axis: tracker.new_speed_sigma_mps and tracker.new_acc_sigma_mps2. These defaults are the old ones (25 and 12);
# the shipped config tightens them (see config/default.toml).
_NEW_SPEED_SIGMA_MPS = 25.0
_NEW_ACC_SIGMA_MPS2 = 12.0
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


# The per-track and per-detection arithmetic below is in plain floats on purpose: every NumPy call on a 3-vector
# or a 3x3 costs 15-50 us on a Pi 3B whatever it computes, and one frame used to make ~150 of them per
# detection (tracker.update with 4 tracks and 4 detections: 13.8 ms -> see docs/INTEGRATION_STATUS.md).
def _jac5(x: float, y: float, z: float) -> tuple[float, float, float, float, float]:
    """_azel_jacobian as (j00, j01, j10, j11, j12); j02 is zero."""
    rho2 = x * x + y * y
    rho2c = max(rho2, 1e-18)
    rho = math.sqrt(rho2c)
    r2 = rho2 + z * z
    return -y / rho2c, x / rho2c, -x * z / (r2 * rho), -y * z / (r2 * rho), rho / r2


def _proj2(j: tuple, c: list) -> tuple[float, float, float]:
    """(m00, m01, m11) of J C J^T for the Jacobian of _jac5 and a 3x3 C given as nested lists."""
    j00, j01, j10, j11, j12 = j
    c0, c1, c2 = c
    a0 = j00 * c0[0] + j01 * c1[0]
    a1 = j00 * c0[1] + j01 * c1[1]
    a2 = j00 * c0[2] + j01 * c1[2]
    b0 = j10 * c0[0] + j11 * c1[0] + j12 * c2[0]
    b1 = j10 * c0[1] + j11 * c1[1] + j12 * c2[1]
    b2 = j10 * c0[2] + j11 * c1[2] + j12 * c2[2]
    return a0 * j00 + a1 * j01, a0 * j10 + a1 * j11 + a2 * j12, b0 * j10 + b1 * j11 + b2 * j12


def _inv_small(m: list) -> list | None:
    """Inverse of a 2x2 or 3x3 given as nested lists (None when singular)."""
    if len(m) == 2:
        (a, b), (c, d) = m
        det = a * d - b * c
        if not det or not math.isfinite(det):
            return None
        return [[d / det, -b / det], [-c / det, a / det]]
    (a, b, c), (d, e, f), (g, h, i) = m
    c11, c12, c13 = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * c11 + b * c12 + c * c13
    if not det or not math.isfinite(det):
        return None
    return [[c11 / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [c12 / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [c13 / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def _angular_view(s: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[az, az_rate, 0, el, el_rate, 0] and the matching position covariance, in closed form.

    For overlays and telemetry only (the solver uses predict/predict_dir/angular_rate); the angular
    accelerations are not filled in."""
    px, py, pz, vx, vy, vz = s[0:6].tolist()
    j = _jac5(px, py, pz)  # rows: d az/dp, d el/dp; the angular rates are the same rows applied to v
    c00, c01, c11 = _proj2(j, S[0:3, 0:3].tolist())
    v00, _, v11 = _proj2(j, S[3:6, 3:6].tolist())
    x = np.array([math.atan2(py, px), j[0] * vx + j[1] * vy, 0.0,
                  math.atan2(pz, math.hypot(px, py)), j[2] * vx + j[3] * vy + j[4] * vz, 0.0], dtype=np.float64)
    P = np.zeros((6, 6), dtype=np.float64)
    P[0, 0], P[3, 3], P[0, 3], P[3, 0] = c00, c11, c01, c01
    P[1, 1], P[4, 4] = v00, v11
    return x, P


def _alloc9() -> tuple[np.ndarray, int, np.ndarray, int]:
    """(s, address, S, address): a 9-vector and a 9x9 on ctypes buffers, so the C filter step gets their addresses
    without ndarray.ctypes (12 us a pointer on a Pi 3B, four pointers a call, ten calls a frame)."""
    b9, b81 = (ctypes.c_double * 9)(), (ctypes.c_double * 81)()
    return (np.frombuffer(b9, dtype=np.float64), ctypes.addressof(b9),
            np.frombuffer(b81, dtype=np.float64).reshape(9, 9), ctypes.addressof(b81))


_I9 = np.eye(9, dtype=np.float64)
_I9.setflags(write=False)
_X0 = np.zeros(6, dtype=np.float64)   # placeholders of a new track's angular view (_sync_views replaces them)
_P0 = np.eye(6, dtype=np.float64)
_X0.setflags(write=False)
_P0.setflags(write=False)
_S_NEW = np.zeros((9, 9), dtype=np.float64)   # a new track's covariance: the position block is filled in
for _i in range(3):
    _S_NEW[3 + _i, 3 + _i] = _NEW_SPEED_SIGMA_MPS ** 2
    _S_NEW[6 + _i, 6 + _i] = _NEW_ACC_SIGMA_MPS2 ** 2
_S_NEW.setflags(write=False)


class CartesianClayTracker(ClayTracker):
    """Multi-target tracker with a 3-D constant-acceleration EKF per clay (world frame, metres)."""

    def __init__(self, cfg: Config, cam: CameraModel, mount: MountModel):
        super().__init__(cfg, cam, mount)
        self._q3 = float(cfg.tracker.jerk_psd_mps3)
        self._twin = bool(getattr(cfg.tracker, "twin_newborn", True))
        bg = np.asarray(getattr(mount, "bore_dir_g", (1.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
        self._bore_g = bg / max(float(np.linalg.norm(bg)), 1e-12)
        self._young_hits = int(getattr(cfg.tracker, "young_hits", 0) or 0)
        yg = math.radians(float(getattr(cfg.tracker, "young_gate_min_deg", 0.0) or 0.0))
        self._young_floor_var = max(self._floor_var, (yg / self._gate) ** 2 if self._gate > 0 else 0.0)
        tc = cfg.tracker
        sv = float(getattr(tc, "new_speed_sigma_mps", 0.0) or _NEW_SPEED_SIGMA_MPS)
        sa = float(getattr(tc, "new_acc_sigma_mps2", 0.0) or _NEW_ACC_SIGMA_MPS2)
        if sv == _NEW_SPEED_SIGMA_MPS and sa == _NEW_ACC_SIGMA_MPS2:
            self._s_new = _S_NEW
        else:
            S = np.zeros((9, 9), dtype=np.float64)
            for i in range(3):
                S[3 + i, 3 + i] = sv * sv
                S[6 + i, 6 + i] = sa * sa
            S.setflags(write=False)
            self._s_new = S
        # the filter step in C when the kernel is built (vision/_measure.c: tbot_track_predict / _update); the
        # NumPy code below is the reference and what runs without it
        self._nat = None
        if bool(getattr(cfg.detector, "native", True)):
            from . import _native  # noqa: PLC0415
            self._nat = _native.track_kernel()
        self._upd_id: int | None = None
        self._addr: dict[int, tuple] = {}   # track id -> (s, its address, S, its address) of arrays made by _alloc9
        self._h3 = (ctypes.c_double * 9)()
        self._z3 = (ctypes.c_double * 3)()
        self._r3 = (ctypes.c_double * 3)()

    # ----------------------------------------------------------------- public API
    def update(self, t_ns: int, detections: list[Detection], R_wg: np.ndarray) -> list[Track]:
        t_ns = int(t_ns)
        rot = np.asarray(R_wg, dtype=np.float64).reshape(3, 3)
        preds = []
        nat = self._nat
        for tr in self.tracks:
            dt_us = max(0, int(round((t_ns - tr.t_ns) * 1e-3)))
            if nat is not None:
                sp, spa, Sp, Spa = _alloc9()
                c = self._addr.get(tr.track_id)
                if c is not None and c[0] is tr.s and c[2] is tr.S:
                    sa, Sa = c[1], c[3]
                else:
                    sa, Sa = tr.s.ctypes.data, tr.S.ctypes.data
                nat[0](sa, Sa, dt_us * 1e-6, self._q3, spa, Spa)
                self._addr[tr.track_id] = (sp, spa, Sp, Spa)
                preds.append((sp, Sp))
                continue
            f, qn = _matrices(dt_us, self._q3)
            preds.append((f @ tr.s, f @ tr.S @ f.T + qn))

        n_det = len(detections)
        if n_det:
            az_m, el_m, sig_az, sig_el, ifov = self._measure(detections, rot)
            usable = np.isfinite(az_m) & np.isfinite(el_m) & np.isfinite(sig_az) & (sig_az > 0.0)
        else:
            az_m = el_m = sig_az = sig_el = ifov = np.zeros(0)
            usable = np.zeros(0, dtype=bool)

        pairs = []
        if n_det:
            az_l, el_l, sa_l, se_l, ok_l = (az_m.tolist(), el_m.tolist(), sig_az.tolist(), sig_el.tolist(),
                                            usable.tolist())
            floor_var, gate2 = self._floor_var, self._gate * self._gate
            inv_cos2 = [1.0 / max(math.cos(e), _MIN_COS_EL) ** 2 if ok else 0.0 for e, ok in zip(el_l, ok_l)]
        for j, (sp, Sp) in enumerate(preds):
            if not n_det:
                break
            px, py, pz = sp[0:3].tolist()
            az_p, el_p = math.atan2(py, px), math.atan2(pz, math.hypot(px, py))
            c00, c01, c11 = _proj2(_jac5(px, py, pz), Sp[0:3, 0:3].tolist())
            # A young track's velocity rests on a few points, one of which may not have been the clay (a speck
            # it was born on): its own sigma is then far too small. Until it has young_hits detections its gate
            # is at least young_gate_min_deg, so the clay can still correct it (2026-11-19: tracks on the clay
            # were 0.15 deg off when confirmed and 0.5-1.5 deg off a frame later, and dropped it).
            fv = self._young_floor_var if self.tracks[j].hits < self._young_hits else floor_var
            for i in range(n_det):
                if not ok_l[i]:
                    continue
                d_az = _wrap_pi(az_l[i] - az_p)
                d_el = el_l[i] - el_p
                s00 = c00 + sa_l[i] ** 2 + fv * inv_cos2[i]
                s11 = c11 + se_l[i] ** 2 + fv
                s01 = c01
                det = s00 * s11 - s01 * s01
                if det <= 0.0 or not math.isfinite(det):
                    continue
                d2 = (s11 * d_az * d_az - 2.0 * s01 * d_az * d_el + s00 * d_el * d_el) / det
                if self._gate >= 0.0 and 0.0 <= d2 <= gate2:
                    # assignment cost = -2 ln(likelihood) + const = d2 + ln det S. With d2 alone a one-frame
                    # track (5 deg wide at 15 fps) is "nearest" to everything and takes the clay from the track
                    # that predicted it to a pixel; the ln det term charges a vague prediction for its width
                    pairs.append((d2 + math.log(det), j, i))

        pairs.sort()
        det_of_track: dict[int, int] = {}
        used_det: set[int] = set()
        twins: list[int] = []
        for _, j, i in pairs:
            if j in det_of_track or i in used_det:
                continue
            det_of_track[j] = i
            used_det.add(i)
            if self.tracks[j].hits == 1 and self._twin:
                # A one-frame track may have been born on something else (a speck 1 deg from the clay): its
                # second point then gives it a velocity tens of deg/s wrong, held with full confidence, and the
                # clay is rejected from the next frame on (2026-11-19, the clay in open sky lost 9 times in 30
                # frames). The detection also starts a clean track; the one that predicts it keeps it.
                twins.append(i)

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
                self._upd_id = tr.track_id
                tr.s, tr.S = self._ekf_update(sp, Sp, float(az_m[i]), float(el_m[i]), float(sig_az[i]),
                                              float(sig_el[i]), det, float(ifov[i]))
                tr.hits += 1
                tr.misses = 0
                tr.last_detection = det
                if tr.hits >= self.cfg.tracker.confirm_hits:
                    tr.confirmed = True
            self._sync_views(tr)
            # a tentative track seen twice survives a two-frame dropout (a clay crossing a light trunk merges with
            # it for a frame or two); a one-hit one does not
            limit = (self.cfg.tracker.max_misses if tr.confirmed
                     else _TENTATIVE_MISSES if tr.hits >= 2 else _TENTATIVE_MISSES - 1)
            if tr.misses <= limit:
                survivors.append(tr)
        self.tracks = survivors
        if self._addr:
            alive = {tr.track_id for tr in survivors}
            self._addr = {k: v for k, v in self._addr.items() if k in alive}
        self._slow = {}

        # new tracks nearest the bore first: the clay is where the gun points
        bore_w = rot @ self._bore_g
        new = [i for i in range(n_det) if i not in used_det and usable[i]]
        dirs = {}
        for i in new + twins:
            ce = math.cos(float(el_m[i]))
            dirs[i] = np.array([ce * math.cos(float(az_m[i])), ce * math.sin(float(az_m[i])), math.sin(float(el_m[i]))])
        new.sort(key=lambda i: -float(dirs[i] @ bore_w))
        twins.sort(key=lambda i: -float(dirs[i] @ bore_w))
        for i in new + twins:
            if not self._make_room(dirs[i], bore_w):
                continue
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
        S = track.S.tolist()
        dth, dt2, h2 = dt * h, dt * dt, h * h
        q = self._q3 * dt ** 5 / 20.0
        # position block of F S F^T + Q: pp + (dt pv + h pa + dt h va) + (...)^T + dt^2 vv + h^2 aa
        cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for i in range(3):
            Si, Svi, Sai = S[i], S[3 + i], S[6 + i]
            for k in range(i, 3):
                Sk, Svk = S[k], S[3 + k]
                c = (Si[k] + dt * (Si[3 + k] + Sk[3 + i]) + h * (Si[6 + k] + Sk[6 + i])
                     + dth * (Svi[6 + k] + Svk[6 + i]) + dt2 * Svi[3 + k] + h2 * Sai[6 + k])
                cov[i][k] = cov[k][i] = c
            cov[i][i] += q
        s = track.s.tolist()
        px, py, pz = s[0] + dt * s[3] + h * s[6], s[1] + dt * s[4] + h * s[7], s[2] + dt * s[5] + h * s[8]
        az, el = math.atan2(py, px), math.atan2(pz, math.hypot(px, py))
        c00, _, c11 = _proj2(_jac5(px, py, pz), cov)
        var = math.cos(el) ** 2 * c00 + c11
        return az, el, math.sqrt(max(float(var), 0.0))

    def predict_xyz(self, track: Track, t_ns: int) -> tuple[float, float, float]:
        """predict_dir as three floats (no array: 10 us less a call on a Pi 3B, a dozen calls a frame)."""
        dt = max(0.0, (int(t_ns) - track.t_ns) * 1e-9)
        h = 0.5 * dt * dt
        s = track.s.tolist()
        px, py, pz = s[0] + dt * s[3] + h * s[6], s[1] + dt * s[4] + h * s[7], s[2] + dt * s[5] + h * s[8]
        n = max(math.sqrt(px * px + py * py + pz * pz), 1e-12)
        return px / n, py / n, pz / n

    def predict_dir(self, track: Track, t_ns: int) -> np.ndarray:
        """Unit world direction at t_ns: the mean position only (no covariance work)."""
        return np.array(self.predict_xyz(track, t_ns), dtype=np.float64)

    def angular_rate(self, track: Track) -> float:
        """Clay angular speed (rad/s) on the sphere: transverse velocity over range."""
        px, py, pz, vx, vy, vz = track.s[0:6].tolist()
        r = max(math.sqrt(px * px + py * py + pz * pz), 1e-6)
        ux, uy, uz = px / r, py / r, pz / r
        along = vx * ux + vy * uy + vz * uz
        tx, ty, tz = vx - along * ux, vy - along * uy, vz - along * uz
        return math.sqrt(tx * tx + ty * ty + tz * tz) / r

    # ----------------------------------------------------------------- internals
    def _ekf_update(self, sp, Sp, az_m, el_m, sig_az, sig_el, det: Detection, ifov: float):
        # The measurement (az, el and, with a size, range) only looks at the position: H is zero outside its
        # first three columns, so the gain is built from the 3 position columns of the covariance.
        px, py, pz = sp[0:3].tolist()
        j00, j01, j10, j11, j12 = _jac5(px, py, pz)
        rows = [[j00, j01, 0.0], [j10, j11, j12]]
        z = [_wrap_pi(az_m - math.atan2(py, px)), el_m - math.atan2(pz, math.hypot(px, py))]
        rdiag = [sig_az ** 2, sig_el ** 2]
        r_size = self._size_range(det, ifov)
        if r_size is not None:
            r_p = math.sqrt(px * px + py * py + pz * pz)
            rn = max(r_p, 1e-9)
            rows.append([px / rn, py / rn, pz / rn])
            z.append(r_size - r_p)
            rdiag.append((float(self.cfg.tracker.size_range_sigma_frac) * r_size) ** 2)
        if self._nat is not None:
            h3, z3, r3 = self._h3, self._z3, self._r3
            m = len(rows)
            for k in range(m):
                h3[3 * k], h3[3 * k + 1], h3[3 * k + 2] = rows[k]
                z3[k] = z[k]
                r3[k] = rdiag[k]
            s_out, soa, S_out, Soa = _alloc9()
            c = self._addr.get(self._upd_id)
            if c is not None and c[0] is sp and c[2] is Sp:
                spa, Spa = c[1], c[3]
            else:
                spa, Spa = sp.ctypes.data, Sp.ctypes.data
            if self._nat[1](spa, Spa, ctypes.addressof(h3), ctypes.addressof(z3), ctypes.addressof(r3), m,
                            _RANGE_MIN_M, _RANGE_MAX_M, soa, Soa) == 0:
                if self._upd_id is not None:
                    self._addr[self._upd_id] = (s_out, soa, S_out, Soa)
                return s_out, S_out
        H3 = np.array(rows, dtype=np.float64)                 # (m, 3)
        PHt = Sp[:, 0:3] @ H3.T                               # (9, m)
        Sm = (H3 @ PHt[0:3]).tolist()
        for k, r_k in enumerate(rdiag):
            Sm[k][k] += r_k
        inv = _inv_small(Sm)
        K = PHt @ (np.array(inv, dtype=np.float64) if inv is not None else np.linalg.inv(np.array(Sm)))
        s = sp + K @ np.array(z, dtype=np.float64)
        A = _I9.copy()
        A[:, 0:3] -= K @ H3
        S = A @ Sp @ A.T + (K * np.array(rdiag, dtype=np.float64)) @ K.T   # Joseph form
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
        ce, se, ca, sa = math.cos(el), math.sin(el), math.cos(az), math.sin(az)
        u = (ce * ca, ce * sa, se)
        e_az = (-sa, ca, 0.0)
        e_el = (u[1] * e_az[2] - u[2] * e_az[1], u[2] * e_az[0] - u[0] * e_az[2], u[0] * e_az[1] - u[1] * e_az[0])
        a, b, c = sig_r ** 2, (r0 * sig_az * ce) ** 2, (r0 * sig_el) ** 2
        S = self._s_new.copy()
        S[0:3, 0:3] = [[a * u[i] * u[k] + b * e_az[i] * e_az[k] + c * e_el[i] * e_el[k] for k in range(3)]
                       for i in range(3)]
        s = np.zeros(9, dtype=np.float64)
        s[0:3] = (r0 * u[0], r0 * u[1], r0 * u[2])
        tr = Track(track_id=self._next_id, t_ns=t_ns, x=_X0, P=_P0, range_m=r0, range_sigma_m=sig_r,
                   hits=1, misses=0, age=1, confirmed=tc.confirm_hits <= 1, last_detection=det, first_t_ns=t_ns)
        tr.s, tr.S = s, S
        self._next_id += 1
        self._sync_views(tr)
        return tr

    def _sync_views(self, tr: Track) -> None:
        """Refresh the angular view and the range estimate from the 3-D state.

        Track.x is [az, az_rate, 0, el, el_rate, 0] (telemetry, overlays, the solver's clay rate). Track.P is NOT
        kept up to date for a cartesian track - nothing reads it, and building it cost more than the filter step
        it described (0.15 ms a track a frame on a Pi 3B): ask angular_cov(track) for it."""
        px, py, pz, vx, vy, vz = tr.s[0:6].tolist()
        j00, j01, j10, j11, j12 = _jac5(px, py, pz)
        tr.x = np.array([math.atan2(py, px), j00 * vx + j01 * vy, 0.0,
                         math.atan2(pz, math.hypot(px, py)), j10 * vx + j11 * vy + j12 * vz, 0.0], dtype=np.float64)
        r = math.sqrt(px * px + py * py + pz * pz)
        n = max(r, 1e-12)
        ux, uy, uz = px / n, py / n, pz / n
        (c00, c01, c02), (c10, c11, c12), (c20, c21, c22) = tr.S[0:3, 0:3].tolist()
        var = (ux * (c00 * ux + c01 * uy + c02 * uz) + uy * (c10 * ux + c11 * uy + c12 * uz)
               + uz * (c20 * ux + c21 * uy + c22 * uz))
        sig = math.sqrt(max(var, 0.0))
        tr.range_m = min(max(r, _RANGE_MIN_M), _RANGE_MAX_M)
        # size samples share a systematic error: never claim better than the correlated floor
        tr.range_sigma_m = max(sig, float(self.cfg.tracker.size_range_bias_frac) * tr.range_m)

    def angular_cov(self, tr: Track) -> np.ndarray:
        """The 6x6 covariance matching Track.x (position and rate blocks; the accelerations are not filled in)."""
        return _angular_view(tr.s, tr.S)[1]


def make_tracker(cfg: Config, cam: CameraModel, mount: MountModel) -> ClayTracker:
    """The tracker named by cfg.tracker.model ("angular" | "cartesian")."""
    model = str(getattr(cfg.tracker, "model", "angular")).lower()
    if model == "cartesian":
        if str(getattr(getattr(cfg, "vision", None), "engine", "legacy")).lower() == "motion":
            from .tracker_motion import MotionTracker  # noqa: PLC0415 - tracker_motion imports this module
            return MotionTracker(cfg, cam, mount)
        return CartesianClayTracker(cfg, cam, mount)
    if model == "angular":
        return ClayTracker(cfg, cam, mount)
    raise ValueError(f"tracker.model must be 'angular' or 'cartesian', got {model!r}")
