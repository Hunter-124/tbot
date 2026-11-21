"""Track management of the motion front end (`vision.engine = "motion"`, `tracker.model = "cartesian"`).

Same filter as `CartesianClayTracker` (3-D constant-acceleration EKF per clay, world frame, metres; the native C step
when it is built), different lifecycle:

1. Live tracks first. Each track is predicted to the frame and takes at most one detection: inside its gate
   (Mahalanobis in az/el at vision.track.gate_sigma, never narrower than gate_min_px) and plausible in size (within
   init_size_ratio of the median of its recent detections). Globally smallest cost first, one detection per track.
   No twins, no young-gate widening, no ln det cost: a track only exists once its velocity is known.
2. Track-before-confirm. The detections no track took build CHAINS in background-stabilised pixels (`Detection.stab`;
   scenery stands still there, a clay moves at its angular rate against the world). Every detection is a 1-point
   chain (a "seed") for two frames; a seed pairs with a later detection when the implied speed against the scenery,
   the size and the contrast sign fit (at most init_max_branches pairs per seed, the most clay-like first). A chain
   of 2+ points predicts its next point by constant velocity (least squares over its points) and takes the
   best-fitting detection inside init_gate_px + init_gate_frac * step (wider by init_gyro_frac of the gun's own
   image motion where stab rests on the gyro alone). A chain of init_hits points over 3+ frame intervals that has
   travelled init_min_travel_px and lies on its line (RMS <= init_max_rms_px) becomes a TRACK: the EKF starts from
   its first point exactly as the parent's `_new_track` does and runs over the other points with their own times
   and attitudes, so the track is born confirmed with position, velocity and range from all of them. Chains are not
   tracks: nothing downstream sees them (`self.chains`, `self.seeds`, `self.stats` are for debugging).
3. Death. A track coasts at most vision.track.max_misses frames and max_coast_ms; one that has stopped moving
   against the scenery (stop_window_ms of detections slower than init_min_speed_dps) dies; of two tracks on the same
   object (within duplicate_deg, similar angular velocity) the older stays.

Detections without `stab` (the legacy detector, `tbot vclay`, tests) get an equivalent: the gnomonic projection of
their world direction, in focal-length pixels, about a reference direction that is re-centred on the bore when the
gun has turned more than 10 degrees from it (the points of live chains are then re-projected).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..optics import CameraModel, MountModel
from ..types import Detection, Track
from .tracker import _MIN_COS_EL, _wrap_pi
from .tracker3d import CartesianClayTracker, _alloc9, _jac5, _matrices, _proj2

_RECENTRE_COS = math.cos(math.radians(10.0))
_MIN_DEN = 0.2          # gnomonic denominator floor (a direction 78 deg off the reference is not projected)
_SEED_FRAMES = 2        # a seed may pair with a detection of the next frame or the one after
_HIST_NS = 400_000_000  # per-track detection history kept for the size median and the stop test
_SIZE_MEDIAN_N = 5
_CELL = 32.0           # grid cell of the chain search, stab px
_LINE_N = 8             # a live track's stab gate is the line through its last this many detections
_POST_BIRTH_HITS = 2    # see _promote
_ACC_TYPICAL_DPS2 = 60.0  # the angular acceleration a clay typically shows (recorded clays: 40-100 deg/s^2) and ...
_ACC_NOISE_PX = 0.5       # ... the centroid noise of a detection: together the weight of a short path's acceleration


class _Pt:
    """One unassigned (or track-assigned) detection with everything the chain logic needs, in plain floats."""

    __slots__ = ("pid", "frame", "t", "sx", "sy", "fb", "wx", "wy", "wz", "bx", "by", "bz", "reg_ok", "diam",
                 "sign", "mass", "det", "az", "el", "sa", "se", "ifov", "kx", "ky", "kz", "sd", "edge")


class _Chain:
    """A tentative line of detections (2+ points), with the running sums of its least-squares constant-velocity fit
    (time u in s from the first point, positions relative to the first point: no cancellation at large stab).
    fit = (t0, tm_s, xm, ym, vx, vy, rms) in stab px and px/s."""

    __slots__ = ("pts", "misses", "mass", "fit", "quad", "_s")

    def __init__(self, a: _Pt, b: _Pt):
        self.pts = [a, b]
        self.misses = 0
        self.mass = a.mass + b.mass
        u, X, Y = (b.t - a.t) * 1e-9, b.sx - a.sx, b.sy - a.sy
        self._s = (2, u, u * u, X, Y, u * X, u * Y, X * X, Y * Y)
        self.fit = _fit_sums(a, self._s)
        self.quad = _NO_QUAD   # the constant-acceleration term on top of fit (see _quad); none with two points

    def with_point(self, p: _Pt) -> tuple:
        """The sums with p added (not applied)."""
        a = self.pts[0]
        n, su, suu, sx, sy, sux, suy, sxx, syy = self._s
        u, X, Y = (p.t - a.t) * 1e-9, p.sx - a.sx, p.sy - a.sy
        return (n + 1, su + u, suu + u * u, sx + X, sy + Y, sux + u * X, suy + u * Y, sxx + X * X, syy + Y * Y)

    def add(self, p: _Pt, sums: tuple, fit: tuple, quad: tuple) -> None:
        self.pts.append(p)
        self.mass += p.mass
        self._s, self.fit, self.quad = sums, fit, quad
        self.misses = 0

    def refit(self, a_max: float, ridge: float = 0.0) -> None:
        """Rebuild the sums from the points (after their stab was re-projected)."""
        a = self.pts[0]
        s = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for p in self.pts:
            u, X, Y = (p.t - a.t) * 1e-9, p.sx - a.sx, p.sy - a.sy
            s[0] += 1
            s[1] += u
            s[2] += u * u
            s[3] += X
            s[4] += Y
            s[5] += u * X
            s[6] += u * Y
            s[7] += X * X
            s[8] += Y * Y
        self._s = tuple(s)
        fit = _fit_sums(a, self._s)
        self.quad = _quad(self.pts, fit, a_max, ridge)
        self.fit = fit[:6] + (self.quad[4],) if len(self.pts) >= 3 else fit

    def pids(self) -> set[int]:
        return {p.pid for p in self.pts}


def _fit_sums(a: _Pt, s: tuple) -> tuple[float, float, float, float, float, float, float]:
    """(t0, tm, xm, ym, vx, vy, rms) from a chain's sums (a = its first point)."""
    n, su, suu, sx, sy, sux, suy, sxx, syy = s
    um, xm, ym = su / n, sx / n, sy / n
    stt = suu - su * um
    if stt <= 1e-18:
        return a.t, um, a.sx + xm, a.sy + ym, 0.0, 0.0, 0.0
    vx, vy = (sux - um * sx) / stt, (suy - um * sy) / stt
    rss = (sxx - sx * xm - vx * vx * stt) + (syy - sy * ym - vy * vy * stt)
    return a.t, um, a.sx + xm, a.sy + ym, vx, vy, math.sqrt(max(rss, 0.0) / n)


def _fit(pts: list[_Pt]) -> tuple[float, float, float, float, float, float, float]:
    """Least-squares constant velocity through the points (time in s from the first point)."""
    n = len(pts)
    t0 = pts[0].t
    if n == 2:
        a, b = pts
        dt = (b.t - t0) * 1e-9
        if dt <= 0.0:
            return t0, 0.0, a.sx, a.sy, 0.0, 0.0, 0.0
        return t0, 0.5 * dt, 0.5 * (a.sx + b.sx), 0.5 * (a.sy + b.sy), (b.sx - a.sx) / dt, (b.sy - a.sy) / dt, 0.0
    ts = [(p.t - t0) * 1e-9 for p in pts]
    tm = sum(ts) / n
    xm = sum(p.sx for p in pts) / n
    ym = sum(p.sy for p in pts) / n
    stt = sxt = syt = 0.0
    for t, p in zip(ts, pts):
        d = t - tm
        stt += d * d
        sxt += d * (p.sx - xm)
        syt += d * (p.sy - ym)
    if stt <= 0.0:
        return t0, tm, xm, ym, 0.0, 0.0, 0.0
    vx, vy = sxt / stt, syt / stt
    rss = 0.0
    for t, p in zip(ts, pts):
        d = t - tm
        ex, ey = p.sx - xm - vx * d, p.sy - ym - vy * d
        rss += ex * ex + ey * ey
    return t0, tm, xm, ym, vx, vy, math.sqrt(rss / n)


_NO_QUAD = (0.0, 0.0, 0.0, 0.0, 0.0)


def _quad(pts: list[_Pt], fit: tuple, a_max: float, ridge: float = 0.0) -> tuple[float, float, float, float, float]:
    """The constant-acceleration term on top of a constant-velocity fit of the same points: (cx, cy, beta, gamma,
    rms) of the path x(d) = xm + vx d + cx (d^2 - beta d - gamma) (same in y), d = time from the fit's mean time
    in s. q(d) = d^2 - beta d - gamma is orthogonal to 1 and d over the points' times, so (xm, vx) of the fit stand
    and cx is the least-squares half-acceleration (px/s^2), limited to a_max / 2 in length.

    A clay does not cross the picture at constant speed: one going away slows down in angle as it gains distance
    (the hit of 2026-11-19 16:02:43: 1700 -> 400 px/s in 0.45 s, 3000 px/s^2), a near one curves under gravity
    (g / range: 3 px of sag in 110 ms at 10 m). A straight evenly timed line fits neither to a pixel.

    ridge: (centroid noise / the half-acceleration a clay typically shows)^2, s^4. Three or four detections say
    little about an acceleration (the quadratic through three points puts the fourth 4.4 sigma of centroid noise
    off, a line 1.5): the estimate is shrunk towards none by sqq / (sqq + ridge), which is ~0.4 for three points
    20 ms apart through the gun's lens, ~0.9 for five, and ~0 through a wide lens, where a clay's acceleration is
    a fraction of a pixel and the path is a line again.

    rms: the residual about that path, the three fitted parameters per axis taken off the count
    (sqrt(SSE / (n - 3)))."""
    n = len(pts)
    if n < 3:
        return _NO_QUAD
    t0, tm, xm, ym, vx, vy = fit[:6]
    ds = [(p.t - t0) * 1e-9 - tm for p in pts]
    s2 = s3 = 0.0
    for d in ds:
        dd = d * d
        s2 += dd
        s3 += dd * d
    if s2 <= 1e-18:
        return _NO_QUAD
    beta, gamma = s3 / s2, s2 / n
    sqq = sqx = sqy = sse = 0.0
    for d, p in zip(ds, pts):
        q = d * d - beta * d - gamma
        rx, ry = p.sx - xm - vx * d, p.sy - ym - vy * d
        sqq += q * q
        sqx += q * rx
        sqy += q * ry
        sse += rx * rx + ry * ry
    if sqq <= 1e-30:
        return 0.0, 0.0, beta, gamma, math.sqrt(sse / max(n - 2, 1))
    cx, cy = sqx / sqq, sqy / sqq
    lam = sqq / (sqq + ridge) if ridge > 0.0 else 1.0
    kx, ky = cx * lam, cy * lam
    c, lim = math.hypot(kx, ky), 0.5 * a_max
    if c > lim:
        kx, ky = kx * lim / c, ky * lim / c
    sse -= sqq * ((2.0 * cx * kx - kx * kx) + (2.0 * cy * ky - ky * ky))
    return kx, ky, beta, gamma, math.sqrt(max(sse, 0.0) / max(n - 3, 1))


def _path(fit: tuple, quad: tuple, t_ns: int, t_end_ns: int | None = None) -> tuple[float, float]:
    """Where the fitted path (constant velocity + _quad) is at t_ns. t_end_ns: the path is trusted as a curve up
    to there (one frame past its last point); beyond it goes on straight at the velocity it has there. The
    acceleration of eight noisy detections, carried on over a dropout of five frames, is pixels of error growing
    with the square of the time: the track's gate left its clay behind and the track died coasting (sim, a clay at
    5 deg/s with 15 % dropouts)."""
    t0, tm, xm, ym, vx, vy = fit[:6]
    d = (t_ns - t0) * 1e-9 - tm
    cx, cy, beta, gamma = quad[:4]
    if cx == 0.0 and cy == 0.0:
        return xm + vx * d, ym + vy * d
    if t_end_ns is not None and t_ns > t_end_ns:
        de = (t_end_ns - t0) * 1e-9 - tm
        q = de * de - beta * de - gamma
        w = 2.0 * de - beta
        return (xm + vx * de + cx * q + (vx + cx * w) * (d - de),
                ym + vy * de + cy * q + (vy + cy * w) * (d - de))
    q = d * d - beta * d - gamma
    return xm + vx * d + cx * q, ym + vy * d + cy * q


def _median(v: list[float]) -> float:
    if not v:
        return 0.0
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _angle_xyz(ax, ay, az, bx, by, bz) -> float:
    return math.acos(max(-1.0, min(1.0, ax * bx + ay * by + az * bz)))


class MotionTracker(CartesianClayTracker):
    """Cartesian EKF tracks born only from detections on one straight, evenly timed line (see module doc)."""

    def __init__(self, cfg: Config, cam: CameraModel, mount: MountModel):
        super().__init__(cfg, cam, mount)
        vt = cfg.vision.track
        self._vt = vt
        self._gate = float(vt.gate_sigma)
        self._gate_min_px = float(vt.gate_min_px)
        fx = float(getattr(cam, "fx", 0.0) or 0.0)
        self._fx = fx if fx > 0.0 else 1.0 / math.radians(1.0 / 35.0)
        # the parent's names, for whoever reads them: the floor as an angle at the image centre
        self._gate_min = self._gate_min_px / self._fx
        self._floor_var = (self._gate_min / self._gate) ** 2 if self._gate > 0 else 0.0
        self._twin = False
        self._young_hits = 0
        self._init_hits = max(2, int(vt.init_hits))
        self._win_ns = int(float(vt.init_window_ms) * 1e6)
        self._coast_ns = int(float(vt.max_coast_ms) * 1e6)
        self._stop_ns = int(float(vt.stop_window_ms) * 1e6)
        rad = math.pi / 180.0
        self._vmin = float(vt.init_min_speed_dps) * rad * self._fx   # stab px/s
        self._vmax = float(vt.init_max_speed_dps) * rad * self._fx
        self._edge_px = max(0.0, float(vt.init_edge_px))
        self._w = float(getattr(cam, "width", 640) or 640)
        self._h = float(getattr(cam, "height", 400) or 400)
        self._vmin_gyro = max(float(vt.init_min_speed_gyro_dps), float(vt.init_min_speed_dps)) * rad * self._fx
        self._amax = max(0.0, float(vt.init_max_accel_dps2)) * rad * self._fx   # stab px/s^2
        self._same_sign = bool(vt.same_sign)
        self._chain_mass_ratio = max(0.0, float(vt.init_mass_ratio))
        self._weak_mass = max(0.0, float(vt.init_weak_mass))
        self._weak_hits = max(0, int(vt.init_weak_extra_hits))
        self._ridge = (_ACC_NOISE_PX / (0.5 * _ACC_TYPICAL_DPS2 * rad * self._fx)) ** 2   # see _quad
        self._ratio = max(1.0, float(vt.init_size_ratio))
        self._dup_rad = math.radians(float(vt.duplicate_deg))
        self._mass_ratio = max(0.0, float(vt.track_mass_ratio))
        self._reset_motion()

    # ----------------------------------------------------------------- public API
    def reset(self) -> None:
        super().reset()
        self._addr = {}
        self._reset_motion()

    def _reset_motion(self) -> None:
        self.chains: list[_Chain] = []
        self.seeds: list[_Pt] = []
        self._hist: dict[int, list[_Pt]] = {}
        self._ref: tuple | None = None    # (r, e1, e2) of the world-direction fallback, 9 floats
        self._sref: tuple | None = None   # the same for the detector's stabilised directions (Detection.stab_dir)
        self._frame = 0
        self._pid = 0
        self._slow = {}
        self.stats: dict[str, int] = {
            "frames": 0, "chains": 0, "seeds": 0, "promotions": 0, "births_refused_table_full": 0,
            "rej_rms": 0, "rej_window": 0, "rej_miss": 0, "rej_cap": 0, "recentre": 0,
            "killed_coast": 0, "killed_stopped": 0, "killed_duplicate": 0, "killed_replaced": 0,
        }

    def update(self, t_ns: int, detections: list[Detection], R_wg: np.ndarray) -> list[Track]:
        t_ns = int(t_ns)
        rot = np.asarray(R_wg, dtype=np.float64).reshape(3, 3)
        self._frame += 1
        self.stats["frames"] += 1
        bgx, bgy, bgz = self._bore_g.tolist()
        (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = rot.tolist()
        bore = (r00 * bgx + r01 * bgy + r02 * bgz, r10 * bgx + r11 * bgy + r12 * bgz,
                r20 * bgx + r21 * bgy + r22 * bgz)
        self._maybe_recentre(bore)

        preds = [self._predict_one(tr, t_ns) for tr in self.tracks]
        pts = self._points(t_ns, detections, rot, bore)

        # ---- 1. live tracks take their detections
        same_sign = self._same_sign
        pairs = []
        if pts and preds:
            gate2 = self._gate * self._gate
            for j, (sp, Sp) in enumerate(preds):
                px, py, pz = sp[0:3].tolist()
                az_p, el_p = math.atan2(py, px), math.atan2(pz, math.hypot(px, py))
                c00, c01, c11 = _proj2(_jac5(px, py, pz), Sp[0:3, 0:3].tolist())
                # generous bound on the gate (rad^2): sigma^2 of the prediction plus a 10x-measurement floor
                max_d2 = gate2 * (c00 + c11 + 10.0 * (self._gate_min_px / self._fx) ** 2) * 4.0 + 1e-8
                tid = self.tracks[j].track_id
                med, med_m = self._medians(tid)
                line = self._stab_line(tid, t_ns, bore)
                for i, p in enumerate(pts):
                    if med > 0.0 and p.diam > 0.0 and not (med / self._ratio <= p.diam <= med * self._ratio):
                        continue
                    if med_m > 0.0 and 0.0 < p.mass * self._mass_ratio < med_m:
                        continue
                    if line is not None:
                        fb, sign, qx, qy, g, lok = line
                        if same_sign and sign and p.sign and sign != p.sign:
                            continue
                        # The tight stab gate holds only where stab is measured on the picture on both sides.
                        # Where it rests on the gyro (open sky, a horizon, recoil) it drifts by pixels a frame, and
                        # the track kept coasting past detections a pixel from its prediction until it died
                        # (presses 105639, 105308, 104806 of 2026-11-17, right after the shot): there the filter's
                        # own gate, the size, the mass and the sign decide.
                        if fb == p.fb and p.sx == p.sx and lok and p.reg_ok:
                            ex, ey = p.sx - qx, p.sy - qy
                            if ex * ex + ey * ey > g * g:
                                continue
                    d_az = _wrap_pi(p.az - az_p)
                    d_el = p.el - el_p
                    if d_az * d_az + d_el * d_el > max_d2:
                        continue   # farther than any gate could reach: skip the pixel size
                    self._ensure_meas(p)
                    fv = (self._gate_min_px * p.ifov / self._gate) ** 2 if self._gate > 0 else 0.0
                    s00 = c00 + p.sa * p.sa + fv / max(math.cos(p.el), _MIN_COS_EL) ** 2
                    s11 = c11 + p.se * p.se + fv
                    det = s00 * s11 - c01 * c01
                    if det <= 0.0 or not math.isfinite(det):
                        continue
                    d2 = (s11 * d_az * d_az - 2.0 * c01 * d_az * d_el + s00 * d_el * d_el) / det
                    if 0.0 <= d2 <= gate2:
                        pairs.append((d2, j, i))
        pairs.sort()
        det_of_track: dict[int, int] = {}
        used: set[int] = set()
        for _, j, i in pairs:
            if j in det_of_track or i in used:
                continue
            det_of_track[j] = i
            used.add(i)

        max_miss = int(self._vt.max_misses)
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
                p = pts[i]
                self._upd_id = tr.track_id
                tr.s, tr.S = self._ekf_update(sp, Sp, p.az, p.el, p.sa, p.se, p.det, p.ifov)
                tr.hits += 1
                tr.misses = 0
                tr.last_detection = p.det
                self._push_hist(tr.track_id, p, t_ns)
            self._sync_views(tr)
            hist = self._hist.get(tr.track_id)
            last_t = hist[-1].t if hist else tr.first_t_ns
            if tr.misses > max_miss or t_ns - last_t > self._coast_ns:
                self.stats["killed_coast"] += 1
                continue
            if tr.misses == 0 and self._stopped(hist):
                self.stats["killed_stopped"] += 1
                continue
            survivors.append(tr)
        self.tracks = survivors

        # ---- 2. chains from what no track took
        free = [p for i, p in enumerate(pts) if i not in used]
        born = self._chains_step(t_ns, free, bore)
        for ch in born:
            self._promote(ch, t_ns)

        # ---- 3. one track per object
        self._dedupe(t_ns)
        alive = {tr.track_id for tr in self.tracks}
        if self._addr:
            self._addr = {k: v for k, v in self._addr.items() if k in alive}
        if len(self._hist) != len(alive):
            self._hist = {k: v for k, v in self._hist.items() if k in alive}
        self._slow = {}
        self.stats["chains"] = len(self.chains)
        self.stats["seeds"] = len(self.seeds)
        return self.tracks

    # ----------------------------------------------------------------- filter steps (the parent's mathematics)
    def _predict_one(self, tr: Track, t_ns: int):
        """The parent's predict step for one track (native C when built), keeping its address cache."""
        dt_us = max(0, int(round((t_ns - tr.t_ns) * 1e-3)))
        nat = self._nat
        if nat is not None:
            sp, spa, Sp, Spa = _alloc9()
            c = self._addr.get(tr.track_id)
            if c is not None and c[0] is tr.s and c[2] is tr.S:
                sa, Sa = c[1], c[3]
            else:
                sa, Sa = tr.s.ctypes.data, tr.S.ctypes.data
            nat[0](sa, Sa, dt_us * 1e-6, self._q3, spa, Spa)
            self._addr[tr.track_id] = (sp, spa, Sp, Spa)
            return sp, Sp
        f, qn = _matrices(dt_us, self._q3)
        return f @ tr.s, f @ tr.S @ f.T + qn

    def _promote(self, ch: _Chain, t_ns: int) -> None:
        if not self._make_room_motion():
            self.stats["births_refused_table_full"] += 1
            return
        pts = ch.pts
        for p in pts:
            self._ensure_meas(p)
        p0 = pts[0]
        tr = self._new_track(p0.t, p0.det, p0.az, p0.el, p0.sa, p0.se, p0.ifov)
        for p in pts[1:]:
            sp, Sp = self._predict_one(tr, p.t)
            self._upd_id = tr.track_id
            tr.s, tr.S = self._ekf_update(sp, Sp, p.az, p.el, p.sa, p.se, p.det, p.ifov)
            tr.t_ns = p.t
        if tr.t_ns < t_ns:   # the line's last point is older than this frame (cannot happen today: kept for safety)
            tr.s, tr.S = self._predict_one(tr, t_ns)
            tr.t_ns = t_ns
        # Credited with at most fire.min_track_hits - _POST_BIRTH_HITS detections: the solver's bar is then only
        # reached by detections the filter itself gated after the birth, never on the frame a path is first
        # believed (2026-11-17 presses 105639 and 105308 fired on that frame: the path was 200 ms old already)
        tr.hits = min(len(pts), max(1, int(self.cfg.fire.min_track_hits) - _POST_BIRTH_HITS))
        tr.misses = 0
        tr.age = pts[-1].frame - p0.frame + 1
        tr.confirmed = True
        tr.first_t_ns = p0.t
        tr.last_detection = pts[-1].det
        self._sync_views(tr)
        self._hist[tr.track_id] = list(pts)
        self.tracks.append(tr)
        self.stats["promotions"] += 1
        gone = ch.pids()
        self.chains = [c for c in self.chains if c is not ch and not (c.pids() & gone)]
        self.seeds = [s for s in self.seeds if s.pid not in gone]

    def _make_room_motion(self) -> bool:
        """A slot for a newborn: the track with the most misses goes (ties: the youngest); a track seen this frame
        never does, so with every track seen the newborn waits (its chain may promote on a later frame)."""
        if len(self.tracks) < int(self.cfg.tracker.max_tracks):
            return True
        missed = [tr for tr in self.tracks if tr.misses > 0]
        if not missed:
            return False
        victim = max(missed, key=lambda tr: (tr.misses, tr.first_t_ns, tr.track_id))
        self.tracks.remove(victim)
        self.stats["killed_replaced"] += 1
        return True

    # ----------------------------------------------------------------- track upkeep
    def _push_hist(self, tid: int, p: _Pt, t_ns: int) -> None:
        h = self._hist.setdefault(tid, [])
        h.append(p)
        lim = t_ns - _HIST_NS
        if h[0].t < lim:
            k = 0
            while k < len(h) - 1 and h[k].t < lim:
                k += 1
            del h[:k]

    def _stab_line(self, tid: int, t_ns: int, bore):
        """Where the track's own recent detections say it is now in stab: (fb, contrast sign, x, y, gate px, whether
        its last detection's stab was measured on the picture) from a constant-acceleration fit to its last _LINE_N
        detections; None before it has two. The gate is the chains' (init_gate_px + init_gate_frac * step): the EKF
        gate alone, many pixels wide for a young track, let it take a speck whenever its clay dropped out."""
        h = self._hist.get(tid)
        if not h or len(h) < 2:
            return None
        last = h[-1]
        pts = [p for p in h[-_LINE_N:] if p.fb == last.fb and p.sx == p.sx]
        if len(pts) < 2:
            return None
        fit = _fit(pts)
        quad = _quad(pts, fit, self._amax, self._ridge)
        if len(pts) >= 5 and quad[4] > float(self._vt.init_max_rms_px):
            # one speck taken while the clay dropped out must not bend the path: drop the worst point once
            def res(p):
                x, y = _path(fit, quad, p.t)
                return math.hypot(p.sx - x, p.sy - y)
            worst = max(pts, key=res)
            pts = [p for p in pts if p is not worst]
            fit = _fit(pts)
            quad = _quad(pts, fit, self._amax, self._ridge)
        vx, vy = fit[4], fit[5]
        frame_ns = (pts[-1].t - pts[0].t) // (len(pts) - 1)
        px_, py_ = _path(fit, quad, t_ns, last.t + frame_ns)
        step = math.hypot(vx, vy) * (t_ns - last.t) * 1e-9
        g = float(self._vt.init_gate_px) + float(self._vt.init_gate_frac) * step
        # ... plus what the fit itself does not know: 3 sigma of its prediction there, from the scatter of its own
        # points (a clean clay path keeps the tight gate, a noisy or short one is not believed to the pixel)
        n = len(pts)
        if n >= 3:
            t0, tm = fit[0], fit[1]
            ds = [(p.t - t0) * 1e-9 - tm for p in pts]
            stt = sum(d * d for d in ds)
            d = (min(t_ns, last.t + frame_ns) - t0) * 1e-9 - tm
            beta, gamma = quad[2], quad[3]
            sqq = sum((x * x - beta * x - gamma) ** 2 for x in ds)
            var = 1.0 / n + (d * d / stt if stt > 0.0 else 0.0)
            if sqq > 0.0 and (quad[0] != 0.0 or quad[1] != 0.0):
                lam = sqq / (sqq + self._ridge)
                var += (lam * (d * d - beta * d - gamma)) ** 2 / sqq
            sig = max(quad[4] * 0.7071, _ACC_NOISE_PX)
            g += min(3.0 * sig * math.sqrt(var), 2.0 * float(self._vt.init_gate_px))
        signs = {p.sign for p in pts if p.sign}
        sign = signs.pop() if len(signs) == 1 else 0
        return last.fb, sign, px_, py_, g, bool(last.reg_ok)

    def _medians(self, tid: int) -> tuple[float, float]:
        """Median diameter and mass of the track's last _SIZE_MEDIAN_N detections (0 = unknown)."""
        h = self._hist.get(tid)
        if not h:
            return 0.0, 0.0
        recent = h[-_SIZE_MEDIAN_N:]
        return (_median([p.diam for p in recent if p.diam > 0.0]),
                _median([p.mass for p in recent if p.mass > 0.0]) if self._mass_ratio > 0.0 else 0.0)

    def _stopped(self, hist: list[_Pt] | None) -> bool:
        """True when the detections of the last stop_window_ms moved slower than init_min_speed_dps against the
        scenery (needs two thirds of the window covered and 3 points)."""
        if not hist or len(hist) < 3 or self._stop_ns <= 0:
            return False
        last = hist[-1]
        lim = last.t - self._stop_ns
        k = len(hist) - 1
        while k > 0 and hist[k - 1].t >= lim:
            k -= 1
        first = hist[k]
        dt = (last.t - first.t) * 1e-9
        if len(hist) - k < 3 or dt < 0.66 * self._stop_ns * 1e-9:
            return False
        if first.fb == last.fb and not first.fb:
            d = math.hypot(last.sx - first.sx, last.sy - first.sy)
        else:   # world directions: the fallback stab is their projection anyway
            d = self._fx * _angle_xyz(first.wx, first.wy, first.wz, last.wx, last.wy, last.wz)
        return d / dt < self._vmin

    def _dedupe(self, t_ns: int) -> None:
        if len(self.tracks) < 2:
            return
        order = sorted(self.tracks, key=lambda tr: (tr.first_t_ns, tr.track_id))
        keep: list[Track] = []
        info = []
        for tr in order:
            dx, dy, dz = self.predict_xyz(tr, t_ns)
            ce = math.cos(float(tr.x[3]))
            ra, re = float(tr.x[1]) * ce, float(tr.x[4])
            dup = False
            for (kx, ky, kz, ka, ke) in info:
                if _angle_xyz(dx, dy, dz, kx, ky, kz) <= self._dup_rad:
                    dv = math.hypot(ra - ka, re - ke)
                    if dv <= max(math.radians(5.0), 0.3 * math.hypot(ka, ke)):
                        dup = True
                        break
            if dup:
                self.stats["killed_duplicate"] += 1
                continue
            keep.append(tr)
            info.append((dx, dy, dz, ra, re))
        if len(keep) != len(self.tracks):
            ids = {tr.track_id for tr in keep}
            self.tracks = [tr for tr in self.tracks if tr.track_id in ids]

    # ----------------------------------------------------------------- stabilised coordinates
    def _maybe_recentre(self, bore) -> None:
        ref = self._ref
        if ref is not None and ref[0] * bore[0] + ref[1] * bore[1] + ref[2] * bore[2] >= _RECENTRE_COS:
            return
        rx, ry, rz = bore
        h = math.hypot(rx, ry)
        if h > 1e-6:
            e1 = (-ry / h, rx / h, 0.0)
        else:
            e1 = (0.0, 1.0, 0.0)
        e2 = (ry * e1[2] - rz * e1[1], rz * e1[0] - rx * e1[2], rx * e1[1] - ry * e1[0])
        self._ref = (rx, ry, rz) + e1 + e2
        if ref is not None:
            self.stats["recentre"] += 1
            for p in self.seeds:
                if p.fb:
                    self._project(p)
            for ch in self.chains:
                if ch.pts[0].fb:
                    for p in ch.pts:
                        self._project(p)
                    ch.refit(self._amax, self._ridge)
            for h in self._hist.values():
                for p in h:
                    if p.fb:
                        self._project(p)

    def _project(self, p: _Pt) -> bool:
        rx, ry, rz, ax, ay, az, bx, by, bz = self._ref
        den = p.wx * rx + p.wy * ry + p.wz * rz
        if den < _MIN_DEN:
            p.sx = p.sy = math.nan
            return False
        f = self._fx / den
        p.sx = f * (p.wx * ax + p.wy * ay + p.wz * az)
        p.sy = -f * (p.wx * bx + p.wy * by + p.wz * bz)
        return True

    def _directions(self, detections: list[Detection], rot: np.ndarray):
        """World unit directions of the detections in plain floats (the parent's _measure_few path without the pixel
        size, which costs four more rays a detection and is needed only for the few that meet a track or are born:
        _ensure_meas). None when the camera or mount has no plain-float form."""
        ray_of = getattr(self.cam, "ray_xyz", None)
        r_gc = getattr(self.mount, "R_gun_cam", None)
        if ray_of is None or r_gc is None or getattr(self.cam, "ifov1", None) is None:
            return None
        (g00, g01, g02), (g10, g11, g12), (g20, g21, g22) = np.asarray(r_gc, dtype=np.float64).tolist()
        (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = rot.tolist()
        out = []
        for d in detections:
            u, v = float(d.u), float(d.v)
            if not (math.isfinite(u) and math.isfinite(v)):
                out.append(None)
                continue
            ray = ray_of(u, v)
            if ray is None:
                return None
            x, y, z = ray
            gx = g00 * x + g01 * y + g02 * z   # camera -> gun (mount.cam_to_gun)
            gy = g10 * x + g11 * y + g12 * z
            gz = g20 * x + g21 * y + g22 * z
            wx = r00 * gx + r01 * gy + r02 * gz   # gun -> world
            wy = r10 * gx + r11 * gy + r12 * gz
            wz = r20 * gx + r21 * gy + r22 * gz
            n = math.sqrt(wx * wx + wy * wy + wz * wz)
            out.append((wx / n, wy / n, wz / n) if n > 0.0 else None)
        return out

    def _ensure_meas(self, p: _Pt) -> None:
        """The measurement sigmas and pixel size of a point made by the lazy path (as the parent's _measure_few)."""
        if p.ifov > 0.0:
            return
        d = p.det
        i = float(self.cam.ifov1(float(d.u), float(d.v)))
        sig = float(self.cfg.tracker.meas_sigma_px) * i
        p.ifov, p.se, p.sa = i, sig, sig / max(math.cos(p.el), _MIN_COS_EL)

    def _points(self, t_ns: int, detections: list[Detection], rot: np.ndarray, bore) -> list[_Pt]:
        if not detections:
            return []
        edge = self._edge_px
        dirs = self._directions(detections, rot)
        if dirs is None:
            az_l, el_l, sa_l, se_l, if_l = (a.tolist() for a in self._measure(detections, rot))
        out = []
        bx, by, bz = bore
        for k, d in enumerate(detections):
            if dirs is not None:
                w = dirs[k]
                if w is None:
                    continue
                wx, wy, wz = w
                az, el = math.atan2(wy, wx), math.asin(max(-1.0, min(1.0, wz)))
                sa = se = ifov = 0.0   # filled in by _ensure_meas when needed
            else:
                az, el, sa, se, ifov = az_l[k], el_l[k], sa_l[k], se_l[k], if_l[k]
                if not (math.isfinite(az) and math.isfinite(el) and math.isfinite(sa) and sa > 0.0):
                    continue
                ce = math.cos(el)
                wx, wy, wz = ce * math.cos(az), ce * math.sin(az), math.sin(el)
            p = _Pt()
            self._pid += 1
            p.pid, p.frame, p.t, p.det = self._pid, self._frame, t_ns, d
            p.az, p.el, p.sa, p.se, p.ifov = az, el, sa, se, ifov
            p.wx, p.wy, p.wz = wx, wy, wz
            p.bx, p.by, p.bz = bx, by, bz
            p.diam = float(d.diameter_px)
            # attached to the picture's border (its extent reaches within init_edge_px of it): what comes in at the
            # edge is scenery far more often than a target, and a target is clear of the edge two frames later
            r = 0.5 * p.diam + edge
            p.edge = edge > 0.0 and not (r <= d.u <= self._w - 1.0 - r and r <= d.v <= self._h - 1.0 - r)
            c = float(d.contrast)
            p.sign = 1 if c > 0.0 else (-1 if c < 0.0 else 0)
            p.mass = float(d.mass) if d.mass else 0.0
            p.reg_ok = bool(d.reg_ok)
            st = d.stab
            sdir = getattr(d, "stab_dir", None)
            p.sd = False
            if sdir is not None:
                # the detector's stabilised frame as a direction, laid out about a reference of our own that
                # follows the target: its `stab` plane is the picture of its first frame, and stretches without
                # bound as the gun turns 60-90 degrees away from where it pointed then
                p.fb = False
                p.sd = True
                p.kx, p.ky, p.kz = float(sdir[0]), float(sdir[1]), float(sdir[2])
            elif st is not None:
                p.fb = False
                p.sx, p.sy = float(st[0]), float(st[1])
            else:
                p.fb = True
                self._project(p)
            out.append(p)
        self._lay_out_stabilised(out)
        return out

    def _lay_out_stabilised(self, pts: list[_Pt]) -> None:
        """sx, sy of the points that carry a stabilised direction: its gnomonic projection in focal-length pixels
        about self._sref, which is moved onto the strongest of them when that one is more than 10 degrees from it
        (the points of live seeds, chains and tracks are then laid out again: their distances change by the
        projection's stretch, a few percent)."""
        sd = [p for p in pts if p.sd]
        if not sd:
            return
        lead = max(sd, key=lambda p: (p.mass, -p.pid))
        ref = self._sref
        if ref is None or ref[0] * lead.kx + ref[1] * lead.ky + ref[2] * lead.kz < _RECENTRE_COS:
            rx, ry, rz = lead.kx, lead.ky, lead.kz
            # e1 across the reference, in the plane that holds least of it (never degenerate)
            ax_, ay_, az_ = (0.0, -rz, ry) if abs(rx) <= min(abs(ry), abs(rz)) else                 ((rz, 0.0, -rx) if abs(ry) <= abs(rz) else (-ry, rx, 0.0))
            n = math.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_)
            e1 = (ax_ / n, ay_ / n, az_ / n)
            e2 = (ry * e1[2] - rz * e1[1], rz * e1[0] - rx * e1[2], rx * e1[1] - ry * e1[0])
            self._sref = (rx, ry, rz) + e1 + e2
            if ref is not None:
                self.stats["recentre"] += 1
                for q in self.seeds:
                    if q.sd:
                        self._project_s(q)
                for ch in self.chains:
                    if ch.pts[0].sd:
                        for q in ch.pts:
                            self._project_s(q)
                        ch.refit(self._amax, self._ridge)
                for h in self._hist.values():
                    for q in h:
                        if q.sd:
                            self._project_s(q)
        for p in sd:
            self._project_s(p)

    def _project_s(self, p: _Pt) -> bool:
        rx, ry, rz, ax, ay, az, bx, by, bz = self._sref
        den = p.kx * rx + p.ky * ry + p.kz * rz
        if den < _MIN_DEN:
            p.sx = p.sy = math.nan
            return False
        f = self._fx / den
        p.sx = f * (p.kx * ax + p.ky * ay + p.kz * az)
        p.sy = f * (p.kx * bx + p.ky * by + p.kz * bz)
        return True

    # ----------------------------------------------------------------- track-before-confirm
    def _gun_px(self, a: _Pt, bx, by, bz) -> float:
        return self._fx * _angle_xyz(a.bx, a.by, a.bz, bx, by, bz)

    def _chains_step(self, t_ns: int, free: list[_Pt], bore) -> list[_Chain]:
        vt = self._vt
        frame = self._frame
        gate0, gfrac, gyro = float(vt.init_gate_px), float(vt.init_gate_frac), float(vt.init_gyro_frac)
        bx, by, bz = bore
        # (unprojectable points and those attached to the picture's border start and extend no path; a live track
        # may still take one)
        free = [p for p in free if p.sx == p.sx and not p.edge]
        extended: set[tuple[int, int]] = set()
        ready: list[_Chain] = []
        keep: list[_Chain] = []
        win = self._win_ns
        max_rms = float(vt.init_max_rms_px)
        amax, ridge = self._amax, self._ridge
        ratio = self._ratio
        mratio = self._chain_mass_ratio
        same_sign = self._same_sign

        def ch_ok(ch: _Chain) -> bool:
            for q in ch.pts:
                if not q.reg_ok:
                    return False
            return True

        # this frame's points on a coarse grid: a gate or a speed ring only looks at the cells it overlaps
        grid: dict[tuple[int, int], list[_Pt]] = {}
        for p in free:
            grid.setdefault((int(p.sx // _CELL), int(p.sy // _CELL)), []).append(p)
        n_cells = len(grid)

        def near(x: float, y: float, r: float) -> list[_Pt]:
            x0, x1, y0, y1 = int((x - r) // _CELL), int((x + r) // _CELL), int((y - r) // _CELL), int((y + r) // _CELL)
            if (x1 - x0 + 1) * (y1 - y0 + 1) >= n_cells:
                return free
            out: list[_Pt] = []
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    c = grid.get((ix, iy))
                    if c:
                        out.extend(c)
            return out

        all_ok = all(p.reg_ok for p in free)
        # chains of 2+ points: the best-fitting detection extends each
        for ch in self.chains:
            if t_ns - ch.pts[0].t > win:
                self.stats["rej_window"] += 1
                continue
            last = ch.pts[-1]
            vx, vy = ch.fit[4], ch.fit[5]
            qx, qy = _path(ch.fit, ch.quad, t_ns)
            step = math.hypot(vx, vy) * (t_ns - last.t) * 1e-9
            base = gate0 + gfrac * step
            best, best_d2 = None, 0.0
            lfb, lsign, ld, lok, lm = last.fb, last.sign, last.diam, last.reg_ok, last.mass
            g2 = base * base
            g_gyro = base + gyro * self._gun_px(last, bx, by, bz) if gyro > 0.0 and not (all_ok and lok) else base
            g2_gyro = g_gyro * g_gyro
            for p in near(qx, qy, g_gyro) if free else free:
                ex, ey = p.sx - qx, p.sy - qy
                d2 = ex * ex + ey * ey
                if d2 > (g2 if (p.reg_ok and lok) else g2_gyro) or (best is not None and d2 >= best_d2):
                    continue
                if p.fb != lfb or (same_sign and lsign and p.sign and p.sign != lsign):
                    continue
                pd = p.diam
                if ld > 0.0 and pd > 0.0 and (ld > ratio * pd or pd > ratio * ld):
                    continue
                pm = p.mass
                if mratio > 0.0 and lm > 0.0 and pm > 0.0 and (lm > mratio * pm or pm > mratio * lm):
                    continue
                best, best_d2 = p, d2
            fit = sums = quad = None
            if best is not None:
                sums = ch.with_point(best)
                fit = _fit_sums(ch.pts[0], sums)
                quad = _quad(ch.pts + [best], fit, amax, ridge)
                fit = fit[:6] + (quad[4],)   # the scatter that counts is about the constant-acceleration path
                if fit[6] > max_rms + (0.0 if (lok and best.reg_ok and ch_ok(ch)) else
                                       self._rms_allow_pts(ch.pts + [best])):
                    # off the path: a speck near where the clay should be (it dropped out) is a miss, not the end
                    self.stats["rej_rms"] += 1
                    best = None
            if best is None:
                ch.misses += 1
                if len(ch.pts) < 3 or ch.misses > 1:
                    self.stats["rej_miss"] += 1
                    continue
                keep.append(ch)
                continue
            ch.add(best, sums, fit, quad)
            extended.add((last.pid, best.pid))
            keep.append(ch)
            if len(ch.pts) >= self._init_hits:
                ready.append(ch)

        # seeds pair with this frame's detections
        vmin, vmax = self._vmin, self._vmax
        nb = max(1, int(vt.init_max_branches))
        pairs = []
        ftup = [(p.sx, p.sy, p.fb, p.sign, p.diam, p) for p in free]
        # The upper speed limit is on what could be shot at: a target crossing the picture faster than vmax is gone
        # before the solver has it. One the gun FOLLOWS is slow in the picture however fast both turn against the
        # world (a ball thrown up from beside the shooter on 2026-11-19 15:50: 130-230 deg/s against the scenery,
        # a few pixels a frame in the picture, detected in 36 frames running and never paired): past vmax against
        # the scenery a pair still counts while its step in the PICTURE is inside vmax.

        for s in self.seeds if free else ():
            dt = (t_ns - s.t) * 1e-9
            if dt <= 0.0:
                continue
            lo, hi = vmin * dt, vmax * dt
            lo2, hi2 = lo * lo, hi * hi
            sx, sy, sfb, ssign, sd, spid, sm = s.sx, s.sy, s.fb, s.sign, s.diam, s.pid, s.mass
            cand = []
            for px_, py_, pfb, psign, pd, p in ftup:
                ex, ey = px_ - sx, py_ - sy
                d2 = ex * ex + ey * ey
                if d2 < lo2:
                    continue
                if d2 > hi2:
                    iu, iv = p.det.u - s.det.u, p.det.v - s.det.v
                    if iu * iu + iv * iv > hi2:
                        continue
                if pfb != sfb or (same_sign and ssign and psign and psign != ssign):
                    continue
                pm = p.mass
                if mratio > 0.0 and sm > 0.0 and pm > 0.0 and (sm > mratio * pm or pm > mratio * sm):
                    continue
                if sd > 0.0 and pd > 0.0:
                    if sd > ratio * pd or pd > ratio * sd:
                        continue
                    sim = sd / pd if sd < pd else pd / sd
                else:
                    sim = 1.0
                if extended and (spid, p.pid) in extended:
                    continue
                cand.append((-(p.mass + 1.0) * sim * sim, p.pid, p, sim))
            if not cand:
                continue
            if len(cand) > nb:
                cand.sort(key=lambda c: (c[0], c[1]))
            for _, ppid, p, sim in cand[:nb]:
                pairs.append((-(s.mass + p.mass + 1.0) * sim * sim, spid, ppid, s, p))
        # only as many new lines as the cap leaves room for (the strongest pairs): most pairs in clutter are chance
        room = max(1, int(vt.max_chains)) - len(keep)
        if len(pairs) > room:
            pairs.sort(key=lambda c: (c[0], c[1], c[2]))
            self.stats["rej_cap"] += len(pairs) - max(room, 0)
            pairs = pairs[:max(room, 0)]
        for _, _, _, s, p in pairs:
            keep.append(_Chain(s, p))

        # seeds: this frame's detections; those of the last _SEED_FRAMES frames stay
        self.seeds = [s for s in self.seeds if frame - s.frame < _SEED_FRAMES] + free

        # promotion candidates: the chains that meet every rule, the strongest first
        born: list[_Chain] = []
        if ready:
            min_travel = float(vt.init_min_travel_px)
            ok = []
            for ch in ready:
                a, b = ch.pts[0], ch.pts[-1]
                if b.frame - a.frame < 3:
                    continue
                if self._weak_hits and len(ch.pts) < self._init_hits + self._weak_hits and                         _median([q.mass for q in ch.pts]) < self._weak_mass:
                    continue   # speck-sized: more detections before it is believed
                if math.hypot(b.sx - a.sx, b.sy - a.sy) < min_travel:
                    continue
                spd = math.hypot(ch.fit[4], ch.fit[5])
                # Where the path rests on the gyro (open sky: no texture to register on), "against the scenery"
                # is against the gyro, and a gyro bias nobody has corrected yet turns everything that stands still
                # at that bias: the yaw gyro read -2.5 deg/s at rest in the recordings of 2026-11-17 to 20, and
                # press 20261119-092521 got a track on a tree top at the picture's corner and one on a far speck
                # that was shot at. Such a path has to be faster than any bias seen.
                on_gyro = 2 * sum(1 for q in ch.pts if q.reg_ok) < len(ch.pts)
                if spd < (self._vmin_gyro if on_gyro else vmin):
                    continue
                if spd > vmax:   # ... unless the gun follows it (see the pairing rule)
                    dt_ab = (b.t - a.t) * 1e-9
                    if dt_ab <= 0.0 or math.hypot(b.det.u - a.det.u, b.det.v - a.det.v) > vmax * dt_ab:
                        continue
                ok.append(ch)
            ok.sort(key=lambda c: (-len(c.pts), c.fit[6], -c.mass, c.pts[0].pid))
            taken: set[int] = set()
            for ch in ok:
                ids = ch.pids()
                if ids & taken:
                    continue
                taken |= ids
                born.append(ch)

        # cap: the longest and strongest lines stay
        cap = max(1, int(vt.max_chains))
        if len(keep) > cap:
            keep.sort(key=lambda c: (-len(c.pts), -c.mass, c.pts[0].pid))
            # a chain about to be born is never dropped by the cap
            bset = {id(c) for c in born}
            head = keep[:cap]
            tail = [c for c in keep[cap:] if id(c) in bset]
            self.stats["rej_cap"] += len(keep) - cap - len(tail)
            keep = head + tail
        self.chains = keep
        return born

    def _rms_allow_pts(self, pts: list[_Pt]) -> float:
        """Extra RMS allowed where stab rests on the gyro alone: init_gyro_frac of the gun's mean image motion per
        step over the chain."""
        if all(p.reg_ok for p in pts):
            return 0.0
        a, b = pts[0], pts[-1]
        n = max(1, len(pts) - 1)
        return float(self._vt.init_gyro_frac) * self._gun_px(a, b.bx, b.by, b.bz) / n
