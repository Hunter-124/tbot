"""Fire solver: when does pulling the trigger now put the pattern on the clay?

Contract: docs/ARCHITECTURE.md [tracker]. Everything is in the WORLD frame with the camera's optical
centre at the origin (the gun only rotates about it). Safety: this module only ever returns a decision;
the arming interlock is enforced again in the actuator layer.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config, LatencyBudget
from ..optics import MountModel
from ..types import FireDecision, Track
from .ballistics import G_MPS2, ShotModel
from .policy import FirePolicy

_UP_W = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_TOF_ITERS = 3  # fixed-point passes on time of flight (converges in 2 at skeet ranges)
_DTOF_STEP_M = 0.5  # finite-difference step for d(ToF)/d(range)
# A closest approach this many sigma outside the pattern is not worth a full solution: P(hit) there is below
# 4e-5 whatever the threshold (the sigma is scaled up for the later exit time before it is compared).
_SUBFRAME_NSIG = 4.0

# Polar quadrature of the aim-error Gaussian over the pattern disc (no scipy, no Bessel overflow).
_QUAD_R_X, _QUAD_R_W = np.polynomial.legendre.leggauss(16)
_QUAD_T_X, _QUAD_T_W = np.polynomial.legendre.leggauss(32)
_QUAD_T_UNIT = 0.5 * (_QUAD_T_X + 1.0)
_QUAD_T_HALF_W = 0.5 * _QUAD_T_W
_QUAD_NSIG = 6.0  # integrate only where the Gaussian has mass; beyond 6 sigma it is < 1e-8


_SERIES_MAX = 40.0  # the series below is used while miss^2 / 2 sigma^2 and pattern^2 / 2 sigma^2 are under this


def _hit_probability(miss: float, pattern: float, sigma: float) -> float:
    """P(|e + miss| < pattern) for an isotropic 2-D Gaussian aim error e ~ N(0, sigma^2 I)."""
    if not (pattern > 0.0) or not math.isfinite(miss):
        return 0.0
    if not (sigma > 0.0):
        return 1.0 if miss < pattern else 0.0
    # |e + miss|^2 / sigma^2 is noncentral chi-square with 2 degrees of freedom, whose distribution function is
    # a Poisson mixture of integer-order gamma functions: a few dozen float operations instead of a
    # 16 x 32-point quadrature (0.9 ms of small NumPy calls on a Pi 3B, three or four times on a firing frame).
    inv = 1.0 / (2.0 * sigma * sigma)
    a = miss * miss * inv
    b = pattern * pattern * inv
    if a < _SERIES_MAX and b < _SERIES_MAX:
        pois = math.exp(-a)          # Poisson(k; a)
        term = math.exp(-b)          # e^-b b^k / k!
        tail = 1.0 - term            # P(k + 1, b) = 1 - e^-b sum_{j<=k} b^j / j!
        total = pois * tail
        k = 0
        k_peak = a if a > b else b
        while True:
            k += 1
            pois *= a / k
            term *= b / k
            tail -= term
            if tail <= 0.0:
                break
            add = pois * tail
            total += add
            if k > k_peak and add < 1e-12:
                break
        return min(max(total, 0.0), 1.0)
    return _hit_probability_quad(miss, pattern, sigma)


def _hit_probability_quad(miss: float, pattern: float, sigma: float) -> float:
    """The same probability by polar quadrature: any miss / pattern / sigma, and the reference for the series."""
    if not (pattern > 0.0) or not math.isfinite(miss):
        return 0.0
    if not (sigma > 0.0):
        return 1.0 if miss < pattern else 0.0
    # Only the annulus within _QUAD_NSIG sigma of the mean contributes.
    lo = max(0.0, miss - _QUAD_NSIG * sigma)
    hi = min(pattern, miss + _QUAD_NSIG * sigma)
    if hi <= lo:
        return 1.0 if miss + _QUAD_NSIG * sigma < pattern else 0.0
    r = 0.5 * (hi - lo) * _QUAD_R_X + 0.5 * (hi + lo)  # (nr,)
    wr = 0.5 * (hi - lo) * _QUAD_R_W
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    if miss <= 0.0:
        ang = 2.0 * math.pi * np.exp(-r * r * inv2s2)
    else:
        # Angular window where |p - m| <= _QUAD_NSIG sigma, i.e. cos(theta) >= c.
        c = (r * r + miss * miss - (_QUAD_NSIG * sigma) ** 2) / (2.0 * r * miss)
        th_max = np.arccos(np.clip(c, -1.0, 1.0))
        th = th_max[:, None] * _QUAD_T_UNIT[None, :]
        wt = th_max[:, None] * _QUAD_T_HALF_W[None, :]
        d2 = r[:, None] ** 2 + miss * miss - 2.0 * miss * r[:, None] * np.cos(th)
        ang = 2.0 * np.sum(wt * np.exp(-d2 * inv2s2), axis=1)  # symmetric in +/- theta
    p = float(np.sum(wr * r * ang)) * inv2s2 / math.pi
    return min(max(p, 0.0), 1.0)


def _rotvec(v: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation matrix of the rotation vector v (axis * angle, radians)."""
    x, y, z = map(float, v)
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = x / theta, y / theta, z / theta
    s, c = math.sin(theta), 1.0 - math.cos(theta)
    return np.array(
        [[1.0 - c * (y * y + z * z), c * x * y - s * z, c * x * z + s * y],
         [c * y * x + s * z, 1.0 - c * (x * x + z * z), c * y * z - s * x],
         [c * z * x - s * y, c * z * y + s * x, 1.0 - c * (x * x + y * y)]],
        dtype=np.float64,
    )


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, az = map(float, a)
    bx, by, bz = map(float, b)
    x, y, z = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    return math.atan2(math.sqrt(x * x + y * y + z * z), ax * bx + ay * by + az * bz)


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm3(v)
    return v / n if n > 0.0 else v


def _norm3(v: np.ndarray) -> float:
    """np.linalg.norm of a 1-D vector (NumPy computes it as sqrt(v . v) too) without its ~65 us overhead."""
    return math.sqrt(float(v @ v))

def _tof(shot, distance: float) -> float:
    # np.interp already clamps at the table ends, so the scalar np.clip in
    # ShotModel._lookup is unnecessary here. Preserve custom shot models.
    if type(shot) is ShotModel:
        return float(np.interp(distance, shot._x, shot._t))
    return float(shot.time_of_flight(distance))



class FireSolver:
    """Decides whether firing *now* is the best moment to break the clay."""

    @staticmethod
    def worst_case_work(cfg: Config) -> float:
        """The most firing solutions one frame can need (the frame that fires needs them all): now, the aim
        one period on, the instant of closest approach, and every frame of the look-ahead."""
        return 2.6 + max(0, int(cfg.fire.horizon_frames))

    def __init__(self, cfg: Config, shot: ShotModel, mount: MountModel, latency: LatencyBudget):
        self.cfg = cfg
        self.shot = shot
        self.mount = mount
        self.latency = latency
        self._atype = str(cfg.actuator.type)
        if self._atype not in latency.cmd_to_break_ms:
            raise ValueError(
                f"no latency budget for actuator type {self._atype!r} "
                f"(known: {sorted(latency.cmd_to_break_ms)})"
            )
        self._muzzle_g = np.asarray(mount.muzzle_pos_g, dtype=np.float64).reshape(3) - np.asarray(
            mount.cam_pos_g, dtype=np.float64
        ).reshape(3)
        self._bore_g = _unit(np.asarray(mount.bore_dir_g, dtype=np.float64).reshape(3))
        self.policy = FirePolicy.from_config(cfg)
        self._aim_prior = math.radians(float(getattr(cfg.fire, "aim_prior_deg", 0.0) or 0.0))
        self._evidence = float(getattr(cfg.fire, "aim_prior_evidence", 0.0) or 0.0)

    def evaluate(
        self,
        now_ns: int,
        tracker,
        attitude,
        armed: bool,
        frame_period_s: float,
        accept=None,
    ) -> FireDecision:
        """accept(track) -> bool may veto targets (the engagement lockout of a multi-shot press)."""
        now_ns = int(now_ns)
        track = tracker.best(accept) if accept is not None else tracker.best()
        if track is None:
            reason = "no confirmed track" if armed else "no confirmed track (not armed)"
            return FireDecision(fire=False, t_ns=now_ns, reason=reason)

        rot_now = np.asarray(attitude.R_wg(now_ns), dtype=np.float64).reshape(3, 3)
        omega = np.asarray(attitude.omega_g(now_ns), dtype=np.float64).reshape(3)
        alpha_fn = getattr(attitude, "alpha_g", None)
        alpha = (np.asarray(alpha_fn(now_ns), dtype=np.float64).reshape(3) if callable(alpha_fn)
                 else np.zeros(3, dtype=np.float64))
        delay_s = self.latency.fire_delay_s(self._atype)
        jitter_s = self.latency.fire_jitter_s(self._atype)
        fc = self.cfg.fire
        # How much the swing rate has changed lately: a hand that is changing its mind will change it again
        # (the uncertainty model in _geometry).
        window_ns = int(float(getattr(fc, "gun_rate_change_window_ms", 100.0)) * 1e6)
        dw = _norm3(omega - np.asarray(attitude.omega_g(now_ns - window_ns), dtype=np.float64).reshape(3)) \
            if window_ns > 0 else 0.0
        g = self._geometry(now_ns, now_ns, track, tracker, rot_now, omega, delay_s, jitter_s, alpha, dw)

        fire = False
        fire_ns = now_ns
        bar = ceiling = float("nan")
        work = 1.0  # firing solutions worked out on this frame (an aim-only one counts 0.6): see WORST_CASE_WORK
        future: list[float] = []
        speed = _norm3(track.s[3:6]) if track.s is not None else float("inf")
        # Only a track that is SURELY slow is not a clay. A young track's speed is mostly its prior (zero mean),
        # so its estimate starts near nothing and climbs as the motion is seen: on the range day a clay was held
        # "too slow" for 20 frames at P(hit) up to 0.51. Its own 1-sigma along the velocity says how far to trust it.
        speed_hi = speed
        k_sig = float(getattr(fc, "min_clay_speed_sigmas", 0.0) or 0.0)
        if k_sig > 0.0 and track.s is not None and getattr(track, "S", None) is not None and speed < float("inf"):
            v = track.s[3:6]
            u = v / speed if speed > 1e-6 else np.array([1.0, 0.0, 0.0])
            Svv = np.asarray(track.S, dtype=np.float64)[3:6, 3:6]
            speed_hi = speed + k_sig * math.sqrt(max(float(u @ Svv @ u), 0.0))
        if track.hits < fc.min_track_hits:
            reason = "track young"
        elif speed_hi < float(fc.min_clay_speed_mps):
            reason = f"too slow for a clay ({speed:.1f} m/s)"
        elif not (fc.min_range_m <= g["range_m"] <= fc.max_range_m):
            reason = "range out of window"
        elif g["clay_el"] < math.radians(fc.min_elevation_deg):
            reason = "below min elevation"
        else:
            period_ns = int(round(float(frame_period_s) * 1e9))
            horizon = int(fc.horizon_frames) if period_ns > 0 else 0
            if horizon >= 1 and bool(getattr(fc, "subframe", True)):
                # The command need not go out on a frame. Bore and clay close on each other almost
                # linearly over one frame period, so the instant of closest approach follows from where
                # shot and clay will be for a command now and for one a frame later; a full solution
                # at that instant says what it is worth. A fast crossing passes through the pattern in
                # 20-40 ms: on frames alone a 30 fps loop (trees in view) steps right over it.
                # (This runs on every frame with a target, so the look ahead is the aim only - no
                # uncertainty, no probability - and the full solution is only worked out when the closest
                # approach can come near the pattern: 7.4 -> 3.6 ms a frame on a Pi 3B.)
                a1 = self._geometry(now_ns, now_ns + period_ns, track, tracker, rot_now, omega, delay_s, jitter_s,
                                    alpha, dw, aim_only=True)
                work += 0.6
                e0 = g["shot_dir_w"] - g["lead_dir_w"]
                de = (a1["shot_dir_w"] - a1["lead_dir_w"]) - e0
                dd = float(de @ de)
                if dd > 0.0:
                    frac = -float(e0 @ de) / dd
                    if 0.02 < frac < 0.98:
                        closest = _norm3(e0 + frac * de)
                        # the uncertainty grows with the time to exit (at most with its square)
                        grow = ((delay_s + frac * period_ns * 1e-9) / delay_s) ** 2 if delay_s > 0.0 else 1.0
                        # (with the aim prior the miss counts for k times itself against sqrt(k) sigma)
                        sg = grow * g["sigma"]
                        pr = self._aim_prior
                        k = pr * pr / (pr * pr + sg * sg) if pr > 0.0 and sg > 0.0 else 1.0
                        if k * closest < g["pattern"] + _SUBFRAME_NSIG * math.sqrt(k) * sg:
                            at_ns = now_ns + int(round(frac * period_ns))
                            gs = self._geometry(now_ns, at_ns, track, tracker, rot_now, omega, delay_s, jitter_s,
                                                alpha, dw)
                            work += 1.0
                            if gs["p_hit"] > g["p_hit"]:
                                g, fire_ns = gs, at_ns
            # The bar this top of P(hit) has to clear (fire/policy.py): fire.min_hit_probability, lowered to
            # what this swing can give, to the chances this target has already let go, and with the time it
            # has been followed. One shot a press: waiting for 0.4 on a clay that only ever offers 0.3 is a
            # clay lost.
            ceiling = self._p_hit(0.0, g["pattern"], g["sigma"])  # what a perfect aim would be worth right now
            bar = self.policy.bar(track.track_id, now_ns, int(track.first_t_ns or 0), g["p_hit"], ceiling)
            if g["p_hit"] < bar:
                reason = f"p_hit {g['p_hit']:.2f} < {bar:.2f}"
            else:
                better_k = 0
                for k in range(1, horizon + 1):
                    gk = self._geometry(
                        now_ns, now_ns + k * period_ns, track, tracker, rot_now, omega, delay_s, jitter_s, alpha, dw,
                        remeasured=True,
                    )
                    future.append(gk["p_hit"])
                    work += 1.0
                    if better_k == 0 and gk["p_hit"] > g["p_hit"]:
                        better_k = k
                if better_k:
                    reason = f"better in {better_k} frames"
                elif not armed:
                    reason = "would fire (not armed)"
                else:
                    fire = True
                    reason = "FIRE"
        if not armed and reason != "would fire (not armed)":
            # The interlock is visible on every disarmed frame; a dry run that would fire still reads
            # "would fire (not armed)".
            reason = f"{reason} (not armed)"

        return FireDecision(
            fire=fire,
            t_ns=now_ns,
            reason=reason,
            track_id=track.track_id,
            t_exit_ns=g["t_exit_ns"],
            miss_rad=g["miss"],
            pattern_rad=g["pattern"],
            sigma_rad=g["sigma"],
            p_hit=g["p_hit"],
            range_m=g["range_m"],
            tof_s=g["tof"],
            lead_dir_w=g["lead_dir_w"],
            bore_dir_w=g["bore_dir_w"],
            extra={
                "armed": bool(armed),
                "actuator": self._atype,
                "fire_delay_s": delay_s,
                "dist_m": g["dist"],
                "clay_el_rad": g["clay_el"],
                "track_sigma_rad": g["track_sigma"],
                "gun_sigma_rad": g["gun_sigma"],
                "gun_alpha_rad_s2": _norm3(alpha),
                "gun_rate_change_rad_s": float(dw),
                # when to send the command: now, or a moment later inside this frame period (fire.subframe)
                "fire_at_ns": int(fire_ns),
                "bar": bar,
                "ceiling": ceiling,
                "work": work,
                "clay_rate_rad_s": g["omega_clay"],
                "future_p_hit": future,
            },
        )

    # ----------------------------------------------------------------- internals
    def _p_hit(self, miss: float, pattern: float, sigma: float) -> float:
        """P(hit) for a predicted miss known to `sigma`.

        The predicted miss is the true one plus an error of `sigma`, and the true one is not just anything: a
        solution exists because a shooter is swinging through the target (fire.aim_prior_deg, 1 sigma per axis
        of where the charge really goes relative to the clay over such frames). With that prior the best
        estimate of the true miss is the predicted one shrunk by k = prior^2 / (prior^2 + sigma^2), known to
        sqrt(k) sigma. Without it (0) every frame near the top of a pass is called far less likely than it is:
        virtual clays through the range-day swings (`tbot vclay --calibration`), frames given 0.1-0.2 hit 44 %
        and 0.3-0.4 hit 65 %. The 54 real frames of that day with a solution agree (Brier 0.134 -> 0.115 with
        the prior; of the 28 frames it then puts under 0.2 none would have hit).

        A prior is not a measurement: with nothing known (a track three frames old, sigma 18 deg) it alone
        would call any frame 0.25-0.36. So it only counts as far as the measurement by itself supports a
        shot: below fire.aim_prior_evidence of plain P(hit) the result is scaled down in proportion. As
        shipped (1.0 deg, 0.2), 58 000 virtual-clay frames: given 0.2-0.3 hit 46 %, 0.4-0.5 62 %, 0.6-0.8 81 %."""
        p_raw = _hit_probability(miss, pattern, sigma)
        prior = self._aim_prior
        if not (prior > 0.0 and sigma > 0.0):
            return p_raw
        k = prior * prior / (prior * prior + sigma * sigma)
        p = _hit_probability(k * miss, pattern, math.sqrt(k) * sigma)
        if p_raw < self._evidence:
            p *= p_raw / self._evidence
        return p

    def _geometry(
        self,
        now_ns: int,
        decide_ns: int,
        track: Track,
        tracker,
        rot_now: np.ndarray,
        omega: np.ndarray,
        delay_s: float,
        jitter_s: float,
        alpha: np.ndarray | None = None,
        rate_change: float = 0.0,
        aim_only: bool = False,
        remeasured: bool = False,
    ) -> dict:
        """Miss angle, uncertainty and hit probability for a trigger command issued at decide_ns.
        aim_only: just where the shot and the clay will be ("shot_dir_w", "lead_dir_w").
        remeasured: decide_ns is a later frame, which will have its own fresh look at the gun: the aim is
        extrapolated from now, but the swing uncertainty is that of a decision made then. Otherwise waiting
        always looks worse than it will be (the uncertainty grows with the square of the time ahead) and the
        shot goes early."""
        t_exit_ns = int(decide_ns) + int(round(delay_s * 1e9))
        dt_exit = (t_exit_ns - now_ns) * 1e-9
        if alpha is None:
            alpha = np.zeros(3, dtype=np.float64)
        # Where the bore will be at exit. A hand does not keep its acceleration: over the 164 ms of the measured
        # trigger delay a constant-acceleration extrapolation is WORSE than no acceleration term at all. Range day
        # 2026-11-17, real swings on real clays, the 135 frames of 9 presses that had a target (10-30 deg/s),
        # bore error p50 / p90 / rms: constant acceleration 1.10 / 2.98 / 2.19 deg, constant rate 0.69 / 1.64 /
        # 1.03, against a 0.75 deg pattern. So the acceleration decays with fire.gun_accel_decay_s (0.02 s):
        #     turned by  omega * dt  +  alpha * tau^2 (dt/tau - 1 + e^(-dt/tau))
        # 0.70 / 1.60 / 1.00 deg at 164 ms, and still the better one at 104 ms (0.41 / 1.23 / 0.84 -> 0.31 /
        # 0.78 / 0.48). 0 = the acceleration is kept (alpha * dt^2 / 2). A decay of the rate itself fits the
        # whole press better but that is the gun being mounted and lowered (100-190 deg/s); on a target it
        # is within the noise, and a sustained fast swing is not in the data to say otherwise.
        fc = self.cfg.fire
        t_acc = float(getattr(fc, "gun_accel_decay_s", 0.0) or 0.0)
        half_dt2 = 0.5 * dt_exit * dt_exit
        k_acc = half_dt2
        if t_acc > 0.0 and dt_exit > 0.0:
            k_acc = t_acc * t_acc * (dt_exit / t_acc - 1.0 + math.exp(-dt_exit / t_acc))
        rot_exit = rot_now @ _rotvec(omega * dt_exit + alpha * k_acc)
        # How far off that is, 1 sigma per axis. What the hand does next is an acceleration nobody measured, so
        # the error grows with the SQUARE of the time ahead (p50 0.31 deg at 104 ms, 0.70 at 164, 0.85 at 190):
        #   fire.gun_accel_floor_dps2 x dt^2/2            the steadiest hand still does this much
        #   fire.gun_rate_drift_per_s x |omega| x dt^2/2  a faster swing changes faster
        #   fire.gun_accel_sigma_frac x the acceleration term above
        #   fire.gun_rate_change_sigma_frac x the mean acceleration of the last gun_rate_change_window_ms x dt^2/2
        #   fire.gun_rate_sigma_dps x dt                  a constant rate error (0: the gyro re-zeroes at rest)
        # Fitted to the same frames; error / sigma (a perfect model reads p50 1.18, p90 2.15, p99 3.03):
        # 104 ms 1.10 / 2.20 / 3.18, 164 ms 1.11 / 2.29 / 3.27, 190 ms 1.04 / 2.26 / 3.19.
        window_s = float(getattr(fc, "gun_rate_change_window_ms", 100.0)) * 1e-3
        half_s2, k_acc_s = half_dt2, k_acc
        if remeasured:
            # a later frame will look again: its uncertainty is one trigger delay's worth, not delay + wait
            dt_sigma = (t_exit_ns - int(decide_ns)) * 1e-9
            half_s2 = 0.5 * dt_sigma * dt_sigma
            k_acc_s = (t_acc * t_acc * (dt_sigma / t_acc - 1.0 + math.exp(-dt_sigma / t_acc))
                       if t_acc > 0.0 and dt_sigma > 0.0 else half_s2)
        else:
            dt_sigma = dt_exit
        gun_sigma = math.sqrt(
            (math.radians(float(getattr(fc, "gun_rate_sigma_dps", 0.0))) * dt_sigma) ** 2
            + (math.radians(float(getattr(fc, "gun_accel_floor_dps2", 0.0))) * half_s2) ** 2
            + (float(getattr(fc, "gun_rate_drift_per_s", 0.0)) * _norm3(omega) * half_s2) ** 2
            + (float(fc.gun_accel_sigma_frac) * _norm3(alpha) * k_acc_s) ** 2
            + (float(getattr(fc, "gun_rate_change_sigma_frac", 0.0)) * float(rate_change)
               / max(window_s, 1e-3) * half_s2) ** 2
        )
        bore_w = _unit(rot_exit @ self._bore_g)
        muzzle_w = rot_exit @ self._muzzle_g

        shot = self.shot
        tabulated = type(shot) is ShotModel
        rng = float(track.range_m)
        t_arr_ns = t_exit_ns
        p_clay = rng * tracker.predict_dir(track, t_arr_ns)
        dist = _norm3(p_clay - muzzle_w)
        tof = _tof(shot, dist)
        for _ in range(_TOF_ITERS):
            t_arr_ns = t_exit_ns + int(round(tof * 1e9))
            p_clay = rng * tracker.predict_dir(track, t_arr_ns)
            dist = _norm3(p_clay - muzzle_w)
            tof = _tof(shot, dist)

        to_clay = p_clay - muzzle_w
        lead_dir = _unit(to_clay)
        elevation = math.asin(max(-1.0, min(1.0, float(bore_w[2]))))
        drop = (float(0.5 * G_MPS2 * tof * tof * np.cos(elevation)) if tabulated
                else float(shot.drop(dist, elevation)))
        to_shot = bore_w * dist - drop * _UP_W
        if aim_only:
            return {"lead_dir_w": lead_dir, "shot_dir_w": _unit(to_shot)}
        miss = _angle_between(to_clay, to_shot)

        _, _, track_sigma = tracker.predict(track, t_arr_ns)
        # Clay angular speed on the sphere, from the filtered rates.
        rate_fn = getattr(tracker, "angular_rate", None)
        if callable(rate_fn):
            omega_clay = float(rate_fn(track))
        else:
            el = float(track.x[3])
            omega_clay = math.hypot(float(track.x[1]) * math.cos(el), float(track.x[4]))
        h = _DTOF_STEP_M
        lo = max(dist - h, 1e-3)
        dtof_dr = (_tof(shot, dist + h) - _tof(shot, lo)) / (
            dist + h - lo
        )
        sigma = math.sqrt(
            track_sigma**2
            + (omega_clay * jitter_s) ** 2
            + (omega_clay * dtof_dr * float(track.range_sigma_m)) ** 2
            + gun_sigma**2
        )
        pattern = (max(shot._pattern_min_m, shot._pattern_ref_m * dist / shot._pattern_ref_range_m)
                   if tabulated else float(shot.pattern_radius(dist))) / dist

        return {
            "t_exit_ns": t_exit_ns,
            "miss": miss,
            "pattern": pattern,
            "sigma": sigma,
            "p_hit": self._p_hit(miss, pattern, sigma),
            "range_m": rng,
            "tof": tof,
            "dist": dist,
            "lead_dir_w": lead_dir,
            "shot_dir_w": _unit(to_shot),
            "bore_dir_w": bore_w,
            "clay_el": math.asin(max(-1.0, min(1.0, float(lead_dir[2])))),
            "track_sigma": track_sigma,
            "gun_sigma": gun_sigma,
            "omega_clay": omega_clay,
        }
