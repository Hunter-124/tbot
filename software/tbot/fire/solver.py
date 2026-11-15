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

_UP_W = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_TOF_ITERS = 3  # fixed-point passes on time of flight (converges in 2 at skeet ranges)
_DTOF_STEP_M = 0.5  # finite-difference step for d(ToF)/d(range)

# Polar quadrature of the aim-error Gaussian over the pattern disc (no scipy, no Bessel overflow).
_QUAD_R_X, _QUAD_R_W = np.polynomial.legendre.leggauss(16)
_QUAD_T_X, _QUAD_T_W = np.polynomial.legendre.leggauss(32)
_QUAD_T_UNIT = 0.5 * (_QUAD_T_X + 1.0)
_QUAD_T_HALF_W = 0.5 * _QUAD_T_W
_QUAD_NSIG = 6.0  # integrate only where the Gaussian has mass; beyond 6 sigma it is < 1e-8


def _hit_probability(miss: float, pattern: float, sigma: float) -> float:
    """P(|e + miss| < pattern) for an isotropic 2-D Gaussian aim error e ~ N(0, sigma^2 I)."""
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
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v

def _tof(shot, distance: float) -> float:
    # np.interp already clamps at the table ends, so the scalar np.clip in
    # ShotModel._lookup is unnecessary here. Preserve custom shot models.
    if type(shot) is ShotModel:
        return float(np.interp(distance, shot._x, shot._t))
    return float(shot.time_of_flight(distance))



class FireSolver:
    """Decides whether firing *now* is the best moment to break the clay."""

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

    def evaluate(
        self,
        now_ns: int,
        tracker,
        attitude,
        armed: bool,
        frame_period_s: float,
    ) -> FireDecision:
        now_ns = int(now_ns)
        track = tracker.best()
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
        g = self._geometry(now_ns, now_ns, track, tracker, rot_now, omega, delay_s, jitter_s, alpha)

        fire = False
        fc = self.cfg.fire
        future: list[float] = []
        if track.hits < fc.min_track_hits:
            reason = "track young"
        elif not (fc.min_range_m <= g["range_m"] <= fc.max_range_m):
            reason = "range out of window"
        elif g["clay_el"] < math.radians(fc.min_elevation_deg):
            reason = "below min elevation"
        elif g["p_hit"] < fc.min_hit_probability:
            reason = f"p_hit {g['p_hit']:.2f} < {fc.min_hit_probability:.2f}"
        else:
            better_k = 0
            period_ns = int(round(float(frame_period_s) * 1e9))
            if period_ns > 0:
                for k in range(1, int(fc.horizon_frames) + 1):
                    gk = self._geometry(
                        now_ns, now_ns + k * period_ns, track, tracker, rot_now, omega, delay_s, jitter_s, alpha
                    )
                    future.append(gk["p_hit"])
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
                "gun_alpha_rad_s2": float(np.linalg.norm(alpha)),
                "clay_rate_rad_s": g["omega_clay"],
                "future_p_hit": future,
            },
        )

    # ----------------------------------------------------------------- internals
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
    ) -> dict:
        """Miss angle, uncertainty and hit probability for a trigger command issued at decide_ns."""
        t_exit_ns = int(decide_ns) + int(round(delay_s * 1e9))
        dt_exit = (t_exit_ns - now_ns) * 1e-9
        if alpha is None:
            alpha = np.zeros(3, dtype=np.float64)
        # Constant angular acceleration over the delay: a brisk swing changes rate by hundreds of deg/s^2, and a
        # constant-rate extrapolation then lands the shot behind (or ahead of) where the bore really is.
        rot_exit = rot_now @ _rotvec(omega * dt_exit + 0.5 * alpha * dt_exit * dt_exit)
        # The acceleration itself is not constant: carry a fraction of its effect as uncertainty, so the solver
        # does not fire confidently in the middle of a violent change of swing.
        gun_sigma = float(self.cfg.fire.gun_accel_sigma_frac) * 0.5 * float(np.linalg.norm(alpha)) * dt_exit * dt_exit
        bore_w = _unit(rot_exit @ self._bore_g)
        muzzle_w = rot_exit @ self._muzzle_g

        shot = self.shot
        tabulated = type(shot) is ShotModel
        rng = float(track.range_m)
        t_arr_ns = t_exit_ns
        p_clay = rng * tracker.predict_dir(track, t_arr_ns)
        dist = float(np.linalg.norm(p_clay - muzzle_w))
        tof = _tof(shot, dist)
        for _ in range(_TOF_ITERS):
            t_arr_ns = t_exit_ns + int(round(tof * 1e9))
            p_clay = rng * tracker.predict_dir(track, t_arr_ns)
            dist = float(np.linalg.norm(p_clay - muzzle_w))
            tof = _tof(shot, dist)

        to_clay = p_clay - muzzle_w
        lead_dir = _unit(to_clay)
        elevation = math.asin(max(-1.0, min(1.0, float(bore_w[2]))))
        drop = (float(0.5 * G_MPS2 * tof * tof * np.cos(elevation)) if tabulated
                else float(shot.drop(dist, elevation)))
        to_shot = bore_w * dist - drop * _UP_W
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
            "p_hit": _hit_probability(miss, pattern, sigma),
            "range_m": rng,
            "tof": tof,
            "dist": dist,
            "lead_dir_w": lead_dir,
            "bore_dir_w": bore_w,
            "clay_el": math.asin(max(-1.0, min(1.0, float(lead_dir[2])))),
            "track_sigma": track_sigma,
            "gun_sigma": gun_sigma,
            "omega_clay": omega_clay,
        }
