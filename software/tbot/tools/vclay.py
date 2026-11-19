"""`tbot vclay <press> ...` - virtual clays flown through recorded gun motion.

The simulator (`tbot sim`) invents the shooter. A recorded press holds a real one: the IMU of a real swing on a
real clay, and the frame times of a real, loaded Pi (20-50 frames a second, 40-60 ms old). This tool keeps
both and replaces only the target:

  * for a moment of the press, a clay is placed where a charge leaving the muzzle THEN would have arrived (give
    or take a part of the pattern), flying with a speed and direction a clay has, as fast across the sky as the
    gun was really swinging (times a factor: swing-through or sustained lead);
  * its detections are made for the recorded frame times (position and size noise, dropouts, the field of
    view, the barrel) and go through the real tracker and the real fire solver, which see the attitude only as
    the live loop does (the IMU up to the last sample read before the decision);
  * every shot is judged against the truth: the bore from the whole IMU record at command + the TRUE trigger
    delay (which may differ from the one the solver assumes: --delay-error-ms), the clay where it really is
    when the charge arrives.

So it answers what neither the simulator nor nine presses can: clays hit out of hundreds, per firing rule, per
frame rate, per trigger delay, with a real hand on the gun. It is a relative test - the clays are built around
the path the bore took, so a chance always exists; the number says how well the rule takes it.

    tbot vclay field_data/2026-11-17-range/presses/*            # the shipped rule on the range-day swings
    tbot vclay <presses> --policies                             # compare firing rules on the same clays
    tbot vclay <presses> --fps 100 --age-ms 20                  # what a faster loop would be worth
    tbot vclay <presses> --delay-error-ms 60                    # what a wrong trigger delay costs
    tbot vclay <presses> --calibration                          # does P(hit) mean what it says?
    tbot vclay <presses> --quick                                # a quick shooter: on the clay 0.35-0.7 s after it appears
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tbot.config import Config, LatencyBudget, add_common_args, load_from_args
from tbot.fire.ballistics import G_MPS2, ShotModel
from tbot.fire.policy import FirePolicy
from tbot.fire.solver import FireSolver
from tbot.imu.attitude import ALPHA_WINDOW_S, _exp9, _mul9
from tbot.optics import combined_mask
from tbot.sim.scenario import _clay_accel
from tbot.tools.score import Session, rig_config
from tbot.types import Detection
from tbot.vision.tracker3d import make_tracker

UP = np.array([0.0, 0.0, 1.0])
PATH_STEP_S = 0.005
MIN_PRESS_S = 0.8          # shorter presses hold no swing to fly a clay through
MAX_TRACKING_DPS = 60.0    # faster than this the gun is being mounted or lowered, not swung on a clay
IMU_FIFO_MS = 5.0          # age of the newest IMU sample when it is read (polled at 100 Hz)
RATE_FACTOR = (0.35, 1.05)  # clay rate across the sky / gun swing rate at the planned shot
RATE_SCATTER_DPS = 4.0     # plus this much of its own, per axis
SPEED_CLASSES = {          # clay speed over the ground, m/s
    "slow": (10.0, 15.0),
    "medium": (15.0, 20.0),
    "fast": (20.0, 26.0),
}
FLY_S = (0.7, 1.6)         # the clay is in the air (and visible) this long before the planned arrival: a deliberate
                           # shooter; --quick flies the crossing 0.35-0.7 s after the clay appears, as a quick one does
QUICK_FLY_S = (0.35, 0.7)


def _unit(v: np.ndarray) -> np.ndarray:
    n = math.sqrt(float(v @ v))
    return v / n if n > 0.0 else v


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    c = np.cross(a, b)
    return math.atan2(math.sqrt(float(c @ c)), float(a @ b))


class CausalAttitude:
    """The whole-press attitude as the live loop saw it at one moment: nothing after `limit_ns` (the last IMU
    sample read before the decision); past it the gun is extrapolated at its last rate, as AttitudeEstimator
    does past its newest sample."""

    ready = True

    def __init__(self, full) -> None:
        self.full = full
        self.limit_ns = 0

    def R_wg(self, t_ns: int) -> np.ndarray:
        t = int(t_ns)
        if t <= self.limit_ns:
            return self.full.R_wg(t)
        R = np.asarray(self.full.R_wg(self.limit_ns), dtype=np.float64).reshape(9)
        w = self.full.omega_g(self.limit_ns)
        dt = (t - self.limit_ns) * 1e-9
        out = _mul9(tuple(float(x) for x in R), _exp9(float(w[0]) * dt, float(w[1]) * dt, float(w[2]) * dt))
        return np.array(out, dtype=np.float64).reshape(3, 3)

    def omega_g(self, t_ns: int) -> np.ndarray:
        return self.full.omega_g(min(int(t_ns), self.limit_ns))

    def alpha_g(self, t_ns: int, window_s: float = ALPHA_WINDOW_S) -> np.ndarray:
        return self.full.alpha_g(min(int(t_ns), self.limit_ns), window_s)


@dataclass
class FrameTime:
    t_exp_ns: int       # mid-exposure
    t_decide_ns: int    # when the solver ran on it
    imu_lag_ns: int     # age of the newest IMU sample at that moment
    armed: bool


@dataclass
class VClay:
    """One virtual clay: its true path (world frame, camera at the origin) and how it was made."""

    t0_ns: int
    path: np.ndarray            # (n, 3) positions every PATH_STEP_S from t0_ns
    t_plan_exit_ns: int         # the moment of the press it was built around
    range_m: float
    speed_mps: float
    rate_dps: float             # across the sky at the planned arrival
    offset_deg: float           # how far from the pattern centre a shot leaving at t_plan_exit_ns would pass
    cls: str
    visible_from_ns: int

    def pos(self, t_ns: int) -> np.ndarray | None:
        x = (int(t_ns) - self.t0_ns) * 1e-9 / PATH_STEP_S
        if x < 0.0 or x > len(self.path) - 1:
            return None
        i = min(int(x), len(self.path) - 2)
        f = x - i
        return (1.0 - f) * self.path[i] + f * self.path[i + 1]


@dataclass
class Row:
    """One armed frame of one clay: what the solver said, and what a command sent then would really have done."""

    t_ns: int
    reason: str
    p_hit: float
    bar: float
    fire: bool
    fire_at_ns: int
    track_age_s: float
    sigma_deg: float
    miss_deg: float
    pattern_deg: float
    range_m: float
    solved: bool
    ceiling: float                   # P(hit) of a perfect aim with this frame's uncertainty
    peak: bool                       # no later frame of the look-ahead promises more
    true_miss_deg: float = float("nan")
    true_hit: bool = False
    true_range_m: float = float("nan")
    gun_err_deg: float = float("nan")
    target_err_deg: float = float("nan")
    track_sigma_deg: float = float("nan")   # the solver's own 1-sigma of the target and of the gun at exit
    gun_sigma_deg: float = float("nan")


@dataclass
class Result:
    cls: str
    press: str
    clay: VClay
    rows: list[Row] = field(default_factory=list)
    shot: Row | None = None
    detections: int = 0
    frames_in_view: int = 0


class Press:
    """A recorded press as a stage for virtual clays: true attitude, frame times, the gun's own calibration."""

    def __init__(self, path: str | Path, cfg: Config, fps: float = 0.0, age_ms: float = 0.0,
                 imu_lag_ms: float | None = None) -> None:
        self.path = Path(path)
        self.name = self.path.name
        s = Session(self.path, cfg=cfg)
        self.session = s
        self.cfg = cfg
        self.cam, self.mount = s.cam, s.mount
        self.R_gc = s.R_gc
        self.full = s.att
        w, h = s.size
        self.size = (w, h)
        cal = s.meta.get("calibration") or {}
        calib = type("Cal", (), {"mask": None, "detector": cal.get("detector"), "intrinsics": cal.get("intrinsics"),
                                 "boresight": cal.get("boresight")})()
        try:
            self.mask = combined_mask(self.cam, self.mount, cfg, calib)
        except Exception:  # noqa: BLE001 - a mask is a refinement; the field of view still applies
            self.mask = None
        arm = [int(e["t_ns"]) for e in s.events if e.get("type") == "arming" and e.get("armed")]
        rel = [int(e["t_ns"]) for e in s.events if e.get("type") == "arming" and not e.get("armed")]
        recoil = [int(e["t_ns"]) for e in s.events if e.get("type") == "recoil"]
        self.t_arm = arm[0] if arm else int(s.t_exp[0])
        self.t_release = rel[0] if rel else int(s.imu_span[1])
        # the swing after a real shot is recoil, not aiming
        self.t_end = min([self.t_release, int(s.imu_span[1])] + [t - int(5e6) for t in recoil])
        self.frames: list[FrameTime] = []
        if fps > 0.0:
            period = int(round(1e9 / fps))
            age = int(round(age_ms * 1e6))
            lag = int(round((IMU_FIFO_MS if imu_lag_ms is None else
                             max(age_ms - 10.0, IMU_FIFO_MS) if imu_lag_ms < 0 else imu_lag_ms) * 1e6))
            t = int(s.t_exp[0])
            while t + age < self.t_end:
                self.frames.append(FrameTime(t, t + age, lag, self.t_arm <= t + age <= self.t_release))
                t += period
        else:
            for r in s.tel:
                st = r.get("stage_ms") or {}
                # The loop reads the IMU again just before the solver (app/run.py), so the newest sample is a FIFO
                # period old. "stale" = as on the first range day: read once, before detection and tracking.
                if imu_lag_ms is None:
                    lag = IMU_FIFO_MS
                elif imu_lag_ms < 0:
                    lag = (st.get("attitude", 1.5) + st.get("detect", 15.0) + st.get("track", 1.5)
                           + float(r.get("decide_ahead_ms") or 0.0) + IMU_FIFO_MS)
                else:
                    lag = imu_lag_ms
                t_dec = int(r["t_ns"])
                if t_dec >= self.t_end:
                    break
                self.frames.append(FrameTime(int(r["t_exposure_ns"]), t_dec, int(lag * 1e6),
                                             bool(r.get("armed")) and t_dec <= self.t_release))

    @property
    def armed_s(self) -> float:
        return (self.t_end - self.t_arm) * 1e-9

    def R(self, t_ns: int) -> np.ndarray:
        return np.asarray(self.full.R_wg(int(t_ns)), dtype=np.float64).reshape(3, 3)

    def swing_dps(self, t_ns: int) -> float:
        w = self.full.omega_g(int(t_ns))
        return math.degrees(math.sqrt(float(w @ w)))

    def pixel(self, dir_w: np.ndarray, t_ns: int) -> tuple[float, float] | None:
        ray_c = (self.R(t_ns) @ self.R_gc).T @ dir_w
        if ray_c[2] <= 0.05:
            return None
        uv = np.asarray(self.cam.ray_to_pixel(ray_c), dtype=np.float64).reshape(2)
        if not np.isfinite(uv).all():
            return None
        u, v = float(uv[0]), float(uv[1])
        w, h = self.size
        if not (3.0 <= u <= w - 4.0 and 3.0 <= v <= h - 4.0):
            return None
        if self.mask is not None and not self.mask[int(round(v)), int(round(u))]:
            return None
        return u, v


class Bench:
    """Makes virtual clays for a press and runs them through tracker and solver."""

    def __init__(self, cfg: Config, true_delay_error_ms: float = 0.0, jitter_ms: float = 1.5,
                 noise_px: float = 0.5, size_noise: float = 0.12, dropout: float = 0.1,
                 cmd_late_ms: float = 0.0, fly_s: tuple[float, float] = FLY_S) -> None:
        self.cfg = cfg
        self.fly_s = (float(fly_s[0]), float(fly_s[1]))
        self.shot = ShotModel(cfg)
        self.budget = LatencyBudget.from_config(cfg, None)
        self.atype = str(cfg.actuator.type)
        self.assumed_delay_s = self.budget.fire_delay_s(self.atype)
        self.true_delay_s = self.assumed_delay_s + true_delay_error_ms * 1e-3
        self.jitter_s = jitter_ms * 1e-3
        self.noise_px = noise_px
        self.size_noise = size_noise
        self.dropout = dropout
        self.cmd_late_s = cmd_late_ms * 1e-3
        # every frame that passes the gates gets its look-ahead, whatever the bar (the rules are compared on it)
        self.cfg_open = copy.deepcopy(cfg)
        self.cfg_open.fire.min_hit_probability = 0.0
        self.cfg_open.fire.min_hit_probability_settled = 0.0
        self.cfg_open.fire.adaptive = False

    # ------------------------------------------------------------------ the clay
    def make_clay(self, press: Press, rng: np.random.Generator, cls: str) -> VClay | None:
        solver = FireSolver(self.cfg, self.shot, press.mount, self.budget)
        bore_g, muzzle_g = solver._bore_g, solver._muzzle_g
        lo = press.t_arm + int(0.25e9)
        hi = press.t_end - int(0.02e9)
        if hi <= lo:
            return None
        for _ in range(40):
            t_exit = int(rng.integers(lo, hi))
            # a moment the shooter was on a target: no mounting or lowering of the gun around it
            if max(press.swing_dps(t_exit - int(k * 0.05e9)) for k in range(0, 6)) > MAX_TRACKING_DPS:
                continue
            R = press.R(t_exit)
            bore_w = _unit(R @ bore_g)
            if bore_w[2] < math.sin(math.radians(float(self.cfg.fire.min_elevation_deg) + 2.0)):
                continue
            rng_m = float(rng.uniform(max(float(self.cfg.fire.min_range_m) + 4.0, 14.0),
                                      min(float(self.cfg.fire.max_range_m) - 4.0, 36.0)))
            tof = float(self.shot.time_of_flight(rng_m))
            drop = 0.5 * G_MPS2 * tof * tof * math.cos(math.asin(max(-1.0, min(1.0, float(bore_w[2])))))
            to_shot = bore_w * rng_m - drop * UP
            pattern = float(self.shot.pattern_radius(rng_m)) / rng_m
            # where in the pattern the clay is when the charge of THIS moment arrives: uniform over 60 % of it
            off = 0.6 * pattern * math.sqrt(float(rng.uniform()))
            e1 = _unit(np.cross(to_shot, UP))
            e2 = _unit(np.cross(to_shot, e1))
            phi = float(rng.uniform(0.0, 2.0 * math.pi))
            p_arr = R @ muzzle_g + to_shot + rng_m * math.tan(off) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            # Across the sky as the gun swings, times a factor, plus a little of its own; the rest of the clay's
            # speed is along the line of sight. The gun overtakes its target (swing-through): on the range-day
            # presses the clay crossed at 9-17 deg/s under a gun turning at 11-29, and bore and clay moved
            # against each other at 2.3 / 7 / 20 deg/s (p10 / p50 / p90, 53 frame pairs). Factor and scatter
            # are chosen to give that: the pattern is crossed in 0.1-0.4 s, not sat in.
            w_w = R @ press.full.omega_g(t_exit)
            los = _unit(p_arr)
            dist = math.sqrt(float(p_arr @ p_arr))
            v_t = np.cross(w_w, los) * dist * float(rng.uniform(RATE_FACTOR[0], RATE_FACTOR[1]))
            v_t = v_t + rng.normal(0.0, math.radians(RATE_SCATTER_DPS) * dist, 3)
            v_t -= float(v_t @ los) * los
            speed = float(rng.uniform(*SPEED_CLASSES[cls]))
            vt = math.sqrt(float(v_t @ v_t))
            if vt > speed:
                continue  # this moment's swing is faster than a clay of this class crosses at this range
            radial = math.sqrt(speed * speed - vt * vt) * (1.0 if rng.uniform() < 0.8 else -1.0)
            vel = v_t + radial * los
            t_arr = t_exit + int(tof * 1e9)
            fly_s = float(rng.uniform(*self.fly_s))  # in the air (and visible) this long before the planned arrival
            t0 = max(t_arr - int(fly_s * 1e9), int(press.session.t_exp[0]))
            t1 = press.t_end + int(0.6e9)
            n_back = int(math.ceil((t_arr - t0) * 1e-9 / PATH_STEP_S)) + 1
            n_fwd = int(math.ceil((t1 - t_arr) * 1e-9 / PATH_STEP_S)) + 1
            back = self._integrate(p_arr, vel, n_back, -PATH_STEP_S)
            fwd = self._integrate(p_arr, vel, n_fwd, PATH_STEP_S)
            path = np.concatenate([back[::-1], fwd[1:]])
            t_path0 = t_arr - (n_back - 1) * int(PATH_STEP_S * 1e9)
            rate = math.degrees(vt / dist)
            return VClay(t0_ns=t_path0, path=path, t_plan_exit_ns=t_exit, range_m=dist, speed_mps=speed,
                         rate_dps=rate, offset_deg=math.degrees(off), cls=cls, visible_from_ns=t0)
        return None

    @staticmethod
    def _integrate(p0: np.ndarray, v0: np.ndarray, n: int, dt: float) -> np.ndarray:
        out = np.empty((n, 3))
        p, v = p0.astype(np.float64).copy(), v0.astype(np.float64).copy()
        out[0] = p
        for i in range(1, n):
            a = _clay_accel(v)
            p = p + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
            out[i] = p
        return out

    # ------------------------------------------------------------------ truth
    def outcome(self, press: Press, clay: VClay, t_cmd_ns: int, solver: FireSolver,
                jitter_s: float = 0.0) -> tuple[float, bool, float, np.ndarray, np.ndarray] | None:
        """(miss angle, hit, range, true bore, true direction to the clay at arrival) for a command at t_cmd_ns."""
        t_exit = int(t_cmd_ns) + int(round((self.true_delay_s + self.cmd_late_s + jitter_s) * 1e9))
        if t_exit > press.session.imu_span[1]:
            return None
        R = press.R(t_exit)
        bore_w = _unit(R @ solver._bore_g)
        muzzle_w = R @ solver._muzzle_g
        p = clay.pos(t_exit)
        if p is None:
            return None
        tof = 0.0
        for _ in range(6):
            dist = math.sqrt(float((p - muzzle_w) @ (p - muzzle_w)))
            tof = float(self.shot.time_of_flight(dist))
            nxt = clay.pos(t_exit + int(tof * 1e9))
            if nxt is None:
                return None
            p = nxt
        to_clay = p - muzzle_w
        dist = math.sqrt(float(to_clay @ to_clay))
        drop = 0.5 * G_MPS2 * tof * tof * math.cos(math.asin(max(-1.0, min(1.0, float(bore_w[2])))))
        to_shot = bore_w * dist - drop * UP
        miss = _angle(to_clay, to_shot)
        pattern = float(self.shot.pattern_radius(dist)) / dist
        return miss, bool(miss <= pattern), dist, bore_w, _unit(to_clay)

    # ------------------------------------------------------------------ one clay through the loop
    def run(self, press: Press, clay: VClay, rng: np.random.Generator, policy: bool = True) -> Result:
        """policy=True: the shipped rule decides (FireSolver as configured). False: every gated frame gets its
        look-ahead and nothing fires, for comparing rules afterwards (replay)."""
        cfg = self.cfg if policy else self.cfg_open
        tracker = make_tracker(cfg, press.cam, press.mount)
        solver = FireSolver(cfg, self.shot, press.mount, self.budget)
        att = CausalAttitude(press.full)
        res = Result(cls=clay.cls, press=press.name, clay=clay)
        d_clay = float(self.cfg.mount.clay_diameter_m)
        prev_exp = None
        fps = 0.0
        nominal = 1.0 / max(float(cfg.camera.fps), 1.0)
        for fr in press.frames:
            dets = []
            p = clay.pos(fr.t_exp_ns) if fr.t_exp_ns >= clay.visible_from_ns else None
            if p is not None:
                dist = math.sqrt(float(p @ p))
                uv = press.pixel(p / dist, fr.t_exp_ns)
                if uv is not None:
                    res.frames_in_view += 1
                    if rng.uniform() >= self.dropout:
                        ifov = float(press.cam.ifov1(uv[0], uv[1]))
                        dia = max(1.0, d_clay / dist / ifov * (1.0 + float(rng.normal(0.0, self.size_noise))))
                        u = uv[0] + float(rng.normal(0.0, self.noise_px))
                        v = uv[1] + float(rng.normal(0.0, self.noise_px))
                        h = max(1.0, 0.45 * dia)
                        dets.append(Detection(t_ns=fr.t_exp_ns, u=u, v=v, area_px=0.785 * dia * h, diameter_px=dia,
                                              contrast=-40.0, snr=20.0, score=0.8,
                                              bbox=(int(u - dia / 2), int(v - h / 2), int(dia) + 1, int(h) + 1)))
                        res.detections += 1
            tracker.update(fr.t_exp_ns, dets, press.R(fr.t_exp_ns))
            if prev_exp is not None and fr.t_exp_ns > prev_exp:
                inst = 1e9 / (fr.t_exp_ns - prev_exp)
                fps = inst if fps <= 0.0 else 0.9 * fps + 0.1 * inst
            prev_exp = fr.t_exp_ns
            if not fr.armed or res.shot is not None:
                continue
            att.limit_ns = fr.t_decide_ns - fr.imu_lag_ns
            period = nominal if fps <= 0.0 else min(0.2, max(nominal, 1.0 / fps))
            d = solver.evaluate(fr.t_decide_ns, tracker, att, True, period)
            if d.track_id is None:
                continue
            solved = d.reason.startswith(("p_hit ", "better in", "FIRE"))
            tr = next((t for t in tracker.tracks if t.track_id == d.track_id), None)
            row = Row(t_ns=fr.t_decide_ns, reason=d.reason, p_hit=float(d.p_hit), bar=float(d.extra.get("bar", 0.0)),
                      fire=bool(d.fire), fire_at_ns=int(d.extra.get("fire_at_ns", fr.t_decide_ns)),
                      track_age_s=(fr.t_decide_ns - int(tr.first_t_ns)) * 1e-9 if tr is not None else 0.0,
                      sigma_deg=math.degrees(d.sigma_rad), miss_deg=math.degrees(d.miss_rad),
                      pattern_deg=math.degrees(d.pattern_rad), range_m=float(d.range_m), solved=solved,
                      ceiling=float(d.extra.get("ceiling", float("nan"))),
                      track_sigma_deg=math.degrees(float(d.extra.get("track_sigma_rad", float("nan")))),
                      gun_sigma_deg=math.degrees(float(d.extra.get("gun_sigma_rad", float("nan")))),
                      peak=d.reason == "FIRE" or (solved and not d.reason.startswith("better in")
                                                  and not any(f > d.p_hit for f in d.extra.get("future_p_hit", []))))
            if solved:
                out = self.outcome(press, clay, row.fire_at_ns, solver,
                                   float(rng.normal(0.0, self.jitter_s)) if self.jitter_s > 0.0 else 0.0)
                if out is not None:
                    miss, hit, dist, bore_true, clay_true = out
                    row.true_miss_deg, row.true_hit, row.true_range_m = math.degrees(miss), hit, dist
                    if d.bore_dir_w is not None:
                        row.gun_err_deg = math.degrees(_angle(np.asarray(d.bore_dir_w), bore_true))
                    if d.lead_dir_w is not None:
                        row.target_err_deg = math.degrees(_angle(np.asarray(d.lead_dir_w), clay_true))
            res.rows.append(row)
            if d.fire and policy:
                res.shot = row
        return res


# --------------------------------------------------------------------------------------------- rules, replayed
def replay(result: Result, policy: FirePolicy) -> Row | None:
    """The shot a firing rule would have taken on a clay run with policy=False (every look-ahead present):
    the first frame at a peak of P(hit) whose P clears the rule's bar."""
    policy.reset()
    for row in result.rows:
        if not row.solved:
            continue
        bar = policy.bar(0, row.t_ns, row.t_ns - int(row.track_age_s * 1e9), row.p_hit, row.ceiling)
        if row.peak and row.p_hit >= bar:
            return row
    return None


def tally(results: list[Result], shots: list[Row | None]) -> dict:
    n = len(results)
    fired = [s for s in shots if s is not None]
    hits = sum(1 for s in fired if s.true_hit)
    out = {"clays": n, "shots": len(fired), "hits": hits,
           "clays_hit_pct": 100.0 * hits / n if n else 0.0,
           "shot_pct": 100.0 * len(fired) / n if n else 0.0,
           "hit_per_shot_pct": 100.0 * hits / len(fired) if fired else 0.0}
    if fired:
        out["p_at_fire_mean"] = float(np.mean([s.p_hit for s in fired]))
        miss = np.array([s.true_miss_deg for s in fired if math.isfinite(s.true_miss_deg)])
        if miss.size:
            out["true_miss_deg_p50"] = float(np.median(miss))
        # how long the gun followed its target before it fired (a quick shooter is on the clay in 0.3-0.5 s)
        out["track_age_ms_p50"] = float(np.median([s.track_age_s for s in fired])) * 1e3
        out["track_age_ms_p90"] = float(np.percentile([s.track_age_s for s in fired], 90)) * 1e3
        for key in ("gun_err_deg", "target_err_deg"):
            v = np.array([getattr(s, key) for s in fired])
            v = v[np.isfinite(v)]
            if v.size:
                out[key + "_p50"] = float(np.median(v))
    return out


P_BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01)


def calibration(results: list[Result]) -> list[dict]:
    """Every solved frame scored as if the command had gone out on it: of the frames given P, how many hit?"""
    rows = [r for res in results for r in res.rows if r.solved and math.isfinite(r.true_miss_deg)]
    out = []
    if not rows:
        return out
    p = np.array([r.p_hit for r in rows])
    hit = np.array([r.true_hit for r in rows])
    for lo, hi in zip(P_BINS, P_BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            out.append({"p_lo": lo, "p_hi": min(hi, 1.0), "frames": int(m.sum()), "p_mean": float(p[m].mean()),
                        "hit_frac": float(hit[m].mean())})
    return out


def make_results(presses: list[Press], bench: Bench, clays_per_press: int, seed: int, policy: bool,
                 classes=tuple(SPEED_CLASSES)) -> list[Result]:
    results = []
    for k, press in enumerate(presses):
        rng = np.random.default_rng([seed, k])
        for n in range(clays_per_press):
            cls = classes[n % len(classes)]
            clay = bench.make_clay(press, rng, cls)
            if clay is None:
                continue
            results.append(bench.run(press, clay, np.random.default_rng([seed, k, n]), policy=policy))
    return results


def named_policies(cfg: Config) -> dict[str, FirePolicy]:
    """The shipped rule and the ones it is measured against."""
    f = cfg.fire
    base = dict(bar=float(f.min_hit_probability), floor=float(f.min_hit_probability_floor),
                reach=float(f.adaptive_reach), regret=float(f.adaptive_regret),
                relax_from_s=float(f.adaptive_relax_from_ms) * 1e-3, relax_to_s=float(f.adaptive_relax_to_ms) * 1e-3,
                relaxed=float(f.min_hit_probability_settled))
    off = dict(reach=0.0, regret=0.0, relax_from_s=0.0, relax_to_s=0.0)
    def flat(x: float) -> FirePolicy:
        return FirePolicy(**{**base, **off, "bar": x, "floor": x, "relaxed": x})

    return {
        "as configured": FirePolicy.from_config(cfg),
        "fixed 0.50": flat(0.50),
        "fixed 0.40": flat(0.40),
        "fixed 0.30": flat(0.30),
        "fixed 0.25": flat(0.25),
        "0.40, then 0.30 from 0.7 s (the step)": FirePolicy(**{**base, **off, "bar": 0.40, "floor": 0.30,
                                                               "relaxed": 0.30, "relax_from_s": 0.7,
                                                               "relax_to_s": 0.7}),
        "bar + reach only": FirePolicy(**{**base, "regret": 0.0, "relax_from_s": 0.0, "relax_to_s": 0.0,
                                          "relaxed": base["bar"]}),
        "bar + regret only": FirePolicy(**{**base, "reach": 0.0, "relax_from_s": 0.0, "relax_to_s": 0.0,
                                           "relaxed": base["bar"]}),
        "bar + relax only": FirePolicy(**{**base, "reach": 0.0, "regret": 0.0}),
    }


def _fmt(name: str, t: dict) -> str:
    extra = ""
    if "true_miss_deg_p50" in t:
        extra = (f"   miss p50 {t['true_miss_deg_p50']:.2f} deg  gun {t.get('gun_err_deg_p50', float('nan')):.2f}"
                 f"  target {t.get('target_err_deg_p50', float('nan')):.2f}")
    if "track_age_ms_p50" in t:
        extra += f"   track age at shot p50 {t['track_age_ms_p50']:.0f} ms"
    return (f"  {name:<40s} clays hit {t['clays_hit_pct']:5.1f} %   shot at {t['shot_pct']:5.1f} %   "
            f"hits per shot {t['hit_per_shot_pct']:5.1f} %{extra}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot vclay", description="virtual clays through recorded gun motion")
    add_common_args(p)
    p.add_argument("presses", nargs="+", metavar="PRESS", help="recorded press directories (with imu.csv)")
    p.add_argument("--clays", type=int, default=60, help="virtual clays per press (default 60)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--policies", action="store_true", help="compare firing rules on the same clays")
    p.add_argument("--calibration", action="store_true", help="P(hit) given against hits, every solved frame")
    p.add_argument("--fps", type=float, default=0.0, help="frames at this steady rate instead of the recorded times")
    p.add_argument("--age-ms", type=float, default=30.0, help="with --fps: frame age when the solver runs")
    p.add_argument("--imu-lag-ms", type=float, default=None,
                   help="age of the newest IMU sample at the decision (default 5: read again just before the "
                        "solver; -1 = stale, read once before detection as on the first range day)")
    p.add_argument("--delay-error-ms", type=float, default=0.0,
                   help="the true trigger delay is this much longer than the solver assumes")
    p.add_argument("--cmd-late-ms", type=float, default=0.0, help="the command goes out this long after planned")
    p.add_argument("--jitter-ms", type=float, default=1.5, help="shot-to-shot spread of the true delay (1 sigma)")
    p.add_argument("--quick", action="store_true",
                   help="a quick shooter: the crossing comes 0.35-0.7 s after the clay appears (default 0.7-1.6 s)")
    p.add_argument("--dropout", type=float, default=0.1, help="share of frames in which the clay is not detected")
    p.add_argument("--json", default=None, metavar="PATH", help="write the results as JSON")
    args = p.parse_args(argv)

    cfg, _calib = load_from_args(args)
    paths = [Path(m) for pat in args.presses for m in (sorted(glob.glob(pat)) or [pat])]
    try:
        cfg = rig_config(cfg, paths[0])
    except Exception as exc:  # noqa: BLE001
        print(f"tbot vclay: {paths[0]}: {exc}", file=sys.stderr)
        return 1
    presses = []
    for path in paths:
        try:
            pr = Press(path, cfg, fps=args.fps, age_ms=args.age_ms, imu_lag_ms=args.imu_lag_ms)
        except Exception as exc:  # noqa: BLE001 - one unreadable press must not stop the rest
            print(f"tbot vclay: {path}: {exc}", file=sys.stderr)
            continue
        if pr.armed_s >= MIN_PRESS_S and len(pr.frames) >= 10:
            presses.append(pr)
    if not presses:
        print("tbot vclay: no press long enough to fly a clay through", file=sys.stderr)
        return 1
    bench = Bench(cfg, true_delay_error_ms=args.delay_error_ms, jitter_ms=args.jitter_ms, dropout=args.dropout,
                  cmd_late_ms=args.cmd_late_ms, fly_s=QUICK_FLY_S if args.quick else FLY_S)
    print(f"{len(presses)} press(es), {args.clays} clays each; trigger delay assumed {bench.assumed_delay_s * 1e3:.0f} ms, "
          f"true {bench.true_delay_s * 1e3:.0f} ms; frames: "
          + (f"{args.fps:.0f} fps, {args.age_ms:.0f} ms old" if args.fps > 0 else "as recorded"))
    report: dict = {"presses": [pr.name for pr in presses], "assumed_delay_ms": bench.assumed_delay_s * 1e3,
                    "true_delay_ms": bench.true_delay_s * 1e3}
    if args.policies or args.calibration:
        results = make_results(presses, bench, args.clays, args.seed, policy=False)
    else:
        results = make_results(presses, bench, args.clays, args.seed, policy=True)
    seen = sum(r.detections for r in results)
    print(f"{len(results)} clays, {seen} detections "
          f"({np.mean([r.detections for r in results]):.0f} a clay), "
          f"{sum(1 for r in results if any(x.solved for x in r.rows))} with a firing solution at some point")
    if args.policies:
        report["policies"] = {}
        for name, pol in named_policies(cfg).items():
            shots = [replay(r, pol) for r in results]
            t = tally(results, shots)
            report["policies"][name] = t
            print(_fmt(name, t))
            for cls in SPEED_CLASSES:
                idx = [i for i, r in enumerate(results) if r.cls == cls]
                tc = tally([results[i] for i in idx], [shots[i] for i in idx])
                report["policies"][name + " / " + cls] = tc
                print(f"      {cls:<8s} {tc['clays_hit_pct']:5.1f} % of {tc['clays']}")
    if args.calibration:
        cal = calibration(results)
        report["calibration"] = cal
        print("  P(hit) given    frames   really hit")
        for b in cal:
            print(f"    {b['p_lo']:.2f} - {b['p_hi']:.2f}   {b['frames']:6d}   {100.0 * b['hit_frac']:5.0f} %")
    if not (args.policies or args.calibration):
        t = tally(results, [r.shot for r in results])
        report["shipped"] = t
        print(_fmt("shipped rule", t))
        for cls in SPEED_CLASSES:
            sub = [r for r in results if r.cls == cls]
            tc = tally(sub, [r.shot for r in sub])
            report["shipped / " + cls] = tc
            print(f"      {cls:<8s} {tc['clays_hit_pct']:5.1f} % of {tc['clays']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
