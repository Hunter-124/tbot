"""FireSolver decisions on synthetic clays and gun swings.

The scene is built from truth: a world clay position (camera optical centre = world origin) projected
through the nominal CameraModel/MountModel into detections, a closed-form gun attitude, and an
independent truth-based miss computation used as the reference for "the analytically best moment".

The tracker is deliberately configured for these scenes (low jerk PSD: the synthetic clays are slow,
distant straight lines whose angular jerk really is ~0; small measurement sigma: detections are near
noise-free). With the shipped defaults the combined 1-sigma on a 20 m/s crossing at 20 m is comparable to
the pattern radius, so p_hit stays below min_hit_probability — that is a property of the configuration,
not of the solver, and these tests exercise the decision logic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbot.config import LatencyBudget, load_config
from tbot.fire.ballistics import ShotModel
from tbot.fire.solver import FireSolver, _hit_probability
from tbot.optics import CameraModel, MountModel, azel_to_dir
from tbot.types import Detection
from tbot.vision.tracker import ClayTracker

W, H = 640, 400
FPS = 120.0
DT_NS = int(round(1e9 / FPS))
T0 = 1_000_000_000
SCENE_SETS = ["tracker.jerk_psd_deg=20.0", "tracker.meas_sigma_px=0.1"]


def _frame_from_azel(az: float, el: float) -> np.ndarray:
    x = np.asarray(azel_to_dir(az, el), dtype=float).reshape(3)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


class SwingAttitude:
    """Gun yawing about world +Z at a constant rate, bore elevation fixed."""

    def __init__(self, az0: float, el0: float, rate_rad_s: float, t0_ns: int = T0):
        self.az0, self.el0, self.rate, self.t0 = az0, el0, rate_rad_s, t0_ns
        self.ready = True

    def R_wg(self, t_ns: int) -> np.ndarray:
        return _frame_from_azel(self.az0 + self.rate * (int(t_ns) - self.t0) * 1e-9, self.el0)

    def omega_g(self, t_ns: int) -> np.ndarray:
        return self.R_wg(t_ns).T @ np.array([0.0, 0.0, self.rate])


class PursuitAttitude:
    """Gun bore exactly on the clay's current position: a shooter with no lead at all."""

    def __init__(self, clay):
        self.clay = clay
        self.ready = True

    def _azel(self, t_s: float) -> tuple[float, float]:
        p = self.clay(t_s)
        n = p / np.linalg.norm(p)
        return math.atan2(n[1], n[0]), math.asin(n[2])

    def R_wg(self, t_ns: int) -> np.ndarray:
        return _frame_from_azel(*self._azel((int(t_ns) - T0) * 1e-9))

    def omega_g(self, t_ns: int) -> np.ndarray:
        h = 1e-3
        t_s = (int(t_ns) - T0) * 1e-9
        r_p, r_m, r = self.R_wg(T0 + int((t_s + h) * 1e9)), self.R_wg(T0 + int((t_s - h) * 1e9)), self.R_wg(t_ns)
        w = (r_p - r_m) / (2.0 * h) @ r.T
        return r.T @ np.array([w[2, 1], w[0, 2], w[1, 0]])


class Scene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cam = CameraModel.nominal(cfg, W, H)
        self.mount = MountModel.from_config(cfg, None)
        self.shot = ShotModel(cfg)
        self.latency = LatencyBudget.from_config(cfg, None)
        self._muzzle_g = np.asarray(self.mount.muzzle_pos_g) - np.asarray(self.mount.cam_pos_g)

    def detect(self, p_w, R_wg, t_ns, rng=None, sigma_px=0.1) -> Detection | None:
        v_c = self.mount.gun_to_cam(R_wg.T @ np.asarray(p_w, dtype=float))
        if v_c[2] <= 0.0:
            return None
        uv = np.asarray(self.cam.ray_to_pixel(v_c[None, :]), dtype=float).reshape(2)
        if not (0.0 <= uv[0] < W and 0.0 <= uv[1] < H):
            return None
        if rng is not None and sigma_px:
            uv = uv + rng.normal(0.0, sigma_px, 2)
        ifov = float(np.asarray(self.cam.ifov(uv[None, :])).reshape(1)[0])
        d_eff = self.cfg.mount.clay_diameter_m / (float(np.linalg.norm(p_w)) * ifov)
        d = math.hypot(d_eff, self.cfg.detector.psf_px)
        return Detection(
            t_ns=t_ns, u=float(uv[0]), v=float(uv[1]), area_px=math.pi * d * d / 4.0, diameter_px=d,
            contrast=-45.0, snr=20.0, score=0.9,
            bbox=(int(uv[0] - d), int(uv[1] - d), max(1, int(2 * d)), max(1, int(2 * d))),
        )

    def truth_miss(self, att, clay, decide_ns: int, delay_s: float) -> float:
        """Miss angle of a shot commanded at decide_ns, computed from truth (no tracker)."""
        t_exit = int(decide_ns) + int(round(delay_s * 1e9))
        rot = att.R_wg(t_exit)
        bore = rot @ np.asarray(self.mount.bore_dir_g, dtype=float)
        muzzle = rot @ self._muzzle_g
        tof = 0.0
        for _ in range(4):
            p = clay((t_exit - T0) * 1e-9 + tof)
            tof = float(self.shot.time_of_flight(float(np.linalg.norm(p - muzzle))))
        to_clay = clay((t_exit - T0) * 1e-9 + tof) - muzzle
        dist = float(np.linalg.norm(to_clay))
        to_shot = bore * dist - float(self.shot.drop(dist)) * np.array([0.0, 0.0, 1.0])
        return math.atan2(float(np.linalg.norm(np.cross(to_clay, to_shot))), float(to_clay @ to_shot))

    def ideal_aim(self, clay, decide_ns: int, delay_s: float) -> tuple[float, float]:
        """Bore (az, el) at shot exit that hits the clay for a command issued at decide_ns."""
        az = el = 0.0
        rot = _frame_from_azel(0.0, 0.0)
        t_exit = int(decide_ns) + int(round(delay_s * 1e9))
        for _ in range(6):
            muzzle = rot @ self._muzzle_g
            tof = 0.0
            for _ in range(4):
                p = clay((t_exit - T0) * 1e-9 + tof)
                tof = float(self.shot.time_of_flight(float(np.linalg.norm(p - muzzle))))
            d = clay((t_exit - T0) * 1e-9 + tof) - muzzle
            d = d + float(self.shot.drop(float(np.linalg.norm(d)))) * np.array([0.0, 0.0, 1.0])
            d /= np.linalg.norm(d)
            az, el = math.atan2(d[1], d[0]), math.asin(d[2])
            rot = _frame_from_azel(az, el)
        return az, el


def _slow_crosser(t_s: float) -> np.ndarray:
    """Distant clay crossing left at 3.5 m/s, ~36 m out, 5 m up (angular rate ~0.1 rad/s)."""
    return np.array([35.0, -7.0 + 3.5 * t_s, 5.0])


def _fast_crosser(t_s: float) -> np.ndarray:
    return np.array([20.0, -12.0 + 20.0 * t_s, 3.0])


def _run(cfg, scene, att, clay, n_frames, armed=True, seed=3, sigma_px=0.1):
    tracker = ClayTracker(cfg, scene.cam, scene.mount)
    solver = FireSolver(cfg, scene.shot, scene.mount, scene.latency)
    rng = np.random.default_rng(seed)
    decisions = []
    for n in range(n_frames):
        t = T0 + n * DT_NS
        rot = att.R_wg(t)
        det = scene.detect(clay(n / FPS), rot, t, rng, sigma_px)
        tracker.update(t, [det] if det is not None else [], rot)
        decisions.append(solver.evaluate(t, tracker, att, armed, 1.0 / FPS))
    return decisions


def _first_fire(decisions):
    return next((n for n, d in enumerate(decisions) if d.fire), None)


@pytest.fixture()
def scene_cfg():
    cfg = load_config(sets=SCENE_SETS + ["actuator.type=servo"])
    return cfg, Scene(cfg)


# ------------------------------------------------------------------ hit probability
def test_hit_probability_matches_the_rayleigh_closed_form():
    for ratio in (0.1, 0.5, 1.0, 2.0, 5.0):
        sigma = 1.0
        assert _hit_probability(0.0, ratio * sigma, sigma) == pytest.approx(
            1.0 - math.exp(-0.5 * ratio**2), abs=1e-3
        )
    # Degenerate aim error: a hit is certain inside the pattern and impossible outside.
    assert _hit_probability(0.5, 1.0, 0.0) == 1.0
    assert _hit_probability(1.5, 1.0, 0.0) == 0.0
    probs = [_hit_probability(m, 1.0, 0.4) for m in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(probs, probs[1:])), "p_hit must fall as the miss grows"
    assert probs[0] > 0.9 and probs[-1] < 0.01


# ------------------------------------------------------------------ decisions
def test_leading_swing_fires_at_the_analytically_best_frame(scene_cfg):
    cfg, scene = scene_cfg
    delay = scene.latency.fire_delay_s("servo")
    rate = 0.5
    best_frame = 70
    az_i, el_i = scene.ideal_aim(_slow_crosser, T0 + best_frame * DT_NS, delay)
    att = SwingAttitude(az_i - rate * (best_frame / FPS + delay), el_i, rate)

    decisions = _run(cfg, scene, att, _slow_crosser, 160)
    truth = [scene.truth_miss(att, _slow_crosser, T0 + n * DT_NS, delay) for n in range(160)]
    assert int(np.argmin(truth)) == best_frame  # scene really is best at best_frame

    fired = _first_fire(decisions)
    assert fired is not None, [d.reason for d in decisions[best_frame - 2:best_frame + 3]]
    assert abs(fired - best_frame) <= 1
    d = decisions[fired]
    assert d.reason == "FIRE"
    assert d.p_hit >= cfg.fire.min_hit_probability
    assert d.miss_rad < d.pattern_rad
    assert d.range_m == pytest.approx(np.linalg.norm(_slow_crosser(fired / FPS)), rel=0.1)
    assert d.tof_s == pytest.approx(float(scene.shot.time_of_flight(d.range_m)), rel=0.1)
    assert d.t_exit_ns == T0 + fired * DT_NS + int(round(delay * 1e9))
    # The bore must be aimed ahead of where the clay is now: that is the whole point of the lead.
    now_dir = _slow_crosser(fired / FPS) / np.linalg.norm(_slow_crosser(fired / FPS))
    assert float(np.dot(d.lead_dir_w, now_dir)) < 1.0
    assert np.linalg.norm(d.bore_dir_w) == pytest.approx(1.0)
    # Only one frame is "the best moment"; the horizon rule must hold the shot before it.
    assert all("better in" in d2.reason for d2 in decisions[fired - 3:fired])


def test_no_lead_pursuit_never_fires(scene_cfg):
    cfg, scene = scene_cfg
    att = PursuitAttitude(_fast_crosser)
    decisions = _run(cfg, scene, att, _fast_crosser, 120)
    assert not any(d.fire for d in decisions)
    live = [d for d in decisions if math.isfinite(d.miss_rad)]
    assert live, "the clay must be tracked"
    assert all(d.miss_rad > d.pattern_rad for d in live)
    assert any(d.reason.startswith("p_hit") for d in live)
    assert max(d.p_hit for d in live) < 0.5 * cfg.fire.min_hit_probability


def test_disarmed_run_reports_the_interlock_but_fills_the_geometry(scene_cfg):
    cfg, scene = scene_cfg
    delay = scene.latency.fire_delay_s("servo")
    rate = 0.5
    best_frame = 70
    az_i, el_i = scene.ideal_aim(_slow_crosser, T0 + best_frame * DT_NS, delay)
    att = SwingAttitude(az_i - rate * (best_frame / FPS + delay), el_i, rate)

    armed = _run(cfg, scene, att, _slow_crosser, 160)
    disarmed = _run(cfg, scene, att, _slow_crosser, 160, armed=False)
    fired = _first_fire(armed)
    assert fired is not None
    assert not any(d.fire for d in disarmed)
    assert disarmed[fired].reason == "would fire (not armed)"
    assert all("not armed" in d.reason for d in disarmed)
    for attr in ("miss_rad", "pattern_rad", "sigma_rad", "p_hit", "range_m", "tof_s"):
        assert math.isfinite(getattr(disarmed[fired], attr))
    assert disarmed[fired].lead_dir_w is not None and disarmed[fired].bore_dir_w is not None
    assert disarmed[fired].p_hit == pytest.approx(armed[fired].p_hit)


def test_actuator_delay_changes_the_fire_time(scene_cfg):
    _, scene = scene_cfg
    # Both actuator budgets, one scene; the slower actuator must commit much earlier in the swing.
    sets = ["tracker.jerk_psd_deg=0.2", "tracker.meas_sigma_px=0.1"]
    cfg_servo = load_config(sets=sets + ["actuator.type=servo"])
    cfg_pq12 = load_config(sets=sets + ["actuator.type=pq12"])
    scene = Scene(cfg_servo)
    delay_servo = scene.latency.fire_delay_s("servo")
    delay_pq12 = scene.latency.fire_delay_s("pq12")
    assert delay_pq12 > delay_servo + 0.5

    rate = 0.25
    best_frame = 150
    az_i, el_i = scene.ideal_aim(_slow_crosser, T0 + best_frame * DT_NS, delay_pq12)
    att = SwingAttitude(az_i - rate * (best_frame / FPS + delay_pq12), el_i, rate)

    fire_pq12 = _first_fire(_run(cfg_pq12, scene, att, _slow_crosser, 320))
    fire_servo = _first_fire(_run(cfg_servo, scene, att, _slow_crosser, 320))
    assert fire_pq12 is not None and fire_servo is not None
    assert abs(fire_pq12 - best_frame) <= 2
    # The gun keeps swinging, so the faster actuator must wait: ~(delay_pq12 - delay_servo) later.
    assert fire_servo > fire_pq12
    assert (fire_servo - fire_pq12) / FPS == pytest.approx(delay_pq12 - delay_servo, rel=0.25)


def test_incoming_clay_with_the_bore_on_it_fires(scene_cfg):
    cfg, scene = scene_cfg
    u = np.array([1.0, 0.06, 0.14])
    u /= np.linalg.norm(u)
    clay = lambda t_s: u * (28.0 - 12.0 * t_s)  # noqa: E731 - straight in: zero angular velocity
    delay = scene.latency.fire_delay_s("servo")
    best_frame = 90
    az_i, el_i = scene.ideal_aim(clay, T0 + best_frame * DT_NS, delay)
    att = SwingAttitude(az_i, el_i, 0.0)

    decisions = _run(cfg, scene, att, clay, 140)
    fired = _first_fire(decisions)
    assert fired is not None, sorted({d.reason for d in decisions})
    # There is no lead to wait for, so the solver fires as soon as the filter has settled enough for
    # P(hit) to clear the gate; before that the young track's own uncertainty holds the shot.
    assert fired <= 20
    d = decisions[fired]
    assert d.extra["clay_rate_rad_s"] < 0.02, "a radially incoming clay has no angular velocity"
    assert d.p_hit >= cfg.fire.min_hit_probability
    assert d.miss_rad < d.pattern_rad
    assert all(x.reason.startswith("p_hit") for x in decisions[cfg.fire.min_track_hits:fired])
    mature = decisions[fired + 20]
    assert mature.p_hit > 0.99 and mature.miss_rad < 0.3 * mature.pattern_rad
    assert mature.sigma_rad < 0.25 * mature.pattern_rad


def test_range_outside_the_window_never_fires(scene_cfg):
    cfg, scene = scene_cfg
    u = np.array([1.0, 0.05, 0.12])
    u /= np.linalg.norm(u)
    clay = lambda t_s: u * (7.0 - 2.0 * t_s)  # noqa: E731 - closer than fire.min_range_m
    delay = scene.latency.fire_delay_s("servo")
    az_i, el_i = scene.ideal_aim(clay, T0 + 80 * DT_NS, delay)
    att = SwingAttitude(az_i, el_i, 0.0)

    decisions = _run(cfg, scene, att, clay, 120)
    assert not any(d.fire for d in decisions)
    tail = decisions[60:]
    assert all(d.reason == "range out of window" for d in tail)
    assert all(d.range_m < cfg.fire.min_range_m for d in tail)
    # ... and the same geometry inside the window does fire, so the gate is what blocked it.
    clay_far = lambda t_s: u * (22.0 - 2.0 * t_s)  # noqa: E731
    az_i, el_i = scene.ideal_aim(clay_far, T0 + 80 * DT_NS, delay)
    assert any(d.fire for d in _run(cfg, scene, SwingAttitude(az_i, el_i, 0.0), clay_far, 120))


def test_no_confirmed_track_and_unknown_actuator(scene_cfg):
    cfg, scene = scene_cfg
    tracker = ClayTracker(cfg, scene.cam, scene.mount)
    solver = FireSolver(cfg, scene.shot, scene.mount, scene.latency)
    att = SwingAttitude(0.0, 0.0, 0.0)
    d = solver.evaluate(T0, tracker, att, True, 1.0 / FPS)
    assert d.fire is False and d.reason == "no confirmed track"
    assert d.track_id is None and math.isnan(d.p_hit)
    assert "not armed" in solver.evaluate(T0, tracker, att, False, 1.0 / FPS).reason

    bad = load_config(sets=SCENE_SETS + ["actuator.type=hydraulic"])
    with pytest.raises(ValueError, match="hydraulic"):
        FireSolver(bad, scene.shot, scene.mount, scene.latency)


class AcceleratingSwing:
    """Gun yawing about world +Z with constant angular acceleration (a shooter speeding up the swing)."""

    def __init__(self, az0: float, el0: float, rate0: float, accel: float, with_alpha: bool = True):
        self.az0, self.el0, self.rate0, self.accel = az0, el0, rate0, accel
        self.ready = True
        if not with_alpha:
            self.alpha_g = None  # an attitude source without the acceleration estimate

    def _rate(self, t_ns: int) -> float:
        return self.rate0 + self.accel * (int(t_ns) - T0) * 1e-9

    def R_wg(self, t_ns: int) -> np.ndarray:
        t = (int(t_ns) - T0) * 1e-9
        return _frame_from_azel(self.az0 + self.rate0 * t + 0.5 * self.accel * t * t, self.el0)

    def omega_g(self, t_ns: int) -> np.ndarray:
        return self.R_wg(t_ns).T @ np.array([0.0, 0.0, self._rate(t_ns)])

    def alpha_g(self, t_ns: int) -> np.ndarray:
        return self.R_wg(t_ns).T @ np.array([0.0, 0.0, self.accel])


def test_solver_extrapolates_the_swing_with_its_angular_acceleration(scene_cfg):
    """A brisk swing (2 rad/s, 9 rad/s^2): with alpha the predicted bore at shot exit matches the truth; a
    constant-rate extrapolation is off by ~alpha*dt^2/2, and the solver's sigma carries part of that."""
    cfg, scene = scene_cfg
    delay = scene.latency.fire_delay_s("servo")
    t = T0 + 40 * DT_NS
    bore_g = np.asarray(scene.mount.bore_dir_g, dtype=float)

    def bore_err_and_sigma(att):
        tracker = ClayTracker(cfg, scene.cam, scene.mount)
        solver = FireSolver(cfg, scene.shot, scene.mount, scene.latency)
        rng = np.random.default_rng(1)
        d = None
        for n in range(41):
            tn = T0 + n * DT_NS
            rot = att.R_wg(tn)
            det = scene.detect(_fast_crosser(n / FPS), rot, tn, rng, 0.1)
            tracker.update(tn, [det] if det is not None else [], rot)
            d = solver.evaluate(tn, tracker, att, True, 1.0 / FPS)
        truth = att.R_wg(d.t_exit_ns) @ bore_g
        truth /= np.linalg.norm(truth)
        return math.degrees(math.acos(min(1.0, float(np.dot(truth, d.bore_dir_w))))), d.sigma_rad

    err_alpha, sig_alpha = bore_err_and_sigma(AcceleratingSwing(-0.6, 0.15, 2.0, 9.0))
    err_const, _ = bore_err_and_sigma(AcceleratingSwing(-0.6, 0.15, 2.0, 9.0, with_alpha=False))
    expect_const = math.degrees(0.5 * 9.0 * delay * delay)
    assert err_const == pytest.approx(expect_const, rel=0.15)
    assert err_alpha < 0.05 * expect_const
    assert sig_alpha >= cfg.fire.gun_accel_sigma_frac * math.radians(expect_const) * 0.99
    assert t  # keeps the frame index explicit
