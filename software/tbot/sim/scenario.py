"""Synthetic clay-target scenarios: clay flight physics + simulated shooter swing [sim slice].

World frame W (docs/ARCHITECTURE.md): gravity aligned, +Z up, the gun's rotation centre (= the camera) at
the origin, +X = the horizontal heading the gun holds at t = 0 (the gun only rotates; translation of the
shoulder is ignored). Clay positions are therefore world-fixed positions relative to the camera.

Scenario geometry is written in a "field" frame (+X downrange toward the crossing/trap, +Y left, +Z up,
origin at the camera) and rotated about Z so that the scenario's hold point lies on +X.

Field dimensions are approximations from the published layouts, not measurements:
- Skeet (NSSA/ISSF): high and low house windows 40.2 m apart, window heights 3.05 m / 1.07 m above ground,
  the crossing point ~4.5 m above ground over the centre stake, ~19-21 m from station 4; targets leave the
  house at 20-22 m/s and must carry ~55 m.
- Trap (ATA): the house is 15 m (16 yd) in front of the station, targets leave at 25-27 m/s within 20 deg
  of straight away and pass 2.4-3.4 m above ground 10 yd beyond the house, carrying 48-52 m.
Camera/shoulder height above ground is taken as 1.5 m, so field heights are stored minus 1.5 m. Every
number marked [INFERENCE] is an estimate, to be replaced when a measurement exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tbot.config import Config

__all__ = ["Scenario", "SCENARIOS", "make_scenario"]

G_MPS2 = 9.80665
AIR_DENSITY_KG_M3 = 1.225

# Clay target: 108 mm x ~25 mm, 105 g (ISSF/ATA standard). The clay flies like a frisbee: its spin axis
# stays put (gyroscopic), it carries a small angle of attack and therefore some lift.
CLAY_MASS_KG = 0.105
CLAY_THICKNESS_M = 0.025  # dome height of a standard clay [INFERENCE +/- 5 mm]
CLAY_DRAG_AREA_M2 = 1.1e-3  # Cd*A [INFERENCE]: with the lift below a 21 m/s skeet target carries ~55 m
CLAY_LIFT_AREA_M2 = 3.6e-3  # Cl*A [INFERENCE]: a clay GLIDES — lift is ~0.95 g at 21 m/s and falls off
# with v^2, which is why a 21 m/s target carries 55 m when a drag-only projectile would manage ~25 m.
CLAY_BANK_DEG = 18.0  # disc banked about its velocity at launch [INFERENCE] (machines throw banked)

SHOULDER_HEIGHT_M = 1.5  # camera height above ground, mounted on a shouldered gun [INFERENCE]

T_LAUNCH_S = 0.3  # the clay appears 0.3 s into the scenario
DURATION_S = 2.5
MAX_FLIGHT_S = 3.0  # the clay is dropped from the truth after this long, even if still above the horizon

_DT_S = 0.002  # integration/knot step of the trajectory and of the swing
_SOLVE_DT_S = 0.01  # coarser step for the launch-angle solve
_OU_DT_S = 0.05  # Ornstein-Uhlenbeck wobble knot step (band limits the hand tremor to ~10 Hz)
_OU_TAU_S = 0.35  # wobble correlation time [INFERENCE]
_REACTION_S = 0.18  # shooter reaction before the swing starts [INFERENCE]
_RAMP_S = 0.12  # smooth build-up of the pursuit gain (keeps omega continuous at launch)
# Swing-through technique: the shooter starts behind the clay, catches it and pulls ahead, so the bore
# sweeps through the firing solution instead of trailing it for ever. The aim point is where the clay
# will be _LEAD_RATE seconds-per-second further into its flight, capped at _MAX_LEAD_S. [INFERENCE]
_LEAD_RATE = 0.50
_MAX_LEAD_S = 0.45
_MAX_EL_RAD = math.radians(85.0)  # the simulated shooter never points straight up


# ------------------------------------------------------------------------------------- interpolation
class _Curve:
    """C1 cubic Hermite curve on a uniform grid.

    value() and deriv() are exactly consistent (deriv is the analytic derivative of value), which is what
    makes the synthetic gyro consistent with R_wg. Outside the grid the value is held and deriv is zero.
    """

    __slots__ = ("t0", "dt", "t_end", "y", "dy", "n", "_vec")

    def __init__(self, t0: float, dt: float, y: np.ndarray, dy: np.ndarray):
        self.y = np.asarray(y, dtype=np.float64)
        self.dy = np.asarray(dy, dtype=np.float64)
        if self.y.shape != self.dy.shape or self.y.shape[0] < 2:
            raise ValueError("curve needs matching value/derivative arrays with >= 2 knots")
        self.t0 = float(t0)
        self.dt = float(dt)
        self.n = int(self.y.shape[0])
        self.t_end = self.t0 + (self.n - 1) * self.dt
        self._vec = self.y.ndim > 1

    def _eval(self, t, want_deriv: bool):
        tt = np.asarray(t, dtype=np.float64)
        scalar = tt.ndim == 0
        tc = np.clip(tt, self.t0, self.t_end)
        s = (tc - self.t0) / self.dt
        i = np.clip(np.floor(s).astype(np.int64), 0, self.n - 2)
        u = s - i
        y0 = self.y[i]
        y1 = self.y[i + 1]
        d0 = self.dy[i]
        d1 = self.dy[i + 1]
        if self._vec:
            u = u[..., None] if u.ndim else u
        u2 = u * u
        u3 = u2 * u
        if want_deriv:
            out = (
                (6.0 * u2 - 6.0 * u) * (y0 - y1) / self.dt
                + (3.0 * u2 - 4.0 * u + 1.0) * d0
                + (3.0 * u2 - 2.0 * u) * d1
            )
            inside = (tt >= self.t0) & (tt <= self.t_end)
            out = np.where(inside[..., None] if self._vec and np.ndim(inside) else inside, out, 0.0)
        else:
            out = (
                (2.0 * u3 - 3.0 * u2 + 1.0) * y0
                + (u3 - 2.0 * u2 + u) * (self.dt * d0)
                + (-2.0 * u3 + 3.0 * u2) * y1
                + (u3 - u2) * (self.dt * d1)
            )
        if scalar:
            return float(out) if not self._vec else np.asarray(out, dtype=np.float64)
        return np.asarray(out, dtype=np.float64)

    def value(self, t):
        return self._eval(t, False)

    def deriv(self, t):
        return self._eval(t, True)


# ------------------------------------------------------------------------------------- clay physics
def _clay_accel(vel: np.ndarray) -> np.ndarray:
    """Gravity + quadratic drag + frisbee-like lift, m/s^2."""
    speed = float(math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]))
    a = np.array([0.0, 0.0, -G_MPS2])
    if speed < 1e-6:
        return a
    vhat = vel / speed
    q = 0.5 * AIR_DENSITY_KG_M3 * speed * speed / CLAY_MASS_KG  # dynamic pressure / mass
    a -= (q * CLAY_DRAG_AREA_M2) * vhat
    up = np.array([0.0, 0.0, 1.0]) - vhat[2] * vhat  # "up" component perpendicular to the velocity
    n = float(math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]))
    if n > 1e-6:
        a += (q * CLAY_LIFT_AREA_M2 / n) * up
    return a


def _integrate(pos0: np.ndarray, vel0: np.ndarray, n_steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """RK4 integration of the clay state; returns (n_steps+1, 3) positions and velocities."""
    pos = np.empty((n_steps + 1, 3), dtype=np.float64)
    vel = np.empty_like(pos)
    p = np.array(pos0, dtype=np.float64)
    v = np.array(vel0, dtype=np.float64)
    pos[0] = p
    vel[0] = v
    h = dt / 6.0
    for i in range(n_steps):
        a1 = _clay_accel(v)
        v2 = v + (0.5 * dt) * a1
        a2 = _clay_accel(v2)
        v3 = v + (0.5 * dt) * a2
        a3 = _clay_accel(v3)
        v4 = v + dt * a3
        a4 = _clay_accel(v4)
        p = p + h * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v = v + h * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        pos[i + 1] = p
        vel[i + 1] = v
    return pos, vel


def _height_at_gate(pos0, hdir, speed, elev, gate_dist, max_t):
    """Height of the clay when it has travelled gate_dist along hdir (-inf if it never gets there)."""
    vel0 = speed * (math.cos(elev) * hdir + np.array([0.0, 0.0, math.sin(elev)]))
    n = int(round(max_t / _SOLVE_DT_S))
    p = np.array(pos0, dtype=np.float64)
    v = vel0.copy()
    s_prev = 0.0
    z_prev = float(p[2])
    for _ in range(n):
        a1 = _clay_accel(v)
        v2 = v + (0.5 * _SOLVE_DT_S) * a1
        a2 = _clay_accel(v2)
        v3 = v + (0.5 * _SOLVE_DT_S) * a2
        a3 = _clay_accel(v3)
        v4 = v + _SOLVE_DT_S * a3
        a4 = _clay_accel(v4)
        p = p + (_SOLVE_DT_S / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v = v + (_SOLVE_DT_S / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        s = float((p[:2] - np.asarray(pos0)[:2]) @ hdir[:2])
        if s >= gate_dist:
            f = (gate_dist - s_prev) / max(s - s_prev, 1e-9)
            return z_prev + f * (float(p[2]) - z_prev)
        s_prev = s
        z_prev = float(p[2])
    return -math.inf


def _launch_velocity(pos0: np.ndarray, hdir: np.ndarray, speed: float, gate_dist: float,
                     gate_z: float) -> np.ndarray:
    """Launch velocity whose trajectory passes gate_z when it has flown gate_dist along hdir."""
    lo, hi = math.radians(-15.0), math.radians(45.0)
    f_lo = _height_at_gate(pos0, hdir, speed, lo, gate_dist, MAX_FLIGHT_S) - gate_z
    f_hi = _height_at_gate(pos0, hdir, speed, hi, gate_dist, MAX_FLIGHT_S) - gate_z
    if not (f_lo < 0.0 < f_hi):
        raise ValueError(f"launch gate {gate_z:.2f} m at {gate_dist:.1f} m is unreachable at {speed} m/s")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if hi - lo < 1e-5:
            break
        if _height_at_gate(pos0, hdir, speed, mid, gate_dist, MAX_FLIGHT_S) - gate_z < 0.0:
            lo = mid
        else:
            hi = mid
    elev = 0.5 * (lo + hi)
    return speed * (math.cos(elev) * hdir + np.array([0.0, 0.0, math.sin(elev)]))


def _disc_axis(vel0: np.ndarray) -> np.ndarray:
    """Clay disc normal: perpendicular to the launch velocity, banked about it; constant in flight."""
    vhat = vel0 / np.linalg.norm(vel0)
    up = np.array([0.0, 0.0, 1.0]) - vhat[2] * vhat
    n = np.linalg.norm(up)
    up = up / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    b = math.radians(CLAY_BANK_DEG)
    axis = math.cos(b) * up + math.sin(b) * np.cross(vhat, up)
    return axis / np.linalg.norm(axis)


# ------------------------------------------------------------------------------------- gun swing
def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _ou_curve(duration_s: float, sigma: float, rng: np.random.Generator) -> _Curve:
    """Ornstein-Uhlenbeck process sampled at _OU_DT_S, smoothly (C1) interpolated."""
    m = int(math.ceil(duration_s / _OU_DT_S)) + 2
    x = np.empty(m, dtype=np.float64)
    a = math.exp(-_OU_DT_S / _OU_TAU_S)
    s = sigma * math.sqrt(max(1.0 - a * a, 0.0))
    x[0] = sigma * float(rng.standard_normal())
    for i in range(1, m):
        x[i] = a * x[i - 1] + s * float(rng.standard_normal())
    dx = np.gradient(x, _OU_DT_S) if sigma > 0.0 else np.zeros_like(x)
    return _Curve(0.0, _OU_DT_S, x, dx)


def _swing_curves(duration_s, launch_t_s, hold_az, hold_el, clay: _Curve | None, lag_s: float,
                  noise_rad: float, static: bool, rng: np.random.Generator) -> tuple[_Curve, _Curve]:
    """Bore azimuth/elevation of the simulated shooter: hold, then first-order pursuit plus OU wobble."""
    n = int(round(duration_s / _DT_S)) + 1
    t = np.arange(n, dtype=np.float64) * _DT_S
    if static:
        zeros = np.zeros(n, dtype=np.float64)
        return (_Curve(0.0, _DT_S, np.full(n, hold_az), zeros),
                _Curve(0.0, _DT_S, np.full(n, hold_el), zeros))

    wob_az = _ou_curve(duration_s, noise_rad, rng)
    wob_el = _ou_curve(duration_s, noise_rad, rng)
    w_az, dw_az = wob_az.value(t), wob_az.deriv(t)
    w_el, dw_el = wob_el.value(t), wob_el.deriv(t)

    az_t = np.full(n, hold_az, dtype=np.float64)
    el_t = np.full(n, hold_el, dtype=np.float64)
    gain = np.zeros(n, dtype=np.float64)
    if clay is not None:
        # Swing-through: the aim point is the clay's position tau(t) seconds further into its flight,
        # tau ramping from -swing_lag_s (bore trailing the clay out of the hold point) up to
        # _MAX_LEAD_S, so the bore overtakes the clay and builds lead through the firing solution.
        # lag_s is added back because the first-order pursuit delays the bore by that much: a trained
        # shooter's bore rides the clay's track, it does not cut the corner of it.
        tau = np.clip(-lag_s + (t - launch_t_s - _REACTION_S) * _LEAD_RATE, -lag_s, _MAX_LEAD_S)
        p = clay.value(t + tau + lag_s)  # (n, 3): held at the launch point before launch
        r = np.linalg.norm(p, axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        a = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
        k0 = int(round(launch_t_s / _DT_S))
        a = a + 2.0 * math.pi * round((hold_az - a[min(k0, n - 1)]) / (2.0 * math.pi))
        az_t = a
        el_t = np.arcsin(np.clip(p[:, 2] / r, -1.0, 1.0))
        s = np.clip((t - launch_t_s - _REACTION_S) / _RAMP_S, 0.0, 1.0)
        gain = (s * s * (3.0 - 2.0 * s)) / max(lag_s, 1e-3)

    y_az = np.empty(n, dtype=np.float64)
    d_az = np.empty(n, dtype=np.float64)
    y_el = np.empty(n, dtype=np.float64)
    d_el = np.empty(n, dtype=np.float64)
    az, el = hold_az, hold_el
    for k in range(n):
        r_az = gain[k] * _wrap(az_t[k] - az)
        r_el = gain[k] * (el_t[k] - el)
        y_az[k] = az + w_az[k]
        d_az[k] = r_az + dw_az[k]
        y_el[k] = el + w_el[k]
        d_el[k] = r_el + dw_el[k]
        if k + 1 < n:  # Heun step
            az_p = az + r_az * _DT_S
            el_p = el + r_el * _DT_S
            r_az2 = gain[k + 1] * _wrap(az_t[k + 1] - az_p)
            r_el2 = gain[k + 1] * (el_t[k + 1] - el_p)
            az += 0.5 * (r_az + r_az2) * _DT_S
            el += 0.5 * (r_el + r_el2) * _DT_S
            el = min(max(el, -_MAX_EL_RAD), _MAX_EL_RAD)
    return _Curve(0.0, _DT_S, y_az, d_az), _Curve(0.0, _DT_S, y_el, d_el)


# ------------------------------------------------------------------------------------- scenario
@dataclass
class Scenario:
    """Ground truth of one synthetic engagement: clay flight and gun attitude over [0, duration_s]."""

    name: str
    description: str
    duration_s: float
    fps: float
    launch_t_s: float
    seed: int
    clay_present: bool
    visible_until_s: float
    clay_diameter_m: float
    clay_thickness_m: float
    clay_axis_w: np.ndarray
    launch_pos_w: np.ndarray
    launch_vel_w: np.ndarray
    hold_el_rad: float
    static_gun: bool
    swing_az: _Curve = field(repr=False)
    swing_el: _Curve = field(repr=False)
    clay_curve: _Curve | None = field(default=None, repr=False)

    @property
    def n_frames(self) -> int:
        return int(math.floor(self.duration_s * self.fps)) + 1

    # ---------------------------------------------------------------- clay
    def clay_pos_w(self, t_s: float) -> np.ndarray | None:
        """Clay position in W (metres, relative to the camera), or None when the scenario has no clay."""
        if self.clay_curve is None:
            return None
        return self.clay_curve.value(float(t_s))

    def clay_vel_w(self, t_s: float) -> np.ndarray | None:
        if self.clay_curve is None:
            return None
        return self.clay_curve.deriv(float(t_s))

    def clay_visible(self, t_s: float) -> bool:
        """True while the clay is airborne, above the horizon and younger than MAX_FLIGHT_S."""
        if self.clay_curve is None:
            return False
        t = float(t_s)
        if t < self.launch_t_s or t > self.launch_t_s + MAX_FLIGHT_S or t > self.clay_curve.t_end:
            return False
        return bool(self.clay_curve.value(t)[2] >= 0.0)

    def clay_range_m(self, t_s: float) -> float:
        p = self.clay_pos_w(t_s)
        return float("nan") if p is None else float(np.linalg.norm(p))

    def clay_dir_w(self, t_s: float) -> np.ndarray | None:
        p = self.clay_pos_w(t_s)
        if p is None:
            return None
        r = float(np.linalg.norm(p))
        return p / r if r > 1e-9 else np.array([1.0, 0.0, 0.0])

    # ---------------------------------------------------------------- gun
    def R_wg(self, t_s: float) -> np.ndarray:
        """Gun -> world rotation: zero roll, bore at (azimuth, elevation) of the simulated swing."""
        az = self.swing_az.value(float(t_s))
        el = self.swing_el.value(float(t_s))
        return _rot_z(az) @ _rot_y(-el)

    def omega_g(self, t_s: float) -> np.ndarray:
        """Angular velocity in the GUN frame, exactly consistent with R_wg (d/dt of the same curves)."""
        el = self.swing_el.value(float(t_s))
        d_az = self.swing_az.deriv(float(t_s))
        d_el = self.swing_el.deriv(float(t_s))
        return np.array([d_az * math.sin(el), -d_el, d_az * math.cos(el)])

    def bore_dir_w(self, t_s: float) -> np.ndarray:
        return self.R_wg(t_s)[:, 0].copy()

    # ---------------------------------------------------------------- truth
    def truth(self, t_s: float) -> dict:
        """Ground truth at scenario time t_s (clay_px is added by the synthetic camera)."""
        return {
            "clay_pos_w": self.clay_pos_w(t_s),
            "clay_dir_w": self.clay_dir_w(t_s),
            "clay_range_m": self.clay_range_m(t_s),
            "R_wg": self.R_wg(t_s),
            "visible": self.clay_visible(t_s),
        }


# Scenario specs in the FIELD frame (+X downrange, +Y left, +Z up, camera at the origin, heights already
# reduced by SHOULDER_HEIGHT_M). "clay": launch point, horizontal heading, speed and the gate the
# trajectory must pass through (distance along the heading, height) which fixes the launch elevation.
# The hold point sits BEHIND the clay's launch point along its track: the simulated shooter uses the
# swing-through technique (start behind the target, overtake it, fire while pulling ahead).
_SPECS: dict[str, dict] = {
    "skeet_crossing": {
        "description": "station 4 high-house target crossing left to right through the centre",
        "hold_point": (19.0, 24.5, 1.9),
        "clay": {"pos": (19.0, 20.1, 1.55), "heading_deg": -90.0, "speed": 21.0,
                 "gate_dist": 20.1, "gate_z": 3.0},
    },
    "skeet_incoming": {
        "description": "incoming target from the far house, descending past the shooter on his right",
        "hold_point": (34.0, 9.5, 2.9),
        "clay": {"pos": (30.0, 8.0, 2.6), "heading_deg": -157.5, "speed": 18.0,
                 "gate_dist": 12.0, "gate_z": 1.6},
    },
    "skeet_outgoing": {
        "description": "low-house target leaving beside the shooter, going away and crossing right",
        "hold_point": (5.0, 6.5, 1.4),
        "clay": {"pos": (3.0, 5.5, 1.3), "heading_deg": -23.1, "speed": 21.0,
                 "gate_dist": 15.0, "gate_z": 3.0},
    },
    "trap_straightaway": {
        "description": "16 yd trap target, 6 deg left of straight away, rising",
        "hold_point": (16.0, -0.2, 0.2),
        "clay": {"pos": (15.0, 0.0, -0.6), "heading_deg": 6.0, "speed": 26.0,
                 "gate_dist": 9.14, "gate_z": 1.4},
    },
    "static_gun": {
        "description": "gun clamped on the crossing point while a skeet crossing target flies through",
        "hold_point": (19.0, 0.0, 3.0),
        "static": True,
        "clay": {"pos": (19.0, 20.1, 1.55), "heading_deg": -90.0, "speed": 21.0,
                 "gate_dist": 20.1, "gate_z": 3.0},
    },
    "no_clay": {
        "description": "empty sky, gun held with hand tremor only (false-alarm test)",
        "hold_point": (20.0, 0.0, 1.8),
        "clay": None,
    },
}

SCENARIOS: dict[str, str] = {k: v["description"] for k, v in _SPECS.items()}


def _visible_until(pos: np.ndarray, launch_t_s: float, dt: float) -> float:
    """End of the first above-horizon stretch of the trajectory (capped at MAX_FLIGHT_S)."""
    z = pos[:, 2]
    above = np.flatnonzero(z >= 0.0)
    end = pos.shape[0] - 1
    if above.size:
        below = np.flatnonzero(z[above[0]:] < 0.0)
        if below.size:
            end = int(above[0] + below[0] - 1)
    else:
        end = 0
    return min(launch_t_s + end * dt, launch_t_s + MAX_FLIGHT_S)


def make_scenario(name: str, cfg: Config, seed: int) -> Scenario:
    """Build the named scenario (SCENARIOS) deterministically from seed."""
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown scenario {name!r} (known: {sorted(_SPECS)})")
    rng = np.random.default_rng([int(seed), 17])
    duration = DURATION_S
    launch = T_LAUNCH_S

    hold_f = np.array(spec["hold_point"], dtype=np.float64)
    az0 = math.atan2(hold_f[1], hold_f[0])
    R_fw = _rot_z(-az0)  # field -> world: the hold point ends up on +X
    hold_w = R_fw @ hold_f
    hold_el = math.atan2(hold_w[2], math.hypot(hold_w[0], hold_w[1]))

    clay_spec = spec.get("clay")
    clay_curve = None
    axis = np.array([0.0, 0.0, 1.0])
    pos_w = np.zeros(3)
    vel_w = np.zeros(3)
    visible_until = launch
    if clay_spec is not None:
        pos_f = np.array(clay_spec["pos"], dtype=np.float64)
        head = math.radians(clay_spec["heading_deg"])
        hdir = np.array([math.cos(head), math.sin(head), 0.0])
        vel_f = _launch_velocity(pos_f, hdir, float(clay_spec["speed"]),
                                 float(clay_spec["gate_dist"]), float(clay_spec["gate_z"]))
        pos_w = R_fw @ pos_f
        vel_w = R_fw @ vel_f
        flight = min(MAX_FLIGHT_S, duration - launch)
        n_steps = int(round(flight / _DT_S))
        P, V = _integrate(pos_w, vel_w, n_steps, _DT_S)
        clay_curve = _Curve(launch, _DT_S, P, V)
        axis = _disc_axis(vel_w)
        visible_until = _visible_until(P, launch, _DT_S)

    static = bool(spec.get("static", False))
    swing_az, swing_el = _swing_curves(
        duration, launch, 0.0, hold_el, clay_curve,
        float(cfg.sim.swing_lag_s), math.radians(float(cfg.sim.swing_noise_deg)), static, rng,
    )
    return Scenario(
        name=name,
        description=spec["description"],
        duration_s=duration,
        fps=float(cfg.camera.fps),
        launch_t_s=launch,
        seed=int(seed),
        clay_present=clay_curve is not None,
        visible_until_s=visible_until,
        clay_diameter_m=float(cfg.mount.clay_diameter_m),
        clay_thickness_m=CLAY_THICKNESS_M,
        clay_axis_w=axis,
        launch_pos_w=pos_w,
        launch_vel_w=vel_w,
        hold_el_rad=hold_el,
        static_gun=static,
        swing_az=swing_az,
        swing_el=swing_el,
        clay_curve=clay_curve,
    )
