"""Exterior ballistics of one shot column (contract: docs/ARCHITECTURE.md, "[optics]").

Point-mass model of a single lead pellet: gravity is treated separately from the drag
deceleration along the flight path, which is what the fire solver needs (time of flight,
remaining velocity, drop, pattern spread). Everything is precomputed once per Config into
distance-indexed tables, so the per-frame solver only interpolates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

__all__ = ["ShotModel", "sphere_cd", "G_MPS2"]

G_MPS2 = 9.80665  # CODATA standard gravity

# Drag coefficient of a smooth sphere vs Mach number at the Reynolds numbers of shot in flight
# (d = 2.3 mm, v = 100-370 m/s => Re = 1.5e4 .. 6e4, i.e. subcritical: Cd ~ 0.47-0.50 at low
# Mach, rising through the transonic peak). Sources: S. F. Hoerner, "Fluid-Dynamic Drag" (1965)
# ch. 6 fig. 20; A. B. Bailey & J. Hiatt, "Sphere drag coefficients for a broad range of Mach and
# Reynolds numbers", AIAA J. 10(11) 1436-1440 (1972); A. B. Bailey & R. F. Starr / Miller &
# Bailey, J. Fluid Mech. 93:449-464 (1979). Values are the subcritical branch of that data.
_CD_MACH = np.array(
    [0.0, 0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40, 1.60, 2.00],
    dtype=np.float64,
)
_CD_VALUE = np.array(
    [0.47, 0.47, 0.48, 0.49, 0.51, 0.56, 0.62, 0.72, 0.83, 0.92, 0.96, 0.98, 1.00, 0.99, 0.97, 0.94],
    dtype=np.float64,
)

_STEP_M = 0.02  # integration step along the flight path
_MIN_STEPS = 64


def sphere_cd(mach):
    """Drag coefficient of a sphere at Mach number(s) `mach` (subcritical Reynolds branch)."""
    m = np.asarray(mach, dtype=np.float64)
    out = np.interp(np.abs(m), _CD_MACH, _CD_VALUE)  # constant outside the tabulated range
    return float(out) if out.ndim == 0 else out


class ShotModel:
    """Point-mass lead-sphere ballistics for cfg.shot, tabulated to cfg.shot.max_range_m.

    Ranges outside [0, max_range_m] are clamped to that interval (documented policy: the solver
    gates on cfg.fire.min/max_range_m, and a clamped table edge is a safer answer for a stray
    range estimate than an exception in the per-frame path).
    """

    __slots__ = (
        "muzzle_velocity_mps",
        "max_range_m",
        "pellet_mass_kg",
        "pellet_area_m2",
        "drag_factor",
        "_x",
        "_t",
        "_v",
        "_pattern_ref_m",
        "_pattern_ref_range_m",
        "_pattern_min_m",
        "_speed_of_sound",
    )

    def __init__(self, cfg: "Config") -> None:
        s = cfg.shot
        v0 = float(s.muzzle_velocity_mps)
        d = float(s.pellet_diameter_mm) * 1e-3
        rho_p = float(s.pellet_density_kg_m3)
        rho_air = float(s.air_density_kg_m3)
        a_sound = float(s.speed_of_sound_mps)
        x_max = float(s.max_range_m)
        if v0 <= 0.0:
            raise ValueError(f"shot.muzzle_velocity_mps must be > 0, got {v0}")
        if d <= 0.0 or rho_p <= 0.0:
            raise ValueError("shot.pellet_diameter_mm and pellet_density_kg_m3 must be > 0")
        if rho_air < 0.0 or a_sound <= 0.0:
            raise ValueError("shot.air_density_kg_m3 must be >= 0 and speed_of_sound_mps > 0")
        if x_max <= 0.0:
            raise ValueError(f"shot.max_range_m must be > 0, got {x_max}")
        if float(s.pattern_ref_range_m) <= 0.0:
            raise ValueError("shot.pattern_ref_range_m must be > 0")

        self.muzzle_velocity_mps = v0
        self.max_range_m = x_max
        self._speed_of_sound = a_sound
        self.pellet_mass_kg = rho_p * (np.pi / 6.0) * d**3
        self.pellet_area_m2 = np.pi * 0.25 * d * d
        # dv/dx = -(0.5 rho A / m) Cd(M) v  =>  drag_factor has units 1/m
        self.drag_factor = 0.5 * rho_air * self.pellet_area_m2 / self.pellet_mass_kg

        self._pattern_ref_m = float(s.pattern_radius_ref_m)
        self._pattern_ref_range_m = float(s.pattern_ref_range_m)
        self._pattern_min_m = float(s.pattern_min_radius_m)
        self._x, self._t, self._v = self._integrate(x_max)

    # ---- integration ----------------------------------------------------------------------
    def _dv_dx(self, v: float) -> float:
        cd = float(np.interp(abs(v) / self._speed_of_sound, _CD_MACH, _CD_VALUE))
        return -self.drag_factor * cd * v

    def _integrate(self, x_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RK4 in distance over the state (t, v): dt/dx = 1/v, dv/dx = -k Cd(M) v."""
        n = max(_MIN_STEPS, int(np.ceil(x_max / _STEP_M)))
        h = x_max / n
        xs = np.linspace(0.0, x_max, n + 1)
        ts = np.empty(n + 1, dtype=np.float64)
        vs = np.empty(n + 1, dtype=np.float64)
        t = 0.0
        v = self.muzzle_velocity_mps
        ts[0] = t
        vs[0] = v
        v_floor = 1e-3  # the pellet never actually stops within max_range_m; guards 1/v
        for i in range(n):
            k1_t, k1_v = 1.0 / v, self._dv_dx(v)
            v2 = max(v + 0.5 * h * k1_v, v_floor)
            k2_t, k2_v = 1.0 / v2, self._dv_dx(v2)
            v3 = max(v + 0.5 * h * k2_v, v_floor)
            k3_t, k3_v = 1.0 / v3, self._dv_dx(v3)
            v4 = max(v + h * k3_v, v_floor)
            k4_t, k4_v = 1.0 / v4, self._dv_dx(v4)
            t += (h / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
            v = max(v + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v), v_floor)
            ts[i + 1] = t
            vs[i + 1] = v
        return xs, ts, vs

    def _lookup(self, range_m, table: np.ndarray):
        r = np.asarray(range_m, dtype=np.float64)
        out = np.interp(np.clip(r, 0.0, self.max_range_m), self._x, table)
        return float(out) if r.ndim == 0 else out

    # ---- public API -----------------------------------------------------------------------
    def time_of_flight(self, range_m) -> float | np.ndarray:
        """Seconds from the muzzle to `range_m` down the shot line."""
        return self._lookup(range_m, self._t)

    def velocity(self, range_m) -> float | np.ndarray:
        """Remaining pellet speed (m/s) at `range_m`."""
        return self._lookup(range_m, self._v)

    def drop(self, range_m, elevation_rad: float = 0.0) -> float | np.ndarray:
        """Gravity drop (m) below the bore line at `range_m` fired at `elevation_rad`."""
        t = np.asarray(self.time_of_flight(range_m), dtype=np.float64)
        d = 0.5 * G_MPS2 * t * t * np.cos(np.asarray(elevation_rad, dtype=np.float64))
        return float(d) if d.ndim == 0 else d

    def pattern_radius(self, range_m) -> float | np.ndarray:
        """Effective pattern core radius (m): linear cone with a near-muzzle floor."""
        r = np.asarray(range_m, dtype=np.float64)
        out = np.maximum(
            self._pattern_min_m, self._pattern_ref_m * r / self._pattern_ref_range_m
        )
        return float(out) if r.ndim == 0 else out
