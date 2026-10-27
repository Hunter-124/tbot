"""Behaviour of tbot.fire.ballistics.ShotModel (#8 lead at 366 m/s unless stated)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tbot.config import load_config
from tbot.fire.ballistics import G_MPS2, ShotModel, sphere_cd

YARD20_M = 18.288
YARD40_M = 36.576


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def shot(cfg):
    return ShotModel(cfg)


def _with_shot(cfg, **kw):
    return dataclasses.replace(cfg, shot=dataclasses.replace(cfg.shot, **kw))


# ------------------------------------------------------------------------------------ drag table
def test_sphere_cd_curve():
    assert sphere_cd(0.0) == pytest.approx(0.47, abs=0.01)
    assert sphere_cd(0.3) == pytest.approx(sphere_cd(0.1), abs=0.01)  # subsonic plateau
    assert sphere_cd(1.05) > 1.7 * sphere_cd(0.3)  # transonic peak
    assert 0.85 < sphere_cd(1.0) < 1.05
    assert sphere_cd(5.0) == sphere_cd(2.0)  # clamped outside the table
    assert sphere_cd(np.array([0.2, 1.05])).shape == (2,)


# -------------------------------------------------------------------------------- muzzle and mass
def test_muzzle_conditions(shot, cfg):
    assert shot.time_of_flight(0.0) == 0.0
    assert shot.velocity(0.0) == pytest.approx(cfg.shot.muzzle_velocity_mps)
    assert shot.drop(0.0) == 0.0
    # #8 lead pellet: published nominal mass is 0.0712 g (410 pellets per ounce)
    assert shot.pellet_mass_kg * 1e3 == pytest.approx(0.0713, abs=0.001)
    assert shot.pellet_area_m2 == pytest.approx(math.pi * (0.00229 / 2) ** 2, rel=1e-9)


# ------------------------------------------------------------------------------- flight behaviour
def test_time_of_flight_and_retained_velocity(shot):
    assert 0.055 <= shot.time_of_flight(YARD20_M) <= 0.068
    assert 0.125 <= shot.time_of_flight(YARD40_M) <= 0.160
    assert 170.0 <= shot.velocity(YARD40_M) <= 215.0
    assert shot.velocity(YARD20_M) > shot.velocity(YARD40_M)


def test_tables_are_monotonic(shot):
    r = np.linspace(0.0, shot.max_range_m, 200)
    t = shot.time_of_flight(r)
    v = shot.velocity(r)
    assert np.all(np.diff(t) > 0.0)
    assert np.all(np.diff(v) < 0.0)
    assert np.all(v > 0.0)
    # drag never lets the shot cover ground faster than the muzzle velocity would
    assert np.all(t[1:] >= r[1:] / shot.muzzle_velocity_mps)


def test_integration_matches_an_independent_euler_solution(cfg, shot):
    """Cross-check the RK4 tables against a plain fine-step Euler integration."""
    s = cfg.shot
    d = s.pellet_diameter_mm * 1e-3
    mass = s.pellet_density_kg_m3 * math.pi / 6.0 * d**3
    area = math.pi * 0.25 * d * d
    k = 0.5 * s.air_density_kg_m3 * area / mass
    n = 40000
    h = YARD40_M / n
    v = s.muzzle_velocity_mps
    t = 0.0
    for _ in range(n):
        t += h / v
        v -= h * k * sphere_cd(v / s.speed_of_sound_mps) * v
    assert shot.velocity(YARD40_M) == pytest.approx(v, rel=2e-3)
    assert shot.time_of_flight(YARD40_M) == pytest.approx(t, rel=2e-3)


def test_denser_air_and_lighter_shot_slow_down_faster(cfg, shot):
    thick = ShotModel(_with_shot(cfg, air_density_kg_m3=1.5))
    assert thick.velocity(YARD40_M) < shot.velocity(YARD40_M)
    assert thick.time_of_flight(YARD40_M) > shot.time_of_flight(YARD40_M)
    vacuum = ShotModel(_with_shot(cfg, air_density_kg_m3=0.0))
    assert vacuum.velocity(YARD40_M) == pytest.approx(cfg.shot.muzzle_velocity_mps, rel=1e-9)
    assert vacuum.time_of_flight(YARD40_M) == pytest.approx(
        YARD40_M / cfg.shot.muzzle_velocity_mps, rel=1e-6
    )
    # #4 lead (3.30 mm) has a better sectional density than #8 and carries further
    big = ShotModel(_with_shot(cfg, pellet_diameter_mm=3.30))
    assert big.velocity(YARD40_M) > shot.velocity(YARD40_M)


# ------------------------------------------------------------------------------------------ drop
def test_drop_is_free_fall_over_the_time_of_flight(shot):
    t = shot.time_of_flight(20.0)
    assert shot.drop(20.0) == pytest.approx(0.5 * 9.81 * t * t, rel=0.05)
    assert shot.drop(40.0) > shot.drop(20.0)
    # firing uphill moves part of the drop along the bore line
    assert shot.drop(20.0, math.radians(30.0)) == pytest.approx(
        0.5 * G_MPS2 * t * t * math.cos(math.radians(30.0)), rel=1e-9
    )
    assert shot.drop(20.0, 0.0) > shot.drop(20.0, math.radians(30.0))


# ---------------------------------------------------------------------------------------- pattern
def test_pattern_radius_is_a_linear_cone_with_a_floor(shot, cfg):
    s = cfg.shot
    assert shot.pattern_radius(s.pattern_ref_range_m) == pytest.approx(s.pattern_radius_ref_m)
    assert shot.pattern_radius(2 * s.pattern_ref_range_m) == pytest.approx(
        2 * s.pattern_radius_ref_m
    )
    assert shot.pattern_radius(0.0) == pytest.approx(s.pattern_min_radius_m)
    r = np.array([0.0, 1.0, 10.0, 25.0])
    assert shot.pattern_radius(r) == pytest.approx(
        np.maximum(s.pattern_min_radius_m, s.pattern_radius_ref_m * r / s.pattern_ref_range_m)
    )


# ------------------------------------------------------------------------------- API conventions
def test_scalar_and_vector_calls_agree(shot):
    r = np.array([5.0, 18.3, 36.6])
    assert isinstance(shot.time_of_flight(10.0), float)
    assert isinstance(shot.velocity(10.0), float)
    assert isinstance(shot.drop(10.0), float)
    assert isinstance(shot.pattern_radius(10.0), float)
    for fn in (shot.time_of_flight, shot.velocity, shot.drop, shot.pattern_radius):
        out = fn(r)
        assert isinstance(out, np.ndarray) and out.shape == (3,)
        assert out == pytest.approx([fn(float(x)) for x in r], rel=1e-12)


def test_ranges_outside_the_table_are_clamped(shot):
    assert shot.time_of_flight(shot.max_range_m + 50.0) == shot.time_of_flight(shot.max_range_m)
    assert shot.velocity(-5.0) == shot.velocity(0.0)
    out = shot.velocity(np.array([-1.0, shot.max_range_m * 2]))
    assert out[0] == pytest.approx(shot.muzzle_velocity_mps)
    assert out[1] == pytest.approx(shot.velocity(shot.max_range_m))


def test_invalid_shot_config_is_rejected(cfg):
    for bad in (
        {"muzzle_velocity_mps": 0.0},
        {"pellet_diameter_mm": 0.0},
        {"max_range_m": 0.0},
        {"pattern_ref_range_m": 0.0},
        {"speed_of_sound_mps": 0.0},
    ):
        with pytest.raises(ValueError):
            ShotModel(_with_shot(cfg, **bad))
