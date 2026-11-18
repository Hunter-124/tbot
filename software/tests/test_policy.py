"""The firing bar (fire/policy.py) and the probability it is compared with (fire/solver.py _p_hit)."""

from __future__ import annotations

import math
import random

import pytest

from tbot.config import LatencyBudget, load_config
from tbot.fire.ballistics import ShotModel
from tbot.fire.policy import FirePolicy
from tbot.fire.solver import FireSolver, _hit_probability, _hit_probability_quad
from tbot.optics import MountModel

S = 1_000_000_000


# ------------------------------------------------------------------ the bar
def test_fixed_bar_is_the_bar():
    p = FirePolicy(0.4, 0.4)
    assert p.bar(1, 5 * S, 1 * S, 0.1, 0.9) == 0.4
    assert p.bar(1, 9 * S, 1 * S, 0.39, 0.2) == 0.4


def test_relax_slides_between_the_two_track_ages():
    p = FirePolicy(0.4, 0.2, relax_from_s=0.2, relax_to_s=0.7, relaxed=0.25)
    first = 10 * S
    assert p.bar(1, first + int(0.1 * S), first, 0.0, 1.0) == pytest.approx(0.40)
    assert p.bar(1, first + int(0.45 * S), first, 0.0, 1.0) == pytest.approx(0.325)
    assert p.bar(1, first + int(0.7 * S), first, 0.0, 1.0) == pytest.approx(0.25)
    assert p.bar(1, first + 3 * S, first, 0.0, 1.0) == pytest.approx(0.25)
    # the fixed rule of the first range day is the same thing as a step
    step = FirePolicy(0.4, 0.3, relax_from_s=0.7, relax_to_s=0.7, relaxed=0.3)
    assert step.bar(1, first + int(0.69 * S), first, 0.0, 1.0) == 0.4
    assert step.bar(1, first + int(0.70 * S), first, 0.0, 1.0) == 0.3


def test_reach_never_asks_for_more_than_a_perfect_aim_could_give_but_keeps_the_floor():
    p = FirePolicy(0.4, 0.2, reach=0.85)
    assert p.bar(1, S, S, 0.0, 0.9) == 0.4                         # a smooth swing: the bar itself
    assert p.bar(1, S, S, 0.0, 0.40) == pytest.approx(0.34)        # a rough one: 85 % of what it allows
    assert p.bar(1, S, S, 0.0, 0.10) == 0.2                        # never under the floor
    assert p.bar(1, S, S, 0.0, float("nan")) == 0.4                # no ceiling known: no relief


def test_regret_takes_the_second_chance_a_target_offers_and_forgets_with_the_target():
    p = FirePolicy(0.4, 0.2, regret=0.9)
    seq = [0.10, 0.25, 0.35, 0.30]                 # rises to a top under the bar ...
    assert all(p.bar(7, S, S, x, 1.0) == 0.4 for x in seq)
    assert p.bar(7, S, S, 0.15, 1.0) == 0.4        # ... and falls to under half of it: that chance is lost
    assert p.bar(7, S, S, 0.20, 1.0) == pytest.approx(0.9 * 0.35)
    assert p.bar(7, S, S, 0.33, 1.0) == pytest.approx(0.9 * 0.35)   # 0.33 is now enough
    assert p.bar(8, S, S, 0.33, 1.0) == 0.4        # another target starts from the bar
    # noise on the way up is not a lost chance: nothing fell to half
    q = FirePolicy(0.4, 0.2, regret=0.9)
    assert all(q.bar(1, S, S, x, 1.0) == 0.4 for x in (0.10, 0.22, 0.18, 0.28, 0.24, 0.36))


def test_policy_from_config_adaptive_and_fixed():
    cfg = load_config()
    p = FirePolicy.from_config(cfg)
    assert cfg.fire.adaptive and p.reach == cfg.fire.adaptive_reach and p.floor == cfg.fire.min_hit_probability_floor
    assert p.base == cfg.fire.min_hit_probability and p.relaxed == cfg.fire.min_hit_probability_settled
    fixed = FirePolicy.from_config(load_config(sets=["fire.adaptive=false", "fire.min_hit_probability=0.5",
                                                     "fire.min_hit_probability_settled=0.3", "fire.settled_ms=400"]))
    assert fixed.reach == 0.0 and fixed.regret == 0.0
    assert fixed.bar(1, S + int(0.39 * S), S, 0.0, 0.1) == 0.5
    assert fixed.bar(1, S + int(0.40 * S), S, 0.0, 0.1) == 0.3


# ------------------------------------------------------------------ the probability
def test_series_hit_probability_equals_the_quadrature():
    rnd = random.Random(1)
    worst = 0.0
    for _ in range(4000):
        sigma = math.radians(10 ** rnd.uniform(-1.5, 0.7))
        pattern = math.radians(rnd.uniform(0.2, 1.5))
        miss = math.radians(rnd.uniform(0.0, 5.0))
        worst = max(worst, abs(_hit_probability(miss, pattern, sigma) - _hit_probability_quad(miss, pattern, sigma)))
    assert worst < 1e-6
    # outside the series' range it falls back to the quadrature
    assert _hit_probability(0.001, 0.02, 0.0005) == pytest.approx(1.0, abs=1e-6)
    assert _hit_probability(0.2, 0.02, 0.0005) == 0.0


def _solver(*sets):
    cfg = load_config(sets=list(sets))
    return FireSolver(cfg, ShotModel(cfg), MountModel.from_config(cfg), LatencyBudget.from_config(cfg, None)), cfg


def test_aim_prior_raises_a_supported_solution_and_never_a_blind_one():
    solver, cfg = _solver()
    plain, _ = _solver("fire.aim_prior_deg=0")
    pattern = math.radians(0.75)
    # a pass with the aim near its top: the prior says what the bench and the range frames say
    miss, sigma = math.radians(0.5), math.radians(0.9)
    assert plain._p_hit(miss, pattern, sigma) == pytest.approx(_hit_probability(miss, pattern, sigma))
    assert solver._p_hit(miss, pattern, sigma) > 1.3 * plain._p_hit(miss, pattern, sigma)
    # nothing known (a track three frames old): the prior alone would say 0.25; it must say next to nothing
    blind = solver._p_hit(math.radians(8.0), pattern, math.radians(18.0))
    assert blind < 0.02
    assert solver._p_hit(0.0, pattern, math.radians(18.0)) < 0.5 * cfg.fire.min_hit_probability_floor
    # continuous where the evidence scaling takes over, and falling with the miss
    ps = [solver._p_hit(math.radians(m), pattern, math.radians(1.2)) for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_worst_case_work_counts_every_solution_a_firing_frame_needs():
    cfg = load_config(sets=["fire.horizon_frames=3"])
    assert FireSolver.worst_case_work(cfg) == pytest.approx(5.6)
