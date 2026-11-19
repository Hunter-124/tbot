"""The trigger delay follows the lever (fire/trigger.py): moving to the pre-travel, parked there, let go, spent."""

from __future__ import annotations

import math

import pytest

from tbot.config import LatencyBudget, load_config
from tbot.fire.ballistics import ShotModel
from tbot.fire.solver import FireSolver
from tbot.fire.trigger import SETTLE_MS, TriggerDelay
from tbot.optics import MountModel

MS = 1_000_000
T0 = 5_000_000_000
# this gun, 2026-11-19: rest 1340, break 1095, fire 1075, lever waits 60 % of the way while the button is held
GUN = ["actuator.type=servo", "actuator.servo.rest_us=1340", "actuator.servo.fire_us=1075",
       "actuator.servo.min_us=1070", "actuator.servo.max_us=1500", "actuator.servo.break_us=1095",
       "actuator.servo.preload_frac=0.6"]
MEASURED = {"actuator": {"servo": {
    "cmd_to_break_ms": {"mean": 38.5, "std": 1.1, "n": 10}, "method": "mic",
    "from_rest": {"cmd_to_break_ms": {"mean": 191.7, "std": 1.4, "n": 10}, "method": "mic"}}}}


def _trigger(sets=(), measured=MEASURED):
    cfg = load_config(sets=GUN + list(sets))
    budget = LatencyBudget.from_config(cfg, measured)
    return cfg, budget, TriggerDelay(cfg, budget)


def test_budget_reads_the_from_rest_delay_from_the_file_and_from_the_config():
    _cfg, budget, _t = _trigger()
    assert budget.cmd_to_break_ms["servo"] == 38.5 and budget.rest_cmd_to_break_ms == {"servo": 191.7}
    assert budget.rest_jitter_ms == {"servo": 1.4}
    cfg = load_config(sets=GUN + ["latency.servo_cmd_to_break_rest_ms=180"])
    assert LatencyBudget.from_config(cfg, None).rest_cmd_to_break_ms == {"servo": 180.0}
    assert LatencyBudget.from_config(load_config(), None).rest_cmd_to_break_ms == {}  # as shipped: one delay


def test_one_delay_when_no_from_rest_figure_is_known():
    _cfg, budget, t = _trigger(measured={"actuator": {"servo": {"cmd_to_break_ms": {"mean": 155.0, "std": 1.0}}}})
    assert not t.two_states
    for armed in (True, False):
        assert t.at(T0, armed) == (155.0, 1.0)
    t.on_arming(True, T0)
    t.on_fire(T0 + 300 * MS)
    assert t.at(T0 + 900 * MS) == (155.0, 1.0)


def test_a_press_moves_the_lever_then_it_waits_parked():
    _cfg, _b, t = _trigger()
    assert t.at(T0, armed=False)[0] == 191.7 and t.state == "rest"      # nobody holds the button
    t.on_arming(True, T0)
    d0, j0 = t.at(T0)
    assert d0 == pytest.approx(191.7) and j0 == pytest.approx(1.4) and t.state == "moving"
    # a command while it moves breaks when a from-rest shot sent at the press would have
    for since in (20, 80, 140):
        d, j = t.at(T0 + since * MS)
        assert d == pytest.approx(191.7 - since) and t.state == "moving"
        assert j == pytest.approx(math.hypot(1.4, 0.1 * since))
    knee = 191.7 - 38.5
    d, j = t.at(T0 + int((knee + 1) * MS))
    assert d == 38.5 and j > 5.0 and t.state == "settling"               # just arrived: not yet trusted to 1 ms
    d, j = t.at(T0 + int((knee + SETTLE_MS + 1) * MS))
    assert (d, j) == (38.5, 1.1) and t.state == "parked"
    assert t.at(T0 + 5_000 * MS) == (38.5, 1.1)
    # the delay never rises on the way, and never drops below the parked one
    seq = [t.at(T0 + k * 5 * MS)[0] for k in range(80)]
    assert all(a >= b for a, b in zip(seq, seq[1:])) and min(seq) == 38.5


def test_release_and_a_new_press_start_again():
    _cfg, _b, t = _trigger()
    t.on_arming(True, T0)
    assert t.at(T0 + 1_000 * MS)[0] == 38.5
    t.on_arming(False, T0 + 1_100 * MS)
    assert t.at(T0 + 1_200 * MS, armed=False)[0] == 191.7
    t.on_arming(True, T0 + 2_000 * MS)
    assert t.at(T0 + 2_050 * MS)[0] == pytest.approx(141.7)


def test_after_a_shot_the_lever_is_at_rest_until_the_next_press():
    _cfg, _b, t = _trigger()
    t.on_arming(True, T0)
    t.on_fire(T0 + 600 * MS)
    assert t.at(T0 + 1_500 * MS) == (191.7, 1.4) and t.state == "spent"
    t.on_arming(False, T0 + 2_000 * MS)
    t.on_arming(True, T0 + 3_000 * MS)
    assert t.at(T0 + 4_000 * MS)[0] == 38.5


def test_pre_travel_given_up_after_preload_max_s_is_an_unknown_delay():
    _cfg, _b, t = _trigger(["actuator.servo.preload_max_s=30"])
    t.on_arming(True, T0)
    assert t.at(T0 + 29_000 * MS) == (38.5, 1.1)
    d, j = t.at(T0 + 30_001 * MS)
    assert t.state == "let go" and d == pytest.approx((38.5 + 191.7) / 2) and j > 40.0
    _cfg, _b, forever = _trigger(["actuator.servo.preload_max_s=0"])
    forever.on_arming(True, T0)
    assert forever.at(T0 + 300_000 * MS) == (38.5, 1.1)


def test_pre_travel_off_or_not_marked_means_every_shot_is_from_rest():
    for sets in (["actuator.servo.preload_frac=0"], ["actuator.servo.break_us=0"],
                 ["actuator.servo.break_us=1400"]):   # a break outside rest..fire: the actuator refuses it too
        _cfg, _b, t = _trigger(sets)
        t.on_arming(True, T0)
        assert t.at(T0 + 2_000 * MS) == (191.7, 1.4), sets


def test_a_press_already_held_when_first_seen_counts_from_then():
    _cfg, _b, t = _trigger()
    assert t.at(T0, armed=True)[0] == pytest.approx(191.7)
    assert t.at(T0 + 400 * MS, armed=True)[0] == 38.5


class _Still:
    ready = True

    def R_wg(self, t_ns):
        import numpy as np
        return np.eye(3)

    def omega_g(self, t_ns):
        import numpy as np
        return np.zeros(3)


class _OneTrack:
    """A tracker with one confirmed target dead ahead (the solver only needs best / predict / predict_dir)."""

    def __init__(self):
        import numpy as np
        from tbot.types import Track
        self._dir = np.array([1.0, 0.0, 0.05]) / math.sqrt(1.0025)
        self.track = Track(track_id=1, t_ns=T0, x=np.zeros(6), P=np.eye(6) * 1e-8, hits=20, range_m=20.0,
                           range_sigma_m=1.0, confirmed=True, first_t_ns=T0 - 500 * MS)
        self.tracks = [self.track]

    def best(self, accept=None):
        return self.track

    def predict_dir(self, track, t_ns):
        return self._dir

    def predict(self, track, t_ns):
        return 0.0, 0.0, 1e-4


def _solver():
    cfg, budget, trig = _trigger()
    solver = FireSolver(cfg, ShotModel(cfg), MountModel.from_config(cfg, None), budget)
    return cfg, budget, trig, solver


def test_solver_plans_for_the_delay_of_the_lever_state():
    try:
        tracker = _OneTrack()
    except TypeError as exc:  # Track grew a required field: this stand-in needs it too
        pytest.skip(f"stand-in track out of date: {exc}")
    cfg, budget, trig, solver = _solver()
    lock_barrel_ms = budget.lock_time_ms + budget.barrel_time_ms
    # without the trigger model: the one delay of the budget
    d = solver.evaluate(T0, tracker, _Still(), True, 0.0125)
    assert (d.t_exit_ns - T0) / MS == pytest.approx(38.5 + lock_barrel_ms, abs=0.01)
    solver.trigger = trig
    trig.on_arming(True, T0)
    d = solver.evaluate(T0 + 50 * MS, tracker, _Still(), True, 0.0125)
    assert d.extra["trigger_state"] == "moving"
    assert (d.t_exit_ns - (T0 + 50 * MS)) / MS == pytest.approx(141.7 + lock_barrel_ms, abs=0.01)
    assert d.extra["fire_delay_s"] * 1e3 == pytest.approx(141.7 + lock_barrel_ms, abs=0.01)
    d = solver.evaluate(T0 + 800 * MS, tracker, _Still(), True, 0.0125)
    assert d.extra["trigger_state"] == "parked"
    assert (d.t_exit_ns - (T0 + 800 * MS)) / MS == pytest.approx(38.5 + lock_barrel_ms, abs=0.01)
    # not armed: a press now would start from rest
    trig.on_arming(False, T0 + 900 * MS)
    d = solver.evaluate(T0 + 950 * MS, tracker, _Still(), False, 0.0125)
    assert (d.t_exit_ns - (T0 + 950 * MS)) / MS == pytest.approx(191.7 + lock_barrel_ms, abs=0.01)
