"""Virtual clays through recorded gun motion (tbot vclay) and recorded presses through today's software
(tbot rescore), on the presses of the first range day."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tbot.config import LatencyBudget, load_config
from tbot.fire.policy import FirePolicy
from tbot.fire.solver import FireSolver
from tbot.tools import rescore as rescore_tool
from tbot.tools import score as score_tool
from tbot.tools import vclay

RANGE = Path(__file__).resolve().parents[1] / "field_data" / "2026-11-17-range"
PRESS = RANGE / "presses" / "20261117-105308-press"      # 2.5 s held, a clay followed, a fire command
needs_press = pytest.mark.skipif(not (PRESS / "imu.csv").exists(), reason="range-day field data not present")
# frames.bin is in git LFS: without it the file is a pointer of a few hundred bytes
needs_frames = pytest.mark.skipif(not (PRESS / "frames.bin").exists() or (PRESS / "frames.bin").stat().st_size < 1e6,
                                  reason="range-day frames (git LFS) not present")


@pytest.fixture(scope="module")
def stage():
    cfg = score_tool.rig_config(load_config(), PRESS)
    return cfg, vclay.Press(PRESS, cfg), vclay.Bench(cfg)


@needs_press
def test_rig_config_takes_the_gun_from_the_press_and_the_decisions_from_today(stage):
    cfg, press, bench = stage
    today = load_config()
    assert cfg.optics.model == "pinhole" and cfg.optics.focal_length_mm == 12.0     # the gun's lens
    assert cfg.shot.pattern_radius_ref_m == pytest.approx(0.30)                      # the gun's choke and load
    assert cfg.fire.min_hit_probability == today.fire.min_hit_probability            # today's rule
    assert cfg.latency.servo_cmd_to_break_ms == today.latency.servo_cmd_to_break_ms == 155.0
    assert bench.assumed_delay_s == pytest.approx(0.164)
    assert press.armed_s > 2.0 and len(press.frames) > 50
    assert any(f.armed for f in press.frames) and not all(f.armed for f in press.frames)


@needs_press
def test_a_virtual_clay_is_where_the_charge_of_its_moment_arrives(stage):
    cfg, press, bench = stage
    solver = FireSolver(cfg, bench.shot, press.mount, LatencyBudget.from_config(cfg, None))
    rng = np.random.default_rng(3)
    made = 0
    for cls in vclay.SPEED_CLASSES:
        for _ in range(8):
            clay = bench.make_clay(press, rng, cls)
            if clay is None:
                continue
            made += 1
            # a command one true trigger delay before the planned exit lands inside the pattern
            t_cmd = clay.t_plan_exit_ns - int(round(bench.true_delay_s * 1e9))
            miss, hit, dist, _bore, _dir = bench.outcome(press, clay, t_cmd, solver)
            assert hit and math.degrees(miss) == pytest.approx(clay.offset_deg, abs=0.05)
            assert cfg.fire.min_range_m < dist < cfg.fire.max_range_m
            lo, hi = vclay.SPEED_CLASSES[cls]
            assert lo <= clay.speed_mps <= hi
    assert made >= 12


@needs_press
def test_the_bench_repeats_and_the_shipped_rule_beats_a_bar_nothing_reaches(stage):
    cfg, press, bench = stage
    a = vclay.make_results([press], bench, 24, seed=5, policy=True)
    b = vclay.make_results([press], bench, 24, seed=5, policy=True)
    assert [(r.shot.t_ns if r.shot else None) for r in a] == [(r.shot.t_ns if r.shot else None) for r in b]
    shipped = vclay.tally(a, [r.shot for r in a])
    assert shipped["shots"] >= 10 and shipped["hits"] >= 6
    for r in a:
        if r.shot is not None:
            assert r.shot.p_hit >= r.shot.bar >= cfg.fire.min_hit_probability_floor
    # the same clays with every look-ahead worked out, a rule replayed over them
    open_runs = vclay.make_results([press], bench, 24, seed=5, policy=False)
    assert all(r.shot is None for r in open_runs)
    never = vclay.tally(open_runs, [vclay.replay(r, FirePolicy(0.95, 0.95)) for r in open_runs])
    replayed = vclay.tally(open_runs, [vclay.replay(r, FirePolicy.from_config(cfg), cfg.fire) for r in open_runs])
    assert never["hits"] < replayed["hits"]
    # replaying the shipped rule is the shipped rule
    assert replayed["shots"] == shipped["shots"] and replayed["hits"] == shipped["hits"]


@needs_press
def test_a_wrong_trigger_delay_shows_in_the_gun_half_of_the_error(stage):
    """The delay the solver assumes against the one the gun has: the same commands, judged at another exit
    time. (In clays hit, 60 ms costs 6 points over 1500 clays of six presses; one press is too few to count
    clays, but the bore error of the shots taken shows it at once.)"""
    cfg, press, _ = stage
    gun_err, hits = [], []
    for error_ms in (0.0, 150.0):
        bench = vclay.Bench(cfg, true_delay_error_ms=error_ms, jitter_ms=0.0)
        res = vclay.make_results([press], bench, 45, seed=9, policy=True)
        t = vclay.tally(res, [r.shot for r in res])
        gun_err.append(t["gun_err_deg_p50"])
        hits.append(t["hits"])
    assert gun_err[1] > 1.5 * gun_err[0]
    assert hits[0] > hits[1]


@needs_press
def test_causal_attitude_knows_nothing_after_its_limit(stage):
    _cfg, press, _ = stage
    att = vclay.CausalAttitude(press.full)
    t = press.t_arm + int(0.8e9)
    att.limit_ns = t
    assert np.allclose(att.R_wg(t - 1000), press.R(t - 1000))
    assert np.array_equal(att.omega_g(t + int(0.1e9)), press.full.omega_g(t))
    # past the limit the gun is extrapolated at its last rate, not read from the future
    ahead = t + int(0.15e9)
    w = press.full.omega_g(t)
    from tbot.imu.attitude import exp_so3
    assert np.allclose(att.R_wg(ahead), press.R(t) @ exp_so3(w * 0.15), atol=1e-9)
    assert not np.allclose(att.R_wg(ahead), press.R(ahead), atol=1e-5)


# ------------------------------------------------------------------ rescore: the real frames again
@needs_frames
def test_rescore_runs_a_real_press_through_todays_software_and_repeats():
    cfg = load_config()
    one = rescore_tool.rescore(PRESS, cfg, every=True)
    two = rescore_tool.rescore(PRESS, cfg)
    assert one["armed_frames"] > 50 and one["delay_ms"] == pytest.approx(164.0, abs=1.0)
    assert [s.get("t_ns") for s in one["shots"]] == [s.get("t_ns") for s in two["shots"]]
    assert len(one["shots"]) == 1, "one shot a press"
    shot = one["shots"][0]
    # the clay of this press was followed and the planned shot is on it (the gun fired at P 0.73 on the day,
    # for a trigger delay it did not have)
    assert shot["verdict"] == "HIT" and shot["miss_deg"] < shot["pattern_deg"]
    assert shot["delay_ms"] == pytest.approx(164.0, abs=1.0)
    assert one["every"]["frames"] >= 5


@needs_frames
def test_rescore_holds_the_button_when_the_shooter_did():
    cfg = load_config()
    session, tel = rescore_tool.rerun(PRESS, cfg)
    armed = [r for r in tel if r["armed"]]
    events = [e for e in session.events if e.get("type") == "arming"]
    press_ns = next(int(e["t_ns"]) for e in events if e["armed"])
    release_ns = next(int(e["t_ns"]) for e in events if not e["armed"])
    assert armed and min(r["t_ns"] for r in armed) >= press_ns
    assert max(r["t_exposure_ns"] for r in armed) <= release_ns
    assert not any(r.get("fired") for r in tel if not r["armed"])
    assert rescore_tool.then(PRESS).startswith("command at P 0.73")
