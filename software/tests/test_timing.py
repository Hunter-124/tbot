"""Time in the loop: the modelled work time of the simulator, the command at the planned instant, the IMU read
again before the solver, and the trigger delay the budget is built from."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tbot.app.run import Pipeline
from tbot.camera import open_source
from tbot.camera.synthetic import SyntheticSource
from tbot.config import Calibration, LatencyBudget, load_config
from tbot.hw.gpio import SimGpio
from tbot.tools.sim_eval import evaluate_run
from tbot.types import MonotonicClock, SimClock

RANGE = Path(__file__).resolve().parents[1] / "field_data" / "2026-11-17-range"


def _cfg(tmp_path, *sets):
    cfg = load_config(sets=["camera.fps=100.0", *sets])
    cfg.paths.calib_dir = str(tmp_path / "calib")
    cfg.gpio.backend = "sim"
    cfg.stream.enabled = False
    cfg.record.mode = "off"
    return cfg


# ------------------------------------------------------------------ clocks
def test_frozen_clock_only_moves_when_moved():
    c = SimClock(1000, frozen=True)
    assert c.now_ns() == 1000 == c.now_ns()
    c.advance(500)
    assert c.now_ns() == 1500
    assert c.sleep_until(9000) == 9000 == c.now_ns()
    assert c.sleep_until(2000) == 9000, "never backwards"


def test_real_clocks_wait_for_the_instant():
    for clock in (MonotonicClock(), SimClock(5_000_000_000)):
        t = clock.now_ns() + 3_000_000
        after = clock.sleep_until(t)
        assert t <= after < t + 200_000_000


def test_waiting_for_the_command_instant_works_on_any_clock_and_gives_up_on_a_stopped_one():
    from tbot.app.run import _wait_until

    class ReadOnly:  # a source clock that can only be read (hitl, latency tools)
        def now_ns(self):
            return time.monotonic_ns()

    class Stopped:
        def now_ns(self):
            return 42

    c = ReadOnly()
    t = c.now_ns() + 2_000_000
    assert _wait_until(c, t) >= t
    assert _wait_until(c, c.now_ns() - 5) is not None, "already past: at once"
    assert _wait_until(Stopped(), 43) is None
    assert _wait_until(SimClock(100, frozen=True), 900) == 900
    assert _wait_until(MonotonicClock(), time.monotonic_ns() + 1_000_000) is not None


# ------------------------------------------------------------------ the simulator's time
def test_modelled_work_time_gives_the_guns_frame_rate_and_frame_age(tmp_path):
    cfg = _cfg(tmp_path, "sim.process_ms=22.0")
    calib = Calibration(dir=tmp_path / "calib")
    src = SyntheticSource(cfg, calib, scenario="skeet_crossing", seed=1)
    pipe = Pipeline(cfg, calib, src, gpio=SimGpio())
    tel = []
    with pipe:
        while (t := pipe.step()) is not None:
            tel.append(t)
    seqs = [t["seq"] for t in tel]
    assert seqs == sorted(set(seqs))
    # 100 fps camera, 22 ms of work + 2 ms tail a frame: every 2nd-3rd frame is seen
    gaps = [b - a for a, b in zip(seqs, seqs[1:])]
    assert set(gaps) <= {2, 3} and src.skipped >= sum(g - 1 for g in gaps)
    assert 38.0 <= len(tel) / src.scenario.duration_s <= 46.0
    # the solver runs on a frame as old as on the gun (range day: 40-48 ms), not a few ms
    ages = sorted(t["age_ms"] for t in tel[5:])
    assert 30.0 < ages[len(ages) // 2] < 50.0


def test_every_frame_without_modelled_work_time(tmp_path):
    cfg = _cfg(tmp_path, "sim.process_ms=0")
    calib = Calibration(dir=tmp_path / "calib")
    src = SyntheticSource(cfg, calib, scenario="static_gun", seed=1)
    src.start()
    seqs = []
    for _ in range(30):
        seqs.append(src.read().seq)
    assert seqs == list(range(30))


def test_the_simulator_repeats_exactly(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.sim.process_ms > 0.0
    calib = Calibration(dir=tmp_path / "calib")
    a = evaluate_run(cfg, calib, "trap_straightaway", 1)
    b = evaluate_run(cfg, calib, "trap_straightaway", 1)
    key = lambda r: json.dumps(r, sort_keys=True, default=str)  # noqa: E731
    assert key(a) == key(b)


def test_command_goes_out_at_the_planned_instant_and_one_delay_before_the_exit(tmp_path):
    cfg = _cfg(tmp_path, "actuator.type=sim")
    cfg.fire.live = True
    calib = Calibration(dir=tmp_path / "calib")
    gpio = SimGpio()
    pipe = Pipeline(cfg, calib, open_source("synthetic:static_gun", cfg, calib), gpio=gpio)
    fired = None
    with pipe:
        gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)
        time.sleep(0.05)  # the arming debounce (a real timer)
        while (t := pipe.step()) is not None:
            if t["fired"]:
                fired = t
                break
    assert fired is not None, "the static-gun scenario gets a shot"
    assert fired["cmd_t_ns"] == fired["fire_at_ns"] >= fired["t_ns"]
    delay_ns = int(round(pipe.budget.fire_delay_s("sim") * 1e9))
    assert fired["decision"]["t_exit_ns"] == fired["fire_at_ns"] + delay_ns


def test_imu_is_read_again_before_the_solver_while_the_button_is_held(tmp_path):
    cfg = _cfg(tmp_path, "sim.process_ms=22.0")
    calib = Calibration(dir=tmp_path / "calib")

    def newest_sample_age(armed: bool) -> float:
        gpio = SimGpio()
        pressed, released = (0, 1) if cfg.arming.active_low else (1, 0)
        pipe = Pipeline(cfg, calib, open_source("synthetic:skeet_crossing", cfg, calib), gpio=gpio)
        ages = []
        with pipe:
            gpio.inject(int(cfg.arming.pin), pressed if armed else released)
            time.sleep(0.05)
            for _ in range(40):
                t = pipe.step()
                assert t["armed"] == armed
                if pipe.attitude.ready and pipe.attitude._ts:
                    ages.append((t["t_ns"] - pipe.attitude._ts[-1]) / 1e6)
        return sorted(ages)[len(ages) // 2]

    # held: the attitude the solver extrapolates from is as new as the IMU allows (2 ms samples), not as old as
    # the frame's detection and tracking (the modelled 22 ms)
    assert newest_sample_age(True) < 3.0
    assert newest_sample_age(False) > 20.0


# ------------------------------------------------------------------ the trigger delay
def test_default_trigger_delay_is_the_measured_one():
    cfg = load_config()
    budget = LatencyBudget.from_config(cfg, None)
    assert budget.cmd_to_break_ms["servo"] == 155.0
    assert budget.fire_delay_s("servo") == pytest.approx(0.164)
    assert cfg.actuator.sim.delay_ms == cfg.latency.sim_cmd_to_break_ms == 155.0


def test_a_measurement_beside_an_estimate_wins_over_the_estimate():
    cfg = load_config()
    lat = {"actuator": {"servo": {
        "cmd_to_break_ms": {"mean": 95.0, "std": 2.5},
        "method": "video estimate (LED at fire cmd, webcam 10 fps)",
        "mic_measurement": {"cmd_to_bang_ms": {"mean": 161.3, "std": 0.96}},
    }}}
    budget = LatencyBudget.from_config(cfg, lat)
    assert budget.cmd_to_break_ms["servo"] == pytest.approx(161.3 - cfg.gun.lock_time_ms)
    assert budget.sources["actuator.servo"] == "measured (microphone)"
    assert budget.jitter_ms["servo"] == 1.5
    # a plain estimate is still used as it is, and a measurement is never overridden by its own microphone note
    plain = {"actuator": {"servo": {"cmd_to_break_ms": {"mean": 95.0, "std": 2.5}, "method": "video estimate"}}}
    assert LatencyBudget.from_config(cfg, plain).cmd_to_break_ms["servo"] == 95.0
    measured = {"actuator": {"servo": {"cmd_to_break_ms": {"mean": 140.0, "std": 2.0}, "method": "recoil (IMU)",
                                       "mic_measurement": {"cmd_to_bang_ms": {"mean": 161.3}}}}}
    assert LatencyBudget.from_config(cfg, measured).cmd_to_break_ms["servo"] == 140.0


def test_the_guns_own_latency_file_of_the_range_day_now_reads_155_ms():
    path = RANGE / "gun" / "calib" / "latency.json"
    if not path.exists():
        pytest.skip("range-day field data not present")
    budget = LatencyBudget.from_config(load_config(), json.loads(path.read_text()))
    assert budget.cmd_to_break_ms["servo"] == pytest.approx(155.3, abs=0.05)
