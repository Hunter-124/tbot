import json
import math
from pathlib import Path

import pytest

from tbot.config import Calibration, load_config
from tbot.fire.ballistics import ShotModel
from tbot.tools.sensitivity import (
    build_report, latency_miss_rad, main, pwm_jitter_sigma_s, range_miss_rad,
)


def test_timing_miss_linear_and_zero():
    assert latency_miss_rad(2.0, 0.0) == 0.0
    assert latency_miss_rad(0.0, 0.02) == 0.0
    assert latency_miss_rad(2.0, 0.02) == pytest.approx(2 * latency_miss_rad(2.0, 0.01))
    assert latency_miss_rad(2.0, -0.01) == latency_miss_rad(2.0, 0.01)


def test_uniform_pwm_sigma():
    assert pwm_jitter_sigma_s(50) == pytest.approx(0.020 / math.sqrt(12))
    with pytest.raises(ValueError):
        pwm_jitter_sigma_s(0)


def test_range_miss_uses_worst_tof_error_and_rejects_clamping():
    shot = ShotModel(load_config())
    rate, r = 1.0, 20.0
    assert range_miss_rad(shot, r, 0, rate) == 0
    expected = max(abs(shot.time_of_flight(r * factor) - shot.time_of_flight(r))
                   for factor in (0.8, 1.2))
    assert range_miss_rad(shot, r, 0.2, rate) == pytest.approx(expected)
    assert range_miss_rad(shot, 50, 0.3, rate) is None


def test_report_geometry_budget_and_repeatability():
    cfg = load_config()
    calib = Calibration(Path("."), latency={"actuator": {"servo": {"cmd_to_break_ms": {"mean": 80, "std": 2}}}})
    report = build_report(cfg, calib, [20], [20])
    assert report == build_report(cfg, calib, [20], [20])
    cross, incoming = report["rows"]
    assert cross["angular_rate_rad_s"] == pytest.approx(1.0)
    assert incoming["angular_rate_rad_s"] == pytest.approx(1 / math.sqrt(2))
    assert cross["pwm_jitter_sigma_s"] == pytest.approx(0.02 / math.sqrt(12))
    assert cross["p_hit_with_pwm"] <= cross["p_hit_without_pwm"]
    assert report["latency_budget"]["servo_command_to_break_ms"] == 80
    assert report["latency_budget"]["servo_source"] == "measured"
    assert all(0 <= x["p_hit_with_pwm"] <= 1 for x in report["rows"])


def test_cli_json_and_domain(capsys):
    assert main(["--range-m", "20", "--speed-mps", "20", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["rows"]) == 2
    assert report["assumptions"]
    with pytest.raises(SystemExit):
        main(["--range-m", "61"])
