"""Offline doctor reports actionable config and calibration failures."""
import json

import numpy as np
import cv2

from tbot.tools import doctor


def _run(capsys, tmp_path, *sets):
    args = ["--calib-dir", str(tmp_path)]
    for item in sets:
        args += ["--set", item]
    rc = doctor.main(args)
    return rc, capsys.readouterr().out


def test_invalid_servo_pulse_and_hold_are_failures(capsys, tmp_path):
    rc, text = _run(capsys, tmp_path, "actuator.servo.rest_us=3000", "actuator.servo.hold_ms=30")
    assert rc == 1
    assert "FAIL rest_us: actuator.servo.rest_us=3000" in text
    assert "FAIL hold_ms:" in text


def test_conflicts_and_fire_safety_gates(capsys, tmp_path):
    rc, text = _run(capsys, tmp_path, "arming.pin=18", "fire.max_shots_per_arm=2", "fire.min_shot_interval_ms=100",
                    "tracker.range_prior_m=50", "actuator.servo.pwm_channel=1")
    assert rc == 1
    assert "FAIL pin map: GPIO18: arming, servo PWM" in text
    assert "FAIL PWM overlay channel:" in text
    assert "FAIL shots per arm:" in text
    assert "FAIL range prior:" in text


def test_shots_per_arm_allows_spaced_doubles_and_triples(capsys, tmp_path):
    _, text = _run(capsys, tmp_path, "fire.max_shots_per_arm=3", "fire.min_shot_interval_ms=250")
    assert "PASS shots per arm:" in text
    _, text = _run(capsys, tmp_path, "fire.max_shots_per_arm=4")
    assert "FAIL shots per arm:" in text


def test_camera_and_calibration_errors(capsys, tmp_path):
    (tmp_path / "intrinsics.json").write_text(json.dumps({"width": 640, "height": 480,
        "model": "fisheye", "K": [[400, 0, 320], [0, 400, 240], [0, 0, 1]], "D": [0, 0, 0, 0]}))
    (tmp_path / "boresight.json").write_text(json.dumps({"rpy_deg": [0, 20, 0]}))
    (tmp_path / "imu.json").write_text(json.dumps({"R_gun_imu": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}))
    assert cv2.imwrite(str(tmp_path / "mask.png"), np.zeros((200, 320), dtype=np.uint8))
    rc, text = _run(capsys, tmp_path, "camera.exposure_us=10000", "camera.fps=300")
    assert rc == 1
    for line in ("FAIL exposure:", "FAIL sensor mode:", "FAIL intrinsics size:",
                 "FAIL boresight angles:", "FAIL IMU axis map:", "WARN mask size:"):
        assert line in text


def test_json_and_missing_bench_warning(capsys, tmp_path):
    rc = doctor.main(["--json", "--calib-dir", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["status"] == "WARN"
    row = next(c for c in report["checks"] if c["group"] == "Processing" and c["name"] == "bench")
    assert row["status"] == "WARN"
    assert "run tbot bench on the Pi" in row["hint"]


def test_pq12_conflicts_only_when_selected(capsys, tmp_path):
    rc, text = _run(capsys, tmp_path, "actuator.pq12.pin_extend=17")
    assert rc == 0
    assert "FAIL pin map:" not in text
    rc, text = _run(capsys, tmp_path, "actuator.type=pq12", "actuator.pq12.pin_extend=17")
    assert rc == 1
    assert "FAIL pin map: GPIO17: arming, PQ12 extend" in text
