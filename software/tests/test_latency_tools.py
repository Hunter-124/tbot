"""Latency and bench tools: LED analysis, the simulated LED camera path, report and bench."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tbot.config import load_config
from tbot.tools import bench, latency_camera, latency_report

FPS = 120.0
PERIOD_NS = int(round(1e9 / FPS))
EXPOSURE_US = 4000.0
EXPOSURE_NS = int(EXPOSURE_US * 1e3)
DELIVERY_MS = 7.5  # exposure end -> frame in user space
BIAS_MS = 1.2  # the sensor timestamps mid-exposure 1.2 ms LATE => the correction is -1.2 ms


def _generated_trials(n: int = 200, seed: int = 3) -> list[dict]:
    """Trials from a known camera: global shutter, fixed delivery latency, biased sensor clock."""
    rng = np.random.default_rng(seed)
    delivery_ns = int(DELIVERY_MS * 1e6)
    bias_ns = int(BIAS_MS * 1e6)
    out: list[dict] = []
    for i in range(n):
        base = 1_000_000_000 + i * 5 * PERIOD_NS  # exposure of frame k starts at base + k*period
        t_on = base + int(rng.uniform(0, PERIOD_NS))
        k = 0
        while True:  # the tool records the first frame that is at least half lit
            start = base + k * PERIOD_NS
            fraction = min(max((start + EXPOSURE_NS - t_on) / EXPOSURE_NS, 0.0), 1.0)
            if fraction >= 0.5:
                break
            k += 1
        out.append(
            {
                "t_on_ns": t_on,
                "t_exposure_ns": start + EXPOSURE_NS // 2 + bias_ns,
                "t_capture_ns": start + EXPOSURE_NS + delivery_ns,
                "exposure_us": EXPOSURE_US,
                "fraction": fraction,
            }
        )
    return out


# ------------------------------------------------------------------------------- analyze_led_trials
def test_analyze_led_trials_recovers_timestamp_offset_and_latency():
    trials = _generated_trials()
    res = latency_camera.analyze_led_trials(trials, 1.0 / FPS)

    assert res["n"] == len(trials)
    assert res["n_partial"] > 0
    # The sensor clock runs 1.2 ms late, so -1.2 ms must be ADDED to t_exposure_ns.
    assert res["timestamp_offset_ns"] == pytest.approx(-BIAS_MS * 1e6, abs=0.2e6)

    raw = res["exposure_to_user_ms"]
    # t_capture - t_on holds the delivery latency plus half an exposure plus up to one frame of phase.
    floor_ms = DELIVERY_MS + EXPOSURE_US / 2e3
    assert raw["min"] >= floor_ms - 1e-6
    assert raw["max"] <= floor_ms + PERIOD_NS / 1e6 + 1e-6
    assert floor_ms < raw["mean"] < floor_ms + PERIOD_NS / 1e6

    # Removing the phase wait with the recovered offset leaves exactly the delivery latency.
    corrected = res["exposure_end_to_user_ms"]
    assert corrected["mean"] == pytest.approx(DELIVERY_MS, abs=0.05)
    assert corrected["max"] - corrected["min"] < 0.05
    assert res["warnings"] == []


def test_analyze_led_trials_warns_when_no_frame_is_partially_lit():
    trials = [
        {
            "t_on_ns": 1_000_000_000 + i * PERIOD_NS,
            "t_exposure_ns": 1_000_000_000 + i * PERIOD_NS + 5_000_000,
            "t_capture_ns": 1_000_000_000 + i * PERIOD_NS + 12_000_000,
            "exposure_us": 100.0,
            "fraction": 1.0,
        }
        for i in range(20)
    ]
    res = latency_camera.analyze_led_trials(trials, 1.0 / FPS)
    assert res["n_partial"] == 0
    assert res["timestamp_offset_ns"] == 0
    assert any("exposure" in w for w in res["warnings"])
    assert res["exposure_to_user_ms"]["n"] == 20


def test_parse_roi_rejects_bad_input():
    assert latency_camera.parse_roi(" 10, 20 ,30,40") == (10, 20, 30, 40)
    with pytest.raises(ValueError):
        latency_camera.parse_roi("10,20,30")
    with pytest.raises(ValueError):
        latency_camera.parse_roi("10,20,0,40")


# ------------------------------------------------------------------------------------ sim LED camera
def _sim_rig():
    from tbot.hw.gpio import SimGpio

    cfg = load_config(sets=["camera.width=320", "camera.height=200", f"camera.fps={FPS}",
                            f"camera.exposure_us={EXPOSURE_US}"])
    led = latency_camera.LedDriver(SimGpio(), cfg.gpio.latency_led_pin)
    cam = latency_camera.SimLedCamera(cfg, led, delivery_latency_ms=DELIVERY_MS,
                                      timestamp_offset_ns=int(-BIAS_MS * 1e6), jitter_ms=0.0)
    led.clock = cam.clock
    return led, cam


def test_locate_led_finds_the_blinking_led_not_the_static_bright_patch():
    led, cam = _sim_rig()
    cam.start()
    found = latency_camera.locate_led(cam, led, toggles=3)
    x, y, w, h = found["roi"]
    lx, ly, lw, lh = cam.led_roi
    # The ROI must sit on the LED disc, not on the (brighter, but constant) static patch.
    assert x <= lx + lw // 2 <= x + w and y <= ly + lh // 2 <= y + h
    assert w <= lw + 4 and h <= lh + 4
    assert found["on_level"] - found["off_level"] > 0.5 * (cam.led_on_dn - cam.led_off_dn)
    assert found["off_level"] < cam.led_off_dn + 2.0


def test_latency_camera_sim_writes_calibration_and_recovers_the_bias(tmp_path, capsys):
    rc = latency_camera.main(
        [
            "--sim",
            "--trials", "30",
            "--calib-dir", str(tmp_path),
            "--set", f"camera.exposure_us={EXPOSURE_US}",
            "--set", "camera.width=320",
            "--set", "camera.height=200",
            "--set", f"camera.fps={FPS}",
            "--sim-latency-ms", str(DELIVERY_MS),
            "--sim-offset-ms", str(-BIAS_MS),
            "--sim-jitter-ms", "0",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    cam = json.loads((tmp_path / "latency.json").read_text())["camera"]
    for key in ("timestamp_offset_ns", "exposure_to_user_ms", "n_partial", "created"):
        assert key in cam
    assert cam["exposure_to_user_ms"]["n"] == 30
    assert cam["n_partial"] >= 1
    assert cam["timestamp_offset_ns"] == pytest.approx(-BIAS_MS * 1e6, abs=0.2e6)
    assert cam["exposure_end_to_user_ms"]["mean"] == pytest.approx(DELIVERY_MS, abs=0.3)
    assert cam["exposure_to_user_ms"]["min"] >= DELIVERY_MS + EXPOSURE_US / 2e3 - 0.5


def test_latency_camera_sim_feeds_the_latency_budget(tmp_path):
    from tbot.config import load_calibration

    assert latency_camera.main(
        ["--sim", "--trials", "12", "--calib-dir", str(tmp_path), "--json",
         "--set", f"camera.exposure_us={EXPOSURE_US}", "--set", "camera.width=320",
         "--set", "camera.height=200", "--set", f"camera.fps={FPS}", "--sim-jitter-ms", "0"]
    ) == 0
    cfg = load_config(sets=[f"paths.calib_dir={tmp_path}"])
    budget = load_calibration(cfg).budget
    assert budget.sources["camera.timestamp_offset"] == "measured"
    assert budget.sources["camera.exposure_to_user"] == "measured"
    assert budget.camera_timestamp_offset_ns == pytest.approx(-1.2e6, abs=0.3e6)


# ------------------------------------------------------------------------------------------- report
def _report(capsys, calib_dir, *extra) -> dict:
    assert latency_report.main(["--calib-dir", str(calib_dir), "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def _stats(mean: float, std: float) -> dict:
    return {"mean": mean, "std": std, "p50": mean, "p95": mean + 2 * std,
            "min": mean - std, "max": mean + 2 * std, "n": 25}


def test_latency_report_defaults_without_any_calibration(tmp_path, capsys):
    rep = _report(capsys, tmp_path)
    assert {c["source"] for c in rep["components"]} == {"default"}
    assert rep["processing"]["source"] == "default"
    cfg = load_config()
    assert rep["processing"]["p95_ms"] == pytest.approx(1000.0 / cfg.camera.fps)
    for name in ("servo", "pq12"):
        act = rep["actuators"][name]
        assert act["decision_to_muzzle_ms"] == pytest.approx(
            getattr(cfg.latency, f"{name}_cmd_to_break_ms") + cfg.gun.lock_time_ms + cfg.gun.barrel_time_ms
        )
    # The PQ12 is the slow option: that is the comparison this report exists for.
    assert rep["actuators"]["pq12"]["clay_drift_m"] > rep["actuators"]["servo"]["clay_drift_m"]
    assert rep["actuators"]["pq12"]["clay_drift_deg"] > rep["actuators"]["servo"]["clay_drift_deg"]
    assert rep["reference_clay"]["tof_s"] > 20.0 / 400.0  # slower than the muzzle velocity


def test_latency_report_uses_measured_servo_numbers(tmp_path, capsys):
    before = _report(capsys, tmp_path)
    (tmp_path / "latency.json").write_text(
        json.dumps(
            {
                "camera": {"timestamp_offset_ns": -1_200_000,
                           "exposure_to_user_ms": _stats(9.25, 0.4)},
                "actuator": {"servo": {"cmd_to_break_ms": _stats(62.0, 2.5)}},
            }
        )
    )
    after = _report(capsys, tmp_path)

    rows = {c["name"]: c for c in after["components"]}
    assert rows["actuator.servo.cmd_to_break"]["source"] == "measured"
    assert rows["actuator.servo.cmd_to_break"]["value"] == pytest.approx(62.0)
    assert rows["actuator.servo.jitter"]["value"] == pytest.approx(2.5)
    assert rows["actuator.pq12.cmd_to_break"]["source"] == "default"
    assert rows["camera.exposure_to_user"]["source"] == "measured"
    assert rows["camera.timestamp_offset"]["value"] == pytest.approx(-1.2)

    assert after["actuators"]["servo"]["decision_to_muzzle_ms"] == pytest.approx(62.0 + 6.0 + 3.0)
    assert after["actuators"]["servo"]["decision_to_muzzle_ms"] < before["actuators"]["servo"][
        "decision_to_muzzle_ms"]
    assert after["actuators"]["servo"]["clay_drift_m"] < before["actuators"]["servo"]["clay_drift_m"]
    assert after["actuators"]["pq12"]["decision_to_muzzle_ms"] == pytest.approx(
        before["actuators"]["pq12"]["decision_to_muzzle_ms"]
    )


def test_latency_report_takes_processing_from_bench_json(tmp_path, capsys):
    (tmp_path / "bench.json").write_text(
        json.dumps({"total_ms": _stats(7.0, 1.5), "resolution": [640, 400]})
    )
    rep = _report(capsys, tmp_path)
    assert rep["processing"]["source"] == "measured"
    assert rep["processing"]["p95_ms"] == pytest.approx(10.0)
    assert "640x400" in rep["processing"]["note"]
    servo = rep["actuators"]["servo"]
    assert servo["photon_to_arrival_ms"] == pytest.approx(
        rep["camera"]["exposure_to_user_ms"] + 10.0 + servo["decision_to_muzzle_ms"]
        + rep["reference_clay"]["tof_ms"]
    )


def test_latency_report_text_output_lists_every_actuator(tmp_path, capsys):
    assert latency_report.main(["--calib-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for name in ("servo", "pq12", "camera.exposure_to_user", "gun.lock_time", "clay moves"):
        assert name in out


# -------------------------------------------------------------------------------------------- bench
def test_bench_times_every_stage_and_writes_bench_json(tmp_path, capsys):
    rc = bench.main(
        [
            "--frames", "20",
            "--calib-dir", str(tmp_path),
            "--set", "camera.width=320",
            "--set", "camera.height=200",
            "--threads", "1",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    data = json.loads((tmp_path / "bench.json").read_text())
    assert data["frames"] == 20
    assert data["resolution"] == [320, 200]
    for stage in bench.STAGES:
        st = data["stages"][stage]
        assert st["n"] == 20
        assert st["p50"] >= 0.0 and st["max"] >= st["p50"]
    assert data["total_ms"]["p50"] >= sum(data["stages"][s]["p50"] for s in bench.PROCESS_STAGES) * 0.5
    assert data["fps_achievable"] > 0.0
    assert data["verdict"] in ("OK", "TOO SLOW")
    assert data["opencv"]["version"] and data["platform"]


def test_run_bench_returns_stats_without_writing(tmp_path):
    from tbot.config import load_calibration

    cfg = load_config(sets=[f"paths.calib_dir={tmp_path}", "camera.width=320", "camera.height=200"])
    calib = load_calibration(cfg)
    result = bench.run_bench(cfg, calib, frames=6, threads=1)
    assert set(bench.STAGES) <= set(result["stages"])
    assert result["frames"] == 6
    assert result["total_ms"]["n"] == 6
    assert not (tmp_path / "bench.json").exists()
