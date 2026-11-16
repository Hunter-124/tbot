"""Recoil detection from the IMU and `tbot latency-recoil`."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tbot.app.run import Pipeline
from tbot.camera import open_source
from tbot.config import load_calibration, load_config
from tbot.fire.recoil import G_MPS2, RecoilDetector, summarise
from tbot.tools import latency_recoil
from tbot.types import ImuSample

MS = 1_000_000


def _samples(t0_ms: float, t1_ms: float, bursts=(), hz: int = 500):
    """Still gun (1 g) with (start_ms, length_ms, g) bursts along the bore."""
    out = []
    step = 1000.0 / hz
    t = t0_ms
    while t < t1_ms:
        ax = 0.0
        for start, length, g in bursts:
            if start <= t < start + length:
                ax = g * G_MPS2
        out.append(ImuSample(t_ns=int(t * MS), gyro=np.zeros(3), accel=np.array([-ax, 0.0, G_MPS2])))
        t += step
    return out


def test_shot_is_detected_at_its_onset():
    det = RecoilDetector()
    det.expect(1000 * MS, shot=1)
    events = det.feed(_samples(900, 1400, bursts=[(1104, 12, 8.0)]))
    assert len(events) == 1 and events[0].detected
    assert events[0].delay_ms == pytest.approx(104.0, abs=2.0)
    assert events[0].peak_g > 7.0
    assert not det.waiting
    assert events[0].as_event()["type"] == "recoil"


def test_a_click_is_not_a_shot_and_a_silent_command_is_reported():
    det = RecoilDetector()
    det.expect(1000 * MS, shot=1)
    # servo slap / dry-fire click: one short spike, then nothing
    events = det.feed(_samples(900, 1700, bursts=[(1095, 2, 6.0)]))
    assert len(events) == 1 and not events[0].detected
    assert events[0].as_event()["type"] == "no_recoil"
    # a click followed by the real shot: the shot wins
    det.expect(2000 * MS, shot=2)
    events = det.feed(_samples(1900, 2600, bursts=[(2060, 2, 6.0), (2110, 15, 8.0)]))
    assert events[0].detected and events[0].delay_ms == pytest.approx(110.0, abs=2.0)


def test_swinging_hard_is_not_a_shot_and_early_samples_are_ignored():
    det = RecoilDetector()
    det.expect(1000 * MS, shot=1)
    # 1.5 g of swing acceleration for a long time, and a spike before the command could have acted
    events = det.feed(_samples(900, 1700, bursts=[(1002, 8, 8.0), (1050, 400, 1.5)]))
    assert len(events) == 1 and not events[0].detected


def test_doubles_resolve_in_order_across_feeds():
    det = RecoilDetector()
    det.expect(1000 * MS, shot=1)
    out = det.feed(_samples(1000, 1200, bursts=[(1100, 14, 8.0)]))
    det.expect(1400 * MS, shot=2)
    out += det.feed(_samples(1200, 1700, bursts=[(1497, 14, 8.0)]))
    assert [e.shot for e in out] == [1, 2]
    assert [round(e.delay_ms) for e in out] == pytest.approx([100, 98], abs=2)


def test_summarise_is_robust_to_a_flyer():
    s = summarise([101.0, 99.0, 100.0, 102.0, 98.0, 180.0])
    assert s["n"] == 6 and s["median"] == pytest.approx(100.5)
    assert s["std"] < 5.0
    assert summarise([]) is None


def test_pipeline_logs_recoil_events(tmp_path):
    cfg = load_config()
    cfg.paths.calib_dir = str(tmp_path / "calib")
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.camera.source = "synthetic:skeet_crossing"
    calib = load_calibration(cfg)
    pipe = Pipeline(cfg, calib, open_source(cfg.camera.source, cfg, calib))
    try:
        pipe.start()
        pipe.step()
        real_read = pipe.imu.read
        pipe.recoil.expect(5000 * MS, shot=1)
        pipe.recoil.expect(6000 * MS, shot=2)
        fake = _samples(5000, 6700, bursts=[(5101, 14, 8.0)])
        pipe.imu.read = lambda: fake
        pipe.attitude.update = lambda samples: None  # the fake samples are not on the source's clock
        pipe.step()
        pipe.imu.read = real_read
    finally:
        pipe.close()
    kinds = [(e["type"], e["shot"]) for e in pipe.events if "recoil" in e["type"]]
    assert kinds == [("recoil", 1), ("no_recoil", 2)]
    assert pipe.recoil_delays_ms == pytest.approx([101.0], abs=2.0)
    assert pipe.no_recoil == 1
    assert pipe.events[0]["expected_ms"] == pytest.approx(
        pipe.budget.cmd_to_break_ms["servo"] + pipe.budget.lock_time_ms)


def _write_session(root, name, events):
    d = root / name
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return d


def test_latency_recoil_tool_writes_the_measured_delay(tmp_path, capsys):
    sessions = tmp_path / "sessions"
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "latency.json").write_text(json.dumps(
        {"actuator": {"servo": {"cmd_to_break_ms": {"mean": 95.0, "std": 2.5}, "method": "video estimate"}},
         "camera": {"timestamp_offset_ns": 30047}}))
    delays = [112.0, 110.0, 114.0, 111.0, 113.0, 112.0]
    for k, d in enumerate(delays):
        evs = [{"type": "fire", "t_ns": k}, {"type": "recoil", "t_ns": k, "cmd_t_ns": 1000 + k, "shot": 1, "delay_ms": d}]
        if k == 0:
            evs.append({"type": "no_recoil", "t_ns": 5, "cmd_t_ns": 99, "shot": 2, "delay_ms": None})
        _write_session(sessions, f"s{k}", evs)
    _write_session(sessions, "dup", [{"type": "recoil", "t_ns": 0, "cmd_t_ns": 1000, "shot": 1, "delay_ms": 112.0}])
    common = ["--calib-dir", str(calib), "--set", f"record.dir={sessions.as_posix()}"]

    assert latency_recoil.main(common) == 0  # report only
    out = capsys.readouterr().out
    assert "shots               : 6" in out and "NO shot: 1" in out and "not written" in out
    assert json.loads((calib / "latency.json").read_text())["actuator"]["servo"]["cmd_to_break_ms"]["mean"] == 95.0

    assert latency_recoil.main(common + ["--write", "--min-shots", "10"]) == 1
    assert latency_recoil.main(common + ["--write"]) == 0
    doc = json.loads((calib / "latency.json").read_text())
    cfg = load_config()
    servo = doc["actuator"]["servo"]
    assert servo["cmd_to_break_ms"]["mean"] == pytest.approx(112.0 - cfg.gun.lock_time_ms)
    assert servo["cmd_to_break_ms"]["std"] >= 1.5
    assert "recoil" in servo["method"] and "estimate" not in servo["method"]
    assert doc["camera"]["timestamp_offset_ns"] == 30047  # the rest of the file is untouched
    assert (calib / "latency.prev.json").exists()
    # and the runtime budget now uses it, labelled as measured
    cfg.paths.calib_dir = str(calib)
    from tbot.config import LatencyBudget

    budget = LatencyBudget.from_config(cfg, load_calibration(cfg).latency)
    assert budget.cmd_to_break_ms["servo"] == pytest.approx(106.0)
    assert budget.sources["actuator.servo"] == "measured"


def test_latency_recoil_tool_without_shots(tmp_path, capsys):
    (tmp_path / "sessions").mkdir()
    rc = latency_recoil.main(["--calib-dir", str(tmp_path), "--set", f"record.dir={(tmp_path / 'sessions').as_posix()}"])
    assert rc == 1 and "no recoil events" in capsys.readouterr().out
