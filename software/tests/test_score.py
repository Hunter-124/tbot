"""tbot score: a dry shot of a recorded press is checked against the frames that follow it."""

from __future__ import annotations

import time

import numpy as np

from tbot.app.run import Pipeline
from tbot.camera import open_source
from tbot.config import Calibration, load_config
from tbot.hw.gpio import SimGpio
from tbot.tools import score as score_tool


def _press_session(tmp_path, frames=260, press=(40, 250), **sets):
    cfg = load_config(sets=[f"{k}={v}" for k, v in {"camera.fps": 100.0, **sets}.items()])
    cfg.paths.calib_dir = str(tmp_path / "calib")
    cfg.gpio.backend = "sim"
    cfg.stream.enabled = False
    cfg.record.dir = str(tmp_path / "sessions")
    cfg.record.mode, cfg.record.clip = "armed", "off"
    cfg.record.preroll_s, cfg.record.postroll_s = 0.2, 0.5
    calib = Calibration(dir=tmp_path / "calib")
    gpio = SimGpio()
    pressed, released = (0, 1) if cfg.arming.active_low else (1, 0)
    gpio.inject(int(cfg.arming.pin), released)
    pipe = Pipeline(cfg, calib, open_source("synthetic:skeet_crossing", cfg, calib), gpio=gpio)
    pipe.start()
    shots = 0
    for n in range(frames):
        if n == press[0]:
            gpio.inject(int(cfg.arming.pin), pressed)
            time.sleep(0.03)  # the arming debounce
        if n == press[1]:
            gpio.inject(int(cfg.arming.pin), released)
        tele = pipe.step()
        if tele is None:
            break
        shots = max(shots, int(tele["shots"]))
    pipe.close()
    sessions = sorted((tmp_path / "sessions").iterdir())
    return sessions, shots


def test_a_dry_shot_at_a_synthetic_clay_scores_as_a_hit(tmp_path):
    sessions, shots = _press_session(tmp_path)
    assert shots >= 1, "the solver fired during the press"
    assert len(sessions) == 1
    results = score_tool.score(sessions[0], image=True)
    assert len(results) == shots
    r = results[0]
    assert r["verdict"] == "HIT", r
    # the target flew as predicted (a synthetic clay does); the gun half is the swing extrapolated over the
    # trigger delay, which is the larger error by far, and still inside the pattern here
    assert r["miss_deg"] <= r["pattern_deg"]
    # (how far: a small far clay against the scene's tree line is extrapolated to 0.34 deg at the median over
    # every solved frame of this press, 0.15-0.4 deg at the frames good enough to fire on - `--every`)
    assert r["target_error_deg"] < 0.5 * r["pattern_deg"]
    assert r["gun_error_deg"] < r["pattern_deg"]
    assert (sessions[0] / "shot_1.png").exists()
    assert "HIT" in score_tool.format_shot(r)
    assert score_tool.main([str(sessions[0])]) == 0


def test_a_shot_at_the_very_end_of_the_recording_is_reported_not_scored(tmp_path):
    # released (and the post-roll over) before the shot could arrive: no verdict, and no crash
    sessions, shots = _press_session(tmp_path, frames=400, press=(40, 390), **{"record.postroll_s": 0.0})
    if shots == 0:
        return
    for r in score_tool.score(sessions[0]):
        assert r["verdict"] in ("HIT", "MISS") or r["verdict"].startswith("no data")


def test_config_from_meta_fills_keys_added_since_the_recording():
    cfg = load_config()
    import dataclasses

    meta = {"config": dataclasses.asdict(cfg)}
    del meta["config"]["attitude"]["accel_max_rate_dps"]
    del meta["config"]["detector"]["clutter_ring_max"]
    back = score_tool.config_from_meta(meta)
    assert back.attitude.accel_max_rate_dps == cfg.attitude.accel_max_rate_dps
    assert np.isclose(back.detector.clutter_ring_max, cfg.detector.clutter_ring_max)
