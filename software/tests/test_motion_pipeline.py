"""The motion front end wired into the pipeline (vision.engine = "motion", the default): engine switch, resets,
failure policy, telemetry schema, replay of a recorded press under both engines, and the overlay's two detail
levels on old and new telemetry."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from tbot.app import overlay
from tbot.app.run import Pipeline, vision_engine
from tbot.config import load_calibration, load_config
from tbot.optics import CameraModel, MountModel

ROOT = Path(__file__).resolve().parents[1]
PRESS = ROOT / "field_data" / "2026-11-17-range" / "presses" / "20261117-104806-press"
DET_FIELDS = 10   # [u, v, diameter_px, snr, contrast_dn, bx, by, bw, bh, area_px]; motion appends [10] mass


def _cfg(tmp_path, engine="motion", *sets, rig=True):
    """rig: the gun's own lens and boresight (fx ~2009 px, from a range-day press): the motion front end's pixel
    thresholds assume ~35 px/deg, which the nominal lens of an empty calibration (fx 467) is far from."""
    cfg = load_config(sets=[f"vision.engine={engine}", *sets])
    cfg.paths.calib_dir = str(tmp_path / "calib")
    if rig and PRESS.is_dir():
        cal = json.loads((PRESS / "meta.json").read_text())["calibration"]
        (tmp_path / "calib").mkdir(parents=True, exist_ok=True)
        for name in ("intrinsics", "boresight"):
            (tmp_path / "calib" / f"{name}.json").write_text(json.dumps(cal[name]))
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.stream.enabled = False
    return cfg, load_calibration(cfg)


def _synthetic_pipe(cfg, calib, scenario, armed=True, seed=1):
    import time

    from tbot.camera.synthetic import SyntheticSource
    from tbot.hw.gpio import open_gpio

    source = SyntheticSource(cfg, calib, scenario=scenario, seed=seed)
    gpio = open_gpio(cfg)
    pipe = Pipeline(cfg, calib, source, gpio=gpio)
    if armed:
        gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)
        t_end = time.monotonic() + 1.0
        while not pipe.arming.armed and time.monotonic() < t_end:
            time.sleep(0.001)
    return pipe, source


def _run(pipe, max_frames=None):
    out = []
    try:
        while (tele := pipe.step()) is not None:
            out.append(json.loads(json.dumps(tele)))   # as telemetry.jsonl has it
            if max_frames is not None and len(out) >= max_frames:
                break
    finally:
        pipe.close()
    return out


# ----------------------------------------------------------------------------------------------- engine switch
def test_default_engine_is_motion_and_builds_the_motion_stages(tmp_path):
    from tbot.vision.motion import MotionDetector
    from tbot.vision.tracker_motion import MotionTracker

    assert load_config().vision.engine == "motion"
    cfg, calib = _cfg(tmp_path)
    pipe, _ = _synthetic_pipe(cfg, calib, "no_clay", armed=False)
    try:
        assert pipe.engine == "motion"
        assert isinstance(pipe.detector, MotionDetector)
        assert isinstance(pipe.tracker, MotionTracker)
        assert pipe.clutter is None, "the IMU scenery filter is legacy-only"
    finally:
        pipe.close()


def test_legacy_engine_builds_the_software_of_2026_11_19(tmp_path):
    from tbot.vision.clutter import StaticFilter
    from tbot.vision.detector import ClayDetector
    from tbot.vision.tracker3d import CartesianClayTracker
    from tbot.vision.tracker_motion import MotionTracker

    cfg, calib = _cfg(tmp_path, "legacy")
    pipe, _ = _synthetic_pipe(cfg, calib, "no_clay", armed=False)
    try:
        assert pipe.engine == "legacy"
        assert isinstance(pipe.detector, ClayDetector)
        assert type(pipe.tracker) is CartesianClayTracker and not isinstance(pipe.tracker, MotionTracker)
        assert isinstance(pipe.clutter, StaticFilter)
    finally:
        pipe.close()


def test_an_unknown_engine_is_a_config_error_at_start(tmp_path):
    cfg, calib = _cfg(tmp_path)
    cfg.vision.engine = "magic"
    with pytest.raises(ValueError, match="vision.engine"):
        _synthetic_pipe(cfg, calib, "no_clay", armed=False)
    assert vision_engine(cfg.__class__.__new__(cfg.__class__)) == "legacy"   # a config object without [vision]


# ----------------------------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("scenario", ["skeet_crossing", "skeet_outgoing", "skeet_incoming"])
def test_a_clay_is_tracked_and_shot_at_in_a_dry_run(tmp_path, scenario):
    """Crossing, going away and incoming: a track on the clay (its detections match the truth), and the solver
    reaches a fire decision. The P(hit) bar is lowered: this is the integration, `tbot sim` scores accuracy."""
    from tbot.tools.sim_eval import evaluate_run

    cfg, calib = _cfg(tmp_path)
    cfg.camera.fps = 50.0
    cfg.fire.min_hit_probability = 0.05
    res = evaluate_run(cfg, calib, scenario, 1)
    assert res["visible_frames"] > 0
    assert res["detection_rate"] > 0.3, res
    assert res["t_first_confirmed_ns"] is not None, "no track on the clay"
    assert res["fires"] >= 1, f"never fired; decision reasons: {res['reasons']}"
    for shot in res["shots"]:
        assert shot["armed_at_fire"] is True
        assert shot["t_fire_ns"] > res["t_first_confirmed_ns"]


def test_dry_run_decides_to_fire_and_never_drives_an_actuator(tmp_path):
    cfg, calib = _cfg(tmp_path)
    cfg.camera.fps = 50.0
    cfg.fire.live = False
    cfg.fire.min_hit_probability = 0.05
    pipe, _ = _synthetic_pipe(cfg, calib, "skeet_crossing")
    assert pipe.actuator is None
    tel = _run(pipe)
    assert sum(t["fired"] for t in tel) >= 1, "the solver never reached a fire decision"
    assert all(t["armed"] for t in tel if t["fired"])


def test_no_clay_no_track_and_nothing_fires(tmp_path):
    cfg, calib = _cfg(tmp_path)
    cfg.camera.fps = 50.0
    pipe, _ = _synthetic_pipe(cfg, calib, "no_clay")
    tel = _run(pipe)
    assert tel
    assert not any(t["fired"] for t in tel)
    assert max(len(t["tracks"]) for t in tel) == 0, "a track was born with no clay in the sky"


def test_not_armed_never_fires(tmp_path):
    from tbot.tools.sim_eval import evaluate_run

    cfg, calib = _cfg(tmp_path)
    cfg.camera.fps = 50.0
    cfg.fire.min_hit_probability = 0.05
    res = evaluate_run(cfg, calib, "skeet_crossing", 1, armed=False)
    assert res["armed_frames"] == 0 and res["fires"] == 0


# ----------------------------------------------------------------------------------------------- telemetry
def test_telemetry_schema(tmp_path):
    cfg, calib = _cfg(tmp_path)
    cfg.camera.fps = 50.0
    pipe, _ = _synthetic_pipe(cfg, calib, "skeet_crossing")
    tel = _run(pipe)
    rows = [r for t in tel for r in t["dets"]]
    assert rows, "no detection in the whole run"
    assert all(len(r) == DET_FIELDS + 1 for r in rows), "motion rows: the legacy 10 fields + mass"
    assert all(r[10] > 0.0 for r in rows), "mass of a motion detection is positive"
    for t in tel:
        v = t["vision"]
        assert v["engine"] == "motion"
        assert set(v) >= {"chains", "born", "reg_ok", "reg_shift_px", "noise_dn", "errors"}
        assert v["errors"] == 0
        assert t["static"] is None and t["scenery_drift_dps"] is None
        assert "noise_dn" in t["detector"] and "scale" in t["detector"]
        assert all(not isinstance(x, list) or len(x) <= 2 for x in t["detector"].values()), "no arrays"
    assert sum(t["vision"]["born"] for t in tel) >= 1
    assert len(json.dumps(tel[-1])) < 20_000, "a telemetry line stays small"


def test_legacy_telemetry_has_no_motion_fields(tmp_path):
    cfg, calib = _cfg(tmp_path, "legacy")
    cfg.camera.fps = 50.0
    pipe, _ = _synthetic_pipe(cfg, calib, "skeet_crossing")
    tel = _run(pipe, max_frames=60)
    assert all(len(r) == DET_FIELDS for t in tel for r in t["dets"])
    assert all("vision" not in t for t in tel)


# ----------------------------------------------------------------------------------------------- robustness
def test_a_detector_that_raises_costs_the_frame_not_the_loop(tmp_path, caplog):
    cfg, calib = _cfg(tmp_path)
    pipe, _ = _synthetic_pipe(cfg, calib, "skeet_crossing")
    calls = {"detect": 0, "reset": 0}
    real_detect, real_reset = pipe.detector.detect, pipe.detector.reset

    def detect(frame, R_wg=None):
        calls["detect"] += 1
        if calls["detect"] in (5, 6):
            raise RuntimeError("boom")
        return real_detect(frame, R_wg=R_wg)

    def reset():
        if calls["detect"] in (5, 6):   # (the detector also resets itself on its first frame)
            calls["reset"] += 1
        real_reset()

    pipe.detector.detect, pipe.detector.reset = detect, reset
    with caplog.at_level(logging.ERROR, logger="tbot.run"):
        tel = _run(pipe, max_frames=12)
    assert len(tel) == 12, "the loop went on"
    assert tel[4]["n_detections"] == 0 and tel[4]["vision"]["errors"] == 1 and tel[5]["vision"]["errors"] == 2
    assert calls["reset"] == 2
    assert any("RuntimeError: boom" in (r.exc_text or "") or r.exc_info for r in caplog.records)


def test_a_tracker_that_raises_is_reset_and_the_frame_has_no_tracks(tmp_path):
    cfg, calib = _cfg(tmp_path)
    pipe, _ = _synthetic_pipe(cfg, calib, "skeet_crossing")
    real = pipe.tracker.update
    n = [0]

    def update(t_ns, dets, R):
        n[0] += 1
        if n[0] == 3:
            raise ValueError("bad")
        return real(t_ns, dets, R)

    pipe.tracker.update = update
    tel = _run(pipe, max_frames=6)
    assert len(tel) == 6 and tel[2]["tracks"] == [] and tel[2]["vision"]["errors"] == 1
    assert tel[2]["decision"]["fire"] is False


def test_camera_restart_resets_the_detector_and_the_tracker(tmp_path):
    cfg, calib = _cfg(tmp_path)
    pipe, source = _synthetic_pipe(cfg, calib, "skeet_crossing")
    seen = []
    real = pipe.detector.reset
    pipe.detector.reset = lambda: (seen.append("det"), real())[1]
    real_t = pipe.tracker.reset
    pipe.tracker.reset = lambda: (seen.append("trk"), real_t())[1]
    try:
        for _ in range(5):
            pipe.step()
        seen.clear()                       # (the detector resets itself on its first frame)
        source.gap_after_restart = True
        pipe.step()
    finally:
        pipe.close()
    assert seen == ["trk", "det", "trk"]   # (a detector reset always takes the motion tracker with it)


def test_a_rebased_stabilised_plane_resets_the_motion_tracker(tmp_path):
    cfg, calib = _cfg(tmp_path)
    pipe, _ = _synthetic_pipe(cfg, calib, "skeet_crossing")
    resets = []
    real = pipe.tracker.reset
    pipe.tracker.reset = lambda: (resets.append(pipe.frames), real())[1]
    try:
        for _ in range(4):
            pipe.step()
        assert resets == []
        pipe.detector.stab_epoch += 1      # as MotionDetector does when it re-bases the plane
        pipe.step()
        pipe.step()
    finally:
        pipe.close()
    assert resets == [4]


def test_detection_in_process_is_not_used_by_the_motion_engine(tmp_path, caplog):
    from tbot.camera.synthetic import SyntheticSource
    from tbot.vision.handover import SensorDetector

    cfg, calib = _cfg(tmp_path)
    source = SyntheticSource(cfg, calib, scenario="no_clay", seed=1)
    source.detects = True             # as a sensor process that detects would say
    with caplog.at_level(logging.WARNING, logger="tbot.run"):
        pipe = Pipeline(cfg, calib, source)
    try:
        assert pipe._remote is False
        assert any("detect_in_process" in r.getMessage() for r in caplog.records)
        tele = pipe.step()
        assert tele is not None and "handover" not in tele
    finally:
        pipe.close()
    with pytest.raises(RuntimeError, match="legacy"):
        SensorDetector(cfg, calib, 640, 400, False)


# ----------------------------------------------------------------------------------------------- recorded press
@pytest.fixture(scope="module")
def press_runs():
    from tbot.tools.rescore import rerun

    if not PRESS.is_dir():
        pytest.skip("range-day press not in this checkout")
    out = {}
    for engine in ("legacy", "motion"):
        _, tel = rerun(PRESS, load_config(sets=[f"vision.engine={engine}"]))
        out[engine] = tel
    return out


def test_a_recorded_press_replays_under_both_engines(press_runs):
    n = sum(1 for line in (PRESS / "frames.csv").read_text().splitlines()[1:] if line.strip())
    for engine, tel in press_runs.items():
        assert len(tel) == n, engine
        assert all(math.isfinite(t["stage_ms"]["total"]) for t in tel)
    assert all(len(r) == DET_FIELDS for t in press_runs["legacy"] for r in t["dets"])
    assert all(len(r) == DET_FIELDS + 1 for t in press_runs["motion"] for r in t["dets"])
    assert all(t["vision"]["errors"] == 0 for t in press_runs["motion"])
    assert any(t["vision"]["reg_ok"] for t in press_runs["motion"])


@pytest.mark.parametrize("detail", ["clean", "debug"])
def test_overlay_draws_old_and_new_telemetry_at_both_levels(press_runs, detail):
    meta = json.loads((PRESS / "meta.json").read_text())
    cfg = load_config()
    w, h = int(meta["width"]), int(meta["height"])
    cal = meta.get("calibration") or {}
    cam = CameraModel.from_calibration(cal.get("intrinsics"), cfg, w, h)
    mount = MountModel.from_config(cfg, cal.get("boresight"))
    old = [json.loads(x) for x in (PRESS / "telemetry.jsonl").read_text().splitlines() if x.strip()]
    gray = np.full((h, w), 120, np.uint8)
    for tel in (old, press_runs["legacy"], press_runs["motion"]):
        busy = max(tel, key=lambda t: (len(t.get("tracks") or []), len(t.get("dets") or [])))
        for rec in (tel[0], busy, tel[-1]):
            out = overlay.draw_record(gray, rec, cam, mount, bore_px=(w / 2, h / 2), detail=detail)
            assert out.shape == (h, w, 3)


def test_clean_overlay_has_no_detection_brackets_or_scenery_marks():
    cfg = load_config()
    cam, mount = CameraModel.nominal(cfg, 640, 400), MountModel.from_config(cfg, None)
    rec = {"seq": 1, "armed": True, "dets": [[100.0, 100.0, 9.0, 30.0, -40.0, 96, 96, 8, 8, 40.0, 900.0],
                                             [400.0, 300.0, 5.0, 8.0, -20.0, 398, 298, 5, 5, 18.0, 200.0]],
           "static": [[500.0, 300.0]], "R_wg": np.eye(3).ravel().tolist(), "tracks": [],
           "decision": {"fire": False, "reason": "no confirmed track"}}
    gray = np.full((400, 640), 150, np.uint8)
    mask = np.full((400, 640), 255, np.uint8)
    mask[300:, 250:390] = 0
    clean = overlay.draw_record(gray, rec, cam, mount, mask=mask, detail="clean")
    debug = overlay.draw_record(gray, rec, cam, mount, mask=mask, detail="debug")
    assert (debug == overlay.C_DET).all(axis=2).sum() >= 20
    assert (clean == overlay.C_DET).all(axis=2).sum() == 0
    assert np.array_equal(clean[100, 100], overlay.C_DET_DOT) and np.array_equal(clean[300, 400], overlay.C_DET_DOT)
    assert np.array_equal(clean[350, 300], [150, 150, 150]), "no mask wash"
    assert (clean[30:380] != 150).any(axis=2).sum() == 2, "two dots and nothing else in the picture"
