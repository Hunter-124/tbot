"""Detection in the sensor process (camera.detect_in_process): vision/handover.py, camera/proc.py, app/run.py.

What must hold: a frame detected in the child gives the pipeline what its own detector would have given it;
the hints (tracks for the scenery filter) arrive as the tracker would predict them; with the hints of the same
frame the single loop uses (`lockstep=0`) a recorded press decides the same; and a child that cannot detect
leaves the pipeline detecting, as before.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tbot.app.run import Pipeline
from tbot.camera.proc import SensorProcess
from tbot.config import Calibration, load_calibration, load_config
from tbot.optics import CameraModel, MountModel
from tbot.types import Detection, Track
from tbot.vision import handover
from tbot.vision.tracker3d import make_tracker

# Detection in a sensor process is the LEGACY front end's (vision/handover.py refuses the motion engine).
PRESSES = Path(__file__).resolve().parents[1] / "field_data" / "2026-11-17-range" / "presses"
PRESS = PRESSES / "20261117-104603-press"   # 128 frames, trees, a swing, one shot
needs_press = pytest.mark.skipif(not (PRESS / "frames.bin").exists(), reason="recorded range presses not here")


# ------------------------------------------------------------------------------------------ wire format
def test_result_round_trip():
    dets = [Detection(t_ns=123, u=10.25, v=20.5, area_px=31.0, diameter_px=6.28, contrast=-41.5, snr=9.75,
                      score=0.625, bbox=(7, 17, 6, 7)),
            Detection(t_ns=123, u=600.0, v=399.5, area_px=4.0, diameter_px=2.26, contrast=12.0, snr=4.5,
                      score=0.25, bbox=(598, 398, 3, 2))]
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r = handover.DetectResult(detections=dets, R_wg=R, filtered=True, imu_ready=True, scale=2, noise_sigma=1.75,
                              timings_ms={"bg": 1.5, "total": np.float64(4.25)},
                              stats={"components": np.int64(12), "candidates": 5, "kept": 3, "measured": 2},
                              static_px=[[1.5, 2.5], [3.0, 4.0]], drift=(12.5, -3.0), drift_updates=7,
                              attitude_ms=0.125, detect_ms=11.5, t_read_ns=10**12, t_sent_ns=10**12 + 5, hints_n=41)
    back = handover.unpack_result(b"xyz" + handover.pack_result(r), 123, offset=3)
    assert back.detections == dets
    assert np.array_equal(back.R_wg, R)
    assert (back.filtered, back.imu_ready, back.scale, back.noise_sigma) == (True, True, 2, 1.75)
    assert back.timings_ms == {"bg": 1.5, "total": 4.25}
    assert back.stats == {"components": 12, "candidates": 5, "kept": 3, "measured": 2}
    assert back.static_px == [[1.5, 2.5], [3.0, 4.0]]
    assert (back.drift, back.drift_updates, back.hints_n) == ((12.5, -3.0), 7, 41)
    assert (back.attitude_ms, back.detect_ms, back.t_read_ns, back.t_sent_ns) == (0.125, 11.5, 10**12, 10**12 + 5)
    empty = handover.unpack_result(handover.pack_result(handover.DetectResult(detections=[], R_wg=np.eye(3))), 5)
    assert empty.detections == [] and not empty.filtered and math.isnan(empty.noise_sigma)


@pytest.mark.parametrize("model", ["cartesian", "angular"])
def test_hints_predict_as_the_tracker_does(model):
    """The child predicts the hinted tracks with the tracker's own code from the states sent: to the bit."""
    cfg = load_config(sets=["vision.engine=legacy", f"tracker.model={model}"])
    cam = CameraModel.nominal(cfg, 640, 400)
    tracker = make_tracker(cfg, cam, MountModel.from_config(cfg, None))
    rng = np.random.default_rng(3)
    tracks = []
    for i in range(3):
        A = rng.normal(size=(9, 9)) * 0.1
        s = np.concatenate([[30.0 + i, 5.0 * i, 8.0], rng.normal(size=3) * 10.0, rng.normal(size=3)])
        B = rng.normal(size=(6, 6)) * 0.01
        tracks.append(Track(track_id=i + 1, t_ns=1_000_000_000 + i, x=rng.normal(size=6) * 0.1, P=B @ B.T,
                            range_m=30.0, range_sigma_m=5.0, hits=9, confirmed=True,
                            s=s if model == "cartesian" else None, S=A @ A.T if model == "cartesian" else None))
    protect, focus = [tracks[2], tracks[0]], [tracks[0], tracks[1]]
    h = handover.unpack_hints(b"P" + handover.pack_hints(17, protect, focus), 1)
    assert h.n == 17 and len(h.tracks) == 3   # a track in both lists travels once
    t = 1_030_000_000
    want = handover.scenery_dirs(tracker, t, protect, focus)
    got = handover.scenery_dirs(tracker, t, [h.tracks[i] for i in h.protect], [h.tracks[i] for i in h.focus])
    for a, b in zip(want, got):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    none = handover.unpack_hints(handover.pack_hints(3, [], []))
    assert none.n == 3 and not none.tracks and not none.protect and not none.focus


# ------------------------------------------------------------------------------------------ recorded press
@needs_press
def test_recorded_press_decides_the_same_through_the_sensor_process():
    """The scenery filter knowing the same tracks as the single loop's (lockstep 0): every frame has the same
    detections and the same decision; one frame stale (as live) still runs the whole press with detections
    made in the child."""
    from tbot.tools.rescore import rerun

    _, single = rerun(PRESS, load_config(sets=["vision.engine=legacy"]))
    _, split = rerun(PRESS, load_config(sets=["vision.engine=legacy"]), handover=0)
    assert len(split) == len(single) > 100
    assert any(r["decision"]["fire"] for r in single), "the press no longer has a shot: pick another"
    for a, b in zip(single, split):
        where = f"frame {a['frames']}"
        assert b["handover"]["hints_age"] in (None, 1), where
        assert a["dets"] == b["dets"], where
        assert a["static"] == b["static"] and a["detector"] == b["detector"], where
        assert a["scenery_drift_dps"] == b["scenery_drift_dps"], where
        assert [t["id"] for t in a["tracks"]] == [t["id"] for t in b["tracks"]], where
        da, db = a["decision"], b["decision"]
        assert (da["fire"], da["track_id"], a["armed"], a["fired"]) == (db["fire"], db["track_id"], b["armed"],
                                                                       b["fired"]), where
        # the child's attitude filter is fed the same samples in other portions: equal to rounding
        assert np.allclose(a["R_wg"], b["R_wg"], rtol=0.0, atol=1e-12), where
        if math.isfinite(da["p_hit"]):
            assert db["p_hit"] == pytest.approx(da["p_hit"], abs=1e-6), where
        assert a["imu_age_ms"] == b["imu_age_ms"] and a["t_ns"] == b["t_ns"], where

    _, stale = rerun(PRESS, load_config(sets=["vision.engine=legacy"]), handover=1)
    assert len(stale) == len(single)
    ages = {r["handover"]["hints_age"] for r in stale}
    assert ages <= {None, 2}, ages
    assert sum(len(r["dets"]) for r in stale) > 0.5 * sum(len(r["dets"]) for r in single)
    assert sum(r["fired"] for r in stale) == sum(r["fired"] for r in single)


# ------------------------------------------------------------------------------------------ live shape
def _cfg(tmp_path, *sets):
    cfg = load_config(sets=["vision.engine=legacy", "sim.process_ms=0.0", "camera.fps=50.0", *sets])
    cfg.paths.calib_dir = str(tmp_path / "calib")
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.stream.enabled = False
    return cfg


def _steps(pipe, n, timeout_s=30.0):
    out, deadline = [], time.monotonic() + timeout_s
    while len(out) < n and time.monotonic() < deadline:
        tele = pipe.step()
        if tele is not None:
            out.append(tele)
    assert len(out) == n, f"only {len(out)} of {n} frames within {timeout_s} s"
    return out


def test_pipeline_takes_detections_from_the_sensor_process(tmp_path):
    cfg = _cfg(tmp_path)
    source = SensorProcess(cfg, load_calibration(cfg), "synthetic:skeet_crossing", detect=True)
    try:
        assert source.detects
        pipe = Pipeline(cfg, load_calibration(cfg), source)
        try:
            tel = _steps(pipe, 40)
        finally:
            pipe.close()
    finally:
        source.stop()
    assert all("handover" in t for t in tel)
    assert all(t["handover"]["sensor_ms"] >= 0.0 for t in tel)
    assert sum(t["n_detections"] for t in tel) > 0, "the child's detector saw nothing of the clay"
    assert any(t["tracks"] for t in tel), "detections from the child never became a track"
    with_dets = next(t for t in tel if t["dets"])
    assert with_dets["detector_ms"]["total"] > 0.0 and "components" in with_dets["detector"]
    assert with_dets["stage_ms"]["detect"] > 0.0
    for key in ("seq", "age_ms", "fps", "dropped", "lost", "cam_restarts", "armed", "decision", "R_wg", "static",
                "scenery_drift_dps", "imu_ready", "ae", "health"):
        assert key in with_dets


def test_off_by_default_and_never_outside_a_sensor_process(tmp_path):
    """camera.detect_in_process is off as shipped; switched on, the pipeline still only uses it for a source
    that says it detects: synthetic, replay and a sensor process asked not to are the single loop."""
    from tbot.camera import open_source

    cfg = _cfg(tmp_path)
    assert cfg.camera.detect_in_process is False
    calib = load_calibration(cfg)
    plain = SensorProcess(cfg, calib, "synthetic:static_gun")   # detect=None: what the config says
    try:
        assert not plain.detects
        pipe = Pipeline(cfg, calib, plain)
        try:
            assert not pipe._remote
            pipe._idle_period = 0.1
            pipe._set_pace(False)            # the single loop rests by itself: the sensor process is never slowed
            assert plain._pace_ns == 0
        finally:
            pipe.close()
    finally:
        plain.stop()
    cfg.camera.detect_in_process = True
    pipe = Pipeline(cfg, calib, open_source("synthetic:static_gun", cfg, calib))
    try:
        assert not pipe._remote
        assert "handover" not in _steps(pipe, 3)[-1]
    finally:
        pipe.close()
    source = SensorProcess(cfg, calib, "synthetic:static_gun", detect=False)
    try:
        assert not source.detects
        source.read(timeout_s=5.0)
        assert source.detection is None
    finally:
        source.stop()
    on = SensorProcess(cfg, calib, "synthetic:static_gun")       # the key alone switches it on
    try:
        assert on.detects
    finally:
        on.stop()


def test_child_that_cannot_detect_leaves_it_to_the_pipeline(tmp_path):
    bad = _cfg(tmp_path, "tracker.model=nonsense")   # the child's detector stage cannot be built with this
    cfg = _cfg(tmp_path)
    source = SensorProcess(bad, Calibration(dir=tmp_path / "calib"), "synthetic:skeet_crossing", detect=True)
    try:
        assert not source.detects
        pipe = Pipeline(cfg, load_calibration(cfg), source)
        try:
            tel = _steps(pipe, 30)
        finally:
            pipe.close()
    finally:
        source.stop()
    assert all("handover" not in t for t in tel)
    assert sum(t["n_detections"] for t in tel) > 0   # detected here, in the single loop


def test_pace_and_respawn_keep_the_detections_coming(tmp_path):
    cfg = _cfg(tmp_path)
    source = SensorProcess(cfg, Calibration(dir=tmp_path / "calib"), "synthetic:static_gun", detect=True,
                           respawn_s=5.0)

    def frames(n, timeout_s=20.0):
        out, deadline = [], time.monotonic() + timeout_s
        while len(out) < n and time.monotonic() < deadline:
            f = source.read(timeout_s=0.2)
            if f is not None:
                assert source.detection is not None, "a frame without detections"
                out.append(f)
        assert len(out) == n
        return out

    try:
        frames(3)
        source.set_pace(0.1)            # button up: every fifth frame of a 50 fps camera
        frames(2)                       # (what was already on its way)
        dropped = source.dropped
        slow = frames(6)
        gaps = np.diff([f.t_capture_ns for f in slow]) / 1e9
        assert gaps.min() > 0.07, gaps
        assert source.dropped > dropped  # the frames left out are counted, as when this side skipped them
        source.set_pace(0.0)
        frames(2)
        fast = frames(6)
        assert (np.diff([f.t_capture_ns for f in fast]) / 1e9).max() < 0.07

        source.set_pace(0.1)
        old = source.child_pid
        source._proc.kill()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not (source.respawns >= 1 and source.streaming):
            source.read(timeout_s=0.1)
        assert source.respawns == 1 and source.child_pid != old and source.detects
        frames(2)
        slow = frames(4)                # the new child was told the pace again
        assert (np.diff([f.t_capture_ns for f in slow]) / 1e9).min() > 0.07
    finally:
        source.stop()


def test_detector_process_with_a_sensor_process_of_its_own(tmp_path):
    """As with the live camera: pipeline <- detector process <- sensor process. Frames arrive with detections,
    a killed detector process is replaced with its sensor process, and stopping leaves neither behind."""
    cfg = _cfg(tmp_path)
    source = SensorProcess(cfg, Calibration(dir=tmp_path / "calib"), "synthetic:skeet_crossing", detect=True,
                           nested=True, respawn_s=5.0)
    sensor_pids = [source.sensor_pid]

    def frames(n, timeout_s=30.0):
        out, deadline = [], time.monotonic() + timeout_s
        while len(out) < n and time.monotonic() < deadline:
            f = source.read(timeout_s=0.2)
            if f is not None:
                assert source.detection is not None
                out.append(f)
        assert len(out) == n
        return out

    try:
        assert source.detects and source.has_imu
        got = frames(20)
        assert [f.seq for f in got] == sorted({f.seq for f in got})
        imu = source.imu().read()
        assert imu and all(b.t_ns > a.t_ns for a, b in zip(imu, imu[1:]))   # relayed in order, once
        source._proc.kill()
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline and not (source.respawns >= 1 and source.streaming):
            source.read(timeout_s=0.1)
        assert source.respawns == 1 and source.detects
        sensor_pids.append(source.sensor_pid)
        assert None not in sensor_pids and sensor_pids[0] != sensor_pids[1] != source.child_pid
        frames(5)
    finally:
        source.stop()
    assert not source._proc.is_alive()
    if sys.platform != "win32":
        for pid in sensor_pids:   # the sensor processes behind both detector processes are gone too
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and _alive(pid):
                time.sleep(0.1)
            assert not _alive(pid), f"sensor process {pid} outlived its detector process"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # (a zombie waiting for init is not alive)
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True
