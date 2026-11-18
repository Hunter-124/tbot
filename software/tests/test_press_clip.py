"""Held-press recording, per-frame telemetry, the annotated clip, the scenery filter and the sensor process."""

import json
import math
import time

import numpy as np
import pytest

from tbot.app.record import SUMMARY_JSON, TELEMETRY_JSONL, RingRecorder
from tbot.app.run import Pipeline
from tbot.camera import open_source
from tbot.config import Calibration, load_config
from tbot.hw.gpio import SimGpio
from tbot.optics import CameraModel, MountModel
from tbot.types import Frame
from tbot.vision.clutter import StaticFilter
from tbot.vision.detector import ClayDetector


def _cfg(tmp_path, **sets):
    # (every frame of the scenario, as these tests count them: no modelled work time)
    cfg = load_config(sets=[f"{k}={v}" for k, v in {"sim.process_ms": 0.0, **sets}.items()])
    cfg.paths.calib_dir = str(tmp_path / "calib")
    cfg.gpio.backend = "sim"
    cfg.stream.enabled = False
    cfg.record.dir = str(tmp_path / "sessions")
    return cfg


def _frames(n, shape=(12, 16), fps=100.0):
    return [Frame(image=np.full(shape, i % 250, np.uint8), seq=i, t_exposure_ns=int(i * 1e9 / fps),
                  t_capture_ns=int(i * 1e9 / fps) + 1_000_000, exposure_us=500.0, gain=1.0) for i in range(n)]


# ------------------------------------------------------------------------------------- ring recorder
def test_ring_recorder_keeps_the_whole_press_and_its_telemetry(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.record.preroll_s, cfg.record.postroll_s = 0.1, 0.05
    done = []
    rec = RingRecorder(cfg, 16, 12, 1, 100.0, on_dump=done.append)
    frames = _frames(200)
    press, release = frames[60].t_capture_ns, frames[150].t_capture_ns  # held 0.9 s: nine times the preroll
    for f in frames:
        if f.t_capture_ns == press:
            rec.hold(press)
        rec.push_telemetry(f.t_capture_ns, {"seq": f.seq, "armed": press <= f.t_capture_ns <= release,
                                            "n_detections": 1, "decision": {"fire": False, "reason": "p_hit 0.12 < 0.50"},
                                            "tracks": [], "stage_ms": {"detect": 3.0}})
        rec.push(f)
        if f.t_capture_ns == release:
            path = rec.trigger(release, tag="press", start_ns=press)
    assert rec.close() == [path] and done == [path]
    seqs = [int(line.split(",")[0]) for line in (path / "frames.csv").read_text().splitlines()[1:]]
    assert seqs == list(range(50, 156))  # 10 preroll + the 91 held frames + 5 postroll
    tele = [json.loads(line) for line in (path / TELEMETRY_JSONL).read_text().splitlines()]
    assert [r["seq"] for r in tele] == seqs
    summary = json.loads((path / SUMMARY_JSON).read_text())
    assert summary["frames"] == 106 and summary["armed_frames"] == 91 and summary["evicted_frames"] == 0
    assert summary["decision_reasons"] == {"p_hit # < #": 91}  # numbers folded: one reason, not one per value
    meta = json.loads((path / "meta.json").read_text())
    assert meta["trigger"]["press_ns"] == [press, release]


def test_ring_recorder_long_press_is_capped_by_ram_not_by_the_preroll(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.record.preroll_s, cfg.record.postroll_s, cfg.record.max_ram_mb = 0.05, 0.0, 1
    rec = RingRecorder(cfg, 256, 256, 1, 100.0)  # 64 kB frames: 1 MB holds 16
    frames = _frames(60, shape=(256, 256))
    rec.hold(frames[10].t_capture_ns)
    for f in frames:
        rec.push(f)
    path = rec.trigger(frames[-1].t_capture_ns, tag="press", start_ns=frames[10].t_capture_ns)
    rec.close()
    summary = json.loads((path / SUMMARY_JSON).read_text())
    assert summary["frames"] == 16 and summary["evicted_frames"] > 0


# ------------------------------------------------------------------------------------- pipeline + clip
def test_pipeline_press_becomes_a_session_with_an_annotated_clip(tmp_path):
    cfg = _cfg(tmp_path, **{"camera.fps": 60.0})
    cfg.record.mode, cfg.record.clip = "armed", "inline"
    cfg.record.preroll_s, cfg.record.postroll_s = 0.2, 0.1
    calib = Calibration(dir=tmp_path / "calib")
    gpio = SimGpio()
    pressed, released = (0, 1) if cfg.arming.active_low else (1, 0)
    gpio.inject(int(cfg.arming.pin), released)
    pipe = Pipeline(cfg, calib, open_source("synthetic:skeet_crossing", cfg, calib), gpio=gpio)
    pipe.start()
    for n in range(110):
        if n == 50:
            gpio.inject(int(cfg.arming.pin), pressed)
            time.sleep(0.03)  # the arming debounce
        if n == 90:
            gpio.inject(int(cfg.arming.pin), released)
        tele = pipe.step()
        assert tele is not None
    pipe.close()
    sessions = sorted((tmp_path / "sessions").iterdir())
    assert len(sessions) == 1 and sessions[0].name.endswith("-press")
    s = sessions[0]
    for name in ("frames.bin", "frames.csv", "imu.csv", "events.jsonl", "meta.json", TELEMETRY_JSONL, SUMMARY_JSON):
        assert (s / name).exists(), name
    tele = [json.loads(line) for line in (s / TELEMETRY_JSONL).read_text().splitlines()]
    armed = [r["armed"] for r in tele]
    assert any(armed) and not armed[0] and not armed[-1]  # preroll and postroll around the held frames
    seen = [r for r in tele if r["dets"]]
    assert seen, "the clay was detected during the press"
    u, v, d, snr, _c, x, y, w, h, _a = seen[0]["dets"][0]
    assert x - 1 <= u <= x + w + 1 and y - 1 <= v <= y + h + 1 and d > 0 and snr > 0
    assert len(seen[0]["R_wg"]) == 9 and "detector" in seen[0]
    clips = [p for p in s.iterdir() if p.stem == "clip"]
    assert len(clips) == 1 and clips[0].stat().st_size > 1000
    import cv2

    cap = cv2.VideoCapture(str(clips[0]))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, image = cap.read()
    cap.release()
    assert ok and image.shape[:2] == (cfg.camera.height, cfg.camera.width)
    assert n_frames == len((s / "frames.csv").read_text().splitlines()) - 1  # every recorded frame is in the clip


def test_draw_record_matches_the_live_overlay_inputs():
    from tbot.app import overlay

    cfg = load_config()
    cam = CameraModel.nominal(cfg, 640, 400)
    mount = MountModel.from_config(cfg, None)
    rec = {"seq": 3, "armed": True, "fired": False, "fps": 99.0, "age_ms": 12.0, "n_detections": 1, "dropped": 0,
           "dets": [[120.0, 90.0, 9.0, 30.0, -40.0, 116, 87, 8, 6, 40.0]], "R_wg": np.eye(3).ravel().tolist(),
           "tracks": [{"id": 1, "az_deg": 0.0, "el_deg": 0.0, "range_m": 20.0, "confirmed": True,
                       "s": [20.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]}],
           "decision": {"fire": False, "reason": "p_hit 0.10 < 0.50", "miss_deg": 1.0, "pattern_deg": 0.8,
                        "sigma_deg": 0.5, "p_hit": 0.1, "lead_dir_w": [1.0, 0.01, 0.0], "bore_dir_w": [1.0, 0.0, 0.0]},
           "static": [[500.0, 300.0]], "exposure_us": 500.0, "gain": 1.0, "stage_ms": {"detect": 4.0}}
    gray = np.full((400, 640), 128, np.uint8)
    out = overlay.draw_record(gray, json.loads(json.dumps(rec)), cam, mount, banner="t+0.100s")
    assert out.shape == (400, 640, 3) and out.dtype == np.uint8
    assert (out == overlay.C_DET).all(axis=2).sum() >= 20          # the detection's bounding box is drawn
    assert (out != 128).any(axis=2).sum() > 500                  # track ring, lead cross, text...


# ------------------------------------------------------------------------------------- detector scale
def _disc_frame(cx, cy, d, contrast=-45.0, shape=(400, 640), seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    ss = 8
    img = np.full(shape, 170.0, np.float32)
    half = int(d) + 6
    yy, xx = np.mgrid[0:2 * half * ss, 0:2 * half * ss].astype(np.float32)
    x0, y0 = int(cx) - half, int(cy) - half
    r2 = ((xx + 0.5) / ss - 0.5 + x0 - cx) ** 2 + ((yy + 0.5) / ss - 0.5 + y0 - cy) ** 2
    m = cv2.resize((r2 <= (d / 2) ** 2).astype(np.float32), (2 * half, 2 * half), interpolation=cv2.INTER_AREA)
    img[y0:y0 + 2 * half, x0:x0 + 2 * half] += contrast * m
    img += rng.normal(0.0, 2.0, shape).astype(np.float32)
    return Frame(image=np.clip(img, 0, 255).astype(np.uint8), seq=0, t_exposure_ns=0, t_capture_ns=0,
                 exposure_us=500.0, gain=1.0)


def test_auto_downscale_follows_the_lens():
    cfg = load_config()
    det = ClayDetector(cfg)
    det.detect(_disc_frame(300.3, 200.7, 5.0))
    assert det.scale == 1                      # 2.8 mm kit lens: a clay at 40 m is 1.3 px
    cfg.optics.focal_length_mm = 12.0
    det = ClayDetector(cfg)
    det.detect(_disc_frame(300.3, 200.7, 12.0))
    assert det.scale == 2                      # 12 mm: 5.4 px at 40 m -> half resolution keeps 2.7 px


@pytest.mark.parametrize("d", [6.0, 12.0, 20.0, 27.0])
def test_downscaled_detection_reports_full_resolution_pixels(d):
    cfg = load_config(sets=["detector.downscale=2"])
    f = _disc_frame(301.4, 187.8, d, seed=int(d))
    dets = ClayDetector(cfg).detect(f)
    assert len(dets) == 1
    det = dets[0]
    assert math.hypot(det.u - 301.4, det.v - 187.8) < 0.5
    assert abs(det.diameter_px - d) < 0.08 * d + 0.3
    x, y, w, h = det.bbox
    assert x <= 301.4 <= x + w and y <= 187.8 <= y + h and abs(w - d) <= 4 and abs(h - d) <= 4
    assert 0.6 * math.pi * d * d / 4 < det.area_px < 1.6 * math.pi * d * d / 4


def test_a_clay_just_above_a_dark_horizon_is_found_at_half_resolution():
    f = _disc_frame(320.0, 250.0, 10.0)
    f.image[262:, :] = 60                      # ground: a step edge 7 px under the clay
    cfg = load_config(sets=["detector.downscale=2"])
    dets = ClayDetector(cfg).detect(f)         # bg_mode auto -> the edge-preserving background
    assert any(math.hypot(d.u - 320.0, d.v - 250.0) < 1.0 for d in dets)
    cfg.detector.bg_mode = "box"               # the box background mixes sky and ground there and loses it
    assert not any(math.hypot(d.u - 320.0, d.v - 250.0) < 1.0 for d in ClayDetector(cfg).detect(f))


# ------------------------------------------------------------------------------------- scenery filter
def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pixels(cam, mount, R_wg, dirs_w):
    d_c = np.asarray(mount.gun_to_cam(np.asarray(dirs_w) @ R_wg))
    return np.asarray(cam.ray_to_pixel(d_c)).reshape(-1, 2)


def test_scenery_filter_drops_world_fixed_candidates_and_keeps_the_mover():
    cfg = load_config(sets=["optics.model=pinhole", "optics.focal_length_mm=12.0"])
    cam = CameraModel.nominal(cfg, 640, 400)
    mount = MountModel.from_config(cfg, None)
    filt = StaticFilter(cfg, cam, mount)
    rng = np.random.default_rng(1)
    fwd = np.array([1.0, 0.0, 0.0])
    scenery = fwd + np.c_[np.zeros(12), rng.uniform(-0.12, 0.12, 12), rng.uniform(-0.07, 0.07, 12)]
    scenery /= np.linalg.norm(scenery, axis=1, keepdims=True)
    kept_clay = dropped = seen = 0
    for i in range(60):
        t = i * 0.01
        yaw = 0.5 * t                                        # the gun swings at 0.5 rad/s ...
        R = _rot_z(yaw)
        clay = np.array([math.cos(yaw + 0.01), math.sin(yaw + 0.01), 0.02])  # ... with the clay: still in the image
        uv = _pixels(cam, mount, R, np.vstack([scenery, clay]))
        uv[:-1] += rng.normal(0.0, 0.7, (12, 2))             # centroid jitter
        inside = (uv[:, 0] > 5) & (uv[:, 0] < 634) & (uv[:, 1] > 5) & (uv[:, 1] < 394)
        idx = np.flatnonzero(inside)
        filt.begin(int(t * 1e9), R)
        keep = idx[filt.filter(uv[idx], np.full(idx.size, 30.0), np.full(idx.size, 6), np.full(idx.size, 6))]
        if i >= 20:                                          # history has built up
            assert 12 in keep, "the clay must survive"
            kept_clay += 1
            visible = [j for j in idx if j != 12]
            seen += len(visible)
            dropped += sum(1 for j in visible if j not in keep)
    assert kept_clay == 40
    assert dropped >= 0.9 * seen                             # scenery that just entered the view is not yet known


def test_scenery_filter_allows_for_parallax_of_near_scenery():
    """A fence post at 8 m shifts against the swing because the camera is ahead of the shooter's pivot."""
    cfg = load_config(sets=["optics.model=pinhole", "optics.focal_length_mm=12.0"])
    cam = CameraModel.nominal(cfg, 640, 400)
    mount = MountModel.from_config(cfg, None)
    lever = float(cfg.detector.static_lever_m)
    post_w = np.array([8.0, 0.3, 0.2])                       # world position, pivot at the origin
    filt = StaticFilter(cfg, cam, mount)
    hits = 0
    for i in range(40):
        t = i * 0.01
        R = _rot_z(0.6 * t)
        cam_pos = R @ np.array([lever, 0.0, 0.0])
        d = post_w - cam_pos
        uv = _pixels(cam, mount, R, (d / np.linalg.norm(d))[None, :])
        filt.begin(int(t * 1e9), R)
        keep = filt.filter(uv, np.array([30.0]), np.array([6]), np.array([6]))
        if i >= 15:
            hits += int(len(keep) == 0)
    assert hits == 25


def test_pipeline_arms_the_scenery_filter_only_with_a_real_attitude(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.record.mode = "off"
    calib = Calibration(dir=tmp_path / "calib")
    pipe = Pipeline(cfg, calib, open_source("synthetic:skeet_crossing", cfg, calib), gpio=SimGpio())
    with pipe:
        for _ in range(60):
            tele = pipe.step()
        assert pipe.detector.candidate_filter is not None and tele["static"] is not None
    cfg.imu.enabled = False
    from tbot.imu.mpu6050 import NullImu

    pipe = Pipeline(cfg, calib, open_source("synthetic:skeet_crossing", cfg, calib), gpio=SimGpio(), imu=NullImu())
    with pipe:
        for _ in range(5):
            tele = pipe.step()
        assert pipe.detector.candidate_filter is None and tele["static"] is None


# ------------------------------------------------------------------------------------- sensor process
def test_sensor_process_delivers_the_same_frames_and_imu(tmp_path):
    from tbot.camera.proc import SensorProcess

    cfg = _cfg(tmp_path, **{"camera.fps": 50.0})
    calib = Calibration(dir=tmp_path / "calib")
    ref = open_source("synthetic:static_gun", cfg, calib)
    ref.start()
    proc = SensorProcess(cfg, calib, "synthetic:static_gun")
    try:
        assert (proc.width, proc.height, proc.is_color) == (ref.width, ref.height, ref.is_color)
        imu = proc.imu()
        assert imu is not None
        frames, samples = [], []
        while len(frames) < 20:
            f = proc.read(timeout_s=5.0)
            assert f is not None
            frames.append(f)
            samples += imu.read()
            time.sleep(0.002)
        seqs = [f.seq for f in frames]
        assert seqs == sorted(set(seqs))
        want = None
        for _ in range(frames[-1].seq + 1):
            want = ref.read()
        assert np.array_equal(frames[-1].image, want.image) and frames[-1].t_exposure_ns == want.t_exposure_ns
        assert samples and all(s.gyro.shape == (3,) and s.accel.shape == (3,) for s in samples)
        assert [s.t_ns for s in samples] == sorted(s.t_ns for s in samples)
    finally:
        proc.stop()
    assert not proc._proc.is_alive()
