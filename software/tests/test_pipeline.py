"""[app] slice: closed-loop sim evaluation, watchdog, debug server, overlay."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from tbot.app import overlay
from tbot.app.run import Pipeline
from tbot.app.stream import DebugServer
from tbot.config import load_calibration, load_config
from tbot.types import Detection, FireDecision, MonotonicClock, Frame, Track

WATCHDOG_MS = 60.0


def base_cfg(tmp_path):
    """Default config with an empty calibration dir, simulated GPIO and no recording/streaming."""
    cfg = load_config()
    cfg.paths.calib_dir = str(tmp_path / "calib")
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.stream.enabled = False
    cfg.imu.enabled = False
    return cfg, load_calibration(cfg)


class FakeActuator:
    """Records commands; never moves anything."""

    kind = "fake"

    def __init__(self):
        self.busy = False
        self.last_fire_ns = None
        self.fires: list[int] = []
        self.aborts: list[float] = []
        self.rests = 0
        self.closed = False

    def rest(self) -> None:
        self.rests += 1

    def fire(self, t_ns: int) -> bool:
        self.fires.append(int(t_ns))
        self.last_fire_ns = int(t_ns)
        return True

    def abort(self) -> None:
        self.aborts.append(time.monotonic())

    def close(self) -> None:
        self.closed = True


class StallingSource:
    """Delivers n_frames of grey noise, then stalls (read() times out forever)."""

    is_color = False
    name = "stalling"

    def __init__(self, n_frames: int, width: int = 160, height: int = 100, fps: float = 50.0):
        self.width = width
        self.height = height
        self.nominal_fps = fps
        self.clock = MonotonicClock()
        self.dropped = 0
        self.n_frames = n_frames
        self.seq = 0
        self.last_frame_wall = None
        self._rng = np.random.default_rng(7)
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def imu(self):
        return None

    def read(self, timeout_s: float = 1.0):
        if self.seq >= self.n_frames:
            time.sleep(min(timeout_s, 0.01))
            return None
        img = np.clip(self._rng.normal(170.0, 2.0, (self.height, self.width)), 0, 255).astype(np.uint8)
        t = self.clock.now_ns()
        frame = Frame(image=img, seq=self.seq, t_exposure_ns=t, t_capture_ns=t, exposure_us=500.0, gain=1.0)
        self.seq += 1
        self.last_frame_wall = time.monotonic()
        time.sleep(1.0 / self.nominal_fps)
        return frame


# --------------------------------------------------------------------------------------------- sim
def test_fire_path_commands_the_actuator_once_per_press(tmp_path):
    """End-to-end fire path, independent of tracking ACCURACY (vision quality is benchmarked by `tbot sim`).

    The probability gate is lowered to 0.05 so the solver reaches a fire decision whatever the current
    tracker/detector accuracy is; what is asserted here is the integration: one press at ONE clay -> exactly
    one command accepted by the actuator layer even though the press allows max_shots_per_arm (a clay that
    was shot at is never engaged again), only while armed, after the track is confirmed, and with a
    complete shot record.
    """
    from tbot.tools.sim_eval import evaluate_run

    cfg, calib = base_cfg(tmp_path)
    cfg.camera.fps = 50.0
    cfg.fire.min_hit_probability = 0.05  # gate reachable regardless of vision accuracy (not a quality claim)
    cfg.fire.max_shots_per_arm = 3       # the press allows several shots; one clay must still get one
    res = evaluate_run(cfg, calib, "skeet_crossing", 1)

    assert res["visible_frames"] > 0, "the clay is never visible: scenario/renderer problem"
    assert res["fires"] >= 1, f"never fired; decision reasons: {res['reasons']}"
    assert res["actuator_kind"] == "sim"
    assert cfg.fire.max_shots_per_arm > 1
    assert len(res["actuator_fire_times"]) == 1, "one clay must get exactly one shot per press"
    assert res["actuator_fire_times"] == [s["t_fire_ns"] for s in res["shots"]]
    assert res["t_first_confirmed_ns"] is not None
    for shot in res["shots"]:
        assert shot["armed_at_fire"] is True, "a shot was commanded while the interlock was open"
        assert shot["t_fire_ns"] > res["t_first_confirmed_ns"], "fired before a track was confirmed"
        assert shot["t_exit_ns"] > shot["t_fire_ns"], "the shot must leave after the command"
        assert math.isfinite(shot["miss_m"]) and math.isfinite(shot["pattern_radius_m"])
        assert shot["range_m"] > 0.0 and shot["tof_s"] > 0.0
        assert math.isfinite(shot["predicted"]["p_hit"])


def test_no_clay_scenario_never_fires(tmp_path):
    from tbot.tools.sim_eval import evaluate_run

    cfg, calib = base_cfg(tmp_path)
    cfg.camera.fps = 30.0
    res = evaluate_run(cfg, calib, "no_clay", 1)

    assert res["frames"] > 0
    assert res["fires"] == 0
    assert res["shots"] == []


def test_interlock_open_blocks_fire_but_still_decides(tmp_path):
    from tbot.tools.sim_eval import evaluate_run

    cfg, calib = base_cfg(tmp_path)
    cfg.camera.fps = 50.0
    res = evaluate_run(cfg, calib, "skeet_crossing", 1, armed=False, max_frames=40)

    assert res["frames"] == 40
    assert res["armed_frames"] == 0
    assert res["fires"] == 0
    assert sum(res["reasons"].values()) == res["frames"], "every frame must produce a decision"
    assert any("arm" in r.lower() for r in res["reasons"]), res["reasons"]


def test_dry_run_never_commands_the_actuator(tmp_path):
    from tbot.camera.synthetic import SyntheticSource
    from tbot.hw.gpio import open_gpio

    cfg, calib = base_cfg(tmp_path)
    cfg.camera.fps = 50.0
    cfg.fire.live = False
    cfg.fire.min_hit_probability = 0.05  # reach a fire decision regardless of vision accuracy
    source = SyntheticSource(cfg, calib, scenario="skeet_crossing", seed=1)
    gpio = open_gpio(cfg)
    act = FakeActuator()
    pipe = Pipeline(cfg, calib, source, gpio=gpio, actuator=act)
    gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)

    dry_fires = 0
    try:
        while True:
            tele = pipe.step()
            if tele is None:
                break
            dry_fires += int(tele["fired"])
    finally:
        pipe.close()

    assert dry_fires >= 1, "the solver never reached a fire decision, so the dry-run check proves nothing"
    assert act.fires == [], "fire.live is false: the actuator must never be commanded to fire"
    assert act.rests == 0


# --------------------------------------------------------------------------------------------- watchdog
def test_watchdog_aborts_when_frames_stop(tmp_path):
    cfg, calib = base_cfg(tmp_path)
    cfg.fire.watchdog_ms = WATCHDOG_MS
    source = StallingSource(n_frames=3)
    act = FakeActuator()
    pipe = Pipeline(cfg, calib, source, actuator=act)

    rc = pipe.run(status=False, idle_end_s=0.5)

    assert rc == 0
    assert pipe.frames == 3
    assert act.aborts, "the watchdog never aborted the actuator"
    delay_s = act.aborts[0] - source.last_frame_wall
    assert delay_s <= 2.5 * WATCHDOG_MS / 1000.0, f"watchdog fired {delay_s * 1e3:.0f} ms after the last frame"
    assert pipe.watchdog_aborts >= 1


def test_telemetry_shape_from_a_live_step(tmp_path):
    cfg, calib = base_cfg(tmp_path)
    source = StallingSource(n_frames=2)
    pipe = Pipeline(cfg, calib, source)
    try:
        tele = pipe.step()
    finally:
        pipe.close()

    assert tele is not None
    assert tele["seq"] == 0 and tele["frames"] == 1
    assert tele["armed"] is False and tele["fired"] is False and tele["live"] is False
    assert tele["age_ms"] >= 0.0
    assert set(tele["stage_ms"]) >= {"read", "imu", "detect", "track", "solve", "total"}
    assert tele["decision"]["fire"] is False and tele["decision"]["reason"]
    assert json.loads(json.dumps(tele))  # no numpy objects may leak into the telemetry
    assert tele["ae"] is None            # this source has no auto-exposure


def test_exposure_out_of_range_is_in_the_telemetry_and_warned_once(tmp_path, caplog):
    from tbot.camera.autoexposure import AeStatus

    cfg, calib = base_cfg(tmp_path)
    source = StallingSource(n_frames=2)
    pipe = Pipeline(cfg, calib, source)
    try:
        source.ae_status = AeStatus("ok", 160.0, 0.0, 0.0, 120.0, 1.0)
        assert pipe._check_exposure(0)["state"] == "ok"
        source.ae_status = AeStatus("too_bright", 255.0, 60.0, -1.6, 9.0, 1.0)   # sky clipped at the shortest exposure
        with caplog.at_level("WARNING"):
            first = pipe._check_exposure(int(1e9))
            pipe._check_exposure(int(2e9))               # 1 s of it: could be a glance at the sun
            assert not caplog.records
            pipe._check_exposure(int(3.1e9))             # still there after 2 s: say so ...
            pipe._check_exposure(int(9e9))               # ... once
        assert first == {"state": "too_bright", "level_dn": 255.0, "clip_pct": 60.0, "error_stops": -1.6}
        assert len(caplog.records) == 1 and "ND filter" in caplog.records[0].getMessage()
        assert json.loads(json.dumps(first))
    finally:
        pipe.close()


# --------------------------------------------------------------------------------------------- stream
def test_debug_server_snapshot_and_telemetry(tmp_path):
    cfg, _ = base_cfg(tmp_path)
    cfg.stream.host = "127.0.0.1"
    cfg.stream.port = 0
    cfg.stream.fps = 0.5  # 2 s between frames: the first encode on a busy Pi 3B can take a good share of 0.2 s
    srv = DebugServer(cfg).start()
    try:
        assert srv.port > 0
        img = np.zeros((32, 48, 3), dtype=np.uint8)
        img[:16] = 200
        tele = {"fps": np.float32(12.5), "armed": True, "miss_deg": float("nan"), "tracks": [{"id": 1}]}
        assert srv.publish(img, tele) is True
        assert srv.publish(img, tele) is False, "publish must be rate limited to stream.fps"

        base = f"http://127.0.0.1:{srv.port}"
        with urllib.request.urlopen(base + "/snapshot.jpg", timeout=5) as r:
            data = r.read()
            assert r.headers["Content-Type"] == "image/jpeg"
        assert data[:2] == b"\xff\xd8", "not a JPEG"

        with urllib.request.urlopen(base + "/telemetry", timeout=5) as r:
            got = json.loads(r.read().decode())
        assert got["armed"] is True
        assert got["fps"] == pytest.approx(12.5)
        assert got["miss_deg"] is None, "NaN must be encoded as null so browsers can parse it"
        assert got["tracks"] == [{"id": 1}]

        with urllib.request.urlopen(base + "/", timeout=5) as r:
            assert b"stream.mjpg" in r.read()
    finally:
        srv.stop()


def test_debug_server_snapshot_before_any_publish(tmp_path):
    cfg, _ = base_cfg(tmp_path)
    cfg.stream.host = "127.0.0.1"
    cfg.stream.port = 0
    srv = DebugServer(cfg).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/snapshot.jpg", timeout=5)
        assert e.value.code == 503
    finally:
        srv.stop()


# --------------------------------------------------------------------------------------------- overlay
def _scene():
    det = Detection(
        t_ns=1_000, u=22.0, v=18.0, area_px=9.0, diameter_px=3.4, contrast=-40.0, snr=12.0, score=0.8,
        bbox=(20, 16, 4, 4),
    )
    track = Track(
        track_id=3, t_ns=1_000, x=np.array([0.10, 0.9, 0.0, 0.05, 0.2, 0.0]), P=np.eye(6) * 1e-6,
        range_m=21.0, range_sigma_m=3.0, hits=7, confirmed=True,
    )
    decision = FireDecision(
        fire=True, t_ns=1_200, reason="FIRE", track_id=3, t_exit_ns=1_300, miss_rad=0.001,
        pattern_rad=0.016, sigma_rad=0.004, p_hit=0.93, range_m=21.0, tof_s=0.062,
        lead_dir_w=np.array([1.0, 0.12, 0.02]), bore_dir_w=np.array([1.0, 0.11, 0.015]),
    )
    telemetry = {
        "fps": 48.5, "age_ms": 12.0, "seq": 11, "n_detections": 1, "dropped": 0, "shots": 1, "armed": True,
        "stage_ms": {"read": 1.0, "detect": 3.2, "track": 0.4, "solve": 0.2},
        "tracks": [{"id": 3}],
    }

    def project_w(d):
        d = np.asarray(d, dtype=np.float64).reshape(3)
        return (30.0 + 120.0 * float(d[1]), 20.0 - 120.0 * float(d[2]))

    return det, track, decision, telemetry, project_w


@pytest.mark.parametrize("color", [False, True])
def test_overlay_draw_returns_bgr_of_input_size(color):
    det, track, decision, telemetry, project_w = _scene()
    rng = np.random.default_rng(3)
    mono = rng.integers(0, 255, (40, 60), dtype=np.uint8)
    image = np.dstack([mono, mono, mono]) if color else mono
    mask = np.full((40, 60), 255, dtype=np.uint8)
    mask[:6, :] = 0
    before = image.copy()

    out = overlay.draw(
        image,
        detections=[det],
        tracks=[track],
        decision=decision,
        mask=mask,
        bore_px=(30.0, 20.0),
        telemetry=telemetry,
        project_w=project_w,
        track_lead={3: np.array([1.0, 0.2, 0.03])},
        armed=True,
        fired=True,
    )

    assert out.shape == (40, 60, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(image, before), "draw must not modify the caller's frame"
    assert not np.array_equal(out[:, :, 0], mono), "nothing was drawn"
    assert out[2, 30, 0] != mono[2, 30], "the masked region must be tinted"


def test_overlay_draw_tolerates_missing_everything():
    img = np.zeros((20, 30), dtype=np.uint8)
    out = overlay.draw(img, detections=[], tracks=[], decision=None, mask=None, bore_px=None, telemetry=None)
    assert out.shape == (20, 30, 3)


def test_overlay_draw_skips_unprojectable_directions():
    det, track, decision, telemetry, _ = _scene()

    out = overlay.draw(
        np.zeros((40, 60), dtype=np.uint8),
        detections=[det],
        tracks=[track],
        decision=decision,
        telemetry=telemetry,
        project_w=lambda d: None,  # everything is behind the camera
        bore_px=(float("nan"), 3.0),
    )
    assert out.shape == (40, 60, 3)


def test_no_shot_while_the_imu_stream_is_stopped(tmp_path):
    """An IMU that stops delivering (I2C fault, sensor process being replaced) freezes the attitude and the
    swing the solver extrapolates: the pipeline must refuse to fire rather than shoot on stale motion."""
    from tbot.camera.synthetic import SyntheticSource
    from tbot.hw.gpio import open_gpio

    cfg, calib = base_cfg(tmp_path)
    cfg.camera.fps = 50.0
    cfg.fire.live = False
    cfg.fire.min_hit_probability = 0.05
    source = SyntheticSource(cfg, calib, scenario="skeet_crossing", seed=1)
    gpio = open_gpio(cfg)
    pipe = Pipeline(cfg, calib, source, gpio=gpio, actuator=FakeActuator())
    gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)
    real_read = pipe.imu.read
    frames = [0]

    def read():
        samples = real_read()
        return samples if frames[0] < 15 else []   # the stream stops before the clay can be shot at

    pipe.imu.read = read
    fired, stale = 0, 0
    try:
        while (tele := pipe.step()) is not None:
            frames[0] += 1
            fired += int(tele["fired"])
            stale += int(tele["decision"]["reason"].startswith("IMU stale"))
            if frames[0] > 15:
                assert tele["imu_age_ms"] is None or tele["imu_age_ms"] > 0
    finally:
        pipe.close()
    assert fired == 0
    assert stale >= 1, "the solver never wanted to fire, so the check proves nothing"
