"""Live pipeline: frame -> IMU -> attitude -> detect -> track -> fire solver -> actuator (`tbot run`).

Safety (docs/ARCHITECTURE.md): cfg.fire.live is false by default and the pipeline then only logs
would-fire decisions; the arming interlock is enforced by the arming switch and the actuator layer; a
watchdog aborts the actuator when frames stop arriving; shutdown always aborts and rests the actuator.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from tbot.app import overlay as overlay_mod
from tbot.app.stream import DebugServer, jsonable
from tbot.camera import open_live_source
from tbot.config import (
    Calibration,
    Config,
    LatencyBudget,
    add_common_args,
    load_from_args,
    resolve_path,
)
from tbot.fire.ballistics import ShotModel
from tbot.fire.engagement import Engagement
from tbot.fire.recoil import RecoilDetector, summarise
from tbot.fire.solver import FireSolver
from tbot.hw.actuator import make_actuator
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import SimGpio, open_gpio
from tbot.hw.health import read_health
from tbot.imu.attitude import AttitudeEstimator, FixedAttitude
from tbot.imu.mpu6050 import NullImu, open_imu
from tbot.optics import CameraModel, MountModel, combined_mask
from tbot.types import Frame, MonotonicClock
from tbot.vision.clutter import StaticFilter
from tbot.vision.detector import ClayDetector
from tbot.vision.tracker import ClayTracker  # noqa: F401 - re-exported for callers
from tbot.vision.tracker3d import make_tracker

log = logging.getLogger("tbot.run")

LEAD_PREVIEW_S = overlay_mod.LEAD_PREVIEW_S  # overlay velocity arrow horizon
FPS_EMA_ALPHA = 0.1
SOLVE_EMA_ALPHA = 0.1
EXPOSURE_WARN_S = 2.0  # the scene must be out of the camera's exposure range this long before it is reported
HOT_TEMP_C = 75.0  # warn before the Pi 3B firmware throttles at 80 C
HEALTH_PERIOD_S = 2.0


def _ms(t0_ns: int, t1_ns: int) -> float:
    return (t1_ns - t0_ns) / 1e6


class Pipeline:
    """One frame in, one telemetry dict out; owns every per-run object the stages need."""

    def __init__(
        self,
        cfg: Config,
        calib: Calibration,
        source,
        gpio=None,
        actuator=None,
        arming=None,
        imu=None,
    ):
        self.cfg = cfg
        self.calib = calib
        self.source = source
        width, height = int(source.width), int(source.height)
        self.cam = CameraModel.from_calibration(calib.intrinsics, cfg, width, height)
        self.mount = MountModel.from_config(cfg, calib.boresight)
        self.mask = combined_mask(self.cam, self.mount, cfg, calib)
        self.detector = ClayDetector(cfg, self.mask)
        self.tracker = make_tracker(cfg, self.cam, self.mount)
        self.clutter = StaticFilter(cfg, self.cam, self.mount) if bool(cfg.detector.static_reject) else None
        self._movers: set[int] = set()  # tracks seen crossing the sky: the scenery filter never takes them
        self.shot = ShotModel(cfg)
        self.budget: LatencyBudget = calib.budget or LatencyBudget.from_config(cfg, calib.latency)
        self.solver = FireSolver(cfg, self.shot, self.mount, self.budget)
        self.engagement = Engagement(cfg, self.tracker)
        self.recoil = RecoilDetector.from_config(cfg)
        self.recoil_delays_ms: list[float] = []
        self.no_recoil = 0

        self._own_imu = imu is None
        self.imu = imu if imu is not None else open_imu(cfg, calib, source)
        self.attitude = FixedAttitude() if isinstance(self.imu, NullImu) else AttitudeEstimator(cfg)

        self._own_gpio = gpio is None
        self.gpio = gpio if gpio is not None else open_gpio(cfg)
        self.arming = arming if arming is not None else ArmingSwitch(self.gpio, cfg)

        self._own_actuator = actuator is None
        if actuator is not None:
            self.actuator = actuator
        elif cfg.fire.live:
            self.actuator = make_actuator(cfg, self.gpio, self.arming)
        else:
            self.actuator = None
            log.info("dry run (fire.live = false): decisions are logged, the actuator is never driven")

        self.nominal_fps = float(source.nominal_fps) or float(cfg.camera.fps)
        if self.nominal_fps <= 0:
            self.nominal_fps = 30.0
        self.frame_period_s = 1.0 / self.nominal_fps
        self.read_timeout_s = max(0.1, 5.0 * self.frame_period_s)
        self.bore_px = self.mount.bore_pixel(self.cam, float(cfg.tracker.range_prior_m))

        # per-frame state (read by the overlay, the tools and the tests)
        self.frames = 0
        self.shots = 0
        self.fps = 0.0
        self.detections: list = []
        self.tracks: list = []
        self.decision = None
        self.last_telemetry: dict | None = None
        self.events: list[dict] = []
        self.health: dict = {}
        self._health_flags = {"under_voltage_now": False, "throttled_now": False, "soft_temp_limit_now": False,
                              "hot": False}
        self._ae_state = ""
        self._ae_since_ns = 0
        self._ae_warned = False

        self.server: DebugServer | None = None
        self.recorder = None
        self._recorder_is_writer = False
        self._press_ns: int | None = None  # source-clock time the current press began (armed recording)
        self._clip_procs: list = []
        self._display = False
        self._window = "tbot"
        self._window_open = False
        self._jsonl = None

        self._started = False
        self._closed = False
        self._stop = threading.Event()
        self._prev_exposure_ns: int | None = None
        self._last_step_ns: int | None = None
        self._expected_solve_ns = 0.0
        self._last_frame_wall: float | None = None
        self._wd_thread: threading.Thread | None = None
        self._wd_stop = threading.Event()
        self._wd_tripped = False
        self.watchdog_aborts = 0
        self._status_wall = 0.0
        self._sig_prev: list = []
        self._health_thread: threading.Thread | None = None
        self._health_stop = threading.Event()

    # ------------------------------------------------------------------ exposure
    def _check_exposure(self, t_ns: int) -> dict | None:
        """Auto-exposure state for the telemetry; warns once each time the scene stays brighter or darker than
        the camera can expose for (shortest exposure still clipped / longest exposure and full gain still dark)."""
        st = getattr(self.source, "ae_status", None)
        if st is None:
            return None
        out_of_range = st.state in ("too_bright", "too_dark")
        if not out_of_range or st.state != self._ae_state:
            self._ae_since_ns, self._ae_warned = t_ns, False
        elif not self._ae_warned and t_ns - self._ae_since_ns >= EXPOSURE_WARN_S * 1e9:
            self._ae_warned = True
            log.warning("exposure: %s", st.message())
        self._ae_state = st.state
        return {"state": st.state, "level_dn": round(float(st.level_dn), 1), "clip_pct": round(float(st.clip_pct), 2),
                "error_stops": round(float(st.error_stops), 2)}

    # ------------------------------------------------------------------ Pi health
    _HEALTH_WARNINGS = (
        ("under_voltage_now", "UNDER-VOLTAGE on the Pi supply (sagging pack or thin wiring)"),
        ("throttled_now", "CPU THROTTLED (hot pod or low supply voltage)"),
        ("soft_temp_limit_now", "SOFT TEMPERATURE LIMIT reached (clock reduced: hot pod)"),
    )

    def _check_health(self, health: dict) -> None:
        """Warn once on each rising edge of an under-voltage / throttling condition."""
        flags = (health or {}).get("throttled") or {}
        for key, message in self._HEALTH_WARNINGS:
            active = bool(flags.get(key))
            if active and not self._health_flags[key]:
                log.warning("health: %s (source %s)", message, (health or {}).get("source"))
            self._health_flags[key] = active
        # early heat warning: the Pi 3B firmware throttles the ARM clock at 80 C, which costs frame rate
        temp = (health or {}).get("cpu_temp_c")
        if temp is not None:
            hot = bool(self._health_flags.get("hot")) and temp > HOT_TEMP_C - 3.0 or temp >= HOT_TEMP_C
            if hot and not self._health_flags.get("hot"):
                log.warning("health: Pi at %.1f C (throttling starts at 80 C): shade/ventilate the pod", temp)
            self._health_flags["hot"] = hot

    def _health_loop(self) -> None:
        """Pi temperature / throttle flags / ARM clock every HEALTH_PERIOD_S, on its own thread: the clock comes
        from a `vcgencmd` child process, and starting one from the frame loop stalled it for 20-40 ms (several
        frames at 100 fps) once a second."""
        while True:
            try:
                self.health = read_health()
            except Exception:  # noqa: BLE001 - read_health never raises; a thread must not die if it ever does
                pass
            if self._health_stop.wait(HEALTH_PERIOD_S):
                return

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the source, the IMU, the debug stream and the recorder (idempotent)."""
        if self._started:
            return
        self._started = True
        self._health_stop.clear()
        self._health_thread = threading.Thread(target=self._health_loop, name="tbot-health", daemon=True)
        self._health_thread.start()
        self.source.start()
        self.imu.start()
        if self.cfg.stream.enabled:
            self.server = DebugServer(self.cfg)
            self.server.clips_dir = resolve_path(self.cfg.record.dir)
            self.server.renders_on_demand = True  # frames are drawn only while somebody is watching (_publish)
            self.server.start()
        self._open_recorder()
        if self.actuator is not None and self.cfg.fire.live:
            self.actuator.rest()  # known state before the first frame; dry run never commands the actuator
        self._last_step_ns = time.monotonic_ns()

    def close(self) -> None:
        """Always-safe shutdown: actuator to rest, then source/IMU/GPIO/recorder/stream down."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wd_stop.set()
        self._health_stop.set()
        if self._wd_thread is not None:
            self._wd_thread.join(timeout=1.0)
            self._wd_thread = None
        if self.actuator is not None:
            try:
                self.actuator.abort()
            finally:
                if self._own_actuator:
                    self.actuator.close()
        try:
            self.source.stop()
        finally:
            self.imu.stop()
        if self.recorder is not None:
            if self._press_ns is not None and not self._recorder_is_writer:
                # stopped with the button still held: the press so far is still a session
                self.recorder.trigger(int(self.source.clock.now_ns()), tag="press", start_ns=self._press_ns)
                self._press_ns = None
            written = self.recorder.close()
            self.recorder = None
            for p in written if isinstance(written, list) else [written]:
                log.info("session written: %s", p)
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        if self._window_open:
            import cv2

            cv2.destroyWindow(self._window)
            self._window_open = False
        if self._own_gpio:
            self.gpio.close()

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ recording
    def _open_recorder(self) -> None:
        cfg = self.cfg
        mode = cfg.record.mode
        if mode == "off":
            return
        from tbot.app.record import RingRecorder, SessionWriter, session_meta_extra

        channels = 3 if bool(self.source.is_color) else 1
        w, h = int(self.source.width), int(self.source.height)
        meta = session_meta_extra(cfg, self.calib, getattr(self.source, "name", None) or cfg.camera.source)
        if mode == "armed":
            self.recorder = RingRecorder(cfg, w, h, channels, self.nominal_fps, meta_extra=meta,
                                         on_dump=self._on_session)
            self._recorder_is_writer = False
            self.arming.on_change(self._on_arming_change)
            log.info("recording every held press (%.1f s before to %.1f s after, clip: %s) to %s",
                     cfg.record.preroll_s, cfg.record.postroll_s, cfg.record.clip, cfg.record.dir)
        elif mode == "continuous":
            path = resolve_path(cfg.record.dir) / (time.strftime("%Y%m%d-%H%M%S") + "-run")
            self.recorder = SessionWriter(path, w, h, channels, self.nominal_fps, meta_extra=meta)
            self._recorder_is_writer = True
            log.info("recording continuously to %s", path)
        else:
            raise ValueError(f"record.mode must be off|armed|continuous, got {mode!r}")

    def _on_arming_change(self, armed: bool, t_ns: int) -> None:
        """Ring recorder: a held press is one session, from the preroll before the button went down to the
        postroll after the operator lets go."""
        if self.recorder is None:
            return
        # The ring windows frames by the SOURCE clock; on hardware it is the same monotonic clock the
        # switch stamps with, on replay/synthetic sources it is not, so window on the source clock.
        now = int(self.source.clock.now_ns())
        self.recorder.event({"t_ns": now, "type": "arming", "armed": bool(armed), "switch_t_ns": int(t_ns)})
        if self._recorder_is_writer:
            return
        if armed:
            self._press_ns = now
            self.recorder.hold(now)
        else:
            start, self._press_ns = self._press_ns, None
            path = self.recorder.trigger(now, tag="press", start_ns=start)
            log.info("press released after %.2f s: recording %s",
                     (now - start) / 1e9 if start is not None else 0.0, path.name)

    def _on_session(self, path: Path) -> None:
        """A press is on the SD card (recorder thread): render its annotated clip without touching the frame loop."""
        mode = str(self.cfg.record.clip)
        log.info("session written: %s", path)
        if mode == "inline":
            from tbot.tools.clip import render

            log.info("clip: %s", render(path))
        elif mode == "process":
            self._clip_procs = [p for p in self._clip_procs if p.poll() is None]
            cmd = [sys.executable, "-m", "tbot.tools.clip", str(path), "--nice", "15"]
            self._clip_procs.append(subprocess.Popen(cmd, cwd=str(resolve_path(".")), stdin=subprocess.DEVNULL))
        elif mode != "off":
            log.warning("record.clip must be off|inline|process, got %r", mode)

    def _record(self, frame: Frame, samples: list, tele: dict | None = None) -> None:
        rec = self.recorder
        if rec is None:
            return
        if self._recorder_is_writer:
            if self.cfg.record.decimate <= 1 or (self.frames % int(self.cfg.record.decimate)) == 0:
                rec.add_frame(frame)
            if samples:
                rec.add_imu(samples)
        else:
            if tele is not None:
                rec.push_telemetry(int(frame.t_capture_ns), tele)
            rec.push(frame)
            if samples:
                rec.push_imu(samples)

    def _on_recoil(self, rec) -> None:
        """A live fire command resolved: the IMU saw the shot (and when), or it did not."""
        expected = self.budget.cmd_to_break_ms[str(self.cfg.actuator.type)] + self.budget.lock_time_ms
        if rec.detected:
            self.recoil_delays_ms.append(float(rec.delay_ms))
            log.info("shot %d: recoil %.1f ms after the command (budget %.1f ms, %+.1f)  peak %.1f g",
                     rec.shot, rec.delay_ms, expected, rec.delay_ms - expected, rec.peak_g)
        else:
            self.no_recoil += 1
            log.warning("shot %d: NO RECOIL within %.0f ms of the command - the gun did not fire (lever slipped, "
                        "safety on, empty chamber or a dud)", rec.shot, self.recoil.window_ns / 1e6)
        ev = rec.as_event()
        ev["expected_ms"] = expected
        self._event(ev)

    def _event(self, ev: dict) -> None:
        self.events.append(ev)
        if self.recorder is not None:
            if self._recorder_is_writer:
                self.recorder.add_event(ev)
            else:
                self.recorder.event(ev)

    # ------------------------------------------------------------------ helpers
    def _projector(self, R_wg: np.ndarray):
        """World direction -> pixel in the frame whose attitude is R_wg (None when behind the camera)."""
        cam, mount = self.cam, self.mount
        R_gw = np.asarray(R_wg, dtype=np.float64).reshape(3, 3).T

        def project(dir_w) -> tuple[float, float] | None:
            if dir_w is None:
                return None
            d_g = R_gw @ np.asarray(dir_w, dtype=np.float64).reshape(3)
            d_c = np.asarray(mount.gun_to_cam(d_g), dtype=np.float64).reshape(1, 3)
            uv = np.asarray(cam.ray_to_pixel(d_c), dtype=np.float64).reshape(-1, 2)
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if not (np.isfinite(u) and np.isfinite(v)):
                return None
            return (u, v)

        return project

    def render_overlay(self, frame: Frame, R_wg: np.ndarray, telemetry: dict) -> np.ndarray:
        """Debug image for the stream / local window (only called when it is going to be shown)."""
        project = self._projector(R_wg)
        lead_ns = int(LEAD_PREVIEW_S * 1e9)
        track_lead = {}
        for t in self.tracks:
            track_lead[t.track_id] = self.tracker.predict_dir(t, int(t.t_ns) + lead_ns)
        return overlay_mod.draw(
            frame.image,
            detections=self.detections,
            tracks=self.tracks,
            decision=self.decision,
            mask=self.mask,
            bore_px=self.bore_px,
            telemetry=telemetry,
            project_w=project,
            track_lead=track_lead,
            armed=bool(telemetry.get("armed", False)),
            fired=bool(telemetry.get("fired", False)),
        )

    def _draw_record(self, image: np.ndarray, record: dict) -> np.ndarray:
        return overlay_mod.draw_record(image, record, self.cam, self.mount, mask=self.mask, bore_px=self.bore_px)

    def _publish(self, frame: Frame, R_wg: np.ndarray, telemetry: dict) -> float:
        """Render + publish/show the overlay when due. Returns the time spent in ms."""
        if self.server is not None and not self._display:
            # stream only: drawn and JPEG-encoded on the stream thread from this frame's record, and only while
            # somebody is watching (10 ms of drawing in the frame loop is a dropped frame at 100 fps)
            self.server.set_telemetry(telemetry)
            if self.server.watched() and self.server.due():
                t0 = time.perf_counter_ns()
                self.server.submit(frame.image, telemetry, self._draw_record)
                return _ms(t0, time.perf_counter_ns())
            return 0.0
        want_stream = self.server is not None and self.server.due()
        if not (want_stream or self._display):
            if self.server is not None:
                self.server.set_telemetry(telemetry)
            return 0.0
        t0 = time.perf_counter_ns()
        bgr = self.render_overlay(frame, R_wg, telemetry)
        if want_stream and self.server is not None:
            self.server.publish(bgr, telemetry)
        if self._display:
            import cv2

            if not self._window_open:
                cv2.namedWindow(self._window)
                self._window_open = True
            cv2.imshow(self._window, bgr)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                log.info("display window: quit requested")
                self._stop.set()
        return _ms(t0, time.perf_counter_ns())

    def _arm_scenery_filter(self, t_ns: int, R_wg: np.ndarray) -> None:
        """Give the detector the scenery filter for this frame, when there is a real attitude to base it on."""
        clutter = self.clutter
        if clutter is None or isinstance(self.attitude, FixedAttitude) or not self.attitude.ready \
                or not bool(self.cfg.detector.static_reject):
            self.detector.candidate_filter = None
            return
        # Tracks that have crossed the sky at several times the filter's own speed floor are clays: keep
        # whatever is detected near them even if they slow to a crawl (going away, or at the top of an arc).
        rate_of = getattr(self.tracker, "angular_rate", None)
        protect = []
        if rate_of is not None:
            det = self.cfg.detector
            floor = 4.0 * float(det.static_tol_px) * clutter.ifov / max(float(det.static_min_age_ms) * 1e-3, 1e-3)
            alive = set()
            for t in self.tracks:
                alive.add(t.track_id)
                if t.confirmed and t.track_id not in self._movers and t.s is not None and rate_of(t) > floor:
                    self._movers.add(t.track_id)
                if t.track_id in self._movers:
                    protect.append(self.tracker.predict_dir(t, t_ns))
            self._movers &= alive
        clutter.begin(t_ns, R_wg, protect_dirs_w=protect)
        self.detector.candidate_filter = clutter.filter

    # ------------------------------------------------------------------ one frame
    def step(self) -> dict | None:
        """Process the next frame. None = no frame (timeout or end of stream)."""
        if not self._started:
            self.start()
        cfg = self.cfg
        t_begin = time.perf_counter_ns()
        frame = self.source.read(timeout_s=self.read_timeout_s)
        t_read = time.perf_counter_ns()
        if frame is None:
            return None
        if getattr(self.source, "gap_after_restart", False):
            # the camera was reopened: tracks from before the outage would be extrapolated across it
            self.source.gap_after_restart = False
            self.tracker.reset()
            if self.clutter is not None:
                self.clutter.reset()
            log.warning("camera restarted (%d so far): tracks reset", int(getattr(self.source, "restarts", 0)))

        samples = self.imu.read()
        t_imu = time.perf_counter_ns()
        if samples:
            self.attitude.update(samples)
            if self.recoil.waiting:
                for rec in self.recoil.feed(samples):
                    self._on_recoil(rec)
        R_wg = np.asarray(self.attitude.R_wg(int(frame.t_exposure_ns)), dtype=np.float64)
        t_att = time.perf_counter_ns()

        self._arm_scenery_filter(int(frame.t_exposure_ns), R_wg)
        self.detections = self.detector.detect(frame)
        t_det = time.perf_counter_ns()

        self.tracks = self.tracker.update(int(frame.t_exposure_ns), self.detections, R_wg)
        t_trk = time.perf_counter_ns()

        solve_start_ns = int(self.source.clock.now_ns())
        decide_ahead_ns = int(self._expected_solve_ns)
        now = solve_start_ns + decide_ahead_ns
        armed = bool(self.arming.armed)
        self.engagement.new_press(int(getattr(self.arming, "press_count", 0)))
        self.engagement.observe(self.tracks, int(frame.t_exposure_ns))
        # the solver looks one decision ahead: that is the real loop period (16-30 fps with trees in view), not
        # the camera's
        period_s = self.frame_period_s if self.fps <= 0.0 else min(0.2, max(self.frame_period_s, 1.0 / self.fps))
        decision = self.solver.evaluate(now, self.tracker, self.attitude, armed, period_s,
                                        accept=self.engagement.accept(now))
        if decision.fire and not self.engagement.spacing_ok(now):
            decision.fire = False
            decision.reason = "shot spacing"
        if decision.fire and self.actuator is not None and cfg.fire.live and bool(getattr(self.actuator, "busy", False)):
            decision.fire = False  # the trigger has not reset yet: do not spend a shot of this press on a refusal
            decision.reason = "actuator busy"
        self.decision = decision
        t_sol = time.perf_counter_ns()
        solve_end_ns = int(self.source.clock.now_ns())
        solve_ns = max(0, solve_end_ns - solve_start_ns)
        self._expected_solve_ns = (
            solve_ns if self.frames == 0
            else (1.0 - SOLVE_EMA_ALPHA) * self._expected_solve_ns + SOLVE_EMA_ALPHA * solve_ns
        )

        fired = False
        cmd_t_ns = None
        if decision.fire and self.arming.try_consume_shot():
            fire_at = max(now, int(decision.extra.get("fire_at_ns", now)))  # may be inside the frame period
            if cfg.fire.live and self.actuator is not None:
                cmd_t_ns = int(self.source.clock.now_ns())
                while cmd_t_ns < fire_at:
                    time.sleep((fire_at - cmd_t_ns) / 1e9)
                    next_ns = int(self.source.clock.now_ns())
                    if next_ns <= cmd_t_ns:
                        # A stopped source clock cannot establish a safe command time.
                        cmd_t_ns = None
                        break
                    cmd_t_ns = next_ns
                if cmd_t_ns is not None:
                    fired = bool(self.actuator.fire(cmd_t_ns))
                    if not fired:
                        log.warning("actuator refused the fire command (interlock open?)")
                    elif not isinstance(self.imu, NullImu) and str(cfg.actuator.type) in ("servo", "pq12"):
                        self.recoil.expect(cmd_t_ns, self.shots + 1)  # a real trigger: watch for the shot
                else:
                    log.warning("actuator command skipped: source clock did not advance")
            else:
                fired = True
                log.info(
                    "FIRE (dry run) track %s  miss %.2f deg  pattern %.2f deg  P(hit) %.2f  range %.1f m",
                    decision.track_id,
                    np.degrees(decision.miss_rad),
                    np.degrees(decision.pattern_rad),
                    decision.p_hit,
                    decision.range_m,
                )
            if fired:
                self.shots += 1
                target = next((t for t in self.tracker.tracks if t.track_id == decision.track_id), None)
                self.engagement.record(decision, int(cmd_t_ns) if cmd_t_ns is not None else int(now), target)

        self.frames += 1
        dt_ns = None if self._prev_exposure_ns is None else int(frame.t_exposure_ns) - self._prev_exposure_ns
        if dt_ns is not None and dt_ns > 0:
            inst = 1e9 / dt_ns
            self.fps = inst if self.fps <= 0.0 else (1.0 - FPS_EMA_ALPHA) * self.fps + FPS_EMA_ALPHA * inst
        self._prev_exposure_ns = int(frame.t_exposure_ns)

        self._check_health(self.health)  # self.health is refreshed by _health_loop, never by the frame loop
        ae = self._check_exposure(int(frame.t_capture_ns))

        tele = {
            "seq": int(frame.seq),
            "frames": self.frames,
            "t_exposure_ns": int(frame.t_exposure_ns),
            "t_ns": now,
            "cmd_t_ns": cmd_t_ns,
            "fire_at_ns": int(decision.extra.get("fire_at_ns", now)) if fired else None,
            "decide_ahead_ms": decide_ahead_ns / 1e6,
            "age_ms": (now - int(frame.t_exposure_ns)) / 1e6,
            "fps": float(self.fps),
            "dropped": int(getattr(self.source, "dropped", 0)),
            "armed": armed,
            "shots": int(self.shots),
            "fired": bool(fired),
            "live": bool(cfg.fire.live),
            "imu_ready": bool(self.attitude.ready),
            "n_detections": len(self.detections),
            "exposure_us": float(frame.exposure_us),
            "gain": float(frame.gain),
            "ae": ae,
            "health": self.health,
            "stage_ms": {
                "read": _ms(t_begin, t_read),
                "imu": _ms(t_read, t_imu),
                "attitude": _ms(t_imu, t_att),
                "detect": _ms(t_att, t_det),
                "track": _ms(t_det, t_trk),
                "solve": _ms(t_trk, t_sol),
            },
            "detector_ms": dict(self.detector.timings_ms),
            "detector": {"scale": int(self.detector.scale), "noise_dn": float(self.detector.noise_sigma),
                         **self.detector.stats},
            # [u, v, diameter_px, snr, contrast_dn, bbox x, y, w, h, area_px] per detection, full-res pixels
            "dets": [[round(float(d.u), 2), round(float(d.v), 2), round(float(d.diameter_px), 2),
                      round(float(d.snr), 1), round(float(d.contrast), 1), *d.bbox, float(d.area_px)]
                     for d in self.detections],
            "static": self.clutter.static_px if self.detector.candidate_filter is not None else None,
            "R_wg": [float(x) for x in R_wg.ravel()],
            "tracks": [
                {
                    "id": int(t.track_id),
                    "az_deg": float(np.degrees(t.az)),
                    "el_deg": float(np.degrees(t.el)),
                    "range_m": float(t.range_m),
                    "range_sigma_m": float(t.range_sigma_m),
                    "hits": int(t.hits),
                    "misses": int(t.misses),
                    "confirmed": bool(t.confirmed),
                    # filter state: cartesian [p, v, a] in m (world, camera origin), else angular [az, el] x 3
                    **({"s": [float(x) for x in t.s]} if t.s is not None else {"x": [float(x) for x in t.x]}),
                }
                for t in self.tracks
            ],
            "decision": {
                "fire": bool(decision.fire),
                "reason": str(decision.reason),
                "track_id": None if decision.track_id is None else int(decision.track_id),
                "miss_deg": float(np.degrees(decision.miss_rad)),
                "pattern_deg": float(np.degrees(decision.pattern_rad)),
                "sigma_deg": float(np.degrees(decision.sigma_rad)),
                "p_hit": float(decision.p_hit),
                "range_m": float(decision.range_m),
                "tof_ms": float(decision.tof_s * 1e3),
                "t_exit_ns": None if decision.t_exit_ns is None else int(decision.t_exit_ns),
                "lead_dir_w": None if decision.lead_dir_w is None else [float(x) for x in decision.lead_dir_w],
                "bore_dir_w": None if decision.bore_dir_w is None else [float(x) for x in decision.bore_dir_w],
            },
        }
        if fired:
            ev = {
                "t_ns": now,
                "cmd_t_ns": cmd_t_ns,
                "type": "fire",
                "live": bool(cfg.fire.live),
                "track_id": tele["decision"]["track_id"],
                "miss_deg": tele["decision"]["miss_deg"],
                "p_hit": tele["decision"]["p_hit"],
                "range_m": tele["decision"]["range_m"],
                "t_exit_ns": tele["decision"]["t_exit_ns"],
                "tof_ms": tele["decision"]["tof_ms"],
                "shots": self.shots,
            }
            self._event(ev)

        self._record(frame, samples, tele)
        if fired and self.recorder is not None and not self._recorder_is_writer and self._press_ns is None:
            self.recorder.trigger(now, tag="fire")  # a shot outside a recorded press (cannot happen on the gun)

        tele["stage_ms"]["overlay"] = self._publish(frame, R_wg, tele)
        tele["stage_ms"]["total"] = _ms(t_begin, time.perf_counter_ns())
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(jsonable(tele)) + "\n")

        self.last_telemetry = tele
        self._last_step_ns = time.monotonic_ns()
        self._last_frame_wall = time.monotonic()
        self._wd_tripped = False
        return tele

    # ------------------------------------------------------------------ loop
    def _watchdog(self) -> None:
        limit_ns = int(float(self.cfg.fire.watchdog_ms) * 1e6)
        period = max(0.002, min(0.05, limit_ns / 4e9))
        while not self._wd_stop.wait(period):
            last = self._last_step_ns
            if last is None or self._wd_tripped:
                continue
            if time.monotonic_ns() - last > limit_ns:
                self._wd_tripped = True
                self.watchdog_aborts += 1
                log.warning("watchdog: no processed frame for %.0f ms -> actuator abort", self.cfg.fire.watchdog_ms)
                if self.actuator is not None:
                    self.actuator.abort()

    def _on_signal(self, signum, _frame) -> None:
        log.info("signal %s: stopping", signal.Signals(signum).name)
        self._stop.set()

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._sig_prev.append((sig, signal.signal(sig, self._on_signal)))
            except (ValueError, OSError):  # pragma: no cover - restricted environments
                pass

    def _restore_signals(self) -> None:
        while self._sig_prev:
            sig, prev = self._sig_prev.pop()
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):  # pragma: no cover
                pass

    def _status(self) -> None:
        tele = self.last_telemetry
        if tele is None:
            return
        d = tele["decision"]
        health = tele.get("health") or {}
        temp = health.get("cpu_temp_c")
        flags = health.get("throttled") or {}
        warn = "".join(
            tag for tag, key in (("UV", "under_voltage_now"), ("THR", "throttled_now")) if flags.get(key)
        )
        hstr = ""
        if temp is not None:
            hstr += f"  {float(temp):4.1f}C"
        if warn:
            hstr += f"  !{warn}"
        ae = tele.get("ae")
        if ae is not None:  # exposure the light asked for; flagged when the scene is outside the camera's range
            flag = {"too_bright": " !BRIGHT", "too_dark": " !DARK"}.get(ae["state"], "")
            hstr += f"  exp {tele['exposure_us']:.0f}us x{tele['gain']:.1f}{flag}"
        print(
            f"[{self.frames:6d}] {tele['fps']:5.1f} fps  age {tele['age_ms']:5.1f} ms  "
            f"cpu {tele['stage_ms']['total']:5.1f} ms  det {tele['n_detections']}  trk {len(tele['tracks'])}  "
            f"{'ARMED' if tele['armed'] else 'SAFE '}  shots {self.shots}  drop {tele['dropped']}{hstr}  "
            f"{'FIRE' if d['fire'] else d['reason']}",
            flush=True,
        )

    def run(
        self,
        max_frames: int | None = None,
        display: bool = False,
        log_jsonl: str | Path | None = None,
        status: bool = True,
        idle_end_s: float | None = None,
        startup_timeout_s: float = 10.0,
    ) -> int:
        """Loop until the source ends, max_frames, SIGINT or a display quit. Returns a process exit code."""
        self._display = bool(display)
        if log_jsonl is not None:
            p = resolve_path(log_jsonl)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(p, "w", buffering=1)
            log.info("telemetry log: %s", p)
        if idle_end_s is None:
            idle_end_s = max(0.75, 3.0 * float(self.cfg.fire.watchdog_ms) / 1000.0)
        self._install_signals()
        rc = 0
        try:
            self.start()
            # Everything alive now (modules, calibration, buffers) lives as long as the run: take it out of the
            # garbage collector's sight, or every full collection walks ~10^5 objects (tens of ms on a Pi 3B,
            # i.e. dropped frames about once a second).
            gc.collect()
            gc.freeze()
            self._wd_stop.clear()
            self._wd_thread = threading.Thread(target=self._watchdog, name="tbot-watchdog", daemon=True)
            self._wd_thread.start()
            t_run0 = time.monotonic()
            idle_fps = float(getattr(self.cfg.camera, "idle_fps", 0.0) or 0.0)
            # only a source on the real clock is paced: a simulated or replayed one has no heat to save
            idle_period = 1.0 / idle_fps if idle_fps > 0.0 and isinstance(self.source.clock, MonotonicClock) else 0.0
            while not self._stop.is_set():
                t_loop = time.monotonic()
                tele = self.step()
                now = time.monotonic()
                if tele is not None and idle_period and not tele["armed"] and not self.arming.armed:
                    # button up: rest until the idle period is over (a press is noticed within one period, and
                    # the tracks are already there because tracking never stopped)
                    self._stop.wait(max(0.0, idle_period - (now - t_loop)))
                if tele is None:
                    if self._last_frame_wall is None:
                        if now - t_run0 > startup_timeout_s:
                            log.error("no frame from %s within %.0f s", self.source.name, startup_timeout_s)
                            rc = 1
                            break
                    elif now - self._last_frame_wall > idle_end_s and not getattr(self.source, "endless", False):
                        log.info("source %s delivered no frame for %.2f s: stopping", self.source.name, idle_end_s)
                        break
                    time.sleep(0.002)
                    continue
                if status and now - self._status_wall >= 1.0:
                    self._status_wall = now
                    self._status()
                if max_frames is not None and self.frames >= max_frames:
                    break
        except KeyboardInterrupt:  # pragma: no cover - SIGINT handler normally wins
            log.info("interrupted")
        finally:
            self._restore_signals()
            self.close()
        log.info(
            "stopped after %d frames, %d shot(s), %d dropped",
            self.frames,
            self.shots,
            int(getattr(self.source, "dropped", 0)),
        )
        stats = summarise(self.recoil_delays_ms)
        if stats or self.no_recoil:
            expected = self.budget.cmd_to_break_ms[str(self.cfg.actuator.type)] + self.budget.lock_time_ms
            if stats:
                log.info("recoil: %d shot(s) detected, command -> ignition median %.1f ms (sigma %.1f); the budget "
                         "assumes %.1f ms. `tbot latency-recoil` turns the recorded shots into calib/latency.json",
                         stats["n"], stats["median"], stats["std"], expected)
            if self.no_recoil:
                log.warning("recoil: %d fire command(s) with no shot", self.no_recoil)
        return rc


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot run",
        description="live pipeline: camera -> detect -> track -> fire solver (dry run unless --live)",
    )
    add_common_args(parser)
    parser.add_argument("--live", action="store_true", help="ARM THE ACTUATOR: drive the trigger when the solver fires")
    parser.add_argument("--stream", action="store_true", help="serve the debug overlay over HTTP")
    parser.add_argument("--port", type=int, default=None, help="debug stream port (implies --stream)")
    parser.add_argument("--display", action="store_true", help="show a local preview window (needs a display)")
    parser.add_argument(
        "--sim-armed", action="store_true", help="simulated GPIO only: hold the arming button down from the start"
    )
    parser.add_argument("--max-frames", type=int, default=None, metavar="N", help="stop after N processed frames")
    parser.add_argument("--record", choices=("off", "armed", "continuous"), default=None, help="override record.mode")
    parser.add_argument("--log-jsonl", default=None, metavar="PATH", help="write one telemetry JSON line per frame")
    parser.add_argument("--quiet", action="store_true", help="no per-second status line")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, calib = load_from_args(args)
    if args.live:
        cfg.fire.live = True
    if args.port is not None:
        cfg.stream.port = int(args.port)
    if args.stream or args.port is not None:
        cfg.stream.enabled = True
    if args.record is not None:
        cfg.record.mode = args.record

    if cfg.camera.source.partition(":")[0] == "picam" and bool(cfg.camera.process):
        from tbot.camera.proc import SensorProcess

        source = SensorProcess(cfg, calib)  # camera + IMU in their own process (see camera/proc.py for why)
        log.info("sensor process %d: %s%s", source.child_pid, source.name, " + IMU" if source.has_imu else "")
    else:
        source = open_live_source(cfg.camera.source, cfg, calib)
    gpio = open_gpio(cfg)
    if args.sim_armed and not isinstance(gpio, SimGpio):
        gpio.close()
        parser.error("--sim-armed only works with a simulated GPIO backend (--set gpio.backend=sim)")
    pipe = Pipeline(cfg, calib, source, gpio=gpio)
    if args.sim_armed:
        pressed = 0 if cfg.arming.active_low else 1
        gpio.inject(int(cfg.arming.pin), pressed)
        log.warning("--sim-armed: the arming interlock is simulated as HELD (no hardware button)")
    if cfg.fire.live:
        log.warning("LIVE: the trigger actuator will be driven while the arming button is held")
    return pipe.run(
        max_frames=args.max_frames,
        display=args.display,
        log_jsonl=args.log_jsonl,
        status=not args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
