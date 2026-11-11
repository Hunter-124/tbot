"""RigSession: safety, motion helpers and the run lifecycle shared by every calrig sub-command.

Safety model (docs/CALRIG.md): nothing moves until the operator typed UNLOADED (real hardware); the firmware
watchdog is fed by the device heartbeat for as long as this process is alive; any exception, Ctrl-C or SIGTERM
ends in STOP (controlled deceleration) then EN 0; e-stop events, watchdog events, a silent telemetry or IMU
stream and a dead link abort the run with RigFault. No firing or actuator code is imported anywhere here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from tbot.calrig import protocol as P
from tbot.calrig.device import DeviceError, RigDevice, RigFault
from tbot.calrig.recorder import ALL_POSES, RunRecorder
from tbot.config import now_iso, resolve_path

if TYPE_CHECKING:  # pragma: no cover
    from tbot.calrig.procedures import Procedure
    from tbot.config import Calibration, Config

__all__ = [
    "RigSession",
    "SetupError",
    "HomeError",
    "Limits",
    "confirm_unloaded",
    "CONFIRM_WORD",
    "neutral_calibration",
    "describe_imu",
    "move_time_s",
]

CONFIRM_WORD = "UNLOADED"
TELEMETRY_SILENCE_S = 1.0
IMU_SILENCE_S = 2.0
PING_PERIOD_S = 1.0
SERVICE_SLICE_S = 0.02
BACKOFF_DEG = 10.0


class SetupError(RuntimeError):
    """Firmware and configuration disagree, or the rig is not in a state to start."""


class HomeError(RuntimeError):
    pass


@dataclass
class Limits:
    """Host-side refusal limits (config, tightened by the firmware caps reported in INFO)."""

    soft_deg: float
    vmax_dps: float
    amax_dps2: float
    vmax_cap_dps: float
    amax_cap_dps2: float


def move_time_s(distance_deg: float, vmax_dps: float, amax_dps2: float) -> float:
    d = abs(distance_deg)
    if d <= 0.0:
        return 0.0
    if d >= vmax_dps ** 2 / amax_dps2:
        return d / vmax_dps + vmax_dps / amax_dps2
    return 2.0 * math.sqrt(d / amax_dps2)


def confirm_unloaded(input_fn: Callable[[str], str] | None = None, out: Callable[[str], None] = print) -> dict:
    """Make the operator type UNLOADED. Anything else (or EOF: no terminal) refuses."""
    out("\n  SAFETY: the gun must be UNLOADED (chamber checked, action open, no ammunition anywhere near),")
    out("  muzzle in a safe direction, clamp tight, the swing arc clear of people and objects, and the")
    out("  e-stop / motor-power cut within reach. The rig never fires and never touches the trigger actuator.")
    ask = input_fn if input_fn is not None else input
    try:
        answer = ask(f"  Type {CONFIRM_WORD} to confirm and enable the motor: ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip() != CONFIRM_WORD:
        raise SetupError(f"not confirmed (expected {CONFIRM_WORD!r}); the motor stays disabled")
    return {"confirmed": True, "method": f"operator typed {CONFIRM_WORD}", "simulated": False, "at": now_iso()}


def neutral_calibration(calib: "Calibration") -> "Calibration":
    """A copy of calib with neutral IMU calibration, so Mpu6050 returns RAW chip-frame samples with no time offset
    (the pattern tbot calib-imu uses; Mpu6050 itself is unchanged for the main app)."""
    return dataclasses.replace(calib, imu={
        "gyro_bias_rad_s": [0.0, 0.0, 0.0],
        "accel_bias_m_s2": [0.0, 0.0, 0.0],
        "accel_scale": [1.0, 1.0, 1.0],
        "R_gun_imu": np.eye(3).tolist(),
        "time_offset_ns": 0,
    })


def describe_imu(cfg: "Config", calib: "Calibration | None", source: str, sample_rate_hz: float | None) -> dict:
    """meta.json 'imu' block. R_gun_imu is the matrix the main app uses today (calib/imu.json over config);
    the samples in imu.csv are raw chip-frame values and have NOT had it applied."""
    r = (getattr(calib, "imu", None) or {}).get("R_gun_imu", cfg.imu.R_gun_imu)
    return {
        "source": source,
        "sample_rate_hz": sample_rate_hz if sample_rate_hz is not None else cfg.imu.sample_rate_hz,
        "gyro_range_dps": cfg.imu.gyro_range_dps,
        "accel_range_g": cfg.imu.accel_range_g,
        "dlpf_hz": cfg.imu.dlpf_hz,
        "R_gun_imu": np.asarray(r, dtype=float).tolist(),
        "samples_are_raw_chip_frame": True,
        "clock": "time.monotonic_ns (ImuSample.t_ns)",
    }


class RigSession:
    def __init__(self, dev: RigDevice, cfg: "Config", imu: Any | None = None, imu_meta: dict | None = None,
                 sim_truth: dict | None = None, log: Callable[[str], None] = print):
        self.dev = dev
        self.clock = dev.clock
        self.cfg = cfg
        self.rc = cfg.calrig
        self.geometry = dev.geometry
        self.ups = dev.geometry.usteps_per_deg
        self.imu = imu
        self.imu_meta = imu_meta
        self.sim_truth = sim_truth
        self.log = log
        self.limits = Limits(self.rc.soft_limit_deg, self.rc.vmax_dps, self.rc.amax_dps2,
                             self.rc.vmax_cap_dps, self.rc.amax_cap_dps2)
        self.vmax_sps = self.rc.vmax_dps * self.ups
        self.amax_sps2 = self.rc.amax_dps2 * self.ups
        self.rec: RunRecorder | None = None
        self.recording = False   # False while homing: HOME re-zeroes the counter, a jump that must stay out of rig.csv
        self.streaming = False
        self._stream_since_ns = 0
        self._next_ping_ns = 0
        self._last_imu_ns = 0
        self._imu_running = False
        self._closed = False

    # ---------------------------------------------------------------------------- context
    def __enter__(self) -> "RigSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.shutdown()
        return False

    # ---------------------------------------------------------------------------- units
    @property
    def vmax_dps(self) -> float:
        return self.vmax_sps / self.ups

    @property
    def amax_dps2(self) -> float:
        return self.amax_sps2 / self.ups

    # ---------------------------------------------------------------------------- setup
    def setup(self) -> dict:
        """INFO, geometry check, soft limits, default rates. Does not enable the driver."""
        dev, rc = self.dev, self.rc
        info = dev.get_info()
        try:
            fw_usteps, fw_rev = int(info["usteps"]), int(info["steps_rev"])
            fw_ratio = float(info.get("ratio", 1.0))
        except (KeyError, ValueError) as exc:
            raise SetupError(f"unexpected INFO reply {info!r}") from exc
        if (fw_usteps, fw_rev) != (rc.microsteps, rc.steps_rev) or abs(fw_ratio - rc.ratio) > 1e-9:
            raise SetupError(
                f"firmware usteps={fw_usteps} steps_rev={fw_rev} ratio={fw_ratio:g} but config [calrig] says "
                f"microsteps={rc.microsteps} steps_rev={rc.steps_rev} ratio={rc.ratio:g}: angles would be wrong. "
                "Fix the MICROSTEPS constant / jumpers or the config.")
        if "vmax_cap" in info:
            self.limits.vmax_cap_dps = min(self.limits.vmax_cap_dps, int(info["vmax_cap"]) / self.ups)
        if "amax_cap" in info:
            self.limits.amax_cap_dps2 = min(self.limits.amax_cap_dps2, int(info["amax_cap"]) / self.ups)
        st = dev.stat()
        if st.estop_pin:
            raise RigFault("the e-stop input (A0) is asserted: release the e-stop (or jumper A0 to GND) first")
        if st.state == "ESTOP":
            dev.command(P.cmd_reset())
        soft = self.geometry.to_usteps(rc.soft_limit_deg)
        dev.command(P.cmd_lim(-soft, soft))
        self.set_rates(self.limits.vmax_dps, self.limits.amax_dps2)
        dev.start_heartbeat(rc.heartbeat_ms / 1000.0)
        return info

    def enable(self) -> None:
        self.dev.command(P.cmd_en(True))

    def start_streaming(self) -> None:
        self.dev.command(P.cmd_stream(True))
        self.streaming = True
        self._stream_since_ns = self.clock.now_ns()

    def set_rates(self, vmax_dps: float, amax_dps2: float) -> None:
        """CFG with host-side refusal above the caps (the firmware would silently clamp); remembers what is in force."""
        L = self.limits
        if not 0.0 < vmax_dps <= L.vmax_cap_dps + 1e-9:
            raise ValueError(f"rate {vmax_dps:g} deg/s outside (0, {L.vmax_cap_dps:g}] deg/s")
        if not 0.0 < amax_dps2 <= L.amax_cap_dps2 + 1e-9:
            raise ValueError(f"acceleration {amax_dps2:g} deg/s^2 outside (0, {L.amax_cap_dps2:g}] deg/s^2")
        r = self.dev.command(P.cmd_cfg(vmax_dps * self.ups, amax_dps2 * self.ups))
        self.vmax_sps = float(r.fields.get("vmax", vmax_dps * self.ups))
        self.amax_sps2 = float(r.fields.get("amax", amax_dps2 * self.ups))

    # ---------------------------------------------------------------------------- service loop
    def service(self) -> None:
        """Called from every wait loop: pump the link, feed the watchdog, drain queues, ping, check faults."""
        dev = self.dev
        dev.poll()
        dev.touch()
        dev.heartbeat_tick()
        now = self.clock.now_ns()
        if self.rec is not None:
            self.rec.add_commands(_drain(dev.cmd_q))
        tele, hall, ev = _drain(dev.telemetry_q), _drain(dev.hall_q), _drain(dev.event_q)
        samples = self.imu.read() if (self.imu is not None and self._imu_running) else []
        if samples:
            self._last_imu_ns = now
        if self.rec is not None and self.recording:
            self.rec.add_telemetry(tele)
            self.rec.add_hall(hall)
            self.rec.add_events(ev)
            self.rec.add_imu(samples)
        if self.rec is not None and now >= self._next_ping_ns:
            self._next_ping_ns = now + int(PING_PERIOD_S * 1e9)
            dev.ping()
        if dev.link_error:
            raise RigFault(dev.link_error)
        if dev.fault:
            raise RigFault(dev.fault)
        if self.streaming and now - max(dev.last_telem_rx_ns, self._stream_since_ns) > TELEMETRY_SILENCE_S * 1e9:
            raise RigFault(f"no telemetry for {TELEMETRY_SILENCE_S:g} s: link or firmware stalled")
        if (self.rec is not None and self._imu_running and self.imu is not None
                and now - self._last_imu_ns > IMU_SILENCE_S * 1e9):
            raise RigFault(f"no IMU samples for {IMU_SILENCE_S:g} s")

    def wait(self, seconds: float) -> None:
        end = self.clock.now_ns() + int(seconds * 1e9)
        while True:
            remaining = (end - self.clock.now_ns()) * 1e-9
            if remaining <= 0.0:
                break
            self.clock.sleep(min(SERVICE_SLICE_S, remaining))
            self.service()

    def wait_for(self, predicate: Callable[[], bool], timeout_s: float) -> bool:
        end = self.clock.now_ns() + int(timeout_s * 1e9)
        while True:
            self.service()
            if predicate():
                return True
            if self.clock.now_ns() >= end:
                return False
            self.clock.sleep(SERVICE_SLICE_S)

    # ---------------------------------------------------------------------------- motion helpers
    def begin_motion(self) -> int:
        """Round-trip STAT (flushes everything the firmware sent so far), insist on an idle, enabled rig, and
        return the event mark to wait on."""
        st = self.dev.stat()
        if st.state == "ESTOP" or not st.en:
            raise RigFault(f"rig not ready (state={st.state} en={st.en})")
        if st.state != "IDLE":
            raise RigFault(f"rig is not idle before a new motion (state={st.state})")
        self.service()
        return self.dev.event_mark()

    def wait_idle(self, mark: int, timeout_s: float) -> list[P.Event]:
        """Wait for a terminal event after `mark`; if the firmware never says DONE, accept STAT state IDLE.

        Raises RigFault on LIMIT / ESTOP / WATCHDOG or timeout.
        """
        dev = self.dev

        def events() -> list[P.Event]:
            return [e for _, _, e in dev.events_since(mark) if e.code in P.TERMINAL_EVENTS]

        end = self.clock.now_ns() + int(timeout_s * 1e9)
        next_stat = self.clock.now_ns() + int(1.5e9)
        while True:
            ev = events()
            if ev:
                break
            self.service()
            now = self.clock.now_ns()
            if now >= next_stat:
                st = dev.stat()
                if st.state == "IDLE" and st.vel == 0:
                    ev = events()
                    break
                next_stat = now + int(1.0e9)
            if now >= end:
                raise RigFault(f"motion did not finish within {timeout_s:.0f} s")
            self.clock.sleep(SERVICE_SLICE_S)
        bad = [e for e in ev if e.code in ("LIMIT", "ESTOP", "WATCHDOG")]
        if bad:
            raise RigFault(f"motion ended by the firmware: {bad[0].code}")
        return ev

    def move_to(self, deg: float, settle_s: float = 0.0) -> None:
        soft = self.limits.soft_deg
        if abs(deg) > soft + 1e-9:
            raise ValueError(f"target {deg:g} deg is outside the +-{soft:g} deg soft limits")
        mark = self.begin_motion()
        st_pos = self.dev.stat().pos
        target = self.geometry.to_usteps(deg)
        t = move_time_s((target - st_pos) / self.ups, self.vmax_sps / self.ups, self.amax_sps2 / self.ups)
        self.dev.command(P.cmd_move(target))
        self.wait_idle(mark, 1.5 * t + 6.0)
        if settle_s > 0.0:
            self.wait(settle_s)

    def move_by(self, deg: float) -> None:
        pos = self.dev.stat().pos
        self.move_to(self.geometry.to_deg(pos) + deg)

    def home(self, direction: int | None = None, rate_dps: float | None = None, to_zero: bool = True) -> dict:
        """Seek the hall edge, re-zero on it and (default) move onto it. Returns the HOMED offset: the counter at
        the edge before re-zeroing, i.e. the distance from the old zero (lost steps on a second homing)."""
        d = int(direction if direction is not None else self.rc.home_dir)
        rate = float(rate_dps if rate_dps is not None else self.rc.home_rate_dps)
        if d not in (1, -1):
            raise ValueError("home direction must be +1 or -1")
        if not 0.0 < rate <= self.limits.vmax_cap_dps:
            raise ValueError(f"home rate {rate:g} deg/s outside (0, {self.limits.vmax_cap_dps:g}]")
        for _ in range(6):  # the sensor must be inactive so that its becoming active is an edge
            st = self.dev.stat()
            if not st.home:
                break
            self.log(f"home sensor already active: backing off {BACKOFF_DEG:g} deg")
            self.move_by(-d * BACKOFF_DEG)
        else:
            raise HomeError("the home sensor stays active after backing off 60 deg: check the hall switch wiring")
        soft = self.limits.soft_deg
        mark = self.begin_motion()
        self.dev.command(P.cmd_home(d, rate * self.ups))
        try:
            ev = self.wait_idle(mark, 2.0 * soft / rate + 20.0)
        except RigFault as exc:
            raise HomeError(f"homing failed ({exc}): hand-rotate the gun nearer the magnet (the seek is limited "
                            f"to +-{soft:g} deg from the power-up position)") from exc
        homed = [e for e in ev if e.code == "HOMED"]
        if not homed or homed[0].arg is None:
            raise HomeError("firmware finished the HOME motion without an `E HOMED` event")
        offset = int(homed[0].arg)
        out = {"dir": d, "rate_dps": rate, "offset_ustep": offset, "offset_deg": offset * self.geometry.deg_per_ustep}
        self.log(f"homed: offset {offset} usteps ({out['offset_deg']:+.3f} deg)")
        if to_zero:
            self.move_to(0.0)
        return out

    # ---------------------------------------------------------------------------- run lifecycle
    def run(self, proc: "Procedure", pose: str, operator: dict, home_first: bool = True,
            root: Path | None = None) -> Path:
        """Record one procedure into a run directory and return it. The caller has already confirmed UNLOADED."""
        if pose not in ALL_POSES:
            raise ValueError(f"pose must be one of {ALL_POSES}")
        proc.validate(self.limits)
        root = Path(root) if root is not None else resolve_path(self.rc.run_root)
        rec = RunRecorder(root, proc.name, self.geometry)
        self.rec = rec
        status, result = "aborted", None
        self.log(f"run directory: {rec.dir}")
        try:
            self.dev.sync_burst(20)
            self._next_ping_ns = self.clock.now_ns() + int(PING_PERIOD_S * 1e9)
            if self.imu is not None:  # before the driver is energised: a dead IMU must fail the run early
                self.imu.start()
                self._imu_running = True
                self._last_imu_ns = self.clock.now_ns()
            self.enable()
            self.start_streaming()
            self.service()
            homing = None
            if proc.needs_home and home_first:
                homing = self.home()  # not recorded: the counter jump at re-zero would corrupt rig.csv
            self.service()
            self.recording = True
            result = proc.run(self)
            if homing is not None:
                result = {**(result or {}), "initial_home": homing}
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            status = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            self._finish_run(rec, proc, pose, operator, status, result)
        return rec.dir

    def _finish_run(self, rec: RunRecorder, proc: "Procedure", pose: str, operator: dict, status: str,
                    result: Any) -> None:
        dev = self.dev
        self.shutdown_motion()
        try:
            if not dev.link_error:
                dev.try_command(P.cmd_stream(False))
                self.streaming = False
                dev.sync_burst(10)
            self.service_quiet()
        except (DeviceError, OSError, KeyboardInterrupt):
            pass
        self._stop_imu()
        info = dict(dev.info_fields)
        meta = {
            "procedure": proc.name,
            "params": proc.params(),
            "result": result,
            "status": status,
            "pose": pose,
            "firmware": info,
            "imu": self.imu_meta,
            "unloaded_confirmation": operator,
            "simulated": self.sim_truth is not None,
            "limits": {"soft_deg": self.limits.soft_deg, "vmax_dps_in_force": self.vmax_dps,
                       "amax_dps2_in_force": self.amax_dps2, "vmax_cap_dps": self.limits.vmax_cap_dps,
                       "amax_cap_dps2": self.limits.amax_cap_dps2},
            "config": dataclasses.asdict(self.rc),
            "rx": {"bad_lines": dev.bad_lines, "unknown_lines": list(dev.unknown_lines)[:20]},
        }
        if self.sim_truth is not None:
            meta["sim_truth"] = self.sim_truth
        rec.finalize(meta, dev.sync.fit())
        self.rec = None

    def service_quiet(self) -> None:
        """Drain whatever is queued into the recorder without raising on faults (shutdown path)."""
        dev, rec = self.dev, self.rec
        dev.poll()
        if rec is not None:
            tele, hall, ev = _drain(dev.telemetry_q), _drain(dev.hall_q), _drain(dev.event_q)
            rec.add_commands(_drain(dev.cmd_q))
            samples = self.imu.read() if (self.imu is not None and self._imu_running) else []
            if self.recording:
                rec.add_telemetry(tele)
                rec.add_hall(hall)
                rec.add_events(ev)
                rec.add_imu(samples)

    def _stop_imu(self) -> None:
        if self.imu is not None and self._imu_running:
            try:
                self.imu.stop()
            except Exception:  # noqa: BLE001
                pass
            self._imu_running = False

    def shutdown_motion(self) -> None:
        """STOP, wait for rest while feeding the watchdog, EN 0. Idempotent, never raises."""
        self.dev.safe_stop()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.shutdown_motion()
        finally:
            self._stop_imu()
            if self.rec is not None:
                self.rec.abort()
            self.dev.close()


def _drain(q) -> list:
    out = []
    while q:
        try:
            out.append(q.popleft())
        except IndexError:
            break
    return out
