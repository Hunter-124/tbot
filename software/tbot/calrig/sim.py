"""In-process simulation of the CalRig: a FakeRig firmware (full protocol) and a synthetic MPU-6050 stream.

Everything runs on a VirtualClock, so `calrig run ... --sim` and the tests need no hardware and no real time.

FakeRig mirrors docs/CALRIG_PROTOCOL.md: trapezoid MOVE / ramped VEL / enveloped SINE / HOME on the hall edge,
soft limits, e-stop latch (pin and command), 1 s motion watchdog, telemetry, hall-edge lines (interpolated to the
exact crossing), micros() with an arbitrary start, drift and uint32 wrap, and random serial latency per line.
Deliberate simulation choices where the spec is silent:
  * HOME while the sensor is already active -> `ERR HOME ACTIVE` (the host backs off first);
  * SINE while moving -> `ERR SINE MOVING` (the host only starts a sine from rest);
  * `E DONE` is emitted when MOVE / SINE / STOP / VEL 0 complete, not after HOME (`E HOMED`) or LIMIT.

SimImu produces RAW chip-frame gyro/accel from the rig's true yaw rate for a configurable mount R_gun_imu
(v_gun = R v_chip, as tbot.imu.mpu6050), pose, per-axis gyro scale error, bias, noise and IMU time offset
(positive: IMU timestamps run LATE, the signal appears `offset` after it physically happened).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import numpy as np

from tbot.calrig import protocol as P
from tbot.calrig.timebase import VirtualClock
from tbot.types import ImuSample

__all__ = ["FakeRig", "SimImu", "SimBundle", "make_sim", "parse_sim_opts", "POSE_UP_GUN", "plan_move", "SineProfile"]

G_MPS2 = 9.80665
# World-up direction in the GUN frame (+X muzzle, +Y gun-left, +Z up when upright) for each pose.
POSE_UP_GUN = {
    "upright": (0.0, 0.0, 1.0),
    "left_side_down": (0.0, -1.0, 0.0),
    "right_side_down": (0.0, 1.0, 0.0),
    "inverted": (0.0, 0.0, -1.0),
    "muzzle_up": (1.0, 0.0, 0.0),      # extra poses, accepted by the analysis; not reachable on the standard saddles
    "muzzle_down": (-1.0, 0.0, 0.0),
}


# ----------------------------------------------------------------------------------------- profiles
class _Plan:
    """Piecewise constant-acceleration profile; positions in usteps, times in s since the plan start."""

    def __init__(self, p0: float, v0: float, target: float, segs: list[tuple[float, float]]):
        self.target = target
        self.rows: list[tuple[float, float, float, float, float]] = []  # t0, p0, v0, acc, dur
        t, p, v = 0.0, p0, v0
        for dur, acc in segs:
            self.rows.append((t, p, v, acc, dur))
            p += v * dur + 0.5 * acc * dur * dur
            v += acc * dur
            t += dur
        self.duration = t

    def eval(self, tau: float) -> tuple[float, float]:
        if tau >= self.duration or not self.rows:
            return self.target, 0.0
        for t0, p0, v0, acc, dur in reversed(self.rows):
            if tau >= t0:
                s = tau - t0
                return p0 + v0 * s + 0.5 * acc * s * s, v0 + acc * s
        t0, p0, v0, acc, dur = self.rows[0]
        return p0, v0


def plan_move(p0: float, v0: float, target: float, vmax: float, amax: float) -> _Plan:
    """Time-optimal trapezoid/triangle from (p0, v0) to rest at target, |a| <= amax, |v| <= vmax.

    A non-zero v0 gives the smooth transition of spec section 2: no velocity step, a motion that cannot stop
    in time brakes through the target and comes back.
    """
    segs: list[tuple[float, float]] = []
    p, v = p0, v0
    e = target - p
    if v != 0.0:
        stop = v * abs(v) / (2.0 * amax)
        if math.copysign(1.0, v) != math.copysign(1.0, e) or abs(stop) > abs(e) + 1e-9:
            segs.append((abs(v) / amax, -math.copysign(amax, v)))
            p += stop
            v = 0.0
            e = target - p
    if abs(e) > 1e-9:
        s = math.copysign(1.0, e)
        u0 = abs(v)
        vp = min(vmax, math.sqrt(amax * abs(e) + 0.5 * u0 * u0))
        d1 = abs(vp * vp - u0 * u0) / (2.0 * amax)
        d3 = vp * vp / (2.0 * amax)
        cruise = max(0.0, abs(e) - d1 - d3)
        if abs(vp - u0) > 1e-12:
            segs.append((abs(vp - u0) / amax, s * amax * (1.0 if vp > u0 else -1.0)))
        if cruise > 1e-12:
            segs.append((cruise / vp, 0.0))
        segs.append((vp / amax, -s * amax))
    return _Plan(p0, v0, target, segs)


class SineProfile:
    """x(t) = p0 + A W(t) sin(2 pi t / T) with raised-cosine amplitude ramps over the first and last cycle."""

    def __init__(self, p0: float, amp: float, period_s: float, cycles: int):
        self.p0, self.amp, self.T, self.N = p0, amp, period_s, int(cycles)
        self.w = 2.0 * math.pi / period_s
        self.duration = period_s * self.N

    def _window(self, tau: float) -> tuple[float, float]:
        T, N = self.T, self.N
        w, dw = 1.0, 0.0
        if tau < T:
            s = math.sin(math.pi * tau / (2.0 * T))
            w1, dw1 = s * s, (math.pi / (2.0 * T)) * math.sin(math.pi * tau / T)
            w, dw = w1, dw1
        rem = N * T - tau
        if rem < T:
            s = math.sin(math.pi * rem / (2.0 * T))
            w2, dw2 = s * s, -(math.pi / (2.0 * T)) * math.sin(math.pi * rem / T)
            w, dw = w * w2, dw * w2 + w * dw2
        return w, dw

    def eval(self, tau: float) -> tuple[float, float]:
        if tau >= self.duration:
            return self.p0, 0.0
        w, dw = self._window(tau)
        s, c = math.sin(self.w * tau), math.cos(self.w * tau)
        return self.p0 + self.amp * w * s, self.amp * (dw * s + w * self.w * c)


# ----------------------------------------------------------------------------------------- FakeRig
@dataclass
class FakeRigConfig:
    geometry: P.Geometry = field(default_factory=P.Geometry)
    vmax_cap_dps: float = 90.0
    amax_cap_dps2: float = 60.0
    vmax_dps: float = 45.0
    amax_dps2: float = 20.0
    hard_lim_deg: float = 100.0
    soft_lim_deg: float = 90.0
    telem_hz: int = 200
    fw_version: str = "0.1-sim"
    boot_us: int = 4_294_000_000   # micros() at boot: just below the uint32 wrap, so every run crosses it
    drift_ppm: float = 35.0        # firmware crystal vs host clock
    latency_s: float = 0.0006      # one-way USB-serial latency, mean of the fixed part
    jitter_s: float = 0.0004       # exponential extra per direction
    hall_start_deg: float = 0.0    # true table angle of the lower hall edge (sensor active in [start, start+width])
    hall_width_deg: float = 6.0
    start_angle_deg: float = -12.0  # true table angle at power-up (the counter starts at 0 there)
    watchdog_s: float = 1.0
    seed: int = 1


class FakeRig:
    """Transport + firmware. write_line() executes a command at the current virtual time; read_line() returns
    the replies and asynchronous lines produced so far (each with its simulated host receive time)."""

    def __init__(self, clock: VirtualClock, cfg: FakeRigConfig | None = None):
        self.clock = clock
        self.cfg = cfg or FakeRigConfig()
        g = self.cfg.geometry
        self.ups = g.usteps_per_deg
        self.rng = np.random.default_rng(self.cfg.seed)
        self.vmax_cap = self.cfg.vmax_cap_dps * self.ups
        self.amax_cap = self.cfg.amax_cap_dps2 * self.ups
        self.vmax = self.cfg.vmax_dps * self.ups
        self.amax = self.cfg.amax_dps2 * self.ups
        self.hard = int(round(self.cfg.hard_lim_deg * self.ups))
        self.lim_lo = -int(round(self.cfg.soft_lim_deg * self.ups))
        self.lim_hi = int(round(self.cfg.soft_lim_deg * self.ups))

        # physical state (usteps; true angle = pos + true_offset)
        self.pos = 0.0
        self.vel = 0.0
        self.true_offset = self.cfg.start_angle_deg * self.ups
        self.en = False
        self.estop_pin = False
        self.estop_latched = False
        self.streaming = False
        self.state = "IDLE"
        self.mode = "idle"            # idle | plan | sine | vel
        self._plan: _Plan | None = None
        self._sine: SineProfile | None = None
        self._tau = 0.0
        self._v_target = 0.0
        self._vel_kind = ""           # "" | "stop" | "jog" | "home_seek" | "home_stop"
        self._home_latched = 0

        z0 = self.cfg.hall_start_deg * self.ups
        self._hall_lo, self._hall_hi = z0, z0 + self.cfg.hall_width_deg * self.ups

        self._t_boot_ns = clock.now_ns()
        self._last_ns = self._t_boot_ns
        self._last_rx_ns = self._t_boot_ns
        self._next_telem_ns = self._t_boot_ns
        self._telem_period_ns = int(round(1e9 / self.cfg.telem_hz))
        self._out: collections.deque[tuple[str, int]] = collections.deque()
        self.lines_sent: list[str] = []   # every command line received (for tests)
        self._hall_active = self._hall_at(self.pos)
        self._emit(f"E {self.t_us(self._t_boot_ns)} BOOT", self._t_boot_ns)
        clock.add_listener(self._on_tick)

    # ------------------------------------------------------------------ time / queue
    def t_us(self, t_host_ns: int) -> int:
        us = (t_host_ns - self._t_boot_ns) / 1000.0 * (1.0 + self.cfg.drift_ppm * 1e-6)
        return (self.cfg.boot_us + int(us)) % P.UINT32

    def _lat(self) -> int:
        return int((self.cfg.latency_s + self.rng.exponential(self.cfg.jitter_s)) * 1e9)

    def _emit(self, line: str, t_emit_ns: int) -> None:
        self._out.append((line, t_emit_ns + self._lat()))

    # ------------------------------------------------------------------ Transport
    def write_line(self, line: str) -> None:
        self.lines_sent.append(line)
        self._last_rx_ns = self.clock.now_ns()  # any received line feeds the watchdog
        self._handle(line.strip())

    def read_line(self, timeout: float = 0.0) -> tuple[str, int] | None:
        return self._out.popleft() if self._out else None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ test hooks
    def press_estop(self) -> None:
        self.estop_pin = True

    def release_estop(self) -> None:
        self.estop_pin = False

    def lose_steps(self, n: float) -> None:
        """Simulate n lost microsteps: the counter keeps its value but the table is n steps short."""
        self.true_offset -= n

    @property
    def true_angle_deg(self) -> float:
        return (self.pos + self.true_offset) / self.ups

    @property
    def true_rate_dps(self) -> float:
        return self.vel / self.ups

    # ------------------------------------------------------------------ helpers
    @property
    def moving(self) -> bool:
        return self.mode != "idle"

    def _hall_at(self, pos: float) -> bool:
        true = pos + self.true_offset
        return self._hall_lo <= true <= self._hall_hi

    def _reply(self, text: str, t_proc_ns: int) -> None:
        self._emit(text, t_proc_ns)

    def _go_idle(self, event: str | None, now_ns: int) -> None:
        self.mode, self.state, self._vel_kind = "idle", ("ESTOP" if self.estop_latched else "IDLE"), ""
        self._plan = self._sine = None
        self.vel = 0.0
        if event:
            self._emit(f"E {self.t_us(now_ns)} {event}", now_ns)

    def _latch_estop(self, now_ns: int) -> None:
        self.estop_latched = True
        self.en = False
        self.mode, self._vel_kind, self._plan, self._sine = "idle", "", None, None
        self.vel = 0.0
        self.state = "ESTOP"
        self._emit(f"E {self.t_us(now_ns)} ESTOP", now_ns)

    def _begin_stop(self) -> None:
        if self.mode == "idle":
            return
        if self.mode in ("plan", "sine"):
            self.mode = "vel"
        self._v_target, self._vel_kind = 0.0, "stop"

    # ------------------------------------------------------------------ command handling
    def _handle(self, line: str) -> None:
        now = self.clock.now_ns()
        t_proc = now + self._lat()            # when the firmware sees the line
        parts = line.split()
        if not parts:
            return
        cmd = parts[0].upper()
        args = parts[1:]

        def ok(extra: str = "") -> None:
            self._reply(f"OK {cmd}{(' ' + extra) if extra else ''}", t_proc)

        def err(code: str, text: str = "") -> None:
            self._reply(f"ERR {cmd} {code} {text}".rstrip(), t_proc)

        def ints(n: int) -> list[int] | None:
            if len(args) != n:
                return None
            try:
                return [int(a) for a in args]
            except ValueError:
                return None

        ready = self.en and not self.estop_latched
        if cmd == "PING":
            a = ints(1)
            if a is None:
                return err("ARGS", "PING <seq>")
            self._reply(f"PONG {a[0]} {self.t_us(t_proc)}", t_proc)
        elif cmd == "INFO":
            c = self.cfg
            ok(f"fw=calrig-uno/{c.fw_version} usteps={c.geometry.microsteps} steps_rev={c.geometry.steps_rev} "
               f"ratio={c.geometry.ratio:g} vmax_cap={int(round(self.vmax_cap))} "
               f"amax_cap={int(round(self.amax_cap))} hard_lim={self.hard} telem_hz={c.telem_hz}")
        elif cmd == "EN":
            a = ints(1)
            if a is None or a[0] not in (0, 1):
                return err("ARGS", "EN <0/1>")
            if a[0] and self.estop_latched:
                return err("ESTOP", "e-stop latched")
            self.en = bool(a[0])
            if not self.en and self.moving:
                self._go_idle(None, now)
            ok(str(a[0]))
        elif cmd == "ESTOP":
            self._latch_estop(now)
            ok()
        elif cmd == "RESET":
            if self.estop_pin:
                return err("ESTOP_PIN", "e-stop input still asserted")
            self.estop_latched = False
            if self.state == "ESTOP":
                self.state = "IDLE"
            ok()
        elif cmd == "CFG":
            a = ints(2)
            if a is None:
                return err("ARGS", "CFG <vmax> <amax>")
            self.vmax = float(min(max(a[0], 1), self.vmax_cap))
            self.amax = float(min(max(a[1], 1), self.amax_cap))
            ok(f"vmax={int(round(self.vmax))} amax={int(round(self.amax))}")
        elif cmd == "LIM":
            a = ints(2)
            if a is None or a[0] >= a[1]:
                return err("ARGS", "LIM <min> <max>")
            self.lim_lo, self.lim_hi = max(a[0], -self.hard), min(a[1], self.hard)
            ok(f"min={self.lim_lo} max={self.lim_hi}")
        elif cmd == "ZERO":
            if self.moving or abs(self.vel) > 0.0:
                return err("MOVING", "table is moving")
            self.true_offset += self.pos
            self.pos = 0.0
            ok()
        elif cmd == "MOVE":
            a = ints(1)
            if a is None:
                return err("ARGS", "MOVE <target>")
            if not ready:
                return err("NOT_READY", "driver disabled or e-stopped")
            if not self.lim_lo <= a[0] <= self.lim_hi:
                return err("LIMIT", "target outside soft limits")
            self._plan = plan_move(self.pos, self.vel, float(a[0]), self.vmax, self.amax)
            self._tau, self.mode, self.state, self._vel_kind = 0.0, "plan", "MOVE", ""
            ok()
        elif cmd == "VEL":
            a = ints(1)
            if a is None:
                return err("ARGS", "VEL <v>")
            if not ready:
                return err("NOT_READY", "driver disabled or e-stopped")
            self._v_target = float(max(-self.vmax, min(self.vmax, a[0])))
            self._start_vel("jog" if a[0] != 0 else "stop", "VEL")
            ok()
        elif cmd == "SINE":
            a = ints(3)
            if a is None or a[0] <= 0 or a[1] <= 0 or a[2] < 1:
                return err("ARGS", "SINE <amp> <period_ms> <cycles>")
            if not ready:
                return err("NOT_READY", "driver disabled or e-stopped")
            if self.moving or abs(self.vel) > 0.0:
                return err("MOVING", "sine starts from rest")
            w = 2.0 * math.pi / (a[1] / 1000.0)
            if a[0] * w * w > self.amax:
                return err("ACCEL", "amp*(2pi/period)^2 exceeds amax")
            if self.pos - a[0] < self.lim_lo or self.pos + a[0] > self.lim_hi:
                return err("LIMIT", "swing exceeds soft limits")
            self._sine = SineProfile(self.pos, float(a[0]), a[1] / 1000.0, a[2])
            self._tau, self.mode, self.state, self._vel_kind = 0.0, "sine", "SINE", ""
            ok()
        elif cmd == "HOME":
            a = ints(2)
            if a is None or a[0] not in (1, -1) or a[1] <= 0:
                return err("ARGS", "HOME <dir> <v>")
            if not ready:
                return err("NOT_READY", "driver disabled or e-stopped")
            if self._hall_at(self.pos):
                return err("ACTIVE", "home sensor already active; back off first")
            self._v_target = float(a[0] * min(a[1], self.vmax))
            self._start_vel("home_seek", "HOME")
            ok()
        elif cmd == "STOP":
            self._begin_stop()
            ok()
        elif cmd == "STREAM":
            a = ints(1)
            if a is None or a[0] not in (0, 1):
                return err("ARGS", "STREAM <0/1>")
            self.streaming = bool(a[0])
            ok(str(a[0]))
        elif cmd == "HB":
            ok()
        elif cmd == "STAT":
            ok(f"t={self.t_us(t_proc)} pos={int(round(self.pos))} vel={int(round(self.vel))} state={self.state} "
               f"en={int(self.en)} home={int(self._hall_at(self.pos))} estop_pin={int(self.estop_pin)}")
        else:
            err("UNKNOWN", "unknown command")

    def _start_vel(self, kind: str, state: str) -> None:
        self.mode, self.state, self._vel_kind, self._plan, self._sine = "vel", state, kind, None, None

    # ------------------------------------------------------------------ the 1 kHz control tick
    def _on_tick(self, now_ns: int) -> None:
        dt = (now_ns - self._last_ns) * 1e-9
        self._last_ns = now_ns
        if dt <= 0.0:
            return
        if self.estop_pin and not self.estop_latched:
            self._latch_estop(now_ns)
        if (self.moving and not (self._vel_kind in ("stop", "home_stop"))
                and (now_ns - self._last_rx_ns) * 1e-9 > self.cfg.watchdog_s):
            self._begin_stop()
            self._emit(f"E {self.t_us(now_ns)} WATCHDOG", now_ns)
            self._last_rx_ns = now_ns

        prev_pos = self.pos
        self._advance(dt, now_ns)
        self._hall_edges(prev_pos, self.pos, now_ns, dt)

        if self.streaming and now_ns >= self._next_telem_ns:
            self._emit(f"T {self.t_us(now_ns)} {int(round(self.pos))} {int(round(self.vel))}", now_ns)
            self._next_telem_ns += self._telem_period_ns
            if self._next_telem_ns <= now_ns:
                self._next_telem_ns = now_ns + self._telem_period_ns

    def _advance(self, dt: float, now_ns: int) -> None:
        if self.mode == "plan":
            self._tau += dt
            if self._tau >= self._plan.duration:
                self.pos = self._plan.target
                self._go_idle("DONE", now_ns)
            else:
                self.pos, self.vel = self._plan.eval(self._tau)
        elif self.mode == "sine":
            self._tau += dt
            if self._tau >= self._sine.duration:
                self.pos = self._sine.p0
                self._go_idle("DONE", now_ns)
            else:
                self.pos, self.vel = self._sine.eval(self._tau)
        elif self.mode == "vel":
            self._advance_vel(dt, now_ns)

    def _advance_vel(self, dt: float, now_ns: int) -> None:
        v, a = self.vel, self.amax
        want = self._v_target
        # soft limits: begin braking early enough that the table rests at or before the limit
        ahead = v * abs(v) / (2.0 * a) + v * dt
        toward_hi = v > 0.0 or (v == 0.0 and want > 0.0)
        toward_lo = v < 0.0 or (v == 0.0 and want < 0.0)
        if (toward_hi and self.pos + ahead >= self.lim_hi) or (toward_lo and self.pos + ahead <= self.lim_lo):
            if self._vel_kind in ("jog", "home_seek"):
                self._emit(f"E {self.t_us(now_ns)} LIMIT", now_ns)
                self._vel_kind = "limit_stop"
            self._v_target = want = 0.0
        dv = max(-a * dt, min(a * dt, want - v))
        v_new = v + dv
        self.pos = min(max(self.pos + 0.5 * (v + v_new) * dt, float(self.lim_lo)), float(self.lim_hi))
        self.vel = v_new
        if want == 0.0 and abs(v_new) < 1e-9:
            kind = self._vel_kind
            if kind == "home_stop":
                latched = self._home_latched
                self.pos -= latched
                self.true_offset += latched
                self._go_idle(f"HOMED {latched}", now_ns)
            elif kind == "limit_stop":
                self._go_idle(None, now_ns)
            else:
                self._go_idle("DONE", now_ns)

    def _hall_edges(self, p0: float, p1: float, now_ns: int, dt: float) -> None:
        active = self._hall_at(p1)
        if active == self._hall_active:
            return
        # entering: the lower edge when moving up / the upper edge when moving down; leaving: the other one
        if active == (p1 > p0):
            edge_true = self._hall_lo
        else:
            edge_true = self._hall_hi
        span = p1 - p0
        frac = 1.0 if span == 0.0 else min(1.0, max(0.0, (edge_true - self.true_offset - p0) / span))
        t_edge = int(now_ns - (1.0 - frac) * dt * 1e9)
        pos_edge = int(round(p0 + frac * span))
        self._emit(f"H {self.t_us(t_edge)} {pos_edge} {int(active)}", now_ns)
        self._hall_active = active
        if active and self.mode == "vel" and self._vel_kind == "home_seek":
            self._home_latched = pos_edge
            self._v_target, self._vel_kind = 0.0, "home_stop"


# ----------------------------------------------------------------------------------------- synthetic IMU
class SimImu:
    """ImuSource-like stream of RAW chip-frame samples driven by FakeRig's true yaw rate."""

    def __init__(self, clock: VirtualClock, rig: FakeRig, pose: str = "upright", R_gun_imu=None,
                 gyro_scale=(0.0, 0.0, 0.0), gyro_bias_dps=(0.0, 0.0, 0.0), gyro_noise_dps: float = 0.03,
                 accel_noise: float = 0.02, time_offset_s: float = 0.0, rate_hz: float = 500.0, seed: int = 2):
        if pose not in POSE_UP_GUN:
            raise ValueError(f"pose must be one of {sorted(POSE_UP_GUN)}")
        self.clock, self.rig, self.pose = clock, rig, pose
        self.R = np.asarray(R_gun_imu if R_gun_imu is not None else np.eye(3), dtype=np.float64)
        self.up_gun = np.array(POSE_UP_GUN[pose])
        self.gyro_scale = np.asarray(gyro_scale, dtype=np.float64)       # fractional error per CHIP axis
        self.gyro_bias = np.deg2rad(np.asarray(gyro_bias_dps, dtype=np.float64))  # rad/s, chip frame
        self.gyro_noise = math.radians(gyro_noise_dps)
        self.accel_noise = accel_noise
        self.time_offset_ns = int(round(time_offset_s * 1e9))
        self.rate_hz = rate_hz
        self.rng = np.random.default_rng(seed)
        self._period_ns = int(round(1e9 / rate_hz))
        self._next_ns = clock.now_ns()
        self._queue: list[ImuSample] = []
        self._started = False
        self.name = "sim-imu"
        clock.add_listener(self._on_tick)

    def start(self) -> None:
        self._started = True
        self._next_ns = self.clock.now_ns()

    def stop(self) -> None:
        self._started = False

    def read(self) -> list[ImuSample]:
        out, self._queue = self._queue, []
        return out

    def _on_tick(self, now_ns: int) -> None:
        if not self._started or now_ns < self._next_ns:
            return
        self._next_ns += self._period_ns
        if self._next_ns <= now_ns:
            self._next_ns = now_ns + self._period_ns
        w_gun = self.up_gun * math.radians(self.rig.true_rate_dps)       # rad/s, +Z-up right-hand yaw
        w_chip = self.R.T @ w_gun
        gyro = (1.0 + self.gyro_scale) * w_chip + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise, 3)
        accel = self.R.T @ (self.up_gun * G_MPS2) + self.rng.normal(0.0, self.accel_noise, 3)
        temp = 30.0 + 0.5 * (now_ns - self.rig._t_boot_ns) * 1e-9 / 60.0
        self._queue.append(ImuSample(t_ns=now_ns + self.time_offset_ns, gyro=gyro, accel=accel, temp_c=temp))


# ----------------------------------------------------------------------------------------- assembly
@dataclass
class SimBundle:
    clock: VirtualClock
    rig: FakeRig
    imu: SimImu
    truth: dict


_FLOAT_OPTS = {"start_deg", "drift_ppm", "latency_ms", "jitter_ms", "hall_start_deg", "hall_width_deg",
               "imu_offset_ms", "gyro_noise_dps", "accel_noise", "imu_rate_hz", "boot_us", "seed"}
_VEC3_OPTS = {"gyro_scale", "gyro_bias_dps", "mount_err_deg"}
_MAT_OPTS = {"r_gun_imu"}


def parse_sim_opts(items: list[str] | None) -> dict:
    """['gyro_scale=0.02,0,-0.01', 'imu_offset_ms=12', ...] -> dict (validated)."""
    out: dict = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"--sim-opt expects KEY=VALUE, got {item!r}")
        try:
            nums = [float(x) for x in val.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"--sim-opt {key}: not numbers: {val!r}") from None
        if key in _FLOAT_OPTS and len(nums) == 1:
            out[key] = nums[0]
        elif key in _VEC3_OPTS and len(nums) in (1, 3):
            out[key] = nums * 3 if len(nums) == 1 else nums
        elif key in _MAT_OPTS and len(nums) == 9:
            out[key] = np.asarray(nums).reshape(3, 3).tolist()
        else:
            known = sorted(_FLOAT_OPTS | _VEC3_OPTS | _MAT_OPTS)
            raise ValueError(f"unknown or malformed --sim-opt {key!r} (known: {', '.join(known)})")
    return out


def _rot_xyz(deg3) -> np.ndarray:
    rx, ry, rz = (math.radians(d) for d in deg3)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_sim(calrig_cfg, imu_cfg, pose: str = "upright", opts: dict | None = None) -> SimBundle:
    """FakeRig + SimImu on a fresh VirtualClock, parameterised from the config and --sim-opt overrides."""
    o = dict(opts or {})
    geom = P.Geometry(calrig_cfg.microsteps, calrig_cfg.steps_rev, calrig_cfg.ratio)
    clock = VirtualClock()
    rc = FakeRigConfig(
        geometry=geom,
        vmax_cap_dps=calrig_cfg.vmax_cap_dps, amax_cap_dps2=calrig_cfg.amax_cap_dps2,
        vmax_dps=calrig_cfg.vmax_dps, amax_dps2=calrig_cfg.amax_dps2,
        hard_lim_deg=calrig_cfg.hard_limit_deg, soft_lim_deg=calrig_cfg.soft_limit_deg,
        telem_hz=int(calrig_cfg.telem_hz),
    )
    for opt, attr, scale in (("start_deg", "start_angle_deg", 1.0), ("drift_ppm", "drift_ppm", 1.0),
                             ("latency_ms", "latency_s", 1e-3), ("jitter_ms", "jitter_s", 1e-3),
                             ("hall_start_deg", "hall_start_deg", 1.0), ("hall_width_deg", "hall_width_deg", 1.0),
                             ("seed", "seed", 1.0)):
        if opt in o:
            setattr(rc, attr, int(o[opt]) if opt == "seed" else o[opt] * scale)
    if "boot_us" in o:
        rc.boot_us = int(o["boot_us"]) % P.UINT32
    rig = FakeRig(clock, rc)

    R_cfg = np.asarray(o.get("r_gun_imu", imu_cfg.R_gun_imu), dtype=np.float64)
    R_true = R_cfg @ _rot_xyz(o["mount_err_deg"]) if "mount_err_deg" in o and "r_gun_imu" not in o else R_cfg
    scale = o.get("gyro_scale", (0.0, 0.0, 0.0))
    bias = o.get("gyro_bias_dps", (0.0, 0.0, 0.0))
    imu = SimImu(clock, rig, pose=pose, R_gun_imu=R_true, gyro_scale=scale, gyro_bias_dps=bias,
                 gyro_noise_dps=o.get("gyro_noise_dps", 0.03), accel_noise=o.get("accel_noise", 0.02),
                 time_offset_s=o.get("imu_offset_ms", 0.0) * 1e-3, rate_hz=o.get("imu_rate_hz", 500.0),
                 seed=int(o.get("seed", 1)) + 1)
    truth = {
        "R_gun_imu": R_true.tolist(),
        "gyro_scale_err_chip": [float(x) for x in scale],
        "gyro_bias_dps_chip": [float(x) for x in bias],
        "imu_time_offset_s": imu.time_offset_ns * 1e-9,
        "gyro_noise_dps": o.get("gyro_noise_dps", 0.03),
        "pose": pose,
        "rig_clock_drift_ppm": rc.drift_ppm,
        "rig_boot_us": rc.boot_us,
        "start_angle_deg": rc.start_angle_deg,
        "hall_edge_deg": [rc.hall_start_deg, rc.hall_start_deg + rc.hall_width_deg],
    }
    return SimBundle(clock, rig, imu, truth)
