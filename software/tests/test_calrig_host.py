"""CalRig host software: protocol, t_us unwrap, clock fit, FakeRig physics/safety, session safety, end-to-end sim run.

Everything runs on a VirtualClock (no hardware, no real waiting) except the two threaded-driver tests, which use a
tiny scripted transport and real (short) time.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import time
from pathlib import Path

import numpy as np
import pytest

from tbot.calrig import protocol as P
from tbot.calrig.clocksync import ClockSync
from tbot.calrig.device import CommandError, CommandTimeout, DeviceError, RigDevice, RigFault, SerialTransport
from tbot.calrig.procedures import PROCEDURES, Procedure, Rate, Sine, Turns, make_procedure
from tbot.calrig.sim import make_sim, parse_sim_opts
from tbot.calrig.session import RigSession, SetupError, confirm_unloaded
from tbot.config import load_config
from tbot.tools import calrig as tool


# ------------------------------------------------------------------------------------------------ helpers
def sim_stack(opts: dict | None = None, pose: str = "upright", imu: bool = True):
    cfg = load_config()
    bundle = make_sim(cfg.calrig, cfg.imu, pose=pose, opts=opts)
    geom = P.Geometry(cfg.calrig.microsteps, cfg.calrig.steps_rev, cfg.calrig.ratio)
    dev = RigDevice(bundle.rig, geom, bundle.clock, threaded=False)
    dev.connect()
    session = RigSession(dev, cfg, imu=bundle.imu if imu else None, log=lambda m: None,
                         imu_meta={"source": "sim", "R_gun_imu": cfg.imu.R_gun_imu} if imu else None,
                         sim_truth=bundle.truth)
    return cfg, bundle, dev, session


def drain(dev: RigDevice, q) -> list:
    dev.poll()
    out = list(q)
    q.clear()
    return out


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------------------ protocol
def test_parse_async_lines():
    assert P.parse_line("T 123456 -400 266\r\n") == P.Telemetry(123456, -400, 266)
    assert P.parse_line("H 99 53 1") == P.HallEdge(99, 53, 1)
    assert P.parse_line("E 5 DONE") == P.Event(5, "DONE", None)
    assert P.parse_line("E 5 HOMED -107") == P.Event(5, "HOMED", -107)
    assert P.parse_line("PONG 7 4294967295") == P.Pong(7, 4294967295)


def test_parse_replies():
    r = P.parse_line("OK STAT t=10 pos=-5 vel=0 state=IDLE en=1 home=0 estop_pin=0")
    assert isinstance(r, P.Reply) and r.ok and r.cmd == "STAT"
    assert r.fields["state"] == "IDLE" and r.int_field("pos") == -5
    r = P.parse_line("OK INFO fw=calrig-uno/1.0 usteps=16 steps_rev=200 ratio=1 vmax_cap=800 amax_cap=533 "
                     "hard_lim=889 telem_hz=200")
    assert r.fields["fw"] == "calrig-uno/1.0" and r.int_field("hard_lim") == 889
    r = P.parse_line("ERR MOVE LIMIT target outside soft limits")
    assert not r.ok and r.cmd == "MOVE" and r.code == "LIMIT" and r.text == "target outside soft limits"
    assert P.parse_line("OK EN 1").fields == {"1": ""}
    assert isinstance(P.parse_line("calrig boot banner"), P.Unknown)
    assert isinstance(P.parse_line(""), P.Unknown)


def test_parse_malformed_raises():
    for bad in ("T 1 2 x", "H a b c", "E zz DONE", "PONG 1 q"):
        with pytest.raises(P.ProtocolError):
            P.parse_line(bad)


def test_command_formatting_and_geometry():
    assert P.cmd_move(-800) == "MOVE -800"
    assert P.cmd_cfg(266.67, 177.8) == "CFG 267 178"
    assert P.cmd_sine(133, 6500, 6) == "SINE 133 6500 6"
    assert P.cmd_home(1, 44.4) == "HOME 1 44"
    assert P.cmd_en(True) == "EN 1" and P.cmd_stream(False) == "STREAM 0"
    with pytest.raises(P.ProtocolError):
        P.cmd_home(0, 10)
    g = P.Geometry(16, 200, 1.0)
    assert g.usteps_per_deg == pytest.approx(8.8889, abs=1e-4)
    assert g.deg_per_ustep == pytest.approx(0.1125)
    assert g.to_usteps(90.0) == 800 and g.to_deg(800) == pytest.approx(90.0)
    assert P.command_name("move 5") == "MOVE"


def test_tus_unwrap_across_wrap():
    u = P.TimeUnwrapper()
    seq = [4294967000, 296, 5000, 4294967290 % P.UINT32]  # the last one is a slightly OLD sample after the wrap
    out = [u.unwrap(t) for t in seq[:3]]
    assert out == [4294967000, 4294967000 + 592, 4294967000 + 592 + 4704]
    assert out == sorted(out)
    # an out-of-order sample from just before the wrap must land just before, not a whole wrap away
    assert abs(u.unwrap(4294967290) - 4294967290) < 10
    # many wraps, steady 100 Hz ticks
    u2 = P.TimeUnwrapper()
    t, last = 4_000_000_000, None
    for _ in range(3000):
        t += 3_000_000  # 3 s steps: 3000 of them = 2.5 h = 2 wraps
        v = u2.unwrap(t % P.UINT32)
        assert last is None or v - last == 3_000_000
        last = v
    assert last - 4_000_000_000 == 3000 * 3_000_000


# ------------------------------------------------------------------------------------------------ clock fit
def test_clock_fit_recovers_offset_and_drift_through_wrap():
    rng = np.random.default_rng(5)
    true_offset_ns, true_drift_ppm = 123_456_789_000, 42.0
    boot_us = (1 << 32) - 20_000_000  # wraps 20 s into the run
    unwrap = P.TimeUnwrapper()
    sync = ClockSync()
    t_host = 5_000_000_000_000
    for i in range(80):
        t_send = t_host + i * 1_000_000_000
        up = int(300_000 + rng.exponential(200_000))
        down = int(300_000 + rng.exponential(200_000))
        if i % 9 == 0:
            down += 20_000_000  # a stalled USB frame: a long, asymmetric round trip that must be rejected
        t_fw_host = t_send + up
        # firmware clock runs `drift` fast: us = (host - offset)/1000 * (1 + drift)
        t_us = int((t_fw_host - true_offset_ns) / 1000.0 * (1 + true_drift_ppm * 1e-6))
        t_us_raw = (boot_us + t_us - (t_host - true_offset_ns) // 1000) % P.UINT32
        sync.add(t_send, t_send + up + down, unwrap.unwrap(t_us_raw))
    fit = sync.fit()
    assert fit.n == 40 and fit.n_total == 80
    assert fit.drift_ppm == pytest.approx(-true_drift_ppm, abs=2.0)  # host-ns per firmware-ns: slope < 1
    mid = t_host + 40_000_000_000
    t_us_mid = int((mid + 300_000 - true_offset_ns) / 1000.0 * (1 + true_drift_ppm * 1e-6))
    raw = (boot_us + t_us_mid - (t_host - true_offset_ns) // 1000) % P.UINT32
    # conversion error well under a camera frame / IMU sample (the offset is only defined to the RTT asymmetry)
    est = fit.to_host_ns(unwrap.unwrap(raw))
    assert abs(est - (mid + 300_000)) < 300_000
    assert fit.resid_us < 400.0


def test_clock_fit_short_baseline_fixes_slope():
    s = ClockSync()
    for i in range(10):
        s.add(i * 10_000_000, i * 10_000_000 + 1_000_000, i * 10_000)
    fit = s.fit()
    assert fit.drift_ppm == 0.0 and fit.n == 5
    assert fit.to_host_ns(0) == pytest.approx(500_000, abs=1)
    assert ClockSync().fit() is None


# ------------------------------------------------------------------------------------------------ FakeRig
def raw_cmd(b, text: str) -> list[str]:
    """Send a command to the FakeRig directly and return the lines it produced."""
    b.rig.write_line(text)
    out = []
    while (x := b.rig.read_line(0.0)) is not None:
        out.append(x[0])
    return out


def telemetry(b, seconds: float, until_idle: bool = False) -> list[P.Telemetry]:
    out, t_end = [], b.clock.now_ns() + int(seconds * 1e9)
    while b.clock.now_ns() < t_end:
        b.clock.sleep(0.05)
        b.rig.write_line("HB")
        for line, _ in iter(lambda: b.rig.read_line(0.0), None):
            m = P.parse_line(line)
            if isinstance(m, P.Telemetry):
                out.append(m)
        if until_idle and b.rig.mode == "idle":
            break
    return out


def test_fakerig_trapezoid_obeys_vmax_amax_and_reaches_target():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu, opts={"latency_ms": 0.0, "jitter_ms": 0.0})
    raw_cmd(b, "EN 1")
    raw_cmd(b, "STREAM 1")
    raw_cmd(b, "CFG 267 178")  # 30 deg/s, 20 deg/s^2
    assert "OK MOVE" in raw_cmd(b, "MOVE 800")
    t = telemetry(b, 12.0, until_idle=True)
    assert b.rig.mode == "idle" and b.rig.pos == 800 and b.rig.state == "IDLE"
    vel = np.array([m.vel for m in t], dtype=float)
    us = np.array([m.t_us for m in t], dtype=float)
    assert vel.max() <= 267 + 1 and vel.max() >= 260           # cruises at vmax
    k = 20                                                     # 100 ms windows: int rounding error is ~10 sps^2
    acc = (vel[k:] - vel[:-k]) / ((us[k:] - us[:-k]) * 1e-6)
    assert np.abs(acc).max() <= 178 * 1.07 and np.abs(acc).max() >= 178 * 0.9
    pos = np.array([m.pos for m in t])
    assert np.all(np.diff(pos) >= 0) and pos.max() <= 800       # no overshoot


def test_fakerig_triangle_move_and_smooth_replacement():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu, opts={"latency_ms": 0.0, "jitter_ms": 0.0})
    for c in ("EN 1", "STREAM 1", "CFG 400 178"):
        raw_cmd(b, c)
    raw_cmd(b, "MOVE 600")
    telemetry(b, 2.0)
    v_before = b.rig.vel
    assert v_before > 100
    # reverse while moving: no velocity step, brakes through zero, then goes the other way
    raw_cmd(b, "MOVE -300")
    t = telemetry(b, 25.0, until_idle=True)
    vel = np.array([v_before] + [m.vel for m in t], dtype=float)
    assert np.abs(np.diff(vel)).max() < 60                      # 5 ms steps: a 178 sps^2 ramp is < 1 sps per ms
    assert b.rig.pos == -300 and b.rig.mode == "idle"


def test_fakerig_soft_limits_and_vel_stops_at_limit():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu, opts={"latency_ms": 0.0, "jitter_ms": 0.0})
    raw_cmd(b, "EN 1")
    assert "ERR MOVE LIMIT target outside soft limits" in raw_cmd(b, "MOVE 801")
    assert "ERR MOVE LIMIT target outside soft limits" in raw_cmd(b, "MOVE -801")
    assert "OK LIM min=-889 max=889" in raw_cmd(b, "LIM -2000 2000")  # clamped inside the +-100 deg hard limit
    raw_cmd(b, "LIM -800 800")
    raw_cmd(b, "STREAM 1")
    raw_cmd(b, "CFG 400 178")
    raw_cmd(b, "VEL 400")
    lines, peak = [], 0.0
    for _ in range(400):
        b.clock.sleep(0.05)
        b.rig.write_line("HB")
        peak = max(peak, b.rig.pos)
        lines += [x[0] for x in iter(lambda: b.rig.read_line(0.0), None)]
        if b.rig.mode == "idle":
            break
    assert b.rig.mode == "idle" and peak <= 800 and b.rig.pos > 790
    assert any(re_line.endswith(" LIMIT") and re_line.startswith("E ") for re_line in lines)
    assert any(x.startswith("ERR SINE LIMIT") for x in raw_cmd(b, "SINE 100 6500 1"))   # 790 + 100 > 800


def test_fakerig_sine_profile_and_accel_refusal():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu, opts={"latency_ms": 0.0, "jitter_ms": 0.0})
    raw_cmd(b, "EN 1")
    raw_cmd(b, "STREAM 1")
    assert any(x.startswith("ERR SINE ACCEL") for x in raw_cmd(b, "SINE 400 2000 2"))
    assert "OK SINE" in raw_cmd(b, "SINE 133 6500 4")  # 15 deg, 6.5 s; the first and last cycle are ramps
    t = telemetry(b, 30.0, until_idle=True)
    pos = np.array([m.pos for m in t])
    assert 125 <= pos.max() <= 134 and -134 <= pos.min() <= -125
    assert b.rig.mode == "idle" and b.rig.pos == 0
    vel = np.array([m.vel for m in t], dtype=float)
    us = np.array([m.t_us for m in t], dtype=float)
    acc = (vel[20:] - vel[:-20]) / ((us[20:] - us[:-20]) * 1e-6)
    assert np.abs(acc).max() <= b.rig.amax * 1.07  # the raised-cosine envelope keeps the ramp inside amax


def test_fakerig_not_ready_and_zero_rules():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu)
    for c in ("MOVE 10", "VEL 5", "SINE 10 6500 1", "HOME 1 40"):
        assert any(x.startswith(f"ERR {c.split()[0]} NOT_READY") for x in raw_cmd(b, c)), c
    raw_cmd(b, "EN 1")
    raw_cmd(b, "MOVE 400")
    b.clock.sleep(0.5)
    assert any(x.startswith("ERR ZERO MOVING") for x in raw_cmd(b, "ZERO"))
    raw_cmd(b, "STOP")
    b.clock.sleep(0.3 * 4)
    raw_cmd(b, "HB")
    b.clock.sleep(3.0)
    assert b.rig.mode == "idle"
    assert "OK ZERO" in raw_cmd(b, "ZERO") and b.rig.pos == 0


def test_estop_latch_pin_and_command():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu, opts={"latency_ms": 0.0, "jitter_ms": 0.0})
    raw_cmd(b, "EN 1")
    raw_cmd(b, "STREAM 1")
    raw_cmd(b, "MOVE 800")
    b.clock.sleep(1.0)
    assert b.rig.vel > 0
    b.rig.press_estop()
    b.clock.sleep(0.002)  # the pin is polled every control tick: latched within ~1 tick
    assert b.rig.estop_latched and not b.rig.en and b.rig.vel == 0.0 and b.rig.state == "ESTOP"
    pos = b.rig.pos
    b.clock.sleep(1.0)
    assert b.rig.pos == pos                                   # nothing moves while latched
    lines = raw_cmd(b, "STAT")
    assert any("state=ESTOP" in x and "en=0" in x and "estop_pin=1" in x for x in lines)
    assert any(x.startswith("ERR MOVE NOT_READY") for x in raw_cmd(b, "MOVE 0"))
    assert any(x.startswith("ERR EN ESTOP") for x in raw_cmd(b, "EN 1"))
    assert any(x.startswith("ERR RESET ESTOP_PIN") for x in raw_cmd(b, "RESET"))
    assert b.rig.estop_latched                                 # still latched: the pin is still asserted
    b.rig.release_estop()
    assert "OK RESET" in raw_cmd(b, "RESET")
    assert not b.rig.en                                        # RESET leaves the driver disabled
    assert any(x.startswith("ERR MOVE NOT_READY") for x in raw_cmd(b, "MOVE 0"))
    assert "OK EN 1" in raw_cmd(b, "EN 1")
    assert "OK MOVE" in raw_cmd(b, "MOVE 0")
    # ESTOP command: immediate, latched
    raw_cmd(b, "VEL 100")
    b.clock.sleep(0.2)
    out = raw_cmd(b, "ESTOP")
    assert "OK ESTOP" in out and b.rig.vel == 0.0 and b.rig.estop_latched and not b.rig.en
    estops = [x for x in out if x.startswith("E ") and x.endswith(" ESTOP")]
    assert len(estops) == 1


def test_watchdog_stops_without_traffic_and_heartbeat_prevents_it():
    cfg = load_config()
    quiet = {"latency_ms": 0.0, "jitter_ms": 0.0}
    # with a heartbeat every 0.5 s a long motion is never interrupted
    b = make_sim(cfg.calrig, cfg.imu, opts=quiet)
    for c in ("EN 1", "CFG 400 178", "VEL 100"):
        raw_cmd(b, c)
    for i in range(60):                      # 6 s
        b.clock.sleep(0.1)
        if i % 5 == 0:
            raw_cmd(b, "HB")
    assert b.rig.mode == "vel" and b.rig.vel > 95
    # the same rig, silent for > 1 s: controlled stop + E WATCHDOG, driver stays enabled
    lines = []
    for _ in range(80):
        b.clock.sleep(0.05)
        lines += [x[0] for x in iter(lambda: b.rig.read_line(0.0), None)]
    assert any(x.startswith("E ") and x.endswith(" WATCHDOG") for x in lines)
    assert b.rig.mode == "idle" and b.rig.vel == 0.0
    assert b.rig.en and not b.rig.estop_latched               # a controlled stop, not an e-stop
    # and 0.9 s of silence is still fine
    b2 = make_sim(cfg.calrig, cfg.imu, opts=quiet)
    for c in ("EN 1", "VEL 200"):
        raw_cmd(b2, c)
    b2.clock.sleep(0.9)
    assert b2.rig.mode == "vel"


def test_hall_edges_and_home_offset_and_lost_steps():
    cfg, b, dev, s = sim_stack({"start_deg": -12.0}, imu=False)
    s.setup()
    s.enable()
    s.start_streaming()
    first = s.home()                                          # power-up -> hall edge
    assert first["offset_ustep"] == pytest.approx(12.0 * 8.8889, abs=2)  # 12 deg of travel from the power-up spot
    assert b.rig.true_angle_deg == pytest.approx(0.0, abs=0.15)
    s.move_to(60.0)
    s.move_to(-60.0)
    s.move_to(0.0)
    b.rig.lose_steps(18)                                      # 2 deg of slipped steps
    second = s.home()                                         # the counter is 18 steps AHEAD of the table
    assert second["offset_ustep"] == pytest.approx(18, abs=2)
    s.shutdown()


def test_hall_edges_are_captured_at_the_zone_boundaries():
    cfg = load_config()
    b = make_sim(cfg.calrig, cfg.imu, opts={"start_deg": -12.0, "latency_ms": 0.0, "jitter_ms": 0.0})
    for c in ("EN 1", "CFG 200 178", "VEL 100"):
        raw_cmd(b, c)
    edges = []
    for _ in range(90):
        b.clock.sleep(0.1)
        for line, _ in iter(lambda: b.rig.read_line(0.0), None):
            m = P.parse_line(line)
            if isinstance(m, P.HallEdge):
                edges.append(m)
        b.rig.write_line("HB")
    assert [e.level for e in edges] == [1, 0]
    ups = 200 * 16 / 360.0
    assert edges[0].pos == pytest.approx(12.0 * ups, abs=1)        # lower edge: 12 deg above the power-up spot
    assert edges[1].pos == pytest.approx(18.0 * ups, abs=1)        # upper edge: 6 deg wide zone
    assert edges[1].t_us > edges[0].t_us


def test_home_backs_off_when_sensor_active():
    cfg, b, dev, s = sim_stack({"start_deg": 2.0}, imu=False)  # starts inside the magnet zone
    s.setup()
    s.enable()
    s.start_streaming()
    out = s.home()
    sent = b.rig.lines_sent
    assert "MOVE" in " ".join(sent[:sent.index(next(c for c in sent if c.startswith("HOME")))])  # backed off first
    assert abs(b.rig.true_angle_deg) < 0.5 and isinstance(out["offset_ustep"], int)
    s.shutdown()


# ------------------------------------------------------------------------------------------------ device (virtual)
def test_device_commands_and_stray_replies():
    cfg, b, dev, s = sim_stack(imu=False)
    info = dev.get_info()
    assert info["usteps"] == "16" and info["hard_lim"] == "889"
    with pytest.raises(CommandError) as ei:
        dev.command("MOVE 10")
    assert ei.value.code == "NOT_READY"
    r = dev.command("MOVE 10", check=False)
    assert not r.ok
    assert dev.command("HB").ok
    st = dev.stat()
    assert st.state == "IDLE" and st.en == 0 and st.pos == 0
    assert dev.link_error is None
    # the unknown command still gets a (negative) reply and does not hang the matcher
    assert dev.command("FROB", check=False).code == "UNKNOWN"
    s.shutdown()


def test_device_t_us_unwrap_end_to_end_and_clock_sync():
    cfg, b, dev, s = sim_stack(imu=False)
    dev.sync_burst(20)
    s.setup()
    s.enable()
    s.start_streaming()
    for _ in range(10):
        b.clock.sleep(1.0)
        dev.poll()
        dev.ping()
    tele = [x for x in drain(dev, dev.telemetry_q)]
    raw = np.array([m.t_us for _, _, m in tele])
    unwrapped = np.array([u for _, u, _ in tele])
    assert (np.diff(raw) < 0).any(), "boot_us is set just below the uint32 wrap: the raw counter must wrap"
    assert np.all(np.diff(unwrapped) > 0) and np.all(np.diff(unwrapped) < 20_000)
    assert unwrapped[-1] % P.UINT32 == raw[-1]
    fit = dev.sync.fit()
    assert fit.drift_ppm == pytest.approx(-35.0, abs=12.0) and fit.resid_us < 500
    # converted rig time lands on the host clock within the RTT asymmetry (sub-ms), not the 1 ms telemetry period
    t_last = b.clock.now_ns()
    err = fit.to_host_ns(unwrapped[-1]) - tele[-1][0]
    assert abs(err) < 3_000_000
    assert t_last >= tele[-1][0] - 5_000_000
    s.shutdown()


def test_confirm_unloaded_requires_exact_word():
    for bad in ("", "yes", "unloaded", "UNLOADED please", "y"):
        with pytest.raises(SetupError):
            confirm_unloaded(lambda _p, a=bad: a, lambda _m: None)

    def eof(_p):
        raise EOFError

    with pytest.raises(SetupError):
        confirm_unloaded(eof, lambda _m: None)
    ok = confirm_unloaded(lambda _p: " UNLOADED \n", lambda _m: None)
    assert ok["confirmed"] is True


# ------------------------------------------------------------------------------------------------ procedures
def test_procedure_validation_and_params():
    cfg, b, dev, s = sim_stack(imu=False)
    lim = s.limits
    Sine().validate(lim)
    with pytest.raises(ValueError, match="acceleration"):
        Sine(amplitude_deg=40.0, period_s=6.5).validate(lim)   # 40 (2pi/6.5)^2 = 37 > 20
    with pytest.raises(ValueError, match="soft limit"):
        Turns(amplitude_deg=95.0).validate(lim)
    with pytest.raises(ValueError, match="deg/s"):
        Turns(rate_dps=120.0).validate(lim)
    Rate().validate(lim)
    with pytest.raises(ValueError, match="never cruises"):
        Rate(rates_dps=(10.0, 70.0)).validate(lim)
    p = make_procedure("rate", {"rates_dps": "10,20", "repeats": "2", "accel_dps2": "none"})
    assert p.rates_dps == (10.0, 20.0) and p.repeats == 2 and p.accel_dps2 is None
    assert make_procedure("hold", {"duration_s": "5"}).duration_s == 5.0
    with pytest.raises(ValueError, match="unknown parameter"):
        make_procedure("turns", {"bogus": 1})
    with pytest.raises(ValueError, match="unknown procedure"):
        make_procedure("spin")
    assert set(PROCEDURES) == {"hold", "turns", "sine", "rate", "home_check"}
    s.shutdown()


class _Scripted(Procedure):
    """Test procedure: starts a long move, then runs `action(session, rig)`."""

    name = "scripted"
    needs_home = False

    def __init__(self, action):
        self.action = action

    def params(self):
        return {}

    def validate(self, limits):
        pass

    def run(self, s):
        s.set_rates(30.0, 20.0)
        s.begin_motion()
        s.dev.command(P.cmd_move(s.geometry.to_usteps(80.0)))
        s.wait(1.0)
        return self.action(s) or {}


def _run_scripted(tmp_path, action):
    cfg, b, dev, s = sim_stack({"latency_ms": 0.1, "jitter_ms": 0.0})
    s.setup()
    exc = None
    try:
        s.run(_Scripted(action), "upright", {"confirmed": False, "simulated": True}, root=tmp_path)
    except BaseException as e:  # noqa: BLE001
        exc = e
    return b, dev, s, exc


def _meta(tmp_path) -> dict:
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return json.loads((d / "meta.json").read_text()), d


def test_ctrl_c_sends_stop_then_en0_and_finalizes_run(tmp_path):
    def action(s):
        raise KeyboardInterrupt

    b, dev, s, exc = _run_scripted(tmp_path, action)
    assert isinstance(exc, KeyboardInterrupt)
    sent = [x for x in b.rig.lines_sent]
    assert "STOP" in sent and "EN 0" in sent
    assert sent.index("STOP") < len(sent) - 1 - sent[::-1].index("EN 0")
    assert not b.rig.en and b.rig.mode == "idle" and b.rig.vel == 0.0   # decelerated to rest, then disabled
    assert 0 < b.rig.pos < 80 * 8.9                                       # it stopped, it did not finish the move
    meta, d = _meta(tmp_path)
    assert meta["status"] == "interrupted"
    assert (d / "rig.csv").exists() and (d / "imu.csv").exists()


def test_exception_path_disables_driver(tmp_path):
    def action(s):
        raise RuntimeError("boom")

    b, dev, s, exc = _run_scripted(tmp_path, action)
    assert isinstance(exc, RuntimeError)
    assert b.rig.lines_sent[-1] in ("EN 0", "STAT", "STREAM 0") or "EN 0" in b.rig.lines_sent
    assert not b.rig.en and b.rig.vel == 0.0
    assert _meta(tmp_path)[0]["status"] == "error: RuntimeError: boom"


def test_estop_during_run_aborts_and_records_event(tmp_path):
    def action(s):
        b_rig = s.dev.transport
        b_rig.press_estop()
        s.wait(2.0)

    b, dev, s, exc = _run_scripted(tmp_path, action)
    assert isinstance(exc, RigFault) and "ESTOP" in str(exc)
    assert b.rig.estop_latched and not b.rig.en
    meta, d = _meta(tmp_path)
    assert meta["status"].startswith("error: RigFault")
    assert "ESTOP" in [r["code"] for r in read_csv(d / "events.csv")]


def test_telemetry_loss_aborts(tmp_path):
    def action(s):
        s.dev.transport.streaming = False   # the firmware silently stops streaming
        s.wait(3.0)

    b, dev, s, exc = _run_scripted(tmp_path, action)
    assert isinstance(exc, RigFault) and "telemetry" in str(exc)
    assert not b.rig.en


def test_heartbeat_keeps_watchdog_quiet_during_long_wait(tmp_path):
    def action(s):
        s.wait(6.0)   # several watchdog periods with the host doing nothing but waiting

    b, dev, s, exc = _run_scripted(tmp_path, action)
    assert exc is None
    _, d = _meta(tmp_path)
    assert "WATCHDOG" not in [r["code"] for r in read_csv(d / "events.csv")]
    assert sum(1 for x in b.rig.lines_sent if x == "HB") >= 3


# ------------------------------------------------------------------------------------------------ end to end
def test_run_turns_sim_end_to_end(tmp_path):
    rc = tool.main(["run", "turns", "--sim", "--out", str(tmp_path), "-p", "repeats=1",
                    "--sim-opt", "gyro_scale=0.02,0.01,-0.015", "--sim-opt", "imu_offset_ms=12",
                    "--sim-opt", "gyro_bias_dps=0.3,-0.2,0.1", "--pose", "left_side_down"])
    assert rc == 0
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert d.name.endswith("_turns") and len(d.name.split("_")[0]) == 15
    assert sorted(p.name for p in d.iterdir()) == ["cmd.csv", "events.csv", "hall.csv", "imu.csv", "meta.json",
                                                   "rig.csv"]
    meta = json.loads((d / "meta.json").read_text())
    for key in ("procedure", "params", "firmware", "deg_per_ustep", "sign_convention", "clock_fit", "imu", "pose",
                "host", "software_version", "start", "end", "unloaded_confirmation", "status"):
        assert key in meta, key
    assert meta["procedure"] == "turns" and meta["pose"] == "left_side_down" and meta["status"] == "completed"
    assert meta["deg_per_ustep"] == pytest.approx(0.1125)
    assert meta["firmware"]["usteps"] == "16"
    assert set(meta["clock_fit"]) >= {"offset_ns", "drift_ppm", "resid_us", "n"} and meta["clock_fit"]["n"] >= 30
    assert meta["clock_fit"]["drift_ppm"] == pytest.approx(-35.0, abs=5.0)
    assert meta["clock_fit"]["resid_us"] < 500
    assert "clock" not in meta
    assert meta["unloaded_confirmation"]["simulated"] is True
    assert meta["imu"]["samples_are_raw_chip_frame"] is True and len(meta["imu"]["R_gun_imu"]) == 3
    assert meta["sim_truth"]["imu_time_offset_s"] == pytest.approx(0.012)

    rig = read_csv(d / "rig.csv")
    assert list(rig[0]) == ["t_ns", "t_us", "pos_ustep", "angle_deg", "vel_deg_s"]
    t = np.array([int(r["t_ns"]) for r in rig])
    ang = np.array([float(r["angle_deg"]) for r in rig])
    assert np.all(np.diff(t) > 0) and 5.0e6 * 0.5 < np.median(np.diff(t)) < 5.0e6 * 1.5   # 200 Hz on the host clock
    assert ang.max() == pytest.approx(90.0, abs=0.2) and ang.min() == pytest.approx(-90.0, abs=0.2)
    assert abs(ang[-1]) < 0.2 and abs(ang[0]) < 1.0              # recorded window starts and ends on the hall zero
    assert max(abs(float(r["vel_deg_s"])) for r in rig) <= 30.1  # the turns rate

    cmds = [r["command"] for r in read_csv(d / "cmd.csv")]
    assert any(c.startswith("PING") for c in cmds) and "MOVE 800" in cmds and "MOVE -800" in cmds
    assert any(c.startswith("CFG 267") for c in cmds) and cmds.count("MOVE 0") >= 2
    assert list(read_csv(d / "hall.csv")[0]) == ["t_ns", "t_us", "pos_ustep", "angle_deg", "level"]
    assert len(read_csv(d / "hall.csv")) >= 2
    ev = [r["code"] for r in read_csv(d / "events.csv")]
    assert ev.count("DONE") >= 3 and "HOMED" not in ev          # the initial homing is outside the recording

    imu = read_csv(d / "imu.csv")
    assert list(imu[0]) == ["t_ns", "gx", "gy", "gz", "ax", "ay", "az", "temp_c"]
    ti = np.array([int(r["t_ns"]) for r in imu])
    assert np.all(np.diff(ti) > 0) and len(imu) > 0.9 * 500 * (t[-1] - t[0]) * 1e-9
    gyro = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in imu])
    accel = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in imu])
    # RAW chip frame: left_side_down puts world-up on gun -Y; chip = R^T gun with the config R_gun_imu
    R = np.array(load_config().imu.R_gun_imu)
    up_chip = R.T @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(accel.mean(axis=0), up_chip * 9.80665, atol=0.05)
    peak_chip = gyro[np.argmax(np.abs(gyro @ up_chip))]
    assert abs(peak_chip @ up_chip) == pytest.approx(math.radians(30.0) * 1.01, rel=0.05)


def test_run_hold_and_info_and_estop_sim(tmp_path, capsys):
    assert tool.main(["run", "hold", "--sim", "--out", str(tmp_path), "-p", "duration_s=5", "--pose", "inverted"]) == 0
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    imu = read_csv(d / "imu.csv")
    chip = np.array([np.mean([float(r[k]) for r in imu]) for k in ("ax", "ay", "az")])
    # inverted: world up is gun -Z; the chip sees it through R_gun_imu (chip = R^T gun)
    R = np.asarray(load_config().imu.R_gun_imu, dtype=float)
    assert chip == pytest.approx(R.T @ np.array([0.0, 0.0, -9.80665]), abs=0.05)
    assert tool.main(["info", "--sim"]) == 0
    out = capsys.readouterr().out
    assert "firmware : fw=calrig-uno/" in out and "0.1125" in out.replace("0.11250", "0.1125")
    assert tool.main(["estop", "--sim"]) == 0
    assert "ESTOP -> OK ESTOP" in capsys.readouterr().out


def test_run_refuses_without_unloaded_confirmation(monkeypatch, tmp_path):
    holder = {}

    def fake_open(cfg_, port, baud):
        bundle = make_sim(cfg_.calrig, cfg_.imu)
        holder["bundle"] = bundle
        geom = P.Geometry(cfg_.calrig.microsteps, cfg_.calrig.steps_rev, cfg_.calrig.ratio)
        return RigDevice(bundle.rig, geom, bundle.clock, threaded=False)

    monkeypatch.setattr(tool, "_open_real_device", fake_open)
    monkeypatch.setattr("builtins.input", lambda _p="": "yes")
    rc = tool.main(["run", "hold", "--no-imu", "--out", str(tmp_path), "-p", "duration_s=2"])
    assert rc == 1
    sent = holder["bundle"].rig.lines_sent
    assert "EN 1" not in sent and not holder["bundle"].rig.en and not any(c.startswith(("MOVE", "HOME", "VEL"))
                                                                         for c in sent)
    assert not list(tmp_path.iterdir())
    for cmd in (["home"], ["jog", "--deg", "5"]):
        holder.clear()
        assert tool.main(cmd) == 1
        assert "EN 1" not in holder["bundle"].rig.lines_sent
    # typing it enables the rig (hold needs no homing)
    monkeypatch.setattr("builtins.input", lambda _p="": "UNLOADED")
    assert tool.main(["run", "hold", "--no-imu", "--out", str(tmp_path), "-p", "duration_s=2"]) == 0
    assert "EN 1" in holder["bundle"].rig.lines_sent
    meta, _ = _meta(tmp_path)
    assert meta["unloaded_confirmation"]["confirmed"] is True and meta["unloaded_confirmation"]["simulated"] is False
    assert not holder["bundle"].rig.en


def test_run_home_check_reports_lost_steps(tmp_path):
    rc = tool.main(["run", "home_check", "--sim", "--out", str(tmp_path), "--no-imu"])
    assert rc == 0
    meta, d = _meta(tmp_path)
    res = meta["result"]
    assert res["ok"] is True and abs(res["lost_deg"]) <= 0.5
    ev = [(r["code"], r["arg"]) for r in read_csv(d / "events.csv")]
    assert [c for c, _ in ev].count("HOMED") == 2


def test_sim_opts_parse():
    o = parse_sim_opts(["gyro_scale=0.02", "imu_offset_ms=5", "r_gun_imu=1,0,0,0,1,0,0,0,1"])
    assert o["gyro_scale"] == [0.02] * 3 and o["imu_offset_ms"] == 5.0 and o["r_gun_imu"][2][2] == 1.0
    with pytest.raises(ValueError):
        parse_sim_opts(["nonsense=1"])
    with pytest.raises(ValueError):
        parse_sim_opts(["gyro_scale=a,b,c"])


def test_cli_registered_and_config_section():
    from tbot import cli

    assert cli.COMMANDS["calrig"][0] == "tbot.tools.calrig"
    c = load_config().calrig
    assert (c.microsteps, c.steps_rev, c.ratio) == (16, 200, 1.0)
    assert (c.vmax_dps, c.amax_dps2, c.vmax_cap_dps, c.amax_cap_dps2) == (45.0, 20.0, 90.0, 60.0)
    assert (c.soft_limit_deg, c.hard_limit_deg, c.telem_hz, c.baud) == (90.0, 100.0, 200, 115200)
    assert c.run_root == "runs/calrig"


def test_analyze_command_delegates(tmp_path):
    pytest.importorskip("tbot.calrig.analysis")
    assert tool.main(["run", "turns", "--sim", "--out", str(tmp_path), "-p", "repeats=1"]) == 0
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    calib = tmp_path / "calib"
    assert tool.main(["analyze", str(d), "--calib-dir", str(calib)]) == 0   # dry run by default
    assert (d / "report.md").exists() and (d / "report.json").exists()
    assert not (calib / "imu.json").exists()
    assert tool.main(["analyze", str(d), "--calib-dir", str(calib), "--apply"]) == 0
    written = json.loads((calib / "imu.json").read_text())
    assert "gyro_bias_rad_s" in written and "time_offset_ns" in written
    assert abs(written["time_offset_ns"]) < 2_000_000                          # the sim IMU has no offset
    assert tool.main(["analyze", str(tmp_path / "nope")]) == 1


# ------------------------------------------------------------------------------------------------ threaded driver
class ScriptedTransport:
    """Minimal firmware stand-in for the threaded reader: replies come from a queue, like a serial port."""

    def __init__(self):
        self.rx: "queue.Queue[tuple[str, int]]" = queue.Queue()
        self.sent: list[str] = []
        self.silent = False
        self.fail_reads = False

    def write_line(self, line: str) -> None:
        self.sent.append(line)
        if self.silent:
            return
        parts = line.split()
        stamp = time.monotonic_ns()
        if parts[0] == "PING":
            self.rx.put(("T 1000 5 0", stamp))                      # an async line racing the reply
            self.rx.put((f"PONG {parts[1]} {(time.monotonic_ns() // 1000) % P.UINT32}", stamp))
        elif parts[0] == "MOVE":
            self.rx.put(("ERR MOVE NOT_READY driver disabled", stamp))
        elif parts[0] == "STAT":
            self.rx.put(("garbage line from a boot banner", stamp))
            self.rx.put(("OK STAT t=1 pos=0 vel=0 state=IDLE en=1 home=0 estop_pin=0", stamp))
        else:
            self.rx.put((f"OK {parts[0]}", stamp))

    def read_line(self, timeout: float):
        if self.fail_reads:
            raise DeviceError("serial read failed: device disconnected")
        try:
            return self.rx.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        pass


def test_threaded_device_matching_heartbeat_and_link_loss():
    tr = ScriptedTransport()
    dev = RigDevice(tr, P.Geometry(), threaded=True, command_timeout_s=0.5)
    dev.connect(timeout_s=2.0)
    assert len(dev.sync) >= 1
    st = dev.stat()                                           # a junk line before the reply is skipped
    assert st.state == "IDLE" and dev.unknown_lines[-1].startswith("garbage")
    with pytest.raises(CommandError):
        dev.command("MOVE 5")
    assert dev.command("MOVE 5", check=False).code == "NOT_READY"
    assert any(isinstance(m, P.Telemetry) for _, _, m in dev.telemetry_q)  # async T lines were queued
    dev.start_heartbeat(0.05)
    deadline = time.monotonic() + 1.0
    while tr.sent.count("HB") < 3 and time.monotonic() < deadline:
        dev.touch()
        time.sleep(0.02)
    assert tr.sent.count("HB") >= 3
    # owner stops touching: the heartbeat thread must stop feeding the firmware watchdog
    time.sleep(2.2)
    n = tr.sent.count("HB")
    time.sleep(0.3)
    assert tr.sent.count("HB") == n
    tr.silent = True
    with pytest.raises(CommandTimeout):
        dev.command("STAT", timeout_s=0.2)
    tr.fail_reads = True
    deadline = time.monotonic() + 1.0
    while dev.link_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "disconnected" in (dev.link_error or "")
    with pytest.raises(RigFault):
        dev.command("STAT", timeout_s=0.2)
    dev.close()


@pytest.mark.skipif(os.name != "posix", reason="pty needed")
def test_serial_transport_line_framing_over_pty():
    pytest.importorskip("serial")
    import pty

    master, slave = pty.openpty()
    try:
        tr = SerialTransport(os.ttyname(slave), 115200)
        os.write(master, b"OK INFO fw=x usteps=16\r\nT 1 2 3\nPON")
        got = []
        for _ in range(10):
            x = tr.read_line(0.05)
            if x:
                got.append(x[0])
        assert got == ["OK INFO fw=x usteps=16", "T 1 2 3"]       # CRLF stripped, partial line held back
        os.write(master, b"G 4 5\n")
        x = tr.read_line(0.2)
        assert x is not None and x[0] == "PONG 4 5" and x[1] > 0
        tr.write_line("PING 9")
        assert os.read(master, 64).endswith(b"PING 9\n")
        tr.close()
    finally:
        os.close(master)
        os.close(slave)


# ------------------------------------------------------------------------------------------------ integrator asks
def test_extra_poses_accepted(tmp_path):
    for pose, up in (("muzzle_up", (1, 0, 0)), ("muzzle_down", (-1, 0, 0))):
        out = tmp_path / pose
        assert tool.main(["run", "hold", "--sim", "--out", str(out), "-p", "duration_s=3", "--pose", pose]) == 0
        (d,) = list(out.iterdir())
        assert json.loads((d / "meta.json").read_text())["pose"] == pose
        imu = read_csv(d / "imu.csv")
        a = np.mean([[float(r[k]) for k in ("ax", "ay", "az")] for r in imu], axis=0)
        R = np.array(load_config().imu.R_gun_imu)
        assert np.allclose(a, R.T @ np.array(up, float) * 9.80665, atol=0.05)


def test_home_check_homes_in_the_same_direction_and_logs_commands_verbatim(tmp_path):
    assert tool.main(["run", "home_check", "--sim", "--out", str(tmp_path), "--no-imu", "--set",
                      "calrig.home_dir=-1", "--sim-opt", "start_deg=15"]) == 0
    _, d = _meta(tmp_path)
    cmds = [r["command"] for r in read_csv(d / "cmd.csv")]
    homes = [c for c in cmds if c.startswith("HOME")]
    assert len(homes) == 2 and homes[0] == homes[1] and homes[0].startswith("HOME -1 ")
    assert "ZERO" not in cmds
    idx = [i for i, c in enumerate(cmds) if c.startswith("HOME")]
    assert any(c.startswith("MOVE") for c in cmds[idx[0]:idx[1]])


def test_imu_calibration_gyro_scale_is_optional_and_backward_compatible():
    from types import SimpleNamespace

    from tbot.imu.mpu6050 import ImuCalibration

    cfg = load_config()
    gyro = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]])
    accel = np.array([[0.0, 0.0, 9.8], [1.0, 1.0, 1.0]])
    base = {"gyro_bias_rad_s": [0.01, 0.02, 0.03], "R_gun_imu": np.eye(3).tolist()}
    plain = ImuCalibration.from_config(cfg, SimpleNamespace(imu=dict(base)))
    assert np.array_equal(plain.gyro_scale, np.ones(3))
    g0 = (gyro - np.array([0.01, 0.02, 0.03]))
    assert np.allclose(plain.apply(accel, gyro)[1], g0)
    scaled = ImuCalibration.from_config(cfg, SimpleNamespace(imu={**base, "gyro_scale": [1.01, 0.98, 1.0]}))
    assert np.allclose(scaled.apply(accel, gyro)[1], g0 * np.array([1.01, 0.98, 1.0]))
    assert np.allclose(scaled.apply(accel, gyro)[0], plain.apply(accel, gyro)[0])   # accel untouched
    # scale is applied in the CHIP frame, before R_gun_imu
    R = np.array(cfg.imu.R_gun_imu)
    both = ImuCalibration.from_config(cfg, SimpleNamespace(imu={"gyro_scale": [1.1, 1.0, 0.9]}))
    expect = (gyro * np.array([1.1, 1.0, 0.9])) @ R.T
    assert np.allclose(both.apply(accel, gyro)[1], expect)
    # direct construction without the new field still works
    old = ImuCalibration(np.zeros(3), np.zeros(3), np.ones(3), np.eye(3), 0)
    assert np.array_equal(old.gyro_scale, np.ones(3))
    with pytest.raises(ValueError):
        ImuCalibration.from_config(cfg, SimpleNamespace(imu={"gyro_scale": [1.0, 0.0, 1.0]}))
    with pytest.raises(ValueError):
        ImuCalibration.from_config(cfg, SimpleNamespace(imu={"gyro_scale": [1.0, 1.0]}))
