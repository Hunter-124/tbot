"""Arming interlock, actuator safety behaviour and the latency-actuator tool (sim backends only)."""

from __future__ import annotations

import json
import time

import pytest

from tbot.config import load_config
from tbot.hw import gpio as gpio_mod
from tbot.hw.actuator import (NoneActuator, Pq12Actuator, ServoActuator, SimActuator, make_actuator)
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import HardwarePwm, SimGpio, SimPwm, open_gpio
from tbot.tools import latency_actuator

DEBOUNCE_MS = 20.0


def make_cfg():
    """Default config on the sim backends, with short motion times so the tests stay fast."""
    cfg = load_config()
    cfg.gpio.backend = "sim"
    cfg.arming.debounce_ms = DEBOUNCE_MS
    cfg.actuator.servo.hold_ms = 30.0
    cfg.actuator.pq12.extend_ms = 30.0
    cfg.actuator.pq12.retract_ms = 20.0
    cfg.actuator.sim.delay_ms = 10.0
    return cfg


def wait_for(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def arm(gpio: SimGpio, cfg, arming: ArmingSwitch) -> None:
    gpio.inject(cfg.arming.pin, 0)  # NO button to GND, active low
    assert wait_for(lambda: arming.armed), "the switch never armed"


def build(cfg):
    gpio = SimGpio()
    arming = ArmingSwitch(gpio, cfg)
    return gpio, arming


# --------------------------------------------------------------------------------------------- arming
def test_short_press_never_arms():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    gpio.inject(cfg.arming.pin, 0)
    time.sleep(DEBOUNCE_MS / 1000.0 / 4)
    gpio.inject(cfg.arming.pin, 1)
    time.sleep(DEBOUNCE_MS / 1000.0 * 2)
    assert arming.armed is False
    assert arming.press_count == 0


def test_stable_press_arms_after_debounce_only():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    gpio.inject(cfg.arming.pin, 0)
    assert arming.armed is False  # not yet: the debounce window has not elapsed
    assert wait_for(lambda: arming.armed, 0.5)
    assert arming.press_count == 1


def test_release_disarms_immediately_and_notifies():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    events: list[tuple[bool, int]] = []
    arming.on_change(lambda armed, t_ns: events.append((armed, t_ns)))
    arm(gpio, cfg, arming)
    gpio.inject(cfg.arming.pin, 1, t_ns=1234)
    assert arming.armed is False  # synchronous: no polling, no debounce on the release path
    assert [e[0] for e in events] == [True, False]
    assert events[-1][1] == 1234 and isinstance(events[-1][1], int)


def test_open_circuit_reads_not_armed():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    assert arming.armed is False  # pull-up high = nothing pressed
    arm(gpio, cfg, arming)
    # Wire break / missed edge: the level changes without an edge callback ever running.
    gpio.levels[cfg.arming.pin] = 1
    assert arming.armed is False
    assert arming.try_consume_shot() is False


def test_max_shots_per_arm_enforced_and_reset_by_a_new_press():
    cfg = make_cfg()
    cfg.fire.max_shots_per_arm = 2
    gpio, arming = build(cfg)
    arm(gpio, cfg, arming)
    assert [arming.try_consume_shot() for _ in range(3)] == [True, True, False]
    assert arming.shots_this_press == 2
    gpio.inject(cfg.arming.pin, 1)
    assert arming.try_consume_shot() is False
    gpio.inject(cfg.arming.pin, 0)
    assert wait_for(lambda: arming.armed, 0.5)
    assert arming.press_count == 2
    assert arming.shots_this_press == 0
    assert arming.try_consume_shot() is True


# --------------------------------------------------------------------------------------------- servo
def test_servo_refuses_fire_unless_armed():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    assert act.fire(time.monotonic_ns()) is False
    assert pwm.pulse_us == pytest.approx(cfg.actuator.servo.rest_us)
    assert act.busy is False and act.last_fire_ns is None


def test_servo_fires_then_returns_to_rest_after_the_hold():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    t = time.monotonic_ns()
    assert act.fire(t) is True
    assert pwm.pulse_us == pytest.approx(cfg.actuator.servo.fire_us)
    assert act.busy is True and act.last_fire_ns == t
    assert act.fire(time.monotonic_ns()) is False  # one cycle at a time
    assert wait_for(lambda: not act.busy, 0.5)
    assert pwm.pulse_us == pytest.approx(cfg.actuator.servo.rest_us)
    assert act.last_rest_ns is not None and act.last_rest_ns > t


def test_servo_disarm_during_hold_returns_to_rest_immediately():
    cfg = make_cfg()
    cfg.actuator.servo.hold_ms = 1000.0
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    gpio.inject(cfg.arming.pin, 1)
    assert pwm.pulse_us == pytest.approx(cfg.actuator.servo.rest_us)
    assert act.busy is False
    n_pulses = len(pwm.pulses)
    time.sleep(0.05)
    assert len(pwm.pulses) == n_pulses  # the hold timer was cancelled, not merely overridden


# --------------------------------------------------------------------------------------------- pq12
def _both_high_ever(writes, pin_extend: int, pin_retract: int) -> bool:
    level = {pin_extend: 0, pin_retract: 0}
    for _, pin, value in writes:
        if pin in level:
            level[pin] = value
            if level[pin_extend] and level[pin_retract]:
                return True
    return False


def _levels_at_end(writes, pin_extend: int, pin_retract: int) -> tuple[int, int]:
    level = {pin_extend: 0, pin_retract: 0}
    for _, pin, value in writes:
        if pin in level:
            level[pin] = value
    return level[pin_extend], level[pin_retract]


def test_pq12_fire_cycle_never_drives_both_pins_high():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    p = cfg.actuator.pq12
    act = Pq12Actuator(cfg, gpio, arming)
    assert act.fire(time.monotonic_ns()) is False  # interlock first
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    assert gpio.read(p.pin_extend) == 1 and gpio.read(p.pin_retract) == 0
    assert wait_for(lambda: not act.busy, 1.0)
    assert _both_high_ever(gpio.writes, p.pin_extend, p.pin_retract) is False
    assert _levels_at_end(gpio.writes, p.pin_extend, p.pin_retract) == (0, 0)
    act.close()
    assert _both_high_ever(gpio.writes, p.pin_extend, p.pin_retract) is False


def test_pq12_abort_mid_extend_retracts_and_cancels_the_pending_timers():
    cfg = make_cfg()
    cfg.actuator.pq12.extend_ms = 1000.0
    gpio, arming = build(cfg)
    p = cfg.actuator.pq12
    act = Pq12Actuator(cfg, gpio, arming)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    time.sleep(0.01)
    gpio.inject(cfg.arming.pin, 1)  # disarm mid-extend
    assert gpio.read(p.pin_extend) == 0 and gpio.read(p.pin_retract) == 1
    assert wait_for(lambda: not act.busy, 1.0)
    assert _levels_at_end(gpio.writes, p.pin_extend, p.pin_retract) == (0, 0)
    n_writes = len(gpio.writes)
    time.sleep(0.05)
    assert len(gpio.writes) == n_writes  # the extend timer never fired after the abort
    assert _both_high_ever(gpio.writes, p.pin_extend, p.pin_retract) is False
    act.close()
    assert _both_high_ever(gpio.writes, p.pin_extend, p.pin_retract) is False
    assert _levels_at_end(gpio.writes, p.pin_extend, p.pin_retract) == (0, 0)


def test_pq12_repeated_aborts_do_not_restart_the_retract():
    """The watchdog may abort every frame; that must not re-energise the motor each time."""
    cfg = make_cfg()
    cfg.actuator.pq12.extend_ms = 1000.0
    cfg.actuator.pq12.retract_ms = 40.0
    gpio, arming = build(cfg)
    p = cfg.actuator.pq12
    act = Pq12Actuator(cfg, gpio, arming)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    for _ in range(4):
        act.abort()
        time.sleep(0.005)
    assert wait_for(lambda: not act.busy, 1.0)
    retract_highs = [w for w in gpio.writes if w[1] == p.pin_retract and w[2] == 1]
    assert len(retract_highs) == 1
    assert _levels_at_end(gpio.writes, p.pin_extend, p.pin_retract) == (0, 0)


def test_pq12_close_leaves_both_pins_low():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    p = cfg.actuator.pq12
    act = Pq12Actuator(cfg, gpio, arming)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    act.close()
    assert gpio.read(p.pin_extend) == 0 and gpio.read(p.pin_retract) == 0
    assert act.fire(time.monotonic_ns()) is False  # closed actuator never moves again
    assert _both_high_ever(gpio.writes, p.pin_extend, p.pin_retract) is False


# --------------------------------------------------------------------------------------------- sim / none
def test_sim_actuator_records_breaks_and_drives_the_break_sense_pin():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    pin = cfg.gpio.break_sense_pin
    gpio.setup_input(pin, "up")
    act = SimActuator(cfg, arming, gpio=gpio, break_pin=pin, hold_ms=5.0)
    assert act.fire(time.monotonic_ns()) is False
    arm(gpio, cfg, arming)
    t = time.monotonic_ns()
    assert act.fire(t) is True
    assert act.fire_times == [t]
    assert act.break_times == [t + int(cfg.actuator.sim.delay_ms * 1e6)]
    assert wait_for(lambda: gpio.read(pin) == 0, 0.5)  # contact closes at the simulated break
    assert wait_for(lambda: gpio.read(pin) == 1, 0.5)  # and re-opens after the return to rest
    assert wait_for(lambda: not act.busy, 0.5)


def test_sim_actuator_abort_reopens_the_contact():
    cfg = make_cfg()
    cfg.actuator.sim.delay_ms = 1.0
    gpio, arming = build(cfg)
    pin = cfg.gpio.break_sense_pin
    gpio.setup_input(pin, "up")
    act = SimActuator(cfg, arming, gpio=gpio, break_pin=pin, hold_ms=1000.0)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    assert wait_for(lambda: gpio.read(pin) == 0, 0.5)
    gpio.inject(cfg.arming.pin, 1)
    assert gpio.read(pin) == 1 and act.busy is False


def test_none_actuator_never_fires():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    act = NoneActuator(arming)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is False
    assert act.busy is False
    act.rest()
    act.abort()
    act.close()


def test_make_actuator_selects_the_sim_backends():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    cfg.actuator.type = "servo"
    servo = make_actuator(cfg, gpio, arming)
    assert servo.kind == "servo" and isinstance(servo.pwm, SimPwm)  # no LgpioGpio -> no sysfs PWM
    cfg.actuator.type = "pq12"
    assert make_actuator(cfg, gpio, arming).kind == "pq12"
    cfg.actuator.type = "sim"
    assert make_actuator(cfg, gpio, arming).kind == "sim"
    cfg.actuator.type = "none"
    assert make_actuator(cfg, gpio, arming).fire(1) is False
    cfg.actuator.type = "hydraulic"
    with pytest.raises(ValueError):
        make_actuator(cfg, gpio, arming)


# --------------------------------------------------------------------------------------------- gpio / pwm
def test_open_gpio_backends():
    cfg = make_cfg()
    assert isinstance(open_gpio(cfg), SimGpio)
    cfg.gpio.backend = "nonsense"
    with pytest.raises(ValueError):
        open_gpio(cfg)


def test_sim_gpio_edges_respect_direction_and_log_writes():
    gpio = SimGpio()
    seen: list[tuple[int, int, int]] = []
    gpio.setup_input(5, "down")
    assert gpio.read(5) == 0
    gpio.watch(5, "rising", lambda pin, level, t_ns: seen.append((pin, level, t_ns)))
    gpio.inject(5, 0)  # no change -> no edge
    gpio.inject(5, 1, t_ns=42)
    gpio.inject(5, 0)  # falling -> not this watcher
    assert seen == [(5, 1, 42)]
    gpio.setup_output(6, 0)
    gpio.write(6, 1)
    assert [(pin, level) for _, pin, level in gpio.writes] == [(6, 0), (6, 1)]


def test_hardware_pwm_missing_overlay_message(tmp_path, monkeypatch):
    monkeypatch.setattr(gpio_mod, "PWM_SYSFS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        HardwarePwm(0, 0, 50.0)
    assert "dtoverlay=pwm,pin=18,func=2" in str(exc.value)


# --------------------------------------------------------------------------------------------- tool
def test_latency_actuator_sim_measures_the_configured_delay(tmp_path, capsys):
    delay_ms = 12.0
    rc = latency_actuator.main([
        "--sim", "--trials", "5", "--calib-dir", str(tmp_path),
        "--set", f"actuator.sim.delay_ms={delay_ms}",
        "--set", "arming.debounce_ms=2.0",
    ])
    assert rc == 0
    data = json.loads((tmp_path / "latency.json").read_text())
    record = data["actuator"]["sim"]
    assert record["cmd_to_break_ms"]["n"] == 5
    assert record["cmd_to_break_ms"]["mean"] == pytest.approx(delay_ms, abs=2.0)
    assert record["cmd_to_rest_ms"]["n"] == 5
    assert record["timeouts"] == 0
    assert set(record["cmd_to_break_ms"]) == {"mean", "std", "p50", "p95", "min", "max", "n"}
    assert record["created"] and "sim" in record["notes"]


def test_latency_actuator_refuses_the_none_actuator(tmp_path):
    assert latency_actuator.main(["--type", "none", "--calib-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "latency.json").exists()


def test_run_trials_pauses_while_the_button_is_released(tmp_path):
    cfg = make_cfg()
    cfg.actuator.sim.delay_ms = 2.0
    gpio, arming = build(cfg)
    sense = latency_actuator.BreakSense(gpio, cfg.gpio.break_sense_pin)
    act = SimActuator(cfg, arming, gpio=gpio, break_pin=cfg.gpio.break_sense_pin, hold_ms=2.0)
    arm(gpio, cfg, arming)
    trials = latency_actuator.run_trials(
        act, sense, arming, trials=2, timeout_s=0.5, wait_range_s=(0.0, 0.0),
        sleep=lambda s: None, log=lambda *a: None,
    )
    assert len(trials) == 2
    assert all(t.complete for t in trials)
    assert all(t.cmd_to_break_ms == pytest.approx(2.0, abs=0.5) for t in trials)

    gpio.inject(cfg.arming.pin, 1)  # operator lets go: the loop must give up instead of firing
    none = latency_actuator.run_trials(
        act, sense, arming, trials=1, timeout_s=0.2, wait_range_s=(0.0, 0.0),
        sleep=lambda s: None, arm_timeout_s=0.05, log=lambda *a: None,
    )
    assert none == []
    assert act.fire_times == [t.cmd_ns for t in trials]
