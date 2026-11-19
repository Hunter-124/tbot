"""Arming interlock, actuator safety behaviour and the latency-actuator tool (sim backends only)."""

from __future__ import annotations

import json
import time

import pytest

from tbot.config import load_config
from tbot.hw import gpio as gpio_mod
from tbot.hw.actuator import (MAX_PRELOAD_FRAC, NoneActuator, Pq12Actuator, ServoActuator, SimActuator,
                              make_actuator, preload_pulse_us)
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
def _driven(pwm) -> list[float]:
    """Every non-zero pulse width the servo was commanded to (0.0 entries are 'PWM off')."""
    return [us for _t, us in pwm.pulses if us]


def test_servo_starts_limp_and_refuses_fire_unless_armed():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    assert pwm.pulse_us is None and pwm.enabled is False  # limp from construction: nothing is driven
    assert act.fire(time.monotonic_ns()) is False
    assert act.busy is False and act.last_fire_ns is None
    assert _driven(pwm) == []


def test_servo_rest_is_a_noop_unless_armed():
    """rest_us is a bench guess until jogged: driving it with nobody holding the button could press the trigger."""
    cfg = make_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    act.rest()
    act.abort()
    assert _driven(pwm) == [] and pwm.pulse_us is None


def test_servo_fires_returns_to_rest_then_goes_limp():
    cfg = make_cfg()
    cfg.actuator.servo.hold_ms = 50.0
    cfg.actuator.servo.return_ms = 50.0
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
    # ... and then it lets go, even with the button still held: no pulse is ever held indefinitely
    assert wait_for(lambda: pwm.pulse_us is None, 0.5)
    assert [round(us) for us in _driven(pwm)] == [round(cfg.actuator.servo.fire_us), round(cfg.actuator.servo.rest_us)]


def test_servo_never_holds_a_pulse_longer_than_hold_plus_return():
    cfg = make_cfg()
    cfg.actuator.servo.hold_ms = 40.0
    cfg.actuator.servo.return_ms = 40.0
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    for _ in range(3):
        t0 = time.monotonic_ns()
        assert act.fire(t0) is True
        assert wait_for(lambda: pwm.pulse_us is None, 0.5)
        driven_s = (pwm.pulses[-1][0] - t0) / 1e9
        assert driven_s < 0.040 + 0.040 + 0.1


def test_servo_disarm_during_hold_pulls_back_to_rest_then_goes_limp():
    cfg = make_cfg()
    cfg.actuator.servo.hold_ms = 1000.0
    cfg.actuator.servo.return_ms = 50.0
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    assert act.fire(time.monotonic_ns()) is True
    gpio.inject(cfg.arming.pin, 1)
    assert pwm.pulse_us == pytest.approx(cfg.actuator.servo.rest_us)
    assert act.busy is False
    assert wait_for(lambda: pwm.pulse_us is None, 0.5)
    time.sleep(1.1)  # past the cancelled hold timer: it must not have re-driven anything
    assert [round(us) for us in _driven(pwm)] == [round(cfg.actuator.servo.fire_us), round(cfg.actuator.servo.rest_us)]


def test_servo_refuses_to_fire_outside_the_calibrated_travel():
    cfg = make_cfg()
    cfg.actuator.servo.min_us, cfg.actuator.servo.max_us = 1400.0, 1600.0
    cfg.actuator.servo.rest_us, cfg.actuator.servo.fire_us = 1500.0, 1670.0
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    assert act.travel_error and "fire_us" in act.travel_error
    assert act.fire(time.monotonic_ns()) is False
    assert _driven(pwm) == []


def test_servo_release_while_idle_sends_nothing():
    cfg = make_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    gpio.inject(cfg.arming.pin, 1)
    time.sleep(0.05)
    assert _driven(pwm) == [] and pwm.pulse_us is None


# --------------------------------------------------------------------------------------------- servo pre-travel
def _preload_cfg(frac: float = 0.5, max_s: float = 30.0):
    cfg = make_cfg()
    s = cfg.actuator.servo
    s.rest_us, s.fire_us, s.break_us = 1420.0, 1530.0, 1510.0
    s.min_us, s.max_us = 1400.0, 1550.0
    s.preload_frac, s.preload_max_s = frac, max_s
    s.hold_ms, s.return_ms = 30.0, 30.0
    return cfg


def test_preload_pulse_is_the_fraction_of_rest_to_break_and_never_reaches_it():
    assert preload_pulse_us(1420, 1530, 1510, 0.5) == pytest.approx(1465.0)
    assert preload_pulse_us(1670, 1330, 1400, 0.5) == pytest.approx(1535.0)  # reversed lever
    assert preload_pulse_us(1420, 1530, 1510, 5.0) == pytest.approx(1420 + MAX_PRELOAD_FRAC * 90)  # capped
    assert preload_pulse_us(1420, 1530, 0.0, 0.5) is None  # break not marked
    assert preload_pulse_us(1420, 1530, 1510, 0.0) is None  # off
    assert preload_pulse_us(1420, 1530, 1600, 0.5) is None  # break beyond fire
    assert preload_pulse_us(1420, 1530, 1400, 0.5) is None  # break on the wrong side of rest
    assert preload_pulse_us(1420, 1530, 1510, 0.5, 1470.0, 1550.0) is None  # outside the travel


def test_servo_pre_travels_while_held_and_rests_then_limps_on_release():
    cfg = _preload_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    assert act.preload_us == pytest.approx(1465.0)
    arm(gpio, cfg, arming)
    assert wait_for(lambda: pwm.pulse_us is not None, 0.5)
    assert pwm.pulse_us == pytest.approx(1465.0) and act.preloaded
    time.sleep(0.1)
    assert pwm.pulse_us == pytest.approx(1465.0)  # held for as long as the button is
    gpio.inject(cfg.arming.pin, 1)
    assert pwm.pulse_us == pytest.approx(1420.0) and not act.preloaded
    assert wait_for(lambda: pwm.pulse_us is None, 0.5)
    assert [round(us) for us in _driven(pwm)] == [1465, 1420]


def test_servo_fires_from_the_pre_travel_and_does_not_re_preload_after_the_shot():
    cfg = _preload_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    assert wait_for(lambda: act.preloaded, 0.5)
    assert act.fire(time.monotonic_ns()) is True
    assert pwm.pulse_us == pytest.approx(1530.0) and not act.preloaded
    assert wait_for(lambda: pwm.pulse_us is None, 0.5)  # rest, then limp, button still held
    assert [round(us) for us in _driven(pwm)] == [1465, 1530, 1420]


def test_servo_fire_from_pre_travel_restarts_the_pwm_frame_but_not_from_limp():
    cfg = _preload_cfg()
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    assert wait_for(lambda: act.preloaded, 0.5)
    assert act.fire(time.monotonic_ns()) is True
    assert pwm.restarts == 1 and pwm.pulse_us == pytest.approx(1530.0)

    cfg2 = make_cfg()  # no pre-travel: the fire starts from limp, which already starts a fresh frame
    gpio2, arming2 = build(cfg2)
    pwm2 = SimPwm(0, 0, cfg2.actuator.servo.frame_hz)
    act2 = ServoActuator(cfg2, pwm2, arming2)
    arm(gpio2, cfg2, arming2)
    assert act2.fire(time.monotonic_ns()) is True
    assert pwm2.restarts == 0


def test_restart_wait_never_cuts_a_pulse():
    from tbot.hw.gpio import restart_wait_ns

    period, pulse, guard = 20_000_000, 1_200_000, 300_000
    assert restart_wait_ns(5_000_000, pulse, period, guard) == 0  # line low, far from the next pulse
    assert restart_wait_ns(500_000, pulse, period, guard) == 1_000_000  # inside the pulse: wait it out + guard
    assert restart_wait_ns(19_900_000, pulse, period, guard) == 100_000 + 1_500_000  # next pulse imminent
    assert restart_wait_ns(45_000_000, pulse, period, guard) == 0  # phase wraps by the period
    for ph in range(0, period, 50_000):
        w = restart_wait_ns(ph, pulse, period, guard)
        assert 0 <= w <= pulse + 2 * guard
        end = (ph + w) % period
        assert pulse + guard <= end <= period - guard


def test_hardware_pwm_restart_toggles_enable_around_the_new_duty(tmp_path, monkeypatch):
    from tbot.hw import gpio as gpio_mod

    ch = tmp_path / "pwmchip0" / "pwm0"
    ch.mkdir(parents=True)
    for name in ("period", "duty_cycle", "enable"):
        (ch / name).write_text("0")
    monkeypatch.setattr(gpio_mod, "PWM_SYSFS_ROOT", str(tmp_path))
    writes: list[tuple[str, str]] = []
    orig = gpio_mod.HardwarePwm._write_attr

    def spy(self, name, value, where=None):
        writes.append((name, str(value)))
        return orig(self, name, value, where)

    monkeypatch.setattr(gpio_mod.HardwarePwm, "_write_attr", spy)
    monkeypatch.setattr(gpio_mod.time, "sleep", lambda s: writes.append(("sleep", f"{s:.6f}")))
    pwm = HardwarePwm(0, 0, 50.0)
    pwm.set_pulse_us(1193.0)
    writes.clear()
    pwm._t_on_ns = time.monotonic_ns() - 5_000_000  # 5 ms into the frame: line low, no wait
    pwm.restart_pulse_us(1075.0)
    assert writes == [("enable", "0"), ("duty_cycle", "1075000"), ("enable", "1")]
    assert pwm.pulse_us == pytest.approx(1075.0)
    writes.clear()
    pwm._t_on_ns = time.monotonic_ns()  # frame just began: the 1075 us pulse is out, wait it out first
    pwm.restart_pulse_us(1100.0)
    assert writes[0][0] == "sleep" and 0.001 < float(writes[0][1]) <= 0.0014
    assert writes[1:] == [("enable", "0"), ("duty_cycle", "1100000"), ("enable", "1")]


def test_servo_pre_travel_times_out_to_limp():
    cfg = _preload_cfg(max_s=0.05)
    gpio, arming = build(cfg)
    pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
    act = ServoActuator(cfg, pwm, arming)
    arm(gpio, cfg, arming)
    assert wait_for(lambda: act.preloaded, 0.5)
    assert wait_for(lambda: pwm.pulse_us is None, 0.5)
    assert arming.armed and not act.preloaded


def test_servo_without_a_marked_break_or_with_preload_disabled_does_nothing_on_press():
    for tweak in ("no_break", "disabled"):
        cfg = _preload_cfg()
        if tweak == "no_break":
            cfg.actuator.servo.break_us = 0.0
        gpio, arming = build(cfg)
        pwm = SimPwm(0, 0, cfg.actuator.servo.frame_hz)
        ServoActuator(cfg, pwm, arming, preload=(tweak != "disabled"))
        arm(gpio, cfg, arming)
        time.sleep(0.05)
        assert _driven(pwm) == [], tweak


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


# ------------------------------------------------------------------ the ready light
def test_status_led_patterns_and_stale_reports():
    from tbot.hw import status_led as sl

    assert sl.level_for(sl.READY, 0.3) == 1 and sl.level_for(sl.READY, 7.7) == 1
    assert sl.level_for(sl.OFF, 0.3) == 0
    assert [sl.level_for(sl.WAIT, t) for t in (0.0, 0.05, 0.1, 0.15, 0.2)] == [1, 1, 0, 0, 1]   # 5 Hz
    assert [sl.level_for(sl.DRY, t) for t in (0.0, 0.9, 1.0, 1.9, 2.0)] == [1, 1, 0, 0, 1]      # 0.5 Hz

    now = [100.0]
    gpio = SimGpio()
    led = sl.StatusLed(gpio, 27, clock=lambda: now[0])
    assert gpio.modes[27] == "out" and gpio.levels[27] == 0
    led.set(sl.READY)
    assert led.shown() == sl.READY
    now[0] += sl.STALE_S + 0.1          # the frame loop stopped reporting: no longer "ready"
    assert led.shown() == sl.WAIT
    led.set(sl.DRY)
    assert led.shown() == sl.DRY
    led.close()
    assert gpio.levels[27] == 0


def test_status_led_thread_lights_and_goes_dark_on_close():
    import time as _time

    from tbot.hw import status_led as sl

    gpio = SimGpio()
    led = sl.StatusLed(gpio, 27, active_high=False)   # LED between 3.3 V and the pin: lit = level 0
    assert gpio.levels[27] == 1
    led.start()
    deadline = _time.monotonic() + 2.0
    while _time.monotonic() < deadline and gpio.levels[27] != 0:
        led.set(sl.READY)
        _time.sleep(0.02)
    assert gpio.levels[27] == 0
    led.close()
    assert gpio.levels[27] == 1


def test_status_led_never_raises_on_a_dead_gpio():
    from tbot.hw import status_led as sl

    class Dead:
        def setup_output(self, pin, level=0):
            raise OSError("no such pin")

        def write(self, pin, level):
            raise OSError("gone")

    led = sl.StatusLed(Dead(), 27)
    led.start()
    led.set(sl.READY)
    led.close()
