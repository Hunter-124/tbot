"""Trigger actuators (contract: docs/ARCHITECTURE.md [hw]).

Safety invariant: `fire()` refuses unless the arming button is held, every actuator subscribes to
`ArmingSwitch.on_change` and `abort()`s to rest the moment it is released, and `abort()` cancels any
pending return/retract timer. All state changes are taken under a per-actuator lock (edges arrive on the
lgpio callback thread, returns on timer threads, fires on the pipeline thread).

`last_rest_ns` (not in the protocol) is the monotonic time at which the return-to-rest command was issued
(servo: end of the hold; PQ12: start of the retract); `tbot latency-actuator` measures the break-sense
re-open against it.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from tbot.config import Config
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import Gpio, HardwarePwm, LgpioGpio, SimGpio, SimPwm

#: SimActuator hold between the simulated break and the simulated return to rest (no config key exists).
SIM_HOLD_MS = 50.0


class Actuator(Protocol):
    kind: str
    busy: bool
    last_fire_ns: int | None

    def rest(self) -> None: ...

    def fire(self, t_ns: int) -> bool: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


class _BaseActuator:
    """Shared interlock subscription, timer bookkeeping and busy flag."""

    kind = "base"

    def __init__(self, arming: ArmingSwitch | None):
        self._arming = arming
        self._lock = threading.RLock()
        self._timers: list[threading.Timer] = []
        self._busy = False
        self._closed = False
        self.last_fire_ns: int | None = None
        self.last_rest_ns: int | None = None
        if arming is not None:
            arming.on_change(self._on_arming)

    # -- helpers ------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def _armed(self) -> bool:
        # No switch wired = never armed: the interlock fails safe.
        return bool(self._arming is not None and self._arming.armed)

    def _on_arming(self, armed: bool, t_ns: int) -> None:
        if not armed:
            self.abort()

    def _schedule(self, delay_s: float, fn) -> None:
        timer = threading.Timer(max(0.0, delay_s), fn)
        timer.daemon = True
        with self._lock:
            self._timers.append(timer)
        timer.start()

    def _cancel_timers(self) -> None:
        with self._lock:
            timers, self._timers = self._timers, []
        for t in timers:
            t.cancel()

    def _join_timers(self, timeout_s: float) -> None:
        with self._lock:
            timers = list(self._timers)
        deadline = time.monotonic() + max(0.0, timeout_s)
        for t in timers:
            t.join(max(0.0, deadline - time.monotonic()))

    def _forget_finished(self) -> None:
        with self._lock:
            self._timers = [t for t in self._timers if t.is_alive()]


class ServoActuator(_BaseActuator):
    """MG996R on hardware PWM: rest pulse -> fire pulse -> back to rest after cfg.actuator.servo.hold_ms."""

    kind = "servo"

    def __init__(self, cfg: Config, pwm: HardwarePwm | SimPwm, arming: ArmingSwitch | None):
        super().__init__(arming)
        s = cfg.actuator.servo
        self.pwm = pwm
        self.rest_us = float(s.rest_us)
        self.fire_us = float(s.fire_us)
        self.hold_s = max(0.0, float(s.hold_ms) / 1000.0)
        self._settle_s = 0.0 if getattr(pwm, "is_sim", False) else min(0.06, 2.0 / float(s.frame_hz))
        self.rest()

    def rest(self) -> None:
        self._cancel_timers()
        with self._lock:
            self.pwm.set_pulse_us(self.rest_us)
            self._busy = False

    def fire(self, t_ns: int) -> bool:
        if not self._armed():
            return False
        with self._lock:
            if self._closed or self._busy:
                return False
            if not self._armed():  # re-check under the lock: the button may have been released meanwhile
                return False
            self.last_fire_ns = int(t_ns)
            self.last_rest_ns = None
            self._busy = True
            self.pwm.set_pulse_us(self.fire_us)
        self._forget_finished()
        self._schedule(self.hold_s, self._return_to_rest)
        return True

    def _return_to_rest(self) -> None:
        with self._lock:
            if not self._busy:
                return
            self.pwm.set_pulse_us(self.rest_us)
            self.last_rest_ns = time.monotonic_ns()
            self._busy = False

    def abort(self) -> None:
        self._cancel_timers()
        with self._lock:
            was_busy = self._busy
            self.pwm.set_pulse_us(self.rest_us)
            if was_busy:
                self.last_rest_ns = time.monotonic_ns()
            self._busy = False

    def close(self) -> None:
        self.abort()
        if self._settle_s:
            time.sleep(self._settle_s)  # let the rest pulse actually go out before the signal stops
        with self._lock:
            self._closed = True
        close = getattr(self.pwm, "close", None)
        if callable(close):
            close()
        else:
            self.pwm.disable()


class Pq12Actuator(_BaseActuator):
    """Actuonix PQ12 through an H-bridge on two GPIOs. Both pins high = shoot-through: never allowed."""

    kind = "pq12"

    def __init__(self, cfg: Config, gpio: Gpio, arming: ArmingSwitch | None):
        super().__init__(arming)
        p = cfg.actuator.pq12
        self._gpio = gpio
        self.pin_extend = int(p.pin_extend)
        self.pin_retract = int(p.pin_retract)
        self.extend_s = max(0.0, float(p.extend_ms) / 1000.0)
        self.retract_s = max(0.0, float(p.retract_ms) / 1000.0)
        self._phase = "idle"  # idle | extend | retract (also guards timers a cancel could not stop)
        gpio.setup_output(self.pin_extend, 0)
        gpio.setup_output(self.pin_retract, 0)

    def _drive(self, extend: int, retract: int) -> None:
        """Set both H-bridge inputs, lows first, so the two are never high at the same instant."""
        extend, retract = int(bool(extend)), int(bool(retract))
        if not extend:
            self._gpio.write(self.pin_extend, 0)
        if not retract:
            self._gpio.write(self.pin_retract, 0)
        if extend:
            self._gpio.write(self.pin_extend, 1)
        if retract:
            self._gpio.write(self.pin_retract, 1)

    def _enter_retract(self) -> None:
        """Drive toward rest (always allowed, armed or not) and stop at the end of retract_ms."""
        self._drive(0, 1)
        self.last_rest_ns = time.monotonic_ns()
        self._phase = "retract"
        self._busy = True

    def rest(self) -> None:
        with self._lock:
            if self._phase == "retract":
                return  # already on its way back
            self._cancel_timers()
            self._enter_retract()
        self._schedule(self.retract_s, self._idle)

    def fire(self, t_ns: int) -> bool:
        if not self._armed():
            return False
        with self._lock:
            if self._closed or self._busy:
                return False
            if not self._armed():
                return False
            self.last_fire_ns = int(t_ns)
            self.last_rest_ns = None
            self._busy = True
            self._phase = "extend"
            self._drive(1, 0)
            self._forget_finished()
        self._schedule(self.extend_s, self._start_retract)
        return True

    def _start_retract(self) -> None:
        with self._lock:
            if self._phase != "extend":
                return
            self._enter_retract()
        self._schedule(self.retract_s, self._idle)

    def _idle(self) -> None:
        with self._lock:
            if self._phase != "retract":
                return
            self._drive(0, 0)
            self._phase = "idle"
            self._busy = False

    def abort(self) -> None:
        with self._lock:
            if self._phase == "retract" and not self._closed:
                return  # already driving back to rest: do not restart the retract timer
            self._cancel_timers()
            if self._closed or self._phase == "idle":
                self._drive(0, 0)  # parked: never stall the motor against the rest end stop
                self._busy = False
                return
            self._enter_retract()
        self._schedule(self.retract_s, self._idle)

    def close(self) -> None:
        self.abort()
        self._join_timers(self.retract_s + 0.5)
        self._cancel_timers()
        with self._lock:
            self._closed = True
            self._phase = "idle"
            self._drive(0, 0)
            self._busy = False


class SimActuator(_BaseActuator):
    """Software actuator with a cfg.actuator.sim.delay_ms command->break delay.

    Records every fire and the resulting (ideal) break time; when a SimGpio and a break-sense pin are
    given it also drives that pin low at the break and high again after the return to rest, so the
    latency tooling can be exercised end to end without hardware.
    """

    kind = "sim"

    def __init__(
        self,
        cfg: Config,
        arming: ArmingSwitch | None,
        gpio: SimGpio | None = None,
        break_pin: int | None = None,
        hold_ms: float = SIM_HOLD_MS,
    ):
        super().__init__(arming)
        self.delay_s = max(0.0, float(cfg.actuator.sim.delay_ms) / 1000.0)
        self.hold_s = max(0.0, float(hold_ms) / 1000.0)
        self._gpio = gpio
        self._break_pin = None if break_pin is None else int(break_pin)
        self.fire_times: list[int] = []
        self.break_times: list[int] = []

    def _sense(self, level: int, t_ns: int) -> None:
        if self._gpio is not None and self._break_pin is not None:
            self._gpio.inject(self._break_pin, level, t_ns)

    def rest(self) -> None:
        self.abort()

    def fire(self, t_ns: int) -> bool:
        if not self._armed():
            return False
        with self._lock:
            if self._closed or self._busy:
                return False
            if not self._armed():
                return False
            t_ns = int(t_ns)
            self.last_fire_ns = t_ns
            self.last_rest_ns = None
            self._busy = True
            self.fire_times.append(t_ns)
            self.break_times.append(t_ns + int(self.delay_s * 1e9))
        self._forget_finished()
        self._schedule(self.delay_s, self._on_break)
        return True

    def _on_break(self) -> None:
        with self._lock:
            if not self._busy or self.last_fire_ns is None:
                return
            # Edge timestamps are the ideal ones (fire + delay), not the timer's wake-up time, so the
            # simulated latency is exactly delay_ms however jittery the host scheduler is.
            t_break = self.break_times[-1]
        self._sense(0, t_break)
        self._schedule(self.hold_s, self._on_rest)

    def _on_rest(self) -> None:
        with self._lock:
            if not self._busy or not self.break_times:
                return
            self.last_rest_ns = self.break_times[-1] + int(self.hold_s * 1e9)
            t_open = self.last_rest_ns + int(self.delay_s * 1e9)
        self._schedule(self.delay_s, lambda: self._on_open(t_open))

    def _on_open(self, t_open: int) -> None:
        with self._lock:
            if not self._busy:
                return
            self._busy = False
        self._sense(1, t_open)

    def abort(self) -> None:
        self._cancel_timers()
        with self._lock:
            was_busy = self._busy
            self._busy = False
            if was_busy:
                self.last_rest_ns = time.monotonic_ns()
        if was_busy:
            self._sense(1, time.monotonic_ns())

    def close(self) -> None:
        self.abort()
        with self._lock:
            self._closed = True


class NoneActuator(_BaseActuator):
    """No trigger hardware fitted: decisions are logged, nothing ever moves."""

    kind = "none"

    def __init__(self, arming: ArmingSwitch | None = None):
        super().__init__(arming)

    def rest(self) -> None:
        return None

    def fire(self, t_ns: int) -> bool:
        return False

    def abort(self) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            self._closed = True


def make_actuator(cfg: Config, gpio: Gpio, arming: ArmingSwitch) -> Actuator:
    """Build the actuator named by cfg.actuator.type (servo | pq12 | sim | none)."""
    kind = str(cfg.actuator.type).lower()
    if kind == "servo":
        s = cfg.actuator.servo
        pwm: HardwarePwm | SimPwm
        if isinstance(gpio, LgpioGpio):
            pwm = HardwarePwm(s.pwm_chip, s.pwm_channel, s.frame_hz)
        else:
            pwm = SimPwm(s.pwm_chip, s.pwm_channel, s.frame_hz)
        return ServoActuator(cfg, pwm, arming)
    if kind == "pq12":
        return Pq12Actuator(cfg, gpio, arming)
    if kind == "sim":
        sim_gpio = gpio if isinstance(gpio, SimGpio) else None
        return SimActuator(cfg, arming, gpio=sim_gpio, break_pin=cfg.gpio.break_sense_pin)
    if kind == "none":
        return NoneActuator(arming)
    raise ValueError(f"actuator.type must be servo|pq12|sim|none, got {cfg.actuator.type!r}")
