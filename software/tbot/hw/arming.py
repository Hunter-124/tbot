"""Arming interlock: the momentary pushbutton that must be HELD for the trigger actuator to move.

Safety invariant (docs/ARCHITECTURE.md): pressed = armed, release or open circuit = NOT armed, and the
disarm path is immediate and un-debounced while the arm path is debounced. Every state read re-samples the
pin, so a missed edge or a dead callback thread can never leave the switch reporting "armed".
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from tbot.config import Config
from tbot.hw.gpio import Gpio


class ArmingSwitch:
    """Debounced view of the arming button on cfg.arming.pin (NO switch to GND, pull-up, active low)."""

    def __init__(self, gpio: Gpio, cfg: Config):
        self._gpio = gpio
        self._cfg = cfg
        self.pin = int(cfg.arming.pin)
        self.active_low = bool(cfg.arming.active_low)
        self.debounce_s = max(0.0, float(cfg.arming.debounce_ms) / 1000.0)
        self._max_shots = int(cfg.fire.max_shots_per_arm)
        self._lock = threading.RLock()
        self._armed = False
        self.press_count = 0
        self.shots_this_press = 0
        self._timer: threading.Timer | None = None
        self._callbacks: list[Callable[[bool, int], None]] = []
        self._closed = False

        gpio.setup_input(self.pin, str(cfg.arming.pull))
        gpio.watch(self.pin, "both", self._on_edge)
        if self._pressed_now():  # button already held at startup: still require a stable debounce window
            self._schedule_arm()

    # -- pin ----------------------------------------------------------------------------------
    def _level_pressed(self, level: int) -> bool:
        return (int(level) == 0) if self.active_low else (int(level) == 1)

    def _pressed_now(self) -> bool:
        return self._level_pressed(self._gpio.read(self.pin))

    # -- state --------------------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        """True only while the button is really held (the pin is re-read on every query)."""
        armed, t_ns = self._sync()
        self._notify(armed, t_ns)
        return armed

    def _sync(self) -> tuple[bool, int | None]:
        """Re-read the pin; drop out of the armed state at once if it is no longer pressed."""
        with self._lock:
            if self._armed and not self._pressed_now():
                self._armed = False
                self._cancel_timer()
                return False, time.monotonic_ns()
            return self._armed, None

    def _notify(self, armed: bool, t_ns: int | None) -> None:
        if t_ns is None:
            return
        for cb in list(self._callbacks):  # outside the lock: callbacks take the actuator's lock
            cb(armed, t_ns)

    def on_change(self, cb: Callable[[bool, int], None]) -> None:
        """Register cb(armed, t_ns) for every armed/disarmed transition."""
        with self._lock:
            self._callbacks.append(cb)

    # -- edges --------------------------------------------------------------------------------
    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_arm(self) -> None:
        self._cancel_timer()
        self._timer = threading.Timer(self.debounce_s, self._arm_if_stable)
        self._timer.daemon = True
        self._timer.start()

    def _on_edge(self, pin: int, level: int, t_ns: int) -> None:
        pressed = self._level_pressed(level)
        notify: tuple[bool, int] | None = None
        with self._lock:
            if self._closed:
                return
            if pressed:
                self._schedule_arm()  # arming waits out the debounce window
            else:
                self._cancel_timer()  # releasing never waits
                if self._armed:
                    self._armed = False
                    notify = (False, int(t_ns))
        if notify is not None:
            self._notify(*notify)

    def _arm_if_stable(self) -> None:
        notify: tuple[bool, int] | None = None
        with self._lock:
            self._timer = None
            if self._closed or self._armed or not self._pressed_now():
                return
            self._armed = True
            self.press_count += 1
            self.shots_this_press = 0
            notify = (True, time.monotonic_ns())
        if notify is not None:
            self._notify(*notify)

    # -- shots --------------------------------------------------------------------------------
    def try_consume_shot(self) -> bool:
        """Claim one shot of this press; False when disarmed or cfg.fire.max_shots_per_arm is used up."""
        armed, t_ns = self._sync()
        granted = False
        if armed:
            with self._lock:
                if self._armed and self.shots_this_press < self._max_shots:
                    self.shots_this_press += 1
                    granted = True
        self._notify(armed, t_ns)
        return granted

    def close(self) -> None:
        """Stop the debounce timer and forget the callbacks (the pin itself belongs to the Gpio)."""
        with self._lock:
            self._closed = True
            self._cancel_timer()
            self._armed = False
            self._callbacks.clear()
