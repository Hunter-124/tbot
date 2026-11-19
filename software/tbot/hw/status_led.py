"""The ready light: one LED that says whether a press of the arming button will be acted on.

    solid       ARMED: the button is held, the gun is live and ready - it is looking for its shot
    flashing    READY and LIVE, button up (once a second): press and go
    short blip  ready, but a DRY RUN (or the paper-fire test mode): a blip every 2 s, held or not - never solid
    flicker     running but not ready (5 a second): no fresh frame, or the IMU / attitude is not there yet
    off         not running (still booting, stopped, no power)

The pipeline reports its state once a frame (`set`); a thread of its own draws the pattern, and calls the state
"not ready" by itself when the reports stop (a frame loop that hangs must not leave the light on). The LED hangs
on a GPIO through a resistor (gpio.status_led_pin, default GPIO26 = header pin 37, ground on pin 39 beside it).
Nothing here can stop the gun: every GPIO error is swallowed after one log line.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("tbot.led")

OFF, WAIT, DRY, READY, ARMED = "off", "wait", "dry", "ready", "armed"
STALE_S = 0.6        # no report for this long = not ready (the idle loop reports 15 times a second)
TICK_S = 0.025
WAIT_PERIOD_S = 0.2  # flicker
READY_PERIOD_S = 1.0 # flashing: half a second on, half off
DRY_PERIOD_S = 2.0   # one short blip in each
DRY_BLIP_S = 0.12


def level_for(state: str, t_s: float) -> int:
    """The LED level (1 = lit) of a state at time t_s."""
    if state == ARMED:
        return 1
    if state == READY:
        return 1 if (t_s % READY_PERIOD_S) < 0.5 * READY_PERIOD_S else 0
    if state == DRY:
        return 1 if (t_s % DRY_PERIOD_S) < DRY_BLIP_S else 0
    if state == WAIT:
        return 1 if (t_s % WAIT_PERIOD_S) < 0.5 * WAIT_PERIOD_S else 0
    return 0


class StatusLed:
    def __init__(self, gpio, pin: int, active_high: bool = True, clock=time.monotonic):
        self._gpio = gpio
        self.pin = int(pin)
        self._on, self._off = (1, 0) if active_high else (0, 1)
        self._clock = clock
        self._state = OFF
        self._beat = clock()
        self._level: int | None = None
        self._dead = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            gpio.setup_output(self.pin, self._off)
        except Exception as exc:  # noqa: BLE001 - a light is never worth a traceback
            log.warning("status LED on GPIO%d not available: %s", self.pin, exc)
            self._dead = True

    def start(self) -> None:
        if self._dead or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="tbot-led", daemon=True)
        self._thread.start()

    def set(self, state: str) -> None:
        """The pipeline's state now (also the sign that it is alive)."""
        self._state = state
        self._beat = self._clock()

    def shown(self, now_s: float | None = None) -> str:
        """The state the light shows: the reported one, or WAIT once the reports have stopped."""
        now_s = self._clock() if now_s is None else now_s
        state = self._state
        if state != OFF and now_s - self._beat > STALE_S:
            return WAIT
        return state

    def _write(self, lit: int) -> None:
        if lit == self._level or self._dead:
            return
        try:
            self._gpio.write(self.pin, self._on if lit else self._off)
            self._level = lit
        except Exception as exc:  # noqa: BLE001
            log.warning("status LED write failed, light abandoned: %s", exc)
            self._dead = True

    def _run(self) -> None:
        while not self._stop.wait(TICK_S):
            now = self._clock()
            self._write(level_for(self.shown(now), now))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._state = OFF
        self._write(0)
