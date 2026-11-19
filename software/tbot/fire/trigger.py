"""The trigger delay for the state the lever is in.

With pre-travel (actuator.servo.preload_frac / break_us) the lever waits most of the way to the sear break while
the arming button is held, and a shot from there breaks in a quarter of the time of one from rest (this gun,
2026-11-19: 38.5 +/- 1.1 ms against 191.7 +/- 1.4). One number for "the" delay is then wrong whenever the lever
is not waiting there, by up to the whole difference (150 ms: 2-4 deg of clay travel, several pattern widths):

  moving    for the first ~150 ms of a press the lever is still on its way to the pre-travel position. A fire
            command sent then just keeps it going, so the sear breaks about when a from-rest shot commanded at the
            press would have: delay = from-rest delay - time since the press, never less than the parked delay.
  parked    the measured pre-travel delay (latency.json actuator.<type>.cmd_to_break_ms).
  let go    the pre-travel is given up after actuator.servo.preload_max_s (the servo goes limp, the lever sags
            back by an unknown amount): the delay is somewhere between the two, and is treated as exactly that -
            the middle, with the spread of not knowing. P(hit) then rarely clears the bar; press again.
  spent     after a shot of this press the lever has gone back to rest and stays there: the from-rest delay.
  no pre-travel configured (or no button held: the next command starts from rest): the from-rest delay.

The from-rest delay is latency.json actuator.<type>.from_rest.cmd_to_break_ms (or
latency.servo_cmd_to_break_rest_ms). When it is not known (0), or is not longer than the parked one, there is
only one delay and this class returns it unchanged - as before pre-travel existed.

The model mirrors ServoActuator (hw/actuator.py) from the outside: the press, the shots and the clock are all it
needs, so a dry run (which has no actuator object) plans exactly as a live one does.
"""

from __future__ import annotations

import math

from ..config import Config, LatencyBudget

#: after the lever arrives at the pre-travel it is taken as still settling for this long (3 PWM frames)
SETTLE_MS = 60.0


class TriggerDelay:
    """cmd -> sear break (ms) and its 1-sigma, for a command sent at a given time."""

    def __init__(self, cfg: Config, budget: LatencyBudget):
        atype = str(cfg.actuator.type)
        self.parked_ms = float(budget.cmd_to_break_ms[atype])
        self.parked_jitter_ms = float(budget.jitter_ms[atype])
        rest = float((getattr(budget, "rest_cmd_to_break_ms", None) or {}).get(atype, 0.0) or 0.0)
        self.rest_ms = rest if rest > self.parked_ms else self.parked_ms
        self.rest_jitter_ms = float((getattr(budget, "rest_jitter_ms", None) or {}).get(atype, 0.0)
                                    or self.parked_jitter_ms)
        self.two_states = self.rest_ms > self.parked_ms
        self.transit_frac = max(0.0, float(getattr(cfg.latency, "servo_transit_jitter_frac", 0.0) or 0.0))
        self.preload = False
        self.preload_max_ns = 0
        if atype == "servo":
            from ..hw.actuator import preload_pulse_us  # the same test the actuator applies
            s = cfg.actuator.servo
            self.preload = preload_pulse_us(s.rest_us, s.fire_us, getattr(s, "break_us", 0.0),
                                            getattr(s, "preload_frac", 0.0), getattr(s, "min_us", 0.0),
                                            getattr(s, "max_us", 0.0)) is not None
            self.preload_max_ns = int(max(0.0, float(getattr(s, "preload_max_s", 0.0) or 0.0)) * 1e9)
        self._press_ns: int | None = None
        self._spent = False
        self.state = "parked" if not self.two_states else "rest"

    # -- what happened ------------------------------------------------------------------------
    def on_arming(self, armed: bool, t_ns: int) -> None:
        """The button went down (the lever starts for the pre-travel) or up (back to rest)."""
        self._press_ns = int(t_ns) if armed else None
        self._spent = False

    def on_fire(self, t_ns: int) -> None:
        """A fire command went out: the lever returns to rest afterwards and waits there."""
        self._spent = True

    # -- the delay ----------------------------------------------------------------------------
    def at(self, t_ns: int, armed: bool = True) -> tuple[float, float]:
        """(cmd -> break ms, 1-sigma ms) for a command sent at t_ns."""
        if not self.two_states:
            return self.parked_ms, self.parked_jitter_ms
        if not self.preload or self._spent:
            self.state = "spent" if self._spent else "rest"
            return self.rest_ms, self.rest_jitter_ms
        if not armed:
            self.state = "rest"  # nobody holds the button: a press now would start the lever from rest
            return self.rest_ms, self.rest_jitter_ms
        press_ns = self._press_ns  # (read once: the button callback runs on another thread)
        if press_ns is None:
            press_ns = self._press_ns = int(t_ns)  # held before anything was seen of the press (start-up, replay)
        since_ns = int(t_ns) - press_ns
        if 0 < self.preload_max_ns <= since_ns:
            self.state = "let go"
            span = self.rest_ms - self.parked_ms
            return self.parked_ms + 0.5 * span, math.hypot(self.rest_jitter_ms, span / math.sqrt(12.0))
        since_ms = max(0.0, since_ns * 1e-6)
        knee_ms = self.rest_ms - self.parked_ms  # the lever reaches the pre-travel about this long after the press
        if since_ms < knee_ms:
            self.state = "moving"
            return self.rest_ms - since_ms, math.hypot(self.rest_jitter_ms, self.transit_frac * since_ms)
        settle = max(0.0, 1.0 - (since_ms - knee_ms) / SETTLE_MS)
        self.state = "settling" if settle > 0.0 else "parked"
        return self.parked_ms, math.hypot(self.parked_jitter_ms, self.transit_frac * knee_ms * settle)
