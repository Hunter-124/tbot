"""`tbot jog [servo|pq12]` — find the trigger actuator's rest/fire positions on the bench, safely.

This is the tool that turns the [INFERENCE] servo pulse widths in config/default.toml into measured
numbers: nudge the lever until it just touches the trigger (that is `rest`), nudge it until the sear
breaks (that is `fire`), `test` the cycle, `save`. The same REPL drives the PQ12 with timed extend /
retract pulses and can swap the H-bridge pins when the rod goes the wrong way.

Safety, enforced here and not merely documented:
* the gun is UNLOADED -- the banner says so before anything can move;
* ANY pulse other than the current rest_us is emitted only while the arming button is HELD;
* releasing the button returns the servo to rest_us / retracts the PQ12 at once, from the arming
  callback, whatever the REPL is doing;
* the PQ12 H-bridge inputs are written lows-first, so the two are never high at the same instant;
* on exit (q, EOF, Ctrl-C, exception) the actuator is parked: the servo PWM is left *at rest_us*
  (still driven, so the lever holds its position), both PQ12 pins low.

Line based, no curses: it works over ssh and over the serial console. `--commands "1500;+;fire;save"`
runs a list non-interactively (scripts and tests), `--sim` runs the whole thing on SimGpio/SimPwm.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Callable, Iterable

from tbot.config import Config, add_common_args, load_from_args, save_local_overrides
from tbot.hw.actuator import Pq12Actuator, ServoActuator
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import Gpio, HardwarePwm, LgpioGpio, SimGpio, SimPwm, open_gpio

__all__ = ["MIN_PULSE_US", "MAX_PULSE_US", "DEFAULT_STEP_US", "JogRepl", "ServoJog", "Pq12Jog",
           "make_jog", "unloaded_banner", "main"]

#: Hobby-servo pulse limits: outside this an MG996R buzzes against its own end stop.
MIN_PULSE_US, MAX_PULSE_US = 500.0, 2500.0
#: `+` / `-` step.
DEFAULT_STEP_US = 10.0
#: REPL poll period while a timed pulse runs (also how fast a release is noticed).
POLL_S = 0.002
#: Printed whenever the interlock drops while something was moving.
SAFE_MESSAGE = "SAFE: arming released"


def unloaded_banner(cfg: Config, kind: str) -> str:
    """The bench procedure, printed before anything can move."""
    lines = [
        f"tbot jog {kind} — trigger actuator bench setup",
        "  1. The gun is UNLOADED. Check the chamber AND the magazine now, then keep it pointed",
        "     in a safe direction. Nothing in this tool needs a round in the gun, ever.",
        f"  2. The actuator moves only while you HOLD the arming button (BCM{cfg.arming.pin} to GND);",
        "     releasing it returns the actuator to rest immediately.",
        "  3. Keep your fingers out of the trigger guard while the lever is moving.",
    ]
    if kind == "servo":
        s = cfg.actuator.servo
        lines.append(f"  4. MG996R on PWM chip {s.pwm_chip} channel {s.pwm_channel} @ {s.frame_hz:g} Hz, "
                     f"powered from its own 5-6 V BEC (never off the Pi's 5 V pin; docs/PROTOTYPING.md §4).")
    else:
        p = cfg.actuator.pq12
        lines.append(f"  4. PQ12 H-bridge: extend BCM{p.pin_extend}, retract BCM{p.pin_retract}; "
                     "check the rod direction with a short pulse before fitting the lever.")
    return "\n".join(lines)


def _clamp_pulse(us: float) -> float:
    return min(MAX_PULSE_US, max(MIN_PULSE_US, float(us)))


# --------------------------------------------------------------------------------------------- base
class JogRepl:
    """Command loop driven by a list of lines, so a script or a test can run it without a tty."""

    kind = "base"

    def __init__(self, cfg: Config, gpio: Gpio, arming: ArmingSwitch, *, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.gpio = gpio
        self.arming = arming
        self.log = log
        self.prompt = f"{self.kind}> "
        self._closed = False

    # -- interlock ----------------------------------------------------------------------------
    def _subscribe(self) -> None:
        """Called last by every subclass __init__: the callback may fire immediately after this."""
        self.arming.on_change(self._on_arming)

    def _on_arming(self, armed: bool, t_ns: int) -> None:
        if not armed:
            self._disarm()

    def _disarm(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def armed(self) -> bool:
        return bool(self.arming.armed)

    def _refuse(self, what: str) -> bool:
        self.log(f"refused: {what} needs the arming button HELD (BCM{self.cfg.arming.pin} to GND)")
        return True

    # -- loop ---------------------------------------------------------------------------------
    def execute(self, line: str) -> bool:
        """Run one command line. Returns False when the session should end."""
        text = str(line).strip()
        if not text or text.startswith("#"):
            return True
        parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("h", "help", "?"):
            self.log(self.help_text())
            return True
        if cmd == "show":
            self.log(self.status_text())
            return True
        if cmd == "save":
            self._cmd_save()
            return True
        return self.dispatch(cmd, args)

    def dispatch(self, cmd: str, args: list[str]) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def run_commands(self, commands: Iterable[str]) -> None:
        for line in commands:
            if not self.execute(line):
                break

    def run(self) -> None:
        """Interactive loop on stdin (line based: works over ssh and the serial console)."""
        self.log(self.help_text())
        self.log(self.status_text())
        while True:
            try:
                line = input(self.prompt)
            except EOFError:
                self.log("")
                return
            if not self.execute(line):
                return

    # -- subclass API -------------------------------------------------------------------------
    def help_text(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def status_text(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _cmd_save(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _save(self, patch: dict, what: str) -> None:
        try:
            path = save_local_overrides(patch)
        except Exception as exc:  # noqa: BLE001 - a bad path must not end the bench session
            self.log(f"save failed: {exc}")
            return
        self.log(f"saved {what} to {path}")

    def close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- servo
class ServoJog(JogRepl):
    """MG996R pulse-width finder: `<us>`, `+`/`-`, `step`, `rest`, `fire`, `test`, `save`, `show`, `q`."""

    kind = "servo"

    def __init__(self, cfg: Config, gpio: Gpio, arming: ArmingSwitch,
                 pwm: HardwarePwm | SimPwm | None = None, *,
                 step_us: float = DEFAULT_STEP_US, log: Callable[[str], None] = print):
        super().__init__(cfg, gpio, arming, log=log)
        s = cfg.actuator.servo
        if pwm is None:
            factory = HardwarePwm if isinstance(gpio, LgpioGpio) else SimPwm
            pwm = factory(int(s.pwm_chip), int(s.pwm_channel), float(s.frame_hz))
        self.pwm = pwm
        self.rest_us = _clamp_pulse(s.rest_us)
        self.fire_us = _clamp_pulse(s.fire_us)
        self.step_us = abs(float(step_us)) or DEFAULT_STEP_US
        self.current_us = self.rest_us
        self._actuator: ServoActuator | None = None
        self.pwm.set_pulse_us(self.rest_us)  # rest is always allowed: it is where the lever belongs
        self._subscribe()

    # -- pulses -------------------------------------------------------------------------------
    def _set_pulse(self, us: float) -> bool:
        target = _clamp_pulse(us)
        if target != float(us):
            self.log(f"clamped {float(us):.0f} -> {target:.0f} us ({MIN_PULSE_US:.0f}-{MAX_PULSE_US:.0f})")
        if target != self.rest_us and not self.armed:
            self._refuse(f"{target:.0f} us")
            return False
        self.pwm.set_pulse_us(target)
        self.current_us = target
        tag = "  (rest)" if target == self.rest_us else ("  (fire)" if target == self.fire_us else "")
        self.log(f"pulse {target:.0f} us{tag}")
        return True

    def _sync_actuator(self) -> None:
        act = self._actuator
        if act is not None:
            act.rest_us = self.rest_us
            act.fire_us = self.fire_us

    def _ensure_actuator(self) -> ServoActuator:
        if self._actuator is None:
            # One actuator for the whole session: each one subscribes to the interlock for life.
            self._actuator = ServoActuator(self.cfg, self.pwm, self.arming)
        self._sync_actuator()
        return self._actuator

    def _disarm(self) -> None:
        act = self._actuator
        if act is not None:
            act.abort()
        self.pwm.set_pulse_us(self.rest_us)
        self.current_us = self.rest_us
        self.log(SAFE_MESSAGE)

    # -- commands -----------------------------------------------------------------------------
    def dispatch(self, cmd: str, args: list[str]) -> bool:
        if cmd[0] in "+-" and len(cmd) > 1:  # "+25" / "-25": relative, never an absolute pulse
            delta = _as_float(cmd, self.log)
            if delta is not None:
                self._set_pulse(self.current_us + delta)
            return True
        if cmd in ("+", "-"):
            self._set_pulse(self.current_us + (self.step_us if cmd == "+" else -self.step_us))
            return True
        if cmd == "step":
            value = _as_float(args[0], self.log) if args else None
            if value is None or value <= 0.0:
                self.log("usage: step <us>  (positive)")
            else:
                self.step_us = float(value)
                self.log(f"step {self.step_us:.0f} us")
            return True
        if cmd == "rest":
            self.rest_us = self.current_us
            self._sync_actuator()
            self.log(f"rest_us = {self.rest_us:.0f} us (the lever rests here; this is the only pulse "
                     "allowed with the button released)")
            return True
        if cmd == "fire":
            self.fire_us = self.current_us
            self._sync_actuator()
            self.log(f"fire_us = {self.fire_us:.0f} us")
            return True
        if cmd == "test":
            self._cmd_test()
            return True
        value = _as_float(cmd, None)
        if value is not None:
            self._set_pulse(value)
            return True
        self.log(f"unknown command {cmd!r}\n{self.help_text()}")
        return True

    def _cmd_test(self) -> None:
        if self.rest_us == self.fire_us:
            self.log("rest_us == fire_us: mark them first (move the lever, then `rest` / `fire`)")
            return
        if not self.armed:
            self._refuse("test")
            return
        act = self._ensure_actuator()
        act.rest()
        self.current_us = self.rest_us
        hold_s = max(0.0, float(self.cfg.actuator.servo.hold_ms) / 1000.0)
        self.log(f"test: {self.rest_us:.0f} -> {self.fire_us:.0f} us, hold {hold_s * 1e3:.0f} ms, "
                 "back to rest")
        if not act.fire(time.monotonic_ns()):
            self.log("fire refused (not armed / busy)")
            return
        deadline = time.monotonic() + hold_s + 1.0
        while act.busy and time.monotonic() < deadline:
            time.sleep(POLL_S)
        act.abort()  # guarantees rest even if the hold timer was late
        self.current_us = self.rest_us
        self.log(f"test done, back at rest {self.rest_us:.0f} us"
                 + ("" if self.armed else f"  ({SAFE_MESSAGE})"))

    def _cmd_save(self) -> None:
        self._save({"actuator": {"servo": {"rest_us": float(self.rest_us), "fire_us": float(self.fire_us)}}},
                   f"rest_us={self.rest_us:.0f} fire_us={self.fire_us:.0f}")

    # -- text ---------------------------------------------------------------------------------
    def help_text(self) -> str:
        return (
            "commands:  <us> go to that pulse | + / - step (also +25 / -25) | step <us> set the step\n"
            "           rest  mark the current pulse as rest_us (lever just off the trigger)\n"
            "           fire  mark the current pulse as fire_us (sear breaks)\n"
            "           test  one full fire/rest cycle through the real actuator\n"
            "           save  write rest_us/fire_us to config/local.toml | show | q"
        )

    def status_text(self) -> str:
        s = self.cfg.actuator.servo
        return (f"servo pwmchip{int(s.pwm_chip)}/pwm{int(s.pwm_channel)} @ {float(s.frame_hz):g} Hz  "
                f"pulse {self.current_us:.0f} us  rest {self.rest_us:.0f}  fire {self.fire_us:.0f}  "
                f"step {self.step_us:.0f}  armed {'YES' if self.armed else 'no'}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        act = self._actuator
        if act is not None:
            act.abort()
        try:
            self.pwm.set_pulse_us(self.rest_us)  # left driven AT rest: the lever holds its position
        except OSError as exc:
            self.log(f"could not park the servo at rest: {exc}")
        self.current_us = self.rest_us
        self.log(f"parked: PWM left at rest {self.rest_us:.0f} us")


# --------------------------------------------------------------------------------------------- pq12
class Pq12Jog(JogRepl):
    """PQ12 H-bridge jogger: `e <ms>`, `r <ms>`, `home`, `test`, `swap`, `save`, `show`, `q`."""

    kind = "pq12"

    def __init__(self, cfg: Config, gpio: Gpio, arming: ArmingSwitch, *,
                 log: Callable[[str], None] = print):
        super().__init__(cfg, gpio, arming, log=log)
        p = cfg.actuator.pq12
        self.pin_extend = int(p.pin_extend)
        self.pin_retract = int(p.pin_retract)
        self.extend_ms = float(p.extend_ms)
        self.retract_ms = float(p.retract_ms)
        self._lock = threading.RLock()
        self._phase = "idle"  # idle | extend | retract
        self._timer: threading.Timer | None = None
        self._actuator: Pq12Actuator | None = None
        self._claim()
        self._subscribe()

    # -- pins ---------------------------------------------------------------------------------
    def _claim(self) -> None:
        self.gpio.setup_output(self.pin_extend, 0)
        self.gpio.setup_output(self.pin_retract, 0)

    def _drive(self, extend: int, retract: int) -> None:
        """Both H-bridge inputs, lows first: the two are never high at the same instant."""
        extend, retract = int(bool(extend)), int(bool(retract))
        with self._lock:
            if not extend:
                self.gpio.write(self.pin_extend, 0)
            if not retract:
                self.gpio.write(self.pin_retract, 0)
            if extend:
                self.gpio.write(self.pin_extend, 1)
            if retract:
                self.gpio.write(self.pin_retract, 1)

    def _cancel_timer(self) -> None:
        with self._lock:
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _stop(self) -> None:
        with self._lock:
            self._drive(0, 0)
            self._phase = "idle"

    def _start_retract(self, ms: float) -> None:
        """Drive toward rest (always allowed) and stop after ms, without blocking the caller."""
        self._cancel_timer()
        with self._lock:
            self._drive(0, 1)
            self._phase = "retract"
            timer = threading.Timer(max(0.0, float(ms)) / 1000.0, self._stop)
            timer.daemon = True
            self._timer = timer
        timer.start()

    def _disarm(self) -> None:
        act = self._actuator
        if act is not None:
            act.abort()
        with self._lock:
            extending = self._phase == "extend"
        if extending:
            self.log(f"{SAFE_MESSAGE}: retracting {self.retract_ms:.0f} ms")
            self._start_retract(self.retract_ms)

    # -- motion -------------------------------------------------------------------------------
    def _extend(self, ms: float) -> None:
        if not self.armed:
            self._refuse(f"extend {ms:.0f} ms")
            return
        self._cancel_timer()
        with self._lock:
            self._drive(1, 0)
            self._phase = "extend"
        self.log(f"extend {ms:.0f} ms on BCM{self.pin_extend}")
        deadline = time.monotonic() + max(0.0, float(ms)) / 1000.0
        while time.monotonic() < deadline:
            if not self.armed:  # the arming callback has already started the retract
                return
            with self._lock:
                if self._phase != "extend":
                    return
            time.sleep(POLL_S)
        with self._lock:
            if self._phase == "extend":
                self._stop()
                self.log("extend done (both pins low)")

    def _retract(self, ms: float) -> None:
        self._start_retract(ms)
        self.log(f"retract {ms:.0f} ms on BCM{self.pin_retract}")
        deadline = time.monotonic() + max(0.0, float(ms)) / 1000.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._phase != "retract":
                    return
            time.sleep(POLL_S)
        self._cancel_timer()
        self._stop()

    def _ensure_actuator(self) -> Pq12Actuator:
        if self._actuator is None:
            self._actuator = Pq12Actuator(self.cfg, self.gpio, self.arming)
        act = self._actuator
        act.pin_extend = self.pin_extend
        act.pin_retract = self.pin_retract
        act.extend_s = max(0.0, self.extend_ms / 1000.0)
        act.retract_s = max(0.0, self.retract_ms / 1000.0)
        self._claim()
        return act

    # -- commands -----------------------------------------------------------------------------
    def dispatch(self, cmd: str, args: list[str]) -> bool:
        if cmd in ("e", "extend", "r", "retract"):
            default = self.extend_ms if cmd in ("e", "extend") else self.retract_ms
            ms = _as_float(args[0], self.log) if args else default
            if ms is None:
                self.log(f"usage: {cmd} <ms>")
                return True
            self._cancel_timer()
            if cmd in ("e", "extend"):
                self._extend(ms)
            else:
                self._retract(ms)
            return True
        if cmd == "home":
            self._cancel_timer()
            self._retract(self.retract_ms)
            self.log("homed (rod fully retracted)")
            return True
        if cmd == "test":
            self._cmd_test()
            return True
        if cmd == "swap":
            self._cancel_timer()
            self._stop()
            self.pin_extend, self.pin_retract = self.pin_retract, self.pin_extend
            self._claim()
            if self._actuator is not None:
                self._ensure_actuator()
            self.log(f"swapped: extend BCM{self.pin_extend}, retract BCM{self.pin_retract}")
            return True
        self.log(f"unknown command {cmd!r}\n{self.help_text()}")
        return True

    def _cmd_test(self) -> None:
        if not self.armed:
            self._refuse("test")
            return
        act = self._ensure_actuator()
        self.log(f"test: extend {self.extend_ms:.0f} ms, retract {self.retract_ms:.0f} ms")
        if not act.fire(time.monotonic_ns()):
            self.log("fire refused (not armed / busy)")
            return
        deadline = time.monotonic() + (self.extend_ms + self.retract_ms) / 1000.0 + 1.0
        while act.busy and time.monotonic() < deadline:
            time.sleep(POLL_S)
        act.abort()
        with self._lock:
            self._phase = "idle"
        self.log("test done" + ("" if self.armed else f"  ({SAFE_MESSAGE})"))

    def _cmd_save(self) -> None:
        self._save(
            {"actuator": {"pq12": {"pin_extend": int(self.pin_extend), "pin_retract": int(self.pin_retract),
                                   "extend_ms": float(self.extend_ms), "retract_ms": float(self.retract_ms)}}},
            f"pin_extend={self.pin_extend} pin_retract={self.pin_retract} "
            f"extend_ms={self.extend_ms:.0f} retract_ms={self.retract_ms:.0f}",
        )

    # -- text ---------------------------------------------------------------------------------
    def help_text(self) -> str:
        return (
            "commands:  e <ms> extend (arming button HELD) | r <ms> retract | home full retract\n"
            "           test  one full fire cycle through the real actuator\n"
            "           swap  swap the H-bridge pins when the rod moves the wrong way\n"
            "           save  write pins + timings to config/local.toml | show | q"
        )

    def status_text(self) -> str:
        return (f"pq12 extend BCM{self.pin_extend}  retract BCM{self.pin_retract}  "
                f"extend {self.extend_ms:.0f} ms  retract {self.retract_ms:.0f} ms  "
                f"phase {self._phase}  armed {'YES' if self.armed else 'no'}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_timer()
        act = self._actuator
        if act is not None:
            act.abort()
            act.close()
        self._stop()
        self.log("parked: both H-bridge inputs low")


# --------------------------------------------------------------------------------------------- glue
def _as_float(text: str, log: Callable[[str], None] | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        if log is not None:
            log(f"not a number: {text!r}")
        return None


def make_jog(cfg: Config, gpio: Gpio, arming: ArmingSwitch, kind: str = "", **kwargs: Any) -> JogRepl:
    """ServoJog or Pq12Jog (default: cfg.actuator.type)."""
    kind = (kind or str(cfg.actuator.type)).lower()
    if kind == "servo":
        return ServoJog(cfg, gpio, arming, **kwargs)
    if kind == "pq12":
        kwargs.pop("step_us", None)
        return Pq12Jog(cfg, gpio, arming, **kwargs)
    raise ValueError(f"tbot jog handles servo | pq12, not {kind!r}")


def _wait(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_S)
    return predicate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot jog",
        description="Bench setup of the trigger actuator (servo pulse widths / PQ12 pulses). "
                    "The gun is UNLOADED and the arming button must be HELD for anything to move.",
    )
    add_common_args(parser)
    parser.add_argument("kind", nargs="?", choices=["servo", "pq12"], default=None,
                        help="actuator to jog (default: actuator.type)")
    parser.add_argument("--commands", default=None,
                        help='run a ";" separated command list instead of reading stdin')
    parser.add_argument("--step-us", type=float, default=DEFAULT_STEP_US,
                        help=f"servo + / - step (default {DEFAULT_STEP_US:g} us)")
    parser.add_argument("--sim", action="store_true", help="no hardware: SimGpio + SimPwm")
    parser.add_argument("--sim-armed", action="store_true",
                        help="--sim with the arming button simulated as held (self-test only)")
    args = parser.parse_args(argv)

    cfg, _calib = load_from_args(args)
    kind = str(args.kind or cfg.actuator.type).lower()
    if kind not in ("servo", "pq12"):
        print(f"actuator.type = {cfg.actuator.type!r}: run `tbot jog servo` or `tbot jog pq12`",
              file=sys.stderr)
        return 2
    if args.sim or args.sim_armed:
        cfg.gpio.backend = "sim"
    if args.step_us <= 0.0:
        parser.error("--step-us must be > 0")

    print(unloaded_banner(cfg, kind))
    gpio = open_gpio(cfg)
    arming = ArmingSwitch(gpio, cfg)
    if args.sim_armed:
        if not isinstance(gpio, SimGpio):
            print("--sim-armed needs the sim GPIO backend", file=sys.stderr)
            arming.close()
            gpio.close()
            return 2
        gpio.inject(cfg.arming.pin, 0)  # simulated operator holding the button
        _wait(lambda: arming.armed, max(1.0, 4.0 * float(cfg.arming.debounce_ms) / 1000.0))
    try:
        repl = make_jog(cfg, gpio, arming, kind, step_us=args.step_us)
    except Exception as exc:  # noqa: BLE001 - no PWM channel / no permission: report, do not traceback
        print(f"cannot open the {kind} actuator: {exc}", file=sys.stderr)
        arming.close()
        gpio.close()
        return 1
    try:
        if args.commands is not None:
            repl.run_commands([c for c in str(args.commands).split(";")])
        else:
            repl.run()
    except KeyboardInterrupt:
        print()
    finally:
        try:
            repl.close()
        finally:
            arming.close()
            gpio.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
