"""`tbot latency-actuator` — fire command -> trigger break latency, measured on the real hardware.

Bench fixture: a microswitch (or a contact pair) on cfg.gpio.break_sense_pin closes to GND exactly when
the trigger breaks; the pin is pulled up, so break = falling edge and sear reset = rising edge. The gun is
UNLOADED. The tool never bypasses the interlock: it only fires while the operator holds the arming button
and pauses the moment it is released.

Results are merged into calib/latency.json under actuator.<type> and feed LatencyBudget.fire_delay_s().
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from dataclasses import dataclass

from tbot.config import Calibration, Config, add_common_args, load_from_args, now_iso, update_json
from tbot.hw.actuator import Actuator, make_actuator
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import Gpio, SimGpio, open_gpio
from tbot.stats import summarize

#: Poll period while waiting for the arming button / the actuator state machine.
POLL_S = 0.002
#: How long the tool waits for the operator to hold the arming button before giving up.
ARM_WAIT_TIMEOUT_S = 300.0
#: Random pause between trials (real runs) so the operator cannot anticipate the shot.
WAIT_RANGE_S = (0.5, 1.5)
#: Shorter pause in --sim mode (nothing to anticipate, and the tests drive this path).
SIM_WAIT_RANGE_S = (0.01, 0.03)

REST_EVENT = {
    "servo": "end of the hold (PWM back to rest_us)",
    "pq12": "start of the retract",
    "sim": "simulated return to rest",
    "none": "n/a",
}


class BreakSense:
    """Edge timestamps of the break-sense contact (closes to GND at the break, re-opens at reset)."""

    def __init__(self, gpio: Gpio, pin: int, pull: str = "up", active_low: bool = True):
        self.pin = int(pin)
        self.active_low = bool(active_low)
        self._lock = threading.Lock()
        self._closed_ns: int | None = None
        self._open_ns: int | None = None
        self._closed_evt = threading.Event()
        self._open_evt = threading.Event()
        gpio.setup_input(self.pin, pull)
        gpio.watch(self.pin, "both", self._on_edge)

    def _on_edge(self, pin: int, level: int, t_ns: int) -> None:
        closed = (int(level) == 0) if self.active_low else (int(level) == 1)
        with self._lock:
            if closed:
                if self._closed_ns is None:
                    self._closed_ns = int(t_ns)
                    self._closed_evt.set()
            elif self._closed_ns is not None and self._open_ns is None:
                self._open_ns = int(t_ns)
                self._open_evt.set()

    def arm_trial(self) -> None:
        with self._lock:
            self._closed_ns = None
            self._open_ns = None
            self._closed_evt.clear()
            self._open_evt.clear()

    def wait_closed(self, timeout_s: float) -> int | None:
        if not self._closed_evt.wait(max(0.0, timeout_s)):
            return None
        with self._lock:
            return self._closed_ns

    def wait_open(self, timeout_s: float) -> int | None:
        if not self._open_evt.wait(max(0.0, timeout_s)):
            return None
        with self._lock:
            return self._open_ns

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed_ns is not None and self._open_ns is None


@dataclass
class Trial:
    """One fire/rest cycle. Times are ns on time.monotonic_ns(); None = the edge never arrived."""

    index: int
    cmd_ns: int
    break_ns: int | None = None
    rest_cmd_ns: int | None = None
    reopen_ns: int | None = None

    @property
    def cmd_to_break_ms(self) -> float | None:
        return None if self.break_ns is None else (self.break_ns - self.cmd_ns) / 1e6

    @property
    def cmd_to_rest_ms(self) -> float | None:
        if self.rest_cmd_ns is None or self.reopen_ns is None:
            return None
        return (self.reopen_ns - self.rest_cmd_ns) / 1e6

    @property
    def complete(self) -> bool:
        return self.cmd_to_break_ms is not None and self.cmd_to_rest_ms is not None


def _wait_armed(arming: ArmingSwitch, timeout_s: float, log) -> bool:
    """Block until the operator holds the arming button (never simulated, never bypassed)."""
    if arming.armed:
        return True
    log("  waiting: HOLD the arming button to continue ...")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if arming.armed:
            return True
        time.sleep(POLL_S * 5)
    return False


def _poll_until(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)


def run_trials(
    actuator: Actuator,
    sense: BreakSense,
    arming: ArmingSwitch,
    *,
    trials: int,
    timeout_s: float,
    wait_range_s: tuple[float, float] = WAIT_RANGE_S,
    rng: random.Random | None = None,
    sleep=time.sleep,
    now_ns=time.monotonic_ns,
    arm_timeout_s: float = ARM_WAIT_TIMEOUT_S,
    log=print,
) -> list[Trial]:
    """Run `trials` fire/rest cycles, pausing whenever the arming button is released.

    `sleep`/`now_ns` are injectable so tests can run the loop without the random inter-trial waits.
    """
    rng = rng or random.Random()
    results: list[Trial] = []
    attempts = 0
    max_attempts = 3 * trials + 10  # refused fires (released button) must not loop forever
    while len(results) < trials and attempts < max_attempts:
        attempts += 1
        if not _wait_armed(arming, arm_timeout_s, log):
            log("  arming button never held: aborting")
            break
        sleep(rng.uniform(*wait_range_s))
        if not arming.armed:
            continue
        sense.arm_trial()
        t_cmd = int(now_ns())
        if not actuator.fire(t_cmd):
            log("  fire refused (not armed / busy) — keep holding the arming button")
            _poll_until(lambda: not actuator.busy, timeout_s)
            continue
        trial = Trial(index=len(results) + 1, cmd_ns=t_cmd)
        trial.break_ns = sense.wait_closed(timeout_s)
        if _poll_until(lambda: getattr(actuator, "last_rest_ns", None) is not None, timeout_s):
            trial.rest_cmd_ns = int(actuator.last_rest_ns)
            trial.reopen_ns = sense.wait_open(timeout_s)
        _poll_until(lambda: not actuator.busy, timeout_s)
        results.append(trial)
        log(f"  trial {trial.index:2d}/{trials}: "
            f"cmd->break {_fmt_ms(trial.cmd_to_break_ms)}  rest->reset {_fmt_ms(trial.cmd_to_rest_ms)}")
    return results


def _fmt_ms(value: float | None) -> str:
    return "  timeout" if value is None else f"{value:8.2f} ms"


def _default_timeout_s(cfg: Config, kind: str) -> float:
    """Generous per-type edge timeout: the slowest expected motion plus margin."""
    if kind == "servo":
        return max(1.0, float(cfg.actuator.servo.hold_ms) / 1000.0 + 1.0)
    if kind == "pq12":
        return (float(cfg.actuator.pq12.extend_ms) + float(cfg.actuator.pq12.retract_ms)) / 1000.0 + 2.0
    return max(1.0, 4.0 * float(cfg.actuator.sim.delay_ms) / 1000.0 + 1.0)


def _procedure(cfg: Config, kind: str) -> str:
    lines = [
        "tbot latency-actuator — bench procedure",
        "  1. The gun is UNLOADED. Check the chamber and the magazine now; keep it pointed in a safe direction.",
        "  2. Fit the break-sense fixture: a microswitch / contact pair that closes to GND exactly at the",
        f"     trigger break, wired between BCM GPIO{cfg.gpio.break_sense_pin} and GND (internal pull-up, break = falling edge).",
        f"  3. Arming button on BCM GPIO{cfg.arming.pin} to GND; the tool only fires while you HOLD it.",
    ]
    if kind == "servo":
        s = cfg.actuator.servo
        lines.append(f"  4. Servo on PWM chip {s.pwm_chip} channel {s.pwm_channel} at {s.frame_hz:g} Hz "
                     f"(rest {s.rest_us:g} us, fire {s.fire_us:g} us, hold {s.hold_ms:g} ms).")
    elif kind == "pq12":
        p = cfg.actuator.pq12
        lines.append(f"  4. PQ12 H-bridge: extend BCM GPIO{p.pin_extend}, retract BCM GPIO{p.pin_retract} "
                     f"(extend {p.extend_ms:g} ms, retract {p.retract_ms:g} ms).")
    lines += [
        "  5. Hold the arming button for the whole run; release it at any time to pause (and to stop).",
        f"  6. Measured: command -> break (falling edge) and {REST_EVENT.get(kind, 'rest command')} -> contact re-open.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tbot latency-actuator",
        description="Measure fire-command -> trigger-break latency with the arming button held.",
    )
    add_common_args(ap)
    ap.add_argument("--trials", type=int, default=20, help="number of fire/rest cycles (default 20)")
    ap.add_argument("--timeout-s", type=float, default=None,
                    help="per-edge timeout (default: from the actuator's own motion times)")
    ap.add_argument("--type", default=None, choices=["servo", "pq12", "sim", "none"],
                    help="override actuator.type")
    ap.add_argument("--sim", action="store_true",
                    help="no hardware: SimGpio + SimActuator inject the break edges (self-test)")
    ap.add_argument("--notes", default="", help="free text stored in calib/latency.json")
    ap.add_argument("--no-write", action="store_true", help="print the stats but do not touch calib/latency.json")
    args = ap.parse_args(argv)

    cfg, calib = load_from_args(args)
    if args.type:
        cfg.actuator.type = args.type
    if args.sim:
        cfg.gpio.backend = "sim"
        if not args.type:
            cfg.actuator.type = "sim"
    kind = str(cfg.actuator.type).lower()
    if kind == "none":
        print("actuator.type = none: nothing to measure (use --type servo|pq12|sim).")
        return 2
    if args.trials < 1:
        print("--trials must be >= 1")
        return 2
    timeout_s = float(args.timeout_s) if args.timeout_s else _default_timeout_s(cfg, kind)

    gpio = open_gpio(cfg)
    arming = ArmingSwitch(gpio, cfg)
    sense = BreakSense(gpio, cfg.gpio.break_sense_pin, pull="up")
    actuator = make_actuator(cfg, gpio, arming)
    rc = 0
    try:
        if args.sim:
            if not isinstance(gpio, SimGpio):
                print("--sim needs the sim GPIO backend")
                return 2
            print(f"--sim: simulated actuator, {cfg.actuator.sim.delay_ms:g} ms command->break delay")
            gpio.inject(cfg.arming.pin, 0)  # simulated operator holding the arming button
            if not _poll_until(lambda: arming.armed, max(1.0, 4.0 * cfg.arming.debounce_ms / 1000.0)):
                print("simulated arming press did not arm the interlock")
                return 1
            wait_range = SIM_WAIT_RANGE_S
        else:
            print(_procedure(cfg, kind))
            wait_range = WAIT_RANGE_S
        actuator.rest()  # park the trigger lever before the first trial (PQ12: a full retract)
        _poll_until(lambda: not actuator.busy, timeout_s)
        print(f"actuator={kind} trials={args.trials} timeout={timeout_s:g} s")
        trials = run_trials(
            actuator, sense, arming,
            trials=args.trials, timeout_s=timeout_s, wait_range_s=wait_range,
            rng=random.Random(), log=print,
        )
    finally:
        try:
            actuator.close()
        finally:
            arming.close()
            gpio.close()

    breaks = [t.cmd_to_break_ms for t in trials if t.cmd_to_break_ms is not None]
    rests = [t.cmd_to_rest_ms for t in trials if t.cmd_to_rest_ms is not None]
    timeouts = sum(1 for t in trials if not t.complete) + (args.trials - len(trials))
    break_stats = summarize(breaks)
    rest_stats = summarize(rests)
    print(f"\ncmd->break ms: {_fmt_stats(break_stats)}")
    print(f"rest->reset ms: {_fmt_stats(rest_stats)}")
    print(f"timeouts: {timeouts} of {args.trials}")
    if not breaks:
        print("no break edge was ever seen: check the fixture wiring and the actuator direction.")
        rc = 1

    if not args.no_write:
        notes = args.notes or (f"{kind} actuator, break-sense GPIO{cfg.gpio.break_sense_pin}"
                               + (", simulated (--sim)" if args.sim else ""))
        _write_latency(calib, kind, break_stats, rest_stats, timeouts, notes)
        print(f"wrote {calib.path('latency.json')}")
    return rc


def _fmt_stats(s: dict) -> str:
    if not s["n"]:
        return "no samples"
    return (f"mean {s['mean']:.2f}  std {s['std']:.2f}  p50 {s['p50']:.2f}  p95 {s['p95']:.2f}  "
            f"min {s['min']:.2f}  max {s['max']:.2f}  n {s['n']}")


def _write_latency(calib: Calibration, kind: str, break_stats: dict, rest_stats: dict,
                   timeouts: int, notes: str) -> dict:
    patch = {
        "actuator": {
            kind: {
                "cmd_to_break_ms": break_stats,
                "cmd_to_rest_ms": rest_stats,
                "timeouts": int(timeouts),
                "created": now_iso(),
                "notes": notes,
            }
        },
        "created": now_iso(),
    }
    return update_json(calib.path("latency.json"), patch)


if __name__ == "__main__":
    raise SystemExit(main())
