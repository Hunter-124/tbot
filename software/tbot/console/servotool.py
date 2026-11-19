"""Trigger-servo helper process for the console (pipeline stopped): `tbot jog servo` driven by JSON lines.

    python -m tbot.console.servotool [--sim]

Every safety rule of tbot.tools.jog holds unchanged (it is the same ServoJog object): anything but the rest
pulse moves only while the arming button is HELD, releasing it goes limp at once, moves are short bursts, and
travel limits creep until marked. Commands on stdin: {"line": "<jog command>"}; after every command and every
change of the button, the state goes to stdout as {"type": "state", ...}. Saving is done by the console itself
(comment-preserving local.toml edit), from the state reported here.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from tbot.config import add_common_args, load_from_args
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import SimGpio, open_gpio
from tbot.tools.jog import ServoJog

_out_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _out_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def state(j: ServoJog) -> dict:
    return {"type": "state", "current_us": j.current_us, "rest_us": j.rest_us, "fire_us": j.fire_us,
            "min_us": j.min_us, "max_us": j.max_us, "limits_set": j.limits_set, "step_us": j.step_us,
            "armed": bool(j.armed), "driving": bool(getattr(j.pwm, "pulse_us", None)),
            "burst_ms": j.burst_s * 1e3, "max_jump_us": j.max_jump_us,
            "hold_ms": float(j.cfg.actuator.servo.hold_ms), "sim": isinstance(j.gpio, SimGpio),
            "break_us": j.break_us, "preload_frac": j.preload_frac, "preload_us": j.preload_us,
            "preload_max_s": float(getattr(j.cfg.actuator.servo, "preload_max_s", 30.0))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot.console.servotool")
    add_common_args(ap)
    ap.add_argument("--sim", action="store_true", help="SimGpio/SimPwm (laptop)")
    args = ap.parse_args(argv)
    cfg, _ = load_from_args(args)
    if args.sim:
        cfg.gpio.backend = "sim"
    if str(cfg.actuator.type) != "servo":
        emit({"type": "fatal", "message": f"actuator.type is {cfg.actuator.type!r}; this page sets up the servo"})
        return 2
    gpio = open_gpio(cfg)
    arming = ArmingSwitch(gpio, cfg)
    try:
        jog = ServoJog(cfg, gpio, arming, log=lambda s: emit({"type": "log", "text": str(s)}))
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fatal", "message": f"cannot open the servo PWM: {exc}"})
        arming.close()
        gpio.close()
        return 1
    arming.on_change(lambda armed, t_ns: emit(state(jog)))
    emit({"type": "ready"})
    emit(state(jog))
    lock = threading.Lock()

    def heartbeat() -> None:  # the button state, also when the sim backend has no edge callbacks
        last = None
        while True:
            time.sleep(0.2)
            a = bool(jog.armed)
            if a != last:
                emit(state(jog))
                last = a

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if c.get("cmd") == "quit":
                break
            if c.get("cmd") == "sim_button" and isinstance(gpio, SimGpio):
                gpio.inject(cfg.arming.pin, 0 if c.get("held") else 1)
                time.sleep(0.05)
                emit(state(jog))
                continue
            text = str(c.get("line", "")).strip()
            if text.split()[:1] == ["save"]:
                continue  # the console saves (keeps local.toml's comments)
            with lock:
                if text and not jog.execute(text):
                    break
            emit(state(jog))
    finally:
        try:
            jog.close()
        finally:
            arming.close()
            gpio.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
