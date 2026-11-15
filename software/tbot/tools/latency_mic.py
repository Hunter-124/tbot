"""Actuator latency from a microphone: fire command -> dry-fire click, one trial per arming-button press.

No break-sense switch needed. Per trial the operator recocks (cycles the action), then presses and HOLDS the
arming button; the tool waits a random 1-3 s, fires once through the real actuator (interlock, one shot per
press, burst-then-limp all in force) and records the microphone around it. Release, recock, press again.

On the bench the lever hitting the trigger is itself loud, so the tool does not trust "first loud sample".
Every trial's audio is saved (trial_NN.wav) with its onset list: each transient above --onset-level after
the command, with its time. A cocked trial shows the lever impact, then the sear/hammer; a trial with the
hammer already down shows the impact only. cmd_to_break_ms uses the LAST strong onset inside the fire hold
(sear + hammer fall), which is what the budget needs; check the per-trial onsets before trusting it.

Times include mic latency (USB/ALSA, a few ms, uncalibrated) and sound flight (~0.3 ms at 10 cm); pass
--mic-latency-ms / --mic-distance-m to remove them once known.

--led lights the latency LED (gpio.latency_led_pin) immediately before each fire command and turns it off
1 s later, so a camera that sees both the LED and the trigger (a webcam beside the gun) times the stroke:
frames from the LED's first lit frame to the trigger fully back = command -> break, independent of the mic.

    tbot latency-mic --trials 10
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from tbot.config import add_common_args, load_from_args, now_iso, update_json
from tbot.hitl.audio import ArecordMic, write_wav
from tbot.hw.actuator import make_actuator
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import open_gpio
from tbot.stats import summarize

SOUND_MPS = 343.0


def find_onsets(x: np.ndarray, fs: float, *, level: float, gap_ms: float = 8.0) -> list[tuple[float, float]]:
    """Transients: (time s, peak) of each burst of |x| >= level, bursts closer than gap_ms merged."""
    idx = np.flatnonzero(np.abs(x) >= level)
    out: list[tuple[float, float]] = []
    if not len(idx):
        return out
    gap = int(gap_ms * fs / 1000)
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap:
            out.append((start / fs, float(np.abs(x[start:prev + 1]).max())))
            start = i
        prev = i
    out.append((start / fs, float(np.abs(x[start:prev + 1]).max())))
    return out


def wait_press(arming: ArmingSwitch, last_count: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if arming.press_count > last_count and arming.armed:
            return True
        time.sleep(0.01)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot latency-mic", description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--device", default="plughw:CARD=Snowball,DEV=0", help="ALSA capture device")
    ap.add_argument("--onset-level", type=float, default=0.6, help="|sample| that counts as a transient (0-1)")
    ap.add_argument("--press-timeout-s", type=float, default=600.0)
    ap.add_argument("--mic-latency-ms", type=float, default=0.0)
    ap.add_argument("--mic-distance-m", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="run directory (default runs/latency-mic-<time>)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--led", action="store_true",
                    help="light the latency LED at each fire command (video timing reference)")
    args = ap.parse_args(argv)
    cfg, calib = load_from_args(args)
    log = lambda s: print(s, flush=True)  # noqa: E731
    out = Path(args.out or f"runs/latency-mic-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    log("tbot latency-mic - gun UNLOADED. Per trial: recock, then press and HOLD the arming button;\n"
        "it fires once after 1-3 s; release; repeat.")
    gpio = open_gpio(cfg)
    arming = ArmingSwitch(gpio, cfg)
    actuator = make_actuator(cfg, gpio, arming)
    led_pin = int(cfg.gpio.latency_led_pin) if args.led else None
    if led_pin is not None:
        gpio.setup_output(led_pin, 0)
    err = getattr(actuator, "travel_error", None)
    if err:
        log(f"refusing: {err}")
        return 2
    time.sleep(max(0.2, 3.0 * float(cfg.arming.debounce_ms) / 1000.0))  # let the debounced reading settle
    if arming.armed:
        log("WARNING: the arming button reads HELD already (clamped?) - remove the clamp: each trial is one "
            "press, and a held button never produces a new one")
    mic = ArecordMic(args.device, keep_s=30.0, latency_ms=args.mic_latency_ms)
    mic.start()
    time.sleep(1.0)
    flight_ms = args.mic_distance_m / SOUND_MPS * 1e3
    hold_s = float(cfg.actuator.servo.hold_ms) / 1000.0
    results: list[dict] = []
    last_press = arming.press_count  # a button already held at start does not count: press again
    try:
        for n in range(1, args.trials + 1):
            log(f"trial {n}/{args.trials}: recock, then press and hold the button")
            if not wait_press(arming, last_press, args.press_timeout_s):
                log("  no press: stopping")
                break
            pressed_at = last_press = arming.press_count
            time.sleep(random.uniform(1.0, 3.0))  # not anticipatable
            if not arming.armed or arming.press_count != pressed_at:
                log("  released before the shot: trial skipped (press again)")
                continue
            if led_pin is not None:
                gpio.write(led_pin, 1)
            t_cmd = time.monotonic_ns()
            if not actuator.fire(t_cmd):
                if led_pin is not None:
                    gpio.write(led_pin, 0)
                log("  fire refused (one shot per press: release and press again)")
                continue
            time.sleep(1.0)
            if led_pin is not None:
                gpio.write(led_pin, 0)
            c = mic.clip(t_cmd - 100_000_000, t_cmd + 900_000_000)
            rec: dict = {"trial": n, "cmd_ns": t_cmd}
            if c is not None:
                x, t_first, fs = c
                write_wav(out / f"trial_{n:02d}.wav", x, int(round(fs)))
                t0 = (t_cmd - t_first) / 1e9  # command time within the clip, s
                ons = [((t - t0) * 1e3 - flight_ms, p) for t, p in find_onsets(x, fs, level=args.onset_level)
                       if t >= t0]
                rec["onsets_ms"] = [[round(t, 2), round(p, 3)] for t, p in ons]
                in_hold = [t for t, _p in ons if t <= hold_s * 1e3 + 60.0]
                if in_hold:
                    rec["first_onset_ms"] = in_hold[0]
                    rec["cmd_to_break_ms"] = in_hold[-1]
            results.append(rec)
            log(f"  trial {n}: onsets after cmd (ms, peak): {rec.get('onsets_ms', [])}  -> release the button")
    finally:
        if led_pin is not None:
            gpio.write(led_pin, 0)
        actuator.close()
        mic.stop()
        arming.close()
        gpio.close()

    (out / "trials.json").write_text(__import__("json").dumps(results, indent=2))
    vals = [r["cmd_to_break_ms"] for r in results if "cmd_to_break_ms" in r]
    log(f"\nsaved {out}/trials.json and trial_NN.wav")
    if not vals:
        log("no transients timed")
        return 1
    st = summarize(vals)
    log(f"cmd->last onset in hold: mean {st['mean']:.2f}  std {st['std']:.2f}  p50 {st['p50']:.2f}  "
        f"min {st['min']:.2f}  max {st['max']:.2f}  n {st['n']}")
    if not args.no_write:
        kind = str(cfg.actuator.type)
        update_json(calib.path("latency.json"), {"actuator": {kind: {
            "cmd_to_break_ms": st, "method": "mic", "device": args.device, "run": str(out),
            "mic_latency_ms": args.mic_latency_ms, "mic_distance_m": args.mic_distance_m,
            "fire_us": float(cfg.actuator.servo.fire_us), "rest_us": float(cfg.actuator.servo.rest_us),
            "notes": args.notes, "created": now_iso()}}, "created": now_iso()})
        log(f"wrote {calib.path('latency.json')} [actuator.{kind}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
