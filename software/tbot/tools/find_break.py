"""`tbot find-break` - find the pulse where the sear breaks, by ear: the IMU hears the hammer fall.

The lever is stepped slowly from well clear of the trigger toward actuator.servo.fire_us; the IMU on the gun
records a sharp knock (0.7-1.7 g on the bench, 2026-11-19) when the hammer falls. The first pulse that knocks is
the break. The lever then goes straight back and limp.

    tbot find-break --unloaded            # EMPTY gun, hammer COCKED, pipeline stopped (tbotctl stop)
    tbot find-break --unloaded --apply    # ... and move rest / fire / min / break in config/local.toml by the shift

`--apply` keeps the geometry the latency calibration was measured with: every pulse moves by (new break - old
break), so the travel from rest to the break, the pre-travel fraction and the measured delays stay what they were.
It is what to run when the trigger set, the horn or the lever has been off the gun (a spline tooth is 14 deg).

Never beyond fire_us, never outside min_us..max_us. Nothing is commanded unless --unloaded is given.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

from tbot.config import add_common_args, load_from_args

G = 9.80665


def knock_g(samples, prev_mag: float | None) -> tuple[float, float | None]:
    """(largest sample-to-sample jump of |accel| in g, last magnitude) over a batch of IMU samples."""
    worst = 0.0
    for s in samples:
        a = s.accel
        mag = math.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2) / G
        if prev_mag is not None:
            worst = max(worst, abs(mag - prev_mag))
        prev_mag = mag
    return worst, prev_mag


def sweep(pwm, imu, start_us: float, end_us: float, step_us: float, dwell_s: float, threshold_g: float,
          sleep=time.sleep, log=print) -> tuple[float | None, list[tuple[float, float]]]:
    """Step from start_us toward end_us; return (first pulse that knocked or None, [(pulse, loudest jump g)])."""
    direction = -1.0 if end_us < start_us else 1.0
    n = int(abs(end_us - start_us) // step_us)
    heard: list[tuple[float, float]] = []
    prev = None
    pwm.set_pulse_us(start_us)
    sleep(0.6)
    _, prev = knock_g(imu.read(), prev)  # the move to the start is not a knock
    for i in range(1, n + 1):
        us = start_us + direction * i * step_us
        pwm.set_pulse_us(us)
        loud = 0.0
        t_end = time.monotonic() + dwell_s
        while True:
            sleep(0.01)
            j, prev = knock_g(imu.read(), prev)
            loud = max(loud, j)
            if loud >= threshold_g or time.monotonic() >= t_end:
                break
        heard.append((us, loud))
        log(f"  {us:7.1f} us   {loud:5.2f} g" + ("   <-- KNOCK" if loud >= threshold_g else ""))
        if loud >= threshold_g:
            return us, heard
    return None, heard


def shifted(servo, new_break: float) -> dict:
    """rest / fire / min / break moved by the shift of the break (the geometry of the latency calibration kept)."""
    shift = float(new_break) - float(servo.break_us)
    out = {"rest_us": float(servo.rest_us) + shift, "fire_us": float(servo.fire_us) + shift,
           "min_us": float(servo.min_us) + shift, "break_us": float(new_break)}
    if out["rest_us"] > float(servo.max_us):
        raise ValueError(f"the shifted rest ({out['rest_us']:.0f} us) is beyond max_us ({servo.max_us:.0f}): "
                         "refit the horn a tooth back instead")
    return out


def apply_to_local(path: Path, values: dict, note: str) -> None:
    """Rewrite the [actuator.servo] keys in config/local.toml in place (comments of other lines kept)."""
    text = path.read_text()
    for key, val in values.items():
        pat = re.compile(rf"^{key}\s*=.*$", re.M)
        line = f"{key} = {val:.1f}".ljust(30) + f"# {note}"
        if not pat.search(text):
            raise ValueError(f"{key} is not in {path}: set it by hand")
        text = pat.sub(line, text, count=1)
    path.with_suffix(".toml.bak-find-break").write_text(path.read_text())
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot find-break", description=__doc__.splitlines()[0])
    add_common_args(ap)
    ap.add_argument("--unloaded", action="store_true", help="the gun is EMPTY and pointed safely (required)")
    ap.add_argument("--step-us", type=float, default=8.0)
    ap.add_argument("--dwell-ms", type=float, default=220.0)
    ap.add_argument("--threshold-g", type=float, default=0.4, help="sample-to-sample |accel| jump that is a knock")
    ap.add_argument("--apply", action="store_true", help="write the shifted pulses to config/local.toml")
    args = ap.parse_args(argv)
    if not args.unloaded:
        print("refused: give --unloaded (EMPTY gun, pointed safely, hammer cocked, pipeline stopped)", file=sys.stderr)
        return 2
    cfg, calib = load_from_args(args)
    s = cfg.actuator.servo
    if str(cfg.actuator.type) != "servo" or not float(getattr(s, "break_us", 0.0)):
        print("needs actuator.type = servo with break_us marked", file=sys.stderr)
        return 2
    from tbot.hw.gpio import HardwarePwm
    from tbot.imu.mpu6050 import Mpu6050

    start, end = float(s.max_us), float(s.fire_us)
    print(f"stepping {start:.0f} -> {end:.0f} us in {args.step_us:g} us steps (old break {s.break_us:.0f}, rest "
          f"{s.rest_us:.0f}); listening for a knock of {args.threshold_g:g} g")
    pwm = HardwarePwm(int(s.pwm_chip), int(s.pwm_channel), float(s.frame_hz))
    imu = Mpu6050(cfg, calib)
    found = None
    try:
        imu.start()
        time.sleep(0.5)
        found, _ = sweep(pwm, imu, start, end, float(args.step_us), float(args.dwell_ms) / 1e3, float(args.threshold_g))
    finally:
        try:
            pwm.set_pulse_us(start)   # straight back, clear of the trigger
            time.sleep(0.4)
            pwm.disable()
        finally:
            pwm.close()
            imu.stop()
    if found is None:
        print("no knock heard: was the hammer cocked? (nothing changed)")
        return 1
    new = shifted(s, found)
    print(f"BREAK at {found:.0f} us (was {s.break_us:.0f}: shift {found - s.break_us:+.0f} us)")
    print("  -> " + "  ".join(f"{k} {v:.0f}" for k, v in new.items()))
    if args.apply:
        from tbot.config import resolve_path

        path = resolve_path("config/local.toml")
        apply_to_local(path, new, f"tbot find-break {time.strftime('%Y-%m-%d %H:%M')}: break heard at {found:.0f} us")
        print(f"written to {path} (old file kept beside it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
