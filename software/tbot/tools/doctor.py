"""Offline configuration and calibration consistency report; never opens hardware."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import socket
import tempfile
import tomllib

import cv2
import numpy as np

from tbot.config import (DEFAULT_CONFIG, LatencyBudget, add_common_args, load_calibration,
                         load_config, now_iso, resolve_path)
from tbot.optics import CameraModel


class Report:
    def __init__(self):
        self.checks = []

    def add(self, group, name, status, detail, hint):
        self.checks.append(dict(group=group, name=name, status=status, detail=str(detail), hint=hint))

    def check(self, group, name, condition, detail, hint, severity="FAIL"):
        self.add(group, name, "PASS" if condition else severity, detail, hint)


def _read_json(path, report):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, ValueError) as exc:
        report.add("Calibration", path.name, "FAIL", f"cannot read {path}: {exc}", "repair or regenerate this calibration file")
        return None


def _age(path):
    days = (dt.datetime.now().timestamp() - path.stat().st_mtime) / 86400
    return max(0.0, days)


def run_checks(cfg):
    r = Report()
    directory = resolve_path(cfg.paths.calib_dir)
    files = {name: _read_json(directory / name, r) for name in
             ("intrinsics.json", "boresight.json", "imu.json", "latency.json", "detector.json", "bench.json")}
    for name, command in (("intrinsics.json", "tbot calib-intrinsics"),
                          ("boresight.json", "tbot calib-boresight"),
                          ("imu.json", "tbot calib-imu"),
                          ("latency.json", "tbot latency-report"),
                          ("detector.json", "tbot calib-detector"),
                          ("mask.png", "tbot calib-detector --auto-mask <session>")):
        path = directory / name
        if not path.exists():
            r.add("Calibration", name + " presence", "WARN", f"{path} missing; nominal values or no mask used", f"run {command}")
        else:
            try:
                age = _age(path)
                r.check("Calibration", name + " age", age <= 365, f"{path}, modified {age:.0f} days ago", f"review and repeat {command} after changes or aging", "WARN")
            except OSError as exc:
                r.add("Calibration", name + " age", "WARN", str(exc), "check file permissions")

    # Match the runtime detector overrides, but still diagnose other files if one is malformed.
    try:
        load_calibration(cfg)
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
        r.add("Calibration", "runtime loading", "FAIL", str(exc), "repair the calibration file and rerun doctor")

    a, s = cfg.actuator, cfg.actuator.servo
    valid_types = ("servo", "pq12", "sim", "none")
    r.check("Actuator", "type", a.type in valid_types, f"actuator.type={a.type}", "use servo, pq12, sim or none")
    if a.type in valid_types:
        r.check("Actuator", "chosen build", a.type == "servo", f"actuator.type={a.type}; chosen build is MG996R servo", "set actuator.type=servo unless deliberately using the fallback", "WARN")
    if a.type == "servo":
        for name, pulse in (("rest_us", s.rest_us), ("fire_us", s.fire_us)):
            r.check("Actuator", name, math.isfinite(pulse) and 500 <= pulse <= 2500,
                    f"actuator.servo.{name}={pulse:g} us (500–2500 us)", "set safe servo pulses with tbot jog servo on an unloaded fixture")
        travel = abs(s.fire_us - s.rest_us)
        measured = (s.min_us > 0 and s.max_us > s.min_us
                    and all(s.min_us <= p <= s.max_us for p in (s.rest_us, s.fire_us)))
        if measured and math.isfinite(travel) and travel > 0:
            # endpoints marked on the real lever with `tbot jog servo` beat the 34° design guess
            r.add("Actuator", "pulse travel", "PASS",
                  f"{travel:g} us, measured endpoints (limits {s.min_us:g}–{s.max_us:g} us)",
                  "re-jog if the lever or trigger set changes")
        else:
            r.check("Actuator", "pulse travel", math.isfinite(travel) and 170 <= travel <= 510,
                    f"{travel:g} us for 34°; expected ~340 us [INFERENCE: ~10 us/degree; tolerance 50%]",
                    "measure servo endpoints and trigger travel with tbot jog servo", "WARN")
        nominal6 = 34 / 60 * 140
        nominal48 = 34 / 60 * 170
        required = nominal48 + 20  # one full 50 Hz command frame, before loading margin
        r.check("Actuator", "hold_ms", math.isfinite(s.hold_ms) and s.hold_ms >= required,
                f"{s.hold_ms:g} ms; 34° takes ~{nominal6:.1f} ms at 6 V or ~{nominal48:.1f} ms at 4.8 V, plus 20 ms frame margin [INFERENCE, unloaded]",
                f"set hold_ms >= {required:.1f} ms and measure loaded travel on an unloaded fixture")
        r.check("Actuator", "frame_hz", s.frame_hz == 50, f"{s.frame_hz:g} Hz; analog servo expects 50 Hz", "set actuator.servo.frame_hz=50")
        r.check("GPIO", "PWM overlay channel", s.pwm_chip == 0 and s.pwm_channel == 0,
                f"GPIO18 overlay pwm,pin=18,func=2 is PWM0, configured pwmchip{s.pwm_chip}/pwm{s.pwm_channel}",
                "set pwm_chip=0 and pwm_channel=0 to match the documented GPIO18 overlay")

    pins = [("arming", cfg.arming.pin), ("latency LED", cfg.gpio.latency_led_pin),
            ("break sense", cfg.gpio.break_sense_pin), ("I2C SDA", 2), ("I2C SCL", 3),
            ("UART TX", 14), ("UART RX", 15)]
    if a.type == "servo":
        pins.append(("servo PWM", 18))
    if a.type == "pq12":
        pins += [("PQ12 extend", a.pq12.pin_extend), ("PQ12 retract", a.pq12.pin_retract)]
    by_pin = {}
    for name, pin in pins:
        by_pin.setdefault(pin, []).append(name)
    conflicts = [f"GPIO{pin}: {', '.join(names)}" for pin, names in by_pin.items() if len(names) > 1]
    invalid = [f"{name}=GPIO{pin}" for name, pin in pins if not 0 <= pin <= 27]
    r.check("GPIO", "pin map", not conflicts and not invalid,
            "; ".join(conflicts + invalid) if conflicts or invalid else ", ".join(f"{name}=GPIO{pin}" for name, pin in pins),
            "assign distinct valid BCM pins; preserve the I2C, UART and PWM wiring")

    fps, w, h = cfg.camera.fps, cfg.camera.width, cfg.camera.height
    valid_rate = math.isfinite(fps) and fps > 0
    period = 1000 / fps if valid_rate else None
    r.check("Camera", "frame rate", valid_rate and w > 0 and h > 0, f"{w}x{h} at {fps:g} fps", "set positive camera dimensions and fps")
    r.check("Camera", "exposure", period is not None and math.isfinite(cfg.camera.exposure_us) and 0 < cfg.camera.exposure_us < period * 1000,
            f"{cfg.camera.exposure_us:g} us vs frame period {period * 1000:.0f} us" if period else f"{cfg.camera.exposure_us:g} us; invalid frame rate",
            "set positive exposure_us below the frame period")
    mode_limits = {(640, 400): 210, (1280, 800): 144}
    limit = mode_limits.get((w, h))
    if limit is None:
        r.add("Camera", "sensor mode", "WARN", f"{w}x{h}: no documented conservative fps limit for this output mode", "check actual OV9281 mode and rate with tbot hwcheck on the Pi")
    else:
        r.check("Camera", "sensor mode", valid_rate and fps <= limit,
                f"{w}x{h} at {fps:g} fps, vendor conservative limit ~{limit} fps", "choose a supported mode or lower camera.fps; verify on the Pi")
    intr = files["intrinsics.json"]
    try:
        cam = CameraModel.from_calibration(intr, cfg, w, h)
        f = cam.fx
        if not math.isfinite(f):
            raise ValueError("non-finite focal length")
        r.add("Camera", "projection", "PASS", f"on-axis fx={f:.1f} px ({'calibrated' if intr else 'nominal'})", "calibrate intrinsics for this camera")
        for distance in (10, 20, 40):
            blur = f * 20 * cfg.camera.exposure_us / 1e6 / distance
            diameter = f * cfg.mount.clay_diameter_m / distance
            area = math.pi * (diameter / 2) ** 2
            r.check("Camera", f"clay {distance} m", cfg.detector.min_area_px <= area <= cfg.detector.max_area_px,
                    f"20 m/s transverse blur ~{blur:.2f} px; face-on clay ~{diameter:.2f} px diameter, ~{area:.1f} px² vs detector {cfg.detector.min_area_px}–{cfg.detector.max_area_px} px² [INFERENCE, on-axis]",
                    "measure detector thresholds on real clays; edge-on clays and segmentation reduce area", "WARN")
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        r.add("Camera", "projection", "FAIL", str(exc), "fix camera dimensions, optics or intrinsics calibration")

    if intr:
        try:
            cw, ch = int(intr["width"]), int(intr["height"])
            compatible = cw > 0 and ch > 0 and w > 0 and h > 0 and abs(cw / ch - w / h) <= .01 * w / h
            r.check("Calibration", "intrinsics size", compatible, f"calibrated {cw}x{ch}, camera {w}x{h}; same-aspect sizes rescale", "recalibrate for the camera aspect ratio")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            r.add("Calibration", "intrinsics size", "FAIL", str(exc), "add valid width and height to intrinsics calibration")
    bore = files["boresight.json"]
    if bore:
        try:
            angles = np.asarray(bore["rpy_deg"], dtype=float)
            ok = angles.shape == (3,) and np.isfinite(angles).all() and np.max(np.abs(angles)) <= 15
            r.check("Calibration", "boresight angles", bool(ok), f"rpy_deg={angles.tolist()} (small correction, <=15°)", "check camera mounting and repeat tbot calib-boresight")
        except (KeyError, ValueError, TypeError) as exc:
            r.add("Calibration", "boresight angles", "FAIL", str(exc), "regenerate boresight.json with rpy_deg")
    imu = files["imu.json"]
    if imu and "R_gun_imu" in imu:
        try:
            matrix = np.asarray(imu["R_gun_imu"], dtype=float)
            # Axis calibration may have small fitted mounting deviations.
            rounded = np.round(matrix)
            ok = (matrix.shape == (3, 3) and np.isfinite(matrix).all() and
                  np.allclose(matrix, rounded, atol=0.1) and np.all(np.abs(rounded).sum(axis=0) == 1) and
                  np.all(np.abs(rounded).sum(axis=1) == 1) and np.linalg.det(rounded) == 1)
            r.check("Calibration", "IMU axis map", bool(ok), f"R_gun_imu={matrix.tolist()}; signed proper permutation within 0.1", "check IMU mounting and rerun tbot calib-imu axes")
        except (ValueError, TypeError) as exc:
            r.add("Calibration", "IMU axis map", "FAIL", str(exc), "repair R_gun_imu or rerun tbot calib-imu axes")
    elif imu:
        r.add("Calibration", "IMU axis map", "WARN", "imu.json has no R_gun_imu; configured map is used", "run tbot calib-imu axes")
    mask_path = directory / "mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            r.add("Calibration", "mask size", "FAIL", f"cannot read {mask_path}", "regenerate mask.png")
        else:
            r.check("Calibration", "mask size", mask.shape == (h, w), f"mask {mask.shape[1]}x{mask.shape[0]}, camera {w}x{h}; runtime resizes masks", "regenerate mask at camera size to avoid resampling artifacts", "WARN")

    try:
        budget = LatencyBudget.from_config(cfg, files["latency.json"])
        for key, value, unit in (("camera.timestamp_offset", budget.camera_timestamp_offset_ns, "ns"),
                                 ("camera.exposure_to_user", budget.exposure_to_user_ms, "ms")):
            source = budget.sources[key]
            r.add("Latency", key, "PASS" if source == "measured" else "WARN", f"{value:g} {unit}, source={source}", "run tbot latency-camera to measure camera timing")
        if a.type in valid_types:
            source = budget.sources[f"actuator.{a.type}"]
            delay = budget.cmd_to_break_ms[a.type]
            jitter = budget.jitter_ms[a.type]
            r.check("Latency", "cmd_to_break_ms", math.isfinite(delay) and delay >= 0 and math.isfinite(jitter) and jitter >= 0,
                    f"{a.type}: {delay:g} ms, jitter {jitter:g} ms, source={source}", "measure tbot latency-actuator on an unloaded fixture")
            if source == "default":
                r.add("Latency", "actuator source", "WARN", "command-to-break and jitter use default estimates", "run tbot latency-actuator on an unloaded fixture")
            if period is not None:
                horizon = cfg.fire.horizon_frames * period
                r.check("Fire gates", "horizon vs jitter", cfg.fire.horizon_frames > 0 and math.isfinite(jitter) and jitter >= 0 and horizon >= jitter,
                        f"{cfg.fire.horizon_frames} frames = {horizon:.1f} ms vs {a.type} jitter {jitter:.1f} ms (1 sigma)", "review horizon_frames and measure actuator jitter", "WARN")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        r.add("Latency", "budget", "FAIL", str(exc), "repair latency.json or rerun latency measurements")
    bench = files["bench.json"]
    if bench is None:
        r.add("Processing", "bench", "WARN", "bench.json missing or unreadable", "run tbot bench on the Pi")
    else:
        try:
            p50, p95 = float(bench["total_ms"]["p50"]), float(bench["total_ms"]["p95"])
            if not all(math.isfinite(v) and v >= 0 for v in (p50, p95)):
                raise ValueError("invalid total_ms percentiles")
            platform = str(bench.get("platform", "unknown"))
            r.check("Processing", "frame budget", period is not None and p95 <= period,
                    f"bench {platform}: total p50={p50:.2f}, p95={p95:.2f} ms vs frame period {period:.2f} ms" if period else "invalid frame rate",
                    "run tbot bench on the Pi with the intended workload and lower fps or optimize if needed", "WARN")
            if platform.lower() not in ("aarch64", "armv7l", "armv6l"):
                r.add("Processing", "benchmark platform", "WARN", f"benchmark platform={platform}; not evidence of Pi performance", "run tbot bench on the Pi")
            if bench.get("resolution") != [w, h] or bench.get("fps_target") != fps:
                r.add("Processing", "benchmark settings", "WARN", f"bench resolution={bench.get('resolution')}, fps={bench.get('fps_target')}; configured {w}x{h} @ {fps:g}", "run tbot bench on the Pi with the current settings")
        except (KeyError, ValueError, TypeError) as exc:
            r.add("Processing", "bench", "FAIL", f"invalid bench.json: {exc}", "run tbot bench on the Pi")

    r.check("Fire gates", "range prior", cfg.fire.min_range_m < cfg.tracker.range_prior_m < cfg.fire.max_range_m,
            f"{cfg.fire.min_range_m:g} < {cfg.tracker.range_prior_m:g} < {cfg.fire.max_range_m:g} m", "place tracker.range_prior_m strictly inside the fire range")
    r.check("Fire gates", "shots per arm",
            1 <= cfg.fire.max_shots_per_arm <= 3 and (cfg.fire.max_shots_per_arm == 1 or cfg.fire.min_shot_interval_ms >= 250.0),
            f"max_shots_per_arm={cfg.fire.max_shots_per_arm}, min_shot_interval_ms={cfg.fire.min_shot_interval_ms:g}",
            "use 1-3 shots per press, with fire.min_shot_interval_ms >= 250 for more than one")
    r.check("Fire gates", "live mode", not cfg.fire.live, f"fire.live={str(cfg.fire.live).lower()}", "set fire.live=false for dry run; enable live only deliberately on a safe rig", "WARN")
    try:
        with DEFAULT_CONFIG.open("rb") as fh:
            default_live = tomllib.load(fh)["fire"]["live"]
        r.check("Fire gates", "default live mode", default_live is False,
                f"config/default.toml fire.live={str(default_live).lower()}",
                "set fire.live=false in config/default.toml")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        r.add("Fire gates", "default live mode", "FAIL", str(exc), "repair config/default.toml")

    record = resolve_path(cfg.record.dir)
    try:
        ancestor = record
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise OSError(f"{ancestor} is not a directory")
        with tempfile.TemporaryFile(dir=ancestor):
            pass
        r.add("Local resources", "record directory", "PASS", f"{record}: {'writable' if record.exists() else f'can be created under {ancestor}'}", "choose a writable record.dir")
    except OSError as exc:
        r.add("Local resources", "record directory", "FAIL", f"{record}: {exc}", "choose a writable record.dir")
    try:
        if not 1 <= cfg.stream.port <= 65535:
            raise ValueError("port must be 1–65535")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((cfg.stream.host, cfg.stream.port))
        r.add("Local resources", "stream port", "PASS", f"{cfg.stream.host}:{cfg.stream.port} available at check time", "choose an available stream.host and port")
    except (OSError, ValueError, OverflowError) as exc:
        r.add("Local resources", "stream port", "WARN", f"{cfg.stream.host}:{cfg.stream.port}: {exc}", "choose an available stream.host and port")
    return r.checks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tbot doctor", description="Offline config, calibration and layout consistency checks; never moves hardware.")
    add_common_args(parser)
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable report")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
        if args.source:
            cfg.camera.source = args.source
        if args.calib_dir:
            cfg.paths.calib_dir = args.calib_dir
        checks = run_checks(cfg)
    except (OSError, ValueError, TypeError) as exc:
        checks = [dict(group="Configuration", name="load", status="FAIL", detail=str(exc), hint="fix the config file or --set assignment")]
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"
    if args.as_json:
        print(json.dumps(dict(created=now_iso(), status=status, checks=checks), indent=2))
    else:
        print("tbot doctor — offline consistency checks (no hardware is opened)")
        for group in dict.fromkeys(c["group"] for c in checks):
            print(f"\n{group}:")
            for c in checks:
                if c["group"] == group:
                    print(f"  {c['status']:<4} {c['name']}: {c['detail']} — {c['hint']}")
        print(f"\nVerdict: {status}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
