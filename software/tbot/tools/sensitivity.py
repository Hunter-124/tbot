"""Error-budget estimates for the MG996R build; no hardware is accessed."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from tbot.config import LatencyBudget, add_common_args, load_from_args
from tbot.fire.ballistics import ShotModel
from tbot.fire.solver import _hit_probability
from tbot.optics import CameraModel

LATENCY_ERRORS_MS = (1, 5, 10, 20)
RANGE_ERRORS = (0.1, 0.2, 0.3)
CAMERA_ERROR_MS = 1.0  # illustrative residual, not the calibrated timestamp correction
SAMPLES = 20000


def latency_miss_rad(angular_rate_rad_s: float, error_s: float) -> float:
    """Small-angle lead error for a fixed timing error."""
    return abs(angular_rate_rad_s * error_s)


def pwm_jitter_sigma_s(frame_hz: float) -> float:
    """Standard deviation of U(0, one PWM frame)."""
    if not math.isfinite(frame_hz) or frame_hz <= 0:
        raise ValueError("PWM frame frequency must be positive and finite")
    return 1.0 / (frame_hz * math.sqrt(12.0))


def range_miss_rad(shot: ShotModel, range_m: float, fraction: float, rate: float) -> float | None:
    """Worst lead error for +/- fractional range error, without clamping ToF."""
    lo, hi = range_m * (1 - fraction), range_m * (1 + fraction)
    if lo < 0 or hi > shot.max_range_m * (1 + 1e-14):
        return None
    hi = min(hi, shot.max_range_m)  # allow roundoff at the tabulated edge
    tof = shot.time_of_flight(range_m)
    dt = max(abs(shot.time_of_flight(lo) - tof), abs(shot.time_of_flight(hi) - tof))
    return latency_miss_rad(rate, dt)


def _max_monotone(good, upper: float) -> tuple[float, bool]:
    """Largest valid value up to upper; bool indicates an untested higher limit."""
    if not good(0.0):
        return 0.0, False
    if good(upper):
        return upper, True
    lo, hi = 0.0, upper
    for _ in range(28):
        mid = (lo + hi) / 2
        if good(mid):
            lo = mid
        else:
            hi = mid
    return lo, False


def build_report(cfg, calib, ranges_m, speeds_mps) -> dict:
    shot = ShotModel(cfg)
    budget = calib.budget or LatencyBudget.from_config(cfg, calib.latency)
    cam = CameraModel.from_calibration(calib.intrinsics, cfg, cfg.camera.width, cfg.camera.height)
    center = np.array([[(cfg.camera.width - 1) / 2, (cfg.camera.height - 1) / 2]], dtype=np.float64)
    tracker_sigma = float(cam.ifov(center)[0]) * float(cfg.tracker.meas_sigma_px)
    hz = float(cfg.actuator.servo.frame_hz)
    pwm_sigma = pwm_jitter_sigma_s(hz)
    # Common random numbers make comparisons reproducible and reduce MC difference noise.
    rng = np.random.default_rng(9217)
    noise_x = rng.standard_normal(SAMPLES) * tracker_sigma
    noise_y = rng.standard_normal(SAMPLES) * tracker_sigma
    phase = rng.uniform(-0.5, 0.5, SAMPLES)
    rows = []
    for geometry, transverse_fraction in (("crossing", 1.0), ("incoming_45deg", 1 / math.sqrt(2))):
        for r in ranges_m:
            pattern = float(shot.pattern_radius(r)) / r
            tof = float(shot.time_of_flight(r))
            for speed in speeds_mps:
                rate = speed * transverse_fraction / r
                base_mc = float(np.mean(noise_x * noise_x + noise_y * noise_y < pattern * pattern))
                def pwm_probability(sigma_s):
                    dx = noise_x + rate * phase * (math.sqrt(12) * sigma_s)
                    return float(np.mean(dx * dx + noise_y * noise_y < pattern * pattern))
                with_pwm = pwm_probability(pwm_sigma)
                def fixed_good(miss):
                    return _hit_probability(miss, pattern, tracker_sigma) >= 0.9
                # Isolated tolerances: each perturbation plus the same Gaussian tracker noise.
                latency_limit, latency_capped = _max_monotone(
                    lambda ms: fixed_good(latency_miss_rad(rate, ms / 1000)), 10000.0)
                range_upper = max(0.0, min(1.0, shot.max_range_m / r - 1.0))
                range_limit, range_capped = _max_monotone(
                    lambda frac: fixed_good(range_miss_rad(shot, r, frac, rate)), range_upper)
                jitter_upper = 4 * pattern / rate
                jitter_limit, jitter_capped = _max_monotone(
                    lambda seconds: pwm_probability(seconds) >= 0.9, jitter_upper)
                range_errors = {str(int(f * 100)): range_miss_rad(shot, r, f, rate) for f in RANGE_ERRORS}
                rows.append({
                    "geometry": geometry, "range_m": r, "speed_mps": speed,
                    "angular_rate_rad_s": rate, "tof_s": tof, "pattern_radius_rad": pattern,
                    "latency_miss_rad": {str(ms): latency_miss_rad(rate, ms / 1000) for ms in LATENCY_ERRORS_MS},
                    "p_hit_without_pwm": base_mc, "p_hit_with_pwm": with_pwm,
                    "pwm_jitter_sigma_s": pwm_sigma, "pwm_miss_sigma_rad": rate * pwm_sigma,
                    "range_miss_rad": range_errors,
                    "camera_timestamp_miss_rad": latency_miss_rad(rate, CAMERA_ERROR_MS / 1000),
                    "tracker_noise_sigma_rad": tracker_sigma,
                    "tolerance": {"baseline_meets_target": _hit_probability(0.0, pattern, tracker_sigma) >= 0.9,
                                  "latency_ms": latency_limit, "latency_capped": latency_capped,
                                  "range_fraction": range_limit, "range_capped": range_capped,
                                  "jitter_sigma_ms": jitter_limit * 1000, "jitter_capped": jitter_capped},
                })
    # Compare representative one-source errors as fractions of the pattern, averaged over rows.
    values = {
        "latency error 10 ms": [x["latency_miss_rad"]["10"] / x["pattern_radius_rad"] for x in rows],
        "PWM jitter 1 sigma": [x["pwm_miss_sigma_rad"] / x["pattern_radius_rad"] for x in rows],
        "range error +/-20%": [x["range_miss_rad"]["20"] / x["pattern_radius_rad"] for x in rows if x["range_miss_rad"]["20"] is not None],
        "camera timestamp residual 1 ms": [x["camera_timestamp_miss_rad"] / x["pattern_radius_rad"] for x in rows],
        "tracker noise 1 sigma": [x["tracker_noise_sigma_rad"] / x["pattern_radius_rad"] for x in rows],
    }
    ranking = sorted(({"source": k, "mean_pattern_fraction": sum(v) / len(v), "cases": len(v)}
                      for k, v in values.items() if v), key=lambda x: x["mean_pattern_fraction"], reverse=True)
    return {
        "assumptions": [
            "Point-mass pellet time of flight and linear-cone effective core pattern from ShotModel; hit = aim error inside pattern disc, not pellet-count probability.",
            "Small-angle angular rate v_transverse/r at the stated range; crossing is fully transverse, incoming is 45 degrees with equal inward and transverse speeds. Constant rate and range during lead.",
            "Tracker angular noise is isotropic Gaussian: configured measurement sigma in pixels times camera centre IFOV; not a measured prediction covariance.",
            "PWM phase U(0,1/frame_hz), mean compensated; Monte Carlo uses 20000 fixed-seed samples and tracker noise. No other errors are combined in PWM probabilities.",
            "Tolerance varies one error at a time with tracker noise only, P(hit)>=0.9; range error is worst of +/- estimates. Jitter tolerance is 1-sigma of centered uniform delay. Capped limits are lower bounds within the model domain.",
            "Camera residual 1 ms is illustrative, not the calibrated offset. Range/latency examples are assumed errors, not measurements; ranking compares their mean fraction of pattern across valid cases.",
        ],
        "calib_dir": str(calib.dir),
        "latency_budget": {"servo_command_to_break_ms": budget.cmd_to_break_ms["servo"],
                           "servo_jitter_ms": budget.jitter_ms["servo"],
                           "servo_source": budget.sources.get("actuator.servo", "default"),
                           "decision_to_muzzle_ms": budget.fire_delay_s("servo") * 1000,
                           "camera_timestamp_offset_ns": budget.camera_timestamp_offset_ns,
                           "camera_timestamp_source": budget.sources.get("camera.timestamp_offset", "default"),
                           "exposure_to_user_ms": budget.exposure_to_user_ms},
        "pwm_frame_hz": hz, "camera_timestamp_residual_ms": CAMERA_ERROR_MS,
        "samples": SAMPLES, "rows": rows, "ranking": ranking,
    }


def format_report(report: dict) -> str:
    b = report["latency_budget"]
    lines = ["MG996R sensitivity analysis — assumptions:"]
    lines += ["  " + s for s in report["assumptions"]]
    lines += [f"  Servo cmd->break {b['servo_command_to_break_ms']:.2f} ms ({b['servo_source']}), "
              f"decision->muzzle {b['decision_to_muzzle_ms']:.2f} ms; budget jitter {b['servo_jitter_ms']:.2f} ms "
              f"(not added to PWM jitter); timestamp correction {b['camera_timestamp_offset_ns']} ns "
              f"({b['camera_timestamp_source']}); PWM {report['pwm_frame_hz']:g} Hz.",
              "", "Geometry and latency miss angles (degrees):",
              "geometry       range speed  rate deg/s  pattern deg   1ms   5ms  10ms  20ms"]
    for x in report["rows"]:
        deg = 180 / math.pi
        m = x["latency_miss_rad"]
        lines.append(f"{x['geometry']:<14} {x['range_m']:5.1f} {x['speed_mps']:5.1f} {x['angular_rate_rad_s']*deg:11.3f} {x['pattern_radius_rad']*deg:12.3f} " + " ".join(f"{m[str(t)]*deg:5.3f}" for t in LATENCY_ERRORS_MS))
    lines += ["", "Range and noise miss angles (degrees); -- = outside ShotModel range:",
              "geometry       range speed  range +/-10%  +/-20%  +/-30%  camera 1ms  tracker sigma  PWM sigma  P(no PWM) P(PWM)"]
    for x in report["rows"]:
        deg = 180 / math.pi
        ms = [x["range_miss_rad"][str(int(f * 100))] for f in RANGE_ERRORS]
        cells = " ".join(f"{v*deg:8.3f}" if v is not None else f"{'--':>8}" for v in ms)
        lines.append(f"{x['geometry']:<14} {x['range_m']:5.1f} {x['speed_mps']:5.1f} {cells} {x['camera_timestamp_miss_rad']*deg:11.3f} {x['tracker_noise_sigma_rad']*deg:14.3f} {x['pwm_miss_sigma_rad']*deg:10.3f} {x['p_hit_without_pwm']:10.3f} {x['p_hit_with_pwm']:6.3f}")
    lines += ["", "Isolated P(hit)>=0.90 tolerances (>= denotes model/search boundary; -- = baseline below 0.90):",
              "geometry       range speed  latency ms  range %  jitter sigma ms"]
    for x in report["rows"]:
        t = x["tolerance"]
        def cell(value, capped):
            return (((">=" if capped else "") + f"{value:.2f}")
                    if t["baseline_meets_target"] else "--")
        lines.append(f"{x['geometry']:<14} {x['range_m']:5.1f} {x['speed_mps']:5.1f} {cell(t['latency_ms'], t['latency_capped']):>11} {cell(t['range_fraction']*100, t['range_capped']):>8} {cell(t['jitter_sigma_ms'], t['jitter_capped']):>16}")
    lines += ["", "What to measure first (mean fraction of pattern for stated representative errors):"]
    for i, item in enumerate(report["ranking"], 1):
        lines.append(f"  {i}. {item['source']}: {item['mean_pattern_fraction']*100:.1f}% ({item['cases']} cases)")
    return "\n".join(lines)


def _positive_list(value: str) -> list[float]:
    try:
        values = [float(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers") from exc
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError("values must be positive and finite")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbot sensitivity", description="MG996R error-budget sensitivity analysis")
    add_common_args(parser)
    parser.add_argument("--range-m", type=_positive_list, default=_positive_list("10,20,30,40"))
    parser.add_argument("--speed-mps", type=_positive_list, default=_positive_list("15,20,25"))
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)
    cfg, calib = load_from_args(args)
    if any(r > cfg.shot.max_range_m for r in args.range_m):
        parser.error("ranges must not exceed shot.max_range_m")
    if not math.isfinite(cfg.actuator.servo.frame_hz) or cfg.actuator.servo.frame_hz <= 0:
        parser.error("actuator.servo.frame_hz must be positive and finite")
    report = build_report(cfg, calib, args.range_m, args.speed_mps)
    print(json.dumps(report, indent=2, allow_nan=False) if args.as_json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
