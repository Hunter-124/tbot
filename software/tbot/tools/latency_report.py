"""`tbot latency-report` — the whole delay chain, config defaults vs measured, per actuator.

Every row says where its number came from ("measured" = calib/latency.json or calib/bench.json,
"default" = config/default.toml). The bottom table is the question the build actually asks: with the
MG996R servo or with the Actuonix PQ12 on the trigger, how far has a crossing clay moved between the
photons that produced the frame and the shot arriving at it.
"""

from __future__ import annotations

import argparse
import json
import math

from tbot.config import (
    Calibration,
    Config,
    LatencyBudget,
    add_common_args,
    load_from_args,
    now_iso,
)

# Reference target: a skeet crossing clay at the station-4 kind of distance, near its ~20 m/s speed
# a second or so after launch (config prior tracker.range_prior_m = 20 m) [INFERENCE].
REFERENCE_RANGE_M = 20.0
REFERENCE_SPEED_MPS = 20.0
ACTUATOR_ORDER = ("servo", "pq12", "sim", "none")


def _read_bench(calib: Calibration) -> dict | None:
    path = calib.path("bench.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def processing_ms(cfg: Config, bench: dict | None) -> tuple[float, str, str]:
    """(frame -> decision time in ms, source, note) from calib/bench.json when it is there."""
    if bench:
        total = bench.get("total_ms")
        if isinstance(total, dict) and isinstance(total.get("p95"), (int, float)):
            res = bench.get("resolution") or [bench.get("width"), bench.get("height")]
            note = f"bench p95, {res[0]}x{res[1]}" if res and res[0] else "bench p95"
            return float(total["p95"]), "measured", note
    # No bench run: assume the pipeline keeps up, i.e. one frame period of processing [INFERENCE].
    return 1000.0 / float(cfg.camera.fps), "default", f"assumed one frame period @ {cfg.camera.fps:g} fps"


def build_report(
    cfg: Config,
    calib: Calibration,
    *,
    range_m: float = REFERENCE_RANGE_M,
    speed_mps: float = REFERENCE_SPEED_MPS,
) -> dict:
    """The full latency budget plus the per-actuator photon -> shot-arrival chain."""
    from tbot.fire.ballistics import ShotModel

    budget = calib.budget if calib.budget is not None else LatencyBudget.from_config(cfg, calib.latency)
    bench = _read_bench(calib)
    proc_ms, proc_src, proc_note = processing_ms(cfg, bench)
    tof_s = float(ShotModel(cfg).time_of_flight(range_m))

    components: list[dict] = [
        {
            "name": "camera.exposure_to_user",
            "value": float(budget.exposure_to_user_ms),
            "unit": "ms",
            "source": budget.sources.get("camera.exposure_to_user", "default"),
            "note": "end of exposure -> frame in user space",
        },
        {
            "name": "camera.timestamp_offset",
            "value": float(budget.camera_timestamp_offset_ns) / 1e6,
            "unit": "ms",
            "source": budget.sources.get("camera.timestamp_offset", "default"),
            "note": "added to the sensor mid-exposure timestamp",
        },
        {
            "name": "processing.p95",
            "value": float(proc_ms),
            "unit": "ms",
            "source": proc_src,
            "note": proc_note,
        },
        {
            "name": "gun.lock_time",
            "value": float(budget.lock_time_ms),
            "unit": "ms",
            "source": "default",
            "note": "sear release -> primer",
        },
        {
            "name": "gun.barrel_time",
            "value": float(budget.barrel_time_ms),
            "unit": "ms",
            "source": "default",
            "note": "primer -> shot leaves the muzzle",
        },
    ]
    types = [t for t in ACTUATOR_ORDER if t in budget.cmd_to_break_ms]
    types += [t for t in sorted(budget.cmd_to_break_ms) if t not in types]
    for t in types:
        src = budget.sources.get(f"actuator.{t}", "default")
        components.append(
            {
                "name": f"actuator.{t}.cmd_to_break",
                "value": float(budget.cmd_to_break_ms[t]),
                "unit": "ms",
                "source": src,
                "note": "fire command -> trigger break",
            }
        )
        components.append(
            {
                "name": f"actuator.{t}.jitter",
                "value": float(budget.jitter_ms[t]),
                "unit": "ms",
                "source": src,
                "note": "1 sigma of the break delay",
            }
        )

    actuators: dict[str, dict] = {}
    for t in types:
        decision_ms = budget.fire_delay_s(t) * 1e3
        photon_ms = float(budget.exposure_to_user_ms) + float(proc_ms) + decision_ms + tof_s * 1e3
        drift_m = speed_mps * photon_ms / 1e3
        actuators[t] = {
            "cmd_to_break_ms": float(budget.cmd_to_break_ms[t]),
            "jitter_ms": float(budget.jitter_ms[t]),
            "source": budget.sources.get(f"actuator.{t}", "default"),
            "decision_to_muzzle_ms": decision_ms,
            "photon_to_arrival_ms": photon_ms,
            "clay_drift_m": drift_m,
            "clay_drift_deg": math.degrees(math.atan2(drift_m, range_m)),
            "jitter_drift_m": speed_mps * float(budget.jitter_ms[t]) / 1e3,
            "configured": t == cfg.actuator.type,
        }

    return {
        "created": now_iso(),
        "calib_dir": str(calib.dir),
        "configured_actuator": cfg.actuator.type,
        "components": components,
        "actuators": actuators,
        "camera": {
            "fps": float(cfg.camera.fps),
            "frame_period_ms": 1000.0 / float(cfg.camera.fps),
            "exposure_us": float(cfg.camera.exposure_us),
            "exposure_to_user_ms": float(budget.exposure_to_user_ms),
            "timestamp_offset_ns": int(budget.camera_timestamp_offset_ns),
        },
        "processing": {"p95_ms": float(proc_ms), "source": proc_src, "note": proc_note},
        "reference_clay": {
            "range_m": float(range_m),
            "speed_mps": float(speed_mps),
            "tof_s": tof_s,
            "tof_ms": tof_s * 1e3,
            "description": "crossing clay, shot time of flight from the muzzle",
        },
    }


def format_report(report: dict) -> str:
    lines: list[str] = []
    ref = report["reference_clay"]
    lines.append(f"latency budget   calib {report['calib_dir']}   configured actuator: "
                 f"{report['configured_actuator']}")
    lines.append("")
    lines.append(f"  {'component':<32}{'value':>10}  {'source':<9} note")
    lines.append(f"  {'-' * 32}{'-' * 10}  {'-' * 9} {'-' * 36}")
    for c in report["components"]:
        lines.append(f"  {c['name']:<32}{c['value']:>10.3f}  {c['source']:<9} {c['note']}")
    lines.append("")
    lines.append(f"  shot time of flight to {ref['range_m']:.0f} m: {ref['tof_ms']:.1f} ms "
                 f"(reference clay {ref['speed_mps']:.0f} m/s crossing)")
    lines.append("")
    lines.append(f"  {'actuator':<10}{'cmd->break':>12}{'jitter':>9}{'decision->muzzle':>18}"
                 f"{'photon->arrival':>17}{'clay moves':>12}{'':>9}")
    lines.append(f"  {'':<10}{'[ms]':>12}{'[ms]':>9}{'[ms]':>18}{'[ms]':>17}{'[m]':>12}{'[deg]':>9}")
    lines.append(f"  {'-' * 67}{'-' * 20}")
    for name, a in report["actuators"].items():
        mark = "*" if a["configured"] else " "
        lines.append(
            f"  {name + mark:<10}{a['cmd_to_break_ms']:>12.1f}{a['jitter_ms']:>9.1f}"
            f"{a['decision_to_muzzle_ms']:>18.1f}{a['photon_to_arrival_ms']:>17.1f}"
            f"{a['clay_drift_m']:>12.2f}{a['clay_drift_deg']:>9.2f}   ({a['source']})"
        )
    lines.append("")
    lines.append("  'clay moves' = how far the reference clay travels between the photons of the frame "
                 "the decision")
    lines.append("  is based on and the shot reaching it: that is the lead the solver has to predict.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot latency-report",
        description="Print the latency budget (config defaults vs measured) and the per-actuator total.",
    )
    add_common_args(parser)
    parser.add_argument("--range-m", type=float, default=REFERENCE_RANGE_M,
                        help=f"reference clay range (default {REFERENCE_RANGE_M:g} m)")
    parser.add_argument("--speed-mps", type=float, default=REFERENCE_SPEED_MPS,
                        help=f"reference clay speed (default {REFERENCE_SPEED_MPS:g} m/s)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    args = parser.parse_args(argv)
    cfg, calib = load_from_args(args)
    if args.range_m <= 0.0:
        parser.error("--range-m must be > 0")
    report = build_report(cfg, calib, range_m=args.range_m, speed_mps=args.speed_mps)
    print(json.dumps(report, indent=2) if args.as_json else format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
