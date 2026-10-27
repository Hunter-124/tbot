"""`tbot sim` — closed-loop evaluation of the whole pipeline on synthetic scenarios.

Every run drives the real Pipeline (detector -> tracker -> solver -> SimActuator) over a SyntheticSource
and then scores it against the scenario's ground truth: detection rate, track latency, prediction error,
and for each shot the TRUE outcome (where the shot actually went, given the actuator's real delay and
the gun attitude at the moment the shot left the muzzle).
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from typing import Sequence

import numpy as np

from tbot.app.run import Pipeline
from tbot.camera.synthetic import SyntheticSource
from tbot.config import Calibration, Config, add_common_args, load_from_args, now_iso, resolve_path, save_json
from tbot.hw.gpio import SimGpio, open_gpio
from tbot.sim.scenario import SCENARIOS
from tbot.stats import summarize

log = logging.getLogger("tbot.sim")

DET_TOL_PX = 2.0  # a detection counts as "found the clay" within this distance of the true clay pixel
LEAD_S = 0.1  # prediction error is measured this far ahead of the frame
TOF_ITERS = 8


def _vec3(x) -> np.ndarray | None:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 3 or not bool(np.all(np.isfinite(a))):
        return None
    return a


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def shot_outcome(source: SyntheticSource, mount, shot, cfg: Config, t_fire_ns: int, delay_s: float) -> dict:
    """Where the shot really went: true bore at exit + true ToF to the clay's true position at arrival."""
    t_exit_ns = int(t_fire_ns + delay_s * 1e9)
    truth_exit = source.truth(t_exit_ns)
    R_wg = np.asarray(truth_exit["R_wg"], dtype=np.float64).reshape(3, 3)
    muzzle_w = R_wg @ (np.asarray(mount.muzzle_pos_g, dtype=np.float64) - np.asarray(mount.cam_pos_g, dtype=np.float64))
    bore_w = R_wg @ np.asarray(mount.bore_dir_g, dtype=np.float64).reshape(3)
    bore_w = bore_w / float(np.linalg.norm(bore_w))

    clay = _vec3(truth_exit.get("clay_pos_w"))
    if clay is None:
        return {
            "t_fire_ns": int(t_fire_ns),
            "t_exit_ns": t_exit_ns,
            "hit": False,
            "reason": "no clay in flight at shot exit",
            "miss_m": float("nan"),
            "pattern_radius_m": float("nan"),
            "range_m": float("nan"),
            "tof_s": float("nan"),
        }

    max_r = float(cfg.shot.max_range_m)
    rng = min(max(float(np.linalg.norm(clay - muzzle_w)), 0.5), max_r)
    tof = 0.0
    t_arrive_ns = t_exit_ns
    for _ in range(TOF_ITERS):
        tof = float(shot.time_of_flight(rng))
        t_arrive_ns = int(t_exit_ns + tof * 1e9)
        nxt = _vec3(source.truth(t_arrive_ns).get("clay_pos_w"))
        if nxt is None:
            break
        clay = nxt
        r_new = min(max(float(np.linalg.norm(clay - muzzle_w)), 0.5), max_r)
        if abs(r_new - rng) < 1e-3:
            rng = r_new
            break
        rng = r_new

    el = math.asin(max(-1.0, min(1.0, float(bore_w[2]))))
    drop = float(shot.drop(rng, el))
    impact = muzzle_w + rng * bore_w - np.array([0.0, 0.0, drop], dtype=np.float64)
    miss = float(np.linalg.norm(clay - impact))
    radius = float(shot.pattern_radius(rng))
    return {
        "t_fire_ns": int(t_fire_ns),
        "t_exit_ns": t_exit_ns,
        "t_arrive_ns": t_arrive_ns,
        "range_m": rng,
        "tof_s": tof,
        "miss_m": miss,
        "pattern_radius_m": radius,
        "hit": bool(miss <= radius),
        "reason": "",
    }


def evaluate_run(
    cfg: Config,
    calib: Calibration,
    scenario: str,
    seed: int,
    actuator_delay_ms: float | None = None,
    armed: bool = True,
    max_frames: int | None = None,
    lead_s: float = LEAD_S,
    det_tol_px: float = DET_TOL_PX,
) -> dict:
    """Run the pipeline once over one scenario/seed and score it against the truth. Returns a metrics dict."""
    cfg = copy.deepcopy(cfg)
    cfg.fire.live = True
    cfg.actuator.type = "sim"
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.stream.enabled = False
    # the synthetic source supplies its own IMU stream, so cfg.imu.enabled is irrelevant here
    if actuator_delay_ms is not None:
        cfg.actuator.sim.delay_ms = float(actuator_delay_ms)
    delay_ms = float(cfg.actuator.sim.delay_ms)
    exit_delay_s = (delay_ms + float(cfg.gun.lock_time_ms) + float(cfg.gun.barrel_time_ms)) / 1000.0

    source = SyntheticSource(cfg, calib, scenario=scenario, seed=seed)
    gpio = open_gpio(cfg)
    if not isinstance(gpio, SimGpio):  # pragma: no cover - open_gpio honours gpio.backend
        gpio.close()
        raise RuntimeError("tbot sim needs the simulated GPIO backend")
    pipe = Pipeline(cfg, calib, source, gpio=gpio)
    if armed:
        gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)

    lead_ns = int(lead_s * 1e9)
    frames = 0
    visible = 0
    detected = 0
    lead_err: list[float] = []
    t_first_visible: int | None = None
    t_first_confirmed: int | None = None
    fire_events: list[tuple[int, dict, bool]] = []
    reasons: dict[str, int] = {}
    armed_frames = 0
    try:
        while True:
            tele = pipe.step()
            if tele is None:
                break
            frames += 1
            t = int(tele["t_exposure_ns"])
            armed_frames += int(bool(tele["armed"]))
            reason = str(tele["decision"]["reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
            if tele["fired"]:
                fire_events.append((int(tele["t_ns"]), dict(tele["decision"]), bool(tele["armed"])))

            truth = source.truth(t)
            px = truth.get("clay_px")
            if bool(truth.get("visible")) and px is not None:
                if t_first_visible is None:
                    t_first_visible = t
                visible += 1
                u0, v0 = float(px[0]), float(px[1])
                if any(math.hypot(float(d.u) - u0, float(d.v) - v0) <= det_tol_px for d in pipe.detections):
                    detected += 1

            best = pipe.tracker.best()
            if best is not None and best.confirmed:
                if t_first_confirmed is None:
                    t_first_confirmed = t
                pred = _vec3(pipe.tracker.predict_dir(best, t + lead_ns))
                true_dir = _vec3(source.truth(t + lead_ns).get("clay_dir_w"))
                if pred is not None and true_dir is not None:
                    lead_err.append(_angle_deg(pred, true_dir))

            if max_frames is not None and frames >= max_frames:
                break
        shots = [
            {
                **shot_outcome(source, pipe.mount, pipe.shot, cfg, t, exit_delay_s),
                "predicted": dec,
                "armed_at_fire": was_armed,
            }
            for t, dec, was_armed in fire_events
        ]
        # The actuator layer's own record of the commands it accepted (SimActuator.fire_times).
        actuator_kind = getattr(pipe.actuator, "kind", None)
        actuator_fire_times = [int(t) for t in getattr(pipe.actuator, "fire_times", [])]
    finally:
        pipe.close()

    track_lat_ms = float("nan")
    if t_first_visible is not None and t_first_confirmed is not None:
        track_lat_ms = (t_first_confirmed - t_first_visible) / 1e6
    hits = sum(1 for s in shots if s["hit"])
    return {
        "scenario": scenario,
        "seed": int(seed),
        "armed": bool(armed),
        "actuator_delay_ms": delay_ms,
        "frames": frames,
        "armed_frames": armed_frames,
        "visible_frames": visible,
        "detected_frames": detected,
        "detection_rate": (detected / visible) if visible else float("nan"),
        "track_latency_ms": track_lat_ms,
        "lead_error_deg": summarize(lead_err),
        "fires": len(fire_events),
        "hits": hits,
        "hit_rate": (hits / len(shots)) if shots else float("nan"),
        "miss_m": summarize([s["miss_m"] for s in shots]),
        "shots": shots,
        "actuator_kind": actuator_kind,
        "actuator_fire_times": actuator_fire_times,
        "t_first_visible_ns": t_first_visible,
        "t_first_confirmed_ns": t_first_confirmed,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------------------------- report
_COLS = (
    ("scenario", 18),
    ("seed", 5),
    ("frames", 7),
    ("vis", 5),
    ("det%", 6),
    ("trk ms", 7),
    ("lead p50", 9),
    ("lead p95", 9),
    ("fires", 6),
    ("hits", 5),
    ("miss m", 7),
)


def _fmt(v: float, nd: int = 2) -> str:
    return "-" if v is None or not math.isfinite(float(v)) else f"{float(v):.{nd}f}"


def format_table(runs: Sequence[dict]) -> str:
    head = "  ".join(f"{name:<{w}}" if i == 0 else f"{name:>{w}}" for i, (name, w) in enumerate(_COLS))
    lines = [head, "-" * len(head)]
    for r in runs:
        lines.append(
            "  ".join(
                [
                    f"{r['scenario']:<18}",
                    f"{r['seed']:>5}",
                    f"{r['frames']:>7}",
                    f"{r['visible_frames']:>5}",
                    f"{_fmt(100.0 * r['detection_rate'], 1):>6}",
                    f"{_fmt(r['track_latency_ms'], 0):>7}",
                    f"{_fmt(r['lead_error_deg']['p50'], 2):>9}",
                    f"{_fmt(r['lead_error_deg']['p95'], 2):>9}",
                    f"{r['fires']:>6}",
                    f"{r['hits']:>5}",
                    f"{_fmt(r['miss_m']['p50'], 2):>7}",
                ]
            )
        )
    return "\n".join(lines)


def summarize_runs(runs: Sequence[dict]) -> dict:
    by_scenario: dict[str, dict] = {}
    for name in dict.fromkeys(r["scenario"] for r in runs):
        rs = [r for r in runs if r["scenario"] == name]
        shots = [s for r in rs for s in r["shots"]]
        by_scenario[name] = {
            "runs": len(rs),
            "detection_rate": summarize([r["detection_rate"] for r in rs]),
            "track_latency_ms": summarize([r["track_latency_ms"] for r in rs]),
            "lead_error_deg_p50": summarize([r["lead_error_deg"]["p50"] for r in rs]),
            "fires": sum(r["fires"] for r in rs),
            "hits": sum(r["hits"] for r in rs),
            "hit_rate": (sum(r["hits"] for r in rs) / len(shots)) if shots else float("nan"),
            "miss_m": summarize([s["miss_m"] for s in shots]),
        }
    total_shots = sum(r["fires"] for r in runs)
    return {
        "runs": len(runs),
        "fires": total_shots,
        "hits": sum(r["hits"] for r in runs),
        "hit_rate": (sum(r["hits"] for r in runs) / total_shots) if total_shots else float("nan"),
        "by_scenario": by_scenario,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot sim", description="closed-loop evaluation of the pipeline on synthetic scenarios"
    )
    add_common_args(parser)
    parser.add_argument(
        "--scenarios", nargs="+", default=None, metavar="NAME",
        help=f"scenarios to run (default: all). available: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--seeds", type=int, default=1, metavar="N", help="run seeds 1..N per scenario")
    parser.add_argument(
        "--actuator-delay-ms", type=float, default=None,
        help="TRUE actuator delay of the simulated trigger (default: actuator.sim.delay_ms)",
    )
    parser.add_argument("--max-frames", type=int, default=None, metavar="N", help="cap the frames per run")
    parser.add_argument("--not-armed", action="store_true", help="never press the arming button (interlock check)")
    parser.add_argument("--out", default=None, metavar="JSON", help="write the full result as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg, calib = load_from_args(args)
    names = list(args.scenarios) if args.scenarios else list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s) {unknown}; available: {', '.join(SCENARIOS)}")
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    runs = []
    for name in names:
        for seed in range(1, int(args.seeds) + 1):
            r = evaluate_run(
                cfg,
                calib,
                name,
                seed,
                actuator_delay_ms=args.actuator_delay_ms,
                armed=not args.not_armed,
                max_frames=args.max_frames,
            )
            runs.append(r)
            print(f"{name:<18} seed {seed}: {r['frames']} frames, {r['fires']} fire(s), {r['hits']} hit(s)", flush=True)

    summary = summarize_runs(runs)
    print()
    print(format_table(runs))
    print()
    print(
        f"total: {summary['fires']} shot(s), {summary['hits']} hit(s), "
        f"hit rate {_fmt(100.0 * summary['hit_rate'], 1)} %"
    )
    if args.out:
        payload = {
            "created": now_iso(),
            "actuator_delay_ms": args.actuator_delay_ms,
            "camera": {"width": cfg.camera.width, "height": cfg.camera.height, "fps": cfg.camera.fps},
            "runs": runs,
            "summary": summary,
        }
        path = save_json(resolve_path(args.out), payload)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
