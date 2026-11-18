"""`tbot sim` — closed-loop evaluation of the whole pipeline on synthetic scenarios.

Every run drives the real Pipeline (detector -> tracker -> solver -> SimActuator) over a SyntheticSource
and then scores it against the scenario's ground truth: detection rate, track latency, prediction error,
and for each shot the TRUE outcome (where the shot actually went, given the actuator's real delay and
the gun attitude at the moment the shot left the muzzle).

Shots are judged as they are fired (closed loop): a shot that hits a clay breaks it at the shot's
arrival time through SyntheticSource.break_clay(index, t_arrive_ns), so later frames show that clay's
fading fragments instead of the clay, and a later shot can only hit the clays still whole. A shot that
breaks nothing but passes within DEBRIS_RADII pattern radii of a fragment is counted as a debris shot.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import time
from typing import Sequence

import numpy as np

from tbot.app.run import Pipeline
from tbot.camera.synthetic import SyntheticSource, T0_NS
from tbot.config import Calibration, Config, add_common_args, load_from_args, now_iso, resolve_path, save_json
from tbot.hw.gpio import SimGpio, open_gpio
from tbot.sim.scenario import SCENARIOS
from tbot.stats import summarize

log = logging.getLogger("tbot.sim")

DET_TOL_PX = 2.0  # a detection counts as "found the clay" within this distance of the true clay pixel
LEAD_S = 0.1  # prediction error is measured this far ahead of the frame
TOF_ITERS = 8
DEBRIS_RADII = 3.0  # a shot that breaks nothing but passes this many pattern radii from a fragment is a debris shot


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


def _intercept(pos_at, t_exit_ns: int, muzzle_w, bore_w, shot, max_r: float):
    """Iterate the shot's time of flight onto a moving point pos_at(t_ns) -> (3,) or None.

    Returns (point at arrival, range, tof, t_arrive_ns, miss distance from the pattern centre, pattern
    radius), or None when pos_at has no position at the shot's exit.
    """
    p = pos_at(t_exit_ns)
    if p is None:
        return None
    rng = min(max(float(np.linalg.norm(p - muzzle_w)), 0.5), max_r)
    tof = 0.0
    t_arrive_ns = t_exit_ns
    for _ in range(TOF_ITERS):
        tof = float(shot.time_of_flight(rng))
        t_arrive_ns = int(t_exit_ns + tof * 1e9)
        nxt = pos_at(t_arrive_ns)
        if nxt is None:
            break
        p = nxt
        r_new = min(max(float(np.linalg.norm(p - muzzle_w)), 0.5), max_r)
        if abs(r_new - rng) < 1e-3:
            rng = r_new
            break
        rng = r_new
    el = math.asin(max(-1.0, min(1.0, float(bore_w[2]))))
    drop = float(shot.drop(rng, el))
    impact = muzzle_w + rng * bore_w - np.array([0.0, 0.0, drop], dtype=np.float64)
    return p, rng, tof, t_arrive_ns, float(np.linalg.norm(p - impact)), float(shot.pattern_radius(rng))


def shot_outcome(source: SyntheticSource, mount, shot, cfg: Config, t_fire_ns: int, delay_s: float) -> dict:
    """Where the shot really went: true bore at exit + true ToF to each clay's true position at arrival.

    The shot is judged against every clay still whole when the shot gets there; the reported geometry
    (range_m, tof_s, miss_m, pattern_radius_m, t_arrive_ns) is that of the nearest such clay in pattern
    radii ("clay" = its index when it is inside the pattern, else None). With no whole clay in flight at
    arrival the nearest clay by miss distance is reported, never as a hit. A shot that breaks nothing
    is a debris shot when its pattern passes within DEBRIS_RADII pattern radii of a fragment of an
    already broken clay. This only judges; the caller reports a hit to the source (break_clay).
    """
    t_exit_ns = int(t_fire_ns + delay_s * 1e9)
    truth_exit = source.truth(t_exit_ns)
    R_wg = np.asarray(truth_exit["R_wg"], dtype=np.float64).reshape(3, 3)
    muzzle_w = R_wg @ (np.asarray(mount.muzzle_pos_g, dtype=np.float64) - np.asarray(mount.cam_pos_g, dtype=np.float64))
    bore_w = R_wg @ np.asarray(mount.bore_dir_g, dtype=np.float64).reshape(3)
    bore_w = bore_w / float(np.linalg.norm(bore_w))
    sc = source.scenario
    max_r = float(cfg.shot.max_range_m)

    def t_s(t_ns: int) -> float:
        return (int(t_ns) - T0_NS) * 1e-9

    candidates = []  # (in flight and whole at arrival, miss / radius, clay index, intercept)
    for c in sc.clays:
        res = _intercept(lambda t_ns, i=c.index: _vec3(sc.clay_pos_w(t_s(t_ns), i)),
                         t_exit_ns, muzzle_w, bore_w, shot, max_r)
        if res is None:
            continue
        # The trajectory curve clamps outside its domain for swing construction.  It is
        # not a stationary target before launch or after the simulated flight ends.
        live = sc.clay_in_flight(t_s(res[3]), c.index)
        candidates.append((live, res[4] / max(res[5], 1e-9), c.index, res))

    # Debris: the fragment the pattern passes closest to (in pattern radii) while it is still visible.
    debris_miss = float("nan")
    debris_radius = float("nan")
    best = math.inf
    for d in sc.debris.values():
        for k in range(d.pos0_w.shape[0]):
            def frag_at(t_ns, d=d, k=k):  # held at the break point before the break
                return d.positions(t_s(t_ns))[k]
            res = _intercept(frag_at, t_exit_ns, muzzle_w, bore_w, shot, max_r)
            if not d.alive(t_s(res[3]))[k]:  # not there (yet / any more) when the shot arrives
                continue
            if res[4] / max(res[5], 1e-9) < best:
                best = res[4] / max(res[5], 1e-9)
                debris_miss, debris_radius = res[4], res[5]

    if not candidates:
        return {
            "t_fire_ns": int(t_fire_ns),
            "t_exit_ns": t_exit_ns,
            "hit": False,
            "clay": None,
            "target_clay": None,
            "debris_shot": False,
            "debris_miss_m": debris_miss,
            "reason": "no clay in flight at shot exit",
            "miss_m": float("nan"),
            "pattern_radius_m": float("nan"),
            "range_m": float("nan"),
            "tof_s": float("nan"),
        }

    live = [cd for cd in candidates if cd[0]]
    if live:
        _, _, idx, (_, rng, tof, t_arrive_ns, miss, radius) = min(live, key=lambda cd: cd[1])
        hit = miss <= radius
        reason = ""
    else:
        _, _, idx, (_, rng, tof, t_arrive_ns, miss, radius) = min(candidates, key=lambda cd: cd[3][4])
        hit = False
        broken = [cd for cd in candidates if cd[2] in sc.broken and sc.broken[cd[2]] <= t_s(cd[3][3])]
        reason = "clay already broken at shot arrival" if len(broken) == len(candidates) \
            else "no clay in flight at shot arrival"
    debris_shot = (not hit) and math.isfinite(debris_miss) and debris_miss <= DEBRIS_RADII * debris_radius
    return {
        "t_fire_ns": int(t_fire_ns),
        "t_exit_ns": t_exit_ns,
        "t_arrive_ns": t_arrive_ns,
        "range_m": rng,
        "tof_s": tof,
        "miss_m": miss,
        "pattern_radius_m": radius,
        "hit": bool(hit),
        "clay": int(idx) if hit else None,
        "target_clay": int(idx),
        "debris_shot": bool(debris_shot),
        "debris_miss_m": debris_miss,
        "reason": reason,
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
        # the debounce runs on a real timer and the simulation does not wait for real time: without this the
        # first frames are armed or not as this computer happens to be fast, and a run does not repeat
        t_end = time.monotonic() + 1.0
        while not pipe.arming.armed and time.monotonic() < t_end:
            time.sleep(0.001)

    lead_ns = int(lead_s * 1e9)
    frames = 0
    visible = 0
    detected = 0
    lead_err: list[float] = []
    t_first_visible: int | None = None
    t_first_confirmed: int | None = None
    fire_events: list[tuple[int, dict, bool]] = []
    shots: list[dict] = []
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
                command_t = tele["cmd_t_ns"] if tele.get("cmd_t_ns") is not None else tele["t_ns"]
                fire_events.append((int(command_t), dict(tele["decision"]), bool(tele["armed"])))
                # Closed loop: judge the shot now and break the clay it hits at the shot's arrival, so
                # the following frames show debris (the arrival is always after the newest frame).
                out = shot_outcome(source, pipe.mount, pipe.shot, cfg, int(command_t), exit_delay_s)
                if out["hit"]:
                    source.break_clay(int(out["clay"]), int(out["t_arrive_ns"]))
                shots.append({**out, "predicted": dict(tele["decision"]), "armed_at_fire": bool(tele["armed"])})

            truth = source.truth(t)
            matched = []
            matched_clay: dict[int, int] = {}  # id(detection) -> clay index
            for c in truth["clays"]:
                px = c.get("clay_px")
                if not (c["visible"] and px is not None):
                    continue
                if t_first_visible is None:
                    t_first_visible = t
                visible += 1
                u0, v0 = float(px[0]), float(px[1])
                m = [d for d in pipe.detections
                     if math.hypot(float(d.u) - u0, float(d.v) - v0) <= det_tol_px]
                detected += int(bool(m))
                for d in m:
                    matched_clay.setdefault(id(d), int(c["index"]))
                matched += m

            # A confirmed background track is not a confirmed clay.  In particular,
            # it can otherwise produce a negative latency before the clay appears.
            matched_tracks = [tr for tr in pipe.tracks if tr.misses == 0 and tr.confirmed
                              and any(tr.last_detection is d for d in matched)]
            if matched_tracks and t_first_confirmed is None:
                t_first_confirmed = t
            best = pipe.tracker.best()
            if best is not None and any(best is tr for tr in matched_tracks):
                ci = matched_clay.get(id(best.last_detection))
                fut = source.truth(t + lead_ns)["clays"]
                if ci is not None and fut[ci]["visible"]:
                    pred = _vec3(pipe.tracker.predict_dir(best, t + lead_ns))
                    true_dir = _vec3(fut[ci]["dir_w"])
                    if pred is not None and true_dir is not None:
                        lead_err.append(_angle_deg(pred, true_dir))

            if max_frames is not None and frames >= max_frames:
                break
        # The actuator layer's own record of the commands it accepted (SimActuator.fire_times).
        actuator_kind = getattr(pipe.actuator, "kind", None)
        actuator_fire_times = [int(t) for t in getattr(pipe.actuator, "fire_times", [])]
    finally:
        pipe.close()

    track_lat_ms = float("nan")
    if t_first_visible is not None and t_first_confirmed is not None:
        track_lat_ms = (t_first_confirmed - t_first_visible) / 1e6
    hits = sum(1 for s in shots if s["hit"])
    sc = source.scenario
    broken = {int(s["clay"]): s for s in shots if s["hit"]}
    break_after_launch_s = [
        (int(s["t_arrive_ns"]) - T0_NS) * 1e-9 - sc.clays[i].launch_t_s for i, s in sorted(broken.items())
    ]
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
        "clays_thrown": sc.n_clays,
        "clays_broken": len(broken),
        "clays_broken_idx": sorted(broken),
        "empty_shots": sum(1 for s in shots if not s["hit"]),  # broke nothing (debris shots included)
        "debris_shots": sum(1 for s in shots if s.get("debris_shot")),
        "break_after_launch_s": break_after_launch_s,  # per broken clay, launch -> breaking shot's arrival
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
    ("broken", 7),
    ("debris", 7),
    ("brk s", 6),
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
                    f"{str(r.get('clays_broken', 0)) + '/' + str(r.get('clays_thrown', 0)):>7}",
                    f"{r.get('debris_shots', 0):>7}",
                    f"{_fmt(_mean(r.get('break_after_launch_s', [])), 2):>6}",
                ]
            )
        )
    return "\n".join(lines)


def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _multi_shot_totals(rs: Sequence[dict]) -> dict:
    """Doubles/debris bookkeeping shared by the per-scenario and the total summary."""
    shots = sum(r["fires"] for r in rs)
    thrown = sum(r.get("clays_thrown", 0) for r in rs)
    broken = sum(r.get("clays_broken", 0) for r in rs)
    return {
        "shots": shots,
        "clays_thrown": thrown,
        "clays_broken": broken,
        "break_rate": (broken / thrown) if thrown else float("nan"),
        "empty_shots": sum(r.get("empty_shots", 0) for r in rs),
        "debris_shots": sum(r.get("debris_shots", 0) for r in rs),
        "shots_per_run": (shots / len(rs)) if rs else float("nan"),
        "break_after_launch_s": summarize([x for r in rs for x in r.get("break_after_launch_s", [])]),
    }


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
            **_multi_shot_totals(rs),
        }
    total_shots = sum(r["fires"] for r in runs)
    return {
        "runs": len(runs),
        "fires": total_shots,
        "hits": sum(r["hits"] for r in runs),
        "hit_rate": (sum(r["hits"] for r in runs) / total_shots) if total_shots else float("nan"),
        **_multi_shot_totals(runs),
        "by_scenario": by_scenario,
    }


def format_scenario_summary(summary: dict) -> str:
    """Per-scenario and total clays broken / thrown, shots, debris shots and break timing."""
    head = (f"{'scenario':<18}  {'runs':>4}  {'shots':>5}  {'shots/run':>9}  {'broken':>9}  "
            f"{'empty':>5}  {'debris':>6}  {'brk p50 s':>9}  {'brk max s':>9}")
    lines = [head, "-" * len(head)]
    rows = list(summary["by_scenario"].items()) + [("total", summary)]
    for name, s in rows:
        lines.append(
            f"{name:<18}  {s['runs']:>4}  {s['shots']:>5}  {_fmt(s['shots_per_run'], 2):>9}  "
            f"{str(s['clays_broken']) + '/' + str(s['clays_thrown']):>9}  {s['empty_shots']:>5}  "
            f"{s['debris_shots']:>6}  {_fmt(s['break_after_launch_s']['p50'], 2):>9}  "
            f"{_fmt(s['break_after_launch_s']['max'], 2):>9}"
        )
    return "\n".join(lines)


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
            print(f"{name:<18} seed {seed}: {r['frames']} frames, {r['fires']} fire(s), {r['hits']} hit(s), "
                  f"{r['clays_broken']}/{r['clays_thrown']} clay(s) broken, {r['debris_shots']} debris shot(s)",
                  flush=True)

    summary = summarize_runs(runs)
    print()
    print(format_table(runs))
    print()
    print(format_scenario_summary(summary))
    print()
    print(
        f"total: {summary['fires']} shot(s), {summary['hits']} hit(s), "
        f"hit rate {_fmt(100.0 * summary['hit_rate'], 1)} %, "
        f"{summary['clays_broken']}/{summary['clays_thrown']} clays broken, "
        f"{summary['debris_shots']} debris shot(s)"
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
