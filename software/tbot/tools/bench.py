"""`tbot bench` — per-stage processing time at the configured resolution.

Runs the real pipeline stages (detector -> tracker -> fire solver -> overlay -> JPEG encode) over
synthetic frames, with the front end vision.engine names (motion: vision/motion.py + the motion tracker;
legacy: the contrast detector + tracker3d; the scenery filter is not timed), or over a live source with --source, and says whether this machine keeps up with
camera.fps. Writes calib/bench.json, which `tbot latency-report` reads for the processing term.
Use --no-overlay for headless processing, --compare FILE for deltas against a previous bench.json,
and --profile [N] / --profile-out FILE for a separate cProfile run without skewing bench timings.

No actuator and no GPIO are touched: the solver is evaluated with armed=True only so the full
decision path (prediction, miss angle, horizon search) is timed, which is what the Pi has to afford.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import platform
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tbot.config import (
    Calibration,
    Config,
    LatencyBudget,
    add_common_args,
    load_from_args,
    now_iso,
    save_json,
)
from tbot.stats import summarize

STAGES = ("read", "detect", "track", "solve", "overlay", "encode")
# "read" is the wait for the next frame, not work this machine has to do: it is reported but left out
# of the per-frame processing total that the verdict uses.
PROCESS_STAGES = ("detect", "track", "solve", "overlay", "encode")
WARMUP_FRAMES = 3


def platform_name() -> str:
    """Raspberry Pi model string when available, else the machine architecture."""
    try:
        raw = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return platform.machine()
    return raw.decode("utf-8", "replace").replace("\x00", "").strip() or platform.machine()


def opencv_info() -> dict:
    info = {
        "version": cv2.__version__,
        "threads": int(cv2.getNumThreads()),
        "use_optimized": bool(cv2.useOptimized()),
    }
    if hasattr(cv2, "getNumberOfCPUs"):
        info["cpus"] = int(cv2.getNumberOfCPUs())
    if hasattr(cv2, "getCPUFeaturesLine"):
        info["cpu_features"] = str(cv2.getCPUFeaturesLine())
    return info


def run_bench(
    cfg: Config,
    calib: Calibration,
    *,
    frames: int = 300,
    scenario: str | None = None,
    source_spec: str | None = None,
    threads: int | None = None,
    seed: int | None = None,
    jpeg_quality: int | None = None,
    no_overlay: bool = False,
    profiler: cProfile.Profile | None = None,
) -> dict:
    """Time every pipeline stage over `frames` frames and return the bench.json payload."""
    from tbot.app.overlay import draw as overlay_draw
    from tbot.camera import open_source
    from tbot.camera.synthetic import SyntheticSource
    from tbot.fire.ballistics import ShotModel
    from tbot.fire.solver import FireSolver
    from tbot.imu.attitude import FixedAttitude
    from tbot.optics import CameraModel, MountModel, combined_mask
    from tbot.app.run import vision_engine
    from tbot.vision import _native
    from tbot.vision.detector import ClayDetector
    from tbot.vision.tracker3d import make_tracker

    engine = vision_engine(cfg)

    if threads is not None:
        cv2.setNumThreads(int(threads))
    if source_spec:
        source = open_source(source_spec, cfg, calib)
    else:
        source = SyntheticSource(cfg, calib, scenario=scenario, seed=seed)

    quality = int(cfg.stream.jpeg_quality if jpeg_quality is None else jpeg_quality)
    budget = calib.budget if calib.budget is not None else LatencyBudget.from_config(cfg, calib.latency)
    attitude = FixedAttitude()
    frame_period_s = 1.0 / float(cfg.camera.fps)

    samples: dict[str, list[float]] = {s: [] for s in STAGES}
    totals: list[float] = []
    detections_total = 0
    tracks_max = 0
    fired = 0
    processed = 0

    source.start()
    try:
        cam = CameraModel.from_calibration(calib.intrinsics, cfg, source.width, source.height)
        mount = MountModel.from_config(cfg, calib.boresight)
        mask = combined_mask(cam, mount, cfg, calib)
        if engine == "motion":
            from tbot.vision.motion import MotionDetector  # noqa: PLC0415

            detector = MotionDetector(cfg, mask, cam=cam, mount=mount)
        else:
            detector = ClayDetector(cfg, mask)
        tracker = make_tracker(cfg, cam, mount)
        solver = FireSolver(cfg, ShotModel(cfg), mount, budget)
        bore_px = mount.bore_pixel(cam, float(cfg.tracker.range_prior_m))

        warmup = min(WARMUP_FRAMES, max(0, frames - 1))
        for i in range(frames + warmup):
            timing = i >= warmup
            if timing and profiler is not None:
                profiler.enable()
            t0 = time.perf_counter_ns()
            frame = source.read(timeout_s=2.0)
            t1 = time.perf_counter_ns()
            if frame is None:
                if timing and profiler is not None:
                    profiler.disable()
                break
            # (the motion detector gets no attitude: a FixedAttitude is no real one, as in the pipeline)
            dets = detector.detect(frame, R_wg=None) if engine == "motion" else detector.detect(frame)
            t2 = time.perf_counter_ns()
            R_wg = attitude.R_wg(frame.t_exposure_ns)
            tracks = tracker.update(frame.t_exposure_ns, dets, R_wg)
            t3 = time.perf_counter_ns()
            decision = solver.evaluate(
                source.clock.now_ns(), tracker, attitude, True, frame_period_s
            )
            t4 = time.perf_counter_ns()
            if no_overlay:
                t5 = t6 = t4
            else:
                bgr = frame.image if frame.is_color else cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
                canvas = overlay_draw(
                    bgr,
                    detections=dets,
                    tracks=tracks,
                    decision=decision,
                    mask=mask,
                    bore_px=bore_px,
                    telemetry={"fps": float(cfg.camera.fps), "armed": False, "dropped": source.dropped},
                )
                t5 = time.perf_counter_ns()
                ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                t6 = time.perf_counter_ns()
                if not ok:
                    raise RuntimeError("cv2.imencode('.jpg') failed")
            if timing and profiler is not None:
                profiler.disable()
            if not timing:
                continue
            processed += 1
            for name, dt in zip(
                STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)
            ):
                samples[name].append(dt / 1e6)
            totals.append((t6 - t1) / 1e6)
            detections_total += len(dets)
            tracks_max = max(tracks_max, len(tracks))
            fired += 1 if decision.fire else 0
    finally:
        if profiler is not None:
            profiler.disable()
        source.stop()

    if processed == 0:
        raise RuntimeError(f"source {getattr(source, 'name', '?')} delivered no frames")

    total_stats = summarize(totals)
    p95 = float(total_stats["p95"])
    fps_achievable = 1000.0 / p95 if p95 > 0 else float("inf")
    result = {
        "created": now_iso(),
        "platform": platform_name(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": opencv_info(),
        # the detector's C measurement kernel: in use, or why not (not built, switched off, stale)
        "native_measure": _native.describe(bool(getattr(cfg.detector, "native", True))),
        "engine": engine,
        "source": getattr(source, "name", "?"),
        "scenario": scenario if source_spec is None else None,
        "frames": processed,
        "resolution": [int(source.width), int(source.height)],
        "jpeg_quality": quality,
        "fps_target": float(cfg.camera.fps),
        "frame_period_ms": frame_period_s * 1e3,
        "stages": {name: summarize(samples[name]) for name in STAGES},
        "process_stages": list(PROCESS_STAGES[:3] if no_overlay else PROCESS_STAGES),
        "overlay_enabled": not no_overlay,
        "total_ms": total_stats,
        "fps_achievable": fps_achievable,
        "verdict": "OK" if p95 <= frame_period_s * 1e3 else "TOO SLOW",
        "detections_per_frame": detections_total / processed,
        "tracks_max": tracks_max,
        "would_fire_frames": fired,
        "dropped": int(getattr(source, "dropped", 0)),
    }
    return result


def format_bench(result: dict) -> str:
    lines = [
        f"bench {result['source']} {result['resolution'][0]}x{result['resolution'][1]} "
        f"({result.get('engine', 'legacy')} front end) "
        f"x {result['frames']} frames on {result['platform']}",
        f"  python {result['python']}  numpy {result['numpy']}  opencv {result['opencv']['version']}"
        f"  threads {result['opencv']['threads']}  optimized {result['opencv']['use_optimized']}",
    ]
    if "cpu_features" in result["opencv"]:
        lines.append(f"  cpu: {result['opencv']['cpu_features']}")
    if "native_measure" in result:
        lines.append(f"  native measurement kernel: {result['native_measure']}")
    lines.append("")
    lines.append(f"  {'stage':<10}{'p50':>9}{'p95':>9}{'max':>9}{'mean':>9}   [ms]")
    lines.append(f"  {'-' * 46}")
    for name in STAGES:
        st = result["stages"][name]
        tag = name if name in result["process_stages"] else f"({name})"
        lines.append(f"  {tag:<10}{st['p50']:>9.2f}{st['p95']:>9.2f}{st['max']:>9.2f}{st['mean']:>9.2f}")
    st = result["total_ms"]
    lines.append(f"  {'total':<10}{st['p50']:>9.2f}{st['p95']:>9.2f}{st['max']:>9.2f}{st['mean']:>9.2f}")
    lines.append("")
    lines.append(
        f"  {result['fps_achievable']:.1f} fps achievable (p95) vs camera.fps {result['fps_target']:g}"
        f" = {result['frame_period_ms']:.2f} ms/frame -> {result['verdict']}"
    )
    lines.append(
        f"  {result['detections_per_frame']:.2f} detections/frame, {result['tracks_max']} track(s) max,"
        f" {result['would_fire_frames']} would-fire frame(s), {result['dropped']} dropped"
    )
    return "\n".join(lines)


def format_comparison(result: dict, previous: dict) -> str:
    """Compare stage latencies against an earlier bench payload, in milliseconds."""
    lines = [
        "Comparison with previous benchmark (current - previous; ms)",
        f"  {'stage':<10}{'p50 delta':>12}{'p95 delta':>12}{'p95 change':>13}",
    ]
    for name in (*STAGES, "total"):
        current = result["total_ms"] if name == "total" else result["stages"].get(name)
        old = previous.get("total_ms") if name == "total" else previous.get("stages", {}).get(name)
        if not current or not old:
            lines.append(f"  {name:<10} {'unavailable':>37}")
            continue
        p50 = current["p50"] - old["p50"]
        p95 = current["p95"] - old["p95"]
        change = f"{p95 / old['p95'] * 100:+.1f}%" if old["p95"] else "n/a"
        lines.append(f"  {name:<10}{p50:>+12.2f}{p95:>+12.2f}{change:>13}")
    lines.append("  read is excluded from the processing total and verdict.")
    return "\n".join(lines)


def _profile_group(filename: str) -> str:
    path = filename.replace("\\", "/")
    for fragment, group in (
        ("/tbot/vision/detector.py", "detector"),
        ("/tbot/vision/motion.py", "detector"),
        ("/tbot/vision/tracker", "tracker"),
        ("/tbot/fire/", "solver"),
        ("/tbot/imu/attitude.py", "attitude"),
        ("/tbot/optics", "optics"),
        ("/tbot/app/overlay.py", "overlay"),
        ("/tbot/camera/", "source"),
    ):
        if fragment in path:
            return group
    return "other"


def format_profile(profiler: cProfile.Profile, limit: int) -> str:
    stats = pstats.Stats(profiler).stats
    lines = [
        "cProfile of a separate measured-frame run (seconds; timings above are unprofiled).",
        "Functions grouped by defining module; shared/library functions appear under other.",
        "Cumulative times include callees and overlap; source cost is not part of the verdict.",
    ]
    groups = ("detector", "tracker", "solver", "attitude", "optics", "overlay", "source", "other")
    for label, index in (("self time", 2), ("cumulative time", 3)):
        top = sorted(stats.items(), key=lambda item: item[1][index], reverse=True)[:limit]
        lines.append(f"\nTop {limit} functions by {label}:")
        for group in groups:
            entries = [(key, values) for key, values in top if _profile_group(key[0]) == group]
            if not entries:
                continue
            lines.append(f"  {group}:")
            for (filename, lineno, func), (primitive, calls, self_s, cumulative_s, _) in entries:
                lines.append(
                    f"    {self_s:9.4f} self {cumulative_s:9.4f} cumulative "
                    f"{calls:7d} calls ({primitive} primitive)  {filename}:{lineno}({func})"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot bench",
        description="Per-stage processing time at the configured resolution; verdict vs camera.fps.",
    )
    add_common_args(parser)
    parser.add_argument("--frames", type=int, default=300, help="frames to time (default 300)")
    parser.add_argument("--scenario", default=None, help="synthetic scenario (default sim.scenario)")
    parser.add_argument("--threads", type=int, default=None, help="cv2.setNumThreads (0 = OpenCV default)")
    parser.add_argument("--seed", type=int, default=None, help="synthetic scenario seed (default sim.seed)")
    parser.add_argument("--profile", nargs="?", const=25, type=int, default=None, metavar="N",
                        help="profile a separate run; show top N functions (default 25)")
    parser.add_argument("--profile-out", type=Path, help="save profile as a .pstats file (implies --profile)")
    parser.add_argument("--no-overlay", action="store_true", help="skip overlay and JPEG encoding")
    parser.add_argument("--compare", type=Path, metavar="FILE", help="compare against a previous bench.json")
    parser.add_argument("--no-write", action="store_true", help="do not write calib/bench.json")
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    args = parser.parse_args(argv)
    cfg, calib = load_from_args(args)
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    if args.profile is not None and args.profile < 1:
        parser.error("--profile N must be >= 1")
    previous = None
    if args.compare is not None:
        try:
            with args.compare.open(encoding="utf-8") as f:
                previous = json.load(f)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load --compare {args.compare}: {exc}")
        if not isinstance(previous, dict) or not isinstance(previous.get("stages"), dict):
            parser.error("--compare must contain a bench.json object with stages")

    result = run_bench(
        cfg,
        calib,
        frames=args.frames,
        scenario=args.scenario,
        source_spec=args.source,
        threads=args.threads,
        seed=args.seed,
        no_overlay=args.no_overlay,
    )
    profiler = None
    if args.profile is not None or args.profile_out is not None:
        profiler = cProfile.Profile()
        run_bench(
            cfg, calib, frames=args.frames, scenario=args.scenario, source_spec=args.source,
            threads=args.threads, seed=args.seed, no_overlay=args.no_overlay, profiler=profiler,
        )
        if args.profile_out is not None:
            profiler.dump_stats(str(args.profile_out))
    path = calib.path("bench.json")
    if not args.no_write:
        save_json(path, result)
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(format_bench(result))
        if not args.no_write:
            print(f"wrote {path}")
    report_stream = sys.stderr if args.as_json else sys.stdout
    if previous is not None:
        print(format_comparison(result, previous), file=report_stream)
    if profiler is not None:
        print(format_profile(profiler, args.profile or 25), file=report_stream)
        if args.profile_out is not None:
            print(f"wrote profile {args.profile_out}", file=report_stream)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
