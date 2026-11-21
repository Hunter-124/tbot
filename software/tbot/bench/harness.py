"""`tbot vbench` - the vision front end scored on recorded presses, at pixel level.

    tbot vbench inject --presses pi-pulls/pi-2026-11-20/sessions field_data/2026-11-17-range/presses \
        --clays-per-press 3 --seed 1 --tag legacy --out bench_data/results/legacy_inject.json
    tbot vbench clean  --presses <same> --tag legacy --out bench_data/results/legacy_clean.json
    tbot vbench inject --presses <same> --set vision.engine=motion --tag motion --out motion_inject.json
    tbot vbench compare bench_data/results/legacy_inject.json motion_inject.json

inject: for each press, N clays (tbot.bench.inject: half followed by the recorded swing, half crossing the
frame on their own), each run separately through the REAL pipeline exactly as `tbot rescore` runs a press
(Pipeline + SimGpio, the button pressed and released when the shooter did, the work time modelled
(sim.process_ms) so the clock and so the run are deterministic). clean: the presses as recorded, nothing
injected: every confirmed track and every fire that is not on a real mover (bench_data/truth/<press>.json,
when it exists) is false. The front end under test is chosen ONLY by --set / --config overrides (unknown keys
fail before anything runs). Presses run in parallel worker processes, each with a timeout; a press that
cannot be used (still being written, zero bytes, no IMU, held for less than 0.8 s) is skipped with a note.

Output: a JSON file (per press, per run, aggregated, by clay kind; the overrides and the git revision) and a
text table (tbot.bench.metrics).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MIN_ARMED_S = 0.8          # vclay.MIN_PRESS_S: a shorter hold has no swing to fly a clay through
DEFAULT_TIMEOUT_S = 900.0


# ------------------------------------------------------------------------------------------------ presses
def find_presses(patterns: list[str]) -> list[Path]:
    """Press directories named by the patterns: a press, a folder of presses, or a glob of either."""
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or [pat]
        for h in hits:
            p = Path(h)
            if (p / "meta.json").exists():
                out.append(p)
            elif p.is_dir():
                out += sorted(c for c in p.iterdir() if c.is_dir() and (c / "meta.json").exists())
    seen, uniq = set(), []
    for p in out:  # one copy of each press (the same press may have been pulled into two folders)
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return sorted(uniq, key=lambda p: p.name)


def press_note(path: Path) -> str | None:
    """None if the press can be benchmarked; else why not."""
    from tbot.bench.inject import arming_intervals, press_ok  # noqa: PLC0415

    why = press_ok(path)
    if why:
        return why
    events = []
    for line in (path / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    holds = arming_intervals(events, 2**62)
    held = max((b - a for a, b in holds if b < 2**62), default=0) * 1e-9
    if held < MIN_ARMED_S:
        return f"button held {held:.2f} s (< {MIN_ARMED_S} s)"
    return None


def select(patterns: list[str], limit: int | None) -> tuple[list[Path], list[dict]]:
    ok, skipped = [], []
    for p in find_presses(patterns):
        why = press_note(p)
        if why:
            skipped.append({"press": p.name, "path": str(p), "reason": why})
        else:
            ok.append(p)
    if limit and len(ok) > limit:
        # spread over the list (sorted by name = by day and time), not the first N of one session
        idx = sorted({int(round(x)) for x in np.linspace(0, len(ok) - 1, int(limit))})
        ok = [ok[i] for i in idx]
    return ok, skipped


def press_key(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def load_movers(press: Path, truth_dir: Path | None) -> list[dict]:
    """The real movers of a press (bench_data/truth/<press>.json, written by another tool), if any."""
    if truth_dir is None:
        return []
    f = Path(truth_dir) / f"{press.name}.json"
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text())
    except ValueError:
        return []
    # Only what a person has looked at: the oracle's raw output (reviewed false, every object "unknown") holds
    # clutter chains next to real movers, and counting those as movers would excuse the false tracks on them.
    if not data.get("reviewed"):
        return []
    return [o for o in (data.get("objects") or []) if str(o.get("label", "")).lower() != "false"]


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------------------------------------ one run
def prepare_config(cfg, press: Path):
    """As tbot.tools.rescore.rerun: today's software on the press's rig, a dry run."""
    from tbot.tools.score import rig_config  # noqa: PLC0415

    cfg = rig_config(cfg, press)
    cfg.fire.live = False
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.stream.enabled = False
    cfg.camera.source = f"replay:{press}"
    return cfg


def run_pipeline(cfg, source) -> list[dict]:
    """The press through the pipeline exactly as tbot.tools.rescore.rerun (single loop), with any source.
    Each telemetry record gets "row": the frames.csv row it was made from."""
    from tbot.app.run import Pipeline  # noqa: PLC0415
    from tbot.hw.gpio import SimGpio  # noqa: PLC0415
    from tbot.tools.rescore import _calibration  # noqa: PLC0415

    calib = _calibration(source.meta, cfg)
    rows, n_frames, events = source._rows, source.n_frames, source.events
    gpio = SimGpio()
    pin = int(cfg.arming.pin)
    pressed, released = (0, 1) if cfg.arming.active_low else (1, 0)
    gpio.inject(pin, released)
    presses = sorted((int(e["t_ns"]), bool(e.get("armed"))) for e in events if e.get("type") == "arming")
    pipe = Pipeline(cfg, calib, source, gpio=gpio)
    tel: list[dict] = []
    k = 0
    try:
        pipe.start()
        for i in range(n_frames):
            t_next = int(rows[i][2])
            while k < len(presses) and presses[k][0] <= t_next:
                gpio.inject(pin, pressed if presses[k][1] else released, presses[k][0])
                if presses[k][1]:
                    time.sleep(pipe.arming.debounce_s + 0.02)  # the debounce runs on a real timer
                    # ... which a loaded machine (8 of these at once) may fire late: wait for it, or the frames
                    # after the press would be armed or not depending on this computer's load
                    deadline = time.monotonic() + 2.0
                    while not pipe.arming.armed and time.monotonic() < deadline:
                        time.sleep(0.002)
                k += 1
            rec = pipe.step()
            if rec is None:
                break
            rec = json.loads(json.dumps(rec, default=float))
            last = getattr(source, "last_row", None)
            rec["row"] = int(last if last is not None else getattr(source, "_index", i + 1) - 1)
            tel.append(rec)
    finally:
        pipe.close()
    return tel


def projector(cam, R_gc):
    """(az_deg, el_deg, R_wg of that frame) -> the track's pixel (tbot.tools.score.Session.target_dir)."""
    from tbot.tools.score import _azel_dir  # noqa: PLC0415

    R_gc = np.asarray(R_gc, dtype=np.float64).reshape(3, 3)

    def project(az_deg: float, el_deg: float, R_wg) -> tuple[float, float] | None:
        R = np.asarray(R_wg, dtype=np.float64).reshape(3, 3)
        ray_c = (R @ R_gc).T @ _azel_dir(az_deg, el_deg)
        if ray_c[2] <= 1e-6:
            return None
        uv = np.asarray(cam.ray_to_pixel(ray_c), dtype=np.float64).reshape(2)
        return (float(uv[0]), float(uv[1])) if np.isfinite(uv).all() else None

    return project


def _camera(press: Path, cfg):
    from tbot.optics import CameraModel, MountModel  # noqa: PLC0415

    meta = json.loads((press / "meta.json").read_text())
    cal = meta.get("calibration") or {}
    cam = CameraModel.from_calibration(cal.get("intrinsics"), cfg, int(meta["width"]), int(meta["height"]))
    mount = MountModel.from_config(cfg, cal.get("boresight"))
    return cam, np.asarray(mount.gun_to_cam(np.eye(3)), dtype=np.float64).reshape(3, 3)


def run_press(spec: dict) -> dict:
    """Every run of one press (in a worker process). spec: press, mode, config, sets, clays, seed, psf,
    level_ref, cross_every, truth_dir."""
    from tbot.bench import inject, metrics  # noqa: PLC0415
    from tbot.camera.replay import ReplaySource  # noqa: PLC0415
    from tbot.config import load_config  # noqa: PLC0415
    from tbot.fire.solver import FireSolver  # noqa: PLC0415

    logging.basicConfig(level=logging.WARNING)
    press = Path(spec["press"])
    cfg = prepare_config(load_config(spec.get("config") or None, spec.get("sets") or None), press)
    work = float(cfg.sim.process_ms)
    movers = load_movers(press, Path(spec["truth_dir"]) if spec.get("truth_dir") else None)
    out = {"press": press.name, "path": str(press), "movers": len(movers), "runs": [], "notes": []}
    t0 = time.monotonic()
    if spec["mode"] == "clean":
        cam, R_gc = _camera(press, cfg)
        tel = run_pipeline(cfg, ReplaySource(press, work_ms=work))
        counts = metrics.score_run(tel, projector(cam, R_gc), None, movers)
        out["runs"].append({"clay": None, "counts": counts})
    else:
        stage, bench, clays = inject.build(press, cfg, int(spec["clays"]), int(spec["seed"]), press_key(press.name),
                                           cross_every=int(spec.get("cross_every", 2)))
        if len(clays) < int(spec["clays"]):
            out["notes"].append(f"{int(spec['clays']) - len(clays)} clay(s) could not be placed")
        project = projector(stage.cam, stage.R_gc)
        solver = FireSolver(cfg, bench.shot, stage.mount, bench.budget)
        back = -(bench.true_delay_s + bench.cmd_late_s)

        for clay in clays:
            src = inject.InjectedSource(press, [clay], stage, work_ms=work, psf_sigma=float(spec.get("psf", 0.6)),
                                        level_ref=str(spec.get("level_ref", "scene")))
            truth = {t.row: t for t in src.truth[clay.id]}
            tel = run_pipeline(cfg, src)

            def outcome(rec, clay=clay):
                d = rec.get("decision") or {}
                t_exit = d.get("t_exit_ns")
                if t_exit is None:  # as the solver planned it: the command at fire_at, then the trigger delay
                    t_exit = int(rec.get("fire_at_ns") or rec["t_ns"]) - int(round(back * 1e9))
                # vclay's criterion with the exit time the solver planned (jitter = -delay: t_cmd is the exit)
                r = bench.outcome(stage.press, clay.path, int(t_exit), solver, jitter_s=back)
                return None if r is None else (math.degrees(r[0]), bool(r[1]))

            counts = metrics.score_run(tel, project, truth, movers, outcome, polarity=clay.polarity)
            vis = sum(1 for t in truth.values() if t.visible)
            out["runs"].append({"clay": {**clay.meta(), "visible_frames": vis,
                                         "drawn_frames": sum(1 for t in truth.values() if t.drawn)},
                                "counts": counts, "truth": src.truth_objects()})
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out


# ------------------------------------------------------------------------------------------------ workers
def _worker_main(spec_path: str, out_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    try:
        res = run_press(spec)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal to the whole run
        res = {"press": Path(spec["press"]).name, "path": spec["press"], "error": f"{exc.__class__.__name__}: {exc}",
               "traceback": traceback.format_exc()}
    tmp = Path(out_path + ".tmp")
    tmp.write_text(json.dumps(res, default=float))
    os.replace(tmp, out_path)
    return 0


def run_all(specs: list[dict], workers: int, timeout_s: float, log=sys.stderr) -> list[dict]:
    """Each spec in a worker process of its own (python -m tbot.bench.harness _worker), at most `workers` at
    once, each killed after timeout_s."""
    results: list[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="vbench_"))
    queue = list(enumerate(specs))
    running: dict[int, tuple] = {}
    n = len(specs)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        while queue or running:
            while queue and len(running) < max(1, workers):
                i, spec = queue.pop(0)
                sp, op, lp = tmpdir / f"{i}.spec.json", tmpdir / f"{i}.out.json", tmpdir / f"{i}.log"
                sp.write_text(json.dumps(spec))
                logf = open(lp, "w")
                proc = subprocess.Popen([sys.executable, "-m", "tbot.bench.harness", "_worker", str(sp), str(op)],
                                        stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
                running[i] = (proc, time.monotonic(), spec, op, lp, logf)
            time.sleep(0.2)
            for i in list(running):
                proc, t0, spec, op, lp, logf = running[i]
                name = Path(spec["press"]).name
                rc = proc.poll()
                if rc is None and time.monotonic() - t0 > timeout_s:
                    proc.kill()
                    proc.wait()
                    logf.close()
                    results.append({"press": name, "path": spec["press"], "error": f"timeout after {timeout_s:.0f} s"})
                    del running[i]
                    print(f"[{len(results)}/{n}] {name}: TIMEOUT", file=log, flush=True)
                    continue
                if rc is None:
                    continue
                logf.close()
                del running[i]
                if op.exists():
                    res = json.loads(op.read_text())
                else:
                    tail = lp.read_text(errors="replace")[-1500:] if lp.exists() else ""
                    res = {"press": name, "path": spec["press"], "error": f"worker exited {rc}", "traceback": tail}
                if "error" in res and not spec.get("_retry"):
                    # once more (an import or config caught half-edited, a transient file error); a press that
                    # fails twice is reported
                    print(f"  {name}: {res['error']} - retrying once", file=log, flush=True)
                    op.unlink(missing_ok=True)
                    queue.append((i, {**spec, "_retry": True}))
                    continue
                results.append(res)
                status = res.get("error") or f"{len(res.get('runs', []))} run(s) in {res.get('seconds', 0):.0f} s"
                print(f"[{len(results)}/{n}] {name}: {status}", file=log, flush=True)
    finally:
        for proc, *_rest in running.values():
            proc.kill()
    return results


# ------------------------------------------------------------------------------------------------ report
def report(mode: str, results: list[dict], skipped: list[dict], args, sets: list[str], configs: list[str]) -> dict:
    from tbot.bench import metrics  # noqa: PLC0415

    ok = [r for r in results if "error" not in r]
    errors = [{"press": r["press"], "error": r["error"], "traceback": r.get("traceback", "")[-2000:]}
              for r in results if "error" in r]
    all_counts = [run["counts"] for r in ok for run in r["runs"]]
    out = {
        "mode": mode, "tag": args.tag, "sets": sets, "config": configs, "git_rev": git_rev(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": getattr(args, "seed", None), "clays_per_press": getattr(args, "clays_per_press", None),
        "level_ref": getattr(args, "level_ref", None), "psf_sigma_px": getattr(args, "psf", None),
        "presses": sorted(r["press"] for r in ok), "skipped": skipped, "errors": errors,
        "aggregate": metrics.aggregate(all_counts),
    }
    if mode == "inject":
        by: dict[str, list] = {}
        for r in ok:
            for run in r["runs"]:
                for key in ("kind", "polarity"):
                    by.setdefault(f"{key}:{run['clay'][key]}", []).append(run["counts"])
        out["by_clay"] = {k: metrics.aggregate(v) for k, v in sorted(by.items())}
    out["per_press"] = {r["press"]: {"metrics": metrics.aggregate([run["counts"] for run in r["runs"]]),
                                     "movers": r.get("movers", 0), "notes": r.get("notes", []),
                                     "seconds": r.get("seconds"),
                                     "runs": [{"clay": run["clay"], "counts": metrics.trim(run["counts"]),
                                               **({"truth": run["truth"]} if "truth" in run else {})}
                                              for run in r["runs"]]}
                        for r in sorted(ok, key=lambda r: r["press"])}
    return out


def print_report(rep: dict, file=sys.stdout) -> None:
    from tbot.bench import metrics  # noqa: PLC0415

    agg = rep["aggregate"]
    print(f"vbench {rep['mode']}  tag={rep.get('tag')}  sets={rep.get('sets')}  git {rep.get('git_rev')}", file=file)
    print(f"{len(rep['presses'])} press(es), {len(rep['skipped'])} skipped, {len(rep['errors'])} failed", file=file)
    print(metrics.table(agg, "aggregate:"), file=file)
    for key, m in (rep.get("by_clay") or {}).items():
        print(f"  [{key}] clays {m.get('clays')}: recall {metrics._fmt(m.get('det_recall'), '{:.3f}')}, "
              f"never tracked {metrics._fmt(m.get('never_tracked_frac'), '{:.2f}')}, "
              f"target ok {metrics._fmt(m.get('target_correct_frac'), '{:.2f}')}, "
              f"hit {metrics._fmt(m.get('clays_hit_pct'), '{:.0f}')} %", file=file)
    print("per press:", file=file)
    hdr = f"  {'press':<24s} {'runs':>4s} {'recall':>7s} {'clut/f':>7s} {'fTrk/s':>7s} {'fFire':>5s} {'hit%':>5s}"
    print(hdr, file=file)
    for name, pp in rep["per_press"].items():
        m = pp["metrics"]
        print(f"  {name:<24s} {m.get('runs', 0):>4d} {metrics._fmt(m.get('det_recall'), '{:.2f}'):>7s} "
              f"{metrics._fmt(m.get('clutter_per_frame_mean'), '{:.2f}'):>7s} "
              f"{metrics._fmt(m.get('false_conf_track_frames_per_armed_s'), '{:.1f}'):>7s} "
              f"{m.get('false_fires', 0):>5d} {metrics._fmt(m.get('clays_hit_pct'), '{:.0f}'):>5s}", file=file)
    for s in rep["skipped"]:
        print(f"  skipped {s['press']}: {s['reason']}", file=file)
    for e in rep["errors"]:
        print(f"  FAILED {e['press']}: {e['error']}", file=file)


# ------------------------------------------------------------------------------------------------ CLI
def _add_run_args(p: argparse.ArgumentParser, inject: bool) -> None:
    p.add_argument("--presses", nargs="+", required=True, metavar="DIR_OR_GLOB",
                   help="press directories, folders of presses, or globs of either")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override selecting the front end under test (repeatable; unknown keys fail)")
    p.add_argument("--config", action="append", default=[], metavar="TOML", help="config override file(s)")
    p.add_argument("--tag", default=None, help="a name for this configuration (in the output)")
    p.add_argument("--out", default=None, metavar="FILE.json",
                   help="result file (default bench_data/results/vbench_<mode>_<tag>.json)")
    p.add_argument("--workers", type=int, default=8, help="worker processes (default 8)")
    p.add_argument("--limit", type=int, default=None, help="at most N presses (spread over the list)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per press (default 900)")
    p.add_argument("--truth-dir", default=str(REPO / "bench_data" / "truth"),
                   help="real movers per press: <dir>/<press>.json (default bench_data/truth)")
    if inject:
        p.add_argument("--clays-per-press", type=int, default=3, help="clays per press, one run each (default 3)")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--cross-every", type=int, default=2,
                       help="every Nth clay crosses the frame on its own, the rest are followed (default 2; 0 = none)")
        p.add_argument("--level-ref", choices=("scene", "local"), default="scene",
                       help="clay brightness relative to the scene level (default) or the local background")
        p.add_argument("--psf", type=float, default=0.6, help="optical PSF sigma, px (default 0.6)")


def _run(mode: str, args) -> int:
    from tbot.bench import metrics  # noqa: PLC0415,F401
    from tbot.config import load_config  # noqa: PLC0415

    try:
        load_config(args.config or None, args.set or None)  # unknown keys fail here, before any worker starts
    except Exception as exc:  # noqa: BLE001
        print(f"tbot vbench: bad configuration override: {exc}", file=sys.stderr)
        return 2
    presses, skipped = select(args.presses, args.limit)
    if not presses:
        print("tbot vbench: no usable press", file=sys.stderr)
        for s in skipped:
            print(f"  skipped {s['press']}: {s['reason']}", file=sys.stderr)
        return 1
    print(f"tbot vbench {mode}: {len(presses)} press(es), {len(skipped)} skipped, {args.workers} workers",
          file=sys.stderr, flush=True)
    specs = [{"press": str(p.resolve()), "mode": mode, "sets": list(args.set),
              "config": [str(Path(c).resolve()) for c in args.config],
              "clays": getattr(args, "clays_per_press", 0), "seed": getattr(args, "seed", 0),
              "cross_every": getattr(args, "cross_every", 2), "level_ref": getattr(args, "level_ref", "scene"),
              "psf": getattr(args, "psf", 0.6), "truth_dir": args.truth_dir} for p in presses]
    t0 = time.monotonic()
    results = run_all(specs, args.workers, args.timeout)
    rep = report(mode, results, skipped, args, list(args.set), list(args.config))
    rep["wall_s"] = round(time.monotonic() - t0, 1)
    out = Path(args.out) if args.out else REPO / "bench_data" / "results" / f"vbench_{mode}_{args.tag or 'untagged'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print_report(rep)
    print(f"wrote {out}")
    return 0 if not rep["errors"] else 1


def _compare(a_path: str, b_path: str) -> int:
    from tbot.bench import metrics  # noqa: PLC0415

    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    na, nb = a.get("tag") or Path(a_path).stem, b.get("tag") or Path(b_path).stem
    print(f"A = {a_path}  ({a.get('mode')}, sets {a.get('sets')}, git {a.get('git_rev')})")
    print(f"B = {b_path}  ({b.get('mode')}, sets {b.get('sets')}, git {b.get('git_rev')})")
    common = sorted(set(a.get("presses", [])) & set(b.get("presses", [])))
    if set(a.get("presses", [])) != set(b.get("presses", [])):
        print(f"note: the press sets differ ({len(a.get('presses', []))} vs {len(b.get('presses', []))}, "
              f"{len(common)} common); the aggregates are over each file's own presses")
    print(metrics.compare_table(a["aggregate"], b["aggregate"], "A " + na, "B " + nb))
    for key in sorted(set(a.get("by_clay") or {}) | set(b.get("by_clay") or {})):
        ma, mb = (a.get("by_clay") or {}).get(key, {}), (b.get("by_clay") or {}).get(key, {})
        print(f"  [{key}] recall {metrics._fmt(ma.get('det_recall'), '{:.3f}')} -> "
              f"{metrics._fmt(mb.get('det_recall'), '{:.3f}')}, clays hit "
              f"{metrics._fmt(ma.get('clays_hit_pct'), '{:.0f}')} -> {metrics._fmt(mb.get('clays_hit_pct'), '{:.0f}')} %")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_worker"]:
        return _worker_main(argv[1], argv[2])
    p = argparse.ArgumentParser(prog="tbot vbench", description="pixel-level benchmark of the vision front end on "
                                                                 "recorded presses (injected clays, clean presses)")
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_run_args(sub.add_parser("inject", help="synthetic clays composited into the presses' frames"), True)
    _add_run_args(sub.add_parser("clean", help="the presses as recorded: every track not on a real mover is false"),
                  False)
    c = sub.add_parser("compare", help="two result files side by side, with deltas")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args(argv)
    if args.cmd == "compare":
        return _compare(args.a, args.b)
    return _run(args.cmd, args)


if __name__ == "__main__":
    raise SystemExit(main())
