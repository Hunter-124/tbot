"""`tbot hitl` — hardware-in-the-loop bench: the rig's camera films a monitor that the rig itself drives.

    tbot hitl run [--tests geometry,display,flash,sweep,detect] [--live --mic DEV] ...   measure
    tbot hitl analyze sessions/hitl-YYYYmmdd-HHMMSS                                     re-analyse a run
    tbot hitl mics [--device DEV --seconds 5]                                           list / level-check mics

The rig serves a target page (http://<rig>:8090/) that is opened full screen on the monitor the camera looks
at; the page syncs its clock to the rig and draws the targets the procedures schedule. Procedure, wiring and
pass criteria: docs/HITL.md. Nothing here bypasses the arming interlock: with --live the operator holds the
arming button for every shot, and the gun is UNLOADED.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tbot.app.stream import jsonable
from tbot.config import add_common_args, load_from_args, now_iso, resolve_path, save_json, update_json
from tbot.stats import summarize

TESTS = ("geometry", "display", "flash", "sweep", "detect")
DEFAULT_TESTS = "geometry,display,flash,sweep"
HITL_JSON = "hitl.json"


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.replace(" ", "").split(",") if v)


# ---------------------------------------------------------------------------------------------- run
def _procedure_text(args, cfg) -> str:
    lines = [
        "tbot hitl — bench procedure",
        "  1. Camera (pod on the gun, or the bare camera) aimed at the monitor, 1-2 m away, the bore line on the",
        "     screen. Monitor at full brightness (PWM dimming flickers under a short exposure), room lights steady.",
        "  2. Open the URL below in a browser ON THAT MONITOR, click Start (full screen). Nothing else on top.",
    ]
    if args.live:
        lines += [
            "  3. The gun is UNLOADED. Check the chamber and the magazine now; muzzle in a safe direction.",
            f"  4. Actuator {cfg.actuator.type} fitted; you HOLD the arming button (BCM GPIO{cfg.arming.pin}) when asked,",
            "     release it after each shot. Releasing it at any time stops the actuator.",
        ]
        if args.mic:
            lines.append(f"  5. Microphone '{args.mic}' {args.mic_distance_m:g} m from the trigger group.")
        if args.break_sense:
            lines.append(f"  5. Break-sense contact on BCM GPIO{cfg.gpio.break_sense_pin} (closes to GND at the break).")
    else:
        lines.append("  3. Dry run: the pipeline decides but never drives the actuator (add --live for trigger pulls).")
    return "\n".join(lines)


def _sim_setup(cfg, calib, store, args):
    """Simulated monitor, camera, GPIO/actuator and microphone (self-test of the whole chain)."""
    from tbot.hitl.audio import SimMic
    from tbot.hitl.geometry import ScreenMap, screen_pose_homography
    from tbot.hitl.sim import SimClickLink, SimDisplay, SimScreenCamera, SimTruth
    from tbot.optics import CameraModel

    truth = SimTruth()
    W, H = 1920, 1080
    cam = CameraModel.from_calibration(calib.intrinsics, cfg, int(cfg.camera.width), int(cfg.camera.height))
    H_true = screen_pose_homography(W, H, distance_m=truth.screen_distance_m, width_m=truth.screen_width_m,
                                    roll_deg=1.5, pitch_deg=-2.0, yaw_deg=3.0,
                                    offset_m=(truth.screen_width_m / 6.0, -0.015))
    display = SimDisplay(store, truth, W, H)
    source = SimScreenCamera(cfg, display, ScreenMap(H_true, W, H, cam), truth, seed=int(args.seed),
                             blur_sigma_px=0.25 * float(cfg.detector.psf_px))  # psf_px is a 4-sigma diameter
    mic = SimMic(latency_ms=truth.mic_latency_ms, true_latency_ms=truth.mic_latency_ms, seed=int(args.seed)) \
        if args.live else None
    return truth, display, source, mic, (lambda gpio: SimClickLink(gpio, cfg.gpio.break_sense_pin, mic, truth,
                                                                     float(cfg.actuator.sim.delay_ms),
                                                                     mic_distance_m=float(args.mic_distance_m)))


def run(args) -> int:
    from tbot.camera import open_source
    from tbot.hitl import procedures as P
    from tbot.hitl.analysis import analyze, format_report
    from tbot.hitl.report_html import write_html
    from tbot.hitl.geometry import ScreenMap
    from tbot.hitl.rig import HumanOperator, Rig, SimOperator, TapActuator
    from tbot.hitl.scene import HitlStore
    from tbot.hitl.server import HitlServer
    from tbot.hw.actuator import make_actuator
    from tbot.hw.arming import ArmingSwitch
    from tbot.hw.gpio import SimGpio, open_gpio
    from tbot.optics import CameraModel, MountModel

    tests = [t for t in TESTS if t in {s.strip() for s in args.tests.split(",") if s.strip()}]
    unknown = {s.strip() for s in args.tests.split(",") if s.strip()} - set(TESTS)
    if unknown:
        print(f"unknown test(s): {sorted(unknown)}; choose from {', '.join(TESTS)}")
        return 2
    cfg, calib = load_from_args(args)
    cfg.record.mode = "off"
    cfg.fire.live = bool(args.live)
    if args.sim:
        cfg.gpio.backend = "sim"
        cfg.imu.enabled = False
        cfg.actuator.type = "sim"
        args.break_sense = bool(args.live)
    if args.live and str(cfg.actuator.type) == "none":
        print("--live needs an actuator (actuator.type = servo | pq12 | sim)")
        return 2
    if args.live and not (args.mic or args.break_sense or args.sim):
        print("--live without --mic or --break-sense measures nothing after the command; add one (docs/HITL.md)")
        return 2

    created = time.strftime("%Y%m%d-%H%M%S")
    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg.record.dir) / f"hitl{'-sim' if args.sim else ''}-{created}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = json.loads(calib.path(HITL_JSON).read_text()) if calib.path(HITL_JSON).exists() else {}

    store = HitlStore(bg=int(args.bg))
    truth = display_sim = mic = None
    link_factory = None
    if args.sim:
        truth, display_sim, source, mic, link_factory = _sim_setup(cfg, calib, store, args)
    else:
        source = open_source(cfg.camera.source, cfg, calib)
    gpio = open_gpio(cfg)
    if args.sim and not isinstance(gpio, SimGpio):
        print("--sim needs the sim GPIO backend")
        return 2
    arming = ArmingSwitch(gpio, cfg)
    sense = None
    if args.break_sense:
        from tbot.tools.latency_actuator import BreakSense

        sense = BreakSense(gpio, cfg.gpio.break_sense_pin, pull="up")
    if link_factory is not None and args.live:
        link_factory(gpio)
    actuator = TapActuator(make_actuator(cfg, gpio, arming)) if args.live else None
    if args.mic and not args.sim:
        from tbot.hitl.audio import ArecordMic

        mic = ArecordMic(args.mic, latency_ms=float(args.mic_latency_ms))

    rig = Rig(cfg, calib, source, store, gpio=gpio, arming=arming, actuator=actuator)
    cam = rig.pipeline.cam
    mount = MountModel.from_config(cfg, calib.boresight)
    operator = None
    if args.live:
        operator = SimOperator(gpio, arming, cfg.arming.pin) if args.sim else HumanOperator(arming)
    bench = P.Bench(rig=rig, store=store, cfg=cfg, cam=cam, mount=mount, live=bool(args.live), bg=int(args.bg),
                    level=int(args.level), out_dir=out_dir, rng=np.random.default_rng(args.seed), operator=operator,
                    mic=mic, sense=sense, actuator=actuator, allow_windowed=bool(args.allow_windowed))

    server = None
    print(_procedure_text(args, cfg))
    rc = 0
    try:
        if args.sim:
            display_sim.start()
        else:
            server = HitlServer(store, host=args.host, port=int(args.port)).start()
            print(f"\n  target page: {server.url}   (state: {server.url}hitl/state)\n")
        if mic is not None:
            mic.start()
        rig.start()
        store.set_status(True, "TBOT HITL: connected, waiting for the rig")

        def stage(name: str, gen) -> None:
            store.state.update({"test": name})
            store.set_status(False)
            print(f"[{name}]")
            rig.drive(gen)

        if "geometry" in tests:
            stage("geometry", P.geometry(bench))
        else:
            if "screen" not in saved:
                print(f"no saved screen map in {calib.path(HITL_JSON)}: include 'geometry' in --tests")
                return 2
            stage("geometry (reused)", P.use_geometry(bench, ScreenMap.from_dict(saved["screen"], cam)))
        if "display" in tests:
            exp = args.display_exposure_us
            if exp is None:
                # ~0.8 frame period: almost every transition lands inside an exposure (partial-frame timing),
                # whatever the phase between the monitor's refresh and the camera's frame clock.
                exp = 0.8e6 * rig.pipeline.frame_period_s if P.supports_exposure(bench) else 0.0
            elif exp and not P.supports_exposure(bench):
                print(f"  this source cannot change exposure at run time: display test at {cfg.camera.exposure_us:g} us")
                exp = 0.0
            stage("display", P.display(bench, trials=int(args.display_trials), range_m=float(args.range_m),
                                       exposure_us=float(exp) or None))
        if "flash" in tests:
            stage("flash", P.flash(bench, trials=int(args.trials), range_m=float(args.range_m)))
        if "sweep" in tests:
            stage("sweep", P.sweep(bench, trials=int(args.trials), speeds_dps=_floats(args.speeds_dps),
                                   range_m=float(args.range_m), directions=args.sweep_dir))
        if "detect" in tests:
            stage("detect", P.detect(bench, ranges_m=_floats(args.detect_ranges), range_m=float(args.range_m)))
    except KeyboardInterrupt:
        print("interrupted: analysing what was measured")
        rc = 130
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        rc = 1
    finally:
        store.set_status(True, "TBOT HITL: finished")
        try:
            rig.close()
        finally:
            if actuator is not None:
                actuator.close()
            arming.close()
            gpio.close()
            if mic is not None:
                mic.stop()
            if display_sim is not None:
                display_sim.stop()
            if server is not None:
                time.sleep(0.3)  # let the page pick up the final status
                server.stop()

    page_t, page_ids = store.frames()
    raw = {
        "meta": {
            "created": now_iso(), "tests": tests, "live": bool(args.live), "actuator": str(cfg.actuator.type),
            "source": getattr(source, "name", cfg.camera.source), "config": cfg.to_dict(),
            "calibration": {"intrinsics": calib.intrinsics, "boresight": calib.boresight, "latency": calib.latency},
            "camera": {"width": int(cam.width), "height": int(cam.height)}, "epoch_ns": store.epoch_ns,
            "mic": None if mic is None else mic.stats(), "mic_distance_m": float(args.mic_distance_m),
            "range_m": float(args.range_m), "bg": int(args.bg), "level": int(args.level),
            "sim_truth": None if truth is None else truth.to_dict(),
            "display_model": None if "display" in tests else saved.get("display"),
            "argv": sys.argv[1:],
        },
        "page": {"t": page_t.tolist(), "ids": [list(i) for i in page_ids], "status": store.page_status},
        "frames": rig.frames,
        **bench.raw,
    }
    with gzip.open(out_dir / "raw.json.gz", "wt") as f:
        json.dump(jsonable(raw), f)
    if not rig.frames or "geometry" not in bench.raw:
        print(f"nothing to analyse; raw data in {out_dir}")
        return rc or 1
    rep = analyze(raw)
    save_json(out_dir / "report.json", rep)
    text = format_report(rep)
    (out_dir / "report.txt").write_text(text + "\n")
    html_path = write_html(rep, out_dir / "report.html")
    print()
    print(json.dumps(jsonable(rep), indent=1) if args.json else text)
    print(f"\nrun saved to {out_dir}")
    print(f"report: {html_path}")

    if not args.no_write and not args.sim:
        patch = {"created": now_iso()}
        if "geometry" in tests and rep.get("geometry"):
            patch["screen"] = rep["geometry"]
        if rep.get("display", {}).get("rise", {}).get("n"):
            patch["display"] = {k: rep["display"][k] for k in ("rise", "fall")}
            patch["display"]["exposure_us"] = (raw.get("display") or {}).get("exposure_us")
        patch["page"] = {k: store.page_status.get(k) for k in ("w", "h", "refresh_hz", "ua")}
        update_json(calib.path(HITL_JSON), patch)
        print(f"updated {calib.path(HITL_JSON)}")
    if args.write_latency and not args.sim:
        _write_latency(calib, cfg, rep)
    if any(c["level"] == "FAIL" for c in rep.get("checks", []) + rep.get("sim_check", [])):
        rc = rc or 1
    return rc


def _write_latency(calib, cfg, rep: dict) -> None:
    """Merge the measured command -> break (flash trials) into calib/latency.json (the solver's budget)."""
    fl = (rep.get("flash") or {}).get("trials", [])
    kind = str(cfg.actuator.type)
    breaks = [t["cmd_to_break_ms"] for t in fl if t.get("cmd_to_break_ms") is not None]
    clicks = [t["cmd_to_click_ms"] for t in fl if t.get("cmd_to_click_ms") is not None]
    if breaks:
        stats, notes = summarize(breaks), "tbot hitl: break-sense contact"
    elif clicks:
        lock = float(cfg.gun.lock_time_ms)
        stats = summarize(c - lock for c in clicks)
        notes = f"tbot hitl: microphone click minus gun.lock_time_ms {lock:g} (lock time not measured)"
    else:
        print("--write-latency: no command -> break/click measurement in this run; latency.json untouched")
        return
    update_json(calib.path("latency.json"), {
        "actuator": {kind: {"cmd_to_break_ms": stats, "created": now_iso(), "notes": notes}},
        "created": now_iso()})
    print(f"wrote actuator.{kind}.cmd_to_break_ms (mean {stats['mean']:.1f} ms, n {stats['n']}) to "
          f"{calib.path('latency.json')}")


# ------------------------------------------------------------------------------------------ analyze
def analyze_cmd(args) -> int:
    from tbot.hitl.analysis import analyze, format_report
    from tbot.hitl.report_html import write_html

    run_dir = Path(args.run_dir)
    path = run_dir / "raw.json.gz"
    if not path.exists():
        print(f"no raw.json.gz in {run_dir}")
        return 2
    with gzip.open(path, "rt") as f:
        raw = json.load(f)
    rep = analyze(raw)
    save_json(run_dir / "report.json", rep)
    text = format_report(rep)
    (run_dir / "report.txt").write_text(text + "\n")
    html_path = write_html(rep, run_dir / "report.html")
    print(json.dumps(jsonable(rep), indent=1) if args.json else text)
    print(f"report: {html_path}")
    return 0


# --------------------------------------------------------------------------------------------- mics
def mics(args) -> int:
    from tbot.hitl.audio import ArecordMic, list_devices

    print(list_devices())
    if not args.device:
        print("\nlevel check: tbot hitl mics --device <name from the list, e.g. plughw:1,0 or default>")
        return 0
    mic = ArecordMic(args.device)
    mic.start()
    print(f"\n{args.device}: snap your fingers / dry-fire near it; peak and RMS in dBFS every 0.25 s")
    try:
        t_end = time.monotonic() + float(args.seconds)
        while time.monotonic() < t_end:
            time.sleep(0.25)
            now = time.monotonic_ns()
            clip = mic.clip(now - 250_000_000, now)
            if clip is None:
                continue
            x = clip[0]
            peak = 20 * math.log10(max(float(np.max(np.abs(x))), 1e-6))
            rms = 20 * math.log10(max(float(np.sqrt(np.mean(x * x))), 1e-6))
            print(f"  peak {peak:6.1f} dBFS  rms {rms:6.1f} dBFS  {'#' * max(0, int((peak + 60) / 2))}")
    finally:
        mic.stop()
    s = mic.stats()
    print(f"fitted rate {s['fs_fitted']:.2f} Hz ({s['ppm']:+.1f} ppm), arrival jitter p95 "
          f"{s['arrival_jitter_ms']['p95']:.2f} ms, overruns {s['overruns']}")
    if s["overruns"]:
        print("WARNING: overruns drop samples and break the sample clock; close other audio apps")
    return 0


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot hitl", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure: geometry, display, flash, sweep, detect")
    add_common_args(r)
    r.add_argument("--tests", default=DEFAULT_TESTS, help=f"comma list of {','.join(TESTS)} (default {DEFAULT_TESTS})")
    r.add_argument("--trials", type=int, default=20, help="flash and sweep trials each (default 20)")
    r.add_argument("--display-trials", type=int, default=30, help="display-latency patches (default 30)")
    r.add_argument("--display-exposure-us", type=float, default=None,
                   help="exposure during the display test (default: 0.8 frame period when the source can "
                        "switch at run time; 0 = keep camera.exposure_us)")
    r.add_argument("--range-m", type=float, default=20.0, help="simulated clay range: target size and bore point")
    r.add_argument("--speeds-dps", default="20,40", help="sweep angular speeds, deg/s (default 20,40; the screen's "
                   "angular width limits the fastest testable crossing, see docs/HITL.md)")
    r.add_argument("--sweep-dir", choices=("auto", "both", "ltr", "rtl"), default="auto",
                   help="sweep direction selection (default auto: use directions with sufficient runway)")
    r.add_argument("--detect-ranges", default="10,20,30,40", help="detect-test ranges, m")
    r.add_argument("--bg", type=int, default=170, help="screen background grey (sky), 0-255")
    r.add_argument("--level", type=int, default=90, help="target grey, 0-255 (darker than bg = clay against sky)")
    r.add_argument("--host", default="0.0.0.0", help="target page bind address")
    r.add_argument("--port", type=int, default=8090, help="target page port (default 8090)")
    r.add_argument("--allow-windowed", action="store_true",
                   help="start measuring without the page in full screen (headless/remote browsers)")
    r.add_argument("--live", action="store_true",
                   help="drive the actuator (arming button held by the operator, gun UNLOADED)")
    r.add_argument("--mic", default=None, help="ALSA capture device for the dry-fire click (tbot hitl mics)")
    r.add_argument("--mic-latency-ms", type=float, default=0.0,
                   help="capture latency subtracted from mic times (calibrate against --break-sense)")
    r.add_argument("--mic-distance-m", type=float, default=0.3, help="mic to trigger group (sound flight time)")
    r.add_argument("--break-sense", action="store_true", help="also time the break-sense contact (gpio.break_sense_pin)")
    r.add_argument("--sim", action="store_true", help="simulated monitor/camera/actuator/mic: self-test, no hardware")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None, help="run directory (default sessions/hitl-<date>)")
    r.add_argument("--no-write", action="store_true", help=f"do not update calib/{HITL_JSON}")
    r.add_argument("--write-latency", action="store_true",
                   help="merge the measured command->break into calib/latency.json (live runs)")
    r.add_argument("--json", action="store_true", help="print the report as JSON")

    a = sub.add_parser("analyze", help="re-analyse a saved run")
    a.add_argument("run_dir")
    a.add_argument("--json", action="store_true")

    m = sub.add_parser("mics", help="list capture devices; level-check one")
    m.add_argument("--device", default=None)
    m.add_argument("--seconds", type=float, default=5.0)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "analyze":
        return analyze_cmd(args)
    return mics(args)


if __name__ == "__main__":
    raise SystemExit(main())
