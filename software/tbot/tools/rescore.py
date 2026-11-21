"""`tbot rescore <press> ...` - run recorded presses again through today's software, and score the result.

A recorded press holds the frames the camera saw, the IMU, and when the button was held. `tbot score` says what
the software of THAT day would have hit. This runs the same frames through the software as it is NOW (detector,
tracker, solver, firing rule, trigger delay), on the gun the press was recorded with (its lens, mount, load and
calibration come from the press), with the button held exactly when the shooter held it, and then scores it the
same way: each planned shot against where the bore really was at the exit time and where the target really was
at the arrival time, and (`--every`) every frame that had a firing solution as if it had fired.

It is the regression test for anything that touches the decision: the answer changes only if the code or the
configuration did. The frames are real, so is the swing; nothing is simulated except the time the gun's own
work takes (`--work-ms`, default sim.process_ms: the Pi 3B's detect + track), which is what makes it repeat.

    tbot rescore field_data/2026-11-17-range/presses/*
    tbot rescore <presses> --set fire.adaptive=false --set fire.min_hit_probability=0.5    # another rule
    tbot rescore <presses> --every                                                         # is P(hit) honest?

    tbot rescore <presses> --handover 1     # detection in a sensor process, tracks one frame stale (as live)
    tbot rescore <presses> --set vision.engine=legacy                                      # the old front end

`--handover N` runs the press the way the gun runs live (camera.detect_in_process): a child process replays the
frames and detects, this process tracks and solves, and the scenery filter knows the tracks as they were N
frames before the one the single loop would use (0 = the same: the answer must then equal the single loop's).

A dry run by construction: no actuator exists in it.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

from tbot.app.run import Pipeline
from tbot.camera.replay import ReplaySource
from tbot.config import Calibration, Config, LatencyBudget, add_common_args, load_from_args
from tbot.hw.gpio import SimGpio
from tbot.tools import score as score_tool


def _calibration(meta: dict, cfg: Config) -> Calibration:
    """The calibration the press was recorded with (what the gun knew that day about its lens, mount and IMU);
    the trigger delay is today's reading of the latency file (config.LatencyBudget)."""
    cal = meta.get("calibration") or {}
    calib = Calibration(dir=Path("."))
    calib.intrinsics = cal.get("intrinsics")
    calib.boresight = cal.get("boresight")
    calib.imu = cal.get("imu")
    calib.latency = cal.get("latency")
    calib.budget = LatencyBudget.from_config(cfg, calib.latency)
    return calib


def rerun(press: str | Path, cfg: Config, work_ms: float | None = None, handover: int | str | None = None,
          raw: bool = False) -> tuple[score_tool.Session, list[dict]]:
    """The press through today's pipeline. Returns (a Session over the new telemetry, that telemetry).
    handover: None = the single loop; N = detection in a sensor process whose scenery filter is N frames
    behind the single loop's (see the module docstring); "free" = the sensor process runs through the press
    as fast as it can and this loop takes its newest frame, as live (`tbot cadence` times that; what it decides
    depends on this computer's speed). raw: the telemetry as the
    pipeline made it (not passed through JSON: the Session returned is then not for scoring)."""
    press = Path(press)
    cfg = score_tool.rig_config(cfg, press)
    cfg.fire.live = False                 # no actuator is built: decisions only
    cfg.gpio.backend = "sim"
    cfg.record.mode = "off"
    cfg.stream.enabled = False
    cfg.camera.source = f"replay:{press}"
    work = float(cfg.sim.process_ms) if work_ms is None else float(work_ms)
    source = ReplaySource(press, work_ms=work)
    calib = _calibration(source.meta, cfg)
    rows, n_frames, events = source._rows, source.n_frames, source.events
    if handover is not None:
        from tbot.camera.proc import SensorProcess  # noqa: PLC0415

        if work <= 0.0:
            raise ValueError("--handover needs a modelled work time (--work-ms > 0): the clock is the session's")
        if str(getattr(cfg.vision, "engine", "legacy")).strip().lower() != "legacy":
            raise ValueError("--handover: detection in a sensor process is the legacy front end's "
                             "(--set vision.engine=legacy); the motion engine detects in the pipeline's own loop")
        source.stop()
        source = SensorProcess(cfg, calib, f"replay:{press}", detect=True, work_ms=work,
                               lockstep=None if handover == "free" else int(handover))
        if not source.detects:
            source.stop()
            raise RuntimeError("the sensor process could not run the detector")
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
            # the button as the shooter held it, set before the frame that first saw it that way
            t_next = int(rows[i][2])
            if handover == "free":  # frames may be skipped: go by the last frame read (one frame late)
                t_next = int(getattr(source, "_t_capture_ns", None) or rows[0][2])
            while k < len(presses) and presses[k][0] <= t_next:
                gpio.inject(pin, pressed if presses[k][1] else released, presses[k][0])
                if presses[k][1]:
                    time.sleep(pipe.arming.debounce_s + 0.02)  # the debounce runs on a real timer
                k += 1
            rec = pipe.step()
            if rec is None:
                break
            if raw:
                rec["wall_ns"] = time.perf_counter_ns()
            tel.append(rec if raw else json.loads(json.dumps(rec, default=float)))
    finally:
        pipe.close()
    return score_tool.Session(press, cfg=cfg, telemetry=tel), tel


def rescore(press: str | Path, cfg: Config, work_ms: float | None = None, every: bool = False,
            handover: int | None = None) -> dict:
    session, tel = rerun(press, cfg, work_ms, handover)
    out = {"press": str(press), "frames": len(tel),
           "armed_frames": sum(1 for r in tel if r.get("armed")),
           "shots": score_tool.score(session),
           "delay_ms": session.budget.fire_delay_s(str(session.cfg.actuator.type)) * 1e3}
    if every:
        out["every_rows"] = score_tool.score(session, every=True)
        out["every"] = score_tool.summarize_every(out["every_rows"])
    return out


def then(press: str | Path) -> str:
    """What the gun did on the day, in a few words."""
    p = Path(press) / "events.jsonl"
    if not p.exists():
        return "-"
    fires = recoil = 0
    p_hit = None
    for line in p.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "fire":
            fires += 1
            p_hit = ev.get("p_hit")
        elif ev.get("type") == "recoil":
            recoil += 1
    if not fires:
        return "no command"
    return f"command at P {p_hit:.2f}" + (", gun fired" if recoil else "")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot rescore",
                                description="recorded presses through today's software, scored against what "
                                            "really happened in them")
    add_common_args(p)
    p.add_argument("presses", nargs="+", metavar="PRESS")
    p.add_argument("--every", action="store_true", help="also score every frame that had a firing solution")
    p.add_argument("--work-ms", type=float, default=None,
                   help="the gun's own work per frame (default sim.process_ms; 0 = this computer's speed)")
    p.add_argument("--handover", type=int, default=None, metavar="N",
                   help="detect in a sensor process, its scenery filter N frames behind the single loop's "
                        "(0 = same answer as the single loop, 1 = as the gun runs live)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg, _calib = load_from_args(args)
    paths = [Path(m) for pat in args.presses for m in (sorted(glob.glob(pat)) or [pat])]
    status = 0
    results = []
    for path in paths:
        try:
            results.append(rescore(path, cfg, args.work_ms, every=args.every, handover=args.handover))
        except Exception as exc:  # noqa: BLE001 - one bad press must not stop the rest
            print(f"tbot rescore: {path}: {exc}", file=sys.stderr)
            status = 1
    if args.json:
        for r in results:
            r.pop("every_rows", None)
        print(json.dumps(results, indent=1, default=float))
        return status
    scored = []
    for r in results:
        name = Path(r["press"]).name
        print(f"{name}: {r['armed_frames']} armed frames, {len(r['shots'])} shot(s) "
              f"(trigger delay {r['delay_ms']:.0f} ms; on the day: {then(r['press'])})")
        for s in r["shots"]:
            print(score_tool.format_shot(s))
            if "hit" in s:
                scored.append(s)
    if scored:
        hits = sum(s["hit"] for s in scored)
        print(f"total: {len(results)} press(es), {sum(len(r['shots']) for r in results)} shot(s), "
              f"{hits}/{len(scored)} of the scored ones would have hit")
    if args.every:
        rows = [x for r in results for x in r.get("every_rows", [])]
        print("every frame with a firing solution, all presses:")
        print(score_tool.format_every(score_tool.summarize_every(rows)))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
