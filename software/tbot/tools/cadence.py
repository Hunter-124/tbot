"""`tbot cadence <press> ...` - how fast, and how late, the frame loop is on THIS computer, one loop against two.

Runs recorded presses (real frames, real swing, the button as it was held) through the pipeline twice:

  single     one loop does it all: read -> attitude -> scenery filter -> detect -> track -> solve
  handover   as the gun runs live with camera.detect_in_process: a sensor process detects each frame the
             moment it has it, this process tracks and solves the frame before (vision/handover.py)

and prints, over the frames with the button held, the frames a second each sustains and how old a frame is when
its decision is made (taken by the loop -> solver done), with where the time goes. The handover run is free
running: the sensor process goes through the press as fast as it can and the loop takes its newest frame, so
the rate is what the pair of processes can do, as with a camera that always has a frame ready (100 fps).

    tbot cadence sessions/20261119-160243-press                  # on the Pi: the numbers that matter

A dry run by construction (see tbot rescore): no actuator exists in it.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import statistics
import sys
from pathlib import Path

from tbot.config import add_common_args, load_from_args
from tbot.tools.rescore import rerun

SKIP = 5  # armed frames left out at the start of each press (buffers warming, the first tracks)


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return statistics.fmean(xs) if xs else float("nan")


def measure(press: str | Path, cfg, handover: bool) -> dict:
    """One press through one path; times in ms, means over the armed frames."""
    _, tel = rerun(press, cfg, handover="free" if handover else None, raw=True)
    rows = [r for r in tel if r.get("armed")][SKIP:]
    if len(rows) < 2:
        raise RuntimeError("no frames with the button held")
    st = [r["stage_ms"] for r in rows]
    main = [s["imu"] + s["attitude"] + s["track"] + s["imu2"] + s["solve"] for s in st]
    out = {
        "frames": len(rows), "of": sum(1 for r in tel if r.get("armed")),
        # one turn of the loop by the wall clock (the wait for the frame and everything after the decision in it)
        "period_ms": statistics.median(b["wall_ns"] - a["wall_ns"] for a, b in zip(rows, rows[1:])) / 1e6,
        "detect_ms": _mean(s["detect"] for s in st), "track_ms": _mean(s["track"] for s in st),
        "solve_ms": _mean(s["solve"] for s in st),
        "dets": _mean(r["n_detections"] for r in rows), "tracks": _mean(len(r["tracks"]) for r in rows),
        "shots": sum(1 for r in tel if r.get("fired")),
    }
    if handover:
        ho = [r["handover"] for r in rows if r.get("handover")]
        if len(ho) != len(rows):
            raise RuntimeError("the sensor process did not detect every frame")
        out["sensor_ms"] = _mean(h["sensor_ms"] for h in ho)   # frame taken -> detections sent
        out["wait_ms"] = _mean(h["wait_ms"] for h in ho)       # sent -> this loop had them
        out["hints_age"] = _mean(h["hints_age"] for h in ho if h["hints_age"])
        out["age_ms"] = _mean(h["sensor_ms"] + h["wait_ms"] + m for h, m in zip(ho, main))
    else:
        out["age_ms"] = _mean(s["read"] + s["detect"] + m for s, m in zip(st, main))
    out["fps"] = 1000.0 / out["period_ms"]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot cadence",
                                description="frames a second and frame age at the decision on recorded presses: "
                                            "one loop against detection in the sensor process")
    add_common_args(p)
    p.add_argument("presses", nargs="+", metavar="PRESS")
    p.add_argument("--only", choices=("single", "handover"), default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg, _calib = load_from_args(args)
    paths = [Path(m) for pat in args.presses for m in (sorted(glob.glob(pat)) or [pat])]
    results, status = [], 0
    for path in paths:
        row = {"press": path.name}
        try:
            if args.only != "handover":
                row["single"] = measure(path, cfg, False)
            if args.only != "single":
                row["handover"] = measure(path, cfg, True)
        except Exception as exc:  # noqa: BLE001 - one bad press must not stop the rest
            print(f"tbot cadence: {path}: {exc}", file=sys.stderr)
            status = 1
            continue
        results.append(row)
        if not args.json:
            print(f"{path.name}")
            for name in ("single", "handover"):
                r = row.get(name)
                if r is None:
                    continue
                extra = (f"  sensor {r['sensor_ms']:5.1f}  hand-over {r['wait_ms']:4.1f}  hints {r['hints_age']:.1f} "
                         f"frames old" if name == "handover" else "")
                print(f"  {name:<9}{r['fps']:6.1f} fps  frame age at the decision {r['age_ms']:5.1f} ms   "
                      f"detect {r['detect_ms']:5.1f}  track {r['track_ms']:4.1f}  solve {r['solve_ms']:4.1f}{extra}   "
                      f"({r['frames']} of {r['of']} armed frames, {r['dets']:.1f} dets, {r['tracks']:.1f} tracks, "
                      f"{r['shots']} shot(s))")
    if args.json:
        print(json.dumps(results, indent=1))
    elif len(results) > 1:
        for name in ("single", "handover"):
            rs = [r[name] for r in results if name in r]
            if rs:
                n = sum(r["frames"] for r in rs)
                print(f"all presses, {name}: {1000.0 * n / sum(r['period_ms'] * r['frames'] for r in rs):.1f} fps, "
                      f"frame age at the decision {sum(r['age_ms'] * r['frames'] for r in rs) / n:.1f} ms")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
