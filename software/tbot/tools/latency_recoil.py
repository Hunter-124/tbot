"""`tbot latency-recoil` — the trigger delay measured from live shots.

`tbot run --live` logs, for every fire command, when the IMU saw the gun recoil (tbot/fire/recoil.py). That
interval is `actuator cmd_to_break + gun lock time`, the one part of the latency budget the bench could only
estimate. This tool collects those events from recorded sessions (events.jsonl) and/or telemetry logs
(--log-jsonl files), prints the statistics, and with --write stores

    actuator.<type>.cmd_to_break_ms = median recoil delay - gun.lock_time_ms

in calib/latency.json (the previous file is kept as latency.prev.json).

    tbot latency-recoil                       # every session under record.dir
    tbot latency-recoil sessions/20261118-*   # some of them
    tbot latency-recoil --write               # apply (needs --min-shots shots, default 5)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from tbot.config import LatencyBudget, add_common_args, load_from_args, now_iso, resolve_path, update_json
from tbot.fire.recoil import summarise

MAX_SANE_MS = 450.0
MIN_SANE_MS = 20.0


def _events_in(path: Path):
    """Recoil/no_recoil event dicts from a session directory, an events.jsonl, or a telemetry jsonl."""
    files = [path / "events.jsonl"] if path.is_dir() else [path]
    for f in files:
        if not f.exists():
            continue
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or "recoil" not in line:
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(ev, dict) and ev.get("type") in ("recoil", "no_recoil"):
                        yield ev
        except OSError as e:
            print(f"tbot latency-recoil: {f}: {e}", file=sys.stderr)


def collect(paths: list[Path]) -> tuple[list[float], int, int]:
    """(delays in ms, commands with no shot, sources that had events). Overlapping ring dumps repeat events:
    each command (cmd_t_ns) is counted once."""
    seen: set = set()
    delays: list[float] = []
    missing = 0
    sources = 0
    for p in paths:
        had = False
        for ev in _events_in(p):
            key = (ev.get("cmd_t_ns"), ev.get("shot"))
            if key in seen:
                continue
            seen.add(key)
            had = True
            if ev["type"] == "no_recoil":
                missing += 1
            elif ev.get("trigger_state") not in (None, "parked"):
                continue  # the lever was not waiting at the pre-travel: not the delay being calibrated here
            elif ev.get("delay_ms") is not None and MIN_SANE_MS <= float(ev["delay_ms"]) <= MAX_SANE_MS:
                delays.append(float(ev["delay_ms"]))
        sources += had
    return delays, missing, sources


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot latency-recoil",
                                description="trigger delay from the recoil of recorded live shots")
    add_common_args(p)
    p.add_argument("paths", nargs="*", help="session directories, events.jsonl or telemetry .jsonl files "
                   "(default: every session under record.dir)")
    p.add_argument("--write", action="store_true", help="store the result in calib/latency.json")
    p.add_argument("--min-shots", type=int, default=5, help="shots needed before --write (default 5)")
    args = p.parse_args(argv)

    cfg, calib = load_from_args(args)
    if args.paths:
        paths = [Path(m) for pat in args.paths for m in (sorted(glob.glob(pat)) or [pat])]
    else:
        root = resolve_path(cfg.record.dir)
        paths = sorted(d for d in root.glob("*") if d.is_dir()) if root.exists() else []
    delays, missing, sources = collect(paths)
    atype = str(cfg.actuator.type)
    budget = calib.budget or LatencyBudget.from_config(cfg, calib.latency)
    lock = float(budget.lock_time_ms)
    now_break = float(budget.cmd_to_break_ms[atype])
    print(f"searched {len(paths)} path(s); {sources} with live-fire events")
    if missing:
        print(f"fire commands with NO shot: {missing}  (lever slip, safety, empty chamber, dud)")
    stats = summarise(delays)
    if stats is None:
        print("no recoil events found: they are logged by `tbot run --live` when a shot is really fired")
        return 1
    new_break = stats["median"] - lock
    print(f"shots               : {stats['n']}")
    print(f"command -> ignition : median {stats['median']:.1f} ms  sigma {stats['std']:.1f}  "
          f"range {stats['min']:.1f} .. {stats['max']:.1f}")
    print(f"lock time (config)  : {lock:.1f} ms")
    print(f"=> {atype} cmd_to_break: {new_break:.1f} ms   (calibration now: {now_break:.1f} ms, "
          f"{new_break - now_break:+.1f})")
    shift_m = abs(new_break - now_break) / 1000.0 * 20.0
    print(f"   a 20 m/s crossing clay moves {shift_m * 100:.0f} cm in that difference")
    if not args.write:
        print("not written (add --write)")
        return 0
    if stats["n"] < int(args.min_shots):
        print(f"tbot latency-recoil: only {stats['n']} shot(s); need {args.min_shots} to write "
              "(--min-shots to change)", file=sys.stderr)
        return 1
    out = calib.path("latency.json")
    if out.exists():
        out.with_name("latency.prev.json").write_bytes(out.read_bytes())
    # IMU sampling (2 ms) and shot-to-shot spread; never claim less jitter than the sampling allows
    std = max(float(stats["std"]), 1.5)
    update_json(out, {
        "created": now_iso(),
        "actuator": {atype: {
            "cmd_to_break_ms": {"mean": new_break, "std": std, "p50": new_break, "p95": stats["p95"] - lock,
                                "min": stats["min"] - lock, "max": stats["max"] - lock, "n": stats["n"]},
            "method": "recoil (IMU, live fire): median command -> ignition minus gun.lock_time_ms",
            "recoil": {"cmd_to_ignition_ms": stats, "lock_time_ms": lock, "no_shot_commands": missing},
            "created": now_iso(),
        }},
    })
    print(f"wrote {out}  (previous kept as latency.prev.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
