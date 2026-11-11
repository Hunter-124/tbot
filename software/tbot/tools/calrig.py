"""`tbot calrig {info,home,jog,run,estop,analyze}` — CalRig B yaw table for IMU calibration (docs/CALRIG.md).

Real hardware: Arduino Uno + CNC shield over USB serial (docs/CALRIG_PROTOCOL.md). `--sim` runs the whole stack
against an in-process FakeRig and synthetic IMU with no hardware (and no UNLOADED prompt). `home`, `jog` and `run`
refuse to energise the motor until the operator types UNLOADED. Ctrl-C = STOP, then EN 0.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from tbot.calrig import protocol as P
from tbot.calrig.device import DeviceError, RigDevice, RigFault, SerialTransport
from tbot.calrig.procedures import PROCEDURES, make_procedure, procedure_help
from tbot.calrig.recorder import ALL_POSES
from tbot.calrig.session import (HomeError, RigSession, SetupError, confirm_unloaded, describe_imu,
                                 neutral_calibration)
from tbot.calrig.sim import make_sim, parse_sim_opts
from tbot.calrig.timebase import RealClock
from tbot.config import load_from_args

__all__ = ["main"]

POSE_HELP = {
    "upright": "gun upright, rib up (normal)",
    "left_side_down": "gun rolled 90 deg onto its LEFT side",
    "right_side_down": "gun rolled 90 deg onto its RIGHT side",
    "inverted": "gun upside down (rib down)",
    "muzzle_up": "gun standing on its butt, muzzle up (extra pose)",
    "muzzle_down": "gun standing on its muzzle, muzzle down (extra pose)",
}


# ------------------------------------------------------------------------------------------ connecting
def _open_real_device(cfg, port: str, baud: int) -> RigDevice:
    """Patch point for tests: the real serial link."""
    clock = RealClock()
    geometry = P.Geometry(cfg.calrig.microsteps, cfg.calrig.steps_rev, cfg.calrig.ratio)
    return RigDevice(SerialTransport(port, baud, clock), geometry, clock, threaded=True)


def _open_session(args, cfg, calib, pose: str, want_imu: bool) -> tuple[RigSession, object | None]:
    """Connect (real or --sim) and return an unconfigured RigSession plus the simulation handle (or None)."""
    geometry = P.Geometry(cfg.calrig.microsteps, cfg.calrig.steps_rev, cfg.calrig.ratio)
    if args.sim:
        bundle = make_sim(cfg.calrig, cfg.imu, pose=pose, opts=parse_sim_opts(args.sim_opt))
        dev = RigDevice(bundle.rig, geometry, bundle.clock, threaded=False)
        want_imu = want_imu and not getattr(args, "no_imu", False)
        imu = bundle.imu if want_imu else None
        imu_meta = describe_imu(cfg, calib, "sim-imu", bundle.imu.rate_hz) if want_imu else None
        truth = bundle.truth
    else:
        port = args.port or cfg.calrig.port
        dev = _open_real_device(cfg, port, args.baud or cfg.calrig.baud)
        imu, imu_meta, truth, bundle = None, None, None, None
        if want_imu:
            imu, imu_meta = _open_real_imu(cfg, calib, args)
    try:
        dev.connect()
    except BaseException:
        dev.close()
        raise
    session = RigSession(dev, cfg, imu=imu, imu_meta=imu_meta, sim_truth=truth, log=lambda m: print(m, flush=True))
    return session, bundle


def _open_real_imu(cfg, calib, args):
    if args.no_imu:
        return None, None
    from tbot.imu.mpu6050 import Mpu6050, is_raspberry_pi

    if not is_raspberry_pi():
        raise SetupError("no MPU-6050 reachable from this machine (not a Raspberry Pi): run the rig from the Pi, "
                         "or pass --no-imu to record the rig only")
    # Neutral calibration: Mpu6050 returns RAW chip-frame samples (no R_gun_imu, bias or time offset), stamped with
    # time.monotonic_ns() = the host clock the rig times are converted to. The main app's behaviour is unchanged.
    imu = Mpu6050(cfg, neutral_calibration(calib))
    return imu, describe_imu(cfg, calib, imu.name, imu.sample_rate_hz)


def _confirm(args) -> dict:
    """Operator confirmation: typed UNLOADED on real hardware, nothing to confirm with --sim."""
    if args.sim:
        return {"confirmed": False, "simulated": True, "method": "--sim: no hardware"}
    return confirm_unloaded()


def _arm(session: RigSession, args) -> dict:
    """Confirm, then enable the driver and start telemetry (home / jog)."""
    operator = _confirm(args)
    session.enable()
    session.start_streaming()
    return operator


# ------------------------------------------------------------------------------------------ sub-commands
def _cmd_info(args, cfg, calib) -> int:
    session, _ = _open_session(args, cfg, calib, "upright", want_imu=False)
    with session:
        info = session.setup()
        st = session.dev.stat()
        session.dev.sync_burst(10)
        fit = session.dev.sync.fit()
        print("firmware :", " ".join(f"{k}={v}" for k, v in info.items()))
        print(f"geometry : {session.ups:.4f} usteps/deg = {session.geometry.deg_per_ustep:.5f} deg/ustep "
              f"({cfg.calrig.microsteps} microsteps, {cfg.calrig.steps_rev} steps/rev, ratio {cfg.calrig.ratio:g})")
        print(f"limits   : soft +-{cfg.calrig.soft_limit_deg:g} deg, vmax {session.vmax_dps:.1f} deg/s "
              f"(cap {session.limits.vmax_cap_dps:g}), amax {session.amax_dps2:.1f} deg/s^2 "
              f"(cap {session.limits.amax_cap_dps2:g})")
        print(f"state    : {st.state} en={st.en} pos={st.pos} ({session.geometry.to_deg(st.pos):+.3f} deg) "
              f"home={st.home} estop_pin={st.estop_pin}")
        if fit is not None:
            print(f"clock    : {fit.n_total} pings, rtt median {fit.rtt_med_us:.0f} us (min {fit.rtt_min_us:.0f}), "
                  f"fit residual {fit.resid_us:.1f} us")
        session.shutdown_motion()  # leaves the driver disabled
    return 0


def _cmd_home(args, cfg, calib) -> int:
    session, _ = _open_session(args, cfg, calib, "upright", want_imu=False)
    with session:
        session.setup()
        _arm(session, args)
        out = session.home(direction=args.dir, rate_dps=args.rate, to_zero=not args.no_return)
        print(json.dumps(out))
    return 0


def _jog_help() -> str:
    return ("jog commands: +  -  (step)   +N -N (relative degrees)   to N (absolute)   step N   rate N (deg/s)   "
            "home   zero   stat   q")


def _jog_execute(session: RigSession, line: str, state: dict) -> bool:
    toks = line.split()
    if not toks:
        return True
    cmd = toks[0].lower()
    try:
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("h", "help", "?"):
            print(_jog_help())
        elif cmd == "stat":
            st = session.dev.stat()
            print(f"state={st.state} pos={st.pos} ({session.geometry.to_deg(st.pos):+.3f} deg) home={st.home} "
                  f"estop_pin={st.estop_pin}")
        elif cmd == "zero":
            session.dev.command(P.cmd_zero())
            print("position zeroed here")
        elif cmd == "home":
            print(json.dumps(session.home()))
        elif cmd == "step" and len(toks) == 2:
            state["step"] = abs(float(toks[1]))
        elif cmd == "rate" and len(toks) == 2:
            state["rate"] = float(toks[1])
            session.set_rates(state["rate"], session.limits.amax_dps2)
        elif cmd == "to" and len(toks) == 2:
            session.move_to(float(toks[1]))
        elif cmd in ("+", "-"):
            session.move_by(state["step"] if cmd == "+" else -state["step"])
        else:
            session.move_by(float(cmd))
    except (ValueError, DeviceError) as exc:
        print(f"refused: {exc}")
    return True


def _cmd_jog(args, cfg, calib) -> int:
    session, _ = _open_session(args, cfg, calib, "upright", want_imu=False)
    with session:
        session.setup()
        _arm(session, args)
        state = {"step": args.step, "rate": args.rate}
        session.set_rates(args.rate, session.limits.amax_dps2)
        if args.deg is not None or args.to is not None:
            if args.to is not None:
                session.move_to(args.to)
            else:
                session.move_by(args.deg)
            st = session.dev.stat()
            print(f"now at {session.geometry.to_deg(st.pos):+.3f} deg")
            return 0
        print(_jog_help())
        while True:
            session.service()
            try:
                line = input("jog> ")
            except EOFError:
                break
            if not _jog_execute(session, line, state):
                break
    return 0


def _cmd_run(args, cfg, calib) -> int:
    if args.list or not args.procedure:
        print("procedures:\n" + procedure_help())
        return 0
    params = {}
    for item in args.param:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        params[key.strip()] = val
    proc = make_procedure(args.procedure, params)
    session, _ = _open_session(args, cfg, calib, args.pose, want_imu=True)
    with session:
        session.setup()
        proc.validate(session.limits)
        est = proc.duration_estimate_s(session.limits) + (60.0 if proc.needs_home and not args.no_home else 0.0)
        print(f"procedure {proc.name} {proc.params()}  pose {args.pose} ({POSE_HELP[args.pose]})")
        print(f"estimated duration {est:.0f} s" + ("" if not args.sim else " (simulated time)"))
        if not args.sim:
            print(f"Put the gun in pose '{args.pose}' with the muzzle safe and the swing arc clear.")
            if proc.needs_home and not args.no_home:
                print("The table will first seek the home magnet at "
                      f"{cfg.calrig.home_rate_dps:g} deg/s (zero = the hall edge).")
        operator = _confirm(args)  # run() enables the driver itself, after the IMU is up
        root = Path(args.out) if args.out else None
        run_dir = session.run(proc, args.pose, operator, home_first=not args.no_home, root=root)
    print(f"run recorded: {run_dir}")
    print(f"analyse with: tbot calrig analyze {run_dir}")
    return 0


def _cmd_estop(args, cfg, calib) -> int:
    session, _ = _open_session(args, cfg, calib, "upright", want_imu=False)
    with session:
        dev = session.dev
        r1 = dev.try_command(P.cmd_estop())
        r2 = dev.try_command(P.cmd_en(False))
        print(f"ESTOP -> {r1.raw if r1 else 'no reply'}; EN 0 -> {r2.raw if r2 else 'no reply'}")
        print("The firmware latches the e-stop until RESET with the A0 input released. Opening the serial port "
              "also resets an Uno, which boots with the driver disabled; for a real emergency cut the motor power.")
    return 0 if r1 is not None else 1


def _cmd_analyze(args, cfg, calib) -> int:
    try:
        from tbot.calrig import analysis
    except ModuleNotFoundError as exc:
        if exc.name == "tbot.calrig.analysis":
            print("error: tbot.calrig.analysis is not available in this checkout")
            return 1
        raise
    dirs = [Path(d) for d in args.run_dir]
    for d in dirs:
        if not (d / "meta.json").exists():
            print(f"error: {d} is not a run directory (no meta.json)")
            return 1
    results = []
    for d in dirs:
        result = analysis.analyze_run(d)
        report = analysis.write_report(result, d)
        print(f"{d.name}: verdict {result.get('verdict')}  report: {report}")
        results.append(result)
    # one run -> its own estimates; several runs (the four poses) -> the combined axis map, bias, scale, offset
    final = results[0] if len(results) == 1 else analysis.combine_runs(results)
    if len(results) > 1:
        print(f"combined {len(results)} runs: verdict {final.get('verdict')}")
    proposal = analysis.propose_calib(final)
    notes = {k: v for k, v in proposal.items() if k.startswith("_")}
    patch = {k: v for k, v in proposal.items() if not k.startswith("_")}
    print("proposed calib/imu.json keys:")
    print(json.dumps(patch, indent=2, default=str) if patch else "  (none: see the report for why)")
    for k, v in notes.items():
        print(f"  note {k}: {v}")
    if args.apply:
        if not patch:
            print("nothing to apply")
            return 1
        out = analysis.apply_to_calib(proposal, cfg.paths.calib_dir, dry_run=False)
        print(f"written to {out['path']}")
    elif patch:
        print("dry run: nothing written; add --apply to merge these keys into calib/imu.json")
    return 0


# ------------------------------------------------------------------------------------------ parser / main
def _common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("link")
    g.add_argument("--port", default=None, help="serial port (default: calrig.port)")
    g.add_argument("--baud", type=int, default=None, help="baud rate (default: calrig.baud)")
    g.add_argument("--sim", action="store_true", help="simulated rig + IMU in-process: no hardware, no prompt")
    g.add_argument("--sim-opt", action="append", default=[], metavar="KEY=VALUE",
                   help="simulation truth, e.g. gyro_scale=0.02,0,0 imu_offset_ms=12 gyro_bias_dps=0.3,0,0 "
                        "mount_err_deg=2,0,0 start_deg=-12 seed=3 (repeatable)")
    c = parser.add_argument_group("configuration")
    c.add_argument("--config", action="append", default=[], metavar="TOML", help="override file(s) (repeatable)")
    c.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one key (repeatable)")
    c.add_argument("--calib-dir", default=None, help="calibration directory (default: paths.calib_dir)")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tbot calrig",
        description="CalRig B motorised yaw table: IMU calibration with the gun UNLOADED (docs/CALRIG.md).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = _common(sub.add_parser("info", help="connect, print firmware INFO / state / clock round trip; motor stays off"))
    p.set_defaults(func=_cmd_info)

    p = _common(sub.add_parser("home", help="seek the hall edge and zero there (asks UNLOADED)"))
    p.add_argument("--dir", type=int, choices=(1, -1), default=None, help="approach direction (default calrig.home_dir)")
    p.add_argument("--rate", type=float, default=None, help="seek rate deg/s (default calrig.home_rate_dps)")
    p.add_argument("--no-return", action="store_true", help="stay where the table stopped instead of moving onto the edge")
    p.set_defaults(func=_cmd_home)

    p = _common(sub.add_parser("jog", help="move the table by hand: --deg/--to once, or an interactive prompt (asks UNLOADED)"))
    p.add_argument("--deg", type=float, default=None, help="relative move in degrees, then exit")
    p.add_argument("--to", type=float, default=None, help="absolute move in degrees from the zero, then exit")
    p.add_argument("--step", type=float, default=5.0, help="interactive step for +/- (deg, default 5)")
    p.add_argument("--rate", type=float, default=10.0, help="jog rate deg/s (default 10)")
    p.set_defaults(func=_cmd_jog)

    p = _common(sub.add_parser(
        "run", help="record a calibration procedure into runs/calrig/<time>_<procedure>/ (asks UNLOADED)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="procedures:\n" + procedure_help()))
    p.add_argument("procedure", nargs="?", choices=sorted(PROCEDURES), help="procedure name")
    p.add_argument("--pose", choices=ALL_POSES, default="upright", help="how the gun lies on the rig (default upright)")
    p.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE",
                   help="procedure parameter, e.g. -p duration_s=60 -p repeats=3 -p rates_dps=10,20,30 (repeatable)")
    p.add_argument("--no-home", action="store_true",
                   help="do not home first: the power-up position is zero (opening the port resets the Uno)")
    p.add_argument("--no-imu", action="store_true", help="record the rig only (no MPU-6050 on this machine)")
    p.add_argument("--out", default=None, help="run root directory (default calrig.run_root)")
    p.add_argument("--list", action="store_true", help="list procedures and their parameters")
    p.set_defaults(func=_cmd_run)

    p = _common(sub.add_parser("estop", help="send ESTOP then EN 0 (cut the motor power for a real emergency)"))
    p.set_defaults(func=_cmd_estop)

    p = sub.add_parser("analyze", help="analyse run directories -> report.json / report.md and a calib/imu.json proposal")
    p.add_argument("run_dir", nargs="+", help="one run, or several poses to combine into the axis map")
    p.add_argument("--apply", action="store_true", help="write the proposal into calib/imu.json (default: dry run)")
    p.add_argument("--config", action="append", default=[], metavar="TOML", help="override file(s) (repeatable)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one key (repeatable)")
    p.add_argument("--calib-dir", default=None, help="calibration directory (default: paths.calib_dir)")
    p.set_defaults(func=_cmd_analyze, sim=False)
    return parser


def _sigterm(_signum, _frame):  # pragma: no cover - signal path
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, calib = load_from_args(args)
    if not getattr(args, "sim", False) and args.cmd != "analyze":
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except ValueError:  # not the main thread
            pass
    try:
        return int(args.func(args, cfg, calib) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted: STOP sent, driver disabled", file=sys.stderr)
        return 130
    except (SetupError, HomeError, RigFault, DeviceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
