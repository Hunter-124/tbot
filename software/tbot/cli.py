"""`tbot <command> ...` — dispatches to each tool module's main(argv) (imported lazily)."""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("tbot.app.run", "live pipeline: camera -> detect -> track -> fire solver (dry run unless fire.live)"),
    "sim": ("tbot.tools.sim_eval", "closed-loop evaluation on synthetic scenarios"),
    "record": ("tbot.app.record", "record a session (frames + timestamps + IMU) for replay"),
    "board": ("tbot.tools.make_board", "write a printable calibration chessboard"),
    "focus": ("tbot.tools.focus", "headless focus and exposure calibration"),
    "calib-intrinsics": ("tbot.tools.calib_intrinsics", "fisheye intrinsic calibration -> calib/intrinsics.json"),
    "calib-boresight": ("tbot.tools.calib_boresight", "camera-to-bore alignment -> calib/boresight.json"),
    "calib-imu": ("tbot.tools.calib_imu", "IMU bias / axis check / camera-IMU time sync -> calib/imu.json"),
    "calib-detector": ("tbot.tools.calib_detector", "detector tuning + barrel/static mask -> calib/detector.json, mask.png"),
    "latency-camera": ("tbot.tools.latency_camera", "LED test: exposure -> userspace latency, timestamp offset"),
    "latency-actuator": ("tbot.tools.latency_actuator", "fire command -> trigger break latency (arming held)"),
    "latency-mic": ("tbot.tools.latency_mic", "fire command -> dry-fire click on a microphone, one trial per recock"),
    "latency-recoil": ("tbot.tools.latency_recoil", "trigger delay from the recoil of recorded live shots -> calib/latency.json"),
    "latency-report": ("tbot.tools.latency_report", "print the latency budget (defaults vs measured)"),
    "bench": ("tbot.tools.bench", "per-stage processing time at the configured resolution"),
    "hwcheck": ("tbot.tools.hwcheck", "check every fitted device: camera, IMU, arming switch, PWM/actuator, power, temperature"),
    "jog": ("tbot.tools.jog", "bench setup of the trigger actuator (servo pulse widths / PQ12 pulses); arming button held"),
    "hitl": ("tbot.tools.hitl", "hardware-in-the-loop bench: camera films a monitor, every delay photon -> trigger break"),
    "doctor": ("tbot.tools.doctor", "offline config, calibration and layout consistency checks (no hardware)"),
    "sensitivity": ("tbot.tools.sensitivity", "error budget: miss vs latency, range, PWM jitter for the MG996R build"),
    "tune": ("tbot.tools.tune", "offline parameter search over the synthetic closed-loop evaluation"),
    "explain": ("tbot.tools.explain", "why did (or didn't) it fire: per-frame decision funnel over a run"),
    "calrig": ("tbot.tools.calrig", "CalRig B yaw table: info, home, jog, run <procedure>, estop, analyze (unloaded gun only)"),
}


def _usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["usage: tbot <command> [options]   (tbot <command> -h for help)", "", "commands:"]
    lines += [f"  {c:<{width}}  {d}" for c, (_, d) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"tbot: unknown command {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[cmd][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
