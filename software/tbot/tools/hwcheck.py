"""`tbot hwcheck` — is every fitted device actually working, and if not, what do I type to fix it.

One row per subsystem (platform, boot config, camera, IMU, GPIO, actuator, power/thermal, calibration)
with a PASS / WARN / FAIL / SKIP verdict and a one-line fix hint. SKIP means "this cannot be answered
here" (no Pi, no picamera2, sim backend) and never fails the run; the exit code is 1 only if something
that *is* fitted is broken.

Nothing in here ever moves the trigger: the actuator check only reports whether the PWM channel /
H-bridge pins are claimable. The gun is UNLOADED for every bench session anyway.

Every check is a plain function `(cfg, calib, env) -> CheckResult` and every system access it makes
comes from the injected `CheckEnv`, so the whole tool is testable on a laptop against fake file trees.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import importlib.util
import json
import os
import platform as _platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from tbot.config import Calibration, Config, LatencyBudget, add_common_args, load_from_args, now_iso
from tbot.hw import gpio as gpio_mod
from tbot.hw.health import describe_throttled, read_health

__all__ = [
    "PASS", "WARN", "FAIL", "SKIP",
    "CheckResult", "CheckEnv", "CHECKS",
    "parse_config_txt", "section_in_force",
    "check_platform", "check_boot_config", "check_camera", "check_imu",
    "check_gpio", "check_actuator", "check_power", "check_calibration",
    "run_checks", "format_table", "main",
]

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_SEVERITY = {PASS: 0, SKIP: 1, WARN: 2, FAIL: 3}

#: Where the firmware config lives (new Bookworm layout first).
CONFIG_TXT_PATHS: tuple[str, ...] = ("/boot/firmware/config.txt", "/boot/config.txt")
#: GPIO18 = PWM0: the pin the MG996R signal is on (dtoverlay=pwm,pin=18,func=2).
SERVO_PWM_PIN = 18
SERVO_PWM_FUNC = 2
#: Capture/sample windows of the live checks.
CAMERA_SECONDS = 1.5
IMU_SECONDS = 0.5
#: Live-rate verdicts: below this fraction of the configured rate is a WARN.
RATE_WARN_FRACTION = 0.90
#: Mean frame level outside this band means the exposure is wrong (clipped black / white).
LEVEL_WARN_LO, LEVEL_WARN_HI = 10.0, 245.0
#: At-rest IMU verdicts.
G_MPS2 = 9.80665
ACCEL_TOLERANCE_FRAC = 0.05
GYRO_REST_LIMIT_DPS = 3.0
#: SoC temperature verdicts (Pi 3B throttles at 80 C, soft limit at 60 C on later boards).
TEMP_WARN_C, TEMP_FAIL_C = 70.0, 80.0
#: How long `--interactive` waits for the operator to press, then release, the arming button.
INTERACTIVE_TIMEOUT_S = 10.0
#: Minimum runtime versions (docs/ARCHITECTURE.md).
MIN_PYTHON = (3, 11)
MIN_NUMPY = (1, 24)
MIN_OPENCV = (4, 6)

#: The calibration artefacts and the tool that writes each one.
CALIB_FILES: tuple[tuple[str, str], ...] = (
    ("intrinsics.json", "tbot calib-intrinsics"),
    ("boresight.json", "tbot calib-boresight"),
    ("imu.json", "tbot calib-imu bias"),
    ("latency.json", "tbot latency-camera / tbot latency-actuator"),
    ("detector.json", "tbot calib-detector"),
    ("bench.json", "tbot bench"),
    ("mask.png", "tbot calib-detector --auto-mask <session>"),
)
#: Without these two the solver is aiming on nominal optics and a nominal bore line.
CALIB_REQUIRED = ("intrinsics.json", "boresight.json")


# --------------------------------------------------------------------------------------------- types
@dataclass
class CheckResult:
    """One row of the table: what was checked, the verdict, and what to type if it is not PASS."""

    name: str
    status: str
    summary: str
    hint: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "summary": self.summary,
                "hint": self.hint, "data": self.data}


@dataclass
class CheckEnv:
    """Every bit of the outside world a check touches, so tests can replace all of it."""

    config_txt_paths: Sequence[str | Path] = CONFIG_TXT_PATHS
    model_path: str | Path = gpio_mod.DEVICE_TREE_MODEL
    #: None = decide from the device-tree model; True/False forces it (tests, odd boards).
    on_pi: bool | None = None
    os_release_path: str | Path = "/etc/os-release"
    i2c_dev_path: str = "/dev/i2c-{bus}"
    gpiochip_path: str = "/dev/gpiochip{chip}"
    pwm_sysfs_root: str | Path = gpio_mod.PWM_SYSFS_ROOT
    health_reader: Callable[[], dict] = read_health
    gpio_factory: Callable[[Config], Any] = gpio_mod.open_gpio
    lgpio_available: Callable[[], bool] = lambda: importlib.util.find_spec("lgpio") is not None
    camera_factory: Callable[[Config, Calibration], Any] | None = None
    camera_info: Callable[[], Sequence[Any]] | None = None
    imu_factory: Callable[[Config, Calibration], Any] | None = None
    camera_seconds: float = CAMERA_SECONDS
    imu_seconds: float = IMU_SECONDS
    interactive: bool = False
    interactive_timeout_s: float = INTERACTIVE_TIMEOUT_S
    log: Callable[[str], None] = print


# --------------------------------------------------------------------------------------------- utils
def _read_text(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def _device_tree_model(env: CheckEnv) -> str:
    text = _read_text(env.model_path) or ""
    return text.replace("\x00", "").strip()


def _on_pi(env: CheckEnv) -> bool:
    if env.on_pi is not None:
        return bool(env.on_pi)
    return "raspberry pi" in _device_tree_model(env).lower()


def _os_pretty_name(env: CheckEnv) -> str:
    for line in (_read_text(env.os_release_path) or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def _version_tuple(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in str(text).split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _not_on_pi(name: str, env: CheckEnv, what: str) -> CheckResult | None:
    """SKIP row for a device that only exists on the Pi; None when we are on one (so the check runs)."""
    if _on_pi(env):
        return None
    return CheckResult(name, SKIP, f"not a Raspberry Pi: {what}",
                       "run `tbot hwcheck` on the Pi (ssh in; docs/PROTOTYPING.md)")


def _mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return _dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _first_line(text: object) -> str:
    return str(text).strip().splitlines()[0] if str(text).strip() else ""


def _worst(results: Iterable[CheckResult]) -> str:
    worst = PASS
    for r in results:
        if _SEVERITY.get(r.status, 0) > _SEVERITY[worst]:
            worst = r.status
    return worst


# --------------------------------------------------------------------------------------------- platform
def check_platform(cfg: Config, calib: Calibration | None = None, env: CheckEnv | None = None) -> CheckResult:
    """Board, OS and the three runtime versions every other module depends on."""
    env = env or CheckEnv()
    model = _device_tree_model(env)
    os_name = _os_pretty_name(env)
    py = _platform.python_version()
    data = {
        "model": model,
        "os": os_name,
        "python": py,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "machine": _platform.machine(),
        "is_raspberry_pi": _on_pi(env),
    }
    old = []
    if _version_tuple(py) < MIN_PYTHON:
        old.append(f"python {py} < {'.'.join(map(str, MIN_PYTHON))}")
    if _version_tuple(np.__version__) < MIN_NUMPY:
        old.append(f"numpy {np.__version__} < {'.'.join(map(str, MIN_NUMPY))}")
    if _version_tuple(cv2.__version__) < MIN_OPENCV:
        old.append(f"opencv {cv2.__version__} < {'.'.join(map(str, MIN_OPENCV))}")
    data["too_old"] = old
    summary = (f"{model or 'unknown board'} | {os_name or 'unknown OS'} | "
               f"python {py} numpy {np.__version__} opencv {cv2.__version__}")
    if old:
        return CheckResult("platform", FAIL, summary,
                           "runtime too old (" + ", ".join(old) + "): sudo apt install -y "
                           "python3-opencv python3-numpy (Pi) / pip install -U numpy opencv-python",
                           data)
    if not data["is_raspberry_pi"]:
        return CheckResult("platform", WARN, summary,
                           "not a Raspberry Pi: every hardware check will SKIP (prototyping host)", data)
    return CheckResult("platform", PASS, summary, "", data)


# --------------------------------------------------------------------------------------------- boot
@dataclass
class Setting:
    """One `key=value` line of config.txt, with where it was and whether the firmware applies it."""

    key: str            # "dtoverlay", "dtparam", "camera_auto_detect", ...
    value: str
    section: str        # the [filter] it sits under ("all" at the top of the file)
    lineno: int
    commented: bool
    in_force: bool      # uncommented and in a section this board matches

    @property
    def overlay(self) -> str:
        """Overlay name of a dtoverlay line ('pwm' for dtoverlay=pwm,pin=18,func=2)."""
        return self.value.split(",", 1)[0].strip().lower() if self.key == "dtoverlay" else ""

    @property
    def params(self) -> dict[str, str]:
        """Parameters of the line: dtoverlay's trailing k=v pairs, or every dtparam k=v."""
        parts = self.value.split(",")
        if self.key == "dtoverlay":
            parts = parts[1:]
        out: dict[str, str] = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip().lower()] = v.strip().lower()
            elif part.strip():
                out[part.strip().lower()] = "on"  # bare dtparam name = enabled
        return out


#: config.txt board filters we can evaluate; anything else ([EDID=..], [HDMI:1], serial) is assumed live.
MODEL_FILTERS = {
    "pi0": "zero", "pi0w": "zero", "pi02": "zero 2", "pi1": "raspberry pi model",
    "pi2": "raspberry pi 2", "pi3": "raspberry pi 3", "pi3+": "raspberry pi 3",
    "pi4": "raspberry pi 4", "pi400": "raspberry pi 400", "pi5": "raspberry pi 5",
    "cm1": "compute module", "cm3": "compute module 3", "cm3+": "compute module 3",
    "cm4": "compute module 4", "cm4s": "compute module 4s", "cm5": "compute module 5",
}


def section_in_force(section: str, model: str = "") -> bool:
    """Does a [filter] section apply to this board? Unknown filters count as live (deliberately loose)."""
    name = section.strip().lower()
    if name in ("", "all"):
        return True
    if name == "none":
        return False
    needle = MODEL_FILTERS.get(name)
    if needle is None:
        return True  # [EDID=...], [HDMI:1], [gpio4=1], serial numbers: not evaluable here
    if not model:
        return True  # board unknown: do not silently discard settings
    low = model.lower()
    if not low.startswith(needle) and needle not in low:
        return False
    if name.endswith("+"):
        return "+" in low or "plus" in low
    return True


def parse_config_txt(text: str, model: str = "") -> list[Setting]:
    """Every key=value line of a config.txt, commented ones included (flagged, never in force)."""
    settings: list[Setting] = []
    section = "all"
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        commented = line.startswith("#")
        if commented:
            line = line.lstrip("#").strip()
            if not line:
                continue
        if line.startswith("[") and line.endswith("]"):
            if not commented:
                section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        if not key or " " in key:
            continue  # prose in a comment, not a setting
        settings.append(Setting(
            key=key,
            value=value.strip(),
            section=section,
            lineno=lineno,
            commented=commented,
            in_force=not commented and section_in_force(section, model),
        ))
    return settings


def _find_overlay(settings: Sequence[Setting], name: str, params: dict[str, str]) -> tuple[bool, str]:
    """(satisfied, note): is `dtoverlay=name` with those params active anywhere?"""
    seen_elsewhere = ""
    for s in settings:
        if s.key != "dtoverlay" or s.overlay != name:
            continue
        have = s.params
        if any(have.get(k) != v for k, v in params.items()):
            if s.in_force:
                seen_elsewhere = f"line {s.lineno} has dtoverlay={s.value}"
            continue
        if s.in_force:
            return True, f"line {s.lineno}"
        seen_elsewhere = (f"commented out at line {s.lineno}" if s.commented
                          else f"only under [{s.section}] at line {s.lineno}")
    return False, seen_elsewhere


def _find_param(settings: Sequence[Setting], name: str, value: str) -> tuple[bool, str]:
    """(satisfied, note) for a dtparam: the last line in force that mentions it wins."""
    note = ""
    satisfied = False
    for s in settings:
        if s.key != "dtparam" or name not in s.params:
            continue
        have = s.params[name]
        if s.in_force:
            satisfied = have == value
            note = f"line {s.lineno}: dtparam={name}={have}" if not satisfied else f"line {s.lineno}"
        elif not satisfied and not note:
            note = (f"commented out at line {s.lineno}" if s.commented
                    else f"only under [{s.section}] at line {s.lineno}")
    return satisfied, note


def check_boot_config(cfg: Config, calib: Calibration | None = None,
                      env: CheckEnv | None = None) -> CheckResult:
    """The firmware overlays the camera, the IMU bus and the servo PWM all depend on."""
    env = env or CheckEnv()
    path = next((Path(p) for p in env.config_txt_paths if Path(p).is_file()), None)
    if path is None:
        where = " or ".join(str(p) for p in env.config_txt_paths)
        return CheckResult("boot", SKIP, f"no config.txt at {where}",
                           "run this on the Pi, or point --config-txt at the boot filesystem")
    text = _read_text(path)
    if text is None:
        return CheckResult("boot", FAIL, f"{path} is not readable",
                           f"sudo chmod a+r {path} (or run hwcheck as root)")
    model = _device_tree_model(env)
    settings = parse_config_txt(text, model)
    servo = str(cfg.actuator.type).lower() == "servo"

    required: list[tuple[str, bool, str]] = []  # (what, satisfied, note)
    ok, note = _find_overlay(settings, "ov9281", {})
    required.append(("dtoverlay=ov9281", ok, note))
    ok, note = _find_param(settings, "i2c_arm", "on")
    required.append(("dtparam=i2c_arm=on", ok, note))
    if servo:
        ok, note = _find_overlay(settings, "pwm", {"pin": str(SERVO_PWM_PIN), "func": str(SERVO_PWM_FUNC)})
        required.append((f"dtoverlay=pwm,pin={SERVO_PWM_PIN},func={SERVO_PWM_FUNC}", ok, note))
        # Analogue audio drives PWM0/PWM1: with it on, the servo channel is not ours.
        ok, note = _find_param(settings, "audio", "off")
        required.append(("dtparam=audio=off", ok, note))

    missing = [what for what, ok, _ in required if not ok]
    notes = {what: note for what, ok, note in required if not ok and note}
    warnings: list[str] = []
    auto = [s for s in settings if s.key == "camera_auto_detect" and s.in_force]
    if auto and auto[-1].value.strip() not in ("0", "off", "false"):
        warnings.append(f"camera_auto_detect={auto[-1].value} (line {auto[-1].lineno}) can claim the CSI port")
    baud = [s for s in settings if s.key == "dtparam" and "i2c_arm_baudrate" in s.params and s.in_force]
    if cfg.imu.enabled and float(cfg.imu.sample_rate_hz) > 250.0 and not baud:
        warnings.append(f"no dtparam=i2c_arm_baudrate: imu.sample_rate_hz={cfg.imu.sample_rate_hz:g} "
                        "wants 400 kHz I2C")

    data = {
        "path": str(path),
        "model": model,
        "required": [{"setting": w, "present": ok, "note": n} for w, ok, n in required],
        "missing": missing,
        "warnings": warnings,
        "actuator_type": cfg.actuator.type,
    }
    present = len(required) - len(missing)
    if missing:
        detail = "; ".join(f"{m} ({notes[m]})" if m in notes else m for m in missing)
        return CheckResult("boot", FAIL, f"{path}: {present}/{len(required)} settings, missing {detail}",
                           f"add {', '.join(missing)} to {path}, then reboot "
                           "(scripts/pi_setup.sh writes the whole block)", data)
    if warnings:
        return CheckResult("boot", WARN, f"{path}: {present}/{len(required)} settings; " + "; ".join(warnings),
                           f"review {path} (scripts/pi_setup.sh writes the recommended block), then reboot",
                           data)
    return CheckResult("boot", PASS, f"{path}: all {present} required settings in force", "", data)


# --------------------------------------------------------------------------------------------- camera
def _camera_info(env: CheckEnv) -> tuple[Sequence[Any] | None, str]:
    """(global_camera_info(), error) — error is a human sentence when picamera2 is unusable."""
    if env.camera_info is not None:
        return env.camera_info(), ""
    if importlib.util.find_spec("picamera2") is None:
        return None, "picamera2 is not installed (Pi only: sudo apt install -y python3-picamera2)"
    try:
        from picamera2 import Picamera2  # noqa: PLC0415 - Pi-only, imported lazily by contract

        return Picamera2.global_camera_info(), ""
    except Exception as exc:  # noqa: BLE001 - libcamera failures are reported, never raised
        return None, f"picamera2 failed to enumerate cameras: {_first_line(exc)}"


def check_camera(cfg: Config, calib: Calibration | None = None, env: CheckEnv | None = None) -> CheckResult:
    """Sensor present, frames flowing at the configured rate, and the exposure in a usable band."""
    env = env or CheckEnv()
    skip = _not_on_pi("camera", env, "no CSI camera on this host")
    if skip is not None:
        return skip
    info, error = _camera_info(env)
    if info is None:
        return CheckResult("camera", FAIL, error,
                           "sudo apt install -y python3-picamera2 && rpicam-hello --list-cameras")
    entries = [str(e) for e in info]
    if not entries:
        return CheckResult("camera", FAIL, "libcamera lists no camera",
                           "check dtoverlay=ov9281 + the CSI ribbon orientation, then "
                           "rpicam-hello --list-cameras", {"cameras": entries})
    is_ov9281 = any("ov9281" in e.lower() for e in entries)

    factory = env.camera_factory
    if factory is None:
        from tbot.camera.picam import PicamSource  # noqa: PLC0415 - Pi-only, imported lazily by contract

        factory = PicamSource
    try:
        source = factory(cfg, calib)
    except Exception as exc:  # noqa: BLE001 - a dead camera is a FAIL row, not a traceback
        return CheckResult("camera", FAIL, f"cannot open the camera: {_first_line(exc)}",
                           "rpicam-hello --list-cameras; check dtoverlay=ov9281 and the ribbon",
                           {"cameras": entries})

    frames = 0
    levels: list[float] = []
    seconds = max(0.05, float(env.camera_seconds))
    try:
        source.start()
        t0 = time.monotonic()
        deadline = t0 + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            frame = source.read(timeout_s=min(1.0, max(0.05, remaining)))
            if frame is None:
                continue
            frames += 1
            levels.append(float(np.mean(frame.image)))
        elapsed = time.monotonic() - t0
        dropped = int(getattr(source, "dropped", 0))
        mode = getattr(source, "sensor_mode", None)
        ae = getattr(source, "ae_status", None)  # auto-exposure verdict on the last frame (None: fixed exposure)
        link = None
        if frames == 0 and env.camera_factory is None:  # real camera, stream still requested: which layer?
            from tbot.camera import linkcheck  # noqa: PLC0415

            link = linkcheck.diagnose(window_s=1.0, source=source)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("camera", FAIL, f"capture failed: {_first_line(exc)}",
                           "rpicam-hello --list-cameras; dmesg | grep -i ov9281", {"cameras": entries})
    finally:
        try:
            source.stop()
        except Exception:  # noqa: BLE001 - already reporting, do not mask the real verdict
            pass

    fps = frames / elapsed if elapsed > 0 else 0.0
    mean_level = float(np.mean(levels)) if levels else float("nan")
    mode_text = _mode_text(mode)
    data = {
        "cameras": entries, "is_ov9281": is_ov9281, "frames": frames, "seconds": elapsed,
        "fps": fps, "nominal_fps": float(cfg.camera.fps), "dropped": dropped,
        "mean_level": None if not levels else mean_level, "sensor_mode": mode_text,
        "size": [int(cfg.camera.width), int(cfg.camera.height)],
    }
    summary = (f"{entries[0]} | {cfg.camera.width}x{cfg.camera.height} mode {mode_text} | "
               f"{frames} frames in {elapsed:.2f} s = {fps:.1f} fps (cfg {cfg.camera.fps:g}) | "
               f"dropped {dropped} | mean level {mean_level:.1f}")
    if frames == 0:
        if link is not None:
            data["link"] = link.as_dict()
            return CheckResult("camera", FAIL, f"no frames in {elapsed:.2f} s - {link.layer}: {link.message}",
                               "link: reseat the CSI ribbon at both ends; sensor: sudo reboot", data)
        return CheckResult("camera", FAIL, f"no frames in {elapsed:.2f} s",
                           "rpicam-hello -t 2000; check the ribbon and dtoverlay=ov9281", data)
    if not is_ov9281:
        return CheckResult("camera", WARN, summary,
                           "the detected sensor is not an OV9281: the optics/mono assumptions do not hold",
                           data)
    if fps < RATE_WARN_FRACTION * float(cfg.camera.fps):
        return CheckResult("camera", WARN, summary,
                           f"below {RATE_WARN_FRACTION:.0%} of camera.fps: lower camera.fps or the "
                           "resolution, or check for a throttled SoC (tbot hwcheck --only power)", data)
    black = getattr(source, "black_level_state", None)  # ok | fixed | failed | unknown (linkcheck.ensure_black_level)
    if black is not None:
        data["black_level"] = black
    if black == "failed":
        return CheckResult("camera", WARN, summary,
                           "the sensor's black level differs from the one the ISP subtracts and could not be set "
                           "(sudo i2ctransfer): dark parts of the image read 0. Run a 1280x800 capture once "
                           "(tbot snap) or check i2c-tools / passwordless sudo", data)
    if ae is not None:
        data["auto_exposure"] = {"state": ae.state, "exposure_us": ae.exposure_us, "gain": ae.gain,
                                 "level_dn": ae.level_dn, "clip_pct": ae.clip_pct, "error_stops": ae.error_stops}
        summary += f" | auto-exposure {ae.exposure_us:.0f} us x{ae.gain:.1f} ({ae.state})"
        if ae.state in ("too_bright", "too_dark"):
            return CheckResult("camera", WARN, summary, ae.message(), data)
    elif levels and (mean_level < LEVEL_WARN_LO or mean_level > LEVEL_WARN_HI):
        return CheckResult("camera", WARN, summary,
                           f"exposure out of band: set camera.exposure_us (now {cfg.camera.exposure_us:g}) "
                           "/ camera.analogue_gain for a mid-grey sky", data)
    return CheckResult("camera", PASS, summary, "", data)


def _mode_text(mode: Any) -> str:
    if mode is None:
        return "unknown"
    size = mode.get("size") if isinstance(mode, dict) else getattr(mode, "size", None)
    if size is None:
        return str(mode)
    fps = mode.get("fps") if isinstance(mode, dict) else getattr(mode, "fps", None)
    text = f"{int(size[0])}x{int(size[1])}"
    try:
        return f"{text}@{float(fps):g}" if fps else text
    except (TypeError, ValueError):
        return text


# --------------------------------------------------------------------------------------------- imu
def check_imu(cfg: Config, calib: Calibration | None = None, env: CheckEnv | None = None) -> CheckResult:
    """Bus present, WHO_AM_I right, and the stream at rest sane (rate, 1 g, no rotation)."""
    env = env or CheckEnv()
    if not bool(cfg.imu.enabled):
        return CheckResult("imu", SKIP, "imu.enabled = false", "--set imu.enabled=true to check the IMU")
    skip = _not_on_pi("imu", env, "no MPU-6050 bus on this host")
    if skip is not None:
        return skip
    dev = Path(env.i2c_dev_path.format(bus=int(cfg.imu.i2c_bus)))
    fix = (f"add dtparam=i2c_arm=on to config.txt, load i2c-dev at boot (sudo ./scripts/pi_setup.sh does both) "
           f"and reboot; {dev} needs the 'i2c' group (sudo usermod -aG i2c $USER)")
    if not dev.exists():
        return CheckResult("imu", FAIL, f"no {dev}", fix, {"device": str(dev)})
    if not os.access(dev, os.R_OK | os.W_OK):
        return CheckResult("imu", FAIL, f"{dev} is not readable/writable by this user", fix,
                           {"device": str(dev)})

    factory = env.imu_factory
    if factory is None:
        from tbot.imu.mpu6050 import Mpu6050  # noqa: PLC0415 - smbus2 is Pi-only, imported lazily

        factory = Mpu6050
    try:
        imu = factory(cfg, calib)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("imu", FAIL, f"cannot create the IMU reader: {_first_line(exc)}", fix,
                           {"device": str(dev)})
    samples: list[Any] = []
    try:
        imu.start()  # resets, programs and verifies WHO_AM_I
        t0 = time.monotonic()
        deadline = t0 + max(0.05, float(env.imu_seconds))
        while time.monotonic() < deadline:
            samples.extend(imu.read())
            time.sleep(0.01)
        samples.extend(imu.read())
        elapsed = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return CheckResult("imu", FAIL, _first_line(exc),
                           f"i2cdetect -y {cfg.imu.i2c_bus} must show 0x{int(cfg.imu.address):02x}; "
                           "check SDA/SCL/3V3/GND and the AD0 pin", {"device": str(dev)})
    finally:
        try:
            imu.stop()
        except Exception:  # noqa: BLE001
            pass

    n = len(samples)
    rate = n / elapsed if elapsed > 0 else 0.0
    data: dict[str, Any] = {
        "device": str(dev), "address": f"0x{int(cfg.imu.address):02x}", "samples": n,
        "seconds": elapsed, "rate_hz": rate, "nominal_rate_hz": float(cfg.imu.sample_rate_hz),
        "errors": int(getattr(imu, "errors", 0)), "overflows": int(getattr(imu, "overflows", 0)),
    }
    if n == 0:
        return CheckResult("imu", FAIL, f"the device answered but produced no samples in {elapsed:.2f} s",
                           "check the FIFO wiring/clock: i2cdetect -y "
                           f"{cfg.imu.i2c_bus}, then tbot hwcheck --only imu", data)
    accel = np.asarray([s.accel for s in samples], dtype=np.float64)
    gyro = np.asarray([s.gyro for s in samples], dtype=np.float64)
    accel_norm = float(np.mean(np.sqrt(np.sum(accel * accel, axis=1))))
    gyro_dps = float(np.mean(np.sqrt(np.sum(gyro * gyro, axis=1)))) * 180.0 / np.pi
    temps = [float(s.temp_c) for s in samples if s.temp_c == s.temp_c]  # drop NaN
    temp_c = float(np.mean(temps)) if temps else float("nan")
    data.update({"accel_norm_m_s2": accel_norm, "gyro_norm_dps": gyro_dps, "temp_c": temp_c})
    summary = (f"{n} samples in {elapsed:.2f} s = {rate:.0f} Hz (cfg {cfg.imu.sample_rate_hz:g}) | "
               f"|a| {accel_norm:.2f} m/s^2 | |w| {gyro_dps:.2f} deg/s | {temp_c:.1f} C")
    if rate < RATE_WARN_FRACTION * float(cfg.imu.sample_rate_hz):
        return CheckResult("imu", WARN, summary,
                           "sample rate low: raise the I2C clock (dtparam=i2c_arm_baudrate=400000) "
                           "or lower imu.sample_rate_hz", data)
    if abs(accel_norm - G_MPS2) > ACCEL_TOLERANCE_FRAC * G_MPS2:
        return CheckResult("imu", WARN, summary,
                           f"|a| is {accel_norm:.2f} m/s^2, not {G_MPS2:.2f}: hold the gun still, then "
                           "run tbot calib-imu bias", data)
    if gyro_dps > GYRO_REST_LIMIT_DPS:
        return CheckResult("imu", WARN, summary,
                           f"{gyro_dps:.1f} deg/s at rest: hold the gun still, then tbot calib-imu bias",
                           data)
    return CheckResult("imu", PASS, summary, "", data)


# --------------------------------------------------------------------------------------------- gpio
def _wait_level(gpio: Any, pin: int, want_pressed: bool, active_low: bool, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        level = int(gpio.read(pin))
        pressed = (level == 0) if active_low else (level == 1)
        if pressed == want_pressed:
            return True
        time.sleep(0.02)
    return False


def check_gpio(cfg: Config, calib: Calibration | None = None, env: CheckEnv | None = None) -> CheckResult:
    """The arming input: chip accessible, lgpio importable, and the button reading the right way round."""
    env = env or CheckEnv()
    skip = _not_on_pi("gpio", env, "no GPIO header on this host")
    if skip is not None:
        return skip
    chip = Path(env.gpiochip_path.format(chip=int(cfg.gpio.chip)))
    fix = "sudo usermod -aG gpio $USER, log in again (apt: python3-lgpio)"
    if not chip.exists():
        return CheckResult("gpio", FAIL, f"no {chip}", fix, {"chip": str(chip)})
    if not os.access(chip, os.R_OK | os.W_OK):
        return CheckResult("gpio", FAIL, f"{chip} is not readable/writable by this user", fix,
                           {"chip": str(chip)})
    if not env.lgpio_available():
        return CheckResult("gpio", FAIL, "the lgpio module is not installed",
                           "sudo apt install -y python3-lgpio", {"chip": str(chip)})

    try:
        gpio = env.gpio_factory(cfg)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("gpio", FAIL, f"cannot open {chip}: {_first_line(exc)}", fix, {"chip": str(chip)})
    pin = int(cfg.arming.pin)
    active_low = bool(cfg.arming.active_low)
    data: dict[str, Any] = {"chip": str(chip), "arming_pin": pin, "backend": type(gpio).__name__,
                            "active_low": active_low, "pull": str(cfg.arming.pull)}
    try:
        if isinstance(gpio, gpio_mod.SimGpio):
            return CheckResult("gpio", SKIP, "gpio.backend resolved to the sim backend: no real pins",
                               "--set gpio.backend=lgpio on the Pi", data)
        gpio.setup_input(pin, str(cfg.arming.pull))
        level = int(gpio.read(pin))
        pressed = (level == 0) if active_low else (level == 1)
        data.update({"level": level, "pressed": pressed})
        summary = (f"{chip} ok, lgpio present | arming BCM{pin} = {level} "
                   f"({'PRESSED' if pressed else 'released'})")
        if env.interactive:
            timeout = float(env.interactive_timeout_s)
            env.log(f"  gun UNLOADED. PRESS AND HOLD the arming button (BCM{pin}) within {timeout:g} s ...")
            if not _wait_level(gpio, pin, True, active_low, timeout):
                data["interactive"] = "no press detected"
                return CheckResult("gpio", FAIL, summary + " | no press detected",
                                   f"check the button between BCM{pin} and GND and the pull-{cfg.arming.pull} "
                                   "setting (pressed must read 0)", data)
            env.log("  press detected. Now RELEASE it ...")
            if not _wait_level(gpio, pin, False, active_low, timeout):
                data["interactive"] = "never released"
                return CheckResult("gpio", FAIL, summary + " | pressed but never released",
                                   f"the button is stuck closed / BCM{pin} is shorted to GND: the trigger "
                                   "would be armed permanently", data)
            data["interactive"] = "press and release detected"
            return CheckResult("gpio", PASS, summary + " | press and release both detected", "", data)
        if pressed:
            return CheckResult("gpio", WARN, summary,
                               "release the arming button (it reads pressed right now); "
                               "re-run with --interactive to test both levels", data)
        return CheckResult("gpio", PASS, summary, "", data)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("gpio", FAIL, f"GPIO access failed: {_first_line(exc)}", fix, data)
    finally:
        try:
            gpio.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------- actuator
def check_actuator(cfg: Config, calib: Calibration | None = None,
                   env: CheckEnv | None = None) -> CheckResult:
    """Can the trigger driver be claimed? Nothing moves here — use `tbot jog` for that."""
    env = env or CheckEnv()
    kind = str(cfg.actuator.type).lower()
    if kind in ("sim", "none"):
        return CheckResult("actuator", SKIP, f"actuator.type = {kind}: no trigger hardware fitted",
                           "--set actuator.type=servo|pq12 once the actuator is wired")
    if kind == "servo":
        return _check_servo(cfg, env)
    if kind == "pq12":
        return _check_pq12(cfg, env)
    return CheckResult("actuator", FAIL, f"unknown actuator.type {cfg.actuator.type!r}",
                       "actuator.type must be servo | pq12 | sim | none")


def _check_servo(cfg: Config, env: CheckEnv) -> CheckResult:
    s = cfg.actuator.servo
    skip = _not_on_pi("actuator", env, "no hardware PWM on this host")
    if skip is not None:
        skip.data = {"kind": "servo", "rest_us": float(s.rest_us), "fire_us": float(s.fire_us)}
        return skip
    root = Path(env.pwm_sysfs_root)
    chip_dir = root / f"pwmchip{int(s.pwm_chip)}"
    channel_dir = chip_dir / f"pwm{int(s.pwm_channel)}"
    fix = (f"add dtoverlay=pwm,pin={SERVO_PWM_PIN},func={SERVO_PWM_FUNC} and dtparam=audio=off to "
           "config.txt, reboot, and make /sys/class/pwm group-writable (udev rule in scripts/pi_setup.sh)")
    data = {"kind": "servo", "chip": str(chip_dir), "channel": int(s.pwm_channel),
            "rest_us": float(s.rest_us), "fire_us": float(s.fire_us), "frame_hz": float(s.frame_hz)}
    if not chip_dir.is_dir():
        return CheckResult("actuator", FAIL, f"no {chip_dir}", fix, data)
    summary = (f"servo on pwmchip{int(s.pwm_chip)}/pwm{int(s.pwm_channel)} @ {s.frame_hz:g} Hz, "
               f"rest {s.rest_us:g} us / fire {s.fire_us:g} us")
    if channel_dir.is_dir():
        writable = os.access(channel_dir / "duty_cycle", os.W_OK)
        data["exported"] = True
        data["writable"] = writable
        if not writable:
            return CheckResult("actuator", FAIL, summary + " | exported but duty_cycle is not writable",
                               fix, data)
        return CheckResult("actuator", PASS, summary + " | channel exported and writable "
                                                      "(not moved by hwcheck)", "", data)
    data["exported"] = False
    export = chip_dir / "export"
    if not os.access(export, os.W_OK):
        data["writable"] = False
        return CheckResult("actuator", FAIL, summary + f" | {export} is not writable", fix, data)
    data["writable"] = True
    return CheckResult("actuator", PASS, summary + " | channel exportable (not moved by hwcheck)", "", data)


def _check_pq12(cfg: Config, env: CheckEnv) -> CheckResult:
    p = cfg.actuator.pq12
    data = {"kind": "pq12", "pin_extend": int(p.pin_extend), "pin_retract": int(p.pin_retract),
            "extend_ms": float(p.extend_ms), "retract_ms": float(p.retract_ms)}
    skip = _not_on_pi("actuator", env, "no GPIO header on this host")
    if skip is not None:
        skip.data = data
        return skip
    chip = Path(env.gpiochip_path.format(chip=int(cfg.gpio.chip)))
    fix = "sudo usermod -aG gpio $USER, log in again; check the H-bridge wiring"
    if not chip.exists():
        return CheckResult("actuator", FAIL, f"no {chip}", fix, data)
    try:
        gpio = env.gpio_factory(cfg)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("actuator", FAIL, f"cannot open {chip}: {_first_line(exc)}", fix, data)
    try:
        if isinstance(gpio, gpio_mod.SimGpio):
            return CheckResult("actuator", SKIP, "gpio.backend resolved to the sim backend: no real pins",
                               "--set gpio.backend=lgpio on the Pi", data)
        gpio.setup_output(int(p.pin_extend), 0)
        gpio.setup_output(int(p.pin_retract), 0)
        levels = [int(gpio.read(int(p.pin_extend))), int(gpio.read(int(p.pin_retract)))]
        data["levels"] = levels
        summary = (f"pq12 H-bridge: extend BCM{p.pin_extend}, retract BCM{p.pin_retract} claimed, "
                   f"driven low (levels {levels[0]},{levels[1]})")
        if any(levels):
            return CheckResult("actuator", FAIL, summary,
                               "a driver input will not go low: check the H-bridge enable/wiring before "
                               "powering the actuator", data)
        return CheckResult("actuator", PASS, summary + " — not moved by hwcheck", "", data)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("actuator", FAIL, f"cannot claim the H-bridge pins: {_first_line(exc)}", fix, data)
    finally:
        try:
            gpio.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------- power
def check_power(cfg: Config | None = None, calib: Calibration | None = None,
                env: CheckEnv | None = None) -> CheckResult:
    """Pi supply and SoC temperature; firmware flags do not measure the separate 7.0 V servo rail."""
    env = env or CheckEnv()
    health = env.health_reader() or {}
    temp = health.get("cpu_temp_c")
    throttled = health.get("throttled")
    freq = health.get("arm_freq_mhz")
    if temp is None and throttled is None and freq is None:
        return CheckResult("power", SKIP, "no Pi firmware health interfaces on this host",
                           "run this on the Pi (vcgencmd / the firmware sysfs node)", dict(health))
    parts = []
    if temp is not None:
        parts.append(f"SoC {temp:.1f} C")
    if freq is not None:
        parts.append(f"ARM {freq:.0f} MHz")
    parts.append(describe_throttled(throttled) if throttled else "throttle flags unavailable")
    summary = " | ".join(parts)
    data = dict(health)

    if throttled and throttled.get("under_voltage_now"):
        return CheckResult("power", FAIL, summary,
                           "UNDER-VOLTAGE NOW: raise the Pi XL4015 output to 5.1 V, shorten/thicken the "
                           "5 V leads, check the 3S pack under load", data)
    if temp is not None and temp > TEMP_FAIL_C:
        return CheckResult("power", FAIL, summary,
                           f"SoC above {TEMP_FAIL_C:g} C: the Pi is throttling hard — add airflow to the "
                           "pod, it sits under a hot barrel", data)
    if throttled and (throttled.get("throttled_now") or throttled.get("arm_frequency_capped_now")
                      or throttled.get("soft_temp_limit_now")):
        return CheckResult("power", WARN, summary,
                           "the firmware is capping the SoC right now: fix cooling/supply before "
                           "trusting frame timing", data)
    if temp is not None and temp > TEMP_WARN_C:
        return CheckResult("power", WARN, summary,
                           f"SoC above {TEMP_WARN_C:g} C: add airflow/a heatsink to the pod", data)
    if throttled and throttled.get("occurred"):
        return CheckResult("power", WARN, summary,
                           "it happened since boot (the flags latch): check the Pi supply; inspect the separate "
                           "servo supply independently; clear with a reboot", data)
    return CheckResult("power", PASS, summary, "", data)


# --------------------------------------------------------------------------------------------- calib
def check_calibration(cfg: Config, calib: Calibration | None = None,
                      env: CheckEnv | None = None) -> CheckResult:
    """Which calibration artefacts exist, when they were made, and what the latency budget rests on."""
    env = env or CheckEnv()
    directory = Path(calib.dir) if calib is not None else Path(cfg.paths.calib_dir)
    files: dict[str, Any] = {}
    present: list[str] = []
    for name, tool in CALIB_FILES:
        path = directory / name
        exists = path.is_file()
        created = ""
        if exists:
            present.append(name)
            if path.suffix == ".json":
                try:
                    payload = json.loads(path.read_text())
                    created = str(payload.get("created", "")) if isinstance(payload, dict) else ""
                except (OSError, ValueError):
                    created = ""
            created = created or _mtime_iso(path)
        files[name] = {"exists": exists, "created": created, "tool": tool, "path": str(path)}
    budget = calib.budget if calib is not None and calib.budget is not None else \
        LatencyBudget.from_config(cfg, calib.latency if calib is not None else None)
    sources = dict(budget.sources)
    measured = sorted(k for k, v in sources.items() if v == "measured")
    missing_required = [n for n in CALIB_REQUIRED if not files[n]["exists"]]
    data = {"dir": str(directory), "files": files, "latency_sources": sources, "measured": measured}
    summary = (f"{directory}: {len(present)}/{len(CALIB_FILES)} artefacts "
               f"({', '.join(present) if present else 'none'}) | latency measured: "
               f"{', '.join(measured) if measured else 'none (config defaults)'}")
    if missing_required:
        tools = ", ".join(files[n]["tool"] for n in missing_required)
        return CheckResult("calib", WARN, summary,
                           f"missing {', '.join(missing_required)}: run {tools}", data)
    if not measured:
        return CheckResult("calib", WARN, summary,
                           "the whole latency budget is config defaults: run tbot latency-camera and "
                           "tbot latency-actuator", data)
    return CheckResult("calib", PASS, summary, "", data)


# --------------------------------------------------------------------------------------------- runner
CHECKS: dict[str, Callable[[Config, Calibration | None, CheckEnv], CheckResult]] = {
    "platform": check_platform,
    "boot": check_boot_config,
    "camera": check_camera,
    "imu": check_imu,
    "gpio": check_gpio,
    "actuator": check_actuator,
    "power": check_power,
    "calib": check_calibration,
}


def run_checks(cfg: Config, calib: Calibration | None = None, env: CheckEnv | None = None,
               only: Sequence[str] | None = None) -> list[CheckResult]:
    """Run the selected checks in order. A check that blows up becomes a FAIL row, never a traceback."""
    env = env or CheckEnv()
    names = list(only) if only else list(CHECKS)
    results: list[CheckResult] = []
    for name in names:
        fn = CHECKS[name]
        try:
            result = fn(cfg, calib, env)
        except Exception as exc:  # noqa: BLE001 - one broken probe must not hide the other rows
            result = CheckResult(name, FAIL, f"check raised {type(exc).__name__}: {_first_line(exc)}",
                                 "this is a bug in tbot: tbot hwcheck --json and file it")
        if not result.name:
            result.name = name
        results.append(result)
    return results


def format_table(results: Sequence[CheckResult], header: str = "") -> str:
    width = max([len(r.name) for r in results] + [6])
    lines = []
    if header:
        lines += [header, ""]
    lines.append(f"  {'check':<{width}}  {'status':<6}  detail")
    lines.append(f"  {'-' * width}  {'-' * 6}  {'-' * 60}")
    for r in results:
        lines.append(f"  {r.name:<{width}}  {r.status:<6}  {r.summary}")
        if r.hint and r.status != PASS:
            lines.append(f"  {'':<{width}}  {'->':<6}  {r.hint}")
    counts = {s: sum(1 for r in results if r.status == s) for s in (PASS, WARN, FAIL, SKIP)}
    lines.append("")
    lines.append("  " + ", ".join(f"{counts[s]} {s}" for s in (PASS, WARN, FAIL, SKIP)))
    worst = _worst(results)
    verdict = {
        FAIL: "something fitted is broken: fix the FAIL rows above before a live session",
        WARN: "usable, but read the WARN rows",
        SKIP: "nothing to check here: run it on the Pi",
        PASS: "every fitted device answered",
    }[worst]
    lines.append(f"  verdict: {worst} — {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, env: CheckEnv | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot hwcheck",
        description="Check every fitted device (camera, IMU, arming switch, actuator, power) and say "
                    "what to type when one is broken. Never moves the trigger.",
    )
    add_common_args(parser)
    parser.add_argument("--json", dest="as_json", action="store_true", help="machine-readable output")
    parser.add_argument("--only", default=None,
                        help=f"comma separated subset of: {','.join(CHECKS)}")
    parser.add_argument("--interactive", action="store_true",
                        help="ask the operator to press and release the arming button and verify both levels")
    parser.add_argument("--config-txt", default=None, help="firmware config.txt to check (default: "
                                                           + " then ".join(CONFIG_TXT_PATHS))
    parser.add_argument("--camera-seconds", type=float, default=CAMERA_SECONDS,
                        help=f"camera capture window (default {CAMERA_SECONDS:g} s)")
    parser.add_argument("--imu-seconds", type=float, default=IMU_SECONDS,
                        help=f"IMU sampling window (default {IMU_SECONDS:g} s)")
    args = parser.parse_args(argv)

    names = list(CHECKS)
    if args.only:
        names = [n.strip().lower() for n in str(args.only).split(",") if n.strip()]
        unknown = [n for n in names if n not in CHECKS]
        if unknown:
            parser.error(f"unknown check(s) {', '.join(unknown)}; known: {', '.join(CHECKS)}")

    cfg, calib = load_from_args(args)
    env = env or CheckEnv()
    replace: dict[str, Any] = {
        "interactive": bool(args.interactive),
        "camera_seconds": float(args.camera_seconds),
        "imu_seconds": float(args.imu_seconds),
    }
    if args.config_txt:
        replace["config_txt_paths"] = (args.config_txt,)
    if args.as_json:
        replace["log"] = lambda msg: print(msg, file=sys.stderr)
    env = dataclasses.replace(env, **replace)

    results = run_checks(cfg, calib, env, names)
    worst = _worst(results)
    if args.as_json:
        print(json.dumps({
            "created": now_iso(),
            "status": worst,
            "actuator_type": cfg.actuator.type,
            "calib_dir": str(calib.dir),
            "checks": [r.to_dict() for r in results],
        }, indent=2, default=str))
    else:
        header = (f"tbot hwcheck   actuator={cfg.actuator.type}   calib {calib.dir}   "
                  f"gun UNLOADED for every bench session")
        print(format_table(results, header))
    return 1 if worst == FAIL else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
