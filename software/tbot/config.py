"""Typed configuration + calibration loading (contract: docs/ARCHITECTURE.md).

config/default.toml holds every key. Override files (--config) and --set key.path=value only change values;
unknown keys are errors so typos never silently fall back to defaults.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _dt
import json
import os
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------------------------- schema
@dataclass
class CameraConfig:
    source: str
    width: int
    height: int
    fps: float
    exposure_us: float
    analogue_gain: float
    auto_exposure: bool
    rotate_180: bool
    emulate_mono: str
    buffer_count: int
    stall_s: float = 0.5
    reopen_timeout_s: float = 5.0


@dataclass
class OpticsConfig:
    model: str
    focal_length_mm: float
    pixel_pitch_um: float
    sensor_width_px: int


@dataclass
class MountConfig:
    cam_pos_mm: list
    cam_rpy_deg: list
    lens_tip_x_mm: float
    muzzle_pos_mm: list
    barrel_axis_z_mm: float
    barrel_od_mm: list
    barrel_mask_margin_px: int
    clay_diameter_m: float


@dataclass
class DetectorConfig:
    mode: str
    polarity: str
    bg_downscale: int
    bg_kernel: int
    k_sigma: float
    min_contrast: float
    noise_every: int
    min_area_px: int
    max_area_px: int
    max_aspect: float
    max_detections: int
    hsv_lo: list
    hsv_hi: list
    color_combine: str
    debug: bool
    psf_px: float
    matched_sigma_px: float = 0.5
    grow_k_sigma: float = 3.0
    max_peak_fraction: float = 0.85


@dataclass
class TrackerConfig:
    gate_sigma: float
    gate_min_deg: float
    confirm_hits: int
    max_misses: int
    jerk_psd_deg: float
    meas_sigma_px: float
    range_prior_m: float
    range_prior_sigma_m: float
    size_range_sigma_frac: float
    size_range_bias_frac: float
    max_tracks: int
    model: str = "angular"
    jerk_psd_mps3: float = 50.0


@dataclass
class ImuConfig:
    enabled: bool
    i2c_bus: int
    address: int
    sample_rate_hz: int
    gyro_range_dps: int
    accel_range_g: int
    dlpf_hz: int
    poll_hz: int
    R_gun_imu: list


@dataclass
class AttitudeConfig:
    accel_gain: float
    accel_gate_g: float
    history_s: float
    init_still_s: float


@dataclass
class GunConfig:
    lock_time_ms: float
    barrel_time_ms: float


@dataclass
class ShotConfig:
    muzzle_velocity_mps: float
    pellet_diameter_mm: float
    pellet_density_kg_m3: float
    air_density_kg_m3: float
    speed_of_sound_mps: float
    pattern_radius_ref_m: float
    pattern_ref_range_m: float
    pattern_min_radius_m: float
    max_range_m: float


@dataclass
class FireConfig:
    live: bool
    min_hit_probability: float
    min_track_hits: int
    horizon_frames: int
    max_shots_per_arm: int
    min_range_m: float
    max_range_m: float
    min_elevation_deg: float
    watchdog_ms: float
    gun_accel_sigma_frac: float = 0.3
    min_shot_interval_ms: float = 250.0
    min_clay_speed_mps: float = 6.0
    duplicate_deg: float = 1.0
    debris_cone_deg: float = 2.0
    debris_spread_deg_s: float = 15.0
    debris_window_ms: float = 1500.0
    debris_birth_margin_ms: float = 50.0
    recoil_settle_ms: float = 100.0


@dataclass
class ArmingConfig:
    pin: int
    active_low: bool
    pull: str
    debounce_ms: float


@dataclass
class ServoConfig:
    pwm_chip: int
    pwm_channel: int
    frame_hz: float
    rest_us: float
    fire_us: float
    hold_ms: float
    return_ms: float = 250.0
    jog_burst_ms: float = 400.0
    min_us: float = 0.0
    max_us: float = 0.0
    max_jump_us: float = 50.0


@dataclass
class Pq12Config:
    pin_extend: int
    pin_retract: int
    extend_ms: float
    retract_ms: float


@dataclass
class SimActuatorConfig:
    delay_ms: float
    model: str = "fixed"
    frame_hz: float = 50.0


@dataclass
class ActuatorConfig:
    type: str
    servo: ServoConfig
    pq12: Pq12Config
    sim: SimActuatorConfig


@dataclass
class GpioConfig:
    backend: str
    chip: int
    latency_led_pin: int
    break_sense_pin: int


@dataclass
class LatencyConfig:
    camera_timestamp_offset_ns: int
    exposure_to_user_ms: float
    servo_cmd_to_break_ms: float
    pq12_cmd_to_break_ms: float
    sim_cmd_to_break_ms: float
    none_cmd_to_break_ms: float
    jitter_ms: float


@dataclass
class StreamConfig:
    enabled: bool
    host: str
    port: int
    fps: float
    jpeg_quality: int


@dataclass
class RecordConfig:
    mode: str
    dir: str
    preroll_s: float
    postroll_s: float
    max_ram_mb: int
    decimate: int


@dataclass
class PathsConfig:
    calib_dir: str


@dataclass
class CalrigConfig:
    port: str
    baud: int
    microsteps: int
    steps_rev: int
    ratio: float
    vmax_dps: float
    amax_dps2: float
    vmax_cap_dps: float
    amax_cap_dps2: float
    soft_limit_deg: float
    hard_limit_deg: float
    telem_hz: int
    heartbeat_ms: int
    home_dir: int
    home_rate_dps: float
    run_root: str


@dataclass
class SimConfig:
    scenario: str
    seed: int
    sky_level: float
    sky_gradient: float
    cloud_amount: float
    clay_contrast: float
    noise_sigma: float
    render_color: bool
    gyro_noise_dps: float
    swing_lag_s: float
    swing_noise_deg: float


@dataclass
class Config:
    camera: CameraConfig
    optics: OpticsConfig
    mount: MountConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    imu: ImuConfig
    attitude: AttitudeConfig
    gun: GunConfig
    shot: ShotConfig
    fire: FireConfig
    arming: ArmingConfig
    actuator: ActuatorConfig
    gpio: GpioConfig
    latency: LatencyConfig
    stream: StreamConfig
    record: RecordConfig
    paths: PathsConfig
    sim: SimConfig
    calrig: CalrigConfig

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------------- paths
def software_root() -> Path:
    """The directory holding config/, calib/, tbot/ (this file's grandparent)."""
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else software_root() / p


DEFAULT_CONFIG = software_root() / "config" / "default.toml"
# Per-rig overrides written by bench tools (e.g. `tbot jog` servo pulses). Applied after default.toml and before
# --config files. TBOT_LOCAL_CONFIG overrides the path; an empty value disables it (the test suite does this).
LOCAL_CONFIG = software_root() / "config" / "local.toml"


def local_config_path() -> Path | None:
    env = os.environ.get("TBOT_LOCAL_CONFIG")
    if env is None:
        return LOCAL_CONFIG
    return resolve_path(env) if env else None


# --------------------------------------------------------------------------------------------- loading
def _build(cls: type, data: dict, where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = set(data) - set(names)
    if unknown:
        raise ValueError(f"unknown config key(s) in [{where}]: {sorted(unknown)}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ValueError(f"missing config key(s) in [{where}]: {missing}")
    kwargs = {}
    for name in names:
        hint = hints[name]
        val = data[name]
        key = f"{where}.{name}" if where else name
        if dataclasses.is_dataclass(hint):
            if not isinstance(val, dict):
                raise ValueError(f"[{key}] must be a table")
            kwargs[name] = _build(hint, val, key)
        elif hint is float:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{key} must be a number, got {val!r}")
            kwargs[name] = float(val)
        elif hint is int:
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{key} must be an integer, got {val!r}")
            kwargs[name] = val
        elif hint is bool:
            if not isinstance(val, bool):
                raise ValueError(f"{key} must be true/false, got {val!r}")
            kwargs[name] = val
        elif hint is str:
            if not isinstance(val, str):
                raise ValueError(f"{key} must be a string, got {val!r}")
            kwargs[name] = val
        elif hint is list:
            if not isinstance(val, list):
                raise ValueError(f"{key} must be an array, got {val!r}")
            kwargs[name] = copy.deepcopy(val)
        else:  # pragma: no cover - schema uses only the types above
            kwargs[name] = val
    return cls(**kwargs)


def _deep_merge(base: dict, over: dict, where: str = "") -> dict:
    for k, v in over.items():
        key = f"{where}.{k}" if where else k
        if k not in base:
            raise ValueError(f"unknown config key: {key}")
        if isinstance(v, dict) and isinstance(base[k], dict):
            _deep_merge(base[k], v, key)
        else:
            base[k] = v
    return base


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text  # bare word -> string


def _apply_set(raw: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"--set expects key.path=value, got {assignment!r}")
    path, text = assignment.split("=", 1)
    parts = path.strip().split(".")
    node = raw
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise ValueError(f"unknown config key: {path}")
        node = node[p]
    if parts[-1] not in node:
        raise ValueError(f"unknown config key: {path}")
    node[parts[-1]] = _parse_value(text.strip())


def load_config(overrides: list[str | Path] | None = None, sets: list[str] | None = None) -> Config:
    """default.toml <- config/local.toml (if present) <- override files <- --set assignments."""
    with open(DEFAULT_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    local = local_config_path()
    if local is not None and local.exists():
        with open(local, "rb") as f:
            _deep_merge(raw, tomllib.load(f), f"<{local.name}>")
    for path in overrides or []:
        with open(resolve_path(path), "rb") as f:
            _deep_merge(raw, tomllib.load(f))
    for s in sets or []:
        _apply_set(raw, s)
    return _build(Config, raw, "")


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)  # JSON string escaping is valid TOML basic-string escaping
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _toml_dumps(data: dict, prefix: str = "") -> str:
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    out = []
    if scalars:
        if prefix:
            out.append(f"[{prefix}]")
        out += [f"{k} = {_toml_value(v)}" for k, v in scalars.items()]
        out.append("")
    for k, v in tables.items():
        out.append(_toml_dumps(v, f"{prefix}.{k}" if prefix else k))
    return "\n".join(out)


def save_local_overrides(patch: dict) -> Path:
    """Deep-merge patch (nested dict of config keys) into config/local.toml after validating it against the schema."""
    path = local_config_path()
    if path is None:
        raise RuntimeError("local config disabled (TBOT_LOCAL_CONFIG is empty)")
    current: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            current = tomllib.load(f)

    def merge(a: dict, b: dict) -> None:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = copy.deepcopy(v)

    merge(current, patch)
    with open(DEFAULT_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    _build(Config, _deep_merge(raw, copy.deepcopy(current)), "")  # raises on unknown keys / wrong types
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Per-rig overrides of config/default.toml (written by tbot bench tools; edit freely).\n\n"
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(header + _toml_dumps(current))
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------------- calibration
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: str | Path, obj: Any) -> Path:
    """Write obj (numpy-aware) as pretty JSON, creating parent dirs. Returns the path."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")
    tmp.replace(path)
    return path


def update_json(path: str | Path, patch: dict) -> dict:
    """Deep-merge patch into the JSON file at path (created if absent) and write it back."""
    path = resolve_path(path)
    data: dict = json.loads(path.read_text()) if path.exists() else {}

    def merge(a: dict, b: dict) -> None:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(data, _jsonable(patch))
    save_json(path, data)
    return data


@dataclass
class LatencyBudget:
    """Delays between a fire decision and the shot leaving the muzzle, plus camera timing."""

    camera_timestamp_offset_ns: int
    exposure_to_user_ms: float
    cmd_to_break_ms: dict[str, float]
    jitter_ms: dict[str, float]
    lock_time_ms: float
    barrel_time_ms: float
    sources: dict[str, str] = field(default_factory=dict)  # item -> "measured" | "default"

    def fire_delay_s(self, actuator_type: str) -> float:
        """Decision -> shot exits the muzzle, seconds."""
        return (self.cmd_to_break_ms[actuator_type] + self.lock_time_ms + self.barrel_time_ms) / 1000.0

    def fire_jitter_s(self, actuator_type: str) -> float:
        return self.jitter_ms[actuator_type] / 1000.0

    @classmethod
    def from_config(cls, cfg: Config, measured: dict | None) -> "LatencyBudget":
        lat = cfg.latency
        types = ("servo", "pq12", "sim", "none")
        cmd = {t: float(getattr(lat, f"{t}_cmd_to_break_ms")) for t in types}
        jit = {t: float(lat.jitter_ms) for t in types}
        sources = {f"actuator.{t}": "default" for t in types}
        cam_off = int(lat.camera_timestamp_offset_ns)
        exp_ms = float(lat.exposure_to_user_ms)
        sources["camera.timestamp_offset"] = "default"
        sources["camera.exposure_to_user"] = "default"
        m = measured or {}
        cam = m.get("camera") or {}
        if "timestamp_offset_ns" in cam:
            cam_off = int(cam["timestamp_offset_ns"])
            sources["camera.timestamp_offset"] = "measured"
        if isinstance(cam.get("exposure_to_user_ms"), dict) and "mean" in cam["exposure_to_user_ms"]:
            exp_ms = float(cam["exposure_to_user_ms"]["mean"])
            sources["camera.exposure_to_user"] = "measured"
        for t, rec in (m.get("actuator") or {}).items():
            stats = rec.get("cmd_to_break_ms") if isinstance(rec, dict) else None
            if isinstance(stats, dict) and "mean" in stats and t in cmd:
                cmd[t] = float(stats["mean"])
                if "std" in stats:
                    jit[t] = float(stats["std"])
                # An operator estimate written into latency.json is labelled as such, not as a measurement.
                method = str(rec.get("method", "")).lower()
                sources[f"actuator.{t}"] = "estimate" if "estimate" in method else "measured"
        return cls(
            camera_timestamp_offset_ns=cam_off,
            exposure_to_user_ms=exp_ms,
            cmd_to_break_ms=cmd,
            jitter_ms=jit,
            lock_time_ms=float(cfg.gun.lock_time_ms),
            barrel_time_ms=float(cfg.gun.barrel_time_ms),
            sources=sources,
        )


@dataclass
class Calibration:
    """Everything under paths.calib_dir. Absent files are None (consumers fall back to config priors)."""

    dir: Path
    intrinsics: dict | None = None
    boresight: dict | None = None
    imu: dict | None = None
    latency: dict | None = None
    detector: dict | None = None
    mask: np.ndarray | None = None  # uint8, 255 = usable
    budget: LatencyBudget | None = None

    def path(self, name: str) -> Path:
        return self.dir / name


def _read_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def load_calibration(cfg: Config) -> Calibration:
    """Load calib JSONs/mask and apply detector.json over cfg.detector (in place)."""
    d = resolve_path(cfg.paths.calib_dir)
    cal = Calibration(dir=d)
    cal.intrinsics = _read_json(d / "intrinsics.json")
    cal.boresight = _read_json(d / "boresight.json")
    cal.imu = _read_json(d / "imu.json")
    cal.latency = _read_json(d / "latency.json")
    cal.detector = _read_json(d / "detector.json")
    if cal.detector:
        merged = dataclasses.asdict(cfg.detector)
        _deep_merge(merged, cal.detector, "detector")
        cfg.detector = _build(DetectorConfig, merged, "detector")
    mask_p = d / "mask.png"
    if mask_p.exists():
        import cv2

        m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"unreadable mask: {mask_p}")
        cal.mask = m
    cal.budget = LatencyBudget.from_config(cfg, cal.latency)
    return cal


# --------------------------------------------------------------------------------------------- CLI glue
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("configuration")
    g.add_argument("--config", action="append", default=[], metavar="TOML",
                   help="override file(s) applied over config/default.toml (repeatable)")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one key, e.g. --set camera.fps=90 (repeatable)")
    g.add_argument("--source", default=None,
                   help="frame source: picam | cv:<index> | video:<path> | replay:<dir> | synthetic[:<scenario>]")
    g.add_argument("--calib-dir", default=None, help="calibration directory (default: paths.calib_dir)")
    return parser


def load_from_args(args: argparse.Namespace) -> tuple[Config, Calibration]:
    cfg = load_config(getattr(args, "config", None), getattr(args, "set", None))
    if getattr(args, "source", None):
        cfg.camera.source = args.source
    if getattr(args, "calib_dir", None):
        cfg.paths.calib_dir = args.calib_dir
    return cfg, load_calibration(cfg)
