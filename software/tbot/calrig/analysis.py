"""CalRig B analysis: a recorded run directory (docs/CALRIG_PROTOCOL.md section 4) -> IMU calibration results.

Public API
    load_run(run_dir) -> Run                         read meta.json + the five CSV files (csv + numpy only)
    analyze_run(run_dir) -> dict                     all estimates of ONE run, each with sigma + pass/fail
    write_report(result, run_dir) -> Path            report.json + report.md next to the run (returns report.md)
    combine_runs(run_dirs) -> dict                   several poses -> R_gun_imu (Kabsch), bias/scale/time aggregate
    propose_calib(result) -> dict                    calib/imu.json keys (keys starting "_" are notes, never written)
    apply_to_calib(proposal, calib_dir, dry_run=True)   the only function that can touch calib/imu.json

Conventions
    * Everything is in the IMU CHIP frame unless named gun/world. gyro rad/s, accel m/s^2, angles deg.
    * Table angle positive = counter-clockwise seen from above = right-hand rotation about world +Z (up).
    * The gyro of a chip whose axis u (chip frame) points up reads +omega*u when the table turns +omega,
      and its accelerometer reads +g*u (specific force points up). So per pose the yaw-axis direction and
      the gravity direction are the same chip-frame vector; the gun-frame "up" of every pose is known
      (POSE_UP_GUN) and R_gun_imu (gun = R @ chip) is the rotation mapping the measured chip vectors onto it.
    * time offset: `late_s` > 0 means the IMU timestamps are LATE (a sample stamped t was measured at t - late_s).
      The calib/imu.json key `time_offset_ns` is ADDED to IMU timestamps, so it is -late_s (in ns).
    * gyro scale: `factor` = gyro / rig angle (1.015 = the gyro reads 1.5 % high); the correction multiplier
      to apply to the gyro is 1/factor.

Pipeline (analyze_run): rig time base from t_us -> rig rate (local linear fit of the angle) -> move/still
segmentation -> bias + gravity from still segments -> yaw axis (regression of gyro on rig rate) -> time offset
(cross-correlation of rig rate with yaw-axis gyro, parabolic refinement) -> scale per move (integral of the
bias-removed gyro over the move window / rig angle change) -> tracking residual + ringing -> hall / HOMED check.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "LIMITS", "LIMIT_BASIS", "POSE_UP_GUN", "Run", "load_run", "analyze_run", "write_report",
    "combine_runs", "propose_calib", "apply_to_calib",
]

G_MPS2 = 9.80665
DEG2RAD = math.pi / 180.0
DEFAULT_DEG_PER_USTEP = 360.0 / (200 * 16)

# ----------------------------------------------------------------------------------------------- limits
# key: (value, unit, basis). Every pass/fail in this module is against one of these. "datasheet" = InvenSense
# MPU-6000/6050 product specification (from memory, not re-checked here); [INFERENCE] = engineering choice.
_LIMIT_TABLE: dict[str, tuple[float, str, str]] = {
    "still_min_s": (2.0, "s", "assignment: a still segment is rig velocity 0 and no motion for >= 2 s; the "
                              "settle margin after a move is trimmed first, so pauses must be >= still_min_s "
                              "+ 2 * SETTLE_S"),
    "gyro_bias_abs_max_dps": (20.0, "deg/s", "datasheet: gyro zero-rate output initial tolerance +-20 deg/s; "
                                             "beyond that the chip is out of spec"),
    "gyro_bias_sigma_max_dps": (0.05, "deg/s", "[INFERENCE] 0.05 deg/s uncorrected = 3 deg/min heading drift; "
                                               "the attitude filter's accel gain removes pitch/roll only"),
    "gyro_bias_spread_max_dps": (0.15, "deg/s", "[INFERENCE] bias disagreement between still segments of one "
                                                "run: >0.15 deg/s means thermal drift or the gun was not still"),
    "accel_norm_tol": (0.05, "fraction", "same 5 % gate as `tbot calib-imu bias` for |accel| vs g"),
    "axis_tilt_max_deg": (5.0, "deg", "[INFERENCE] mounting tolerance of a printed pod; calib_imu axes warns at 20"),
    "axis_gravity_max_deg": (5.0, "deg", "[INFERENCE] angle between the table axis seen by the gyro and by the "
                                         "accelerometer; large = rig axis not vertical or gyro/accel misaligned"),
    "scale_abs_max_pct": (3.0, "%", "datasheet: gyro sensitivity scale factor tolerance +-3 % at 25 C"),
    "scale_sigma_max_pct": (0.3, "%", "[INFERENCE] a scale uncertainty above 0.3 % is no better than the datasheet "
                                      "tolerance is worth correcting"),
    "scale_direction_diff_max_pct": (0.5, "%", "[INFERENCE] CW and CCW moves must agree: a difference points to "
                                               "a bias error, a timing error or strap slip, not scale"),
    "time_offset_abs_max_ms": (20.0, "ms", "[INFERENCE] larger than 2 camera frames at 120 fps: suspect the "
                                           "clock fit or a dropped-sample bug"),
    "time_offset_sigma_max_ms": (1.0, "ms", "[INFERENCE] 1 ms of timestamp error is 0.05 deg at 50 deg/s swing"),
    "time_offset_corr_min": (0.9, "-", "[INFERENCE] normalised correlation at the peak; lower = no clean "
                                      "rate signal (rig or gyro not following the motion)"),
    "hall_repeat_max_deg": (0.5, "deg", "[INFERENCE] A3144 edge + ISR jitter is ~0.1-0.2 deg at the magnet "
                                        "radius; 0.5 deg is < 1/3 of a full step (1.8 deg), a lost full step "
                                        "always exceeds it"),
    "homed_offset_max_deg": (0.5, "deg", "same basis as hall_repeat_max_deg: a HOMED offset is the step error "
                                         "accumulated since the previous homing"),
    "tracking_rms_max_deg": (0.3, "deg", "[INFERENCE] gyro-integrated vs rig angle after scale removal; "
                                         "gyro noise contributes < 0.02 deg per move, so 0.3 deg is slip/ringing"),
    "tracking_max_deg": (1.0, "deg", "[INFERENCE] below one full step (1.8 deg); more means the strap slipped "
                                     "or the gun rang hard"),
    "ringing_amp_max_dps": (1.0, "deg/s", "[INFERENCE] post-move gyro ringing amplitude of the gun on the "
                                          "strap; above this the settle margin must grow"),
    "axis_fit_resid_max_deg": (3.0, "deg", "[INFERENCE] RMS Kabsch residual over all poses: clamping "
                                           "tolerance of the V-saddles"),
    "R_config_diff_max_deg": (5.0, "deg", "[INFERENCE] fitted R_gun_imu vs config R_gun_imu: more than a mount "
                                          "tolerance means the configured axis map is wrong"),
}
LIMITS: dict[str, float] = {k: v[0] for k, v in _LIMIT_TABLE.items()}
LIMIT_BASIS: dict[str, str] = {k: f"{v[1]}: {v[2]}" for k, v in _LIMIT_TABLE.items()}

# Gun frame: +X muzzle, +Y left, +Z up (docs: AXES_POSES in tbot.tools.calib_imu). Direction that points UP in the
# gun frame for each rig pose = the table axis seen from the gun. Rolling about the barrel never makes X vertical.
POSE_UP_GUN: dict[str, tuple[float, float, float]] = {
    "upright": (0.0, 0.0, 1.0),
    "inverted": (0.0, 0.0, -1.0),
    "left_side_down": (0.0, -1.0, 0.0),
    "right_side_down": (0.0, 1.0, 0.0),
    "muzzle_up": (1.0, 0.0, 0.0),       # not a rig pose in the spec; accepted so chip Z can be exercised
    "muzzle_down": (-1.0, 0.0, 0.0),
}

# Analysis constants (algorithm parameters, not pass/fail limits).
RATE_WIN_S = 0.20            # local-linear window for the rig rate; 0.2 s minimises the lag sigma on the default
                             # profiles (0.1125 deg quantisation noise at 0.1 s, corner rounding above 0.3 s)
MOVE_THRESH_DPS = 1.0        # |rig rate| above this = moving
MERGE_GAP_S = 0.25           # moving bursts closer than this are one move (turnarounds, sine zero crossings)
MIN_MOVE_S = 0.5
MIN_MOVE_PEAK_DPS = 3.0
SETTLE_S = 0.5               # trimmed off a still segment on each side that touches a move
RUN_EDGE_TRIM_S = 0.2
WINDOW_PAD_S = 1.0           # still padding included around a move for the scale integral / tracking
NET_MOVE_MIN_DEG = 10.0      # a move counts for the integral scale only if the net angle is at least this
STILL_GYRO_PTP_DPS = 0.5     # reject a still segment whose 0.5 s block means differ by more than this
LAG_SEARCH_S = 0.25
RING_START_S = 0.1
RING_MAX_S = 2.5
RING_MIN_S = 0.6
RING_SNR = 8.0
RING_MIN_DPS = 0.05
RING_HEAD_S = 0.6            # first part of each tail used for the initial-amplitude fit
RING_BAND_HZ = (0.7, 40.0)
TEMP_SLOPE_MIN_SPAN_C = 1.0
FULL_STEP_DEG = 1.8


# ------------------------------------------------------------------------------------------- run loading
@dataclass(eq=False)
class Run:
    """One run directory in memory. All time stamps are int64 host-clock nanoseconds (`t_ns`)."""

    path: Path
    meta: dict
    rig: dict[str, np.ndarray]       # t_ns t_us pos_ustep angle_deg vel_deg_s (file order = arrival order)
    cmd_t_ns: np.ndarray
    cmd: list[str]
    hall: dict[str, np.ndarray]      # t_ns t_us pos_ustep angle_deg level
    events: dict[str, Any]           # t_ns t_us (arrays), code (list[str]), arg (float array, NaN if blank)
    imu: dict[str, np.ndarray]       # t_ns, gyro (N,3), accel (N,3), temp_c


def _read_table(path: Path, columns: tuple[str, ...], required: bool) -> dict[str, list[str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} is missing (docs/CALRIG_PROTOCOL.md section 4)")
        return {c: [] for c in columns}
    with path.open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        if required:
            raise ValueError(f"{path} is empty")
        return {c: [] for c in columns}
    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    out: dict[str, list[str]] = {}
    for c in columns:
        if c in header:
            i = header.index(c)
            out[c] = [r[i].strip() if i < len(r) else "" for r in body]
        else:
            out[c] = []
    first = columns[0]
    if required and not out[first] and body:
        raise ValueError(f"{path}: header {header} lacks column {first!r}")
    return out


def _f(col: list[str]) -> np.ndarray:
    out = np.empty(len(col), dtype=np.float64)
    for i, s in enumerate(col):
        try:
            out[i] = float(s)
        except ValueError:
            out[i] = math.nan
    return out


def _i64(col: list[str]) -> np.ndarray:
    out = np.empty(len(col), dtype=np.int64)
    for i, s in enumerate(col):
        try:
            out[i] = int(s)
        except ValueError:
            try:
                out[i] = int(float(s))
            except ValueError:
                out[i] = -1
    return out


def _col_or_nan(tab: dict[str, list[str]], name: str, n: int) -> np.ndarray:
    return _f(tab[name]) if len(tab[name]) == n else np.full(n, math.nan)


def load_run(run_dir: str | Path) -> Run:
    """Read a run directory. meta.json, rig.csv and imu.csv are required, the rest may be empty/missing."""
    d = Path(run_dir)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    t = _read_table(d / "rig.csv", ("t_ns", "t_us", "pos_ustep", "angle_deg", "vel_deg_s"), True)
    n = len(t["t_ns"])
    rig = {"t_ns": _i64(t["t_ns"]), "t_us": _i64(t["t_us"]) if len(t["t_us"]) == n else np.full(n, -1, np.int64),
           "pos_ustep": _col_or_nan(t, "pos_ustep", n), "angle_deg": _col_or_nan(t, "angle_deg", n),
           "vel_deg_s": _col_or_nan(t, "vel_deg_s", n)}

    t = _read_table(d / "cmd.csv", ("t_ns", "command"), False)
    cmd_t = _i64(t["t_ns"])
    cmd = list(t["command"])

    t = _read_table(d / "hall.csv", ("t_ns", "t_us", "pos_ustep", "angle_deg", "level"), False)
    n = len(t["t_ns"])
    hall = {"t_ns": _i64(t["t_ns"]), "t_us": _i64(t["t_us"]) if len(t["t_us"]) == n else np.full(n, -1, np.int64),
            "pos_ustep": _col_or_nan(t, "pos_ustep", n), "angle_deg": _col_or_nan(t, "angle_deg", n),
            "level": _col_or_nan(t, "level", n)}

    t = _read_table(d / "events.csv", ("t_ns", "t_us", "code", "arg"), False)
    n = len(t["t_ns"])
    events = {"t_ns": _i64(t["t_ns"]), "t_us": _i64(t["t_us"]) if len(t["t_us"]) == n else np.full(n, -1, np.int64),
              "code": list(t["code"]), "arg": _col_or_nan(t, "arg", n)}

    t = _read_table(d / "imu.csv", ("t_ns", "gx", "gy", "gz", "ax", "ay", "az", "temp_c"), True)
    n = len(t["t_ns"])
    gyro = np.column_stack([_f(t["gx"]), _f(t["gy"]), _f(t["gz"])]) if n else np.zeros((0, 3))
    accel = np.column_stack([_f(t["ax"]), _f(t["ay"]), _f(t["az"])]) if n else np.zeros((0, 3))
    imu = {"t_ns": _i64(t["t_ns"]), "gyro": gyro, "accel": accel, "temp_c": _col_or_nan(t, "temp_c", n)}
    good = np.all(np.isfinite(gyro), axis=1) & np.all(np.isfinite(accel), axis=1) & (imu["t_ns"] > 0)
    order = np.argsort(imu["t_ns"][good], kind="stable")
    imu = {k: v[good][order] for k, v in imu.items()}
    return Run(d, meta, rig, cmd_t, cmd, hall, events, imu)


# ------------------------------------------------------------------------------------------- small helpers
def _clean(obj: Any) -> Any:
    """numpy/NaN-safe JSON-able copy (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _rot_angle_deg(R: np.ndarray) -> float:
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cu = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cu))))


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R a_i - b_i| (rows). Same convention as tbot.tools.calib_imu.kabsch."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    d = 1.0 if np.linalg.det(U @ Vt) > 0.0 else -1.0
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _unwrap_us(t_us: np.ndarray) -> np.ndarray:
    t = np.asarray(t_us, dtype=np.int64)
    d = np.diff(t) % (1 << 32)
    return t[0] + np.concatenate(([0], np.cumsum(d)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (i0, i1) index ranges of consecutive True values."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    idx = np.flatnonzero(np.diff(m))
    return [(int(a), int(b) - 1) for a, b in zip(idx[::2], idx[1::2])]


def _neff_inflation(res: np.ndarray) -> float:
    """Variance inflation (1+rho)/(1-rho) of the mean of an AR(1)-like residual (lag-1 autocorrelation rho)."""
    r = np.asarray(res, dtype=np.float64).reshape(-1)
    if r.size < 8:
        return 1.0
    r = r - r.mean()
    den = float(r @ r)
    if den <= 0.0:
        return 1.0
    rho = float(np.clip((r[1:] @ r[:-1]) / den, 0.0, 0.95))
    return (1.0 + rho) / (1.0 - rho)


def _parabola_peak(x: np.ndarray, y: np.ndarray, i: int) -> tuple[float, float]:
    """Vertex (x, y) of the parabola through samples i-1, i, i+1 (uniform spacing); the sample itself at the ends."""
    if i <= 0 or i >= len(x) - 1:
        return float(x[i]), float(y[i])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    den = y0 - 2.0 * y1 + y2
    if den >= 0.0:
        return float(x[i]), y1
    delta = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    return float(x[i] + delta * (x[1] - x[0])), y1 - 0.25 * (y0 - y2) * delta


def _clock_info(meta: dict) -> dict:
    for key in ("clock_fit", "clock", "clocksync", "clock_sync"):
        v = meta.get(key)
        if isinstance(v, dict):
            return v
    return {}


# ------------------------------------------------------------------------------------------- preparation
@dataclass(eq=False)
class _Prep:
    t0_ns: int
    ti: np.ndarray            # IMU times, s
    gyro: np.ndarray          # rad/s chip frame (raw)
    accel: np.ndarray
    temp: np.ndarray
    tr: np.ndarray            # rig times, s (fitted from t_us when possible)
    ang: np.ndarray           # rig angle, deg
    tg: np.ndarray            # uniform grid for the rig rate
    rg: np.ndarray            # rig rate on tg, deg/s
    rig_info: dict
    moves: list[dict] = field(default_factory=list)
    stills: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _rig_time(rig: dict[str, np.ndarray], t0_ns: int) -> tuple[np.ndarray, dict]:
    """Rig times in s from the host clock t_ns, smoothed by a line through the firmware t_us when consistent."""
    t_ns = rig["t_ns"]
    raw = (t_ns - t0_ns).astype(np.float64) * 1e-9
    info: dict[str, Any] = {"source": "t_ns", "jitter_us": None, "drift_ppm": None}
    t_us = rig["t_us"]
    if len(t_ns) >= 20 and np.all(t_us >= 0):
        u = _unwrap_us(t_us).astype(np.float64)
        u -= u[0]
        if np.all(np.diff(u) > 0):
            x = (t_ns - t_ns[0]).astype(np.float64)
            slope, icpt = np.polyfit(u, x, 1)
            resid = x - (slope * u + icpt)
            rms = float(resid.std())
            info["jitter_us"] = rms * 1e-3
            info["drift_ppm"] = (slope / 1000.0 - 1.0) * 1e6
            if rms < 20e6 and float(np.abs(resid).max()) < 100e6:
                fit_ns = slope * u + icpt
                info["source"] = "t_us_fit"
                return (float(t_ns[0] - t0_ns) + fit_ns) * 1e-9, info
    return raw, info


def _local_slope(ag: np.ndarray, dt: float) -> np.ndarray:
    """Zero-lag derivative of a uniformly sampled signal: slope of a local-linear fit over RATE_WIN_S."""
    w = max(5, int(round(RATE_WIN_S / dt)) | 1)
    m = w // 2
    k = np.arange(-m, m + 1, dtype=np.float64)
    coef = k / (dt * float(k @ k))
    return np.correlate(np.pad(ag, m, mode="edge"), coef, mode="valid")


def _uniform_rate(t: np.ndarray, ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle (deg) -> (uniform grid, rate deg/s) with a symmetric local-linear (zero-lag) derivative."""
    dt = float(np.median(np.diff(t)))
    if not dt > 0.0:
        raise ValueError("rig time stamps are not increasing")
    n = int(round((t[-1] - t[0]) / dt)) + 1
    tg = t[0] + dt * np.arange(n)
    return tg, _local_slope(np.interp(tg, t, ang), dt)


def _prepare(run: Run) -> _Prep:
    imu, rig = run.imu, run.rig
    if len(imu["t_ns"]) < 100:
        raise ValueError(f"imu.csv has only {len(imu['t_ns'])} samples")
    if len(rig["t_ns"]) < 20:
        raise ValueError(f"rig.csv has only {len(rig['t_ns'])} rows")
    t0 = int(min(imu["t_ns"][0], rig["t_ns"][0]))
    ti = (imu["t_ns"] - t0).astype(np.float64) * 1e-9
    tr, info = _rig_time(rig, t0)
    dpu = float(run.meta.get("deg_per_ustep") or DEFAULT_DEG_PER_USTEP)
    ang = rig["angle_deg"].copy()
    if not np.all(np.isfinite(ang)):
        ang = rig["pos_ustep"] * dpu
    ok = np.isfinite(ang) & np.isfinite(tr)
    tr, ang = tr[ok], ang[ok]
    order = np.argsort(tr, kind="stable")
    tr, ang = tr[order], ang[order]
    keep = np.concatenate(([True], np.diff(tr) > 0))
    tr, ang = tr[keep], ang[keep]
    tg, rg = _uniform_rate(tr, ang)
    p = _Prep(t0, ti, imu["gyro"], imu["accel"], imu["temp_c"], tr, ang, tg, rg, info)
    p.moves, p.stills = _segment(p)
    return p


def _segment(p: _Prep) -> tuple[list[dict], list[tuple[float, float]]]:
    mask = np.abs(p.rg) > MOVE_THRESH_DPS
    bursts: list[list[float]] = []
    for i0, i1 in _runs(mask):
        t0, t1 = float(p.tg[i0]), float(p.tg[i1])
        if bursts and t0 - bursts[-1][1] < MERGE_GAP_S:
            bursts[-1][1] = t1
        else:
            bursts.append([t0, t1])
    moves: list[dict] = []
    for t0, t1 in bursts:
        i0, i1 = np.searchsorted(p.tg, t0), np.searchsorted(p.tg, t1, side="right")
        peak = float(np.abs(p.rg[i0:i1]).max()) if i1 > i0 else 0.0
        if t1 - t0 >= MIN_MOVE_S and peak >= MIN_MOVE_PEAK_DPS:
            moves.append({"t0": t0, "t1": t1, "peak_dps": peak})
    lo = max(float(p.ti[0]), float(p.tr[0]))
    hi = min(float(p.ti[-1]), float(p.tr[-1]))
    # still segments: gaps between motion bursts (ALL bursts, also short ones), trimmed beside motion
    edges = [(a, b) for a, b in bursts]
    stills: list[tuple[float, float]] = []
    cursor, prev_motion = lo, False
    for a, b in edges + [(hi, hi)]:
        s0 = cursor + (SETTLE_S if prev_motion else RUN_EDGE_TRIM_S)
        s1 = a - (SETTLE_S if a < hi else RUN_EDGE_TRIM_S)
        if s1 - s0 >= LIMITS["still_min_s"]:
            stills.append((s0, s1))
        cursor, prev_motion = max(cursor, b), True
    for k, m in enumerate(moves):  # window with still padding, clipped to half the gap to the neighbours
        prev_end = moves[k - 1]["t1"] if k > 0 else lo
        next_start = moves[k + 1]["t0"] if k + 1 < len(moves) else hi
        m["w0"] = m["t0"] - min(WINDOW_PAD_S, max(0.0, (m["t0"] - prev_end) / 2.0 if k > 0 else m["t0"] - lo))
        m["w1"] = m["t1"] + min(WINDOW_PAD_S, max(0.0, (next_start - m["t1"]) / 2.0 if k + 1 < len(moves)
                                                  else hi - m["t1"]))
    return moves, stills


def _slice(ti: np.ndarray, a: float, b: float) -> slice:
    return slice(int(np.searchsorted(ti, a)), int(np.searchsorted(ti, b)))


def _move_mask(p: _Prep, pad: float = 0.0) -> np.ndarray:
    m = np.zeros(len(p.ti), dtype=bool)
    for mv in p.moves:
        m[_slice(p.ti, mv["w0"] - pad, mv["w1"] + pad)] = True
    return m


# ------------------------------------------------------------------------------------------- estimates
def _status(pass_: bool | None, **kw: Any) -> dict:
    d = {"status": "ok" if pass_ is not None else "insufficient_data", "pass": pass_}
    d.update(kw)
    return d


def _insufficient(reason: str) -> dict:
    return {"status": "insufficient_data", "pass": None, "reason": reason}


def _estimate_bias_gravity(p: _Prep) -> tuple[dict, dict]:
    segs = []
    rejected = 0
    for a, b in p.stills:
        sl = _slice(p.ti, a, b)
        g = p.gyro[sl]
        n = len(g)
        if n < 50:
            continue
        nb = max(1, int((b - a) / 0.5))
        edges = np.linspace(0, n, nb + 1).astype(int)
        bm = np.array([g[edges[j]:edges[j + 1]].mean(axis=0) for j in range(nb)])
        if nb >= 2 and float(np.ptp(bm, axis=0).max()) > STILL_GYRO_PTP_DPS * DEG2RAD:
            rejected += 1
            continue
        sem_white = g.std(axis=0, ddof=1) / math.sqrt(n)
        sem_blocks = bm.std(axis=0, ddof=1) / math.sqrt(nb) if nb >= 3 else np.zeros(3)
        temp = p.temp[sl]
        temp = temp[np.isfinite(temp)]
        segs.append({"t0": a, "t1": b, "n": n, "mean": g.mean(axis=0), "sem": np.maximum(sem_white, sem_blocks),
                     "std": g.std(axis=0, ddof=1), "temp": float(temp.mean()) if temp.size else math.nan,
                     "accel": p.accel[sl].mean(axis=0)})
    if not segs:
        why = "no still segment of >= %.1f s (rig at rest, %.1f s settle margin trimmed)" % (
            LIMITS["still_min_s"], SETTLE_S)
        if rejected:
            why += f"; {rejected} candidate(s) rejected because the gyro moved while the rig was still"
        return _insufficient(why), _insufficient(why)

    w = np.array([s["n"] for s in segs], dtype=np.float64)
    means = np.array([s["mean"] for s in segs])
    sems = np.array([s["sem"] for s in segs])
    bias = (w[:, None] * means).sum(axis=0) / w.sum()
    sig_stat = np.sqrt(((w[:, None] * sems) ** 2).sum(axis=0)) / w.sum()
    spread = np.ptp(means, axis=0) if len(segs) > 1 else np.zeros(3)
    sig_spread = means.std(axis=0, ddof=1) / math.sqrt(len(segs)) if len(segs) > 1 else np.zeros(3)
    sigma = np.maximum(sig_stat, sig_spread)
    noise = np.array([s["std"] for s in segs]).mean(axis=0)

    temps = np.array([s["temp"] for s in segs])
    tfin = np.isfinite(temps)
    temp_c = float(temps[tfin].mean()) if tfin.any() else math.nan
    slope = None
    if tfin.sum() >= 3 and float(np.ptp(temps[tfin])) >= TEMP_SLOPE_MIN_SPAN_C:
        slope = [float(np.polyfit(temps[tfin], means[tfin, i], 1)[0]) / DEG2RAD for i in range(3)]

    checks = {
        "abs": bool(np.all(np.abs(bias) <= LIMITS["gyro_bias_abs_max_dps"] * DEG2RAD)),
        "sigma": bool(np.all(sigma <= LIMITS["gyro_bias_sigma_max_dps"] * DEG2RAD)),
        "spread": bool(np.all(spread <= LIMITS["gyro_bias_spread_max_dps"] * DEG2RAD)),
    }
    bias_est = _status(all(checks.values()), value_rad_s=bias, sigma_rad_s=sigma, value_dps=bias / DEG2RAD,
                       sigma_dps=sigma / DEG2RAD, noise_dps=noise / DEG2RAD, n_segments=len(segs),
                       total_s=float(sum(s["t1"] - s["t0"] for s in segs)),
                       segments=[{"t0": s["t0"], "t1": s["t1"], "bias_dps": s["mean"] / DEG2RAD,
                                  "temp_c": s["temp"]} for s in segs],
                       spread_dps=spread / DEG2RAD, rejected_segments=rejected, temp_c=temp_c,
                       temp_slope_dps_per_c=slope, checks=checks,
                       limits={k: LIMITS[k] for k in ("gyro_bias_abs_max_dps", "gyro_bias_sigma_max_dps",
                                                      "gyro_bias_spread_max_dps")})

    acc = (w[:, None] * np.array([s["accel"] for s in segs])).sum(axis=0) / w.sum()
    norm = float(np.linalg.norm(acc))
    gravity = _status(abs(norm / G_MPS2 - 1.0) <= LIMITS["accel_norm_tol"], mean_m_s2=acc, norm_m_s2=norm,
                      norm_g=norm / G_MPS2, dir=acc / norm, limit_tol=LIMITS["accel_norm_tol"])
    return bias_est, gravity


def _rate_rad(p: _Prep) -> np.ndarray:
    return p.rg * DEG2RAD


def _axis_fit(p: _Prep, bias: np.ndarray, d: float, mask: np.ndarray) -> dict:
    """Regress the bias-removed gyro on the (delayed) rig rate: g = k * r + c. Returns k (3), sigma_k (3)."""
    r = np.interp(p.ti[mask] - d, p.tg, _rate_rad(p), left=0.0, right=0.0)
    G = p.gyro[mask] - bias
    rc = r - r.mean()
    den = float(rc @ rc)
    if den <= 0.0:
        raise ValueError("rig rate is zero everywhere in the motion windows")
    k = (rc @ (G - G.mean(axis=0))) / den
    res = G - G.mean(axis=0) - np.outer(rc, k)
    infl = max(_neff_inflation(res[:, i]) for i in range(3))
    sig_k = res.std(axis=0, ddof=2) * math.sqrt(infl) / math.sqrt(den)
    return {"k": k, "sigma_k": sig_k, "resid_rms": res.std(axis=0), "inflation": infl}


def _corr_curve(y: np.ndarray, ti: np.ndarray, tg: np.ndarray, rr: np.ndarray, lags: np.ndarray) -> np.ndarray:
    # elementwise products + sum instead of `@`: a threaded BLAS ddot on 1e4 samples is slower than this on a loaded box
    out = np.empty(len(lags))
    yy = float((y * y).sum())
    for j, d in enumerate(lags):
        r = np.interp(ti - d, tg, rr, left=0.0, right=0.0)
        rr2 = float((r * r).sum())
        out[j] = float((y * r).sum()) / math.sqrt(rr2 * yy) if rr2 > 0.0 and yy > 0.0 else 0.0
    return out


def _lag_search(y, ti, tg, rr, center, half, step) -> tuple[float, float, bool]:
    """Peak lag of corr(y(t), r(t-d)): coarse grid then 0.1 ms grid + parabola. -> (lag, corr, at_edge)."""
    lags = center + np.arange(-half, half + step * 0.5, step)
    c = _corr_curve(y, ti, tg, rr, lags)
    i = int(np.argmax(c))
    edge = i == 0 or i == len(lags) - 1
    fine = lags[i] + np.arange(-2 * step, 2 * step + 1e-12, step / 10.0)
    cf = _corr_curve(y, ti, tg, rr, fine)
    j = int(np.argmax(cf))
    d, peak = _parabola_peak(fine, cf, j)
    return d, peak, edge


def _estimate_time_offset(p: _Prep, y: np.ndarray, resid_us: float) -> dict:
    """y: bias-removed yaw-axis gyro (rad/s, sign so that table + = positive). late_s > 0: IMU stamps late."""
    mask = _move_mask(p, 0.3)
    if int(mask.sum()) < 100:
        return _insufficient("no rig motion")
    rr = _rate_rad(p)
    ym, tm = y[mask], p.ti[mask]
    d, peak, edge = _lag_search(ym, tm, p.tg, rr, 0.0, LAG_SEARCH_S, 0.001)
    r = np.interp(tm - d, p.tg, rr, left=0.0, right=0.0)
    a = float(ym @ r) / float(r @ r)
    res = ym - a * r
    drdt = np.interp(tm - d, p.tg, np.gradient(rr, p.tg), left=0.0, right=0.0)
    fisher = a * a * float(drdt @ drdt)
    sig_crlb = float(res.std()) * math.sqrt(_neff_inflation(res)) / math.sqrt(fisher) if fisher > 0 else math.inf
    per = []
    for mv in p.moves:
        sl = _slice(p.ti, mv["w0"], mv["w1"])
        if sl.stop - sl.start < 100 or mv["t1"] - mv["t0"] < 1.0:
            continue
        dm, cm, _ = _lag_search(y[sl], p.ti[sl], p.tg, rr, d, 0.02, 0.0005)
        per.append(dm)
    emp = float(np.std(per, ddof=1)) / math.sqrt(len(per)) if len(per) >= 2 else 0.0
    sigma = math.sqrt(max(sig_crlb, emp) ** 2 + (resid_us * 1e-6) ** 2)
    ok = (abs(d) * 1e3 <= LIMITS["time_offset_abs_max_ms"] and sigma * 1e3 <= LIMITS["time_offset_sigma_max_ms"]
          and peak >= LIMITS["time_offset_corr_min"] and not edge)
    return _status(ok, late_s=d, late_ms=d * 1e3, sigma_s=sigma, sigma_ms=sigma * 1e3,
                   time_offset_ns=-int(round(d * 1e9)), correlation=peak, at_search_limit=edge,
                   sigma_crlb_ms=sig_crlb * 1e3, sigma_segments_ms=emp * 1e3, clock_resid_us=resid_us,
                   per_move_late_ms=[x * 1e3 for x in per], convention="late_s>0 = IMU timestamps late; "
                   "calib key time_offset_ns = -late_s*1e9 is ADDED to IMU timestamps",
                   limits={k: LIMITS[k] for k in ("time_offset_abs_max_ms", "time_offset_sigma_max_ms",
                                                   "time_offset_corr_min")})


def _estimate_scale(p: _Prep, y: np.ndarray, d: float, bias_sigma: float, cos_tilt: float = 1.0) -> dict:
    """Integral scale per move: gyro angle over [w0, w1] (IMU time shifted by d) / rig angle change.

    The chip axis is tilted by acos(cos_tilt) from the table axis, so it only sees omega*cos_tilt: divided out.
    """
    ang_rad = p.ang * DEG2RAD
    rr = _rate_rad(p)
    # gyro put through the SAME zero-lag derivative as the rig angle (integrate -> resample -> local slope) so the
    # slope regression is not biased by the rate filter's gain on oscillating moves
    th_g = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(p.ti))))
    y_filt = _local_slope(np.interp(p.tg + d, p.ti, th_g), float(p.tg[1] - p.tg[0]))
    moves = []
    for mv in p.moves:
        a, b = mv["w0"], mv["w1"]
        sl = _slice(p.ti, a + d, b + d)
        t, yy = p.ti[sl], y[sl]
        if len(t) < 50:
            continue
        g_int = float(np.sum(0.5 * (yy[1:] + yy[:-1]) * np.diff(t)))
        da = float(np.interp(t[-1] - d, p.tr, ang_rad) - np.interp(t[0] - d, p.tr, ang_rad))
        sg = _slice(p.tg, a, b)
        rg_, yg_ = rr[sg], y_filt[sg]
        slope = float(np.sum(yg_ * rg_)) / float(np.sum(rg_ * rg_)) / cos_tilt if float(np.sum(rg_ * rg_)) > 0 \
            else math.nan
        net = abs(da) >= NET_MOVE_MIN_DEG * DEG2RAD
        moves.append({"t0": mv["t0"], "t1": mv["t1"], "delta_deg": da / DEG2RAD,
                      "ratio": g_int / da / cos_tilt if net else math.nan, "slope": slope, "net": net,
                      "sigma_bias": bias_sigma * (t[-1] - t[0]) / abs(da) if net else math.nan})
    if not moves:
        return _insufficient("no complete move")
    net = [m for m in moves if m["net"]]
    method = "integral"
    if net:
        vals = np.array([m["ratio"] for m in net])
        dirs = np.array([m["delta_deg"] > 0 for m in net])
        sys_ = float(np.mean([m["sigma_bias"] for m in net]))
        n_pos, n_neg = int(dirs.sum()), int((~dirs).sum())
        factor = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        sem = std / math.sqrt(len(vals))
        sigma = math.sqrt(sem ** 2 + (sys_ * abs(n_pos - n_neg) / len(vals)) ** 2)
        per_dir = {"ccw": _dir_stats(vals[dirs]), "cw": _dir_stats(vals[~dirs])}
    else:
        method = "slope"
        vals = np.array([m["slope"] for m in moves if math.isfinite(m["slope"])])
        if vals.size == 0:
            return _insufficient("no usable move")
        factor = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        sigma = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.05
        per_dir = {}
    err_pct = (factor - 1.0) * 100.0
    diff = math.nan
    if "ccw" in per_dir and per_dir["ccw"]["n"] and per_dir["cw"]["n"]:
        diff = (per_dir["ccw"]["factor"] - per_dir["cw"]["factor"]) * 100.0
    checks = {"abs": abs(err_pct) <= LIMITS["scale_abs_max_pct"],
              "sigma": sigma * 100.0 <= LIMITS["scale_sigma_max_pct"],
              "direction": (not math.isfinite(diff)) or abs(diff) <= LIMITS["scale_direction_diff_max_pct"]}
    return _status(all(checks.values()), factor=factor, error_pct=err_pct, sigma_pct=sigma * 100.0,
                   spread_pct=std * 100.0, correction=1.0 / factor, method=method, tilt_cos=cos_tilt, n_moves=len(vals),
                   per_direction=per_dir, direction_diff_pct=diff, checks=checks,
                   moves=[{k: v for k, v in m.items() if k != "sigma_bias"} for m in moves],
                   limits={k: LIMITS[k] for k in ("scale_abs_max_pct", "scale_sigma_max_pct",
                                                   "scale_direction_diff_max_pct")})


def _dir_stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"n": 0, "factor": math.nan, "error_pct": math.nan, "std_pct": math.nan}
    return {"n": int(v.size), "factor": float(v.mean()), "error_pct": float(v.mean() - 1.0) * 100.0,
            "std_pct": float(v.std(ddof=1)) * 100.0 if v.size > 1 else 0.0}


def _estimate_tracking(p: _Prep, y: np.ndarray, d: float, factor: float) -> dict:
    """Gyro-integrated angle (per move window, scale removed) vs rig angle. RMS/max in deg."""
    ang_rad = p.ang * DEG2RAD
    res_all, raw_all = [], []
    worst = 0.0
    for mv in p.moves:
        sl = _slice(p.ti, mv["w0"] + d, mv["w1"] + d)
        t, yy = p.ti[sl], y[sl]
        if len(t) < 50:
            continue
        th_g = np.concatenate(([0.0], np.cumsum(0.5 * (yy[1:] + yy[:-1]) * np.diff(t))))
        th_r = np.interp(t - d, p.tr, ang_rad)
        th_r = th_r - th_r[0]
        res_all.append((th_g / factor - th_r) / DEG2RAD)
        raw_all.append((th_g - th_r) / DEG2RAD)
        worst = max(worst, float(np.abs(res_all[-1]).max()))
    if not res_all:
        return _insufficient("no move window")
    r = np.concatenate(res_all)
    rms = float(np.sqrt(np.mean(r ** 2)))
    raw = np.concatenate(raw_all)
    ok = rms <= LIMITS["tracking_rms_max_deg"] and worst <= LIMITS["tracking_max_deg"]
    return _status(ok, rms_deg=rms, max_deg=worst, raw_rms_deg=float(np.sqrt(np.mean(raw ** 2))),
                   scale_removed=factor, limits={k: LIMITS[k] for k in ("tracking_rms_max_deg",
                                                                        "tracking_max_deg")})


def _estimate_ringing(p: _Prep, y: np.ndarray, d: float) -> dict:
    """Spectrum of the yaw-axis gyro in the rig-still tail after each move (averaged power)."""
    wins = []
    for k, mv in enumerate(p.moves):
        a = mv["t1"] + RING_START_S + d
        nxt = p.moves[k + 1]["t0"] if k + 1 < len(p.moves) else p.tr[-1]
        b = min(mv["t1"] + RING_MAX_S, nxt - RING_START_S) + d
        b = min(b, float(p.ti[-1]))
        if b - a >= RING_MIN_S:
            wins.append(y[_slice(p.ti, a, b)])
    if not wins:
        return _insufficient("no still tail of >= %.1f s after a move" % RING_MIN_S)
    fs = (len(p.ti) - 1) / float(p.ti[-1] - p.ti[0])
    n = min(len(w) for w in wins)
    nfft = 1 << int(math.ceil(math.log2(max(n, 16) * 8)))
    win = np.hanning(n)
    power = np.zeros(nfft // 2 + 1)
    for w in wins:
        w = w[:n] / DEG2RAD
        w = w - np.polyval(np.polyfit(np.arange(n), w, 1), np.arange(n))
        power += np.abs(np.fft.rfft(w * win, nfft)) ** 2
    amp = np.sqrt(power / len(wins)) * 2.0 / win.sum()
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (f >= RING_BAND_HZ[0]) & (f <= RING_BAND_HZ[1])
    ib = np.flatnonzero(band)
    i = int(ib[np.argmax(amp[ib])])
    floor = float(np.median(amp[ib]))
    fpk, _ = _parabola_peak(f, np.log(np.maximum(amp, 1e-30)), i)
    visible = bool(amp[i] >= RING_SNR * floor and amp[i] >= RING_MIN_DPS)
    res_hz = fs / n
    # amplitude right after the move: LS fit of a sinusoid at the peak frequency to the first RING_HEAD_S of each tail
    nh = min(n, int(RING_HEAD_S * fs))
    th = np.arange(nh) / fs
    A = np.column_stack([np.sin(2 * np.pi * fpk * th), np.cos(2 * np.pi * fpk * th), np.ones(nh), th])
    heads = []
    for w in wins:
        c = np.linalg.lstsq(A, w[:nh] / DEG2RAD, rcond=None)[0]
        heads.append(math.hypot(c[0], c[1]))
    head = float(np.mean(heads))
    return _status(not (visible and head > LIMITS["ringing_amp_max_dps"]), visible=visible,
                   freq_hz=fpk if visible else None, amp_dps=float(amp[i]), amp_initial_dps=head, floor_dps=floor,
                   snr=float(amp[i] / floor) if floor > 0 else math.inf, resolution_hz=res_hz, n_windows=len(wins),
                   window_s=n / fs, peak_candidate_hz=fpk,
                   note="resolution_hz is the bin width of the window: it is the honest frequency uncertainty",
                   limits={"ringing_amp_max_dps": LIMITS["ringing_amp_max_dps"]})


def _estimate_hall(run: Run, p: _Prep) -> dict:
    h = run.hall
    ev = run.events
    if len(h["t_ns"]) == 0:
        return _insufficient("hall.csv has no edges")
    dpu = float(run.meta.get("deg_per_ustep") or DEFAULT_DEG_PER_USTEP)
    # reference marks: HOMED events (position re-zeroed) and ZERO commands
    marks: list[tuple[int, str, float, int]] = []   # t_ns, kind, offset, dir
    home_cmds = [(int(t), c.split()) for t, c in zip(run.cmd_t_ns, run.cmd) if c.upper().startswith("HOME")]
    for t, code, arg in zip(ev["t_ns"], ev["code"], ev["arg"]):
        if code.upper() == "HOMED":
            prior = [c for tc, c in home_cmds if tc <= t]
            try:
                direction = int(float(prior[-1][1])) if prior and len(prior[-1]) > 1 else 0
            except ValueError:
                direction = 0
            marks.append((int(t), "HOMED", float(arg), direction))
    for t, c in zip(run.cmd_t_ns, run.cmd):
        if c.split() and c.split()[0].upper() == "ZERO":
            marks.append((int(t), "ZERO", 0.0, 0))
    marks.sort()
    mark_t = np.array([m[0] for m in marks], dtype=np.int64)

    # edges: epoch, level, motion direction
    tt = (h["t_ns"] - p.t0_ns).astype(np.float64) * 1e-9
    rate = np.interp(tt, p.tg, p.rg)
    epoch = np.searchsorted(mark_t, h["t_ns"], side="left") if len(marks) else np.zeros(len(tt), dtype=int)
    referenced = [False] + [m[1] == "HOMED" for m in marks]   # epoch k+1 is referenced iff mark k is a HOMED
    pos = h["pos_ustep"]
    groups: dict[tuple, list[float]] = {}
    pooled: dict[tuple, list[float]] = {}
    for i in range(len(tt)):
        if not (math.isfinite(pos[i]) and math.isfinite(h["level"][i])):
            continue
        direction = 0 if abs(rate[i]) < 0.2 else (1 if rate[i] > 0 else -1)
        key = (int(epoch[i]), int(h["level"][i]), direction)
        groups.setdefault(key, []).append(float(pos[i]))
        if referenced[int(epoch[i])]:
            pooled.setdefault((int(h["level"][i]), direction), []).append(float(pos[i]))
    rows = []
    worst = 0.0
    stds = []
    for scope, src in (("epoch", groups), ("referenced", pooled)):
        for key, v in src.items():
            if len(v) < 2:
                continue
            rng = (max(v) - min(v)) * dpu
            worst = max(worst, rng)
            stds.append(float(np.std(v, ddof=1)) * dpu)
            lvl, direction = key[-2], key[-1]
            rows.append({"scope": scope, "level": lvl, "direction": direction, "n": len(v),
                         "range_deg": rng, "std_deg": stds[-1], "mean_pos_deg": float(np.mean(v)) * dpu,
                         **({"epoch": key[0]} if scope == "epoch" else {})})
    # HOMED offsets with a previous same-direction HOMED and no ZERO in between = accumulated step error
    offsets = []
    for k, (t, kind, off, direction) in enumerate(marks):
        if kind != "HOMED":
            continue
        prev = marks[k - 1] if k > 0 else None
        valid = (prev is not None and prev[1] == "HOMED" and prev[3] == direction and direction != 0
                 and referenced[k])
        offsets.append({"t_s": (t - p.t0_ns) * 1e-9, "offset_ustep": off, "offset_deg": off * dpu,
                        "dir": direction, "referenced": bool(valid),
                        "lost_full_steps": off * dpu / FULL_STEP_DEG if valid else None})
    valid_offs = [o for o in offsets if o["referenced"]]
    if not rows and not valid_offs:
        return _insufficient("no repeated same-direction hall edges and no referenced HOMED offsets "
                             "(%d edges, %d HOMED events)" % (len(tt), len(offsets)))
    worst_off = max((abs(o["offset_deg"]) for o in valid_offs), default=0.0)
    checks = {"repeatability": worst <= LIMITS["hall_repeat_max_deg"],
              "homed_offsets": worst_off <= LIMITS["homed_offset_max_deg"]}
    return _status(all(checks.values()), repeatability_deg=worst, std_deg=float(np.sqrt(np.mean(np.square(stds))))
                   if stds else math.nan, worst_homed_offset_deg=worst_off if valid_offs else None, groups=rows,
                   homed=offsets, n_edges=len(tt), n_referenced_offsets=len(valid_offs), checks=checks,
                   lost_full_steps_est=(round(max((o["lost_full_steps"] for o in valid_offs),
                                                  key=abs)) if valid_offs else None),
                   note="rows scope=epoch: spread of same-direction edges between reference events; "
                        "scope=referenced: pooled over epochs that start at a HOMED event",
                   limits={k: LIMITS[k] for k in ("hall_repeat_max_deg", "homed_offset_max_deg")})


# ------------------------------------------------------------------------------------------- analyze_run
def analyze_run(run_dir: str | Path) -> dict:
    """Run every estimate on one run directory. Never raises for missing data of a single estimate."""
    run = load_run(run_dir)
    p = _prepare(run)
    warns = list(p.warnings)
    est: dict[str, dict] = {}
    pose = run.meta.get("pose")
    if p.rig_info["source"] != "t_us_fit":
        warns.append("rig times taken from t_ns directly (t_us missing/inconsistent)")
    if not p.moves:
        warns.append("no rig motion found: only bias/gravity/hall can be estimated")

    bias_est, gravity = _estimate_bias_gravity(p)
    est["gyro_bias"], est["gravity"] = bias_est, gravity
    bias = np.asarray(bias_est["value_rad_s"]) if bias_est["status"] == "ok" else np.zeros(3)
    if bias_est["status"] != "ok":
        warns.append("no bias estimate: motion analysis uses zero bias (scale/lag still valid, "
                     "tracking pessimistic)")
    bias_sigma = float(np.max(bias_est["sigma_rad_s"])) if bias_est["status"] == "ok" else 0.0

    axis_est: dict
    y = None
    d_lag = 0.0
    mask = _move_mask(p, 0.3)
    if int(mask.sum()) < 100:
        for k in ("yaw_axis", "time_offset", "gyro_scale", "tracking", "ringing"):
            est[k] = _insufficient("no rig motion")
    else:
        fit0 = _axis_fit(p, bias, 0.0, mask)
        i_ax = int(np.argmax(np.abs(fit0["k"])))
        sign = 1.0 if fit0["k"][i_ax] >= 0 else -1.0
        y = sign * (p.gyro[:, i_ax] - bias[i_ax])
        clock = _clock_info(run.meta)
        resid_us = float(clock.get("resid_us", 0.0) or 0.0)
        est["time_offset"] = _estimate_time_offset(p, y, resid_us)
        if est["time_offset"]["status"] == "ok":
            d_lag = float(est["time_offset"]["late_s"])
        fit = _axis_fit(p, bias, d_lag, mask)
        est["yaw_axis"] = _axis_estimate(fit, i_ax, sign, gravity)
        cos_tilt = math.cos(math.radians(est["yaw_axis"]["tilt_deg"]))
        est["gyro_scale"] = _estimate_scale(p, y, d_lag, bias_sigma, cos_tilt)
        factor = est["gyro_scale"].get("factor", 1.0) if est["gyro_scale"]["status"] == "ok" else 1.0
        est["gyro_scale"]["chip_axis"] = "xyz"[i_ax]
        est["gyro_scale"]["chip_axis_index"] = i_ax
        est["gyro_scale"]["chip_axis_sign"] = int(sign)
        est["tracking"] = _estimate_tracking(p, y, d_lag, float(factor))
        est["ringing"] = _estimate_ringing(p, y, d_lag)
    est["hall_repeatability"] = _estimate_hall(run, p)
    if bias_est["status"] != "ok":   # scale / tracking integrate the bias error: their numbers are not trustworthy
        for k in ("gyro_scale", "tracking"):
            if est[k]["pass"] is not None:
                est[k].update(pass_was=est[k]["pass"], status="unreliable_no_bias", **{"pass": None})

    core = ("gyro_bias", "yaw_axis", "gyro_scale", "time_offset")
    checks = [{"name": k, "pass": est[k].get("pass")} for k in
              ("gyro_bias", "gravity", "yaw_axis", "gyro_scale", "time_offset", "tracking", "ringing",
               "hall_repeatability") if k in est]
    if any(est[k]["pass"] is False for k in est):
        verdict = "FAIL"
    elif any(est[k]["pass"] is None for k in core):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    dur = float(max(p.ti[-1], p.tr[-1]) - min(p.ti[0], p.tr[0]))
    return _clean({
        "kind": "run", "run_dir": str(run.path), "procedure": run.meta.get("procedure"), "pose": pose,
        "meta": {"R_gun_imu_in_use": (run.meta.get("imu") or {}).get("R_gun_imu"),
                 "deg_per_ustep": run.meta.get("deg_per_ustep"), "start": run.meta.get("start")},
        "verdict": verdict, "checks": checks, "warnings": warns,
        "stats": {"duration_s": dur, "n_imu": len(p.ti), "n_rig": len(p.tr),
                  "imu_rate_hz": (len(p.ti) - 1) / float(p.ti[-1] - p.ti[0]),
                  "rig_rate_hz": (len(p.tr) - 1) / float(p.tr[-1] - p.tr[0]), "rig_clock": p.rig_info,
                  "n_moves": len(p.moves), "n_still": len(p.stills)},
        "segments": {"moves": [{"t0": m["t0"], "t1": m["t1"], "peak_dps": m["peak_dps"]} for m in p.moves],
                     "still": [{"t0": a, "t1": b} for a, b in p.stills]},
        "estimates": est, "limits": LIMITS,
    })


def _axis_estimate(fit: dict, i_ax: int, sign: float, gravity: dict) -> dict:
    k, sk = fit["k"], fit["sigma_k"]
    mag = float(np.linalg.norm(k))
    khat = k / mag
    off = [j for j in range(3) if j != i_ax]
    leak = [float(abs(k[j] / k[i_ax])) * 100.0 for j in off]
    tilt = math.degrees(math.hypot(k[off[0]], k[off[1]]) / abs(k[i_ax]))
    sig_tilt = math.degrees(math.hypot(sk[off[0]], sk[off[1]]) / abs(k[i_ax]))
    ax_grav = math.nan
    if gravity.get("status") == "ok":
        ax_grav = _angle_deg(khat, np.asarray(gravity["dir"]))
    checks = {"tilt": tilt <= LIMITS["axis_tilt_max_deg"],
              "gravity": (not math.isfinite(ax_grav)) or ax_grav <= LIMITS["axis_gravity_max_deg"]}
    out = _status(all(checks.values()), chip_axis="xyz"[i_ax], chip_axis_index=i_ax, sign=int(sign),
                  label=("+" if sign > 0 else "-") + "xyz"[i_ax], direction=khat, magnitude=mag,
                  k=k, sigma_k=sk, tilt_deg=tilt, tilt_sigma_deg=sig_tilt,
                  leakage_pct={"xyz"[j]: l for j, l in zip(off, leak)}, axis_vs_gravity_deg=ax_grav,
                  dominance=float(abs(k[i_ax]) / (np.sort(np.abs(k))[-2] + 1e-30)), checks=checks,
                  note="tilt = angle between the measured table axis and the nearest chip axis "
                       "(biased high by ~sigma for small tilts)",
                  limits={k2: LIMITS[k2] for k2 in ("axis_tilt_max_deg", "axis_gravity_max_deg")})
    return out


# ------------------------------------------------------------------------------------------- combine
def combine_runs(run_dirs: list, R_config: Any = None) -> dict:
    """Several runs (different poses) -> full chip->gun rotation and aggregated bias / scale / time offset.

    Entries may be run directories or already-analysed result dicts. `R_config` overrides the config
    imu.R_gun_imu used for the comparison (default: tbot.config.load_config()).
    """
    results = [r if isinstance(r, dict) else analyze_run(r) for r in run_dirs]
    warns: list[str] = []
    rows = []
    for r in results:
        e = r["estimates"]
        ya, bi, gr = e.get("yaw_axis", {}), e.get("gyro_bias", {}), e.get("gravity", {})
        if ya.get("status") != "ok":
            warns.append(f"{r['run_dir']}: no yaw axis estimate, skipped")
            continue
        row = {"run_dir": r["run_dir"], "pose": r.get("pose"), "axis": ya["label"],
               "axis_index": ya["chip_axis_index"], "axis_dir": ya["direction"], "tilt_deg": ya["tilt_deg"],
               "gravity_dir": gr.get("dir") if gr.get("status") == "ok" else None,
               "axis_vs_gravity_deg": ya["axis_vs_gravity_deg"], "verdict": r["verdict"]}
        if bi.get("status") == "ok":
            row["bias_rad_s"], row["bias_sigma"] = bi["value_rad_s"], bi["sigma_rad_s"]
            row["temp_c"] = bi.get("temp_c")
        sc = e.get("gyro_scale", {})
        if sc.get("status") == "ok":
            row["scale_factor"], row["scale_sigma"] = sc["factor"], sc["sigma_pct"] / 100.0
        to = e.get("time_offset", {})
        if to.get("status") == "ok":
            row["late_s"], row["late_sigma_s"] = to["late_s"], to["sigma_s"]
        rows.append(row)
    out: dict[str, Any] = {"kind": "combined", "runs": rows, "warnings": warns, "limits": LIMITS}

    # --- R_gun_imu by Kabsch over (chip direction -> gun up of the pose)
    pairs_a, pairs_b, labels = [], [], []
    for row in rows:
        up = POSE_UP_GUN.get(str(row["pose"]))
        if up is None:
            warns.append(f"{row['run_dir']}: unknown pose {row['pose']!r}, not used for R_gun_imu")
            continue
        up = np.asarray(up, dtype=np.float64)
        pairs_a.append(np.asarray(row["axis_dir"])), pairs_b.append(up), labels.append((row["pose"], "gyro"))
        if row["gravity_dir"] is not None:
            pairs_a.append(np.asarray(row["gravity_dir"])), pairs_b.append(up), labels.append((row["pose"], "accel"))
    fit = None
    ups = [np.asarray(POSE_UP_GUN[row["pose"]]) for row in rows if row["pose"] in POSE_UP_GUN]
    independent = any(np.linalg.norm(np.cross(u, v)) > 0.5 for u, v in itertools.combinations(ups, 2))
    if not independent:
        warns.append("need two runs whose table axes are not parallel in the gun frame (e.g. upright + "
                     "left_side_down) to fit R_gun_imu")
    else:
        A, B = np.array(pairs_a), np.array(pairs_b)
        R = _kabsch(A, B)
        resid = [_angle_deg(R @ a, b) for a, b in zip(A, B)]
        sel = lambda kind: [i for i, lb in enumerate(labels) if lb[1] == kind]
        sub = {}
        for kind in ("gyro", "accel"):
            idx = sel(kind)
            if len(idx) >= 2 and any(np.linalg.norm(np.cross(B[i], B[j])) > 0.5 for i, j in
                                     itertools.combinations(idx, 2)):
                sub[kind] = _kabsch(A[idx], B[idx])
        fit = {"R_gun_imu": R, "residual_deg": [{"pose": lb[0], "source": lb[1], "deg": r}
                                                 for lb, r in zip(labels, resid)],
               "residual_rms_deg": float(np.sqrt(np.mean(np.square(resid)))),
               "residual_max_deg": float(max(resid)), "n_pairs": len(resid),
               "gyro_only_vs_accel_only_deg": _rot_angle_deg(sub["gyro"] @ sub["accel"].T)
               if len(sub) == 2 else None,
               "note": "all table axes lie in the gun Y-Z plane for the four rig poses: the rotation is "
                       "determined (two non-parallel directions) but gun X is only fixed by handedness"}
        snap, snap_dev = _snap_axis_map(R)
        fit["snapped_axis_map"] = snap
        fit["snapped_deviation_deg"] = snap_dev
        fit["axis_map_text"] = _axis_map_text(snap)
        out["fit"] = fit
        out["compare"] = _compare_R(R, snap, R_config, results)
    # --- aggregated estimates
    out["gyro_bias"] = _agg_bias(rows)
    out["time_offset"] = _agg_time(rows)
    out["gyro_scale"] = _agg_scale(rows)
    chk = []
    if fit is not None:
        chk.append({"name": "R_fit_residual", "pass": fit["residual_rms_deg"] <= LIMITS["axis_fit_resid_max_deg"]})
        cmp_ = out["compare"]
        if cmp_.get("config_diff_deg") is not None:
            chk.append({"name": "R_matches_config", "pass": cmp_["config_diff_deg"] <= LIMITS["R_config_diff_max_deg"]})
    for k in ("gyro_bias", "time_offset", "gyro_scale"):
        if out[k].get("pass") is not None:
            chk.append({"name": k, "pass": out[k]["pass"]})
    out["checks"] = chk
    out["verdict"] = ("FAIL" if any(c["pass"] is False for c in chk) else
                      "INCOMPLETE" if fit is None else "PASS")
    out["poses"] = sorted({str(r["pose"]) for r in rows})
    return _clean(out)


def _snap_axis_map(R: np.ndarray) -> tuple[np.ndarray, float]:
    best, best_dev = None, 1e9
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = np.zeros((3, 3))
            for i, j in enumerate(perm):
                P[i, j] = signs[i]
            if np.linalg.det(P) < 0.5:
                continue
            dev = _rot_angle_deg(R @ P.T)
            if dev < best_dev:
                best, best_dev = P, dev
    return best, best_dev


def _axis_map_text(P: np.ndarray) -> list[str]:
    out = []
    for i, gun in enumerate("XYZ"):
        j = int(np.argmax(np.abs(P[i])))
        out.append(f"gun {gun} = {'+' if P[i, j] > 0 else '-'}chip {'XYZ'[j]}")
    return out


def _compare_R(R: np.ndarray, snap: np.ndarray, R_config: Any, results: list[dict]) -> dict:
    cfg = None
    src = "argument"
    if R_config is not None:
        cfg = np.asarray(R_config, dtype=np.float64)
    else:
        try:
            from tbot.config import load_config
            cfg = np.asarray(load_config().imu.R_gun_imu, dtype=np.float64)
            src = "tbot.config load_config().imu.R_gun_imu"
        except Exception as exc:  # config unavailable: report, do not guess
            src = f"unavailable ({exc})"
    ident = np.eye(3)
    out: dict[str, Any] = {
        "config_R": cfg, "config_source": src,
        "config_diff_deg": _rot_angle_deg(R @ cfg.T) if cfg is not None else None,
        "snapped_matches_config": bool(np.allclose(snap, cfg, atol=1e-6)) if cfg is not None else None,
        "layout_identity_diff_deg": _rot_angle_deg(R @ ident.T),
        "layout_note": "design/layout.md as-built IMU note: board axes = gun axes, i.e. identity board-to-gun; "
                       "chip orientation on the board is unconfirmed, so identity is a prior, not a reference",
    }
    in_use = next((np.asarray(r["meta"]["R_gun_imu_in_use"], dtype=np.float64) for r in results
                   if r.get("meta", {}).get("R_gun_imu_in_use")), None)
    out["in_use_R"] = in_use
    out["in_use_diff_deg"] = _rot_angle_deg(R @ in_use.T) if in_use is not None else None
    return out


def _wmean(vals: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = 1.0 / np.maximum(sig, 1e-12) ** 2
    mean = (w * vals).sum(axis=0) / w.sum(axis=0)
    sig_stat = 1.0 / np.sqrt(w.sum(axis=0))
    spread = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros_like(mean)
    return mean, np.maximum(sig_stat, spread / math.sqrt(len(vals))), spread


def _agg_bias(rows: list[dict]) -> dict:
    sel = [r for r in rows if "bias_rad_s" in r]
    if not sel:
        return _insufficient("no run has a bias estimate")
    v = np.array([r["bias_rad_s"] for r in sel])
    s = np.array([r["bias_sigma"] for r in sel])
    mean, sig, spread = _wmean(v, s)
    temps = [r["temp_c"] for r in sel if r.get("temp_c") is not None and math.isfinite(r["temp_c"])]
    ok = bool(np.all(sig <= LIMITS["gyro_bias_sigma_max_dps"] * DEG2RAD)
              and np.all(spread <= LIMITS["gyro_bias_spread_max_dps"] * DEG2RAD)
              and np.all(np.abs(mean) <= LIMITS["gyro_bias_abs_max_dps"] * DEG2RAD))
    return _status(ok, value_rad_s=mean, sigma_rad_s=sig, value_dps=mean / DEG2RAD, sigma_dps=sig / DEG2RAD,
                   spread_between_poses_dps=spread / DEG2RAD, n_runs=len(sel),
                   temp_c=float(np.mean(temps)) if temps else None,
                   note="spread between poses above noise = g-sensitivity or thermal drift")


def _agg_time(rows: list[dict]) -> dict:
    sel = [r for r in rows if "late_s" in r]
    if not sel:
        return _insufficient("no run has a time offset estimate")
    v = np.array([[r["late_s"]] for r in sel])
    s = np.array([[r["late_sigma_s"]] for r in sel])
    mean, sig, spread = _wmean(v, s)
    late, sig_s, spread_s = float(mean[0]), float(sig[0]), float(spread[0])
    ok = (abs(late) * 1e3 <= LIMITS["time_offset_abs_max_ms"] and sig_s * 1e3 <= LIMITS["time_offset_sigma_max_ms"])
    return _status(ok, late_s=late, late_ms=late * 1e3, sigma_s=sig_s, sigma_ms=sig_s * 1e3,
                   spread_ms=spread_s * 1e3, time_offset_ns=-int(round(late * 1e9)), n_runs=len(sel))


def _agg_scale(rows: list[dict]) -> dict:
    per_axis: dict[int, list[dict]] = {}
    for r in rows:
        if "scale_factor" in r:
            per_axis.setdefault(r["axis_index"], []).append(r)
    if not per_axis:
        return _insufficient("no run has a scale estimate")
    factor: list[float | None] = [None, None, None]
    sigma: list[float | None] = [None, None, None]
    for i, rs in per_axis.items():
        v = np.array([[r["scale_factor"]] for r in rs])
        s = np.array([[max(r["scale_sigma"], 1e-6)] for r in rs])
        m, sg, _ = _wmean(v, s)
        factor[i], sigma[i] = float(m[0]), float(sg[0])
    have = [i for i in range(3) if factor[i] is not None]
    ok = all(abs(factor[i] - 1.0) * 100 <= LIMITS["scale_abs_max_pct"]
             and sigma[i] * 100 <= LIMITS["scale_sigma_max_pct"] for i in have)
    return _status(ok, factor=factor, sigma=sigma, correction=[None if f is None else 1.0 / f for f in factor],
                   axes_measured=["xyz"[i] for i in have],
                   axes_missing=["xyz"[i] for i in range(3) if factor[i] is None],
                   note="a chip axis is only measured when it carries the table rate in some pose; the four rig "
                        "poses roll about the barrel, so the chip axis along the barrel is never measured")


# ------------------------------------------------------------------------------------------- proposals
def propose_calib(result: dict) -> dict:
    """calib/imu.json keys for a run result or a combined result. Keys starting with "_" are notes only.

    gyro_bias_rad_s [3] (chip frame) and time_offset_ns come from any result; R_gun_imu needs a combined
    result with a good fit; gyro_scale [3] (multiplier applied to the chip gyro) needs all three axes and is not
    consumed by tbot.imu.mpu6050.ImuCalibration yet.
    """
    prop: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    if result.get("kind") == "combined":
        bias, to, sc = result.get("gyro_bias", {}), result.get("time_offset", {}), result.get("gyro_scale", {})
    else:
        e = result["estimates"]
        bias, to, sc = e.get("gyro_bias", {}), e.get("time_offset", {}), e.get("gyro_scale", {})
    if bias.get("status") == "ok":
        sig = max(bias["sigma_dps"])
        if sig <= LIMITS["gyro_bias_sigma_max_dps"]:
            prop["gyro_bias_rad_s"] = list(bias["value_rad_s"])
            if bias.get("temp_c") is not None:
                prop["temp_c"] = bias["temp_c"]
        else:
            blocked["gyro_bias_rad_s"] = f"sigma {sig:.3f} deg/s above limit"
    else:
        blocked["gyro_bias_rad_s"] = bias.get("reason", "no estimate")
    if "late_s" in to:
        if to["sigma_ms"] <= LIMITS["time_offset_sigma_max_ms"] and not to.get("at_search_limit") and \
                to.get("correlation", 1.0) >= LIMITS["time_offset_corr_min"]:
            prop["time_offset_ns"] = int(to["time_offset_ns"])
        else:
            blocked["time_offset_ns"] = "uncertainty/correlation/search limit check failed"
    else:
        blocked["time_offset_ns"] = to.get("reason", "no estimate")
    if result.get("kind") == "combined":
        fit = result.get("fit")
        if fit and fit["residual_rms_deg"] <= LIMITS["axis_fit_resid_max_deg"]:
            prop["R_gun_imu"] = fit["R_gun_imu"]
        else:
            blocked["R_gun_imu"] = "no fit" if not fit else "fit residual above limit"
        if sc.get("status") == "ok" and not sc.get("axes_missing"):
            prop["gyro_scale"] = list(sc["correction"])
        else:
            blocked["gyro_scale"] = ("axes not measured: %s" % ",".join(sc.get("axes_missing", []))
                                     if sc.get("status") == "ok" else sc.get("reason", "no estimate"))
    else:
        blocked["R_gun_imu"] = "needs combine_runs over >= 2 poses"
        blocked["gyro_scale"] = "needs combine_runs (a single run measures one chip axis only)"
    prop["_blocked"] = blocked
    prop["_notes"] = ["gyro_bias_rad_s is in the CHIP frame; time_offset_ns is ADDED to IMU timestamps "
                      "(-late); gyro_scale multiplies the chip gyro (1/factor)"]
    return prop


def apply_to_calib(proposal: dict, calib_dir: str | Path, dry_run: bool = True) -> dict:
    """Merge the proposal into <calib_dir>/imu.json with tbot.config.update_json -- only when dry_run=False."""
    from tbot.config import now_iso, resolve_path, update_json

    patch = {k: v for k, v in proposal.items() if not k.startswith("_")}
    path = resolve_path(Path(calib_dir) / "imu.json")
    if not patch:
        return {"path": str(path), "patch": {}, "written": False, "result": None}
    patch["created"] = now_iso()
    if dry_run:
        current = json.loads(path.read_text()) if path.exists() else {}
        return {"path": str(path), "patch": _clean(patch), "written": False, "result": {**current, **_clean(patch)}}
    return {"path": str(path), "patch": _clean(patch), "written": True, "result": update_json(path, patch)}


# ------------------------------------------------------------------------------------------- report
def write_report(result: dict, run_dir: str | Path) -> Path:
    """Write report.json and report.md into run_dir; returns the path of report.md."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    clean = _clean(result)
    (d / "report.json").write_text(json.dumps(clean, indent=2) + "\n")
    md = d / "report.md"
    md.write_text(_render_md(clean))
    return md


def _pf(v: Any) -> str:
    return {True: "PASS", False: "FAIL", None: "n/a"}[v]


def _v3(v: Any, f: str = "%+.4f") -> str:
    return "[" + ", ".join(f % x for x in v) + "]" if v is not None else "n/a"


def _render_md(r: dict) -> str:
    L: list[str] = []
    if r.get("kind") == "combined":
        L += ["# CalRig B combined analysis", "", f"Verdict: **{r['verdict']}**  poses: {', '.join(r['poses'])}", ""]
        L += ["| run | pose | yaw axis | tilt deg | axis-vs-gravity deg |", "|---|---|---|---|---|"]
        for x in r["runs"]:
            L.append("| %s | %s | %s | %.2f | %s |" % (Path(x["run_dir"]).name, x["pose"], x["axis"], x["tilt_deg"],
                                                      "n/a" if x["axis_vs_gravity_deg"] is None
                                                      else "%.2f" % x["axis_vs_gravity_deg"]))
        fit = r.get("fit")
        if fit:
            L += ["", "## R_gun_imu (gun = R @ chip)", "", "```"] + [_v3(row, "%+.5f") for row in fit["R_gun_imu"]] + [
                "```", "", "Axis map: " + "; ".join(fit["axis_map_text"]) + "  (snapped deviation %.2f deg)" %
                fit["snapped_deviation_deg"],
                "", "Residual rms %.2f deg, max %.2f deg over %d pairs." % (
                    fit["residual_rms_deg"], fit["residual_max_deg"], fit["n_pairs"])]
            c = r["compare"]
            if c.get("config_diff_deg") is not None:
                L.append("Difference to config R_gun_imu: %.2f deg (snapped map %s config)." % (
                    c["config_diff_deg"], "equals" if c["snapped_matches_config"] else "DIFFERS from"))
            L.append("Difference to layout.md identity (as-built board axes = gun axes): %.1f deg. %s" % (
                c["layout_identity_diff_deg"], c["layout_note"]))
        b = r["gyro_bias"]
        if b["status"] == "ok":
            L += ["", "Gyro bias (chip, deg/s): %s +- %s, spread between poses %s" % (
                _v3(b["value_dps"]), _v3(b["sigma_dps"], "%.4f"), _v3(b["spread_between_poses_dps"], "%.4f"))]
        t = r["time_offset"]
        if t["status"] == "ok":
            L.append("Time offset: IMU stamps late by %+.3f ms +- %.3f (spread %.3f ms)" % (
                t["late_ms"], t["sigma_ms"], t["spread_ms"]))
        s = r["gyro_scale"]
        if s["status"] == "ok":
            L.append("Gyro scale factor per chip axis (xyz): %s; not measured: %s" % (
                ", ".join("n/a" if f is None else "%.4f" % f for f in s["factor"]), s["axes_missing"] or "none"))
    else:
        e = r["estimates"]
        L += ["# CalRig B run report", "", f"Run `{Path(r['run_dir']).name}`  procedure {r['procedure']}  pose "
              f"{r['pose']}  verdict **{r['verdict']}**", ""]
        st = r["stats"]
        L.append("%.1f s, %d IMU samples (%.0f Hz), %d rig rows (%.0f Hz), %d moves, %d still segments; rig clock "
                 "source %s." % (st["duration_s"], st["n_imu"], st["imu_rate_hz"], st["n_rig"], st["rig_rate_hz"],
                                 st["n_moves"], st["n_still"], st["rig_clock"]["source"]))
        for w in r["warnings"]:
            L.append(f"- WARNING: {w}")
        L += ["", "| check | result |", "|---|---|"] + [f"| {c['name']} | {_pf(c['pass'])} |" for c in r["checks"]]
        b = e["gyro_bias"]
        L += ["", "## Gyro bias (chip frame)"]
        if b["status"] == "ok":
            L += ["", "%s deg/s +- %s (%d still segments, %.1f s); noise %s deg/s; spread %s" % (
                _v3(b["value_dps"]), _v3(b["sigma_dps"], "%.4f"), b["n_segments"], b["total_s"],
                _v3(b["noise_dps"], "%.3f"), _v3(b["spread_dps"], "%.4f")),
                  "Temperature %s C." % ("n/a" if b["temp_c"] is None else "%.2f" % b["temp_c"])]
        else:
            L += ["", b["reason"]]
        g = e["gravity"]
        if g["status"] == "ok":
            L.append("Gravity (chip): %s m/s^2, |a| = %.4f g." % (_v3(g["mean_m_s2"]), g["norm_g"]))
        y = e.get("yaw_axis", {})
        L += ["", "## Yaw axis"]
        if y.get("status") == "ok":
            L += ["", "Table rate on chip %s (|k| = %.4f); tilt %.2f +- %.2f deg; leakage %%: %s; axis vs gravity %s deg." % (
                y["label"], y["magnitude"], y["tilt_deg"], y["tilt_sigma_deg"], y["leakage_pct"],
                "n/a" if y["axis_vs_gravity_deg"] is None else "%.2f" % y["axis_vs_gravity_deg"])]
        else:
            L += ["", y.get("reason", "n/a")]
        s = e.get("gyro_scale", {})
        L += ["", "## Gyro scale"]
        if s.get("status") == "ok":
            L += ["", "factor %.5f (%+.3f %%) +- %.3f %%, spread %.3f %%, %d moves (%s), correction x%.5f. CCW %s CW %s." % (
                s["factor"], s["error_pct"], s["sigma_pct"], s["spread_pct"], s["n_moves"], s["method"],
                s["correction"], s["per_direction"].get("ccw"), s["per_direction"].get("cw"))]
        else:
            L += ["", s.get("reason", "n/a")]
        t = e.get("time_offset", {})
        L += ["", "## Time offset (positive = IMU timestamps late)"]
        if t.get("status") == "ok":
            L += ["", "%+.3f ms +- %.3f ms (corr %.4f) -> time_offset_ns %d." % (
                t["late_ms"], t["sigma_ms"], t["correlation"], t["time_offset_ns"])]
        else:
            L += ["", t.get("reason", "n/a")]
        tr = e.get("tracking", {})
        L += ["", "## Rig tracking (gyro angle vs rig angle)"]
        if tr.get("status") == "ok":
            L += ["", "RMS %.3f deg, max %.3f deg (scale removed; raw RMS %.3f deg)." % (
                tr["rms_deg"], tr["max_deg"], tr["raw_rms_deg"])]
        else:
            L += ["", tr.get("reason", "n/a")]
        rg = e.get("ringing", {})
        L += ["", "## Post-move ringing"]
        if rg.get("status") == "ok":
            L += ["", ("%.2f Hz (bin width %.2f Hz), amplitude %.2f deg/s in the first %.1f s (window mean %.2f), "
                      "SNR %.1f" % (rg["freq_hz"], rg["resolution_hz"], rg["amp_initial_dps"], RING_HEAD_S,
                                    rg["amp_dps"], rg["snr"])) if rg["visible"]
                  else "no ringing above the noise floor (peak %.3f deg/s, SNR %.1f)." % (rg["amp_dps"], rg["snr"])]
        else:
            L += ["", rg.get("reason", "n/a")]
        h = e.get("hall_repeatability", {})
        L += ["", "## Hall repeatability / lost steps"]
        if h.get("status") == "ok":
            L += ["", "edge repeatability %.3f deg; worst referenced HOMED offset %s deg; estimated lost full steps %s." % (
                h["repeatability_deg"], h["worst_homed_offset_deg"], h["lost_full_steps_est"])]
        else:
            L += ["", h.get("reason", "n/a")]
    L += ["", "## Limits", "", "| limit | value | basis |", "|---|---|---|"]
    for k, v in LIMITS.items():
        L.append(f"| {k} | {v} | {LIMIT_BASIS[k]} |")
    return "\n".join(L) + "\n"
