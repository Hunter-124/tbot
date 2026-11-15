"""`tbot calib-imu {bias,axes,sync}` — IMU bias, axis map and camera-IMU time offset -> calib/imu.json.

Contract: docs/ARCHITECTURE.md [imu]. All three sub-tools write calib/imu.json keys through update_json,
so each one only touches what it measured.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tbot.config import Calibration, add_common_args, load_from_args, now_iso, resolve_path, update_json
from tbot.imu.attitude import log_so3
from tbot.imu.mpu6050 import G_MPS2, Mpu6050
from tbot.stats import summarize
from tbot.types import ImuSample

__all__ = ["main", "estimate_time_offset_s", "kabsch", "camera_rotation_rate", "is_still_pose"]

# Optical-flow settings for the sync measurement (textured scene, swinging gun).
MAX_CORNERS = 200
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE = 8
LK_WIN = (21, 21)
LK_LEVELS = 3
LK_MAX_ITER = 30
LK_EPS = 0.01
FORWARD_BACKWARD_TOL_PX = 1.0
MIN_TRACKED_POINTS = 8
OUTLIER_FACTOR = 3.0
MIN_SYNC_SAMPLES = 8

# --auto pose capture: a pose counts once it is held this still and differs from the earlier ones.
STILL_GYRO_DPS = 3.0
STILL_ACCEL_STD = 0.25        # m/s^2, per axis
STILL_G_TOL = 0.08
NEW_POSE_DEG = 45.0
MOVED_DPS = 15.0              # a pose is only taken after the gun has moved this fast since the prompt

AXES_POSES = (
    ("LEVEL: gun upright and level (rib up)", np.array([0.0, 0.0, 1.0])),
    ("MUZZLE UP: point the muzzle straight up", np.array([1.0, 0.0, 0.0])),
    ("LEFT SIDE DOWN: roll the gun 90 deg onto its left side", np.array([0.0, -1.0, 0.0])),
)


# ------------------------------------------------------------------------------------ pure helpers
def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R a_i - b_i| for paired direction sets ((N,3) each)."""
    A = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if A.shape != B.shape or A.shape[0] < 2:
        raise ValueError(f"kabsch needs matching point sets with >= 2 rows, got {A.shape} and {B.shape}")
    U, _, Vt = np.linalg.svd(B.T @ A)
    d = 1.0 if np.linalg.det(U @ Vt) > 0.0 else -1.0
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _kabsch_robust(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kabsch with one median-based outlier rejection pass (moving objects, bad LK tracks)."""
    R = kabsch(a, b)
    resid = np.arccos(np.clip(np.sum((a @ R.T) * b, axis=1), -1.0, 1.0))
    thr = max(OUTLIER_FACTOR * float(np.median(resid)), math.radians(0.2))
    keep = resid <= thr
    if MIN_TRACKED_POINTS <= int(keep.sum()) < a.shape[0]:
        R = kabsch(a[keep], b[keep])
    return R


def camera_rotation_rate(img0: np.ndarray, img1: np.ndarray, dt_s: float, cam: Any,
                         mask: np.ndarray | None = None) -> np.ndarray | None:
    """Angular velocity (rad/s, CAMERA frame) between two frames from LK flow on distant scene points.

    Returns None when too few points survive. Scene points are world-fixed, so the rotation that maps
    the rays of frame 0 onto the rays of frame 1 is the inverse of the camera's own rotation.
    """
    import cv2

    if dt_s <= 0.0:
        return None
    p0 = cv2.goodFeaturesToTrack(img0, maxCorners=MAX_CORNERS, qualityLevel=CORNER_QUALITY,
                                 minDistance=CORNER_MIN_DISTANCE, mask=mask)
    if p0 is None or len(p0) < MIN_TRACKED_POINTS:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, LK_MAX_ITER, LK_EPS)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, winSize=LK_WIN,
                                          maxLevel=LK_LEVELS, criteria=criteria)
    if p1 is None:
        return None
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, winSize=LK_WIN,
                                           maxLevel=LK_LEVELS, criteria=criteria)
    if p0b is None:
        return None
    a = p0.reshape(-1, 2).astype(np.float64)
    b = p1.reshape(-1, 2).astype(np.float64)
    back = p0b.reshape(-1, 2).astype(np.float64)
    good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    good &= np.hypot(a[:, 0] - back[:, 0], a[:, 1] - back[:, 1]) < FORWARD_BACKWARD_TOL_PX
    if int(good.sum()) < MIN_TRACKED_POINTS:
        return None
    r0 = cam.pixel_to_ray(a[good])
    r1 = cam.pixel_to_ray(b[good])
    finite = np.all(np.isfinite(r0), axis=1) & np.all(np.isfinite(r1), axis=1)
    if int(finite.sum()) < MIN_TRACKED_POINTS:
        return None
    R = _kabsch_robust(r0[finite], r1[finite])
    return -log_so3(R) / float(dt_s)


def estimate_time_offset_s(cam_t_s, cam_rates, imu_t_s, imu_rates,
                           max_lag_s: float = 0.15, lag_step_s: float = 0.001) -> tuple[float, float]:
    """Lag to ADD to the IMU timestamps so the gyro lines up with the camera-derived rates.

    Both rate signals must be in the same frame (gun). Returns (offset_s, normalised correlation at the
    peak, 0..1), the peak refined by parabolic interpolation of the correlation curve.
    """
    ct = np.asarray(cam_t_s, dtype=np.float64).reshape(-1)
    cr = np.asarray(cam_rates, dtype=np.float64).reshape(ct.size, -1)
    it = np.asarray(imu_t_s, dtype=np.float64).reshape(-1)
    ir = np.asarray(imu_rates, dtype=np.float64).reshape(it.size, -1)
    if it.size < 2:
        raise ValueError("need at least 2 IMU samples")
    if cr.shape[1] != ir.shape[1]:
        raise ValueError(f"rate signals have different widths: {cr.shape[1]} vs {ir.shape[1]}")
    order = np.argsort(it)
    it, ir = it[order], ir[order]
    valid = np.isfinite(ct) & np.all(np.isfinite(cr), axis=1)
    valid &= (ct >= it[0] + max_lag_s) & (ct <= it[-1] - max_lag_s)
    if int(valid.sum()) < MIN_SYNC_SAMPLES:
        raise ValueError(f"only {int(valid.sum())} usable camera rate samples inside the IMU window "
                         f"(need {MIN_SYNC_SAMPLES})")
    ct, cr = ct[valid], cr[valid]
    a = (cr - cr.mean(axis=0)).reshape(-1)
    na = float(np.linalg.norm(a))
    if na <= 0.0:
        raise ValueError("the camera rate signal is constant: rotate the gun while recording")
    steps = int(round(float(max_lag_s) / float(lag_step_s)))
    if steps < 1:
        raise ValueError("max_lag_s must be at least one lag step")
    lags = np.arange(-steps, steps + 1, dtype=np.float64) * float(lag_step_s)
    scores = np.zeros(lags.size, dtype=np.float64)
    sampled = np.empty_like(cr)
    for i, lag in enumerate(lags):
        t_query = ct - lag
        for ax in range(cr.shape[1]):
            sampled[:, ax] = np.interp(t_query, it, ir[:, ax])
        b = (sampled - sampled.mean(axis=0)).reshape(-1)
        nb = float(np.linalg.norm(b))
        scores[i] = float(a @ b) / (na * nb) if nb > 0.0 else 0.0
    i = int(np.argmax(scores))
    if 0 < i < lags.size - 1:
        s0, s1, s2 = float(scores[i - 1]), float(scores[i]), float(scores[i + 1])
        den = s0 - 2.0 * s1 + s2
        delta = 0.5 * (s0 - s2) / den if den != 0.0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        return float(lags[i] + delta * float(lag_step_s)), s1 - 0.25 * (s0 - s2) * delta
    return float(lags[i]), float(scores[i])


# ------------------------------------------------------------------------------------ device helpers
def _uncalibrated(calib: Calibration) -> Calibration:
    """A copy of calib whose IMU calibration is neutral: samples come out raw, in the CHIP frame."""
    return dataclasses.replace(calib, imu={
        "gyro_bias_rad_s": [0.0, 0.0, 0.0],
        "accel_bias_m_s2": [0.0, 0.0, 0.0],
        "accel_scale": [1.0, 1.0, 1.0],
        "R_gun_imu": np.eye(3).tolist(),
        "time_offset_ns": 0,
    })


def _capture(imu: Mpu6050, seconds: float, settle_s: float = 0.25) -> dict[str, np.ndarray]:
    """Drain `seconds` of samples after a settle window. Arrays: t_ns, gyro, accel, temp_c."""
    end = time.monotonic() + settle_s
    while time.monotonic() < end:
        imu.read()
        time.sleep(0.01)
    imu.read()
    rows: list[ImuSample] = []
    end = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < end:
        rows.extend(imu.read())
        time.sleep(0.02)
    rows.extend(imu.read())
    if not rows:
        raise RuntimeError("no IMU samples arrived: check the wiring and `i2cdetect -y 1`")
    return {
        "t_ns": np.array([s.t_ns for s in rows], dtype=np.int64),
        "gyro": np.array([np.asarray(s.gyro, dtype=np.float64) for s in rows], dtype=np.float64),
        "accel": np.array([np.asarray(s.accel, dtype=np.float64) for s in rows], dtype=np.float64),
        "temp_c": np.array([s.temp_c for s in rows], dtype=np.float64),
    }


def is_still_pose(gyro: np.ndarray, accel: np.ndarray, bias: np.ndarray,
                  prior_dirs: list[np.ndarray]) -> bool:
    """True when a window of chip-frame samples is a held pose that differs from every earlier one.

    Held: bias-removed gyro below STILL_GYRO_DPS on average, accel steady (per-axis std below
    STILL_ACCEL_STD) and |a| within STILL_G_TOL of g. New: gravity more than NEW_POSE_DEG away from every
    direction in prior_dirs (unit vectors).
    """
    if gyro.shape[0] < 10:
        return False
    rate_dps = float(np.degrees(np.linalg.norm(gyro - bias, axis=1)).mean())
    if rate_dps > STILL_GYRO_DPS or float(accel.std(axis=0).max()) > STILL_ACCEL_STD:
        return False
    m = accel.mean(axis=0)
    n = float(np.linalg.norm(m))
    if abs(n / G_MPS2 - 1.0) > STILL_G_TOL:
        return False
    return all(_angle_deg(m, d) > NEW_POSE_DEG for d in prior_dirs)


def _capture_still_pose(imu: Mpu6050, seconds: float, bias: np.ndarray, prior_dirs: list[np.ndarray],
                        timeout_s: float) -> dict[str, np.ndarray] | None:
    """Stream until the last `seconds` of samples are a held, new pose (is_still_pose); return that window."""
    rows: list[ImuSample] = []
    t_end = time.monotonic() + timeout_s
    t_status = time.monotonic()
    moved = False
    imu.read()
    while time.monotonic() < t_end:
        time.sleep(0.05)
        rows.extend(imu.read())
        if not rows:
            continue
        horizon = int(rows[-1].t_ns) - int(seconds * 1e9)
        while rows and int(rows[0].t_ns) < horizon:
            rows.pop(0)
        span_s = (int(rows[-1].t_ns) - int(rows[0].t_ns)) * 1e-9
        gyro = np.array([np.asarray(s.gyro, dtype=np.float64) for s in rows])
        accel = np.array([np.asarray(s.accel, dtype=np.float64) for s in rows])
        if time.monotonic() - t_status > 2.0:
            t_status = time.monotonic()
            m = accel.mean(axis=0)
            print(f"  ... gravity (chip) {_fmt_vec(m / max(np.linalg.norm(m), 1e-9), '%+6.2f')}  "
                  f"rate {float(np.degrees(np.linalg.norm(gyro - bias, axis=1)).mean()):5.1f} deg/s",
                  flush=True)
        if not moved:
            # only take a pose the gun was moved into after the prompt (not the one it is resting in)
            if float(np.degrees(np.linalg.norm(gyro[-10:] - bias, axis=1)).max()) > MOVED_DPS:
                moved = True
                print("  moving...", flush=True)
            continue
        if span_s >= 0.95 * seconds and is_still_pose(gyro, accel, bias, prior_dirs):
            print("  captured", flush=True)
            return {"t_ns": np.array([s.t_ns for s in rows], dtype=np.int64), "gyro": gyro, "accel": accel,
                    "temp_c": np.array([s.temp_c for s in rows], dtype=np.float64)}
    return None


def _fmt_vec(v: Iterable[float], fmt: str = "%+9.4f") -> str:
    return "[" + ", ".join(fmt % float(x) for x in v) + "]"


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cu = u / np.linalg.norm(u)
    cv = v / np.linalg.norm(v)
    return math.degrees(math.acos(float(np.clip(cu @ cv, -1.0, 1.0))))


# ------------------------------------------------------------------------------------ sub-commands
def _cmd_bias(args: argparse.Namespace) -> int:
    cfg, calib = load_from_args(args)
    imu = Mpu6050(cfg, _uncalibrated(calib))
    print(f"Keep the gun completely still for {args.seconds:.0f} s (muzzle resting, hands off).")
    try:
        imu.start()
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    try:
        data = _capture(imu, args.seconds)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        imu.stop()

    gyro, accel = data["gyro"], data["accel"]
    bias = gyro.mean(axis=0)
    span_s = float(data["t_ns"][-1] - data["t_ns"][0]) * 1e-9
    print(f"\n{len(gyro)} samples over {span_s:.2f} s "
          f"({len(gyro) / max(span_s, 1e-9):.1f} Hz, {imu.overflows} FIFO overflows, {imu.errors} I2C errors)")
    for i, axis in enumerate("xyz"):
        s = summarize(np.degrees(gyro[:, i]))
        print(f"  gyro {axis}: mean {s['mean']:+8.4f} deg/s   std {s['std']:7.4f}   "
              f"p95 {s['p95']:+8.4f}")
    worst = float(np.degrees(gyro.std(axis=0)).max())
    if worst > 1.0:
        print(f"  WARNING: gyro noise {worst:.2f} deg/s is high — was the gun really still?")

    a_mean = accel.mean(axis=0)
    norm = float(np.linalg.norm(a_mean))
    print(f"  accel mean (chip frame) {_fmt_vec(a_mean)} m/s^2")
    print(f"  |accel| = {norm:.4f} m/s^2 = {norm / G_MPS2:.4f} g "
          f"({100.0 * (norm / G_MPS2 - 1.0):+.2f} % off g)")
    if abs(norm / G_MPS2 - 1.0) > 0.05:
        print("  WARNING: accel magnitude is more than 5 % off g — run `calib-imu axes` and check "
              "the accel_scale/accel_bias entries in calib/imu.json")
    temps = data["temp_c"][np.isfinite(data["temp_c"])]
    temp = float(temps.mean()) if temps.size else float("nan")
    print(f"  die temperature {temp:.2f} C (gyro bias drifts with it: redo this when the pod is cold)")

    if args.no_write:
        print("\n--no-write: calib/imu.json not touched")
        return 0
    path = calib.path("imu.json")
    update_json(path, {"gyro_bias_rad_s": bias.tolist(), "temp_c": temp, "created": now_iso()})
    print(f"\nwrote gyro_bias_rad_s {_fmt_vec(bias, '%+9.6f')} -> {path}")
    return 0


def _cmd_axes(args: argparse.Namespace) -> int:
    cfg, calib = load_from_args(args)
    imu = Mpu6050(cfg, _uncalibrated(calib))
    R_current = np.asarray((calib.imu or {}).get("R_gun_imu", cfg.imu.R_gun_imu), dtype=np.float64)
    try:
        imu.start()
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    measured: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    bias = np.asarray((calib.imu or {}).get("gyro_bias_rad_s", [0.0, 0.0, 0.0]), dtype=np.float64)
    try:
        for prompt, want in AXES_POSES:
            if args.auto:
                print(f"\n{prompt} — hold it still; captures automatically after {args.seconds:.0f} s still",
                      flush=True)
                data = _capture_still_pose(imu, args.seconds, bias, measured, args.timeout)
                if data is None:
                    print(f"  error: no still, new pose within {args.timeout:.0f} s")
                    return 1
            else:
                input(f"\n{prompt} — hold it, then press Enter to capture {args.seconds:.0f} s: ")
                data = _capture(imu, args.seconds)
            m = data["accel"].mean(axis=0)
            n = float(np.linalg.norm(m))
            if n < 1.0:
                print(f"  error: |accel| = {n:.3f} m/s^2, the sensor is not reporting gravity")
                return 1
            got_gun = R_current @ (m / n)
            err = _angle_deg(got_gun, want)
            print(f"  chip frame {_fmt_vec(m)} m/s^2 (|a| = {n / G_MPS2:.3f} g)")
            print(f"  with the current R_gun_imu -> gun {_fmt_vec(got_gun)}, "
                  f"expected {_fmt_vec(want)}: {err:.2f} deg off")
            measured.append(m / n)
            expected.append(want / np.linalg.norm(want))
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    finally:
        imu.stop()

    R_fit = kabsch(np.array(measured), np.array(expected))
    print("\nbest-fit R_gun_imu (Kabsch over the three poses):")
    for row in R_fit:
        print("  " + _fmt_vec(row, "%+8.5f"))
    for (prompt, want), m in zip(AXES_POSES, measured):
        print(f"  residual {prompt.split(':')[0]:<15s} {_angle_deg(R_fit @ m, want):5.2f} deg")
    delta = math.degrees(float(np.linalg.norm(log_so3(R_fit @ R_current.T))))
    print(f"  it differs from the current R_gun_imu by {delta:.2f} deg")
    if delta > 20.0:
        print("  WARNING: that is more than a mounting tolerance — check the pose prompts and the "
              "axis map in design/layout.md before writing it")

    if not args.write:
        print("\nnot written (pass --write to store it in calib/imu.json)")
        return 0
    path = calib.path("imu.json")
    update_json(path, {"R_gun_imu": R_fit.tolist(), "created": now_iso()})
    print(f"\nwrote R_gun_imu -> {path}")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    cfg, calib = load_from_args(args)
    from tbot.optics import CameraModel, MountModel, combined_mask

    try:
        session = _load_session(args.session)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}")
        return 1
    frames = session["frames"]
    t_frames = session["t_ns"]
    imu_t = session["imu_t_ns"]
    imu_gyro = session["imu_gyro"]
    if imu_t.size < 2:
        print(f"error: {args.session}/imu.csv has no IMU samples — record the session with the IMU running")
        return 1
    if frames.shape[0] < 2:
        print(f"error: {args.session} has fewer than 2 frames")
        return 1

    height, width = int(frames.shape[1]), int(frames.shape[2])
    cam = CameraModel.from_calibration(calib.intrinsics, cfg, width, height)
    mount = MountModel.from_config(cfg, calib.boresight)
    mask = combined_mask(cam, mount, cfg, calib) if not args.no_mask else None

    import cv2

    step = max(1, int(args.step))
    mid_t_s: list[float] = []
    rates: list[np.ndarray] = []
    t0 = int(t_frames[0])
    used = 0
    for k in range(0, frames.shape[0] - step, step):
        img0, img1 = frames[k], frames[k + step]
        if img0.ndim == 3:
            img0 = cv2.cvtColor(np.ascontiguousarray(img0), cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(np.ascontiguousarray(img1), cv2.COLOR_BGR2GRAY)
        else:
            img0 = np.ascontiguousarray(img0)
            img1 = np.ascontiguousarray(img1)
        dt_s = float(int(t_frames[k + step]) - int(t_frames[k])) * 1e-9
        omega_c = camera_rotation_rate(img0, img1, dt_s, cam, mask)
        if omega_c is None:
            continue
        rates.append(np.asarray(mount.cam_to_gun(omega_c), dtype=np.float64).reshape(3))
        mid_t_s.append((0.5 * (int(t_frames[k]) + int(t_frames[k + step])) - t0) * 1e-9)
        used += 1
    print(f"{used} of {max(0, frames.shape[0] - step)} frame pairs gave a rotation "
          f"({width}x{height}, step {step})")
    if used < MIN_SYNC_SAMPLES:
        print("error: too few usable frame pairs — record a textured scene with a clear swing")
        return 1

    cam_rates = np.array(rates, dtype=np.float64)
    cam_t_s = np.array(mid_t_s, dtype=np.float64)
    imu_t_s = (imu_t.astype(np.float64) - float(t0)) * 1e-9
    peak_dps = float(np.degrees(np.abs(cam_rates).max()))
    print(f"  peak image rotation rate {peak_dps:.1f} deg/s, "
          f"peak gyro rate {float(np.degrees(np.abs(imu_gyro).max())):.1f} deg/s")
    if peak_dps < 5.0:
        print("  WARNING: the gun barely moved; swing it back and forth while recording")
    try:
        offset_s, score = estimate_time_offset_s(cam_t_s, cam_rates, imu_t_s, imu_gyro,
                                                 max_lag_s=args.max_lag_ms / 1000.0,
                                                 lag_step_s=args.lag_step_ms / 1000.0)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    print(f"\ntime offset to ADD to IMU timestamps: {offset_s * 1000.0:+.2f} ms "
          f"(correlation {score:.3f})")
    if score < args.min_score:
        print(f"error: correlation {score:.3f} is below --min-score {args.min_score:.2f}; "
              "not written (record a longer, more dynamic session)")
        return 1
    if args.no_write:
        print("--no-write: calib/imu.json not touched")
        return 0
    # The recording's IMU timestamps already carry the stored offset, so the measurement is a residual.
    prior_ns = int((calib.imu or {}).get("time_offset_ns", 0) or 0)
    total_ns = prior_ns + int(round(offset_s * 1e9))
    path = calib.path("imu.json")
    update_json(path, {"time_offset_ns": total_ns, "created": now_iso()})
    print(f"wrote time_offset_ns {total_ns} (was {prior_ns}) -> {path}")
    return 0


# ------------------------------------------------------------------------------------ session reader
def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _read_csv(path: Path, columns: tuple[str, ...]) -> dict[str, list[str]]:
    """Read a session CSV by column name (header) or by position (no header)."""
    with path.open(newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    if not rows:
        return {c: [] for c in columns}
    header = rows[0]
    if all(_is_number(x) for x in header):
        idx = {name: i for i, name in enumerate(columns)}
        body = rows
    else:
        missing = [c for c in columns if c not in header]
        if missing:
            raise ValueError(f"{path.name} is missing column(s) {missing}")
        idx = {name: header.index(name) for name in columns}
        body = rows[1:]
    width = max(idx.values()) + 1
    body = [r for r in body if len(r) >= width]
    return {name: [r[i] for r in body] for name, i in idx.items()}


def _load_session(path: str | Path) -> dict[str, Any]:
    """Read a recorded session (meta.json + frames.bin + frames.csv + imu.csv)."""
    d = resolve_path(path)
    meta = json.loads((d / "meta.json").read_text())
    width, height = int(meta["width"]), int(meta["height"])
    channels = int(meta.get("channels", 1) or 1)
    dtype = np.dtype(meta.get("dtype", "uint8"))
    frame_bytes = width * height * channels * dtype.itemsize
    n_file = (d / "frames.bin").stat().st_size // frame_bytes
    cols = _read_csv(d / "frames.csv", ("seq", "t_exposure_ns"))
    n = min(n_file, len(cols["t_exposure_ns"]))
    if n < 1:
        raise ValueError(f"{d} contains no frames")
    shape = (n, height, width) if channels == 1 else (n, height, width, channels)
    frames = np.memmap(d / "frames.bin", dtype=dtype, mode="r", shape=shape)
    t_ns = np.array([int(float(x)) for x in cols["t_exposure_ns"][:n]], dtype=np.int64)

    imu_path = d / "imu.csv"
    if imu_path.exists():
        icols = _read_csv(imu_path, ("t_ns", "gx", "gy", "gz", "ax", "ay", "az"))
        imu_t = np.array([int(float(x)) for x in icols["t_ns"]], dtype=np.int64)
        imu_gyro = np.array([[float(gx), float(gy), float(gz)] for gx, gy, gz
                             in zip(icols["gx"], icols["gy"], icols["gz"])], dtype=np.float64)
        imu_accel = np.array([[float(ax), float(ay), float(az)] for ax, ay, az
                              in zip(icols["ax"], icols["ay"], icols["az"])], dtype=np.float64)
    else:
        imu_t = np.zeros(0, dtype=np.int64)
        imu_gyro = np.zeros((0, 3), dtype=np.float64)
        imu_accel = np.zeros((0, 3), dtype=np.float64)
    return {"dir": d, "meta": meta, "frames": frames, "t_ns": t_ns,
            "imu_t_ns": imu_t, "imu_gyro": imu_gyro, "imu_accel": imu_accel}


# ------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot calib-imu",
        description="IMU calibration: static bias, axis map check, camera-IMU time offset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bias = sub.add_parser("bias", help="static gyro bias + accel magnitude check (gun still)")
    p_bias.add_argument("--seconds", type=float, default=10.0, help="capture duration (default 10)")
    p_bias.add_argument("--no-write", action="store_true", help="measure and print only")
    add_common_args(p_bias)

    p_axes = sub.add_parser("axes", help="guided 3-pose check/solve of R_gun_imu")
    p_axes.add_argument("--seconds", type=float, default=2.0, help="capture per pose (default 2)")
    p_axes.add_argument("--write", action="store_true", help="store the fitted R_gun_imu")
    p_axes.add_argument("--auto", action="store_true",
                        help="no Enter presses: capture each pose once it is held still (and new)")
    p_axes.add_argument("--timeout", type=float, default=180.0, help="--auto: seconds to wait per pose")
    add_common_args(p_axes)

    p_sync = sub.add_parser("sync", help="camera-IMU time offset from a recorded swing")
    p_sync.add_argument("--session", required=True, help="session directory recorded while swinging")
    p_sync.add_argument("--step", type=int, default=1, help="frame stride for the flow pairs")
    p_sync.add_argument("--max-lag-ms", type=float, default=150.0, help="search range (default +/-150)")
    p_sync.add_argument("--lag-step-ms", type=float, default=1.0, help="lag grid step (default 1)")
    p_sync.add_argument("--min-score", type=float, default=0.5, help="minimum correlation to write")
    p_sync.add_argument("--no-mask", action="store_true", help="do not mask the barrel/static mask")
    p_sync.add_argument("--no-write", action="store_true", help="measure and print only")
    add_common_args(p_sync)

    args = parser.parse_args(argv)
    return {"bias": _cmd_bias, "axes": _cmd_axes, "sync": _cmd_sync}[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
