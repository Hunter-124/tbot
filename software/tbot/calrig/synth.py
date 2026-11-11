"""Synthetic CalRig B runs in the exact docs/CALRIG_PROTOCOL.md section 4 format, with a known ground truth.

Used by the analysis tests (tests/test_calrig_analysis.py). Physics modelled: a trapezoidal / sinusoidal table
profile, the chip frame = R_gun_imu^T applied to the gun-frame table axis of the pose, per-axis gyro bias and scale
error, IMU stamps late by `time_offset_s`, white noise, optional post-move ringing, quantised rig counts, hall edges
and optional lost steps. Firmware t_us starts just below the uint32 wrap so the unwrap path is exercised.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tbot.calrig.analysis import POSE_UP_GUN

__all__ = ["NOMINAL_R_GUN_IMU", "DEFAULT_PROGRAM", "rot_about", "write_synthetic_run"]

NOMINAL_R_GUN_IMU = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
DEFAULT_PROGRAM: tuple = (("still", 3.0), ("move", 60.0), ("still", 4.0), ("move", -60.0), ("still", 4.0),
                          ("move", 85.0), ("still", 4.0), ("move", -170.0), ("still", 4.0), ("move", 85.0),
                          ("still", 4.0))
G = 9.80665
DPU = 360.0 / (200 * 16)
FINE_DT = 2e-4
HOST_T0_NS = 1_700_000_000_000_000_000
US0 = (1 << 32) - 2_000_000


def rot_about(axis: Sequence[float], deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def _trap(t: np.ndarray, D: float, vmax: float, amax: float) -> tuple[np.ndarray, float]:
    da_abs = abs(D)
    vp = min(vmax, math.sqrt(amax * da_abs))
    ta = vp / amax
    dacc = 0.5 * amax * ta * ta
    tc = (da_abs - 2 * dacc) / vp
    T = 2 * ta + tc
    th = np.where(t < ta, 0.5 * amax * t ** 2,
                  np.where(t < ta + tc, dacc + vp * (t - ta),
                           np.where(t < T, da_abs - 0.5 * amax * (T - t) ** 2, da_abs)))
    return math.copysign(1.0, D) * th, T


def _profile(program: Sequence[tuple], vmax: float, amax: float) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Counted angle (deg) on a FINE_DT grid and the (kind, t0, t1) of each segment."""
    pieces, marks = [], []
    th0, t_cursor = 0.0, 0.0
    for item in program:
        kind = item[0]
        if kind == "still":
            T = float(item[1])
            n = int(round(T / FINE_DT))
            th = np.zeros(n)
        elif kind == "move":
            D = float(item[1])
            _, T = _trap(np.zeros(1), D, vmax, amax)
            n = int(math.ceil(T / FINE_DT))
            th, _ = _trap(np.arange(n) * FINE_DT, D, vmax, amax)
        elif kind == "sine":
            amp, period, cycles = float(item[1]), float(item[2]), float(item[3])
            T = period * cycles
            n = int(round(T / FINE_DT))
            t = np.arange(n) * FINE_DT
            env = np.minimum(np.sin(np.pi * np.minimum(t / period, 1.0) / 2) ** 2,
                             np.sin(np.pi * np.minimum((T - t) / period, 1.0) / 2) ** 2)
            th = amp * env * np.sin(2 * np.pi * t / period)
            D = 0.0
        else:
            raise ValueError(kind)
        pieces.append(th0 + th)
        marks.append((kind, t_cursor, t_cursor + n * FINE_DT))
        t_cursor += n * FINE_DT
        th0 += th[-1] if kind != "move" else float(item[1])
    return np.concatenate(pieces), np.arange(sum(len(x) for x in pieces)) * FINE_DT, marks


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def write_synthetic_run(
    run_dir: str | Path, *, pose: str = "upright", R_gun_imu: np.ndarray | None = None,
    gyro_bias_rad_s: Sequence[float] = (0.0061, -0.0037, 0.0021),
    gyro_scale_err: Sequence[float] = (0.015, -0.02, 0.01), time_offset_s: float = 0.003,
    gyro_noise_dps: float = 0.05, accel_noise: float = 0.03, program: Sequence[tuple] = DEFAULT_PROGRAM,
    vmax_dps: float = 45.0, amax_dps2: float = 20.0, imu_rate_hz: float = 500.0, telem_hz: float = 200.0,
    ring_hz: float = 0.0, ring_dps: float = 0.0, ring_tau_s: float = 0.5,
    rig_jitter_s: float = 0.0, temp0_c: float = 31.0, temp_slope_c_per_s: float = 0.02,
    hall_window_deg: tuple[float, float] = (-3.0, 3.0), hall_jitter_ustep: float = 0.4,
    lost_ustep: Sequence[tuple[int, int]] = (), homed: Sequence[tuple[float, int]] = ((0.5, 40),),
    seed: int = 1,
) -> dict:
    """Write one run directory; returns the ground truth dict (R_gun_imu, chip axis, bias, scale, ...).

    `program` items: ("still", s), ("move", signed_deg), ("sine", amp_deg, period_s, cycles).
    `lost_ustep`: (segment_index, n_ustep): the table physically slips n_ustep (not counted) just after the
    segment ends. `time_offset_s` > 0: IMU stamps late. `homed`: (t_s, offset_ustep) HOMED events.
    """
    rng = np.random.default_rng(seed)
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    R = np.asarray(NOMINAL_R_GUN_IMU if R_gun_imu is None else R_gun_imu, dtype=np.float64)
    up = np.asarray(POSE_UP_GUN[pose], dtype=np.float64)
    axis_chip = R.T @ up
    bias = np.asarray(gyro_bias_rad_s, dtype=np.float64)
    scale = 1.0 + np.asarray(gyro_scale_err, dtype=np.float64)

    th_cnt, tf, marks = _profile(program, vmax_dps, amax_dps2)
    total = float(tf[-1]) + 1.0
    # lost steps: physical angle = counted + slip(t), slip ramps over 50 ms after the segment end
    slip = np.zeros_like(th_cnt)
    for seg, n in lost_ustep:
        te = marks[seg][2]
        slip += n * DPU * np.clip((tf - te) / 0.05, 0.0, 1.0)
    th_phys = th_cnt + slip
    w_phys = np.gradient(th_phys, FINE_DT)  # deg/s

    # ---- IMU
    n_imu = int((total - 0.5) * imu_rate_hz)
    ti = 0.25 + np.arange(n_imu) / imu_rate_hz
    tt = ti - time_offset_s
    om = np.interp(tt, tf, w_phys, left=0.0, right=0.0)
    if ring_hz > 0.0 and ring_dps > 0.0:
        for kind, t0, t1 in marks:
            if kind == "still":
                continue
            dt = tt - t1
            om = om + np.where(dt > 0, ring_dps * np.exp(-np.maximum(dt, 0) / ring_tau_s)
                               * np.sin(2 * np.pi * ring_hz * dt), 0.0)
    gyro = (om * math.pi / 180.0)[:, None] * axis_chip[None, :]
    gyro = gyro * scale + bias + rng.normal(0.0, gyro_noise_dps * math.pi / 180.0, gyro.shape)
    accel = (R.T @ (G * up))[None, :] + rng.normal(0.0, accel_noise, (n_imu, 3))
    temp = temp0_c + temp_slope_c_per_s * ti
    imu_t_ns = HOST_T0_NS + np.round(ti * 1e9).astype(np.int64)
    _write_csv(d / "imu.csv", ("t_ns", "gx", "gy", "gz", "ax", "ay", "az", "temp_c"),
               [(int(t), *("%.7g" % v for v in g), *("%.5g" % v for v in a), "%.2f" % tc)
                for t, g, a, tc in zip(imu_t_ns, gyro, accel, temp)])

    # ---- rig telemetry
    tr = np.arange(0.2, total - 0.2, 1.0 / telem_hz)
    ang = np.interp(tr, tf, th_cnt)
    pos = np.round(ang / DPU).astype(np.int64)
    vel = np.interp(tr, tf, np.gradient(th_cnt, FINE_DT))
    jit = np.round(rng.normal(0.0, rig_jitter_s, tr.size) * 1e9).astype(np.int64) if rig_jitter_s > 0 else 0
    rig_t_ns = HOST_T0_NS + np.round(tr * 1e9).astype(np.int64) + jit
    rig_t_us = (US0 + np.round(tr * 1e6).astype(np.int64)) % (1 << 32)
    _write_csv(d / "rig.csv", ("t_ns", "t_us", "pos_ustep", "angle_deg", "vel_deg_s"),
               [(int(a), int(b), int(c), "%.4f" % (c * DPU), "%.4f" % v)
                for a, b, c, v in zip(rig_t_ns, rig_t_us, pos, vel)])

    # ---- hall
    lo, hi = hall_window_deg
    active = (th_phys >= lo) & (th_phys <= hi)
    idx = np.flatnonzero(np.diff(active.astype(np.int8))) + 1
    hall_rows = []
    for i in idx:
        te = tf[i]
        p = int(round(th_cnt[i] / DPU + rng.normal(0.0, hall_jitter_ustep)))
        hall_rows.append((HOST_T0_NS + int(round(te * 1e9)), int((US0 + round(te * 1e6)) % (1 << 32)), p,
                          "%.4f" % (p * DPU), int(active[i])))
    _write_csv(d / "hall.csv", ("t_ns", "t_us", "pos_ustep", "angle_deg", "level"), hall_rows)

    # ---- events + commands
    ev = [(HOST_T0_NS, US0 % (1 << 32), "BOOT", "")]
    cmd = [(HOST_T0_NS, "STREAM 1")]
    for th, off in homed:
        ev.append((HOST_T0_NS + int(round(th * 1e9)), int((US0 + round(th * 1e6)) % (1 << 32)), "HOMED", off))
        cmd.append((HOST_T0_NS + int(round((th - 0.3) * 1e9)), "HOME 1 5000"))
    for kind, t0, t1 in marks:
        if kind != "still":
            cmd.append((HOST_T0_NS + int(round(t0 * 1e9)), "MOVE 0" if kind == "move" else "SINE"))
            ev.append((HOST_T0_NS + int(round(t1 * 1e9)), int((US0 + round(t1 * 1e6)) % (1 << 32)), "DONE", ""))
    ev.sort(key=lambda r: r[0])
    cmd.sort(key=lambda r: r[0])
    _write_csv(d / "events.csv", ("t_ns", "t_us", "code", "arg"), ev)
    _write_csv(d / "cmd.csv", ("t_ns", "command"), cmd)

    meta = {"procedure": "synthetic", "pose": pose, "deg_per_ustep": DPU,
            "firmware": {"fw": "calrig-uno/synthetic", "usteps": 16},
            "sign_convention": "angle positive = CCW seen from above",
            "clock": {"offset_ns": HOST_T0_NS - US0 * 1000, "drift_ppm": 0.0, "resid_us": 20.0, "n": 60},
            "imu": {"sample_rate_hz": imu_rate_hz, "gyro_range_dps": 1000, "R_gun_imu": NOMINAL_R_GUN_IMU.tolist()},
            "unloaded_confirmed": True, "synthetic": True}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"R_gun_imu": R, "pose": pose, "axis_chip": axis_chip, "axis_index": int(np.argmax(np.abs(axis_chip))),
            "axis_sign": int(np.sign(axis_chip[np.argmax(np.abs(axis_chip))])), "bias_rad_s": bias,
            "scale_err": np.asarray(gyro_scale_err), "time_offset_s": time_offset_s, "marks": marks,
            "lost_ustep": list(lost_ustep), "total_s": total, "noise_dps": gyro_noise_dps}
