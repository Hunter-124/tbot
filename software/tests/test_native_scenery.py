"""The scenery filter's native kernel (tbot_scenery in tbot/vision/_measure.c) keeps, drops and orders the
candidates exactly as StaticFilter's Python pass, which is its reference.

Skipped when the shared library is not built (scripts/build_native.sh), the state the rest of the suite must
also pass in.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tbot.config import load_calibration, load_config
from tbot.optics import CameraModel, MountModel
from tbot.vision import _native
from tbot.vision.clutter import StaticFilter

pytestmark = pytest.mark.skipif(_native.scenery_kernel() is None,
                                reason=f"native scenery filter: {_native.status} / {_native.scenery_status}")

W, H = 640, 400
PRESSES = Path(__file__).resolve().parents[1] / "field_data" / "2026-11-17-range" / "presses"


def _cams(cfg):
    K = [[2009.0, 0.0, 319.5], [0.0, 2011.0, 199.5], [0.0, 0.0, 1.0]]
    return [
        CameraModel.nominal(cfg, W, H),
        CameraModel(W, H, K, [-0.31, 0.12, 0.0009, -0.0004, 0.02], "pinhole"),
        CameraModel(W, H, K, [-0.31, 0.12, 0.0009, -0.0004, 0.02, 0.01, -0.02, 0.003], "pinhole"),
        CameraModel(W, H, K, [-0.31, 0.12, 0.0009, -0.0004, 0.02, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
                    "pinhole"),   # thin prism: no plain-float model, the kernel gets OpenCV's rays
        CameraModel(W, H, [[420.0, 0.0, 319.5], [0.0, 420.0, 199.5], [0.0, 0.0, 1.0]],
                    [-0.03, 0.004, -0.001, 0.0002], "fisheye"),
    ]


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _pair(cam, mount, sets=()):
    nat = StaticFilter(load_config(sets=[*sets, "detector.native=true"]), cam, mount)
    ref = StaticFilter(load_config(sets=[*sets, "detector.native=false"]), cam, mount)
    return nat, ref


def _same(nat, ref, a, b, where):
    assert a.tolist() == b.tolist(), where
    assert nat.n_static == ref.n_static and nat.n_new == ref.n_new and nat.n_focus == ref.n_focus, where
    assert nat.static_px == ref.static_px, where
    assert nat.native_used and not ref.native_used, where


@pytest.mark.parametrize("cam_i", range(5))
def test_swinging_through_scenery_gives_the_same_candidates(cam_i):
    cfg = load_config()
    cam = _cams(cfg)[cam_i]
    mount = MountModel.from_config(cfg, {"rpy_deg": [0.4, -1.9, -1.7]})
    nat, ref = _pair(cam, mount)
    rng = np.random.default_rng(7 + cam_i)
    R_gc = nat._R_gc
    # fixed scenery: world directions spread over a 30 x 15 degree patch around +z of the starting camera
    scen_cam = np.stack([rng.uniform(-0.26, 0.26, 300), rng.uniform(-0.13, 0.13, 300), np.ones(300)], axis=1)
    scen_cam /= np.linalg.norm(scen_cam, axis=1, keepdims=True)
    scen_w = scen_cam @ (_rot(0, 0, 0) @ R_gc).T   # camera -> world at the start
    checked = static_seen = 0
    for k in range(160):
        t = 1_000_000_000 + k * 10_000_000
        R_wg = _rot(0.004 * k + 0.002 * math.sin(k / 7), 0.001 * k, 0.0005 * math.sin(k / 5))
        R_wc = R_wg @ R_gc
        cam_dirs = scen_w @ R_wc   # world -> this camera (rows)
        ok = cam_dirs[:, 2] > 0.2
        px = np.asarray(cam.ray_to_pixel(cam_dirs[ok]))
        px = px[np.isfinite(px).all(axis=1)]
        px = px[(px[:, 0] > 2) & (px[:, 0] < W - 3) & (px[:, 1] > 2) & (px[:, 1] < H - 3)]
        px = px[rng.random(len(px)) < 0.6]
        px = px + rng.normal(0, 0.3, px.shape)
        movers = np.stack([rng.uniform(5, W - 6, 4), rng.uniform(5, H - 6, 4)], axis=1)
        uv = np.concatenate([px, movers])[rng.permutation(len(px) + 4)]
        n = len(uv)
        areas = rng.integers(1, 200, n).astype(np.int32)
        w = rng.integers(1, 18, n).astype(np.int32)
        h = rng.integers(1, 18, n).astype(np.int32)
        if k % 11 == 5:   # some ties in solidity
            areas[:] = 4
            w[:] = 2
            h[:] = 2
        kw = {}
        if k % 3 == 0:
            pick = R_wc @ np.asarray(cam.pixel_to_ray(uv[: max(1, n // 5)])).reshape(-1, 3).T
            kw["protect_dirs_w"] = pick.T
        if k % 4 == 1:
            f = (R_wc @ np.asarray(cam.pixel_to_ray(uv[:2] + 3.0)).reshape(-1, 3).T).T
            kw.update(focus_dirs_w=f, others_max=3)
            if k % 8 == 1:
                kw["focus_radius_rad"] = [40 * nat.ifov, 5 * nat.ifov]
        for filt in (nat, ref):
            filt.begin(t, R_wg, **kw)
        args = (uv, areas, w, h) if k % 13 else (uv,)   # now and then without sizes
        a, b = nat.filter(*args), ref.filter(*args)
        _same(nat, ref, a, b, f"frame {k}")
        checked += n
        static_seen += nat.n_static
    assert checked > 3000 and static_seen > 1000   # the scenery was there and was found


def test_reset_and_a_clock_going_back():
    cfg = load_config()
    cam = CameraModel.nominal(cfg, W, H)
    mount = MountModel.from_config(cfg, None)
    nat, ref = _pair(cam, mount)
    uv = np.array([[100.0, 100.0], [300.5, 200.25], [500.0, 310.0]])
    for t in (1_000_000_000, 1_050_000_000, 1_100_000_000, 900_000_000, 950_000_000, 1_200_000_000):
        for filt in (nat, ref):
            filt.begin(t, np.eye(3))
        _same(nat, ref, nat.filter(uv), ref.filter(uv), t)
    for filt in (nat, ref):
        filt.reset()
        filt.begin(2_000_000_000, np.eye(3))
    _same(nat, ref, nat.filter(uv), ref.filter(uv), "after reset")
    assert nat.filter(np.zeros((0, 2))).size == 0


def _presses():
    if not PRESSES.is_dir():
        return []
    return sorted(d for d in PRESSES.iterdir() if (d / "frames.bin").is_file() and (d / "imu.csv").is_file())


@pytest.mark.parametrize("press", _presses(), ids=lambda p: p.name)
def test_range_presses_give_the_same_scenery_and_detections(press):
    """The whole pipeline over a recorded press with its IMU: the filter sees real clutter and attitude."""
    from tbot.app.run import Pipeline
    from tbot.camera.replay import ReplaySource
    from tbot.hw.gpio import open_gpio

    runs = []
    for native in ("true", "false"):
        # detector.native also switches the measurement kernel: both runs then differ only by rounding there
        cfg = load_config(sets=["gpio.backend=sim", "record.mode=off", "stream.enabled=false",
                                f"detector.native={native}"])
        calib = load_calibration(cfg)
        gpio = open_gpio(cfg)
        pipe = Pipeline(cfg, calib, ReplaySource(str(press), realtime=False), gpio=gpio)
        gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)
        out = []
        used = 0
        while (tele := pipe.step()) is not None:
            c = pipe.clutter
            out.append((c.n_static, c.n_new, c.n_focus, tuple(map(tuple, c.static_px)),
                        tuple((round(d["u"], 2), round(d["v"], 2)) for d in tele.get("detections", []))))
            used += bool(c is not None and c.native_used)
        pipe.close()
        runs.append((out, used))
    (a, used_a), (b, used_b) = runs
    assert len(a) == len(b) and used_a > 0 and used_b == 0
    assert sum(x[0] for x in a) > 0, "no scenery in this press"
    for k, (x, y) in enumerate(zip(a, b)):
        assert x == y, f"{press.name} frame {k}"
