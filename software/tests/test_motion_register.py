"""Registration of the motion detector (tbot/vision/motion.py): what it may and may not conclude from a scene.

- a scene that is one straight edge (a horizon) fixes the shift across the edge only: nothing is applied along it,
  and the frame does not count as registered;
- a textured scene registers in full;
- the reference lags shrink while the gun swings, so the leading edge of a swing is not blind.

The synthetic world and renderer are test_motion_detector's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import test_motion_detector as T
from tbot.vision import motion as M
from tbot.vision.motion import MotionDetector


@pytest.fixture(scope="module")
def cam():
    return T.make_cam()


@pytest.fixture(scope="module")
def mount():
    return T.make_mount()


@pytest.fixture(scope="module")
def rays(cam):
    return T.make_rays(cam)


@pytest.fixture(scope="module")
def world():
    return T.make_world()


def make_horizon():
    """Level bands (a banded sky over a terraced bank: an edge every 40 px, so the texture guard is satisfied) and
    nothing that fixes a sideways shift."""
    yy = np.arange(T.TEX_H, dtype=np.float32)[:, None]
    t = 120.0 + 45.0 * np.sign(np.sin(2.0 * math.pi * yy / 80.0)) + 0.02 * yy
    return T._blur(np.broadcast_to(t, (T.TEX_H, T.TEX_W)).astype(np.float32).copy(), 1.5)


def traced(det):
    """det.stats of every frame, in order (T.run returns after the last one)."""
    log = []
    real = det.detect

    def detect(frame, R_wg=None):
        out = real(frame, R_wg)
        log.append(dict(det.stats))
        return out

    det.detect = detect
    return log


# ------------------------------------------------------------------------------------------- the tensor
def test_texture_tensor_is_the_sum_over_textured_usable_pixels():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (37, 53), dtype=np.uint8)
    valid = np.where(rng.random((37, 53)) < 0.8, 255, 0).astype(np.uint8)
    tex = M.texture_map(img, 40)
    sxx = syy = sxy = 0
    for y in range(36):
        for x in range(52):
            if tex[y, x] and valid[y, x]:
                dx = int(img[y, x + 1]) - int(img[y, x])
                dy = int(img[y + 1, x]) - int(img[y, x])
                sxx += dx * dx
                syy += dy * dy
                sxy += dx * dy
    assert sxx > 0 and syy > 0
    assert M.texture_tensor(img, tex, valid) == (sxx, syy, sxy)
    assert M.texture_tensor(img, np.zeros_like(tex), valid) == (0, 0, 0)
    assert M.texture_tensor(img, tex, np.zeros_like(valid)) == (0, 0, 0)


# ------------------------------------------------------------------------------------------- a horizon
def test_a_horizon_fixes_the_shift_across_it_only(mount, rays, cam):
    """The attitude drifts 3 deg/s in pitch and in yaw (4.2 px per 40 ms each). The edge shows the pitch error and
    says nothing about the yaw error: the correction is vertical, no sideways shift is invented (it was +-20 px
    from frame to frame on the simulator's horizon), the frame is not "registered", and nothing is detected."""
    det = MotionDetector(T._cfg(), None, cam, mount)
    log = traced(det)
    modes, sx, sy, n_det = [], [], [], 0
    res = T.run(det, make_horizon(), mount, rays, cam, n=36, yaw=10.0, gyro_bias_dps=(0.0, 3.0, 3.0))
    for k, (_, dets, _, _) in enumerate(res):
        if k < T.WARM:
            continue
        info = log[k]["reg_a"]
        modes.append(info["mode"])
        sx.append(info["shift"][0])
        sy.append(info["shift"][1])
        assert info["ok"] is False
        assert all(not d.reg_ok for d in dets)
        n_det += len(dets)
    assert modes.count("1d") >= 0.8 * len(modes), modes
    assert max(abs(x) for x in sx) <= 0.5, sx
    across = [abs(y) for y, m in zip(sy, modes) if m == "1d"]
    lag_a = float(np.median([st["lag_ms"][0] for st in log[T.WARM:]])) * 1e-3   # (an edge scene has the sky's lags)
    want = math.radians(3.0) * T.FX * lag_a     # the pitch drift over lag A
    assert np.median(across) == pytest.approx(want, abs=1.5), (np.median(across), want)
    assert n_det == 0


def test_texture_registers_in_full(world, mount, rays, cam):
    det = MotionDetector(T._cfg(), None, cam, mount)
    log = traced(det)
    T.run(det, world, mount, rays, cam, n=30, yaw=20.0, gyro_bias_dps=(0.0, 2.0, 2.0))
    modes = [st["reg_a"]["mode"] for st in log[T.WARM:]]
    assert all(st["reg_a"]["cond"][1] >= M.TENSOR_MIN for st in log[T.WARM:])
    assert modes.count("2d") >= 0.9 * len(modes), modes


# ------------------------------------------------------------------------------------------- the lags
def test_reference_lags_are_the_configured_ones_on_a_still_gun(world, mount, rays, cam):
    det = MotionDetector(T._cfg(), None, cam, mount)
    T.run(det, world, mount, rays, cam, n=12)
    assert det.stats["lag_ms"] == (40.0, 100.0)


def test_reference_lags_shrink_with_the_swing(world, mount, rays, cam):
    det = MotionDetector(T._cfg(), None, cam, mount)
    T.run(det, world, mount, rays, cam, n=12, yaw=80.0)
    la, lb = det.stats["lag_ms"]
    pan = math.radians(80.0) * T.FX                       # px/s
    assert lb == pytest.approx(1e3 * M.LAG_SHIFT_FRAC * T.W / pan, rel=0.05)   # ~34 ms instead of 100
    assert la == pytest.approx(0.4 * lb, rel=0.05)


@pytest.mark.parametrize("yaw", [80.0, -80.0])
def test_a_mover_near_the_leading_edge_of_a_fast_swing_is_seen(world, mount, rays, cam, yaw):
    """Two discs held at fixed places in the picture (the shooter's view of clays he is swinging with), one fifth
    of the width from either side, the gun swinging at 80 deg/s: with the 100 ms reference 280 px of the leading
    side had no reference at all."""
    det = MotionDetector(T._cfg(), None, cam, mount)
    spots = [(0.2 * T.W, 150.0), (0.8 * T.W, 250.0)]
    state = {"k": 0}

    def post(img):
        for u, v in spots:
            T.draw_disc(img, u, v, 8.0, 235.0)
        state["k"] += 1

    res = T.run(det, world, mount, rays, cam, n=30, yaw=yaw, post=post)
    seen = [0, 0]
    n = 0
    for _, dets, _, _ in res[T.WARM:]:
        n += 1
        for i, (u, v) in enumerate(spots):
            seen[i] += any(math.hypot(d.u - u, d.v - v) <= 3.0 for d in dets)
    assert min(seen) >= 0.9 * n, (seen, n)


# ------------------------------------------------------------------------------------------- open sky
def test_reference_lags_stretch_on_open_sky(mount, rays, cam):
    det = MotionDetector(T._cfg(), None, cam, mount)
    T.run(det, T.make_sky(), mount, rays, cam, n=24)
    assert det.stats["lag_ms"] == (40.0 * M.SKY_LAG_GAIN, 100.0 * M.SKY_LAG_GAIN)


@pytest.mark.parametrize("value", [235.0, 60.0])
def test_a_slow_clay_on_open_sky_is_seen(mount, rays, cam, value):
    """8 px across, 3 deg/s against a still gun: 4 px per 40 ms, half its own size. Against the 40 ms reference
    most of it cancels itself; against the references of open sky (100 and 250 ms) it stands clear."""
    det = MotionDetector(T._cfg(), None, cam, mount)
    res = T.run(det, T.make_sky(), mount, rays, cam, n=50, disc=T._mover(cam, mount, 3.0, 8.0, value, az0=0.03))
    warm = int(round(0.100 * M.SKY_LAG_GAIN / (T.DT_NS * 1e-9))) + 3   # lag B (250 ms) of history, and a little
    hit = n = 0
    for _, dets, truth, _ in res[warm:]:
        if truth is None:
            continue
        n += 1
        hit += any(math.hypot(d.u - truth[0], d.v - truth[1]) <= 2.0 for d in dets)
        assert len(dets) <= 1, [(d.u, d.v) for d in dets]
    assert n >= 20 and hit >= 0.9 * n, (hit, n)
