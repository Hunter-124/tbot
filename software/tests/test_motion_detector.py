"""Motion-first detector (tbot/vision/motion.py) on synthetic scenes with a known camera motion, and on real presses.

The synthetic world is a texture on the (azimuth, elevation) sphere; each frame is rendered through the same
CameraModel / MountModel / R_wg chain the tracker uses (pixel -> cam.pixel_to_ray -> mount R_gun_cam -> R_wg), so a
detector that gets the sign of the roll or of a rotation axis wrong fails here: the scenery would not cancel.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from tbot.config import load_config
from tbot.optics import CameraModel, MountModel, rot_rpy
from tbot.types import Frame
from tbot.vision import motion as M
from tbot.vision.motion import MotionDetector

W, H = 640, 400
FX = 2009.0
F_TEX = FX              # texture px per radian (the camera's own scale)
TEX_W, TEX_H = 3600, 1400
U0, V0 = 600.0, 700.0   # texture pixel of az = 0, el = 0
DT_NS = 20_000_000


def _cfg(scale=2, sets=()):
    return load_config(sets=[f"vision.scale={scale}", *sets])


def make_cam():
    K = np.array([[FX, 0, (W - 1) / 2], [0, FX, (H - 1) / 2], [0, 0, 1]])
    return CameraModel(W, H, K, np.zeros(5), "pinhole")


def make_mount():
    return MountModel.from_config(load_config())


def make_rays(cam):
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return cam.pixel_to_ray(np.stack([xs.ravel(), ys.ravel()], 1))


def _blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)


def make_world():
    """Textured world: smooth structure, fine texture, strong specks and edges, blurred by a ~1 px PSF."""
    rng = np.random.default_rng(7)
    t = 120 + 45 * _blur(rng.standard_normal((TEX_H, TEX_W)).astype(np.float32), 18) * 18 / 3
    t += 20 * _blur(rng.standard_normal((TEX_H, TEX_W)).astype(np.float32), 1.5) * 1.5 * 2
    for _ in range(1500):  # specks
        x, y = rng.integers(2, TEX_W - 4), rng.integers(2, TEX_H - 4)
        t[y:y + 2, x:x + 2] += rng.choice([-90, 90])
    for _ in range(250):  # edges / slabs
        x, y = rng.integers(0, TEX_W - 80), rng.integers(0, TEX_H - 80)
        t[y:y + rng.integers(4, 80), x:x + rng.integers(4, 80)] += rng.choice([-40, 40])
    return np.clip(_blur(t, 1.3), 5, 250).astype(np.float32)


def make_sky():
    """Plain sky: a smooth vertical gradient, no texture."""
    yy = np.arange(TEX_H, dtype=np.float32)[:, None]
    return np.broadcast_to(150.0 + 40.0 * yy / TEX_H, (TEX_H, TEX_W)).astype(np.float32).copy()


def make_sky_trees(sky):
    """The sky with a strongly textured tree line below el = -3.9 deg (the bottom quarter of a level frame)."""
    rng = np.random.default_rng(3)
    t = sky.copy()
    y0 = int(V0 + math.radians(3.9) * F_TEX)
    tex = 60 + 50 * _blur(rng.standard_normal((TEX_H - y0, TEX_W)).astype(np.float32), 2.0) * 4
    t[y0:] = np.clip(_blur(tex, 0.8), 5, 250)
    return t


@pytest.fixture(scope="module")
def cam():
    return make_cam()


@pytest.fixture(scope="module")
def mount():
    return make_mount()


@pytest.fixture(scope="module")
def rays(cam):
    return make_rays(cam)


@pytest.fixture(scope="module")
def world():
    return make_world()


@pytest.fixture(scope="module")
def sky():
    return make_sky()


@pytest.fixture(scope="module")
def sky_trees(sky):
    return make_sky_trees(sky)


def R_of(t_s, yaw_dps=0.0, pitch_dps=0.0, roll_dps=0.0, pitch0=0.0):
    d = math.radians
    return rot_rpy(d(roll_dps * t_s), d(pitch0 + pitch_dps * t_s), d(yaw_dps * t_s))


def render(tex, R_wg, mount, rays, gain=1.0):
    R_wc = R_wg @ mount.R_gun_cam
    d = rays @ R_wc.T
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2], -1, 1))
    mx = (U0 - az * F_TEX).reshape(H, W).astype(np.float32)
    my = (V0 - el * F_TEX).reshape(H, W).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if gain != 1.0:
        img = (img - 4.0) * gain + 4.0
    return img


def world_pixel(az, el, R_wg, mount, cam):
    """Pixel of world direction (az, el) (radians) in the frame with attitude R_wg."""
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    R_wc = R_wg @ mount.R_gun_cam
    return cam.ray_to_pixel(R_wc.T @ d)


def draw_disc(img, u, v, diam, value):
    r = diam / 2.0
    x0, x1 = int(math.floor(u - r - 2)), int(math.ceil(u + r + 2))
    y0, y1 = int(math.floor(v - r - 2)), int(math.ceil(v + r + 2))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W - 1), min(y1, H - 1)
    if x1 <= x0 or y1 <= y0:
        return
    ss = 4
    xs = np.arange(x0 * ss, (x1 + 1) * ss) / ss + (0.5 / ss - 0.5)
    ys = np.arange(y0 * ss, (y1 + 1) * ss) / ss + (0.5 / ss - 0.5)
    inside = ((xs[None, :] - u) ** 2 + (ys[:, None] - v) ** 2 <= r * r).astype(np.float32)
    cov = inside.reshape(y1 - y0 + 1, ss, x1 - x0 + 1, ss).mean(axis=(1, 3))
    img[y0:y1 + 1, x0:x1 + 1] = img[y0:y1 + 1, x0:x1 + 1] * (1 - cov) + value * cov


def to_u8(img, rng, sigma=1.5):
    return np.clip(np.rint(img + rng.normal(0, sigma, img.shape)), 0, 255).astype(np.uint8)


def run(det, tex, mount, rays, cam, n=40, yaw=0.0, pitch=0.0, roll=0.0, disc=None, times=None, gain_fn=None,
        imu=True, gyro_bias_dps=(0.0, 0.0, 0.0), seed=1, post=None):
    """Render n frames and detect. disc(t, R_wg) -> (u, v, diam, value) or None. Returns per frame
    (t_s, dets, truth (u, v) | None, R_wg)."""
    rng = np.random.default_rng(seed)
    out = []
    ts = times if times is not None else [k * DT_NS for k in range(n)]
    for k, t_ns in enumerate(ts):
        t = t_ns * 1e-9
        R = R_of(t, yaw, pitch, roll)
        img = render(tex, R, mount, rays, gain_fn(k) if gain_fn else 1.0)
        truth = None
        if disc is not None:
            dd = disc(t, R)
            if dd is not None:
                u, v, diam, val = dd
                draw_disc(img, u, v, diam, val)
                truth = (u, v, diam)
        if post is not None:
            post(img)
        fr = Frame(image=to_u8(img, rng), seq=k, t_exposure_ns=1_000_000_000 + int(t_ns), t_capture_ns=0)
        Rg = R
        if any(gyro_bias_dps):   # the IMU's attitude drifts: the registration has to correct it
            Rg = R_of(t, *gyro_bias_dps) @ R
        dets = det.detect(fr, Rg if imu else None)
        out.append((t, dets, truth, R))
    return out


WARM = 6  # frames before both references exist and the ring has lag_b behind it


# ------------------------------------------------------------------------------- integer primitives
def test_primitives_match_their_definitions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (201, 323), dtype=np.uint8)
    d = M.box_down2(a)
    s = a[:200, :322].astype(np.int32)
    want = (s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2] + 2) >> 2
    assert np.array_equal(d, want)
    # block-integer warp: remap path == plain gather definition
    v = rng.integers(0, 256, a.shape, dtype=np.uint8)
    m = np.array([[math.cos(0.04), -math.sin(0.04), 13.3], [math.sin(0.04), math.cos(0.04), -21.6], [0, 0, 1]])
    dx, dy, _ = M.block_offsets(m, *a.shape)
    o, ov = np.empty_like(a), np.empty_like(a)
    M.warp_gather(a, v, dx, dy, M.WARP_BLOCK, o, ov)
    r, rv = M.warp_gather_ref(a, v, dx, dy, M.WARP_BLOCK)
    assert np.array_equal(o, r) and np.array_equal(ov, rv)
    # block warp error against the exact transform <= ~0.6 px at this roll
    xc = M.block_centres(*a.shape)
    assert np.abs(dx - (m[0, 0] * xc[None] + m[0, 1] * np.arange(a.shape[0])[:, None] + m[0, 2] - xc[None])).max() < 0.5 + 1e-9
    # fixed-point gain
    for g, off in ((300, -500), (200, 900), (256, 128), (1024, -20000)):
        out = np.empty_like(a)
        M.apply_gain_q8(a, g, off, out)
        assert np.array_equal(out, np.clip((a.astype(np.int64) * g + off) >> 8, 0, 255))
    # envelope residual == its definition (3x3 max/min inside the frame)
    b = rng.integers(0, 256, a.shape, dtype=np.uint8)
    c = rng.integers(0, 256, a.shape, dtype=np.uint8)
    va = np.where(rng.random(a.shape) < 0.9, 255, 0).astype(np.uint8)
    vb = np.where(rng.random(a.shape) < 0.9, 255, 0).astype(np.uint8)
    bufs = {k: np.empty_like(a) for k in ("hi", "lo", "t1", "t2")}
    D = np.empty_like(a)
    M.envelope_residual(a, b, c, va, vb, None, 3, bufs, D)
    pad = lambda x, val: np.pad(x.astype(np.int32), 1, constant_values=val)  # noqa: E731
    def mx(x):
        p = pad(x, -1)
        return np.max([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)], 0)
    def mn(x):
        p = pad(x, 999)
        return np.min([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)], 0)
    A = a.astype(np.int32)
    br = np.minimum(np.clip(A - mx(b), 0, 255), np.clip(A - mx(c), 0, 255))
    dk = np.minimum(np.clip(mn(b) - A, 0, 255), np.clip(mn(c) - A, 0, 255))
    want = np.maximum(br, dk) & va & vb
    assert np.array_equal(D, want)
    # cell sums, CFAR threshold cells and the compare
    SD = M.cell_sums(D, 8)
    ND = M.cell_sums(np.ones_like(D), 8)
    sc = np.zeros((26, 41), np.int64)
    for i in range(26):
        for j in range(41):
            sc[i, j] = int(D[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.int64).sum())
    assert np.array_equal(SD, sc)
    thr, mD = M.threshold_cells(SD, ND, SD, ND, 1.0, 4.0, 4.0, 6.0, 2.0, 8)
    i, j = 10, 20
    sl = (slice(i - 2, i + 3), slice(j - 2, j + 3))
    m = SD[sl].sum() / ND[sl].sum()
    assert thr[i, j] == min(255, int(math.floor(4.0 + 6.0 * m + 2.0 * max(0.0, m - 1.0 / M.ABS_TO_SIGMA))))
    binary = M.compare_cells(D, thr, 8)
    big = np.repeat(np.repeat(thr, 8, 0), 8, 1)[:a.shape[0], :a.shape[1]]
    assert np.array_equal(binary != 0, D > big)
    # SAD surface: shift found where it was put, masked pixels excluded
    f = rng.integers(0, 256, (100, 160), dtype=np.uint8)
    g = np.roll(np.roll(f, 2, 1), -1, 0)        # g(p + (2, -1)) = f(p)
    vv = np.full_like(f, 255)
    sums, cnts = M.sad_surface(f, g, vv, vv, 0, 0, 3)
    j, i = np.unravel_index(np.argmin(sums / cnts), sums.shape)
    assert (i - 3, j - 3) == (2, -1)


# ----------------------------------------------------------------------------- (1) static scenery
@pytest.mark.parametrize("yaw,pitch,roll", [(0.0, 0.0, 0.0), (20.0, -5.0, 10.0), (80.0, 10.0, 10.0)])
def test_static_scene_gives_nothing(world, mount, rays, cam, yaw, pitch, roll):
    det = MotionDetector(_cfg(), None, cam, mount)
    res = run(det, world, mount, rays, cam, n=40, yaw=yaw, pitch=pitch, roll=roll, gyro_bias_dps=(1.5, -1.0, 0.0))
    n = [len(d) for _, d, _, _ in res[WARM:]]
    assert sum(n) == 0, n


def test_static_scene_scale1(world, mount, rays, cam):
    det = MotionDetector(_cfg(scale=1), None, cam, mount)
    res = run(det, world, mount, rays, cam, n=25, yaw=20.0, roll=10.0, gyro_bias_dps=(1.5, -1.0, 0.0))
    assert sum(len(d) for _, d, _, _ in res[WARM:]) == 0


def test_roll_sign_is_proved_by_the_prior(world, mount, rays, cam):
    """With the gyro the only source of rotation (registration refines translation only), a fast roll cancels
    only if the image roll sign the prior derives from the camera model is right."""
    det = MotionDetector(_cfg(), None, cam, mount)
    res = run(det, world, mount, rays, cam, n=20, roll=40.0)
    assert sum(len(d) for _, d, _, _ in res[WARM:]) == 0
    # and the prior's image rotation is MINUS the gun roll about the bore in this convention
    Ra, Rb = R_of(0.0), R_of(0.1, roll_dps=40.0)
    m = det._prior(Rb @ mount.R_gun_cam, Ra @ mount.R_gun_cam, 0.1)   # cur (rolled) -> ref
    ang = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    assert ang == pytest.approx(4.0, abs=0.05)   # cur -> ref rotates +4 deg: the image rolled by -4 deg


# --------------------------------------------------------------------------------- (2) moving disc
def _mover(cam, mount, rate_dps, diam, value, az0=0.02, el0=0.0, ang=0.6):
    w = math.radians(rate_dps)
    def f(t, R):
        az = az0 - w * math.cos(ang) * t
        el = el0 + w * math.sin(ang) * t
        u, v = world_pixel(az, el, R, mount, cam)
        return float(u), float(v), diam, value
    return f


def centred(cam, mount, rate, diam, value, cam_yaw_dps, t_c=0.3, ang=math.pi - 0.3):
    """A _mover that crosses the middle of the frame at t_c while the gun swings at cam_yaw_dps."""
    wx, wy = -math.radians(rate) * math.cos(ang), math.radians(rate) * math.sin(ang)   # az / el rates of _mover
    return _mover(cam, mount, rate, diam, value, az0=-(wx - math.radians(cam_yaw_dps)) * t_c, el0=-wy * t_c,
                  ang=ang)


@pytest.mark.parametrize("rate", [5.0, 20.0, 60.0])
@pytest.mark.parametrize("diam", [4.0, 8.0, 20.0])
@pytest.mark.parametrize("value", [235.0, 15.0])
def test_moving_disc_is_detected(world, mount, rays, cam, rate, diam, value):
    det = MotionDetector(_cfg(), None, cam, mount)
    # moving left and a little up against the world while the gun swings left at 20 deg/s: centred at t = 0.3 s
    res = run(det, world, mount, rays, cam, n=30, yaw=20.0, roll=5.0, gyro_bias_dps=(1.0, 0.5, 0.0),
              disc=centred(cam, mount, rate, diam, value, 20.0))
    # Frame differencing sees only the part of a mover that left its old footprint: the centroid and size are
    # exact once it moves at least its own size per lag_a, and the residual where references A and B BOTH still
    # had it (a ghost) is gone once it moves its size between B and A. Slower than that it is still detected
    # (on its leading crescent), with the tolerance of its own radius.
    move_a = math.radians(rate) * 0.040 * FX     # px per lag_a against the world
    move_ab = math.radians(rate) * 0.060 * FX    # px between the two references
    tight = move_a >= diam
    no_ghost = move_ab >= diam
    tol = 1.5 if tight else diam / 2 + 2.0
    hit = one = sized = n = 0
    for _, dets, truth, _ in res[WARM:]:
        e = diam + 6
        if truth is None or not (e < truth[0] < W - e and e < truth[1] < H - e):
            continue
        n += 1
        near = [d for d in dets if math.hypot(d.u - truth[0], d.v - truth[1]) <= tol]
        if near:
            hit += 1
            one += len(dets) == 1
            if abs(near[0].diameter_px - diam) <= 0.25 * diam:
                sized += 1
    assert n >= 15
    assert hit >= 0.9 * n, (hit, n)
    if no_ghost:
        assert one >= 0.9 * n, (one, n)        # exactly one candidate: no ghost where it was, no scenery
    if tight:
        assert sized >= 0.9 * n, (sized, n)


# ------------------------------------------------------------------------------------ (3) stab
def test_stab_static_point_constant_and_disc_rate(world, mount, rays, cam):
    det = MotionDetector(_cfg(), None, cam, mount)
    rate = 20.0
    pts, stabs, ts = [], [], []
    disc = centred(cam, mount, rate, 8.0, 235.0, 30.0, t_c=0.3, ang=math.pi)
    rng = np.random.default_rng(5)
    for k in range(30):
        t = k * DT_NS * 1e-9
        R = R_of(t, 30.0, 0.0, 10.0)
        img = render(world, R, mount, rays)
        u, v, dm, val = disc(t, R)
        draw_disc(img, u, v, dm, val)
        fr = Frame(image=to_u8(img, rng), seq=k, t_exposure_ns=10**9 + k * DT_NS, t_capture_ns=0)
        dets = det.detect(fr, R_of(t, 0.0, 0.5, 1.0) @ R)
        p = world_pixel(0.10, 0.0, R, mount, cam)       # a static world point
        if 0 < p[0] < W and 0 < p[1] < H:
            pts.append(det.to_stab(float(p[0]), float(p[1])))
        if k >= WARM:
            near = [d for d in dets if math.hypot(d.u - u, d.v - v) < 3]
            if near:
                stabs.append(near[0].stab)
                ts.append(t)
    pts = np.array(pts[-20:])
    assert len(pts) >= 20
    assert np.abs(pts - pts.mean(0)).max() <= 1.5
    st = np.array(stabs)
    assert len(st) >= 15
    vel = np.polyfit(ts, st[:, 0], 1)[0], np.polyfit(ts, st[:, 1], 1)[0]
    assert math.hypot(*vel) == pytest.approx(math.radians(rate) * FX, rel=0.08)


# ---------------------------------------------------------------------------- (4) exposure step
def test_exposure_steps_give_nothing(world, mount, rays, cam):
    det = MotionDetector(_cfg(), None, cam, mount)
    gains = [1.0, 1.0, 1.0, 1.3, 1.3, 0.7, 0.7, 1.0, 1.3, 0.91, 0.91, 0.91, 1.18, 0.7, 1.0] * 3
    res = run(det, world, mount, rays, cam, n=len(gains), yaw=20.0, gain_fn=lambda k: gains[k] * 0.8)
    assert sum(len(d) for _, d, _, _ in res[WARM:]) == 0


# ------------------------------------------------------------------------- (5) irregular timing
def test_irregular_frame_spacing(world, mount, rays, cam):
    rng = np.random.default_rng(11)
    ts = np.cumsum(rng.integers(10, 91, 40)) * 1_000_000
    det = MotionDetector(_cfg(), None, cam, mount)
    res = run(det, world, mount, rays, cam, times=list(ts), yaw=20.0, roll=5.0,
              disc=centred(cam, mount, 30.0, 8.0, 235.0, 20.0, t_c=float(ts[20]) * 1e-9))
    hit = n = static = 0
    for _, dets, truth, _ in res[WARM:]:
        if truth is None or not (15 < truth[0] < W - 15 and 15 < truth[1] < H - 15):
            static += len(dets)
            continue
        n += 1
        hit += any(math.hypot(d.u - truth[0], d.v - truth[1]) < 2.0 for d in dets)
        static += sum(1 for d in dets if math.hypot(d.u - truth[0], d.v - truth[1]) >= 2.0)
    assert n >= 8 and hit >= 0.9 * n, (hit, n)
    assert static == 0


# ------------------------------------------------------------------------------------ (6) no IMU
def test_no_imu(world, mount, rays, cam):
    det = MotionDetector(_cfg(), None, cam, mount)
    res = run(det, world, mount, rays, cam, n=40, yaw=20.0, pitch=3.0, imu=False,
              disc=centred(cam, mount, 30.0, 8.0, 15.0, 20.0, t_c=0.5))
    hit = n = other = 0
    for _, dets, truth, _ in res[WARM + 4:]:
        ok = truth is not None and 15 < truth[0] < W - 15 and 15 < truth[1] < H - 15
        if ok:
            n += 1
            hit += any(math.hypot(d.u - truth[0], d.v - truth[1]) < 2.0 for d in dets)
        other += sum(1 for d in dets if not ok or math.hypot(d.u - truth[0], d.v - truth[1]) >= 2.0)
    assert n >= 10 and hit >= 0.9 * n, (hit, n)
    assert other == 0


# ----------------------------------------------------------------------------------- (7) masked
def test_masked_region_gives_nothing(world, mount, rays, cam):
    mask = np.full((H, W), 255, np.uint8)
    mask[150:300, 200:500] = 0
    det = MotionDetector(_cfg(), mask, cam, mount)
    # a disc crossing the masked rectangle, the camera swinging
    def disc(t, R):
        return 210 + 1000 * t, 225.0, 10.0, 235.0
    res = run(det, world, mount, rays, cam, n=30, yaw=20.0, disc=disc)
    for _, dets, truth, _ in res[WARM:]:
        for d in dets:
            assert not (195 <= d.u <= 505 and 145 <= d.v <= 305), (d.u, d.v)
    # and it is seen once out of the mask
    assert any(dets for _, dets, truth, _ in res[WARM:] if truth[0] > 520)


# ------------------------------------------------------------------- texture guard / open sky
@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("swing", [3.0, 10.0, 30.0])
@pytest.mark.parametrize("drift", [0.0, 2.0])
def test_open_sky_followed_clay_is_never_cancelled(sky, mount, rays, cam, scale, swing, drift):
    """Plain sky + noise, a gun-fixed dark bar (masked), the gun swinging, a disc the shooter follows (held still
    or drifting 2 px/frame across the swing): seen every frame, registration not trusted (prior stands), stab
    advancing at the swing rate."""
    mask = np.full((H, W), 255, np.uint8)
    mask[360:, 250:390] = 0
    det = MotionDetector(_cfg(scale), mask, cam, mount)

    def bar(img):
        img[365:, 260:380] = 25.0          # gun-fixed: in the image, not the world

    def disc(t, R):
        return 300.0, 150.0 + drift * t / (DT_NS * 1e-9), 8.0, 40.0
    res = run(det, sky, mount, rays, cam, n=30, yaw=swing, disc=disc, post=bar)
    seen, oks, st, ts = 0, [], [], []
    for t, dets, truth, _ in res[WARM:]:
        near = [d for d in dets if math.hypot(d.u - truth[0], d.v - truth[1]) < 3.0]
        seen += bool(near)
        assert len(dets) <= 1 + len(near), [(d.u, d.v) for d in dets]
        oks.append(det.stats["reg_a"]["ok"])
        if near:
            st.append(near[0].stab)
            ts.append(t)
    n = len(res) - WARM
    assert seen == n, (seen, n)
    assert not any(oks)
    st = np.array(st)
    vx = np.polyfit(ts, st[:, 0], 1)[0]
    assert abs(vx) == pytest.approx(math.radians(swing) * FX, rel=0.1)


def test_slow_mover_on_sky_is_seen_by_the_slow_test(sky_trees, mount, rays, cam):
    """vision.slow_gain: a disc going 1.5 deg/s against the world over the sky of a frame the tree line registers
    (a clay going away once the gun steadies) moves less than its own size per lag_a and cancels itself in the A/B
    test; compared also with B and a frame about slow_gain x B's lag back, it is seen in every frame, and nothing
    on the trees is."""
    seen = {}
    for gain in (0.0, 2.5):
        det = MotionDetector(_cfg(2, sets=(f"vision.slow_gain={gain}",)), None, cam, mount)
        res = run(det, sky_trees, mount, rays, cam, n=40, disc=centred(cam, mount, 1.5, 8.0, 140.0, 0.0, t_c=0.4),
                  gyro_bias_dps=(1.0, 0.5, 0.0))
        n = 0
        seen[gain] = 0
        for _, dets, truth, _ in res[15:]:     # 15 frames: the slow test's reference is 260 ms back
            n += 1
            near = [d for d in dets if math.hypot(d.u - truth[0], d.v - truth[1]) < 7.0]
            seen[gain] += bool(near)
            assert len(dets) == len(near), [(d.u, d.v) for d in dets]
        if gain:
            assert det.stats["slow"]["lag_ms"] >= 200.0
    assert seen[0.0] <= n // 5, seen          # the A/B test alone: (almost) never
    assert seen[2.5] == n, seen


@pytest.mark.parametrize("scale", [1, 2])
def test_tree_line_registers_and_stays_quiet(sky_trees, mount, rays, cam, scale):
    # (with the local residual field, which is off by default: vision.local_field)
    det = MotionDetector(_cfg(scale, sets=("vision.local_field=true",)), None, cam, mount)

    def disc(t, R):
        return 330.0, 120.0, 8.0, 40.0          # held still in the image, on the sky part
    res = run(det, sky_trees, mount, rays, cam, n=30, yaw=10.0, disc=disc, gyro_bias_dps=(2.0, 1.0, 0.0))
    seen = 0
    for _, dets, truth, _ in res[WARM:]:
        assert det.stats["reg_a"]["ok"] and det.stats["reg_b"]["ok"]
        for d in dets:
            assert math.hypot(d.u - truth[0], d.v - truth[1]) < 3.0, (d.u, d.v)   # nothing on the trees
        seen += len(dets) == 1
        # the local-field cell holding only the disc on plain sky has no residual of its own
        fok = det.stats["reg_a"].get("field_ok")
        assert fok is not None
        gr, gc = fok.shape
        assert not fok[int(truth[1] / H * gr), int(truth[0] / W * gc)]
    assert seen == len(res) - WARM


# ------------------------------------------------------------------------- (8) real press regression
PRESS = Path(__file__).resolve().parents[1] / "pi-pulls" / "pi-2026-11-20" / "sessions" / "20261120-072712-press"
# The thrown ball, verified on the contact sheets (2026-11-20). Rows 123 and 126-131 are left out: the ball hangs at
# the top of its flight there, still against the world for over 100 ms (it sits on the same twig of the tree in the
# current frame and both registered references), which a motion detector cannot see by design; a bird crossing the
# sky (46,319 -> 479,9 over rows 119-128) is then the strongest candidate, rightly.
BALL = {
    104: (299, 373), 105: (304, 364), 106: (306, 359), 107: (308, 354), 108: (310, 349), 109: (311, 342),
    110: (313, 340), 111: (314, 336), 112: (313, 331), 113: (314, 328), 114: (313, 325), 115: (313, 322),
    116: (312, 318), 117: (311, 316), 118: (309, 314), 119: (306, 309), 120: (303, 306), 121: (301, 302),
    122: (299, 298), 124: (297, 283), 125: (296, 276), 132: (315, 178), 133: (321, 137), 134: (328, 104),
    135: (331, 78), 136: (337, 46),
    158: (343, 14), 159: (341, 23), 160: (340, 33), 161: (338, 41), 162: (337, 51), 163: (335, 60), 164: (333, 70),
    165: (330, 80), 166: (329, 89), 167: (327, 98), 168: (325, 107), 169: (325, 115), 170: (323, 122),
    171: (322, 130), 172: (320, 136), 173: (319, 142), 174: (317, 147), 175: (316, 151), 176: (315, 155),
    177: (316, 157), 178: (316, 158), 179: (316, 159), 180: (318, 157), 181: (319, 156), 182: (321, 154),
    183: (322, 150), 184: (326, 144), 185: (328, 139), 186: (331, 131),
}


@pytest.mark.skipif(not PRESS.exists(), reason="press 20261120-072712 not present")
def test_real_press_ball():
    counts, top = {}, {}
    for k, fr, dets, det in M.iter_press(PRESS, start=60, stop=191):
        counts[k] = det.stats.get("candidates", len(dets))
        top[k] = dets[0] if dets else None
    bad = [(r, None if top[r] is None else (round(top[r].u), round(top[r].v)))
           for r, (u, v) in BALL.items() if top[r] is None or math.hypot(top[r].u - u, top[r].v - v) > 12.0]
    assert not bad, bad
    assert np.median(list(counts.values())) <= 20


# -------------------------------------------------- stab_dir from a gun that starts pointing down
@pytest.mark.parametrize("native", [True, False])
def test_stab_dir_survives_a_90_degree_swing(world, mount, rays, cam, native):
    """The gun starts pitched 75 deg down (static, textured world), comes up through level over 1.5 s and then
    holds still while a disc crosses: the stabilised plane of the first frame is useless 90 deg away (stab
    stretched or not finite), the stabilised direction is not: consecutive stab_dir of a static world point agree,
    and the disc's step between frames is its true angular step."""
    det = MotionDetector(_cfg(sets=[f"vision.native={'true' if native else 'false'}"]), None, cam, mount)
    t_up = 1.5
    def R_at(t):
        return R_of(min(t, t_up), pitch_dps=60.0, pitch0=-75.0)
    R_end = R_at(t_up)
    R_wc_end = R_end @ mount.R_gun_cam
    d0 = R_wc_end @ cam.pixel_to_ray(np.array([W * 0.35, H * 0.5]))
    az0, el0 = math.atan2(d0[1], d0[0]), math.asin(d0[2])
    rate = math.radians(20.0)
    def disc_dir(t):
        az = az0 - rate * (t - t_up)
        return np.array([math.cos(el0) * math.cos(az), math.cos(el0) * math.sin(az), math.sin(el0)])
    rng = np.random.default_rng(9)
    pt_prev = disc_prev = None
    static_err, ratios, stabs = [], [], []
    p_world = R_wc_end @ cam.pixel_to_ray(np.array([W * 0.7, H * 0.3]))   # a static point in the final view
    n = int((t_up + 0.5) / (DT_NS * 1e-9))
    for k in range(n):
        t = k * DT_NS * 1e-9
        R = R_at(t)
        img = render(world, R, mount, rays)
        truth = None
        if t >= t_up:
            u, v = cam.ray_to_pixel((R @ mount.R_gun_cam).T @ disc_dir(t))
            draw_disc(img, u, v, 10.0, 235.0)
            truth = (u, v)
        fr = Frame(image=to_u8(img, rng), seq=k, t_exposure_ns=10**9 + k * DT_NS, t_capture_ns=0)
        dets = det.detect(fr, R)
        S = det._slots[det._order[-1]].S
        KS = (det._Kinv @ S).tolist()
        q = cam.ray_to_pixel((R @ mount.R_gun_cam).T @ p_world)
        if t >= t_up - 0.3 and np.all(np.isfinite(q)) and 0 < q[0] < W and 0 < q[1] < H:
            dd = np.array(M._stab_dir(KS, float(q[0]), float(q[1])))
            if pt_prev is not None:
                static_err.append(math.acos(min(1.0, float(dd @ pt_prev))))
            pt_prev = dd
        if truth is not None and k * DT_NS * 1e-9 >= t_up + 2 * DT_NS * 1e-9:
            near = [d for d in dets if math.hypot(d.u - truth[0], d.v - truth[1]) < 3.0]
            if near:
                dd = np.array(near[0].stab_dir)
                assert abs(np.linalg.norm(dd) - 1.0) < 1e-12
                stabs.append(near[0].stab)
                if disc_prev is not None:
                    true_step = math.acos(min(1.0, float(disc_dir(t) @ disc_dir(t - DT_NS * 1e-9))))
                    ratios.append(math.acos(min(1.0, float(dd @ disc_prev))) / true_step)
                disc_prev = dd
            else:
                disc_prev = None
    assert len(static_err) >= 15 and max(static_err) < 0.002, max(static_err)
    assert len(ratios) >= 10
    assert abs(float(np.median(ratios)) - 1.0) < 0.05, ratios
    # the plane of the first frame is stretched far out (or not finite) up there
    st = np.array(stabs)
    assert (not np.all(np.isfinite(st))) or np.abs(st).max() > 10 * W
    assert det.stab_epoch == 0
