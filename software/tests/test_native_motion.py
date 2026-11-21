"""The motion detector's native kernel (tbot_motion_* in tbot/vision/_measure.c) gives exactly what the Python
reference in tbot/vision/motion.py gives: every primitive on random inputs (odd sizes, offsets pointing outside
the frame, masks, empty and saturated images), and the whole detector (MotionDetector with vision.native on and
off: identical Detection lists, every field, and identical registration stats) on a synthetic moving scene and on
a real press when one is present.

Skipped when the shared library is not built (scripts/build_native.sh).
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np
import pytest

from tbot.vision import _native
from tbot.vision import motion as M

K = _native.motion_kernel()
pytestmark = pytest.mark.skipif(K is None, reason=f"native motion kernel: {_native.status} / {_native.motion_status}")

SIZES = [(1, 1), (2, 3), (7, 5), (33, 17), (50, 80), (101, 161), (200, 320), (199, 321)]


def _p(a):
    return None if a is None else a.ctypes.data


def _imgs(rng, h, w, kind):
    if kind == "zero":
        return np.zeros((h, w), np.uint8)
    if kind == "sat":
        return np.full((h, w), 255, np.uint8)
    if kind == "few":
        return rng.choice(np.array([0, 1, 254, 255], np.uint8), (h, w))
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _mask(rng, h, w, frac=0.85):
    return np.where(rng.random((h, w)) < frac, 255, 0).astype(np.uint8)


KINDS = ["rand", "zero", "sat", "few"]


# ------------------------------------------------------------------------------------------ primitives
@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("kind", KINDS)
def test_box_down2_texture_tensor(h, w, kind):
    rng = np.random.default_rng(h * 1000 + w)
    a = _imgs(rng, h, w, kind)
    if h >= 2 and w >= 2:
        d = np.empty((h // 2, w // 2), np.uint8)
        K.box_down2(_p(a), h, w, _p(d))
        assert np.array_equal(d, M.box_down2(a))
    for thr in (0, 4, 200):
        t = np.empty_like(a)
        v = _mask(rng, h, w)
        ten = np.zeros(3, np.int64)
        K.texture(_p(a), h, w, thr, _p(t), _p(v), _p(ten))
        want = M.texture_map(a, thr)
        assert np.array_equal(t, want)
        assert tuple(int(x) for x in ten) == M.texture_tensor(a, want, v)


def test_numpy_summation_order():
    """The kernel's float64 sums (block_offsets' mean residual, the local field's) follow this NumPy's."""
    rng = np.random.default_rng(77)
    for n in list(range(1, 300)) + [1000, 8191, 8192, 8193, 16384, 16385, 30000, 70001]:
        a = rng.standard_normal(n) * 10.0 ** rng.integers(-4, 5, n)
        assert K.sum(_p(a), n, K.np_chunk) == float(a.sum()), n
        assert K.sum(_p(a), n, K.np_chunk) / n == float(a.reshape(2, -1).mean() if n % 2 == 0 else a.mean()), n


@pytest.mark.parametrize("trial", range(40))
def test_block_offsets(trial):
    rng = np.random.default_rng(trial)
    h, w = int(rng.integers(1, 210)), int(rng.integers(1, 330))
    block = int(rng.choice([16, 16, 5, 1]))
    m = np.eye(3)
    m[:2, :2] += rng.normal(0, 0.03, (2, 2))
    m[:2, 2] = rng.normal(0, 30, 2)
    if trial % 3 == 0:
        m[2, :2] = rng.normal(0, 1e-4, 2)
    rows = 2 if trial % 5 == 4 else 3
    nb = (w + block - 1) // block
    extra = (rng.normal(0, 2, (h, nb)), rng.normal(0, 2, (h, nb))) if trial % 2 else None
    dx, dy = np.empty((h, nb), np.int32), np.empty((h, nb), np.int32)
    rx, ry = np.empty((h, nb)), np.empty((h, nb))
    res = np.empty(2)
    mm = np.ascontiguousarray(m[:rows])
    ex = None if extra is None else np.ascontiguousarray(extra[0])
    ey = None if extra is None else np.ascontiguousarray(extra[1])
    K.block_offsets(_p(mm), rows, h, w, block, _p(ex), _p(ey), _p(dx), _p(dy), _p(rx), _p(ry), _p(res), K.np_chunk)
    wdx, wdy, wres, (wrx, wry) = M.block_offsets(mm, h, w, block, extra=extra, full=True)
    assert np.array_equal(dx, wdx) and np.array_equal(dy, wdy)
    assert np.array_equal(rx, wrx) and np.array_equal(ry, wry)
    assert res.tobytes() == wres.tobytes()


@pytest.mark.parametrize("trial", range(30))
def test_warp_gather(trial):
    rng = np.random.default_rng(100 + trial)
    h, w = int(rng.integers(1, 210)), int(rng.integers(1, 330))
    src, sv = _imgs(rng, h, w, "rand"), _mask(rng, h, w)
    nb = (w + 15) // 16
    big = 40000 if trial % 7 == 0 else 2 * max(h, w)   # far outside the frame, int16 wrap included
    dx = rng.integers(-big, big + 1, (h, nb)).astype(np.int32)
    dy = rng.integers(-big, big + 1, (h, nb)).astype(np.int32)
    if trial % 2:
        dx //= 20
        dy //= 20
    o, ov = np.empty_like(src), np.empty_like(src)
    K.warp(_p(src), _p(sv), h, w, _p(dx), _p(dy), 16, _p(o), _p(ov))
    r, rv = np.empty_like(src), np.empty_like(src)
    M.warp_gather(src, sv, dx, dy, 16, r, rv)
    assert np.array_equal(o, r) and np.array_equal(ov, rv)


def test_gain():
    rng = np.random.default_rng(3)
    a = _imgs(rng, 37, 53, "rand")
    a[0, :2] = (0, 255)
    for g, o in ((256, 128), (300, -500), (200, 900), (1024, -20000), (64, 300000), (0, -1), (4000, 5)):
        out = np.empty_like(a)
        K.gain(_p(a), a.size, g, o, _p(out))
        want = np.empty_like(a)
        M.apply_gain_q8(a, g, o, want)
        assert np.array_equal(out, want), (g, o)


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("env", [1, 2, 3, 4, 5])
def test_envelope(h, w, env):
    rng = np.random.default_rng(h + w + env)
    cur, wa, wb = (_imgs(rng, h, w, k) for k in ("rand", "rand", "few"))
    va, vb, mk = _mask(rng, h, w), _mask(rng, h, w), _mask(rng, h, w, 0.7)
    for mask in (None, mk):
        D = np.empty_like(cur)
        scratch = np.empty(5 * h * w, np.uint8)
        K.envelope(_p(cur), _p(wa), _p(wb), _p(va), _p(vb), _p(mask), h, w, env, _p(scratch), _p(D))
        bufs = {k: np.empty_like(cur) for k in ("hi", "lo", "t1", "t2")}
        want = np.empty_like(cur)
        M.envelope_residual(cur, wa, wb, va, vb, mask, env, bufs, want)
        assert np.array_equal(D, want)


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("cell", [2, 8, 7])
def test_cell_sums_absdiff(h, w, cell):
    rng = np.random.default_rng(h * w + cell)
    a, b = _imgs(rng, h, w, "rand"), _imgs(rng, h, w, "few")
    v = _mask(rng, h, w)
    v[rng.random((h, w)) < 0.1] = 7          # validity that is not 0 / 255: bitwise semantics
    ch, cw = -(-h // cell), -(-w // cell)
    for rs in (1, 2, 3):
        out = np.empty((ch, cw), np.int64)
        K.cell_sums(_p(a), h, w, cell, rs, _p(out))
        assert np.array_equal(out, M.cell_sums(a, cell, rs))
        s, c = np.empty((ch, cw), np.int64), np.empty((ch, cw), np.int64)
        K.absdiff_sums(_p(a), _p(b), _p(v), h, w, cell, rs, _p(s), _p(c))
        ws, wc = M.cell_absdiff_sums(a, b, v, cell, rs)
        assert np.array_equal(s, ws) and np.array_equal(c, wc)


@pytest.mark.parametrize("trial", range(40))
def test_noise_threshold_compare(trial):
    rng = np.random.default_rng(500 + trial)
    cell = int(rng.choice([8, 4, 5]))
    h, w = int(rng.integers(1, 210)), int(rng.integers(1, 330))
    ch, cw = -(-h // cell), -(-w // cell)
    full = cell * (-(-cell // 2))
    SA = rng.integers(0, 40 * full, (ch, cw)).astype(np.int64)
    NA = rng.integers(0, full + 1, (ch, cw)).astype(np.int64)
    if trial % 4 == 0:
        NA[...] = 0
    SD = rng.integers(0, 255 * cell * cell, (ch, cw)).astype(np.int64)
    ND = np.maximum(rng.integers(0, cell * cell + 1, (ch, cw)).astype(np.int64), 1)
    sig = K.noise(_p(SA), _p(NA), ch * cw, cell, 2, M.NOISE_QUANTILE, M.NOISE_MIN_FILL, M.ABS_TO_SIGMA)
    want = M.noise_sigma_from_cells(SA, NA, cell)
    assert (math.isnan(sig) and math.isnan(want)) or sig == want
    for s_in in (sig, float("nan"), 0.0, 3.7):
        thr = np.empty((ch, cw), np.uint8)
        mD = np.empty((ch, cw))
        prm = [float(x) for x in rng.uniform(0, 8, 4)]
        assert K.threshold(_p(SD), _p(ND), _p(SA), _p(NA), ch, cw, s_in, *prm, M.ABS_TO_SIGMA, M.CFAR_BOX, M.CFAR_GUARD,
                           _p(thr), _p(mD)) == 0
        wthr, wmD = M.threshold_cells(SD, ND, SA, NA, s_in, *prm, cell)
        assert np.array_equal(thr, wthr) and mD.tobytes() == np.ascontiguousarray(wmD).tobytes()
    D = _imgs(rng, h, w, "rand")
    b = np.empty_like(D)
    K.compare(_p(D), h, w, _p(thr), cw, cell, _p(b))
    assert np.array_equal(b, M.compare_cells(D, thr, cell))


@pytest.mark.parametrize("trial", range(25))
def test_sad_surface_and_cells(trial):
    rng = np.random.default_rng(900 + trial)
    h, w = int(rng.integers(3, 110)), int(rng.integers(3, 170))
    cur, ref = _imgs(rng, h, w, "rand"), _imgs(rng, h, w, "few" if trial % 3 == 0 else "rand")
    vc, vr = _mask(rng, h, w), _mask(rng, h, w)
    r = int(rng.integers(0, 4))
    cx, cy = int(rng.integers(-w - 3, w + 3)) // (1 if trial % 2 else 8), int(rng.integers(-h - 3, h + 3)) // 8
    region = None if trial % 3 else (int(rng.integers(0, w)), int(rng.integers(0, h)), w, h)
    x0, y0, x1, y1 = (0, 0, w, h) if region is None else region
    n = 2 * r + 1
    s, c = np.empty((n, n), np.int64), np.empty((n, n), np.int64)
    K.sad(_p(cur), _p(ref), _p(vc), _p(vr), h, w, cx, cy, r, x0, y0, x1, y1, _p(s), _p(c))
    ws, wc = M.sad_surface(cur, ref, vc, vr, cx, cy, r, region)
    assert np.array_equal(s, ws) and np.array_equal(c, wc)
    ys = np.linspace(0, h, 4).astype(np.int32)
    xs = np.linspace(0, w, 5).astype(np.int32)
    r = 2
    s = np.empty((3, 4, 5, 5), np.int64)
    c = np.empty_like(s)
    K.sad_cells(_p(cur), _p(ref), _p(vc), _p(vr), h, w, r, _p(ys), 3, _p(xs), 4, _p(s), _p(c))
    ws, wc = M.sad_cells(cur, ref, vc, vr, r, ys, xs)
    assert np.array_equal(s, ws) and np.array_equal(c, wc)


@pytest.mark.parametrize("trial", range(60))
def test_photometric_fit(trial):
    rng = np.random.default_rng(1300 + trial)
    h, w = int(rng.integers(1, 110)), int(rng.integers(1, 170))
    ref = rng.integers(0, 256, (h, w)).astype(np.uint8)
    if trial % 4 == 1:
        ref = np.clip(rng.normal(120, 1.0, (h, w)), 0, 255).astype(np.uint8)    # flat: ratio-of-means branch
    a = rng.uniform(0.2, 5.0)
    cur = np.clip(ref * a + rng.normal(rng.uniform(-50, 50), 3, (h, w)), 0, 255).astype(np.uint8)
    if trial % 5 == 2:
        cur[...] = 0
    valid = (rng.random((h, w)) < (0.02 if trial % 6 == 3 else 0.9)).astype(np.uint8)
    ab = np.empty(2)
    gq = np.empty(2, np.int32)
    K.photometric(_p(cur), _p(ref), _p(valid), h, w, M.BLACK_DN, M.GAIN_MIN, M.GAIN_MAX, M.FIT_STD_MIN, _p(ab), _p(gq))
    wa, wb, wg = M.photometric_fit(cur, ref, valid)
    assert ab[0] == wa and ab[1] == wb and tuple(int(x) for x in gq) == wg


def _measure_reference(binary, D, cur, wa, mD, cell, s, S, min_area, max_area, max_cand):
    """MotionDetector._measure's labelling + numbers on one binary image (the Python path)."""
    import cv2
    x, y, bw, bh = cv2.boundingRect(binary)
    if bw == 0:
        return 0, 0, []
    hw, ww = binary.shape
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = min(ww, x + bw + 1), min(hw, y + bh + 1)
    roi = binary[y0:y1, x0:x1]
    joined = cv2.dilate(roi, M._rect(3))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    class _V:
        pass
    det = M.MotionDetector.__new__(M.MotionDetector)
    det.vc = _V()
    det.vc.min_area_px, det.vc.max_area_px, det.vc.max_candidates = min_area, max_area, max_cand
    det.scale = s
    det._Kinv = np.eye(3)
    det.stats = {}
    det._bufs = _V()
    det._bufs.wa = wa
    slot = _V()
    slot.work, slot.S, slot.t = cur, S, 0
    dets = det._measure(slot, lab, st, x0, y0, D, mD, cell, {"ok": True}, {"ok": True}, roi)
    return int(n), det.stats.get("candidates", 0), [(d, m) for d, m in zip(dets, det.stats.get("minor_px", []))]


@pytest.mark.parametrize("trial", range(40))
def test_measure(trial):
    rng = np.random.default_rng(2000 + trial)
    h, w = int(rng.integers(1, 120)), int(rng.integers(1, 200))
    cell = 8
    if trial % 10 == 0:
        binary = np.zeros((h, w), np.uint8)
    else:
        binary = np.where(rng.random((h, w)) < rng.choice([0.002, 0.02, 0.1, 0.4]), 255, 0).astype(np.uint8)
    D = rng.integers(1, rng.choice([3, 256]), (h, w)).astype(np.uint8)    # few distinct values: ties
    cur, wa = _imgs(rng, h, w, "rand"), _imgs(rng, h, w, "rand")
    ch, cw = -(-h // cell), -(-w // cell)
    mD = rng.uniform(0, 3, (ch, cw))
    S = np.eye(3)
    S[:2, :] += rng.normal(0, 0.01, (2, 3))
    S[:2, 2] = rng.normal(0, 500, 2)
    S[2, :2] = rng.normal(0, 1e-5, 2)
    s = int(rng.choice([1, 2]))
    min_area, max_area, max_cand = (1, 2500, 12) if trial % 3 else (2, 30, int(rng.choice([0, 3, -2, 50])))
    out = np.zeros((max(max_cand, 0) + 64, _native.MOTION_DET_COLS))
    mp = _native.MotionMeas()
    mp.h, mp.w, mp.cell, mp.scale = h, w, cell, s
    mp.min_area, mp.max_area, mp.max_cand = min_area, max_area, max_cand
    joined = np.empty(h * w, np.uint8)
    runs = np.empty((h * (w // 2 + 2), 4), np.int32)
    mp.cap_runs = runs.shape[0]
    mp.binary, mp.D, mp.cur, mp.wa, mp.mD = _p(binary), _p(D), _p(cur), _p(wa), _p(mD)
    mp.joined, mp.runs, mp.out = _p(joined), _p(runs), _p(out)
    mp.S[:] = S.ravel().tolist()
    mp.pixel_box_var, mp.score_mass = M.PIXEL_BOX_VAR, M.SCORE_MASS
    assert K.measure(ctypes.byref(mp)) == 0
    n, cand, want = _measure_reference(binary, D, cur, wa, mD, cell, s, S, min_area, max_area, max_cand)
    assert mp.n_labels == n
    if n > 1:
        assert mp.n_cand == cand
    assert mp.n_out == len(want)
    for k, (d, minor) in enumerate(want):
        row = out[k].tolist()
        got = (row[0], row[1], row[2], row[3], row[4], row[5], row[6], (int(row[7]), int(row[8]), int(row[9]),
               int(row[10])), (row[11], row[12]), row[13], row[14])
        exp = (d.u, d.v, d.area_px, d.diameter_px, d.contrast, d.snr, d.score, tuple(d.bbox), tuple(d.stab), d.mass,
               minor)
        assert got == exp, (k, got, exp)


# ---------------------------------------------------------------------------------- whole detector
def _strict(x):
    """A comparable form that tells -0.0 from 0.0 and compares arrays by bytes."""
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, dict):
        return tuple(sorted((k, _strict(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_strict(v) for v in x)
    if isinstance(x, (float, np.floating)):
        return ("f", float(x).hex())
    if hasattr(x, "__dataclass_fields__"):
        return tuple((k, _strict(getattr(x, k))) for k in x.__dataclass_fields__)
    return x


def _pair(cfg_sets, mask, cam, mount):
    from tbot.config import load_config
    from tbot.vision.motion import MotionDetector
    on = MotionDetector(load_config(sets=[*cfg_sets, "vision.native=true"]), mask, cam, mount)
    off = MotionDetector(load_config(sets=[*cfg_sets, "vision.native=false"]), mask, cam, mount)
    return on, off


def _same_frame(on, off, fr, R, k):
    d1 = on.detect(fr, R)
    d2 = off.detect(fr, R)
    assert on._nat is not None and off._nat is None
    assert _strict(d1) == _strict(d2), k
    s1 = {kk: v for kk, v in on.stats.items()}
    s2 = {kk: v for kk, v in off.stats.items()}
    assert _strict(s1) == _strict(s2), k
    assert _strict(on.noise_sigma) == _strict(off.noise_sigma), k
    assert on.stab_epoch == off.stab_epoch
    assert _strict(on.to_stab(100.0, 50.0)) == _strict(off.to_stab(100.0, 50.0))
    return len(d1)


@pytest.mark.parametrize("scale", [2, 1])
@pytest.mark.parametrize("scene", ["world", "sky_trees", "horizon"])
@pytest.mark.parametrize("imu", [True, False])
def test_detector_parity_synthetic(scale, scene, imu):
    import tests.test_motion_detector as T
    from tbot.types import Frame
    cam, mount = T.make_cam(), T.make_mount()
    rays = T.make_rays(cam)
    if scene == "world":
        tex = T.make_world()
    elif scene == "sky_trees":
        tex = T.make_sky_trees(T.make_sky())
    else:                                   # an edge-only scene: the fine search re-centres across it only
        tex = T.make_sky()
        tex[int(T.V0 + math.radians(2.0) * T.F_TEX):] = 70.0
        tex = T._blur(tex, 1.2)
    mask = np.full((T.H, T.W), 255, np.uint8)
    mask[360:, 250:390] = 0
    on, off = _pair([f"vision.scale={scale}"], mask, cam, mount)
    disc = T.centred(cam, mount, 30.0, 8.0, 235.0, 20.0, t_c=0.3)
    disc2 = T.centred(cam, mount, 60.0, 5.0, 20.0, 20.0, t_c=0.4, ang=0.4)
    rng = np.random.default_rng(4)
    n = 0
    for k in range(26):
        t = k * T.DT_NS * 1e-9
        R = T.R_of(t, 20.0, 3.0, 5.0)
        img = T.render(tex, R, mount, rays, 1.0 if k % 7 else 1.15)
        for d in (disc, disc2):
            u, v, dm, val = d(t, R)
            T.draw_disc(img, u, v, dm, val)
        fr = Frame(image=T.to_u8(img, rng), seq=k, t_exposure_ns=10**9 + k * T.DT_NS, t_capture_ns=0)
        Rg = T.R_of(t, 1.0, -0.5, 0.0) @ R if imu else None
        n += _same_frame(on, off, fr, Rg, k)
    assert n > 10


@pytest.mark.parametrize("scene", ["world", "sky_trees"])
def test_detector_parity_slow_test(scene):
    """The slow-mover test (vision.slow_gain) through the C primitives == through their NumPy references."""
    import tests.test_motion_detector as T
    from tbot.types import Frame
    cam, mount = T.make_cam(), T.make_mount()
    rays = T.make_rays(cam)
    tex = T.make_world() if scene == "world" else T.make_sky_trees(T.make_sky())
    on, off = _pair(["vision.scale=2", "vision.slow_gain=2.5"], None, cam, mount)
    slow = T.centred(cam, mount, 1.5, 8.0, 140.0, 0.0, t_c=0.4)
    fast = T.centred(cam, mount, 30.0, 8.0, 40.0, 5.0, t_c=0.5)
    rng = np.random.default_rng(5)
    n = added = 0
    for k in range(36):
        t = k * T.DT_NS * 1e-9
        R = T.R_of(t, 5.0, 1.0, 0.0)
        img = T.render(tex, R, mount, rays, 1.0 if k % 9 else 1.1)
        for d in (slow, fast):
            u, v, dm, val = d(t, R)
            T.draw_disc(img, u, v, dm, val)
        fr = Frame(image=T.to_u8(img, rng), seq=k, t_exposure_ns=10**9 + k * T.DT_NS, t_capture_ns=0)
        n += _same_frame(on, off, fr, T.R_of(t, 1.0, -0.5, 0.0) @ R, k)
        added += (on.stats.get("slow") or {}).get("px", 0)
    assert n > 10 and added > 0


PRESSES = [Path(__file__).resolve().parents[1] / "pi-pulls" / "pi-2026-11-20" / "sessions" / p
           for p in ("20261120-072712-press", "20261119-160243-press")]


@pytest.mark.parametrize("press", PRESSES, ids=lambda p: p.name)
def test_detector_parity_real_press(press):
    if not press.exists():
        pytest.skip(f"{press.name} not present")
    from tbot.camera.replay import ReplaySource
    cfg, cam, mount, mask, session = M._press_objects(press)
    from tbot.config import load_config  # noqa: F401
    import dataclasses
    v_on = dataclasses.replace(cfg.vision, native=True)
    v_off = dataclasses.replace(cfg.vision, native=False)
    on = M.MotionDetector(dataclasses.replace(cfg, vision=v_on), mask, cam, mount)
    off = M.MotionDetector(dataclasses.replace(cfg, vision=v_off), mask, cam, mount)
    src = ReplaySource(press)
    k = n = 0
    while k < 200:
        fr = src.read()
        if fr is None:
            break
        n += _same_frame(on, off, fr, session.R(fr.t_exposure_ns), k)
        k += 1
    assert k > 50


@pytest.mark.parametrize("trial", range(36))
def test_motion_kernel(trial):
    """tbot_motion_kernel (the fused row pass, or the plane-by-plane one for cells over 256 px) == motion_kernel."""
    rng = np.random.default_rng(3000 + trial)
    h, w = (200, 320) if trial % 4 == 0 else (int(rng.integers(3, 140)), int(rng.integers(3, 230)))
    env = int(rng.choice([1, 2, 3, 3, 4, 5]))
    cell = int(rng.choice([8, 8, 5, 2, 300]))
    cur, ra, rb = (_imgs(rng, h, w, "rand" if trial % 5 else "few") for _ in range(3))
    sv = _mask(rng, h, w, 0.95)
    mask = None if trial % 3 == 0 else _mask(rng, h, w, 0.8)
    nb = (w + 15) // 16
    offs = [rng.integers(-6, 7, (h, nb)).astype(np.int32) for _ in range(4)]
    ga, gb = (int(rng.integers(150, 400)), int(rng.integers(-3000, 3000))), (256, 128)
    prm = (float(rng.uniform(2, 6)), float(rng.uniform(2, 6)), float(rng.uniform(0, 8)), float(rng.uniform(0, 3)))
    bufs = M.KernelBuffers(h, w, cell)
    D, binary, thr, mD, sig = M.motion_kernel(cur, ra, rb, sv, mask, offs[:2], offs[2:], ga, gb, env, cell, *prm, bufs)
    want = [x.copy() for x in (D, binary, thr, bufs.wa, bufs.wb, bufs.va, bufs.vb)]
    kp = _native.MotionKParams()
    ch, cw = -(-h // cell), -(-w // cell)
    out = [np.zeros((h, w), np.uint8) for _ in range(6)] + [np.zeros((ch, cw), np.uint8)]
    cells = [np.zeros((ch, cw), np.int64) for _ in range(4)]
    cmD = np.zeros((ch, cw))
    scratch = np.zeros(6 * h * w + 64, np.uint8)
    kp.h, kp.w, kp.block, kp.env, kp.cell, kp.row_step, kp.cfar_box = h, w, 16, env, cell, M.NOISE_ROW_STEP, M.CFAR_BOX
    kp.cfar_guard = M.CFAR_GUARD
    kp.ga, kp.oa, kp.gb, kp.ob = *ga, *gb
    kp.cur, kp.ref_a, kp.ref_b, kp.src_valid, kp.mask = _p(cur), _p(ra), _p(rb), _p(sv), _p(mask)
    kp.dxa, kp.dya, kp.dxb, kp.dyb = (_p(o) for o in offs)
    kp.D, kp.binary, kp.wa, kp.wb, kp.va, kp.vb, kp.thr = (_p(o) for o in out)
    kp.mD = _p(cmD)
    kp.SD, kp.ND, kp.SA, kp.NA = (_p(c) for c in cells)
    kp.scratch = _p(scratch)
    kp.t_min_dn, kp.noise_k, kp.cfar_gain, kp.texture_gain = prm
    kp.abs_to_sigma, kp.noise_quantile, kp.noise_min_fill = M.ABS_TO_SIGMA, M.NOISE_QUANTILE, M.NOISE_MIN_FILL
    assert K.kernel(ctypes.byref(kp)) == 0
    got = [out[0], out[1], out[6], out[2], out[3], out[4], out[5]]
    for k, (g_, w_) in enumerate(zip(got, want)):
        assert np.array_equal(g_, w_), k
    assert cmD.tobytes() == np.ascontiguousarray(mD).tobytes()
    assert _strict(kp.sigma_n) == _strict(sig)
