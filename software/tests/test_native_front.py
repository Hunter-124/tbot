"""The detector's native front end (tbot_front_* and tbot_noise_sigma in tbot/vision/_measure.c) gives exactly what
the OpenCV / NumPy code it replaces gives: the same components with the same numbers, the same stands-alone
verdicts, the same box means and the same noise sigma.

Skipped when the shared library is not built (scripts/build_native.sh).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tbot.vision import _native
from tbot.vision import detector as D

pytestmark = pytest.mark.skipif(_native.front_kernel() is None,
                                reason=f"native front end: {_native.status} / {_native.front_status}")


def _opencv_components(b8):
    """ClayDetector._measure's OpenCV labelling: (n, labels, stats, centroids, rectangle of the set pixels)."""
    h, w = b8.shape
    rx, ry, rw, rh = cv2.boundingRect(b8)
    if rw == 0:
        return 1, np.zeros((h, w), np.int32), None, None, (0, 0, 0, 0)
    if rw * rh * 2 > b8.size:
        n, lab, st, ce = cv2.connectedComponentsWithStats(b8, connectivity=8)
    else:
        n, sub, st, ce = cv2.connectedComponentsWithStats(np.ascontiguousarray(b8[ry:ry + rh, rx:rx + rw]),
                                                          connectivity=8)
        ce += (rx, ry)
        lab = np.zeros((h, w), np.int32)
        lab[ry:ry + rh, rx:rx + rw] = sub
        st[:, cv2.CC_STAT_LEFT] += rx
        st[:, cv2.CC_STAT_TOP] += ry
    return n, lab, st, ce, (rx, ry, rw, rh)


def _opencv_alone(b8, st, roi, ring_max):
    rx, ry, rw, rh = roi
    ii = cv2.integral(b8[ry:ry + rh, rx:rx + rw])
    unit = int(ii[-1, -1]) // max(1, int(st[1:, 4].sum()))
    bx, by = st[1:, 0].astype(np.int64) - rx, st[1:, 1].astype(np.int64) - ry
    bw, bh = st[1:, 2].astype(np.int64), st[1:, 3].astype(np.int64)
    m = np.clip(np.maximum(bw, bh), D.CLUTTER_RING_MIN_PX, D.CLUTTER_RING_MAX_PX)
    x0, y0 = np.maximum(bx - m, 0), np.maximum(by - m, 0)
    x1, y1 = np.minimum(bx + bw + m, rw), np.minimum(by + bh + m, rh)
    inside = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) // max(1, unit)
    ring = (x1 - x0) * (y1 - y0) - bw * bh
    return (inside - st[1:, 4]) <= ring_max * np.maximum(ring, 1)


def test_labelling_and_the_stands_alone_test_match_opencv_on_random_images():
    label, alone = _native.front_kernel()[:2]
    rng = np.random.default_rng(1)
    compared = 0
    for trial in range(700):
        h, w = (400, 640) if trial % 40 == 0 else (int(rng.integers(5, 90)), int(rng.integers(9, 130)))
        dens = float(rng.choice([0.002, 0.02, 0.1, 0.3, 0.5, 0.7])) if (h, w) != (400, 640) else 0.01
        b = rng.random((h, w)) < dens
        if trial % 3 == 0:      # set pixels confined to a corner: only their rectangle is labelled
            m = np.zeros((h, w), bool)
            y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
            m[y0:y0 + max(2, h // 3), x0:x0 + max(2, w // 3)] = True
            b &= m
        if trial % 7 == 0:
            b = cv2.dilate(b.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
        b8 = np.ascontiguousarray(b.view(np.uint8) * (255 if trial % 2 else 1))
        fr = _native.Front(h, w)
        for _ in range(2):      # the second call clears and rewrites the label image
            n = label(fr.ref, b8.ctypes.data)
        n2, lab, st, ce, roi = _opencv_components(b8)
        where = f"trial {trial} ({h}x{w}, density {dens})"
        assert n == n2 and tuple(fr.state.roi) == roi, where
        assert np.array_equal(fr.labels, lab), where
        if n == 1:
            continue
        assert np.array_equal(fr.stats[1:n], st[1:]), where
        assert np.array_equal(fr.cents[1:n], ce[1:]), where    # bit for bit
        idx = np.arange(n - 1, dtype=np.intp)
        flags = np.empty(n - 1, np.uint8)
        for ring_max in (0.05, 0.25):
            alone(fr.ref, idx.ctypes.data, n - 1, D.CLUTTER_RING_MIN_PX, D.CLUTTER_RING_MAX_PX, ring_max,
                  flags.ctypes.data)
            assert np.array_equal(flags.view(bool), _opencv_alone(b8, st, roi, ring_max)), where
        compared += 1
    assert compared > 500


def test_labelling_says_so_when_a_frame_is_beyond_its_capacity_and_leaves_nothing_behind():
    label = _native.front_kernel()[0]
    rng = np.random.default_rng(2)
    fr = _native.Front(400, 640)
    sparse = np.ascontiguousarray((rng.random((400, 640)) < 0.01).view(np.uint8))
    assert label(fr.ref, sparse.ctypes.data) > 1 and fr.labels.any()
    dense = np.ascontiguousarray((rng.random((400, 640)) < 0.5).view(np.uint8))   # ~64000 runs
    assert label(fr.ref, dense.ctypes.data) == -1
    assert not fr.labels.any() and fr.state.n_runs == 0
    # and the detector then labels with OpenCV: the same detections as with the kernel off
    assert label(fr.ref, sparse.ctypes.data) == _opencv_components(sparse)[0]


def test_box_mean_and_cell_statistics_match_this_opencv():
    box, cells = _native.front_kernel()[3:5]
    rng = np.random.default_rng(3)
    done = 0
    for trial in range(300):
        s = int(rng.integers(2, 9))
        rounding = D._area_rounding(s)
        if rounding is None:     # this OpenCV rounds some other way: the detector keeps cv2.resize
            continue
        h, w = s * int(rng.integers(3, 60)), s * int(rng.integers(3, 90))
        src = (rng.integers(0, 256, (h, w), dtype=np.uint8) if trial % 2
               else (rng.integers(0, 2, (h, w), dtype=np.uint8) * int(rng.integers(1, 256))).astype(np.uint8))
        got = np.empty((h // s, w // s), np.uint8)
        assert box(src.ctypes.data, h, w, s, rounding, got.ctypes.data) == 0
        assert np.array_equal(got, cv2.resize(src, (w // s, h // s), interpolation=cv2.INTER_AREA)), (trial, s)
        cell = 2 * s
        if h % cell == 0 and w % cell == 0:
            a8 = cv2.resize(src, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            a8 = cv2.convertScaleAbs(a8, alpha=float(D.LOCAL_GAIN))
            ref = cv2.resize(a8, (w // cell, h // cell), interpolation=cv2.INTER_AREA)
            got = np.empty_like(ref)
            assert cells(src.ctypes.data, h, w, cell, int(D.LOCAL_GAIN), rounding, got.ctypes.data) == 0
            assert np.array_equal(got, ref), (trial, cell)
        done += 1
    assert done > 100
    assert box(src.ctypes.data, 10, 10, 4, 0, got.ctypes.data) == -1     # not a multiple of the scale


def test_noise_sigma_is_the_python_estimate_to_the_last_bit():
    noise = _native.front_kernel()[2]
    rng = np.random.default_rng(4)
    tiled = 0
    for trial in range(120):
        h, w = (400, 640) if trial % 2 else (int(rng.integers(40, 300)), int(rng.integers(60, 400)))
        sd = float(rng.choice([0.5, 1.5, 3, 8, 30]))
        bg = np.full((h, w), 120, np.uint8)
        gray = np.clip(120 + rng.normal(0, sd, (h, w)) + (rng.random((h, w)) < 0.01) * 80, 0, 255).astype(np.uint8)
        mask = None if trial % 3 else (rng.random((h, w)) < float(rng.choice([0.1, 0.6, 0.95])))
        step = max(1, int(round((h * w / D.NOISE_TARGET_SAMPLES) ** 0.5)))
        for lazy in (True, False):
            c = D._LazyContrast(gray, bg) if lazy else cv2.subtract(gray, bg, dtype=cv2.CV_16S)
            ref = D._tiled_sigma(c[::step, ::step], mask[::step, ::step] if mask is not None else None)
            arrays = (gray.ctypes.data, bg.ctypes.data, None) if lazy else (None, None, c.ctypes.data)
            q = noise(*arrays, mask.ctypes.data if mask is not None else None, h, w, step, *D.NOISE_TILES,
                      D.DEV_MAX, D.NOISE_TILE_MIN_SAMPLES, D.NOISE_QUANTILE, D.NOISE_TILE_QUANTILE,
                      D.NOISE_SIGMA_FLOOR)
            if ref is None:
                assert q != q, trial       # NaN: the detector then runs the Python estimate
            else:
                assert q == ref, (trial, lazy, q, ref)
                tiled += 1
    assert tiled > 100
