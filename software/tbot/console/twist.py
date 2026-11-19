"""Bore-laser twist: find the laser dot in each frame and fit the circle it traces; the centre is the bore pixel.

The method of calib_data/bs20ft_2026-11-18/twist_fit.py (the one that worked on 2026-11-16 and 2026-11-18):
difference-of-Gaussians peak in a window, weighted centroid, then a circle fit that drops points off the ring.
Every threshold is relative to the run itself (no fixed DN numbers): a dot counts when its peak is at least a
quarter of the run's strong dots and no second peak is half as strong.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def find_dot(gray: np.ndarray, window: tuple[float, float, float] | None = None) -> tuple[float, float, float, float] | None:
    """(x, y, snr, snr2) of the brightest compact spot (full-frame pixels), or None.

    snr is the DoG peak over the window's robust noise; snr2 the same for the best peak more than 12 px away
    (an ambiguous frame, e.g. a reflection, has snr2 close to snr).
    """
    h, w = gray.shape[:2]
    if window is not None:
        cx, cy, r = window
        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(w, int(cx + r)), min(h, int(cy + r))
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    dog = cv2.GaussianBlur(roi, (0, 0), 1.5) - cv2.GaussianBlur(roi, (0, 0), 6)
    noise = float(np.median(np.abs(dog))) * 1.48 + 1e-3
    y, x = np.unravel_index(int(np.argmax(dog)), dog.shape)
    snr = float(dog[y, x]) / noise
    d2 = dog.copy()
    d2[max(0, y - 12):y + 12, max(0, x - 12):x + 12] = -1e9
    snr2 = float(d2.max()) / noise
    ya, yb, xa, xb = max(0, y - 7), min(dog.shape[0], y + 8), max(0, x - 7), min(dog.shape[1], x + 8)
    wts = dog[ya:yb, xa:xb].clip(0)
    s = float(wts.sum())
    if s <= 0:
        return None
    yy, xx = np.mgrid[ya:yb, xa:xb]
    return float((wts * xx).sum() / s + x0), float((wts * yy).sum() / s + y0), snr, snr2


def _fit(p: np.ndarray) -> tuple[float, float, float]:
    a = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    c = np.linalg.lstsq(a, b, rcond=None)[0]
    cx, cy = float(c[0]), float(c[1])
    return cx, cy, math.sqrt(max(0.0, float(c[2]) + cx * cx + cy * cy))


def fit_ring(rows: list[tuple[float, float, float, float]]) -> dict:
    """Circle through the dot positions. rows: (x, y, snr, snr2). Raises ValueError when it cannot be trusted."""
    if len(rows) < 12:
        raise ValueError(f"only {len(rows)} frames with a dot (need 12+): twist slower, or check the exposure")
    a = np.asarray(rows, dtype=np.float64)
    strong = np.percentile(a[:, 2], 90)
    good = (a[:, 2] > 0.25 * strong) & (a[:, 3] < 0.5 * a[:, 2])
    p = a[good][:, :2]
    if len(p) < 12:
        raise ValueError(f"only {len(p)} clean dots of {len(a)}: a reflection or a second light is competing; "
                         "narrow the search window (click on the dot) or dim the room")
    keep = np.ones(len(p), bool)
    for _ in range(5):  # drop points off the ring (fingers pushing the laser, reflections)
        cx, cy, r = _fit(p[keep])
        res = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r
        s = 1.4826 * float(np.median(np.abs(res[keep])))
        keep = np.abs(res) < max(3 * s, 2.0)
        if keep.sum() < 8:
            raise ValueError("the dots do not lie on a circle: turn the laser in the bore, not the gun")
    cx, cy, r = _fit(p[keep])
    res = np.hypot(p[keep, 0] - cx, p[keep, 1] - cy) - r
    ang = np.degrees(np.arctan2(p[keep, 1] - cy, p[keep, 0] - cx)) % 360
    sectors = int(len(np.unique((ang // 30).astype(int))))
    rng = np.random.default_rng(0)
    q = p[keep]
    bs = np.array([_fit(q[rng.integers(0, len(q), len(q))])[:2] for _ in range(200)])
    sd = bs.std(axis=0)
    scatter = float(np.std(res))
    # quality: the ring must be covered all round and the centre known to well under a pixel or two
    sd_max = float(max(sd))
    if sectors >= 10 and sd_max <= 1.0:
        grade = "excellent"
    elif sectors >= 8 and sd_max <= 2.0:
        grade = "good"
    elif sectors >= 6 and sd_max <= 4.0:
        grade = "fair"
    else:
        grade = "poor"
    return {
        "cx": cx, "cy": cy, "r": r, "scatter_px": scatter, "sd_px": [float(sd[0]), float(sd[1])],
        "frames": int(len(a)), "clean": int(good.sum()), "on_ring": int(keep.sum()), "sectors": sectors,
        "grade": grade,
        "points": [[round(float(x), 2), round(float(y), 2), bool(k)] for (x, y), k in zip(p, keep)],
    }
