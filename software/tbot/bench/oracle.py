"""Offline, non-causal "oracle" that labels moving objects in recorded presses (ground truth for the bench).

Pipeline (per press):
  1. Camera rotation between frames: gyro prior, then LK-tracked corners and a RANSAC rotation-only fit
     (K R K^-1), chained into a per-frame orientation O_k (camera -> press world frame).
  2. Every frame k is compared with registered past AND future frames at several lags (~35/90/180 ms):
     global rotation warp + a heavily smoothed DIS optical-flow correction (parallax, sub-pixel error),
     photometric gain/offset fitted from the image pair itself (auto-exposure), then an envelope
     difference (cur - dilate(ref), erode(ref) - cur).  min(past, future) per lag kills ghosts,
     max over lags keeps slow movers.
  3. Blobs of that difference -> candidates -> world directions (gnomonic plane of the press's mean view).
  4. Candidates that recur at the same world spot >=150 ms apart are static clutter and are dropped.
  5. Batch linking: constant-velocity triplet seeds in world coordinates, greedy extension both ways
     with gaps, then only long, smooth, consistently moving chains are kept.
  6. Each kept chain is re-measured on every frame of its span against a median background built from
     refs in which the object is elsewhere: centroid and extent in full-res pixels.

usage: python -m tbot.bench.oracle PRESS_DIR... [--out-dir bench_data/truth] [--sheets DIR] [--jobs N] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

W, H = 640, 400
LAGS_MS = (35.0, 90.0, 180.0)
MIN_SCENE_REL_PX = 3.0   # object must move >= this vs. the scenery around it in ~80 ms


# ----------------------------------------------------------------------------------------- data
class Press:
    def __init__(self, d: str):
        self.dir = d.rstrip("/\\")
        self.name = os.path.basename(self.dir)
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        n = int(self.meta["frames"])
        fb = os.path.join(d, "frames.bin")
        if not self.meta.get("complete") or os.path.getsize(fb) != n * H * W:
            raise ValueError("incomplete press %s" % d)
        self.fr = np.fromfile(fb, dtype=np.uint8).reshape(n, H, W)
        self.N = n
        fc = np.genfromtxt(os.path.join(d, "frames.csv"), delimiter=",", names=True)
        self.t = fc["t_exposure_ns"].astype(np.int64)[:n]
        if len(self.t) < n:
            raise ValueError("frames.csv short")
        K = np.array(self.meta["calibration"]["intrinsics"]["K"], float)
        self.K = K
        self.Ki = np.linalg.inv(K)
        self.fx = K[0, 0]
        imu = np.genfromtxt(os.path.join(d, "imu.csv"), delimiter=",", names=True)
        self.it = imu["t_ns"].astype(np.int64)
        g = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
        # gun (+X muzzle, +Y left, +Z up) -> camera (x right, y down, z forward)
        self.wc = np.stack([-g[:, 1], -g[:, 2], g[:, 0]], 1)
        self.events = []
        p = os.path.join(d, "events.jsonl")
        if os.path.exists(p):
            self.events = [json.loads(l) for l in open(p) if l.strip()]

    def gyro_rot(self, t0, t1):
        """Rotation taking camera rays at t0 to rays of the same static point at t1 (gyro only)."""
        if t1 == t0 or len(self.it) < 2:
            return np.eye(3)
        sgn = 1.0
        if t1 < t0:
            t0, t1, sgn = t1, t0, -1.0
        ts = np.clip(np.linspace(t0, t1, max(3, int((t1 - t0) / 1e6) + 1)), self.it[0], self.it[-1])
        w = np.stack([np.interp(ts, self.it, self.wc[:, a]) for a in range(3)], 1)
        th = (0.5 * (w[1:] + w[:-1]) * (np.diff(ts) * 1e-9)[:, None]).sum(0) * sgn
        R, _ = cv2.Rodrigues(-th.reshape(3, 1))
        return R


# ----------------------------------------------------------------------------------------- geometry
def _rays(Ki, pts):
    h = np.c_[pts, np.ones(len(pts))] @ Ki.T
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def _proj(K, r):
    x = r @ K.T
    return x[..., :2] / x[..., 2:3]


def _kabsch(A, B, w=None):
    """R with B ~ R A (rows are unit rays)."""
    if w is None:
        w = np.ones(len(A))
    Hm = (A * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


_rng = np.random.default_rng(1)


def _ransac_rot(K, a, b, pa, pb, thr=1.2, iters=128):
    n = len(a)
    i1 = _rng.integers(0, n, iters)
    i2 = _rng.integers(0, n, iters)
    ok = i1 != i2
    i1, i2 = i1[ok], i2[ok]
    Hm = a[i1][:, :, None] * b[i1][:, None, :] + a[i2][:, :, None] * b[i2][:, None, :]
    U, S, Vt = np.linalg.svd(Hm)
    V = np.transpose(Vt, (0, 2, 1))
    Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))
    D = np.zeros((len(d), 3, 3))
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 2, 2] = d
    Rs = V @ D @ Ut
    pr = _proj(K, np.einsum("mij,nj->mni", Rs, a))
    err = np.linalg.norm(pr - pb[None], axis=2)
    cnt = (err < thr).sum(1)
    best = int(np.argmax(cnt))
    inl = err[best] < thr
    R = Rs[best]
    for _ in range(3):
        if inl.sum() < 3:
            break
        R = _kabsch(a[inl], b[inl])
        e = np.linalg.norm(_proj(K, a @ R.T) - pb, axis=1)
        inl = e < thr
    e = np.linalg.norm(_proj(K, a @ R.T) - pb, axis=1)
    return R, inl, e


def _norm_gain(cur, ref):
    m1 = float(np.median(cur))
    m0 = float(np.median(ref))
    return 1.0 if m0 < 2 else max(0.2, min(5.0, m1 / m0))


class Registrar:
    """Pairwise rotation (and local residual flow) between frames of one press, cached."""

    def __init__(self, P: Press):
        self.P = P
        self.rot = {}
        self.feat = {}
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.O = None
        self.quality = {}

    def _features(self, j):
        if j not in self.feat:
            img = self.P.fr[j]
            p = cv2.goodFeaturesToTrack(img, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7)
            self.feat[j] = np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2)
        return self.feat[j]

    def rel(self, j, i, prior=None, fast=False):
        """R taking rays of static points in frame j to frame i."""
        key = (j, i)
        if key in self.rot:
            return self.rot[key]
        if (i, j) in self.rot:
            return self.rot[(i, j)].T
        P = self.P
        if prior is None:
            prior = P.gyro_rot(P.t[j], P.t[i])
        ref = P.fr[j]
        cur = P.fr[i]
        g = _norm_gain(cur, ref)
        refn = cv2.convertScaleAbs(ref, alpha=g)
        pj = self._features(j)
        R = prior
        nin = 0
        rms = -1.0
        if len(pj) >= 12:
            pi0 = _proj(P.K, _rays(P.Ki, pj) @ prior.T).astype(np.float32)
            inside = (pi0[:, 0] > -50) & (pi0[:, 0] < W + 50) & (pi0[:, 1] > -50) & (pi0[:, 1] < H + 50)
            pjj, pi0 = pj[inside], pi0[inside]
            if len(pjj) >= 12:
                if fast and len(pjj) > 250:
                    sel = np.linspace(0, len(pjj) - 1, 250).astype(int)
                    pjj, pi0 = pjj[sel], pi0[sel]
                lk = dict(winSize=(21, 21), maxLevel=2 if fast else 4,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
                pi, st, _ = cv2.calcOpticalFlowPyrLK(refn, cur, pjj.reshape(-1, 1, 2), pi0.reshape(-1, 1, 2).copy(),
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, refn, pi, pjj.reshape(-1, 1, 2).copy(),
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                pi = pi.reshape(-1, 2)
                fb = np.linalg.norm(pb.reshape(-1, 2) - pjj, axis=1)
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.6)
                ok &= (pi[:, 0] >= 2) & (pi[:, 0] < W - 2) & (pi[:, 1] >= 2) & (pi[:, 1] < H - 2)
                if ok.sum() >= 10:
                    a = _rays(P.Ki, pjj[ok])
                    b = _rays(P.Ki, pi[ok])
                    Rr, inl, e = _ransac_rot(P.K, a, b, pjj[ok], pi[ok])
                    if inl.sum() < max(8, 0.25 * ok.sum()):
                        Rr, inl, e = _ransac_rot(P.K, a, b, pjj[ok], pi[ok], thr=2.5)
                    if inl.sum() >= max(8, 0.2 * ok.sum()):
                        R = Rr
                        nin = int(inl.sum())
                        rms = float(np.sqrt(np.mean(e[inl] ** 2)))
        if nin == 0:
            R = self._phase_fallback(j, i, prior, g)
        self.quality[key] = (nin, rms)
        self.rot[key] = R
        return R

    def _phase_fallback(self, j, i, prior, g):
        P = self.P
        Hm = P.K @ prior @ P.Ki
        s = 2
        ref = cv2.warpPerspective(P.fr[j], Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a = cv2.resize(P.fr[i], (W // s, H // s), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(ref, (W // s, H // s), interpolation=cv2.INTER_AREA).astype(np.float32) * g
        han = cv2.createHanningWindow((W // s, H // s), cv2.CV_32F)
        (sx, sy), resp = cv2.phaseCorrelate(b - b.mean(), a - a.mean(), han)
        if resp < 0.08 or abs(sx) * s > 40 or abs(sy) * s > 40:
            return prior
        du, dv = sx * s, sy * s
        Rc, _ = cv2.Rodrigues(np.array([-dv / P.fx, du / P.fx, 0.0]).reshape(3, 1))
        return Rc @ prior

    def orient(self):
        """Per-frame orientation O_k (camera k -> world of frame 0), chained from consecutive pairs."""
        if self.O is not None:
            return self.O
        P = self.P
        O = [np.eye(3)]
        for k in range(1, P.N):
            R = self.rel(k - 1, k)
            O.append(O[-1] @ R.T)
        self.O = np.array(O)
        return self.O

    def pair_rot(self, j, i):
        O = self.orient()
        prior = O[i].T @ O[j]
        if abs(i - j) == 1:
            return self.rel(j, i)
        return self.rel(j, i, prior, fast=True)

    def warp(self, j, i, local=True):
        """Reference frame j warped into frame i (float32, photometrically matched), valid mask."""
        P = self.P
        R = self.pair_rot(j, i)
        Hm = P.K @ R @ P.Ki
        ref = P.fr[j]
        cur = P.fr[i]
        wr = cv2.warpPerspective(ref, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.full((H, W), 255, np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST)
        a, b = _photo_fit(cur, wr, valid)
        wrf = wr.astype(np.float32) * a + b
        if local:
            wr8 = np.clip(wrf, 0, 255).astype(np.uint8)
            c2 = cv2.resize(cur, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            r2 = cv2.resize(wr8, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            fl = self.dis.calc(c2, r2, None) * 2.0
            fl = _smooth_flow(fl, c2)
            mapx = _MX + fl[..., 0]
            mapy = _MY + fl[..., 1]
            wrf = cv2.remap(wrf, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            valid = cv2.remap(valid, mapx, mapy, interpolation=cv2.INTER_NEAREST, borderValue=0)
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
        return wrf, valid > 0


def _compose_maps(Hm, mapx, mapy):
    """Source coords in the reference for destination pixel x: H^-1 applied to (x + flow)."""
    Hi = np.linalg.inv(Hm)
    X = Hi[0, 0] * mapx + Hi[0, 1] * mapy + Hi[0, 2]
    Y = Hi[1, 0] * mapx + Hi[1, 1] * mapy + Hi[1, 2]
    Z = Hi[2, 0] * mapx + Hi[2, 1] * mapy + Hi[2, 2]
    return (X / Z).astype(np.float32), (Y / Z).astype(np.float32)


def _photo_fit(cur, wr, valid):
    """Robust cur ~ a*ref + b over valid unsaturated pixels."""
    c = cur[::5, ::5].astype(np.float32).ravel()
    r = wr[::5, ::5].astype(np.float32).ravel()
    m = (valid[::5, ::5].ravel() > 0) & (c < 250) & (r < 250) & (c > 2) & (r > 2)
    if m.sum() < 500:
        return 1.0, 0.0
    c, r = c[m], r[m]
    a, b = 1.0, 0.0
    for it in range(4):
        res = c - (a * r + b)
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1.0
        k = np.abs(res - np.median(res)) < 2.5 * s if it else np.ones(len(c), bool)
        if k.sum() < 200:
            break
        A = np.stack([r[k], np.ones(k.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, c[k], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if not (0.2 < a < 5):
            return _norm_gain(cur, wr), 0.0
    return a, b


_MX, _MY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def _smooth_flow(fl, cur):
    """fl, cur at half resolution; returns full-res flow, median over ~40 px blocks, zero where flat."""
    s = 4
    h2, w2 = cur.shape
    small = cv2.resize(fl, (w2 // s, h2 // s), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3)
    tex = cv2.resize(np.abs(gx) + np.abs(gy), (w2 // s, h2 // s), interpolation=cv2.INTER_AREA)
    small[tex < 12] = 0
    small = np.ascontiguousarray(small)
    for c in range(2):
        small[..., c] = cv2.medianBlur(np.ascontiguousarray(small[..., c]), 5)
    small = np.clip(small, -8, 8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------------------- difference
def ref_frames(t, k, lag_ms):
    L = int(lag_ms * 1e6)
    prev = np.nonzero(t[:k] <= t[k] - L)[0]
    nxt = np.nonzero(t[k + 1:] >= t[k] + L)[0]
    p = int(prev[-1]) if len(prev) else None
    n = int(k + 1 + nxt[0]) if len(nxt) else None
    return p, n


_K3 = np.ones((3, 3), np.uint8)


def frame_diff(reg: Registrar, k):
    """Positive / negative envelope differences of frame k (float32) and noise level."""
    P = reg.P
    cur = P.fr[k].astype(np.float32)
    pos = np.zeros((H, W), np.float32)
    neg = np.zeros((H, W), np.float32)
    vall = np.zeros((H, W), bool)
    used = []
    for L in LAGS_MS:
        p, n = ref_frames(P.t, k, L)
        pair = [x for x in (p, n) if x is not None]
        if len(pair) < 2:
            p2, n2 = ref_frames(P.t, k, 2 * L)
            extra = p2 if p is not None else n2
            if extra is not None and extra not in pair:
                pair.append(extra)
        if len(pair) < 2:
            continue
        dp = None
        for j in pair[:2]:
            wr, val = reg.warp(j, k)
            hi = cv2.dilate(wr, _K3)
            lo = cv2.erode(wr, _K3)
            a = cur - hi
            b = lo - cur
            a[~val] = 0
            b[~val] = 0
            if dp is None:
                dp, dn, vv = a, b, val
            else:
                dp = np.minimum(dp, a)
                dn = np.minimum(dn, b)
                vv = vv & val
        used.append(tuple(pair[:2]))
        pos = np.maximum(pos, dp)
        neg = np.maximum(neg, dn)
        vall |= vv
    pos[~vall] = 0
    neg[~vall] = 0
    return pos, neg, used


def extract_cands(pos, neg, cur, thr_min=8.0, max_n=300):
    D = np.maximum(pos, neg)
    v = D[D > 0]
    noise = 1.4826 * float(np.median(v)) if len(v) > 1000 else 2.0
    thr = max(thr_min, 3.0 * noise)
    mask = (D >= thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros((0, 9), np.float32), thr
    idx = lab.ravel()
    Dw = D.ravel()
    sw = np.bincount(idx, weights=Dw, minlength=n)
    yy, xx = np.mgrid[0:H, 0:W]
    su = np.bincount(idx, weights=Dw * xx.ravel(), minlength=n)
    sv = np.bincount(idx, weights=Dw * yy.ravel(), minlength=n)
    sp = np.bincount(idx, weights=(pos.ravel() > neg.ravel()) * Dw, minlength=n)
    order = np.argsort(idx, kind="stable")
    starts = np.searchsorted(idx[order], np.arange(n))
    pk = np.maximum.reduceat(Dw[order], starts)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area > 6000:
            continue
        if area < 2 and pk[i] < 2 * thr:
            continue
        pol = 1.0 if sp[i] > 0.5 * sw[i] else -1.0
        out.append((su[i] / sw[i], sv[i] / sw[i], area, pk[i], w, h, pol, sw[i], 0))
    out = np.array(out, np.float32).reshape(-1, 9)
    if len(out) > max_n:
        out = out[np.argsort(-out[:, 7])[:max_n]]
    return out, thr


# ----------------------------------------------------------------------------------------- linking
class World:
    """Gnomonic plane (px units) of the press's mean viewing direction."""

    def __init__(self, P: Press, O):
        z = O[:, :, 2].mean(0)
        z /= np.linalg.norm(z)
        x = np.cross([0, 1.0, 0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.B = np.stack([x, y, z])  # rows: world -> plane frame
        self.P, self.O = P, O

    def to_plane(self, k, uv):
        r = _rays(self.P.Ki, np.asarray(uv, float).reshape(-1, 2)) @ self.O[k].T @ self.B.T
        z = np.maximum(r[:, 2], 0.2)
        return self.P.fx * r[:, :2] / z[:, None], r[:, 2]

    def to_image(self, k, XY):
        XY = np.asarray(XY, float).reshape(-1, 2)
        r = np.c_[XY / self.P.fx, np.ones(len(XY))] @ self.B @ self.O[k]
        return _proj(self.P.K, r)


def _grid_neighbors(Pts, cell):
    keys = np.floor(Pts / cell).astype(np.int64)
    d = {}
    for i, (a, b) in enumerate(keys):
        d.setdefault((a, b), []).append(i)
    return keys, d


def registration_sigma(reg):
    """Per-frame registration uncertainty (px) of frame k vs k-1: LK fit rms when the feature fit worked,
    otherwise a fraction of the inter-frame rotation (gyro / phase-correlation fallback)."""
    P = reg.P
    sig = np.zeros(P.N)
    for k in range(1, P.N):
        nin, rms = reg.quality.get((k - 1, k), (0, -1.0))
        R = reg.rot.get((k - 1, k), np.eye(3))
        ang_px = P.fx * float(np.linalg.norm(cv2.Rodrigues(R)[0]))
        if nin >= 30 and rms >= 0:
            sig[k] = 0.2 + rms
        elif nin > 0:
            sig[k] = 0.5 + rms + 0.01 * ang_px
        else:
            sig[k] = 1.0 + 0.04 * ang_px
    return sig


def static_filter(C, t, r=3.0, min_dt_ms=150.0, max_dt_ms=1200.0, need=2):
    """C: per-frame arrays (plane X, Y, ok, area, ...). A candidate is static clutter when candidates of similar
    size sit at the same world spot in >= `need` other frames that are >= min_dt_ms away."""
    allp, fid, area = [], [], []
    for f, c in enumerate(C):
        allp.append(c[:, :2])
        area.append(c[:, 3])
        fid += [f] * len(c)
    if not fid:
        return [np.zeros(0, bool) for _ in C]
    allp = np.concatenate(allp)
    area = np.maximum(np.concatenate(area), 2.0)
    fid = np.array(fid)
    tt = t[fid] / 1e6
    keys, grid = _grid_neighbors(allp, r)
    stat = np.zeros(len(allp), bool)
    for i in range(len(allp)):
        a, b = keys[i]
        cand = []
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                cand += grid.get((a + da, b + db), [])
        if len(cand) <= need:
            continue
        cand = np.array(cand)
        dt = np.abs(tt[cand] - tt[i])
        rr = max(r, 0.25 * np.sqrt(area[i]))
        m = (dt >= min_dt_ms) & (dt <= max_dt_ms)
        m &= np.hypot(allp[cand, 0] - allp[i, 0], allp[cand, 1] - allp[i, 1]) <= rr
        m &= np.abs(np.log(area[cand] / area[i])) < np.log(4.0)
        stat[i] = len(set(fid[cand[m]].tolist())) >= need
    out, o = [], 0
    for c in C:
        out.append(stat[o:o + len(c)])
        o += len(c)
    return out


_WCACHE = {n: np.linspace(0.4, 1.0, n) for n in range(1, 9)}


def _predict(chain, tq):
    """chain: list of (t_s, X, Y). Weighted LS of the last points: constant velocity, or constant
    acceleration once there are >= 6 points over >= 120 ms. Returns prediction and speed (px/s)."""
    pts = np.asarray(chain[-8:], float)
    n = len(pts)
    if n == 1:
        return pts[0, 1:], 0.0
    ts = pts[:, 0] - pts[-1, 0]
    w = _WCACHE[n]
    dtq = tq - pts[-1, 0]
    if n >= 6 and -ts[0] >= 0.12:
        A = np.stack([np.ones(n), ts, ts * ts], 1)
        Aw = A * w[:, None]
        sol = np.linalg.solve(Aw.T @ A, Aw.T @ pts[:, 1:])
        return sol[0] + sol[1] * dtq + sol[2] * dtq * dtq, float(np.hypot(*sol[1]))
    sw = w.sum()
    tm = (w * ts).sum() / sw
    xm = (w[:, None] * pts[:, 1:]).sum(0) / sw
    vt = (w * (ts - tm) ** 2).sum()
    if vt <= 1e-12:
        return xm, 0.0
    vel = (w[:, None] * (ts - tm)[:, None] * (pts[:, 1:] - xm)).sum(0) / vt
    return xm + vel * (dtq - tm), float(np.hypot(*vel))


# candidate columns in C: 0 X, 1 Y, 2 ok, 3 area, 4 peak, 5 w, 6 h, 7 polarity, 8 sumD
def link(C, t, vmax_px_s=5300.0, max_gap_s=0.2, max_gap_f=8, min_len=6, regsig=None):
    """Greedy strongest-first tracker over the whole press in world-plane coordinates.
    regsig[k]: registration uncertainty (px) of frame k relative to k-1; gates grow with it."""
    N = len(C)
    S = np.cumsum(np.r_[0.0, regsig[1:]]) if regsig is not None else np.zeros(N)
    ts = t.astype(np.float64) / 1e9
    claimed = [np.zeros(len(c), bool) for c in C]
    chains = []

    def extend(chain, direction):
        f = chain[-1][0] if direction > 0 else chain[0][0]
        miss = 0
        while True:
            f += direction
            if f < 0 or f >= N:
                break
            seq = chain if direction > 0 else chain[::-1]
            sgn = 1.0 if direction > 0 else -1.0
            pts = [(sgn * ts[a], C[a][b, 0], C[a][b, 1]) for a, b in seq[-8:]]
            tq = sgn * ts[f]
            dt = tq - pts[-1][0]
            if dt > max_gap_s or miss > max_gap_f:
                break
            if len(seq) < 5 and (miss > 2 or dt > 0.1):
                break
            pred, sp = _predict(pts, tq)
            gate = 2.5 + 0.08 * sp * dt + (3.0 if len(seq) < 4 else 1.0) * dt / 0.03 + 2.0 * abs(S[f] - S[seq[-1][0]])
            c = C[f]
            if len(c):
                d = np.hypot(c[:, 0] - pred[0], c[:, 1] - pred[1])
                bad = (c[:, 2] < 0.5) | claimed[f] | (d > gate)
                if not bad.all():
                    aref = np.median([max(C[a][b, 3], 2.0) for a, b in seq[-4:]])
                    lr = np.abs(np.log(np.maximum(c[:, 3], 2.0) / aref))
                    bad |= (lr > np.log(5.0)) & (aref >= 8)
                    cost = d / gate + 0.4 * lr
                    cost[bad] = 1e9
                    j = int(np.argmin(cost))
                    if cost[j] < 1e8:
                        if direction > 0:
                            chain.append((f, j))
                        else:
                            chain.insert(0, (f, j))
                        miss = 0
                        continue
            miss += 1
        return chain

    seeds = []
    for f1 in range(N):
        c1 = C[f1]
        ok1 = np.nonzero(c1[:, 2] > 0.5)[0] if len(c1) else []
        if not len(ok1):
            continue
        for f2 in range(f1 + 1, min(N, f1 + 4)):
            dt12 = ts[f2] - ts[f1]
            if dt12 > 0.12:
                break
            c2 = C[f2]
            ok2 = np.nonzero(c2[:, 2] > 0.5)[0] if len(c2) else []
            if not len(ok2):
                continue
            d12 = c2[ok2, None, :2] - c1[None, ok1, :2]
            dist = np.hypot(d12[..., 0], d12[..., 1])
            ar = np.abs(np.log(np.maximum(c2[ok2, None, 3], 2) / np.maximum(c1[None, ok1, 3], 2)))
            pb, pa = np.nonzero((dist <= vmax_px_s * dt12 + 3) & (dist >= 0.3) & (ar < np.log(4.0)))
            if not len(pa):
                continue
            A = c1[ok1[pa], :2]
            B = c2[ok2[pb], :2]
            for f3 in range(f2 + 1, min(N, f2 + 4)):
                dt23 = ts[f3] - ts[f2]
                if dt23 > 0.12:
                    break
                c3 = C[f3]
                ok3 = np.nonzero(c3[:, 2] > 0.5)[0] if len(c3) else []
                if not len(ok3):
                    continue
                pred = B + (B - A) * (dt23 / dt12)
                step = np.hypot(*(pred - B).T)
                gate = 2.0 + 0.15 * step + 1.5 * dt23 / dt12 + 2.0 * (S[f3] - S[f1])
                dd = np.hypot(c3[None, ok3, 0] - pred[:, None, 0], c3[None, ok3, 1] - pred[:, None, 1])
                jj = np.argmin(dd, 1)
                good = dd[np.arange(len(pred)), jj] <= gate
                if not good.any():
                    continue
                # require a 4th consistent point (vectorised) before the expensive greedy extension
                Cc = c3[ok3[jj], :2]
                has4 = np.zeros(len(good), bool)
                gi = np.nonzero(good)[0]
                for f4 in range(f3 + 1, min(N, f3 + 4)):
                    dt34 = ts[f4] - ts[f3]
                    if dt34 > 0.12:
                        break
                    c4 = C[f4]
                    ok4 = np.nonzero(c4[:, 2] > 0.5)[0] if len(c4) else []
                    if not len(ok4):
                        continue
                    v = (Cc[gi] - A[gi]) / (dt12 + dt23)
                    p4 = Cc[gi] + v * dt34
                    g4 = 2.5 + 0.15 * np.hypot(*(v * dt34).T) + 1.5 * dt34 / (dt12 + dt23)
                    d4 = np.hypot(c4[None, ok4, 0] - p4[:, None, 0], c4[None, ok4, 1] - p4[:, None, 1]).min(1)
                    has4[gi[d4 <= g4]] = True
                good &= has4
                # the seed must actually move (static jitter that escaped the static filter)
                good &= np.hypot(*(Cc - A).T) >= np.maximum(2.5, 25.0 * (dt12 + dt23))
                for s_ in np.nonzero(good)[0]:
                    tri = ((f1, int(ok1[pa[s_]])), (f2, int(ok2[pb[s_]])), (f3, int(ok3[jj[s_]])))
                    score = min(C[f][j, 8] for f, j in tri)
                    seeds.append((score, tri))
    seeds.sort(key=lambda x: -x[0])
    for score, tri in seeds:
        if any(claimed[f][j] for f, j in tri):
            continue
        ch = extend(extend(list(tri), 1), -1)
        strong = len(ch) >= 4 and all(C[f][j, 4] >= 60 and C[f][j, 3] >= 15 for f, j in ch)
        if len(ch) >= min_len or strong:
            chains.append(ch)
            for f, j in ch:
                claimed[f][j] = True
    return chains


def chain_stats(C, t, ch):
    ts = np.array([t[f] for f, _ in ch], float) / 1e9
    XY = np.array([C[f][j, :2] for f, j in ch])
    n = len(ch)
    travel = float(np.max(np.hypot(*(XY - XY[0]).T)))
    res = []
    for i in range(n):
        lo, hi = max(0, i - 3), min(n, i + 4)
        if hi - lo < 5:
            continue
        tt = ts[lo:hi] - ts[i]
        A = np.stack([tt ** p for p in range(3)], 1)
        sol, *_ = np.linalg.lstsq(A, XY[lo:hi], rcond=None)
        res.append(float(np.hypot(*(XY[i] - sol[0]))))
    rr = float(np.median(res)) if res else 99.0
    dur = ts[-1] - ts[0]
    speed = travel / max(dur, 1e-3)
    area = np.array([C[f][j, 3] for f, j in ch])
    peak = np.array([C[f][j, 4] for f, j in ch])
    return dict(n=n, travel=travel, res=rr, dur=float(dur), speed=float(speed),
                area=float(np.median(area)), peak=float(np.median(peak)))


def select_chains(C, t, chains, min_n=6, min_travel=10.0, max_res=1.5, regsig=None):
    keep = []
    for ch in chains:
        s = chain_stats(C, t, ch)
        weak = s["area"] < 8 and s["peak"] < 45
        # clutter density: other live candidates within 25 px of the chain points
        dens = []
        for f, j in ch:
            c = C[f]
            d = np.hypot(c[:, 0] - c[j, 0], c[:, 1] - c[j, 1])
            dens.append(int(((d < 25) & (c[:, 2] > 0.5)).sum()) - 1)
        s["clutter"] = float(np.mean(dens))
        need_n = min_n if not (s["area"] >= 15 and s["peak"] >= 60) else 4
        mr = max_res + 0.0015 * s["speed"]
        if regsig is not None:
            s["regsig"] = float(np.median([regsig[f] for f, _ in ch]))
            mr += 1.5 * s["regsig"]
        if weak:
            need_n = min_n + 4
            mr = min(mr, 1.0)
            if s["travel"] < 25:
                continue
        strong = s["area"] >= 15 and s["peak"] >= 60
        if s["clutter"] > 1.0 and not strong:
            need_n = max(need_n, min_n + 2 + int(4 * s["clutter"]))
        if s["n"] >= need_n and s["travel"] >= min_travel and s["res"] <= mr:
            keep.append((ch, s))
    return keep


def scene_relative(reg, ch, raw, n_pairs=10, lag_s=0.08):
    """Median displacement (px) of the object relative to the scene around it, over pairs of chain points
    ~lag_s apart. Scene motion = global rotation + median LK flow of textured points in an annulus around
    the object. Clutter stuck to the scenery (parallax, lens distortion, branches) gives ~0-2 px."""
    P = reg.P
    ts = np.array([P.t[f] for f, _ in ch], float) / 1e9
    pairs = []
    for i in range(len(ch)):
        j = int(np.searchsorted(ts, ts[i] + lag_s))
        if j < len(ch) and ts[j] - ts[i] < 2.5 * lag_s:
            pairs.append((i, j))
    if not pairs:
        pairs = [(0, len(ch) - 1)]
    pairs = [pairs[int(x)] for x in np.unique(np.linspace(0, len(pairs) - 1, n_pairs).round())]
    rel = []
    lk = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for i, j in pairs:
        fa, ia = ch[i]
        fb, ib = ch[j]
        pa = raw[fa][ia, :2].astype(float)
        pb = raw[fb][ib, :2].astype(float)
        sz = max(raw[fa][ia, 4], raw[fa][ia, 5])
        R = reg.pair_rot(fa, fb)
        q = _proj(P.K, _rays(P.Ki, pa[None]) @ R.T)[0]
        r_in, r_out = max(8.0, 1.0 * sz + 5), max(40.0, 2 * sz + 20)
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (int(pa[0]), int(pa[1])), int(r_out), 255, -1)
        cv2.circle(mask, (int(pa[0]), int(pa[1])), int(r_in), 0, -1)
        img_a, img_b = P.fr[fa], P.fr[fb]
        g = _norm_gain(img_b, img_a)
        img_a = cv2.convertScaleAbs(img_a, alpha=g)
        pts = cv2.goodFeaturesToTrack(img_a, 60, 0.02, 4, mask=mask, blockSize=5)
        delta = np.zeros(2)
        if pts is not None and len(pts) >= 5:
            pts = pts.reshape(-1, 2)
            pr = _proj(P.K, _rays(P.Ki, pts) @ R.T).astype(np.float32)
            nb, st, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, pts.reshape(-1, 1, 2), pr.reshape(-1, 1, 2).copy(),
                                                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            okk = st.ravel() == 1
            if okk.sum() >= 5:
                dl = nb.reshape(-1, 2)[okk] - pr[okk]
                delta = np.median(dl, 0)
                if np.linalg.norm(delta) > 25:
                    delta = np.zeros(2)
        rel.append(float(np.linalg.norm(pb - (q + delta))))
    return float(np.median(rel)) if rel else 0.0


# ----------------------------------------------------------------------------------------- measurement
def _merge_chains(keep, C, t, world):
    """Join kept chains that are the same object split by a gap (end of one predicts start of other)."""
    keep = sorted(keep, key=lambda x: x[0][0][0])
    merged = True
    while merged:
        merged = False
        for a in range(len(keep)):
            for b in range(len(keep)):
                if a == b:
                    continue
                A, B = keep[a][0], keep[b][0]
                if A[-1][0] >= B[0][0]:
                    continue
                ta, tb = t[A[-1][0]] / 1e9, t[B[0][0]] / 1e9
                if tb - ta > 0.4:
                    continue
                pts = [(t[f] / 1e9, C[f][j, 0], C[f][j, 1]) for f, j in A]
                pred, sp = _predict(pts, tb)
                d = np.hypot(*(C[B[0][0]][B[0][1], :2] - pred))
                if d < 4 + 0.15 * sp * (tb - ta) + 30 * (tb - ta):
                    keep[a] = (A + B, keep[a][1])
                    del keep[b]
                    merged = True
                    break
            if merged:
                break
    return keep


def measure_chain(reg, world, C, ch, cand_raw, max_ext=6, regsig=None):
    """Re-measure the object on every frame of the chain's span, then keep extending past both ends
    (extrapolated in world coordinates, re-measured) until it is lost for 2 frames.
    Returns sorted list of [k, u, v, w, h]."""
    P = reg.P
    t = P.t
    pts = {f: (float(C[f][j, 0]), float(C[f][j, 1]), float(max(cand_raw[f][j, 4], cand_raw[f][j, 5])))
           for f, j in ch}

    def model():
        ks = sorted(pts)
        return ks, np.array([t[k] for k in ks], float) / 1e9, np.array([pts[k][:2] for k in ks]), \
            np.array([pts[k][2] for k in ks])

    def xy_at(tq, M):
        ks, ts, XY, _ = M
        i = int(np.searchsorted(ts, tq))
        lo, hi = max(0, i - 4), min(len(ts), i + 4)
        if hi - lo < 3:
            lo, hi = max(0, hi - 4), min(len(ts), lo + 4)
        tt = ts[lo:hi] - tq
        deg = min(2, hi - lo - 1)
        if tq < ts[0] or tq > ts[-1]:
            deg = min(deg, 1)   # extrapolate linearly
        A = np.stack([tt ** p for p in range(deg + 1)], 1)
        sol, *_ = np.linalg.lstsq(A, XY[lo:hi], rcond=None)
        return sol[0]

    def size_at(tq, M):
        ks, ts, XY, S = M
        i = min(int(np.searchsorted(ts, tq)), len(ts) - 1)
        return float(np.median(S[max(0, i - 3):i + 4]))

    out = {}
    M = model()
    k0, k1 = M[0][0], M[0][-1]
    ref_peak = float(np.median([cand_raw[f][j, 3] for f, j in ch]))
    for k in range(k0, k1 + 1):
        tq = t[k] / 1e9
        uv = world.to_image(k, xy_at(tq, M))[0]
        if not (-5 <= uv[0] < W + 5 and -5 <= uv[1] < H + 5):
            continue
        m = _measure_at(reg, world, k, uv, size_at(tq, M), lambda tt: xy_at(tt, M))
        if m is not None:
            out[k] = m
    # extend past the ends
    for direction in (1, -1):
        k = k1 if direction > 0 else k0
        miss = 0
        n_ext = 0
        while miss < 2 and n_ext < 40:
            k += direction
            if k < 0 or k >= P.N:
                break
            M = model()
            tq = t[k] / 1e9
            if regsig is not None and regsig[max(k, k - direction)] > 2.0:
                break   # registration across this step is unreliable: do not extrapolate through it
            if abs(tq - (M[1][-1] if direction > 0 else M[1][0])) > 0.15:
                break
            uv = world.to_image(k, xy_at(tq, M))[0]
            if not (-5 <= uv[0] < W + 5 and -5 <= uv[1] < H + 5):
                break
            sz = size_at(tq, M)
            m = _measure_at(reg, world, k, uv, sz, lambda tt: xy_at(tt, M), strict=True, ref_peak=ref_peak)
            if m is None:
                miss += 1
                continue
            miss = 0
            n_ext += 1
            out[k] = m
            XY, _ = world.to_plane(k, np.array([m[:2]]))
            pts[k] = (float(XY[0, 0]), float(XY[0, 1]), float(max(m[2], m[3])))
    return [[k] + out[k] for k in sorted(out)]


def _measure_at(reg, world, k, uv, sz, xy_at, strict=False, ref_peak=None):
    P = reg.P
    t = P.t
    r = int(max(16, 2.5 * sz + 10))
    x0, y0 = int(round(uv[0])) - r, int(round(uv[1])) - r
    x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    refs = []
    for L in (45.0, 90.0, 140.0, 200.0, 300.0):
        for j in ref_frames(t, k, L):
            if j is None or j in refs:
                continue
            # where is the object at time of j, in frame-k pixel coords?
            pj = world.to_image(k, xy_at(t[j] / 1e9))[0]
            if np.hypot(*(pj - uv)) >= sz + 6:
                refs.append(j)
    if len(refs) < 2:
        return None
    refs = refs[:6]
    stack = []
    for j in refs:
        wr, val = reg.warp(j, k)
        crop = wr[cy0:cy1, cx0:cx1].copy()
        crop[~val[cy0:cy1, cx0:cx1]] = np.nan
        stack.append(crop)
    stack = np.array(stack)
    good = np.isfinite(stack).sum(0)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(stack, axis=0)
        spread = np.nanmax(stack, 0) - np.nanmin(stack, 0)
    bg[good < 2] = np.nan
    cur = P.fr[k][cy0:cy1, cx0:cx1].astype(np.float32)
    diff = cur - bg
    diff[np.isnan(diff)] = 0
    spread[np.isnan(spread)] = 255
    ad = np.abs(diff) - 0.5 * spread
    lu, lv = uv[0] - cx0, uv[1] - cy0
    yy, xx = np.mgrid[0:cy1 - cy0, 0:cx1 - cx0]
    near = (xx - lu) ** 2 + (yy - lv) ** 2 <= (max(4.0, 0.8 * sz) + (3 if strict else 4)) ** 2
    if not near.any():
        return None
    pk = float(ad[near].max())
    if pk < (10 if strict else 7):
        return None
    thr = max(5.0, 0.35 * pk)
    mask = (ad >= thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, bd = None, 1e9
    for i in range(1, n):
        comp = lab == i
        if not (comp & near).any():
            continue
        d = np.hypot(cen[i][0] - lu, cen[i][1] - lv)
        if d < bd:
            best, bd = i, d
    if best is None:
        return None
    comp = lab == best
    # absorb other components within the object's extent (fragmented object)
    for i in range(1, n):
        if i != best and np.hypot(cen[i][0] - cen[best][0], cen[i][1] - cen[best][1]) < 0.7 * sz + 3:
            comp |= lab == i
    wts = np.where(comp, np.abs(diff), 0)
    s = wts.sum()
    u = float((wts * xx).sum() / s) + cx0
    v = float((wts * yy).sum() / s) + cy0
    ys, xs = np.nonzero(comp)
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    if w > 3 * sz + 12 or h > 3 * sz + 12:
        return None
    if strict and (max(w, h) > 2.0 * sz + 6 or max(w, h) < 0.4 * sz - 2):
        return None
    if strict and ref_peak is not None and not (0.5 * ref_peak <= pk + 0.5 * float(spread[comp].mean()) <= 2.0 * ref_peak + 10):
        return None
    return [round(u, 2), round(v, 2), w, h]


def dedupe_objects(objs):
    """Drop an object that sits on top of a bigger one (fragment of the same thing) in >= 3 shared frames."""
    def med_size(o):
        return float(np.median([f[3] * f[4] for f in o["frames"]]))
    objs = sorted(objs, key=lambda o: -med_size(o) * len(o["frames"]))
    keep = []
    for o in objs:
        dup = False
        fo = {f[0]: f for f in o["frames"]}
        for q in keep:
            shared = 0
            close = 0
            for f in q["frames"]:
                g = fo.get(f[0])
                if g is None:
                    continue
                shared += 1
                if np.hypot(f[1] - g[1], f[2] - g[2]) <= max(f[3], f[4]) / 2 + 3:
                    close += 1
            if close >= 3 and close >= 0.5 * shared:
                dup = True
                break
        if not dup:
            keep.append(o)
    return sorted(keep, key=lambda o: o["frames"][0][0])


# ----------------------------------------------------------------------------------------- main
def analyse(press_dir, verbose=False, fe_cache=None):
    """fe_cache: optional pickle path caching the (slow) registration + candidate front end."""
    import pickle
    P = Press(press_dir)
    reg = Registrar(P)
    raw = None
    if fe_cache and os.path.exists(fe_cache):
        D = pickle.load(open(fe_cache, "rb"))
        reg.rot, reg.quality, raw = D["rot"], D["quality"], D["raw"]
    O = reg.orient()
    world = World(P, O)
    if raw is None:
        raw = []
        for k in range(P.N):
            pos, neg, used = frame_diff(reg, k)
            c, thr = extract_cands(pos, neg, P.fr[k])
            raw.append(c)
            if verbose and k % 50 == 0:
                print(P.name, k, len(c), "thr %.1f" % thr, flush=True)
        if fe_cache:
            pickle.dump(dict(rot=reg.rot, quality=reg.quality, raw=raw), open(fe_cache, "wb"))
    C = []
    for k, c in enumerate(raw):
        if len(c):
            XY, z = world.to_plane(k, c[:, :2])
            C.append(np.c_[XY, (z > 0.3).astype(np.float32), c[:, 2:]].astype(np.float64))
        else:
            C.append(np.zeros((0, 10)))
    st = static_filter(C, P.t)
    for c, s in zip(C, st):
        if len(c):
            c[s, 2] = 0
    regsig = registration_sigma(reg)
    chains = link(C, P.t, regsig=regsig)
    keep = select_chains(C, P.t, chains, regsig=regsig)
    keep2 = []
    for ch, s in keep:
        s["rel"] = scene_relative(reg, ch, raw)
        if s["rel"] >= MIN_SCENE_REL_PX:
            keep2.append((ch, s))
    rejected = [(ch, s) for ch, s in keep if s["rel"] < MIN_SCENE_REL_PX]
    keep = _merge_chains(keep2, C, P.t, world)
    objs = []
    for ch, s in keep:
        fr = measure_chain(reg, world, C, ch, raw, regsig=regsig)
        if len(fr) < 6:
            continue
        objs.append(dict(frames=fr, stats=s, chain=ch))
    objs = dedupe_objects(objs)
    reg.rejected = rejected
    return P, reg, world, raw, C, st, chains, objs


def extract(press_dir, verbose=False, return_all=False):
    P, reg, world, raw, C, st, chains, objs = analyse(press_dir, verbose)
    res = {"press": P.name, "frames": P.N, "objects": [], "reviewed": False, "note": ""}
    for i, o in enumerate(objs):
        fr = o["frames"]
        res["objects"].append({"id": i + 1, "label": "unknown", "note": "",
                               "frames": [[int(f[0]), f[1], f[2], f[3], f[4]] for f in fr]})
    if return_all:
        return res, (P, reg, world, raw, C, st, chains, objs)
    return res


def load_frames(press_dir):
    m = json.load(open(os.path.join(press_dir, "meta.json")))
    n = int(m["frames"])
    return np.fromfile(os.path.join(press_dir, "frames.bin"), dtype=np.uint8).reshape(n, H, W)


def review_strip(press_dir, obj, out_png, n=12, crop=96, up=3, cols=4):
    fr = load_frames(press_dir)
    F = obj["frames"]
    idx = np.unique(np.linspace(0, len(F) - 1, min(n, len(F))).round().astype(int))
    tiles = []
    for i in idx:
        k, u, v, w, h = F[i][:5]
        img = fr[k]
        x0, y0 = int(round(u)) - crop // 2, int(round(v)) - crop // 2
        pad = cv2.copyMakeBorder(img, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=0)
        c = pad[y0 + crop:y0 + 2 * crop, x0 + crop:x0 + 2 * crop]
        c = cv2.resize(c, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        cu, cv_ = (u - x0) * up, (v - y0) * up
        hw, hh = (w / 2 + 3) * up, (h / 2 + 3) * up
        for sx in (-1, 1):
            for sy in (-1, 1):
                px, py = int(cu + sx * hw), int(cv_ + sy * hh)
                cv2.line(c, (px, py), (px - sx * 8, py), (0, 0, 255), 1)
                cv2.line(c, (px, py), (px, py - sy * 8), (0, 0, 255), 1)
        cv2.putText(c, "row %d  %dx%d" % (k, w, h), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        tiles.append(c)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    mid = F[len(F) // 2][0]
    th = cv2.cvtColor(fr[mid], cv2.COLOR_GRAY2BGR)
    pts = np.array([[f[1], f[2]] for f in F], np.int32)
    cv2.polylines(th, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(th, (int(F[len(F) // 2][1]), int(F[len(F) // 2][2])), 12, (0, 255, 255), 1)
    cv2.putText(th, "%s obj %d rows %d-%d (n=%d) thumb=row %d" % (os.path.basename(press_dir.rstrip("/\\")),
                obj["id"], F[0][0], F[-1][0], len(F), mid), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA)
    gw = grid.shape[1]
    th = cv2.resize(th, (gw, int(H * gw / W))) if gw < W else th
    if th.shape[1] < gw:
        th = cv2.copyMakeBorder(th, 0, 0, 0, gw - th.shape[1], cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite(out_png, np.vstack([grid, th]))


def _run_one(args):
    press_dir, out_dir, sheets, force = args
    name = os.path.basename(press_dir.rstrip("/\\"))
    out = os.path.join(out_dir, name + ".json")
    if os.path.exists(out) and not force:
        try:
            if json.load(open(out)).get("reviewed"):
                return name, "kept (reviewed)"
        except Exception:
            pass
    try:
        res = extract(press_dir)
    except ValueError as e:
        return name, "skip: %s" % e
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    if sheets:
        os.makedirs(sheets, exist_ok=True)
        for o in res["objects"]:
            review_strip(press_dir, o, os.path.join(sheets, "%s_obj%d.png" % (name, o["id"])))
    return name, "%d objects" % len(res["objects"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("press_dirs", nargs="+")
    ap.add_argument("--out-dir", default="bench_data/truth")
    ap.add_argument("--sheets", default=None, help="review strip dir (default OUT_DIR/sheets)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="overwrite files already marked reviewed")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    sheets = a.sheets or os.path.join(a.out_dir, "sheets")
    jobs = [(d, a.out_dir, sheets, a.force) for d in a.press_dirs]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for name, msg in ex.map(_run_one, jobs):
                print(name, msg, flush=True)
    else:
        for j in jobs:
            print(*_run_one(j), flush=True)


if __name__ == "__main__":
    main()
