"""Scenery filter: candidates that stay put in the WORLD while the gun turns are not clays.

The detector finds small blobs that stand out from their surroundings. Against open sky that is the clay;
with trees, roofs, wires or a fence in view it is also every leaf gap and knot, and they crowd the clay out of
the few detections that are measured and tracked (real stills + a pasted clay: 5 % of frames kept the clay).
What separates them is motion: the IMU says how the gun turned between frames, so a blob's pixel can be turned
into a direction in the world. Scenery keeps its world direction from frame to frame, a flying clay does not
(even one the shooter is swinging with: it is nearly still in the IMAGE and moving in the world).

`StaticFilter.filter` is the detector's `candidate_filter`: it gets every candidate's centroid before the
expensive measurement, drops those that were already there, in the world, a little earlier, and returns the
rest in priority order.

"Already there" allows for parallax. The camera sits `static_lever_m` ahead of the point the shooter turns
about, so turning also moves it sideways by dp = lever x (change of the optical axis), and scenery at range R
shifts by -dp/R against the turn: nothing at infinity, most for the nearest scenery (`static_min_range_m`).
A candidate at direction c matches an earlier candidate d when
    c = d - lambda * dp_perp   for some 0 <= lambda <= 1 / static_min_range_m
to within `static_tol_px` (plus 1 % of the turn for focal-length and timing error). That is a short line
segment trailing each old candidate, not a circle around it: a clay swung with moves ALONG the turn, the
opposite way, and a circle wide enough for a fence at 6 m would swallow a quarter of a cluttered frame.
Compared against a few frames `static_min_age_ms` .. `static_max_age_ms` old.

Limits, by design:
  * a clay with almost no angular motion (dead straight away or straight in, below ~tol / min_age =
    1.7 deg/s by default), or one drifting slowly against the swing, looks exactly like scenery here.
    `protect` keeps candidates near a track that has already been seen crossing the sky, so a clay that
    slows down is not lost; one that never moved across the sky is.
  * scenery nearer than static_min_range_m shifts more than allowed and passes as a slow mover (the tracker's
    speed gate deals with it).
  * no IMU, no filter: without attitude a still image region says nothing (the pipeline switches it off).
"""

from __future__ import annotations

import ctypes
import math
from collections import deque

import numpy as np

from tbot.vision import _native

__all__ = ["StaticFilter"]

NEW_MARGIN_PX = 4.0   # a candidate this close to where the old frame ended was not in view then
COMPARE_FRAMES = 4    # old frames a candidate is compared with, spread over the age window
TURN_TOL = 0.01       # extra tolerance per radian turned (focal length +-0.3 %, IMU/camera timing)
STATIC_LISTED = 24    # dropped candidates reported per frame (telemetry / overlay)
SIZE_TOL = 0.15       # extra matching tolerance as a fraction of a candidate's own extent


class StaticFilter:
    def __init__(self, cfg, cam, mount) -> None:
        self.cfg = cfg
        self.cam = cam
        self.mount = mount
        # camera -> gun rotation (mount.gun_to_cam is v_c = v_g @ R_gun_cam)
        self._R_gc = np.asarray(mount.gun_to_cam(np.eye(3)), dtype=np.float64).reshape(3, 3)
        self._hist: deque = deque()  # (t_ns, R_wc, dirs_w (N,3))
        self._t_ns = 0
        self._R_wc: np.ndarray | None = None
        self._protect: np.ndarray | None = None
        self._protect_cos = 1.0
        self._focus: np.ndarray | None = None
        self._focus_cos: np.ndarray | float = 1.0
        self._others_max = 0
        self.n_focus = 0  # last frame: candidates kept because they are where a followed target should be
        self.static_px: list[list[float]] = []  # last frame: the candidates dropped as scenery (full-res px)
        self.n_static = 0
        self.n_new = 0
        ifov = np.asarray(cam.ifov(np.array([[0.5 * cam.width, 0.5 * cam.height]], dtype=np.float64))).ravel()
        self.ifov = float(ifov[0]) if ifov.size and math.isfinite(float(ifov[0])) and ifov[0] > 0 else 1e-3
        # the frame's edges (NEW_MARGIN_PX inside) on the tangent plane x/z, y/z of the camera frame
        m, w, h = NEW_MARGIN_PX, float(cam.width), float(cam.height)
        edge = np.asarray(cam.pixel_to_ray(np.array([[m, 0.5 * h], [w - 1 - m, 0.5 * h], [0.5 * w, m],
                                                     [0.5 * w, h - 1 - m]], dtype=np.float64))).reshape(4, 3)
        self._view = (float(edge[0, 0] / edge[0, 2]), float(edge[1, 0] / edge[1, 2]),
                      float(edge[2, 1] / edge[2, 2]), float(edge[3, 1] / edge[3, 2]))
        self._focus_cos_arr: np.ndarray | None = None
        self._np: _native.SceneryParams | None = None   # native kernel parameters, filled on first use
        self._flags = np.zeros(0, np.uint8)
        self._spx = np.zeros((_native.SCENERY_LISTED, 2), np.float64)
        self.native_used = False   # last frame went through the C kernel

    def reset(self) -> None:
        self._hist.clear()
        self._R_wc = None
        self.static_px = []
        self.n_static = self.n_new = 0

    # ------------------------------------------------------------------ per frame
    def begin(self, t_ns: int, R_wg: np.ndarray, protect_dirs_w=None, protect_px: float = 25.0,
              focus_dirs_w=None, others_max: int = 0, focus_radius_rad=None) -> None:
        """This frame's attitude (before detector.detect). protect_dirs_w: world directions (N,3) of tracks
        known to be clays; candidates within protect_px of one are never called scenery.
        focus_dirs_w: where the targets being followed should be in this frame (world directions), the main
        one first. Candidates within protect_px of one (or within its focus_radius_rad, when that is wider:
        how far off the tracker itself thinks its prediction may be) go first. Of the rest only the best
        others_max are kept (0 = all) - but only in a frame where the MAIN target has a candidate in its
        zone: when it is not where it should be (a turn, a missed frame) the frame is searched as usual, or
        the target would be skipped again and again while its prediction drifts. Each
        measurement costs ~6 ms on a Pi 3B, and while there is something to shoot at, the frame rate on it is
        worth more than a prompt look at every new speck of tree that the swing brings into view."""
        self._t_ns = int(t_ns)
        self._R_wc = np.asarray(R_wg, dtype=np.float64).reshape(3, 3) @ self._R_gc
        if focus_dirs_w is not None and len(focus_dirs_w):
            f = np.asarray(focus_dirs_w, dtype=np.float64).reshape(-1, 3)
            # C order: the native kernel reads it as rows
            self._focus = np.ascontiguousarray(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12))
            floor = float(protect_px) * self.ifov
            if focus_radius_rad is not None and len(focus_radius_rad) == len(f):
                self._focus_cos = np.cos(np.minimum(np.maximum(np.asarray(focus_radius_rad, dtype=np.float64),
                                                               floor), 0.5 * math.pi))
            else:
                self._focus_cos = math.cos(floor)
            self._focus_cos_arr = None   # made when the native kernel needs it
            self._others_max = max(0, int(others_max))
        else:
            self._focus = None
        self.n_focus = 0
        if protect_dirs_w is not None and len(protect_dirs_w):
            p = np.asarray(protect_dirs_w, dtype=np.float64).reshape(-1, 3)
            self._protect = np.ascontiguousarray(p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12))
            self._protect_cos = math.cos(float(protect_px) * self.ifov)
        else:
            self._protect = None
        self.static_px = []
        self.n_static = self.n_new = 0

    def _old_frames(self, t: int, min_age: float) -> list:
        """Up to COMPARE_FRAMES history entries at least min_age old, spread from the newest such to the oldest."""
        hist = self._hist
        n_old = 0
        for h in hist:  # oldest first
            if t - h[0] < min_age:
                break
            n_old += 1
        if n_old <= COMPARE_FRAMES:
            return [hist[i] for i in range(n_old)]
        pick = sorted({int(round(k * (n_old - 1) / (COMPARE_FRAMES - 1))) for k in range(COMPARE_FRAMES)})
        return [hist[i] for i in pick]

    def filter(self, uv: np.ndarray, areas: np.ndarray | None = None, widths: np.ndarray | None = None,
               heights: np.ndarray | None = None) -> np.ndarray:
        """Indices of the candidates to keep, best first: candidates known to move, then candidates that were
        out of view `static_min_age_ms` ago (nothing is known about them yet), the most compact first within
        each (a clay fills its bounding box, a tangle of branches does not); scenery is left out.
        uv: (N,2) full-resolution pixel centroids; areas / widths / heights: the components' pixel counts and
        bounding boxes. Also records the candidates for the frames to come.

        Runs on every frame that has a candidate, so it is written for few NumPy calls (each costs 20-50 us
        on a Pi 3B whatever the array size): the old frames are compared in one stacked pass and the per-frame
        geometry is plain float arithmetic. Measured there for one candidate: 5.7 ms -> 2.6 ms."""
        det = self.cfg.detector
        n = int(len(uv))
        if n == 0 or self._R_wc is None:
            return np.arange(n)
        uv = np.ascontiguousarray(uv, dtype=np.float64).reshape(n, 2)
        t = self._t_ns
        min_age = float(det.static_min_age_ms) * 1e6
        max_age = float(det.static_max_age_ms) * 1e6
        hist = self._hist
        if hist and t < hist[-1][0]:  # the clock went backwards (source restart / replay loop)
            hist.clear()
        while hist and t - hist[0][0] > max_age:
            hist.popleft()

        old = [h for h in self._old_frames(t, min_age) if len(h[2])]
        self.native_used = False
        fn = _native.scenery_kernel() if bool(getattr(det, "native", True)) else None
        if fn is not None:
            keep = self._filter_native(fn, uv, areas, widths, heights, old, det)
            if keep is not None:
                self.native_used = True
                return keep
        return self._filter_python(uv, areas, widths, heights, old, det)

    def _filter_python(self, uv, areas, widths, heights, old, det) -> np.ndarray:
        """The reference for _measure.c's tbot_scenery (and the path when the library is not built)."""
        n = int(len(uv))
        hist = self._hist
        t = self._t_ns
        R_now = self._R_wc
        rays = np.asarray(self.cam.pixel_to_ray(uv), dtype=np.float64).reshape(n, 3)
        dirs = rays @ R_now.T
        static = None
        new = None  # no history yet (start-up, after a reset): nothing is known
        if old:
            tol0 = float(det.static_tol_px) * self.ifov
            lever = float(det.static_lever_m)
            lam_max = 1.0 / max(float(det.static_min_range_m), 0.5)
            r = R_now.ravel().tolist()
            zx, zy, zz = r[2], r[5], r[8]  # optical axis in the world
            tols, gs, counts = [], [], []
            for _t, R_f, d_f in old:
                f = R_f.ravel().tolist()
                tr = sum(a * b for a, b in zip(f, r))  # trace(R_f^T R_now)
                turn = math.acos(min(1.0, max(-1.0, 0.5 * (tr - 1.0))))
                # sideways move of the camera since that frame, and the direction shift per (1/range) it causes
                dx, dy, dz = lever * (zx - f[2]), lever * (zy - f[5]), lever * (zz - f[8])
                along = dx * zx + dy * zy + dz * zz
                gs.append((-(dx - along * zx), -(dy - along * zy), -(dz - along * zz)))
                tols.append(tol0 + TURN_TOL * turn)
                counts.append(len(d_f))
            D = old[0][2] if len(old) == 1 else np.concatenate([h[2] for h in old])   # (M, 3)
            G = np.repeat(np.array(gs, dtype=np.float64), counts, axis=0)               # (M, 3)
            tol = np.repeat(np.array(tols, dtype=np.float64), counts)                   # (M,)
            gg = np.einsum("ij,ij->i", G, G)
            e2 = 2.0 - 2.0 * (dirs @ D.T)                       # |c - d|^2 for unit vectors, (n, M)
            np.maximum(e2, 0.0, out=e2)
            eg = dirs @ G.T
            eg -= np.einsum("ij,ij->i", D, G)                    # (c - d) . g
            lam = eg / np.maximum(gg, 1e-12)
            np.clip(lam, 0.0, lam_max, out=lam)
            e2 += lam * (lam * gg - 2.0 * eg)                    # |c - d - lam g|^2
            if widths is not None and heights is not None:
                # a big ragged blob's centroid wanders with the threshold: allow it a share of its own size
                size_tol = (SIZE_TOL * self.ifov) * np.maximum(np.asarray(widths, dtype=np.float64),
                                                              np.asarray(heights, dtype=np.float64))
                lim = tol[None, :] + size_tol[:, None]
            else:
                lim = tol[None, :]
            static = (e2 <= lim * lim).any(axis=1)
            # was this direction in view in the newest comparison frame? If not, nothing is known about it yet.
            # (tangent-plane test against the frame's edges: exact for the pinhole lenses this is used with,
            # a pixel or two out at the corners of a distorted one - NEW_MARGIN_PX covers it)
            then = dirs @ old[-1][1]
            x0, x1, y0, y1 = self._view
            with np.errstate(invalid="ignore", divide="ignore"):
                tx = then[:, 0] / then[:, 2]
                ty = then[:, 1] / then[:, 2]
                new = ~((then[:, 2] > 1e-6) & (tx >= x0) & (tx <= x1) & (ty >= y0) & (ty <= y1))
        if static is None:
            static = np.zeros(n, dtype=bool)
        if new is None:
            new = np.ones(n, dtype=bool)
        if self._protect is not None and static.any():
            near = ((dirs @ self._protect.T) >= self._protect_cos).any(axis=1)
            static &= ~near
        hist.append((t, R_now, dirs))

        self.n_static = int(static.sum())
        self.n_new = int((new & ~static).sum())
        if self.n_static:
            self.static_px = np.round(uv[static][:STATIC_LISTED], 1).tolist()
        keep = np.flatnonzero(~static)
        if keep.size > 1:
            if areas is not None and widths is not None and heights is not None:
                box = np.maximum(np.asarray(widths, dtype=np.float64) * np.asarray(heights, dtype=np.float64), 1.0)
                # solid AND with some size: a one-pixel speck fills its box perfectly and is the last thing
                # worth one of the few measurements a cluttered frame gets
                fill = np.asarray(areas, dtype=np.float64) / box * np.sqrt(np.asarray(areas, dtype=np.float64))
            else:
                fill = np.ones(n)
            keep = keep[np.lexsort((-fill[keep], new[keep]))]  # movers before unknowns, solid and sizeable first
        if self._focus is not None and keep.size:
            hit = (dirs[keep] @ self._focus.T) >= self._focus_cos   # (kept, targets)
            near = hit.any(axis=1)
            self.n_focus = int(near.sum())
            others = keep[~near]
            if self._others_max and hit[:, 0].any():
                others = others[:self._others_max]
            keep = np.concatenate((keep[near], others))
        return keep

    # ------------------------------------------------------------------ native (tbot/vision/_measure.c)
    def _native_params(self) -> _native.SceneryParams:
        """The per-camera part of the kernel's parameters (once)."""
        p = _native.SceneryParams()
        cam = self.cam
        fast = getattr(cam, "_fast", None)
        if fast is None:
            p.model = _native.RAYS_GIVEN   # a model with no plain-float form: rays from cam.pixel_to_ray
        else:
            fx, fy, cx, cy, skew, k = fast
            p.fx, p.fy, p.cx, p.cy, p.skew = fx, fy, cx, cy, skew
            if cam.model == "fisheye":
                p.model = _native.RAYS_FISHEYE
            elif k is None:
                p.model = _native.RAYS_PINHOLE_IDEAL
            else:
                p.model = _native.RAYS_PINHOLE
            for i, v in enumerate(k or ()):
                p.k[i] = float(v)
        from tbot import optics
        p.newton_iters = int(optics._NEWTON_ITERS)
        p.undistort_iters = int(optics._UNDISTORT_ITERS)
        p.newton_tol = float(optics._NEWTON_TOL)
        p.turn_tol = TURN_TOL
        p.size_tol = SIZE_TOL * self.ifov
        for i, v in enumerate(self._view):
            p.view[i] = v
        p.static_px = self._spx.ctypes.data
        return p

    @staticmethod
    def _arr(a):
        """(address, type code, keep-alive) of a candidate attribute array for the kernel."""
        if a is None:
            return None, 0, None
        if not isinstance(a, np.ndarray) or a.dtype not in _native.ARR_TYPES or not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float64)
        return a.ctypes.data, _native.ARR_TYPES[a.dtype], a

    def _filter_native(self, fn, uv, areas, widths, heights, old, det) -> np.ndarray | None:
        n = int(len(uv))
        if len(old) > _native.SCENERY_MAX_OLD:
            return None
        p = self._np
        if p is None:
            p = self._np = self._native_params()
        R_now = self._R_wc
        if not R_now.flags.c_contiguous:
            R_now = self._R_wc = np.ascontiguousarray(R_now)
        rays = None
        if p.model == _native.RAYS_GIVEN:
            rays = np.ascontiguousarray(self.cam.pixel_to_ray(uv), dtype=np.float64).reshape(n, 3)
            p.rays = rays.ctypes.data
        p.n = n
        p.uv = uv.ctypes.data
        p.R_now = R_now.ctypes.data
        p.n_old = len(old)
        for i, h in enumerate(old):
            # entries the native path made carry their arrays' addresses (each .ctypes costs a few us)
            p.old_m[i] = len(h[2])
            p.old_R[i], p.old_dirs[i] = (h[3], h[4]) if len(h) > 4 else (h[1].ctypes.data, h[2].ctypes.data)
        p.tol0 = float(det.static_tol_px) * self.ifov
        p.lever = float(det.static_lever_m)
        p.lam_max = 1.0 / max(float(det.static_min_range_m), 0.5)
        sized = widths is not None and heights is not None
        a_p, a_t, a_keep = self._arr(areas if sized else None)
        w_p, w_t, w_keep = self._arr(widths if sized else None)
        h_p, h_t, h_keep = self._arr(heights if sized else None)
        p.areas, p.areas_t, p.widths, p.widths_t, p.heights, p.heights_t = a_p, a_t, w_p, w_t, h_p, h_t
        if self._protect is not None:
            p.protect, p.n_protect, p.protect_cos = self._protect.ctypes.data, len(self._protect), self._protect_cos
        else:
            p.protect, p.n_protect = None, 0
        if self._focus is not None:
            fc = self._focus_cos_arr
            if fc is None:
                c = self._focus_cos
                fc = self._focus_cos_arr = (np.ascontiguousarray(c, dtype=np.float64) if isinstance(c, np.ndarray)
                                            else np.full(len(self._focus), float(c)))
            p.focus, p.focus_cos, p.n_focus, p.others_max = (self._focus.ctypes.data, fc.ctypes.data,
                                                             len(self._focus), self._others_max)
        else:
            p.focus, p.focus_cos, p.n_focus = None, None, 0
        dirs = np.empty((n, 3), np.float64)
        keep = np.empty(n, np.intp)
        if self._flags.size < n:
            self._flags = np.zeros(max(n, 2 * self._flags.size, 64), np.uint8)
        d_addr = dirs.ctypes.data
        p.dirs, p.keep, p.flags = d_addr, keep.ctypes.data, self._flags.ctypes.data
        rc = fn(ctypes.addressof(p))
        del a_keep, w_keep, h_keep, rays
        if rc != 0:
            return None
        self._hist.append((self._t_ns, R_now, dirs, p.R_now, d_addr))
        self.n_static = int(p.n_static)
        self.n_new = int(p.n_new)
        self.n_focus = int(p.n_focus_kept)
        if self.n_static:
            self.static_px = self._spx[:p.n_listed].tolist()
        return keep[:p.n_keep]
