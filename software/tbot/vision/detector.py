"""Clay detection on mono (and optionally colour) frames — contract: docs/ARCHITECTURE.md [detector].

The OV9281 is monochrome and a 108 mm clay spans only ~1.5-8 px at 8-40 m with the 640x400 prior
(fx = 466.7 px), so detection is *local contrast* against a smoothly varying sky, never hue: a box-blur
background on a downscaled image is subtracted, the residual is smoothed by a small Gaussian matched to a
distant clay (`detector.matched_sigma_px`) and thresholded in robust noise sigmas OF THAT SMOOTHED IMAGE, and
the surviving connected components are measured with sub-pixel, contrast-weighted moments on the raw residual.
The matched filter is what finds a 25-30 m clay whose light is spread over 2-4 pixels, none of which clears the
per-pixel threshold alone, and it suppresses single-pixel spikes (hot pixels) relative to the noise.
Colour sources (laptop webcam) may additionally AND/OR an HSV orange mask.

Everything is vectorised (no per-pixel Python) and works on preallocated buffers so the Pi 3B does not
reallocate ~1 MB per frame at 120 fps.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import cv2
import numpy as np

from tbot.config import Config
from tbot.types import Detection, Frame

# 8-bit quantisation alone is uniform over 1 DN => sigma = 1/sqrt(12); never report less than that,
# so snr stays finite on synthetic/flat imagery.
NOISE_SIGMA_FLOOR = 1.0 / 12.0**0.5
# Noise sigma is estimated on a strided subsample: this many samples is plenty for a MAD (<1 % error)
# and costs ~30 kB instead of a full-frame sort.
NOISE_TARGET_SAMPLES = 20000
# Below this many usable samples the mask is ignored for the noise estimate (the image is then almost
# entirely masked and the few remaining pixels cannot support a robust estimate).
NOISE_MIN_SAMPLES = 64
# Absolute deviations are histogrammed up to this many DN (the quantile of interest sits at ~1 sigma).
DEV_MAX = 64
# Safety valve for pathological frames (e.g. colour OR-mode on a scene full of orange): measure at most
# this many components per frame, largest first. A real sky yields a handful.
CANDIDATE_LIMIT = 64
# With the matched filter on, a blob is MEASURED on the raw residual above this many raw sigmas (the
# smoothed, grown detection footprint is fatter than the object and would bias the size up ~12 %).
MEAS_SEED_SIGMAS = 2.5
# Hysteresis growth reaches this far from a seed pixel (an edge-on clay is ~2 px long at 30 m).
GROW_RADIUS_PX = 2
# Matched/grown components may exceed max_area_px by this factor before their raw footprint is measured.
COMPONENT_AREA_HEADROOM = 3
# Local-contrast test (matched mode): ring distance beyond the blob's half-size, and the required margin.
LOCAL_RING_PX = 3
LOCAL_K_SIGMA = 4.5
# The matched-filter image is kept in int16 at this many counts per DN (integer noise estimator, finer steps).
MATCHED_SCALE = 4
# Variance of a unit pixel box (uniform over 1 px): the smallest second moment anything can have.
PIXEL_BOX_VAR = 1.0 / 12.0
# Seed weights (thresholded pixels only) are soft-thresholded at this many sigma so noise barely
# counts: E[(|n| - 1.5 sigma)+] = 0.059 sigma.
MOMENT_FLOOR_SIGMAS = 1.5
# Adaptive moments: the Gaussian weight is this multiple of the current object estimate. 1.0 is the
# classic matched weight (best S/N) but it biases a flat disc low because the profile is not Gaussian;
# a broader weight trades a little noise for a near-unbiased second moment (measured in the report).
ADAPTIVE_ALPHA = 2.5
ADAPTIVE_ITERS = 6
ADAPTIVE_TOL = 0.02
ADAPTIVE_DAMPING = 0.5
# A Gaussian-matched moment over-reports a FLAT disc (a clay has no Gaussian tails, so dividing the
# weight out over-corrects) by a scale-invariant factor. Fitted over D = 2-10 px x system blur
# 1.15-2.7 px (see the slice report): dividing by 1.09 leaves <= 7 % bias anywhere in that box, for
# both disc-dominated and blur-dominated blobs.
PROFILE_FACTOR = 1.09
# Half-window in sigmas of the Gaussian weight (clipped to the min/max half-widths below).
WINDOW_SIGMAS = 3.0
ADAPTIVE_MIN_HALF = 5
ADAPTIVE_MAX_HALF = 40
# A Gaussian weight narrower than this samples almost nothing and the deconvolution is unstable.
MIN_WEIGHT_VAR = 0.5

_OFF3_Y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
_OFF3_X = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

_POLARITIES = ("bright", "dark", "both")
_MODES = ("auto", "mono", "color")
_COMBINES = ("and", "or")


class _LazyContrast:
    """gray - bg (int16) computed only for the slices/indices actually read.

    With the matched filter on, the full-frame raw residual is only needed on a subsample (noise estimate)
    and inside a few measurement windows; a full uint8 -> int16 subtraction costs ~2.2 ms on a Pi 3B."""

    def __init__(self, gray: np.ndarray, bg: np.ndarray):
        self.gray, self.bg = gray, bg
        self.shape = gray.shape

    def __getitem__(self, idx):
        return self.gray[idx].astype(np.int16) - self.bg[idx].astype(np.int16)

    def full(self) -> np.ndarray:
        return cv2.subtract(self.gray, self.bg, dtype=cv2.CV_16S)


class ClayDetector:
    """Local-contrast clay detector.

    `cfg.detector` is re-read on every `detect()` call, so a tuner may mutate it live.
    """

    def __init__(self, cfg: Config, mask: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.timings_ms: dict[str, float] = {}
        self.debug: dict[str, np.ndarray] = {}
        self.noise_sigma: float = float("nan")
        self.match_sigma: float = float("nan")  # noise sigma of the matched-filter image, in DN
        self.frames_seen: int = 0
        self._since_noise: int = 0
        self._key: tuple | None = None
        self._mask_src: np.ndarray | None = None
        self._mask: np.ndarray | None = None  # bool, runtime size
        self._mask_key: tuple | None = None
        self.set_mask(mask)

    # ------------------------------------------------------------------ mask
    def set_mask(self, mask: np.ndarray | None) -> None:
        """Usable-pixel mask (255/non-zero = usable). Any size; resized NEAREST to the frame size."""
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim != 2:
                raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
        self._mask_src = mask
        self._mask = None
        self._mask_key = None

    def _mask_for(self, h: int, w: int) -> np.ndarray | None:
        if self._mask_src is None:
            return None
        if self._mask_key == (h, w):
            return self._mask
        src = self._mask_src
        if src.shape != (h, w):
            u8 = src if src.dtype == np.uint8 else (src != 0).astype(np.uint8) * 255
            src = cv2.resize(u8, (w, h), interpolation=cv2.INTER_NEAREST)
        self._mask = np.ascontiguousarray(src != 0)
        self._mask_key = (h, w)
        return self._mask

    # --------------------------------------------------------------- buffers
    def _ensure_buffers(self, h: int, w: int, down: int, color: bool) -> None:
        key = (h, w, down, color)
        if self._key == key:
            return
        dw = max(1, int(round(w / down)))
        dh = max(1, int(round(h / down)))
        self._small = np.empty((dh, dw), np.uint8)
        self._smooth = np.empty((dh, dw), np.uint8)
        self._bg = np.empty((h, w), np.uint8)
        self._contrast = np.empty((h, w), np.int16)
        self._matched = np.empty((h, w), np.int16)
        self._gsm = np.empty((h, w), np.uint8)
        # inRange writes byte masks; bool views also work with NumPy's logical operations.
        self._bin = np.empty((h, w), np.uint8).view(np.bool_)
        self._bin2 = np.empty((h, w), np.uint8).view(np.bool_)
        self._orange = np.empty((h, w), np.uint8) if color else None
        self._gray = np.empty((h, w), np.uint8) if color else None
        self._hsv = np.empty((h, w, 3), np.uint8) if color else None
        self._key = key

    # ---------------------------------------------------------------- detect
    def detect(self, frame: Frame) -> list[Detection]:
        """Detections in one frame, best score first, at most cfg.detector.max_detections."""
        t0 = time.perf_counter()
        det = self.cfg.detector
        if det.mode not in _MODES:
            raise ValueError(f"detector.mode must be one of {_MODES}, got {det.mode!r}")
        if det.polarity not in _POLARITIES:
            raise ValueError(f"detector.polarity must be one of {_POLARITIES}, got {det.polarity!r}")
        if det.color_combine not in _COMBINES:
            raise ValueError(f"detector.color_combine must be one of {_COMBINES}, got {det.color_combine!r}")

        img = frame.image
        if img.dtype != np.uint8:
            raise ValueError(f"frame image must be uint8, got {img.dtype}")
        if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
            raise ValueError(f"frame image must be (H,W) mono or (H,W,3) BGR, got shape {img.shape}")
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        is_color = img.ndim == 3
        # mode "color" on a mono frame has no hue to use: contrast only.
        use_hsv = is_color and det.mode in ("auto", "color")
        h, w = img.shape[0], img.shape[1]
        down = max(1, int(det.bg_downscale))
        self._ensure_buffers(h, w, down, is_color)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dst=self._gray) if is_color else img

        # ---- background and signed contrast -------------------------------
        k = max(1, int(det.bg_kernel))
        cv2.resize(gray, (self._small.shape[1], self._small.shape[0]), dst=self._small,
                   interpolation=cv2.INTER_AREA)
        cv2.blur(self._small, (k, k), dst=self._smooth)
        cv2.resize(self._smooth, (w, h), dst=self._bg, interpolation=cv2.INTER_LINEAR)
        ms = float(getattr(det, "matched_sigma_px", 0.0) or 0.0)
        if ms > 0.0 and not use_hsv:
            contrast = _LazyContrast(gray, self._bg)  # raw residual on demand (see _LazyContrast)
        else:
            contrast = cv2.subtract(gray, self._bg, dst=self._contrast, dtype=cv2.CV_16S)
        t_bg = time.perf_counter()

        # ---- matched filter, noise sigma and threshold ---------------------
        if ms > 0.0:
            # Matched filter on the 8-bit image (OpenCV's fixed-point uint8 Gaussian is fast on ARM; an int16
            # separable filter cost ~4 ms on the Pi 3B), then the same background subtraction as the plain
            # contrast: smoothing is linear and the background is already smooth, so this IS the smoothed
            # contrast, at 1 DN resolution.
            ks = 2 * int(math.ceil(3.0 * ms)) + 1
            cv2.GaussianBlur(gray, (ks, ks), ms, dst=self._gsm, borderType=cv2.BORDER_REFLECT)
            cv2.subtract(self._gsm, self._bg, dst=self._matched, dtype=cv2.CV_16S)
            stat = self._matched
            unit = 1.0
        else:
            stat = contrast
            unit = 1.0
        mask = self._mask_for(h, w)
        if self._since_noise <= 0 or not np.isfinite(self.noise_sigma):
            self.noise_sigma = self._estimate_sigma(contrast, mask)
            self.match_sigma = self._estimate_sigma(stat, mask) / unit if ms > 0.0 else self.noise_sigma
            self._since_noise = max(1, int(det.noise_every))
        self._since_noise -= 1
        self.frames_seen += 1
        sigma = self.noise_sigma
        # the detection statistic is thresholded in its own sigmas; the DN floor (min_contrast) is applied
        # to each component's raw peak in _measure, since smoothing lowers a small blob's peak
        thr_dn = float(det.k_sigma) * self.match_sigma if ms > 0.0 else max(float(det.k_sigma) * sigma,
                                                                             float(det.min_contrast))
        thr = thr_dn * unit

        binary = self._bin
        if math.isfinite(thr):
            # Contrast is integral in [-255, 255], so strict float comparisons reduce to
            # inclusive integer ranges. OpenCV writes the byte mask without converting the
            # entire frame to floating point, and accepts reusable destination buffers.
            top = int(255 * unit)
            bright_lo = max(-top - 1, min(top + 1, math.floor(thr) + 1))
            dark_hi = max(-top - 1, min(top + 1, math.ceil(-thr) - 1))
            if det.polarity == "bright":
                cv2.inRange(stat, bright_lo, top, dst=binary.view(np.uint8))
            elif det.polarity == "dark":
                cv2.inRange(stat, -top, dark_hi, dst=binary.view(np.uint8))
            else:
                cv2.inRange(stat, bright_lo, top, dst=binary.view(np.uint8))
                cv2.inRange(stat, -top, dark_hi, dst=self._bin2.view(np.uint8))
                np.logical_or(binary, self._bin2, out=binary)
        elif det.polarity == "bright":
            np.greater(stat, thr, out=binary)
        elif det.polarity == "dark":
            np.less(stat, -thr, out=binary)
        else:
            np.greater(stat, thr, out=binary)
            np.less(stat, -thr, out=self._bin2)
            np.logical_or(binary, self._bin2, out=binary)

        if use_hsv:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=self._hsv)
            lo = np.asarray(det.hsv_lo, dtype=np.uint8)
            hi = np.asarray(det.hsv_hi, dtype=np.uint8)
            cv2.inRange(hsv, lo, hi, dst=self._orange)
            orange = self._orange.view(np.bool_)
            if det.color_combine == "and":
                np.logical_and(binary, orange, out=binary)
            else:
                np.logical_or(binary, orange, out=binary)
        if mask is not None:
            np.logical_and(binary, mask, out=binary)
        gk = float(getattr(det, "grow_k_sigma", 0.0) or 0.0)
        grow_low = gk * (thr / float(det.k_sigma)) if 0.0 < gk < float(det.k_sigma) and math.isfinite(thr) else None
        t_thr = time.perf_counter()

        # ---- components ---------------------------------------------------
        # Hue-selected blobs (colour sources) may have no luma contrast at all: measure them the plain way.
        matched_on = ms > 0.0 and not use_hsv
        dets = self._measure(binary, contrast, sigma, det, frame.t_exposure_ns, mask,
                             stat if matched_on else None, self.match_sigma * unit if matched_on else None,
                             grow_low if matched_on else None)
        t_cc = time.perf_counter()

        dets.sort(key=lambda d: d.score, reverse=True)
        if len(dets) > int(det.max_detections):
            dets = dets[: int(det.max_detections)]

        if det.debug:
            full = contrast.full() if isinstance(contrast, _LazyContrast) else contrast.copy()
            debug = {"contrast": full, "binary": np.where(binary, 255, 0).astype(np.uint8)}
            if ms > 0.0:
                debug["matched"] = self._matched.copy()
            if use_hsv:
                debug["orange"] = np.where(orange, 255, 0).astype(np.uint8)
            self.debug = debug
        elif self.debug:
            self.debug = {}

        self.timings_ms = {
            "bg": (t_bg - t0) * 1e3,
            "threshold": (t_thr - t_bg) * 1e3,
            "components": (t_cc - t_thr) * 1e3,
            "total": (time.perf_counter() - t0) * 1e3,
        }
        return dets

    @staticmethod
    @lru_cache(maxsize=8)
    def _matched_kernel(ms: float) -> np.ndarray:
        ks = 2 * int(math.ceil(3.0 * ms)) + 1
        k = cv2.getGaussianKernel(ks, ms).astype(np.float32) * np.float32(MATCHED_SCALE ** 0.5)
        k.setflags(write=False)
        return k

    # ------------------------------------------------------------- hysteresis
    @staticmethod
    def _grown_enough(labels, stats, lid: int, stat: np.ndarray, low: float, det, mask, kernel) -> bool:
        """Hysteresis for one small seed component: does it reach min_area_px once same-polarity pixels above
        `low` within GROW_RADIUS_PX of it are added? Local (a few pixels), not a flood fill: an edge-on clay's
        partial pixels count, a faint cloud edge or sky gradient cannot be followed, and a dark seed only takes
        dark pixels (a big dark blob is ringed by a bright halo where it pulls its own background down)."""
        r = GROW_RADIUS_PX
        h, w = labels.shape
        x0, y0 = int(stats[lid, cv2.CC_STAT_LEFT]), int(stats[lid, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[lid, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[lid, cv2.CC_STAT_HEIGHT])
        wx0, wy0, wx1, wy1 = max(0, x0 - r), max(0, y0 - r), min(w, x1 + r), min(h, y1 + r)
        seed = labels[wy0:wy1, wx0:wx1] == lid
        st = stat[wy0:wy1, wx0:wx1]
        sign = 1 if int(st[seed].sum()) >= 0 else -1
        weak = (st * sign) > low
        if mask is not None:
            weak &= mask[wy0:wy1, wx0:wx1]
        near = cv2.dilate(seed.view(np.uint8), kernel).view(np.bool_)
        return int(np.count_nonzero(seed | (near & weak))) >= int(det.min_area_px)

    # ----------------------------------------------------------------- noise
    def _estimate_sigma(self, contrast: np.ndarray, mask: np.ndarray | None) -> float:
        h, w = contrast.shape
        step = max(1, int(round((h * w / NOISE_TARGET_SAMPLES) ** 0.5)))
        sub = contrast[::step, ::step]
        if mask is not None:
            vals = sub[mask[::step, ::step]]
            if vals.size < NOISE_MIN_SAMPLES:
                vals = sub.reshape(-1)
        else:
            vals = sub.reshape(-1)
        if vals.size == 0:
            return NOISE_SIGMA_FLOOR
        return _robust_sigma(vals)

    # ------------------------------------------------------------ components
    def _measure(self, binary: np.ndarray, contrast: np.ndarray, sigma: float,
                 det, t_ns: int, mask: np.ndarray | None,
                 matched: np.ndarray | None = None, matched_sigma: float | None = None,
                 grow_low: float | None = None) -> list[Detection]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.view(np.uint8), connectivity=8)
        if n <= 1:
            return []
        areas = stats[1:, cv2.CC_STAT_AREA]
        ws = stats[1:, cv2.CC_STAT_WIDTH]
        hs = stats[1:, cv2.CC_STAT_HEIGHT]
        long_side = np.maximum(ws, hs).astype(np.float64)
        short_side = np.maximum(np.minimum(ws, hs), 1).astype(np.float64)
        if matched is None:
            keep = (
                (areas >= int(det.min_area_px))
                & (areas <= int(det.max_area_px))
                & (long_side / short_side <= float(det.max_aspect))
            )
        else:
            # Matched/grown footprints are a little fatter than the object (growth is local, <= GROW_RADIUS_PX):
            # screen components with headroom here - large sky structures and streaks still fail - and apply
            # the exact max_area / aspect limits to each blob's RAW footprint once it is found (below).
            min_seed = 1 if grow_low is not None else int(det.min_area_px)  # small seeds are grown below
            keep = (
                (areas >= min_seed)
                & (areas <= COMPONENT_AREA_HEADROOM * int(det.max_area_px))
                & (long_side / short_side <= float(det.max_aspect))
            )
        idx = np.nonzero(keep)[0]
        if idx.size and matched is not None and grow_low is not None:
            # vectorised pre-screen of tiny seeds (the Python loop below costs ~0.1 ms each on a Pi 3B): the
            # raw residual somewhere in the 3x3 around the seed's corner pixel must reach the DN floor
            small = areas[idx] < int(det.min_area_px)
            if small.any():
                ys = stats[idx[small] + 1, cv2.CC_STAT_TOP]
                xs = stats[idx[small] + 1, cv2.CC_STAT_LEFT]
                hh, ww = contrast.shape
                yy = np.clip(ys[:, None] + _OFF3_Y[None, :], 0, hh - 1)
                xx = np.clip(xs[:, None] + _OFF3_X[None, :], 0, ww - 1)
                best = np.abs(contrast[yy, xx].astype(np.int32)).max(axis=1)
                ok = np.ones(idx.size, dtype=bool)
                ok[np.flatnonzero(small)] = best >= float(det.min_contrast)
                idx = idx[ok]
        if idx.size == 0:
            return []
        if idx.size > CANDIDATE_LIMIT:  # keep the largest; a real frame never gets here
            idx = idx[np.argsort(areas[idx])[::-1][:CANDIDATE_LIMIT]]

        k_sigma = max(float(det.k_sigma), 1e-6)
        h_img, w_img = contrast.shape
        floor = MOMENT_FLOOR_SIGMAS * sigma
        alpha_half = ADAPTIVE_ALPHA**0.5
        # Radius over which a blob pulls its own background down (box kernel on the downscaled image).
        moat_half = max(ADAPTIVE_MIN_HALF,
                        min(ADAPTIVE_MAX_HALF, int(0.5 * max(1, int(det.bg_kernel)) * max(1, int(det.bg_downscale))) - 1))
        out: list[Detection] = []
        grow_k = np.ones((2 * GROW_RADIUS_PX + 1, 2 * GROW_RADIUS_PX + 1), np.uint8)
        for i in idx:
            lid = int(i) + 1
            if grow_low is not None and int(stats[lid, cv2.CC_STAT_AREA]) < int(det.min_area_px):
                if not self._grown_enough(labels, stats, lid, matched, grow_low, det, mask, grow_k):
                    continue
            x0 = int(stats[lid, cv2.CC_STAT_LEFT])
            y0 = int(stats[lid, cv2.CC_STAT_TOP])
            bw = int(stats[lid, cv2.CC_STAT_WIDTH])
            bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
            area = float(stats[lid, cv2.CC_STAT_AREA])
            # Window: ~3 sigma of the Gaussian weight (which is sqrt(alpha) wider than the blob,
            # sigma_blob ~ bbox/4), and never wider than the background kernel: the border ring has to
            # sit inside the moat it is there to measure.
            half = min(moat_half, max(ADAPTIVE_MIN_HALF,
                                      int(WINDOW_SIGMAS * alpha_half * max(bw, bh) / 4.0 + 0.5)))
            cxi = x0 + 0.5 * (bw - 1)
            cyi = y0 + 0.5 * (bh - 1)
            px0 = max(0, int(round(cxi)) - half)
            py0 = max(0, int(round(cyi)) - half)
            px1 = min(w_img, int(round(cxi)) + half + 1)
            py1 = min(h_img, int(round(cyi)) + half + 1)
            lab_r = labels[py0:py1, px0:px1]
            own = lab_r == lid
            c_r = contrast[py0:py1, px0:px1].astype(np.float32)
            core = c_r[own]
            if core.size == 0:  # a huge, oddly shaped component whose pixels miss the window: not a clay
                continue
            peak = float(core[int(np.argmax(np.abs(core)))])
            if matched is not None:
                if abs(peak) < float(det.min_contrast):  # DN floor on the raw residual
                    continue
                m_core = matched[py0:py1, px0:px1][own].astype(np.float32)
                m_peak = float(m_core[int(np.argmax(np.abs(m_core)))])
                peak_sign = 1.0 if m_peak >= 0.0 else -1.0
                if peak * peak_sign < 0.0:  # the raw extreme fights the matched polarity: use the matched sign
                    peak = float(core[int(np.argmax(core * peak_sign))])
                sp = 1.0 if peak >= 0.0 else -1.0
                comp = lab_r == lid
                pol = sp * c_r
                ry, rx = np.unravel_index(int(np.argmax(np.where(comp, pol, -np.inf))), pol.shape)
                # speck filter: a hot pixel holds (nearly) all of its 3x3 neighbourhood's contrast
                nb = np.maximum(pol[max(0, ry - 1):ry + 2, max(0, rx - 1):rx + 2], 0.0)
                if float(pol[ry, rx]) > float(det.max_peak_fraction) * float(nb.sum()):
                    continue
                # local-contrast test: a clay stands out from its immediate surroundings (the optics blur is
                # small); a patch of sky texture is part of a broad shallow gradient. Peak over the median of
                # the ring LOCAL_RING_PX out (beyond the clay itself) must exceed LOCAL_K_SIGMA raw sigmas.
                rr = LOCAL_RING_PX + max(int(stats[lid, cv2.CC_STAT_WIDTH]), int(stats[lid, cv2.CC_STAT_HEIGHT])) // 2
                ya, yb, xa, xb = ry - rr, ry + rr, rx - rr, rx + rr
                if ya >= 0 and xa >= 0 and yb < pol.shape[0] and xb < pol.shape[1]:
                    ring = np.concatenate((pol[ya, xa:xb + 1], pol[yb, xa:xb + 1], pol[ya + 1:yb, xa], pol[ya + 1:yb, xb]))
                    if float(pol[ry, rx]) - float(np.median(ring)) < LOCAL_K_SIGMA * sigma:
                        continue
                # Measure on the RAW footprint exactly as without the matched filter (pixels above the raw
                # threshold), so every blob the plain detector could see is measured identically; a faint blob
                # with no such pixel falls back to its pixels above MEAS_SEED_SIGMAS raw sigmas.
                raw_thr = max(k_sigma * sigma, float(det.min_contrast))
                raw_fp = comp & (pol > raw_thr)
                seed_thr = raw_thr
                if not raw_fp.any():
                    seed_thr = MEAS_SEED_SIGMAS * sigma
                    raw_fp = comp & (pol > seed_thr)
                raw_fp[ry, rx] = True
                fy_, fx_ = np.nonzero(raw_fp)
                x0, y0 = px0 + int(fx_.min()), py0 + int(fy_.min())
                bw, bh = int(fx_.max() - fx_.min()) + 1, int(fy_.max() - fy_.min()) + 1
                area = float(raw_fp.sum())
                if area > int(det.max_area_px) or max(bw, bh) / max(min(bw, bh), 1) > float(det.max_aspect):
                    continue
                gy, gx = py0 + int(ry), px0 + int(rx)
                half = min(moat_half, max(ADAPTIVE_MIN_HALF,
                                          int(WINDOW_SIGMAS * alpha_half * max(bw, bh) / 4.0 + 0.5)))
                cxi = x0 + 0.5 * (bw - 1)
                cyi = y0 + 0.5 * (bh - 1)
                px0 = max(0, int(round(cxi)) - half)
                py0 = max(0, int(round(cyi)) - half)
                px1 = min(w_img, int(round(cxi)) + half + 1)
                py1 = min(h_img, int(round(cyi)) + half + 1)
                lab_r = labels[py0:py1, px0:px1]
                c_r = contrast[py0:py1, px0:px1].astype(np.float32)
                own = (lab_r == lid) & (sp * c_r > seed_thr)
                if py0 <= gy < py1 and px0 <= gx < px1:
                    own[gy - py0, gx - px0] = True
                core = c_r[own]
            sign = np.float32(1.0 if peak >= 0.0 else -1.0)
            # UNTHRESHOLDED flux of this blob's polarity. Noise is zero-mean here, so it adds variance
            # to the moments but no bias; pixels owned by another blob (or masked out) are dropped.
            img_w = sign * c_r
            keep_px = (lab_r == lid) | (lab_r == 0) if matched is not None else own | (lab_r == 0)
            if mask is not None:
                keep_px = keep_px & mask[py0:py1, px0:px1]
            # The blob sits inside its own background estimate, so the box blur leaves a shallow moat
            # around it; a Gaussian weight integrates that moat at large radius and shrinks the moment.
            # The median of the window's border ring measures the moat (and any residual sky tilt).
            ring = np.concatenate((img_w[0, :], img_w[-1, :], img_w[1:-1, 0], img_w[1:-1, -1]))
            img_w = np.where(keep_px, img_w - np.float32(np.median(ring)), np.float32(0.0))

            # Seed: contrast-weighted centroid/second moments of the thresholded pixels.
            ys, xs = np.nonzero(own)
            wv = np.maximum(core * sign - floor, np.float32(0.0))
            wsum = float(wv.sum())
            if wsum <= 0.0:  # hue-only blob (colour OR mode): weight its pixels equally
                wv = np.ones(core.shape, np.float32)
                wsum = float(wv.size)
            fx = xs.astype(np.float32)
            fy = ys.astype(np.float32)
            cx = float((fx * wv).sum() / wsum)
            cy = float((fy * wv).sum() / wsum)
            dx0 = fx - cx
            dy0 = fy - cy
            sxx = float((wv * dx0 * dx0).sum() / wsum) + PIXEL_BOX_VAR
            syy = float((wv * dy0 * dy0).sum() / wsum) + PIXEL_BOX_VAR
            sxy = float((wv * dx0 * dy0).sum() / wsum)

            cx, cy, oxx, oxy, oyy = _adaptive_moments(img_w, cx, cy, sxx, sxy, syy)
            # Major axis of the OBSERVED (object * blur) profile; the tracker removes
            # cfg.detector.psf_px from it in quadrature to get the clay itself.
            mid = 0.5 * (oxx + oyy)
            rad = ((0.5 * (oxx - oyy)) ** 2 + oxy * oxy) ** 0.5
            lam = max(mid + rad, PIXEL_BOX_VAR)
            diameter = 4.0 * lam**0.5 / PROFILE_FACTOR
            snr = (abs(m_peak) / matched_sigma) if matched is not None else abs(peak) / sigma
            # 0.5 at the detection threshold, saturating towards 1; shrunk for non-compact blobs.
            s_snr = snr / (snr + k_sigma)
            compact = min(1.0, area / (0.25 * np.pi * diameter * diameter))
            out.append(
                Detection(
                    t_ns=int(t_ns),
                    u=cx + px0,
                    v=cy + py0,
                    area_px=area,
                    diameter_px=diameter,
                    contrast=peak,
                    snr=snr,
                    score=float(s_snr * (0.5 + 0.5 * compact)),
                    bbox=(x0, y0, bw, bh),
                )
            )
        return out


@lru_cache(maxsize=82)
def _moment_axes(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates for the small adaptive windows; shared and never modified."""
    return np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32).reshape(-1, 1)


def _adaptive_moments(img: np.ndarray, cx: float, cy: float,
                      sxx: float, sxy: float, syy: float) -> tuple[float, float, float, float, float]:
    """Adaptive (Gaussian-weighted, weight-deconvolved) centroid and second moments of `img`.

    `img` is a small background-subtracted window whose flux has already been flipped to the blob's
    polarity (noise therefore averages to zero and adds no bias). The blob is measured with a Gaussian
    weight K and the weight is divided out again: for a Gaussian object, M^-1 = O^-1 + K^-1, so
    O = (M^-1 - K^-1)^-1 (the 2-D form of sigma_o^2 = M sigma_w^2 / (sigma_w^2 - M)). Iterating the
    weight onto the object estimate converges in 2-3 passes on real blobs.

    Returns (cx, cy, oxx, oxy, oyy) in window coordinates; on a degenerate window the seed moments
    come back unchanged.
    """
    h, w = img.shape
    xs, ys = _moment_axes(h, w)
    oxx, oxy, oyy = sxx, sxy, syy
    for _ in range(ADAPTIVE_ITERS):
        kxx = max(ADAPTIVE_ALPHA * oxx, MIN_WEIGHT_VAR)
        kyy = max(ADAPTIVE_ALPHA * oyy, MIN_WEIGHT_VAR)
        kxy = ADAPTIVE_ALPHA * oxy
        det_k = kxx * kyy - kxy * kxy
        if det_k <= 0.0:
            break
        ixx, ixy, iyy = kyy / det_k, -kxy / det_k, kxx / det_k
        dx = xs - np.float32(cx)
        dy = ys - np.float32(cy)
        r2 = ixx * dx * dx + (2.0 * ixy) * dx * dy + iyy * dy * dy
        flux = np.exp(-0.5 * r2) * img
        s = float(flux.sum())
        if not s > 0.0:
            break
        mx = float((flux * dx).sum()) / s
        my = float((flux * dy).sum()) / s
        mxx = float((flux * dx * dx).sum()) / s - mx * mx
        myy = float((flux * dy * dy).sum()) / s - my * my
        mxy = float((flux * dx * dy).sum()) / s - mx * my
        cx += mx
        cy += my
        det_m = mxx * myy - mxy * mxy
        if det_m <= 0.0:
            break
        # O^-1 = M^-1 - K^-1
        pxx = myy / det_m - ixx
        pyy = mxx / det_m - iyy
        pxy = -mxy / det_m - ixy
        det_p = pxx * pyy - pxy * pxy
        if det_p <= 0.0 or pxx <= 0.0 or pyy <= 0.0:
            # The weight is narrower than the object: widen it and look again.
            oxx, oyy, oxy = oxx * 2.0, oyy * 2.0, oxy * 2.0
            continue
        nxx, nxy, nyy = pyy / det_p, -pxy / det_p, pxx / det_p
        scale = max(nxx, nyy, 1e-3)
        done = (abs(nxx - oxx) < ADAPTIVE_TOL * scale and abs(nyy - oyy) < ADAPTIVE_TOL * scale
                and abs(nxy - oxy) < ADAPTIVE_TOL * scale)
        # Undamped, the weight over/undershoots the object and the iterate oscillates, so the answer
        # would depend on the parity of the last iteration.
        oxx += ADAPTIVE_DAMPING * (nxx - oxx)
        oyy += ADAPTIVE_DAMPING * (nyy - oyy)
        oxy += ADAPTIVE_DAMPING * (nxy - oxy)
        if done:
            break
    return cx, cy, oxx, oxy, oyy


def _erfinv(y: float) -> float:
    """Inverse error function by bisection (no scipy; called once per noise refresh)."""
    lo, hi = 0.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _robust_sigma(vals: np.ndarray) -> float:
    """Gaussian sigma from the MAD of integer-valued samples, corrected for quantisation.

    The contrast is an integer number of DN, so a plain median of |x - med| can only land on whole
    numbers: at the 1-3 DN noise of this sensor that quantises sigma to 1.48 / 2.97 DN and biases the
    threshold low. Instead the empirical fraction F = P(|x - med| <= k + 0.5) is read at the integer
    bin k closest to the median and inverted against the Gaussian:
    sigma = (k + 0.5) / (sqrt(2) * erfinv(F)); for a continuous F = 0.5 this is exactly 1.4826 * MAD.
    """
    n = int(vals.size)
    med = int(np.partition(vals.reshape(-1), n // 2)[n // 2])
    dev = np.abs(vals.astype(np.int32) - med)
    np.minimum(dev, DEV_MAX, out=dev)
    counts = np.bincount(dev.reshape(-1), minlength=DEV_MAX + 1)
    cum = np.cumsum(counts) / float(n)
    k = int(np.argmin(np.abs(cum - 0.5)))
    # Away from these bounds the inversion is ill-conditioned (and a saturated frame hits them).
    frac = min(max(float(cum[k]), 0.05), 0.98)
    sigma = (k + 0.5) / (math.sqrt(2.0) * _erfinv(frac))
    return max(sigma, NOISE_SIGMA_FLOOR)
