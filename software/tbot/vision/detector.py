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
# Noise sigma is estimated on a strided subsample: this many samples is plenty for a MAD (~1.5 % error; the
# threshold is 6 sigma, so that moves it by a tenth of a sigma) and each refresh is a slow frame: two estimates
# cost ~3.4 ms on a Pi 3B at 20000 samples, a third of the frame period at 100 fps.
NOISE_TARGET_SAMPLES = 6000
# Below this many usable samples the mask is ignored for the noise estimate (the image is then almost
# entirely masked and the few remaining pixels cannot support a robust estimate).
NOISE_MIN_SAMPLES = 64
# The noise sigma is re-estimated at once (not after noise_every frames) when a frame's exposure x gain differs
# from the estimate's by more than this factor: the auto-exposure has moved, and with it the noise in DN (gain
# 1 -> 8 is 1 -> 4.5 DN on the OV9281). A stale sigma is a threshold that is wrong for up to noise_every frames.
NOISE_EXPOSURE_STEP = 1.2
# Absolute deviations are histogrammed up to this many DN (the quantile of interest sits at ~1 sigma).
DEV_MAX = 64
# The robust sigma is read at this quantile of |deviation| (0.5 = the classic MAD). Lower = still right when
# more of the frame is texture than sky: trees and roofs filling 60 % of the picture must not double the
# threshold over the sky.
NOISE_QUANTILE = 0.35
# Safety valve for pathological frames (e.g. colour OR-mode on a scene full of orange): measure at most
# this many components per frame, largest first. A real sky yields a handful.
CANDIDATE_LIMIT = 64
# Local threshold (_local_threshold): fixed-point gain of the block means; the neighbourhood (in blocks) the
# smallest block mean is taken over, which must be wider than the biggest blob to detect or the blob raises
# its own threshold (5 x 8 = 40 detection px); and the bias of that smallest-of-25 of 16-sample means (measured
# on Gaussian noise of 4-8 DN: the smallest reads 0.66 of the truth). The local threshold only ever RAISES the
# frame-wide one: at the sub-DN noise of a clear sky an integer |statistic| averages to almost nothing (0.43
# of the truth at 0.7 DN) and would put the threshold far too low.
LOCAL_GAIN = 8
LOCAL_NEIGHBOURHOOD = 5
LOCAL_MIN_BIAS = 1.5
# With a candidate_filter the survivors arrive in priority order and at most this many are measured: bounded
# frame time in a cluttered scene (each measurement is 0.6-3 ms on a Pi 3B).
FILTERED_CANDIDATE_LIMIT = 24
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

# Candidates per frame from which the rejection tests run vectorised (_prescreen, ~4.5 ms on a Pi 3B mostly
# fixed cost) instead of one by one in _measure's loop (~0.6 ms per candidate).
PRESCREEN_MIN_CANDIDATES = 8

_OFF3_Y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
_OFF3_X = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

# detector.downscale = 0 (auto): the largest of these factors that still leaves a clay at fire.max_range_m this
# many detection pixels across (the detector is tuned for 2-10 px blobs: PROFILE_FACTOR, matched_sigma_px).
AUTO_DOWNSCALES = (4, 2, 1)
AUTO_MIN_CLAY_PX = 2.5

_POLARITIES = ("bright", "dark", "both")
_MODES = ("auto", "mono", "color")
# box: mean over the kernel - what the small-blob size calibration (PROFILE_FACTOR, psf tests) was made with.
# morph: edge-preserving (see _detect) - measured at downscale 2 on 4-27 px discs: sizes +3.5 % and flatter
# than box (-6..+5 %), synthetic recall 829/829 against 817/829 (box loses the clay just above the horizon).
# auto: morph when the detector runs reduced (a long lens), box at full resolution.
_BG_MODES = ("auto", "box", "morph")
MORPH_SIZE_BIAS = 1.035
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
        self.stats: dict = {}  # last frame: connected components, candidates after the size/shape screen, measured
        # optional f(uv (N,2) full-res centroids, areas (N,)) -> indices to keep, best first (run.py sets it)
        self.candidate_filter = None
        self._labels: np.ndarray | None = None  # full-frame label image when only a sub-rectangle was labelled
        self._cells: np.ndarray | None = None   # local-noise buffers (_local_threshold)
        self._labels_roi = (0, 0, 0, 0)
        self._since_noise: int = 0
        self._noise_total: float = 0.0  # exposure_us x gain of the frame the noise sigma was estimated on
        self._key: tuple | None = None
        self._mask_src: np.ndarray | None = None
        self._mask: np.ndarray | None = None  # bool, runtime size
        self._mask_key: tuple | None = None
        self.scale: int = 1  # detection runs on the frame reduced by this factor (detector.downscale)
        self._img_scale: int = 1  # the reduction of the image _detect is working on right now
        self._morph_bg = False
        self._scale_key: tuple | None = None
        self._down: np.ndarray | None = None
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
        # 0xFF where usable: AND-ing it into a 0/1 (or 0/255) byte mask keeps the value, and is one cv2 pass
        self._mask_u8 = np.where(self._mask, np.uint8(255), np.uint8(0))
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
        self._morph = np.empty((dh, dw), np.uint8)
        self._bg = np.empty((h, w), np.uint8)
        self._contrast = np.empty((h, w), np.int16)
        self._gsm = np.empty((h, w), np.uint8)
        self._absd = np.empty((h, w), np.uint8)
        # inRange writes byte masks; bool views also work with NumPy's logical operations.
        self._bin = np.empty((h, w), np.uint8).view(np.bool_)
        self._bin2 = np.empty((h, w), np.uint8).view(np.bool_)
        self._orange = np.empty((h, w), np.uint8) if color else None
        self._gray = np.empty((h, w), np.uint8) if color else None
        self._hsv = np.empty((h, w, 3), np.uint8) if color else None
        self._key = key

    # ------------------------------------------------------------- downscale
    def _scale_for(self, w: int) -> int:
        """Reduction factor for a frame `w` pixels wide: detector.downscale, or with 0 the factor that keeps a
        far clay AUTO_MIN_CLAY_PX across (nominal focal length from [optics]; calibration changes it by a few %)."""
        want = int(getattr(self.cfg.detector, "downscale", 1) or 0)
        key = (want, w)
        if self._scale_key == key:
            return self.scale
        if want >= 1:
            scale = want
        else:
            cfg = self.cfg
            pitch_mm = float(cfg.optics.pixel_pitch_um) * 1e-3 * float(cfg.optics.sensor_width_px) / max(w, 1)
            far_px = (float(cfg.optics.focal_length_mm) / pitch_mm * float(cfg.mount.clay_diameter_m)
                      / max(float(cfg.fire.max_range_m), 1e-3))
            scale = next((f for f in AUTO_DOWNSCALES if far_px / f >= AUTO_MIN_CLAY_PX), 1)
        self._scale_key, self.scale = key, max(1, int(scale))
        return self.scale

    # ---------------------------------------------------------------- detect
    def detect(self, frame: Frame) -> list[Detection]:
        """Detections in one frame, best score first, at most cfg.detector.max_detections.

        With detector.downscale > 1 the whole detector runs on the frame averaged down by that factor (a long
        lens makes the clay 5-27 px across at 640x400: four times the pixels for the same decision, and bigger
        than the 2-10 px the measurement is tuned for). Detections come back in full-resolution pixels, and
        diameter_px has the extra blur of the s x s pixel box removed, so cfg.detector.psf_px (the tracker
        subtracts it) keeps meaning the system blur at full resolution."""
        img = frame.image
        s = self._scale_for(int(img.shape[1])) if getattr(img, "ndim", 0) in (2, 3) else 1
        if s <= 1 or img.dtype != np.uint8 or img.shape[0] < 2 * s or img.shape[1] < 2 * s:
            self._img_scale = 1
            return self._detect(frame, img)
        self._img_scale = s
        t0 = time.perf_counter()
        h, w = img.shape[0], img.shape[1]
        dh, dw = h // s, w // s
        shape = (dh, dw) + tuple(img.shape[2:])
        if self._down is None or self._down.shape != shape:
            self._down = np.empty(shape, np.uint8)
        src = img if (dh * s, dw * s) == (h, w) else img[:dh * s, :dw * s]
        if not src.flags["C_CONTIGUOUS"]:
            src = np.ascontiguousarray(src)
        cv2.resize(src, (dw, dh), dst=self._down, interpolation=cv2.INTER_AREA)
        t_down = (time.perf_counter() - t0) * 1e3
        dets = self._detect(frame, self._down)
        box4 = 16.0 * (s * s - 1) / 12.0  # (4 sigma)^2 of the s-pixel box, less that of one pixel, in full-res px^2
        off = 0.5 * (s - 1)
        for d in dets:
            d.u = d.u * s + off
            d.v = d.v * s + off
            d.area_px = d.area_px * (s * s)
            full = d.diameter_px * s
            d.diameter_px = max(full * full - box4, 4.0 * PIXEL_BOX_VAR) ** 0.5
            x, y, bw, bh = d.bbox
            d.bbox = (x * s, y * s, bw * s, bh * s)
        if self.debug:
            self.debug = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST) for k, v in self.debug.items()}
        self.timings_ms["downscale"] = t_down
        self.timings_ms["total"] = (time.perf_counter() - t0) * 1e3
        return dets

    def _detect(self, frame: Frame, img: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        det = self.cfg.detector
        if det.mode not in _MODES:
            raise ValueError(f"detector.mode must be one of {_MODES}, got {det.mode!r}")
        if det.polarity not in _POLARITIES:
            raise ValueError(f"detector.polarity must be one of {_POLARITIES}, got {det.polarity!r}")
        if str(getattr(det, "bg_mode", "box")) not in _BG_MODES:
            raise ValueError(f"detector.bg_mode must be one of {_BG_MODES}, got {det.bg_mode!r}")
        if det.color_combine not in _COMBINES:
            raise ValueError(f"detector.color_combine must be one of {_COMBINES}, got {det.color_combine!r}")

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
        # bg_downscale is in full-resolution pixels: on a frame already reduced by `scale` the background
        # image is reduced by the rest, so the kernel covers the same patch of sky either way
        down = max(1, int(det.bg_downscale) // max(1, self._img_scale))
        self._ensure_buffers(h, w, down, is_color)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dst=self._gray) if is_color else img

        # ---- background and signed contrast -------------------------------
        k = max(1, int(det.bg_kernel))
        cv2.resize(gray, (self._small.shape[1], self._small.shape[0]), dst=self._small,
                   interpolation=cv2.INTER_AREA)
        mode = str(getattr(det, "bg_mode", "box"))
        # auto: each background with the scale it was validated at (see _BG_MODES)
        self._morph_bg = mode == "morph" or (mode == "auto" and self._img_scale >= 2)
        if self._morph_bg:
            # Edge-preserving background: closing removes dark things narrower than the kernel, opening bright
            # ones, and a step edge (horizon, tree line, roof) passes through both unchanged, so one after the
            # other leaves the scene without its small structures. A clay keeps its full contrast right next
            # to an edge, where a box blur mixes sky and ground over the kernel width and buries it, and it
            # sits in no moat of its own making.
            # The order matters next to an edge. Closing first removes a DARK clay before the opening can
            # merge it with dark ground a few pixels below (the strip of sky between them is narrower than the
            # kernel): that is the clay over the tree line, so it is the order for dark and both polarities.
            # Opening first is the mirror image, for a bright clay beside something bright.
            se = _rect_kernel(k)
            first, second = ((cv2.MORPH_OPEN, cv2.MORPH_CLOSE) if det.polarity == "bright"
                             else (cv2.MORPH_CLOSE, cv2.MORPH_OPEN))
            cv2.morphologyEx(self._small, first, se, dst=self._morph)
            cv2.morphologyEx(self._morph, second, se, dst=self._smooth)
        else:
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
            # gsm - bg in int16 is only read in small windows (and a noise subsample): the full-frame threshold
            # is taken on uint8 below, exactly (a full int16 subtract + two inRange cost ~3.5 ms on a Pi 3B)
            stat = _LazyContrast(self._gsm, self._bg)
            unit = 1.0
        else:
            stat = contrast
            unit = 1.0
        mask = self._mask_for(h, w)
        total = float(frame.exposure_us) * float(frame.gain)  # 0 for sources that do not report exposure
        moved = total > 0.0 and self._noise_total > 0.0 and not (
            1.0 / NOISE_EXPOSURE_STEP < total / self._noise_total < NOISE_EXPOSURE_STEP)
        if self._since_noise <= 0 or moved or not np.isfinite(self.noise_sigma):
            # (both estimates on one frame: ~3.4 ms on a Pi 3B every noise_every frames)
            self._noise_total = total
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
        if isinstance(stat, _LazyContrast) and math.isfinite(thr):
            # stat = gsm - bg is an integer in [-255, 255], so stat > thr <=> stat > floor(thr), and
            # |stat| = absdiff(gsm, bg); the saturating uint8 subtract gives max(stat, 0) / max(-stat, 0).
            t_int = int(min(255, max(-1, math.floor(thr))))
            if det.polarity == "both":
                cv2.absdiff(self._gsm, self._bg, dst=self._absd)
            elif det.polarity == "bright":
                cv2.subtract(self._gsm, self._bg, dst=self._absd)
            else:
                cv2.subtract(self._bg, self._gsm, dst=self._absd)
            cell = int(getattr(det, "local_noise_cell", 0) or 0)
            if cell > 0 and w >= 4 * cell and h >= 4 * cell:
                self._local_threshold(det, t_int, cell, w, h, binary.view(np.uint8))
            else:
                cv2.threshold(self._absd, t_int, 1, cv2.THRESH_BINARY, dst=binary.view(np.uint8))
        elif math.isfinite(thr):
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
            cv2.bitwise_and(binary.view(np.uint8), self._mask_u8, dst=binary.view(np.uint8))
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
                debug["matched"] = stat.full() if isinstance(stat, _LazyContrast) else stat.copy()
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

    def _local_threshold(self, det, t_global: int, cell: int, w: int, h: int, out: np.ndarray) -> None:
        """Threshold |matched statistic| against the LOCAL noise where that is higher than the frame's.

        A frame is rarely one kind of scene: quiet sky above, a busy tree line below. The frame-wide sigma
        (a low quantile, so it follows the quiet part) is right for the sky and makes every leaf gap in the
        trees a candidate. Here the mean |statistic| of each `cell` x `cell` block gives a local scale, each
        block takes the SMALLEST of its neighbourhood, and the threshold is the larger of the frame-wide one
        and k_sigma x that. Smallest-of means a clay cannot raise its own threshold (it covers a couple of
        blocks at most), and a clay just above the tree line is judged by the sky next to it, not the trees."""
        cw, ch = max(1, w // cell), max(1, h // cell)
        if self._cells is None or self._cells.shape != (ch, cw) or self._thrmap.shape != (h, w):
            self._cells = np.empty((ch, cw), np.uint8)
            self._cells2 = np.empty((ch, cw), np.uint8)
            self._thrmap = np.empty((h, w), np.uint8)
            self._abs8 = np.empty((h // 2, w // 2), np.uint8)
        # Block statistics from every other pixel (a quarter of the work; 16 samples a block is enough for a
        # scale that only matters where it is several times the frame's), as 8 x |stat| saturating at 255:
        # 1/8 DN resolution for the block means, and nothing above 32 DN (a clay, an edge) counts for more.
        cv2.resize(self._absd, (w // 2, h // 2), dst=self._abs8, interpolation=cv2.INTER_NEAREST)
        cv2.convertScaleAbs(self._abs8, dst=self._abs8, alpha=float(LOCAL_GAIN))
        cv2.resize(self._abs8, (cw, ch), dst=self._cells, interpolation=cv2.INTER_AREA)
        cv2.erode(self._cells, _rect_kernel(LOCAL_NEIGHBOURHOOD), dst=self._cells2)
        cv2.blur(self._cells2, (3, 3), dst=self._cells)
        # mean |x| -> sigma: 1.2533 (two-sided) or 2.5066 (one polarity: the other half was clipped to 0),
        # times LOCAL_MIN_BIAS for taking the smallest block mean of the neighbourhood; then the threshold in
        # DN per block, never under the frame-wide one, and only then up to the image size
        to_sigma = (1.2533 if det.polarity == "both" else 2.5066) * LOCAL_MIN_BIAS / LOCAL_GAIN
        cv2.convertScaleAbs(self._cells, dst=self._cells, alpha=float(det.k_sigma) * to_sigma)
        cv2.max(self._cells, max(0, int(t_global)), dst=self._cells)
        cv2.resize(self._cells, (w, h), dst=self._thrmap, interpolation=cv2.INTER_LINEAR)
        cv2.compare(self._absd, self._thrmap, cv2.CMP_GT, dst=out)

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
        # Label only the bounding box of the pixels that passed the threshold: a clear sky has none (most
        # frames), a lone clay a few dozen, and labelling the whole frame costs ~1 ms on a Pi 3B either way.
        b8 = binary.view(np.uint8)
        rx, ry, rw, rh = cv2.boundingRect(b8)
        self.stats = {"components": 0, "candidates": 0}
        if rw == 0 or rh == 0:
            return []
        if rw * rh * 2 > b8.size:
            n, labels, stats, cents = cv2.connectedComponentsWithStats(b8, connectivity=8)
        else:
            n, sub, stats, cents = cv2.connectedComponentsWithStats(
                np.ascontiguousarray(b8[ry:ry + rh, rx:rx + rw]), connectivity=8)
            cents += (rx, ry)
            labels = self._labels
            if labels is None or labels.shape != b8.shape:
                labels = self._labels = np.zeros(b8.shape, np.int32)
            else:
                x0, y0, w0, h0 = self._labels_roi
                labels[y0:y0 + h0, x0:x0 + w0] = 0
            labels[ry:ry + rh, rx:rx + rw] = sub
            self._labels_roi = (rx, ry, rw, rh)
            stats[:, cv2.CC_STAT_LEFT] += rx
            stats[:, cv2.CC_STAT_TOP] += ry
        self.stats["components"] = int(n) - 1
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
        self.stats["candidates"] = int(idx.size)
        ordered = False
        if self.candidate_filter is not None and idx.size:
            # e.g. the scenery filter (vision/clutter.py): drops candidates before they cost a measurement and
            # says which of the rest matter most. Centroids go out in full-resolution pixels.
            s = self._img_scale
            order = self.candidate_filter(cents[idx + 1] * s + 0.5 * (s - 1), areas[idx] * (s * s),
                                          ws[idx] * s, hs[idx] * s)
            idx = idx[np.asarray(order, dtype=np.intp)]
            ordered = True
            self.stats["kept"] = int(idx.size)
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
        if ordered:
            idx = idx[:FILTERED_CANDIDATE_LIMIT]  # already in priority order
        elif idx.size > CANDIDATE_LIMIT:  # keep the largest; a real frame never gets here
            idx = idx[np.argsort(areas[idx])[::-1][:CANDIDATE_LIMIT]]

        k_sigma = max(float(det.k_sigma), 1e-6)
        h_img, w_img = contrast.shape
        floor = MOMENT_FLOOR_SIGMAS * sigma
        alpha_half = ADAPTIVE_ALPHA**0.5
        # Radius over which a blob pulls its own background down (box kernel on the downscaled image).
        moat_half = max(ADAPTIVE_MIN_HALF,
                        min(ADAPTIVE_MAX_HALF, int(0.5 * max(1, int(det.bg_kernel))
                                                   * max(1, int(det.bg_downscale) // max(1, self._img_scale))) - 1))
        out: list[Detection] = []
        grow_k = np.ones((2 * GROW_RADIUS_PX + 1, 2 * GROW_RADIUS_PX + 1), np.uint8)
        grown_done = False
        if (idx.size >= PRESCREEN_MIN_CANDIDATES and matched is not None and isinstance(contrast, _LazyContrast)
                and isinstance(matched, _LazyContrast) and int(det.min_area_px) <= 2):
            # all of the loop's rejection tests at once (a real sky gives tens of candidates per frame and the
            # per-candidate loop costs ~0.6 ms each on a Pi 3B); the loop then only measures the survivors
            idx = _prescreen(idx, labels, stats, binary, contrast, matched, sigma, det, mask, grow_low,
                             moat_half, alpha_half)
            grown_done = True
        for i in idx:
            lid = int(i) + 1
            if (not grown_done and grow_low is not None
                    and int(stats[lid, cv2.CC_STAT_AREA]) < int(det.min_area_px)):
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
            c_r = _window_f32(contrast, py0, py1, px0, px1)
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
                    if float(pol[ry, rx]) - _median(ring) < LOCAL_K_SIGMA * sigma:
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
                c_r = _window_f32(contrast, py0, py1, px0, px1)
                own = (lab_r == lid) & (sp * c_r > seed_thr)
                if py0 <= gy < py1 and px0 <= gx < px1:
                    own[gy - py0, gx - px0] = True
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
            img_w = np.where(keep_px, img_w - np.float32(_median(ring)), np.float32(0.0))

            # Seed: contrast-weighted centroid/second moments of the thresholded pixels (one cv2.moments call:
            # double accumulation, and ~25 fewer small NumPy calls than the explicit sums).
            wimg = np.where(own, np.maximum(sign * c_r - floor, np.float32(0.0)), np.float32(0.0))
            mo = cv2.moments(wimg)
            if not mo["m00"] > 0.0:  # hue-only blob (colour OR mode): weight its pixels equally
                mo = cv2.moments(own.astype(np.float32))
            wsum = mo["m00"]
            cx = mo["m10"] / wsum
            cy = mo["m01"] / wsum
            sxx = mo["mu20"] / wsum + PIXEL_BOX_VAR
            syy = mo["mu02"] / wsum + PIXEL_BOX_VAR
            sxy = mo["mu11"] / wsum

            cx, cy, oxx, oxy, oyy = _adaptive_moments(img_w, cx, cy, sxx, sxy, syy)
            # Major axis of the OBSERVED (object * blur) profile; the tracker removes
            # cfg.detector.psf_px from it in quadrature to get the clay itself.
            mid = 0.5 * (oxx + oyy)
            rad = ((0.5 * (oxx - oyy)) ** 2 + oxy * oxy) ** 0.5
            lam = max(mid + rad, PIXEL_BOX_VAR)
            diameter = 4.0 * lam**0.5 / (PROFILE_FACTOR * MORPH_SIZE_BIAS if self._morph_bg else PROFILE_FACTOR)
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


@lru_cache(maxsize=8)
def _rect_kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


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
        dx = xs - np.float32(cx)                     # (w,)
        dy = ys - np.float32(cy)                     # (h, 1)
        # -0.5 r2 built from 1-D pieces (each NumPy call on a small window costs ~20 us on a Pi 3B)
        e = (np.float32(-ixy) * dy) * dx
        e += np.float32(-0.5 * ixx) * dx * dx
        e += np.float32(-0.5 * iyy) * dy * dy
        flux = np.exp(e, out=e)
        flux *= img
        mo = cv2.moments(flux)                       # raw + central moments in one call (double sums)
        s = mo["m00"]
        if not s > 0.0:
            break
        mx = mo["m10"] / s - cx
        my = mo["m01"] / s - cy
        mxx = mo["mu20"] / s
        myy = mo["mu02"] / s
        mxy = mo["mu11"] / s
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


_OFF5_Y, _OFF5_X = (a.reshape(-1) for a in np.mgrid[-GROW_RADIUS_PX:GROW_RADIUS_PX + 1,
                                                     -GROW_RADIUS_PX:GROW_RADIUS_PX + 1])
_OFF5_NOT_C = ~((_OFF5_Y == 0) & (_OFF5_X == 0))


@lru_cache(maxsize=32)
def _ring_offsets(rr: int) -> tuple[np.ndarray, np.ndarray]:
    """(dy, dx) of the square ring at Chebyshev distance rr, in _measure's ring order."""
    a = np.arange(-rr, rr + 1)
    m = np.arange(-rr + 1, rr)
    dy = np.concatenate((np.full(a.size, -rr), np.full(a.size, rr), m, m))
    dx = np.concatenate((a, a, np.full(m.size, -rr), np.full(m.size, rr)))
    return dy, dx


def _prescreen(idx, labels, stats, binary, contrast, matched, sigma, det, mask, grow_low, moat_half, alpha_half):
    """Vectorised form of _measure's per-candidate rejection tests in matched mode: hysteresis growth of a
    1-pixel seed (_grown_enough with min_area_px <= 2), the raw DN floor, the matched polarity, the speck
    filter and the local-contrast ring test. Returns the surviving candidate indices, in input order; the
    loop in _measure repeats the tests on them (and gets the same answers)."""
    if idx.size == 0:
        return idx
    h_img, w_img = labels.shape
    gray, bg, gsm = contrast.gray, contrast.bg, matched.gray
    lids = idx + 1
    area = stats[lids, cv2.CC_STAT_AREA]
    alive = np.ones(idx.size, dtype=bool)

    # ---- growth of 1-pixel seeds: a same-polarity matched pixel above grow_low within GROW_RADIUS_PX
    if grow_low is not None:
        seed = np.flatnonzero(area < int(det.min_area_px))
        if seed.size:
            sy = stats[lids[seed], cv2.CC_STAT_TOP].astype(np.int64)
            sx = stats[lids[seed], cv2.CC_STAT_LEFT].astype(np.int64)
            s0 = gsm[sy, sx].astype(np.int32) - bg[sy, sx].astype(np.int32)
            sgn = np.where(s0 >= 0, 1, -1)
            yy = sy[:, None] + _OFF5_Y[None, :]
            xx = sx[:, None] + _OFF5_X[None, :]
            inside = (yy >= 0) & (yy < h_img) & (xx >= 0) & (xx < w_img) & _OFF5_NOT_C[None, :]
            yc, xc = np.clip(yy, 0, h_img - 1), np.clip(xx, 0, w_img - 1)
            st = gsm[yc, xc].astype(np.int32) - bg[yc, xc].astype(np.int32)
            weak = inside & ((st * sgn[:, None]) > grow_low)
            if mask is not None:
                weak &= mask[yc, xc]
            alive[seed[~weak.any(axis=1)]] = False

    # ---- the candidate's measurement window (as in _measure)
    x0 = stats[lids, cv2.CC_STAT_LEFT].astype(np.int64)
    y0 = stats[lids, cv2.CC_STAT_TOP].astype(np.int64)
    bw = stats[lids, cv2.CC_STAT_WIDTH].astype(np.int64)
    bh = stats[lids, cv2.CC_STAT_HEIGHT].astype(np.int64)
    half = np.minimum(moat_half, np.maximum(ADAPTIVE_MIN_HALF, np.floor(
        WINDOW_SIGMAS * alpha_half * np.maximum(bw, bh) / 4.0 + 0.5).astype(np.int64)))
    rcx = np.round(x0 + 0.5 * (bw - 1)).astype(np.int64)  # np.round = Python round (half to even)
    rcy = np.round(y0 + 0.5 * (bh - 1)).astype(np.int64)
    px0, py0 = np.maximum(0, rcx - half), np.maximum(0, rcy - half)
    px1, py1 = np.minimum(w_img, rcx + half + 1), np.minimum(h_img, rcy + half + 1)

    # ---- per window size, all candidates' windows as one (K, n, n) stack; padding outside the image is
    # marked invalid, which is exactly the clipping of _measure's window. Flattened argmax over a stack row
    # visits the valid pixels in the window's row-major order, so ties resolve as in the loop.
    keep = np.zeros(idx.size, dtype=bool)
    big = np.int32(1 << 20)
    for hw in np.unique(half[alive]):
        g = np.flatnonzero(alive & (half == hw))
        n = 2 * int(hw) + 1
        ar = np.arange(n)
        Y = (rcy[g] - hw)[:, None, None] + ar[None, :, None]
        X = (rcx[g] - hw)[:, None, None] + ar[None, None, :]
        valid = (Y >= 0) & (Y < h_img) & (X >= 0) & (X < w_img)
        Yc, Xc = np.clip(Y, 0, h_img - 1), np.clip(X, 0, w_img - 1)
        bgw = bg[Yc, Xc].astype(np.int32)
        c = gray[Yc, Xc].astype(np.int32) - bgw
        own = valid & (labels[Yc, Xc] == lids[g][:, None, None])
        K = g.size
        cf = c.reshape(K, -1)
        of = own.reshape(K, -1)
        rows = np.arange(K)

        def first_max(key):
            return np.argmax(np.where(of, key, -big), axis=1)

        present = of.any(axis=1)                                   # core.size == 0 -> rejected
        peak = cf[rows, first_max(np.abs(cf))]
        floor_ok = np.abs(peak) >= float(det.min_contrast)
        mf = (gsm[Yc, Xc].astype(np.int32) - bgw).reshape(K, -1)
        m_peak = mf[rows, first_max(np.abs(mf))]
        peak_sign = np.where(m_peak >= 0, 1, -1)
        alt = cf[rows, first_max(cf * peak_sign[:, None])]
        peak = np.where(peak * peak_sign < 0, alt, peak)
        sp = np.where(peak >= 0, 1, -1)
        pol = cf * sp[:, None]
        j = first_max(pol)
        ppk = pol[rows, j].astype(np.float64)
        jy, jx = j // n, j % n                                     # peak in stack coordinates

        # speck filter: 3x3 around the peak, clipped to the window
        yy = jy[:, None] + _OFF3_Y[None, :]
        xx = jx[:, None] + _OFF3_X[None, :]
        inb = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
        fl = np.clip(yy, 0, n - 1) * n + np.clip(xx, 0, n - 1)
        vf = valid.reshape(K, -1)
        nb_ok = inb & vf[rows[:, None], fl]
        nb_sum = np.where(nb_ok, np.maximum(pol[rows[:, None], fl], 0), 0).sum(axis=1).astype(np.float64)
        speck_ok = ~(ppk > float(det.max_peak_fraction) * nb_sum)

        # local-contrast ring test, where the ring fits in the (clipped) window
        ry, rx = rcy[g] - hw + jy, rcx[g] - hw + jx               # image coordinates
        rr = LOCAL_RING_PX + np.maximum(bw[g], bh[g]) // 2
        fits = (ry - rr >= py0[g]) & (rx - rr >= px0[g]) & (ry + rr < py1[g]) & (rx + rr < px1[g])
        ring_ok = np.ones(K, dtype=bool)
        for r in np.unique(rr[fits]):
            k = np.flatnonzero(fits & (rr == r))
            dy, dx = _ring_offsets(int(r))
            ya, xa = ry[k, None] + dy[None, :], rx[k, None] + dx[None, :]
            ring = np.sort(sp[k, None] * (gray[ya, xa].astype(np.int32) - bg[ya, xa].astype(np.int32)), axis=1)
            lo, hi = (ring.shape[1] - 1) // 2, ring.shape[1] // 2
            med = (ring[:, lo] + ring[:, hi]).astype(np.float32) / np.float32(2)   # np.median of the float32 ring
            ring_ok[k] = ~(ppk[k] - med.astype(np.float64) < LOCAL_K_SIGMA * sigma)

        keep[g] = present & floor_ok & speck_ok & ring_ok
    return idx[keep & alive]

def _window_f32(contrast, py0: int, py1: int, px0: int, px1: int) -> np.ndarray:
    """contrast[py0:py1, px0:px1] as float32; a _LazyContrast window costs one cv2.subtract instead of three
    NumPy calls."""
    if isinstance(contrast, _LazyContrast):
        return cv2.subtract(contrast.gray[py0:py1, px0:px1], contrast.bg[py0:py1, px0:px1], dtype=cv2.CV_32F)
    return contrast[py0:py1, px0:px1].astype(np.float32)


def _median(a: np.ndarray) -> float:
    """np.median of a small 1-D array (same value: mean of the two middle elements), without np.median's
    Python-level overhead (~80 us per call on a Pi 3B)."""
    n = int(a.size)
    lo, hi = (n - 1) // 2, n // 2
    p = np.partition(a, (lo, hi))
    return float(p[lo]) if lo == hi else float((p[lo] + p[hi]) / a.dtype.type(2))


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
    k = int(np.argmin(np.abs(cum - NOISE_QUANTILE)))
    # Away from these bounds the inversion is ill-conditioned (and a saturated frame hits them).
    frac = min(max(float(cum[k]), 0.05), 0.98)
    sigma = (k + 0.5) / (math.sqrt(2.0) * _erfinv(frac))
    return max(sigma, NOISE_SIGMA_FLOOR)
