"""Motion-first detection (`vision.engine = "motion"`): what moves against the world, not what has contrast.

The single-frame detector finds everything with local contrast and leaves it to later stages to tell a clay from
a crack in the concrete or a speck of foliage. Here the scenery is cancelled at PIXEL level instead: two earlier
frames (A ~ vision.lag_a_ms ago, B ~ vision.lag_b_ms ago) are registered onto the current one (gyro prior, then
an integer SAD search on low-resolution pyramids), and a pixel is a candidate only where the current frame leaves
"what the scenery could look like here" (the min..max envelope of each registered reference over a 3 x 3
neighbourhood) in BOTH references. A clay differs from both; its old position differs from one only (no ghost);
a leaf, a crack, a cloud edge, registration error of a pixel or so, differ from neither.

Pipeline per frame (all at the working resolution = 1/vision.scale of the frame, scale 2 by default):

  prep       2x2 box mean to the working image, box pyramid to 1/reg_downscale and 1/(2 reg_downscale) of
             the full frame (box_down2)
  register   for A and B: prior = the homography of the relative camera rotation (IMU attitude through the
             camera model, plus a running correction learnt from earlier registrations: gyro bias), or the last
             frame-to-frame motion without an IMU; coarse SAD search at 1/(2 reg_downscale) (+-ceil(reg_search_px
             / coarse) px, +-8 without IMU), fine SAD search +-1 around it at 1/reg_downscale, parabolic
             sub-pixel. Masked pixels never take part, and the result counts (reg_ok) only when the unmasked
             overlap holds enough textured AREA (TEXTURE_FRAC: a clay or a ball never does) and the SAD surface
             has a clear minimum; otherwise the prior stands. Then a coarse residual field (FIELD_GRID cells,
             +-FIELD_R fine px each, same texture guard, 3x3 median) for the parallax of near scenery. Photometric
             gain and offset (cur ~ a ref + b) by matching percentiles on the fine level.
  kernel     motion_kernel(): block-integer warp of both references (warp_gather; offsets = global homography +
             field), fixed-point gain (apply_gain_q8), envelope residual per polarity combined over A and B
             (envelope_residual), cell sums of the residual and of |cur - warped A| (cell_sums,
             cell_absdiff_sums), noise sigma and CFAR threshold per cell (noise_sigma_from_cells,
             threshold_cells: Python, one value per cell), compare (compare_cells)
  components 8-connected components of the thresholded residual (1-px gaps closed) inside its bounding ROI
  measure    integer moment sums per component -> centroid, size, mass; strongest max_candidates kept

The functions marked "C port" below are pure functions on arrays with integer semantics; they are the reference
a C kernel must reproduce bit for bit. OpenCV is used inside them only where its result is exactly that integer
definition (3x3 min/max, saturating subtract, min/max, AND, nearest block expansion); this is checked in
tests/test_motion_detector.py.

Coordinates: Detection.(u, v), bbox, area_px and diameter_px are in FULL-resolution pixels; stab is in
full-resolution "background-stabilised" pixels (S_k maps frame-k pixels to a plane in which static scenery stands
still: S_k = S_A o (cur -> A global registration, homography); the local field is not in it). min_area_px / max_area_px, envelope_px and cfar_cell_px are in
WORKING pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tbot.types import Detection, Frame

# ------------------------------------------------------------------------------------------------ constants
WARP_BLOCK = 16          # block-integer warp: one integer source offset per (row, 16-column block)
RING = 40                # frames kept for references (400 ms at 100 frames/s: the long lags of open sky)
SRC_MARGIN = 3           # a warped pixel whose source lies this close to the frame edge / mask is invalid
BLACK_DN = 4.0           # sensor black level (gain fallback on textureless pairs)
GAIN_MIN, GAIN_MAX = 0.25, 4.0
NOIMU_COARSE_R = 8       # coarse SAD window without IMU (coarse px)
FINE_R = 1               # fine SAD window around the coarse result
FINE_RECENTRE = 2        # times the fine window may move when its minimum lies on its edge
MIN_OVERLAP_FRAC = 0.25  # registration needs at least this much of the low-res frame in the overlap
FIT_STD_MIN = 2.0        # reference spread (DN, as a std) below which the photometric fit is a ratio of means
NOISE_ROW_STEP = 2       # noise estimate: every 2nd row
NOISE_QUANTILE = 0.10    # ... of the cell means of |cur - warped A|, the quietest tenth
NOISE_MIN_FILL = 0.75    # a cell counts for the noise estimate when this much of it is valid
ABS_TO_SIGMA = 1.0 / (2.0 / math.sqrt(math.pi))  # mean |x - y| of two N(0, s) = 2 s / sqrt(pi) -> s
CFAR_BOX = 5             # threshold smoothing over 5 x 5 cells
SCORE_MASS = 400.0       # score = mass / (mass + SCORE_MASS), full-resolution DN px
PIXEL_BOX_VAR = 1.0 / 12.0
CORR_GAIN = 0.3         # running correction of the IMU prior: gain per registered frame ...
CORR_MAX = 300.0         # ... and its limit, px/s
TEXTURE_DN = 4           # texture guard: a fine-level pixel is textured when a neighbour differs by more than this
TEXTURE_FRAC = 0.03      # global registration counts only with this fraction of the overlap textured
SKY_LAG_GAIN = 2.5       # where the scene cannot hold a registration (open sky: nothing to cancel, so nothing that a
                         # longer lag could leave behind) both lags are this much longer: a clay going away at
                         # 3 deg/s moves 4 px in 40 ms, less than its own size, and cancels itself
SLOW_CLEAR_PX = 4        # the slow-mover test adds nothing within this of an A/B candidate (working px)
LAG_SHIFT_FRAC = 0.15    # the reference lags shrink so the scenery moved at most this part of the frame width since B
TENSOR_MIN = 3.0        # ... and in full only when the weaker eigenvalue of the textured pixels' gradient tensor is
                         # at least this, DN^2 per overlap pixel (fine level); the stronger one alone: across the edge
CELL_TEXTURE_FRAC = 0.05  # ... and a cell of the local field with this fraction of the cell
FIELD_GRID = (3, 4)      # local residual field: rows x columns of cells over the fine level
FIELD_R = 2              # its SAD window, fine px around the global solution


# ======================================================================== pure integer functions (C port)
def box_down2(src: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
    """C port. uint8 (H, W) -> uint8 (H//2, W//2): dst[y, x] = (s00 + s01 + s10 + s11 + 2) >> 2 of the 2x2
    block at (2y, 2x) (round half up). Odd last row/column dropped."""
    h, w = src.shape[0] // 2, src.shape[1] // 2
    s = src[:2 * h, :2 * w].astype(np.uint16)
    out = ((s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2] + 2) >> 2).astype(np.uint8)
    if dst is None:
        return out
    dst[...] = out
    return dst


def min_down2(src: np.ndarray) -> np.ndarray:
    """uint8 (H, W) -> (H//2, W//2): min of each 2x2 block (masks: a low-res pixel is usable only if all its
    source pixels are). Set-up only (once per mask)."""
    h, w = src.shape[0] // 2, src.shape[1] // 2
    s = src[:2 * h, :2 * w]
    return np.minimum(np.minimum(s[0::2, 0::2], s[0::2, 1::2]), np.minimum(s[1::2, 0::2], s[1::2, 1::2]))


def block_centres(h: int, w: int, block: int = WARP_BLOCK) -> np.ndarray:
    """x of the centre of each warp block (float64 (nb,)): the middle of its columns."""
    nb = (w + block - 1) // block
    x0 = np.arange(nb) * block
    return (x0 + np.minimum(x0 + block, w) - 1) * 0.5


def block_offsets(minv: np.ndarray, h: int, w: int, block: int = WARP_BLOCK, extra=None, full: bool = False):
    """Integer source offsets of the block-integer warp (Python, floats; the C side receives the tables).

    minv: (2, 3) or (3, 3) float64 affine, destination pixel -> source pixel (same image size, pixel centres at
    integers). For row y and block b (columns b*block .. min(w, (b+1)*block) - 1, centre xc = middle of those
    columns): (sx, sy) = minv @ (xc, y, 1) (+ extra[0][y, b], extra[1][y, b]: the local residual field);
    dx[y, b] = floor(sx - xc + 0.5), dy[y, b] = floor(sy - y + 0.5).
    Returns dx, dy (int32 (h, nb)) and the mean rounding residual (float64 (2,): mean over all (y, b) of the
    exact offset minus the integer one); with full=True also the residual arrays (rx, ry) float64 (h, nb)."""
    xc = block_centres(h, w, block)
    ys = np.arange(h, dtype=np.float64)
    m = np.asarray(minv, dtype=np.float64)
    if m.shape[0] == 3:
        den = m[2, 0] * xc[None, :] + m[2, 1] * ys[:, None] + m[2, 2]
    else:
        den = 1.0
    ex = (m[0, 0] * xc[None, :] + m[0, 1] * ys[:, None] + m[0, 2]) / den - xc[None, :]
    ey = (m[1, 0] * xc[None, :] + m[1, 1] * ys[:, None] + m[1, 2]) / den - ys[:, None]
    if extra is not None:
        ex = ex + extra[0]
        ey = ey + extra[1]
    dx = np.floor(ex + 0.5)
    dy = np.floor(ey + 0.5)
    rx, ry = ex - dx, ey - dy
    res = np.array([float(rx.mean()), float(ry.mean())])
    if full:
        return dx.astype(np.int32), dy.astype(np.int32), res, (rx, ry)
    return dx.astype(np.int32), dy.astype(np.int32), res


def texture_map(img: np.ndarray, thr: int, dst: np.ndarray | None = None) -> np.ndarray:
    """C port. uint8 (h, w) -> uint8 0 / 255: 255 where max(|img[y, x+1] - img[y, x]|, |img[y+1, x] - img[y, x]|)
    > thr; the last row and column are 0."""
    h, w = img.shape
    out = np.zeros((h, w), np.uint8) if dst is None else dst
    out[...] = 0
    a = img.astype(np.int16)
    g = np.maximum(np.abs(a[:-1, 1:] - a[:-1, :-1]), np.abs(a[1:, :-1] - a[:-1, :-1]))
    out[:-1, :-1] = np.where(g > int(thr), 255, 0)
    return out


def texture_tensor(img: np.ndarray, tex: np.ndarray, valid: np.ndarray) -> tuple[int, int, int]:
    """C port. Gradient structure tensor of the textured usable pixels: the integer sums (sum dx*dx, sum dy*dy,
    sum dx*dy) over the pixels with tex != 0 and valid != 0, with texture_map's forward differences
    dx = img[y, x+1] - img[y, x], dy = img[y+1, x] - img[y, x] (the last row and column never count: tex is 0
    there). Its eigenvalues say whether the texture fixes a shift in both directions or across an edge only."""
    a = img.astype(np.int64)
    m = (tex[:-1, :-1] != 0) & (valid[:-1, :-1] != 0)
    dx = (a[:-1, 1:] - a[:-1, :-1])[m]
    dy = (a[1:, :-1] - a[:-1, :-1])[m]
    return int((dx * dx).sum()), int((dy * dy).sum()), int((dx * dy).sum())


class _GatherIdx:
    """Cached grids and map buffers of warp_gather at one size (implementation detail of the NumPy path)."""

    def __init__(self, h: int, w: int, block: int) -> None:
        self.h, self.w, self.block = h, w, block
        self.nb = (w + block - 1) // block
        self.X = np.ascontiguousarray(np.broadcast_to(np.arange(w, dtype=np.int16)[None, :], (h, w)))
        self.Y = np.ascontiguousarray(np.broadcast_to(np.arange(h, dtype=np.int16)[:, None], (h, w)))
        self.ex = np.empty((h, self.nb * block), np.int16)
        self.map = np.empty((h, w, 2), np.int16)


def warp_gather(src: np.ndarray, src_valid: np.ndarray, dx: np.ndarray, dy: np.ndarray, block: int,
                dst: np.ndarray, dst_valid: np.ndarray, _g: _GatherIdx | None = None) -> None:
    """C port. Block-integer warp. src, src_valid, dst, dst_valid: uint8 (H, W); dx, dy: int32 (H, nb).
    For every destination pixel (y, x), b = x // block: sy = y + dy[y, b], sx = x + dx[y, b];
        inside the frame:  dst[y, x] = src[sy, sx], dst_valid[y, x] = src_valid[sy, sx]
        outside:           dst[y, x] = 0,           dst_valid[y, x] = 0
    (src_valid is also 0 within SRC_MARGIN of the frame edge and of the mask.) Here: cv2.remap, INTER_NEAREST,
    integer map, BORDER_CONSTANT 0 - an exact gather (warp_gather_ref is the plain definition)."""
    h, w = src.shape
    g = _g if _g is not None else _GatherIdx(h, w, block)
    m = g.map
    cv2.resize(dx.astype(np.int16), (g.nb * block, h), dst=g.ex, interpolation=cv2.INTER_NEAREST)
    np.add(g.X, g.ex[:, :w], out=m[:, :, 0])
    cv2.resize(dy.astype(np.int16), (g.nb * block, h), dst=g.ex, interpolation=cv2.INTER_NEAREST)
    np.add(g.Y, g.ex[:, :w], out=m[:, :, 1])
    cv2.remap(src, m, None, cv2.INTER_NEAREST, dst=dst, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.remap(src_valid, m, None, cv2.INTER_NEAREST, dst=dst_valid, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_gather_ref(src, src_valid, dx, dy, block):
    """The definition of warp_gather in plain NumPy (tests)."""
    h, w = src.shape
    b = np.arange(w) // block
    sy = np.arange(h)[:, None] + dy[:, b]
    sx = np.arange(w)[None, :] + dx[:, b]
    inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
    syc, sxc = np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)
    return np.where(inside, src[syc, sxc], 0).astype(np.uint8), np.where(inside, src_valid[syc, sxc], 0).astype(np.uint8)


def gain_lut(g_q8: int, o_q8: int) -> np.ndarray:
    """lut[v] = clamp((v * g_q8 + o_q8) >> 8, 0, 255) (arithmetic shift = floor). o_q8 includes the +128 that
    makes the shift round to nearest."""
    v = np.arange(256, dtype=np.int64)
    return np.clip((v * int(g_q8) + int(o_q8)) >> 8, 0, 255).astype(np.uint8)


def apply_gain_q8(img: np.ndarray, g_q8: int, o_q8: int, dst: np.ndarray) -> None:
    """C port. uint8 -> uint8: dst = clamp((img * g_q8 + o_q8) >> 8, 0, 255), >> = floor (arithmetic)."""
    if g_q8 == 256 and o_q8 == 128:
        if dst is not img:
            np.copyto(dst, img)
        return
    cv2.LUT(img, gain_lut(g_q8, o_q8), dst=dst)


def envelope_residual(cur, wa, wb, va, vb, mask, env: int, bufs: dict, D: np.ndarray) -> None:
    """C port (env = 3). All uint8 (H, W); va, vb, mask 0 / 255.
        hiX = max of wX over the env x env neighbourhood (pixels inside the frame only), loX = min likewise
        bright = min(sat(cur - hiA), sat(cur - hiB)), dark = min(sat(loA - cur), sat(loB - cur))
        D = max(bright, dark) & va & vb & mask         (sat = saturate to 0..255)"""
    k = _rect(env)
    hi, lo, t1, t2 = bufs["hi"], bufs["lo"], bufs["t1"], bufs["t2"]
    cv2.dilate(wa, k, dst=hi)
    cv2.erode(wa, k, dst=lo)
    cv2.subtract(cur, hi, dst=t1)          # bright vs A
    cv2.subtract(lo, cur, dst=t2)          # dark vs A
    cv2.dilate(wb, k, dst=hi)
    cv2.erode(wb, k, dst=lo)
    cv2.subtract(cur, hi, dst=hi)          # bright vs B
    cv2.subtract(lo, cur, dst=lo)          # dark vs B
    cv2.min(t1, hi, dst=t1)
    cv2.min(t2, lo, dst=t2)
    cv2.max(t1, t2, dst=D)
    cv2.bitwise_and(D, va, dst=D)
    cv2.bitwise_and(D, vb, dst=D)
    if mask is not None:
        cv2.bitwise_and(D, mask, dst=D)


def cell_sums(img: np.ndarray, cell: int, row_step: int = 1) -> np.ndarray:
    """Sum of img (uint8 (H, W)) per cell x cell cell (rows y % row_step == 0 only), int64 (ceil(H/cell),
    ceil(W/cell)); partial cells at the right/bottom edge sum what they have. (Here: an integral image.)"""
    h, w = img.shape
    ch, cw = -(-h // cell), -(-w // cell)
    src = img if row_step == 1 else np.ascontiguousarray(img[::row_step])
    ii = cv2.integral(src, sdepth=cv2.CV_32S)
    ye = np.minimum(np.arange(ch + 1) * cell, h)
    ye = -(-ye // row_step)                       # first sampled row at or after each cell edge
    xe = np.minimum(np.arange(cw + 1) * cell, w)
    c = ii[ye][:, xe].astype(np.int64)
    return c[1:, 1:] - c[:-1, 1:] - c[1:, :-1] + c[:-1, :-1]


def cell_absdiff_sums(cur, wa, valid, cell: int, row_step: int = NOISE_ROW_STEP):
    """C port. Noise statistic of the frame: per cell, over rows y % row_step == 0, the sum of |cur - wa| at
    pixels with valid != 0 and the number of such pixels. Returns (sums, counts), int64 (ch, cw)."""
    a = cv2.absdiff(cur, wa)
    cv2.bitwise_and(a, valid, dst=a)
    return cell_sums(a, cell, row_step), cell_sums((valid != 0).view(np.uint8), cell, row_step)


def noise_sigma_from_cells(sums: np.ndarray, counts: np.ndarray, cell: int, row_step: int = NOISE_ROW_STEP) -> float:
    """Python (tiny). sigma_n = ABS_TO_SIGMA * (the NOISE_QUANTILE-th smallest cell mean of |cur - wa| over the
    cells at least NOISE_MIN_FILL valid); NaN when no cell qualifies. The quietest cells are sky or flat
    ground, where the registered difference is sensor noise (of two frames: hence 2 s / sqrt(pi))."""
    full = cell * (-(-cell // row_step))
    ok = counts >= NOISE_MIN_FILL * full
    if not np.any(ok):
        return float("nan")
    means = sums[ok] / counts[ok]
    k = int(NOISE_QUANTILE * (means.size - 1))
    return float(np.partition(means, k)[k]) * ABS_TO_SIGMA


def threshold_cells(SD, ND, SA, NA, sigma_n: float, t_min_dn: float, noise_k: float, cfar_gain: float,
                    texture_gain: float, cell: int, row_step: int = NOISE_ROW_STEP):
    """Python (tiny: one value per cell). The CFAR threshold of each cell, uint8:
        floor = max(t_min_dn, noise_k * sigma_n)
        mD    = (box sum of SD) / (box sum of ND)              mean residual D around the cell
        mA    = (box sum of SA) / (box sum of NA)              mean |cur - warped A| around the cell
        exc   = max(0, mA - sigma_n / ABS_TO_SIGMA)            ... less what sensor noise alone gives
        thr   = clamp(floor(floor + cfar_gain * mD + texture_gain * exc), 0, 255)
    box = the CFAR_BOX x CFAR_BOX cells around, clipped to the grid. SD, ND: cell sums of D and pixel counts
    (cell_sums, all rows); SA, NA: cell_absdiff_sums (every row_step-th row, valid pixels). Texture that does
    not register to the pixel (sub-pixel misregistration x gradient, leaf flutter, parallax) raises its own bar
    through mA, open sky keeps the floor."""
    S, N = _box_clipped(SD, CFAR_BOX), _box_clipped(ND, CFAR_BOX)
    sa, na = _box_clipped(SA, CFAR_BOX), _box_clipped(NA, CFAR_BOX)
    sig = sigma_n if math.isfinite(sigma_n) else 0.0
    fl = max(float(t_min_dn), float(noise_k) * sig)
    mD = S / np.maximum(N, 1)
    mA = np.where(na > 0, sa / np.maximum(na, 1), 0.0)
    exc = np.maximum(0.0, mA - sig / ABS_TO_SIGMA)
    thr = np.clip(np.floor(fl + float(cfar_gain) * mD + float(texture_gain) * exc), 0, 255).astype(np.uint8)
    return thr, mD


def compare_cells(D: np.ndarray, thr: np.ndarray, cell: int, binary: np.ndarray | None = None) -> np.ndarray:
    """C port. binary[y, x] = 255 if D[y, x] > thr[y // cell, x // cell] else 0 (uint8)."""
    h, w = D.shape
    ch, cw = thr.shape
    big = cv2.resize(thr, (cw * cell, ch * cell), interpolation=cv2.INTER_NEAREST)
    if big.shape != (h, w):
        big = np.ascontiguousarray(big[:h, :w])
    if binary is None:
        binary = np.empty((h, w), np.uint8)
    cv2.compare(D, big, cv2.CMP_GT, dst=binary)
    return binary


def _box_clipped(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over the k x k neighbourhood clipped to the array (int64)."""
    r = k // 2
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1), np.int64)
    np.cumsum(np.cumsum(a, 0), 1, out=ii[1:, 1:])
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    return ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]


def sad_surface(cur: np.ndarray, ref: np.ndarray, vcur: np.ndarray, vref: np.ndarray, cx: int, cy: int, r: int,
                region: tuple[int, int, int, int] | None = None):
    """C port. cur, ref: uint8 (h, w) (ref already warped by the prior and gain-normalised); vcur, vref: uint8
    0 / 255; region = (x0, y0, x1, y1) of cur (default the whole image). For each shift (sx, sy) = (cx + i,
    cy + j), i, j in -r..r:
        sum[j + r, i + r] = sum over pixels p = (x, y) of the region with vcur[p] != 0, q = (x + sx, y + sy)
                            inside the image and vref[q] != 0 of |cur[p] - ref[q]|;  cnt = number of such p
    Returns sum, cnt: int64 (2r + 1, 2r + 1). Minimum at (sx, sy) <=> cur(p) ~ ref(p + s)."""
    h, w = cur.shape
    x0, y0, x1, y1 = (0, 0, w, h) if region is None else region
    n = 2 * r + 1
    sums = np.zeros((n, n), np.int64)
    cnts = np.zeros((n, n), np.int64)
    for j in range(n):
        sy = int(cy) + j - r
        ya, yb = max(y0, -sy), min(y1, h - sy)
        if yb <= ya:
            continue
        for i in range(n):
            sx = int(cx) + i - r
            xa, xb = max(x0, -sx), min(x1, w - sx)
            if xb <= xa:
                continue
            m = cv2.bitwise_and(vcur[ya:yb, xa:xb], vref[ya + sy:yb + sy, xa + sx:xb + sx])
            sums[j, i] = int(cv2.norm(cur[ya:yb, xa:xb], ref[ya + sy:yb + sy, xa + sx:xb + sx], cv2.NORM_L1, m))
            cnts[j, i] = cv2.countNonZero(m)
    return sums, cnts


def sad_cells(cur: np.ndarray, ref: np.ndarray, vcur: np.ndarray, vref: np.ndarray, r: int, ys, xs):
    """C port. sad_surface(cur, ref, vcur, vref, 0, 0, r, region) for every cell of the grid with row edges ys
    and column edges xs (region of cell (a, b) = (xs[b], ys[a], xs[b+1], ys[a+1])).
    Returns sums, cnts: int64 (rows, cols, 2r + 1, 2r + 1)."""
    h, w = cur.shape
    n = 2 * r + 1
    gr, gc = len(ys) - 1, len(xs) - 1
    sums = np.zeros((gr, gc, n, n), np.int64)
    cnts = np.zeros((gr, gc, n, n), np.int64)
    ad = np.zeros((h, w), np.uint8)
    mk = np.zeros((h, w), np.uint8)
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    for j in range(n):
        sy = j - r
        ya, yb = max(0, -sy), min(h, h - sy)
        for i in range(n):
            sx = i - r
            xa, xb = max(0, -sx), min(w, w - sx)
            ad[...] = 0
            mk[...] = 0
            m = cv2.bitwise_and(vcur[ya:yb, xa:xb], vref[ya + sy:yb + sy, xa + sx:xb + sx])
            m = cv2.min(m, 1)
            mk[ya:yb, xa:xb] = m
            ad[ya:yb, xa:xb] = cv2.multiply(cv2.absdiff(cur[ya:yb, xa:xb], ref[ya + sy:yb + sy, xa + sx:xb + sx]), m)
            for arr, out in ((ad, sums), (mk, cnts)):
                ii = cv2.integral(arr, sdepth=cv2.CV_32S)
                c = ii[ys][:, xs].astype(np.int64)
                out[:, :, j, i] = c[1:, 1:] - c[:-1, 1:] - c[1:, :-1] + c[:-1, :-1]
    return sums, cnts


class KernelBuffers:
    """Preallocated working-resolution buffers of motion_kernel."""

    def __init__(self, h: int, w: int, cell: int) -> None:
        self.shape = (h, w)
        self.wa = np.empty((h, w), np.uint8)
        self.wb = np.empty((h, w), np.uint8)
        self.va = np.empty((h, w), np.uint8)
        self.vb = np.empty((h, w), np.uint8)
        self.D = np.empty((h, w), np.uint8)
        self.binary = np.empty((h, w), np.uint8)
        self.env = {k: np.empty((h, w), np.uint8) for k in ("hi", "lo", "t1", "t2")}
        self.thr = np.empty((-(-h // cell), -(-w // cell)), np.uint8)
        self.gather = _GatherIdx(h, w, WARP_BLOCK)


def motion_kernel(cur, ref_a, ref_b, src_valid, mask, offs_a, offs_b, gain_a, gain_b, env: int, cell: int,
                  t_min_dn: float, noise_k: float, cfar_gain: float, texture_gain: float, bufs: KernelBuffers):
    """The hot path (one function, so a C kernel can replace it whole). Working-resolution uint8 images.

    offs_X = (dx, dy) int32 (H, nb) block offsets of reference X; gain_X = (g_q8, o_q8).
    C port: everything but noise_sigma_from_cells / threshold_cells (Python, one value per cell).
    Returns (D, binary, thr_cells, mD_cells, sigma_n)."""
    warp_gather(ref_a, src_valid, offs_a[0], offs_a[1], WARP_BLOCK, bufs.wa, bufs.va, bufs.gather)
    warp_gather(ref_b, src_valid, offs_b[0], offs_b[1], WARP_BLOCK, bufs.wb, bufs.vb, bufs.gather)
    apply_gain_q8(bufs.wa, gain_a[0], gain_a[1], bufs.wa)
    apply_gain_q8(bufs.wb, gain_b[0], gain_b[1], bufs.wb)
    envelope_residual(cur, bufs.wa, bufs.wb, bufs.va, bufs.vb, mask, env, bufs.env, bufs.D)
    valid = bufs.env["t1"]
    cv2.bitwise_and(bufs.va, bufs.vb, dst=valid)
    if mask is not None:
        cv2.bitwise_and(valid, mask, dst=valid)
    SA, NA = cell_absdiff_sums(cur, bufs.wa, valid, cell)
    SD = cell_sums(bufs.D, cell)
    ND = cell_sums((valid != 0).view(np.uint8), cell)
    ND = np.maximum(ND, 1)
    sigma_n = noise_sigma_from_cells(SA, NA, cell)
    thr, mD = threshold_cells(SD, ND, SA, NA, sigma_n, t_min_dn, noise_k, cfar_gain, texture_gain, cell)
    bufs.thr[...] = thr
    compare_cells(bufs.D, bufs.thr, cell, bufs.binary)
    return bufs.D, bufs.binary, bufs.thr, mD, sigma_n


# =============================================================================================== helpers
_RECTS: dict = {}


def _rect(k: int) -> np.ndarray:
    if k not in _RECTS:
        _RECTS[k] = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return _RECTS[k]


def _level(f: float) -> np.ndarray:
    """3x3 affine full-res pixel -> pixel of the image reduced by f (pixel centres: x_l = (x + 0.5) / f - 0.5)."""
    o = 0.5 / f - 0.5
    return np.array([[1.0 / f, 0.0, o], [0.0, 1.0 / f, o], [0.0, 0.0, 1.0]])


_LEVELS: dict = {}


def _at_level(m: np.ndarray, f: float) -> np.ndarray:
    if f not in _LEVELS:       # (L, inv(L)) of each level computed once: the same numbers every call
        L = _level(f)
        _LEVELS[f] = (L, np.linalg.inv(L))
    L, Li = _LEVELS[f]
    return L @ m @ Li


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _apply(m: np.ndarray, u: float, v: float) -> tuple[float, float]:
    w = m[2, 0] * u + m[2, 1] * v + m[2, 2]
    return (float((m[0, 0] * u + m[0, 1] * v + m[0, 2]) / w), float((m[1, 0] * u + m[1, 1] * v + m[1, 2]) / w))


def _rigid(theta: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def _src_valid(mask_lvl: np.ndarray | None, h: int, w: int, margin: int) -> np.ndarray:
    """255 where a pixel may serve as a warp source: unmasked, margin px from the mask and from the frame edge."""
    v = np.full((h, w), 255, np.uint8) if mask_lvl is None else np.where(mask_lvl > 0, 255, 0).astype(np.uint8)
    if mask_lvl is not None and margin > 0:
        v = cv2.erode(v, _rect(2 * margin - 1), borderType=cv2.BORDER_CONSTANT, borderValue=255)
    if margin > 0:
        v[:margin] = 0
        v[-margin:] = 0
        v[:, :margin] = 0
        v[:, -margin:] = 0
    return v


def photometric_fit(cur: np.ndarray, ref: np.ndarray, valid: np.ndarray):
    """Python (tiny). cur ~ a ref + b on the valid low-res overlap (every 2nd row and column), by matching the
    10th / 50th / 90th percentiles of the two: a = (cur90 - cur10) / (ref90 - ref10), b = cur50 - a ref50.
    Symmetric in the two frames (a least-squares slope of cur on ref is pulled below 1 by the reference's own
    noise and misregistration: 0.9 on a textured scene with no exposure change at all, a 5 DN error on bright
    texture) and indifferent to the few pixels a mover changes. A reference with less than FIT_STD_MIN DN of
    spread (open sky) falls back to the ratio of means above BLACK_DN. a clamped to [GAIN_MIN, GAIN_MAX].
    Returns a, b, (g_q8, o_q8)."""
    m = valid[::2, ::2] != 0
    x = ref[::2, ::2][m]
    y = cur[::2, ::2][m]
    a, b = 1.0, 0.0
    if x.size >= 16:
        x10, x50, x90 = np.percentile(x, (10, 50, 90))
        y10, y50, y90 = np.percentile(y, (10, 50, 90))
        if x90 - x10 < 2.56 * FIT_STD_MIN:
            a = (float(y.mean()) - BLACK_DN) / max(float(x.mean()) - BLACK_DN, 1.0)
            a = min(max(a, GAIN_MIN), GAIN_MAX)
            b = BLACK_DN * (1.0 - a)
        else:
            a = min(max(float(y90 - y10) / float(x90 - x10), GAIN_MIN), GAIN_MAX)
            b = float(y50) - a * float(x50)
    g = int(round(a * 256.0))
    o = int(round(b * 256.0)) + 128
    return a, b, (g, o)


def _stab_dir(KS, u: float, v: float) -> tuple[float, float, float]:
    """Unit direction, in the detector's stabilised frame, of full-res pixel (u, v) of a frame whose plane map is
    S: normalize(K^-1 S (u, v, 1)), KS = (K^-1 S) as nested lists. The homogeneous vector's sign is kept."""
    x = KS[0][0] * u + KS[0][1] * v + KS[0][2]
    y = KS[1][0] * u + KS[1][1] * v + KS[1][2]
    z = KS[2][0] * u + KS[2][1] * v + KS[2][2]
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


def _parabola(sm: float, s0: float, sp: float) -> tuple[float, bool]:
    den = sm - 2.0 * s0 + sp
    if den <= 0.0:
        return 0.0, False
    return float(min(0.5, max(-0.5, 0.5 * (sm - sp) / den))), True


# ============================================================================================== detector
class _Slot:
    __slots__ = ("work", "fine", "coarse", "tex", "tensor", "t", "R_wc", "S", "used", "addr")

    def __init__(self, shape_w, shape_f, shape_c) -> None:
        self.work = np.empty(shape_w, np.uint8)
        self.fine = np.empty(shape_f, np.uint8)
        self.coarse = np.empty(shape_c, np.uint8)
        self.tex = np.empty(shape_f, np.uint8)
        # addresses of work, fine, coarse, tex for the native kernel (the arrays live as long as the slot)
        self.addr = (self.work.ctypes.data, self.fine.ctypes.data, self.coarse.ctypes.data, self.tex.ctypes.data)
        self.tensor = (0, 0, 0)     # texture_tensor of the fine level over its usable pixels
        self.t = 0
        self.R_wc = None
        self.S = np.eye(3)
        self.used = False


class _NativeMotion:
    """The C kernel (tbot_motion_* in tbot/vision/_measure.c) for one frame size: its ctypes structures preset
    with the detector's buffers, and the calls detect() and _register make. Every result is the Python path's,
    bit for bit (tests/test_native_motion.py). A buffer that is not a C-contiguous array of the expected type
    raises TypeError here (the detector then runs the Python path)."""

    def __init__(self, K, det: "MotionDetector", H: int, W: int) -> None:
        import ctypes  # noqa: PLC0415

        from tbot.vision import _native as N  # noqa: PLC0415
        if FINE_R != 1 or FINE_RECENTRE != 2 or FIELD_R > 4 or FIELD_GRID[0] * FIELD_GRID[1] > 64:
            raise TypeError("the C registration is written for FINE_R 1, FINE_RECENTRE 2, FIELD_R <= 4")
        P = N.ptr
        u8 = np.uint8
        self.K, self.det, self.N = K, det, N
        self.H, self.W = H, W
        self.steps = int(round(math.log2(det.scale)))
        self.scratch = np.empty(2 * (H // 2) * (W // 2) + 1, u8)
        self.tensor = np.zeros(3, np.int64)
        hw, ww = det._wshape
        hf, wf = det._fshape
        hc, wc = det._cshape
        self.curv_f = P(det._curv_f, u8)
        # ---- registration (tbot_motion_reg1 / reg2)
        nbf, nbc, nbw = -(-wf // WARP_BLOCK), -(-wc // WARP_BLOCK), -(-ww // WARP_BLOCK)
        n = max(hf * nbf, hc * nbc, 1)
        self.dx, self.dy = np.zeros(n, np.int32), np.zeros(n, np.int32)
        self.rx, self.ry = np.zeros(n), np.zeros(n)
        gr, gc = FIELD_GRID
        self.ys = np.linspace(0, hf, gr + 1).astype(int).astype(np.int32)
        self.xs = np.linspace(0, wf, gc + 1).astype(int).astype(np.int32)
        self.field = np.zeros((gr, gc, 2))
        self.field_ok = np.zeros((gr, gc), u8)
        r = self.reg = N.MotionReg()
        r.hc, r.wc, r.hf, r.wf, r.block = hc, wc, hf, wf, WARP_BLOCK
        r.gr, r.gc, r.field_r = gr, gc, FIELD_R
        r.srcv_c, r.srcv_f = P(det._srcv_c, u8), P(det._srcv_f, u8)
        r.curv_c, r.curv_f = P(det._curv_c, u8), self.curv_f
        r.wrc, r.vrc = P(det._tmp_c[0], u8), P(det._tmp_c[1], u8)
        r.wrf, r.vrf = P(det._tmp_f[0], u8), P(det._tmp_f[1], u8)
        r.dx, r.dy, r.rx, r.ry = (a.ctypes.data for a in (self.dx, self.dy, self.rx, self.ry))
        r.ys, r.xs = self.ys.ctypes.data, self.xs.ctypes.data
        r.field, r.field_ok = self.field.ctypes.data, self.field_ok.ctypes.data
        r.min_overlap_frac, r.black_dn, r.gain_min, r.gain_max = MIN_OVERLAP_FRAC, BLACK_DN, GAIN_MIN, GAIN_MAX
        r.fit_std_min, r.cell_texture_frac = FIT_STD_MIN, CELL_TEXTURE_FRAC
        r.np_chunk = K.np_chunk
        self.reg_ref = ctypes.byref(r)
        # ---- kernel (tbot_motion_offsets, tbot_motion_kernel)
        x0, x1, tx, y0, y1, ty = det._ftab
        self.ftab = (np.ascontiguousarray(x0, np.int32), np.ascontiguousarray(x1, np.int32),
                     np.ascontiguousarray(tx, np.float64), np.ascontiguousarray(y0, np.int32),
                     np.ascontiguousarray(y1, np.int32), np.ascontiguousarray(ty, np.float64))
        self.ftab_p = tuple(a.ctypes.data for a in self.ftab)
        self.ex, self.ey = np.zeros((hw, nbw)), np.zeros((hw, nbw))
        self.offs = [np.zeros((hw, nbw), np.int32) for _ in range(4)]
        self.rxw, self.ryw = np.zeros((hw, nbw)), np.zeros((hw, nbw))
        self.mat = np.zeros((3, 3))
        b = det._bufs
        cell = b.thr.shape
        self.mD = np.zeros(cell)
        self.cells = [np.zeros(cell, np.int64) for _ in range(4)]
        self.kscratch = np.empty(6 * hw * ww, u8)
        kp = self.kp = N.MotionKParams()
        kp.h, kp.w, kp.block = hw, ww, WARP_BLOCK
        kp.row_step, kp.cfar_box = NOISE_ROW_STEP, CFAR_BOX
        kp.src_valid, kp.mask = P(det._srcv_w, u8), P(det._mask_d, u8)
        kp.dxa, kp.dya, kp.dxb, kp.dyb = (a.ctypes.data for a in self.offs)
        kp.wa, kp.wb, kp.va, kp.vb = P(b.wa, u8), P(b.wb, u8), P(b.va, u8), P(b.vb, u8)
        kp.D, kp.binary, kp.thr = P(b.D, u8), P(b.binary, u8), P(b.thr, u8)
        kp.mD = self.mD.ctypes.data
        kp.SD, kp.ND, kp.SA, kp.NA = (a.ctypes.data for a in self.cells)
        kp.scratch = self.kscratch.ctypes.data
        kp.abs_to_sigma, kp.noise_quantile, kp.noise_min_fill = ABS_TO_SIGMA, NOISE_QUANTILE, NOISE_MIN_FILL
        self.kp_ref = ctypes.byref(kp)
        # ---- components + measure (tbot_motion_measure)
        self.joined = np.empty(hw * ww, u8)
        self.runs = np.empty((hw * (ww // 2 + 2), 4), np.int32)
        self.out = np.zeros((1, N.MOTION_DET_COLS))
        mp = self.mp = N.MotionMeas()
        mp.h, mp.w, mp.scale, mp.cap_runs = hw, ww, det.scale, self.runs.shape[0]
        mp.binary, mp.D, mp.wa, mp.mD = kp.binary, kp.D, kp.wa, kp.mD
        mp.joined, mp.runs = self.joined.ctypes.data, self.runs.ctypes.data
        mp.out = self.out.ctypes.data
        mp.pixel_box_var, mp.score_mass = PIXEL_BOX_VAR, SCORE_MASS
        self.mp_ref = ctypes.byref(mp)

    def prep(self, img: np.ndarray, slot: _Slot) -> tuple[int, int, int]:
        """box_down2 chain, texture_map, texture_tensor into the slot; returns the tensor."""
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        a = slot.addr
        self.K.prep(img.ctypes.data, self.H, self.W, self.steps, self.det._n_fine, self.scratch.ctypes.data,
                    a[0], a[1], a[2], a[3], TEXTURE_DN, self.curv_f, self.tensor.ctypes.data)
        t = self.tensor.tolist()
        return t[0], t[1], t[2]

    def reg1(self, cur: _Slot, ref: _Slot, lc, lf, r_c: int, need_x: bool = True, need_y: bool = True):
        """MotionDetector._search_py in C: None or (depth, edge, okx, oky, fin, delta_f)."""
        r = self.reg
        r.cur_c, r.cur_f, r.tex_f = cur.addr[2], cur.addr[1], cur.addr[3]
        r.ref_c, r.ref_f = ref.addr[2], ref.addr[1]
        r.lc[:] = lc.ravel().tolist()
        r.lf[:] = lf.ravel().tolist()
        r.r_c = int(r_c)
        r.need_x, r.need_y = (1 if need_x else 0), (1 if need_y else 0)
        if self.K.reg1(self.reg_ref):
            raise MemoryError("tbot_motion_reg1")
        if r.early:
            return None
        c = r.curv
        return (r.depth, bool(r.edge), bool(c & 1), bool(c & 2), bool(c & 4),
                np.array([r.delta[0], r.delta[1]]))

    def reg2(self, lm, use_field: bool, rd: float):
        """The fit and the local field of _register (after reg1, same cur / ref): a, b, gq, field or None."""
        r = self.reg
        r.lm[:] = lm.ravel().tolist()
        r.use_field = 1 if use_field else 0
        r.reg_min_response = float(self.det.vc.reg_min_response)
        if self.K.reg2(self.reg_ref, rd):
            raise MemoryError("tbot_motion_reg2")
        return r.a, r.b, (r.gq[0], r.gq[1]), (self.field.copy() if use_field else None)

    def kernel(self, slot: _Slot, A: _Slot, B: _Slot, ma, mb, info_a: dict, info_b: dict):
        """block_offsets (+ _field_extra) of both references and motion_kernel: (D, binary, thr, mD, sigma_n)."""
        K, vc, kp, ft = self.K, self.det.vc, self.kp, self.ftab_p
        hw, ww = self.det._wshape
        gr, gc = FIELD_GRID
        s = float(self.det.scale)
        for m, info, dx, dy in ((ma, info_a, self.offs[0], self.offs[1]), (mb, info_b, self.offs[2], self.offs[3])):
            np.copyto(self.mat, m)
            f = info["field"]
            fp = None
            if f is not None:
                if f.shape != (gr, gc, 2) or f.dtype != np.float64 or not f.flags.c_contiguous:
                    f = np.ascontiguousarray(f, np.float64)
                    if f.shape != (gr, gc, 2):
                        raise TypeError("field of an unexpected shape")
                fp = f.ctypes.data
            K.offsets(self.mat.ctypes.data, hw, ww, WARP_BLOCK, fp, gr, gc, ft[0], ft[1], ft[2], ft[3], ft[4], ft[5],
                      s, self.ex.ctypes.data, self.ey.ctypes.data, dx.ctypes.data, dy.ctypes.data,
                      self.rxw.ctypes.data, self.ryw.ctypes.data)
        kp.cur, kp.ref_a, kp.ref_b = slot.addr[0], A.addr[0], B.addr[0]
        kp.ga, kp.oa = info_a["gq"]
        kp.gb, kp.ob = info_b["gq"]
        kp.env, kp.cell = max(1, int(vc.envelope_px)), max(2, int(vc.cfar_cell_px))
        kp.t_min_dn, kp.noise_k = float(vc.t_min_dn), float(vc.noise_k)
        kp.cfar_gain, kp.texture_gain = float(vc.cfar_gain), float(vc.texture_gain)
        if K.kernel(self.kp_ref):
            raise MemoryError("tbot_motion_kernel")
        b = self.det._bufs
        return b.D, b.binary, b.thr, self.mD, kp.sigma_n

    def measure(self, slot: _Slot):
        """Components and measurement of the frame's binary image: (labels incl. background or 0, candidates,
        detection rows as lists), or None when a capacity is exceeded (the caller takes the Python path)."""
        vc, mp = self.det.vc, self.mp
        mc = int(vc.max_candidates)
        if self.out.shape[0] < max(mc, 1):
            self.out = np.zeros((max(mc, 1), self.N.MOTION_DET_COLS))
            mp.out = self.out.ctypes.data
        mp.cell = max(2, int(vc.cfar_cell_px))
        mp.min_area, mp.max_area, mp.max_cand = int(vc.min_area_px), int(vc.max_area_px), mc
        mp.cur = slot.addr[0]
        mp.S[:] = slot.S.ravel().tolist()
        if self.K.measure(self.mp_ref):
            return None
        return mp.n_labels, mp.n_cand, self.out[:mp.n_out].tolist()


class MotionDetector:
    """Detections of what moves against the world (see the module docstring)."""

    def __init__(self, cfg, mask: np.ndarray | None = None, cam=None, mount=None) -> None:
        self.cfg = cfg
        self.vc = cfg.vision
        self.cam = cam
        self.mount = mount
        self.timings_ms: dict[str, float] = {}
        self.stats: dict = {}
        self.noise_sigma: float = float("nan")
        self.debug: dict = {}
        self.keep_debug = False
        self.candidate_filter = None  # accepted for the ClayDetector surface; not used
        self.scale = max(1, int(self.vc.scale))
        # local residual registration field (parallax of near scenery): vision.local_field
        self.use_field = bool(getattr(self.vc, "local_field", True))
        self.stab_epoch = 0  # kept for the pipeline's check; the plane is no longer re-based, so it stays 0
        self._mask_src = None
        self._size = None
        self._nat = None             # _NativeMotion for the current frame size (vision.native and the C kernel built)
        self._cam_size = None
        self._R_gc = None
        self.set_mask(mask)
        self.reset()

    # ---------------------------------------------------------------- surface
    def set_mask(self, mask: np.ndarray | None) -> None:
        """Usable-pixel mask (non-zero = usable), any size (resized NEAREST to the frame)."""
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim != 2:
                raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
        self._mask_src = mask
        self._size = None  # rebuild the per-size state on the next frame

    def reset(self) -> None:
        """Forget all earlier frames (camera restart)."""
        self._slots: list[_Slot] = []
        self._order: list[int] = []  # slot indices, oldest first
        self._vel = None             # (tx, ty) px/s of cur -> ref translation, for the prior without IMU
        self._corr_rate = np.zeros(2)  # running registration correction to the IMU prior, px/s (gyro bias)
        self._last_t = None

    # ------------------------------------------------------------ per size
    def _setup(self, H: int, W: int) -> None:
        s = self.scale
        rd = max(1, int(self.vc.reg_downscale))
        if rd % s or (rd // s) & (rd // s - 1):
            raise ValueError(f"vision.reg_downscale ({rd}) must be scale ({s}) times a power of 2")
        self._rd = rd
        self._n_fine = int(round(math.log2(rd // s)))          # box_down2 steps working -> fine
        hw, ww = H // s, W // s
        self._wshape = (hw, ww)
        shp = [(hw, ww)]
        for _ in range(self._n_fine + 1):
            shp.append((shp[-1][0] // 2, shp[-1][1] // 2))
        self._fshape, self._cshape = shp[self._n_fine], shp[self._n_fine + 1]
        self._pyr_bufs = [np.empty(x, np.uint8) for x in shp[1:]]
        self._wbuf = np.empty((hw, ww), np.uint8)
        # masks at every level (a low-res pixel is usable only if all its source pixels are)
        mw = None
        if self._mask_src is not None:
            src = self._mask_src
            u8 = src if src.dtype == np.uint8 else (src != 0).astype(np.uint8) * 255
            if u8.shape != (H, W):
                u8 = cv2.resize(np.ascontiguousarray(u8), (W, H), interpolation=cv2.INTER_NEAREST)
            m = np.where(u8 > 0, 255, 0).astype(np.uint8)
            for _ in range(int(round(math.log2(s)))):
                m = min_down2(m)
            mw = np.ascontiguousarray(m[:hw, :ww])
        self._mask_w = mw
        # the residual mask: the mask, and the outermost envelope_px // 2 rows / columns of the frame (the
        # envelope there sees only one side of the reference: registration error would show)
        r = max(1, int(self.vc.envelope_px) // 2)
        dm = np.full((hw, ww), 255, np.uint8) if mw is None else mw.copy()
        dm[:r] = 0
        dm[-r:] = 0
        dm[:, :r] = 0
        dm[:, -r:] = 0
        self._mask_d = dm
        levels = [mw]
        for _ in range(self._n_fine + 1):
            levels.append(None if levels[-1] is None else min_down2(levels[-1]))
        self._mask_f, self._mask_c = levels[self._n_fine], levels[self._n_fine + 1]
        self._srcv_w = _src_valid(mw, hw, ww, SRC_MARGIN)
        self._srcv_f = _src_valid(self._mask_f, *self._fshape, 1)
        self._srcv_c = _src_valid(self._mask_c, *self._cshape, 1)
        self._curv_f = np.full(self._fshape, 255, np.uint8) if self._mask_f is None else self._mask_f
        self._curv_c = np.full(self._cshape, 255, np.uint8) if self._mask_c is None else self._mask_c
        self._n_valid_f = int(np.count_nonzero(self._curv_f))    # usable fine-level pixels (conditioning scale)
        cell = max(2, int(self.vc.cfar_cell_px))
        self._bufs = KernelBuffers(hw, ww, cell)
        self._lvl_f = _GatherIdx(*self._fshape, WARP_BLOCK)
        self._lvl_c = _GatherIdx(*self._cshape, WARP_BLOCK)
        self._tmp_f = (np.empty(self._fshape, np.uint8), np.empty(self._fshape, np.uint8))
        self._tmp_c = (np.empty(self._cshape, np.uint8), np.empty(self._cshape, np.uint8))
        # camera model in full-res pixels of this frame size
        cam = self.cam
        if cam is None or (cam.width, cam.height) != (W, H):
            from tbot.optics import CameraModel  # noqa: PLC0415
            if cam is None:
                cam = CameraModel.nominal(self.cfg, W, H)
            else:
                sx, sy = W / cam.width, H / cam.height
                K = cam.K.copy()
                K[0, 0] *= sx
                K[0, 1] *= sx
                K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
                K[1, 1] *= sy
                K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
                cam = CameraModel(W, H, K, cam.D, cam.model)
        self._cam_w = cam
        self._K = np.asarray(cam.K, dtype=np.float64)
        self._Kinv = np.linalg.inv(self._K)   # full-res pixel -> camera ray
        self._pinhole = (str(getattr(cam, "model", "")).lower() == "pinhole"
                         and not np.any(np.asarray(getattr(cam, "D", 0.0), dtype=np.float64)))
        if self.mount is not None:
            self._R_gc = np.asarray(self.mount.R_gun_cam, dtype=np.float64)
        else:
            from tbot.optics import NOMINAL_R_GUN_CAM  # noqa: PLC0415
            self._R_gc = NOMINAL_R_GUN_CAM
        gx = np.linspace(0.0, 1.0, 5) * (W - 1)
        gy = np.linspace(0.0, 1.0, 5) * (H - 1)
        self._grid = np.array([(x, y) for y in gy for x in gx], dtype=np.float64)
        self._grid_rays = cam.pixel_to_ray(self._grid)
        self._slot_pool = [_Slot((hw, ww), self._fshape, self._cshape) for _ in range(RING)]
        # per-size constants, computed once (the same expressions as per frame before: bit-identical)
        self._ftab = self._field_tables(FIELD_GRID[0], FIELD_GRID[1], H, W)
        self._lw = _level(float(s))
        self._lwi = np.linalg.inv(self._lw)
        self._nat = None
        if bool(getattr(self.vc, "native", True)):
            from tbot.vision import _native  # noqa: PLC0415
            k = _native.motion_kernel()
            if k is not None:
                try:
                    self._nat = _NativeMotion(k, self, H, W)
                except TypeError:          # a buffer that is not a C-contiguous array of the expected type
                    self._nat = None
        self._size = (H, W)
        self.reset()

    # ---------------------------------------------------------------- prior
    def _prior(self, R_cur, R_ref, dt_s: float) -> np.ndarray:
        """3x3 full-res homography cur pixel -> ref pixel predicted before registration: the relative camera
        rotation mapped through the camera model on a 5 x 5 grid and fitted (exact for a pinhole without
        distortion; not a small-angle or rigid approximation: at 80 deg/s over 100 ms the perspective change of
        a pure rotation is 1-2.5 px at the frame edges), plus the registration's running correction rate."""
        if R_cur is not None and R_ref is not None:
            R = R_ref.T @ R_cur          # camera(cur) -> camera(ref)
            if self._pinhole:
                # a pinhole without distortion (the gun's 12 mm lens): the homography of a rotation IS K R K^-1,
                # no grid and no fit (OpenCV's fit of 25 points took over a millisecond a reference on the Pi)
                Hm = self._K @ R @ self._Kinv
                if abs(Hm[2, 2]) > 1e-12:
                    Hm = Hm / Hm[2, 2]
                    c = self._corr_rate
                    return _translate(c[0] * dt_s, c[1] * dt_s) @ Hm
            rays = self._grid_rays @ R.T
            q = self._cam_w.ray_to_pixel(rays)
            ok = np.isfinite(q).all(axis=1)
            if ok.sum() >= 4:
                Hm, _ = cv2.findHomography(self._grid[ok], q[ok], 0)
                if Hm is not None:
                    Hm = Hm / Hm[2, 2]
                    c = self._corr_rate
                    return _translate(c[0] * dt_s, c[1] * dt_s) @ Hm
        if self._vel is not None:
            return _rigid(0.0, self._vel[0] * dt_s, self._vel[1] * dt_s)
        return np.eye(3)

    # ------------------------------------------------------------ register
    def _register(self, cur: _Slot, ref: _Slot, imu: bool, want_oracle: bool = False):
        """cur -> ref full-res 3x3 affine, and an info dict."""
        dt = (cur.t - ref.t) * 1e-9
        prior = self._prior(cur.R_wc, ref.R_wc, dt)
        rd = float(self._rd)
        fc = 2.0 * rd
        # ---- coarse: block warp of the reference by the prior, ratio-of-means gain, SAD search
        lc = _at_level(prior, fc)
        lf = _at_level(prior, rd)
        info = {"prior": (float(prior[0, 2]), float(prior[1, 2])), "ok": False, "depth": 0.0, "gain": 1.0,
                "shift": (0.0, 0.0), "dt_ms": dt * 1e3, "gq": (256, 128), "field": None, "tex": 0.0}
        r_c = NOIMU_COARSE_R if not imu else max(1, int(math.ceil(float(self.vc.reg_search_px) / fc)))
        wf, vf = self._tmp_f
        # conditioning: does the texture fix the shift in BOTH directions? A horizon, a roof line or a banded sky
        # fixes it across the edge only: along it the SAD surface is a flat valley and its minimum is noise (the
        # simulator's horizon: +-20 px sideways from frame to frame, "registered"; recorded presses with the weaker
        # eigenvalue under TENSOR_MIN: corrections of 3 px median, 14 px at the 90th percentile, against 0.7 and
        # 5 above it). Then the fine search re-centres across the edge only, only the component across the edge is
        # applied, and the frame does not count as registered (reg_ok false: stab rests partly on the gyro).
        sxx, syy, sxy = cur.tensor
        tr, df = float(sxx + syy), math.hypot(float(sxx - syy), 2.0 * float(sxy))
        qmin = TENSOR_MIN * self._n_valid_f
        two_d = 0.5 * (tr - df) >= qmin
        one_d = not two_d and 0.5 * (tr + df) >= qmin
        th = 0.5 * math.atan2(2.0 * float(sxy), float(sxx - syy))
        e = np.array([math.cos(th), math.sin(th)])         # the dominant gradient direction: across the edge
        need_x = not one_d or abs(e[0]) >= abs(e[1])
        need_y = not one_d or abs(e[1]) > abs(e[0])
        nat = self._nat
        if nat is not None:
            # coarse + fine search in C (tbot_motion_reg1): the same buffers wf, vf and numbers as below
            got = nat.reg1(cur, ref, lc, lf, r_c, need_x, need_y)
            if got is None:
                return prior, info
            depth, edge, okx, oky, fin, delta_f = got
            n_ovl, n_tex = nat.reg.n_ovl, nat.reg.n_tex     # counted in C exactly as below
        else:
            got = self._search_py(cur, ref, lc, lf, r_c, wf, vf, need_x, need_y)
            if got is None:
                return prior, info
            depth, edge, okx, oky, fin, delta_f = got
            # texture guard counts: the unmasked overlap and its textured part
            ovl = (vf != 0) & (self._curv_f != 0)
            n_ovl = int(np.count_nonzero(ovl))
            n_tex = int(np.count_nonzero(ovl & (cur.tex != 0)))
        ok_curv = bool(okx and oky and fin)
        delta = delta_f * rd                                   # full-res, in ref pixels
        lim = float(self.vc.reg_search_px) if imu else NOIMU_COARSE_R * fc
        # texture guard: the unmasked overlap must hold enough textured AREA (never just the target itself)
        textured = n_tex >= TEXTURE_FRAC * max(n_ovl, 1)
        ok_across = fin and (okx if need_x else oky)
        if one_d:
            delta = e * float(delta @ e)
        base = (textured and depth >= float(self.vc.reg_min_response)
                and float(np.hypot(*delta)) <= lim + rd)
        ok = bool(base and two_d and ok_curv)
        partial = bool(base and one_d and ok_across)
        info.update(depth=float(depth), ok=ok, edge=bool(edge), curv=ok_curv, tex=n_tex / max(n_ovl, 1),
                    delta=(float(delta[0]), float(delta[1])), mode="2d" if ok else "1d" if partial else "prior",
                    cond=(0.5 * (tr + df) / max(self._n_valid_f, 1), 0.5 * (tr - df) / max(self._n_valid_f, 1)))
        m = prior.copy()
        if ok or partial:
            m = _translate(float(delta[0]), float(delta[1])) @ m
            info["shift"] = (float(delta[0]), float(delta[1]))
        # ---- photometric fit on the fine level, aligned by the final transform
        if nat is not None:
            a, b, gq, field = nat.reg2(_at_level(m, rd), bool(ok and self.use_field), rd)
            info["gain"] = a
            info["offset"] = b
            info["gq"] = gq
            info["field"] = field
            if field is not None:
                self._field_ok = nat.field_ok.astype(bool)
                info["field_ok"] = self._field_ok.copy()
        else:
            dxf, dyf, _, (rxf, ryf) = block_offsets(_at_level(m, rd), *self._fshape, full=True)
            warp_gather(ref.fine, self._srcv_f, dxf, dyf, WARP_BLOCK, wf, vf, self._lvl_f)
            fitv = (vf != 0) & (self._curv_f != 0)
            a, b, gq = photometric_fit(cur.fine, wf, fitv.view(np.uint8))
            info["gain"] = a
            info["offset"] = b
            info["gq"] = gq
            info["field"] = None
            if ok and self.use_field:
                apply_gain_q8(wf, gq[0], gq[1], wf)
                info["field"] = self._local_field(cur, wf, vf, rxf, ryf, rd)
                info["field_ok"] = self._field_ok.copy()
        if want_oracle:
            info["oracle"] = _phase_oracle(cur.fine, ref.fine, _at_level(prior, rd)) * rd
        return m, info

    def _search_py(self, cur: _Slot, ref: _Slot, lc, lf, r_c: int, wf, vf, need_x: bool = True,
                   need_y: bool = True):
        """The search of _register (Python reference of tbot_motion_reg1): coarse block warp of the reference by
        the prior, ratio-of-means gain, coarse SAD search; fine search +-1 around it with recentring, parabolic
        sub-pixel; the fine window re-centres only along the axes asked for (need_x, need_y: both, or the
        one across an edge-only scene). Leaves the gained fine warp in wf, vf. Returns None (overlap too small) or (depth, edge, okx, oky,
        fin, delta_f): the parabola flags in x and y, whether the last fine surface is finite everywhere,
        and delta_f in fine px (ref space)."""
        dx, dy, res_c = block_offsets(lc, *self._cshape)
        wc, vcv = self._tmp_c
        warp_gather(ref.coarse, self._srcv_c, dx, dy, WARP_BLOCK, wc, vcv, self._lvl_c)
        both = (vcv != 0) & (self._curv_c != 0)
        if both.sum() < MIN_OVERLAP_FRAC * both.size:
            return None
        mc = float(cur.coarse[both].mean())
        mr = float(wc[both].mean())
        a0 = min(max((mc - BLACK_DN) / max(mr - BLACK_DN, 1.0), GAIN_MIN), GAIN_MAX)
        g0 = (int(round(a0 * 256)), int(round(BLACK_DN * (1 - a0) * 256)) + 128)
        apply_gain_q8(wc, g0[0], g0[1], wc)
        sums, cnts = sad_surface(cur.coarse, wc, self._curv_c, vcv, 0, 0, r_c)
        with np.errstate(invalid="ignore", divide="ignore"):
            sad = np.where(cnts >= MIN_OVERLAP_FRAC * both.size, sums / np.maximum(cnts, 1), np.inf)
        j, i = np.unravel_index(int(np.argmin(sad)), sad.shape)
        smin = float(sad[j, i])
        fin = sad[np.isfinite(sad)]
        med = float(np.median(fin)) if fin.size else 0.0
        depth = (med - smin) / max(med, 1e-6) if math.isfinite(smin) else 0.0
        edge = r_c > 0 and (abs(i - r_c) == r_c or abs(j - r_c) == r_c)
        delta_c = np.array([i - r_c, j - r_c], dtype=np.float64) - res_c
        # ---- fine: same at 1/rd, +-1 around the coarse result, parabolic sub-pixel
        dxf, dyf, res_f = block_offsets(lf, *self._fshape)
        warp_gather(ref.fine, self._srcv_f, dxf, dyf, WARP_BLOCK, wf, vf, self._lvl_f)
        apply_gain_q8(wf, g0[0], g0[1], wf)
        centre = np.floor(2.0 * delta_c + res_f + 0.5).astype(int)
        okx = oky = fin = False
        sub = np.zeros(2)
        for _ in range(FINE_RECENTRE + 1):
            fs, fn = sad_surface(cur.fine, wf, self._curv_f, vf, int(centre[0]), int(centre[1]), FINE_R)
            fsad = np.where(fn > 0, fs / np.maximum(fn, 1), np.inf)
            jj, ii = np.unravel_index(int(np.argmin(fsad)), fsad.shape)
            if np.isfinite(fsad[jj, ii]) and ((need_x and ii in (0, 2)) or (need_y and jj in (0, 2))):
                centre = centre + np.array([ii - 1 if need_x else 0, jj - 1 if need_y else 0])
                continue
            sx, okx = _parabola(fsad[1, 0], fsad[1, 1], fsad[1, 2])
            sy, oky = _parabola(fsad[0, 1], fsad[1, 1], fsad[2, 1])
            sub = np.array([sx, sy])
            fin = bool(np.isfinite(fsad).all())
            break
        return depth, edge, okx, oky, fin, centre + sub - res_f

    def _local_field(self, cur: _Slot, wf, vf, rxf, ryf, rd: float) -> np.ndarray:
        """Residual registration field (parallax of near scenery) on a FIELD_GRID of cells of the fine level:
        per cell a SAD search +-FIELD_R fine px around the global solution with a parabolic sub-pixel fit; a cell
        counts only with enough textured area, a clear minimum off the window edge and positive curvature, else
        0. Then a 3x3 median over the grid (clipped at its edges). Returns (rows, cols, 2) full-res px, ref
        space (added to the source offset)."""
        h, w = self._fshape
        gr, gc = FIELD_GRID
        ys = np.linspace(0, h, gr + 1).astype(int)
        xs = np.linspace(0, w, gc + 1).astype(int)
        xcb = block_centres(h, w)
        f = np.zeros((gr, gc, 2))
        self._field_ok = np.zeros((gr, gc), bool)
        ovl = cv2.bitwise_and(vf, self._curv_f)
        nvs = np.add.reduceat(np.add.reduceat((ovl != 0).astype(np.int64), ys[:-1], 0), xs[:-1], 1)
        nts = np.add.reduceat(np.add.reduceat(((ovl != 0) & (cur.tex != 0)).astype(np.int64), ys[:-1], 0),
                              xs[:-1], 1)
        cand = (nvs >= 0.5 * np.diff(ys)[:, None] * np.diff(xs)[None, :]) & (nts >= CELL_TEXTURE_FRAC * nvs)
        if not cand.any():
            return f
        all_s, all_c = sad_cells(cur.fine, wf, self._curv_f, vf, FIELD_R, ys, xs)
        for a in range(gr):
            for b in range(gc):
                if not cand[a, b]:
                    continue
                x0, x1, y0, y1 = xs[b], xs[b + 1], ys[a], ys[a + 1]
                nv = int(nvs[a, b])
                sums, cnts = all_s[a, b], all_c[a, b]
                sad = np.where(cnts >= 0.5 * nv, sums / np.maximum(cnts, 1), np.inf)
                j, i = np.unravel_index(int(np.argmin(sad)), sad.shape)
                if not np.isfinite(sad).all() or i in (0, 2 * FIELD_R) or j in (0, 2 * FIELD_R):
                    continue
                med = float(np.median(sad))
                if (med - float(sad[j, i])) / max(med, 1e-6) < float(self.vc.reg_min_response):
                    continue
                sx, okx = _parabola(sad[j, i - 1], sad[j, i], sad[j, i + 1])
                sy, oky = _parabola(sad[j - 1, i], sad[j, i], sad[j + 1, i])
                if not (okx and oky):
                    continue
                bsel = (xcb >= x0) & (xcb < x1)
                mrx = float(rxf[y0:y1][:, bsel].mean()) if bsel.any() else 0.0
                mry = float(ryf[y0:y1][:, bsel].mean()) if bsel.any() else 0.0
                f[a, b] = ((i - FIELD_R + sx - mrx) * rd, (j - FIELD_R + sy - mry) * rd)
                self._field_ok[a, b] = True
        out = np.zeros_like(f)
        for a in range(gr):
            for b in range(gc):
                nb_ = f[max(0, a - 1):a + 2, max(0, b - 1):b + 2].reshape(-1, 2)
                out[a, b] = np.median(nb_, axis=0)
        return out

    def _field_extra(self, field: np.ndarray | None, H: int, W: int):
        """The field (full-res px) bilinearly interpolated at every working-resolution warp block centre and
        converted to working px: (ex, ey) float64 (h, nb), or None."""
        if field is None or not np.any(field):
            return None
        s = float(self.scale)
        gr, gc = field.shape[:2]
        tab = self._ftab if (gr, gc) == FIELD_GRID else self._field_tables(gr, gc, H, W)
        x0, x1, tx, y0, y1, ty = tab
        out = []
        for c in range(2):
            F = field[:, :, c]
            top = F[y0][:, x0] * (1 - tx)[None, :] + F[y0][:, x1] * tx[None, :]
            bot = F[y1][:, x0] * (1 - tx)[None, :] + F[y1][:, x1] * tx[None, :]
            out.append((top * (1 - ty)[:, None] + bot * ty[:, None]) / s)
        return out[0], out[1]

    def _field_tables(self, gr: int, gc: int, H: int, W: int):
        """Where _field_extra samples a gr x gc field: (x0, x1, tx) per working warp block column, (y0, y1, ty)
        per working row (depends on the frame size only)."""
        s = float(self.scale)
        hw, ww = self._wshape
        # cell centres in full-res px
        ccx = (np.arange(gc) + 0.5) * W / gc - 0.5
        ccy = (np.arange(gr) + 0.5) * H / gr - 0.5
        bx = (block_centres(hw, ww) + 0.5) * s - 0.5
        by = (np.arange(hw) + 0.5) * s - 0.5
        fx = np.clip(np.interp(bx, ccx, np.arange(gc)), 0, gc - 1)
        fy = np.clip(np.interp(by, ccy, np.arange(gr)), 0, gr - 1)
        x0 = np.minimum(np.floor(fx).astype(int), max(gc - 2, 0))
        y0 = np.minimum(np.floor(fy).astype(int), max(gr - 2, 0))
        tx = fx - x0
        ty = fy - y0
        x1 = np.minimum(x0 + 1, gc - 1)
        y1 = np.minimum(y0 + 1, gr - 1)
        return x0, x1, tx, y0, y1, ty

    # ---------------------------------------------------------------- refs
    def _pan_px_s(self, t: int, R_wc) -> float:
        """How fast the scenery crosses the picture now, full-res px/s: from the turn of the optical axis since the
        newest earlier frame (IMU), else from the last registration."""
        if R_wc is not None and self._order:
            prev = self._slots[self._order[-1]]
            dt = (t - prev.t) * 1e-9
            if prev.R_wc is not None and dt > 0.0:
                c = float(np.dot(R_wc[:, 2], prev.R_wc[:, 2]))
                return math.acos(max(-1.0, min(1.0, c))) * float(self._cam_w.K[0, 0]) / dt
        if self._vel is not None:
            return math.hypot(self._vel[0], self._vel[1])
        return 0.0

    def _pick(self, t: int, R_wc=None, tensor=None):
        """Slot indices of references A and B (B != A), or None.

        The lags shrink while the gun swings: a pixel is compared only where BOTH references show the same scenery,
        so the strip of picture that came into view since reference B is blind, and that strip is the leading edge
        of the swing, where the clay the shooter is swinging onto is. Both lags are scaled so that the scenery has
        moved at most LAG_SHIFT_FRAC of the frame width since B (at 70 deg/s: 100 ms -> 39 ms; the simulator's
        crossing clay was in view and unseen for 12 frames of such a swing). What the long lag is for, a slow
        mover, is not what a fast swing follows."""
        if not self._order:
            return None, None
        ts = np.array([self._slots[k].t for k in self._order], dtype=np.int64)
        la_ms, lb_ms = float(self.vc.lag_a_ms), float(self.vc.lag_b_ms)
        if tensor is not None and self._size is not None:
            # (the same test as _register's: the weaker eigenvalue of the textured pixels' gradient tensor)
            sxx, syy, sxy = tensor
            weak = 0.5 * (float(sxx + syy) - math.hypot(float(sxx - syy), 2.0 * float(sxy)))
            if weak < TENSOR_MIN * max(1, int(np.count_nonzero(self._curv_f))):
                la_ms, lb_ms = la_ms * SKY_LAG_GAIN, lb_ms * SKY_LAG_GAIN
        pan = self._pan_px_s(t, R_wc)
        if pan > 0.0 and self._size is not None:
            k = min(1.0, LAG_SHIFT_FRAC * self._size[1] / (pan * max(lb_ms, la_ms) * 1e-3))
            la_ms, lb_ms = la_ms * k, lb_ms * k
        self.stats["lag_ms"] = (round(la_ms, 1), round(lb_ms, 1))
        la = int(la_ms * 1e6)
        lb = int(lb_ms * 1e6)
        ia = int(np.argmin(np.abs(ts - (t - la))))
        if len(ts) < 2:
            return self._order[ia], None
        cost = np.abs(ts - (t - lb)).astype(np.float64)
        cost[ia] = np.inf
        ib = int(np.argmin(cost))
        return self._order[ia], self._order[ib]

    def _push(self, slot: _Slot) -> None:
        idx = self._slots.index(slot)
        self._order.append(idx)
        if len(self._order) > RING - 1:   # keep one slot free for the next frame
            old = self._order.pop(0)
            self._slots[old].used = False

    def _take_slot(self) -> _Slot:
        if not self._slots:
            self._slots = self._slot_pool
            for s in self._slots:
                s.used = False
        for s in self._slots:
            if not s.used:
                s.used = True
                return s
        old = self._order.pop(0)
        self._slots[old].used = True
        return self._slots[old]

    # -------------------------------------------------------------- detect
    def detect(self, frame: Frame, R_wg: np.ndarray | None = None, oracle: bool = False) -> list[Detection]:
        t0 = time.perf_counter()
        img = frame.image
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            raise ValueError(f"frame image must be uint8, got {img.dtype}")
        H, W = img.shape
        if self._size != (H, W):
            self._setup(H, W)
        t = int(frame.t_exposure_ns)
        if self._last_t is not None and t <= self._last_t:
            self.reset()
        self._last_t = t
        vc = self.vc
        s = self.scale
        slot = self._take_slot()
        nat = self._nat
        # ---- prep: working image and pyramid straight into the ring slot
        if nat is not None:
            slot.tensor = nat.prep(img, slot)
        elif s == 1:
            np.copyto(slot.work, img)
        else:
            src = img
            steps = int(round(math.log2(s)))
            for k in range(steps):
                dst = slot.work if k == steps - 1 else np.empty((src.shape[0] // 2, src.shape[1] // 2), np.uint8)
                box_down2(src, dst)
                src = dst
        lv = slot.work
        for k in range(self._n_fine if nat is None else 0):
            dst = slot.fine if k == self._n_fine - 1 else self._pyr_bufs[k]
            box_down2(lv, dst)
            lv = dst
        if nat is None:
            if self._n_fine == 0:
                np.copyto(slot.fine, slot.work)
            box_down2(slot.fine, slot.coarse)
            texture_map(slot.fine, TEXTURE_DN, slot.tex)
            slot.tensor = texture_tensor(slot.fine, slot.tex, self._curv_f)
        slot.t = t
        imu = R_wg is not None
        slot.R_wc = (np.asarray(R_wg, dtype=np.float64).reshape(3, 3) @ self._R_gc) if imu else None
        t_prep = time.perf_counter()
        self.stats = {"components": 0, "candidates": 0, "refs": 0}
        self.timings_ms = {}
        ia, ib = self._pick(t, slot.R_wc, slot.tensor)
        if ia is None:
            slot.S = np.eye(3)
            self._push(slot)
            self.timings_ms = {"prep": (t_prep - t0) * 1e3, "total": (time.perf_counter() - t0) * 1e3}
            return []
        A = self._slots[ia]
        m_a, info_a = self._register(slot, A, imu, oracle)
        dt_a = (t - A.t) * 1e-9
        if dt_a > 0:
            cu, cv_ = (W - 1) * 0.5, (H - 1) * 0.5
            pu, pv = _apply(m_a, cu, cv_)
            self._vel = ((pu - cu) / dt_a, (pv - cv_) / dt_a)
            if imu and info_a["ok"]:
                # the IMU prior's systematic error (gyro bias, time offset) learnt from the registration
                sh = np.asarray(info_a["shift"]) / dt_a
                self._corr_rate = np.clip(self._corr_rate + CORR_GAIN * sh, -CORR_MAX, CORR_MAX)
        slot.S = A.S @ m_a
        # S is a homography fixed only up to scale: keep it at unit norm so the chain never over- or underflows.
        # (No translation re-base: stab_dir is a direction, a translated plane is not a rotated one. stab_epoch
        # stays 0.)
        slot.S = slot.S / np.linalg.norm(slot.S)
        self.stats["reg_a"] = info_a
        if ib is None:
            self._push(slot)
            self.timings_ms = {"prep": (t_prep - t0) * 1e3, "register": (time.perf_counter() - t_prep) * 1e3,
                               "total": (time.perf_counter() - t0) * 1e3}
            return []
        B = self._slots[ib]
        m_b, info_b = self._register(slot, B, imu, oracle)
        self.stats["reg_b"] = info_b
        self.stats["refs"] = 2
        t_reg = time.perf_counter()
        # ---- kernel
        Lw, Lwi = self._lw, self._lwi
        cell = max(2, int(vc.cfar_cell_px))
        if nat is not None:
            D, binary, thr, mD, sig = nat.kernel(slot, A, B, Lw @ m_a @ Lwi, Lw @ m_b @ Lwi, info_a, info_b)
        else:
            oa = block_offsets(Lw @ m_a @ Lwi, *self._wshape, extra=self._field_extra(info_a["field"], H, W))
            ob = block_offsets(Lw @ m_b @ Lwi, *self._wshape, extra=self._field_extra(info_b["field"], H, W))
            D, binary, thr, mD, sig = motion_kernel(
                slot.work, A.work, B.work, self._srcv_w, self._mask_d, oa[:2], ob[:2], info_a["gq"], info_b["gq"],
                max(1, int(vc.envelope_px)), cell, float(vc.t_min_dn), float(vc.noise_k), float(vc.cfar_gain),
                float(vc.texture_gain), self._bufs)
        self.noise_sigma = sig
        if float(getattr(vc, "slow_gain", 0.0) or 0.0) > 1.0:
            self._slow_pass(slot, B, D, binary, sig)
        t_ker = time.perf_counter()
        # ---- components
        dets: list[Detection] = []
        rows = nat.measure(slot) if nat is not None else None
        if rows is not None:
            # labels, moments and the per-candidate numbers in C (tbot_motion_measure): Detection objects here
            n_lab, n_cand, out = rows
            n = 0
            if n_lab > 0:
                self.stats["components"] = n_lab - 1
                self.stats["candidates"] = n_cand
            t_cc = time.perf_counter()
            reg_ok = bool(info_a.get("ok") and info_b.get("ok"))
            ts = int(slot.t)
            minor = []
            KS = (self._Kinv @ slot.S).tolist() if out else None
            for r in out:
                dets.append(Detection(
                    t_ns=ts, u=r[0], v=r[1], area_px=r[2], diameter_px=r[3], contrast=r[4], snr=r[5], score=r[6],
                    bbox=(int(r[7]), int(r[8]), int(r[9]), int(r[10])), stab=(r[11], r[12]), mass=r[13],
                    reg_ok=reg_ok, stab_dir=_stab_dir(KS, r[0], r[1])))
                minor.append(r[14])
            if minor:
                self.stats["minor_px"] = minor
            x = y = bw = bh = 0
        else:
            x, y, bw, bh = cv2.boundingRect(binary)
        n = 0
        if bw > 0 and bh > 0:
            # components of the thresholded pixels with 1-px gaps closed (a 3x3 max first): one object whose
            # residual is split by a thin line of sub-threshold pixels is one candidate. Measured on the
            # thresholded pixels only.
            hw, ww = binary.shape
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(ww, x + bw + 1), min(hw, y + bh + 1)
            roi = binary[y0:y1, x0:x1]
            joined = cv2.dilate(roi, _rect(3))
            n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
            self.stats["components"] = int(n) - 1
            x, y = x0, y0
        if rows is None:
            t_cc = time.perf_counter()
        if n > 1:
            dets = self._measure(slot, lab, st, x, y, D, mD, cell, info_a, info_b, roi)
        self._push(slot)
        if self.keep_debug:
            self.debug = {"D": D.copy(), "binary": binary.copy(), "warped_a": self._bufs.wa.copy()}
        t_end = time.perf_counter()
        self.timings_ms = {"prep": (t_prep - t0) * 1e3, "register": (t_reg - t_prep) * 1e3,
                           "kernel": (t_ker - t_reg) * 1e3, "components": (t_cc - t_ker) * 1e3,
                           "measure": (t_end - t_cc) * 1e3, "total": (t_end - t0) * 1e3}
        return dets

    def _slow_pass(self, slot: _Slot, B: _Slot, D: np.ndarray, binary: np.ndarray, sig: float) -> None:
        """The slow-mover test, merged into D and binary in place (vision.slow_gain > 1).

        A target that moves less than its own size against the scenery between reference A and now (a clay going
        away at 2-5 deg/s once the gun steadies: 52 % of such frames on sky are seen, 90 % of faster ones) cancels
        itself in the A/B test. Here the current frame is compared with B and with C, an older frame about
        slow_gain x B's lag back (at least slow_min_extra_ms older than B): the same envelope residual, so a mover
        must differ from both and leaves no ghost. C is laid over the current frame through the stabilised plane
        (C.S^-1 S: the registrations chained since C, no search of its own); where that is not good to the pixel
        (texture, parallax over a quarter of a second) the residual is high everywhere around and this test's own
        CFAR bar rises with it, so it adds what open sky and flat ground show. Only pixels over its own bar and away
        from the A/B test's own candidates are added (D raised to the slow residual there).

        Off by default (vision.slow_gain = 0): on the 68 recorded presses it lifts what is seen (sky 71 -> 74 % of
        the frames, targets under 4 px 27 -> 38 %) but hits the same virtual clays and fires on the same targets,
        and on the Pi 3B it costs 10 ms a frame through these primitives (a second full-frame residual pass)."""
        vc = self.vc
        g = float(vc.slow_gain)
        t = int(slot.t)
        lag_b = t - int(B.t)
        if lag_b <= 0:
            return
        want = t - int(g * lag_b)
        newest = int(B.t) - int(float(getattr(vc, "slow_min_extra_ms", 60.0)) * 1e6)
        C, best = None, None
        for k in self._order:
            s_ = self._slots[k]
            if s_ is slot or s_.t > newest or s_.S is None:
                continue
            d = abs(int(s_.t) - want)
            if best is None or d < best:
                C, best = s_, d
        if C is None:
            return
        try:
            m_c = np.linalg.inv(C.S) @ slot.S
        except np.linalg.LinAlgError:
            return
        if not np.isfinite(m_c).all() or abs(m_c[2, 2]) < 1e-12:
            return
        m_c = m_c / m_c[2, 2]
        hw, ww = self._wshape
        sb = getattr(self, "_slow_bufs", None)
        if sb is None or sb["wc"].shape != (hw, ww):
            sb = self._slow_bufs = {
                "wc": np.empty((hw, ww), np.uint8), "vc": np.empty((hw, ww), np.uint8),
                "Ds": np.empty((hw, ww), np.uint8), "bs": np.empty((hw, ww), np.uint8),
                "valid": np.empty((hw, ww), np.uint8), "gather": _GatherIdx(hw, ww, WARP_BLOCK),
                "env": {k: np.empty((hw, ww), np.uint8) for k in ("hi", "lo", "t1", "t2")}}
        bufs = self._bufs
        wc, vcv, Ds, bs, valid = sb["wc"], sb["vc"], sb["Ds"], sb["bs"], sb["valid"]
        cur = slot.work
        env = max(1, int(vc.envelope_px))
        cell = max(2, int(vc.cfar_cell_px))
        # the C primitives of the same steps (each held equal to its NumPy reference: tests/test_native_motion.py)
        K = self._nat.K if self._nat is not None else None
        mm = np.ascontiguousarray(self._lw @ m_c @ self._lwi, dtype=np.float64)
        if K is not None:
            nb = (ww + WARP_BLOCK - 1) // WARP_BLOCK
            ch, cw = -(-hw // cell), -(-ww // cell)
            nk = sb.get("nat")
            if nk is None or nk["thr"].shape != (ch, cw):
                nk = sb["nat"] = {
                    "dx": np.zeros((hw, nb), np.int32), "dy": np.zeros((hw, nb), np.int32),
                    "rx": np.zeros((hw, nb)), "ry": np.zeros((hw, nb)), "res": np.zeros(2),
                    "ab": np.zeros(2), "gq": np.zeros(2, np.int32), "scratch": np.empty(5 * hw * ww, np.uint8),
                    "v01": np.empty((hw, ww), np.uint8), "SA": np.zeros((ch, cw), np.int64),
                    "NA": np.zeros((ch, cw), np.int64), "SD": np.zeros((ch, cw), np.int64),
                    "ND": np.zeros((ch, cw), np.int64), "thr": np.zeros((ch, cw), np.uint8),
                    "mD": np.zeros((ch, cw))}
            A_ = lambda a: a.ctypes.data  # noqa: E731
            K.block_offsets(A_(mm), 3, hw, ww, WARP_BLOCK, None, None, A_(nk["dx"]), A_(nk["dy"]), A_(nk["rx"]),
                            A_(nk["ry"]), A_(nk["res"]), K.np_chunk)
            K.warp(A_(C.work), A_(self._srcv_w), hw, ww, A_(nk["dx"]), A_(nk["dy"]), WARP_BLOCK, A_(wc), A_(vcv))
        else:
            dx, dy, _ = block_offsets(mm, hw, ww)
            warp_gather(C.work, self._srcv_w, dx, dy, WARP_BLOCK, wc, vcv, sb["gather"])
        cv2.bitwise_and(bufs.vb, vcv, dst=valid)
        cv2.bitwise_and(valid, self._mask_d, dst=valid)
        if K is not None:
            K.photometric(A_(cur), A_(wc), A_(valid), hw, ww, BLACK_DN, GAIN_MIN, GAIN_MAX, FIT_STD_MIN, A_(nk["ab"]),
                          A_(nk["gq"]))
            K.gain(A_(wc), hw * ww, int(nk["gq"][0]), int(nk["gq"][1]), A_(wc))
            K.envelope(A_(cur), A_(bufs.wb), A_(wc), A_(bufs.vb), A_(vcv), A_(self._mask_d), hw, ww, env,
                       A_(nk["scratch"]), A_(Ds))
        else:
            _, _, gq = photometric_fit(cur, wc, valid)
            apply_gain_q8(wc, gq[0], gq[1], wc)
            envelope_residual(cur, bufs.wb, wc, bufs.vb, vcv, self._mask_d, env, sb["env"], Ds)
        # edges of C are not open sky: where C's own envelope is wide (a tree line, a roof, a wire: laid over through
        # the chained registrations it is off by a pixel or two, and its residual is an edge-thin line that a
        # cell-mean bar does not raise) nothing is added. A target is not in C where it is now: C there shows what
        # is behind it.
        k = _rect(env)
        hi, lo = sb["env"]["hi"], sb["env"]["lo"]
        cv2.dilate(wc, k, dst=hi)
        cv2.erode(wc, k, dst=lo)
        cv2.subtract(hi, lo, dst=hi)
        wmax = int(min(255, math.ceil(float(getattr(vc, "slow_edge_k", 2.0)) *
                                      max(float(vc.t_min_dn), float(vc.noise_k) * (sig if math.isfinite(sig) else 0.0)))))
        cv2.compare(hi, wmax, cv2.CMP_LE, dst=lo)
        cv2.bitwise_and(Ds, lo, dst=Ds)
        prm = (float(vc.t_min_dn), float(vc.noise_k), float(vc.cfar_gain), float(vc.texture_gain))
        done = False
        if K is not None:
            SA, NA, SD, ND, thr = nk["SA"], nk["NA"], nk["SD"], nk["ND"], nk["thr"]
            K.absdiff_sums(A_(cur), A_(wc), A_(valid), hw, ww, cell, NOISE_ROW_STEP, A_(SA), A_(NA))
            K.cell_sums(A_(Ds), hw, ww, cell, 1, A_(SD))
            cv2.min(valid, 1, dst=nk["v01"])
            K.cell_sums(A_(nk["v01"]), hw, ww, cell, 1, A_(ND))
            np.maximum(ND, 1, out=ND)
            if K.threshold(A_(SD), A_(ND), A_(SA), A_(NA), ch, cw, sig, *prm, ABS_TO_SIGMA, CFAR_BOX, A_(thr),
                           A_(nk["mD"])) == 0:
                K.compare(A_(Ds), hw, ww, A_(thr), cw, cell, A_(bs))
                done = True
        if not done:
            SA, NA = cell_absdiff_sums(cur, wc, valid, cell)
            SD = cell_sums(Ds, cell)
            ND = np.maximum(cell_sums((valid != 0).view(np.uint8), cell), 1)
            thr, _ = threshold_cells(SD, ND, SA, NA, sig, *prm, cell)
            compare_cells(Ds, thr, cell, bs)
        # only what the A/B test did not see: nothing within SLOW_CLEAR_PX of its candidates (their measurement, and
        # so every track the A/B test feeds, stays exactly as it was)
        if cv2.countNonZero(binary):
            near = cv2.dilate(binary, _rect(2 * SLOW_CLEAR_PX + 1))
            cv2.bitwise_not(near, dst=near)
            cv2.bitwise_and(bs, near, dst=bs)
        n = int(cv2.countNonZero(bs))
        self.stats["slow"] = {"lag_ms": round((t - int(C.t)) * 1e-6, 1), "px": n}
        if n:
            np.copyto(D, np.maximum(D, Ds), where=bs != 0)
            cv2.bitwise_or(binary, bs, dst=binary)

    def to_stab(self, u: float, v: float) -> tuple[float, float] | None:
        """Full-res pixel (u, v) of the LAST frame given to detect() in background-stabilised pixels."""
        if not self._order:
            return None
        return _apply(self._slots[self._order[-1]].S, u, v)

    # ------------------------------------------------------------- measure
    def _measure(self, slot, lab, st, rx, ry, D, mD, cell, info_a, info_b, roi) -> list[Detection]:
        vc = self.vc
        s = self.scale
        h, w = lab.shape
        nl = st.shape[0]
        pts = cv2.findNonZero(roi).reshape(-1, 2)          # thresholded pixels only
        L = lab[pts[:, 1], pts[:, 0]]
        areas = np.bincount(L, minlength=nl)
        okl = (areas >= int(vc.min_area_px)) & (areas <= int(vc.max_area_px))
        okl[0] = False
        keep = np.nonzero(okl)[0]
        self.stats["candidates"] = int(keep.size)
        if keep.size == 0:
            return []
        sel = okl[L]
        pts, L = pts[sel], L[sel]
        px = pts[:, 0].astype(np.int64)
        py = pts[:, 1].astype(np.int64)
        gx = px + rx
        gy = py + ry
        wgt = D[gy, gx].astype(np.int64)
        bx0 = np.full(nl, 1 << 30)
        by0 = np.full(nl, 1 << 30)
        bx1 = np.full(nl, -1)
        by1 = np.full(nl, -1)
        np.minimum.at(bx0, L, gx)
        np.minimum.at(by0, L, gy)
        np.maximum.at(bx1, L, gx)
        np.maximum.at(by1, L, gy)
        # integer moment sums per component (C port: int64 sums over the component's pixels)
        s0 = np.bincount(L, weights=wgt, minlength=nl)
        sx = np.bincount(L, weights=wgt * gx, minlength=nl)
        sy = np.bincount(L, weights=wgt * gy, minlength=nl)
        sxx = np.bincount(L, weights=wgt * gx * gx, minlength=nl)
        syy = np.bincount(L, weights=wgt * gy * gy, minlength=nl)
        sxy = np.bincount(L, weights=wgt * gx * gy, minlength=nl)
        # peak pixel per component: largest D, ties -> first in raster order
        order = np.lexsort((-(gy * w + gx), wgt, L))
        last = np.r_[np.nonzero(np.diff(L[order]))[0], order.size - 1]
        pk = order[last]
        peak_lab = L[pk]
        masses = s0[keep]
        rank = keep[np.argsort(-masses, kind="stable")][: int(vc.max_candidates)]
        peak_of = dict(zip(peak_lab.tolist(), pk.tolist()))
        wa = self._bufs.wa
        cur = slot.work
        reg_ok = bool(info_a.get("ok") and info_b.get("ok"))
        Sm = slot.S
        KS = (self._Kinv @ Sm).tolist()
        out = []
        f_area = float(s * s)
        box = 1.0  # area-sampled pixels add their own box (var 1/12 px^2) to the blob
        for lid in rank.tolist():
            m0 = float(s0[lid])
            if m0 <= 0:
                continue
            cx, cy = sx[lid] / m0, sy[lid] / m0
            vxx = sxx[lid] / m0 - cx * cx - PIXEL_BOX_VAR * box
            vyy = syy[lid] / m0 - cy * cy - PIXEL_BOX_VAR * box
            vxy = sxy[lid] / m0 - cx * cy
            tr, det = vxx + vyy, vxx * vyy - vxy * vxy
            disc = math.sqrt(max(tr * tr * 0.25 - det, 0.0))
            lmax, lmin = tr * 0.5 + disc, max(tr * 0.5 - disc, 0.0)
            diam = max(1.0, 4.0 * math.sqrt(max(lmax, 0.0)) * s)
            k = peak_of[lid]
            qx, qy = int(gx[k]), int(gy[k])
            pkv = float(wgt[k])
            contrast = float(int(cur[qy, qx]) - int(wa[qy, qx]))
            ci, cj = qy // cell, qx // cell
            local = float(mD[ci, cj])
            u = (cx + 0.5) * s - 0.5
            v = (cy + 0.5) * s - 0.5
            st_u, st_v = _apply(Sm, u, v)
            mass = m0 * f_area
            bx, by = int(bx0[lid]), int(by0[lid])
            d = Detection(
                t_ns=int(slot.t), u=float(u), v=float(v), area_px=float(areas[lid]) * f_area,
                diameter_px=float(diam), contrast=contrast, snr=float(pkv / max(local, 1.0)),
                score=float(mass / (mass + SCORE_MASS)),
                bbox=(bx * s, by * s, int(bx1[lid] - bx + 1) * s, int(by1[lid] - by + 1) * s),
                stab=(float(st_u), float(st_v)), mass=float(mass), reg_ok=reg_ok, stab_dir=_stab_dir(KS, float(u), float(v)))
            d_minor = max(1.0, 4.0 * math.sqrt(lmin) * s)
            out.append(d)
            self.stats.setdefault("minor_px", []).append(d_minor)
        return out


# ============================================================================================ test oracle
def _phase_oracle(cur_l: np.ndarray, ref_l: np.ndarray, minv_l: np.ndarray) -> np.ndarray:
    """Reference registration (NOT the shipped path): warpAffine of ref by the prior + phase correlation.
    Returns the translation correction in ref pixels of this level, comparable with _register's delta."""
    h, w = cur_l.shape
    r = cv2.warpAffine(ref_l.astype(np.float32), np.ascontiguousarray(minv_l[:2]), (w, h),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    c = cur_l.astype(np.float32)
    han = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (sx, sy), _resp = cv2.phaseCorrelate(r - r.mean(), c - c.mean(), han)
    # phaseCorrelate: c(p) ~ r(p - shift); r(q) = ref(minv q) => c(p) = ref(minv(p - shift)) ~ ref(minv p - shift)
    return -np.array([sx, sy])


# ============================================================================================== press tools
def _press_objects(press_dir, cfg=None):
    from tbot.config import Calibration, load_config  # noqa: PLC0415
    from tbot.optics import CameraModel, MountModel, combined_mask  # noqa: PLC0415
    from tbot.tools.score import Session, rig_config  # noqa: PLC0415
    press = Path(press_dir)
    if cfg is None:
        cfg = rig_config(load_config(), press)
    meta = json.loads((press / "meta.json").read_text())
    cal = meta.get("calibration") or {}
    w, h = int(meta["width"]), int(meta["height"])
    cam = CameraModel.from_calibration(cal.get("intrinsics"), cfg, w, h)
    mount = MountModel.from_config(cfg, cal.get("boresight"))
    calib = Calibration(dir=press, intrinsics=cal.get("intrinsics"), boresight=cal.get("boresight"))
    mask = combined_mask(cam, mount, cfg, calib)
    session = Session(press, cfg)
    return cfg, cam, mount, mask, session


def iter_press(press_dir, cfg=None, start: int = 0, stop: int | None = None, imu: bool = True,
               oracle: bool = False):
    """Yields (row, frame, detections, detector) for every frame of a press from row 0 (the ring needs the
    frames before `start`; detections before `start` are computed and not yielded)."""
    from tbot.camera.replay import ReplaySource  # noqa: PLC0415
    cfg, cam, mount, mask, session = _press_objects(press_dir, cfg)
    det = MotionDetector(cfg, mask, cam, mount)
    src = ReplaySource(press_dir)
    first = max(0, start - 12)
    k = 0
    while True:
        fr = src.read()
        if fr is None or (stop is not None and k >= stop):
            break
        if k >= first:
            R = session.R(fr.t_exposure_ns) if imu else None
            dets = det.detect(fr, R, oracle=oracle and k >= start)
            if k >= start:
                yield k, fr, dets, det
        k += 1


def run_press(press_dir, cfg=None, start: int = 0, stop: int | None = None) -> list[list[Detection]]:
    """Detections of every frame start..stop-1 of a recorded press (attitude re-estimated from its IMU)."""
    return [d for _, _, d, _ in iter_press(press_dir, cfg, start, stop)]


def _telemetry_dets(press: Path) -> dict:
    out = {}
    p = press / "telemetry.jsonl"
    if not p.exists():
        return out
    for i, line in enumerate(p.read_text().splitlines()):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[rec.get("seq", i)] = rec.get("dets") or []
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tbot.vision.motion", description=__doc__.splitlines()[0])
    ap.add_argument("press")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--scale", type=float, default=0.5, help="tile scale of the contact sheet")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="compare registration with phase correlation")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    from tbot.config import load_config  # noqa: PLC0415
    from tbot.tools.score import rig_config  # noqa: PLC0415
    press = Path(a.press)
    cfg = rig_config(load_config(sets=a.set), press)
    teld = _telemetry_dets(press)
    tiles, times, counts, odiff = [], {}, [], []
    for k, fr, dets, det in iter_press(press, cfg, a.start, a.stop, imu=not a.no_imu, oracle=a.oracle):
        for key, v in det.timings_ms.items():
            times.setdefault(key, []).append(v)
        if det.stats.get("refs") == 2:
            counts.append(len(dets))
        if (k - a.start) % a.step:
            continue
        ra, rb = det.stats.get("reg_a", {}), det.stats.get("reg_b", {})
        txt = ""
        for nm, r in (("A", ra), ("B", rb)):
            if r:
                txt += " %s %4.0fms pr%+6.1f,%+6.1f sh%+5.1f,%+5.1f d%.2f%s g%.2f" % (
                    nm, r["dt_ms"], *r["prior"], *r["shift"], r["depth"], "" if r["ok"] else "!", r["gain"])
                if "oracle" in r:
                    o = r["oracle"]
                    dd = (o[0] - r["shift"][0], o[1] - r["shift"][1]) if r["ok"] else (o[0], o[1])
                    odiff.append((k, nm, r["ok"], r["depth"], dd))
                    txt += " pc%+5.1f,%+5.1f" % (o[0], o[1])
        top = " ".join("%d/%d@%d,%d" % (d.mass, d.area_px, d.u, d.v) for d in dets[:4])
        if not a.quiet:
            print("%4d n=%3d sig%.2f |%s | %s" % (k, len(dets), det.noise_sigma, txt, top))
        if a.sheet:
            im = cv2.cvtColor(np.ascontiguousarray(fr.image), cv2.COLOR_GRAY2BGR) if fr.image.ndim == 2 else fr.image.copy()
            for i, d in enumerate(reversed(dets)):
                col = (0, 0, 255) if d is dets[0] else (0, 160, 255)
                x, y, w, h = d.bbox
                cv2.rectangle(im, (x - 3, y - 3), (x + w + 2, y + h + 2), col, 1)
            old = teld.get(fr.seq, [])
            for dd in old:
                cv2.circle(im, (int(dd[0]), int(dd[1])), 9, (0, 255, 255), 1)
            cv2.putText(im, "%d new %d old %d" % (k, len(dets), len(old)), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if a.scale != 1.0:
                im = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            tiles.append(im)
    for key, v in times.items():
        print("%-11s p50 %.3f ms  p95 %.3f ms" % (key, np.percentile(v, 50), np.percentile(v, 95)))
    if counts:
        print("candidates/frame median %.1f p95 %.1f max %d (%d frames)" % (
            np.median(counts), np.percentile(counts, 95), max(counts), len(counts)))
    if odiff:
        good = [np.hypot(*d[4]) for d in odiff if d[2]]
        if good:
            print("registration vs phase correlation (reg_ok pairs, full-res px): median %.2f p90 %.2f max %.2f n %d"
                  % (np.median(good), np.percentile(good, 90), max(good), len(good)))
        print("reg_ok %d / %d" % (sum(1 for d in odiff if d[2]), len(odiff)))
    if a.sheet and tiles:
        rows = []
        for r in range(0, len(tiles), a.cols):
            row = tiles[r:r + a.cols]
            while len(row) < a.cols:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        cv2.imwrite(a.sheet, np.vstack(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
