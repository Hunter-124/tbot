"""Find a bore-laser dot in camera frames (used by `tbot calib-boresight --laser`).

The dot is the brightest *compact* thing in the frame once the smooth background is removed. Two sources of
background are supported: a frame taken with the beam blocked (best: everything static cancels, so it works
in a lit room or outdoors) or, without one, a heavily blurred copy of the frame itself. The barrel is masked
out (it is glossy and reflects the beam back into the lens).

A measurement is the median of the per-frame centroids, so a single bad frame cannot move it, and the scatter
across frames says whether the gun was still.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_PEAK_DN = 12.0        # the dot must stand this far above its background (8-bit DN) ...
MIN_PEAK_SIGMAS = 8.0     # ... and this many robust noise sigmas
RIVAL_FRACTION = 0.6      # a second spot this strong (relative to the dot) makes the frame ambiguous
MAX_DOT_FRACTION = 0.08   # a "dot" wider than this fraction of the image width is not a laser dot
SATURATED_DN = 250.0
MAX_SCATTER_PX = 1.5      # frame-to-frame centroid scatter above this = the gun (or the laser) moved


class LaserDotError(ValueError):
    """No usable dot (the message says what to change)."""


@dataclass
class Dot:
    u: float
    v: float
    peak_dn: float          # height above the background
    diameter_px: float      # equivalent diameter of the half-maximum region
    saturated: bool
    sigma_dn: float


@dataclass
class DotMeasurement:
    u: float
    v: float
    n_frames: int
    n_found: int
    scatter_px: float
    peak_dn: float
    diameter_px: float
    saturated: bool
    warnings: list[str] = field(default_factory=list)


def _gray_f32(image: np.ndarray) -> np.ndarray:
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    if image.dtype == np.uint16:  # 10-bit raw in 16-bit containers -> 8-bit scale
        g *= 255.0 / max(255.0, float(g.max()))
    return g


def find_dot(image: np.ndarray, background: np.ndarray | None = None, mask: np.ndarray | None = None) -> Dot:
    """Sub-pixel centre of the laser dot in one frame.

    background: a frame with the beam blocked (same size), or None to use the frame's own smooth background.
    mask: uint8, 0 = ignore (barrel); same size as the image.
    """
    gray = _gray_f32(image)
    h, w = gray.shape
    if background is not None:
        bg = _gray_f32(background)
        if bg.shape != gray.shape:
            raise LaserDotError(f"background frame is {bg.shape[1]}x{bg.shape[0]}, the image is {w}x{h}")
    else:
        k = max(15, (w // 12) | 1)
        bg = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), k if k % 2 else k + 1).astype(np.float32)
    diff = gray - bg
    valid = np.ones((h, w), dtype=bool) if mask is None else (np.asarray(mask) != 0)
    if valid.shape != (h, w):
        raise LaserDotError("mask and image sizes differ")
    if not valid.any():
        raise LaserDotError("the mask leaves no image to search")
    diff[~valid] = 0.0

    resid = diff[valid]
    med = float(np.median(resid))
    sigma = 1.4826 * float(np.median(np.abs(resid - med))) + 0.5  # +0.5 DN: quantisation floor
    smooth = cv2.GaussianBlur(diff, (0, 0), 1.0)
    _, peak, _, loc = cv2.minMaxLoc(smooth, mask=valid.astype(np.uint8))
    need = max(MIN_PEAK_DN, MIN_PEAK_SIGMAS * sigma)
    if peak < need:
        raise LaserDotError(
            f"no laser dot: the brightest compact spot is only {peak:.0f} DN above its background (need "
            f"{need:.0f}). Is the beam on the target and in view? Raise camera.exposure_us / analogue_gain, "
            "dim the room, or use a filter-free lens (an IR-pass filter blocks a red or green laser)")

    # the dot = the connected half-maximum region around the peak
    _, labels = cv2.connectedComponents((smooth >= 0.5 * peak).astype(np.uint8), connectivity=8)
    region = labels == labels[loc[1], loc[0]]
    area = int(region.sum())
    diameter = 2.0 * math.sqrt(area / math.pi)
    if diameter > MAX_DOT_FRACTION * w:
        raise LaserDotError(
            f"the brightest region is {diameter:.0f} px wide: that is glare or a lamp, not a laser dot. "
            "Lower camera.exposure_us, or take a background with the beam blocked (--laser-bg)")

    # anything comparably bright elsewhere makes the choice a guess
    grown = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=int(max(4, round(1.5 * diameter)))) > 0
    rival = float(smooth[valid & ~grown].max()) if (valid & ~grown).any() else 0.0
    if rival >= RIVAL_FRACTION * peak:
        raise LaserDotError(
            f"ambiguous: a second spot is {100.0 * rival / peak:.0f} % as bright as the brightest one "
            "(reflection, lamp or window). Take a background with the beam blocked (--laser-bg), lower the "
            "exposure, or move the target away from shiny surfaces")

    # intensity-weighted centroid of the region, grown by one pixel so the wings count
    win = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    wgt = np.where(win & valid, np.clip(diff, 0.0, None), 0.0).astype(np.float64)
    total = float(wgt.sum())
    if total <= 0.0:
        raise LaserDotError("no laser dot (empty centroid window)")
    ys, xs = np.nonzero(wgt)
    u = float((wgt[ys, xs] * xs).sum() / total)
    v = float((wgt[ys, xs] * ys).sum() / total)
    return Dot(u, v, float(peak), float(diameter), bool(float(gray[region].max()) >= SATURATED_DN), sigma)


def measure_dot(frames: list[np.ndarray], background: np.ndarray | None = None,
                mask: np.ndarray | None = None) -> DotMeasurement:
    """Median dot position over several frames of a still gun."""
    if not frames:
        raise LaserDotError("no frames")
    dots: list[Dot] = []
    last_error: LaserDotError | None = None
    for image in frames:
        try:
            dots.append(find_dot(image, background, mask))
        except LaserDotError as e:
            last_error = e
    if len(dots) < max(1, (len(frames) + 1) // 2):
        raise LaserDotError(f"dot found in only {len(dots)} of {len(frames)} frames: {last_error}")
    uv = np.array([[d.u, d.v] for d in dots], dtype=np.float64)
    centre = np.median(uv, axis=0)
    scatter = float(np.sqrt(np.mean(np.sum((uv - centre) ** 2, axis=1)))) if len(dots) > 1 else 0.0
    m = DotMeasurement(
        u=float(centre[0]), v=float(centre[1]), n_frames=len(frames), n_found=len(dots), scatter_px=scatter,
        peak_dn=float(np.median([d.peak_dn for d in dots])),
        diameter_px=float(np.median([d.diameter_px for d in dots])),
        saturated=bool(np.mean([d.saturated for d in dots]) > 0.5),
    )
    if scatter > MAX_SCATTER_PX:
        m.warnings.append(f"dot moved {scatter:.1f} px rms between frames: the gun or the laser is not still")
    if m.saturated and m.diameter_px > 6.0:
        m.warnings.append(f"dot is saturated and {m.diameter_px:.0f} px wide: lower camera.exposure_us for a "
                          "tighter centre")
    if len(dots) < len(frames):
        m.warnings.append(f"dot found in {len(dots)} of {len(frames)} frames")
    return m


def circle_centre(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and rms radius of the dots seen while the laser is turned in the bore.

    A bore laser's beam is never exactly on the bore axis: turning it in the bore makes the dot trace a
    circle whose centre is the bore line. With evenly spaced turns the mean is that centre; with three or
    more points an algebraic circle fit also handles uneven spacing.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mean = pts.mean(axis=0)
    centre = mean
    if len(pts) >= 3:
        d = pts - mean
        A = np.column_stack([2.0 * d, np.ones(len(d))])
        b = (d * d).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        fit = mean + sol[:2]
        spread = float(np.sqrt((d * d).sum(axis=1).mean()))
        # a near-collinear set gives a wild centre: keep the fit only when it stays inside the cloud
        if np.all(np.isfinite(fit)) and float(np.linalg.norm(fit - mean)) <= spread:
            centre = fit
    radius = float(np.sqrt(((pts - centre) ** 2).sum(axis=1).mean())) if len(pts) > 1 else 0.0
    return centre, radius


def annotate(image: np.ndarray, dots: list[tuple[float, float]], centre: tuple[float, float],
             predicted: tuple[float, float] | None = None) -> np.ndarray:
    """Debug picture: each measured dot (green), the solved bore point (magenta), the previous prediction (cyan)."""
    g = np.clip(_gray_f32(image), 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    scale = 3 if vis.shape[1] <= 800 else 1
    if scale > 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def pt(p):
        return int(round((p[0] + 0.5) * scale - 0.5)), int(round((p[1] + 0.5) * scale - 0.5))

    for d in dots:
        cv2.circle(vis, pt(d), 4 * scale, (0, 255, 0), 1, cv2.LINE_AA)
    if predicted is not None:
        cv2.drawMarker(vis, pt(predicted), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14 * scale, 1, cv2.LINE_AA)
    cv2.drawMarker(vis, pt(centre), (255, 0, 255), cv2.MARKER_CROSS, 22 * scale, 1, cv2.LINE_AA)
    cv2.putText(vis, "green: laser dot(s)   magenta: bore   cyan: previous calibration", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
