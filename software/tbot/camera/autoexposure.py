"""Software auto-exposure for the OV9281: keeps the sky well exposed as light and scenery change.

Why not libcamera's AE: it trades exposure time freely up to the frame period (20 ms at 50 fps), and a long
exposure smears a 20 m/s clay over many pixels. Here exposure TIME is capped (`ae_max_exposure_us`, the motion
blur budget) and only after it hits the cap does analogue GAIN rise (noise instead of blur).

Metering: median of the frame (subsampled) is driven toward `ae_target_dn`, but never so far that the bright
tail (99.5th percentile, the sky) passes `ae_highlight_dn` - a clipped sky hides a clay. A frame with clipped
highlights is pulled down fast.

The controller is closed-loop on the APPLIED exposure: each frame reports the ExposureTime / AnalogueGain it
was actually taken with (picamera2 metadata), so the 2-3 frame control latency of the sensor cannot make it
overshoot - it only computes from (image, the settings that produced that image).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["AeSettings", "AutoExposure", "meter"]

HIGHLIGHT_PCT = 99.5
CLIP_DN = 250.0


@dataclasses.dataclass(frozen=True)
class AeSettings:
    target_dn: float = 110.0        # median brightness goal
    highlight_dn: float = 235.0     # the 99.5th percentile must stay below this
    min_exposure_us: float = 20.0
    max_exposure_us: float = 500.0  # motion-blur budget: exposure never goes above this, gain rises instead
    min_gain: float = 1.0
    max_gain: float = 8.0
    every: int = 3                  # frames between corrections (sensor applies a change 2-3 frames later)
    deadband: float = 0.08          # relative brightness error that is left alone (no hunting)
    max_step: float = 2.0           # largest change per correction, either way

    @classmethod
    def from_camera(cls, cam) -> "AeSettings":
        return cls(target_dn=float(cam.ae_target_dn), highlight_dn=float(cam.ae_highlight_dn),
                   min_exposure_us=float(cam.ae_min_exposure_us), max_exposure_us=float(cam.ae_max_exposure_us),
                   min_gain=float(cam.ae_min_gain), max_gain=float(cam.ae_max_gain), every=int(cam.ae_every))


def meter(image: np.ndarray, step: int = 4) -> tuple[float, float]:
    """(median, 99.5th percentile) of a subsampled 8-bit mono (or BGR) frame, in DN."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., 1]  # green ~ luma; the rig is mono anyway
    sub = a[::step, ::step]
    if sub.dtype != np.uint8:
        sub = np.clip(sub, 0, 255).astype(np.uint8)
    hist = np.bincount(sub.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    n = cdf[-1]
    if n <= 0:
        return 0.0, 0.0
    med = float(np.searchsorted(cdf, 0.5 * n))
    hi = float(np.searchsorted(cdf, HIGHLIGHT_PCT / 100.0 * n))
    return med, hi


class AutoExposure:
    """`update(image, applied_exposure_us, applied_gain)` -> new (exposure_us, gain) to command, or None."""

    def __init__(self, settings: AeSettings, exposure_us: float, gain: float) -> None:
        s = settings
        if not (0 < s.min_exposure_us <= s.max_exposure_us and 0 < s.min_gain <= s.max_gain):
            raise ValueError(f"auto-exposure limits are inconsistent: {s}")
        if not (0 < s.target_dn < s.highlight_dn <= 255):
            raise ValueError(f"auto-exposure needs 0 < target_dn < highlight_dn <= 255: {s}")
        self.s = s
        self.exposure_us, self.gain = self.split(float(exposure_us) * float(gain))
        self._n = 0
        self.last_meter: tuple[float, float] | None = None

    @property
    def total_min(self) -> float:
        return self.s.min_exposure_us * self.s.min_gain

    @property
    def total_max(self) -> float:
        return self.s.max_exposure_us * self.s.max_gain

    def split(self, total: float) -> tuple[float, float]:
        """Exposure time first (up to the blur cap), then gain."""
        s = self.s
        total = min(max(total, self.total_min), self.total_max)
        exposure = min(max(total / s.min_gain, s.min_exposure_us), s.max_exposure_us)
        gain = min(max(total / exposure, s.min_gain), s.max_gain)
        return exposure, gain

    def correction(self, median: float, highlight: float) -> float:
        """Pure: multiplicative exposure change for a frame with this median / 99.5th percentile."""
        s = self.s
        k = s.target_dn / max(median, 1.0)
        k = min(k, s.highlight_dn / max(highlight, 1.0))  # never push the sky into clipping
        if highlight >= CLIP_DN:
            k = min(k, 0.5)  # clipped: the true level is unknown, step down hard
        if abs(math.log(k)) < math.log1p(s.deadband):
            return 1.0
        return min(max(k, 1.0 / s.max_step), s.max_step)

    def update(self, image: np.ndarray, applied_exposure_us: float, applied_gain: float) -> tuple[float, float] | None:
        self._n += 1
        if self._n % max(1, self.s.every):
            return None
        med, hi = meter(image)
        self.last_meter = (med, hi)
        k = self.correction(med, hi)
        if k == 1.0:
            return None
        applied = max(float(applied_exposure_us), 1e-3) * max(float(applied_gain), 1e-3)
        exposure, gain = self.split(applied * k)
        if (abs(exposure - self.exposure_us) <= 0.01 * self.exposure_us
                and abs(gain - self.gain) <= 0.01 * self.gain):
            return None  # pinned at a limit: nothing new to command
        self.exposure_us, self.gain = exposure, gain
        return exposure, gain
