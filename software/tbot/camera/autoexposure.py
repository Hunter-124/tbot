"""Auto-exposure for the gun camera (OV9281 through libcamera), from dense woods at dusk to an open field at noon.

Why not libcamera's AE: it trades exposure time freely up to the frame period and meters for a pleasing picture.
This controller meters for the clay detector and knows what each setting costs it.

What is metered: the HIGHLIGHTS. The 98th percentile of the frame (the sky, or the brightest thing there is) is
held at `ae_target_dn`. A clipped sky hides a clay completely; a dark tree line still shows one, because the
image is tone-curved (shadows are stretched). So the bright end is pinned, with about two stops of headroom above
it, and the shadows fall where they fall. A dark barrel or a patch of ground cannot pull the exposure up, and a
glint or the sun's disc (under 2 % of the frame) cannot pull it down.

How light is bought (the "program line", see `split`), cheapest first:
  1. exposure time up to `ae_max_exposure_us` - the motion-blur budget. Free.
  2. analogue gain up to `ae_knee_gain` - a little noise.
  3. exposure time beyond the blur budget, up to `ae_dim_exposure_us` - blur, but 4x the light of step 4.
  4. analogue gain up to `ae_max_gain` - noise.

How it moves. The camera gun-swings: the view changes by a whole frame width in the time one exposure command
takes to act (measured 7 frames = 140 ms at 50 fps, see MEASURED below). So:
  * Every frame is turned into "the exposure this scene wants" from the image AND the settings that frame was
    actually taken with (libcamera reports them per frame), through the sensor's real tone curve. That number
    does not depend on commands still on their way, so the controller can jump straight to the answer in one
    command and cannot overshoot or oscillate on the control delay.
  * Too bright is corrected at once (a clipped frame is blind); a clipped frame steps down hard, by more the
    more of it is clipped, and is refined when that lands.
  * Too dark is corrected slowly: nothing for `ae_hold_s`, then at `ae_up_stops_per_s`. Under-exposure is
    harmless for a second or two, and this is what keeps the sky's exposure while the gun passes over trees and
    comes back to the sky. A black frame (lens covered, walked indoors) is not waited on.
  * Small errors (inside the deadband) are left alone, so a steady scene gets no commands at all.

MEASURED on the rig, 2026-11-16 (Pi 3B, libcamera 0.8, picamera2 0.3.39, 640x400 10-bit, YUV420 main stream):
  * Control delay: set_controls() acts on the frame `buffer_count + 3` later (7 frames with the default 4
    buffers; 8 at 120 fps), exposure and gain together. Never 2-3.
  * Tone curve: the ISP applies the tuning file's rpi.contrast gamma (OV9281_GAMMA below, checked against an
    exposure sweep within 3 %). DN 128 is 18 % of full scale, DN 160 is 26 %. Doubling the exposure does NOT
    double the DN, which is why this module works in linear units (`ToneCurve`).
  * Exposure time is in sensor lines of ~9.5 us (9, 19, 28, 38 ...); libcamera makes up the remainder with
    DigitalGain (reported per frame, up to 2x), so the product exposure x gain x digital gain is continuous.
    Minimum 9 us. Analogue gain 1 .. 15.94 in 1/16 steps.
  * Noise against gain at the same light: gain 1 -> 16 raises the signal 10.7x and the noise 5x: more gain never
    lowers the signal-to-noise ratio, it only raises both. Hence gain goes all the way up before the image is
    left dark.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import cv2
import numpy as np

__all__ = ["AeSettings", "AeStatus", "AutoExposure", "Meter", "ToneCurve", "meter", "OV9281_GAMMA"]

# rpi.contrast gamma_curve of libcamera's ov9281_mono.json (16-bit linear in -> 16-bit out, piecewise linear).
OV9281_GAMMA: tuple[int, ...] = (
    0, 0, 1024, 5040, 2048, 9338, 3072, 12356, 4096, 15312, 5120, 18051, 6144, 20790, 7168, 23193,
    8192, 25744, 9216, 27942, 10240, 30035, 11264, 32005, 12288, 33975, 13312, 35815, 14336, 37600,
    15360, 39168, 16384, 40642, 18432, 43379, 20480, 45749, 22528, 47753, 24576, 49621, 26624, 51253,
    28672, 52698, 30720, 53796, 32768, 54876, 36864, 57012, 40960, 58656, 45056, 59954, 49152, 61183,
    53248, 62355, 57344, 63419, 61440, 64476, 65535, 65535)

CLIP_MARGIN_DN = 5.0     # a pixel within this of white is clipped
BLACK_MARGIN_DN = 8.0    # highlights within this of black: the frame is black, its level says nothing
BLACK_JUMP = 16.0        # exposure step up from a black frame (4 stops; refined when it lands)
# Exposure step down from a clipped frame, by the fraction of it that is clipped: (fraction, divisor) knots,
# log-interpolated. A fully white frame says nothing about how far over it is, and a step that lands dark is
# still a measurement (down to 40x under the target) while one that lands white is another blind round trip.
CLIP_STEPS = ((0.02, 3.0), (0.5, 8.0), (1.0, 16.0))
DEADBAND = 0.12          # relative exposure error that is left alone
DEADBAND_INNER = 0.04    # a slow correction, once started, runs until the error is this small
FAST_OVER = 1.5          # over-exposed by more than this factor: corrected at once
SLOW_TAU_S = 0.15        # smoothing of the wanted exposure for slow corrections (rides out lamp flicker)
OVER_HOLD_S = 0.15       # a mild over-exposure (under FAST_OVER) is corrected after this long
SLEW_PERIOD_S = 0.08     # slow corrections are sent no more often than this
SLEW_MAX_DT_S = 0.25     # a slow step never covers more time than this (after a pause in the frames)
ACQUIRE_TIMEOUT_S = 1.0  # after a jump, corrections stay immediate until it has landed, or this long
LANDED_TOL = 0.08        # applied exposure within this of the commanded one = the command has landed
METER_DOWN = 4           # the frame is block-averaged by this before metering (noise and specks drop out)


class ToneCurve:
    """Output DN <-> linear sensor signal (fraction of full scale) for an 8-bit stream.

    `gamma_points`: the ISP tone curve as flat (in, out, in, out, ...) values, any common scale (libcamera's
    rpi.contrast gamma_curve). `black_dn` / `white_dn`: the stream's range (0 / 255 full range, 16 / 235 video).
    """

    def __init__(self, gamma_points: Sequence[float] = OV9281_GAMMA, black_dn: float = 0.0,
                 white_dn: float = 255.0) -> None:
        pts = np.asarray(gamma_points, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2 or not (np.all(np.diff(pts[:, 0]) > 0) and np.all(np.diff(pts[:, 1]) > 0)):
            raise ValueError("tone curve needs at least two points, strictly increasing in and out")
        if not 0 <= black_dn < white_dn <= 255:
            raise ValueError(f"tone curve needs 0 <= black_dn < white_dn <= 255, got {black_dn}, {white_dn}")
        self._x = (pts[:, 0] - pts[0, 0]) / (pts[-1, 0] - pts[0, 0])
        self._y = (pts[:, 1] - pts[0, 1]) / (pts[-1, 1] - pts[0, 1])
        self.black_dn, self.white_dn = float(black_dn), float(white_dn)
        self._dn = np.arange(256, dtype=np.float64)
        self.lut = np.interp(np.clip((self._dn - black_dn) / (white_dn - black_dn), 0.0, 1.0), self._y, self._x)
        self._lut = [float(v) for v in self.lut]  # plain floats: linear_of runs once per frame on a Pi 3B

    @classmethod
    def linear(cls, black_dn: float = 0.0, white_dn: float = 255.0) -> "ToneCurve":
        return cls((0.0, 0.0, 1.0, 1.0), black_dn, white_dn)

    def linear_of(self, dn: float) -> float:
        """Linear signal (0..1) at an output level, fractional DN allowed."""
        dn = min(max(float(dn), 0.0), 255.0)
        k = min(int(dn), 254)
        return self._lut[k] + (dn - k) * (self._lut[k + 1] - self._lut[k])

    def dn_of(self, linear: float) -> float:
        """Output level of a linear signal (0..1)."""
        out = float(np.interp(min(max(linear, 0.0), 1.0), self._x, self._y))
        return self.black_dn + out * (self.white_dn - self.black_dn)


@dataclasses.dataclass(frozen=True)
class AeSettings:
    target_dn: float = 160.0        # where the highlight percentile is held (output DN)
    highlight_pct: float = 98.0     # the metered percentile: brighter areas smaller than the rest may clip
    min_exposure_us: float = 9.0
    max_exposure_us: float = 500.0  # motion-blur budget: exposure stays below this while gain <= knee_gain
    knee_gain: float = 4.0          # gain reached before the exposure is allowed past the blur budget
    dim_exposure_us: float = 4000.0  # longest exposure in dim light (a 20 m/s clay moves its own width in 5.4 ms)
    min_gain: float = 1.0
    max_gain: float = 15.9
    every: int = 1                  # meter every n-th frame
    hold_s: float = 0.5             # an under-exposed scene is left alone this long (trees between two skies)
    up_stops_per_s: float = 1.5     # then the exposure rises at this rate
    frame_s: float = 0.02           # frame period, used only when update() is not given a time

    @classmethod
    def from_camera(cls, cam) -> "AeSettings":
        return cls(target_dn=float(cam.ae_target_dn), highlight_pct=float(cam.ae_highlight_pct),
                   min_exposure_us=float(cam.ae_min_exposure_us), max_exposure_us=float(cam.ae_max_exposure_us),
                   knee_gain=float(cam.ae_knee_gain), dim_exposure_us=float(cam.ae_dim_exposure_us),
                   min_gain=float(cam.ae_min_gain), max_gain=float(cam.ae_max_gain), every=int(cam.ae_every),
                   hold_s=float(cam.ae_hold_s), up_stops_per_s=float(cam.ae_up_stops_per_s),
                   frame_s=1.0 / max(float(cam.fps), 1e-3))


@dataclasses.dataclass(frozen=True)
class Meter:
    level_dn: float     # the highlight percentile, output DN (interpolated inside its histogram bin)
    level_lin: float    # the same as linear signal, 0..1
    clip_frac: float    # fraction of the (block-averaged) frame at white
    median_dn: float


@dataclasses.dataclass(frozen=True)
class AeStatus:
    """What the controller made of the last metered frame."""

    state: str          # ok | adjusting | too_bright | too_dark
    level_dn: float     # metered highlight level
    clip_pct: float     # percent of the frame at white
    error_stops: float  # exposure the scene wants over the commanded one (+ = wants more). On a clipped or
                        # black frame the true error is unknown: this is the step taken, a lower bound
    exposure_us: float  # commanded
    gain: float

    def message(self) -> str:
        """One line for a log; empty when there is nothing to say."""
        if self.state == "too_bright":
            return (f"scene too bright for the camera: {self.clip_pct:.0f} % of the frame is clipped at the "
                    f"shortest exposure ({self.exposure_us:.0f} us, gain {self.gain:.1f}), at least "
                    f"{abs(self.error_stops):.1f} stops over - fit an ND filter or a smaller aperture")
        if self.state == "too_dark":
            under = f"{self.error_stops:.1f} stops under" if self.error_stops > 0.05 else "only just exposed"
            return (f"scene too dark for the camera: {under} at the longest exposure and highest gain allowed "
                    f"({self.exposure_us:.0f} us, gain {self.gain:.1f}). In daylight: is the lens cap on or the "
                    "lens covered/fogged? Nothing can be tracked like this")
        return ""


def meter(image: np.ndarray, curve: ToneCurve, pct: float = 98.0, down: int = METER_DOWN) -> Meter:
    """Highlight percentile, clipped fraction and median of an 8-bit mono (or BGR) frame.

    The frame is block-averaged `down` x `down` first: sensor noise at high gain would otherwise push a high
    percentile up by two noise sigmas, and single hot pixels and glints would count as highlights."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., 1]  # green ~ luma; the rig is mono anyway
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    a = np.ascontiguousarray(a)
    # block average by halving (OpenCV's 2x area resize is vectorised: 0.2 ms for 640x400 -> 160x100 on a
    # Pi 3B, against 1.4 ms for one 4x step)
    while down >= 2 and min(a.shape) >= 4:
        step = 2 if down % 2 == 0 else down
        a = cv2.resize(a, (a.shape[1] // step, a.shape[0] // step), interpolation=cv2.INTER_AREA)
        down //= step
    hist = cv2.calcHist([a], [0], None, [256], [0, 256]).reshape(-1)  # float32, exact for counts below 2^24
    cdf = np.cumsum(hist)
    n = int(cdf[-1])
    if n <= 0:
        return Meter(0.0, 0.0, 0.0, 0.0)

    def level(p: float) -> float:
        want = min(max(p, 0.0), 1.0) * n
        k = int(min(np.searchsorted(cdf, want), 255))
        below = float(cdf[k - 1]) if k > 0 else 0.0
        inside = float(hist[k])
        return k - 0.5 + ((want - below) / inside if inside > 0 else 0.5)  # bin k spans k-0.5 .. k+0.5

    dn = min(max(level(pct / 100.0), 0.0), 255.0)
    clip_k = int(math.ceil(curve.white_dn - CLIP_MARGIN_DN))
    clipped = n - (int(cdf[clip_k - 1]) if clip_k > 0 else 0)
    return Meter(dn, curve.linear_of(dn), clipped / n, min(max(level(0.5), 0.0), 255.0))


class AutoExposure:
    """`update(image, applied_exposure_us, applied_gain, t_s)` -> new (exposure_us, gain) to command, or None.

    `applied_*` are the settings the frame was really taken with (per-frame metadata; pass analogue x digital
    gain as the gain). `status` describes the last metered frame."""

    def __init__(self, settings: AeSettings, exposure_us: float, gain: float,
                 curve: ToneCurve | None = None) -> None:
        s = settings
        if not (0 < s.min_exposure_us <= s.max_exposure_us and 0 < s.min_gain <= s.max_gain):
            raise ValueError(f"auto-exposure limits are inconsistent: {s}")
        self.curve = curve or ToneCurve()
        self.clip_dn = self.curve.white_dn - CLIP_MARGIN_DN
        self.black_dn = self.curve.black_dn + BLACK_MARGIN_DN
        if not (self.black_dn < s.target_dn < self.clip_dn):
            raise ValueError(f"ae_target_dn must lie between {self.black_dn:g} and {self.clip_dn:g}: {s.target_dn}")
        if not 50.0 <= s.highlight_pct < 100.0:
            raise ValueError(f"ae_highlight_pct must be in [50, 100): {s.highlight_pct}")
        if s.hold_s < 0 or s.up_stops_per_s <= 0:
            raise ValueError(f"ae_hold_s must be >= 0 and ae_up_stops_per_s > 0: {s}")
        self.s = s
        self.target_lin = self.curve.linear_of(s.target_dn)
        # the program line's corner points
        self._knee_gain = min(max(s.knee_gain, s.min_gain), s.max_gain)
        self._dim_exposure = max(s.dim_exposure_us, s.max_exposure_us)
        self.exposure_us, self.gain = self.split(float(exposure_us) * float(gain))
        self.total = self.exposure_us * self.gain
        self._n = 0
        self._t: float | None = None
        self._acquire = True                    # start-up: corrections are immediate, both ways
        self._acquire_t: float | None = None    # when the current acquisition began
        self._cmd_t = -math.inf                 # time of the last command
        self._slow: float | None = None         # smoothed log of the wanted exposure
        self._err_since: float | None = None    # since when the slow error has been outside the deadband ...
        self._err_sign = 0                      # ... on this side
        self._slewing = False
        self.last_meter: Meter | None = None
        self.status = AeStatus("adjusting", 0.0, 0.0, 0.0, self.exposure_us, self.gain)

    # ------------------------------------------------------------------ program line
    @property
    def total_min(self) -> float:
        return self.s.min_exposure_us * self.s.min_gain

    @property
    def total_max(self) -> float:
        return self._dim_exposure * self.s.max_gain

    def split(self, total: float) -> tuple[float, float]:
        """(exposure_us, gain) for an exposure x gain product: exposure to the blur budget, gain to the knee,
        exposure to the dim-light limit, gain to its maximum. Continuous and monotonic in both."""
        s = self.s
        total = min(max(total, self.total_min), self.total_max)
        if total <= s.max_exposure_us * s.min_gain:
            return max(total / s.min_gain, s.min_exposure_us), s.min_gain
        if total <= s.max_exposure_us * self._knee_gain:
            return s.max_exposure_us, total / s.max_exposure_us
        if total <= self._dim_exposure * self._knee_gain:
            return total / self._knee_gain, self._knee_gain
        return self._dim_exposure, min(total / self._dim_exposure, s.max_gain)

    # ------------------------------------------------------------------ control
    def wanted(self, m: Meter, applied_total: float) -> tuple[str, float]:
        """Pure: (kind, exposure x gain this scene wants) from a metered frame and the product it was taken at.
        kind: "ok" (the level is a measurement), "clip" / "black" (it is only a bound: a step is returned)."""
        if m.level_dn >= self.clip_dn:
            fracs, steps = zip(*CLIP_STEPS)
            return "clip", applied_total / math.exp(float(np.interp(m.clip_frac, fracs, np.log(steps))))
        if m.level_dn <= self.black_dn or m.level_lin <= 0.0:
            return "black", applied_total * BLACK_JUMP
        return "ok", applied_total * self.target_lin / m.level_lin

    def update(self, image: np.ndarray, applied_exposure_us: float, applied_gain: float,
               t_s: float | None = None) -> tuple[float, float] | None:
        self._n += 1
        if self._n % max(1, self.s.every):
            return None
        t = float(t_s) if t_s is not None else self._n * self.s.frame_s
        m = meter(image, self.curve, self.s.highlight_pct)
        applied = max(float(applied_exposure_us), 1e-3) * max(float(applied_gain), 1e-3)
        return self.step(m, applied, t)

    def step(self, m: Meter, applied_total: float, t: float) -> tuple[float, float] | None:
        """One control step from a metered frame (see the module docstring for the rules)."""
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        self.last_meter = m
        if self._acquire_t is None:
            self._acquire_t = t
        cmd = self.total
        kind, want_raw = self.wanted(m, applied_total)
        want = min(max(want_raw, self.total_min), self.total_max)
        err = math.log(want / cmd)  # > 0: the scene wants more exposure than is commanded
        band = math.log1p(DEADBAND)
        landed = abs(math.log(applied_total / cmd)) <= LANDED_TOL
        new: float | None = None

        if kind != "ok":
            # the level is only a bound; frames still on their way from before the step repeat the same bound
            if (kind == "clip" and err < -band) or (kind == "black" and err > band):
                new = want
                self._acquire, self._acquire_t = True, t
        elif self._acquire:
            if abs(err) > band:
                new = want
            elif landed:
                self._acquire = False
            if t - self._acquire_t > ACQUIRE_TIMEOUT_S:
                self._acquire = False  # e.g. lamp flicker: stop chasing every frame, track instead
        elif err < -math.log(FAST_OVER):
            new = want
        if new is not None or self._acquire or self._slow is None:
            # a jump (or still acquiring): slow tracking restarts from here
            self._slow = math.log(new if new is not None else want)
            self._slewing = False
            self._err_since, self._err_sign = None, 0
        elif kind == "ok":
            new = self._slow_step(want, cmd, t, dt)

        self._set_status(kind, m, want_raw, err, band, landed)
        if new is None:
            return None
        exposure, gain = self.split(new)
        if abs(math.log(exposure * gain / cmd)) < 0.01:
            return None  # pinned at a limit: nothing new to command
        self.exposure_us, self.gain, self.total = exposure, gain, exposure * gain
        self._cmd_t = t
        return exposure, gain

    def _slow_step(self, want: float, cmd: float, t: float, dt: float) -> float | None:
        """Tracking, small or dark-side errors: smoothed, held, then slewed."""
        s = self.s
        lw = math.log(want)
        self._slow = lw if self._slow is None else self._slow + (1.0 - math.exp(-dt / SLOW_TAU_S)) * (lw - self._slow)
        e = self._slow - math.log(cmd)
        now = lw - math.log(cmd)  # this frame's own error: the smoothed one lags a scene that has gone back
        if (abs(e) <= math.log1p(DEADBAND_INNER if self._slewing else DEADBAND)
                or (not self._slewing and (abs(now) <= math.log1p(DEADBAND) or (now > 0) != (e > 0)))):
            self._slewing = False
            self._err_since, self._err_sign = None, 0
            return None
        sign = 1 if e > 0 else -1
        if self._err_since is None or sign != self._err_sign:
            self._err_since, self._err_sign, self._slewing = t, sign, False
        if t - self._err_since < (s.hold_s if e > 0 else min(s.hold_s, OVER_HOLD_S)) and not self._slewing:
            return None
        self._slewing = True
        since = t - self._cmd_t
        if since < SLEW_PERIOD_S:
            return None
        if e < 0:
            return math.exp(self._slow)  # mildly over-exposed (under FAST_OVER): one step
        limit = s.up_stops_per_s * math.log(2.0) * min(since, SLEW_MAX_DT_S)
        return cmd * math.exp(min(e, limit))

    def _set_status(self, kind: str, m: Meter, want_raw: float, err: float, band: float, landed: bool) -> None:
        cmd = self.total
        stops = math.log2(want_raw / cmd)
        # out of range is only said of a frame taken AT the limit, not of one still on its way from before
        at_min = landed and cmd <= self.total_min * 1.02
        at_max = landed and cmd >= self.total_max / 1.02
        if at_min and (kind == "clip" or stops < -math.log2(FAST_OVER)):
            state = "too_bright"
        elif at_max and (kind == "black" or stops > -math.log2(1.0 + DEADBAND)):
            # held at the longest exposure and the highest gain, whatever the level: a covered lens lets enough light
            # through to meet the target there (backyard 2026-11-21: lens covered, 4 ms x16, level 127 of 160, AE
            # "ok", zero detections in 10 presses). Daylight needs 9-50 us at gain 1; nothing moving can be tracked
            # at 4 ms, so the limit itself is the warning, from the configured limits, not a brightness threshold
            state = "too_dark"
        elif self._acquire or self._slewing or abs(err) > band:
            state = "adjusting"
        else:
            state = "ok"
        self.status = AeStatus(state, m.level_dn, 100.0 * m.clip_frac, stops, self.exposure_us, self.gain)
