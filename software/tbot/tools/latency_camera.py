"""`tbot latency-camera` — photon -> userspace latency and the camera timestamp offset.

An LED on cfg.gpio.latency_led_pin (330 ohm to GND) sits in front of the lens. The tool toggles it at
random phases to the frame clock and looks at the first frame that shows it lit. The brightness of a
partially exposed frame says where inside the exposure window the LED came on (OV9281 = global
shutter, every row integrates over the same window), which pins the sensor timestamp to the monotonic
clock that the fire solver runs on.

Results go to calib/latency.json "camera" (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np

from tbot.camera import open_source
from tbot.camera.base import SourceBase
from tbot.config import add_common_args, load_from_args, now_iso, update_json
from tbot.hw.gpio import open_gpio
from tbot.stats import summarize
from tbot.types import Frame, SimClock

# A frame counts as "lit" once the ROI level crosses half way between the measured off and on levels.
LIT_THRESHOLD = 0.5
# Fractions outside this band carry no timing information (fully dark / fully lit exposure).
PARTIAL_LO = 0.05
PARTIAL_HI = 0.95
# Below this many partial frames the timestamp offset is too noisy to trust.
MIN_PARTIAL_TRIALS = 5
# The LED must raise the ROI by at least this much (DN) over the dark level to be usable.
MIN_LED_CONTRAST_DN = 10.0
# Give up on a trial after this much wall/sim time without the expected light level.
TRIAL_TIMEOUT_S = 0.5


# ------------------------------------------------------------------------------------------ helpers
def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _roi_level(image: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """Mean grey level inside the ROI (x, y, w, h)."""
    x, y, w, h = roi
    patch = _gray(image)[y : y + h, x : x + w]
    return float(np.asarray(patch, dtype=np.float64).mean())


def parse_roi(text: str) -> tuple[int, int, int, int]:
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise ValueError(f"--roi wants x,y,w,h (got {text!r})")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"--roi width/height must be positive (got {text!r})")
    return x, y, w, h


class LedDriver:
    """Drives the latency LED and keeps the timestamp of every level change."""

    def __init__(self, gpio, pin: int, clock=None):
        self.gpio = gpio
        self.pin = int(pin)
        self.clock = clock  # set to the source's clock before the first write
        self.level = 0
        self.events: list[tuple[int, int]] = []
        gpio.setup_output(self.pin, 0)

    def now_ns(self) -> int:
        return int(self.clock.now_ns()) if self.clock is not None else time.monotonic_ns()

    def write(self, level: int) -> int:
        """Set the LED and return the timestamp taken immediately after the write."""
        level = 1 if level else 0
        self.gpio.write(self.pin, level)
        t = self.now_ns()
        self.level = level
        self.events.append((t, level))
        return t

    def level_at(self, t_ns: int) -> int:
        level = 0
        for t, lv in self.events:
            if t > t_ns:
                break
            level = lv
        return level

    def lit_fraction(self, t0_ns: int, t1_ns: int) -> float:
        """Fraction of the window [t0, t1) during which the LED was on."""
        if t1_ns <= t0_ns:
            return float(self.level_at(t0_ns))
        lit = 0
        prev_t, prev_level = t0_ns, self.level_at(t0_ns)
        for t, level in self.events:
            if t <= t0_ns:
                continue
            if t >= t1_ns:
                break
            if prev_level:
                lit += t - prev_t
            prev_t, prev_level = t, level
        if prev_level:
            lit += t1_ns - prev_t
        return lit / float(t1_ns - t0_ns)


# --------------------------------------------------------------------------------------- sim camera
class SimLedCamera(SourceBase):
    """Fake global-shutter camera that sees the LED driven through SimGpio (`--sim`).

    Frame k integrates over [t0 + k*period, +exposure]; the pixel level of the LED disc is linear in
    the fraction of that window during which the LED was on. The reported mid-exposure timestamp is
    biased by -timestamp_offset_ns (so a correct analysis recovers timestamp_offset_ns), and the frame
    reaches user space delivery_latency_ms after the end of the exposure. Like PicamSource, it adds the
    already-calibrated applied_offset_ns to every timestamp, so a re-run measures only the residual.
    """

    def __init__(
        self,
        cfg,
        led: LedDriver,
        *,
        delivery_latency_ms: float = 7.5,
        timestamp_offset_ns: int = -1_200_000,
        jitter_ms: float = 0.3,
        noise_sigma: float = 1.5,
        background_dn: float = 28.0,
        led_off_dn: float = 45.0,
        led_on_dn: float = 215.0,
        seed: int = 0,
        t0_ns: int = 1_000_000_000,
        applied_offset_ns: int = 0,
    ):
        super().__init__()
        self.width = int(cfg.camera.width)
        self.height = int(cfg.camera.height)
        self.is_color = False
        self.nominal_fps = float(cfg.camera.fps)
        self.name = "sim-led"
        self.clock = SimClock(t0_ns, frozen=True)
        self.led = led
        self.exposure_us = float(cfg.camera.exposure_us)
        self.period_ns = int(round(1e9 / self.nominal_fps))
        self.exposure_ns = int(round(self.exposure_us * 1e3))
        self.delivery_ns = int(round(delivery_latency_ms * 1e6))
        self.timestamp_offset_ns = int(timestamp_offset_ns)
        self.applied_timestamp_offset_ns = int(applied_offset_ns)
        self.jitter_ns = float(jitter_ms) * 1e6
        self.led_off_dn = float(led_off_dn)
        self.led_on_dn = float(led_on_dn)
        self._t0 = int(t0_ns)
        self._seq = 0
        self._rng = np.random.default_rng(seed)

        h, w = self.height, self.width
        static = np.full((h, w), float(background_dn), dtype=np.float64)
        # A bright but constant patch: the ROI search must key on what CHANGES, not on what is bright.
        sy, sx = h // 6, w // 6
        static[sy : sy + max(4, h // 8), sx : sx + max(4, w // 8)] = 230.0
        self._static = static
        cy, cx = int(h * 0.6), int(w * 0.65)
        radius = max(3.0, min(h, w) * 0.03)
        yy, xx = np.ogrid[:h, :w]
        self._disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius
        self.led_roi = (
            int(cx - radius - 1),
            int(cy - radius - 1),
            int(2 * radius + 3),
            int(2 * radius + 3),
        )
        # A small pool of noise fields keeps rendering cheap while staying deterministic.
        self._noise = [self._rng.normal(0.0, float(noise_sigma), (h, w)) for _ in range(8)]

    def sleep_until(self, t_ns: int) -> None:
        """Advance simulated time (the tool uses this instead of time.sleep on a simulated clock)."""
        self.clock.set(max(int(self.clock.now_ns()), int(t_ns)))

    def _render(self, fraction: float) -> np.ndarray:
        noise = self._noise[int(self._rng.integers(0, len(self._noise)))]
        img = self._static + noise
        level = self.led_off_dn + float(fraction) * (self.led_on_dn - self.led_off_dn)
        img[self._disc] = level + noise[self._disc]
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        now = int(self.clock.now_ns())
        k = self._seq
        while True:
            start = self._t0 + k * self.period_ns
            end = start + self.exposure_ns
            jitter = abs(self._rng.normal(0.0, self.jitter_ns)) if self.jitter_ns > 0 else 0.0
            capture = end + self.delivery_ns + int(jitter)
            if capture > now:
                break
            k += 1
        self._seq = k + 1
        self.clock.set(capture)
        fraction = self.led.lit_fraction(start, end)
        t_exposure = start + self.exposure_ns // 2 - self.timestamp_offset_ns + self.applied_timestamp_offset_ns
        return Frame(
            image=self._render(fraction),
            seq=k,
            t_exposure_ns=int(t_exposure),
            t_capture_ns=int(capture),
            exposure_us=self.exposure_us,
            gain=1.0,
        )


# ------------------------------------------------------------------------------------- measurement
def _read_frame(source, timeout_s: float) -> Frame:
    frame = source.read(timeout_s=timeout_s)
    if frame is None:
        raise RuntimeError(f"no frame from {getattr(source, 'name', 'source')} within {timeout_s:.1f} s")
    return frame


def _wait_until(source, deadline_ns: int, period_ns: int, timeout_s: float) -> None:
    """Keep the frame queue drained until one frame period before the deadline, then wait precisely.

    The final sub-frame wait is what decorrelates the LED switch-on from the frame clock: leaving the
    drain loop always lands just after a frame arrived.
    """
    clock = source.clock
    while clock.now_ns() < deadline_ns - period_ns:
        _read_frame(source, timeout_s)
    sleep_until = getattr(source, "sleep_until", None)
    if sleep_until is not None:
        sleep_until(deadline_ns)
        return
    remaining_s = (deadline_ns - clock.now_ns()) / 1e9
    if remaining_s > 0:
        time.sleep(remaining_s)


def locate_led(
    source,
    led: LedDriver,
    *,
    toggles: int = 6,
    settle_frames: int = 3,
    frames_per_state: int = 4,
    roi: tuple[int, int, int, int] | None = None,
    timeout_s: float = 2.0,
) -> dict:
    """Toggle the LED and find it by differencing the mean on/off images.

    Returns {"roi", "off_level", "on_level", "peak_dn"}; the levels are ROI means in DN.
    """
    on_sum: np.ndarray | None = None
    off_sum: np.ndarray | None = None
    n_on = n_off = 0
    for _ in range(max(1, toggles)):
        for level in (1, 0):
            led.write(level)
            for _ in range(settle_frames):
                _read_frame(source, timeout_s)
            for _ in range(frames_per_state):
                img = np.asarray(_gray(_read_frame(source, timeout_s).image), dtype=np.float64)
                if level:
                    on_sum = img.copy() if on_sum is None else on_sum + img
                    n_on += 1
                else:
                    off_sum = img.copy() if off_sum is None else off_sum + img
                    n_off += 1
    mean_on = on_sum / n_on
    mean_off = off_sum / n_off
    diff = cv2.blur(mean_on - mean_off, (3, 3))
    peak = float(diff.max())
    if peak < MIN_LED_CONTRAST_DN:
        raise RuntimeError(
            f"the LED changes the image by only {peak:.1f} DN (need {MIN_LED_CONTRAST_DN:.0f}): "
            "check the wiring on the latency LED pin, point the LED at the lens, or raise the exposure"
        )
    if roi is None:
        binary = (diff >= LIT_THRESHOLD * peak).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if n_labels < 2:
            raise RuntimeError("no changing blob found for the LED; pass --roi x,y,w,h")
        weight = np.bincount(labels.ravel(), weights=np.maximum(diff, 0.0).ravel(), minlength=n_labels)
        weight[0] = -1.0  # background
        best = int(np.argmax(weight))
        x, y, w, h = (int(v) for v in stats[best, :4])
        pad = 1
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(diff.shape[1] - x, w + 2 * pad)
        h = min(diff.shape[0] - y, h + 2 * pad)
        roi = (x, y, w, h)
    x, y, w, h = roi
    if x < 0 or y < 0 or x + w > diff.shape[1] or y + h > diff.shape[0]:
        raise ValueError(f"roi {roi} is outside the {diff.shape[1]}x{diff.shape[0]} image")
    off_level = float(mean_off[y : y + h, x : x + w].mean())
    on_level = float(mean_on[y : y + h, x : x + w].mean())
    if on_level - off_level < MIN_LED_CONTRAST_DN:
        raise RuntimeError(
            f"ROI {roi} only brightens by {on_level - off_level:.1f} DN when the LED is on; "
            "pick a tighter --roi or move the LED closer to the lens"
        )
    return {"roi": (int(x), int(y), int(w), int(h)), "off_level": off_level, "on_level": on_level,
            "peak_dn": peak}


def run_led_trials(
    source,
    led: LedDriver,
    *,
    trials: int,
    roi: tuple[int, int, int, int],
    off_level: float,
    on_level: float,
    rng: np.random.Generator,
    min_wait_s: float = 0.030,
    max_wait_s: float = 0.250,
    progress=None,
) -> list[dict]:
    """N LED switch-ons at random phases; one record per trial for the first lit frame.

    Each record: {"trial","seq","t_on_ns","t_exposure_ns","t_capture_ns","exposure_us","fraction",
    "level","frames_waited","dropped"}.
    """
    span = on_level - off_level
    threshold = off_level + LIT_THRESHOLD * span
    period_ns = int(round(1e9 / float(source.nominal_fps))) if source.nominal_fps > 0 else 10_000_000
    read_timeout_s = max(1.0, 4.0 * period_ns / 1e9)
    max_frames = max(4, int(math.ceil(TRIAL_TIMEOUT_S * 1e9 / period_ns)))
    records: list[dict] = []
    led.write(0)
    for i in range(int(trials)):
        wait_s = float(rng.uniform(min_wait_s, max_wait_s))
        _wait_until(source, source.clock.now_ns() + int(wait_s * 1e9), period_ns, read_timeout_s)
        dropped0 = int(getattr(source, "dropped", 0))
        t_on = led.write(1)
        frame = None
        for waited in range(max_frames):
            frame = _read_frame(source, read_timeout_s)
            level = _roi_level(frame.image, roi)
            if level >= threshold:
                break
        else:
            led.write(0)
            raise RuntimeError(
                f"trial {i + 1}: the ROI never reached {threshold:.1f} DN after the LED went on "
                "(LED not in view, or the ROI moved)"
            )
        fraction = min(max((level - off_level) / span, 0.0), 1.0)
        records.append(
            {
                "trial": i,
                "seq": int(frame.seq),
                "t_on_ns": int(t_on),
                "t_exposure_ns": int(frame.t_exposure_ns),
                "t_capture_ns": int(frame.t_capture_ns),
                "exposure_us": float(frame.exposure_us),
                "fraction": float(fraction),
                "level": float(level),
                "frames_waited": int(waited),
                "dropped": int(getattr(source, "dropped", 0)) - dropped0,
            }
        )
        led.write(0)
        dark = 0
        for _ in range(max_frames):
            if _roi_level(_read_frame(source, read_timeout_s).image, roi) < threshold:
                dark += 1
                if dark >= 2:
                    break
            else:
                dark = 0
        else:
            raise RuntimeError(f"trial {i + 1}: the ROI stayed lit after the LED was switched off")
        if progress is not None:
            progress(i + 1, int(trials))
    return records


def analyze_led_trials(trials, frame_period_s: float, *, phase_offset_ns: float | None = None) -> dict:
    """Camera latency statistics from the records of run_led_trials (pure function).

    exposure_to_user_ms = t_capture - t_on: it contains up to one frame period of phase wait, so its
    minimum is the honest "exposure end -> user space" number and its spread is the frame clock.
    exposure_end_to_user_ms removes that phase wait using the recovered timestamp offset.
    timestamp_offset_ns must be ADDED to the sensor's mid-exposure timestamp to put it on the
    monotonic clock: for a global shutter the LED came on at (exposure end - fraction*exposure).
    phase_offset_ns, if given, replaces this run's own offset estimate in exposure_end_to_user_ms (used
    when a tighter offset from an earlier run is kept).
    """
    user_ms: list[float] = []
    offsets_ns: list[float] = []
    corr: list[tuple[int, int, float]] = []  # (t_capture, t_exposure, exposure_ns)
    exposures_us: list[float] = []
    for rec in trials:
        t_on = int(rec["t_on_ns"])
        t_cap = int(rec["t_capture_ns"])
        t_exp = int(rec["t_exposure_ns"])
        exp_ns = float(rec.get("exposure_us", 0.0)) * 1e3
        fraction = float(rec["fraction"])
        user_ms.append((t_cap - t_on) / 1e6)
        exposures_us.append(float(rec.get("exposure_us", 0.0)))
        if exp_ns > 0.0:
            corr.append((t_cap, t_exp, exp_ns))
            if PARTIAL_LO < fraction < PARTIAL_HI:
                offsets_ns.append(t_on - (t_exp + 0.5 * exp_ns - fraction * exp_ns))

    warnings: list[str] = []
    n_partial = len(offsets_ns)
    # A switch-on lands inside an exposure window with probability exposure/frame period, so half a
    # frame period of exposure makes about half the trials usable for the offset.
    advice = (f"raise camera.exposure_us for this test (now "
              f"{np.mean(exposures_us) if exposures_us else float('nan'):.0f} us, try "
              f"{0.5 * float(frame_period_s) * 1e6:.0f} us) so LED switch-ons land inside an exposure")
    if n_partial == 0:
        offset_ns, offset_sigma = 0.0, float("nan")
        warnings.append(f"no partially lit frame: the timestamp offset was NOT measured - {advice}")
    else:
        offset_ns = float(np.mean(offsets_ns))
        offset_sigma = float(np.std(offsets_ns, ddof=1)) if n_partial > 1 else 0.0
        if n_partial < MIN_PARTIAL_TRIALS:
            warnings.append(
                f"only {n_partial} partially lit frame(s) (want {MIN_PARTIAL_TRIALS}): the timestamp "
                f"offset is noisy - run more trials or {advice}"
            )
    phase_ns = offset_ns if phase_offset_ns is None else float(phase_offset_ns)
    end_to_user_ms = [(t_cap - (t_exp + phase_ns + 0.5 * exp_ns)) / 1e6 for t_cap, t_exp, exp_ns in corr]
    if not corr:
        warnings.append("no frame reported an exposure time; the phase-corrected latency is unavailable")

    return {
        "exposure_to_user_ms": summarize(user_ms),
        "exposure_end_to_user_ms": summarize(end_to_user_ms),
        "timestamp_offset_ns": int(round(offset_ns)),
        "timestamp_offset_sigma_ns": offset_sigma,
        "n": len(user_ms),
        "n_partial": n_partial,
        "frame_period_ms": float(frame_period_s) * 1e3,
        "exposure_us_mean": float(np.mean(exposures_us)) if exposures_us else float("nan"),
        "warnings": warnings,
    }


def _fmt_stats(name: str, st: dict) -> str:
    return (
        f"  {name:<26} mean {st['mean']:8.3f}  p50 {st['p50']:8.3f}  p95 {st['p95']:8.3f}"
        f"  min {st['min']:8.3f}  max {st['max']:8.3f}  n {st['n']}"
    )


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot latency-camera",
        description="LED test: exposure -> userspace latency and camera timestamp offset.",
    )
    add_common_args(parser)
    parser.add_argument("--trials", type=int, default=100, help="LED switch-ons to time (default 100)")
    parser.add_argument("--roi", default=None, metavar="X,Y,W,H", help="LED window (default: find it)")
    parser.add_argument("--pin", type=int, default=None, help="LED BCM pin (default gpio.latency_led_pin)")
    parser.add_argument("--toggles", type=int, default=6, help="on/off cycles used to locate the LED")
    parser.add_argument("--min-wait-ms", type=float, default=30.0, help="shortest random wait per trial")
    parser.add_argument("--max-wait-ms", type=float, default=250.0, help="longest random wait per trial")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for the random waits")
    parser.add_argument("--sim", action="store_true", help="simulated LED camera + SimGpio (no hardware)")
    parser.add_argument("--sim-latency-ms", type=float, default=7.5, help="--sim: exposure end -> user space")
    parser.add_argument("--sim-offset-ms", type=float, default=-1.2, help="--sim: true timestamp offset")
    parser.add_argument("--sim-jitter-ms", type=float, default=0.3, help="--sim: delivery jitter (1 sigma)")
    parser.add_argument("--no-write", action="store_true", help="do not touch calib/latency.json")
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    args = parser.parse_args(argv)

    cfg, calib = load_from_args(args)
    cfg.camera.auto_exposure = False  # timing is read from pixel levels: exposure must not move
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.max_wait_ms < args.min_wait_ms:
        parser.error("--max-wait-ms must be >= --min-wait-ms")
    try:
        roi = parse_roi(args.roi) if args.roi else None
    except ValueError as exc:
        parser.error(str(exc))
    quiet = args.as_json
    if args.sim:
        cfg.gpio.backend = "sim"

    gpio = open_gpio(cfg)
    pin = cfg.gpio.latency_led_pin if args.pin is None else args.pin
    led = LedDriver(gpio, pin)
    if args.sim:
        source = SimLedCamera(
            cfg,
            led,
            delivery_latency_ms=args.sim_latency_ms,
            timestamp_offset_ns=int(round(args.sim_offset_ms * 1e6)),
            jitter_ms=args.sim_jitter_ms,
            seed=args.seed,
            applied_offset_ns=int(calib.budget.camera_timestamp_offset_ns) if calib.budget is not None else 0,
        )
    else:
        source = open_source(cfg.camera.source, cfg, calib)
    led.clock = source.clock

    def progress(done: int, total: int) -> None:
        step = max(1, total // 10)
        if not quiet and (done % step == 0 or done == total):
            print(f"  trial {done}/{total}", flush=True)

    try:
        source.start()
        if not quiet:
            print(f"source {source.name} {source.width}x{source.height} @ {source.nominal_fps:g} fps, "
                  f"LED on BCM {pin} ({cfg.gpio.backend})")
        found = locate_led(source, led, toggles=args.toggles, roi=roi)
        if not quiet:
            print(f"LED ROI {found['roi']}  off {found['off_level']:.1f} DN  on {found['on_level']:.1f} DN "
                  f"(peak change {found['peak_dn']:.1f} DN)")
        records = run_led_trials(
            source,
            led,
            trials=args.trials,
            roi=found["roi"],
            off_level=found["off_level"],
            on_level=found["on_level"],
            rng=np.random.default_rng(args.seed),
            min_wait_s=args.min_wait_ms / 1e3,
            max_wait_s=args.max_wait_ms / 1e3,
            progress=progress,
        )
    finally:
        try:
            led.write(0)
        finally:
            source.stop()
            gpio.close()

    fps = float(source.nominal_fps) if source.nominal_fps > 0 else float(cfg.camera.fps)
    result = analyze_led_trials(records, 1.0 / fps)
    # The source already adds the calibrated offset to t_exposure_ns, so the analysis recovered only the
    # residual: add the applied part back to get the absolute offset. (The phase-corrected latency is
    # unaffected - it uses applied + residual through t_exposure_ns.)
    applied_ns = int(getattr(source, "applied_timestamp_offset_ns", 0))
    result["applied_offset_ns"] = applied_ns
    if result["n_partial"]:
        result["residual_offset_ns"] = result["timestamp_offset_ns"]
        result["timestamp_offset_ns"] += applied_ns
        prev = (calib.latency or {}).get("camera", {}) if getattr(calib, "latency", None) else {}
        prev_sigma = prev.get("timestamp_offset_sigma_ns")
        if (
            "timestamp_offset_ns" in prev
            and isinstance(prev_sigma, (int, float))
            and np.isfinite(prev_sigma)
            and not result["timestamp_offset_sigma_ns"] < prev_sigma
        ):
            result["warnings"].append(
                f"kept the offset on disk ({prev['timestamp_offset_ns'] / 1e6:+.3f} ms, sigma "
                f"{prev_sigma / 1e6:.3f} ms): this run's estimate "
                f"({result['timestamp_offset_ns'] / 1e6:+.3f} ms, sigma "
                f"{result['timestamp_offset_sigma_ns'] / 1e6:.3f} ms) is no tighter"
            )
            result["timestamp_offset_ns"] = int(prev["timestamp_offset_ns"])
            result["timestamp_offset_sigma_ns"] = float(prev_sigma)
            result["offset_kept_from_disk"] = True
            # Re-derive the phase-corrected latency with the kept offset, not this run's noisy one. The frames'
            # t_exposure_ns already include the applied offset, so pass the kept offset relative to it.
            redo = analyze_led_trials(records, 1.0 / fps, phase_offset_ns=result["timestamp_offset_ns"] - applied_ns)
            result["exposure_end_to_user_ms"] = redo["exposure_end_to_user_ms"]
    result["roi"] = list(found["roi"])
    result["off_level"] = found["off_level"]
    result["on_level"] = found["on_level"]
    result["source"] = source.name
    result["dropped"] = int(sum(r["dropped"] for r in records))

    path = calib.path("latency.json")
    if not args.no_write:
        camera = dict(result)
        camera["created"] = now_iso()
        if result["n_partial"] == 0:
            # Nothing measured the offset: leave whatever is already on disk / the config default
            # rather than writing a fake zero that the budget would then call "measured".
            camera.pop("timestamp_offset_ns")
        update_json(path, {"camera": camera, "created": now_iso()})
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{len(records)} trials, frame period {result['frame_period_ms']:.3f} ms, "
              f"exposure {result['exposure_us_mean']:.0f} us, {result['n_partial']} partially lit")
        print(_fmt_stats("exposure->user (raw, ms)", result["exposure_to_user_ms"]))
        print(_fmt_stats("exposure end->user (ms)", result["exposure_end_to_user_ms"]))
        offset = (f"{result['timestamp_offset_ns'] / 1e6:+8.3f} ms (sigma "
                  f"{result['timestamp_offset_sigma_ns'] / 1e6:.3f} ms, ADD to t_exposure_ns)"
                  if result["n_partial"] else "  not measured")
        print(f"  timestamp offset          {offset}")
        if result["dropped"]:
            print(f"  note: {result['dropped']} frame(s) dropped during the trials")
        for w in result["warnings"]:
            print(f"  WARNING: {w}")
        if not args.no_write:
            print(f"wrote {path} [camera]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
