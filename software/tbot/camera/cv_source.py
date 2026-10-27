"""cv2.VideoCapture frame source: `cv:<index>` (live webcam) and `video:<path>` (file playback).

Laptop prototyping path. A colour webcam can stand in for the mono OV9281 through
`camera.emulate_mono`: "luma" (gray) or "red" (R channel = what a mono sensor behind an orange/red
long-pass filter sees); "none" keeps BGR so the colour branch of the detector can be exercised.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from tbot.camera.base import SourceBase
from tbot.types import Frame, ImuSource, MonotonicClock, SimClock

__all__ = ["CvSource", "EMULATE_MONO_MODES"]

EMULATE_MONO_MODES = ("none", "luma", "red")

# Simulated timestamps start here (contract: synthetic/replay clocks start at 1 s).
_SIM_T0_NS = 1_000_000_000


class CvSource(SourceBase):
    """OpenCV capture source.

    Live (`cv:<index>`): a background thread grabs continuously and keeps only the newest frame
    (`dropped` counts the rest) on the real monotonic clock; the capture driver gives no per-frame
    timestamp, so t_capture is taken at grab() and t_exposure is t_capture minus the configured
    exposure->userspace latency.

    File (`video:<path>`): strictly sequential, never drops, and runs on a SimClock driven by the
    file's presentation timestamps (CAP_PROP_POS_MSEC, falling back to seq / fps).
    """

    def __init__(self, cfg, spec: str) -> None:
        super().__init__()
        kind, _, arg = spec.partition(":")
        if kind not in ("cv", "video"):
            raise ValueError(f"CvSource takes 'cv:<index>' or 'video:<path>', got {spec!r}")
        if not arg:
            raise ValueError(f"{kind} source needs an argument: {kind}:<{'index' if kind == 'cv' else 'path'}>")

        self._emulate = str(cfg.camera.emulate_mono)
        if self._emulate not in EMULATE_MONO_MODES:
            raise ValueError(f"camera.emulate_mono must be one of {EMULATE_MONO_MODES}, got {self._emulate!r}")
        self._live = kind == "cv"
        self.name = spec
        self._exposure_us = float(cfg.camera.exposure_us)
        self._gain = float(cfg.camera.analogue_gain)
        self._latency_ns = int(round(float(cfg.latency.exposure_to_user_ms) * 1e6))

        if self._live:
            try:
                index = int(arg)
            except ValueError as exc:
                raise ValueError(f"cv source needs a numeric device index, got {arg!r}") from exc
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.camera.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.camera.height))
                cap.set(cv2.CAP_PROP_FPS, float(cfg.camera.fps))
            target = f"camera index {index}"
        else:
            path = Path(arg).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"video file not found: {path}")
            cap = cv2.VideoCapture(str(path))
            target = str(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {target} (unsupported format or device busy)")
        self._cap = cap

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = width if width > 0 else int(cfg.camera.width)
        self.height = height if height > 0 else int(cfg.camera.height)
        self.nominal_fps = fps if np.isfinite(fps) and fps > 0.0 else float(cfg.camera.fps)
        self.is_color = self._emulate == "none"
        self._period_ns = int(round(1e9 / max(self.nominal_fps, 1e-3)))

        self.clock = MonotonicClock() if self._live else SimClock(_SIM_T0_NS)
        self._seq = 0
        self._last_exposure_ns = -1
        self._cond = threading.Condition()
        self._slot: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ended = False

    # ----------------------------------------------------------------------------- conversion
    def _convert(self, image: np.ndarray) -> np.ndarray:
        """Apply camera.emulate_mono to a decoded frame (BGR, or already-gray input)."""
        if image.ndim == 2:
            return image if self._emulate == "none" else np.ascontiguousarray(image)
        if self._emulate == "luma":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._emulate == "red":
            return np.ascontiguousarray(image[:, :, 2])
        return image

    def _wrap(self, image: np.ndarray, t_exposure_ns: int, t_capture_ns: int) -> Frame:
        out = self._convert(image)
        self.height, self.width = int(out.shape[0]), int(out.shape[1])
        self.is_color = out.ndim == 3
        frame = Frame(
            image=out,
            seq=self._seq,
            t_exposure_ns=int(t_exposure_ns),
            t_capture_ns=int(t_capture_ns),
            # cv2 exposes no per-frame metadata: report the configured nominal exposure/gain.
            exposure_us=self._exposure_us,
            gain=self._gain,
        )
        self._seq += 1
        return frame

    # ----------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._live and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._grab_loop, name="cv-capture", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._cap.release()
        with self._cond:
            self._slot = None
            self._cond.notify_all()

    # ----------------------------------------------------------------------------- live path
    def _grab_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            if not self._cap.grab():
                # A USB camera can miss a frame; only a persistent failure means end of stream.
                failures += 1
                if failures < 5:
                    time.sleep(0.01)
                    continue
                with self._cond:
                    self._ended = True
                    self._cond.notify_all()
                return
            t_capture_ns = time.monotonic_ns()
            ok, image = self._cap.retrieve()
            if not ok or image is None:
                failures += 1
                if failures >= 5:
                    with self._cond:
                        self._ended = True
                        self._cond.notify_all()
                    return
                continue
            failures = 0
            frame = self._wrap(image, t_capture_ns - self._latency_ns, t_capture_ns)
            with self._cond:
                if self._slot is not None:
                    self.dropped += 1
                self._slot = frame
                self._cond.notify_all()

    def _read_live(self, timeout_s: float) -> Frame | None:
        deadline = time.monotonic() + float(timeout_s)
        with self._cond:
            while self._slot is None:
                if self._ended or self._stop.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(remaining)
            frame, self._slot = self._slot, None
            return frame

    # ----------------------------------------------------------------------------- file path
    def _read_file(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            self._ended = True
            return None
        # FFMPEG reports the presentation time of the frame just decoded; fall back to seq / fps
        # when the container has no usable timestamps (and keep the series strictly increasing).
        pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        t_exposure_ns = _SIM_T0_NS + self._seq * self._period_ns
        if np.isfinite(pos_ms) and pos_ms >= 0.0:
            from_pts = _SIM_T0_NS + int(round(float(pos_ms) * 1e6))
            if from_pts > self._last_exposure_ns:
                t_exposure_ns = from_pts
        if t_exposure_ns <= self._last_exposure_ns:
            t_exposure_ns = self._last_exposure_ns + self._period_ns
        self._last_exposure_ns = t_exposure_ns
        t_capture_ns = t_exposure_ns + self._latency_ns
        frame = self._wrap(image, t_exposure_ns, t_capture_ns)
        self.clock.set(t_capture_ns)
        return frame

    # ----------------------------------------------------------------------------- consumer
    def read(self, timeout_s: float = 1.0) -> Frame | None:
        if self._live:
            return self._read_live(timeout_s)
        if self._ended:
            return None
        return self._read_file()

    def imu(self) -> ImuSource | None:
        return None
