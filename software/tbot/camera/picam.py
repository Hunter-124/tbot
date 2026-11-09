"""Picamera2 frame source for the innomaker CAM-MIPIOV9281 (OV9281 mono global shutter).

picamera2 is imported lazily (inside the class) so laptop runs never touch it.
The main stream is YUV420; only the Y plane is used, which is the sensor's mono image.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Sequence

import cv2
import numpy as np

from tbot.camera.base import SourceBase
from tbot.types import Frame, ImuSource, MonotonicClock

__all__ = ["PicamSource", "select_sensor_mode", "extract_y_plane", "exposure_mid_ns"]

_NO_PICAMERA2 = (
    "picamera2 is not available: this source only runs on a Raspberry Pi with libcamera.\n"
    "  Pi:      sudo apt install -y python3-picamera2 python3-opencv python3-numpy\n"
    "  laptop:  use --source cv:0, video:<path>, replay:<session_dir> or synthetic"
)
_NO_CAMERA = (
    "no camera detected by libcamera. For the innomaker OV9281 check:\n"
    "  * /boot/firmware/config.txt contains 'dtoverlay=ov9281' (and 'camera_auto_detect=0'), then reboot\n"
    "  * the CSI ribbon is seated the right way round (contacts toward the board)\n"
    "  * 'libcamera-hello --list-cameras' lists the sensor"
)

# Largest plausible difference between SensorTimestamp and time.monotonic_ns() before warning.
# CLOCK_BOOTTIME includes suspend time; CLOCK_MONOTONIC does not (see class docstring).
_CLOCK_SANITY_NS = 1_000_000_000


def _mode_size(mode: Any) -> tuple[int, int]:
    """(width, height) of a picamera2 sensor-mode entry (dict-like or attribute style)."""
    size = mode.get("size") if isinstance(mode, dict) else getattr(mode, "size", None)
    if size is None:
        raise ValueError(f"sensor mode without a size: {mode!r}")
    w, h = size
    return int(w), int(h)


def _mode_fps(mode: Any) -> float:
    fps = mode.get("fps") if isinstance(mode, dict) else getattr(mode, "fps", None)
    try:
        return float(fps)
    except (TypeError, ValueError):
        return 0.0


def select_sensor_mode(modes: Sequence[Any], width: int, height: int) -> Any:
    """Pick the sensor mode to drive the raw stream with for a `width` x `height` output.

    Exact size wins; otherwise the smallest mode that still covers the requested size (so the ISP
    downscales rather than upscales); otherwise the largest mode available. Ties go to the faster
    mode. On the OV9281 this picks 640x400 (binned) for the default 640x400 output.
    """
    if not modes:
        raise ValueError("the camera reports no sensor modes")
    w, h = int(width), int(height)
    indexed = list(enumerate(modes))
    exact = [(i, m) for i, m in indexed if _mode_size(m) == (w, h)]
    if exact:
        pool, biggest = exact, False
    else:
        covering = [(i, m) for i, m in indexed if _mode_size(m)[0] >= w and _mode_size(m)[1] >= h]
        pool, biggest = (covering, False) if covering else (indexed, True)
    sign = -1.0 if biggest else 1.0

    def key(item: tuple[int, Any]) -> tuple[float, float, int]:
        i, m = item
        mw, mh = _mode_size(m)
        return (sign * float(mw * mh), -_mode_fps(m), i)

    return min(pool, key=key)[1]


def extract_y_plane(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Copy the luma plane out of a (possibly stride-padded) YUV420 buffer.

    picamera2 hands out the main stream as a (height*3/2, stride) uint8 array whose rows are padded
    to the DMA stride; the first `height` rows hold Y, the rest the subsampled chroma. The result is
    always a fresh contiguous (height, width) array so the request can be released immediately.
    """
    a = np.asarray(array)
    if a.ndim != 2:
        raise ValueError(f"YUV420 buffer must be 2-D (rows, stride), got shape {a.shape}")
    rows, stride = int(a.shape[0]), int(a.shape[1])
    if rows < height or stride < width:
        raise ValueError(f"YUV420 buffer {rows}x{stride} is smaller than the {height}x{width} Y plane")
    y = a[:height, :width]
    if y.dtype != np.uint8:
        y = y.astype(np.uint8)
        return np.ascontiguousarray(y)
    return y.copy()


def exposure_mid_ns(sensor_timestamp_ns: int, exposure_us: float, offset_ns: int = 0) -> int:
    """Mid-exposure time from the kernel start-of-frame timestamp, plus the calibrated offset."""
    return int(sensor_timestamp_ns) - int(round(float(exposure_us) * 1000.0 / 2.0)) + int(offset_ns)


class PicamSource(SourceBase):
    """Live OV9281 frames through Picamera2.

    A background thread owns the camera: it dequeues requests, copies the Y plane, releases the
    request and keeps only the newest frame (`dropped` counts the overwritten ones), so a slow
    consumer can never stall the camera or exhaust the buffer pool.

    Timing: on the Raspberry Pi pipeline SensorTimestamp approximates the CSI start-of-frame
    interrupt, after the first pixels have been exposed:
    https://github.com/raspberrypi/picamera2/issues/821
    For the global shutter we assume mid-exposure is half an ExposureTime earlier, and add the
    measured camera.timestamp_offset_ns from `tbot latency-camera`. Libcamera specifies the clock
    as CLOCK_BOOTTIME, while monotonic_ns uses CLOCK_MONOTONIC; they agree only without suspend.
    Re-measure the offset after suspend. CSI timing and the achieved frame rate must be measured on
    the actual Pi; FrameDurationLimits requests a rate, it does not guarantee it.
    """

    def __init__(self, cfg, calib) -> None:
        super().__init__()
        cam = cfg.camera
        self.name = "picam"
        self.is_color = False
        self.width = int(cam.width)
        self.height = int(cam.height)
        self.nominal_fps = float(cam.fps)
        self.clock = MonotonicClock()
        self._rotate_180 = bool(cam.rotate_180)
        self._exposure_us = float(cam.exposure_us)
        self._gain = float(cam.analogue_gain)
        budget = getattr(calib, "budget", None)
        self._offset_ns = int(
            budget.camera_timestamp_offset_ns if budget is not None else cfg.latency.camera_timestamp_offset_ns
        )

        self._cond = threading.Condition()
        self._slot: Frame | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._clock_warned = False
        self._camera_closed = False

        try:
            from picamera2 import Picamera2  # noqa: PLC0415 - Pi-only, imported lazily by contract
        except ImportError as exc:
            raise RuntimeError(_NO_PICAMERA2) from exc
        if not Picamera2.global_camera_info():
            raise RuntimeError(_NO_CAMERA)
        self._picam2 = Picamera2()
        self.sensor_mode = select_sensor_mode(self._picam2.sensor_modes, self.width, self.height)

        frame_us = int(round(1e6 / max(self.nominal_fps, 1e-3)))
        controls: dict[str, Any] = {
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": bool(cam.auto_exposure),
        }
        if not cam.auto_exposure:
            controls["ExposureTime"] = int(round(self._exposure_us))
            controls["AnalogueGain"] = self._gain
        config = self._picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "YUV420"},
            raw={"size": _mode_size(self.sensor_mode)},
            buffer_count=int(cam.buffer_count),
            controls=controls,
        )
        self._picam2.configure(config)

    # ----------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._camera_closed:
            raise RuntimeError("camera source cannot be restarted after stop()")
        if self._thread is not None:
            return
        self._stop.clear()
        self._picam2.start()
        self._thread = threading.Thread(target=self._capture_loop, name="picam-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if not self._camera_closed:
            self._camera_closed = True
            try:
                self._picam2.stop()
            finally:
                self._picam2.close()
        with self._cond:
            self._slot = None
            self._cond.notify_all()

    def set_exposure_us(self, exposure_us: float) -> None:
        """Change the manual exposure while streaming (`tbot hitl` display test). Frames carry the applied
        ExposureTime in their metadata, so the switch-over frames report their real exposure."""
        self._exposure_us = float(exposure_us)
        self._picam2.set_controls({"AeEnable": False, "ExposureTime": int(round(self._exposure_us))})

    # ----------------------------------------------------------------------------- capture
    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                request = self._picam2.capture_request()
                t_capture_ns = time.monotonic_ns()
                try:
                    array = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                if not self._stop.is_set():
                    self._publish(self._make_frame(array, metadata, t_capture_ns))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer in read()
            if not self._stop.is_set():
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()

    def _make_frame(self, array: np.ndarray, metadata: dict, t_capture_ns: int) -> Frame:
        image = extract_y_plane(array, self.width, self.height)
        if self._rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        exposure_us = float(metadata.get("ExposureTime", self._exposure_us))
        gain = float(metadata.get("AnalogueGain", self._gain))
        sensor_ts = metadata.get("SensorTimestamp")
        if sensor_ts is None:
            t_exposure_ns = t_capture_ns - int(round(exposure_us * 1000.0 / 2.0)) + self._offset_ns
        else:
            if not self._clock_warned and abs(int(sensor_ts) - t_capture_ns) > _CLOCK_SANITY_NS:
                self._clock_warned = True
                print(
                    f"tbot: SensorTimestamp ({int(sensor_ts)} ns) is {abs(int(sensor_ts) - t_capture_ns) / 1e9:.1f} s "
                    "away from time.monotonic_ns(); measure the offset with 'tbot latency-camera'",
                    file=sys.stderr,
                )
            t_exposure_ns = exposure_mid_ns(int(sensor_ts), exposure_us, self._offset_ns)
        self._seq += 1
        return Frame(
            image=image,
            seq=self._seq - 1,
            t_exposure_ns=t_exposure_ns,
            t_capture_ns=t_capture_ns,
            exposure_us=exposure_us,
            gain=gain,
        )

    def _publish(self, frame: Frame) -> None:
        with self._cond:
            if self._slot is not None:
                self.dropped += 1
            self._slot = frame
            self._cond.notify_all()

    # ----------------------------------------------------------------------------- consumer
    def read(self, timeout_s: float = 1.0) -> Frame | None:
        deadline = time.monotonic() + float(timeout_s)
        with self._cond:
            while self._slot is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or self._stop.is_set():
                    return None
                self._cond.wait(remaining)
            if self._error is not None:
                error, self._error = self._error, None
                raise RuntimeError(f"camera capture thread failed: {error}") from error
            frame, self._slot = self._slot, None
            return frame

    def imu(self) -> ImuSource | None:
        return None
