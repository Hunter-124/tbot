"""Picamera2 frame source for the innomaker CAM-MIPIOV9281 (OV9281 mono global shutter).

picamera2 is imported lazily (inside the class) so laptop runs never touch it.
The main stream is YUV420; only the Y plane is used, which is the sensor's mono image.
"""

from __future__ import annotations

import sys
import threading
from concurrent import futures
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
        self.start_probe = None  # linkcheck.LinkDiagnosis of the last start (Pi only), see _probe_link

        try:
            from picamera2 import Picamera2  # noqa: PLC0415 - Pi-only, imported lazily by contract
        except ImportError as exc:
            raise RuntimeError(_NO_PICAMERA2) from exc
        if not Picamera2.global_camera_info():
            raise RuntimeError(_NO_CAMERA)
        self._picam2 = Picamera2()
        self.sensor_mode = select_sensor_mode(self._picam2.sensor_modes, self.width, self.height)

        frame_us = int(round(1e6 / max(self.nominal_fps, 1e-3)))
        # libcamera's AE stays off: tbot's controller (camera/autoexposure.py) caps the exposure time for motion
        # blur, which libcamera's would not. auto_exposure=false means the fixed exposure_us / analogue_gain.
        self._ae = None
        if cam.auto_exposure:
            from tbot.camera.autoexposure import AeSettings, AutoExposure  # noqa: PLC0415
            self._ae = AutoExposure(AeSettings.from_camera(cam), self._exposure_us, self._gain)
            self._exposure_us, self._gain = self._ae.exposure_us, self._ae.gain
        controls: dict[str, Any] = {
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": False,
            "ExposureTime": int(round(self._exposure_us)),
            "AnalogueGain": self._gain,
        }
        config = self._picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "YUV420"},
            raw={"size": _mode_size(self.sensor_mode)},
            buffer_count=int(cam.buffer_count),
            controls=controls,
        )
        self._picam2.configure(config)

    @property
    def applied_timestamp_offset_ns(self) -> int:
        """The calibrated offset already added to every frame's t_exposure_ns (`tbot latency-camera`
        measures only the residual on top of it)."""
        return self._offset_ns

    # ----------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._camera_closed:
            raise RuntimeError("camera source cannot be restarted after stop()")
        if self._thread is not None:
            return
        self._stop.clear()
        self.start_probe = None
        self._picam2.start()
        self._thread = threading.Thread(target=self._capture_loop, name="picam-capture", daemon=True)
        self._thread.start()
        threading.Thread(target=self._probe_link, name="picam-probe", daemon=True).start()

    PROBE_WINDOW_S = 0.7

    def _probe_link(self) -> None:
        """Measure the CSI link in the first PROBE_WINDOW_S of the stream (sensor register + unicam interrupt
        count). It has to be now: about 1 s after a start without frames libcamera's frontend timeout stops
        the sensor itself, after which neither measurement says anything. Result: self.start_probe."""
        from tbot.camera import linkcheck  # noqa: PLC0415

        before = linkcheck.unicam_interrupts()
        if before is None:
            return  # not a Pi with unicam
        seq0 = self._seq
        time.sleep(0.1)
        streaming = linkcheck.sensor_streaming()
        time.sleep(max(0.0, self.PROBE_WINDOW_S - 0.1))
        after = linkcheck.unicam_interrupts()
        delta = None if after is None else max(0, after - before)
        self.start_probe = linkcheck.verdict(streaming, delta, self._seq - seq0, self.PROBE_WINDOW_S)

    STOP_TIMEOUT_S = 3.0
    CAPTURE_WAIT_S = 1.0

    def _cancel_jobs(self) -> None:
        """Drop picamera2's pending jobs. picamera2 runs stop() as a job queued BEHIND any pending
        capture_request(); on a stalled CSI link that request never completes, so without this stop()
        deadlocks and the camera stays acquired (a reopen then fails with "Camera in Running state").
        cancel_all_and_flush() is picamera2's API for exactly this ("the camera cable has become dislodged")."""
        cancel = getattr(self._picam2, "cancel_all_and_flush", None)
        if cancel is not None:
            try:
                cancel()
            except Exception as exc:  # noqa: BLE001
                print(f"tbot: camera job cancel failed: {exc}", file=sys.stderr)

    def stop(self) -> None:
        """Stop and close the camera. Pending capture jobs are cancelled first (see _cancel_jobs), and the
        stop/close still runs on a daemon helper thread that is abandoned after STOP_TIMEOUT_S, so a dead
        camera can never hang the calling tool."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if not self._camera_closed:
            self._camera_closed = True
            done = threading.Event()

            def _close() -> None:
                self._cancel_jobs()
                try:
                    self._picam2.stop()
                except Exception as exc:  # noqa: BLE001 - a broken camera may fail to stop cleanly
                    print(f"tbot: camera stop failed: {exc}", file=sys.stderr)
                finally:
                    try:
                        self._picam2.close()
                    except Exception as exc:  # noqa: BLE001
                        print(f"tbot: camera close failed: {exc}", file=sys.stderr)
                    done.set()

            threading.Thread(target=_close, name="picam-close", daemon=True).start()
            if not done.wait(self.STOP_TIMEOUT_S):
                print(f"tbot: camera did not stop within {self.STOP_TIMEOUT_S:g} s (stalled CSI link); "
                      "abandoning it", file=sys.stderr)
        if thread is not None:
            thread.join(timeout=1.0)
        with self._cond:
            self._slot = None
            self._cond.notify_all()

    def set_exposure_us(self, exposure_us: float) -> None:
        """Change the manual exposure while streaming (`tbot hitl` display test). Frames carry the applied
        ExposureTime in their metadata, so the switch-over frames report their real exposure."""
        self._exposure_us = float(exposure_us)
        self._ae = None  # a manual exposure request ends auto-exposure for this source
        self._picam2.set_controls({"AeEnable": False, "ExposureTime": int(round(self._exposure_us))})

    def _auto_expose(self, frame: Frame) -> None:
        ae = self._ae
        if ae is None:
            return
        new = ae.update(frame.image, frame.exposure_us, frame.gain)
        if new is not None:
            self._exposure_us, self._gain = new
            self._picam2.set_controls({"AeEnable": False, "ExposureTime": int(round(new[0])),
                                       "AnalogueGain": float(new[1])})

    # ----------------------------------------------------------------------------- capture
    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    # bounded wait: an unbounded one leaves a job that never completes on a stalled link
                    request = self._picam2.capture_request(wait=self.CAPTURE_WAIT_S)
                except (TimeoutError, futures.TimeoutError, futures.CancelledError):
                    if self._stop.is_set():
                        break
                    self._cancel_jobs()  # do not pile up one stuck job per second
                    continue
                t_capture_ns = time.monotonic_ns()
                try:
                    array = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                if not self._stop.is_set():
                    frame = self._make_frame(array, metadata, t_capture_ns)
                    self._publish(frame)
                    self._auto_expose(frame)
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
