"""Sensor process: the live camera (and the IMU) acquired in a child process (`camera.process = true`).

Why: Python runs one thread at a time. Measured on the Pi 3B at 100 fps, the picamera2 capture thread (buffer
copies, auto-exposure metering) took 33 % of a core, the MPU-6050 polling thread 22 % and picamera2's own event
threads 20 %, all of it holding the interpreter lock the vision loop needs: every detector step ran 2-3x slower
live than on stored frames (12 ms against 4 ms with nothing in view). In a child process that work runs on
another core and the pipeline process has a single thread.

The child opens the source exactly as the pipeline would (`open_live_source`: a picam wrapped in
RecoveringSource, auto-exposure and all) plus the IMU, and sends
  * each frame's pixels through a small shared-memory ring (a sequence lock per slot: a slot that was
    overwritten while it was being copied is detected and counted as a drop), and
  * frame metadata and IMU samples over a pipe.
The parent side is two thin proxies with the usual interfaces: `SensorProcess` (a FrameSource) and
`SensorProcess.imu()` (an ImuSource). Timestamps are CLOCK_MONOTONIC on both sides, so nothing shifts.

If the child dies, read() raises: the pipeline stops and systemd restarts the service (camera stalls are
already handled inside the child by RecoveringSource).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import select
import signal
import struct
import time
from multiprocessing import shared_memory

import numpy as np

from tbot.camera.base import SourceBase
from tbot.types import Frame, ImuSample, MonotonicClock

__all__ = ["SensorProcess"]

log = logging.getLogger(__name__)

SLOTS = 4            # frames in the shared ring: the reader always takes the newest, so 2 would do
HEADER = 16          # per slot: int64 sequence before the pixels, int64 after (equal and even = complete)
START_TIMEOUT_S = 20.0
IMU_COLS = 7         # gyro xyz, accel xyz, temp (float64); the int64 timestamps travel beside them
AE_PERIOD_NS = 200_000_000  # the auto-exposure status is sent on a change of state, else at 5 Hz

# Pipe messages are raw bytes with a one-byte tag (pickling a tuple per frame and the stock Connection.poll()
# cost the reader ~2.5 ms a frame on the Pi 3B):
#   F + _FRAME  one frame is in the ring       I + n int64 times + n x IMU_COLS float64   IMU samples
#   A + pickle  auto-exposure status           N + pickle  source description (first message)
#   E + utf-8   the child failed               B           the child is exiting
_FRAME = struct.Struct("<qqqqqddqqB")  # slot, stamp, seq, t_exposure_ns, t_capture_ns, exposure_us, gain,
#                                        frames dropped by the source, restarts, first-after-restart flag


def _pack_imu(samples) -> bytes:
    n = len(samples)
    t = np.empty(n, dtype=np.int64)
    a = np.empty((n, IMU_COLS), dtype=np.float64)
    for i, s in enumerate(samples):
        t[i] = s.t_ns
        a[i, 0:3] = s.gyro
        a[i, 3:6] = s.accel
        a[i, 6] = s.temp_c
    return b"I" + t.tobytes() + a.tobytes()


def _unpack_imu(msg: bytes) -> list[ImuSample]:
    n = (len(msg) - 1) // (8 * (1 + IMU_COLS))
    t = np.frombuffer(msg, dtype=np.int64, count=n, offset=1)
    a = np.frombuffer(msg, dtype=np.float64, count=n * IMU_COLS, offset=1 + 8 * n).reshape(n, IMU_COLS)
    return [ImuSample(int(t[i]), a[i, 0:3].copy(), a[i, 3:6].copy(), float(a[i, 6])) for i in range(n)]


def _child(conn, cfg, calib, spec: str, want_imu: bool, stop_evt, parent_pid: int) -> None:
    """Child process body: acquire, publish, until told to stop or the parent is gone."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C reaches the whole process group: the parent decides
    try:
        signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())
    except (ValueError, OSError):
        pass
    source = imu = shm = None
    try:
        from tbot.camera import open_live_source  # noqa: PLC0415
        from tbot.imu.mpu6050 import NullImu, open_imu  # noqa: PLC0415

        source = open_live_source(spec, cfg, calib)
        if want_imu:
            imu = open_imu(cfg, calib, source)
            if isinstance(imu, NullImu):
                imu = None
        channels = 3 if bool(source.is_color) else 1
        conn.send_bytes(b"N" + pickle.dumps({
            "width": int(source.width), "height": int(source.height), "channels": channels,
            "nominal_fps": float(source.nominal_fps), "name": str(getattr(source, "name", spec)),
            "endless": bool(getattr(source, "endless", False)), "imu": imu is not None,
            "applied_timestamp_offset_ns": int(getattr(source, "applied_timestamp_offset_ns", 0) or 0),
            "pid": os.getpid(),
        }))
        shm_name = conn.recv_bytes().decode()
        try:
            shm = shared_memory.SharedMemory(name=shm_name, track=False)  # 3.13+: the parent owns the segment
        except TypeError:
            shm = shared_memory.SharedMemory(name=shm_name)
        frame_bytes = int(source.width) * int(source.height) * channels
        slot_bytes = HEADER + frame_bytes
        buf = np.ndarray((SLOTS * slot_bytes,), dtype=np.uint8, buffer=shm.buf)
        heads = [np.ndarray((2,), dtype=np.int64, buffer=shm.buf, offset=i * slot_bytes) for i in range(SLOTS)]
        source.start()
        if imu is not None:
            imu.start()
        k = 0
        ae_state, ae_sent_ns = None, 0
        while not stop_evt.is_set():
            frame = source.read(timeout_s=0.05)
            if imu is not None:
                samples = imu.read()
                if samples:
                    conn.send_bytes(_pack_imu(samples))
            if frame is None:
                if os.getppid() != parent_pid:  # orphaned: the pipeline is gone
                    break
                continue
            slot = k % SLOTS
            head = heads[slot]
            head[0] = 2 * k + 1                                      # odd: being written
            o = slot * slot_bytes + HEADER
            buf[o:o + frame_bytes] = np.ascontiguousarray(frame.image).reshape(-1)
            head[1] = 2 * k + 2
            head[0] = 2 * k + 2                                      # even and equal: complete
            gap = bool(getattr(source, "gap_after_restart", False))
            if gap:
                source.gap_after_restart = False
            try:
                ae = getattr(source, "ae_status", None)
            except Exception:  # noqa: BLE001
                ae = None
            if ae is not None:
                state = getattr(ae, "state", None)
                if state != ae_state or int(frame.t_capture_ns) - ae_sent_ns >= AE_PERIOD_NS:
                    ae_state, ae_sent_ns = state, int(frame.t_capture_ns)
                    conn.send_bytes(b"A" + pickle.dumps(ae))
            conn.send_bytes(b"F" + _FRAME.pack(
                slot, 2 * k + 2, int(frame.seq), int(frame.t_exposure_ns), int(frame.t_capture_ns),
                float(frame.exposure_us), float(frame.gain), int(getattr(source, "dropped", 0)),
                int(getattr(source, "restarts", 0)), 1 if gap else 0))
            k += 1
    except (BrokenPipeError, EOFError, KeyboardInterrupt):
        pass
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which raises it
        try:
            conn.send_bytes(b"E" + f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        for dev in (imu, source):
            if dev is not None:
                try:
                    dev.stop()
                except Exception:  # noqa: BLE001
                    pass
        if shm is not None:
            shm.close()
        try:
            conn.send_bytes(b"B")
            conn.close()
        except Exception:  # noqa: BLE001
            pass


class _ImuProxy:
    """ImuSource fed by the sensor process."""

    name = "sensor-process-imu"

    def __init__(self, owner: "SensorProcess") -> None:
        self._owner = owner

    def start(self) -> None:
        self._owner.start()

    def stop(self) -> None:
        pass  # the source's stop() ends the process

    def read(self) -> list[ImuSample]:
        owner = self._owner
        owner._pump()
        chunks, owner._imu_chunks = owner._imu_chunks, []
        out: list[ImuSample] = []
        for msg in chunks:
            out.extend(_unpack_imu(msg))
        return out


class SensorProcess(SourceBase):
    """FrameSource proxy for a source (and IMU) running in a child process. See the module docstring."""

    def __init__(self, cfg, calib, spec: str | None = None, want_imu: bool = True) -> None:
        super().__init__()
        self.clock = MonotonicClock()
        self.spec = spec or str(cfg.camera.source)
        self._ctx = mp.get_context("spawn")  # never fork a process that has OpenCV/libcamera threads
        self._conn, child_conn = self._ctx.Pipe(duplex=True)
        self._stop_evt = self._ctx.Event()
        self._proc = self._ctx.Process(target=_child, name="tbot-sensors", daemon=True,
                                       args=(child_conn, cfg, calib, self.spec, bool(want_imu), self._stop_evt,
                                             os.getpid()))
        self._proc.start()
        child_conn.close()
        self._shm = None
        self._closed = False
        self._started = False
        self._imu_chunks: list = []
        self._latest: tuple | None = None
        self._skipped = 0       # frames this side replaced with a newer one, or lost to an overwritten slot
        self._child_dropped = 0
        self.restarts = 0
        self.gap_after_restart = False
        self.ae_status = None
        self._error: str | None = None
        self._poller = None
        if hasattr(select, "poll"):  # one registered poll object: Connection.poll() builds a selector per call
            self._poller = select.poll()
            self._poller.register(self._conn.fileno(), select.POLLIN)
        info = self._wait_info()
        self.width, self.height = int(info["width"]), int(info["height"])
        self._channels = int(info["channels"])
        self.is_color = self._channels == 3
        self.nominal_fps = float(info["nominal_fps"])
        self.name = str(info["name"])
        self.endless = bool(info["endless"])
        self.has_imu = bool(info["imu"])
        self.applied_timestamp_offset_ns = int(info["applied_timestamp_offset_ns"])
        self.child_pid = int(info["pid"])
        self._frame_bytes = self.width * self.height * self._channels
        self._slot_bytes = HEADER + self._frame_bytes
        self._shm = shared_memory.SharedMemory(create=True, size=SLOTS * self._slot_bytes)
        self._buf = np.ndarray((SLOTS * self._slot_bytes,), dtype=np.uint8, buffer=self._shm.buf)
        self._heads = [np.ndarray((2,), dtype=np.int64, buffer=self._shm.buf, offset=i * self._slot_bytes)
                       for i in range(SLOTS)]
        for head in self._heads:
            head[:] = 0
        self._shape = (self.height, self.width) if self._channels == 1 else (self.height, self.width, 3)
        self._conn.send_bytes(self._shm.name.encode())  # the child starts streaming as soon as it has the ring
        self._imu = _ImuProxy(self) if self.has_imu else None

    # ----------------------------------------------------------------------------- start-up
    def _wait_info(self) -> dict:
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._ready(0.1):
                try:
                    msg = self._conn.recv_bytes()
                except (EOFError, OSError):
                    break
                if msg[:1] == b"N":
                    return pickle.loads(msg[1:])
                if msg[:1] == b"E":
                    self._abort()
                    raise RuntimeError(msg[1:].decode("utf-8", "replace").split(": ", 1)[-1])
            elif not self._proc.is_alive():
                break
        self._abort()
        raise RuntimeError(f"the sensor process did not open {self.spec!r} (exit code {self._proc.exitcode})")

    def _abort(self) -> None:
        self._stop_evt.set()
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()

    @property
    def dropped(self) -> int:
        return self._child_dropped + self._skipped

    @dropped.setter
    def dropped(self, value: int) -> None:  # SourceBase.__init__ assigns it
        self._skipped = int(value)

    # ----------------------------------------------------------------------------- pipe
    def _ready(self, timeout_s: float) -> bool:
        if self._poller is not None:
            return bool(self._poller.poll(max(0.0, timeout_s) * 1000.0))
        return self._conn.poll(timeout_s)

    def _pump(self, timeout_s: float = 0.0) -> None:
        """Take everything waiting on the pipe (blocking up to timeout_s for the first message)."""
        conn = self._conn
        try:
            if not self._ready(timeout_s):
                return
            while True:
                msg = conn.recv_bytes()
                tag = msg[:1]
                if tag == b"F":
                    if self._latest is not None:
                        self._skipped += 1
                    self._latest = _FRAME.unpack_from(msg, 1)
                elif tag == b"I":
                    self._imu_chunks.append(msg)
                elif tag == b"A":
                    self.ae_status = pickle.loads(msg[1:])
                elif tag == b"E":
                    self._error = msg[1:].decode("utf-8", "replace")
                if not self._ready(0.0):
                    return
        except (EOFError, OSError):
            if self._error is None and not self._closed:
                self._error = "the sensor process closed its pipe"

    def start(self) -> None:
        self._started = True  # the child streams from the moment it has the ring

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_evt.set()
        deadline = time.monotonic() + 6.0   # the camera's own stop can take 3 s on a stalled link
        while self._proc.is_alive() and time.monotonic() < deadline:
            self._pump(0.05)                # keep the pipe drained so the child is never stuck in send()
        if self._proc.is_alive():
            log.warning("sensor process did not exit: terminating it")
            self._proc.terminate()
        self._proc.join(timeout=2.0)
        try:
            self._conn.close()
        except OSError:
            pass
        if self._shm is not None:
            self._buf = self._heads = None
            try:
                self._shm.close()
                self._shm.unlink()
            except (OSError, BufferError):
                pass
            self._shm = None

    # ----------------------------------------------------------------------------- frames
    def read(self, timeout_s: float = 1.0) -> Frame | None:
        if self._closed:
            return None
        deadline = time.monotonic() + float(timeout_s)
        while True:
            self._pump(0.0)
            if self._latest is None:
                remaining = deadline - time.monotonic()
                if self._error is not None:
                    raise RuntimeError(f"sensor process failed: {self._error}")
                if remaining <= 0.0:
                    if not self._proc.is_alive():
                        raise RuntimeError(f"sensor process died (exit code {self._proc.exitcode})")
                    return None
                self._pump(min(remaining, 0.05))
                continue
            slot, stamp, seq, t_exp, t_cap, exposure_us, gain, child_dropped, restarts, gap = self._latest
            self._latest = None
            head = self._heads[slot]
            o = slot * self._slot_bytes + HEADER
            image = self._buf[o:o + self._frame_bytes].copy()
            if int(head[0]) != stamp or int(head[1]) != stamp:
                self._skipped += 1  # the ring lapped this reader mid-copy: the frame is torn, wait for the next
                continue
            self._child_dropped = int(child_dropped)
            self.restarts = int(restarts)
            if gap:
                self.gap_after_restart = True
            return Frame(image=image.reshape(self._shape), seq=int(seq), t_exposure_ns=int(t_exp),
                         t_capture_ns=int(t_cap), exposure_us=float(exposure_us), gain=float(gain))

    def imu(self):
        return self._imu
