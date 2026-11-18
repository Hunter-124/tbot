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
  * frame metadata, IMU samples and a heartbeat over a pipe.
The parent side is two thin proxies with the usual interfaces: `SensorProcess` (a FrameSource) and
`SensorProcess.imu()` (an ImuSource). Timestamps are CLOCK_MONOTONIC on both sides, so nothing shifts.

Recovery, in two layers:
  1. inside the child, RecoveringSource closes and reopens a camera that stalls (`camera.stall_s`) or whose
     capture thread fails: a CSI hiccup costs about a second;
  2. the parent replaces the whole child (`respawn_s`, live cameras only) when it dies, stops talking (no
     heartbeat for `hang_s`: wedged in a driver call) or delivers no frame for `camera.respawn_s` although
     alive (the in-process reopen is not working: libcamera left in a bad state). A fresh process starts
     libcamera from nothing, which is what a service restart did, without the pipeline's 15 s cold start.
     Meanwhile read() returns None (the pipeline's watchdog keeps the actuator safe), and the first frame
     afterwards carries `gap_after_restart` so tracks that span the outage are dropped. Respawns back off
     (1, 2, 4 ... 15 s) while they keep failing.
Without respawn (replay/synthetic specs, tests) a dead child raises from read() as before.
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
HEARTBEAT_S = 0.25   # the child says something at least this often, frames or not
HANG_S = 3.0         # parent: a child silent this long is wedged (12 missed heartbeats)
RESPAWN_BACKOFF_S = (0.0, 1.0, 2.0, 4.0, 8.0, 15.0)

# Pipe messages are raw bytes with a one-byte tag (pickling a tuple per frame and the stock Connection.poll()
# cost the reader ~2.5 ms a frame on the Pi 3B):
#   F + _FRAME  one frame is in the ring       I + n int64 times + n x IMU_COLS float64   IMU samples
#   A + pickle  auto-exposure status           N + pickle  source description (first message)
#   E + utf-8   the child failed               B           the child is exiting
#   H           heartbeat (nothing else to say)
_FRAME = struct.Struct("<qqqqqddqqBq")  # slot, stamp, seq, t_exposure_ns, t_capture_ns, exposure_us, gain,
#                                        frames dropped by the source, restarts, first-after-restart flag,
#                                        frames the sensor sent that never arrived (timestamp gaps)


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


class _StopFlag:
    """The child's stop request: SIGTERM, or an S message from the parent on the pipe. (A multiprocessing
    Event is a named semaphore, and at every service stop their cleanup raced the resource tracker and
    filled the log with tracebacks.)"""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        if not self._set:
            try:
                if self._conn.poll(0) and self._conn.recv_bytes() == b"S":
                    self._set = True
            except (EOFError, OSError):
                self._set = True  # the parent is gone
        return self._set


def _child(conn, cfg, calib, spec: str, want_imu: bool, parent_pid: int) -> None:
    """Child process body: acquire, publish, until told to stop or the parent is gone."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C reaches the whole process group: the parent decides
    stop_evt = _StopFlag(conn)
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
        if shm_name == "S":  # told to stop before it was given the ring
            return
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
        said = time.monotonic()
        while not stop_evt.is_set():
            frame = source.read(timeout_s=0.05)
            if imu is not None:
                samples = imu.read()
                if samples:
                    conn.send_bytes(_pack_imu(samples))
                    said = time.monotonic()
            if frame is None:
                if os.getppid() != parent_pid:  # orphaned: the pipeline is gone
                    break
                if time.monotonic() - said >= HEARTBEAT_S:
                    conn.send_bytes(b"H")
                    said = time.monotonic()
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
                int(getattr(source, "restarts", 0)), 1 if gap else 0, int(getattr(source, "lost", 0) or 0)))
            said = time.monotonic()
            k += 1
    except (BrokenPipeError, EOFError, KeyboardInterrupt):
        pass
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which raises it or respawns
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
    """FrameSource proxy for a source (and IMU) running in a child process. See the module docstring.

    respawn_s: replace a child that delivers no frame for this long (0 = never respawn: a dead child raises
    from read()). None = `camera.respawn_s` for a live camera (picam), 0 for anything else.
    """

    def __init__(self, cfg, calib, spec: str | None = None, want_imu: bool = True,
                 respawn_s: float | None = None, hang_s: float = HANG_S) -> None:
        super().__init__()
        self.clock = MonotonicClock()
        self.spec = spec or str(cfg.camera.source)
        self._cfg, self._calib, self._want_imu = cfg, calib, bool(want_imu)
        live = self.spec.partition(":")[0] == "picam"
        if respawn_s is None:
            respawn_s = float(getattr(cfg.camera, "respawn_s", 8.0)) if live else 0.0
        self.respawn_s = float(respawn_s)
        self.hang_s = float(hang_s)
        self._ctx = mp.get_context("spawn")  # never fork a process that has OpenCV/libcamera threads
        self._shm = None
        self._closed = False
        self._started = False
        self._imu_chunks: list = []
        self._latest: tuple | None = None
        self._skipped = 0       # frames this side replaced with a newer one, or lost to an overwritten slot
        self._child_dropped = 0
        self._child_restarts = 0
        self._child_lost = 0
        self._base_dropped = 0  # what earlier children counted
        self._base_restarts = 0
        self._base_lost = 0
        self._seq_base = 0      # seq keeps rising across children
        self._last_seq = -1
        self.respawns = 0       # children replaced (each is also one restart)
        self.last_error: str | None = None
        self.gap_after_restart = False
        self.ae_status = None
        self._error: str | None = None
        self._info: dict | None = None
        self._pending_info: dict | None = None
        self._state = "starting"
        self._fresh_child = False
        self._backoff_i = 0
        self._next_spawn_t = 0.0
        self._spawn()
        info = self._wait_info()
        self.width, self.height = int(info["width"]), int(info["height"])
        self._channels = int(info["channels"])
        self.is_color = self._channels == 3
        self.nominal_fps = float(info["nominal_fps"])
        self.name = str(info["name"])
        self.endless = bool(info["endless"]) or self.respawn_s > 0.0
        self.has_imu = bool(info["imu"])
        self.applied_timestamp_offset_ns = int(info["applied_timestamp_offset_ns"])
        self._frame_bytes = self.width * self.height * self._channels
        self._slot_bytes = HEADER + self._frame_bytes
        self._shm = shared_memory.SharedMemory(create=True, size=SLOTS * self._slot_bytes)
        self._buf = np.ndarray((SLOTS * self._slot_bytes,), dtype=np.uint8, buffer=self._shm.buf)
        self._heads = [np.ndarray((2,), dtype=np.int64, buffer=self._shm.buf, offset=i * self._slot_bytes)
                       for i in range(SLOTS)]
        self._shape = (self.height, self.width) if self._channels == 1 else (self.height, self.width, 3)
        self._attach(info)
        self._imu = _ImuProxy(self) if self.has_imu else None

    # ----------------------------------------------------------------------------- the child
    def _spawn(self) -> None:
        self._conn, child_conn = self._ctx.Pipe(duplex=True)
        self._proc = self._ctx.Process(target=_child, name="tbot-sensors", daemon=True,
                                       args=(child_conn, self._cfg, self._calib, self.spec, self._want_imu,
                                             os.getpid()))
        self._proc.start()
        child_conn.close()
        self._poller = None
        if hasattr(select, "poll"):  # one registered poll object: Connection.poll() builds a selector per call
            self._poller = select.poll()
            self._poller.register(self._conn.fileno(), select.POLLIN)
        self._state = "starting"
        self._error = None
        self._pending_info = None
        self._latest = None
        now = time.monotonic()
        self._spawn_t = self._last_msg_t = self._last_frame_t = now

    def _wait_info(self) -> dict:
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            self._pump(0.1)
            if self._pending_info is not None:
                info, self._pending_info = self._pending_info, None
                return info
            if self._error is not None:
                self._abort()
                raise RuntimeError(self._error.split(": ", 1)[-1])
            if not self._proc.is_alive() and not self._ready(0.0):
                break
        self._abort()
        raise RuntimeError(f"the sensor process did not open {self.spec!r} (exit code {self._proc.exitcode})")

    def _attach(self, info: dict) -> None:
        """The child described itself: hand it the ring (it starts streaming as soon as it has it)."""
        self.child_pid = int(info["pid"])
        for head in self._heads:
            head[:] = 0
        self._conn.send_bytes(self._shm.name.encode())
        self._state = "running"
        self._last_msg_t = self._last_frame_t = time.monotonic()

    def _request_stop(self) -> None:
        try:
            self._conn.send_bytes(b"S")
        except (OSError, ValueError):
            pass

    def _abort(self) -> None:
        self._request_stop()
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()

    def _kill_child(self, polite: bool = True) -> None:
        """End the current child, politely first unless it is hung; never returns with it holding the camera."""
        proc = self._proc
        self._request_stop()
        deadline = time.monotonic() + (1.5 if polite else 0.0)
        while proc.is_alive() and time.monotonic() < deadline:
            self._pump(0.05)  # keep the pipe drained so the child is never stuck in send()
        for end in (proc.terminate, proc.kill) if polite else (proc.kill,):
            if not proc.is_alive():
                break
            end()
            proc.join(timeout=1.0)
        if proc.is_alive():
            log.error("sensor process %s survived SIGKILL (stuck in the kernel?)", proc.pid)
        try:
            self._conn.close()
        except OSError:
            pass
        self._poller = None

    def _respawn(self, reason: str, hung: bool = False) -> None:
        """Replace the child (now, or after the backoff when respawns keep failing)."""
        now = time.monotonic()
        if self._state != "down":
            log.warning("sensor process: %s; starting a new one (respawn #%d)", reason, self.respawns + 1)
            self.last_error = reason
            self._state = "stopping"  # drained while it dies, never attached
            self._kill_child(polite=not hung)
            self._base_dropped += self._child_dropped
            self._base_restarts += self._child_restarts
            self._base_lost += self._child_lost
            self._child_dropped = self._child_restarts = self._child_lost = 0
            self._seq_base = self._last_seq + 1
            self._imu_chunks = []
            self._state = "down"
            wait = RESPAWN_BACKOFF_S[min(self._backoff_i, len(RESPAWN_BACKOFF_S) - 1)]
            self._backoff_i += 1
            self._next_spawn_t = now + wait
            if wait > 0.0:
                log.warning("sensor process: next start in %.0f s", wait)
        if now >= self._next_spawn_t:
            self.respawns += 1
            self._fresh_child = True
            self._spawn()

    def _supervise(self) -> None:
        """No frame this time: decide whether the child is still worth waiting for."""
        if self.respawn_s <= 0.0 or self._closed:
            return
        now = time.monotonic()
        if self._state == "down":
            self._respawn(self.last_error or "down")
            return
        if self._error is not None and self._error != "the sensor process closed its pipe":
            self._respawn(f"it failed: {self._error}")
        elif not self._proc.is_alive() and not self._ready(0.0):
            self._respawn(f"it died (exit code {self._proc.exitcode})")
        elif self._error is not None:
            self._proc.join(timeout=0.5)
            self._respawn(f"it died (exit code {self._proc.exitcode})" if not self._proc.is_alive()
                          else "it closed its pipe")
        elif self._state == "starting" and now - self._spawn_t > START_TIMEOUT_S:
            self._respawn(f"it did not open the camera within {START_TIMEOUT_S:.0f} s")
        elif self._state == "running" and now - self._last_msg_t > self.hang_s:
            self._respawn(f"it has said nothing for {now - self._last_msg_t:.1f} s (hung)", hung=True)
        elif self._state == "running" and now - self._last_frame_t > self.respawn_s:
            self._respawn(f"no frame for {now - self._last_frame_t:.1f} s")

    @property
    def dropped(self) -> int:
        return self._base_dropped + self._child_dropped + self._skipped

    @dropped.setter
    def dropped(self, value: int) -> None:  # SourceBase.__init__ assigns it
        self._skipped = int(value)

    @property
    def restarts(self) -> int:
        """Camera reopens inside the children plus children replaced."""
        return self._base_restarts + self._child_restarts + self.respawns

    @property
    def lost(self) -> int:
        """Frames the sensor sent that never reached the source (gaps in the sensor timestamps)."""
        return self._base_lost + self._child_lost

    @property
    def streaming(self) -> bool:
        return self._state == "running" and not self._closed

    # ----------------------------------------------------------------------------- pipe
    def _ready(self, timeout_s: float) -> bool:
        try:
            if self._poller is not None:
                return bool(self._poller.poll(max(0.0, timeout_s) * 1000.0))
            return self._conn.poll(timeout_s)
        except (OSError, ValueError):
            return False

    def _pump(self, timeout_s: float = 0.0) -> None:
        """Take everything waiting on the pipe (blocking up to timeout_s for the first message)."""
        if self._state == "down":
            return
        conn = self._conn
        try:
            if not self._ready(timeout_s):
                return
            self._last_msg_t = time.monotonic()
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
                elif tag == b"N":
                    self._pending_info = pickle.loads(msg[1:])
                elif tag == b"E":
                    self._error = msg[1:].decode("utf-8", "replace")
                if not self._ready(0.0):
                    break
        except (EOFError, OSError):
            if self._error is None and not self._closed:
                self._error = "the sensor process closed its pipe"
        if self._state == "starting" and self._pending_info is not None and self._shm is not None:
            info, self._pending_info = self._pending_info, None
            same = (int(info["width"]), int(info["height"]), int(info["channels"])) == \
                (self.width, self.height, self._channels)
            if not same:
                self._error = f"the new sensor process opened {info['width']}x{info['height']}, not " \
                              f"{self.width}x{self.height}"
                return
            self._attach(info)
            log.warning("sensor process %d: %s streaming again", self.child_pid, self.name)

    def start(self) -> None:
        self._started = True  # the child streams from the moment it has the ring

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._state != "down":
            self._state = "stopping"
            self._request_stop()
            deadline = time.monotonic() + 6.0   # the camera's own stop can take 3 s on a stalled link
            while self._proc.is_alive() and time.monotonic() < deadline:
                self._pump(0.05)                # keep the pipe drained so the child is never stuck in send()
            if self._proc.is_alive():
                log.warning("sensor process did not exit: terminating it")
                self._proc.terminate()
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1.0)
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
        respawning = self.respawn_s > 0.0
        while True:
            self._pump(0.0)
            if self._latest is None:
                if respawning:
                    self._supervise()
                elif self._error is not None:
                    raise RuntimeError(f"sensor process failed: {self._error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    if not respawning and not self._proc.is_alive():
                        raise RuntimeError(f"sensor process died (exit code {self._proc.exitcode})")
                    return None
                if self._state == "down":
                    time.sleep(min(remaining, 0.02))
                else:
                    self._pump(min(remaining, 0.05))
                continue
            slot, stamp, seq, t_exp, t_cap, exposure_us, gain, child_dropped, restarts, gap, lost = self._latest
            self._latest = None
            head = self._heads[slot]
            o = slot * self._slot_bytes + HEADER
            image = self._buf[o:o + self._frame_bytes].copy()
            if int(head[0]) != stamp or int(head[1]) != stamp:
                self._skipped += 1  # the ring lapped this reader mid-copy: the frame is torn, wait for the next
                continue
            self._child_dropped = int(child_dropped)
            self._child_restarts = int(restarts)
            self._child_lost = int(lost)
            self._last_frame_t = time.monotonic()
            if gap or self._fresh_child:
                self.gap_after_restart = True
            if self._fresh_child:  # a replacement child delivers: the next failure starts the backoff over
                self._fresh_child = False
                self._backoff_i = 0
            seq = int(seq) + self._seq_base
            if seq <= self._last_seq:
                seq = self._last_seq + 1
            self._last_seq = seq
            return Frame(image=image.reshape(self._shape), seq=seq, t_exposure_ns=int(t_exp),
                         t_capture_ns=int(t_cap), exposure_us=float(exposure_us), gain=float(gain))

    def imu(self):
        return self._imu
