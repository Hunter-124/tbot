"""Session recording: `SessionWriter`, `RingRecorder` and `tbot record`.

The on-disk format is documented in tbot/camera/replay.py (the reader) and docs/ARCHITECTURE.md.
Both writers run their I/O on a background thread so the capture/processing loop is never blocked
by the SD card.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from tbot.camera import open_source
from tbot.camera.replay import (
    EVENTS_JSONL,
    FRAMES_BIN,
    FRAMES_CSV,
    FRAMES_CSV_HEADER,
    IMU_CSV,
    IMU_CSV_HEADER,
    META_JSON,
    SESSION_VERSION,
)
from tbot.config import add_common_args, load_from_args, now_iso, resolve_path, save_json
from tbot.types import Frame, ImuSample, SimClock

__all__ = ["SessionWriter", "RingRecorder", "session_meta_extra", "main"]

_SENTINEL = object()


def session_meta_extra(cfg, calib=None, source: str | None = None) -> dict:
    """The provenance block stored in meta.json: source spec, full config, calibration in force."""
    calibration: dict = {}
    if calib is not None:
        for key in ("intrinsics", "boresight", "imu", "latency"):
            value = getattr(calib, key, None)
            if value:
                calibration[key] = value
    return {"source": source or cfg.camera.source, "config": cfg.to_dict(), "calibration": calibration}


class SessionWriter:
    """Writes one session directory from a background thread.

    `add_frame`/`add_imu`/`add_event` hand the data to a bounded queue and return immediately; when
    the queue is full the item is dropped and counted in `dropped` (the capture loop must never
    stall). `drop_when_full=False` makes the calls block instead — used by RingRecorder, which dumps
    from its own thread and must not lose frames.
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        channels: int,
        fps: float,
        meta_extra: dict | None = None,
        queue_size: int = 128,
        drop_when_full: bool = True,
    ) -> None:
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 (mono) or 3 (BGR), got {channels}")
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.fps = float(fps)
        self.shape = (self.height, self.width) if self.channels == 1 else (self.height, self.width, self.channels)
        self.frame_bytes = self.height * self.width * self.channels
        self.meta_extra = dict(meta_extra or {})
        self.dropped = 0
        self.frames_written = 0
        self.imu_written = 0
        self.events_written = 0
        self.closed = False

        self._drop_when_full = bool(drop_when_full)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._error: BaseException | None = None
        self._bin = open(self.path / FRAMES_BIN, "wb")
        self._frames_csv = open(self.path / FRAMES_CSV, "w", newline="")
        self._imu_csv = open(self.path / IMU_CSV, "w", newline="")
        self._events = open(self.path / EVENTS_JSONL, "w")
        self._frames_writer = csv.writer(self._frames_csv)
        self._frames_writer.writerow(FRAMES_CSV_HEADER)
        self._imu_writer = csv.writer(self._imu_csv)
        self._imu_writer.writerow(IMU_CSV_HEADER)
        self._write_meta(complete=False)
        self._thread = threading.Thread(target=self._run, name="session-writer", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------------------------- producer side
    def add_frame(self, frame: Frame) -> bool:
        """Queue one frame. Returns False when it was dropped (queue full)."""
        image = frame.image
        if image.dtype != np.uint8 or tuple(image.shape) != self.shape:
            raise ValueError(
                f"frame {image.shape} {image.dtype} does not match the session format {self.shape} uint8"
            )
        return self._put(("frame", frame))

    def add_imu(self, samples) -> bool:
        """Queue IMU samples (gun frame). Empty lists are ignored."""
        samples = list(samples)
        if not samples:
            return True
        return self._put(("imu", samples))

    def add_event(self, event: dict) -> bool:
        """Queue one event object; it must carry "t_ns" and "type"."""
        if "t_ns" not in event or "type" not in event:
            raise ValueError(f'event needs "t_ns" and "type", got keys {sorted(event)}')
        return self._put(("event", dict(event)))

    def _put(self, item) -> bool:
        if self.closed:
            raise RuntimeError(f"session writer for {self.path} is closed")
        if self._drop_when_full:
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self.dropped += 1
                return False
        while True:
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                if not self._thread.is_alive():
                    self.dropped += 1
                    return False

    # ----------------------------------------------------------------------------- writer thread
    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    return
                kind, payload = item
                if kind == "frame":
                    self._write_frame(payload)
                elif kind == "imu":
                    self._write_imu(payload)
                else:
                    self._events.write(json.dumps(payload) + "\n")
                    self.events_written += 1
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller in close()
            self._error = exc
            # Drain so a blocking producer cannot deadlock on a dead writer.
            while True:
                try:
                    if self._queue.get_nowait() is _SENTINEL:
                        return
                except queue.Empty:
                    return

    def _write_frame(self, frame: Frame) -> None:
        image = frame.image
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        self._bin.write(image.data)
        self._frames_writer.writerow(
            [frame.seq, frame.t_exposure_ns, frame.t_capture_ns, repr(float(frame.exposure_us)), repr(float(frame.gain))]
        )
        self.frames_written += 1

    def _write_imu(self, samples: list[ImuSample]) -> None:
        for s in samples:
            g = np.asarray(s.gyro, dtype=np.float64).reshape(3)
            a = np.asarray(s.accel, dtype=np.float64).reshape(3)
            # repr(float(...)) round-trips exactly; repr() of a numpy scalar does not (numpy 2.x).
            self._imu_writer.writerow(
                [int(s.t_ns)]
                + [repr(float(v)) for v in (g[0], g[1], g[2], a[0], a[1], a[2], s.temp_c)]
            )
            self.imu_written += 1

    # ----------------------------------------------------------------------------- lifecycle
    def _write_meta(self, complete: bool) -> None:
        meta = dict(self.meta_extra)
        meta.update({
            "version": SESSION_VERSION,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": "uint8",
            "fps": self.fps,
            "frames": self.frames_written,
            "created": now_iso(),
            "complete": complete,
        })
        save_json(self.path / META_JSON, meta)

    def close(self) -> Path:
        """Flush, stop the thread and finalise meta.json. Returns the session directory."""
        if self.closed:
            return self.path
        self.closed = True
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=30.0)
        for handle in (self._bin, self._frames_csv, self._imu_csv, self._events):
            handle.close()
        self._write_meta(complete=self._error is None)
        if self._error is not None:
            raise RuntimeError(f"session writer failed for {self.path}: {self._error}") from self._error
        return self.path

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _PendingDump:
    __slots__ = ("t0_ns", "t1_ns", "path", "tag")

    def __init__(self, t0_ns: int, t1_ns: int, path: Path, tag: str) -> None:
        self.t0_ns = t0_ns
        self.t1_ns = t1_ns
        self.path = path
        self.tag = tag


class RingRecorder:
    """RAM ring of the most recent frames/IMU/events, dumped around a trigger.

    Holds `record.preroll_s` of frames, capped at `record.max_ram_mb`; `trigger(t_ns)` reserves a
    session directory and dumps `[t_ns - preroll_s, t_ns + postroll_s]` once the postroll frames have
    arrived (or at `close()`), on a background thread. Window membership uses t_capture_ns, the time
    the frame reached user space, which is the clock the pipeline triggers on.
    """

    def __init__(self, cfg, width: int, height: int, channels: int, fps: float, meta_extra: dict | None = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.fps = float(fps)
        self.dir = resolve_path(cfg.record.dir)
        self.preroll_ns = int(round(float(cfg.record.preroll_s) * 1e9))
        self.postroll_ns = int(round(float(cfg.record.postroll_s) * 1e9))
        self.max_bytes = int(cfg.record.max_ram_mb) * 1024 * 1024
        self.meta_extra = dict(meta_extra) if meta_extra is not None else session_meta_extra(cfg)
        self.evicted = 0  # frames thrown away by the RAM cap while a dump still wanted them
        self.sessions: list[Path] = []

        self._lock = threading.Lock()
        self._frames: deque[Frame] = deque()
        self._imu: deque[ImuSample] = deque()
        self._events: deque[dict] = deque()
        self._bytes = 0
        self._pending: list[_PendingDump] = []
        self._threads: list[threading.Thread] = []
        self._issued: set[Path] = set()
        self._closed = False

    # ----------------------------------------------------------------------------- introspection
    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def ram_bytes(self) -> int:
        return self._bytes

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ----------------------------------------------------------------------------- input
    def push(self, frame: Frame) -> None:
        ready: list[tuple[_PendingDump, list, list, list]] = []
        with self._lock:
            self._frames.append(frame)
            self._bytes += int(frame.image.nbytes)
            self._trim_locked(int(frame.t_capture_ns))
            newest = int(frame.t_capture_ns)
            for dump in list(self._pending):
                if newest >= dump.t1_ns:
                    self._pending.remove(dump)
                    ready.append((dump, *self._snapshot_locked(dump)))
        for dump, frames, samples, events in ready:
            self._spawn_dump(dump, frames, samples, events)

    def push_imu(self, samples) -> None:
        with self._lock:
            self._imu.extend(samples)

    def event(self, event: dict) -> None:
        if "t_ns" not in event or "type" not in event:
            raise ValueError(f'event needs "t_ns" and "type", got keys {sorted(event)}')
        with self._lock:
            self._events.append(dict(event))

    # ----------------------------------------------------------------------------- trigger / dump
    def trigger(self, t_ns: int, tag: str = "trigger") -> Path:
        """Reserve a session for [t_ns - preroll, t_ns + postroll] and return its path."""
        t_ns = int(t_ns)
        path = self._reserve_path(tag)
        dump = _PendingDump(t_ns - self.preroll_ns, t_ns + self.postroll_ns, path, tag)
        ready = None
        with self._lock:
            newest = int(self._frames[-1].t_capture_ns) if self._frames else None
            if newest is not None and newest >= dump.t1_ns:
                ready = self._snapshot_locked(dump)
            else:
                self._pending.append(dump)
        if ready is not None:
            self._spawn_dump(dump, *ready)
        return path

    def close(self) -> list[Path]:
        """Dump everything still pending with the frames on hand and wait for the writers."""
        pending: list[tuple[_PendingDump, list, list, list]] = []
        with self._lock:
            self._closed = True
            for dump in self._pending:
                pending.append((dump, *self._snapshot_locked(dump)))
            self._pending.clear()
        for dump, frames, samples, events in pending:
            self._spawn_dump(dump, frames, samples, events)
        for thread in list(self._threads):
            thread.join(timeout=60.0)
        self._threads.clear()
        with self._lock:
            self._frames.clear()
            self._imu.clear()
            self._events.clear()
            self._bytes = 0
            return list(self.sessions)

    # ----------------------------------------------------------------------------- internals
    def _reserve_path(self, tag: str) -> Path:
        base = self.dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
        path, n = base, 1
        while path in self._issued or path.exists():
            path = base.with_name(f"{base.name}-{n}")
            n += 1
        self._issued.add(path)
        return path

    def _trim_locked(self, newest_ns: int) -> None:
        keep_from = newest_ns - self.preroll_ns
        for dump in self._pending:
            keep_from = min(keep_from, dump.t0_ns)
        while len(self._frames) > 1 and int(self._frames[0].t_capture_ns) < keep_from:
            self._bytes -= int(self._frames.popleft().image.nbytes)
        # Hard RAM cap: drop the oldest frames even if a pending dump wanted them.
        while len(self._frames) > 1 and self._bytes > self.max_bytes:
            self._bytes -= int(self._frames.popleft().image.nbytes)
            self.evicted += 1
        oldest = int(self._frames[0].t_capture_ns) if self._frames else keep_from
        horizon = min(keep_from, oldest)
        while self._imu and int(self._imu[0].t_ns) < horizon:
            self._imu.popleft()
        while self._events and int(self._events[0].get("t_ns", horizon)) < horizon:
            self._events.popleft()

    def _snapshot_locked(self, dump: _PendingDump) -> tuple[list, list, list]:
        frames = [f for f in self._frames if dump.t0_ns <= int(f.t_capture_ns) <= dump.t1_ns]
        samples = [s for s in self._imu if dump.t0_ns <= int(s.t_ns) <= dump.t1_ns]
        events = [e for e in self._events if dump.t0_ns <= int(e.get("t_ns", dump.t0_ns)) <= dump.t1_ns]
        return frames, samples, events

    def _spawn_dump(self, dump: _PendingDump, frames: list, samples: list, events: list) -> None:
        thread = threading.Thread(
            target=self._write_dump,
            args=(dump, frames, samples, events),
            name=f"ring-dump-{dump.tag}",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def _write_dump(self, dump: _PendingDump, frames: list, samples: list, events: list) -> None:
        meta = dict(self.meta_extra)
        meta["trigger"] = {"t_ns": dump.t0_ns + self.preroll_ns, "tag": dump.tag,
                           "window_ns": [dump.t0_ns, dump.t1_ns]}
        writer = SessionWriter(
            dump.path, self.width, self.height, self.channels, self.fps,
            meta_extra=meta, queue_size=64, drop_when_full=False,
        )
        try:
            writer.add_imu(samples)
            for event in events:
                writer.add_event(event)
            for frame in frames:
                writer.add_frame(frame)
        finally:
            writer.close()
        with self._lock:
            self.sessions.append(dump.path)


# ------------------------------------------------------------------------------------- tbot record
def _open_imu(cfg, calib, source):
    """tbot.imu.mpu6050.open_imu when that module is importable, else the source's own IMU."""
    try:
        from tbot.imu.mpu6050 import open_imu
    except ImportError:
        return source.imu()
    return open_imu(cfg, calib, source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot record",
        description="record a session (frames + timestamps + IMU) for replay and offline tuning",
    )
    add_common_args(parser)
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after this much frame time (default: until end of stream / Ctrl-C)")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="session directory (default: <record.dir>/YYYYmmdd-HHMMSS-rec)")
    parser.add_argument("--decimate", type=int, default=None, metavar="N",
                        help="write every Nth frame (default: record.decimate)")
    args = parser.parse_args(argv)
    cfg, calib = load_from_args(args)

    decimate = int(args.decimate if args.decimate is not None else cfg.record.decimate)
    if decimate < 1:
        parser.error("--decimate must be >= 1")
    if args.out:
        out = resolve_path(Path(args.out).expanduser())
    else:
        out = resolve_path(Path(cfg.record.dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-rec")

    source = open_source(cfg.camera.source, cfg, calib)
    simulated = isinstance(source.clock, SimClock)
    imu = _open_imu(cfg, calib, source)
    writer: SessionWriter | None = None
    kept = read_frames = imu_count = 0
    t_first: int | None = None
    t_last: int | None = None
    timeouts = 0
    status = 0

    print(f"tbot record: source {cfg.camera.source}, out {out}"
          + (f", {args.seconds:g} s" if args.seconds else "")
          + (f", decimate {decimate}" if decimate > 1 else ""))
    source.start()
    if imu is not None:
        imu.start()
    next_report = time.monotonic() + 0.5
    try:
        while True:
            frame = source.read(timeout_s=1.0)
            if frame is None:
                if simulated:
                    break
                timeouts += 1
                if timeouts >= 10:
                    print("\ntbot record: no frames for 10 s — giving up", file=sys.stderr)
                    status = 1
                    break
                continue
            timeouts = 0
            read_frames += 1
            if t_first is None:
                t_first = int(frame.t_capture_ns)
                writer = SessionWriter(
                    out, frame.width, frame.height, 3 if frame.is_color else 1, source.nominal_fps,
                    meta_extra=session_meta_extra(cfg, calib),
                )
                writer.add_event({"t_ns": t_first, "type": "record_start",
                                  "source": cfg.camera.source, "decimate": decimate})
            t_last = int(frame.t_capture_ns)
            if imu is not None:
                samples = imu.read()
                if samples:
                    imu_count += len(samples)
                    writer.add_imu(samples)
            if (read_frames - 1) % decimate == 0:
                writer.add_frame(frame)
                kept += 1
            now = time.monotonic()
            if now >= next_report:
                next_report = now + 0.5
                span_s = (t_last - t_first) / 1e9
                fps = kept / span_s if span_s > 0 else 0.0
                print(f"\r  {kept} frames  {span_s:6.1f} s  {fps:6.1f} fps  "
                      f"dropped {source.dropped}+{writer.dropped}  imu {imu_count}", end="", flush=True)
            if args.seconds is not None and (t_last - t_first) >= args.seconds * 1e9:
                break
    except KeyboardInterrupt:
        print("\ntbot record: interrupted")
    finally:
        if writer is not None and t_last is not None:
            writer.add_event({"t_ns": t_last, "type": "record_stop", "frames": kept})
        if imu is not None:
            imu.stop()
        source.stop()
        if writer is not None:
            writer.close()

    if writer is None:
        print("tbot record: no frames captured", file=sys.stderr)
        return 1
    span_s = (t_last - t_first) / 1e9 if t_first is not None and t_last is not None else 0.0
    print(f"\r  {kept} frames  {span_s:6.1f} s  "
          f"{kept / span_s if span_s > 0 else 0.0:6.1f} fps  "
          f"dropped {source.dropped}+{writer.dropped}  imu {imu_count}")
    print(f"tbot record: wrote {writer.frames_written} frames, {writer.imu_written} IMU samples to {out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
