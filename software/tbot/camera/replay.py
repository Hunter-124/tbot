"""Recorded-session format (reader) — see docs/ARCHITECTURE.md "Session format".

A session is a directory:
  meta.json      {"version":1,"width","height","channels","dtype":"uint8","fps","frames",
                  "source","config":{...},"calibration":{...},"created","complete"}
  frames.bin     raw concatenated frames, C order, uint8
  frames.csv     seq,t_exposure_ns,t_capture_ns,exposure_us,gain
  imu.csv        t_ns,gx,gy,gz,ax,ay,az,temp_c   (gun frame, rad/s, m/s^2)
  events.jsonl   one JSON object per line, each with "t_ns" and "type"

The writer (tbot.app.record.SessionWriter) imports the names below so both ends stay in step.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from tbot.camera.base import SourceBase
from tbot.types import Frame, ImuSample, ImuSource, SimClock

__all__ = [
    "ReplaySource",
    "SESSION_VERSION",
    "META_JSON",
    "FRAMES_BIN",
    "FRAMES_CSV",
    "IMU_CSV",
    "EVENTS_JSONL",
    "FRAMES_CSV_HEADER",
    "IMU_CSV_HEADER",
]

SESSION_VERSION = 1
META_JSON = "meta.json"
FRAMES_BIN = "frames.bin"
FRAMES_CSV = "frames.csv"
IMU_CSV = "imu.csv"
EVENTS_JSONL = "events.jsonl"
FRAMES_CSV_HEADER = ["seq", "t_exposure_ns", "t_capture_ns", "exposure_us", "gain"]
IMU_CSV_HEADER = ["t_ns", "gx", "gy", "gz", "ax", "ay", "az", "temp_c"]

_SIM_T0_NS = 1_000_000_000


def _read_csv(path: Path, header: list[str]) -> Iterator[list[str]]:
    """Yield data rows of a session CSV, checking the header."""
    if not path.exists():
        return
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return
        if [c.strip() for c in first] != header:
            raise ValueError(f"{path}: expected header {','.join(header)}, got {','.join(first)}")
        for row in reader:
            if row:
                yield row


class _ReplayImu:
    """ImuSource over a session's imu.csv, released in step with the frames being replayed."""

    def __init__(self, samples: list[ImuSample], limit_ns: Callable[[], int | None]) -> None:
        self._samples = samples
        self._limit_ns = limit_ns
        self._next = 0

    def start(self) -> None:
        """No-op: the samples come from the file."""

    def stop(self) -> None:
        """No-op."""

    def read(self) -> list[ImuSample]:
        limit = self._limit_ns()
        if limit is None:
            return []
        start = self._next
        n = len(self._samples)
        i = start
        while i < n and self._samples[i].t_ns <= limit:
            i += 1
        self._next = i
        return self._samples[start:i]


class ReplaySource(SourceBase):
    """Replays a recorded session: memory-mapped frames on a SimClock.

    realtime=False (default): every frame is delivered, as fast as the consumer asks for them.
    realtime=True: frames whose capture time is already behind the simulated clock are skipped
    (counted in `dropped`), reproducing what a live source would have done to a slow consumer.
    """

    def __init__(self, path: str | Path, realtime: bool = False, work_ms: float = 0.0) -> None:
        """work_ms > 0: the loop's own work per frame (detect + track) is modelled instead of taken from this
        computer: the clock stands at each frame's arrival, moves on by work_ms when the pipeline reports the
        frame processed, and the IMU samples of that time are there for the solver. A replay then repeats
        exactly and decides as late as the gun does."""
        super().__init__()
        self._work_ns = int(round(float(work_ms) * 1e6))
        self.path = Path(path).expanduser()
        meta_path = self.path / META_JSON
        if not meta_path.exists():
            raise FileNotFoundError(f"not a session directory (no {META_JSON}): {self.path}")
        self.meta: dict = json.loads(meta_path.read_text())
        dtype = self.meta.get("dtype", "uint8")
        if dtype != "uint8":
            raise ValueError(f"{meta_path}: unsupported frame dtype {dtype!r} (only uint8)")
        self.width = int(self.meta["width"])
        self.height = int(self.meta["height"])
        self.channels = int(self.meta["channels"])
        if self.channels not in (1, 3):
            raise ValueError(f"{meta_path}: channels must be 1 or 3, got {self.channels}")
        self.is_color = self.channels == 3
        self.nominal_fps = float(self.meta.get("fps", 0.0))
        self.name = f"replay:{self.path}"
        self.realtime = bool(realtime)

        self._rows = [
            (int(r[0]), int(r[1]), int(r[2]), float(r[3]), float(r[4]))
            for r in _read_csv(self.path / FRAMES_CSV, FRAMES_CSV_HEADER)
        ]
        frame_bytes = self.height * self.width * self.channels
        bin_path = self.path / FRAMES_BIN
        n_bin = bin_path.stat().st_size // frame_bytes if bin_path.exists() else 0
        # A session killed mid-write can have fewer frames than meta claims: trust what is on disk.
        n = min(len(self._rows), int(n_bin))
        self.n_frames = n
        shape = (n, self.height, self.width) if self.channels == 1 else (n, self.height, self.width, self.channels)
        if n > 0:
            self._frames = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=shape)
        else:
            self._frames = np.empty(shape, dtype=np.uint8)

        self._samples = [
            ImuSample(
                t_ns=int(r[0]),
                gyro=np.array([float(r[1]), float(r[2]), float(r[3])], dtype=np.float64),
                accel=np.array([float(r[4]), float(r[5]), float(r[6])], dtype=np.float64),
                temp_c=float(r[7]),
            )
            for r in _read_csv(self.path / IMU_CSV, IMU_CSV_HEADER)
        ]
        events_path = self.path / EVENTS_JSONL
        self.events: list[dict] = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                line = line.strip()
                if line:
                    self.events.append(json.loads(line))

        self._index = 0
        self._t_capture_ns: int | None = None
        self._imu: _ReplayImu | None = None
        self.clock = SimClock(self._rows[0][2] if self._rows else _SIM_T0_NS, frozen=self._work_ns > 0)

    def processed(self) -> None:
        if self._work_ns > 0:
            self.clock.advance(self._work_ns)

    def _imu_limit_ns(self) -> int | None:
        if self._t_capture_ns is None:
            return None
        return max(self._t_capture_ns, int(self.clock.now_ns())) if self._work_ns > 0 else self._t_capture_ns

    # ----------------------------------------------------------------------------- lifecycle
    def stop(self) -> None:
        self._frames = np.empty((0, self.height, self.width), dtype=np.uint8)

    # ----------------------------------------------------------------------------- consumer
    def read(self, timeout_s: float = 1.0) -> Frame | None:
        i = self._index
        if i >= self.n_frames:
            return None
        if self.realtime:
            now = self.clock.now_ns()
            while i + 1 < self.n_frames and self._rows[i + 1][2] <= now:
                i += 1
                self.dropped += 1
            wait_ns = self._rows[i][2] - self.clock.now_ns()
            if wait_ns > 0:
                if wait_ns > float(timeout_s) * 1e9:
                    self._index = i
                    time.sleep(float(timeout_s))
                    return None
                time.sleep(wait_ns / 1e9)
        seq, t_exposure_ns, t_capture_ns, exposure_us, gain = self._rows[i]
        self._index = i + 1
        self._t_capture_ns = t_capture_ns
        self.clock.set(t_capture_ns)
        return Frame(
            image=self._frames[i],
            seq=seq,
            t_exposure_ns=t_exposure_ns,
            t_capture_ns=t_capture_ns,
            exposure_us=exposure_us,
            gain=gain,
        )

    def imu(self) -> ImuSource | None:
        """The session's IMU stream, or None when the recording has no samples."""
        if not self._samples:
            return None
        if self._imu is None:
            self._imu = _ReplayImu(self._samples, self._imu_limit_ns)
        return self._imu
