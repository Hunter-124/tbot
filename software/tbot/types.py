"""Shared data types and clocks (contract: docs/ARCHITECTURE.md).

All timestamps are int nanoseconds on time.monotonic_ns()'s clock, or on a SimClock standing in for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class Frame:
    """One upright camera frame.

    image: (H, W) uint8 mono or (H, W, 3) uint8 BGR.
    t_exposure_ns: mid-exposure time. t_capture_ns: time the frame reached user space.
    """

    image: np.ndarray
    seq: int
    t_exposure_ns: int
    t_capture_ns: int
    exposure_us: float = 0.0
    gain: float = 1.0

    @property
    def is_color(self) -> bool:
        return self.image.ndim == 3

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class ImuSample:
    """Calibrated IMU sample in the GUN frame (+X muzzle, +Y gun-left, +Z up)."""

    t_ns: int
    gyro: np.ndarray  # (3,) rad/s
    accel: np.ndarray  # (3,) m/s^2 (specific force: reads +g on +Z when level and still)
    temp_c: float = float("nan")


@dataclass(slots=True)
class Detection:
    """A clay candidate in one frame. (u, v) distorted pixel centroid in the upright runtime image."""

    t_ns: int  # frame mid-exposure time
    u: float
    v: float
    area_px: float
    diameter_px: float  # equivalent-circle diameter, sqrt(4*area/pi), or second-moment based
    contrast: float  # signed peak contrast in DN (positive = brighter than background)
    snr: float  # |contrast| / noise sigma
    score: float  # 0..1
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass(slots=True)
class Track:
    """A tracked clay in the WORLD frame (azimuth/elevation, constant-acceleration state)."""

    track_id: int
    t_ns: int  # time of the state estimate (last update)
    x: np.ndarray  # (6,) [az, az_rate, az_acc, el, el_rate, el_acc]  rad, rad/s, rad/s^2
    P: np.ndarray  # (6, 6) covariance
    range_m: float
    range_sigma_m: float
    hits: int = 0
    misses: int = 0
    age: int = 0  # frames since creation
    confirmed: bool = False
    last_detection: Detection | None = None
    first_t_ns: int = 0
    s: np.ndarray | None = None  # cartesian tracker: (9,) world [p, v, a] in m, m/s, m/s^2 (camera origin)
    S: np.ndarray | None = None  # cartesian tracker: (9, 9) covariance

    @property
    def az(self) -> float:
        return float(self.x[0])

    @property
    def el(self) -> float:
        return float(self.x[3])


@dataclass(slots=True)
class FireDecision:
    fire: bool
    t_ns: int  # decision time ("now")
    reason: str  # why (not) — always set
    track_id: int | None = None
    t_exit_ns: int | None = None  # predicted shot exit time if fired now
    miss_rad: float = float("nan")  # predicted miss angle if fired now
    pattern_rad: float = float("nan")  # angular radius of the effective shot pattern at the clay distance
    sigma_rad: float = float("nan")  # combined 1-sigma angular uncertainty (track, latency jitter, range)
    p_hit: float = float("nan")  # P(clay inside the pattern) for a shot commanded now
    range_m: float = float("nan")
    tof_s: float = float("nan")
    lead_dir_w: np.ndarray | None = None  # where the clay will be (world dir from muzzle) at shot arrival
    bore_dir_w: np.ndarray | None = None  # predicted bore direction at t_exit
    extra: dict = field(default_factory=dict)


class Clock(Protocol):
    def now_ns(self) -> int: ...


class MonotonicClock:
    """Real clock: time.monotonic_ns()."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.monotonic_ns()


class SimClock:
    """Simulated clock for synthetic/replay sources.

    now_ns() = last set() value + real wall time elapsed since that set(), so the real processing
    latency of the pipeline is reproduced on simulated timestamps.
    """

    __slots__ = ("_t_ns", "_wall0", "_frozen")

    def __init__(self, t0_ns: int = 1_000_000_000, frozen: bool = False):
        self._t_ns = int(t0_ns)
        self._wall0 = time.perf_counter_ns()
        self._frozen = frozen  # frozen=True: now_ns() ignores wall time (fully deterministic tests)

    def set(self, t_ns: int) -> None:
        self._t_ns = int(t_ns)
        self._wall0 = time.perf_counter_ns()

    def advance(self, dt_ns: int) -> None:
        self._t_ns += int(dt_ns)

    def now_ns(self) -> int:
        if self._frozen:
            return self._t_ns
        return self._t_ns + (time.perf_counter_ns() - self._wall0)


class ImuSource(Protocol):
    """Stream of calibrated gun-frame IMU samples (Mpu6050, NullImu, replay, synthetic)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self) -> list[ImuSample]:
        """All new samples since the last call, time ordered."""
        ...
