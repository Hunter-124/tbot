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
    # motion front end (vision/motion.py); the legacy detector leaves these at their defaults
    stab: tuple[float, float] | None = None  # centroid in background-stabilised pixels (scenery stands still there)
    # the same as a unit direction in the detector's stabilised frame: the plane of `stab` is the picture of the
    # first frame, stretched without bound as the gun turns 60-90 degrees away from it; a direction is not
    stab_dir: tuple[float, float, float] | None = None
    mass: float = 0.0  # summed residual against the registered background, DN * px
    reg_ok: bool = True  # stab rests on image registration (False: on the gyro alone, e.g. open sky)


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

    def sleep_until(self, t_ns: int) -> int:
        """Wait for t_ns; returns the time it then is."""
        now = time.monotonic_ns()
        while now < t_ns:
            time.sleep((t_ns - now) / 1e9)
            now = time.monotonic_ns()
        return now


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

    def sleep_until(self, t_ns: int) -> int:
        """Wait for t_ns; returns the time it then is. A frozen clock simply moves there."""
        if self._frozen:
            if t_ns > self._t_ns:
                self._t_ns = int(t_ns)
            return self._t_ns
        now = self.now_ns()
        while now < t_ns:
            time.sleep((t_ns - now) / 1e9)
            nxt = self.now_ns()
            if nxt <= now:  # a clock that does not move cannot be waited on
                break
            now = nxt
        return now


class ImuSource(Protocol):
    """Stream of calibrated gun-frame IMU samples (Mpu6050, NullImu, replay, synthetic)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self) -> list[ImuSample]:
        """All new samples since the last call, time ordered."""
        ...
