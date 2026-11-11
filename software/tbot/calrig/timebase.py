"""Clocks for the rig host code: the real monotonic clock and a deterministic virtual one for --sim and tests.

The Pi's IMU stamps samples with `time.monotonic_ns()` (tbot.types.MonotonicClock, used by Mpu6050), so the rig
host clock is the same call: firmware `t_us` is converted to it by clocksync and everything lands on one axis.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

__all__ = ["Timebase", "RealClock", "VirtualClock"]


class Timebase(Protocol):
    virtual: bool

    def now_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class RealClock:
    """time.monotonic_ns() / time.sleep()."""

    virtual = False

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        if seconds > 0.0:
            time.sleep(seconds)


class VirtualClock:
    """Deterministic simulated time. sleep() advances it in `tick_s` steps and calls every listener per tick.

    Listeners (FakeRig, SimImu) are called in registration order with the new time in ns, so a simulated rig
    and its IMU see exactly the same timeline and a 20 s procedure costs only CPU time, not 20 s.
    """

    virtual = True

    def __init__(self, start_ns: int = 1_000_000_000_000, tick_s: float = 0.001):
        self._now = int(start_ns)
        self.tick_ns = max(1, int(round(tick_s * 1e9)))
        self._listeners: list[Callable[[int], None]] = []

    def now_ns(self) -> int:
        return self._now

    def add_listener(self, fn: Callable[[int], None]) -> None:
        self._listeners.append(fn)

    def sleep(self, seconds: float) -> None:
        target = self._now + int(round(seconds * 1e9))
        while self._now < target:
            self._now = min(self._now + self.tick_ns, target)
            for fn in self._listeners:
                fn(self._now)
