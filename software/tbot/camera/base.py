"""FrameSource protocol (contract: docs/ARCHITECTURE.md)."""

from __future__ import annotations

from typing import Protocol

from tbot.types import Clock, Frame, ImuSource, MonotonicClock


class FrameSource(Protocol):
    width: int
    height: int
    is_color: bool
    nominal_fps: float
    clock: Clock
    name: str
    dropped: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        """Newest unread frame (older unread frames are dropped). None = timeout or end of stream."""
        ...

    def imu(self) -> ImuSource | None:
        """The source's own IMU stream (replay, synthetic), else None."""
        ...


class SourceBase:
    """Optional base for sources: context manager + defaults. Subclasses set the protocol attributes."""

    width: int = 0
    height: int = 0
    is_color: bool = False
    nominal_fps: float = 0.0
    name: str = "source"
    dropped: int = 0

    def __init__(self) -> None:
        self.clock: Clock = MonotonicClock()
        self.dropped = 0

    def start(self) -> None:  # pragma: no cover - overridden
        pass

    def stop(self) -> None:  # pragma: no cover - overridden
        pass

    def imu(self) -> ImuSource | None:
        return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
