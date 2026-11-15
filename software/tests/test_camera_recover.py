"""RecoveringSource: a stalled or crashed live camera is reopened, seq stays continuous."""

from __future__ import annotations

import numpy as np

from tbot.camera.recover import RecoveringSource
from tbot.types import Frame


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class FakeCam:
    """Delivers `good` frames, then behaves per `then`: 'stall' (None forever) or 'raise'."""

    def __init__(self, clock: FakeClock, good: int, then: str = "stall", fail_open: bool = False):
        if fail_open:
            raise RuntimeError("camera busy")
        self.clock = clock
        self.good = good
        self.then = then
        self.n = 0
        self.started = self.stopped = False
        self.dropped = 2
        self.name = "fake"

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def imu(self):
        return None

    def read(self, timeout_s: float = 1.0):
        if self.n < self.good:
            self.n += 1
            self.clock.t += 0.02
            return Frame(image=np.zeros((2, 2), np.uint8), seq=self.n - 1, t_exposure_ns=0, t_capture_ns=0)
        self.clock.t += timeout_s
        if self.then == "raise":
            raise RuntimeError("camera capture thread failed: frontend timeout")
        return None


def _make(plan, clock):
    """Factory that returns cameras according to plan: list of (good, then, fail_open)."""
    made = []

    def factory():
        good, then, fail_open = plan[min(len(made), len(plan) - 1)]
        made.append(None)
        cam = FakeCam(clock, good, then, fail_open)
        made[-1] = cam
        return cam

    return factory, made


def _drain(src, n_reads):
    frames = []
    for _ in range(n_reads):
        f = src.read(timeout_s=0.1)
        if f is not None:
            frames.append(f)
    return frames


def test_stall_reopens_and_keeps_seq_continuous():
    clock = FakeClock()
    factory, made = _make([(5, "stall", False), (5, "stall", False)], clock)
    src = RecoveringSource(factory, stall_s=0.5, backoff_s=0.0, clock=clock, sleep=clock.sleep)
    src.start()
    frames = _drain(src, 30)
    assert len(made) >= 2 and made[0].stopped
    assert src.restarts >= 1
    assert [f.seq for f in frames] == list(range(len(frames)))
    assert len(frames) >= 10
    assert src.endless is True
    assert src.name == "fake"  # passthrough to the live source
    assert src.dropped >= 4    # drops of the replaced camera are kept


def test_capture_crash_reopens_and_flags_the_gap():
    clock = FakeClock()
    factory, made = _make([(3, "raise", False), (100, "stall", False)], clock)
    src = RecoveringSource(factory, stall_s=0.5, backoff_s=0.0, clock=clock, sleep=clock.sleep)
    src.start()
    assert len(_drain(src, 3)) == 3
    assert src.read(timeout_s=0.1) is None   # the crash is absorbed, not raised
    assert src.restarts == 1 and src.gap_after_restart
    f = src.read(timeout_s=0.1)
    assert f is not None and f.seq == 3


def test_failed_reopens_back_off_and_eventually_recover():
    clock = FakeClock()
    plan = [(2, "stall", False), (0, "stall", True), (0, "stall", True), (1000, "stall", False)]
    factory, made = _make(plan, clock)
    src = RecoveringSource(factory, stall_s=0.5, backoff_s=0.5, backoff_max_s=2.0, clock=clock, sleep=clock.sleep)
    src.start()
    frames = _drain(src, 200)
    assert src.failed_opens == 2
    assert src.restarts == 1
    assert len(frames) > 10
    assert [f.seq for f in frames] == list(range(len(frames)))


def test_dead_camera_is_retried_with_backoff_not_hammered():
    clock = FakeClock()
    factory, made = _make([(0, "stall", False)], clock)
    src = RecoveringSource(factory, stall_s=0.5, open_timeout_s=1.0, backoff_s=1.0, backoff_max_s=4.0,
                           clock=clock, sleep=clock.sleep)
    src.start()
    _drain(src, 300)  # ~30 s of fake time at 0.1 s per read
    assert 3 <= len(made) <= 12
