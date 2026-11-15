"""RecoveringSource: keep a live camera streaming through CSI stalls and capture-thread crashes.

A live camera can stop delivering frames without ending (libcamera "Camera frontend has timed out",
a loose CSI ribbon, a crashed capture thread). The plain sources report that as "no frame", which the
pipeline would treat as the end of the stream. This wrapper instead tears the stalled source down and
opens a fresh one from a factory, with backoff, for as long as it keeps failing. Frames are renumbered
so `seq` stays strictly increasing across restarts; `restarts` and `gap_after_restart` let the
pipeline drop tracks that span the outage.

Frames lost during an outage are gone: recovery gets the camera back for the next clay, it cannot
recover the flight it missed.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any, Callable

from tbot.types import Frame

__all__ = ["RecoveringSource"]

log = logging.getLogger(__name__)


class RecoveringSource:
    """Wraps a live FrameSource built by `factory`; restarts it when it stalls or raises.

    stall_s: no frame for this long (after the first frame of a source instance) = stalled.
    open_timeout_s: a fresh source that has not produced its first frame within this = stalled.
    backoff_s / backoff_max_s: wait between failed attempts (doubles up to the max).
    stop_timeout_s: a stuck stop() is abandoned after this (it runs on a helper thread).
    """

    endless = True  # the pipeline must not treat a gap in frames as end of stream

    def __init__(self, factory: Callable[[], Any], stall_s: float = 0.5, open_timeout_s: float = 5.0,
                 backoff_s: float = 0.5, backoff_max_s: float = 5.0, stop_timeout_s: float = 3.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._factory = factory
        self.stall_s = float(stall_s)
        self.open_timeout_s = float(open_timeout_s)
        self.backoff_s = float(backoff_s)
        self.backoff_max_s = float(backoff_max_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self._clock = clock
        self._sleep = sleep
        self.restarts = 0
        self.failed_opens = 0
        self.last_error: str | None = None
        self.gap_after_restart = False   # the next frame is the first after a restart
        self._dropped_before = 0         # drops counted by earlier source instances
        self._seq_next = 0
        self._inner = factory()          # the first open raises: no camera at start-up is a real error
        self._started = False
        self._inner_started = False
        self._inner_t0 = 0.0
        self._last_frame_t: float | None = None
        self._next_attempt_t = 0.0
        self._backoff_cur = self.backoff_s

    # ------------------------------------------------------------------ protocol passthrough
    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def dropped(self) -> int:
        inner = self._inner
        return self._dropped_before + (int(getattr(inner, "dropped", 0)) if inner is not None else 0)

    def imu(self):
        return self._inner.imu() if self._inner is not None else None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._start_inner()

    def stop(self) -> None:
        self._started = False
        self._stop_inner()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _start_inner(self) -> None:
        self._inner.start()
        self._inner_started = True
        self._inner_t0 = self._clock()
        self._last_frame_t = None

    def _stop_inner(self) -> None:
        inner, self._inner_started = self._inner, False
        if inner is None:
            return
        self._dropped_before += int(getattr(inner, "dropped", 0))
        done = threading.Event()

        def _stop() -> None:
            try:
                inner.stop()
            except Exception as exc:  # noqa: BLE001 - a broken camera may fail to stop cleanly
                log.warning("camera: stop() of the stalled source failed: %s", exc)
            finally:
                done.set()

        threading.Thread(target=_stop, name="camera-stop", daemon=True).start()
        if not done.wait(self.stop_timeout_s):
            log.warning("camera: stop() of the stalled source hung for %.1f s; abandoning it", self.stop_timeout_s)

    # ------------------------------------------------------------------ recovery
    def _restart(self, reason: str) -> None:
        """Tear the current source down and try to bring a new one up (one attempt, then backoff)."""
        now = self._clock()
        if now < self._next_attempt_t:
            return
        log.warning("camera: %s; restarting the camera (restart #%d)", reason, self.restarts + 1)
        self.last_error = reason
        self._stop_inner()
        self._inner = None
        try:
            self._inner = self._factory()
            self._start_inner()
        except Exception as exc:  # noqa: BLE001 - keep trying; the camera may come back
            self.failed_opens += 1
            self.last_error = f"reopen failed: {exc}"
            log.warning("camera: reopen failed (%s); next try in %.1f s", exc, self._backoff_cur)
            if self._inner is not None:
                self._stop_inner()
                self._inner = None
            self._next_attempt_t = self._clock() + self._backoff_cur
            self._backoff_cur = min(self.backoff_max_s, 2.0 * self._backoff_cur)
            return
        self.restarts += 1
        self.gap_after_restart = True
        # a source that opens but stays dead gets the backoff too, so a hard fault is not hammered
        self._next_attempt_t = self._clock() + self._backoff_cur
        self._backoff_cur = min(self.backoff_max_s, 2.0 * self._backoff_cur)

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        if not self._started:
            self.start()
        if self._inner is None or not self._inner_started:
            self._restart("camera is down")
            if self._inner is None or not self._inner_started:
                self._sleep(min(timeout_s, 0.05))
                return None
        try:
            frame = self._inner.read(timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001 - capture thread died: recover instead of crashing
            self._restart(f"capture failed: {exc}")
            return None
        now = self._clock()
        if frame is None:
            if self._last_frame_t is None:
                if now - self._inner_t0 > self.open_timeout_s:
                    self._restart(f"no first frame within {self.open_timeout_s:.1f} s")
            elif now - self._last_frame_t > self.stall_s:
                self._restart(f"no frame for {now - self._last_frame_t:.2f} s")
            return None
        self._last_frame_t = now
        self._backoff_cur = self.backoff_s
        self._next_attempt_t = 0.0
        frame = dataclasses.replace(frame, seq=self._seq_next)
        self._seq_next += 1
        return frame
