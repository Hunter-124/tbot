"""Scene program and frame log shared by the rig and the target page (protocol: docs/HITL.md).

Times on the wire are `rig_ms`: float milliseconds since `HitlStore.epoch_ns` on the rig's monotonic clock.
Screen coordinates are the page canvas's device pixels (origin top-left, +y down). An item is drawn on a
display frame whose rig time t satisfies t0 <= t < t1; a disc's centre at dt = t - t0 (ms) is
x0 + vx*dt + 0.5*ax*dt^2 (same for y). The page logs one entry per requestAnimationFrame callback.
"""

from __future__ import annotations

import bisect
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Disc:
    id: int
    t0: float  # rig_ms
    t1: float | None  # rig_ms, None = until removed
    x0: float
    y0: float
    r: float
    level: int
    vx: float = 0.0  # px/ms
    vy: float = 0.0
    ax: float = 0.0  # px/ms^2
    ay: float = 0.0
    kind: str = "disc"

    def pos(self, t_ms: float) -> tuple[float, float]:
        dt = float(t_ms) - self.t0
        return (self.x0 + self.vx * dt + 0.5 * self.ax * dt * dt,
                self.y0 + self.vy * dt + 0.5 * self.ay * dt * dt)


@dataclass
class Rect:
    id: int
    t0: float
    t1: float | None
    x: float
    y: float
    w: float
    h: float
    level: int
    kind: str = "rect"


Item = Disc | Rect


def active(item: Item, t_ms: float) -> bool:
    return item.t0 <= t_ms and (item.t1 is None or t_ms < item.t1)


class HitlStore:
    """Thread-safe scene program (rig -> page) and frame log (page -> rig).

    `post_log` is called by the HTTP server (or the simulated display) with the page's batches; the
    procedures read the log to find when an item was first drawn.
    """

    def __init__(self, clock_ns=time.monotonic_ns, bg: int = 170):
        self._clock_ns = clock_ns
        self.epoch_ns = int(clock_ns())
        self._lock = threading.Lock()
        self._version = 0
        self._bg = int(bg)
        self._status = True
        self._message = "TBOT HITL: waiting for the rig"
        self._items: dict[int, Item] = {}
        self._next_id = 1
        self._frame_t: list[float] = []  # rig_ms of every logged display frame, ascending
        self._frame_ids: list[tuple[int, ...]] = []
        self._first_drawn: dict[int, float] = {}
        self.page_status: dict = {}
        self.last_post_ns: int | None = None
        self.posts = 0
        self.state: dict = {}

    # ---------------------------------------------------------------- clock
    def now_ns(self) -> int:
        return int(self._clock_ns())

    def rig_ms(self, t_ns: int | None = None) -> float:
        t = self.now_ns() if t_ns is None else int(t_ns)
        return (t - self.epoch_ns) / 1e6

    def ns(self, rig_ms: float) -> int:
        return self.epoch_ns + int(round(float(rig_ms) * 1e6))

    # ---------------------------------------------------------------- program (rig side)
    def _bump(self) -> None:
        self._version += 1

    def new_id(self) -> int:
        with self._lock:
            i = self._next_id
            self._next_id += 1
            return i

    def add(self, item: Item) -> int:
        with self._lock:
            self._items[item.id] = item
            self._bump()
        return item.id

    def end(self, item_id: int, t1_ms: float | None = None) -> None:
        """Stop drawing an item from t1 (default: now)."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return
            t1 = self.rig_ms() if t1_ms is None else float(t1_ms)
            item.t1 = t1 if item.t1 is None else min(item.t1, t1)
            self._bump()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bump()

    def prune(self, before_ms: float) -> None:
        """Forget items that ended before `before_ms` (keeps the program small)."""
        with self._lock:
            dead = [i for i, it in self._items.items() if it.t1 is not None and it.t1 < before_ms]
            for i in dead:
                del self._items[i]
            if dead:
                self._bump()

    def set_background(self, level: int) -> None:
        with self._lock:
            self._bg = int(level)
            self._bump()

    @property
    def bg(self) -> int:
        return self._bg

    def set_status(self, show: bool, message: str | None = None) -> None:
        with self._lock:
            self._status = bool(show)
            if message is not None:
                self._message = str(message)
            self._bump()

    def items(self) -> list[Item]:
        with self._lock:
            return list(self._items.values())

    def program(self) -> dict:
        with self._lock:
            return {
                "v": self._version,
                "bg": self._bg,
                "status": self._status,
                "message": self._message,
                "items": [asdict(it) for it in self._items.values()],
            }

    # ---------------------------------------------------------------- frame log (page side)
    def post_log(self, payload: dict, received_ns: int) -> None:
        frames = payload.get("frames") or []
        with self._lock:
            for fr in frames:
                try:
                    t = float(fr["t"])
                    ids = tuple(int(i) for i in (fr.get("ids") or ()))
                except (KeyError, TypeError, ValueError):
                    continue
                if self._frame_t and t <= self._frame_t[-1]:
                    continue  # duplicate/out-of-order batch (page retry): the log stays ascending
                self._frame_t.append(t)
                self._frame_ids.append(ids)
                for i in ids:
                    self._first_drawn.setdefault(i, t)
            status = payload.get("status")
            if isinstance(status, dict):
                self.page_status = dict(status)
            self.last_post_ns = int(received_ns)
            self.posts += 1

    def rig_state(self) -> dict:
        with self._lock:
            return dict(self.state, frames_logged=len(self._frame_t), posts=self.posts)

    def first_drawn(self, item_id: int) -> float | None:
        """rig_ms of the first display frame that drew the item (None until the page logged it)."""
        with self._lock:
            return self._first_drawn.get(int(item_id))

    def logged_until(self) -> float | None:
        with self._lock:
            return self._frame_t[-1] if self._frame_t else None

    def frames(self, t0_ms: float | None = None, t1_ms: float | None = None) -> tuple[np.ndarray, list[tuple]]:
        """Logged display frames with t0 <= t < t1: (times rig_ms, ids per frame)."""
        with self._lock:
            lo = 0 if t0_ms is None else bisect.bisect_left(self._frame_t, float(t0_ms))
            hi = len(self._frame_t) if t1_ms is None else bisect.bisect_left(self._frame_t, float(t1_ms))
            return np.asarray(self._frame_t[lo:hi], dtype=np.float64), list(self._frame_ids[lo:hi])

    def connected(self, max_age_s: float = 1.0) -> bool:
        """The page posted recently and reports a synced clock."""
        last = self.last_post_ns
        if last is None:
            return False
        return (self.now_ns() - last) <= max_age_s * 1e9 and bool(self.page_status.get("synced"))

    def screen_size(self) -> tuple[int, int] | None:
        s = self.page_status
        try:
            w, h = int(s["w"]), int(s["h"])
        except (KeyError, TypeError, ValueError):
            return None
        return (w, h) if w > 0 and h > 0 else None
