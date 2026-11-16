"""Recoil detection: did the gun fire after a trigger command, and how long did it take?

The IMU sits on the gun, and a 12 ga shot drives the accelerometer far past anything a swing produces (it
clips at the configured range). The first sample of that burst after a fire command marks ignition, so

    recoil delay = fire command -> ignition = actuator cmd_to_break + gun lock time

which is the part of the latency budget that could only be estimated on the bench (video of the lever,
+/-10 ms). Every live shot therefore measures it for free; `tbot latency-recoil` turns the logged events
into calib/latency.json. A command with no recoil after it means the trigger did not break (lever slipped,
safety on, empty chamber).

A dry-fire click or the servo hitting the trigger is a single short spike; a shot holds the accelerometer
high for many milliseconds. `confirm_samples` over-threshold samples inside `confirm_ms` tell them apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G_MPS2 = 9.80665

__all__ = ["RecoilDetector", "RecoilEvent", "summarise"]


@dataclass
class RecoilEvent:
    shot: int
    cmd_t_ns: int
    detected: bool
    t_ns: int | None = None        # first over-threshold sample (ignition)
    delay_ms: float | None = None  # t_ns - cmd_t_ns
    peak_g: float | None = None

    def as_event(self) -> dict:
        return {
            "type": "recoil" if self.detected else "no_recoil",
            "t_ns": int(self.t_ns if self.t_ns is not None else self.cmd_t_ns),
            "cmd_t_ns": int(self.cmd_t_ns),
            "shot": int(self.shot),
            "delay_ms": self.delay_ms,
            "peak_g": self.peak_g,
        }


class RecoilDetector:
    """Feed it every IMU sample; tell it about each fire command; it returns one RecoilEvent per command."""

    def __init__(self, threshold_g: float = 3.0, min_delay_ms: float = 15.0, window_ms: float = 500.0,
                 confirm_samples: int = 3, confirm_ms: float = 20.0):
        self.threshold = float(threshold_g) * G_MPS2
        self.min_delay_ns = int(float(min_delay_ms) * 1e6)
        self.window_ns = int(float(window_ms) * 1e6)
        self.confirm_samples = max(1, int(confirm_samples))
        self.confirm_ns = int(float(confirm_ms) * 1e6)
        self._pending: list[dict] = []
        self.events: list[RecoilEvent] = []

    @classmethod
    def from_config(cls, cfg) -> "RecoilDetector":
        f = cfg.fire
        return cls(threshold_g=float(getattr(f, "recoil_detect_g", 3.0)),
                   window_ms=float(getattr(f, "recoil_window_ms", 500.0)))

    @property
    def waiting(self) -> bool:
        return bool(self._pending)

    def expect(self, cmd_t_ns: int, shot: int) -> None:
        """A fire command was issued at cmd_t_ns (same clock as the IMU samples)."""
        self._pending.append({"cmd": int(cmd_t_ns), "shot": int(shot), "onset": None, "count": 0, "peak": 0.0})

    def feed(self, samples) -> list[RecoilEvent]:
        """Returns the events that completed with these samples (usually none)."""
        if not self._pending or not samples:
            return []
        done: list[RecoilEvent] = []
        for s in samples:
            if not self._pending:
                break
            t = int(s.t_ns)
            a = s.accel
            ax, ay, az = float(a[0]), float(a[1]), float(a[2])
            excess = abs(math.sqrt(ax * ax + ay * ay + az * az) - G_MPS2)
            p = self._pending[0]
            since = t - p["cmd"]
            if since < self.min_delay_ns:
                continue
            if p["onset"] is not None and t - p["onset"] > self.confirm_ns:
                if p["count"] >= self.confirm_samples:
                    done.append(self._finish(p, True))
                    continue
                p["onset"], p["count"], p["peak"] = None, 0, 0.0  # a click, not a shot: keep looking
            if since > self.window_ns:
                done.append(self._finish(p, False))
                continue
            if excess >= self.threshold:
                if p["onset"] is None:
                    p["onset"] = t
                p["count"] += 1
                p["peak"] = max(p["peak"], excess)
        self.events.extend(done)
        return done

    def _finish(self, p: dict, detected: bool) -> RecoilEvent:
        self._pending.remove(p)
        if not detected:
            return RecoilEvent(p["shot"], p["cmd"], False)
        return RecoilEvent(p["shot"], p["cmd"], True, int(p["onset"]), (p["onset"] - p["cmd"]) / 1e6,
                           p["peak"] / G_MPS2 + 1.0)


def summarise(delays_ms: list[float]) -> dict | None:
    """Robust statistics of recoil delays (median, MAD-based sigma): one flyer must not move the budget."""
    vals = sorted(float(d) for d in delays_ms if d is not None and math.isfinite(float(d)))
    if not vals:
        return None
    n = len(vals)

    def q(f: float) -> float:
        x = f * (n - 1)
        i = int(x)
        return vals[i] if i + 1 >= n else vals[i] + (x - i) * (vals[i + 1] - vals[i])

    med = q(0.5)
    mad = sorted(abs(v - med) for v in vals)[n // 2] if n > 1 else 0.0
    return {"n": n, "median": med, "mean": sum(vals) / n, "std": 1.4826 * mad, "p05": q(0.05), "p95": q(0.95),
            "min": vals[0], "max": vals[-1]}
