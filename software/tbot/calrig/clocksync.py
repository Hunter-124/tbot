"""Firmware micros() -> host monotonic_ns, by a least-squares line over PING round trips.

Contract: docs/CALRIG_PROTOCOL.md section 3. Each sample is (t_send_ns, t_recv_ns, t_us) with t_us UNWRAPPED
(protocol.TimeUnwrapper). The host time of the firmware sample is taken at the round-trip midpoint; only the
lowest-RTT half of the samples is kept (a long RTT means the midpoint is a poor estimate: the delay was
asymmetric), then
    t_host_ns = offset_ns + (1 + drift_ppm * 1e-6) * t_us * 1000
is fitted by least squares. With fewer than MIN_SPAN_S of baseline the slope is fixed at 1 (drift_ppm = 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["ClockFit", "ClockSync", "MIN_SPAN_S", "KEEP_FRACTION"]

KEEP_FRACTION = 0.5
MIN_SPAN_S = 5.0  # below this baseline a drift estimate is noise
MIN_DRIFT_SAMPLES = 4


@dataclass(frozen=True)
class ClockFit:
    offset_ns: float
    drift_ppm: float
    resid_us: float   # RMS residual of the kept samples (us)
    n: int            # samples kept in the fit
    n_total: int = 0  # samples offered
    rtt_med_us: float = 0.0
    rtt_min_us: float = 0.0

    @property
    def slope(self) -> float:
        return 1.0 + self.drift_ppm * 1e-6

    def to_host_ns(self, t_us):
        """Unwrapped firmware microseconds (scalar or array) -> host monotonic ns (float; int for a scalar)."""
        out = self.offset_ns + self.slope * (np.asarray(t_us, dtype=np.float64) * 1000.0)
        return int(round(float(out))) if out.ndim == 0 else out

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (float(v) if isinstance(v, float) else v) for k, v in d.items()}


class ClockSync:
    """Accumulates PING round trips and fits them on demand."""

    def __init__(self) -> None:
        self._samples: list[tuple[int, int, int]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, t_send_ns: int, t_recv_ns: int, t_us: int) -> None:
        if t_recv_ns < t_send_ns:
            raise ValueError("t_recv_ns < t_send_ns")
        self._samples.append((int(t_send_ns), int(t_recv_ns), int(t_us)))

    def fit(self) -> ClockFit | None:
        """Least-squares fit of the kept samples; None until there is at least one sample."""
        if not self._samples:
            return None
        s = np.asarray(self._samples, dtype=np.float64)
        rtt = s[:, 1] - s[:, 0]
        n_keep = max(1, int(np.ceil(len(s) * KEEP_FRACTION)))
        keep = np.argsort(rtt, kind="stable")[:n_keep]
        x = s[keep, 2] * 1000.0                    # firmware ns
        y = 0.5 * (s[keep, 0] + s[keep, 1])        # host ns at the round-trip midpoint
        # centre both axes: raw monotonic ns are ~1e15 and float64 would lose the sub-microsecond bits
        xm, ym = x.mean(), y.mean()
        span_s = (x.max() - x.min()) / 1e9
        if len(keep) >= MIN_DRIFT_SAMPLES and span_s >= MIN_SPAN_S:
            dx = x - xm
            slope = float(np.dot(dx, y - ym) / np.dot(dx, dx))
        else:
            slope = 1.0
        offset = ym - slope * xm
        resid = y - (offset + slope * x)
        return ClockFit(
            offset_ns=float(offset),
            drift_ppm=(slope - 1.0) * 1e6,
            resid_us=float(np.sqrt(np.mean(resid ** 2)) / 1000.0),
            n=int(len(keep)),
            n_total=int(len(s)),
            rtt_med_us=float(np.median(rtt) / 1000.0),
            rtt_min_us=float(rtt.min() / 1000.0),
        )
