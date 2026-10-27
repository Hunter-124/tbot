"""Summary statistics shared by the latency/bench tools (format used in calib/latency.json)."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def summarize(values: Iterable[float]) -> dict:
    """{"mean","std","p50","p95","min","max","n"} of finite values; NaN fields when empty."""
    a = np.asarray([float(v) for v in values], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "p50": nan, "p95": nan, "min": nan, "max": nan, "n": 0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }
