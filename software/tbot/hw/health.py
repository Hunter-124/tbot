"""Raspberry Pi power and thermal health (contract: docs/ARCHITECTURE.md [hw]).

The pod runs off an XL4015 buck from a 3S Li-ion pack with an MG996R (or a PQ12) hanging off the same
5 V rail, so the firmware's under-voltage flag is the single most useful diagnostic on the whole rig: a
sagging rail shows up here long before it shows up as a missed shot. Sources, in order of preference:

* ``/sys/class/thermal/thermal_zone0/temp``                     -- SoC temperature in milli-degrees C
* ``/sys/devices/platform/soc/soc:firmware/get_throttled``      -- the firmware throttle word (no fork)
* ``vcgencmd get_throttled`` / ``vcgencmd measure_clock arm``   -- fallback when that sysfs node is absent

Nothing here ever raises and nothing here is Pi-only: off a Pi every field is simply None. Reads are
cached and rate limited to one refresh per second (the firmware mailbox is not free and `tbot hwcheck`,
the pipeline telemetry and the debug stream all ask for the same numbers).

Bit layout of the throttle word (Raspberry Pi firmware ``get_throttled``): bits 0-3 are the state *now*,
bits 16-19 the same conditions *since boot*.
"""

from __future__ import annotations

import copy
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = [
    "CPU_TEMP_PATH",
    "THROTTLED_SYSFS_PATH",
    "VCGENCMD",
    "THROTTLE_BITS",
    "parse_throttled",
    "describe_throttled",
    "read_health",
    "clear_cache",
]

#: SoC temperature, milli-degrees C.
CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"
#: Firmware throttle word without forking vcgencmd (raspberrypi-firmware driver).
THROTTLED_SYSFS_PATH = "/sys/devices/platform/soc/soc:firmware/get_throttled"
#: Fallback helper (``/usr/bin/vcgencmd``); None disables the subprocess path entirely.
VCGENCMD = "vcgencmd"
#: vcgencmd must answer fast or not at all: this is called from check tools and telemetry.
VCGENCMD_TIMEOUT_S = 1.0
#: One firmware refresh per second at most; everything in between is served from the cache.
REFRESH_S = 1.0

#: bit -> (key, human text). The same conditions are reported at bit + OCCURRED_SHIFT "since boot".
THROTTLE_BITS: dict[int, tuple[str, str]] = {
    0: ("under_voltage", "under-voltage"),
    1: ("arm_frequency_capped", "ARM frequency capped"),
    2: ("throttled", "throttled"),
    3: ("soft_temp_limit", "soft temperature limit"),
}
OCCURRED_SHIFT = 16


# --------------------------------------------------------------------------------------------- decode
def parse_throttled(value: int) -> dict:
    """Decode the firmware throttle word into flat booleans plus `now`/`occurred` key lists.

    Keys: ``raw``, ``hex``, ``<name>_now``, ``<name>_occurred`` for every name in THROTTLE_BITS,
    ``now`` / ``occurred`` (lists of the set names) and ``ok`` (nothing set at all).
    """
    raw = int(value)
    if raw < 0:
        raise ValueError(f"throttle word must be unsigned, got {raw}")
    out: dict[str, Any] = {"raw": raw, "hex": f"0x{raw:x}"}
    now: list[str] = []
    occurred: list[str] = []
    for bit, (key, _text) in THROTTLE_BITS.items():
        is_now = bool((raw >> bit) & 1)
        was = bool((raw >> (bit + OCCURRED_SHIFT)) & 1)
        out[f"{key}_now"] = is_now
        out[f"{key}_occurred"] = was
        if is_now:
            now.append(key)
        if was:
            occurred.append(key)
    out["now"] = now
    out["occurred"] = occurred
    out["ok"] = not now and not occurred
    return out


def describe_throttled(decoded: dict | None) -> str:
    """One-line human summary of a parse_throttled() result."""
    if not decoded:
        return "unknown"
    texts = {key: text for key, text in THROTTLE_BITS.values()}
    parts = []
    if decoded.get("now"):
        parts.append("NOW: " + ", ".join(texts.get(k, k) for k in decoded["now"]))
    if decoded.get("occurred"):
        parts.append("since boot: " + ", ".join(texts.get(k, k) for k in decoded["occurred"]))
    return "; ".join(parts) if parts else "no throttling"


# --------------------------------------------------------------------------------------------- readers
def _read_text(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text()
    except (OSError, ValueError):
        return None


def _read_cpu_temp_c(path: str | Path | None) -> float | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip()) / 1000.0
    except ValueError:
        return None


def _read_throttled_sysfs(path: str | Path | None) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip(), 0)
    except ValueError:
        return None


def _run_command(args: Sequence[str], timeout_s: float) -> str | None:
    """stdout of a successful command, else None (missing binary, non-zero exit, timeout)."""
    try:
        proc = subprocess.run(list(args), capture_output=True, text=True,
                              timeout=max(0.1, float(timeout_s)), check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _after_equals(text: str | None) -> str | None:
    if not text or "=" not in text:
        return None
    return text.split("=", 1)[1].strip()


def _vcgencmd_throttled(run, cmd: Sequence[str], timeout_s: float) -> int | None:
    value = _after_equals(run([*cmd, "get_throttled"], timeout_s))  # "throttled=0x0"
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def _vcgencmd_arm_mhz(run, cmd: Sequence[str], timeout_s: float) -> float | None:
    value = _after_equals(run([*cmd, "measure_clock", "arm"], timeout_s))  # "frequency(48)=1200000000"
    if value is None:
        return None
    try:
        hz = float(value)
    except ValueError:
        return None
    return hz / 1e6 if hz > 0 else None


# --------------------------------------------------------------------------------------------- cache
_LOCK = threading.Lock()
#: (key, t_s, health) of the last refresh; a different key always refreshes.
_CACHE: tuple[tuple, float, dict] | None = None


def clear_cache() -> None:
    """Forget the cached reading (tests, and after a long sleep)."""
    global _CACHE
    with _LOCK:
        _CACHE = None


def _as_command(vcgencmd: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if vcgencmd is None:
        return None
    if isinstance(vcgencmd, (str, Path)):
        return (str(vcgencmd),)
    cmd = tuple(str(a) for a in vcgencmd)
    return cmd or None


def read_health(
    now_s: float | None = None,
    *,
    temp_path: str | Path | None = CPU_TEMP_PATH,
    throttled_path: str | Path | None = THROTTLED_SYSFS_PATH,
    vcgencmd: str | Sequence[str] | None = VCGENCMD,
    timeout_s: float = VCGENCMD_TIMEOUT_S,
    refresh_s: float = REFRESH_S,
    run: Callable[[Sequence[str], float], str | None] | None = None,
) -> dict:
    """Current SoC temperature, throttle flags and ARM clock. Never raises; absent sources are None.

    Returns ``{"cpu_temp_c": float|None, "throttled": dict|None, "arm_freq_mhz": float|None,
    "source": {"cpu_temp_c": str|None, "throttled": str|None, "arm_freq_mhz": str|None},
    "available": bool, "t_s": float}``. ``now_s`` (default time.monotonic()) drives the rate limit, so
    tests can step time; the paths and the vcgencmd command are injectable for the same reason.
    """
    global _CACHE
    now = time.monotonic() if now_s is None else float(now_s)
    cmd = _as_command(vcgencmd)
    key = (str(temp_path or ""), str(throttled_path or ""), cmd)
    with _LOCK:
        cached = _CACHE
    if cached is not None and cached[0] == key and 0.0 <= now - cached[1] < max(0.0, float(refresh_s)):
        return copy.deepcopy(cached[2])

    runner = run or _run_command
    source: dict[str, str | None] = {"cpu_temp_c": None, "throttled": None, "arm_freq_mhz": None}

    cpu_temp_c = _read_cpu_temp_c(temp_path)
    if cpu_temp_c is not None:
        source["cpu_temp_c"] = "sysfs"

    word = _read_throttled_sysfs(throttled_path)
    if word is not None:
        source["throttled"] = "sysfs"
    elif cmd is not None:
        word = _vcgencmd_throttled(runner, cmd, timeout_s)
        if word is not None:
            source["throttled"] = "vcgencmd"

    throttled = None
    if word is not None:
        try:
            throttled = parse_throttled(word)
        except ValueError:
            throttled = None
            source["throttled"] = None

    arm_freq_mhz = None
    if cmd is not None:
        arm_freq_mhz = _vcgencmd_arm_mhz(runner, cmd, timeout_s)
        if arm_freq_mhz is not None:
            source["arm_freq_mhz"] = "vcgencmd"

    health = {
        "cpu_temp_c": cpu_temp_c,
        "throttled": throttled,
        "arm_freq_mhz": arm_freq_mhz,
        "source": source,
        "available": any(v is not None for v in (cpu_temp_c, throttled, arm_freq_mhz)),
        "t_s": now,
    }
    with _LOCK:
        _CACHE = (key, now, copy.deepcopy(health))
    return health
