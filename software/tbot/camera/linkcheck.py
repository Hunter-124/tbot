"""Where does a dead CSI camera fail? Sensor, CSI link or frame format — measured, not guessed.

When libcamera reports "Camera frontend has timed out", three layers can be at fault, and only one of
them is fixable by restarting software:

  sensor  - the OV9281 is not streaming (register 0x0100 != 1) although a stream was requested:
            driver / power state; a camera restart or reboot can fix it.
  link    - the sensor IS streaming but the Pi's CSI receiver (unicam) sees no packets at all
            (its interrupt counter does not move). That does NOT mean every lane is dead: one bad
            conductor of the clock pair is enough, because no packet is received until the clock
            lane completes its LP->HS handshake. `sudo tbot csi-lanes` shows each wire live.
            Restarts cannot fix it.
  frames  - unicam interrupts arrive but no frame completes: packets are corrupted on the way
            (marginal contact, interference) or the format does not match.

`diagnose()` must be called WHILE a stream is requested (the source is started), because the sensor
only streams then. It needs no root for the interrupt counter; the sensor register read uses
`sudo -n i2ctransfer` and is skipped if that is not available.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import subprocess
import time
from typing import Callable

__all__ = ["LinkDiagnosis", "diagnose", "unicam_interrupts", "sensor_streaming", "verdict"]

_PROC_INTERRUPTS = "/proc/interrupts"
_SENSOR_GLOB = "/sys/bus/i2c/devices/*-0060"  # OV9281 at 0x60
MODE_SELECT_REG = 0x0100  # 1 = streaming


@dataclasses.dataclass(frozen=True)
class LinkDiagnosis:
    layer: str              # "ok" | "sensor" | "link" | "frames" | "unknown"
    sensor_streaming: bool | None
    csi_interrupts: int | None   # unicam interrupts counted during the sample window
    window_s: float
    message: str            # one sentence for the operator

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def unicam_interrupts(path: str = _PROC_INTERRUPTS) -> int | None:
    """Total unicam interrupts across CPUs, or None when there is no unicam line (not a Pi)."""
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    total = None
    for line in lines:
        if "unicam" not in line:
            continue
        fields = line.split()
        count = 0
        for tok in fields[1:]:
            if not tok.isdigit():
                break
            count += int(tok)
        total = (total or 0) + count
    return total


def _sensor_bus_addr() -> tuple[int, int] | None:
    for dev in sorted(glob.glob(_SENSOR_GLOB)):
        try:
            with open(os.path.join(dev, "name"), encoding="ascii") as f:
                if not any(k in f.read() for k in ("9281", "9282")):
                    continue
        except OSError:
            continue
        bus, _, addr = os.path.basename(dev).partition("-")
        try:
            return int(bus), int(addr, 16)
        except ValueError:
            continue
    return None


def sensor_streaming(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> bool | None:
    """Reads OV9281 register 0x0100 (mode_select). None when it cannot be read (no sudo, no sensor)."""
    loc = _sensor_bus_addr()
    if loc is None:
        return None
    bus, addr = loc
    cmd = ["sudo", "-n", "i2ctransfer", "-f", "-y", str(bus),
           f"w2@0x{addr:02x}", f"0x{MODE_SELECT_REG >> 8:02x}", f"0x{MODE_SELECT_REG & 0xFF:02x}", "r1"]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=3.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().split()[0], 16) & 0x01 == 1
    except (ValueError, IndexError):
        return None


def verdict(streaming: bool | None, interrupts: int | None, frames: int, window_s: float) -> LinkDiagnosis:
    """Pure decision from the measurements (unit-tested)."""
    if frames > 0:
        return LinkDiagnosis("ok", streaming, interrupts, window_s, f"{frames} frame(s) arrived: the camera works")
    if interrupts is None:
        return LinkDiagnosis("unknown", streaming, None, window_s,
                             "no frames, and no unicam CSI receiver found to inspect (not a Pi?)")
    if streaming is False:
        return LinkDiagnosis("sensor", False, interrupts, window_s,
                             "the sensor is NOT streaming although a stream was requested: driver/power state - "
                             "a camera restart or reboot should bring it back")
    if interrupts == 0:
        head = ("the sensor reports it is streaming, but" if streaming else
                "the sensor state could not be read (no sudo i2ctransfer), and")
        return LinkDiagnosis("link", streaming, 0, window_s,
                             f"{head} the Pi's CSI receiver got 0 packets in {window_s:.1f} s: at least one MIPI "
                             "conductor is not getting through (one bad wire of the clock pair is enough). Restarts "
                             "and reboots cannot fix this; stop this tool and run `sudo tbot csi-lanes` to see which "
                             "wire, live, while you handle the ribbon")
    return LinkDiagnosis("frames", streaming, interrupts, window_s,
                         f"the CSI receiver saw {interrupts} interrupts in {window_s:.1f} s but no frame completed: "
                         "packets are arriving corrupted (marginal contact or interference) or the format does not match")


def down_card(width: int, height: int, title: str, message: str):
    """A BGR placeholder frame for the debug stream while the camera is down (so the page says why)."""
    import textwrap

    import cv2
    import numpy as np

    img = np.full((int(height), int(width), 3), 30, dtype=np.uint8)
    cv2.putText(img, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    y = 64
    for line in textwrap.wrap(message, width=max(20, int(width) // 9)):
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        y += 20
    return img


def diagnose(window_s: float = 1.5, frames_seen: Callable[[], int] | None = None,
             sleep: Callable[[float], None] = time.sleep, source=None) -> LinkDiagnosis:
    """The link verdict for a camera that delivers no frames.

    Prefers `source.start_probe` (PicamSource measures the link in the first 0.7 s of each start, before
    libcamera's ~1 s frontend timeout stops the sensor). Otherwise samples the CSI interrupt counter over
    `window_s` and reads the sensor state now, which is only meaningful within ~1 s of a stream start.
    `frames_seen()` (optional) returns how many frames the caller received during the window."""
    if source is not None and hasattr(source, "start_probe"):
        deadline = time.monotonic() + 1.5  # a probe may be in progress on a just-restarted camera
        while True:
            probe = getattr(source, "start_probe", None)
            if isinstance(probe, LinkDiagnosis):
                return probe
            if time.monotonic() >= deadline:
                break
            sleep(0.05)
    before = unicam_interrupts()
    streaming = sensor_streaming()
    sleep(window_s)
    after = unicam_interrupts()
    delta = None if before is None or after is None else max(0, after - before)
    frames = int(frames_seen()) if frames_seen is not None else 0
    return verdict(streaming, delta, frames, window_s)
