"""`sudo tbot csi-lanes` — live view of every MIPI wire at the Pi's CSI receiver, while you handle the cable.

A dead camera ("Camera frontend has timed out", 0 unicam interrupts) does not mean a dead link: the
receiver only raises an interrupt once the CLOCK lane has completed its low-power -> high-speed
handshake, so a single bad conductor of the clock pair silences a link whose other five wires are
fine. The unicam D-PHY exposes the instantaneous level of each lane's two wires (bits 31/30 of the
CLK / DAT0 / DAT1 registers, 1 = wire at low-power HIGH) and a clock-lane sequence-error flag (bit 29).
Sampled ~40 kHz while the sensor streams, a healthy wire is seen low for a few percent of samples
(every burst pulls both wires of a lane below the LP threshold); a wire that never goes low is not
following the sensor: open contact, short, or a dead driver/receiver on that conductor.

This tool starts a raw V4L2 stream itself (v4l2-ctl: no libcamera, so no 1 s frontend timeout stops
the sensor) and prints one line per interval. Reseat / press on the ribbon ends and watch the line:
the moment the stuck wire starts moving and frames arrive it prints LINK UP.

Needs root (/dev/mem). Stop anything holding the camera first (tbot focus / run).
The bit meanings of 31/30 are measured, not documented [INFERENCE from lane behaviour on this rig].
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import mmap
import os
import re
import struct
import subprocess
import sys
import time

from tbot.camera import linkcheck

__all__ = ["LaneState", "lane_state", "summarise", "unicam_base", "main"]

REG_STA, REG_CLK, REG_DAT0, REG_DAT1 = 0x04, 0x10, 0x18, 0x1C
BIT_A, BIT_B, BIT_ERR = 1 << 31, 1 << 30, 1 << 29
#: A wire counts as moving when it is seen low in at least this fraction of samples.
MOVING_FRACTION = 0.005
#: 15-pin Pi camera FFC pins of each lane's pair (N, P).
FFC_PINS = {"CLK": (8, 9), "D0": (2, 3), "D1": (5, 6)}


@dataclasses.dataclass(frozen=True)
class LaneState:
    name: str
    low_a: float   # fraction of samples with the bit-31 wire low
    low_b: float   # fraction of samples with the bit-30 wire low
    error: bool    # sequence-error flag seen (clock lane)

    @property
    def moving(self) -> tuple[bool, bool]:
        return self.low_a >= MOVING_FRACTION, self.low_b >= MOVING_FRACTION

    @property
    def verdict(self) -> str:
        a, b = self.moving
        if a and b:
            return "ok"
        if a or b:
            return "ONE WIRE STUCK"
        return "idle"


def lane_state(name: str, hist: dict[int, int]) -> LaneState:
    """Pure: register-value histogram of one lane -> per-wire activity."""
    n = sum(hist.values()) or 1
    low_a = sum(c for v, c in hist.items() if not v & BIT_A) / n
    low_b = sum(c for v, c in hist.items() if not v & BIT_B) / n
    return LaneState(name, low_a, low_b, any(v & BIT_ERR for v in hist))


def summarise(lanes: list[LaneState], interrupts: int) -> str:
    """Pure: one operator sentence from the lane states and the unicam interrupt count."""
    if interrupts > 0:
        return f"LINK UP - {interrupts} CSI interrupts: frames are arriving"
    stuck = [s for s in lanes if s.verdict == "ONE WIRE STUCK"]
    if stuck:
        parts = [f"{s.name} pair (FFC pins {FFC_PINS[s.name][0]}/{FFC_PINS[s.name][1]})" for s in stuck]
        return "one wire not following the sensor on: " + ", ".join(parts)
    if all(s.verdict == "idle" for s in lanes):
        return "no lane activity at all: sensor not streaming, unpowered, or the whole ribbon is out"
    if all(s.verdict == "ok" for s in lanes):
        return "all six wires move but no packet completes: corrupted data or format mismatch"
    return "partial lane activity: " + ", ".join(f"{s.name}={s.verdict}" for s in lanes)


def unicam_base(iomem: str = "/proc/iomem") -> int | None:
    try:
        with open(iomem, encoding="ascii", errors="replace") as f:
            for line in f:
                m = re.match(r"\s*([0-9a-f]+)-[0-9a-f]+ : .*csi.* unicam", line)
                if m and int(m.group(1), 16):
                    return int(m.group(1), 16)
    except OSError:
        pass
    return None


def _sample(mem: mmap.mmap, seconds: float) -> dict[str, dict[int, int]]:
    regs = {"CLK": REG_CLK, "D0": REG_DAT0, "D1": REG_DAT1}
    hist = {k: collections.Counter() for k in regs}
    unpack = struct.unpack_from
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for _ in range(200):
            for k, off in regs.items():
                hist[k][unpack("<I", mem, off)[0]] += 1
    return hist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot csi-lanes", description=__doc__.split("\n\n")[0])
    ap.add_argument("--interval", type=float, default=0.5, help="seconds per printed line (default 0.5)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (default: until Ctrl-C)")
    ap.add_argument("--device", default="/dev/video0", help="unicam video node (default /dev/video0)")
    args = ap.parse_args(argv)

    if os.geteuid() != 0:
        print("csi-lanes reads the CSI receiver registers through /dev/mem: run it with sudo", file=sys.stderr)
        return 2
    base = unicam_base()
    if base is None:
        print("no unicam CSI receiver in /proc/iomem (not a Pi 0-4, or the camera overlay is not loaded)",
              file=sys.stderr)
        return 2
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mem = mmap.mmap(fd, 4096, offset=base)
    stream = None
    t_end = time.monotonic() + args.seconds if args.seconds > 0 else None
    was_up = False
    print("wire columns: % of samples each wire is seen low (healthy while streaming: a few %, both wires alike)")
    try:
        while t_end is None or time.monotonic() < t_end:
            if stream is None or stream.poll() is not None:
                stream = subprocess.Popen(["v4l2-ctl", "-d", args.device, "--stream-mmap=4"],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.8)  # sensor register load
            irq0 = linkcheck.unicam_interrupts() or 0
            hist = _sample(mem, args.interval)
            irq = (linkcheck.unicam_interrupts() or 0) - irq0
            lanes = [lane_state(k, h) for k, h in hist.items()]
            # the clock-lane error flag is sticky: write it back to clear it for the next interval
            clk = struct.unpack_from("<I", mem, REG_CLK)[0]
            if clk & BIT_ERR:
                struct.pack_into("<I", mem, REG_CLK, clk)
            cols = "  ".join(f"{s.name} {100 * s.low_a:4.1f}/{100 * s.low_b:4.1f}%{'!' if s.error else ' '}"
                             f"{s.verdict:<14}" for s in lanes)
            print(f"{time.strftime('%H:%M:%S')}  {cols} | {summarise(lanes, irq)}", flush=True)
            up = irq > 0
            if up != was_up:
                print("\a>>> LINK UP: hold it exactly there <<<" if up else ">>> link lost <<<", flush=True)
                was_up = up
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None and stream.poll() is None:
            stream.terminate()
            try:
                stream.wait(timeout=3)
            except subprocess.TimeoutExpired:
                stream.kill()
        mem.close()
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
