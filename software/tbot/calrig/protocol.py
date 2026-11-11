"""CalRig serial protocol: line parsing, command formatting, unit conversion, t_us unwrapping. Pure, no I/O.

Contract: docs/CALRIG_PROTOCOL.md section 2. Positions are int32 microsteps, velocities microsteps/s,
accelerations microsteps/s^2, `t_us` the firmware micros() (uint32, wraps every 71.6 min).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "BAUD",
    "UINT32",
    "Geometry",
    "Reply",
    "Pong",
    "Telemetry",
    "HallEdge",
    "Event",
    "Unknown",
    "ProtocolError",
    "parse_line",
    "TimeUnwrapper",
    "cmd_ping",
    "cmd_info",
    "cmd_en",
    "cmd_estop",
    "cmd_reset",
    "cmd_cfg",
    "cmd_lim",
    "cmd_zero",
    "cmd_move",
    "cmd_vel",
    "cmd_sine",
    "cmd_home",
    "cmd_stop",
    "cmd_stream",
    "cmd_hb",
    "cmd_stat",
    "command_name",
    "sine_peak_accel_dps2",
    "MOTION_COMMANDS",
    "STATES",
    "TERMINAL_EVENTS",
    "FAULT_EVENTS",
]

BAUD = 115200
UINT32 = 1 << 32
STATES = ("IDLE", "MOVE", "VEL", "SINE", "HOME", "ESTOP")
MOTION_COMMANDS = ("MOVE", "VEL", "SINE", "HOME")
TERMINAL_EVENTS = ("DONE", "LIMIT", "HOMED", "ESTOP", "WATCHDOG")  # motion ended (or was ended by the firmware)
FAULT_EVENTS = ("ESTOP", "WATCHDOG")


class ProtocolError(ValueError):
    """A line that is recognisably protocol but malformed."""


# ------------------------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class Geometry:
    """Table degrees <-> microsteps (docs/CALRIG_PROTOCOL.md section 1: 200 * MICROSTEPS / 360 per degree)."""

    microsteps: int = 16
    steps_rev: int = 200
    ratio: float = 1.0

    @property
    def usteps_per_deg(self) -> float:
        return self.steps_rev * self.microsteps * self.ratio / 360.0

    @property
    def deg_per_ustep(self) -> float:
        return 1.0 / self.usteps_per_deg

    def to_usteps(self, deg: float) -> int:
        return int(round(deg * self.usteps_per_deg))

    def to_deg(self, usteps: float) -> float:
        return usteps / self.usteps_per_deg


# ------------------------------------------------------------------------------------------ line types
@dataclass(frozen=True)
class Reply:
    """`OK <CMD> [k=v ...]` or `ERR <CMD> <code> <text>`."""

    ok: bool
    cmd: str
    fields: dict = field(default_factory=dict)  # OK: key=value pairs (strings), bare tokens map to ""
    code: str = ""                              # ERR: machine-readable code
    text: str = ""                              # ERR: free text
    raw: str = ""

    def int_field(self, key: str) -> int:
        return int(self.fields[key])


@dataclass(frozen=True)
class Pong:
    seq: int
    t_us: int


@dataclass(frozen=True)
class Telemetry:
    t_us: int
    pos: int
    vel: int


@dataclass(frozen=True)
class HallEdge:
    t_us: int
    pos: int
    level: int


@dataclass(frozen=True)
class Event:
    t_us: int
    code: str
    arg: int | None = None


@dataclass(frozen=True)
class Unknown:
    raw: str


def _int(token: str, what: str, line: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ProtocolError(f"bad {what} {token!r} in {line!r}") from None


def parse_line(line: str) -> Reply | Pong | Telemetry | HallEdge | Event | Unknown:
    """Classify and parse one firmware line (trailing CR/LF tolerated).

    Lines that are not protocol (boot banners, garbage after a reset) come back as Unknown; a recognised
    line with bad fields raises ProtocolError (the reader counts it and carries on).
    """
    raw = line.strip()
    if not raw:
        return Unknown(raw)
    parts = raw.split()
    head = parts[0]
    if head == "T" and len(parts) == 4:
        return Telemetry(_int(parts[1], "t_us", raw), _int(parts[2], "pos", raw), _int(parts[3], "vel", raw))
    if head == "H" and len(parts) == 4:
        return HallEdge(_int(parts[1], "t_us", raw), _int(parts[2], "pos", raw), _int(parts[3], "level", raw))
    if head == "E" and len(parts) >= 3:
        arg = _int(parts[3], "arg", raw) if len(parts) >= 4 else None
        return Event(_int(parts[1], "t_us", raw), parts[2], arg)
    if head == "PONG" and len(parts) == 3:
        return Pong(_int(parts[1], "seq", raw), _int(parts[2], "t_us", raw))
    if head == "OK" and len(parts) >= 2:
        fields: dict[str, str] = {}
        for tok in parts[2:]:
            k, sep, v = tok.partition("=")
            fields[k] = v if sep else ""
        return Reply(True, parts[1], fields, raw=raw)
    if head == "ERR" and len(parts) >= 3:
        return Reply(False, parts[1], code=parts[2], text=" ".join(parts[3:]), raw=raw)
    return Unknown(raw)


# ------------------------------------------------------------------------------------------ t_us wrap
class TimeUnwrapper:
    """uint32 microsecond counter -> monotonically extended integer microseconds.

    The first value seen is the origin. Each later value is placed at the 2^32-congruent point NEAREST to the
    previous one, so a stream that interleaves lines sampled slightly out of order (PONG vs T) neither
    jumps a whole wrap backwards nor forwards.
    """

    def __init__(self) -> None:
        self._last: int | None = None

    def unwrap(self, t_us: int) -> int:
        t = int(t_us) % UINT32
        if self._last is None:
            self._last = t
            return t
        diff = (t - self._last + UINT32 // 2) % UINT32 - UINT32 // 2
        self._last += diff
        return self._last


# ------------------------------------------------------------------------------------------ commands
def cmd_ping(seq: int) -> str:
    return f"PING {int(seq)}"


def cmd_info() -> str:
    return "INFO"


def cmd_en(on: bool) -> str:
    return f"EN {1 if on else 0}"


def cmd_estop() -> str:
    return "ESTOP"


def cmd_reset() -> str:
    return "RESET"


def cmd_cfg(vmax_sps: float, amax_sps2: float) -> str:
    return f"CFG {int(round(vmax_sps))} {int(round(amax_sps2))}"


def cmd_lim(lo: int, hi: int) -> str:
    return f"LIM {int(lo)} {int(hi)}"


def cmd_zero() -> str:
    return "ZERO"


def cmd_move(target: int) -> str:
    return f"MOVE {int(target)}"


def cmd_vel(v_sps: float) -> str:
    return f"VEL {int(round(v_sps))}"


def cmd_sine(amp: int, period_ms: int, cycles: int) -> str:
    if cycles < 1 or period_ms < 1:
        raise ProtocolError("SINE needs cycles >= 1 and period_ms >= 1")
    return f"SINE {int(amp)} {int(period_ms)} {int(cycles)}"


def cmd_home(direction: int, v_sps: float) -> str:
    if direction not in (1, -1):
        raise ProtocolError("HOME direction must be +1 or -1")
    return f"HOME {direction} {int(round(v_sps))}"


def cmd_stop() -> str:
    return "STOP"


def cmd_stream(on: bool) -> str:
    return f"STREAM {1 if on else 0}"


def cmd_hb() -> str:
    return "HB"


def cmd_stat() -> str:
    return "STAT"


def command_name(line: str) -> str:
    """The upper-case command word of a command line (the reply echoes it, PING excepted: PONG)."""
    return line.split(None, 1)[0].upper() if line.strip() else ""


def sine_peak_accel_dps2(amp_deg: float, period_s: float) -> float:
    """Peak acceleration of a steady sinusoid A sin(2 pi t / T): the firmware's SINE acceptance test."""
    return abs(amp_deg) * (2.0 * math.pi / period_s) ** 2
