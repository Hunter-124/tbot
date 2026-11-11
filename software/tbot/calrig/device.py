"""Serial driver for the CalRig firmware: command/reply matching, async-line queues, clock-sync pings.

Contract: docs/CALRIG_PROTOCOL.md. One reader pulls lines from a Transport and sorts them:
  * `OK ...` / `ERR ...` / `PONG ...` complete the single in-flight command (commands are serialised);
  * `T`, `H`, `E` lines are timestamped (host receive time + unwrapped firmware t_us) and queued.
With `threaded=True` (the real serial link) a reader thread does this; with `threaded=False` (FakeRig on a
VirtualClock) the owner calls poll() and every wait loop pumps it, so simulated time stays deterministic.
Pyserial is imported lazily; nothing here imports firing or actuator code.
"""

from __future__ import annotations

import collections
import itertools
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from tbot.calrig import protocol as P
from tbot.calrig.clocksync import ClockSync
from tbot.calrig.timebase import RealClock, Timebase

__all__ = [
    "Transport",
    "SerialTransport",
    "RigDevice",
    "Stat",
    "DeviceError",
    "CommandError",
    "CommandTimeout",
    "RigFault",
]

MAX_QUEUE = 400_000  # ~30 min of 200 Hz telemetry; the recorder drains far more often


class DeviceError(RuntimeError):
    pass


class CommandError(DeviceError):
    """The firmware answered `ERR <CMD> <code> <text>`."""

    def __init__(self, command: str, reply: P.Reply):
        super().__init__(f"{command!r} refused: ERR {reply.cmd} {reply.code} {reply.text}".rstrip())
        self.command = command
        self.reply = reply
        self.code = reply.code


class CommandTimeout(DeviceError):
    pass


class RigFault(DeviceError):
    """E-stop, watchdog, lost telemetry or a dead link: the run cannot continue."""


class Transport(Protocol):
    def write_line(self, line: str) -> None: ...

    def read_line(self, timeout: float) -> tuple[str, int] | None:
        """Next complete line and its host receive time (ns), or None after `timeout` seconds."""
        ...

    def close(self) -> None: ...


class SerialTransport:
    """pyserial link. Lines are stamped with the host clock when the bytes were read."""

    def __init__(self, port: str, baud: int = P.BAUD, clock: Timebase | None = None):
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DeviceError("pyserial is not installed: `pip install pyserial` "
                              "(Pi: `sudo apt install python3-serial`)") from exc
        self._serial_mod = serial
        self.clock = clock or RealClock()
        try:
            # NOTE: opening the port toggles DTR, which resets an Arduino Uno (position and zero are lost).
            self._ser = serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)
        except (OSError, serial.SerialException) as exc:
            raise DeviceError(f"cannot open {port}: {exc}") from exc
        self._buf = bytearray()
        self._lines: collections.deque[tuple[str, int]] = collections.deque()

    def write_line(self, line: str) -> None:
        try:
            self._ser.write(line.encode("ascii") + b"\n")
            self._ser.flush()
        except (OSError, self._serial_mod.SerialException) as exc:
            raise DeviceError(f"serial write failed: {exc}") from exc

    def read_line(self, timeout: float) -> tuple[str, int] | None:
        if not self._lines:
            try:
                self._ser.timeout = max(0.0, timeout)
                chunk = self._ser.read(max(1, self._ser.in_waiting))
            except (OSError, self._serial_mod.SerialException) as exc:
                raise DeviceError(f"serial read failed: {exc}") from exc
            if chunk:
                stamp = self.clock.now_ns()
                self._buf += chunk
                while True:
                    nl = self._buf.find(b"\n")
                    if nl < 0:
                        break
                    raw = bytes(self._buf[:nl])
                    del self._buf[:nl + 1]
                    self._lines.append((raw.decode("ascii", errors="replace").strip("\r"), stamp))
                if len(self._buf) > 4096:  # runaway garbage without a newline
                    self._buf.clear()
        return self._lines.popleft() if self._lines else None

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001 - closing a dead port must not raise
            pass


@dataclass(frozen=True)
class Stat:
    t_us: int
    pos: int
    vel: int
    state: str
    en: int
    home: int
    estop_pin: int


@dataclass
class _Pending:
    cmd: str
    seq: int | None = None
    done: bool = False
    reply: Any = None
    t_rx_ns: int = 0


class RigDevice:
    def __init__(self, transport: Transport, geometry: P.Geometry, clock: Timebase | None = None,
                 threaded: bool = True, command_timeout_s: float = 1.0):
        self.transport = transport
        self.geometry = geometry
        self.clock = clock or RealClock()
        self.threaded = threaded
        self.command_timeout_s = command_timeout_s

        self.unwrapper = P.TimeUnwrapper()
        self.sync = ClockSync()
        self.info_fields: dict[str, str] = {}
        self.fault: str | None = None        # first fault seen: dead link, e-stop or watchdog event
        self.link_error: str | None = None   # the serial link itself is dead: no command can complete
        self.booted = False
        self.bad_lines = 0
        self.unknown_lines: collections.deque[str] = collections.deque(maxlen=200)
        self.last_rx_ns = 0
        self.last_telem_rx_ns = 0
        self.last_tx_ns = 0

        # async-line queues: (t_rx_ns, t_us_unwrapped, message), drained by the recorder
        self.telemetry_q: collections.deque = collections.deque(maxlen=MAX_QUEUE)
        self.hall_q: collections.deque = collections.deque(maxlen=MAX_QUEUE)
        self.event_q: collections.deque = collections.deque(maxlen=MAX_QUEUE)
        self.event_log: list[tuple[int, int, P.Event]] = []   # never drained: wait/mark helpers read it
        self.cmd_q: collections.deque = collections.deque()   # (t_ns, command line) for cmd.csv

        self._cond = threading.Condition()
        self._cmd_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: _Pending | None = None
        self._ping_seq = itertools.count(1)
        self._stop_evt = threading.Event()
        self._reader: threading.Thread | None = None
        self._hb_thread: threading.Thread | None = None
        self._hb_period_s = 0.0
        self._last_touch_ns = 0

    # ---------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.threaded and self._reader is None:
            self._reader = threading.Thread(target=self._reader_loop, name="calrig-reader", daemon=True)
            self._reader.start()

    def close(self) -> None:
        self._stop_evt.set()
        for t in (self._hb_thread, self._reader):
            if t is not None and t is not threading.current_thread():
                t.join(timeout=1.0)
        self._hb_thread = self._reader = None
        self.transport.close()

    def connect(self, timeout_s: float = 8.0) -> None:
        """Start the reader and PING until the firmware answers (an Uno needs ~2 s to boot after the port opens)."""
        self.start()
        end = self.clock.now_ns() + int(timeout_s * 1e9)
        last_err = ""
        while True:
            try:
                if self.ping(timeout_s=0.4) is not None:
                    return
            except DeviceError as exc:
                last_err = str(exc)
            if self.clock.now_ns() >= end:
                raise DeviceError("no answer from the rig firmware" + (f" ({last_err})" if last_err else "")
                                  + ": check the USB cable, the port and that calrig_uno is flashed")
            self.clock.sleep(0.1)

    # ---------------------------------------------------------------------------- reader
    def _reader_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._read_once(0.05)
            except DeviceError as exc:
                self._set_fault(str(exc), link=True)
                return
            except Exception as exc:  # noqa: BLE001 - a dying reader must latch a fault, never vanish silently
                self._set_fault(f"reader thread crashed: {exc!r}", link=True)
                return

    def poll(self, max_lines: int = 100_000) -> int:
        """Non-threaded mode: dispatch every line the transport has ready. No-op when the reader thread runs."""
        if self.threaded:
            return 0
        n = 0
        while n < max_lines and self._read_once(0.0):
            n += 1
        return n

    def _read_once(self, timeout: float) -> bool:
        got = self.transport.read_line(timeout)
        if got is None:
            return False
        self._dispatch(got[0], got[1])
        return True

    def _set_fault(self, text: str, link: bool = False) -> None:
        with self._cond:
            if self.fault is None:
                self.fault = text
            if link and self.link_error is None:
                self.link_error = text
            self._cond.notify_all()

    def _dispatch(self, line: str, t_rx_ns: int) -> None:
        try:
            msg = P.parse_line(line)
        except P.ProtocolError:
            self.bad_lines += 1
            return
        with self._cond:
            self.last_rx_ns = t_rx_ns
            if isinstance(msg, P.Telemetry):
                self.last_telem_rx_ns = t_rx_ns
                self.telemetry_q.append((t_rx_ns, self.unwrapper.unwrap(msg.t_us), msg))
            elif isinstance(msg, P.HallEdge):
                self.hall_q.append((t_rx_ns, self.unwrapper.unwrap(msg.t_us), msg))
            elif isinstance(msg, P.Event):
                item = (t_rx_ns, self.unwrapper.unwrap(msg.t_us), msg)
                self.event_q.append(item)
                self.event_log.append(item)
                if msg.code == "BOOT":
                    self.booted = True
                elif msg.code in P.FAULT_EVENTS and self.fault is None:
                    self.fault = f"firmware event {msg.code}"
            elif isinstance(msg, P.Pong):
                u = self.unwrapper.unwrap(msg.t_us)
                p = self._pending
                if p is not None and p.cmd == "PING" and p.seq == msg.seq and not p.done:
                    p.reply, p.t_rx_ns, p.done = (msg, u), t_rx_ns, True
            elif isinstance(msg, P.Reply):
                p = self._pending
                if p is not None and p.cmd == msg.cmd and not p.done:
                    p.reply, p.t_rx_ns, p.done = msg, t_rx_ns, True
                # else: reply to a fire-and-forget line (HB), or stale: dropped
            else:
                self.unknown_lines.append(msg.raw)
            self._cond.notify_all()

    # ---------------------------------------------------------------------------- waiting
    def wait_until(self, predicate, timeout_s: float) -> bool:
        """Block until predicate() is true (True) or timeout_s elapses (False). Pumps the link if unthreaded."""
        if self.threaded:
            with self._cond:
                return bool(self._cond.wait_for(lambda: predicate() or self.link_error is not None, timeout_s)
                            and predicate())
        end = self.clock.now_ns() + int(timeout_s * 1e9)
        while True:
            self.poll()
            if predicate():
                return True
            if self.clock.now_ns() >= end:
                return False
            self.clock.sleep(0.001)

    # ---------------------------------------------------------------------------- tx
    def _write(self, line: str) -> int:
        with self._write_lock:
            t = self.clock.now_ns()
            self.transport.write_line(line)
            self.last_tx_ns = t
            self.cmd_q.append((t, line))
        return t

    def command(self, line: str, timeout_s: float | None = None, check: bool = True) -> P.Reply:
        """Send one command and wait for its reply. ERR replies raise CommandError unless check=False."""
        name = P.command_name(line)
        with self._cmd_lock:
            pend = _Pending(name)
            with self._cond:
                self._pending = pend
            try:
                self._write(line)
                if not self.wait_until(lambda: pend.done, timeout_s or self.command_timeout_s):
                    if self.link_error:
                        raise RigFault(self.link_error)
                    raise CommandTimeout(f"no reply to {line!r} within {timeout_s or self.command_timeout_s} s")
            finally:
                with self._cond:
                    self._pending = None
        reply: P.Reply = pend.reply
        if check and not reply.ok:
            raise CommandError(line, reply)
        return reply

    def try_command(self, line: str, timeout_s: float = 0.5) -> P.Reply | None:
        """Best-effort command for shutdown paths: never raises, None on failure."""
        try:
            return self.command(line, timeout_s=timeout_s, check=False)
        except (DeviceError, OSError):
            return None

    def ping(self, timeout_s: float = 0.5) -> tuple[int, int, int] | None:
        """One PING round trip -> (t_send_ns, t_recv_ns, t_us_unwrapped), also fed to the clock-sync fit."""
        seq = next(self._ping_seq)
        with self._cmd_lock:
            pend = _Pending("PING", seq=seq)
            with self._cond:
                self._pending = pend
            try:
                t_send = self._write(P.cmd_ping(seq))
                if not self.wait_until(lambda: pend.done, timeout_s):
                    if self.link_error:
                        raise RigFault(self.link_error)
                    return None
            finally:
                with self._cond:
                    self._pending = None
        _, t_us = pend.reply
        t_recv = max(pend.t_rx_ns, t_send)
        self.sync.add(t_send, t_recv, t_us)
        return t_send, t_recv, t_us

    def sync_burst(self, n: int = 20, gap_s: float = 0.02) -> int:
        """n PINGs spaced gap_s apart (before / after a run); returns how many were answered."""
        ok = 0
        for _ in range(n):
            ok += self.ping() is not None
            self.clock.sleep(gap_s)
        return ok

    # ---------------------------------------------------------------------------- typed helpers
    def get_info(self) -> dict[str, str]:
        reply = self.command(P.cmd_info())
        self.info_fields = dict(reply.fields)
        return self.info_fields

    def stat(self) -> Stat:
        f = self.command(P.cmd_stat()).fields
        try:
            return Stat(int(f["t"]), int(f["pos"]), int(f["vel"]), f["state"], int(f["en"]), int(f["home"]),
                        int(f["estop_pin"]))
        except (KeyError, ValueError) as exc:
            raise DeviceError(f"malformed STAT reply {f!r}") from exc

    def event_mark(self) -> int:
        return len(self.event_log)

    def events_since(self, mark: int) -> list[tuple[int, int, P.Event]]:
        return self.event_log[mark:]

    # ---------------------------------------------------------------------------- heartbeat
    def start_heartbeat(self, period_s: float) -> None:
        """HB whenever nothing else was sent for period_s, only while the owner is alive (touch()).

        Threaded devices get a thread so a long blocking call in the owner still keeps the firmware watchdog
        fed; touch() liveness stops it when the owner itself hangs. Unthreaded devices rely on service().
        """
        self._hb_period_s = float(period_s)
        self.touch()
        if self.threaded and self._hb_thread is None:
            self._hb_thread = threading.Thread(target=self._hb_loop, name="calrig-heartbeat", daemon=True)
            self._hb_thread.start()

    def touch(self) -> None:
        """The owner's main loop is alive."""
        self._last_touch_ns = self.clock.now_ns()

    def heartbeat_tick(self) -> None:
        if self._hb_period_s <= 0.0 or self.fault is not None:
            return
        now = self.clock.now_ns()
        if now - self._last_touch_ns > 2_000_000_000:
            return  # owner hung or blocked in operator input: let the firmware watchdog do its job
        if now - self.last_tx_ns >= int(self._hb_period_s * 1e9):
            try:
                self._write(P.cmd_hb())
            except DeviceError as exc:
                self._set_fault(str(exc), link=True)

    def _hb_loop(self) -> None:
        while not self._stop_evt.wait(max(0.02, self._hb_period_s / 4.0)):
            self.heartbeat_tick()

    # ---------------------------------------------------------------------------- shutdown
    def safe_stop(self, max_wait_s: float = 8.0) -> None:
        """STOP (controlled decel), wait for the table to rest while feeding the watchdog, then EN 0.

        A second Ctrl-C while waiting skips straight to EN 0. Never raises.
        """
        try:
            self.touch()
            self.try_command(P.cmd_stop())
            end = self.clock.now_ns() + int(max_wait_s * 1e9)
            while self.clock.now_ns() < end:
                self.touch()
                self.heartbeat_tick()
                r = self.try_command(P.cmd_stat(), timeout_s=0.3)
                if r is None or not r.ok or r.fields.get("state") in ("IDLE", "ESTOP"):
                    break
                self.clock.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.try_command(P.cmd_en(False))
            except KeyboardInterrupt:
                pass
