"""GPIO and PWM backends (contract: docs/ARCHITECTURE.md [hw]).

`lgpio` is imported lazily inside LgpioGpio so the laptop path never touches it; the sim backends are
pure Python and are what the tests, the synthetic pipeline and `--sim` tools run on.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from tbot.config import Config

PULLS = ("up", "down", "none")
EDGES = ("rising", "falling", "both")

#: Where the kernel exposes hardware PWM channels (overridable for tests/odd kernels).
PWM_SYSFS_ROOT = "/sys/class/pwm"
#: How long to wait for udev to create/chown an exported PWM channel.
PWM_EXPORT_TIMEOUT_S = 1.5

#: Margin kept around each servo pulse when the frame is restarted (restart_pulse_us): the frame phase is inferred
#: from the time `enable` was written, which is good to tens of us.
PWM_RESTART_GUARD_NS = 300_000


def restart_wait_ns(phase_ns: int, pulse_ns: int, period_ns: int, guard_ns: int = PWM_RESTART_GUARD_NS) -> int:
    """How long to wait before restarting a running PWM frame so no pulse is cut short.

    phase_ns: time since the current frame began (the line is high for the first pulse_ns of it). Restarting is
    safe once the pulse plus guard is over and while the next pulse is more than guard away; otherwise wait until
    the next pulse plus guard has passed (at most one pulse plus two guards).
    """
    phase = int(phase_ns) % int(period_ns)
    safe_from = int(pulse_ns) + int(guard_ns)
    safe_to = int(period_ns) - int(guard_ns)
    if safe_from <= phase <= safe_to:
        return 0
    if phase < safe_from:
        return safe_from - phase
    return int(period_ns) - phase + safe_from
#: Path the auto backend reads to recognise a Raspberry Pi.
DEVICE_TREE_MODEL = "/proc/device-tree/model"


class Gpio(Protocol):
    """Minimal digital IO used by the arming switch, actuators and latency fixtures."""

    def setup_output(self, pin: int, level: int = 0) -> None: ...

    def write(self, pin: int, level: int) -> None: ...

    def setup_input(self, pin: int, pull: str = "up") -> None: ...

    def read(self, pin: int) -> int: ...

    def watch(self, pin: int, edge: str, callback: Callable[[int, int, int], None]) -> None: ...

    def close(self) -> None: ...


def _check_pull(pull: str) -> str:
    pull = str(pull).lower()
    if pull not in PULLS:
        raise ValueError(f"pull must be one of {PULLS}, got {pull!r}")
    return pull


def _check_edge(edge: str) -> str:
    edge = str(edge).lower()
    if edge not in EDGES:
        raise ValueError(f"edge must be one of {EDGES}, got {edge!r}")
    return edge


def _edge_matches(edge: str, level: int) -> bool:
    return edge == "both" or (edge == "rising" and level == 1) or (edge == "falling" and level == 0)


# --------------------------------------------------------------------------------------------- lgpio
class LgpioGpio:
    """Real Pi GPIO through the `lgpio` daemonless library (apt: python3-lgpio)."""

    def __init__(self, chip: int = 0):
        import lgpio  # lazy: Pi-only

        self._lg = lgpio
        self.chip = int(chip)
        self._h = lgpio.gpiochip_open(self.chip)
        self._lock = threading.RLock()
        self._claims: dict[int, str] = {}  # pin -> "in" | "out" | "alert"
        self._pull_flags: dict[int, int] = {}
        self._callbacks: list = []
        self._watchers: dict[int, list[tuple[str, Callable[[int, int, int], None]]]] = {}
        # lgpio reports an ns event timestamp whose clock (CLOCK_MONOTONIC on recent gpio uAPI kernels,
        # CLOCK_REALTIME on older ones) is not documented reliably. Sample the offset between the two
        # clocks now, then decide on the first event which one the kernel is using and keep that choice.
        self._rt_to_mono_ns = time.monotonic_ns() - time.time_ns()
        self._ts_mode: str | None = None

    # -- claims -------------------------------------------------------------------------------
    def _free(self, pin: int) -> None:
        if self._claims.pop(pin, None) is not None:
            self._lg.gpio_free(self._h, pin)

    def _pull_flag(self, pull: str) -> int:
        return {
            "up": self._lg.SET_PULL_UP,
            "down": self._lg.SET_PULL_DOWN,
            "none": self._lg.SET_PULL_NONE,
        }[_check_pull(pull)]

    def setup_output(self, pin: int, level: int = 0) -> None:
        pin = int(pin)
        with self._lock:
            if self._claims.get(pin) != "out":
                self._free(pin)
                self._lg.gpio_claim_output(self._h, pin, int(bool(level)))
                self._claims[pin] = "out"
            else:
                self._lg.gpio_write(self._h, pin, int(bool(level)))

    def write(self, pin: int, level: int) -> None:
        with self._lock:
            self._lg.gpio_write(self._h, int(pin), int(bool(level)))

    def setup_input(self, pin: int, pull: str = "up") -> None:
        pin = int(pin)
        flag = self._pull_flag(pull)
        with self._lock:
            self._pull_flags[pin] = flag
            if self._claims.get(pin) == "alert":
                return  # an alert claim already reads the pin; re-claiming would drop the callbacks
            self._free(pin)
            self._lg.gpio_claim_input(self._h, pin, flag)
            self._claims[pin] = "in"

    def read(self, pin: int) -> int:
        with self._lock:
            return int(self._lg.gpio_read(self._h, int(pin)))

    def watch(self, pin: int, edge: str, callback: Callable[[int, int, int], None]) -> None:
        pin, edge = int(pin), _check_edge(edge)
        with self._lock:
            first = pin not in self._watchers
            self._watchers.setdefault(pin, []).append((edge, callback))
            if not first:
                return  # one alert claim per pin; _dispatch fans out to every watcher
            flag = self._pull_flags.get(pin, self._lg.SET_PULL_NONE)
            if self._claims.get(pin) != "alert":
                self._free(pin)
                self._lg.gpio_claim_alert(self._h, pin, self._lg.BOTH_EDGES, flag)
                self._claims[pin] = "alert"
            self._callbacks.append(self._lg.callback(self._h, pin, self._lg.BOTH_EDGES, self._dispatch))

    def _event_ns(self, ts_ns: int) -> int:
        """lgpio event timestamp -> time.monotonic_ns()."""
        mono = time.monotonic_ns()
        ts_ns = int(ts_ns)
        if self._ts_mode is None:
            self._ts_mode = "monotonic" if abs(ts_ns - mono) <= abs(ts_ns - time.time_ns()) else "realtime"
        t = ts_ns if self._ts_mode == "monotonic" else ts_ns + self._rt_to_mono_ns
        # A stamp more than a second away from now cannot be this event: fall back to the read time.
        return t if abs(t - mono) < 1_000_000_000 else mono

    def _dispatch(self, chip: int, gpio: int, level: int, ts_ns: int) -> None:
        if level > 1:  # 2 = watchdog timeout, not a level change
            return
        t_ns = self._event_ns(ts_ns)
        with self._lock:
            watchers = list(self._watchers.get(int(gpio), ()))
        for edge, cb in watchers:
            if _edge_matches(edge, int(level)):
                cb(int(gpio), int(level), t_ns)

    def close(self) -> None:
        with self._lock:
            for cb in self._callbacks:
                try:
                    cb.cancel()
                except Exception:
                    pass
            self._callbacks.clear()
            self._watchers.clear()
            for pin in list(self._claims):
                try:
                    self._free(pin)
                except Exception:
                    pass
            if self._h is not None:
                self._lg.gpiochip_close(self._h)
                self._h = None


# --------------------------------------------------------------------------------------------- sim
class SimGpio:
    """In-memory GPIO: `inject()` drives inputs (watchers fire synchronously), writes are logged."""

    def __init__(self, clock=None):
        self._clock = clock
        self._lock = threading.RLock()
        self.levels: dict[int, int] = {}
        self.modes: dict[int, str] = {}
        self.pulls: dict[int, str] = {}
        self.writes: list[tuple[int, int, int]] = []  # (t_ns, pin, level)
        self._watchers: dict[int, list[tuple[str, Callable[[int, int, int], None]]]] = {}
        self.closed = False

    def _now(self) -> int:
        return self._clock.now_ns() if self._clock is not None else time.monotonic_ns()

    def setup_output(self, pin: int, level: int = 0) -> None:
        pin = int(pin)
        with self._lock:
            self.modes[pin] = "out"
        self.write(pin, level)

    def write(self, pin: int, level: int) -> None:
        pin, level = int(pin), int(bool(level))
        with self._lock:
            self.levels[pin] = level
            self.writes.append((self._now(), pin, level))

    def setup_input(self, pin: int, pull: str = "up") -> None:
        pin, pull = int(pin), _check_pull(pull)
        with self._lock:
            self.modes[pin] = "in"
            self.pulls[pin] = pull
            self.levels[pin] = 1 if pull == "up" else 0

    def read(self, pin: int) -> int:
        with self._lock:
            return int(self.levels.get(int(pin), 0))

    def watch(self, pin: int, edge: str, callback: Callable[[int, int, int], None]) -> None:
        pin, edge = int(pin), _check_edge(edge)
        with self._lock:
            self._watchers.setdefault(pin, []).append((edge, callback))

    def inject(self, pin: int, level: int, t_ns: int | None = None) -> None:
        """Drive an input pin; watchers for a matching edge run synchronously on the caller's thread."""
        pin, level = int(pin), int(bool(level))
        with self._lock:
            previous = self.levels.get(pin)
            self.levels[pin] = level
            if previous == level:
                return
            watchers = list(self._watchers.get(pin, ()))
            t = self._now() if t_ns is None else int(t_ns)
        for edge, cb in watchers:
            if _edge_matches(edge, level):
                cb(pin, level, t)

    def close(self) -> None:
        with self._lock:
            self._watchers.clear()
            self.closed = True


# --------------------------------------------------------------------------------------------- factory
def is_raspberry_pi() -> bool:
    try:
        model = Path(DEVICE_TREE_MODEL).read_bytes().decode("utf-8", "replace")
    except OSError:
        return False
    return "raspberry pi" in model.lower()


def open_gpio(cfg: Config) -> Gpio:
    """Open the backend named by cfg.gpio.backend ("auto" = lgpio on a Pi where it imports, else sim)."""
    backend = str(cfg.gpio.backend).lower()
    chip = int(cfg.gpio.chip)
    if backend == "sim":
        return SimGpio()
    if backend == "lgpio":
        return LgpioGpio(chip)
    if backend != "auto":
        raise ValueError(f"gpio.backend must be auto|lgpio|sim, got {cfg.gpio.backend!r}")
    if is_raspberry_pi():
        try:
            return LgpioGpio(chip)
        except (ImportError, OSError, AttributeError):
            pass  # no lgpio / no permission on /dev/gpiochip*: prototype against the sim backend
    return SimGpio()


# --------------------------------------------------------------------------------------------- PWM
class HardwarePwm:
    """Kernel hardware PWM through sysfs (servo: GPIO18 = pwmchip0/pwm0, `dtoverlay=pwm,pin=18,func=2`)."""

    is_sim = False

    def __init__(self, chip: int, channel: int, frame_hz: float):
        self.chip, self.channel = int(chip), int(channel)
        self.frame_hz = float(frame_hz)
        if self.frame_hz <= 0:
            raise ValueError(f"frame_hz must be > 0, got {frame_hz}")
        self.period_ns = int(round(1e9 / self.frame_hz))
        self._root = Path(PWM_SYSFS_ROOT) / f"pwmchip{self.chip}"
        self._dir = self._root / f"pwm{self.channel}"
        self._exported = False
        self._enabled = False
        self.pulse_us: float | None = None
        self._lock = threading.RLock()
        self._open()

    # -- sysfs --------------------------------------------------------------------------------
    def _hint(self) -> str:
        return (
            f"hardware PWM channel {self.channel} on {self._root} is not available.\n"
            "  * enable it in /boot/firmware/config.txt: dtoverlay=pwm,pin=18,func=2 (GPIO18 = PWM0), reboot;\n"
            "  * check that the pwm sysfs class exists: ls /sys/class/pwm/"
        )

    def _perm_hint(self, path: Path) -> str:
        return (
            f"no permission to write {path}.\n"
            "  * add your user to the 'gpio' group (sudo usermod -aG gpio $USER, then log in again), and\n"
            '  * install a udev rule so the pwm sysfs files are group-writable, e.g. /etc/udev/rules.d/99-pwm.rules:\n'
            '      SUBSYSTEM=="pwm", ACTION=="add", RUN+="/bin/chgrp -R gpio /sys%p", RUN+="/bin/chmod -R g=u /sys%p"\n'
            "  * or run this tool with sudo."
        )

    def _write_attr(self, name: str, value: object, where: Path | None = None) -> None:
        path = (where or self._dir) / name
        try:
            with open(path, "w") as f:
                f.write(str(value))
        except PermissionError as exc:
            raise PermissionError(self._perm_hint(path)) from exc
        except FileNotFoundError as exc:
            raise FileNotFoundError(self._hint()) from exc

    def _open(self) -> None:
        if not self._root.is_dir():
            raise FileNotFoundError(self._hint())
        if not self._dir.is_dir():
            self._write_attr("export", self.channel, where=self._root)
            self._exported = True
            deadline = time.monotonic() + PWM_EXPORT_TIMEOUT_S
            while time.monotonic() < deadline:  # udev needs a moment to create and chown the channel
                if (self._dir / "period").exists():
                    break
                time.sleep(0.02)
            else:
                raise FileNotFoundError(f"{self._dir} did not appear after export.\n{self._hint()}")
            time.sleep(0.05)  # let the udev chgrp/chmod rule land before writing attributes
        # duty must never exceed the period while reprogramming it
        self._write_attr("duty_cycle", 0)
        self._write_attr("period", self.period_ns)
        self._write_attr("enable", 1)
        self._enabled = True
        self._t_on_ns = time.monotonic_ns()

    # -- API ----------------------------------------------------------------------------------
    def set_pulse_us(self, us: float) -> None:
        """Set the pulse width in microseconds (clamped to one frame).

        On a running channel the new width starts with the NEXT frame (0 - 1/frame_hz later); from limp the frame
        starts at once. restart_pulse_us() starts a fresh frame now instead.
        """
        duty_ns = int(round(max(0.0, min(float(us) * 1000.0, float(self.period_ns)))))
        with self._lock:
            if not self._enabled:
                # width first, then switch on: the first pulse out is already the right one. (The other way
                # round the frame starts with the old width - zero after disable() - and the new one lands in
                # the middle of it: a first pulse of the wrong length, on the trigger side for this lever.)
                self._write_attr("duty_cycle", duty_ns)
                self._write_attr("enable", 1)
                self._enabled = True
                self._t_on_ns = time.monotonic_ns()
            else:
                self._write_attr("duty_cycle", duty_ns)
            self.pulse_us = duty_ns / 1000.0

    def restart_pulse_us(self, us: float) -> None:
        """Set the pulse width AND start a new frame now, so the new pulse goes out at once rather than up to one
        frame (20 ms at 50 Hz) later. Waits out the current pulse first (<= ~2 ms) so none is ever cut short."""
        duty_ns = int(round(max(0.0, min(float(us) * 1000.0, float(self.period_ns)))))
        with self._lock:
            if not self._enabled or self.pulse_us is None:
                self.set_pulse_us(us)
                return
            wait = restart_wait_ns(time.monotonic_ns() - self._t_on_ns, int(round(self.pulse_us * 1000.0)),
                                   self.period_ns)
            if wait:
                time.sleep(wait / 1e9)
            self._write_attr("enable", 0)  # PWEN off/on restarts the frame counter
            self._write_attr("duty_cycle", duty_ns)
            self._write_attr("enable", 1)
            self._t_on_ns = time.monotonic_ns()
            self.pulse_us = duty_ns / 1000.0

    def disable(self) -> None:
        """Stop emitting pulses (servo goes limp, PWM pin idles low)."""
        with self._lock:
            if self._enabled:
                # Never cut a pulse short: a servo reads a truncated pulse as a command further toward the
                # short-pulse end of its travel, which on this gun is the trigger side. Wait out the pulse that
                # is going out (as restart_pulse_us does), set the width to zero - a width change takes effect
                # with the next frame - let that empty frame begin, and only then stop the channel. The frame
                # phase is only an estimate after a long hold (two clocks), hence both waits.
                if self.pulse_us:
                    wait = restart_wait_ns(time.monotonic_ns() - self._t_on_ns, int(round(self.pulse_us * 1000.0)),
                                           self.period_ns)
                    if wait:
                        time.sleep(wait / 1e9)
                    self._write_attr("duty_cycle", 0)
                    time.sleep((self.period_ns + 2_000_000) / 1e9)
                else:
                    self._write_attr("duty_cycle", 0)
                self._write_attr("enable", 0)
                self._enabled = False
            self.pulse_us = None

    def close(self) -> None:
        with self._lock:
            try:
                self.disable()
            finally:
                if self._exported:
                    try:
                        self._write_attr("unexport", self.channel, where=self._root)
                    except OSError:
                        pass
                    self._exported = False


class SimPwm:
    """HardwarePwm's API in memory: every commanded pulse is logged as (t_ns, us)."""

    is_sim = True

    def __init__(self, chip: int, channel: int, frame_hz: float):
        self.chip, self.channel = int(chip), int(channel)
        self.frame_hz = float(frame_hz)
        if self.frame_hz <= 0:
            raise ValueError(f"frame_hz must be > 0, got {frame_hz}")
        self.period_ns = int(round(1e9 / self.frame_hz))
        self.pulses: list[tuple[int, float]] = []
        self.pulse_us: float | None = None
        self.enabled = True
        self.restarts = 0
        self._lock = threading.RLock()

    def set_pulse_us(self, us: float) -> None:
        duty_ns = int(round(max(0.0, min(float(us) * 1000.0, float(self.period_ns)))))
        with self._lock:
            self.enabled = True
            self.pulse_us = duty_ns / 1000.0
            self.pulses.append((time.monotonic_ns(), self.pulse_us))

    def restart_pulse_us(self, us: float) -> None:
        with self._lock:
            if self.enabled and self.pulse_us is not None:
                self.restarts += 1
            self.set_pulse_us(us)

    def disable(self) -> None:
        with self._lock:
            self.enabled = False
            self.pulse_us = None
            self.pulses.append((time.monotonic_ns(), 0.0))

    def close(self) -> None:
        self.disable()
