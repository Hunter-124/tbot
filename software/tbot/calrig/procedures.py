"""Named calibration procedures. Each one validates against the rig limits BEFORE anything moves, then drives a
RigSession; the recorder captures everything, the procedure only returns a small result dict for meta.json.

    hold        still for N s with the driver holding: gyro bias (and noise)
    turns       +-A deg moves at a fixed rate, repeated, returning to 0: gyro scale per axis, axis map and sign
    sine        enveloped sinusoid about 0: IMU time offset (cross-correlation), rate/accel dynamics
    rate        constant-rate sweeps at several rates: scale vs rate, rate-dependent bias
    home_check  home, excursion, home again: the second HOMED offset is the lost-step count
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from tbot.calrig import protocol as P

if TYPE_CHECKING:  # pragma: no cover
    from tbot.calrig.session import Limits, RigSession

__all__ = ["Procedure", "Hold", "Turns", "Sine", "Rate", "HomeCheck", "PROCEDURES", "make_procedure",
           "procedure_help"]

MIN_CRUISE_DEG = 10.0  # a "constant-rate" sweep needs at least this much cruise at the commanded rate
SETTLE_S = 3.5         # pause between moves: analysis needs >= 3 s still (2 s + 0.5 s trimmed off each side)


class Procedure:
    name: ClassVar[str] = ""
    needs_home: ClassVar[bool] = False
    summary: ClassVar[str] = ""

    def params(self) -> dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[call-overload]

    def validate(self, limits: "Limits") -> None:
        raise NotImplementedError

    def duration_estimate_s(self, limits: "Limits") -> float:
        return 0.0

    def run(self, s: "RigSession") -> dict[str, Any]:
        raise NotImplementedError


def _check_rate(rate: float, limits: "Limits", what: str = "rate") -> None:
    if not 0.0 < rate <= limits.vmax_cap_dps + 1e-9:
        raise ValueError(f"{what} {rate:g} deg/s outside (0, {limits.vmax_cap_dps:g}] deg/s")


def _check_accel(accel: float, limits: "Limits") -> None:
    if not 0.0 < accel <= limits.amax_cap_dps2 + 1e-9:
        raise ValueError(f"acceleration {accel:g} deg/s^2 outside (0, {limits.amax_cap_dps2:g}] deg/s^2")


def _ramp_time(dist: float, v: float, a: float) -> float:
    from tbot.calrig.session import move_time_s

    return move_time_s(dist, v, a)


@dataclass
class Hold(Procedure):
    name: ClassVar[str] = "hold"
    summary: ClassVar[str] = "stand still, driver enabled: gyro bias and noise"
    duration_s: float = 120.0

    def validate(self, limits: "Limits") -> None:
        if not 1.0 <= self.duration_s <= 3600.0:
            raise ValueError("duration_s must be within 1..3600 s")

    def duration_estimate_s(self, limits: "Limits") -> float:
        return self.duration_s

    def run(self, s: "RigSession") -> dict[str, Any]:
        s.log(f"holding still for {self.duration_s:g} s: do not touch the gun or the table")
        s.wait(self.duration_s)
        return {"duration_s": self.duration_s}


@dataclass
class Turns(Procedure):
    name: ClassVar[str] = "turns"
    needs_home: ClassVar[bool] = True
    summary: ClassVar[str] = "0 -> +A -> -A -> 0 at a fixed rate, repeated: gyro scale, axis map"
    amplitude_deg: float = 90.0
    rate_dps: float = 30.0
    repeats: int = 2
    settle_s: float = SETTLE_S
    accel_dps2: float | None = None   # default: the configured amax

    def _accel(self, limits: "Limits") -> float:
        return limits.amax_dps2 if self.accel_dps2 is None else self.accel_dps2

    def validate(self, limits: "Limits") -> None:
        if not 5.0 <= self.amplitude_deg <= limits.soft_deg + 1e-9:
            raise ValueError(f"amplitude_deg must be within 5..{limits.soft_deg:g} (the soft limit)")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.settle_s < 0.0:
            raise ValueError("settle_s must be >= 0")
        _check_rate(self.rate_dps, limits)
        _check_accel(self._accel(limits), limits)

    def duration_estimate_s(self, limits: "Limits") -> float:
        a, v, A = self._accel(limits), self.rate_dps, self.amplitude_deg
        one = _ramp_time(A, v, a) * 2 + _ramp_time(2 * A, v, a)
        return self.settle_s + self.repeats * (one + 3 * self.settle_s)

    def run(self, s: "RigSession") -> dict[str, Any]:
        a = self._accel(s.limits)
        s.set_rates(self.rate_dps, a)
        A = self.amplitude_deg
        reached = min(self.rate_dps, math.sqrt(a * A))
        s.log(f"turns: +-{A:g} deg at {self.rate_dps:g} deg/s (peak {reached:g}), accel {a:g} deg/s^2, "
              f"{self.repeats} repeat(s)")
        s.wait(self.settle_s)
        for i in range(self.repeats):
            for target in (A, -A, 0.0):
                s.log(f"  repeat {i + 1}/{self.repeats}: -> {target:+g} deg")
                s.move_to(target, settle_s=self.settle_s)
        return {"moves": 3 * self.repeats, "peak_rate_dps_reached": reached, "accel_dps2": a}


@dataclass
class Sine(Procedure):
    name: ClassVar[str] = "sine"
    needs_home: ClassVar[bool] = True
    summary: ClassVar[str] = "enveloped sinusoid about 0: IMU time offset and dynamics"
    amplitude_deg: float = 15.0
    period_s: float = 6.5
    cycles: int = 6
    settle_s: float = SETTLE_S

    def validate(self, limits: "Limits") -> None:
        if self.amplitude_deg <= 0.0 or self.period_s <= 0.0 or self.cycles < 1:
            raise ValueError("amplitude_deg and period_s must be > 0 and cycles >= 1")
        if self.amplitude_deg > limits.soft_deg + 1e-9:
            raise ValueError(f"amplitude {self.amplitude_deg:g} deg exceeds the +-{limits.soft_deg:g} deg soft limits")
        accel = P.sine_peak_accel_dps2(self.amplitude_deg, self.period_s)
        if accel > limits.amax_dps2 + 1e-9:
            raise ValueError(
                f"sine peak acceleration {accel:.2f} deg/s^2 (A (2 pi / T)^2) exceeds amax {limits.amax_dps2:g} "
                f"deg/s^2: lower the amplitude or lengthen the period (the firmware refuses it too)")
        speed = self.amplitude_deg * 2.0 * math.pi / self.period_s
        if speed > limits.vmax_dps + 1e-9:
            raise ValueError(f"sine peak rate {speed:.2f} deg/s exceeds vmax {limits.vmax_dps:g} deg/s")
        if int(round(self.period_s * 1000.0)) < 1:
            raise ValueError("period_s too short")

    def duration_estimate_s(self, limits: "Limits") -> float:
        return self.period_s * self.cycles + 2 * self.settle_s

    def run(self, s: "RigSession") -> dict[str, Any]:
        s.set_rates(s.limits.vmax_dps, s.limits.amax_dps2)  # SINE is checked against the amax in force
        amp = s.geometry.to_usteps(self.amplitude_deg)
        accel = P.sine_peak_accel_dps2(self.amplitude_deg, self.period_s)
        s.log(f"sine: {self.amplitude_deg:g} deg, {self.period_s:g} s, {self.cycles} cycles "
              f"(A w^2 = {accel:.2f} of amax {s.amax_dps2:.2f} deg/s^2)")
        s.wait(self.settle_s)
        mark = s.begin_motion()
        s.dev.command(P.cmd_sine(amp, int(round(self.period_s * 1000.0)), self.cycles))
        s.wait_idle(mark, self.period_s * self.cycles + 10.0)
        s.wait(self.settle_s)
        return {"peak_accel_dps2": accel, "peak_rate_dps": self.amplitude_deg * 2.0 * math.pi / self.period_s}


def _floats(v: Any) -> tuple[float, ...]:
    if isinstance(v, (int, float)):
        return (float(v),)
    return tuple(float(x) for x in v)


@dataclass
class Rate(Procedure):
    name: ClassVar[str] = "rate"
    needs_home: ClassVar[bool] = True
    summary: ClassVar[str] = "constant-rate sweeps at several rates: scale and bias versus rate"
    rates_dps: tuple = (10.0, 20.0, 30.0, 45.0)
    span_deg: float = 160.0            # sweep from -span/2 to +span/2
    repeats: int = 1                   # back-and-forth passes per rate
    approach_dps: float = 20.0         # positioning moves
    settle_s: float = SETTLE_S
    accel_dps2: float | None = None

    def __post_init__(self) -> None:
        self.rates_dps = _floats(self.rates_dps)

    def _accel(self, limits: "Limits") -> float:
        return limits.amax_dps2 if self.accel_dps2 is None else self.accel_dps2

    def validate(self, limits: "Limits") -> None:
        if not self.rates_dps:
            raise ValueError("rates_dps is empty")
        if not 20.0 <= self.span_deg <= 2.0 * limits.soft_deg + 1e-9:
            raise ValueError(f"span_deg must be within 20..{2 * limits.soft_deg:g}")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        a = self._accel(limits)
        _check_accel(a, limits)
        _check_rate(self.approach_dps, limits, "approach_dps")
        for r in self.rates_dps:
            _check_rate(r, limits)
            ramps = r * r / a                    # distance of the ramp up plus the ramp down
            if ramps + MIN_CRUISE_DEG > self.span_deg:
                vmax_ok = math.sqrt(max(0.0, (self.span_deg - MIN_CRUISE_DEG) * a))
                raise ValueError(f"rate {r:g} deg/s never cruises within a {self.span_deg:g} deg sweep at "
                                 f"{a:g} deg/s^2 (ramps take {ramps:.0f} deg): max {vmax_ok:.1f} deg/s, or widen "
                                 f"the span / raise the acceleration")

    def duration_estimate_s(self, limits: "Limits") -> float:
        a = self._accel(limits)
        t = _ramp_time(self.span_deg / 2, self.approach_dps, a) * 2
        for r in self.rates_dps:
            t += self.repeats * 2 * (_ramp_time(self.span_deg, r, a) + self.settle_s)
        return t

    def run(self, s: "RigSession") -> dict[str, Any]:
        a = self._accel(s.limits)
        half = self.span_deg / 2.0
        cruise = {}
        s.set_rates(self.approach_dps, a)
        s.move_to(-half, settle_s=self.settle_s)
        for r in self.rates_dps:
            s.set_rates(r, a)
            cruise[f"{r:g}"] = self.span_deg - r * r / a
            s.log(f"rate {r:g} deg/s: {self.repeats} x (-{half:g} <-> +{half:g}), {cruise[f'{r:g}']:.0f} deg of cruise")
            for _ in range(self.repeats):
                s.move_to(half, settle_s=self.settle_s)
                s.move_to(-half, settle_s=self.settle_s)
        s.set_rates(self.approach_dps, a)
        s.move_to(0.0, settle_s=self.settle_s)
        return {"rates_dps": list(self.rates_dps), "cruise_deg": cruise, "accel_dps2": a}


@dataclass
class HomeCheck(Procedure):
    name: ClassVar[str] = "home_check"
    summary: ClassVar[str] = "home, excursion, home again: the second HOMED offset is the lost-step count"
    excursion_deg: float = 60.0
    rate_dps: float = 30.0
    tolerance_deg: float = 0.5
    home_dir: int | None = None

    def validate(self, limits: "Limits") -> None:
        if not 5.0 <= self.excursion_deg <= limits.soft_deg + 1e-9:
            raise ValueError(f"excursion_deg must be within 5..{limits.soft_deg:g}")
        _check_rate(self.rate_dps, limits)
        if self.tolerance_deg <= 0.0:
            raise ValueError("tolerance_deg must be > 0")
        if self.home_dir not in (None, 1, -1):
            raise ValueError("home_dir must be 1 or -1")

    def duration_estimate_s(self, limits: "Limits") -> float:
        return 2 * (2 * limits.soft_deg / 5.0) + 4 * _ramp_time(self.excursion_deg, self.rate_dps, limits.amax_dps2)

    def run(self, s: "RigSession") -> dict[str, Any]:
        d = int(self.home_dir if self.home_dir is not None else s.rc.home_dir)  # SAME direction both times
        first = s.home(direction=d)
        s.set_rates(self.rate_dps, s.limits.amax_dps2)
        for target in (self.excursion_deg, -self.excursion_deg, 0.0):
            s.log(f"excursion -> {target:+g} deg")
            s.move_to(target, settle_s=0.5)
        second = s.home(direction=d)
        lost_deg = second["offset_deg"]
        ok = abs(lost_deg) <= self.tolerance_deg
        s.log(f"second HOMED offset {second['offset_ustep']} usteps = {lost_deg:+.3f} deg "
              f"({'OK' if ok else 'LOST STEPS'}, tolerance {self.tolerance_deg:g} deg)")
        return {"first_home": first, "second_home": second, "lost_ustep": second["offset_ustep"],
                "lost_deg": lost_deg, "tolerance_deg": self.tolerance_deg, "ok": ok}


PROCEDURES: dict[str, type[Procedure]] = {c.name: c for c in (Hold, Turns, Sine, Rate, HomeCheck)}


def _coerce(cls: type, key: str, value: Any) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    if key not in fields:
        raise ValueError(f"{cls.name}: unknown parameter {key!r} (valid: {', '.join(fields)})")  # type: ignore[attr-defined]
    default = fields[key].default
    if isinstance(value, str):
        text = value.strip()
        if isinstance(default, tuple):
            return tuple(float(x) for x in text.strip("()[]").split(",") if x.strip())
        if text.lower() in ("none", "null", ""):
            return None
        try:
            num = float(text)
        except ValueError:
            raise ValueError(f"{cls.name}.{key}: not a number: {value!r}") from None
        return int(num) if isinstance(default, int) and not isinstance(default, bool) and num == int(num) else num
    return value


def make_procedure(name: str, params: dict[str, Any] | None = None) -> Procedure:
    if name not in PROCEDURES:
        raise ValueError(f"unknown procedure {name!r}; choose from {', '.join(PROCEDURES)}")
    cls = PROCEDURES[name]
    kwargs = {k: _coerce(cls, k, v) for k, v in (params or {}).items()}
    return cls(**kwargs)


def procedure_help() -> str:
    lines = []
    for name, cls in PROCEDURES.items():
        fields = ", ".join(f"{f.name}={getattr(cls(), f.name)!r}" for f in dataclasses.fields(cls))
        lines.append(f"  {name:<11} {cls.summary}\n              params: {fields}")
    return "\n".join(lines)
