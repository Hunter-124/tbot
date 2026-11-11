"""MPU-6050 (GY-521) IMU over I2C: FIFO mode with a background reader thread.

Contract: docs/ARCHITECTURE.md [imu]. Registers, scalings and the DLPF table come from the InvenSense
MPU-6000/6050 Register Map rev 4.2 (RM-MPU-6000A-00) and Product Specification rev 3.4 (PS-MPU-6000A-00).
smbus2/smbus are imported lazily so the laptop path never needs them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tbot.types import Clock, ImuSample, ImuSource, MonotonicClock

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tbot.camera.base import FrameSource
    from tbot.config import Calibration, Config

__all__ = [
    "Mpu6050",
    "NullImu",
    "open_imu",
    "is_raspberry_pi",
    "ImuCalibration",
    "TimestampRegressor",
    "parse_fifo",
    "dlpf_cfg_for",
    "gyro_fs_sel",
    "accel_fs_sel",
    "sample_rate_div",
]

# ----------------------------------------------------------------------------------- register map
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_FIFO_EN = 0x23
REG_INT_STATUS = 0x3A
REG_TEMP_OUT_H = 0x41
REG_USER_CTRL = 0x6A
REG_PWR_MGMT_1 = 0x6B
REG_FIFO_COUNT_H = 0x72
REG_FIFO_COUNT_L = 0x73
REG_FIFO_R_W = 0x74
REG_WHO_AM_I = 0x75

PWR1_DEVICE_RESET = 0x80
PWR1_CLK_PLL_XGYRO = 0x01
USER_CTRL_FIFO_EN = 0x40
USER_CTRL_FIFO_RESET = 0x04
INT_STATUS_FIFO_OFLOW = 0x10
FIFO_EN_ACCEL_GYRO = 0x78  # ACCEL + GYRO_X/Y/Z -> 12 bytes per sample, accel first

WHO_AM_I_VALUE = 0x68
SAMPLE_BYTES = 12
FIFO_CAPACITY_BYTES = 1024
MAX_SAMPLES_PER_READ = FIFO_CAPACITY_BYTES // SAMPLE_BYTES
SMBUS_BLOCK_MAX = 32  # SMBus block read limit (smbus2 read_i2c_block_data)

G_MPS2 = 9.80665
DEG2RAD = np.pi / 180.0
GYRO_OUTPUT_RATE_HZ = 1000.0  # with DLPF_CFG 1..6 (DLPF_CFG 0 would be 8 kHz; never used here)
RESET_WAIT_S = 0.1
WAKE_WAIT_S = 0.01
TEMP_INTERVAL_S = 1.0
MAX_QUEUED_SAMPLES = 20000  # ~40 s at 500 Hz; oldest are dropped if nobody calls read()

GYRO_LSB_PER_DPS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
ACCEL_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
DLPF_BANDWIDTH_HZ = {1: 188.0, 2: 98.0, 3: 42.0, 4: 20.0, 5: 10.0, 6: 5.0}

I2C_HELP = (
    "check the wiring and `i2cdetect -y {bus}` (expect 0x{addr:02x}); the I2C bus needs "
    "`dtparam=i2c_arm=on` in /boot/firmware/config.txt"
)


# ----------------------------------------------------------------------------------- pure helpers
def dlpf_cfg_for(bandwidth_hz: float) -> int:
    """DLPF_CFG 1..6 whose bandwidth is nearest to bandwidth_hz (all give a 1 kHz gyro output rate)."""
    return min(DLPF_BANDWIDTH_HZ, key=lambda cfg: abs(DLPF_BANDWIDTH_HZ[cfg] - float(bandwidth_hz)))


def gyro_fs_sel(range_dps: int) -> int:
    """FS_SEL for a full-scale gyro range in deg/s."""
    order = sorted(GYRO_LSB_PER_DPS)  # 250, 500, 1000, 2000 deg/s -> FS_SEL 0, 1, 2, 3
    if range_dps not in GYRO_LSB_PER_DPS:
        raise ValueError(f"imu.gyro_range_dps must be one of {sorted(GYRO_LSB_PER_DPS)}, got {range_dps}")
    return order.index(range_dps)


def accel_fs_sel(range_g: int) -> int:
    """AFS_SEL for a full-scale accelerometer range in g."""
    order = sorted(ACCEL_LSB_PER_G)  # 2, 4, 8, 16 g -> AFS_SEL 0, 1, 2, 3
    if range_g not in ACCEL_LSB_PER_G:
        raise ValueError(f"imu.accel_range_g must be one of {sorted(ACCEL_LSB_PER_G)}, got {range_g}")
    return order.index(range_g)


def sample_rate_div(sample_rate_hz: float) -> int:
    """SMPLRT_DIV for a requested output rate: rate = 1000 Hz / (1 + SMPLRT_DIV)."""
    if sample_rate_hz <= 0:
        raise ValueError(f"imu.sample_rate_hz must be > 0, got {sample_rate_hz}")
    div = int(round(GYRO_OUTPUT_RATE_HZ / float(sample_rate_hz))) - 1
    return int(min(255, max(0, div)))


def parse_fifo(data: bytes | bytearray, accel_lsb_per_g: float, gyro_lsb_per_dps: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """FIFO bytes (accel x,y,z then gyro x,y,z, big-endian int16) -> (accel m/s^2, gyro rad/s), chip frame.

    Returns (N, 3) float64 arrays; a trailing partial sample is ignored.
    """
    buf = bytes(data)
    n = len(buf) // SAMPLE_BYTES
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    raw = np.frombuffer(buf[: n * SAMPLE_BYTES], dtype=">i2").reshape(n, 6).astype(np.float64)
    accel = raw[:, 0:3] * (G_MPS2 / float(accel_lsb_per_g))
    gyro = raw[:, 3:6] * (DEG2RAD / float(gyro_lsb_per_dps))
    return accel, gyro


def parse_temperature(data: bytes | bytearray) -> float:
    """TEMP_OUT_H/L -> degrees C (register map: temp_C = raw/340 + 36.53)."""
    raw = int(np.frombuffer(bytes(data)[:2], dtype=">i2")[0])
    return raw / 340.0 + 36.53


@dataclass(eq=False)  # numpy fields: identity comparison only
class ImuCalibration:
    """calib/imu.json applied to raw chip-frame samples (biases/scale in the CHIP frame).

    gyro (rad/s) -> (raw - gyro_bias_rad_s) * gyro_scale per chip axis -> R_gun_imu. gyro_scale is optional in
    calib/imu.json (written by `tbot calrig analyze --apply`: the multiplier to apply to the chip gyro).
    """

    gyro_bias_rad_s: np.ndarray
    accel_bias_m_s2: np.ndarray
    accel_scale: np.ndarray
    R_gun_imu: np.ndarray
    time_offset_ns: int
    gyro_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))

    @classmethod
    def from_config(cls, cfg: "Config", calib: "Calibration | None") -> "ImuCalibration":
        imu_json = getattr(calib, "imu", None) or {}
        gyro_bias = _vec3(imu_json.get("gyro_bias_rad_s"), 0.0, "gyro_bias_rad_s")
        accel_bias = _vec3(imu_json.get("accel_bias_m_s2"), 0.0, "accel_bias_m_s2")
        accel_scale = _vec3(imu_json.get("accel_scale"), 1.0, "accel_scale")
        gyro_scale = _vec3(imu_json.get("gyro_scale"), 1.0, "gyro_scale")
        if np.any(gyro_scale <= 0.0):
            raise ValueError("gyro_scale must be positive")
        rot = imu_json.get("R_gun_imu")
        if rot is None:
            rot = cfg.imu.R_gun_imu
        R = np.asarray(rot, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R_gun_imu must be 3x3, got shape {R.shape}")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0:
            raise ValueError("R_gun_imu is not a right-handed rotation matrix")
        return cls(gyro_bias, accel_bias, accel_scale, R,
                   int(imu_json.get("time_offset_ns", 0) or 0), gyro_scale)

    def apply(self, accel_chip: np.ndarray, gyro_chip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw chip-frame SI samples -> calibrated GUN-frame samples ((N,3) or (3,))."""
        accel = (np.asarray(accel_chip, dtype=np.float64) - self.accel_bias_m_s2) * self.accel_scale
        gyro = (np.asarray(gyro_chip, dtype=np.float64) - self.gyro_bias_rad_s) * self.gyro_scale
        return accel @ self.R_gun_imu.T, gyro @ self.R_gun_imu.T


def _vec3(value: Any, fill: float, name: str) -> np.ndarray:
    if value is None:
        return np.full(3, fill, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.size != 3:
        raise ValueError(f"{name} must have 3 elements, got {v.size}")
    return v


class TimestampRegressor:
    """Maps a cumulative FIFO sample index to a host timestamp.

    The MPU's internal oscillator is specified to +/-1 % (PS-MPU-6000A-00 §6.1) and in practice drifts a
    few %, so the true sample period is fitted online: the k-th sample of a read is at
    anchor + k*period, and the read itself happens at most one period after its last sample. A sliding
    least-squares window over (index of the last sample in a read, read time) gives the period; the
    anchor is the lower envelope of those points (reads can only be late, never early).
    """

    __slots__ = ("window", "min_points", "_nominal", "_k", "_t", "_n", "_pos", "_k0", "_t0",
                 "_period", "_anchor_rel")

    def __init__(self, nominal_period_ns: float, window: int = 256, min_points: int = 4):
        if nominal_period_ns <= 0:
            raise ValueError("nominal_period_ns must be > 0")
        self.window = max(2, int(window))
        self.min_points = max(2, int(min_points))
        self._nominal = float(nominal_period_ns)
        self._k = np.zeros(self.window, dtype=np.float64)
        self._t = np.zeros(self.window, dtype=np.float64)
        self._n = 0
        self._pos = 0
        self._k0 = 0
        self._t0 = 0
        self._period = self._nominal
        self._anchor_rel = 0.0

    @property
    def period_ns(self) -> float:
        return self._period

    @property
    def anchor_ns(self) -> float:
        """Host time of sample index 0 (diagnostic; use time_of() for exact timestamps)."""
        return float(self._t0) + self._anchor_rel - self._period * float(self._k0)

    @property
    def n(self) -> int:
        return self._n

    def reset(self, nominal_period_ns: float | None = None) -> None:
        """Forget everything; optionally change the nominal period."""
        if nominal_period_ns is not None:
            if nominal_period_ns <= 0:
                raise ValueError("nominal_period_ns must be > 0")
            self._nominal = float(nominal_period_ns)
        self._n = 0
        self._pos = 0
        self._period = self._nominal
        self._anchor_rel = 0.0

    def reanchor(self) -> None:
        """Drop the observations (indices restart) but keep the learnt period as the new prior."""
        self._nominal = self._period
        self._n = 0
        self._pos = 0
        self._anchor_rel = 0.0

    def update(self, index: int, t_read_ns: int) -> None:
        """Record that sample `index` (cumulative) was in a FIFO read that completed at t_read_ns."""
        if self._n == 0:
            self._k0 = int(index)
            self._t0 = int(t_read_ns)
        self._k[self._pos] = float(int(index) - self._k0)
        self._t[self._pos] = float(int(t_read_ns) - self._t0)
        self._pos = (self._pos + 1) % self.window
        self._n = min(self._n + 1, self.window)
        k = self._k[: self._n]
        t = self._t[: self._n]
        if self._n >= self.min_points:
            dk = k - k.mean()
            skk = float(dk @ dk)
            if skk > 0.0:
                slope = float(dk @ (t - t.mean())) / skk
                # Reject nonsense (a lost/duplicated block would show up as a wild slope).
                if 0.5 * self._nominal <= slope <= 2.0 * self._nominal:
                    self._period = slope
        self._anchor_rel = float(np.min(t - self._period * k))

    def time_of(self, index):
        """Host timestamp(s) of cumulative sample index/indices (int ns)."""
        idx = np.asarray(index, dtype=np.float64)
        t = float(self._t0) + self._anchor_rel + self._period * (idx - float(self._k0))
        out = np.rint(t).astype(np.int64)
        return int(out) if out.ndim == 0 else out


# ----------------------------------------------------------------------------------- the device
class Mpu6050:
    """MPU-6050 over I2C in FIFO mode, drained by a background thread (ImuSource).

    `bus` injects an SMBus-like object (write_byte_data / read_byte_data / read_i2c_block_data and
    optionally i2c_rdwr); when it is None the bus is opened in start() with smbus2 (fallback smbus).
    """

    def __init__(self, cfg: "Config", calib: "Calibration", bus: Any | None = None):
        imu = cfg.imu
        self.cfg = cfg
        self.address = int(imu.address)
        self.i2c_bus = int(imu.i2c_bus)
        self.divider = sample_rate_div(imu.sample_rate_hz)
        self.sample_rate_hz = GYRO_OUTPUT_RATE_HZ / (self.divider + 1)
        self.dlpf_cfg = dlpf_cfg_for(imu.dlpf_hz)
        self.gyro_fs_sel = gyro_fs_sel(int(imu.gyro_range_dps))
        self.accel_fs_sel = accel_fs_sel(int(imu.accel_range_g))
        self.gyro_lsb_per_dps = GYRO_LSB_PER_DPS[int(imu.gyro_range_dps)]
        self.accel_lsb_per_g = ACCEL_LSB_PER_G[int(imu.accel_range_g)]
        self.poll_hz = float(max(1.0, imu.poll_hz))
        self.calibration = ImuCalibration.from_config(cfg, calib)
        self.clock: Clock = MonotonicClock()
        self.name = f"mpu6050@{self.i2c_bus}:0x{self.address:02x}"

        self.overflows = 0
        self.errors = 0
        self.dropped = 0
        self.sample_count = 0
        self.last_error: str | None = None
        self.temp_c = float("nan")

        self._bus = bus
        self._own_bus = bus is None
        self._i2c_msg = None
        self._regressor = TimestampRegressor(1e9 / self.sample_rate_hz)
        self._index = 0
        self._last_t_ns = 0
        self._next_temp_ns = 0
        self._queue: list[ImuSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ ImuSource
    def start(self) -> None:
        if self._thread is not None:
            return
        if self._bus is None:
            self._bus = self._open_bus()
        self._i2c_msg = self._find_i2c_msg()
        self.configure()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mpu6050", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0 / self.poll_hz + 1.0)
        if self._own_bus and self._bus is not None:
            close = getattr(self._bus, "close", None)
            if close is not None:
                close()
            self._bus = None

    def read(self) -> list[ImuSample]:
        """All samples collected since the last call, time ordered."""
        with self._lock:
            out, self._queue = self._queue, []
        return out

    # ------------------------------------------------------------------ bus plumbing
    def _open_bus(self) -> Any:
        try:
            from smbus2 import SMBus  # type: ignore[import-not-found]
        except ImportError:
            try:
                from smbus import SMBus  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "no I2C library found: `sudo apt install python3-smbus2` on the Pi "
                    "(or `pip install smbus2`)"
                ) from exc
        try:
            return SMBus(self.i2c_bus)
        except OSError as exc:
            raise RuntimeError(
                f"cannot open I2C bus {self.i2c_bus}: {exc}; "
                + I2C_HELP.format(bus=self.i2c_bus, addr=self.address)
            ) from exc

    def _find_i2c_msg(self):
        """smbus2.i2c_msg when the bus supports combined transfers (one transaction per FIFO read)."""
        if not hasattr(self._bus, "i2c_rdwr"):
            return None
        try:
            from smbus2 import i2c_msg  # type: ignore[import-not-found]
        except ImportError:
            return None
        return i2c_msg

    def _write(self, reg: int, value: int) -> None:
        self._bus.write_byte_data(self.address, reg, value)

    def _read_byte(self, reg: int) -> int:
        return int(self._bus.read_byte_data(self.address, reg))

    def _read_block(self, reg: int, length: int) -> bytes:
        return bytes(self._bus.read_i2c_block_data(self.address, reg, length))

    def _read_fifo(self, length: int) -> bytes:
        if self._i2c_msg is not None:
            write = self._i2c_msg.write(self.address, [REG_FIFO_R_W])
            read = self._i2c_msg.read(self.address, length)
            self._bus.i2c_rdwr(write, read)
            return bytes(bytearray(list(read)))
        out = bytearray()
        remaining = length
        while remaining > 0:
            chunk = min(SMBUS_BLOCK_MAX, remaining)
            out += self._read_block(REG_FIFO_R_W, chunk)
            remaining -= chunk
        return bytes(out)

    # ------------------------------------------------------------------ configuration
    def configure(self) -> None:
        """Reset and program the device, then start the FIFO. Raises RuntimeError if it is not there."""
        try:
            self._write(REG_PWR_MGMT_1, PWR1_DEVICE_RESET)
            time.sleep(RESET_WAIT_S)
            self._write(REG_PWR_MGMT_1, PWR1_CLK_PLL_XGYRO)  # PLL with X gyro reference
            time.sleep(WAKE_WAIT_S)
            who = self._read_byte(REG_WHO_AM_I)
        except OSError as exc:
            raise RuntimeError(
                f"MPU-6050 at 0x{self.address:02x} on I2C bus {self.i2c_bus} does not respond: {exc}; "
                + I2C_HELP.format(bus=self.i2c_bus, addr=self.address)
            ) from exc
        if who != WHO_AM_I_VALUE:
            raise RuntimeError(
                f"WHO_AM_I is 0x{who:02x}, expected 0x{WHO_AM_I_VALUE:02x} at address "
                f"0x{self.address:02x} on I2C bus {self.i2c_bus}: this is not an MPU-6050; "
                + I2C_HELP.format(bus=self.i2c_bus, addr=self.address)
            )
        self._write(REG_CONFIG, self.dlpf_cfg)
        self._write(REG_SMPLRT_DIV, self.divider)
        self._write(REG_GYRO_CONFIG, self.gyro_fs_sel << 3)
        self._write(REG_ACCEL_CONFIG, self.accel_fs_sel << 3)
        self._write(REG_FIFO_EN, FIFO_EN_ACCEL_GYRO)
        self._index = 0
        self._last_t_ns = 0
        self._next_temp_ns = 0
        self._regressor.reset(1e9 / self.sample_rate_hz)
        self._reset_fifo()

    def _reset_fifo(self) -> None:
        self._write(REG_USER_CTRL, 0x00)  # FIFO_RESET only acts while FIFO_EN is clear
        self._write(REG_USER_CTRL, USER_CTRL_FIFO_RESET)
        self._write(REG_USER_CTRL, USER_CTRL_FIFO_EN)
        self._index = 0
        self._regressor.reanchor()

    # ------------------------------------------------------------------ reader thread
    def _loop(self) -> None:
        period = 1.0 / self.poll_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self._poll()
            except OSError as exc:  # transient I2C errors must not kill the stream
                self.errors += 1
                self.last_error = str(exc)
            next_t += period
            delay = next_t - time.monotonic()
            if delay <= 0.0:
                next_t = time.monotonic()
            else:
                self._stop.wait(delay)

    def _poll(self) -> None:
        if self._read_byte(REG_INT_STATUS) & INT_STATUS_FIFO_OFLOW:
            # The FIFO wrapped: the sample stream is no longer contiguous, so drop it and re-anchor.
            self.overflows += 1
            self._reset_fifo()
            return
        count = (self._read_byte(REG_FIFO_COUNT_H) << 8) | self._read_byte(REG_FIFO_COUNT_L)
        n = min(count // SAMPLE_BYTES, MAX_SAMPLES_PER_READ)
        if n <= 0:
            self._maybe_read_temperature()
            return
        data = self._read_fifo(n * SAMPLE_BYTES)
        t_read_ns = self.clock.now_ns()
        accel_chip, gyro_chip = parse_fifo(data, self.accel_lsb_per_g, self.gyro_lsb_per_dps)
        n = accel_chip.shape[0]
        if n == 0:
            return
        self._regressor.update(self._index + n - 1, t_read_ns)
        times = self._regressor.time_of(np.arange(self._index, self._index + n, dtype=np.int64))
        times = times + self.calibration.time_offset_ns
        self._index += n
        self.sample_count += n
        accel, gyro = self.calibration.apply(accel_chip, gyro_chip)
        self._maybe_read_temperature()
        temp = self.temp_c
        fresh: list[ImuSample] = []
        last = self._last_t_ns
        for i in range(n):
            t = int(times[i])
            if t <= last:  # a refined fit must never reorder what was already published
                t = last + 1
            last = t
            fresh.append(ImuSample(t_ns=t, gyro=gyro[i], accel=accel[i], temp_c=temp))
        self._last_t_ns = last
        with self._lock:
            self._queue.extend(fresh)
            excess = len(self._queue) - MAX_QUEUED_SAMPLES
            if excess > 0:
                del self._queue[:excess]
                self.dropped += excess

    def _maybe_read_temperature(self) -> None:
        now = self.clock.now_ns()
        if now < self._next_temp_ns:
            return
        self._next_temp_ns = now + int(TEMP_INTERVAL_S * 1e9)
        self.temp_c = parse_temperature(self._read_block(REG_TEMP_OUT_H, 2))


class NullImu:
    """No IMU: a valid, permanently empty sample stream."""

    name = "null-imu"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> list[ImuSample]:
        return []


def is_raspberry_pi() -> bool:
    """True on Raspberry Pi hardware (device-tree model string)."""
    try:
        model = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return False
    return b"Raspberry Pi" in model


def open_imu(cfg: "Config", calib: "Calibration", source: "FrameSource | None") -> ImuSource:
    """The source's own IMU stream, else a real MPU-6050 on a Pi, else NullImu."""
    own = source.imu() if source is not None else None
    if own is not None:
        return own
    if cfg.imu.enabled and is_raspberry_pi():
        return Mpu6050(cfg, calib)
    return NullImu()
