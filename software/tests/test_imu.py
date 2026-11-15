"""Tests for the [imu] slice: MPU-6050 driver, attitude estimator, camera-IMU time sync."""

from __future__ import annotations

import math
import struct
import time

import numpy as np
import pytest

from tbot.config import Calibration, load_config
from tbot.imu.attitude import AttitudeEstimator, FixedAttitude, exp_so3, log_so3
from tbot.imu.mpu6050 import (
    ACCEL_LSB_PER_G,
    G_MPS2,
    GYRO_LSB_PER_DPS,
    ImuCalibration,
    Mpu6050,
    NullImu,
    TimestampRegressor,
    accel_fs_sel,
    dlpf_cfg_for,
    gyro_fs_sel,
    open_imu,
    parse_fifo,
    sample_rate_div,
)
from tbot.tools.calib_imu import estimate_time_offset_s, kabsch
from tbot.types import ImuSample

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


# --------------------------------------------------------------------------------- fake hardware
class FakeMpu:
    """Register file + FIFO of an MPU-6050 whose samples accrue in real time (SMBus-like API)."""

    FIFO_CAPACITY = 1024

    def __init__(self, sample_rate_hz=500.0, who_am_i=0x68,
                 accel_raw=(0, 0, 4096), gyro_raw=(100, -200, 300), temp_raw=-3920):
        self.regs: dict[int, int] = {}
        self.writes: list[tuple[int, int]] = []
        self.fifo = bytearray()
        self.rate = float(sample_rate_hz)
        self.who_am_i = who_am_i
        self.temp_raw = temp_raw
        self.sample = struct.pack(">6h", *accel_raw, *gyro_raw)
        self.enabled = False
        self.t0: float | None = None
        self.generated = 0
        self.overflow = False
        self.over_reads = 0
        self.closed = False

    # -- device behaviour
    def _generate(self):
        if not self.enabled or self.t0 is None:
            return
        target = int((time.monotonic() - self.t0) * self.rate)
        for _ in range(target - self.generated):
            if len(self.fifo) + len(self.sample) > self.FIFO_CAPACITY:
                self.overflow = True
                break
            self.fifo += self.sample
        self.generated = max(self.generated, target)

    def write_byte_data(self, addr, reg, value):
        self.writes.append((reg, value))
        self.regs[reg] = value
        if reg == REG_USER_CTRL:
            if value & 0x04:
                self.fifo.clear()
                self.generated = 0
                self.t0 = None
                self.overflow = False
            self.enabled = bool(value & 0x40)
            if self.enabled and self.t0 is None:
                self.t0 = time.monotonic()
                self.generated = 0

    def read_byte_data(self, addr, reg):
        if reg == REG_WHO_AM_I:
            return self.who_am_i
        if reg == REG_INT_STATUS:
            self._generate()
            flag = 0x10 if self.overflow else 0x00
            self.overflow = False  # INT_STATUS is cleared by reading it
            return flag
        if reg == REG_FIFO_COUNT_H:
            self._generate()
            return (len(self.fifo) >> 8) & 0xFF
        if reg == REG_FIFO_COUNT_L:
            return len(self.fifo) & 0xFF
        return self.regs.get(reg, 0)

    def read_i2c_block_data(self, addr, reg, length):
        assert length <= 32, "SMBus block reads are limited to 32 bytes"
        if reg == REG_FIFO_R_W:
            out = self.fifo[:length]
            del self.fifo[:length]
            if len(out) < length:
                self.over_reads += 1
                out = out + bytearray(length - len(out))
            return list(out)
        if reg == REG_TEMP_OUT_H:
            return list(struct.pack(">h", self.temp_raw))
        return [0] * length

    def close(self):
        self.closed = True


def make_cfg(**sets):
    return load_config(sets=[f"{k}={v}" for k, v in sets.items()])


def calib(tmp_path, imu=None):
    return Calibration(dir=tmp_path, imu=imu)


# --------------------------------------------------------------------------------- pure parsing
@pytest.mark.parametrize("accel_g, gyro_dps", [(2, 250), (4, 500), (8, 1000), (16, 2000)])
def test_parse_fifo_scales_each_range(accel_g, gyro_dps):
    raw = (1234, -5678, 4096, -100, 250, 32767)
    accel, gyro = parse_fifo(struct.pack(">6h", *raw), ACCEL_LSB_PER_G[accel_g], GYRO_LSB_PER_DPS[gyro_dps])
    assert accel.shape == (1, 3) and gyro.shape == (1, 3)
    expect_a = np.array(raw[:3], dtype=np.float64) * G_MPS2 / ACCEL_LSB_PER_G[accel_g]
    expect_g = np.radians(np.array(raw[3:], dtype=np.float64) / GYRO_LSB_PER_DPS[gyro_dps])
    assert np.allclose(accel[0], expect_a)
    assert np.allclose(gyro[0], expect_g)
    # 1 g on chip +Z at the configured range is always 9.80665 m/s^2.
    one_g = struct.pack(">6h", 0, 0, int(ACCEL_LSB_PER_G[accel_g]), 0, 0, 0)
    a, _ = parse_fifo(one_g, ACCEL_LSB_PER_G[accel_g], GYRO_LSB_PER_DPS[gyro_dps])
    assert a[0, 2] == pytest.approx(G_MPS2, abs=1e-9)


def test_parse_fifo_multiple_samples_and_partial_tail():
    blocks = [struct.pack(">6h", i, i + 1, i + 2, i + 3, i + 4, i + 5) for i in range(4)]
    data = b"".join(blocks) + b"\x01\x02\x03"  # trailing partial sample is dropped
    accel, gyro = parse_fifo(data, ACCEL_LSB_PER_G[8], GYRO_LSB_PER_DPS[1000])
    assert accel.shape == (4, 3)
    assert accel[3, 0] == pytest.approx(3 * G_MPS2 / ACCEL_LSB_PER_G[8])
    assert gyro[0, 2] == pytest.approx(math.radians(5 / GYRO_LSB_PER_DPS[1000]))
    empty_a, empty_g = parse_fifo(b"", 4096.0, 32.8)
    assert empty_a.shape == (0, 3) and empty_g.shape == (0, 3)


def test_register_field_helpers():
    assert dlpf_cfg_for(98) == 2 and dlpf_cfg_for(188) == 1 and dlpf_cfg_for(5) == 6
    assert dlpf_cfg_for(50) == 3  # nearest of 42 / 98
    assert [gyro_fs_sel(r) for r in (250, 500, 1000, 2000)] == [0, 1, 2, 3]
    assert [accel_fs_sel(r) for r in (2, 4, 8, 16)] == [0, 1, 2, 3]
    assert sample_rate_div(500) == 1 and sample_rate_div(1000) == 0 and sample_rate_div(100) == 9
    assert sample_rate_div(1) == 255  # clamped to the register width
    with pytest.raises(ValueError):
        gyro_fs_sel(123)
    with pytest.raises(ValueError):
        accel_fs_sel(3)


def test_calibration_maps_chip_axes_to_the_gun_frame(tmp_path):
    """R_gun_imu from the config maps chip axes to gun axes (as-built: identity, layout.md "As-built IMU");
    a non-trivial map is checked with a rotated config."""
    cfg = make_cfg()
    cal = ImuCalibration.from_config(cfg, calib(tmp_path))
    accel = np.array([[0.0, 0.0, G_MPS2]])  # level gun: specific force along chip +Z
    gyro = np.array([[1.0, 2.0, 3.0]])
    a_gun, g_gun = cal.apply(accel, gyro)
    assert np.allclose(a_gun[0], [0.0, 0.0, G_MPS2])
    assert np.allclose(g_gun[0], [1.0, 2.0, 3.0])
    # the old pod_cap mount: chip +X -> gun -Z, chip +Y -> gun -Y, chip +Z -> gun -X
    cfg.imu.R_gun_imu = [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]
    cal = ImuCalibration.from_config(cfg, calib(tmp_path))
    a_gun, g_gun = cal.apply(accel, gyro)
    assert np.allclose(a_gun[0], [-G_MPS2, 0.0, 0.0])
    assert np.allclose(g_gun[0], [-3.0, -2.0, -1.0])


def test_calibration_applies_bias_scale_then_rotation(tmp_path):
    cfg = make_cfg()
    cal = ImuCalibration.from_config(cfg, calib(tmp_path, imu={
        "gyro_bias_rad_s": [0.01, -0.02, 0.03],
        "accel_bias_m_s2": [0.5, -0.5, 1.0],
        "accel_scale": [1.01, 0.99, 1.02],
        "time_offset_ns": -1234,
    }))
    assert cal.time_offset_ns == -1234
    accel = np.array([[1.0, 2.0, 3.0]])
    gyro = np.array([[0.11, 0.0, -0.07]])
    a_gun, g_gun = cal.apply(accel, gyro)
    a_chip = (accel[0] - np.array([0.5, -0.5, 1.0])) * np.array([1.01, 0.99, 1.02])
    g_chip = gyro[0] - np.array([0.01, -0.02, 0.03])
    R = np.array(cfg.imu.R_gun_imu, dtype=np.float64)
    assert np.allclose(a_gun[0], R @ a_chip)
    assert np.allclose(g_gun[0], R @ g_chip)


def test_calibration_rejects_a_non_rotation(tmp_path):
    cfg = make_cfg()
    with pytest.raises(ValueError):
        ImuCalibration.from_config(cfg, calib(tmp_path, imu={"R_gun_imu": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}))
    with pytest.raises(ValueError):
        ImuCalibration.from_config(cfg, calib(tmp_path, imu={"gyro_bias_rad_s": [0.0, 1.0]}))


# --------------------------------------------------------------------------------- timestamping
def test_timestamp_regression_recovers_a_fast_oscillator():
    """A 1.5 % fast MPU clock must be recovered to 0.1 % from jittered, latency-delayed reads."""
    rng = np.random.default_rng(4)
    nominal_ns = 2_000_000.0  # 500 Hz
    true_period = nominal_ns / 1.015
    reg = TimestampRegressor(nominal_ns)
    t0 = 1_000_000_000
    index = 0
    for read in range(300):
        # Read every ~10 ms; the block ends on whatever sample is complete, plus I2C/scheduler latency.
        t_read = t0 + read * 10_000_000 + int(rng.uniform(0, 900_000))
        last = int((t_read - t0) / true_period)
        if last < index:
            continue
        index = last
        reg.update(index, t_read)
    assert reg.period_ns == pytest.approx(true_period, rel=1e-3)
    # The anchor is the lower envelope: predicted sample times never run ahead of the read that
    # carried them, and stay within one period behind.
    t_pred = reg.time_of(index)
    assert -2.0 * true_period < t_pred - (t0 + index * true_period) < 2.0 * true_period


def test_timestamp_regressor_before_any_fit_and_after_reanchor():
    reg = TimestampRegressor(2_000_000.0)
    reg.update(9, 1_000_000_000)
    assert reg.period_ns == 2_000_000.0
    assert reg.time_of(9) == 1_000_000_000
    assert reg.time_of(10) == 1_002_000_000
    for k in range(1, 50):
        reg.update(9 + 5 * k, 1_000_000_000 + int(k * 5 * 2_100_000))
    assert reg.period_ns == pytest.approx(2_100_000.0, rel=1e-6)
    reg.reanchor()
    assert reg.n == 0
    assert reg.period_ns == pytest.approx(2_100_000.0, rel=1e-6)  # period survives, indices restart
    reg.update(0, 5_000_000_000)
    assert reg.time_of(0) == 5_000_000_000


# --------------------------------------------------------------------------------- driver
def test_driver_init_sequence_and_sample_stream(tmp_path):
    cfg = make_cfg(**{"imu.sample_rate_hz": 500, "imu.poll_hz": 200})
    bus = FakeMpu(sample_rate_hz=500.0)
    imu = Mpu6050(cfg, calib(tmp_path), bus=bus)
    assert imu.read() == []
    imu.start()
    try:
        time.sleep(0.4)
        samples = imu.read()
    finally:
        imu.stop()

    assert bus.writes[:10] == [
        (REG_PWR_MGMT_1, 0x80),      # device reset
        (REG_PWR_MGMT_1, 0x01),      # PLL with X gyro reference
        (REG_CONFIG, 2),             # DLPF 98 Hz
        (REG_SMPLRT_DIV, 1),         # 1000 / (1 + 1) = 500 Hz
        (REG_GYRO_CONFIG, 2 << 3),   # +/-1000 deg/s
        (REG_ACCEL_CONFIG, 2 << 3),  # +/-8 g
        (REG_FIFO_EN, 0x78),         # accel + gyro xyz
        (REG_USER_CTRL, 0x00),
        (REG_USER_CTRL, 0x04),       # FIFO reset
        (REG_USER_CTRL, 0x40),       # FIFO enable
    ]
    assert bus.over_reads == 0
    assert imu.errors == 0
    assert len(samples) > 80  # 0.4 s at 500 Hz, minus whatever a loaded scheduler costs
    t = np.array([s.t_ns for s in samples], dtype=np.int64)
    assert np.all(np.diff(t) > 0)
    assert float(np.median(np.diff(t))) == pytest.approx(2_000_000.0, rel=0.1)
    assert 0 < int(t[-1] - t[0]) <= 500_000_000
    # Chip +Z reads +1 g (4096 LSB at +/-8 g); as-built R_gun_imu is the identity (layout.md "As-built IMU").
    assert np.allclose(samples[0].accel, [0.0, 0.0, G_MPS2], atol=1e-9)
    chip_gyro = np.radians(np.array([100.0, -200.0, 300.0]) / GYRO_LSB_PER_DPS[1000])
    assert np.allclose(samples[0].gyro, chip_gyro, atol=1e-12)
    assert samples[0].temp_c == pytest.approx(25.0, abs=0.1)
    assert bus.closed is False  # an injected bus is not ours to close


def test_driver_recovers_from_a_fifo_overflow(tmp_path):
    cfg = make_cfg(**{"imu.sample_rate_hz": 500, "imu.poll_hz": 100})
    bus = FakeMpu(sample_rate_hz=500.0)
    imu = Mpu6050(cfg, calib(tmp_path), bus=bus)
    imu.start()
    try:
        time.sleep(0.1)
        before = len(imu.read())
        bus.overflow = True
        time.sleep(0.15)
        imu.read()
        time.sleep(0.15)
        after = imu.read()
    finally:
        imu.stop()
    assert before > 0
    assert imu.overflows >= 1
    reset_writes = [w for w in bus.writes if w == (REG_USER_CTRL, 0x04)]
    assert len(reset_writes) >= 2  # the init reset plus the overflow recovery
    assert len(after) > 10, "the stream must resume after an overflow"
    t = np.array([s.t_ns for s in after], dtype=np.int64)
    assert np.all(np.diff(t) > 0)


def test_driver_rejects_a_foreign_chip(tmp_path):
    cfg = make_cfg()
    bus = FakeMpu(who_am_i=0x71)
    imu = Mpu6050(cfg, calib(tmp_path), bus=bus)
    with pytest.raises(RuntimeError) as err:
        imu.start()
    message = str(err.value)
    assert "0x71" in message and "i2cdetect" in message and "dtparam=i2c_arm=on" in message


def test_null_imu_and_open_imu(tmp_path):
    cfg = make_cfg(**{"imu.enabled": "false"})
    null = NullImu()
    null.start()
    assert null.read() == []
    null.stop()

    class Source:
        def __init__(self, own):
            self._own = own

        def imu(self):
            return self._own

    own = NullImu()
    assert open_imu(cfg, calib(tmp_path), Source(own)) is own
    # No source IMU and imu.enabled false -> NullImu even on a Pi.
    assert isinstance(open_imu(cfg, calib(tmp_path), Source(None)), NullImu)
    assert isinstance(open_imu(cfg, calib(tmp_path), None), NullImu)


# --------------------------------------------------------------------------------- attitude
def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def still_sample(t_ns: int, R_wg: np.ndarray) -> ImuSample:
    """A still gun at attitude R_wg measures gravity's specific force in the gun frame."""
    return ImuSample(t_ns=t_ns, gyro=np.zeros(3), accel=R_wg.T @ np.array([0.0, 0.0, G_MPS2]))


def test_attitude_integrates_a_constant_yaw_rate():
    cfg = make_cfg(**{"attitude.init_still_s": 0.0, "imu.sample_rate_hz": 500})
    est = AttitudeEstimator(cfg)
    assert est.ready is False
    assert np.allclose(est.R_wg(0), np.eye(3))
    t0 = 1_000_000_000
    est.update([still_sample(t0, np.eye(3))])
    assert est.ready is True
    rate = math.radians(90.0)
    g = np.array([0.0, 0.0, G_MPS2])
    est.update([ImuSample(t_ns=t0 + i * 2_000_000, gyro=np.array([0.0, 0.0, rate]), accel=g)
                for i in range(1, 501)])
    R = est.R_wg(t0 + 1_000_000_000)
    yaw_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    assert yaw_deg == pytest.approx(90.0, abs=0.2)
    bore_w = R @ np.array([1.0, 0.0, 0.0])
    assert abs(math.degrees(math.asin(bore_w[2]))) < 0.05  # yaw must not leak into elevation
    assert np.allclose(est.omega_g(t0 + 1_000_000_000), [0.0, 0.0, rate])


def test_attitude_gravity_init_with_the_muzzle_pitched_up():
    cfg = make_cfg(**{"attitude.init_still_s": 0.3, "imu.sample_rate_hz": 500})
    est = AttitudeEstimator(cfg)
    R_true = rot_y(-math.radians(20.0))  # muzzle up 20 deg (columns = gun axes in world)
    t0 = 1_000_000_000
    samples = [still_sample(t0 + i * 2_000_000, R_true) for i in range(151)]
    est.update(samples[:50])
    assert est.ready is False, "initialisation must wait for the full still window"
    est.update(samples[50:])
    assert est.ready is True
    R = est.R_wg(samples[-1].t_ns)
    bore_w = R @ np.array([1.0, 0.0, 0.0])
    assert math.degrees(math.asin(bore_w[2])) == pytest.approx(20.0, abs=0.2)
    assert abs(math.degrees(math.atan2(bore_w[1], bore_w[0]))) < 0.2  # heading 0 by definition
    assert np.allclose(R, R_true, atol=2e-3)


def test_attitude_accel_correction_bounds_gyro_bias_drift():
    cfg = make_cfg(**{"attitude.init_still_s": 0.0, "imu.sample_rate_hz": 200})
    rate_hz, seconds = 200, 30.0
    bias = math.radians(0.5)  # deg/s of pitch bias
    t0 = 1_000_000_000
    dt_ns = int(1e9 / rate_hz)
    g = np.array([0.0, 0.0, G_MPS2])

    est = AttitudeEstimator(cfg)
    est.update([still_sample(t0, np.eye(3))])
    est.update([ImuSample(t_ns=t0 + i * dt_ns, gyro=np.array([0.0, bias, 0.0]), accel=g)
                for i in range(1, int(rate_hz * seconds) + 1)])
    up_g = est.R_wg(t0 + int(seconds * 1e9))[2]  # world up expressed in the gun frame
    tilt_deg = math.degrees(math.acos(min(1.0, abs(float(up_g @ np.array([0.0, 0.0, 1.0]))))))
    assert tilt_deg < 1.0
    assert est.accel_updates > 1000

    # Without the accelerometer (out of gate) the same bias drifts the full 15 deg: the correction
    # is what keeps the estimate honest, not luck.
    free = AttitudeEstimator(cfg)
    free.update([still_sample(t0, np.eye(3))])
    free.update([ImuSample(t_ns=t0 + i * dt_ns, gyro=np.array([0.0, bias, 0.0]), accel=np.zeros(3))
                 for i in range(1, int(rate_hz * seconds) + 1)])
    up_free = free.R_wg(t0 + int(seconds * 1e9))[2]
    drift_deg = math.degrees(math.acos(min(1.0, abs(float(up_free @ np.array([0.0, 0.0, 1.0]))))))
    assert drift_deg == pytest.approx(0.5 * seconds, abs=0.2)


def test_attitude_interpolates_and_extrapolates_analytically():
    cfg = make_cfg(**{"attitude.init_still_s": 0.0, "imu.sample_rate_hz": 500})
    est = AttitudeEstimator(cfg)
    t0 = 1_000_000_000
    est.update([still_sample(t0, np.eye(3))])
    omega = np.array([0.3, -0.7, 1.1])
    dt_ns = 2_000_000
    # accel = 0 keeps the complementary correction out of the way: pure gyro integration.
    est.update([ImuSample(t_ns=t0 + i * dt_ns, gyro=omega, accel=np.zeros(3)) for i in range(1, 101)])

    def analytic(t_ns: int) -> np.ndarray:
        return exp_so3(omega * (t_ns - t0) * 1e-9)

    assert np.allclose(est.R_wg(t0 + 100 * dt_ns), analytic(t0 + 100 * dt_ns), atol=1e-9)
    between = t0 + 60 * dt_ns + int(0.37 * dt_ns)
    assert np.allclose(est.R_wg(between), analytic(between), atol=1e-9)
    beyond = t0 + 100 * dt_ns + 3_700_000
    assert np.allclose(est.R_wg(beyond), analytic(beyond), atol=1e-9)
    assert np.allclose(est.omega_g(beyond), omega)
    # History is bounded by attitude.history_s.
    assert est.history_len <= est.max_history


def test_attitude_handles_duplicate_and_reordered_samples():
    cfg = make_cfg(**{"attitude.init_still_s": 0.0, "imu.sample_rate_hz": 500})
    est = AttitudeEstimator(cfg)
    t0 = 1_000_000_000
    est.update([still_sample(t0, np.eye(3))])
    omega = np.array([0.0, 0.0, 1.0])
    good = [ImuSample(t_ns=t0 + i * 2_000_000, gyro=omega, accel=np.zeros(3)) for i in range(1, 11)]
    est.update(good + [good[3], good[0]])  # stale repeats must not rewind or double-integrate
    assert np.allclose(est.R_wg(good[-1].t_ns), exp_so3(omega * 20e-3), atol=1e-9)


def test_rotation_helpers_round_trip():
    rng = np.random.default_rng(11)
    for _ in range(20):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0.0, math.pi - 1e-3)
        R = exp_so3(v)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(log_so3(R), v, atol=1e-9)
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3))
    assert np.allclose(log_so3(np.eye(3)), np.zeros(3))
    axis = np.array([0.0, 1.0, 0.0])
    assert np.linalg.norm(log_so3(exp_so3(axis * (math.pi - 1e-9)))) == pytest.approx(math.pi, abs=1e-6)


def test_fixed_attitude_is_the_identity_world():
    fixed = FixedAttitude()
    assert fixed.ready is True
    assert np.allclose(fixed.R_wg(123), np.eye(3))
    assert np.allclose(fixed.omega_g(123), np.zeros(3))
    fixed.update([])


# --------------------------------------------------------------------------------- time sync
def _swing(t: np.ndarray) -> np.ndarray:
    """Band-limited three-axis rate signal (deterministic)."""
    rng = np.random.default_rng(3)
    freqs = rng.uniform(0.4, 3.0, size=(3, 4))
    phases = rng.uniform(0.0, 2 * math.pi, size=(3, 4))
    amps = rng.uniform(0.3, 2.0, size=(3, 4))
    out = np.zeros((t.size, 3))
    for ax in range(3):
        for k in range(4):
            out[:, ax] += amps[ax, k] * np.sin(2 * math.pi * freqs[ax, k] * t + phases[ax, k])
    return out


@pytest.mark.parametrize("truth_s, step_ms", [(0.023, 1.0), (-0.0073, 4.0)])
def test_time_offset_estimator_recovers_a_known_offset(truth_s, step_ms):
    rng = np.random.default_rng(17)
    cam_t = np.arange(0.0, 4.0, 1.0 / 120.0)
    imu_t = np.arange(-0.3, 4.3, 1.0 / 500.0)
    cam_rates = _swing(cam_t) + rng.normal(0.0, 0.01, size=(cam_t.size, 3))
    imu_rates = _swing(imu_t + truth_s)  # the IMU clock needs +truth_s to line up with the camera
    offset, score = estimate_time_offset_s(cam_t, cam_rates, imu_t, imu_rates,
                                           lag_step_s=step_ms / 1000.0)
    assert offset == pytest.approx(truth_s, abs=0.002)
    assert score > 0.9


def test_time_offset_estimator_rejects_useless_input():
    t = np.arange(0.0, 2.0, 0.01)
    rates = np.zeros((t.size, 3))
    with pytest.raises(ValueError):
        estimate_time_offset_s(t, rates, t, rates)  # no signal to correlate
    with pytest.raises(ValueError):
        estimate_time_offset_s(t[:3], _swing(t[:3]), t, _swing(t))  # too few camera samples


def test_kabsch_recovers_a_known_rotation():
    rng = np.random.default_rng(5)
    R_true = exp_so3(np.array([0.05, -0.11, 0.3]))
    a = rng.normal(size=(40, 3))
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = a @ R_true.T
    R = kabsch(a, b)
    assert np.allclose(R, R_true, atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    noisy = b + rng.normal(0.0, 1e-3, size=b.shape)
    assert np.linalg.norm(log_so3(kabsch(a, noisy) @ R_true.T)) < 5e-3
    with pytest.raises(ValueError):
        kabsch(a[:1], b[:1])


def test_attitude_alpha_recovers_a_constant_angular_acceleration():
    """alpha_g is the gyro slope over the last 30 ms: the solver extrapolates the swing with it."""
    cfg = make_cfg(**{"attitude.init_still_s": 0.0, "imu.sample_rate_hz": 500})
    est = AttitudeEstimator(cfg)
    t0 = 1_000_000_000
    assert np.allclose(est.alpha_g(t0), 0.0)  # no history yet
    est.update([still_sample(t0, np.eye(3))])
    alpha = np.array([1.5, -8.0, 4.0])  # rad/s^2, a brisk swing
    g = np.array([0.0, 0.0, G_MPS2])
    rng = np.random.default_rng(3)
    est.update([ImuSample(t_ns=t0 + i * 2_000_000, gyro=alpha * (i * 0.002) + rng.normal(0.0, 1e-3, 3), accel=g)
                for i in range(1, 101)])
    assert est.alpha_g(t0 + 200_000_000) == pytest.approx(alpha, abs=0.3)
    # a constant rate has no acceleration
    est2 = AttitudeEstimator(cfg)
    est2.update([still_sample(t0, np.eye(3))])
    est2.update([ImuSample(t_ns=t0 + i * 2_000_000, gyro=np.array([0.0, 0.0, 2.0]), accel=g) for i in range(1, 51)])
    assert np.allclose(est2.alpha_g(t0 + 100_000_000), 0.0, atol=1e-9)
