"""Pi health decoding, `tbot hwcheck` verdicts and the `tbot jog` bench REPL (no hardware)."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from tbot.config import Calibration, load_calibration, load_config
from tbot.hw import health as health_mod
from tbot.hw.arming import ArmingSwitch
from tbot.hw.gpio import SimGpio, SimPwm
from tbot.tools import hwcheck, jog
from tbot.tools.hwcheck import FAIL, PASS, SKIP, WARN, CheckEnv
from tbot.types import ImuSample

DEBOUNCE_MS = 20.0
G = 9.80665

GOOD_CONFIG_TXT = """\
# tbot block
camera_auto_detect=0
dtoverlay=ov9281
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=400000
dtoverlay=pwm,pin=18,func=2
dtparam=audio=off
enable_uart=1
"""


def make_cfg():
    """Default config on the sim backends with short motion times."""
    cfg = load_config()
    cfg.gpio.backend = "sim"
    cfg.arming.debounce_ms = DEBOUNCE_MS
    cfg.actuator.servo.hold_ms = 30.0
    cfg.actuator.pq12.extend_ms = 40.0
    cfg.actuator.pq12.retract_ms = 30.0
    return cfg


def wait_for(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def arm(gpio: SimGpio, cfg, arming: ArmingSwitch) -> None:
    gpio.inject(cfg.arming.pin, 0)  # NO button to GND, active low
    assert wait_for(lambda: arming.armed), "the switch never armed"


def off_pi_env(tmp_path, **kwargs) -> CheckEnv:
    """A CheckEnv whose every system path points into tmp_path (so a real Pi cannot leak in)."""
    defaults = dict(
        config_txt_paths=(tmp_path / "config.txt",),
        model_path=tmp_path / "no-model",
        os_release_path=tmp_path / "no-os-release",
        i2c_dev_path=str(tmp_path / "i2c-{bus}"),
        gpiochip_path=str(tmp_path / "gpiochip{chip}"),
        pwm_sysfs_root=tmp_path / "pwm",
        health_reader=lambda: {},
        log=lambda msg: None,
    )
    defaults.update(kwargs)
    return CheckEnv(**defaults)


@pytest.fixture(autouse=True)
def _clear_health_cache():
    health_mod.clear_cache()
    yield
    health_mod.clear_cache()


# --------------------------------------------------------------------------------------------- health
def test_parse_throttled_decodes_now_and_since_boot_bits():
    d = health_mod.parse_throttled(0x50005)  # bits 0, 2, 16, 18
    assert d["under_voltage_now"] is True and d["throttled_now"] is True
    assert d["under_voltage_occurred"] is True and d["throttled_occurred"] is True
    assert d["arm_frequency_capped_now"] is False and d["soft_temp_limit_now"] is False
    assert d["arm_frequency_capped_occurred"] is False and d["soft_temp_limit_occurred"] is False
    assert d["now"] == ["under_voltage", "throttled"]
    assert d["occurred"] == ["under_voltage", "throttled"]
    assert d["raw"] == 0x50005 and d["hex"] == "0x50005" and d["ok"] is False

    clean = health_mod.parse_throttled(0)
    assert clean["ok"] is True and clean["now"] == [] and clean["occurred"] == []
    assert health_mod.parse_throttled(0x20000)["arm_frequency_capped_occurred"] is True


def test_read_health_from_a_fake_sysfs_tree(tmp_path):
    temp = tmp_path / "temp"
    temp.write_text("47123\n")
    throttled = tmp_path / "get_throttled"
    throttled.write_text("0x50000\n")  # under-voltage + throttling happened since boot

    h = health_mod.read_health(0.0, temp_path=temp, throttled_path=throttled, vcgencmd=None)
    assert h["cpu_temp_c"] == pytest.approx(47.123)
    assert h["throttled"]["under_voltage_occurred"] is True
    assert h["throttled"]["under_voltage_now"] is False
    assert h["source"] == {"cpu_temp_c": "sysfs", "throttled": "sysfs", "arm_freq_mhz": None}
    assert h["arm_freq_mhz"] is None and h["available"] is True


def test_read_health_without_any_source_returns_none_fields(tmp_path):
    h = health_mod.read_health(0.0, temp_path=tmp_path / "nope", throttled_path=tmp_path / "nope2",
                               vcgencmd=None)
    assert h["cpu_temp_c"] is None and h["throttled"] is None and h["arm_freq_mhz"] is None
    assert h["available"] is False


def test_read_health_is_rate_limited_to_one_refresh_per_second(tmp_path):
    temp = tmp_path / "temp"
    throttled = tmp_path / "get_throttled"
    temp.write_text("40000")
    throttled.write_text("0x0")
    kwargs = dict(temp_path=temp, throttled_path=throttled, vcgencmd=None)

    first = health_mod.read_health(100.0, **kwargs)
    temp.write_text("80000")
    cached = health_mod.read_health(100.5, **kwargs)
    assert cached["cpu_temp_c"] == first["cpu_temp_c"] == pytest.approx(40.0)
    fresh = health_mod.read_health(101.5, **kwargs)
    assert fresh["cpu_temp_c"] == pytest.approx(80.0)


def test_read_health_falls_back_to_vcgencmd(tmp_path):
    def fake_run(args, timeout_s):
        assert args[0] == "vcgencmd" and timeout_s > 0
        if list(args[1:]) == ["get_throttled"]:
            return "throttled=0x1\n"
        if list(args[1:]) == ["measure_clock", "arm"]:
            return "frequency(48)=1200000000\n"
        return None

    h = health_mod.read_health(0.0, temp_path=tmp_path / "nope", throttled_path=tmp_path / "nope2",
                               vcgencmd="vcgencmd", run=fake_run)
    assert h["throttled"]["under_voltage_now"] is True
    assert h["source"]["throttled"] == "vcgencmd"
    assert h["arm_freq_mhz"] == pytest.approx(1200.0)
    assert "under-voltage" in health_mod.describe_throttled(h["throttled"])


# --------------------------------------------------------------------------------------------- boot
def write_config_txt(tmp_path, text: str):
    path = tmp_path / "config.txt"
    path.write_text(text)
    return path


def test_boot_config_passes_when_every_required_setting_is_in_force(tmp_path):
    write_config_txt(tmp_path, GOOD_CONFIG_TXT)
    result = hwcheck.check_boot_config(make_cfg(), None, off_pi_env(tmp_path))
    assert result.status == PASS
    assert result.data["missing"] == []


def test_boot_config_fails_when_an_overlay_is_missing(tmp_path):
    text = GOOD_CONFIG_TXT.replace("dtoverlay=ov9281\n", "")
    write_config_txt(tmp_path, text)
    result = hwcheck.check_boot_config(make_cfg(), None, off_pi_env(tmp_path))
    assert result.status == FAIL
    assert result.data["missing"] == ["dtoverlay=ov9281"]
    assert "dtoverlay=ov9281" in result.hint and "config.txt" in result.hint


def test_boot_config_does_not_count_a_commented_line(tmp_path):
    text = GOOD_CONFIG_TXT.replace("dtparam=audio=off", "#dtparam=audio=off")
    write_config_txt(tmp_path, text)
    result = hwcheck.check_boot_config(make_cfg(), None, off_pi_env(tmp_path))
    assert result.status == FAIL
    assert result.data["missing"] == ["dtparam=audio=off"]
    note = next(r["note"] for r in result.data["required"] if r["setting"] == "dtparam=audio=off")
    assert "commented" in note


def test_boot_config_skips_a_section_this_board_does_not_match(tmp_path):
    model = tmp_path / "model"
    model.write_text("Raspberry Pi 3 Model B Rev 1.2\x00")
    text = GOOD_CONFIG_TXT.replace("dtoverlay=ov9281\n", "[pi5]\ndtoverlay=ov9281\n[all]\n")
    write_config_txt(tmp_path, text)
    result = hwcheck.check_boot_config(make_cfg(), None, off_pi_env(tmp_path, model_path=model))
    assert result.status == FAIL
    assert result.data["missing"] == ["dtoverlay=ov9281"]
    assert hwcheck.section_in_force("pi3", "Raspberry Pi 3 Model B Rev 1.2") is True
    assert hwcheck.section_in_force("none", "Raspberry Pi 3 Model B Rev 1.2") is False
    assert hwcheck.section_in_force("HDMI:1", "Raspberry Pi 3 Model B Rev 1.2") is True


def test_boot_config_only_wants_the_pwm_overlay_for_the_servo(tmp_path):
    cfg = make_cfg()
    cfg.actuator.type = "pq12"
    text = GOOD_CONFIG_TXT.replace("dtoverlay=pwm,pin=18,func=2\n", "").replace("dtparam=audio=off\n", "")
    write_config_txt(tmp_path, text)
    assert hwcheck.check_boot_config(cfg, None, off_pi_env(tmp_path)).status == PASS
    # ... but the servo build needs both of them.
    assert hwcheck.check_boot_config(make_cfg(), None, off_pi_env(tmp_path)).status == FAIL


def test_boot_config_skips_when_there_is_no_config_txt(tmp_path):
    result = hwcheck.check_boot_config(make_cfg(), None, off_pi_env(tmp_path))
    assert result.status == SKIP and "config.txt" in result.summary


# --------------------------------------------------------------------------------------------- power
def fake_health(temp_c=45.0, word=0):
    return {"cpu_temp_c": temp_c, "throttled": health_mod.parse_throttled(word),
            "arm_freq_mhz": 1200.0, "source": {}, "available": True}


@pytest.mark.parametrize("health, expected", [
    (fake_health(), PASS),
    (fake_health(word=0x1), FAIL),              # under-voltage right now
    (fake_health(word=0x10000), WARN),          # under-voltage since boot
    (fake_health(word=0x4), WARN),              # capped right now
    (fake_health(temp_c=75.0), WARN),
    (fake_health(temp_c=85.0), FAIL),
    ({}, SKIP),                                 # no firmware interfaces (laptop)
])
def test_power_check_verdicts(tmp_path, health, expected):
    env = off_pi_env(tmp_path, health_reader=lambda: health)
    assert hwcheck.check_power(make_cfg(), None, env).status == expected


def test_power_check_undervoltage_beats_a_hot_soc(tmp_path):
    env = off_pi_env(tmp_path, health_reader=lambda: fake_health(temp_c=85.0, word=0x1))
    result = hwcheck.check_power(make_cfg(), None, env)
    assert result.status == FAIL and "XL4015" in result.hint


# --------------------------------------------------------------------------------------------- calib
def test_calibration_check_on_an_empty_directory(tmp_path):
    cfg = make_cfg()
    cfg.paths.calib_dir = str(tmp_path / "calib")
    result = hwcheck.check_calibration(cfg, Calibration(dir=tmp_path / "calib"), off_pi_env(tmp_path))
    assert result.status == WARN
    assert "tbot calib-intrinsics" in result.hint
    assert all(not f["exists"] for f in result.data["files"].values())


def test_calibration_check_on_a_populated_directory(tmp_path):
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    (calib_dir / "intrinsics.json").write_text(json.dumps({"created": "2026-01-02T03:04:05+00:00"}))
    (calib_dir / "boresight.json").write_text(json.dumps({"created": "2026-01-02T03:04:06+00:00"}))
    (calib_dir / "latency.json").write_text(json.dumps(
        {"actuator": {"servo": {"cmd_to_break_ms": {"mean": 118.0, "std": 4.0}}}}))
    cfg = make_cfg()
    cfg.paths.calib_dir = str(calib_dir)
    result = hwcheck.check_calibration(cfg, load_calibration(cfg), off_pi_env(tmp_path))
    assert result.status == PASS
    assert result.data["files"]["intrinsics.json"]["created"] == "2026-01-02T03:04:05+00:00"
    assert result.data["files"]["mask.png"]["exists"] is False
    assert result.data["latency_sources"]["actuator.servo"] == "measured"
    assert "actuator.servo" in result.data["measured"]


# --------------------------------------------------------------------------------------------- camera
class FakeFrame:
    def __init__(self, image):
        self.image = image


class FakeSource:
    """Just enough of FrameSource for check_camera: a frame every `period_s`."""

    def __init__(self, cfg, calib, level=120.0, period_s=0.005):
        self.width, self.height = int(cfg.camera.width), int(cfg.camera.height)
        self.level = float(level)
        self.period_s = float(period_s)
        self.dropped = 0
        self.started = False
        self.stopped = False
        self.sensor_mode = {"size": (640, 400), "fps": 120.0}

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self, timeout_s=1.0):
        time.sleep(min(self.period_s, timeout_s))
        return FakeFrame(np.full((8, 8), self.level, dtype=np.uint8))


def pi_env(tmp_path, **kwargs):
    """off_pi_env with the board forced to 'this is a Pi', so the hardware checks actually run."""
    kwargs.setdefault("on_pi", True)
    return off_pi_env(tmp_path, **kwargs)


def camera_env(tmp_path, **kwargs):
    return pi_env(tmp_path, camera_info=lambda: [{"Model": "ov9281", "Id": "fake"}],
                  camera_seconds=0.05, **kwargs)


def test_camera_check_passes_at_the_configured_rate(tmp_path):
    cfg = make_cfg()
    cfg.camera.fps = 50.0
    env = camera_env(tmp_path, camera_factory=lambda c, k: FakeSource(c, k, period_s=0.002))
    result = hwcheck.check_camera(cfg, None, env)
    assert result.status == PASS, result.summary
    assert result.data["frames"] > 0 and result.data["is_ov9281"] is True
    assert result.data["mean_level"] == pytest.approx(120.0)


def test_camera_check_warns_when_frames_are_too_slow(tmp_path):
    cfg = make_cfg()
    cfg.camera.fps = 1000.0  # the fake cannot deliver anywhere near this
    env = camera_env(tmp_path, camera_factory=lambda c, k: FakeSource(c, k, period_s=0.005))
    result = hwcheck.check_camera(cfg, None, env)
    assert result.status == WARN and "camera.fps" in result.hint


def test_camera_check_warns_on_a_black_frame(tmp_path):
    env = camera_env(tmp_path, camera_factory=lambda c, k: FakeSource(c, k, level=2.0, period_s=0.002))
    result = hwcheck.check_camera(make_cfg(), None, env)
    assert result.status == WARN and "exposure" in result.hint


def test_camera_check_fails_when_libcamera_lists_nothing(tmp_path):
    env = pi_env(tmp_path, camera_info=lambda: [])
    result = hwcheck.check_camera(make_cfg(), None, env)
    assert result.status == FAIL and "ov9281" in result.hint


# --------------------------------------------------------------------------------------------- imu
class FakeImu:
    """Produces samples at `rate_hz` of real time, with a chosen accel magnitude."""

    def __init__(self, cfg, calib, rate_hz=None, accel_g=1.0, gyro_dps=0.0):
        self.rate_hz = float(cfg.imu.sample_rate_hz if rate_hz is None else rate_hz)
        self.accel = np.array([0.0, 0.0, float(accel_g) * G])
        self.gyro = np.array([float(gyro_dps) * np.pi / 180.0, 0.0, 0.0])
        self.errors = 0
        self.overflows = 0
        self._t0 = None
        self._produced = 0

    def start(self):
        self._t0 = time.monotonic()

    def stop(self):
        pass

    def read(self):
        if self._t0 is None:
            return []
        want = int((time.monotonic() - self._t0) * self.rate_hz)
        n = max(0, want - self._produced)
        self._produced = want
        now = time.monotonic_ns()
        return [ImuSample(t_ns=now + i, gyro=self.gyro, accel=self.accel, temp_c=31.0)
                for i in range(n)]


def imu_env(tmp_path, factory, **kwargs):
    (tmp_path / "i2c-1").write_text("")
    return pi_env(tmp_path, imu_factory=factory, imu_seconds=0.2, **kwargs)


def test_imu_check_passes_on_a_still_sensor(tmp_path):
    cfg = make_cfg()
    env = imu_env(tmp_path, lambda c, k: FakeImu(c, k))
    result = hwcheck.check_imu(cfg, None, env)
    assert result.status == PASS, result.summary
    assert result.data["samples"] > 0
    assert result.data["accel_norm_m_s2"] == pytest.approx(G, rel=1e-6)
    assert result.data["temp_c"] == pytest.approx(31.0)


def test_imu_check_warns_when_gravity_is_wrong(tmp_path):
    env = imu_env(tmp_path, lambda c, k: FakeImu(c, k, accel_g=0.5))
    result = hwcheck.check_imu(make_cfg(), None, env)
    assert result.status == WARN and "calib-imu" in result.hint


def test_imu_check_warns_when_the_gun_is_moving(tmp_path):
    env = imu_env(tmp_path, lambda c, k: FakeImu(c, k, gyro_dps=20.0))
    result = hwcheck.check_imu(make_cfg(), None, env)
    assert result.status == WARN and "still" in result.hint


def test_imu_check_warns_on_a_low_sample_rate(tmp_path):
    cfg = make_cfg()
    env = imu_env(tmp_path, lambda c, k: FakeImu(c, k, rate_hz=cfg.imu.sample_rate_hz * 0.3))
    result = hwcheck.check_imu(cfg, None, env)
    assert result.status == WARN and "i2c_arm_baudrate" in result.hint


def test_imu_check_fails_when_the_device_does_not_answer(tmp_path):
    class DeadImu(FakeImu):
        def start(self):
            raise RuntimeError("WHO_AM_I is 0x00, expected 0x68\nsecond line of help")

    env = imu_env(tmp_path, lambda c, k: DeadImu(c, k))
    result = hwcheck.check_imu(make_cfg(), None, env)
    assert result.status == FAIL
    assert result.summary == "WHO_AM_I is 0x00, expected 0x68"  # only the first line in the table
    assert "i2cdetect" in result.hint


def test_imu_check_skips_off_a_pi(tmp_path):
    env = off_pi_env(tmp_path)  # no /dev/i2c-1 under tmp_path
    assert hwcheck.check_imu(make_cfg(), None, env).status == SKIP


# --------------------------------------------------------------------------------------------- gpio
class FakeGpio:
    """A non-sim Gpio stand-in so the GPIO/actuator checks can be exercised off a Pi."""

    def __init__(self, levels=None):
        self.levels = dict(levels or {})
        self.modes: dict[int, str] = {}
        self.writes: list[tuple[int, int]] = []
        self.closed = False

    def setup_output(self, pin, level=0):
        self.modes[int(pin)] = "out"
        self.write(pin, level)

    def write(self, pin, level):
        self.levels[int(pin)] = int(bool(level))
        self.writes.append((int(pin), int(bool(level))))

    def setup_input(self, pin, pull="up"):
        self.modes[int(pin)] = "in"
        self.levels.setdefault(int(pin), 1)

    def read(self, pin):
        return int(self.levels.get(int(pin), 0))

    def watch(self, pin, edge, callback):
        pass

    def close(self):
        self.closed = True


def gpio_env(tmp_path, cfg, gpio, **kwargs):
    (tmp_path / f"gpiochip{cfg.gpio.chip}").write_text("")
    return pi_env(tmp_path, gpio_factory=lambda c: gpio, lgpio_available=lambda: True, **kwargs)


def test_gpio_check_reports_a_released_button(tmp_path):
    cfg = make_cfg()
    gpio = FakeGpio({cfg.arming.pin: 1})
    result = hwcheck.check_gpio(cfg, None, gpio_env(tmp_path, cfg, gpio))
    assert result.status == PASS
    assert result.data["pressed"] is False and gpio.closed is True


def test_gpio_check_warns_while_the_button_is_held(tmp_path):
    cfg = make_cfg()
    gpio = FakeGpio({cfg.arming.pin: 0})
    result = hwcheck.check_gpio(cfg, None, gpio_env(tmp_path, cfg, gpio))
    assert result.status == WARN and "release the arming button" in result.hint


def test_gpio_check_interactive_verifies_both_levels(tmp_path):
    cfg = make_cfg()
    gpio = FakeGpio({cfg.arming.pin: 1})
    env = gpio_env(tmp_path, cfg, gpio, interactive=True, interactive_timeout_s=1.0)

    def operator():
        time.sleep(0.05)
        gpio.levels[cfg.arming.pin] = 0  # press
        time.sleep(0.05)
        gpio.levels[cfg.arming.pin] = 1  # release

    thread = threading.Thread(target=operator, daemon=True)
    thread.start()
    result = hwcheck.check_gpio(cfg, None, env)
    thread.join(2.0)
    assert result.status == PASS and result.data["interactive"] == "press and release detected"


def test_gpio_check_interactive_fails_when_nobody_presses(tmp_path):
    cfg = make_cfg()
    gpio = FakeGpio({cfg.arming.pin: 1})
    env = gpio_env(tmp_path, cfg, gpio, interactive=True, interactive_timeout_s=0.1)
    result = hwcheck.check_gpio(cfg, None, env)
    assert result.status == FAIL and "no press detected" in result.summary


def test_gpio_check_skips_on_the_sim_backend(tmp_path):
    cfg = make_cfg()
    result = hwcheck.check_gpio(cfg, None, gpio_env(tmp_path, cfg, SimGpio()))
    assert result.status == SKIP


def test_gpio_check_skips_off_a_pi(tmp_path):
    assert hwcheck.check_gpio(make_cfg(), None, off_pi_env(tmp_path)).status == SKIP


# --------------------------------------------------------------------------------------------- actuator
def test_actuator_check_reports_an_exportable_pwm_channel(tmp_path):
    chip = tmp_path / "pwm" / "pwmchip0"
    chip.mkdir(parents=True)
    (chip / "export").write_text("")
    result = hwcheck.check_actuator(make_cfg(), None, pi_env(tmp_path))
    assert result.status == PASS and "not moved by hwcheck" in result.summary
    assert result.data["exported"] is False


def test_actuator_check_reports_an_exported_channel(tmp_path):
    channel = tmp_path / "pwm" / "pwmchip0" / "pwm0"
    channel.mkdir(parents=True)
    (channel / "duty_cycle").write_text("0")
    result = hwcheck.check_actuator(make_cfg(), None, pi_env(tmp_path))
    assert result.status == PASS and result.data["exported"] is True


def test_actuator_check_skips_without_a_pwm_chip_off_a_pi(tmp_path):
    assert hwcheck.check_actuator(make_cfg(), None, off_pi_env(tmp_path)).status == SKIP


def test_actuator_check_drives_both_pq12_pins_low(tmp_path):
    cfg = make_cfg()
    cfg.actuator.type = "pq12"
    gpio = FakeGpio()
    result = hwcheck.check_actuator(cfg, None, gpio_env(tmp_path, cfg, gpio))
    assert result.status == PASS
    assert result.data["levels"] == [0, 0]
    assert gpio.levels[cfg.actuator.pq12.pin_extend] == 0
    assert gpio.levels[cfg.actuator.pq12.pin_retract] == 0


def test_actuator_check_skips_for_a_software_actuator(tmp_path):
    cfg = make_cfg()
    cfg.actuator.type = "sim"
    assert hwcheck.check_actuator(cfg, None, off_pi_env(tmp_path)).status == SKIP


# --------------------------------------------------------------------------------------------- main
def test_hwcheck_main_json_runs_off_a_pi(tmp_path, capsys):
    write_config_txt(tmp_path, GOOD_CONFIG_TXT)
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    env = off_pi_env(tmp_path)
    code = hwcheck.main(["--json", "--only", "boot,power,calib",
                         "--config-txt", str(tmp_path / "config.txt"),
                         "--calib-dir", str(calib_dir)], env=env)
    report = json.loads(capsys.readouterr().out)
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses == {"boot": PASS, "power": SKIP, "calib": WARN}
    assert report["status"] == WARN and code == 0
    assert all({"name", "status", "summary", "hint", "data"} <= set(c) for c in report["checks"])


def test_hwcheck_main_exits_1_on_a_failing_check(tmp_path, capsys):
    write_config_txt(tmp_path, GOOD_CONFIG_TXT.replace("dtparam=i2c_arm=on\n", ""))
    code = hwcheck.main(["--only", "boot", "--config-txt", str(tmp_path / "config.txt")],
                        env=off_pi_env(tmp_path))
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out and "dtparam=i2c_arm=on" in out
    assert "verdict: FAIL" in out


def test_hwcheck_main_rejects_an_unknown_check(tmp_path):
    with pytest.raises(SystemExit):
        hwcheck.main(["--only", "nosuchcheck"], env=off_pi_env(tmp_path))


# --------------------------------------------------------------------------------------------- jog
def build_servo_jog(cfg=None):
    cfg = cfg or make_cfg()
    gpio = SimGpio()
    arming = ArmingSwitch(gpio, cfg)
    pwm = SimPwm(cfg.actuator.servo.pwm_chip, cfg.actuator.servo.pwm_channel, cfg.actuator.servo.frame_hz)
    log: list[str] = []
    repl = jog.ServoJog(cfg, gpio, arming, pwm, log=log.append)
    return cfg, gpio, arming, pwm, repl, log


def build_pq12_jog(cfg=None):
    cfg = cfg or make_cfg()
    cfg.actuator.type = "pq12"
    gpio = SimGpio()
    arming = ArmingSwitch(gpio, cfg)
    log: list[str] = []
    repl = jog.Pq12Jog(cfg, gpio, arming, log=log.append)
    return cfg, gpio, arming, repl, log


def assert_never_both_high(gpio: SimGpio, pin_a: int, pin_b: int) -> None:
    level = {pin_a: 0, pin_b: 0}
    for _t, pin, value in gpio.writes:
        if pin in level:
            level[pin] = value
        assert not (level[pin_a] and level[pin_b]), f"H-bridge shoot-through after writing {pin}={value}"


def _jog_driven(pwm) -> list[int]:
    return [round(us) for _t, us in pwm.pulses if us]


def _with_travel(cfg, lo=1300.0, hi=1700.0):
    cfg.actuator.servo.min_us, cfg.actuator.servo.max_us = lo, hi
    return cfg


def test_servo_jog_starts_limp_and_never_moves_while_the_button_is_released():
    """Start-up must not drive rest_us: an unverified rest pulse can sit stalled against the trigger."""
    cfg, gpio, arming, pwm, repl, log = build_servo_jog()
    assert pwm.pulse_us is None and _jog_driven(pwm) == []
    try:
        repl.run_commands(["1520", "+", "-", "test", str(round(cfg.actuator.servo.rest_us))])
    finally:
        repl.close()
    assert _jog_driven(pwm) == []  # not even the rest pulse
    assert pwm.pulse_us is None
    assert any("refused" in line for line in log)
    arming.close()
    gpio.close()


def test_servo_jog_creeps_until_the_travel_limits_are_marked():
    """The lever's valid travel is small: before lo / hi exist, no move may jump more than max_jump_us."""
    cfg, gpio, arming, pwm, repl, log = build_servo_jog()
    jump = cfg.actuator.servo.max_jump_us
    try:
        arm(gpio, cfg, arming)
        repl.execute("1670")  # the old default rest guess: far from centre, refused
        assert _jog_driven(pwm) == []
        repl.execute(str(round(jog.NEUTRAL_US + jump)))  # within one jump of centre: allowed
        repl.execute("+")
        repl.execute("+25")
        repl.execute(str(round(jog.NEUTRAL_US + jump + 10 + 25 + jump + 5)))  # one jump + 5 from last: refused
        assert _jog_driven(pwm) == [round(jog.NEUTRAL_US + jump), round(jog.NEUTRAL_US + jump + 10),
                                    round(jog.NEUTRAL_US + jump + 35)]
        assert pwm.pulse_us is None  # every move was a burst
        repl.execute("hi")
        repl.execute("save")  # only one end marked: refused
        assert any("before saving" in line for line in log)
    finally:
        repl.close()
        arming.close()
        gpio.close()


def test_servo_jog_moves_in_short_bursts_while_armed():
    cfg, gpio, arming, pwm, repl, log = build_servo_jog()
    try:
        arm(gpio, cfg, arming)
        repl.execute("1520")
        assert _jog_driven(pwm) == [1520] and pwm.pulse_us is None  # drove, then let go
        repl.execute("+")
        assert _jog_driven(pwm)[-1] == round(1520.0 + jog.DEFAULT_STEP_US) and pwm.pulse_us is None
        repl.execute("fire")
        assert repl.fire_us == pytest.approx(1520.0 + jog.DEFAULT_STEP_US)

        gpio.inject(cfg.arming.pin, 1)  # operator lets go
        assert pwm.pulse_us is None
        assert any(jog.SAFE_MESSAGE in line for line in log)
        n = len(_jog_driven(pwm))
        repl.execute("1520")  # nothing can move again until it is held once more
        assert len(_jog_driven(pwm)) == n
    finally:
        repl.close()
        arming.close()
        gpio.close()


def test_servo_jog_refuses_moves_outside_the_calibrated_travel():
    cfg, gpio, arming, pwm, repl, log = build_servo_jog(_with_travel(make_cfg(), 1400.0, 1600.0))
    try:
        arm(gpio, cfg, arming)
        repl.execute("1400")
        repl.execute("1600")
        repl.execute("1610")
        repl.execute("1390")
        repl.execute("-")  # from 1600
        assert _jog_driven(pwm) == [1400, 1600, 1590]
        assert sum("outside the calibrated travel" in line for line in log) == 2
    finally:
        repl.close()
        arming.close()
        gpio.close()


def test_servo_jog_clamps_pulses_to_the_servo_range():
    cfg, gpio, arming, pwm, repl, log = build_servo_jog(_with_travel(make_cfg(), 500.0, 2500.0))
    try:
        arm(gpio, cfg, arming)
        repl.execute("400")
        repl.execute("3000")
        repl.execute("-100")
        assert _jog_driven(pwm) == [round(jog.MIN_PULSE_US), round(jog.MAX_PULSE_US),
                                    round(jog.MAX_PULSE_US - 100.0)]
    finally:
        repl.close()
        arming.close()
        gpio.close()


def test_servo_jog_test_cycle_fires_returns_and_goes_limp():
    cfg, gpio, arming, pwm, repl, log = build_servo_jog(_with_travel(make_cfg()))
    cfg.actuator.servo.hold_ms = 30.0
    cfg.actuator.servo.return_ms = 30.0
    try:
        arm(gpio, cfg, arming)
        repl.run_commands(["1500", "rest", "1400", "fire", "test"])
        driven = _jog_driven(pwm)
        assert driven[-2:] == [1400, 1500], driven  # fire burst, then rest burst
        assert pwm.pulse_us is None
        assert any("test done" in line for line in log)
    finally:
        repl.close()
        arming.close()
        gpio.close()


def test_servo_jog_test_and_save_refuse_without_travel_limits():
    cfg, gpio, arming, pwm, repl, log = build_servo_jog()
    try:
        arm(gpio, cfg, arming)
        repl.run_commands(["1500", "rest", "1460", "fire", "test", "save"])
        assert _jog_driven(pwm) == [1500, 1460]  # the two marking moves only: no fire cycle
        assert any("test needs the travel limits" in line for line in log)
        assert any("before saving" in line for line in log)
    finally:
        repl.close()
        arming.close()
        gpio.close()


def test_servo_jog_save_round_trips_through_local_toml(monkeypatch, tmp_path):
    monkeypatch.setenv("TBOT_LOCAL_CONFIG", str(tmp_path / "local.toml"))
    cfg, gpio, arming, pwm, repl, log = build_servo_jog()
    try:
        arm(gpio, cfg, arming)
        # creep out to each end of the travel, mark it, then mark rest / fire inside it
        repl.run_commands(["1540", "hi", "1500", "rest", "1460", "fire", "1450", "lo", "save"])
    finally:
        repl.close()
        arming.close()
        gpio.close()
    assert (tmp_path / "local.toml").exists()
    assert any("saved" in line for line in log)
    reloaded = load_config()
    assert reloaded.actuator.servo.rest_us == pytest.approx(1500.0)
    assert reloaded.actuator.servo.fire_us == pytest.approx(1460.0)
    assert reloaded.actuator.servo.min_us == pytest.approx(1450.0)
    assert reloaded.actuator.servo.max_us == pytest.approx(1540.0)


def test_pq12_jog_refuses_to_extend_while_the_button_is_released():
    cfg, gpio, arming, repl, log = build_pq12_jog()
    pin_e = cfg.actuator.pq12.pin_extend
    try:
        repl.execute("e 20")
    finally:
        repl.close()
    assert all(not (pin == pin_e and level == 1) for _t, pin, level in gpio.writes)
    assert any("refused" in line for line in log)
    arming.close()
    gpio.close()


def test_pq12_jog_release_aborts_the_extend_and_never_shoots_through():
    cfg, gpio, arming, repl, log = build_pq12_jog()
    pin_e, pin_r = cfg.actuator.pq12.pin_extend, cfg.actuator.pq12.pin_retract
    try:
        arm(gpio, cfg, arming)
        release = threading.Timer(0.02, lambda: gpio.inject(cfg.arming.pin, 1))
        release.daemon = True
        release.start()
        repl.execute("e 200")  # returns early: the release aborts it
        release.join(1.0)
        assert any(jog.SAFE_MESSAGE in line for line in log)
        assert wait_for(lambda: gpio.read(pin_r) == 0, 1.0), "the retract pulse never stopped"
        repl.execute("r 20")
        repl.execute("q")
    finally:
        repl.close()
    assert gpio.read(pin_e) == 0 and gpio.read(pin_r) == 0
    assert_never_both_high(gpio, pin_e, pin_r)
    arming.close()
    gpio.close()


def test_pq12_jog_test_cycle_and_swap_persist(monkeypatch, tmp_path):
    monkeypatch.setenv("TBOT_LOCAL_CONFIG", str(tmp_path / "local.toml"))
    cfg, gpio, arming, repl, log = build_pq12_jog()
    pin_e, pin_r = cfg.actuator.pq12.pin_extend, cfg.actuator.pq12.pin_retract
    try:
        arm(gpio, cfg, arming)
        repl.execute("test")
        assert any("test done" in line for line in log)
        repl.execute("swap")
        repl.execute("save")
    finally:
        repl.close()
        arming.close()
        gpio.close()
    assert_never_both_high(gpio, pin_e, pin_r)
    reloaded = load_config()
    assert reloaded.actuator.pq12.pin_extend == pin_r
    assert reloaded.actuator.pq12.pin_retract == pin_e


def test_jog_main_runs_a_command_list_on_the_sim_backend(capsys):
    code = jog.main(["servo", "--sim", "--sim-armed", "--commands", "show;1540;+;show;q",
                     "--set", "arming.debounce_ms=5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "UNLOADED" in out
    assert "pulse 1540 us" in out
    assert "PWM off: servo limp" in out


def test_jog_main_refuses_an_actuator_it_cannot_jog(capsys):
    code = jog.main(["--set", "actuator.type=sim"])
    assert code == 2
    assert "tbot jog servo" in capsys.readouterr().err


# ------------------------------------------------------------------ tbot find-break
def test_find_break_stops_at_the_first_knock_and_shifts_the_whole_calibration(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from tbot.tools import find_break as fb

    class Pwm:
        def __init__(self):
            self.us = []

        def set_pulse_us(self, us):
            self.us.append(float(us))

    class Imu:
        """Quiet until the lever is at or beyond 1203 us, then one 1 g knock."""

        def __init__(self, pwm):
            self.pwm, self.knocked = pwm, False

        def read(self):
            g = fb.G
            quiet = SimpleNamespace(accel=np.array([0.0, 0.0, g]))
            if not self.knocked and self.pwm.us and self.pwm.us[-1] <= 1203.0 and len(self.pwm.us) > 1:
                self.knocked = True
                return [quiet, SimpleNamespace(accel=np.array([0.0, 0.0, 2.0 * g])), quiet]
            return [quiet, quiet]

    pwm = Pwm()
    found, heard = fb.sweep(pwm, Imu(pwm), 1500.0, 1075.0, 8.0, 0.0, 0.4, sleep=lambda s: None, log=lambda m: None)
    assert found == 1500.0 - 8.0 * 38            # 1196: the first step at or past 1203
    assert pwm.us[-1] == found and min(pwm.us) == found   # it stopped there: nothing further was commanded
    assert all(loud < 0.4 for _, loud in heard[:-1]) and heard[-1][1] >= 0.4

    servo = SimpleNamespace(rest_us=1340.0, fire_us=1075.0, min_us=1070.0, max_us=1500.0, break_us=1095.0)
    new = fb.shifted(servo, found)
    assert new == {"rest_us": 1441.0, "fire_us": 1176.0, "min_us": 1171.0, "break_us": 1196.0}
    with pytest.raises(ValueError):
        fb.shifted(servo, 1300.0)                 # rest would land beyond max_us

    local = tmp_path / "local.toml"
    local.write_text("[actuator.servo]\nrest_us = 1340.0   # old\nfire_us = 1075.0\nmin_us = 1070.0\nmax_us = 1500.0\n"
                     "break_us = 1095.0\npreload_frac = 0.6\n")
    fb.apply_to_local(local, new, "test")
    text = local.read_text()
    assert "rest_us = 1441.0" in text and "break_us = 1196.0" in text and "preload_frac = 0.6" in text
    assert "max_us = 1500.0" in text and (tmp_path / "local.toml.bak-find-break").exists()
    assert fb.main([]) == 2                       # refused without --unloaded


def test_find_break_reports_nothing_when_the_hammer_is_down():
    from types import SimpleNamespace

    import numpy as np

    from tbot.tools import find_break as fb

    class Pwm:
        us = None

        def set_pulse_us(self, us):
            self.us = us

    quiet = SimpleNamespace(accel=np.array([0.0, 0.0, fb.G]))
    imu = SimpleNamespace(read=lambda: [quiet, quiet])
    pwm = Pwm()
    found, heard = fb.sweep(pwm, imu, 1500.0, 1075.0, 8.0, 0.0, 0.4, sleep=lambda s: None, log=lambda m: None)
    assert found is None and pwm.us >= 1075.0 and len(heard) == 53
