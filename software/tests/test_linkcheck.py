"""Camera link diagnosis (tbot.camera.linkcheck) and the bounded PicamSource.stop()."""

from __future__ import annotations

import subprocess
import threading
import time

import numpy as np

from tbot.camera import linkcheck as lc

PROC = """           CPU0       CPU1       CPU2       CPU3
 17:      12345          0          0          0 ARMCTRL-level   1 Edge      3f00b880.mailbox
 81:        120         30          0          0 ARMCTRL-level  71 Edge      unicam_capture0
IPI0:       100        200        300        400  Rescheduling interrupts
"""


def test_unicam_interrupts_sums_all_cpus(tmp_path):
    p = tmp_path / "interrupts"
    p.write_text(PROC)
    assert lc.unicam_interrupts(str(p)) == 150
    p.write_text(PROC.replace("unicam_capture0", "something_else"))
    assert lc.unicam_interrupts(str(p)) is None
    assert lc.unicam_interrupts(str(tmp_path / "missing")) is None


def test_verdict_names_the_failing_layer():
    assert lc.verdict(True, 400, 50, 1.0).layer == "ok"
    link = lc.verdict(True, 0, 0, 1.5)
    assert link.layer == "link" and "cannot fix" in link.message and "0 packets" in link.message
    assert lc.verdict(None, 0, 0, 1.5).layer == "link"  # sensor unreadable, still no packets
    assert lc.verdict(False, 0, 0, 1.5).layer == "sensor"
    frames = lc.verdict(True, 37, 0, 1.5)
    assert frames.layer == "frames" and "37" in frames.message
    assert lc.verdict(True, None, 0, 1.5).layer == "unknown"


def test_sensor_streaming_parses_i2ctransfer(monkeypatch):
    monkeypatch.setattr(lc, "_sensor_bus_addr", lambda: (10, 0x60))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="0x01\n", stderr="")

    assert lc.sensor_streaming(run) is True
    assert calls[0][-4:] == ["w2@0x60", "0x01", "0x00", "r1"] and calls[0][4:6] == ["-y", "10"]
    assert lc.sensor_streaming(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "0x00", "")) is False
    assert lc.sensor_streaming(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "sudo: a password")) is None
    monkeypatch.setattr(lc, "_sensor_bus_addr", lambda: None)
    assert lc.sensor_streaming(run) is None


def test_black_level_is_set_to_what_the_isp_subtracts(monkeypatch):
    monkeypatch.setattr(lc, "_sensor_bus_addr", lambda: (10, 0x60))

    class Sensor:
        """i2ctransfer against a two-byte register at 0x4002; `stuck` = writes are ignored."""

        def __init__(self, value, stuck=False, rc=0):
            self.value, self.stuck, self.rc, self.writes = value, stuck, rc, 0

        def __call__(self, cmd, **kw):
            args = cmd[cmd.index("-y") + 2:]
            assert args[0] in ("w2@0x60", "w4@0x60") and args[1:3] == ["0x40", "0x02"]
            if self.rc:
                return subprocess.CompletedProcess(cmd, self.rc, "", "sudo: a password is required")
            if args[0] == "w4@0x60":
                self.writes += 1
                if not self.stuck:
                    self.value = (int(args[3], 16) << 8) | int(args[4], 16)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 0, f"0x{self.value >> 8:02x} 0x{self.value & 255:02x}\n", "")

    power_on = Sensor(0x0010)                      # 640x400 straight after power-on
    assert lc.ensure_black_level(power_on) == ("fixed", 16) and power_on.value == 0x0040
    assert lc.ensure_black_level(power_on) == ("ok", 64) and power_on.writes == 1     # nothing written twice
    assert lc.ensure_black_level(Sensor(0x0010, stuck=True)) == ("failed", 16)
    assert lc.ensure_black_level(Sensor(0x0010, rc=1)) == ("unknown", None)           # no passwordless sudo
    monkeypatch.setattr(lc, "_sensor_bus_addr", lambda: None)
    assert lc.ensure_black_level(power_on) == ("unknown", None)                       # not a Pi


def test_diagnose_measures_the_interrupt_delta(monkeypatch):
    counts = iter([1000, 1000])
    monkeypatch.setattr(lc, "unicam_interrupts", lambda: next(counts))
    monkeypatch.setattr(lc, "sensor_streaming", lambda: True)
    d = lc.diagnose(window_s=0.0, sleep=lambda s: None)
    assert d.layer == "link" and d.csi_interrupts == 0


def test_diagnose_prefers_the_start_probe_of_the_source(monkeypatch):
    class Src:
        start_probe = lc.verdict(True, 0, 0, 0.7)

    monkeypatch.setattr(lc, "unicam_interrupts", lambda: (_ for _ in ()).throw(AssertionError("sampled")))
    d = lc.diagnose(source=Src())
    assert d.layer == "link" and d.window_s == 0.7


def test_down_card_is_a_frame_of_the_stream_size():
    img = lc.down_card(640, 400, "CAMERA DOWN (link)", "x " * 200)
    assert img.shape == (400, 640, 3) and img.dtype == np.uint8 and img.max() > 100


class _HungPicam2:
    """picamera2 on a stalled CSI link: stop() never returns."""

    def __init__(self):
        self.release = threading.Event()
        self.closed = False

    def stop(self):
        self.release.wait(30.0)

    def close(self):
        self.closed = True


def test_picam_stop_is_bounded_when_the_camera_hangs(monkeypatch):
    from tbot.camera.picam import PicamSource

    src = PicamSource.__new__(PicamSource)  # skip __init__ (needs picamera2)
    src._stop = threading.Event()
    src._thread = None
    src._camera_closed = False
    src._cond = threading.Condition()
    src._slot = None
    src._picam2 = _HungPicam2()
    monkeypatch.setattr(PicamSource, "STOP_TIMEOUT_S", 0.2)
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 2.0
    src._picam2.release.set()
