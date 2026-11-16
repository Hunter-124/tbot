"""`tbot calib-boresight --laser`: automatic bore-laser dot finding, wobble removal, --check."""

from __future__ import annotations

import builtins
import json
import math

import cv2
import numpy as np
import pytest

from tbot.config import load_config
from tbot.optics import CameraModel, MountModel
from tbot.tools import calib_boresight as cb
from tbot.tools import laser_dot
from tbot.types import Frame

W, H = 640, 400


def _scene(seed: int = 0) -> np.ndarray:
    """A dim room: gradient, noise, a dark barrel across the top."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = 40.0 + 20.0 * x / W + 10.0 * y / H + rng.normal(0.0, 1.5, (H, W)).astype(np.float32)
    return img


def _add_dot(img: np.ndarray, u: float, v: float, peak: float = 150.0, sigma: float = 1.3) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return img + peak * np.exp(-((x - u) ** 2 + (y - v) ** 2) / (2.0 * sigma * sigma))


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _true_pixel(cfg, rpy_deg, range_m: float) -> tuple[float, float]:
    cam = CameraModel.from_calibration(None, cfg, W, H)
    mount = MountModel.from_config(cfg, {"rpy_deg": list(rpy_deg), "poi_offset_rad": [0.0, 0.0]})
    return mount.bore_pixel(cam, range_m)


# ------------------------------------------------------------------------------------ dot finder
def test_find_dot_is_subpixel_accurate():
    for u, v in ((300.3, 210.7), (120.0, 333.25), (511.6, 250.1)):
        d = laser_dot.find_dot(_u8(_add_dot(_scene(), u, v)))
        assert (d.u, d.v) == pytest.approx((u, v), abs=0.15)
        assert d.peak_dn > 60 and not d.saturated


def test_find_dot_saturated_dot_keeps_its_centre():
    d = laser_dot.find_dot(_u8(_add_dot(_scene(), 250.4, 180.6, peak=900.0, sigma=2.5)))
    assert d.saturated
    assert (d.u, d.v) == pytest.approx((250.4, 180.6), abs=0.3)


def test_find_dot_refuses_an_empty_scene_and_an_ambiguous_one():
    with pytest.raises(laser_dot.LaserDotError, match="no laser dot"):
        laser_dot.find_dot(_u8(_scene()))
    two = _add_dot(_add_dot(_scene(), 200.0, 200.0), 420.0, 150.0, peak=120.0)
    with pytest.raises(laser_dot.LaserDotError, match="ambiguous"):
        laser_dot.find_dot(_u8(two))


def test_find_dot_refuses_glare():
    glare = _add_dot(_scene(), 300.0, 200.0, peak=400.0, sigma=40.0)
    with pytest.raises(laser_dot.LaserDotError, match="glare"):
        laser_dot.find_dot(_u8(glare))


def test_background_frame_removes_a_lamp():
    lamp = _add_dot(_scene(), 420.0, 150.0, peak=200.0, sigma=3.0)
    with_dot = _add_dot(lamp, 200.0, 220.0, peak=80.0)
    d = laser_dot.find_dot(_u8(with_dot), background=_u8(lamp))
    assert (d.u, d.v) == pytest.approx((200.0, 220.0), abs=0.2)


def test_mask_hides_a_reflection_on_the_barrel():
    img = _add_dot(_add_dot(_scene(), 320.0, 30.0, peak=200.0), 300.0, 215.0, peak=100.0)
    mask = np.full((H, W), 255, np.uint8)
    mask[:80] = 0
    d = laser_dot.find_dot(_u8(img), mask=mask)
    assert (d.u, d.v) == pytest.approx((300.0, 215.0), abs=0.2)


def test_measure_dot_takes_the_median_and_flags_motion():
    frames = [_u8(_add_dot(_scene(k), 300.0, 200.0)) for k in range(9)]
    frames[4] = _u8(_add_dot(_scene(4), 330.0, 200.0))  # one outlier
    m = laser_dot.measure_dot(frames)
    assert (m.u, m.v) == pytest.approx((300.0, 200.0), abs=0.15)
    moving = [_u8(_add_dot(_scene(k), 300.0 + 2.0 * k, 200.0)) for k in range(9)]
    assert any("not still" in w for w in laser_dot.measure_dot(moving).warnings)


def test_circle_centre_from_uneven_turns():
    c = np.array([310.0, 205.0])
    ang = np.radians([10.0, 80.0, 200.0, 260.0])  # not evenly spaced
    pts = c + 4.0 * np.column_stack([np.cos(ang), np.sin(ang)])
    centre, radius = laser_dot.circle_centre(pts)
    assert centre == pytest.approx(c, abs=1e-6)
    assert radius == pytest.approx(4.0, abs=1e-6)
    # two points: the midpoint
    centre2, _ = laser_dot.circle_centre(np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert centre2 == pytest.approx([1.0, 0.0])
    # collinear points must not throw the centre far away
    centre3, _ = laser_dot.circle_centre(np.array([[0.0, 0.0], [1.0, 0.001], [2.0, 0.0]]))
    assert np.linalg.norm(centre3 - [1.0, 0.0]) < 1.0


# ------------------------------------------------------------------------------------ the tool
def test_laser_image_recovers_a_known_mount_rotation(tmp_path):
    cfg = load_config()
    true_rpy = [0.0, -1.2, -3.5]
    u, v = _true_pixel(cfg, true_rpy, 15.0)
    still = tmp_path / "dot.png"
    cv2.imwrite(str(still), _u8(_add_dot(_scene(), u, v)))
    rc = cb.main(["--calib-dir", str(tmp_path), "--laser", "--image", str(still), "--range-m", "15"])
    assert rc == 0
    doc = json.loads((tmp_path / "boresight.json").read_text())
    assert doc["method"] == "laser"
    assert doc["rpy_deg"][1] == pytest.approx(true_rpy[1], abs=0.03)
    assert doc["rpy_deg"][2] == pytest.approx(true_rpy[2], abs=0.03)
    assert doc["laser"]["turns"] == 1
    assert (tmp_path / "boresight_laser.png").exists()


def test_laser_needs_a_range_and_reports_a_missing_dot(tmp_path, capsys):
    still = tmp_path / "empty.png"
    cv2.imwrite(str(still), _u8(_scene()))
    assert cb.main(["--calib-dir", str(tmp_path), "--laser", "--image", str(still)]) == 2
    assert cb.main(["--calib-dir", str(tmp_path), "--laser", "--image", str(still), "--range-m", "15"]) == 1
    assert "no laser dot" in capsys.readouterr().err
    assert not (tmp_path / "boresight.json").exists()


class _TurnSource:
    """Live-source stand-in: the dot sits at dots[turn]; _next_turn() is what pressing Enter does."""

    width, height, is_color, nominal_fps, name, dropped = W, H, False, 50.0, "fake", 0

    def __init__(self, dots, blocked_first: bool = False):
        self.dots = list(dots)
        self.turn = 0
        self.blocked = blocked_first
        self.seq = 0
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def read(self, timeout_s: float = 1.0):
        self.seq += 1
        img = _scene(self.seq)
        img = _add_dot(img, 500.0, 300.0, peak=180.0, sigma=3.0)  # a lamp, always on
        if not self.blocked:
            img = _add_dot(img, *self.dots[min(self.turn, len(self.dots) - 1)], peak=110.0)
        return Frame(image=_u8(img), t_exposure_ns=self.seq, t_capture_ns=self.seq, seq=self.seq,
                     exposure_us=500.0, gain=1.0)


def _frame_fields_ok() -> bool:
    try:
        Frame(image=np.zeros((1, 1), np.uint8), t_exposure_ns=0, t_capture_ns=0, seq=0, exposure_us=1.0, gain=1.0)
        return True
    except TypeError:
        return False


@pytest.mark.skipif(not _frame_fields_ok(), reason="Frame signature changed: update _TurnSource")
def test_laser_turns_with_background_removes_wobble_and_the_lamp(tmp_path, monkeypatch, capsys):
    cfg = load_config()
    true_rpy = [0.0, -1.7, -4.0]
    u, v = _true_pixel(cfg, true_rpy, 20.0)
    ang = np.radians([0.0, 90.0, 180.0, 270.0])
    dots = [(u + 3.0 * math.cos(a), v + 3.0 * math.sin(a)) for a in ang]  # laser 3 px off the bore axis
    src = _TurnSource(dots, blocked_first=False)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if "UNBLOCK" in prompt:
            src.blocked = False
        elif "BLOCK" in prompt:
            src.blocked = True
        else:
            src.turn += 1
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)
    import tbot.camera

    monkeypatch.setattr(tbot.camera, "open_source", lambda spec, cfg, calib: src)
    rc = cb.main(["--calib-dir", str(tmp_path), "--laser", "--laser-turns", "4", "--laser-bg",
                  "--laser-frames", "5", "--range-m", "20"])
    assert rc == 0, capsys.readouterr()
    assert src.stopped and len(prompts) == 5
    doc = json.loads((tmp_path / "boresight.json").read_text())
    assert doc["rpy_deg"][1] == pytest.approx(true_rpy[1], abs=0.03)
    assert doc["rpy_deg"][2] == pytest.approx(true_rpy[2], abs=0.03)
    assert doc["laser"]["wobble_px"] == pytest.approx(3.0, abs=0.2)
    assert doc["laser"]["background"] is True
    # without the background frame the lamp is brighter than the dot and the tool must refuse, not guess
    src2 = _TurnSource(dots)
    monkeypatch.setattr(tbot.camera, "open_source", lambda spec, cfg, calib: src2)
    out2 = tmp_path / "other"
    out2.mkdir()
    assert cb.main(["--calib-dir", str(out2), "--laser", "--laser-frames", "3", "--range-m", "20"]) == 1
    assert src2.stopped and not (out2 / "boresight.json").exists()


def test_check_reports_the_error_and_writes_nothing(tmp_path, capsys):
    cfg = load_config()
    true_rpy = [0.0, -1.0, 2.0]
    still = tmp_path / "dot.png"
    cv2.imwrite(str(still), _u8(_add_dot(_scene(), *_true_pixel(cfg, true_rpy, 10.0))))
    assert cb.main(["--calib-dir", str(tmp_path), "--laser", "--image", str(still), "--range-m", "10"]) == 0
    before = (tmp_path / "boresight.json").read_text()
    # the same mount seen at 25 m: parallax moves the dot, the calibration must still agree
    cv2.imwrite(str(still), _u8(_add_dot(_scene(1), *_true_pixel(cfg, true_rpy, 25.0))))
    capsys.readouterr()
    assert cb.main(["--calib-dir", str(tmp_path), "--laser", "--image", str(still), "--range-m", "25",
                    "--check"]) == 0
    assert "PASS" in capsys.readouterr().out
    # a camera knocked 0.5 deg in yaw is caught
    cv2.imwrite(str(still), _u8(_add_dot(_scene(2), *_true_pixel(cfg, [0.0, -1.0, 2.5], 25.0))))
    assert cb.main(["--calib-dir", str(tmp_path), "--laser", "--image", str(still), "--range-m", "25",
                    "--check"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "0.5" in out
    assert (tmp_path / "boresight.json").read_text() == before


def test_recalibrating_keeps_one_step_of_undo(tmp_path):
    assert cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", "300,200"]) == 0
    first = json.loads((tmp_path / "boresight.json").read_text())
    assert cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", "310,205"]) == 0
    prev = json.loads((tmp_path / "boresight.prev.json").read_text())
    assert prev["rpy_deg"] == first["rpy_deg"]
