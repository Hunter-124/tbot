"""Auto-exposure controller (tbot.camera.autoexposure) and camera-mode helpers (tbot.tools.snap)."""

from __future__ import annotations

import numpy as np
import pytest

from tbot.camera.autoexposure import AeSettings, AutoExposure, meter
from tbot.tools import snap

S = AeSettings(target_dn=110, highlight_dn=235, min_exposure_us=20, max_exposure_us=500, min_gain=1, max_gain=8,
               every=1)


def scene(total: float, reflect: np.ndarray) -> np.ndarray:
    """A linear sensor: DN = reflectance * exposure*gain, clipped."""
    return np.clip(reflect * total, 0, 255).astype(np.uint8)


def test_meter_median_and_highlight():
    img = np.full((40, 40), 100, np.uint8)
    img[:2, :] = 240  # 5 % bright rows
    med, hi = meter(img, step=1)
    assert med == 100 and hi == 240


def test_split_fills_exposure_before_gain():
    ae = AutoExposure(S, 100, 1)
    assert ae.split(300) == (300, 1)
    assert ae.split(2000) == (500, 4)
    assert ae.split(1e9) == (500, 8)
    assert ae.split(1) == (20, 1)


@pytest.mark.parametrize("light", [0.01, 0.1, 1.0, 5.0])
def test_closed_loop_converges_without_clipping_the_sky(light):
    rng = np.random.default_rng(0)
    reflect = np.full((100, 160), 0.5 * light)
    reflect[:30] = 1.0 * light  # bright sky at the top
    reflect += rng.normal(0, 0.01 * light, reflect.shape)
    ae = AutoExposure(S, 100, 1)
    exp, gain = ae.exposure_us, ae.gain
    for _ in range(30):
        new = ae.update(scene(exp * gain, reflect), exp, gain)
        if new is not None:
            exp, gain = new
    med, hi = meter(scene(exp * gain, reflect))
    if light >= 0.1:
        assert hi <= 240 and (med >= 100 or hi >= 215)  # on target, or held back by the sky
        assert exp <= 500 and gain >= 1
    else:  # too dark even at the limits: pinned at max exposure and max gain
        assert (exp, gain) == (500, 8)


def test_clipped_frame_steps_down_hard_and_deadband_holds():
    ae = AutoExposure(S, 500, 4)
    assert ae.correction(200, 255) <= 0.5
    assert ae.correction(110, 200) == 1.0
    assert ae.correction(60, 120) == pytest.approx(110 / 60)


def test_bad_limits_rejected():
    with pytest.raises(ValueError):
        AutoExposure(AeSettings(min_exposure_us=600, max_exposure_us=500), 100, 1)


class _F:
    def __init__(self, image):
        self.image = image


def test_snap_numbering_sharpest_and_durable_png(tmp_path):
    assert snap.next_index(tmp_path) == 1
    (tmp_path / "snap_0007.png").write_bytes(b"x")
    (tmp_path / "other.png").write_bytes(b"x")
    assert snap.next_index(tmp_path) == 8
    rng = np.random.default_rng(1)
    sharp = (rng.random((50, 50)) > 0.5).astype(np.uint8) * 255
    blurred = np.full((50, 50), 128, np.uint8)
    i, score = snap.pick_sharpest([_F(blurred), _F(sharp), _F(blurred)])
    assert i == 1 and score > 0
    p = tmp_path / "snap_0008.png"
    snap.save_png(p, sharp)
    import cv2
    assert np.array_equal(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), sharp)
    assert not list(tmp_path.glob("*.part"))
