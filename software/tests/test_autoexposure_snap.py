"""Auto-exposure controller (tbot.camera.autoexposure) on a simulated camera, and camera-mode helpers (tbot.tools.snap).

The simulated camera has what was measured on the rig: commands act 7 frames later, exposure time comes in
9.5 us lines made up by digital gain, the image is tone-curved (libcamera's OV9281 gamma) and clips at 255."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tbot.camera.autoexposure import AeSettings, AutoExposure, Meter, OV9281_GAMMA, ToneCurve, meter
from tbot.camera.picam import ae_limits, tone_curve_for
from tbot.tools import snap

S = AeSettings()
CURVE = ToneCurve()
LINE_US = 9.5
SKY = 2e-3      # luminance of an open sky in simulator units: wants ~125 us at gain 1
FPS = 50.0


class SimCam:
    """scene luminance -> 8-bit frames through exposure quantisation, noise, the tone curve and clipping;
    a command reaches the sensor `delay` frames after it is given."""

    def __init__(self, settings=S, exposure_us=125.0, gain=1.0, delay=7, fps=FPS, seed=0):
        self.ae = AutoExposure(settings, exposure_us, gain, CURVE)
        self.dt = 1.0 / fps
        self.pipe = [(self.ae.exposure_us, self.ae.gain)] * (delay + 1)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.rows: list[SimpleNamespace] = []

    def frame(self, lum):
        e_req, g = self.pipe[0]
        e = max(1, int(e_req / LINE_US)) * LINE_US
        dgain = max(e_req / e, 1.0)
        lin = lum * e * g * dgain
        electrons = np.clip(lin, 0, dgain) / (g * dgain) * 7000.0   # ~7000 e- full well at gain 1
        lin = lin + self.rng.normal(0, 1, lin.shape) * np.sqrt(electrons + 9.0) / 7000.0 * g * dgain
        dn = np.interp(np.clip(lin, 0, 1), CURVE._x, CURVE._y) * 255.0
        return np.clip(np.rint(dn), 0, 255).astype(np.uint8), e, g * dgain

    def run(self, scene, seconds):
        """scene(t) -> luminance array. Appends one row per frame and returns the rows of this call."""
        first = len(self.rows)
        for _ in range(int(round(seconds / self.dt))):
            img, e, g = self.frame(scene(self.t))
            new = self.ae.update(img, e, g, self.t)
            self.pipe = self.pipe[1:] + [(self.ae.exposure_us, self.ae.gain)]
            m = meter(img, CURVE)
            self.rows.append(SimpleNamespace(t=self.t, exposure_us=e, gain=g, level=m.level_dn, clip=m.clip_frac,
                                             state=self.ae.status.state, command=new is not None))
            self.t += self.dt
        return self.rows[first:]


def sky(level=SKY, trees=0.0, shape=(100, 160)):
    """A sky with a 30 % vertical gradient; the bottom `trees` fraction of the rows is a tree line 12x darker."""
    h, w = shape
    lum = level * (1.0 - 0.3 * np.linspace(0, 1, h))[:, None] * np.ones((1, w))
    n = int(round(trees * h))
    if n:
        lum[h - n:] = level * 0.08 * (0.6 + 0.8 * np.random.default_rng(1).random((n, w)))
    return lum


def blind(rows):
    """Frames with more than the allowed 2 % at white."""
    return sum(r.clip > 0.02 for r in rows)


def commands(rows):
    return sum(r.command for r in rows)


def on_target(row):
    return abs(math.log(CURVE.linear_of(row.level) / CURVE.linear_of(S.target_dn))) < math.log(1.2)


# ------------------------------------------------------------------------------------ tone curve / metering
def test_tone_curve_matches_the_measured_sensor():
    # exposure sweep on the rig (video range, black 16 / white 235): median 21 DN at gain 1, 75 DN at gain 15.875
    c = ToneCurve(OV9281_GAMMA, 16, 235)
    assert c.linear_of(75) / c.linear_of(21) == pytest.approx(15.875, rel=0.06)
    full = ToneCurve()
    assert full.linear_of(0) == 0 and full.linear_of(255) == 1
    assert full.linear_of(128) == pytest.approx(0.18, abs=0.01)      # mid grey
    assert full.dn_of(full.linear_of(160.0)) == pytest.approx(160.0, abs=0.01)
    assert ToneCurve.linear().linear_of(51) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        ToneCurve((0, 0, 1, 1, 2, 1))


def test_meter_reads_highlights_ignores_small_glints_and_counts_clipping():
    img = np.full((80, 160), 60, np.uint8)
    img[:40] = 150                       # half the frame is sky
    img[50:54, 20:24] = 255              # a glint: one 4x4 block of 800
    m = meter(img, CURVE)
    assert 149 <= m.level_dn <= 151 and m.clip_frac < 0.01 and 59 <= m.median_dn <= 151
    img[:8] = 255                        # a tenth of the frame clipped
    m = meter(img, CURVE)
    assert m.level_dn >= 250 and m.clip_frac == pytest.approx(0.1, abs=0.01)


def test_meter_is_not_biased_by_noise():
    rng = np.random.default_rng(0)
    clean = np.full((400, 640), 140.0)
    noisy = np.clip(clean + rng.normal(0, 8, clean.shape), 0, 255).astype(np.uint8)   # gain-16 noise
    assert meter(noisy, CURVE).level_dn == pytest.approx(140, abs=5)


# ------------------------------------------------------------------------------------ program line
def test_program_line_buys_light_in_order():
    ae = AutoExposure(S, 100, 1)
    assert ae.split(1) == (9, 1)                         # below the sensor's shortest
    assert ae.split(300) == (300, 1)                     # exposure first
    assert ae.split(1000) == (500, 2)                    # then gain to the knee
    assert ae.split(4000) == (1000, 4)                   # then exposure past the blur budget
    assert ae.split(32000) == (4000, 8)                  # then gain
    assert ae.split(1e9) == (4000, 15.9)
    totals = np.geomspace(5, 1e5, 400)
    e, g = np.array([ae.split(t) for t in totals]).T
    assert np.all(np.diff(e) >= 0) and np.all(np.diff(g) >= 0)        # never trades one back for the other
    inside = (totals >= ae.total_min) & (totals <= ae.total_max)
    assert np.allclose((e * g)[inside], totals[inside])


def test_bad_settings_rejected():
    with pytest.raises(ValueError):
        AutoExposure(AeSettings(min_exposure_us=600, max_exposure_us=500), 100, 1)
    with pytest.raises(ValueError):
        AutoExposure(AeSettings(target_dn=252), 100, 1)
    with pytest.raises(ValueError):
        AutoExposure(AeSettings(highlight_pct=100), 100, 1)


# ------------------------------------------------------------------------------------ closed loop
@pytest.mark.parametrize("lum", [3e-3, 3e-4, 1e-4, 3e-5, 1e-5])
def test_finds_the_light_within_two_round_trips_and_then_sends_nothing(lum):
    cam = SimCam(exposure_us=500, gain=1)
    rows = cam.run(lambda t: sky(lum), 4.0)
    assert all(on_target(r) for r in rows[17:])          # 2 x (7 frames of delay + 1)
    assert commands(rows) <= 2 and commands(rows[17:]) == 0
    assert blind(rows[17:]) == 0 and rows[-1].state == "ok"


def test_one_command_is_enough_when_the_frame_is_measurable():
    cam = SimCam(exposure_us=60, gain=1)                  # one stop under: no clipping, a plain measurement
    rows = cam.run(lambda t: sky(), 1.0)
    assert commands(rows) == 1 and rows[0].command and all(on_target(r) for r in rows[8:])


def test_gun_comes_up_from_trees_into_the_sky():
    cam = SimCam()
    cam.run(lambda t: sky(trees=1.0), 5.0)               # long enough to have exposed for the trees
    assert cam.ae.total > 5 * 125
    rows = cam.run(lambda t: sky(), 2.0)
    assert blind(rows) <= 8                               # the control delay and not a frame more
    assert all(on_target(r) for r in rows[17:]) and commands(rows) <= 2


def test_sky_exposure_is_kept_while_the_gun_passes_over_trees():
    cam = SimCam()
    cam.run(lambda t: sky(), 2.0)
    held = cam.ae.total
    dip = cam.run(lambda t: sky(trees=1.0), 0.4)          # shorter than ae_hold_s
    back = cam.run(lambda t: sky(), 1.0)
    assert commands(dip) == 0 and commands(back) == 0 and cam.ae.total == held
    assert blind(back) == 0 and all(on_target(r) for r in back)


def test_a_second_over_trees_does_not_blind_the_return_to_the_sky():
    cam = SimCam()
    cam.run(lambda t: sky(), 2.0)
    cam.run(lambda t: sky(trees=1.0), 1.0)
    back = cam.run(lambda t: sky(), 1.0)
    assert blind(back) == 0 and all(on_target(r) for r in back[9:])


def test_darker_scene_is_followed_slowly_and_fully():
    cam = SimCam()
    cam.run(lambda t: sky(), 2.0)
    start = cam.ae.total
    rows = cam.run(lambda t: sky(SKY / 8), 4.0)           # three stops darker, for good
    totals = np.array([r.exposure_us * r.gain for r in rows])
    assert totals[int(0.5 * FPS) - 1] == pytest.approx(start, rel=0.05)            # held
    assert np.log2(totals[int(1.5 * FPS)] / start) == pytest.approx(1.5, abs=0.5)  # ~1.5 stops/s after the hold
    assert np.diff(np.log2(totals)).max() <= 0.45                                  # no single step above ~0.4 stop
    assert all(on_target(r) for r in rows[int(3.2 * FPS):])


def test_brighter_scene_is_corrected_at_once():
    cam = SimCam()
    cam.run(lambda t: sky(), 2.0)
    rows = cam.run(lambda t: sky(SKY * 2.5), 1.0)         # bright cloud: over, not clipped
    assert rows[0].command and blind(rows) == 0 and all(on_target(r) for r in rows[8:])


def test_tree_line_moving_through_the_frame_does_not_move_the_exposure():
    cam = SimCam()
    cam.run(lambda t: sky(), 1.0)
    rows = cam.run(lambda t: sky(trees=float(np.clip(0.5 + 0.45 * np.sin(2 * np.pi * t / 1.5), 0, 1))), 6.0)
    assert commands(rows) == 0 and blind(rows) == 0


def test_no_hunting_on_lamp_flicker_or_at_high_gain():
    cam = SimCam()
    cam.run(lambda t: sky(), 1.0)
    rows = cam.run(lambda t: sky(SKY * (1 + 0.15 * np.sin(2 * np.pi * 3.7 * t))), 8.0)
    assert commands(rows) == 0
    dim = SimCam(exposure_us=4000, gain=8)
    dim.run(lambda t: sky(1e-5, trees=0.5), 1.0)
    assert commands(dim.run(lambda t: sky(1e-5, trees=0.5), 10.0)) == 0


@pytest.mark.parametrize("delay, fps", [(5, 50.0), (8, 120.0), (12, 50.0)])
def test_any_control_delay_converges_without_overshoot(delay, fps):
    cam = SimCam(delay=delay, fps=fps)
    cam.run(lambda t: sky(trees=1.0), 5.0)
    rows = cam.run(lambda t: sky(), 2.0)
    assert blind(rows) <= delay + 1 and commands(rows) <= 2
    assert all(on_target(r) for r in rows[2 * (delay + 1) + 1:])


def test_black_frame_is_not_waited_on_and_cap_off_recovers():
    cam = SimCam()
    cam.run(lambda t: sky(), 1.0)
    dark = cam.run(lambda t: sky(1e-9), 0.2)              # lens covered
    assert dark[0].command                                # at once, not after ae_hold_s
    cam.run(lambda t: sky(1e-9), 2.0)
    assert cam.ae.status.state == "too_dark" and cam.ae.total == pytest.approx(cam.ae.total_max)
    rows = cam.run(lambda t: sky(), 2.0)                  # uncovered: 500x too bright
    assert blind(rows) <= 17 and all(on_target(r) for r in rows[26:])


def test_out_of_range_scenes_are_reported():
    bright = SimCam()
    bright.run(lambda t: sky(1.0), 1.0)                   # clipped at 9 us, gain 1
    st = bright.ae.status
    assert st.state == "too_bright" and (st.exposure_us, st.gain) == (9, 1) and "ND filter" in st.message()
    dark = SimCam()
    dark.run(lambda t: sky(2e-7), 2.0)
    st = dark.ae.status
    assert st.state == "too_dark" and st.error_stops > 1 and "too dark" in st.message()
    ok = SimCam()
    ok.run(lambda t: sky(), 1.0)
    assert ok.ae.status.state == "ok" and ok.ae.status.message() == ""


def test_decisions_come_from_the_applied_exposure_not_the_commanded_one():
    ae = AutoExposure(S, 125, 1, CURVE)
    level = CURVE.dn_of(CURVE.linear_of(S.target_dn) * 3)             # a frame 3x over, taken at 375 us
    m = Meter(level, CURVE.linear_of(level), 0.0, level)
    assert ae.step(m, 375.0, 0.0) is None                             # 375 / 3 = the 125 us already commanded
    assert ae.step(m, 750.0, 0.02)[0] == pytest.approx(250, rel=0.02)


# ------------------------------------------------------------------------------------ camera glue
def test_ae_limits_follow_the_camera_and_the_frame_period():
    controls = {"ExposureTime": (9, 10631860, 20000), "AnalogueGain": (1.0, 15.9375, 1.0)}
    s = ae_limits(S, controls, 20000)
    assert (s.min_exposure_us, s.max_exposure_us, s.dim_exposure_us, s.max_gain) == (9, 500, 4000, 15.9)
    s = ae_limits(dataclasses.replace(S, max_gain=16.0, min_exposure_us=5.0), controls, 3333)
    assert (s.min_exposure_us, s.dim_exposure_us, s.max_gain) == (9, 3033, 15.9375)   # 300 fps: fits the frame
    assert ae_limits(S, {}, 20000) == S


def test_tone_curve_is_read_from_the_tuning_file():
    class Cam:
        files = {"ov9281_mono.json": {"rpi.contrast": {"ce_enable": 0, "gamma_curve": [0, 0, 100, 50, 200, 200]}}}

        @classmethod
        def load_tuning_file(cls, name):
            if name not in cls.files:
                raise RuntimeError("no such tuning file")
            return cls.files[name]

        @staticmethod
        def find_tuning_algo(tuning, name):
            return tuning[name]

    props = {"Model": "ov9281", "ColorFilterArrangement": 5}
    assert tone_curve_for(Cam, props).linear_of(63.75) == pytest.approx(0.5, abs=0.01)        # the file's curve
    assert np.allclose(tone_curve_for(Cam, {"Model": "imx296"}).lut, CURVE.lut)        # no file: built in
    Cam.files["ov9281_mono.json"]["rpi.contrast"]["ce_enable"] = 1
    assert np.allclose(tone_curve_for(Cam, props).lut, CURVE.lut)                      # adaptive: not trusted


# ------------------------------------------------------------------------------------ camera mode (tbot snap)
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
