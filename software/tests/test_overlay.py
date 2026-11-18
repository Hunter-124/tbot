"""Debug overlay (tbot.app.overlay): HUD, marks, old/sparse telemetry records, cost."""

import json
import math
import time

import numpy as np
import pytest

from tbot.app import overlay
from tbot.config import load_config
from tbot.optics import CameraModel, MountModel


@pytest.fixture(scope="module")
def rig():
    cfg = load_config()
    return CameraModel.nominal(cfg, 640, 400), MountModel.from_config(cfg, None)


def _record(**over):
    """A busy record in today's format (tbot.app.run Pipeline.step), JSON round-tripped like telemetry.jsonl."""
    rec = {
        "seq": 3, "armed": True, "fired": False, "live": False, "imu_ready": True, "shots": 2, "fps": 49.5,
        "age_ms": 31.0, "n_detections": 2, "dropped": 4, "exposure_us": 420.0, "gain": 1.5,
        "ae": {"state": "too_bright", "level_dn": 230.0, "clip_pct": 3.0, "error_stops": 1.2},
        "health": {"cpu_temp_c": 71.5, "throttled": {"under_voltage_now": True, "throttled_now": False}},
        "dets": [[120.0, 90.0, 9.0, 30.0, -40.0, 116, 87, 8, 6, 40.0],
                 [400.0, 300.0, 5.0, 8.0, -20.0, 398, 298, 5, 5, 18.0]],
        "static": [[500.0, 300.0], [30.0, 10.0], [600.0, 395.0]],
        "R_wg": np.eye(3).ravel().tolist(),
        "tracks": [
            {"id": 1, "az_deg": 0.0, "el_deg": 0.0, "range_m": 20.0, "confirmed": True,
             "s": [20.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]},
            {"id": 2, "az_deg": 3.0, "el_deg": 2.0, "range_m": None, "confirmed": False,
             "x": [0.05, 0.1, 0.0, 0.03, 0.0, 0.0]},
        ],
        "decision": {"fire": False, "reason": "p_hit 0.10 < 0.50", "track_id": 1, "miss_deg": 1.0,
                     "pattern_deg": 0.8, "sigma_deg": 0.5, "p_hit": 0.1, "range_m": 20.0,
                     "lead_dir_w": [1.0, 0.01, 0.0], "bore_dir_w": [1.0, 0.0, 0.0]},
        "stage_ms": {"read": 1.0, "attitude": 0.7, "detect": 4.0, "track": 1.1, "solve": 0.9, "overlay": 0.0,
                     "total": 8.2},
    }
    rec.update(over)
    return json.loads(json.dumps(rec))


def _gray(value=150):
    return np.full((400, 640), value, np.uint8)


def test_draw_record_busy_frame(rig):
    cam, mount = rig
    gray = _gray()
    before = gray.copy()
    mask = np.full((400, 640), 255, np.uint8)
    mask[180:220, 0:40] = 0
    out = overlay.draw_record(gray, _record(), cam, mount, mask=mask, bore_px=(320.0, 210.0), banner="t+0.100s")
    assert out.shape == (400, 640, 3) and out.dtype == np.uint8
    assert np.array_equal(gray, before), "the caller's frame is not modified"
    assert (out == overlay.C_DET).all(axis=2).sum() >= 20               # detection brackets in their colour
    assert np.median(out[2:overlay.TOP_H - 2, :]) < 100                # the status band darkens the top
    assert np.median(out[-6:, :]) < 100                                 # ... and the decision band the bottom
    assert out[200, 10, 0] > out[200, 10, 2]                            # the mask is washed blue
    assert np.array_equal(out[150, 600], [150, 150, 150])              # free sky is left alone
    assert not (out == overlay.C_FIRE).all(axis=2)[0:4, 100:540].any()  # no fire flash


def test_fire_flash_and_pattern(rig):
    cam, mount = rig
    rec = _record(fired=True, decision={**_record()["decision"], "fire": True, "reason": "FIRE", "p_hit": 0.8})
    out = overlay.draw_record(_gray(), rec, cam, mount)
    red = (out == overlay.C_FIRE).all(axis=2)
    assert red[0:4, 100:540].mean() > 0.9 and red[396:400, 100:540].mean() > 0.9   # red frame round the picture
    assert red[overlay.TOP_H + 8:overlay.TOP_H + 40, 300:340].mean() > 0.5         # the FIRE plate


@pytest.mark.parametrize("rec", [
    {},
    {"seq": 7},
    {"seq": 7, "armed": None, "fired": None, "fps": None, "age_ms": None, "shots": None, "health": None,
     "ae": None, "exposure_us": None, "gain": None, "static": None, "stage_ms": None, "decision": None},
    # an early-format record: short detection rows, no filter state, no live/imu/health/exposure keys
    {"seq": 1, "armed": True, "fps": 30.0, "n_detections": 1, "dets": [[100.0, 100.0, 6.0, 12.0]],
     "tracks": [{"id": 4, "az_deg": 1.0, "el_deg": 0.5, "confirmed": True}],
     "decision": {"fire": False, "reason": "no target"}, "R_wg": np.eye(3).ravel().tolist()},
    # nulls and junk where numbers belong (NaN is written as null)
    {"seq": 2, "dets": [[None, 5.0, 3.0, 2.0, 0.0, None, None, None, None, 0.0], [1.0], []],
     "tracks": [{"id": 5, "az_deg": None, "el_deg": 0.0}, {"az_deg": 1.0, "el_deg": 1.0}, {"id": "x"},
                {"id": 6, "az_deg": 2.0, "el_deg": 1.0, "range_m": None, "s": [None] * 9}],
     "decision": {"fire": True, "reason": None, "miss_deg": None, "pattern_deg": None, "sigma_deg": None,
                  "p_hit": None, "lead_dir_w": [0.0, 0.0, 0.0], "bore_dir_w": None},
     "R_wg": np.eye(3).ravel().tolist(), "static": [[None, 3.0], [1e9, 2.0]],
     "health": {"cpu_temp_c": None, "throttled": None}, "ae": {"state": None}},
    # pointing backwards: nothing projects
    {"seq": 3, "R_wg": (-np.eye(3)).ravel().tolist(),
     "tracks": [{"id": 1, "az_deg": 0.0, "el_deg": 0.0, "confirmed": True, "s": [20, 0, 0, 0, 5, 0, 0, 0, 0]}],
     "decision": {"fire": False, "reason": "x", "lead_dir_w": [1.0, 0.0, 0.0], "bore_dir_w": [1.0, 0.0, 0.0],
                  "pattern_deg": 0.8}},
])
def test_draw_record_tolerates_old_and_sparse_records(rig, rec):
    cam, mount = rig
    out = overlay.draw_record(_gray(), json.loads(json.dumps(rec)), cam, mount, mask=None, bore_px=None)
    assert out.shape == (400, 640, 3) and out.dtype == np.uint8


@pytest.mark.parametrize("shape", [(400, 640), (40, 60), (12, 16), (1, 1), (400, 640, 3), (400, 640, 1)])
def test_draw_any_size(shape, rig):
    cam, mount = rig
    img = np.full(shape, 90, np.uint8)
    out = overlay.draw(img, telemetry=_record(), armed=True, fired=True, bore_px=(5.0, 5.0))
    assert out.shape == shape[:2] + (3,) and out.dtype == np.uint8
    out = overlay.draw_record(img, _record(), cam, mount, banner="HELD t+0.010s  frame 1/2")
    assert out.shape == shape[:2] + (3,)


def test_status_band_drops_detail_on_a_narrow_frame():
    tele = _record()
    for w in (640, 300, 120, 30):
        out = overlay.draw(np.zeros((60, w), np.uint8), telemetry=tele, armed=False)
        assert out.shape == (60, w, 3)


def test_projector_batch_matches_single(rig):
    cam, mount = rig
    R = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    proj = overlay.projector(cam, mount, R)
    dirs = np.array([[0.8, 0.6, 0.05], [0.79, 0.62, -0.02], [-1.0, 0.0, 0.0], [0.82, 0.57, 0.1]])
    many = proj.many(dirs)
    for d, uv in zip(dirs, many):
        one = proj(d)
        if one is None:
            assert not np.isfinite(uv).all()
        else:
            assert np.allclose(one, uv, atol=1e-6)
    pts = overlay._project_all(proj, [dirs[0], None, dirs[2]])
    assert pts[1] is None and pts[2] is None and pts[0] == overlay._ipt(proj(dirs[0]))
    assert overlay._project_all(lambda d: proj(d), [dirs[0]]) == [pts[0]]  # a projector without .many


def test_firing_bar_is_read_from_the_reason():
    assert float(overlay._BAR_RE.search("p_hit 0.12 < 0.50").group(1)) == 0.5
    assert float(overlay._BAR_RE.search("p_hit 1e-05 < 0.35 (not armed)").group(1)) == 0.35
    assert overlay._BAR_RE.search("track young") is None


def test_pattern_circle_radius_follows_the_angle(rig):
    cam, mount = rig
    proj = overlay.projector(cam, mount, np.eye(3))
    bore = np.array([1.0, 0.0, 0.0])
    c = proj(bore)
    r1 = math.dist(c, proj(overlay._cone_dir(bore, math.radians(0.5))))
    r2 = math.dist(c, proj(overlay._cone_dir(bore, math.radians(1.0))))
    assert r1 > 1.0 and abs(r2 / r1 - 2.0) < 0.05
    assert overlay._cone_dir(bore, None) is None and overlay._cone_dir([0.0, 0.0, 0.0], 0.01) is None


def test_overlay_cost(rig):
    """Generous laptop bound: the overlay runs on the Pi's CPU next to the frame loop, so a regression to
    full-frame float work or thick anti-aliased strokes everywhere should show up here."""
    cam, mount = rig
    rec = _record(fired=True)
    rec["dets"] = rec["dets"] * 4
    rec["static"] = [[20.0 * i, 200.0 + 5 * (i % 7)] for i in range(24)]
    rec["tracks"] = [dict(t, id=i) for i, t in enumerate(rec["tracks"] * 3)]
    gray = (np.random.default_rng(0).random((400, 640)) * 255).astype(np.uint8)
    mask = np.full((400, 640), 255, np.uint8)
    mask[300:, 250:390] = 0
    for _ in range(3):
        overlay.draw_record(gray, rec, cam, mount, mask=mask, bore_px=(320.0, 200.0), banner="t+0.1s")
    times = []
    for _ in range(25):
        t0 = time.perf_counter()
        overlay.draw_record(gray, rec, cam, mount, mask=mask, bore_px=(320.0, 200.0), banner="t+0.1s")
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 0.015, f"overlay median {np.median(times) * 1e3:.1f} ms"
