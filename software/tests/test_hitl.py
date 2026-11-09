"""Deterministic HITL geometry, timing, scene and simulated integration tests."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tbot.config import load_config
from tbot.hitl import analysis
from tbot.hitl.geometry import ScreenMap, screen_pose_homography
from tbot.hitl.scene import Disc, HitlStore
from tbot.optics import CameraModel
from tbot.tools import hitl


def _screen_map():
    cam = CameraModel.nominal(load_config(), 640, 400)
    h = screen_pose_homography(1920, 1080, distance_m=0.9, width_m=0.53,
                               roll_deg=2, pitch_deg=-4, yaw_deg=6)
    return ScreenMap(h, 1920, 1080, cam)


def test_screen_map_fit_projection_inverse_radius_and_serialization():
    truth = _screen_map()
    x, y = np.meshgrid(np.linspace(150, 1770, 9), np.linspace(100, 980, 7))
    points = np.column_stack((x.ravel(), y.ravel()))
    rng = np.random.default_rng(41)
    pixels = truth.to_pixel(points) + rng.normal(0, 0.025, (len(points), 2))
    fitted = ScreenMap.fit(points, pixels, truth.cam, 1920, 1080)
    held_out = np.array([[220, 230], [950, 520], [1600, 850], [1300, 160]])
    np.testing.assert_allclose(fitted.to_pixel(held_out), truth.to_pixel(held_out), atol=0.1)
    assert fitted.rms_px < 0.05
    assert fitted.n_points == len(points)
    np.testing.assert_allclose(fitted.from_pixel(fitted.to_pixel(held_out)), held_out, atol=1e-6)

    center = np.array([960.0, 540.0])
    angular_radius = 0.002
    radius = fitted.radius_for(center, angular_radius)
    rays = fitted.to_ray([center + [-radius, 0], center + [radius, 0]])
    diameter = math.atan2(np.linalg.norm(np.cross(*rays)), rays[0] @ rays[1])
    assert diameter == pytest.approx(2 * angular_radius, rel=0.01)

    saved = fitted.to_dict()
    restored = ScreenMap.from_dict(saved, truth.cam)
    assert restored.to_dict() == saved
    np.testing.assert_allclose(restored.to_pixel(held_out), fitted.to_pixel(held_out))
    other_cam = CameraModel.nominal(load_config(), 320, 200)
    with pytest.raises(ValueError, match="camera is"):
        ScreenMap.from_dict(saved, other_cam)


def test_store_log_replay_first_drawn_and_program():
    store = HitlStore(clock_ns=lambda: 1_000_000_000)
    initial = store.program()
    item = Disc(id=store.new_id(), t0=10, t1=None, x0=50, y0=60, r=4, level=200)
    store.add(item)
    added = store.program()
    assert added["v"] > initial["v"]
    assert added["items"][0]["t1"] is None
    store.end(item.id, 30)
    ended = store.program()
    assert ended["v"] > added["v"]
    assert ended["items"][0]["t1"] == 30
    store.post_log({"frames": [{"t": 1, "ids": []}, {"t": 2, "ids": [item.id]},
                               {"t": 3, "ids": [item.id, 99]}]}, 1)
    store.post_log({"frames": [{"t": 2, "ids": [77]}, {"t": 3, "ids": []},
                               {"t": 4, "ids": [99]}, {"t": 3.5, "ids": [77]}]}, 2)
    times, ids = store.frames()
    np.testing.assert_array_equal(times, [1, 2, 3, 4])
    assert ids == [(), (item.id,), (item.id, 99), (99,)]
    assert store.first_drawn(item.id) == 2
    assert store.first_drawn(99) == 3
    assert store.first_drawn(77) is None


def _display_case(exposure_fraction, n=150):
    """Generate exposure-averaged square patches with independently phased camera frames."""
    rng = np.random.default_rng(24)
    period_ms, height, d0, scan = 8.0, 1080, 12.0, 4.0
    exposure_ms = period_ms * exposure_fraction
    frames, trials, page_t, page_ids = [], [], [], []
    for i in range(n):
        draw = 1000.0 * i + 100.0
        off_draw = draw + 200.0
        row = [108.0, 540.0, 972.0][i % 3]
        delay = d0 + scan * row / height
        page_t.extend([draw, off_draw])
        page_ids.extend([{i + 1}, set()])
        start = len(frames)
        phase = rng.uniform(0, period_ms)
        for te in np.arange(draw - 40 + phase, off_draw + 60, period_ms):
            lo, hi = te - exposure_ms / 2, te + exposure_ms / 2
            lit = max(0.0, min(hi, off_draw + delay) - max(lo, draw + delay)) / exposure_ms
            frames.append({"te": int(round(te * 1e6)), "tc": int(round(te * 1e6)),
                           "exp_us": exposure_ms * 1000, "probe": {"patch": 30 + 150 * lit}})
        trials.append({"item": i + 1, "row": "patch", "y": row, "t1": off_draw,
                       "frames": [start, len(frames)]})
    ctx = analysis.Context(cfg=None, cam=None, mount=None, shot=None, budget=None,
                           smap=SimpleNamespace(screen_h=height), epoch_ns=0,
                           page_t=np.asarray(page_t), page_ids=page_ids, frames=frames,
                           mic_distance_m=0.0)
    return analysis.analyze_display(ctx, {"trials": trials}), d0, scan


def test_display_recovers_scanout_from_partial_exposures():
    result, d0, scan = _display_case(0.8)
    for edge in ("rise", "fall"):
        model = result[edge]
        assert model["n"] == 150
        assert model["d0_ms"] == pytest.approx(d0, abs=0.3)
        assert model["scanout_ms"] == pytest.approx(scan, abs=0.3)
        assert model["partial"] > 0.6 * model["n"]
        assert sum(s["method"] == "partial" for s in result["samples"][edge]) == model["partial"]


def test_display_short_exposures_report_brackets_and_warning():
    result, _, _ = _display_case(0.00001, n=30)
    for edge in ("rise", "fall"):
        assert result[edge]["n"] == 30
        assert all(s["method"] == "bracket" for s in result["samples"][edge])
        assert result[edge]["bracket_uncert_ms"] > 0
    assert any("few partially lit frames" in warning for warning in result["warnings"])


def test_sweep_truth_late_shot_is_behind_clay():
    smap = _screen_map()
    trajectory = {"t0": 0.0, "x0": 900.0, "y0": 540.0, "vx": 0.5}
    tof_s, exit_ms = 0.05, 200.0
    aim_xy = [trajectory["x0"] + trajectory["vx"] * (exit_ms + 1000 * tof_s), trajectory["y0"]]
    bore = smap.to_ray([aim_xy])[0]
    mount = SimpleNamespace(muzzle_pos_g=np.zeros(3), cam_pos_g=np.zeros(3), bore_dir_g=bore,
                            cam_to_gun=lambda ray: ray)
    shot = SimpleNamespace(time_of_flight=lambda distance: tof_s, drop=lambda distance, angle: 0.0,
                           pattern_radius=lambda distance: 0.5)
    ctx = SimpleNamespace(smap=smap, mount=mount, shot=shot, epoch_ns=0)
    on_time = analysis._shot_miss(ctx, trajectory, np.eye(3), exit_ms * 1e6, 20.0, 0.0, 0.0)
    late = analysis._shot_miss(ctx, trajectory, np.eye(3), (exit_ms + 12) * 1e6, 20.0, 0.0, 0.0)
    assert on_time["miss_deg"] == pytest.approx(0, abs=1e-8)
    assert late["miss_deg"] > 0
    assert late["along_deg"] > 0
    assert late["late_ms"] == pytest.approx(12, abs=0.1)


def test_sim_live_geometry_and_flash_end_to_end(tmp_path):
    """Slow test: real-time simulated rig, actuator and microphone (about 20–30 seconds)."""
    out = tmp_path / "run"
    assert hitl.main(["run", "--sim", "--live", "--tests", "geometry,flash", "--trials", "3",
                      "--out", str(out), "--no-write"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["flash"]["decisions"] == 3
    assert report["sim_check"]
    assert all(check["level"] == "PASS" for check in report["sim_check"])
