"""tbot.bench: clays composited into recorded frames (inject) and the scores against truth (metrics)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from tbot.bench import inject, metrics
from tbot.config import load_config
from tbot.tools.score import rig_config

PRESS = Path(__file__).resolve().parents[1] / "field_data" / "2026-11-17-range" / "presses" / "20261117-104603-press"
# frames.bin is in git LFS: without it the file is a pointer of a few hundred bytes
needs_press = pytest.mark.skipif(not (PRESS / "frames.bin").exists() or (PRESS / "frames.bin").stat().st_size < 1e6
                                 or not (PRESS / "imu.csv").exists(), reason="range-day press (git LFS) not present")


@pytest.fixture(scope="module")
def built():
    cfg = rig_config(load_config(), PRESS)
    stage, bench, clays = inject.build(PRESS, cfg, 4, seed=3, press_key=11)
    assert clays, "no clay could be placed in the test press"
    return cfg, stage, bench, clays


def _good_rows(src, clay, min_major=3.0, margin=12):
    w, h = src.stage.size
    out = []
    for t in src.truth[clay.id]:
        if not t.visible or t.major_px < min_major or not (t.bg_std < 6.0):
            continue
        if not (margin <= t.u <= w - 1 - margin and margin <= t.v <= h - 1 - margin):
            continue
        r = int(math.ceil(t.major_px)) + 4
        y, x = int(round(t.v)), int(round(t.u))
        if (src.stage.mask[y - r:y + r + 1, x - r:x + r + 1] == 0).any():
            continue
        out.append(t)
    return out


def _dark(clay):
    return dataclasses.replace(clay, polarity="dark", k_top=0.4, k_bot=0.4, noise_dn=0.0)


@needs_press
def test_the_injected_clay_is_where_the_truth_says(built):
    _cfg, stage, _bench, clays = built
    errs = []
    for clay in clays:
        clay = _dark(clay)
        src = inject.InjectedSource(PRESS, [clay], stage)
        for t in _good_rows(src, clay)[::3]:
            diff = src.original(t.row).astype(np.float64) - src.render(t.row).astype(np.float64)
            r = int(math.ceil(t.major_px)) + 5
            y0, x0 = int(round(t.v)) - r, int(round(t.u)) - r
            d = np.clip(diff[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1], 0.0, None)
            assert d.sum() > 0.0, f"clay {clay.id} not drawn at row {t.row}"
            yy, xx = np.mgrid[y0:y0 + d.shape[0], x0:x0 + d.shape[1]]
            cu, cv = float((d * xx).sum() / d.sum()), float((d * yy).sum() / d.sum())
            errs.append(math.hypot(cu - t.u, cv - t.v))
            # nothing is drawn away from the clay
            far = np.abs(diff).copy()
            far[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1] = 0.0
            assert far.max() == 0.0
    assert len(errs) >= 5
    assert max(errs) < 0.5, f"centroid errors {sorted(errs)[-5:]}"


@needs_press
def test_size_scales_with_range(built):
    _cfg, stage, _bench, clays = built
    ratios, sizes = [], []
    for clay in clays:
        clay = _dark(clay)
        far = dataclasses.replace(clay, id=clay.id + "far",
                                  path=dataclasses.replace(clay.path, path=clay.path.path * 2.0))
        src = inject.InjectedSource(PRESS, [clay, far], stage)
        near_src = inject.InjectedSource(PRESS, [clay], stage)
        far_src = inject.InjectedSource(PRESS, [far], stage)
        truth_far = {t.row: t for t in src.truth[far.id]}
        for t in _good_rows(near_src, clay, min_major=6.0)[::2]:
            tf = truth_far[t.row]
            # the truth: same direction, twice the range, half the size
            assert tf.range_m == pytest.approx(2.0 * t.range_m, rel=1e-6)
            assert tf.u == pytest.approx(t.u, abs=1e-6) and tf.v == pytest.approx(t.v, abs=1e-6)
            sizes.append(tf.major_px / t.major_px)
            # the drawing: the summed darkening is contrast x projected area, a quarter at twice the range
            o = near_src.original(t.row).astype(np.float64)
            a_near = float((o - near_src.render(t.row)).sum())
            a_far = float((o - far_src.render(t.row)).sum())
            ratios.append(a_near / a_far)
    assert len(ratios) >= 3
    assert np.allclose(sizes, 0.5, rtol=1e-6)
    assert 3.5 < float(np.median(ratios)) < 4.5, ratios


@needs_press
def test_deterministic_from_the_seed(built):
    cfg, stage, bench, clays = built
    again = inject.make_clays(stage, bench, 4, seed=3, press_key=11)
    other = inject.make_clays(stage, bench, 4, seed=4, press_key=11)
    assert [c.meta() for c in again] == [c.meta() for c in clays]
    assert [c.meta() for c in other] != [c.meta() for c in clays]
    a = inject.InjectedSource(PRESS, clays[:1], stage)
    b = inject.InjectedSource(PRESS, again[:1], stage)
    rows = [t.row for t in a.truth[clays[0].id] if t.visible][:5]
    assert rows
    for r in rows:
        assert np.array_equal(a.render(r), b.render(r))


@needs_press
def test_the_press_is_never_written(built):
    _cfg, stage, _bench, clays = built
    path = PRESS / "frames.bin"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    mtime = path.stat().st_mtime_ns
    src = inject.InjectedSource(PRESS, clays, stage)
    drawn = 0
    while (f := src.read()) is not None:
        orig = src.original(src.last_row)
        drawn += int(not np.array_equal(f.image, orig))
        assert f.image is not orig
    assert drawn > 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert path.stat().st_mtime_ns == mtime


# ------------------------------------------------------------------------------------------------ metrics
def _hand_made():
    """10 frames 40 ms apart, armed from frame 2. The clay moves along v = 50; the detector finds it in all but
    frames 0 and 4, and reports a speck at (300, 300) in every frame (two in frame 4). Track 1 follows the clay
    (confirmed from frame 5), track 2 sits on the speck (confirmed from frame 3). The solver targets track 2
    until frame 5, then track 1, and fires on it in frame 8. A real mover at (500, 100) in frame 9 is detected."""
    truth = {k: {"exists": True, "u": 100.0 + 10 * k, "v": 50.0, "in_image": True, "masked": False,
                 "major_px": 10.0, "bg_std": 2.0} for k in range(10)}
    movers = [{"id": "bird", "label": "bird", "frames": [[9, 500.0, 100.0, 6.0, 6.0]]}]
    tel = []
    for k in range(10):
        u = 100.0 + 10 * k
        dets = [] if k in (0, 4) else [[u + 1.0, 50.5, 10, 20, -50, 0, 0, 10, 4, 30]]
        dets.append([300.0, 300.0, 5, 10, -20, 0, 0, 4, 4, 10])
        if k == 4:
            dets.append([310.0, 200.0, 5, 10, -20, 0, 0, 4, 4, 10])
        if k == 9:
            dets.append([501.0, 101.0, 6, 10, -20, 0, 0, 6, 6, 20])
        # the test projector takes (az, el) as the pixel itself
        tracks = [{"id": 1, "az_deg": u + 2.0, "el_deg": 50.0, "confirmed": k >= 5},
                  {"id": 2, "az_deg": 300.0, "el_deg": 300.0, "confirmed": k >= 3}]
        tel.append({"row": k, "t_exposure_ns": k * 40_000_000, "t_ns": k * 40_000_000 + 30_000_000,
                    "armed": k >= 2, "dets": dets, "tracks": tracks, "R_wg": list(np.eye(3).ravel()),
                    "decision": {"track_id": 2 if k < 6 else 1, "p_hit": 0.5, "range_m": 25.0},
                    "fired": k == 8})
    return tel, truth, movers


def _project(az, el, R):
    return az, el


def test_metrics_on_a_hand_made_press():
    tel, truth, movers = _hand_made()
    c = metrics.score_run(tel, _project, truth, movers, outcome=lambda rec: (0.2, True), polarity="dark")
    m = metrics.aggregate([c])
    assert m["det_recall"] == pytest.approx(0.8)
    assert m["det_recall_by"]["size:8-16"] == {"frames": 10, "recall": pytest.approx(0.8)}
    assert m["det_recall_by"]["bg:sky"]["frames"] == 10 and m["det_recall_by"]["polarity:dark"]["frames"] == 10
    assert m["clutter_per_frame_mean"] == pytest.approx(1.1)
    assert m["clutter_per_frame_p95"] == pytest.approx(1.55)
    assert m["real_mover_dets"] == 1
    assert m["armed_s"] == pytest.approx(0.28)
    assert m["false_conf_track_frames"] == 7 and m["false_conf_track_ids"] == 1
    assert m["false_conf_track_frames_per_armed_s"] == pytest.approx(7 / 0.28)
    assert m["false_conf_track_ids_per_armed_s"] == pytest.approx(1 / 0.28)
    assert m["ttt_ms_median"] == pytest.approx(120.0) and m["never_tracked_frac"] == 0.0
    assert m["target_correct_frac"] == pytest.approx(0.5) and m["target_false_frac"] == pytest.approx(0.5)
    assert m["truth_fires"] == 1 and m["false_fires"] == 0
    assert m["clays_hit_pct"] == 100.0 and m["clays_fired_at_pct"] == 100.0


def test_metrics_clean_mode_counts_every_confirmed_track_and_fire_as_false():
    tel, _truth, movers = _hand_made()
    m = metrics.aggregate([metrics.score_run(tel, _project, None, movers)])
    assert "det_recall" not in m
    assert m["false_conf_track_frames"] == 7 + 5 and m["false_conf_track_ids"] == 2
    assert m["false_fires"] == 1
    assert m["false_fires_per_armed_min"] == pytest.approx(1 / (0.28 / 60.0))
    # the clay's detections are clutter now; the real mover's are not
    assert m["clutter_per_frame_mean"] == pytest.approx(1.9) and m["real_mover_dets"] == 1


def test_a_track_on_a_real_mover_is_not_false():
    tel, truth, _movers = _hand_made()
    movers = [{"id": "speck", "label": "real", "frames": [[k, 300.0, 300.0, 4.0, 4.0] for k in range(10)]}]
    m = metrics.aggregate([metrics.score_run(tel, _project, truth, movers)])
    assert m["false_conf_track_frames"] == 0 and m["real_mover_conf_track_frames"] == 7
    assert m["target_false_frac"] == 0.0
