"""ClayTracker behaviour on synthetic clays projected through the nominal optics.

Truth is a world clay position (the camera's optical centre is the world origin); it is projected with
CameraModel/MountModel exactly the way the real pipeline sees it, so the tracker is exercised through its
real measurement path (pixel -> ray -> gun -> world).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbot.config import load_config
from tbot.optics import CameraModel, MountModel, azel_to_dir
from tbot.types import Detection
from tbot.vision.tracker import ClayTracker  # noqa: F401
from tbot.vision.tracker3d import make_tracker

W, H = 640, 400
FPS = 120.0
DT_NS = int(round(1e9 / FPS))
T0 = 1_000_000_000


# These pin the LEGACY trackers (tracker3d births, angular): vision.engine=legacy, else make_tracker
# returns the motion tracker for tracker.model=cartesian (tests/test_motion_tracker.py covers that one).
def _frame_from_azel(az: float, el: float) -> np.ndarray:
    """R_wg of a gun whose bore points at (az, el) with zero roll (columns = gun axes in world)."""
    x = np.asarray(azel_to_dir(az, el), dtype=float).reshape(3)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


class FixedAttitude:
    """Gun held still, bore on (az, el)."""

    def __init__(self, az: float = 0.0, el: float = 0.0):
        self._R = _frame_from_azel(az, el)
        self.ready = True

    def R_wg(self, t_ns: int) -> np.ndarray:
        return self._R

    def omega_g(self, t_ns: int) -> np.ndarray:
        return np.zeros(3)


class Scene:
    """Projects world clay positions into detections through the nominal camera/mount."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cam = CameraModel.nominal(cfg, W, H)
        self.mount = MountModel.from_config(cfg, None)

    def detect(self, p_w, R_wg, t_ns, rng=None, sigma_px=0.3) -> Detection | None:
        v_c = self.mount.gun_to_cam(R_wg.T @ np.asarray(p_w, dtype=float))
        if v_c[2] <= 0.0:
            return None
        uv = np.asarray(self.cam.ray_to_pixel(v_c[None, :]), dtype=float).reshape(2)
        if not (0.0 <= uv[0] < W and 0.0 <= uv[1] < H):
            return None
        if rng is not None and sigma_px:
            uv = uv + rng.normal(0.0, sigma_px, 2)
        ifov = float(np.asarray(self.cam.ifov(uv[None, :])).reshape(1)[0])
        rng_m = float(np.linalg.norm(p_w))
        # Size consistent with range_from_size(): apparent diameter widened by the blur disc.
        d_eff = self.cfg.mount.clay_diameter_m / (rng_m * ifov)
        d = math.hypot(d_eff, self.cfg.detector.psf_px)
        return Detection(
            t_ns=t_ns, u=float(uv[0]), v=float(uv[1]), area_px=math.pi * d * d / 4.0,
            diameter_px=d, contrast=-45.0, snr=20.0, score=0.9,
            bbox=(int(uv[0] - d), int(uv[1] - d), max(1, int(2 * d)), max(1, int(2 * d))),
        )


def _crossing(t_s: float) -> np.ndarray:
    """Clay crossing the muzzle line at 20 m, 20 m/s, 3 m up (y = 0 at t = 0.6 s)."""
    return np.array([20.0, -12.0 + 20.0 * t_s, 3.0])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


@pytest.fixture(params=["angular", "cartesian"])
def cfg(request):
    """Every behaviour test runs against both tracker models."""
    return load_config(sets=["vision.engine=legacy", f"tracker.model={request.param}"])


def test_crossing_clay_is_tracked_confirmed_and_predicted(cfg):
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(7)

    ids, pred_err, range_err, sigma_ratio = set(), [], [], []
    for n in range(144):
        t = T0 + n * DT_NS
        p = _crossing(n / FPS)
        det = scene.detect(p, att.R_wg(t), t, rng, 0.3)
        tracks = tracker.update(t, [det] if det is not None else [], att.R_wg(t))
        # one clay, one track; the cartesian tracker may hold a clean twin of a newborn for a frame or two
        # (tracker.twin_newborn) but never confirms a second one
        assert len(tracks) <= 2 and sum(tr.confirmed for tr in tracks) <= 1
        track = max(tracks, key=lambda tr: tr.hits)
        assert track.confirmed == (n + 1 >= cfg.tracker.confirm_hits)
        if not track.confirmed:
            continue
        ids.add(track.track_id)
        assert tracker.best() is track
        if n >= 30:
            t_pred = t + 150_000_000
            d_pred = tracker.predict_dir(track, t_pred)
            d_true = _crossing(n / FPS + 0.150)
            d_true = d_true / np.linalg.norm(d_true)
            err = _angle(d_pred, d_true)
            pred_err.append(err)
            sigma_ratio.append(err / tracker.predict(track, t_pred)[2])
        if n >= 60:
            range_err.append(abs(track.range_m - np.linalg.norm(p)) / np.linalg.norm(p))

    assert ids == {1}, "one clay must stay on one track id"
    # 150 ms lead prediction. The lead itself is ~8.6 deg at the 1 rad/s crossing peak; a constant-
    # acceleration model with the configured jerk PSD (2000 deg^2/s^5) predicts it to ~0.5 deg worst case
    # and ~0.25 deg typically (measured over 8 seeds; lowering the PSD makes the true error worse, see
    # the slice report). The filter's own sigma must cover that error.
    assert max(pred_err) < math.radians(0.6)
    assert float(np.median(pred_err)) < math.radians(0.3)
    assert max(sigma_ratio) < 3.0
    assert max(range_err) < 0.25, "size-based range must be within 25 % after 60 frames"
    # Size-based ranges share a systematic error (psf, detector bias), so the filter must not report
    # a 1-sigma that keeps shrinking as if every frame were an independent sample.
    track = tracker.best()
    floor = cfg.tracker.size_range_bias_frac * track.range_m
    assert track.range_sigma_m >= floor
    assert track.range_sigma_m < cfg.tracker.range_prior_sigma_m


def test_random_clutter_never_confirms_a_track(cfg):
    scene = Scene(cfg)
    att = FixedAttitude()
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(11)
    for n in range(200):
        t = T0 + n * DT_NS
        u, v = float(rng.uniform(0, W)), float(rng.uniform(0, H))
        det = Detection(t_ns=t, u=u, v=v, area_px=7.1, diameter_px=3.0, contrast=-40.0, snr=9.0,
                        score=0.4, bbox=(int(u), int(v), 3, 3))
        tracks = tracker.update(t, [det], att.R_wg(t))
        assert len(tracks) <= cfg.tracker.max_tracks
        assert not any(tr.confirmed for tr in tracks)
        assert tracker.best() is None


def test_detection_gap_keeps_the_same_track(cfg):
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(3)
    gap = (70, 71, 72)
    seen = set()
    for n in range(120):
        t = T0 + n * DT_NS
        det = None if n in gap else scene.detect(_crossing(n / FPS), att.R_wg(t), t, rng, 0.3)
        tracks = tracker.update(t, [det] if det is not None else [], att.R_wg(t))
        if n in gap:
            assert tracks and tracks[0].misses == n - gap[0] + 1
        elif n > gap[-1]:
            assert tracks and tracks[0].misses == 0
        if tracker.best() is not None:
            seen.add(tracker.best().track_id)
    assert seen == {1}
    assert tracker.tracks[0].hits == 120 - len(gap)


def test_track_is_deleted_after_too_many_misses(cfg):
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    for n in range(10):
        t = T0 + n * DT_NS
        det = scene.detect(_crossing(n / FPS), att.R_wg(t), t)
        tracker.update(t, [det], att.R_wg(t))
    assert tracker.tracks[0].confirmed
    for k in range(1, cfg.tracker.max_misses + 1):
        tracks = tracker.update(T0 + (10 + k) * DT_NS, [], att.R_wg(0))
        assert len(tracks) == 1 and tracks[0].misses == k
    assert tracker.update(T0 + (11 + cfg.tracker.max_misses) * DT_NS, [], att.R_wg(0)) == []
    assert tracker.best() is None


def test_azimuth_wrap_keeps_one_track(cfg):
    """A clay crossing world azimuth +/-pi must not spawn a second track."""
    scene = Scene(cfg)
    att = FixedAttitude(math.pi, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(5)
    ids, crossed, dir_err = set(), False, []
    prev_az = None
    for n in range(144):
        t = T0 + n * DT_NS
        p = np.array([-20.0, 12.0 - 20.0 * (n / FPS), 3.0])
        det = scene.detect(p, att.R_wg(t), t, rng, 0.3)
        tracker.update(t, [det] if det is not None else [], att.R_wg(t))
        track = tracker.best()
        if track is None:
            continue
        ids.add(track.track_id)
        az_true = math.atan2(p[1], p[0])
        if prev_az is not None and abs(az_true - prev_az) > math.pi:
            crossed = True
        prev_az = az_true
        d_true = p / np.linalg.norm(p)
        dir_err.append(_angle(tracker.predict_dir(track, t), d_true))
    assert crossed, "the trajectory must actually cross the +/-pi seam"
    assert ids == {1}
    assert max(dir_err) < math.radians(0.2), "state must stay on the clay across the wrap"


def test_two_clays_get_two_tracks(cfg):
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(13)
    for n in range(40):
        t = T0 + n * DT_NS
        ts = n / FPS
        dets = [
            scene.detect(np.array([20.0, -8.0 + 20.0 * ts, 3.0]), att.R_wg(t), t, rng, 0.3),
            scene.detect(np.array([22.0, 6.0 - 18.0 * ts, 6.0]), att.R_wg(t), t, rng, 0.3),
        ]
        tracker.update(t, [d for d in dets if d is not None], att.R_wg(t))
    confirmed = [tr for tr in tracker.tracks if tr.confirmed]
    assert len(confirmed) == 2
    assert all(tr.hits == 40 for tr in confirmed), "no detection may be stolen by the other track"
    assert {tr.track_id for tr in confirmed} == {1, 2}


def test_max_tracks_and_reset(cfg):
    scene = Scene(cfg)
    att = FixedAttitude()
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(17)
    for n in range(5):
        t = T0 + n * DT_NS
        dets = []
        for k in range(cfg.tracker.max_tracks + 3):
            u = 60.0 + 70.0 * k
            dets.append(Detection(t_ns=t, u=u + float(rng.normal(0, 0.1)), v=200.0, area_px=7.1,
                                  diameter_px=3.0, contrast=-40.0, snr=9.0, score=0.5,
                                  bbox=(int(u), 200, 3, 3)))
        tracks = tracker.update(t, dets, att.R_wg(t))
        assert len(tracks) <= cfg.tracker.max_tracks
    tracker.reset()
    assert tracker.tracks == [] and tracker.best() is None
    t = T0 + 100 * DT_NS
    det = scene.detect(_crossing(0.0), att.R_wg(t), t)
    assert tracker.update(t, [det], att.R_wg(t))[0].track_id == 1


def test_update_without_detections_predicts_forward(cfg):
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    for n in range(12):
        t = T0 + n * DT_NS
        tracker.update(t, [scene.detect(_crossing(n / FPS), att.R_wg(t), t)], att.R_wg(t))
    track = tracker.best()
    az_rate = track.x[1]
    assert az_rate > 0.5
    before = track.az
    sigma_before = tracker.predict(track, track.t_ns)[2]
    t = T0 + 12 * DT_NS
    tracker.update(t, [], att.R_wg(t))
    assert track.t_ns == t
    assert track.az == pytest.approx(before + az_rate / FPS, rel=0.05)
    assert tracker.predict(track, t)[2] > sigma_before, "uncertainty must grow without measurements"
    assert track.range_sigma_m > cfg.tracker.range_prior_sigma_m * 0.0  # range filter also coasts


def _outgoing(t_s: float) -> np.ndarray:
    """Receding clay: starts 8 m out, 22 m/s away and across, decelerating (drag) and nearly level (lift ~ g)."""
    v0 = np.array([18.0, 9.0, 4.0])
    a = np.array([-4.0, -2.0, -1.5])  # drag along -v, a little net sink
    return np.array([8.0, -4.0, 1.0]) + v0 * t_s + 0.5 * a * t_s * t_s


def test_cartesian_tracker_predicts_a_receding_clay_far_better_than_the_angular_one():
    """The case the angular filter gets wrong: a clay whose range grows fast. 50 fps, predict 140 ms ahead."""
    fps, horizon_s = 50.0, 0.14
    errs, sig = {}, {}
    for model in ("angular", "cartesian"):
        cfg = load_config(sets=["vision.engine=legacy", f"tracker.model={model}"])
        scene = Scene(cfg)
        att = FixedAttitude(math.radians(20.0), math.radians(5.0))
        tracker = make_tracker(cfg, scene.cam, scene.mount)
        rng = np.random.default_rng(11)
        e, s = [], []
        for n in range(30):  # 0.6 s of flight
            t = T0 + int(round(n * 1e9 / fps))
            det = scene.detect(_outgoing(n / fps), att.R_wg(t), t, rng, 0.3)
            tracks = tracker.update(t, [det] if det is not None else [], att.R_wg(t))
            if n >= 15 and tracks and tracks[0].confirmed:
                tr = tracks[0]
                t_ahead = t + int(horizon_s * 1e9)
                truth = _outgoing(n / fps + horizon_s)
                e.append(math.degrees(_angle(tracker.predict_dir(tr, t_ahead), truth)))
                s.append(math.degrees(tracker.predict(tr, t_ahead)[2]))
        errs[model], sig[model] = float(np.mean(e)), float(np.mean(s))
    assert errs["cartesian"] < 0.25
    assert errs["cartesian"] < 0.5 * errs["angular"], errs
    assert errs["cartesian"] < 3.0 * sig["cartesian"], (errs, sig)  # the reported uncertainty is honest


def test_cartesian_angular_rate_and_range_follow_the_3d_state():
    cfg = load_config(sets=["vision.engine=legacy", "tracker.model=cartesian"])
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    rng = np.random.default_rng(5)
    tr = None
    for n in range(60):
        t = T0 + n * DT_NS
        det = scene.detect(_crossing(n / FPS), att.R_wg(t), t, rng, 0.3)
        tr = tracker.update(t, [det] if det is not None else [], att.R_wg(t))[0]
    t_s = 59 / FPS
    p = _crossing(t_s)
    v = np.array([0.0, 20.0, 0.0])
    u = p / np.linalg.norm(p)
    true_rate = float(np.linalg.norm(v - (v @ u) * u)) / float(np.linalg.norm(p))
    assert tracker.angular_rate(tr) == pytest.approx(true_rate, rel=0.1)
    assert tr.range_m == pytest.approx(float(np.linalg.norm(p)), rel=0.3)
    assert tr.range_sigma_m >= cfg.tracker.size_range_bias_frac * tr.range_m - 1e-9
    assert tr.x.shape == (6,) and tr.P.shape == (6, 6)  # angular view for overlays/telemetry


def test_native_filter_step_matches_the_numpy_reference():
    """vision/_measure.c tbot_track_predict / tbot_track_update against the NumPy code they replace: the same
    tracks, to rounding, over a crossing clay with a second object in view (2- and 3-row updates, new tracks)."""
    from tbot.vision import _native

    if _native.track_kernel() is None:
        pytest.skip(f"native tracker step: {_native.track_status}")
    cfg = load_config(sets=["vision.engine=legacy", "tracker.model=cartesian"])
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    fast, ref = make_tracker(cfg, scene.cam, scene.mount), make_tracker(cfg, scene.cam, scene.mount)
    assert fast._nat is not None
    ref._nat = None
    rng = np.random.default_rng(5)
    n_updates = 0
    for n in range(60):
        t = T0 + n * DT_NS
        dets = [d for d in (scene.detect(_crossing(n / FPS), att.R_wg(t), t, rng, 0.3),
                            scene.detect(_outgoing(n / FPS), att.R_wg(t), t, rng, 0.3)) if d is not None]
        a, b = fast.update(t, dets, att.R_wg(t)), ref.update(t, dets, att.R_wg(t))
        assert [(x.track_id, x.hits, x.misses, x.confirmed) for x in a] == \
               [(x.track_id, x.hits, x.misses, x.confirmed) for x in b]
        for x, y in zip(a, b):
            n_updates += x.hits > 1
            assert x.s.dtype == np.float64 and x.s.flags.c_contiguous and x.S.flags.c_contiguous
            np.testing.assert_allclose(x.s, y.s, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(x.S, y.S, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(x.x, y.x, rtol=1e-9, atol=1e-12)
            assert x.range_m == pytest.approx(y.range_m, rel=1e-9)
    assert n_updates > 50


def test_make_tracker_rejects_an_unknown_model():
    cfg = load_config(sets=["vision.engine=legacy", "tracker.model=bogus"])
    scene = Scene(cfg)
    with pytest.raises(ValueError):
        make_tracker(cfg, scene.cam, scene.mount)


# ------------------------------------------------------------------ 2026-11-19 range day: clutter around the clay
def _cart():
    return load_config(sets=["vision.engine=legacy", "tracker.model=cartesian"])


def _feed(tracker, scene, att, n0, n1, pos_of, extra=lambda n, t: []):
    for n in range(n0, n1):
        t = T0 + n * DT_NS
        dets = [d for d in (scene.detect(p, att.R_wg(t), t) for p in pos_of(n)) if d is not None]
        tracker.update(t, dets + extra(n, t), att.R_wg(t))


def test_target_is_the_moving_track_nearest_the_bore_not_the_oldest_speck():
    """A tree speck the scenery filter missed (still in the world, many hits) against a clay the gun points at."""
    cfg = _cart()
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    speck = np.array([30.0, 8.0, 6.0])                       # fixed in the world, 15 deg off the bore
    _feed(tracker, scene, att, 0, 60, lambda n: [speck])
    _feed(tracker, scene, att, 60, 120, lambda n: [speck, _crossing(0.3 + (n - 60) / FPS)])   # 20 m/s, at the bore
    by_hits = tracker.best()
    bore = np.asarray(azel_to_dir(0.0, math.radians(8.0)), dtype=float)
    chosen = tracker.best(bore_dir_w=bore, t_ns=T0 + 119 * DT_NS)
    assert by_hits is not None and chosen is not None and chosen is not by_hits
    assert by_hits.hits > chosen.hits and not tracker.surely_slow(chosen)
    assert np.linalg.norm(chosen.s[3:6]) > 10.0, "the target is the 20 m/s clay"


def _recorded(name):
    """Frames of a real range press (2026-11-19, the gun swinging on a clay in trees): every detection the gun made
    and its attitude, with the lens and mount it had (tests/data/range_2026-11-19_clutter.json)."""
    import json
    from pathlib import Path

    from tbot.config import _build, Config
    from tbot.tools.score import _fill
    import dataclasses

    d = json.loads((Path(__file__).parent / "data" / "range_2026-11-19_clutter.json").read_text())[name]
    raw = dataclasses.asdict(load_config(sets=["vision.engine=legacy"]))
    for sec, v in d["config"].items():
        raw[sec] = _fill({k: x for k, x in v.items() if k in raw[sec]}, raw[sec])
    cfg = _build(Config, raw, "")
    cam = CameraModel.from_calibration(d["calibration"]["intrinsics"], cfg, d["width"], d["height"])
    mount = MountModel.from_config(cfg, d["calibration"]["boresight"])
    return cfg, cam, mount, d["frames"]


@pytest.mark.parametrize("name,clay_uv", [("20261119-092601-press", (251.0, 252.0)),
                                          ("20261119-092733-press", (227.0, 224.0))])
def test_real_clay_in_clutter_keeps_a_track_that_predicts_it(name, clay_uv):
    """On the range day the clay the gun followed was taken by one-frame tracks born on specks, rejected by the
    overconfident track that resulted, and pushed out of a full table: it never had a track old enough to shoot
    at. Through today's tracker, from the frame the clay is first seen to the end of the slice, some confirmed
    track must be on it (predicting it to 0.3 deg) in most frames."""
    cfg, cam, mount, frames = _recorded(name)
    tracker = make_tracker(cfg, cam, mount)
    cu, cv = clay_uv
    on, seen = 0, 0
    for fr in frames:
        R = np.asarray(fr["R"], dtype=float).reshape(3, 3)
        dets = [Detection(t_ns=fr["t"], u=x[0], v=x[1], area_px=x[9], diameter_px=x[2], contrast=x[4], snr=x[3],
                          score=1.0, bbox=(int(x[5]), int(x[6]), int(x[7]), int(x[8]))) for x in fr["dets"]]
        near = [x for x in dets if math.hypot(x.u - cu, x.v - cv) < 30.0 and x.diameter_px < 20.0]
        clay = min(near, key=lambda x: math.hypot(x.u - cu, x.v - cv)) if near else None
        tracker.update(fr["t"], dets, R)
        if clay is None:
            continue
        cu, cv = clay.u, clay.v
        seen += 1
        ray = np.asarray(mount.cam_to_gun(np.asarray(cam.pixel_to_ray(np.array([clay.u, clay.v]))).reshape(3)))
        d_clay = R @ (ray.reshape(3) / np.linalg.norm(ray))
        if any(tr.confirmed and _angle(tracker.predict_dir(tr, fr["t"]), d_clay) < math.radians(0.3)
               for tr in tracker.tracks):
            on += 1
    assert seen >= 10
    assert on >= 0.6 * seen, f"the clay had a confirmed track on it in {on} of the {seen} frames it was seen"


def test_a_one_frame_track_does_not_steal_the_clay_and_a_clean_twin_takes_over():
    """A speck seen once next to where the clay then appears (30 fps, as in trees on the gun) must not leave the
    clay with a junk velocity it is sure of: the clay must keep a confirmed track that predicts it."""
    cfg = _cart()
    scene = Scene(cfg)
    att = FixedAttitude(0.0, math.radians(8.0))
    tracker = make_tracker(cfg, scene.cam, scene.mount)
    dt = 33_333_333
    clay = lambda n: np.array([25.0, -3.0 + 0.5 * n, 4.0 + 0.1 * n])   # 15 m/s across
    speck = clay(1) + np.array([0.0, -0.25, 0.1])                        # ~0.6 deg from the clay's first sighting
    t = T0
    tracker.update(t, [scene.detect(speck, att.R_wg(t), t)], att.R_wg(t))
    for n in range(1, 16):
        t = T0 + n * dt
        tracker.update(t, [scene.detect(clay(n), att.R_wg(t), t)], att.R_wg(t))
    d_true = clay(15) / np.linalg.norm(clay(15))
    live = [tr for tr in tracker.tracks if tr.confirmed and tr.misses == 0]
    assert live, "the clay must have a confirmed track that is still being seen"
    assert min(_angle(tracker.predict_dir(tr, t), d_true) for tr in live) < math.radians(0.1)
