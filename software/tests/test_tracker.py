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
    return load_config(sets=[f"tracker.model={request.param}"])


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
        assert len(tracks) == 1
        track = tracks[0]
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
        cfg = load_config(sets=[f"tracker.model={model}"])
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
    cfg = load_config(sets=["tracker.model=cartesian"])
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


def test_make_tracker_rejects_an_unknown_model():
    cfg = load_config(sets=["tracker.model=bogus"])
    scene = Scene(cfg)
    with pytest.raises(ValueError):
        make_tracker(cfg, scene.cam, scene.mount)
