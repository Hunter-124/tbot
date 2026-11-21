"""MotionTracker (vision/tracker_motion.py): tracks are born only from detections on one straight, evenly timed line.

Synthetic scenes through the nominal optics, 50 frames/s (the Pi's rate). Truth lives in the world frame; detections
are projected through CameraModel/MountModel the way the pipeline sees them. `stab` (background-stabilised pixels) is
the gnomonic projection of the true world direction about +X in focal-length pixels: scenery stands still there, as
in vision/motion.py. Clutter is drawn uniform over the frame (or a 150 x 100 px patch of scenery) every frame.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tbot.config import load_config
from tbot.optics import CameraModel, MountModel, azel_to_dir
from tbot.types import Detection
from tbot.vision.tracker3d import CartesianClayTracker, make_tracker
from tbot.vision.tracker_motion import MotionTracker

W, H = 640, 400
FPS = 50.0
DT_NS = int(round(1e9 / FPS))
T0 = 1_000_000_000
D2R = math.pi / 180.0

CLAY = 0.97      # Detection.score marks whose detection it is (the tracker ignores score)
CLAY_B = 0.98
CLUTTER = 0.5


def _frame_from_azel(az: float, el: float) -> np.ndarray:
    x = np.asarray(azel_to_dir(az, el), dtype=float).reshape(3)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _rodrigues(v: np.ndarray, k: np.ndarray, a: float) -> np.ndarray:
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * float(k @ v) * (1.0 - math.cos(a))


class Scene:
    def __init__(self, sets=()):
        self.cfg = load_config(sets=list(sets))
        # the rig's calibrated 12 mm lens (calib/intrinsics.json of 2026-11-20: pinhole, f = 2009 px, ~35 px/deg);
        # CameraModel.nominal still describes the old 2.8 mm lens (8 px/deg) and the px-valued [vision.track] keys
        # are set for this one
        self.cam = CameraModel(W, H, [[2009.0, 0.0, 319.5], [0.0, 2009.0, 199.5], [0.0, 0.0, 1.0]], np.zeros(5),
                               "pinhole")
        self.mount = MountModel.from_config(self.cfg, None)
        self.fx = self.cam.fx

    def tracker(self):
        return make_tracker(self.cfg, self.cam, self.mount)

    def stab(self, w: np.ndarray) -> np.ndarray:
        """(N, 2) stabilised px of world directions (N, 3): u-like to the right, v-like down."""
        w = np.atleast_2d(w)
        return np.stack([-self.fx * w[:, 1] / w[:, 0], -self.fx * w[:, 2] / w[:, 0]], axis=1)

    def to_pixel(self, w: np.ndarray, R: np.ndarray):
        v_c = self.mount.gun_to_cam(R.T @ np.asarray(w, dtype=float))
        if v_c[2] <= 0.0:
            return None
        uv = np.asarray(self.cam.ray_to_pixel(v_c[None, :]), dtype=float).reshape(2)
        if not (2.0 <= uv[0] < W - 2.0 and 2.0 <= uv[1] < H - 2.0):
            return None
        return uv

    def to_world(self, uv: np.ndarray, R: np.ndarray) -> np.ndarray:
        rays_c = np.asarray(self.cam.pixel_to_ray(uv), dtype=float).reshape(-1, 3)
        rays_g = np.asarray(self.mount.cam_to_gun(rays_c), dtype=float).reshape(-1, 3)
        w = rays_g @ R.T
        return w / np.linalg.norm(w, axis=1, keepdims=True)

    def clutter(self, rng, R, t_ns, n=12, patch=None, with_stab=True, mass=(50.0, 200.0)) -> list[Detection]:
        """n specks uniform over the frame, or over a patch of scenery (patch = (centre stab x, y, w, h)); summed
        residual uniform in `mass` (the default: leaf flicker)."""
        if patch is None:
            uv = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n)], axis=1)
            w = self.to_world(uv, R)
            st = self.stab(w)
        else:
            cx, cy, pw, ph = patch
            st = np.stack([rng.uniform(cx - pw / 2, cx + pw / 2, n), rng.uniform(cy - ph / 2, cy + ph / 2, n)], axis=1)
            # stab -> world (inverse gnomonic) -> pixel
            w = np.stack([np.ones(n), -st[:, 0] / self.fx, -st[:, 1] / self.fx], axis=1)
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            v_c = np.asarray(self.mount.gun_to_cam(w @ R), dtype=float).reshape(-1, 3)
            front = v_c[:, 2] > 0.0
            uv = np.asarray(self.cam.ray_to_pixel(v_c[front]), dtype=float).reshape(-1, 2)
            inside = (uv[:, 0] >= 2.0) & (uv[:, 0] < W - 2.0) & (uv[:, 1] >= 2.0) & (uv[:, 1] < H - 2.0)
            st, uv = st[front][inside], uv[inside]
        sizes = rng.uniform(3.0, 6.0, len(uv))
        mass = rng.uniform(mass[0], mass[1], len(uv))
        sign = rng.choice([-1.0, 1.0], len(uv))
        out = []
        for k in range(len(uv)):
            d = float(sizes[k])
            out.append(Detection(t_ns=t_ns, u=float(uv[k, 0]), v=float(uv[k, 1]), area_px=math.pi * d * d / 4,
                                 diameter_px=d, contrast=float(sign[k] * 25.0), snr=6.0, score=CLUTTER,
                                 bbox=(0, 0, 1, 1), stab=(float(st[k, 0]), float(st[k, 1])) if with_stab else None,
                                 mass=float(mass[k])))
        return out

    def clay(self, rng, w, R, t_ns, range_m=30.0, sigma=0.7, with_stab=True, score=CLAY, mass=1500.0):
        uv = self.to_pixel(w, R)
        if uv is None:
            return None
        n = rng.normal(0.0, sigma, 2)
        st = self.stab(w)[0] + n
        uv = uv + n
        ifov = float(np.asarray(self.cam.ifov(uv[None, :])).reshape(-1)[0])
        d = math.hypot(self.cfg.mount.clay_diameter_m / (range_m * ifov), self.cfg.detector.psf_px)
        return Detection(t_ns=t_ns, u=float(uv[0]), v=float(uv[1]), area_px=math.pi * d * d / 4, diameter_px=d,
                         contrast=-40.0, snr=20.0, score=score, bbox=(0, 0, 1, 1),
                         stab=(float(st[0]), float(st[1])) if with_stab else None, mass=float(mass))


def _swing(rate_dps: float, amp_deg: float = 6.0):
    """R_wg(t) of a gun swinging sinusoidally in azimuth with this peak rate and amplitude, slightly raised."""
    if rate_dps <= 0.0:
        R = _frame_from_azel(0.0, 2.0 * D2R)
        return lambda t: R
    amp = amp_deg * D2R
    om = rate_dps * D2R / amp
    return lambda t: _frame_from_azel(amp * math.sin(om * t), 2.0 * D2R + 0.3 * amp * math.sin(0.7 * om * t))


class ClayPath:
    """Constant world angular velocity: direction d0 rotated about a fixed axis at omega."""

    def __init__(self, az0_deg, el0_deg, heading_deg, rate_dps):
        self.d0 = np.asarray(azel_to_dir(az0_deg * D2R, el0_deg * D2R), dtype=float).reshape(3)
        e_az = np.array([-math.sin(az0_deg * D2R), math.cos(az0_deg * D2R), 0.0])
        e_el = np.cross(self.d0, e_az)
        m = math.cos(heading_deg * D2R) * e_az + math.sin(heading_deg * D2R) * e_el
        self.k = np.cross(self.d0, m)
        self.k /= np.linalg.norm(self.k)
        self.om = rate_dps * D2R

    def at(self, t_s: float) -> np.ndarray:
        return _rodrigues(self.d0, self.k, self.om * t_s)


def _clay_tracks(tracker, score=CLAY):
    return [tr for tr in tracker.tracks if tr.last_detection is not None and tr.last_detection.score == score]


# ------------------------------------------------------------------------------------------------ (a) clutter only
def _false_tracks(scene, seed, rate, n_frames=300, n=12, patch=None, with_stab=True, mass=(50.0, 200.0)):
    rng = np.random.default_rng(seed)
    tracker = scene.tracker()
    R_of = _swing(rate)
    born = 0
    for f in range(n_frames):
        t = T0 + f * DT_NS
        R = R_of(f / FPS)
        tracker.update(t, scene.clutter(rng, R, t, n=n, patch=patch, with_stab=with_stab, mass=mass), R)
        born = max(born, tracker.stats["promotions"])
    return born, tracker


@pytest.mark.parametrize("rate", [0.0, 30.0, 80.0])
def test_uniform_clutter_never_makes_a_track(rate):
    scene = Scene()
    total = 0
    for seed in range(20):
        born, tr = _false_tracks(scene, seed, rate)
        total += born
        assert not tr.tracks
    assert total == 0


def test_clutter_in_a_patch_of_scenery_never_makes_a_track():
    scene = Scene()
    patch = (0.0, 0.0, 150.0, 100.0)   # around the bore, fixed in the scenery
    total = 0
    for seed in range(10):
        for rate in (0.0, 30.0):
            born, _ = _false_tracks(scene, 100 + seed, rate, patch=patch)
            total += born
    assert total == 0


def test_clutter_without_stab_never_makes_a_track():
    scene = Scene()
    for seed in range(5):
        born, _ = _false_tracks(scene, 200 + seed, 30.0, with_stab=False)
        assert born == 0


@pytest.mark.parametrize("rate", [0.0, 30.0, 80.0])
def test_strong_clutter_never_makes_a_track(rate):
    """Specks as strong as a clay (summed residual 3000-8000, above init_strong_mass), 4 a frame over the picture
    (the recorded presses: 1 detection a frame that is not a target on average, 5 at the 95th percentile, most of
    them weak): a strong path is believed after init_strong_hits detections, and random ones still never line up."""
    scene = Scene()
    vt = scene.cfg.vision.track
    assert vt.init_strong_hits < vt.init_hits and vt.init_strong_mass <= 3000.0
    total = 0
    for seed in range(10):
        born, tr = _false_tracks(scene, 300 + seed, rate, n=4, mass=(3000.0, 8000.0))
        total += born
    assert total == 0


def test_strong_flicker_in_a_patch_waits_for_init_hits():
    """A crowded patch of strong flicker (12 specks a frame in 150 x 100 px) is where chance lines come from: a path
    through it is not alone (init_strong_radius_px), so it needs init_hits detections like any other. Without the
    isolation test, lines of init_strong_hits made 170 tracks here; with it, no more than with the rule off."""
    patch = (0.0, 0.0, 150.0, 100.0)

    def births(scene):
        return sum(_false_tracks(scene, 400 + seed, 0.0, patch=patch, mass=(3000.0, 8000.0))[0] for seed in range(5))

    on = births(Scene())
    off = births(Scene(["vision.track.init_strong_hits=0"]))
    lonely = births(Scene(["vision.track.init_strong_max_neighbours=1000"]))
    assert on <= off
    assert lonely > 5 * max(off, 1)   # what the isolation test is there for


# ------------------------------------------------------------------------------------------------ (b, c, d) one clay
def _run_clay(scene, seed, rate_dps, follow=False, with_stab=True, clutter=12, dropout=0.15, gun_rate=0.0,
              heading=10.0, mass=1500.0):
    rng = np.random.default_rng(seed)
    tracker = scene.tracker()
    span = 16.0 if not follow else 20.0
    dur = span / rate_dps
    path = ClayPath(-8.0 if not follow else 0.0, 2.0, heading, rate_dps)
    R_of = _swing(gun_rate, 2.0)   # small enough that the clay stays in the picture
    n_frames = int(dur * FPS)
    seen = 0                 # clay detections handed to the tracker so far (since it last entered the picture)
    run = 0                  # consecutive frames without its detection
    fair = 0                 # longest such count while it had no track
    born_at = None
    ids = set()
    stolen = 0
    birth = None
    for f in range(n_frames):
        t = T0 + f * DT_NS
        w = path.at(f / FPS)
        if follow:
            # the gun's bore 1.5 deg behind the clay along its path, with a little hand shake
            lag = path.at(f / FPS - 1.5 / rate_dps)
            az, el = math.atan2(lag[1], lag[0]), math.asin(lag[2])
            R = _frame_from_azel(az + 0.02 * D2R * math.sin(7 * f), el - 0.5 * D2R)
        else:
            R = R_of(f / FPS)
        dets = scene.clutter(rng, R, t, n=clutter, with_stab=with_stab) if clutter else []
        cd = None
        if rng.random() >= dropout:
            cd = scene.clay(rng, w, R, t, with_stab=with_stab, mass=mass)
            if cd is not None:
                dets.append(cd)
                seen += 1
        run = 0 if cd is not None else run + 1
        if born_at is None and (run >= 2 or scene.to_pixel(w, R) is None):
            seen = 0   # a line may miss one frame, not two: its count restarts (as it does out of the picture)
        if born_at is None:
            fair = max(fair, seen)
        rng.shuffle(dets)
        tracker.update(t, dets, R)
        mine = _clay_tracks(tracker)
        for tr in tracker.tracks:
            if tr.track_id in ids and tr.misses == 0 and tr.last_detection.score != CLAY:
                stolen += 1   # the clay's track took a speck (its clay dropped out that frame)
        if mine and born_at is None:
            born_at = seen
            tr = mine[0]
            # velocity at birth against the truth, deg/s on the sphere
            v_true = (path.at(f / FPS + 1e-3) - path.at(f / FPS - 1e-3)) / 2e-3
            ce = math.cos(tr.x[3])
            az_r, el_r = tr.x[1] * ce, tr.x[4]
            e_az = np.array([-math.sin(tr.x[0]), math.cos(tr.x[0]), 0.0])
            d = np.asarray(azel_to_dir(tr.x[0], tr.x[3]), dtype=float).reshape(3)
            e_el = np.cross(d, e_az)
            birth = (math.hypot(az_r - float(v_true @ e_az), el_r - float(v_true @ e_el)) / D2R,
                     math.hypot(az_r, el_r) / D2R)
        ids |= {tr.track_id for tr in mine}
    return dict(fair=fair, born_at=born_at, ids=ids, stolen=stolen, birth=birth, tracker=tracker, seen=seen,
                n_tracks_end=len(tracker.tracks))


@pytest.mark.parametrize("rate", [5.0, 20.0, 60.0])
def test_a_clay_amid_clutter_gets_one_track_quickly(rate):
    scene = Scene()
    k = scene.cfg.vision.track.init_hits
    for seed in range(6):
        r = _run_clay(scene, 1000 + seed, rate)
        assert r["born_at"] is not None, f"seed {seed}: no track ({r['fair']} detections in a row)"
        assert k <= r["born_at"] <= k + 2, f"seed {seed}: born at clay detection {r['born_at']}"
        assert len(r["ids"]) == 1, f"seed {seed}: ids {r['ids']}"
        assert r["stolen"] == 0
        assert r["n_tracks_end"] <= 1
        err, _ = r["birth"]
        assert err <= max(1.5, 0.15 * rate), f"seed {seed}: velocity error at birth {err:.2f} deg/s"
        assert r["tracker"].stats["promotions"] == 1


@pytest.mark.parametrize("rate", [5.0, 20.0, 60.0])
def test_a_strong_clay_is_believed_sooner(rate):
    """A clay whose detections are strong (a clay at 14-30 m: summed residual 3000-40000) is born after
    init_strong_hits of them, one weaker than init_strong_mass after init_hits."""
    scene = Scene()
    vt = scene.cfg.vision.track
    for seed in range(6):
        strong = _run_clay(scene, 1000 + seed, rate, mass=2.0 * vt.init_strong_mass)
        weak = _run_clay(scene, 1000 + seed, rate, mass=0.5 * vt.init_strong_mass)
        assert strong["born_at"] is not None and vt.init_strong_hits <= strong["born_at"] <= vt.init_strong_hits + 2
        assert weak["born_at"] is not None and vt.init_hits <= weak["born_at"] <= vt.init_hits + 2
        assert len(strong["ids"]) == 1 and strong["stolen"] == 0
        err, _ = strong["birth"]
        assert err <= max(2.0, 0.2 * rate), f"seed {seed}: velocity error at birth {err:.2f} deg/s"


@pytest.mark.parametrize("rate", [20.0, 60.0])
@pytest.mark.parametrize("with_stab", [True, False])
def test_a_clay_the_gun_follows(rate, with_stab):
    scene = Scene()
    k = scene.cfg.vision.track.init_hits
    for seed in range(4):
        r = _run_clay(scene, 2000 + seed, rate, follow=True, with_stab=with_stab)
        assert r["born_at"] is not None and k <= r["born_at"] <= k + 2, r["born_at"]
        assert len(r["ids"]) == 1
        assert r["stolen"] == 0
        assert r["birth"][0] <= max(1.5, 0.15 * rate)
        if not with_stab:
            assert r["tracker"].stats["recentre"] > 0   # the fallback reference moved with the gun


@pytest.mark.parametrize("rate", [5.0, 20.0, 60.0])
def test_stab_none_uses_the_world_direction(rate):
    scene = Scene()
    k = scene.cfg.vision.track.init_hits
    for seed in range(4):
        r = _run_clay(scene, 3000 + seed, rate, with_stab=False, gun_rate=30.0)
        assert r["born_at"] is not None and k <= r["born_at"] <= k + 2, r["born_at"]
        assert len(r["ids"]) == 1
        assert r["stolen"] == 0


def test_a_clay_while_the_gun_swings():
    scene = Scene()
    for seed in range(4):
        r = _run_clay(scene, 4000 + seed, 20.0, gun_rate=30.0)
        assert r["born_at"] is not None and len(r["ids"]) == 1 and r["stolen"] == 0


# ------------------------------------------------------------------------------------------------ (e) two clays
def test_two_clays_crossing_keep_their_tracks():
    scene = Scene()
    for seed in range(4):
        rng = np.random.default_rng(5000 + seed)
        tracker = scene.tracker()
        a = ClayPath(-7.0, 4.0, -25.0, 20.0)
        b = ClayPath(-7.0, -2.0, 25.0, 20.0)   # the paths cross about 7 deg on, ~0.35 s in
        R = _swing(0.0)(0.0)
        owner: dict[int, float] = {}
        for f in range(int(0.75 * FPS)):
            t = T0 + f * DT_NS
            dets = scene.clutter(rng, R, t, n=12)
            for path, sc, dt_s in ((a, CLAY, 0.0), (b, CLAY_B, 0.03)):   # b is 30 ms behind a at the crossing
                if rng.random() >= 0.15:
                    d = scene.clay(rng, path.at(f / FPS - dt_s), R, t, score=sc)
                    if d is not None:
                        dets.append(d)
            tracker.update(t, dets, R)
            for tr in tracker.tracks:
                if tr.misses == 0:
                    sc = tr.last_detection.score
                    assert sc in (CLAY, CLAY_B), "a track took clutter"
                    assert owner.setdefault(tr.track_id, sc) == sc, "tracks swapped"
        assert sorted(owner.values()) == [CLAY, CLAY_B], owner
        assert len(tracker.tracks) == 2


# ------------------------------------------------------------------------------------------------ (f) a mover stops
def test_a_mover_that_stops_is_dropped():
    scene = Scene()
    for seed in range(3):
        rng = np.random.default_rng(6000 + seed)
        tracker = scene.tracker()
        path = ClayPath(-5.0, 2.0, 0.0, 20.0)
        R = _swing(0.0)(0.0)
        t_stop = 0.4
        gone_at = None
        had = False
        for f in range(int(1.2 * FPS)):
            t_s = f / FPS
            t = T0 + f * DT_NS
            w = path.at(min(t_s, t_stop))
            dets = scene.clutter(rng, R, t, n=12)
            d = scene.clay(rng, w, R, t, sigma=0.3)
            dets.append(d)
            tracker.update(t, dets, R)
            mine = _clay_tracks(tracker) + [tr for tr in tracker.tracks if tr.misses > 0]
            if t_s < t_stop:
                had = had or bool(mine)
            elif not tracker.tracks and gone_at is None:
                gone_at = t_s
            if gone_at is not None:
                assert not tracker.tracks, f"a track came back at {t_s:.2f} s"
        assert had
        assert gone_at is not None and gone_at - t_stop <= 0.22, gone_at


# ------------------------------------------------------------------------------------------------ (g) solver surface
def test_solver_interface_matches_the_parent_tracker():
    scene = Scene()
    legacy = make_tracker(load_config(sets=["vision.engine=legacy"]), scene.cam, scene.mount)
    motion = scene.tracker()
    assert type(legacy) is CartesianClayTracker and isinstance(motion, MotionTracker)
    rng = np.random.default_rng(7)
    path = ClayPath(-6.0, 2.0, 15.0, 20.0)
    R_of = _swing(30.0)
    for f in range(30):
        t = T0 + f * DT_NS
        R = R_of(f / FPS)
        d = scene.clay(rng, path.at(f / FPS), R, t, sigma=0.3)
        legacy.update(t, [d], R)
        motion.update(t, [d], R)
    bore = R @ scene.mount.bore_dir_g
    b_l, b_m = legacy.best(bore_dir_w=bore, t_ns=t), motion.best(bore_dir_w=bore, t_ns=t)
    assert b_m is not None and b_l is not None and b_m is motion.tracks[0]
    assert motion.best() is b_m
    gate = motion._gate_min
    for dt_ms in (0, 50, 150):
        tt = t + dt_ms * 1_000_000
        dl, dm = legacy.predict_dir(b_l, tt), motion.predict_dir(b_m, tt)
        ang = math.acos(min(1.0, float(dl @ dm)))
        assert ang <= gate, (dt_ms, ang / D2R)
        truth = path.at((f + dt_ms / 20.0) / FPS)
        assert math.acos(min(1.0, float(truth @ dm))) <= gate
        az, el, sig = motion.predict(b_m, tt)
        assert math.isfinite(az) and math.isfinite(el) and 0.0 < sig < math.radians(1.0)
        xyz = motion.predict_xyz(b_m, tt)
        assert np.allclose(xyz, dm)
    assert abs(motion.angular_rate(b_m) - 20.0 * D2R) < 2.0 * D2R
    assert motion.surely_slow(b_m) in (False, True)
    # (born credited with fire.min_track_hits - 2 of its init_hits detections: see MotionTracker._promote)
    credit = scene.cfg.vision.track.init_hits - (scene.cfg.fire.min_track_hits - 2)
    assert b_m.confirmed and b_m.hits == 30 - max(0, credit) and b_m.first_t_ns == T0
    motion.reset()
    assert not motion.tracks and not motion.chains and motion.stats["promotions"] == 0


def test_birth_state_is_confirmed_and_two_hits_short_of_the_solvers_bar():
    scene = Scene()
    tracker = scene.tracker()
    rng = np.random.default_rng(8)
    path = ClayPath(-6.0, 2.0, 0.0, 20.0)
    R = _swing(0.0)(0.0)
    k = scene.cfg.vision.track.init_hits
    for f in range(k):
        t = T0 + f * DT_NS
        tracker.update(t, [scene.clay(rng, path.at(f / FPS), R, t, sigma=0.3)], R)
        if f < k - 1:
            assert not tracker.tracks
    (tr,) = tracker.tracks
    # two detections short of fire.min_track_hits: the solver cannot fire on the frame a path is first believed
    assert tr.hits == min(k, scene.cfg.fire.min_track_hits - 2)
    assert tr.confirmed and tr.first_t_ns == T0 and tr.t_ns == t and tr.misses == 0
    assert not tracker.chains and not tracker.seeds
    for f in range(k, k + 2):
        assert tr.hits < scene.cfg.fire.min_track_hits
        t = T0 + f * DT_NS
        tracker.update(t, [scene.clay(rng, path.at(f / FPS), R, t, sigma=0.3)], R)
        (tr,) = tracker.tracks
    assert tr.hits == scene.cfg.fire.min_track_hits


def test_a_target_the_gun_follows_is_tracked_however_fast_both_turn():
    """150 deg/s against the scenery is over init_max_speed_dps. With the gun following (the bore a degree behind
    it) it moves a few pixels a frame in the picture and gets its track; crossing the picture of a still gun at
    that rate (105 px a frame, gone in six frames) it gets none."""
    import dataclasses

    scene = Scene()
    rate = 150.0
    assert rate > scene.cfg.vision.track.init_max_speed_dps
    rng = np.random.default_rng(3)
    path = ClayPath(-22.0, 2.0, 0.0, rate)
    tracker = scene.tracker()

    def with_dir(d, w):   # as the motion detector gives it: the direction in the stabilised frame
        return dataclasses.replace(d, stab_dir=tuple(float(x) for x in w))

    for f in range(15):
        t = T0 + f * DT_NS
        lag = path.at(f / FPS - 1.0 / rate)
        R = _frame_from_azel(math.atan2(lag[1], lag[0]), math.asin(lag[2]))
        d = scene.clay(rng, path.at(f / FPS), R, t, sigma=0.3, range_m=3.0)   # 8 m/s, as thrown from close by
        assert d is not None
        tracker.update(t, [with_dir(d, path.at(f / FPS))], R)
    assert len(_clay_tracks(tracker)) == 1 and tracker.stats["promotions"] == 1

    tracker = scene.tracker()
    R = _frame_from_azel(0.0, 2.0 * D2R)
    path = ClayPath(-8.5, 2.0, 0.0, rate)
    seen = 0
    for f in range(10):
        t = T0 + f * DT_NS
        d = scene.clay(rng, path.at(f / FPS), R, t, sigma=0.3, range_m=3.0)
        seen += d is not None
        tracker.update(t, [with_dir(d, path.at(f / FPS))] if d is not None else [], R)
    assert seen >= scene.cfg.vision.track.init_hits, "it was in the picture long enough to be born"
    assert not tracker.tracks and tracker.stats["promotions"] == 0


@pytest.mark.parametrize("az0", [0.0, 70.0, 130.0])
def test_a_stabilised_direction_gives_a_track_wherever_the_gun_points(az0):
    """Detection.stab_dir (a unit direction in the detector's stabilised frame; the true world direction here) is
    what the path is built from when it is there. 70 degrees from the axis of the detector's plane `stab` is
    stretched 8 times, at 130 degrees it does not exist: it is given as garbage here and must not matter."""
    import dataclasses

    scene = Scene()
    k = scene.cfg.vision.track.init_hits
    rate = 20.0
    for seed in range(3):
        rng = np.random.default_rng(40 + seed)
        path = ClayPath(az0, 2.0, 10.0, rate)
        tracker = scene.tracker()
        born = None
        for f in range(40):
            t = T0 + f * DT_NS
            w = path.at(f / FPS)
            lag = path.at(f / FPS - 1.5 / rate)
            R = _frame_from_azel(math.atan2(lag[1], lag[0]), math.asin(lag[2]) - 0.5 * D2R)
            d = scene.clay(rng, w, R, t, sigma=0.4)
            assert d is not None
            # (the centroid noise of the detection, 0.4 px, as an angle on the direction)
            n = rng.normal(0.0, 0.4 / scene.fx, 3)
            sdir = w + n - w * float(w @ n)
            sdir /= np.linalg.norm(sdir)
            d = dataclasses.replace(d, stab=(1e7 * (f % 3 - 1), float("nan")), stab_dir=tuple(float(x) for x in sdir))
            dets = [dataclasses.replace(c, stab_dir=tuple(scene.to_world(np.array([[c.u, c.v]]), R)[0]))
                    for c in scene.clutter(rng, R, t, n=6)] + [d]
            tracker.update(t, dets, R)
            if born is None and _clay_tracks(tracker):
                born = f + 1
        assert born is not None and k <= born <= k + 2, (az0, seed, born)
        assert len(_clay_tracks(tracker)) == 1 and tracker.stats["promotions"] == 1


# ------------------------------------------------------------------------------------------------ (h) legacy
def test_legacy_engine_keeps_the_old_tracker():
    scene = Scene()
    cfg = load_config(sets=["vision.engine=legacy"])
    assert type(make_tracker(cfg, scene.cam, scene.mount)) is CartesianClayTracker
    assert type(make_tracker(load_config(), scene.cam, scene.mount)) is MotionTracker
    ang = load_config(sets=["tracker.model=angular"])
    assert not isinstance(make_tracker(ang, scene.cam, scene.mount), CartesianClayTracker)


# ------------------------------------------------------------------------------------------------ cost
def test_per_frame_cost_is_small():
    scene = Scene()
    rng = np.random.default_rng(9)
    tracker = scene.tracker()
    path = ClayPath(-8.0, 2.0, 10.0, 20.0)
    R = _swing(0.0)(0.0)
    frames = []
    for f in range(40):
        t = T0 + f * DT_NS
        dets = scene.clutter(rng, R, t, n=12)
        dets.append(scene.clay(rng, path.at(f / FPS), R, t))
        frames.append((t, dets))
    for t, dets in frames[:10]:
        tracker.update(t, dets, R)
    t0 = time.perf_counter()
    for t, dets in frames[10:]:
        tracker.update(t, dets, R)
    per = (time.perf_counter() - t0) / 30
    assert per < 3e-3, f"{per * 1e3:.2f} ms a frame"
