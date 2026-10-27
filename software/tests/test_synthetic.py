"""Synthetic camera + IMU [sim slice]: does the rendered scene agree with the ground truth it claims?"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbot.camera.synthetic import T0_NS, SyntheticSource
from tbot.config import Calibration, load_config, resolve_path
from tbot.sim.scenario import SCENARIOS, make_scenario

G_MPS2 = 9.80665


def make_cfg(**overrides):
    """Default config with a low frame rate (fewer frames per scenario) plus dotted overrides."""
    cfg = load_config()
    cfg.camera.fps = 30.0
    for key, value in overrides.items():
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, value)
    return cfg


def make_calib(cfg) -> Calibration:
    """Empty calibration: the nominal camera model, independent of what is in calib/."""
    return Calibration(dir=resolve_path(cfg.paths.calib_dir))


def open_source(cfg, scenario, seed=1) -> SyntheticSource:
    src = SyntheticSource(cfg, make_calib(cfg), scenario=scenario, seed=seed)
    src.start()
    return src


def all_frames(src):
    out = []
    while True:
        frame = src.read()
        if frame is None:
            return out
        out.append(frame)


def closest_frame(src):
    """Frame with the clay visible in the image at its shortest range, and its truth."""
    best = None
    while True:
        frame = src.read()
        if frame is None:
            return best
        truth = src.truth(frame.t_exposure_ns)
        if truth["clay_px"] is None:
            continue
        if best is None or truth["clay_range_m"] < best[1]["clay_range_m"]:
            best = (frame, truth)


def blob_stats(image, u, v, expected_d_px):
    """Centroid and major axis of the clay blob, measured against a local median background."""
    gray = image.astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    half = int(max(4, round(2.0 * expected_d_px)))
    x0, x1 = int(round(u)) - half, int(round(u)) + half + 1
    y0, y1 = int(round(v)) - half, int(round(v)) + half + 1
    window = gray[y0:y1, x0:x1]
    contrast = np.median(window) - window  # the clay is darker than the sky in mono
    peak = float(contrast.max())
    weight = np.where(contrast > 0.35 * peak, contrast, 0.0)
    total = float(weight.sum())
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mu = float((weight * xs).sum() / total)
    mv = float((weight * ys).sum() / total)
    cxx = float((weight * (xs - mu) ** 2).sum() / total)
    cyy = float((weight * (ys - mv) ** 2).sum() / total)
    cxy = float((weight * (xs - mu) * (ys - mv)).sum() / total)
    lam = 0.5 * (cxx + cyy) + math.sqrt(0.25 * (cxx - cyy) ** 2 + cxy * cxy)
    return mu, mv, peak, 4.0 * math.sqrt(max(lam, 0.0))


def rodrigues(omega, dt):
    theta = float(np.linalg.norm(omega)) * dt
    if theta < 1e-12:
        return np.eye(3)
    k = omega / np.linalg.norm(omega)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rot_error_deg(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A.T @ B) - 1.0) * 0.5, -1.0, 1.0))))


# ------------------------------------------------------------------------------------ determinism
def test_same_seed_renders_identical_frames_and_other_seeds_differ():
    cfg = make_cfg()
    a = open_source(cfg, "trap_straightaway", seed=5)
    b = open_source(cfg, "trap_straightaway", seed=5)
    c = open_source(cfg, "trap_straightaway", seed=6)
    for _ in range(6):
        fa, fb, fc = a.read(), b.read(), c.read()
        assert np.array_equal(fa.image, fb.image)
        assert not np.array_equal(fa.image, fc.image)
        assert fa.t_exposure_ns == fb.t_exposure_ns == fc.t_exposure_ns



# ------------------------------------------------------------------------------------ geometry
def test_clay_blob_lands_where_truth_says_and_has_the_right_angular_size():
    cfg = make_cfg()
    src = open_source(cfg, "skeet_crossing", seed=1)
    frame, truth = closest_frame(src)
    u, v = truth["clay_px"]
    ifov = float(np.asarray(src.cam.ifov(np.array([[u, v]], dtype=np.float64))).ravel()[0])
    expected_d = cfg.mount.clay_diameter_m / truth["clay_range_m"] / ifov
    mu, mv, peak, major = blob_stats(frame.image, u, v, expected_d)
    assert peak > 4.0 * cfg.sim.noise_sigma, "the clay must stand out of the sensor noise"
    assert math.hypot(mu - u, mv - v) < 1.0
    assert abs(major - expected_d) <= 0.4 * expected_d


def test_clay_grows_as_it_approaches_and_truth_matches_its_physics():
    cfg = make_cfg()
    src = open_source(cfg, "skeet_crossing", seed=1)
    samples = []
    for frame in all_frames(src):
        truth = src.truth(frame.t_exposure_ns)
        if truth["clay_px"] is not None:
            samples.append((truth["clay_range_m"], truth["clay_pos_w"], truth["clay_dir_w"]))
    assert len(samples) > 20
    for rng_m, pos, direction in samples:
        assert 5.0 < rng_m < 45.0
        assert np.allclose(direction * rng_m, pos, atol=1e-9)
        assert pos[2] >= 0.0  # visible means above the horizon
    ranges = [s[0] for s in samples]
    assert min(ranges) < ranges[0]  # the crossing target closes in before going away again


def test_clay_pixel_is_none_when_the_clay_is_outside_the_image():
    cfg = make_cfg()
    src = open_source(cfg, "static_gun", seed=1)
    visible, in_image = 0, 0
    for frame in all_frames(src):
        truth = src.truth(frame.t_exposure_ns)
        visible += int(truth["visible"])
        in_image += int(truth["clay_px"] is not None)
        if truth["clay_px"] is not None:
            u, v = truth["clay_px"]
            assert 0 <= u < src.width and 0 <= v < src.height
    assert visible > in_image > 10  # the clamped gun loses the clay out of the frame part of the pass


# ------------------------------------------------------------------------------------ imu
def test_integrating_the_synthetic_gyro_reproduces_the_attitude():
    cfg = make_cfg()
    cfg.sim.gyro_noise_dps = 0.0
    src = open_source(cfg, "skeet_crossing", seed=1)
    imu = src.imu()
    imu.start()
    samples = []
    while True:
        frame = src.read()
        if frame is None:
            break
        new = imu.read()
        assert all(s.t_ns <= frame.t_capture_ns for s in new)
        samples.extend(new)
    assert len(samples) > 500
    times = [s.t_ns for s in samples]
    assert times == sorted(set(times))  # every sample exactly once, in order

    R = src.scenario.R_wg((samples[0].t_ns - T0_NS) * 1e-9)
    for a, b in zip(samples, samples[1:]):
        R = R @ rodrigues(0.5 * (a.gyro + b.gyro), (b.t_ns - a.t_ns) * 1e-9)
    truth = src.scenario.R_wg((samples[-1].t_ns - T0_NS) * 1e-9)
    assert rot_error_deg(R, truth) < 0.5


def test_accelerometer_reads_gravity_in_the_gun_frame():
    cfg = make_cfg()
    src = open_source(cfg, "static_gun", seed=2)
    imu = src.imu()
    imu.start()
    src.read()
    samples = imu.read()
    assert samples
    R = src.scenario.R_wg(0.0)
    for s in samples:
        assert np.allclose(s.accel, R.T @ np.array([0.0, 0.0, G_MPS2]), atol=0.1)
        assert abs(float(np.linalg.norm(s.accel)) - G_MPS2) < 0.1


def test_imu_is_silent_before_it_is_started():
    cfg = make_cfg()
    src = open_source(cfg, "no_clay", seed=1)
    imu = src.imu()
    src.read()
    assert imu.read() == []
    imu.start()
    assert imu.read()


# ------------------------------------------------------------------------------------ scenarios
def test_every_scenario_builds_and_only_no_clay_is_empty():
    cfg = make_cfg()
    for name in SCENARIOS:
        for seed in (1, 2, 7):
            sc = make_scenario(name, cfg, seed)
            assert sc.duration_s > 1.0 and sc.n_frames > 10
            if name == "no_clay":
                assert not sc.clay_present
                assert sc.clay_pos_w(1.0) is None
                continue
            assert sc.clay_present
            assert 12.0 <= float(np.linalg.norm(sc.launch_vel_w)) <= 30.0
            assert sc.visible_until_s > sc.launch_t_s + 0.8
            assert abs(float(np.linalg.norm(sc.clay_axis_w)) - 1.0) < 1e-9


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        make_scenario("free_fall", make_cfg(), 1)


def test_no_clay_never_reports_a_target():
    cfg = make_cfg()
    src = open_source(cfg, "no_clay", seed=3)
    frames = all_frames(src)
    assert len(frames) > 20
    for frame in frames:
        truth = src.truth(frame.t_exposure_ns)
        assert truth["visible"] is False
        assert truth["clay_px"] is None
        assert truth["clay_pos_w"] is None


def test_static_gun_holds_its_attitude_and_reports_no_rotation():
    cfg = make_cfg()
    src = open_source(cfg, "static_gun", seed=1)
    imu = src.imu()
    imu.start()
    R0 = None
    gyro = []
    while True:
        frame = src.read()
        if frame is None:
            break
        R = src.truth(frame.t_exposure_ns)["R_wg"]
        R0 = R if R0 is None else R0
        assert np.allclose(R, R0, atol=1e-12)
        gyro.extend(s.gyro for s in imu.read())
        assert np.allclose(src.scenario.omega_g(frame.seq / cfg.camera.fps), 0.0, atol=1e-12)
    assert len(gyro) > 500
    assert np.all(np.abs(np.mean(np.array(gyro), axis=0)) < 1e-3)  # noise only, no bias


def test_the_shooter_swings_through_the_clay():
    cfg = make_cfg()
    sc = make_scenario("skeet_crossing", cfg, 1)
    lead = []
    for k in range(0, 260):
        t = k * 0.01
        if not (sc.clay_visible(t) and sc.clay_visible(t + 0.01)):
            continue
        d = sc.clay_dir_w(t)
        nxt = sc.clay_dir_w(t + 0.01)
        along = nxt - float(nxt @ d) * d
        norm = float(np.linalg.norm(along))
        if norm < 1e-9:
            continue
        # signed angle of the bore ahead of the clay along the clay's own track
        lead.append(float((sc.bore_dir_w(t) - d) @ (along / norm)))
    assert len(lead) > 50
    assert min(lead) < -0.02, "the bore must start behind the clay"
    assert max(lead) > 0.05, "the bore must pull ahead of the clay"
    crossings = [i for i in range(1, len(lead)) if lead[i - 1] < 0.0 <= lead[i]]
    assert crossings, "the bore must sweep through the clay direction during the visible flight"


def test_gun_attitude_has_no_roll_and_matches_its_angular_rate():
    cfg = make_cfg()
    sc = make_scenario("skeet_outgoing", cfg, 4)
    for t in (0.101, 0.503, 1.007, 1.701, 2.399):  # off the interpolation knots
        R = sc.R_wg(t)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        left = R[:, 1]  # gun-left stays horizontal when the roll is zero
        assert abs(float(left[2])) < 1e-12
        h = 1e-4
        numeric = (sc.R_wg(t + h) - sc.R_wg(t - h)) / (2.0 * h)
        skew = R.T @ numeric
        omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        assert np.allclose(omega, sc.omega_g(t), atol=1e-4)


# ------------------------------------------------------------------------------------ rendering
def test_frames_are_uint8_of_the_configured_shape_with_a_dark_barrel():
    cfg = make_cfg()
    src = open_source(cfg, "skeet_crossing", seed=1)
    frame = src.read()
    assert frame.image.dtype == np.uint8
    assert frame.image.shape == (cfg.camera.height, cfg.camera.width)
    assert src.mask.shape == frame.image.shape
    barrel = src.mask == 0
    assert barrel.any(), "the barrel is in the camera's field of view (layout.md geometry)"
    image = frame.image.astype(np.float64)
    assert np.median(image[barrel]) < 0.5 * np.median(image[~barrel])


def test_color_mode_renders_an_orange_clay_on_a_blue_sky():
    cfg = make_cfg(**{"sim.render_color": True})
    frame, truth = closest_frame(open_source(cfg, "skeet_crossing", seed=1))
    assert frame.image.shape == (cfg.camera.height, cfg.camera.width, 3)
    assert frame.image.dtype == np.uint8
    u, v = truth["clay_px"]
    b, g, r = (int(x) for x in frame.image[int(round(v)), int(round(u))])
    assert r > g > b, f"clay pixel should be orange, got BGR {(b, g, r)}"


def test_frame_timing_follows_the_configured_rate_and_latency():
    cfg = make_cfg()
    src = open_source(cfg, "no_clay", seed=1)
    expected_gap = int(round(0.5 * cfg.camera.exposure_us * 1e3)) + int(round(cfg.latency.exposure_to_user_ms * 1e6))
    period_ns = 1e9 / cfg.camera.fps
    frames = all_frames(src)
    assert len(frames) == src.scenario.n_frames
    assert src.read() is None  # end of stream stays the end of stream
    assert src.dropped == 0
    for i, frame in enumerate(frames):
        assert frame.seq == i
        assert frame.t_exposure_ns == T0_NS + int(round(i * period_ns))
        assert frame.t_capture_ns - frame.t_exposure_ns == expected_gap
    assert (frames[-1].t_exposure_ns - T0_NS) * 1e-9 <= src.scenario.duration_s + 1e-9
