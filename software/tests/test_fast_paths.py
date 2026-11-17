"""The plain-float fast paths of the frame loop give the same numbers as the array code they replace.

On a Pi 3B every NumPy call on a 3-vector or a 3x3 costs 15-50 us whatever it computes, so the per-detection
and per-track arithmetic (camera rays, tracker update, the solver's look ahead) is written in plain floats.
Each test here states the reference with ordinary matrix algebra and compares.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from tbot.config import load_config
from tbot.optics import CameraModel, MountModel
from tbot.types import Detection
from tbot.vision.clutter import StaticFilter
from tbot.vision.tracker import ClayTracker
from tbot.vision.tracker3d import CartesianClayTracker, _azel_jacobian, _inv_small, _jac5, _proj2

W, H = 640, 400


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _cams():
    K = [[2009.0, 0.0, 319.5], [0.0, 2011.0, 199.5], [0.0, 0.0, 1.0]]
    return [
        CameraModel(W, H, K, [0.0] * 5, "pinhole"),
        CameraModel(W, H, K, [-0.31, 0.12, 0.0009, -0.0004, 0.02], "pinhole"),
        CameraModel(W, H, K, [-0.31, 0.12, 0.0009, -0.0004, 0.02, 0.01, -0.02, 0.003], "pinhole"),
        CameraModel(W, H, [[420.0, 0.0, 319.5], [0.0, 420.0, 199.5], [0.0, 0.0, 1.0]],
                    [-0.03, 0.004, -0.001, 0.0002], "fisheye"),
    ]


def test_single_pixel_ray_matches_the_array_path():
    pts = np.array([[0.0, 0.0], [639.0, 399.0], [321.3, 187.7], [15.5, 380.25], [600.0, 12.0]])
    for cam in _cams():
        batch = np.asarray(cam.pixel_to_ray(pts)).reshape(-1, 3)   # OpenCV / vectorised Newton
        for p, ref in zip(pts, batch):
            ray = cam.ray_xyz(float(p[0]), float(p[1]))
            assert ray is not None
            assert np.allclose(ray, ref, atol=1e-12)
            assert np.allclose(cam.pixel_to_ray(p), ref, atol=1e-12)
            assert math.isclose(cam.ifov1(float(p[0]), float(p[1])), float(cam.ifov(pts)[list(map(tuple, pts)).index(tuple(p))]),
                                rel_tol=1e-9)


def test_pinhole_with_thin_prism_terms_stays_with_opencv():
    K = [[2009.0, 0.0, 319.5], [0.0, 2009.0, 199.5], [0.0, 0.0, 1.0]]
    D = [-0.2, 0.1, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0]
    cam = CameraModel(W, H, K, D, "pinhole")
    assert cam.ray_xyz(100.0, 100.0) is None
    und = cv2.undistortPoints(np.array([[[100.0, 100.0]]]), cam.K, cam.D).reshape(2)
    ref = np.array([und[0], und[1], 1.0]) / math.sqrt(und[0] ** 2 + und[1] ** 2 + 1.0)
    assert np.allclose(cam.pixel_to_ray(np.array([100.0, 100.0])), ref, atol=1e-12)
    assert cam.ifov1(100.0, 100.0) == pytest.approx(float(cam.ifov(np.array([100.0, 100.0]))), rel=1e-9)


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _det(u, v, d=6.0):
    return Detection(t_ns=0, u=u, v=v, area_px=20.0, diameter_px=d, contrast=-40.0, snr=30.0, score=0.9,
                     bbox=(int(u) - 3, int(v) - 3, 6, 6) if math.isfinite(u) else (0, 0, 6, 6))


def test_measuring_a_few_detections_matches_the_array_path(cfg):
    cam = CameraModel.nominal(cfg, W, H)
    mount = MountModel.from_config(cfg, {"rpy_deg": [0.4, -1.9, -1.7]})
    trk = ClayTracker(cfg, cam, mount)
    dets = [_det(321.3, 187.7), _det(12.0, 390.5), _det(600.2, 20.1)]
    R = _rot(0.7, 0.35, -0.1)
    fast = trk._measure_few(dets, R)
    assert fast is not None
    # the reference: the same quantities through the array functions
    uv = np.array([[d.u, d.v] for d in dets])
    rays_w = np.asarray(mount.cam_to_gun(np.asarray(cam.pixel_to_ray(uv)).reshape(-1, 3))) @ R.T
    az = np.arctan2(rays_w[:, 1], rays_w[:, 0])
    el = np.arcsin(rays_w[:, 2] / np.linalg.norm(rays_w, axis=1))
    ifov = np.asarray(cam.ifov(uv))
    sig = float(cfg.tracker.meas_sigma_px) * ifov
    for got, ref in zip(fast, (az, el, sig / np.maximum(np.cos(el), 0.2), sig, ifov)):
        assert np.allclose(got, ref, rtol=1e-9, atol=1e-13)
    # a non-finite pixel is left to the array path
    assert trk._measure_few([_det(float("nan"), 3.0)], R) is None


def test_small_matrix_helpers():
    p = np.array([12.0, -7.0, 4.5])
    J = _azel_jacobian(p)
    j = _jac5(*p)
    assert np.allclose([j[0], j[1], 0.0, j[2], j[3], j[4]], J.ravel())
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3))
    C = A @ A.T + np.eye(3)
    M = J @ C @ J.T
    assert np.allclose(_proj2(j, C.tolist()), (M[0, 0], M[0, 1], M[1, 1]))
    for n in (2, 3):
        S = (A @ A.T + np.diag([1e-8, 1e-8, 2.0]))[:n, :n]
        assert np.allclose(_inv_small(S.tolist()), np.linalg.inv(S), rtol=1e-9)
    assert _inv_small([[1.0, 2.0], [2.0, 4.0]]) is None


def _reference_update(sp, Sp, az_m, el_m, sig_az, sig_el, r_size, r_sigma):
    """The EKF update written out with the full 9-column measurement matrix (Joseph form)."""
    p = sp[0:3]
    az_p, el_p = math.atan2(p[1], p[0]), math.atan2(p[2], math.hypot(p[0], p[1]))
    H = np.zeros((2, 9))
    H[:, 0:3] = _azel_jacobian(p)
    z = np.array([(az_m - az_p + math.pi) % (2 * math.pi) - math.pi, el_m - el_p])
    R = np.diag([sig_az ** 2, sig_el ** 2])
    if r_size is not None:
        r_p = float(np.linalg.norm(p))
        Hr = np.zeros((1, 9))
        Hr[0, 0:3] = p / r_p
        H = np.vstack([H, Hr])
        z = np.append(z, r_size - r_p)
        R = np.diag([sig_az ** 2, sig_el ** 2, r_sigma ** 2])
    K = Sp @ H.T @ np.linalg.inv(H @ Sp @ H.T + R)
    A = np.eye(9) - K @ H
    S = A @ Sp @ A.T + K @ R @ K.T
    return sp + K @ z, 0.5 * (S + S.T)


def test_tracker_update_and_prediction_match_the_matrix_form(cfg):
    cam = CameraModel.nominal(cfg, W, H)
    mount = MountModel.from_config(cfg, None)
    trk = CartesianClayTracker(cfg, cam, mount)
    R = _rot(0.2, 0.3, 0.0)
    trk.update(1_000_000_000, [_det(300.0, 180.0, 7.0)], R)
    trk.update(1_010_000_000, [_det(306.0, 178.5, 7.2)], R)
    trk.update(1_020_000_000, [_det(312.2, 177.1, 7.1)], R)
    tr = trk.tracks[0]
    assert tr.hits == 3
    sp, Sp = trk._propagate(tr, tr.t_ns + 10_000_000)
    det = _det(318.1, 175.4, 7.3)
    az, el, sig_az, sig_el, ifov = (float(v[0]) for v in trk._measure([det], R))
    for d in (det, _det(318.1, 175.4, 0.0)):   # with and without a size (range) measurement
        r_size = trk._size_range(d, ifov)
        ref_s, ref_S = _reference_update(sp, Sp, az, el, sig_az, sig_el, r_size,
                                         None if r_size is None else float(cfg.tracker.size_range_sigma_frac) * r_size)
        s, S = trk._ekf_update(sp.copy(), Sp.copy(), az, el, sig_az, sig_el, d, ifov)
        assert np.allclose(s, ref_s, rtol=1e-9, atol=1e-12)
        assert np.allclose(S, ref_S, rtol=1e-8, atol=1e-14)

    # prediction: the position block of F S F^T + Q through the Jacobian
    t = tr.t_ns + 137_000_000
    dt = 0.137
    f, q = np.eye(9), trk._q3 * dt ** 5 / 20.0
    for i in range(3):
        f[i, 3 + i], f[i, 6 + i], f[3 + i, 6 + i] = dt, 0.5 * dt * dt, dt
    p = (f @ tr.s)[0:3]
    cov = (f @ tr.S @ f.T)[0:3, 0:3] + q * np.eye(3)
    J = _azel_jacobian(p)
    C = J @ cov @ J.T
    el_ref = math.atan2(p[2], math.hypot(p[0], p[1]))
    az_p, el_p, sigma = trk.predict(tr, t)
    assert az_p == pytest.approx(math.atan2(p[1], p[0]), abs=1e-12)
    assert el_p == pytest.approx(el_ref, abs=1e-12)
    assert sigma == pytest.approx(math.sqrt(math.cos(el_ref) ** 2 * C[0, 0] + C[1, 1]), rel=1e-9)
    assert np.allclose(trk.predict_dir(tr, t), p / np.linalg.norm(p), atol=1e-13)
    v = tr.s[3:6]
    u = tr.s[0:3] / np.linalg.norm(tr.s[0:3])
    assert trk.angular_rate(tr) == pytest.approx(np.linalg.norm(v - (v @ u) * u) / np.linalg.norm(tr.s[0:3]), rel=1e-9)
    # the range view of the state
    assert tr.range_m == pytest.approx(np.linalg.norm(tr.s[0:3]))
    assert tr.range_sigma_m >= math.sqrt(u @ tr.S[0:3, 0:3] @ u) - 1e-12


def test_followed_targets_go_first_and_the_rest_are_limited(cfg):
    cam = CameraModel.nominal(cfg, W, H)
    mount = MountModel.from_config(cfg, None)
    filt = StaticFilter(cfg, cam, mount)
    R = np.eye(3)
    uv = np.array([[100.0, 100.0], [320.0, 200.0], [500.0, 300.0], [50.0, 350.0], [600.0, 50.0]])
    areas = np.array([200.0, 6.0, 150.0, 90.0, 40.0])   # the followed target (index 1) is the smallest
    wh = np.array([16.0, 3.0, 14.0, 10.0, 7.0])
    target_w = (R @ filt._R_gc) @ np.asarray(cam.pixel_to_ray(np.array([322.0, 201.0])))

    filt.begin(1_000_000_000, R)
    plain = filt.filter(uv, areas, wh, wh)
    assert sorted(plain.tolist()) == [0, 1, 2, 3, 4] and plain[0] != 1   # by solidity and size: the target is last

    filt.reset()
    filt.begin(1_000_000_000, R, focus_dirs_w=[target_w], others_max=2)
    kept = filt.filter(uv, areas, wh, wh)
    assert kept[0] == 1 and len(kept) == 3 and filt.n_focus == 1
    assert kept[1:].tolist() == [k for k in plain.tolist() if k != 1][:2]   # the best two of the others, in order

    filt.reset()
    filt.begin(1_000_000_000, R, focus_dirs_w=[target_w], others_max=0)       # 0 = no limit, the target still first
    assert filt.filter(uv, areas, wh, wh).tolist() == [1] + [k for k in plain.tolist() if k != 1]

    # a wide zone (the tracker unsure of its prediction) reaches a candidate 60 px away; the default 25 px does not
    far_w = (R @ filt._R_gc) @ np.asarray(cam.pixel_to_ray(np.array([380.0, 200.0])))
    filt.reset()
    filt.begin(1_000_000_000, R, focus_dirs_w=[far_w], others_max=1)
    kept = filt.filter(uv, areas, wh, wh)
    # ... and a frame in which the main target is not where it should be is searched as usual: no limit
    assert kept.tolist() == plain.tolist() and filt.n_focus == 0
    filt.reset()
    filt.begin(1_000_000_000, R, focus_dirs_w=[far_w], others_max=1, focus_radius_rad=[80.0 * filt.ifov])
    kept = filt.filter(uv, areas, wh, wh)
    assert kept[0] == 1 and len(kept) == 2 and filt.n_focus == 1

    # two targets, only the second one found: the main one is missing, so no limit
    filt.reset()
    filt.begin(1_000_000_000, R, focus_dirs_w=[far_w, target_w], others_max=1)
    kept = filt.filter(uv, areas, wh, wh)
    assert kept[0] == 1 and len(kept) == 5
