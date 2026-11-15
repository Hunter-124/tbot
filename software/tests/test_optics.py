"""Behaviour of tbot.optics: projection, mount geometry, barrel masking, helpers."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from tbot.config import Calibration, load_config
from tbot.optics import (
    CameraModel,
    MountModel,
    azel_to_dir,
    barrel_mask,
    combined_mask,
    dir_to_azel,
    range_from_size,
    rot_rpy,
)

FISHEYE_D = np.array([-0.02, 0.01, -0.005, 0.001])  # realistic M12 wide-angle residuals
PINHOLE_D = np.array([-0.05, 0.01, 0.001, -0.001, 0.0])


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _pinhole_cfg(cfg):
    return dataclasses.replace(cfg, optics=dataclasses.replace(cfg.optics, model="pinhole"))


def _pixel_grid(w, h, n_u=21, n_v=13):
    uu, vv = np.meshgrid(np.linspace(0.0, w - 1, n_u), np.linspace(0.0, h - 1, n_v))
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


# ------------------------------------------------------------------------------ intrinsics prior
def test_nominal_focal_length_follows_sensor_binning(cfg):
    full = CameraModel.nominal(cfg, 1280, 800)
    binned = CameraModel.nominal(cfg, 640, 400)
    assert full.fx == pytest.approx(933.3333, abs=1e-3)
    assert full.fy == pytest.approx(933.3333, abs=1e-3)
    assert binned.fx == pytest.approx(466.6667, abs=1e-3)
    assert binned.fy == pytest.approx(466.6667, abs=1e-3)
    assert (binned.cx, binned.cy) == (319.5, 199.5)
    assert (full.cx, full.cy) == (639.5, 399.5)
    assert binned.model == "fisheye"
    assert binned.D.shape == (4,)
    assert np.all(binned.D == 0.0)


def test_nominal_pinhole_carries_five_coefficients(cfg):
    cam = CameraModel.nominal(_pinhole_cfg(cfg), 640, 400)
    assert cam.model == "pinhole"
    assert cam.D.shape == (5,)


def test_rejects_wrong_distortion_length(cfg):
    K = CameraModel.nominal(cfg, 640, 400).K
    with pytest.raises(ValueError):
        CameraModel(640, 400, K, np.zeros(5), "fisheye")
    with pytest.raises(ValueError):
        CameraModel(640, 400, K, np.zeros(4), "equirect")


# ------------------------------------------------------------------------------- fisheye optics
def test_fisheye_pixel_ray_round_trip(cfg):
    cam = CameraModel(640, 400, CameraModel.nominal(cfg, 640, 400).K, FISHEYE_D, "fisheye")
    uv = _pixel_grid(cam.width, cam.height)
    back = cam.ray_to_pixel(cam.pixel_to_ray(uv))
    assert np.isfinite(back).all()
    assert np.abs(back - uv).max() < 1e-3


def test_fisheye_matches_cv2_fisheye_project_points(cfg):
    cam = CameraModel(640, 400, CameraModel.nominal(cfg, 640, 400).K, FISHEYE_D, "fisheye")
    rays = cam.pixel_to_ray(_pixel_grid(cam.width, cam.height))
    ref, _ = cv2.fisheye.projectPoints(
        np.ascontiguousarray(rays).reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        cam.K,
        cam.D.reshape(4, 1),
    )
    ours = cam.ray_to_pixel(rays)
    assert np.abs(np.asarray(ref).reshape(-1, 2) - ours).max() < 1e-3


def test_fisheye_round_trip_up_to_80_degrees(cfg):
    cam = CameraModel(640, 400, CameraModel.nominal(cfg, 640, 400).K, FISHEYE_D, "fisheye")
    theta, phi = np.meshgrid(np.radians(np.linspace(0.0, 80.0, 41)), np.linspace(0.0, 2 * np.pi, 17))
    rays = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    ).reshape(-1, 3)
    uv = cam.ray_to_pixel(rays)
    assert np.isfinite(uv).all()
    again = cam.ray_to_pixel(cam.pixel_to_ray(uv))
    assert np.abs(again - uv).max() < 1e-3
    # the recovered ray must be the same direction, not just the same pixel
    cos = np.clip(np.sum(cam.pixel_to_ray(uv) * rays, axis=1), -1.0, 1.0)
    assert np.degrees(np.arccos(cos)).max() < 1e-4


def test_fisheye_is_equidistant_without_distortion(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    theta = np.radians(np.array([0.0, 10.0, 30.0, 45.0]))
    rays = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    uv = cam.ray_to_pixel(rays)
    assert uv[:, 0] - cam.cx == pytest.approx(cam.fx * theta, abs=1e-9)
    assert uv[:, 1] == pytest.approx(np.full(4, cam.cy), abs=1e-9)


# ------------------------------------------------------------------------------- pinhole optics
def test_pinhole_pixel_ray_round_trip(cfg):
    cam = CameraModel(640, 400, CameraModel.nominal(cfg, 640, 400).K, PINHOLE_D, "pinhole")
    uv = _pixel_grid(cam.width, cam.height)
    back = cam.ray_to_pixel(cam.pixel_to_ray(uv))
    assert np.abs(back - uv).max() < 1e-3
    rays = cam.pixel_to_ray(uv)
    assert np.linalg.norm(rays, axis=1) == pytest.approx(np.ones(rays.shape[0]), abs=1e-12)


# --------------------------------------------------------------------------- shapes and edge cases
def test_single_vectors_keep_their_shape(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    ray = cam.pixel_to_ray(np.array([100.0, 50.0]))
    assert ray.shape == (3,)
    uv = cam.ray_to_pixel(ray)
    assert uv.shape == (2,)
    assert uv == pytest.approx(np.array([100.0, 50.0]), abs=1e-9)
    assert isinstance(cam.ifov(np.array([100.0, 50.0])), float)
    assert cam.ifov(np.array([[100.0, 50.0], [10.0, 10.0]])).shape == (2,)


def test_rays_at_or_behind_the_image_plane_are_nan(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    uv = cam.ray_to_pixel(np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert np.isnan(uv[0]).all()
    assert np.isnan(uv[1]).all()
    assert np.isfinite(uv[2]).all()


def test_bad_shapes_raise(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    with pytest.raises(ValueError):
        cam.pixel_to_ray(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        cam.ray_to_pixel(np.zeros((4, 2)))


# ---------------------------------------------------------------------------- calibration loading
def test_from_calibration_rescales_about_pixel_centres(cfg):
    intr = {
        "model": "fisheye",
        "width": 1280,
        "height": 800,
        "K": [[933.3333, 0.0, 645.5], [0.0, 933.3333, 403.5], [0.0, 0.0, 1.0]],
        "D": [0.01, -0.002, 0.0, 0.0],
    }
    cam = CameraModel.from_calibration(intr, cfg, 640, 400)
    assert cam.fx == pytest.approx(466.66665, abs=1e-5)
    assert cam.fy == pytest.approx(466.66665, abs=1e-5)
    assert cam.cx == pytest.approx((645.5 + 0.5) * 0.5 - 0.5, abs=1e-9)  # 322.5
    assert cam.cy == pytest.approx((403.5 + 0.5) * 0.5 - 0.5, abs=1e-9)  # 201.5
    assert cam.D == pytest.approx(np.array(intr["D"]))
    assert (cam.width, cam.height) == (640, 400)


def test_from_calibration_at_native_size_is_untouched(cfg):
    intr = {
        "model": "fisheye",
        "width": 640,
        "height": 400,
        "K": [[470.0, 0.0, 322.0], [0.0, 471.0, 198.0], [0.0, 0.0, 1.0]],
        "D": [0.0, 0.0, 0.0, 0.0],
    }
    cam = CameraModel.from_calibration(intr, cfg, 640, 400)
    assert cam.K == pytest.approx(np.array(intr["K"]))


def test_from_calibration_aspect_mismatch_raises(cfg):
    intr = {
        "model": "fisheye",
        "width": 1280,
        "height": 720,
        "K": [[933.0, 0.0, 639.5], [0.0, 933.0, 359.5], [0.0, 0.0, 1.0]],
        "D": [0.0, 0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError):
        CameraModel.from_calibration(intr, cfg, 640, 400)


def test_from_calibration_without_file_falls_back_to_nominal(cfg):
    cam = CameraModel.from_calibration(None, cfg, 640, 400)
    nom = CameraModel.nominal(cfg, 640, 400)
    assert cam.K == pytest.approx(nom.K)
    assert cam.model == nom.model


# ------------------------------------------------------------------------------------------ ifov
def test_ifov_matches_the_analytic_equidistant_value(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    assert cam.ifov(np.array([cam.cx, cam.cy])) == pytest.approx(1.0 / cam.fx, rel=1e-6)
    # 200 px off axis: radial step 1/f, tangential step sin(theta)/theta/f; ifov is their mean
    theta = 200.0 / cam.fx
    expect = (1.0 + math.sin(theta) / theta) / (2.0 * cam.fx)
    assert cam.ifov(np.array([cam.cx + 200.0, cam.cy])) == pytest.approx(expect, rel=1e-6)


# ----------------------------------------------------------------------------- geometry helpers
def test_range_from_size_inverts_the_apparent_size(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    ifov = 1.0 / cam.fx
    clay = cfg.mount.clay_diameter_m
    px = clay / (20.0 * ifov)
    assert px == pytest.approx(2.52, abs=0.01)
    assert range_from_size(px, ifov, clay, 0.0) == pytest.approx(20.0, rel=1e-6)
    # blur inflates the blob, so ignoring the psf would under-estimate the range
    blurred = math.sqrt(px**2 + 1.5**2)
    assert range_from_size(blurred, ifov, clay, 1.5) == pytest.approx(20.0, rel=1e-6)
    assert range_from_size(blurred, ifov, clay, 0.0) < 20.0
    # a sub-pixel blob is floored at 0.5 px effective diameter, never zero/infinite range
    assert range_from_size(0.2, ifov, clay, 1.5) == pytest.approx(clay / (0.5 * ifov), rel=1e-9)
    with pytest.raises(ValueError):
        range_from_size(3.0, 0.0, clay, 0.0)


def test_rot_rpy_is_zyx_and_orthonormal():
    assert rot_rpy(0.0, 0.0, 0.0) == pytest.approx(np.eye(3))
    assert rot_rpy(0.0, 0.0, math.pi / 2) @ np.array([1.0, 0.0, 0.0]) == pytest.approx(
        np.array([0.0, 1.0, 0.0]), abs=1e-12
    )
    assert rot_rpy(0.0, math.pi / 2, 0.0) @ np.array([1.0, 0.0, 0.0]) == pytest.approx(
        np.array([0.0, 0.0, -1.0]), abs=1e-12
    )
    r, p, y = 0.1, -0.2, 0.3
    R = rot_rpy(r, p, y)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    rz = rot_rpy(0.0, 0.0, y)
    ry = rot_rpy(0.0, p, 0.0)
    rx = rot_rpy(r, 0.0, 0.0)
    assert R == pytest.approx(rz @ ry @ rx, abs=1e-12)


def test_azel_conventions_and_round_trip():
    az, el = dir_to_azel(np.array([0.0, 1.0, 0.0]))
    assert (az, el) == pytest.approx((math.pi / 2, 0.0))
    az, el = dir_to_azel(np.array([0.0, 0.0, 2.0]))
    assert el == pytest.approx(math.pi / 2)
    rng = np.random.default_rng(7)
    d = rng.normal(size=(50, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    az, el = dir_to_azel(d)
    assert azel_to_dir(az, el) == pytest.approx(d, abs=1e-12)
    assert azel_to_dir(0.3, 0.2).shape == (3,)


# ------------------------------------------------------------------------------------ mount model
def test_camera_axes_map_to_the_gun_frame(cfg):
    mount = MountModel.from_config(cfg, None)
    assert mount.cam_to_gun(np.array([0.0, 0.0, 1.0])) == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert mount.cam_to_gun(np.array([1.0, 0.0, 0.0])) == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)
    assert mount.cam_to_gun(np.array([0.0, 1.0, 0.0])) == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)
    v = np.array([[0.2, -0.3, 0.9], [1.0, 2.0, 3.0]])
    assert mount.gun_to_cam(mount.cam_to_gun(v)) == pytest.approx(v, abs=1e-12)


def test_bore_pixel_parallax(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    mount = MountModel.from_config(cfg, None)
    far = mount.bore_pixel(cam, 1000.0)
    assert far[0] == pytest.approx(cam.cx, abs=0.05)
    assert far[1] == pytest.approx(cam.cy, abs=0.05)
    near = mount.bore_pixel(cam, 10.0)
    dx = (mount.muzzle_pos_g[0] - mount.cam_pos_g[0]) + 10.0
    dz = mount.muzzle_pos_g[2] - mount.cam_pos_g[2]
    assert dz == pytest.approx((cfg.mount.muzzle_pos_mm[2] - cfg.mount.cam_pos_mm[2]) / 1000.0, abs=1e-9)
    assert dz > 0.0  # as-built: 54 mm below the bore
    expect_v = cam.cy - cam.fy * math.atan2(dz, dx)
    assert near[1] < cam.cy  # the bore line sits ABOVE the optical axis at short range
    assert near[1] == pytest.approx(expect_v, abs=0.5)
    assert near[0] == pytest.approx(cam.cx, abs=1e-6)
    assert mount.bore_point_g(10.0) == pytest.approx(mount.muzzle_pos_g + np.array([10.0, 0, 0]))


def test_boresight_rpy_rotates_the_camera(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    base = MountModel.from_config(cfg, None).bore_pixel(cam, 1000.0)
    yawed = MountModel.from_config(cfg, {"rpy_deg": [0.0, 0.0, 1.0]}).bore_pixel(cam, 1000.0)
    # +yaw turns the camera toward gun-left (+Y), so the scene -- and the bore -- shift right
    assert yawed[0] - base[0] == pytest.approx(cam.fx * math.radians(1.0), abs=0.05)
    assert yawed[1] == pytest.approx(base[1], abs=0.05)
    pitched = MountModel.from_config(cfg, {"rpy_deg": [0.0, 1.0, 0.0]}).bore_pixel(cam, 1000.0)
    # Ry(+1 deg) about gun +Y tips the camera nose-down, so the bore appears higher up
    assert pitched[1] - base[1] == pytest.approx(-cam.fy * math.radians(1.0), abs=0.05)


def test_poi_offset_moves_the_shot_line(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    mount = MountModel.from_config(cfg, {"poi_offset_rad": [0.01, 0.005]})
    assert mount.bore_dir_g == pytest.approx(
        np.array([math.cos(0.005) * math.cos(0.01), math.cos(0.005) * math.sin(0.01), math.sin(0.005)]),
        abs=1e-12,
    )
    base = MountModel.from_config(cfg, None).bore_pixel(cam, 30.0)
    off = mount.bore_pixel(cam, 30.0)
    assert off[0] < base[0]  # +yaw = gun-left = image-left
    assert off[1] < base[1]  # +pitch = up = smaller row
    assert base[0] - off[0] == pytest.approx(cam.fx * 0.01, rel=0.05)
    assert base[1] - off[1] == pytest.approx(cam.fy * 0.005, rel=0.05)


# ------------------------------------------------------------------------------------ barrel mask
@pytest.fixture(scope="module")
def nominal_mask(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    return cam, barrel_mask(cam, MountModel.from_config(cfg, None), cfg)


def test_barrel_mask_blocks_the_top_centre_only(nominal_mask):
    cam, mask = nominal_mask
    assert mask.shape == (cam.height, cam.width)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[60, 319] == 0  # barrel
    for u, v in ((319, 300), (20, 380), (620, 380), (20, 200), (620, 200)):
        assert mask[v, u] == 255, f"({u},{v}) should be usable sky"
    assert 0.0 < (mask == 0).mean() < 0.25


def test_barrel_mask_lower_edge_is_the_muzzle_silhouette(cfg, nominal_mask):
    _, mask = nominal_mask
    masked_rows = np.flatnonzero(mask[:, 319] == 0)
    assert masked_rows.min() == 0  # the barrel runs off the top of the frame
    assert 120 <= masked_rows.max() <= 150
    assert np.array_equal(masked_rows, np.arange(masked_rows.max() + 1))  # solid, no holes


def test_barrel_mask_margin_grows_the_silhouette(cfg):
    cam = CameraModel.nominal(cfg, 640, 400)
    mount = MountModel.from_config(cfg, None)
    tight = dataclasses.replace(cfg, mount=dataclasses.replace(cfg.mount, barrel_mask_margin_px=0))
    wide = dataclasses.replace(cfg, mount=dataclasses.replace(cfg.mount, barrel_mask_margin_px=6))
    m0 = barrel_mask(cam, mount, tight)
    m6 = barrel_mask(cam, mount, wide)
    edge0 = np.flatnonzero(m0[:, 319] == 0).max()
    edge6 = np.flatnonzero(m6[:, 319] == 0).max()
    assert edge6 - edge0 == 6
    assert np.all((m6 == 0) | (m0 == 255))  # the margin only ever adds masked pixels
    # the geometric edge is the muzzle rim, seen from the optical centre (as-built: ~8.7 deg above the axis)
    m = cfg.mount
    rim_z = m.barrel_axis_z_mm - m.barrel_od_mm[-1][1] / 2.0 - m.cam_pos_mm[2]
    expect = cam.cy - cam.fy * math.atan2(rim_z, m.muzzle_pos_mm[0] - m.cam_pos_mm[0])
    assert edge0 == pytest.approx(expect, abs=1.5)


def test_barrel_mask_scales_with_resolution(cfg):
    big = CameraModel.nominal(cfg, 1280, 800)
    mount = MountModel.from_config(cfg, None)
    mask = barrel_mask(big, mount, cfg)
    assert mask.shape == (800, 1280)
    assert mask[120, 639] == 0
    assert mask[600, 639] == 255


def test_combined_mask_intersects_the_user_mask(cfg, nominal_mask):
    cam, barrel = nominal_mask
    mount = MountModel.from_config(cfg, None)
    calib = Calibration(dir=Path("."))
    assert np.array_equal(combined_mask(cam, mount, cfg, calib), barrel)

    half = np.full((200, 320), 255, dtype=np.uint8)  # half resolution: left half blocked
    half[:, :160] = 0
    calib.mask = half
    out = combined_mask(cam, mount, cfg, calib)
    assert out.shape == barrel.shape
    assert np.all(out[:, :320] == 0)
    assert np.array_equal(out[:, 320:], barrel[:, 320:])
