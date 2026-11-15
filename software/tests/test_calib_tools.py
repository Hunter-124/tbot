"""Calibration tools: board generation, intrinsic calibration, boresight solve.

The intrinsic tests render a chessboard through a KNOWN fisheye model (K, D below) and check that the
tool recovers it: every pixel of a supersampled image is back-projected to a ray with
cv2.fisheye.undistortPoints, intersected with the board plane, and the board texture is sampled there
(cv2.remap), so the images obey the fisheye model exactly rather than approximately.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import cv2
import numpy as np
import pytest

from tbot.config import load_config, save_json
from tbot.optics import CameraModel, MountModel
from tbot.tools import calib_boresight as cb
from tbot.tools import calib_intrinsics as ci
from tbot.tools import make_board as mb

IMG_W, IMG_H = 640, 400
TRUE_K = np.array([[466.67, 0.0, 319.5], [0.0, 466.67, 199.5], [0.0, 0.0, 1.0]], dtype=np.float64)
TRUE_D = np.array([-0.03, 0.01, 0.0, 0.0], dtype=np.float64)
PATTERN = (9, 6)  # inner corners
SQUARE_M = 0.025
SUPERSAMPLE = 2  # render at 2x and box-average down: no aliasing on the 25 mm squares
TEXEL_PER_M = 2000.0  # board texture resolution (50 px per square)
N_POSES = 24
POSE_SEED = 5


# ------------------------------------------------------------------ synthetic board rendering
@lru_cache(maxsize=1)
def _texture() -> tuple[np.ndarray, float, float]:
    """Chessboard texture with a one-square white margin; returns (image, width_m, height_m)."""
    cols, rows = PATTERN
    sq = int(round(SQUARE_M * TEXEL_PER_M))
    w, h = (cols + 3) * sq, (rows + 3) * sq
    tex = np.full((h, w), 245, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                tex[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 20
    return tex, w / TEXEL_PER_M, h / TEXEL_PER_M


@lru_cache(maxsize=1)
def _ray_grid() -> np.ndarray:
    """(W*SS*H*SS, 3) camera rays of the supersampled image under the true fisheye model."""
    s = SUPERSAMPLE
    k = TRUE_K.copy()
    k[0, 0] *= s
    k[1, 1] *= s
    k[0, 2] = (TRUE_K[0, 2] + 0.5) * s - 0.5
    k[1, 2] = (TRUE_K[1, 2] + 0.5) * s - 0.5
    u, v = np.meshgrid(np.arange(IMG_W * s, dtype=np.float64), np.arange(IMG_H * s, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    und = np.asarray(cv2.fisheye.undistortPoints(pts, k, TRUE_D), dtype=np.float64).reshape(-1, 2)
    return np.concatenate([und, np.ones((und.shape[0], 1), dtype=np.float64)], axis=1)


def _rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]))


def _render(R_board_cam: np.ndarray, t: np.ndarray) -> np.ndarray:
    """One view of the board at pose (R, t) (board centre at t, metres, camera frame)."""
    tex, tex_w, tex_h = _texture()
    rays = _ray_grid()
    s = SUPERSAMPLE
    rt = R_board_cam.T
    rb = rays @ rt.T
    tb = rt @ t
    ok = np.abs(rb[:, 2]) > 1e-9
    scale = np.where(ok, tb[2] / np.where(ok, rb[:, 2], 1.0), -1.0)
    bx = scale * rb[:, 0] - tb[0]
    by = scale * rb[:, 1] - tb[1]
    map_x = ((bx + tex_w * 0.5) * TEXEL_PER_M).astype(np.float32).reshape(IMG_H * s, IMG_W * s)
    map_y = ((by + tex_h * 0.5) * TEXEL_PER_M).astype(np.float32).reshape(IMG_H * s, IMG_W * s)
    behind = (scale <= 0.0).reshape(IMG_H * s, IMG_W * s)
    map_x[behind] = -1.0
    map_y[behind] = -1.0
    big = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=245)
    return cv2.resize(big, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def _poses(n: int = N_POSES, seed: int = POSE_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Varied board poses: tilts to 40 deg, distances 0.35-0.95 m, spread across the frame."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        R = _rot(math.radians(rng.uniform(-40.0, 40.0)), math.radians(rng.uniform(-40.0, 40.0)),
                 math.radians(rng.uniform(-25.0, 25.0)))
        tz = rng.uniform(0.35, 0.95)
        lim_x = max(0.0, tz * math.tan(math.radians(33.0)) - 0.16)
        lim_y = max(0.0, tz * math.tan(math.radians(22.0)) - 0.11)
        out.append((R, np.array([rng.uniform(-lim_x, lim_x), rng.uniform(-lim_y, lim_y), tz])))
    return out


@lru_cache(maxsize=1)
def _views() -> tuple[np.ndarray, ...]:
    return tuple(_render(R, t) for R, t in _poses())


@lru_cache(maxsize=1)
def _detected() -> tuple[np.ndarray, ...]:
    found = [ci.find_corners(img, PATTERN) for img in _views()]
    return tuple(c for c in found if c is not None)


@lru_cache(maxsize=1)
def _calibration() -> dict:
    return ci.calibrate_views(list(_detected()), PATTERN, SQUARE_M, (IMG_W, IMG_H), "fisheye")


# ---------------------------------------------------------------------------------- make_board
def test_board_image_matches_the_print_geometry(tmp_path):
    out = tmp_path / "board.png"
    assert mb.main(["--cols", "9", "--rows", "6", "--square-mm", "25", "--dpi", "300",
                    "--out", str(out)]) == 0
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    sq = 25.0 * 300.0 / 25.4  # 295.27 px per square
    assert img.shape[0] == int(round(sq)) * (6 + 1) + 3 * int(round(sq))  # board + 2 margins + footer
    assert img.shape[1] >= int(round(sq)) * (9 + 1) + 2 * int(round(sq))
    # a quiet zone of at least one square on every side of the board, as the detector needs
    margin = int(round(sq))
    assert img[:margin, :].min() > 200
    assert img[:, :margin].min() > 200
    assert img[:, -margin:].min() > 200


def test_board_corners_are_detectable_and_correctly_spaced(tmp_path):
    out = tmp_path / "board.png"
    mb.main(["--cols", "9", "--rows", "6", "--square-mm", "20", "--dpi", "200", "--out", str(out)])
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    corners = ci.find_corners(img, (9, 6))
    assert corners is not None, "the printed board must be detectable by the calibration tool"
    grid = corners.reshape(6, 9, 2)
    dx = np.abs(np.diff(grid[:, :, 0], axis=1)).mean()
    dy = np.abs(np.diff(grid[:, :, 1], axis=0)).mean()
    drawn = mb.square_px(20.0, 200.0)
    assert abs(drawn - 20.0 * 200.0 / 25.4) <= 0.5, "squares are rounded to the nearest pixel"
    assert abs(dx - drawn) < 0.1
    assert abs(dy - drawn) < 0.1


def test_board_refuses_a_square_too_small_to_print():
    with pytest.raises(ValueError):
        mb.render_board(9, 6, 0.5, 72.0)  # 1.4 px per square
    with pytest.raises(ValueError):
        mb.render_board(2, 6, 25.0, 300.0)


# ----------------------------------------------------------------------------- calib_intrinsics
def test_corners_are_found_in_most_synthetic_views():
    assert len(_detected()) >= 12
    for corners in _detected():
        assert corners.shape == (PATTERN[0] * PATTERN[1], 2)
        assert corners.min() > -1.0 and corners[:, 0].max() < IMG_W and corners[:, 1].max() < IMG_H


def test_calibrate_views_recovers_the_true_fisheye_model():
    result = _calibration()
    K = np.asarray(result["K"], dtype=np.float64)
    assert result["model"] == "fisheye"
    assert result["n_views"] >= 12
    assert len(result["D"]) == 4
    assert result["rms_px"] < 0.3
    assert abs(K[0, 0] - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.01
    assert abs(K[1, 1] - TRUE_K[1, 1]) / TRUE_K[1, 1] < 0.01
    assert abs(K[0, 2] - TRUE_K[0, 2]) < 2.0
    assert abs(K[1, 2] - TRUE_K[1, 2]) < 2.0
    assert len(result["per_view_rms"]) == result["n_views"]
    assert max(result["per_view_rms"]) < 1.0


def test_calibrated_model_points_where_the_true_model_points():
    """What consumers actually use is pixel -> ray: inside the field the board covered, the recovered
    model must agree with the true one to a small fraction of a clay's angular size (a 108 mm clay at
    25 m subtends 0.25 deg)."""
    cfg = load_config()
    result = _calibration()
    truth = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    cam = CameraModel.from_calibration(result, cfg, IMG_W, IMG_H)
    seen = np.concatenate([np.asarray(c, dtype=np.float64) for c in _detected()])
    lo, hi = seen.min(axis=0), seen.max(axis=0)
    u, v = np.meshgrid(np.linspace(lo[0], hi[0], 9), np.linspace(lo[1], hi[1], 7))
    uv = np.stack([u.ravel(), v.ravel()], axis=1)
    a = truth.pixel_to_ray(uv)
    b = cam.pixel_to_ray(uv)
    angle_deg = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)))
    assert angle_deg.max() < 0.2


def test_calibrate_views_rejects_a_corrupted_view():
    good = list(_detected())
    bad = good[0].copy()
    bad[::3] += 4.0  # a mis-detected board: part of the grid shifted
    result = ci.calibrate_views(good + [bad], PATTERN, SQUARE_M, (IMG_W, IMG_H), "fisheye")
    assert len(good) in result["rejected_views"] or result["n_views"] == len(good)
    assert result["rejected_views"], "the outlier view must be dropped"
    K = np.asarray(result["K"], dtype=np.float64)
    assert abs(K[0, 0] - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.01
    assert result["rms_px"] < 0.3


def test_calibrate_views_validates_its_inputs():
    good = list(_detected())
    with pytest.raises(ValueError):
        ci.calibrate_views(good[:3], PATTERN, SQUARE_M, (IMG_W, IMG_H), "fisheye")
    with pytest.raises(ValueError):
        ci.calibrate_views(good, PATTERN, SQUARE_M, (IMG_W, IMG_H), "equirect")
    with pytest.raises(ValueError):
        ci.calibrate_views(good, (8, 6), SQUARE_M, (IMG_W, IMG_H), "fisheye")  # wrong corner count


def test_pinhole_model_gives_five_distortion_coefficients():
    result = ci.calibrate_views(list(_detected()), PATTERN, SQUARE_M, (IMG_W, IMG_H), "pinhole")
    assert result["model"] == "pinhole"
    assert len(result["D"]) == 5
    assert result["rms_px"] < 1.0


def test_fov_matches_the_fisheye_model_geometry():
    fov = ci.fov_deg(TRUE_K, TRUE_D, "fisheye", (IMG_W, IMG_H))
    # equidistant model: theta_d = (u - cx) / fx, theta_d = theta (1 + k1 t^2 + k2 t^4)
    def theta_of(theta_d: float) -> float:
        t = theta_d
        for _ in range(60):
            f = t * (1.0 + TRUE_D[0] * t ** 2 + TRUE_D[1] * t ** 4) - theta_d
            df = 1.0 + 3.0 * TRUE_D[0] * t ** 2 + 5.0 * TRUE_D[1] * t ** 4
            t -= f / df
        return t
    h = math.degrees(theta_of((IMG_W - 0.5 - 319.5) / 466.67) + theta_of((319.5 + 0.5) / 466.67))
    v = math.degrees(theta_of((IMG_H - 0.5 - 199.5) / 466.67) + theta_of((199.5 + 0.5) / 466.67))
    assert abs(fov["h"] - h) < 0.2
    assert abs(fov["v"] - v) < 0.2
    assert fov["d"] > fov["h"] > fov["v"]


def test_nominal_fx_follows_the_datasheet_prior():
    cfg = load_config()
    assert ci.nominal_fx(cfg, 1280) == pytest.approx(2.8 / 0.003, rel=1e-9)
    assert ci.nominal_fx(cfg, 640) == pytest.approx(2.8 / 0.003 / 2.0, rel=1e-9)


def test_sharpness_drops_when_the_view_is_blurred():
    view = _views()[1]
    corners = ci.find_corners(view, PATTERN)
    sharp = ci.sharpness(view, corners)
    assert sharp > ci.SHARPNESS_MIN
    for sigma in (1.0, 2.0):
        blurred = cv2.GaussianBlur(view, (0, 0), sigma)
        assert ci.sharpness(blurred, corners) < sharp / 2.0


def test_view_selector_takes_new_poses_and_skips_duplicates():
    views = _views()
    sel = ci.ViewSelector(PATTERN, (IMG_W, IMG_H))
    corners, reason = sel.offer(views[0])
    assert corners is not None and reason == "accepted"
    again, reason = sel.offer(views[0])
    assert again is None and "already covered" in reason
    assert sel.offer(np.full((IMG_H, IMG_W), 128, dtype=np.uint8)) == (None, "no board")
    before = sel.coverage_frac
    assert sel.offer(views[1])[0] is not None
    assert sel.coverage_frac >= before
    assert len(sel.accepted) == 2
    assert sel.coverage.sum() < sel.coverage.size  # two views cannot cover the whole frame
    assert len(sel.missing_cells()) == int(sel.coverage.size - sel.coverage.sum())
    assert sel.coverage_hint().startswith("still need ")
    sel.coverage[:] = True
    assert sel.coverage_hint() == "coverage complete"
    assert sel.missing_cells() == []


def test_view_selector_rejects_views_below_the_sharpness_gate():
    sel = ci.ViewSelector(PATTERN, (IMG_W, IMG_H), sharpness_min=1e9)
    accepted, reason = sel.offer(_views()[0])
    assert accepted is None and reason.startswith("blurred")
    assert sel.accepted == []


def test_cli_from_dir_writes_the_contract_schema(tmp_path):
    img_dir = tmp_path / "views"
    img_dir.mkdir()
    for i, img in enumerate(_views()):
        cv2.imwrite(str(img_dir / f"v{i:02d}.png"), img)
    rc = ci.main(["--from-dir", str(img_dir), "--calib-dir", str(tmp_path),
                  "--cols", "9", "--rows", "6", "--square-mm", "25"])
    assert rc == 0
    doc = json.loads((tmp_path / "intrinsics.json").read_text())
    for key in ("model", "width", "height", "K", "D", "rms_px", "n_views", "created", "notes"):
        assert key in doc, key
    assert doc["width"] == IMG_W and doc["height"] == IMG_H
    assert np.asarray(doc["K"]).shape == (3, 3)
    assert set(doc["fov_deg"]) == {"h", "v", "d"}
    assert len(doc["per_view_rms"]) == doc["n_views"]
    assert doc["rms_px"] < 0.3
    assert (tmp_path / "intrinsics_coverage.png").exists()
    cam = CameraModel.from_calibration(doc, load_config(), IMG_W, IMG_H)
    assert cam.fx == pytest.approx(doc["K"][0][0])


def test_cli_refuses_a_calibration_with_too_few_views(tmp_path):
    img_dir = tmp_path / "views"
    img_dir.mkdir()
    for i, img in enumerate(_views()[:5]):
        cv2.imwrite(str(img_dir / f"v{i:02d}.png"), img)
    assert ci.main(["--from-dir", str(img_dir), "--calib-dir", str(tmp_path)]) == 1
    assert not (tmp_path / "intrinsics.json").exists()
    assert ci.main(["--from-dir", str(img_dir), "--calib-dir", str(tmp_path), "--force"]) == 0
    doc = json.loads((tmp_path / "intrinsics.json").read_text())
    assert doc["n_views"] == 5
    assert "force" in doc["notes"].lower()


def test_cli_reports_a_directory_without_boards(tmp_path):
    img_dir = tmp_path / "views"
    img_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"v{i}.png"), np.full((IMG_H, IMG_W), 120, dtype=np.uint8))
    assert ci.main(["--from-dir", str(img_dir), "--calib-dir", str(tmp_path)]) == 1
    assert ci.main(["--from-dir", str(tmp_path / "nothing"), "--calib-dir", str(tmp_path)]) == 1


def test_cli_calibrates_from_a_recorded_session(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    views = _views()
    with open(session / "frames.bin", "wb") as f:
        for img in views:
            f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    (session / "meta.json").write_text(json.dumps({
        "version": 1, "width": IMG_W, "height": IMG_H, "channels": 1, "dtype": "uint8",
        "fps": 120.0, "source": "synthetic", "config": {}, "calibration": {}, "created": "test"}))
    rc = ci.main(["--session", str(session), "--every", "2", "--calib-dir", str(tmp_path)])
    assert rc == 0
    doc = json.loads((tmp_path / "intrinsics.json").read_text())
    assert doc["n_views"] >= 8
    assert abs(doc["K"][0][0] - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.02
    assert sorted(p.name for p in (tmp_path / "intrinsics_views").glob("*.png"))[0] == "view_000.png"


# ------------------------------------------------------------------------------ calib_boresight
def _write_intrinsics(calib_dir) -> dict:
    doc = {"model": "fisheye", "width": IMG_W, "height": IMG_H,
           "K": [[float(x) for x in row] for row in TRUE_K], "D": [float(x) for x in TRUE_D],
           "rms_px": 0.12, "n_views": 20, "created": "test", "notes": "test fixture"}
    save_json(calib_dir / "intrinsics.json", doc)
    return doc


def _true_mount(cfg, rpy_deg):
    return MountModel.from_config(cfg, {"rpy_deg": list(rpy_deg), "poi_offset_rad": [0.0, 0.0]})


def test_boresight_recovers_a_known_mount_rotation(tmp_path):
    cfg = load_config()
    _write_intrinsics(tmp_path)
    cam = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    true_rpy = [0.0, -0.3, 0.5]
    u, v = _true_mount(cfg, true_rpy).bore_pixel(cam, 25.0)
    rc = cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", f"{u},{v}"])
    assert rc == 0
    doc = json.loads((tmp_path / "boresight.json").read_text())
    assert doc["rpy_deg"][1] == pytest.approx(true_rpy[1], abs=0.01)
    assert doc["rpy_deg"][2] == pytest.approx(true_rpy[2], abs=0.01)
    assert doc["rpy_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert doc["range_m"] == 25.0
    assert doc["bore_pixel"] == pytest.approx([u, v])
    assert doc["width"] == IMG_W and doc["height"] == IMG_H
    assert "created" in doc
    # the calibrated mount reproduces the observed pixel
    mount = MountModel.from_config(cfg, doc)
    assert mount.bore_pixel(cam, 25.0) == pytest.approx((u, v), abs=1e-6)


def test_boresight_recovers_roll_from_a_plumb_line(tmp_path):
    cfg = load_config()
    _write_intrinsics(tmp_path)
    cam = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    true_rpy = [1.0, -0.3, 0.5]
    mount = _true_mount(cfg, true_rpy)
    u, v = mount.bore_pixel(cam, 25.0)
    below = mount.muzzle_pos_g + np.array([25.0, 0.0, -1.0]) - mount.cam_pos_g  # plumb line, 1 m lower
    u2, v2 = cam.ray_to_pixel(mount.gun_to_cam(below))
    rc = cb.main(["--calib-dir", str(tmp_path), "--range-m", "25",
                  "--pixel", f"{u},{v}", "--pixel2", f"{u2},{v2}"])
    assert rc == 0
    doc = json.loads((tmp_path / "boresight.json").read_text())
    assert doc["rpy_deg"][0] == pytest.approx(1.0, abs=0.05)
    assert doc["rpy_deg"][1] == pytest.approx(true_rpy[1], abs=0.01)
    assert doc["rpy_deg"][2] == pytest.approx(true_rpy[2], abs=0.01)


def test_boresight_without_a_plumb_line_keeps_the_known_roll(tmp_path):
    cfg = load_config()
    _write_intrinsics(tmp_path)
    cam = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    mount = _true_mount(cfg, [1.0, -0.3, 0.5])
    u, v = mount.bore_pixel(cam, 25.0)
    rc = cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", f"{u},{v}",
                  "--roll-deg", "1.0"])
    assert rc == 0
    doc = json.loads((tmp_path / "boresight.json").read_text())
    assert doc["rpy_deg"][0] == pytest.approx(1.0, abs=1e-9)
    assert doc["rpy_deg"][1] == pytest.approx(-0.3, abs=0.01)
    assert doc["rpy_deg"][2] == pytest.approx(0.5, abs=0.01)


def test_boresight_poi_offset_is_written_without_touching_the_rotation(tmp_path):
    cfg = load_config()
    _write_intrinsics(tmp_path)
    cam = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    u, v = _true_mount(cfg, [0.0, -0.3, 0.5]).bore_pixel(cam, 25.0)
    assert cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", f"{u},{v}"]) == 0
    first = json.loads((tmp_path / "boresight.json").read_text())

    # patterning: the pattern centre is 10 cm gun-left and 5 cm high at 25 m
    assert cb.main(["--calib-dir", str(tmp_path), "--poi-offset-cm", "10,5@25"]) == 0
    doc = json.loads((tmp_path / "boresight.json").read_text())
    assert doc["poi_offset_rad"][0] == pytest.approx(math.atan2(0.10, 25.0), rel=1e-12)
    assert doc["poi_offset_rad"][1] == pytest.approx(math.atan2(0.05, 25.0), rel=1e-12)
    assert doc["poi_offset_rad"] == pytest.approx([0.004, 0.002], abs=1e-5)
    assert doc["rpy_deg"] == first["rpy_deg"]
    assert doc["bore_pixel"] == first["bore_pixel"]
    # the shot direction now leans toward the pattern centre
    bore = MountModel.from_config(cfg, doc).bore_dir_g
    assert bore[1] / bore[0] == pytest.approx(0.10 / 25.0, rel=1e-6)
    assert bore[2] == pytest.approx(math.sin(math.atan2(0.05, 25.0)), rel=1e-6)


def test_boresight_poi_parsing_rejects_nonsense():
    with pytest.raises(ValueError):
        cb.parse_poi("10,5")
    with pytest.raises(ValueError):
        cb.parse_poi("10,5@0")
    with pytest.raises(ValueError):
        cb.parse_pixel("320")
    assert cb.parse_pixel(" 320.5 , 199 ") == (320.5, 199.0)


def test_boresight_rejects_impossible_requests(tmp_path):
    _write_intrinsics(tmp_path)
    assert cb.main(["--calib-dir", str(tmp_path)]) == 2  # nothing to do
    assert cb.main(["--calib-dir", str(tmp_path), "--pixel", "320,200"]) == 2  # no --range-m
    assert cb.main(["--calib-dir", str(tmp_path), "--pixel", "900,200", "--range-m", "25"]) == 2
    assert not (tmp_path / "boresight.json").exists()


def test_boresight_refuses_a_wild_correction(tmp_path):
    cfg = load_config()
    _write_intrinsics(tmp_path)
    cam = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    assert cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", "10,10"]) == 1
    assert not (tmp_path / "boresight.json").exists()
    assert cb.main(["--calib-dir", str(tmp_path), "--range-m", "25", "--pixel", "10,10", "--force"]) == 0
    doc = json.loads((tmp_path / "boresight.json").read_text())
    mount = MountModel.from_config(cfg, doc)
    assert mount.bore_pixel(cam, 25.0) == pytest.approx((10.0, 10.0), abs=1e-6)


def test_boresight_range_changes_the_parallax_correction(tmp_path):
    """The same pixel at a different range means a different mount rotation (camera/bore offset)."""
    cfg = load_config()
    _write_intrinsics(tmp_path)
    cam = CameraModel(IMG_W, IMG_H, TRUE_K, TRUE_D, "fisheye")
    px = _true_mount(cfg, [0.0, 0.0, 0.0]).bore_pixel(cam, 25.0)
    near, _ = cb.solve_mount_rpy(cam, cfg, px, 10.0)
    far, _ = cb.solve_mount_rpy(cam, cfg, px, 25.0)
    assert far == pytest.approx(np.zeros(3), abs=1e-9)
    # the camera sits below the bore, so the bore line at 10 m is seen higher than at 25 m: the mount
    # correction must pitch the camera ray up (pitch is positive downward, Ry about gun +Y)
    mount = _true_mount(cfg, [0.0, 0.0, 0.0])
    d10 = mount.muzzle_pos_g + np.array([10.0, 0.0, 0.0]) - mount.cam_pos_g
    d25 = mount.muzzle_pos_g + np.array([25.0, 0.0, 0.0]) - mount.cam_pos_g
    expected = math.atan2(d10[2], d10[0]) - math.atan2(d25[2], d25[0])
    assert expected > 0.0
    assert math.degrees(near[1]) == pytest.approx(-math.degrees(expected), abs=1e-6)


# ------------------------------------------------------------------------------- boresight from the barrel
def _barrel_still(cfg, rpy_deg, w=1280, h=800):
    """Synthetic still: a dark barrel (the model's own silhouette for this mount error) on a lit wall."""
    from tbot.optics import barrel_mask
    cam = CameraModel.nominal(cfg, w, h)
    mount = MountModel.from_config(cfg, {"rpy_deg": list(rpy_deg), "poi_offset_rad": [0.0, 0.0]})
    m0 = cfg.mount.barrel_mask_margin_px
    cfg.mount.barrel_mask_margin_px = 0
    try:
        barrel = barrel_mask(cam, mount, cfg) == 0
    finally:
        cfg.mount.barrel_mask_margin_px = m0
    img = np.full((h, w), 170.0, np.float32)
    img[barrel] = 60.0
    img = cv2.GaussianBlur(img, (0, 0), 3.0)  # out-of-focus barrel edges
    img += np.random.default_rng(0).normal(0.0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("rpy", [(0.0, -2.0, -4.0), (0.0, 1.0, 2.5)])
def test_barrel_boresight_recovers_a_known_mount_error(tmp_path, rpy):
    cfg = load_config(sets=["camera.rotate_180=false"])
    path = tmp_path / "barrel.png"
    cv2.imwrite(str(path), _barrel_still(cfg, rpy))
    rc = cb.main(["--calib-dir", str(tmp_path), "--barrel", str(path), "--set", "camera.rotate_180=false"])
    assert rc == 0
    out = json.loads((tmp_path / "boresight.json").read_text())
    assert out["method"] == "barrel"
    assert out["rpy_deg"][1] == pytest.approx(rpy[1], abs=0.25)  # pitch
    assert out["rpy_deg"][2] == pytest.approx(rpy[2], abs=0.25)  # yaw


def test_barrel_apex_direction_follows_the_taper():
    cfg = load_config()
    cfg.mount.barrel_od_mm = [[500.0, 22.0], [819.0, 22.0]]
    assert cb.barrel_apex_dir_g(cfg) == pytest.approx([1.0, 0.0, 0.0])  # straight barrel: the bore itself
    cfg.mount.barrel_od_mm = [[500.0, 21.5], [819.0, 22.5]]  # widens toward the muzzle: apex behind
    d = cb.barrel_apex_dir_g(cfg)
    assert d[0] > 0.99 and d[2] < 0.0  # forward and slightly down
    cfg.mount.barrel_od_mm = [[500.0, 24.0], [819.0, 20.5]]  # narrows: apex ahead, above the lens
    assert cb.barrel_apex_dir_g(cfg)[2] > 0.0


def test_barrel_boresight_refuses_an_image_without_a_barrel(tmp_path):
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((800, 1280), 150, np.uint8))
    assert cb.main(["--calib-dir", str(tmp_path), "--barrel", str(path)]) == 1
    assert not (tmp_path / "boresight.json").exists()
