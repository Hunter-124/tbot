"""CalRig B analysis: synthetic runs in the docs/CALRIG_PROTOCOL.md section 4 format with known truth.

Truth used (unless a test says otherwise): gyro bias (0.0061, -0.0037, 0.0021) rad/s, gyro scale error
(+1.5 %, -2.0 %, +1.0 %) on chip x/y/z, IMU timestamps 3 ms LATE, 0.05 deg/s gyro noise (white), chip mounted
2 deg off the nominal axis map. Tolerances are stated at each assert and are > 3 sigma of the reported uncertainty
of the seeded run.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tbot.calrig import analysis as A
from tbot.calrig.synth import NOMINAL_R_GUN_IMU, rot_about, write_synthetic_run

BIAS = np.array([0.0061, -0.0037, 0.0021])
SCALE_ERR = (0.015, -0.02, 0.01)
R_TRUE = rot_about((1.0, 0.0, 0.0), 2.0) @ NOMINAL_R_GUN_IMU
DPS = math.pi / 180.0


def _angle(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


@pytest.fixture(scope="module")
def upright(tmp_path_factory):
    d = tmp_path_factory.mktemp("run_upright")
    truth = write_synthetic_run(d, pose="upright", R_gun_imu=R_TRUE, gyro_bias_rad_s=BIAS, gyro_scale_err=SCALE_ERR,
                                time_offset_s=0.003, ring_hz=4.5, ring_dps=1.0, hall_jitter_ustep=1.0, seed=11)
    return d, truth, A.analyze_run(d)


@pytest.fixture(scope="module")
def poses(tmp_path_factory):
    """upright, inverted, left/right side down, muzzle up: one analysed run each."""
    out = {}
    for i, pose in enumerate(("upright", "left_side_down", "right_side_down", "inverted", "muzzle_up")):
        d = tmp_path_factory.mktemp("run_" + pose)
        write_synthetic_run(d, pose=pose, R_gun_imu=R_TRUE, gyro_bias_rad_s=BIAS, gyro_scale_err=SCALE_ERR,
                            time_offset_s=0.003, seed=20 + i)
        out[pose] = (d, A.analyze_run(d))
    return out


# ------------------------------------------------------------------------------------------- loading
def test_load_run_reads_spec_files(upright):
    d, truth, _ = upright
    run = A.load_run(d)
    assert run.meta["pose"] == "upright"
    assert run.imu["gyro"].shape[1] == 3 and run.imu["accel"].shape == run.imu["gyro"].shape
    assert run.imu["t_ns"].dtype == np.int64 and np.all(np.diff(run.imu["t_ns"]) > 0)
    assert run.rig["pos_ustep"].size == run.rig["t_ns"].size > 1000
    assert {"BOOT", "HOMED", "DONE"} <= set(run.events["code"])
    assert any(c.startswith("HOME") for c in run.cmd)
    assert run.hall["level"].size > 0


# ------------------------------------------------------------------------------------------- single run
def test_bias_recovered(upright):
    _, _, res = upright
    b = res["estimates"]["gyro_bias"]
    assert b["pass"] is True
    err_dps = (np.array(b["value_rad_s"]) - BIAS) / DPS
    assert np.all(np.abs(err_dps) < 0.01), err_dps            # tolerance 0.01 deg/s (observed ~0.003)
    assert max(b["sigma_dps"]) < 0.02
    assert b["n_segments"] >= 4
    assert b["temp_c"] == pytest.approx(31.4, abs=0.5)


def test_gravity_direction(upright):
    _, truth, res = upright
    g = res["estimates"]["gravity"]
    want = truth["R_gun_imu"].T @ np.array([0.0, 0.0, 1.0])
    assert g["pass"] is True
    assert np.degrees(np.arccos(np.dot(g["dir"], want))) < 0.1


def test_yaw_axis_identified(upright):
    _, truth, res = upright
    y = res["estimates"]["yaw_axis"]
    assert (y["chip_axis_index"], y["sign"]) == (truth["axis_index"], truth["axis_sign"]) == (0, -1)
    assert y["tilt_deg"] == pytest.approx(2.0, abs=0.3)        # mounted 2 deg off; scale errors distort ~0.07
    assert y["leakage_pct"]["y"] == pytest.approx(3.5, abs=0.5)
    assert y["axis_vs_gravity_deg"] < 0.5
    assert y["pass"] is True


def test_scale_recovered(upright):
    _, _, res = upright
    s = res["estimates"]["gyro_scale"]
    assert s["method"] == "integral" and s["n_moves"] >= 4
    assert s["per_direction"]["ccw"]["n"] >= 2 and s["per_direction"]["cw"]["n"] >= 2
    assert s["error_pct"] == pytest.approx(1.5, abs=0.1)       # tolerance 0.1 % (observed < 0.02)
    assert s["spread_pct"] < 0.2 and s["sigma_pct"] < 0.1
    assert s["pass"] is True


def test_time_offset_recovered(upright):
    _, _, res = upright
    t = res["estimates"]["time_offset"]
    assert t["late_ms"] == pytest.approx(3.0, abs=0.5)         # tolerance 0.5 ms; sigma ~0.2 ms (smooth profile)
    assert abs(t["late_ms"] - 3.0) < 3 * t["sigma_ms"]
    assert t["time_offset_ns"] == -round(t["late_s"] * 1e9)    # key is ADDED to IMU stamps: negative of "late"
    assert t["correlation"] > 0.99 and not t["at_search_limit"]
    assert t["pass"] is True


def test_tracking_and_ringing(upright):
    _, _, res = upright
    tr = res["estimates"]["tracking"]
    assert tr["rms_deg"] < 0.1 and tr["max_deg"] < 0.3 and tr["pass"] is True
    assert tr["raw_rms_deg"] > 0.5                              # unremoved 1.5 % scale shows up in the raw residual
    rg = res["estimates"]["ringing"]
    assert rg["visible"] is True
    assert rg["freq_hz"] == pytest.approx(4.5, abs=max(0.3, rg["resolution_hz"]))
    assert rg["amp_initial_dps"] == pytest.approx(0.6, abs=0.3)


def test_hall_repeatability_clean_run(upright):
    _, _, res = upright
    h = res["estimates"]["hall_repeatability"]
    assert h["pass"] is True
    assert h["repeatability_deg"] <= 3 * 0.1125 + 1e-9         # jitter sigma 1 ustep -> range of a few usteps
    assert h["n_referenced_offsets"] == 0                       # only one HOMED: nothing to compare it with


def test_verdict_pass_and_json_safe(upright):
    _, _, res = upright
    assert res["verdict"] == "PASS", [c for c in res["checks"] if c["pass"] is not True]
    json.dumps(res)                                              # no numpy / NaN leaks


def test_negative_time_offset_and_rig_jitter(tmp_path):
    """IMU stamps EARLY by 2.2 ms; rig t_ns carries 2 ms of host jitter (t_us is exact and wraps)."""
    write_synthetic_run(tmp_path, R_gun_imu=R_TRUE, time_offset_s=-0.0022, rig_jitter_s=0.002, seed=5,
                        gyro_bias_rad_s=BIAS, gyro_scale_err=SCALE_ERR)
    res = A.analyze_run(tmp_path)
    assert res["stats"]["rig_clock"]["source"] == "t_us_fit"
    assert res["stats"]["rig_clock"]["jitter_us"] == pytest.approx(2000.0, rel=0.2)
    t = res["estimates"]["time_offset"]
    assert t["late_ms"] == pytest.approx(-2.2, abs=1.0)
    assert t["time_offset_ns"] > 0                                # early stamps -> add a positive offset


def test_sine_only_run_uses_slope_scale(tmp_path):
    prog = (("still", 3.0), ("sine", 8.0, 2.5, 4), ("still", 4.0))
    write_synthetic_run(tmp_path, R_gun_imu=R_TRUE, program=prog, gyro_bias_rad_s=BIAS, gyro_scale_err=SCALE_ERR,
                        seed=6)
    s = A.analyze_run(tmp_path)["estimates"]["gyro_scale"]
    assert s["method"] == "slope"
    assert s["error_pct"] == pytest.approx(1.5, abs=0.3)          # tolerance 0.3 %


def test_lost_steps_flagged(tmp_path):
    """The table slips 16 usteps (one full step, 1.8 deg) uncounted after the first move; the last HOMED offset
    reports it. Hall edges, HOMED offset and the gyro-vs-rig tracking must all flag it."""
    write_synthetic_run(tmp_path, R_gun_imu=R_TRUE, lost_ustep=[(1, 16)], homed=((0.5, 40), (38.0, 16)), seed=7,
                        gyro_bias_rad_s=BIAS, gyro_scale_err=SCALE_ERR)
    res = A.analyze_run(tmp_path)
    h = res["estimates"]["hall_repeatability"]
    assert h["pass"] is False
    assert h["repeatability_deg"] == pytest.approx(1.8, abs=0.35)
    assert h["worst_homed_offset_deg"] == pytest.approx(1.8, abs=0.01)
    assert h["lost_full_steps_est"] == 1
    assert res["estimates"]["tracking"]["max_deg"] > 1.2
    assert res["verdict"] == "FAIL"


def test_homed_offset_needs_reference(tmp_path):
    """A ZERO command between two HOMED events un-references the second offset."""
    write_synthetic_run(tmp_path, R_gun_imu=R_TRUE, homed=((0.5, 40), (38.0, 9)), seed=8)
    with (tmp_path / "cmd.csv").open("a") as fh:
        fh.write("%d,ZERO\n" % (1_700_000_000_000_000_000 + int(20e9)))
    h = A.analyze_run(tmp_path)["estimates"]["hall_repeatability"]
    assert h["n_referenced_offsets"] == 0


def test_insufficient_data_does_not_raise(tmp_path):
    write_synthetic_run(tmp_path / "nomotion", R_gun_imu=R_TRUE, program=(("still", 12.0),), seed=9)
    res = A.analyze_run(tmp_path / "nomotion")
    assert res["verdict"] == "INCOMPLETE"
    assert res["estimates"]["gyro_bias"]["status"] == "ok"
    for k in ("yaw_axis", "time_offset", "gyro_scale", "tracking", "ringing"):
        assert res["estimates"][k]["pass"] is None
    write_synthetic_run(tmp_path / "nostill", R_gun_imu=R_TRUE, program=(("move", 60.0), ("move", -60.0)), seed=9)
    res = A.analyze_run(tmp_path / "nostill")
    assert res["estimates"]["gyro_bias"]["pass"] is None
    assert res["estimates"]["yaw_axis"]["status"] == "ok"       # motion analysis still runs without a bias
    assert res["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------------------------------- combine
def test_combine_recovers_R_gun_imu(poses):
    res = A.combine_runs([poses[k][1] for k in ("upright", "left_side_down", "right_side_down", "inverted")],
                         R_config=NOMINAL_R_GUN_IMU)
    fit = res["fit"]
    R = np.array(fit["R_gun_imu"])
    assert _angle(R @ R_TRUE.T) < 0.3                            # tolerance 0.3 deg (observed ~0.05)
    assert fit["residual_rms_deg"] < 0.3
    assert fit["gyro_only_vs_accel_only_deg"] < 0.3
    assert fit["axis_map_text"] == ["gun X = -chip Z", "gun Y = -chip Y", "gun Z = -chip X"]
    assert fit["snapped_deviation_deg"] == pytest.approx(2.0, abs=0.3)
    cmp_ = res["compare"]
    assert cmp_["config_diff_deg"] == pytest.approx(2.0, abs=0.3) and cmp_["snapped_matches_config"] is True
    assert cmp_["layout_identity_diff_deg"] > 170.0              # identity (layout.md as-built) is a 180 deg flip away
    assert res["poses"] == ["inverted", "left_side_down", "right_side_down", "upright"]
    assert res["verdict"] == "PASS"
    # aggregates: bias, time offset and the two measurable chip axes
    assert np.all(np.abs((np.array(res["gyro_bias"]["value_rad_s"]) - BIAS) / DPS) < 0.01)
    assert res["time_offset"]["late_ms"] == pytest.approx(3.0, abs=1.0)
    f = res["gyro_scale"]["factor"]
    assert f[0] == pytest.approx(1.015, abs=0.001) and f[1] == pytest.approx(0.98, abs=0.001) and f[2] is None
    assert res["gyro_scale"]["axes_missing"] == ["z"]


def test_combine_flags_wrong_config_map(poses):
    res = A.combine_runs([poses[k][1] for k in ("upright", "left_side_down")], R_config=np.eye(3))
    assert res["compare"]["config_diff_deg"] > 170.0
    assert next(c for c in res["checks"] if c["name"] == "R_matches_config")["pass"] is False
    assert res["verdict"] == "FAIL"


def test_combine_needs_two_axes(poses):
    res = A.combine_runs([poses[k][1] for k in ("upright", "inverted")], R_config=NOMINAL_R_GUN_IMU)
    assert "fit" not in res and res["verdict"] == "INCOMPLETE"
    assert any("not parallel" in w for w in res["warnings"])
    assert "R_gun_imu" not in A.propose_calib(res)


def test_combine_all_three_axes_gives_gyro_scale(poses):
    res = A.combine_runs([poses[k][1] for k in ("upright", "left_side_down", "muzzle_up")], R_config=NOMINAL_R_GUN_IMU)
    assert res["gyro_scale"]["axes_missing"] == []
    assert res["gyro_scale"]["factor"][2] == pytest.approx(1.01, abs=0.001)
    prop = A.propose_calib(res)
    assert prop["gyro_scale"] == pytest.approx([1 / 1.015, 1 / 0.98, 1 / 1.01], abs=0.001)
    assert _angle(np.array(prop["R_gun_imu"]) @ R_TRUE.T) < 0.3


# ------------------------------------------------------------------------------------------- proposal / report
def test_propose_single_run_and_apply(upright, tmp_path):
    _, _, res = upright
    prop = A.propose_calib(res)
    assert set(k for k in prop if not k.startswith("_")) == {"gyro_bias_rad_s", "temp_c", "time_offset_ns"}
    assert "R_gun_imu" in prop["_blocked"] and "gyro_scale" in prop["_blocked"]
    assert prop["time_offset_ns"] == pytest.approx(-3.0e6, abs=1.0e6)

    calib = tmp_path / "calib"
    dry = A.apply_to_calib(prop, calib)                          # default dry_run=True: nothing written
    assert dry["written"] is False and not (calib / "imu.json").exists()
    assert "gyro_bias_rad_s" in dry["patch"] and "_blocked" not in dry["patch"]

    (calib).mkdir()
    (calib / "imu.json").write_text(json.dumps({"accel_bias_m_s2": [0.1, 0.2, 0.3], "R_gun_imu": np.eye(3).tolist()}))
    out = A.apply_to_calib(prop, calib, dry_run=False)
    assert out["written"] is True
    on_disk = json.loads((calib / "imu.json").read_text())
    assert on_disk["accel_bias_m_s2"] == [0.1, 0.2, 0.3]           # untouched keys survive (update_json deep merge)
    assert on_disk["gyro_bias_rad_s"] == pytest.approx(prop["gyro_bias_rad_s"])
    assert "_blocked" not in on_disk and "created" in on_disk

    from tbot.config import load_config
    from tbot.imu.mpu6050 import ImuCalibration
    ic = ImuCalibration.from_config(load_config(), SimpleNamespace(imu=on_disk))
    assert ic.time_offset_ns == prop["time_offset_ns"]
    assert ic.gyro_bias_rad_s == pytest.approx(on_disk["gyro_bias_rad_s"])


def test_write_report(upright, poses, tmp_path):
    d, _, res = upright
    md = A.write_report(res, d)
    assert md.name == "report.md" and md.exists()
    rep = json.loads((d / "report.json").read_text())
    assert rep["verdict"] == "PASS" and rep["estimates"]["gyro_scale"]["error_pct"] == pytest.approx(1.5, abs=0.1)
    text = md.read_text()
    assert "positive = IMU timestamps late" in text and "hall_repeat_max_deg" in text
    comb = A.combine_runs([poses[k][1] for k in ("upright", "left_side_down")], R_config=NOMINAL_R_GUN_IMU)
    md2 = A.write_report(comb, tmp_path)
    assert "R_gun_imu" in md2.read_text() and json.loads((tmp_path / "report.json").read_text())["kind"] == "combined"


def test_limits_documented():
    assert set(A.LIMITS) == set(A.LIMIT_BASIS)
    assert all(A.LIMIT_BASIS[k].strip() for k in A.LIMITS)
