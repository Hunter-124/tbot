"""Multi-clay scenarios, breaks and debris [sim slice]: doubles render, hits leave fragments, debris shots count."""

from __future__ import annotations

import hashlib
import math
import platform
from types import SimpleNamespace

import numpy as np
import pytest

from tbot.camera.synthetic import T0_NS
from tbot.sim.scenario import SCENARIOS, make_scenario
from tbot.tools.sim_eval import (
    DEBRIS_RADII, evaluate_run, format_scenario_summary, format_table, shot_outcome, summarize_runs,
)
from test_synthetic import all_frames, make_calib, make_cfg, open_source

SINGLE = ("skeet_crossing", "skeet_incoming", "skeet_outgoing", "trap_straightaway", "static_gun", "no_clay")
DOUBLES = ("skeet_doubles", "trap_doubles")


def ns(t_s: float) -> int:
    return T0_NS + int(round(t_s * 1e9))


def darkest_near(image, px, half=3) -> float:
    """Darkest pixel within half px of px, relative to the median of a wider window (negative = darker)."""
    u, v = int(round(px[0])), int(round(px[1]))
    g = image.astype(np.float64)
    wide = g[max(v - 12, 0):v + 13, max(u - 12, 0):u + 13]
    near = g[max(v - half, 0):v + half + 1, max(u - half, 0):u + half + 1]
    return float(near.min() - np.median(wide))


# ------------------------------------------------------------------------------------ single clay unchanged
# sha256 over every frame, its truth (clay_px, visible, clay_pos_w, R_wg) and the IMU stream, seed 3 at
# 30 fps, recorded on the code BEFORE multi-clay support (commit efe8893) on x86-64 Windows. The exact
# bytes depend on the float/OpenCV build, so the check only runs on x86-64.
_PRE_MULTI_CLAY_SHA256 = {
    "skeet_crossing": "73d9b32b8bd5d9727c73231c645f48a2959f1c20ce6861040c987db289c8610c",
    "skeet_incoming": "7bd5cdfbf2875b888c81beabae574997327cdc00f5a68ee4375a53a0c35598e7",
    "skeet_outgoing": "1a9ff91608affb7a864b94674b87ee4e689d44c00d3a569f56689323a603c3ae",
    "trap_straightaway": "cc57e864119b24b5ea4b7cef0bc41d5505a5e684a8c32587f4f67b9e81deb9f8",
    "static_gun": "0b3e01d5518018fc7bd359717eb50d44a91714230b51ff2faaa8868274094516",
    "no_clay": "10526ca5375818a656b9be58b42689448c59e37b8882794817aa746062beff38",
}


def stream_hash(name: str, seed: int = 3) -> str:
    src = open_source(make_cfg(), name, seed=seed)
    imu = src.imu()
    imu.start()
    h = hashlib.sha256()
    while True:
        f = src.read()
        if f is None:
            break
        h.update(f.image.tobytes())
        tr = src.truth(f.t_exposure_ns)
        h.update(repr((tr["clay_px"], tr["visible"])).encode())
        if tr["clay_pos_w"] is not None:
            h.update(np.asarray(tr["clay_pos_w"]).tobytes())
        h.update(np.asarray(tr["R_wg"]).tobytes())
        for s in imu.read():
            h.update(np.asarray(s.gyro).tobytes())
            h.update(np.asarray(s.accel).tobytes())
    return h.hexdigest()


# The same streams on x86-64 Linux with OpenCV 5 / NumPy 2 (Fedora 44, 2026-11-19): the renderer's bytes differ
# from the Windows build above, so that build has its own set. Recorded on the code of commit e03ff3e (the one that
# added the Windows hashes) and equal there to the code of 2026-11-19: nothing had changed the streams in between.
_PRE_MULTI_CLAY_SHA256_LINUX_CV5 = {
    "skeet_crossing": "41131a5c7c4d6a2ee7b35ce04e741ebb6b27a80228054fba211a0370d21fb599",
    "skeet_incoming": "57c33432d4004f5b28756375ffaf0fb3e2d4738d44d20a5af507244836b4e990",
    "skeet_outgoing": "886e1dfe410463f6fbf1f034ffb56b8229c7e955fa1da4ec5dda5166d7f8d3de",
    "trap_straightaway": "74f7237c046033cdcf332cab86a0ed6760df53e2a50b0cddfb0d0e074b2d84e5",
    "static_gun": "6ba94992d46e11567690ae8cf903ebdaaa2ee185861263c7e2e908f52295e3cf",
    "no_clay": "6a33efc5be01332377e3a2eff8888bcdd686c0c9e35905a9546f8741c880aee7",
}


def _golden_hashes() -> dict | None:
    """The golden set of this machine's build; None where none was recorded (the bytes depend on the float and
    OpenCV build, so a set is only good for the build it was recorded on)."""
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return None
    if platform.system() == "Windows":
        return _PRE_MULTI_CLAY_SHA256
    import cv2

    if platform.system() == "Linux" and cv2.__version__.split(".")[0] == "5" and np.__version__.split(".")[0] == "2":
        return _PRE_MULTI_CLAY_SHA256_LINUX_CV5
    return None


@pytest.mark.parametrize("name", SINGLE)
def test_single_clay_scenarios_are_byte_identical_to_before_multi_clay(name):
    golden = _golden_hashes()
    if golden is None:
        pytest.skip("no golden hashes recorded for this platform / OpenCV / NumPy build")
    assert stream_hash(name) == golden[name]


def test_single_clay_scenarios_have_exactly_one_clay():
    cfg = make_cfg()
    for name in SINGLE:
        sc = make_scenario(name, cfg, 1)
        assert sc.n_clays == (0 if name == "no_clay" else 1)
        if sc.n_clays:
            assert sc.clays[0].curve is sc.clay_curve
            assert np.array_equal(sc.clays[0].launch_vel_w, sc.launch_vel_w)


# ------------------------------------------------------------------------------------ doubles
@pytest.mark.parametrize("name", DOUBLES)
def test_doubles_render_two_clays_where_truth_says(name):
    assert name in SCENARIOS
    cfg = make_cfg(**{"sim.noise_sigma": 0.0})
    src = open_source(cfg, name, seed=1)
    sc = src.scenario
    assert sc.n_clays == 2
    assert sc.clays[0].launch_t_s == sc.clays[1].launch_t_s  # thrown together
    both = 0
    for frame in all_frames(src):
        truth = src.truth(frame.t_exposure_ns)
        pxs = [c["clay_px"] for c in truth["clays"]]
        if any(p is None for p in pxs) or max(c["range_m"] for c in truth["clays"]) > 35.0:
            continue  # a 108 mm clay much further out is only a faint pixel
        a, b = pxs
        if math.hypot(a[0] - b[0], a[1] - b[1]) < 8.0:
            continue  # overlapping pair: no separate blobs to look at
        both += 1
        for p in pxs:
            assert darkest_near(frame.image, p) < -10.0, "each clay must be drawn"
        assert truth["visible"] and truth["clay_px"] in pxs  # the primary clay is one of them
    assert both > 5, "the pair must be in the image together for a while"


def test_doubles_are_deterministic_and_the_clays_differ():
    cfg = make_cfg()
    a, b = make_scenario("skeet_doubles", cfg, 4), make_scenario("skeet_doubles", cfg, 4)
    for t in (0.0, 0.7, 1.4, 2.1):
        assert np.array_equal(a.R_wg(t), b.R_wg(t))
        assert np.array_equal(a.clay_pos_w(t, 1), b.clay_pos_w(t, 1))
    high, low = a.clays
    assert high.launch_vel_w[1] * low.launch_vel_w[1] < 0.0 or high.launch_vel_w @ low.launch_vel_w < 0.0
    assert low.launch_pos_w[2] < high.launch_pos_w[2]  # low house window below the high house's


def test_shooter_moves_to_the_second_clay_after_the_first_breaks_without_rewriting_the_past():
    cfg = make_cfg()
    ref = make_scenario("skeet_doubles", cfg, 2)
    sc = make_scenario("skeet_doubles", cfg, 2)
    t_break = 0.75
    assert sc.break_clay(0, t_break)
    for t in np.arange(0.0, t_break + 0.15, 0.01):  # nothing changes before the shooter could react
        assert np.array_equal(sc.R_wg(t), ref.R_wg(t))
    # Some time after the switch the bore is much nearer the second clay than when he stays on clay 0.
    t = t_break + 0.6
    near = lambda s: math.degrees(math.acos(float(np.clip(s.bore_dir_w(t) @ s.clay_dir_w(t, 1), -1, 1))))
    assert near(sc) < near(ref) - 5.0
    assert near(sc) < 10.0


# ------------------------------------------------------------------------------------ breaks and debris
def test_a_forced_hit_replaces_the_clay_with_fading_fragments():
    cfg = make_cfg(**{"sim.noise_sigma": 0.0})
    cfg.camera.fps = 50.0
    whole = open_source(cfg, "skeet_crossing", seed=1)
    hit = open_source(cfg, "skeet_crossing", seed=1)
    t_break = 0.9
    assert hit.break_clay(0, ns(t_break))
    assert not hit.break_clay(0, ns(t_break + 0.1)), "a clay breaks once"
    n = int(cfg.sim.debris_fragments)
    assert 4 <= n <= 10
    seen_frags = 0
    for fa, fb in zip(all_frames(whole), all_frames(hit)):
        t = fa.seq / cfg.camera.fps
        ta, tb = whole.truth(fa.t_exposure_ns), hit.truth(fb.t_exposure_ns)
        if t < t_break - 0.01:
            assert np.array_equal(fa.image, fb.image), "nothing changes before the break"
            assert len(tb["fragments_w"]) == 0
            continue
        if t < t_break + 0.005:
            continue
        assert tb["visible"] is False and tb["clay_px"] is None
        assert not tb["clays"][0]["intact"]
        if ta["clay_px"] is not None:
            # the whole clay is gone from where it would have been ...
            assert darkest_near(fb.image, ta["clay_px"], half=1) > darkest_near(fa.image, ta["clay_px"], half=1) + 10
        frag = tb["fragments_w"]
        if t < t_break + 0.3:
            assert len(frag) == n, "every fragment lives for a while"
            # ... and the fragments start at the break point and spread out with about the clay's velocity
            centre = hit.scenario.clay_pos_w(t_break, 0)
            spread = np.linalg.norm(frag - frag.mean(axis=0), axis=1)
            assert float(np.max(np.linalg.norm(frag - centre, axis=1))) > 0.0
            assert float(np.max(spread)) < 6.0 * cfg.sim.debris_spread_mps * (t - t_break) + 0.05
            diff = fb.image.astype(int) - fa.image.astype(int)
            assert (diff < -3).sum() >= 2, "the fragments are drawn"
            seen_frags += 1
        if t > t_break + 1.31 * cfg.sim.debris_life_s:
            assert len(frag) == 0, "fragments disappear after their lifetime"
            assert np.array_equal(fa.image, fb.image) or ta["clay_px"] is not None
    assert seen_frags > 5


def test_fragments_are_deterministic_per_seed_and_independent_of_other_streams():
    cfg = make_cfg()
    a, b, c = (make_scenario("trap_doubles", cfg, s) for s in (5, 5, 6))
    for sc in (a, b, c):
        assert sc.break_clay(1, 0.8)
    for t in (0.8, 0.95, 1.2):
        assert np.array_equal(a.fragments_w(t)[0], b.fragments_w(t)[0])
        assert np.array_equal(a.fragments_w(t)[2], b.fragments_w(t)[2])
    assert not np.array_equal(a.debris[1].vel0_w, c.debris[1].vel0_w)
    # rendering too
    sa, sb = open_source(cfg, "trap_doubles", seed=5), open_source(cfg, "trap_doubles", seed=5)
    sa.break_clay(0, ns(0.7))
    sb.break_clay(0, ns(0.7))
    for fa, fb in zip(all_frames(sa), all_frames(sb)):
        assert np.array_equal(fa.image, fb.image)


def test_a_miss_or_an_unlaunched_clay_cannot_break():
    sc = make_scenario("skeet_crossing", make_cfg(), 1)
    assert not sc.break_clay(0, 0.1), "not launched yet"
    assert not sc.break_clay(3, 1.0), "no such clay"
    assert not sc.broken and not sc.debris
    assert sc.clay_visible(1.0)


def test_starting_the_source_again_throws_whole_clays():
    src = open_source(make_cfg(), "skeet_doubles", seed=1)
    ref = make_scenario("skeet_doubles", make_cfg(), 1)
    assert src.break_clay(0, ns(0.8))
    src.start()
    assert not src.scenario.broken and not src.scenario.debris
    assert np.array_equal(src.scenario.R_wg(2.0), ref.R_wg(2.0))


# ------------------------------------------------------------------------------------ evaluation
class StubShot:
    """Straight, drop-free shot with a fixed time of flight: the judge's geometry becomes exact."""

    TOF_S = 0.05

    def time_of_flight(self, r):
        return self.TOF_S

    def drop(self, r, el):
        return 0.0

    def pattern_radius(self, r):
        return 0.4


def aimed_mount(sc, t_exit_s, dir_w):
    """A mount whose bore at t_exit_s points along dir_w (camera = muzzle at the origin)."""
    R = sc.R_wg(t_exit_s)
    return SimpleNamespace(muzzle_pos_g=np.zeros(3), cam_pos_g=np.zeros(3), bore_dir_g=R.T @ dir_w)


def judge(src, dir_w, t_fire_s, delay_s=0.08):
    mount = aimed_mount(src.scenario, t_fire_s + delay_s, dir_w)
    return shot_outcome(src, mount, StubShot(), make_cfg(), ns(t_fire_s), delay_s)


def test_shot_at_debris_counts_as_a_debris_shot_and_never_as_a_hit():
    cfg = make_cfg()
    src = open_source(cfg, "skeet_crossing", seed=1)
    sc = src.scenario
    t_break, t_fire = 0.9, 1.0
    assert src.break_clay(0, ns(t_break))
    t_arrive = t_fire + 0.08 + StubShot.TOF_S
    frag = sc.debris[0].positions(t_arrive)[0]
    out = judge(src, frag / np.linalg.norm(frag), t_fire)
    assert out["hit"] is False and out["clay"] is None
    assert out["debris_shot"] is True
    assert out["debris_miss_m"] < 0.05
    assert out["reason"] == "clay already broken at shot arrival"

    # aimed well away from everything: an empty shot, not a debris shot
    away = sc.R_wg(t_fire)[:, 1]
    out = judge(src, away, t_fire)
    assert out["hit"] is False and out["debris_shot"] is False
    assert out["debris_miss_m"] > DEBRIS_RADII * 0.4


def test_shot_judged_against_every_clay_hits_the_one_it_points_at():
    cfg = make_cfg()
    src = open_source(cfg, "skeet_doubles", seed=1)
    sc = src.scenario
    t_fire = 1.0
    t_arrive = t_fire + 0.08 + StubShot.TOF_S
    for i in (1, 0):
        out = judge(src, sc.clay_dir_w(t_arrive, i), t_fire)
        assert out["hit"] is True and out["clay"] == i and out["target_clay"] == i
        assert out["miss_m"] < 0.01
        assert out["debris_shot"] is False
    # break clay 0, then shoot through its fragments at the intact clay 1: a hit, not a debris shot
    assert src.break_clay(0, ns(0.95))
    out = judge(src, sc.clay_dir_w(t_arrive, 1), t_fire)
    assert out["hit"] is True and out["clay"] == 1 and out["debris_shot"] is False
    # the broken clay can no longer be hit
    out = judge(src, sc.clay_dir_w(t_arrive, 0), t_fire)
    assert out["hit"] is False and out["clay"] is None


def test_summary_counts_clays_shots_and_debris():
    def run(scenario, thrown, shots):
        broken = sorted({s["clay"] for s in shots if s["hit"]})
        return {
            "scenario": scenario, "seed": 1, "frames": 100, "visible_frames": 50, "detection_rate": 1.0,
            "track_latency_ms": 40.0, "lead_error_deg": {"p50": 0.1, "p95": 0.2},
            "fires": len(shots), "hits": len(broken), "miss_m": {"p50": 0.1}, "shots": shots,
            "clays_thrown": thrown, "clays_broken": len(broken),
            "empty_shots": sum(not s["hit"] for s in shots),
            "debris_shots": sum(s["debris_shot"] for s in shots),
            "break_after_launch_s": [0.6] * len(broken),
        }

    hit0 = {"hit": True, "clay": 0, "debris_shot": False, "miss_m": 0.1}
    hit1 = {"hit": True, "clay": 1, "debris_shot": False, "miss_m": 0.2}
    debris = {"hit": False, "clay": None, "debris_shot": True, "miss_m": 1.0}
    empty = {"hit": False, "clay": None, "debris_shot": False, "miss_m": 3.0}
    runs = [run("skeet_doubles", 2, [hit0, hit1]), run("skeet_doubles", 2, [hit0, debris]),
            run("skeet_crossing", 1, [empty])]
    s = summarize_runs(runs)
    d = s["by_scenario"]["skeet_doubles"]
    assert (d["shots"], d["clays_broken"], d["clays_thrown"], d["debris_shots"], d["empty_shots"]) == (4, 3, 4, 1, 1)
    assert d["shots_per_run"] == 2.0 and d["break_after_launch_s"]["n"] == 3
    assert (s["shots"], s["clays_broken"], s["clays_thrown"], s["debris_shots"]) == (5, 3, 5, 1)
    assert s["fires"] == 5 and s["hits"] == 3  # the old totals still work
    table = format_table(runs)
    assert "2/2" in table and "debris" in table
    assert "total" in format_scenario_summary(s)


def test_evaluate_run_on_doubles_reports_the_new_fields(tmp_path):
    cfg = make_cfg()
    res = evaluate_run(cfg, make_calib(cfg), "trap_doubles", 1, max_frames=60)
    assert res["clays_thrown"] == 2
    assert res["clays_broken"] == res["hits"] == len(res["break_after_launch_s"])
    assert res["debris_shots"] <= res["empty_shots"] == res["fires"] - res["hits"]
    for shot in res["shots"]:
        assert "debris_shot" in shot and "clay" in shot
