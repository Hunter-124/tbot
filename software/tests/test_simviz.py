"""tbot simviz: the graphics watch a sim run without changing it, and draw it (short runs, capped frames)."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from tbot.config import load_calibration, load_config
from tbot.tools import sim_eval, simviz

SCENARIO, SEED, FRAMES = "trap_straightaway", 1, 50  # fires and breaks its clay within the first ~50 frames


def _cfg(tmp_path):
    cfg = load_config()
    cfg.paths.calib_dir = str(tmp_path / "calib")
    return cfg, load_calibration(cfg)


def _canon(res: dict) -> str:
    def fix(x):
        if isinstance(x, float) and not math.isfinite(x):
            return repr(x)
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        if isinstance(x, np.generic):
            return fix(x.item())
        return x

    return json.dumps(fix(res), sort_keys=True, default=str)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    cfg, calib = _cfg(tmp_path_factory.mktemp("simviz"))
    rec, res = simviz.record_run(cfg, calib, SCENARIO, SEED, max_frames=FRAMES)
    plain = sim_eval.evaluate_run(cfg, calib, SCENARIO, SEED, max_frames=FRAMES)
    return cfg, calib, rec, res, plain


def test_watching_a_run_changes_nothing_in_it(recorded):
    _, _, rec, res, plain = recorded
    assert _canon(res) == _canon(plain)
    assert len(rec.steps) == res["frames"] == FRAMES
    assert len(rec.shots) == res["fires"]
    for st in rec.steps:
        assert st.image is not None and st.image.shape[:2] == (int(rec.cam.height), int(rec.cam.width))
    for g, s in zip(rec.shots, res["shots"]):  # the drawn shot is the judged shot
        assert g.outcome == ("hit" if s["hit"] else ("debris" if s["debris_shot"] else "miss"))
        assert g.range_m == pytest.approx(s["range_m"])
        if g.clay_at_w is not None:  # pattern centre to the clay: the judge's miss distance
            assert float(np.linalg.norm(g.clay_at_w - g.impact_w)) == pytest.approx(s["miss_m"], abs=1e-6)


def test_render_outputs_writes_a_video_and_a_summary(recorded, tmp_path):
    _, _, rec, res, _ = recorded
    out = simviz.render_outputs(rec, res, tmp_path, fps=30.0, slowmo=1.0, hold_s=0.2, max_video_frames=24)
    png = cv2.imread(str(out["png"]))
    assert png is not None and png.shape == (simviz.SUM_H, simviz.SUM_W, 3)
    assert float(png.std()) > 10.0  # not blank
    cap = cv2.VideoCapture(str(out["video"]))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok and frame.shape == (simviz.VIDEO_H, simviz.VIDEO_W, 3)
    assert float(frame.std()) > 10.0


def test_frames_show_the_shot_and_its_outcome(recorded):
    _, _, rec, res, _ = recorded
    scene = simviz.build_scene(rec, res)
    for t in (0.0, 0.5 * scene.t_end_video, scene.t_end_video):
        img = simviz.video_frame(scene, t, 0.25)
        assert img.shape == (simviz.VIDEO_H, simviz.VIDEO_W, 3) and img.dtype == np.uint8
    if not scene.shots:
        pytest.skip("this build did not fire in the capped run")
    s = scene.shots[0]
    assert s.t_fire < s.t_exit < s.t_arrive
    assert s.pellet(0.5 * (s.t_exit + s.t_arrive)) is not None and s.pellet(s.t_arrive + 0.01) is None
    after = simviz.video_frame(scene, min(s.t_arrive + 0.05, scene.t_end_video), 0.25)
    colour = np.array(simviz.OUTCOME[s.outcome], dtype=np.int16)
    assert int((np.abs(after.astype(np.int16) - colour).sum(axis=2) < 12).sum()) > 50  # the outcome is drawn


def test_clay_zoom_is_opt_in_and_only_touches_the_camera_panel(recorded):
    _, _, rec, res, _ = recorded
    scene = simviz.build_scene(rec, res)
    idx = next((i for i, st in enumerate(rec.steps) if st.truth_px), None)
    if idx is None:
        pytest.skip("no frame with the clay in view in the capped run")
    t = rec.steps[idx].t_s
    plain = simviz.video_frame(scene, t, 0.25)
    marked = simviz.video_frame(scene, t, 0.25, clay_zoom=True)
    assert np.array_equal(plain, simviz.video_frame(scene, t, 0.25, clay_zoom=False))  # default unchanged
    _, cam, *_ = simviz._layout(*simviz._Renderer(scene).cam_size)
    x, y, w, h = cam
    diff = np.abs(plain.astype(np.int16) - marked.astype(np.int16)).sum(axis=2) > 0
    assert diff[y:y + h, x:x + w].any()  # the window is drawn in the camera panel...
    diff[y:y + h, x:x + w] = False
    assert not diff.any()  # ...and nowhere else


def test_contact_sheet_and_reason_grouping(tmp_path):
    a = np.full((90, 160, 3), 40, np.uint8)
    b = np.full((90, 160, 3), 200, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), a)
    cv2.imwrite(str(tmp_path / "b.png"), b)
    p = simviz.contact_sheet([tmp_path / "a.png", tmp_path / "b.png", tmp_path / "a.png"], tmp_path / "sheet.png",
                             cols=2, scale=1.0)
    sheet = cv2.imread(str(p))
    assert sheet.shape[1] == 2 * 160 + 3 * 8 and sheet.shape[0] > 2 * 90
    assert simviz.reason_kind("p_hit 0.10 < 0.21") == simviz.reason_kind("p_hit 0.00 < 0.33")
    assert simviz.reason_kind("track young") == "track young"


def test_cli_and_sim_render_flag(tmp_path, capsys):
    calib = str(tmp_path / "calib")
    assert simviz.main(["--calib-dir", calib, "--scenarios", "no_clay", "--max-frames", "12", "--out",
                        str(tmp_path / "viz"), "--slowmo", "1", "--hold", "0"]) == 0
    assert (tmp_path / "viz" / "no_clay_seed1.png").exists()
    assert (tmp_path / "viz" / "no_clay_seed1.mp4").exists() or (tmp_path / "viz" / "no_clay_seed1.avi").exists()
    capsys.readouterr()
    assert sim_eval.main(["--calib-dir", calib, "--scenarios", "no_clay", "--max-frames", "12"]) == 0
    plain = capsys.readouterr().out
    assert sim_eval.main(["--calib-dir", calib, "--scenarios", "no_clay", "--max-frames", "12",
                          "--render", str(tmp_path / "r")]) == 0
    drawn = capsys.readouterr().out
    assert (tmp_path / "r" / "no_clay_seed1.png").exists()
    # the report is the same; the flag only adds the lines naming what it drew
    assert [ln for ln in drawn.splitlines() if "drawn:" not in ln] == plain.splitlines()
