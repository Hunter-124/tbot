"""HITL analysis: raw bench evidence -> per-stage latencies, display timing, lead accuracy, detection.

Pure functions of the run's `raw` dict (written as raw.json), so `tbot hitl analyze <run>` reproduces the
report offline. Time bases: rig frames carry monotonic ns; the page log carries rig_ms relative to
`meta.epoch_ns`. The definitions of every interval are in docs/HITL.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tbot.config import Config, LatencyBudget, _build
from tbot.fire.ballistics import ShotModel
from tbot.hitl.geometry import ScreenMap
from tbot.optics import CameraModel, MountModel, range_from_size
from tbot.stats import summarize

SPEED_OF_SOUND_MPS = 343.0
_UP = np.array([0.0, 0.0, 1.0])
#: Fractions of the on-off swing that count as "dark" / "lit" for a display-test frame.
PARTIAL_LO, PARTIAL_HI = 0.05, 0.95
#: Display-test frames this long after the draw are assumed fully lit (display latency is far below it).
LIT_AFTER_MS = 100.0


@dataclass
class Context:
    cfg: Config
    cam: CameraModel
    mount: MountModel
    shot: ShotModel
    budget: LatencyBudget
    smap: ScreenMap | None
    epoch_ns: int
    page_t: np.ndarray  # rig_ms per logged display frame
    page_ids: list
    frames: list
    mic_distance_m: float
    first_idx: dict | None = None

    def ns(self, rig_ms: float) -> int:
        return self.epoch_ns + int(round(float(rig_ms) * 1e6))

    def first_drawn_ns(self, item_id: int) -> int | None:
        if self.first_idx is None:
            self.first_idx = {}
            for k, ids in enumerate(self.page_ids):
                for i in ids:
                    self.first_idx.setdefault(i, k)
        k = self.first_idx.get(int(item_id))
        return None if k is None else self.ns(self.page_t[k])

    def first_undrawn_ns(self, item_id: int, after_ms: float) -> int | None:
        """First display frame at/after `after_ms` that no longer drew the item."""
        start = int(np.searchsorted(self.page_t, after_ms - 1e-6))
        for t, ids in zip(self.page_t[start:], self.page_ids[start:]):
            if item_id not in ids:
                return self.ns(t)
        return None


def context_from_raw(raw: dict) -> Context:
    meta = raw["meta"]
    cfg = _build(Config, meta["config"], "")
    cal = meta.get("calibration") or {}
    cam = CameraModel.from_calibration(cal.get("intrinsics"), cfg, int(meta["camera"]["width"]),
                                       int(meta["camera"]["height"]))
    mount = MountModel.from_config(cfg, cal.get("boresight"))
    budget = LatencyBudget.from_config(cfg, cal.get("latency"))
    geo = raw.get("geometry")
    smap = ScreenMap.from_dict(geo["map"], cam) if geo else None
    page = raw.get("page") or {}
    order = np.argsort(np.asarray(page.get("t", []), dtype=np.float64), kind="stable")
    t = np.asarray(page.get("t", []), dtype=np.float64)[order]
    ids = [set(page["ids"][i]) for i in order]
    return Context(cfg=cfg, cam=cam, mount=mount, shot=ShotModel(cfg), budget=budget, smap=smap,
                   epoch_ns=int(meta["epoch_ns"]), page_t=t, page_ids=ids, frames=raw["frames"],
                   mic_distance_m=float(meta.get("mic_distance_m", 0.0)))


def _frames(ctx: Context, span) -> list[dict]:
    i0, i1 = span
    return ctx.frames[int(i0):int(i1)]


def _exp_ns(rec: dict) -> float:
    return float(rec["exp_us"]) * 1e3


def _read_end_ns(rec: dict) -> float:
    s = rec["stage"]
    return rec["now"] - (s.get("imu", 0.0) + s.get("attitude", 0.0) + s.get("detect", 0.0) + s.get("track", 0.0)) * 1e6


# ------------------------------------------------------------------------------------------ page
def page_summary(ctx: Context, status: dict) -> dict:
    t = ctx.page_t
    out = {"status": status, "frames_logged": int(len(t))}
    if len(t) > 2:
        dt = np.diff(t)
        med = float(np.median(dt))
        out.update({"frame_ms_p50": med, "frame_ms_p99": float(np.percentile(dt, 99)),
                    "refresh_hz": 1000.0 / med if med > 0 else float("nan"),
                    "late_frames": int(np.sum(dt > 1.5 * med)),
                    "late_pct": float(100.0 * np.mean(dt > 1.5 * med))})
    return out


def display_period_ms(ctx: Context) -> float:
    if len(ctx.page_t) < 3:
        return float("nan")
    return float(np.median(np.diff(ctx.page_t)))


# --------------------------------------------------------------------------------------- display
def _transition(frames: list[dict], probe: str, t_draw_ns: int, off: float, on: float) -> dict | None:
    """Photon time of one level step seen by `probe` (partial exposure if any, else the frame bracket)."""
    span = on - off
    if abs(span) < 1e-6:
        return None
    seq = []
    for rec in frames:
        if probe not in rec.get("probe", {}):
            continue
        frac = (rec["probe"][probe] - off) / span
        seq.append((rec["te"], _exp_ns(rec), frac))
    seq.sort()
    k = next((j for j, (te, e, f) in enumerate(seq) if te + e / 2 >= t_draw_ns and f >= 0.5), None)
    if k is None:
        return None
    partial = [j for j in (k - 1, k) if 0 <= j < len(seq) and PARTIAL_LO < seq[j][2] < PARTIAL_HI
               and seq[j][0] + seq[j][1] / 2 >= t_draw_ns]
    if partial:
        j = min(partial, key=lambda jj: abs(seq[jj][2] - 0.5))
        te, e, f = seq[j]
        return {"photon_ns": te + e / 2 - f * e, "method": "partial", "uncert_ns": 0.0}
    dark = [j for j in range(k) if seq[j][2] <= PARTIAL_LO]
    if not dark:
        return None
    te_d, e_d, _ = seq[dark[-1]]
    te_l, e_l, _ = seq[k]
    lo, hi = te_d + e_d / 2, te_l - e_l / 2
    if hi < lo:
        lo, hi = hi, lo
    return {"photon_ns": 0.5 * (lo + hi), "method": "bracket", "uncert_ns": 0.5 * (hi - lo)}


def analyze_display(ctx: Context, disp: dict) -> dict:
    """Display latency (rAF timestamp -> photon) per row, rise and fall, with a linear scan-out model."""
    H = ctx.smap.screen_h if ctx.smap is not None else 1.0
    rise, fall, levels = [], [], []
    for tr in disp["trials"]:
        frames = _frames(ctx, tr["frames"])
        probe = tr["row"]
        t_on = ctx.first_drawn_ns(tr["item"])
        t_off = ctx.first_undrawn_ns(tr["item"], tr["t1"]) if t_on is not None else None
        if t_on is None or t_off is None:
            continue
        lv = [(r["te"], _exp_ns(r), r["probe"][probe]) for r in frames if probe in r.get("probe", {})]
        off_lv = [v for te, e, v in lv if te + e / 2 < t_on]
        on_lv = [v for te, e, v in lv if te - e / 2 > t_on + LIT_AFTER_MS * 1e6 and te + e / 2 < t_off]
        if len(off_lv) < 1 or len(on_lv) < 1:
            continue
        off, on = float(np.median(off_lv)), float(np.median(on_lv))
        levels.append({"trial": tr.get("trial"), "row": probe, "y": tr["y"],
                       "off_dn": off, "on_dn": on, "contrast_dn": on - off})
        a = _transition(frames, probe, t_on, off, on)
        if a is not None:
            rise.append({"y": tr["y"], "d_ms": (a["photon_ns"] - t_on) / 1e6, "method": a["method"],
                         "uncert_ms": a["uncert_ns"] / 1e6, "contrast_dn": on - off})
        after = [r for r in frames if r["te"] - _exp_ns(r) / 2 > t_on + LIT_AFTER_MS * 1e6]
        b = _transition(after, probe, t_off, on, off)
        if b is not None:
            fall.append({"y": tr["y"], "d_ms": (b["photon_ns"] - t_off) / 1e6, "method": b["method"],
                         "uncert_ms": b["uncert_ns"] / 1e6})

    def model(samples: list[dict]) -> dict:
        if not samples:
            return {"n": 0}
        y = np.array([s["y"] for s in samples]) / H
        d = np.array([s["d_ms"] for s in samples])
        if len(set(np.round(y, 3))) >= 2 and len(samples) >= 4:
            A = np.column_stack([np.ones_like(y), y])
            (d0, scan), *_ = np.linalg.lstsq(A, d, rcond=None)
        else:
            d0, scan = float(np.mean(d)), 0.0
        resid = d - (d0 + scan * y)
        rows = {}
        for s in samples:
            rows.setdefault(round(s["y"]), []).append(s["d_ms"])
        return {"n": len(samples), "d0_ms": float(d0), "scanout_ms": float(scan),
                "resid_std_ms": float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
                "partial": sum(1 for s in samples if s["method"] == "partial"),
                "bracket_uncert_ms": float(np.mean([s["uncert_ms"] for s in samples if s["method"] == "bracket"]))
                if any(s["method"] == "bracket" for s in samples) else 0.0,
                "per_row": {str(k): summarize(v) for k, v in sorted(rows.items())}}

    out = {"rise": model(rise), "fall": model(fall), "samples": {"rise": rise, "fall": fall},
           "levels": levels}
    warnings = []
    if any(v["on_dn"] >= 250 or v["off_dn"] >= 250 for v in levels):
        warnings.append("display levels saturated (on or off >= 250 DN): lower camera.analogue_gain / "
                        "--display-exposure-us, or change --bg/--level")
    if any(abs(v["contrast_dn"]) < 8 for v in levels):
        warnings.append("display contrast < 8 DN: raise camera.analogue_gain / --display-exposure-us, "
                        "or widen the --bg/--level difference")
    if out["rise"].get("n", 0) and out["rise"]["partial"] < 5:
        warnings.append("few partially lit frames: the display latency is from frame brackets (+-half a frame "
                        "period each); for a sub-ms number rerun `--tests display` with camera.exposure_us near half "
                        "the frame period")
    out["warnings"] = warnings
    return out


def display_latency_ms(disp_res: dict | None, y: float, H: float) -> float | None:
    if not disp_res or not disp_res.get("rise", {}).get("n"):
        return None
    m = disp_res["rise"]
    return m["d0_ms"] + m["scanout_ms"] * (y / H)


# ----------------------------------------------------------------------------------------- flash
def _target_uv(ctx: Context, xy) -> np.ndarray:
    return ctx.smap.to_pixel([xy])[0]


def _gate_px(ctx: Context, xy, r: float) -> float:
    r_cam = r * float(ctx.smap.px_angle([xy])[0]) / float(ctx.cam.ifov(np.array([_target_uv(ctx, xy)]))[0])
    return max(3.0, 1.5 * r_cam + 2.0)


def _near_det(rec: dict, uv, gate: float) -> list | None:
    best = None
    for d in rec["dets"]:
        dist = math.hypot(d[0] - uv[0], d[1] - uv[1])
        if dist <= gate and (best is None or dist < best[0]):
            best = (dist, d)
    return None if best is None else best[1]


def _actuation(ctx: Context, tr: dict) -> dict:
    """cmd->break (break-sense) and cmd->click (mic, minus the sound's flight time)."""
    out = {}
    t_cmd = tr.get("t_cmd_ns")
    if t_cmd is None:
        return out
    if tr.get("break_ns") is not None:
        out["cmd_to_break_ms"] = (tr["break_ns"] - t_cmd) / 1e6
    mic = tr.get("mic") or {}
    onset = mic.get("onset")
    if onset is not None:
        acoustic_ns = ctx.mic_distance_m / SPEED_OF_SOUND_MPS * 1e9
        out["click_ns"] = onset["t_ns"] - acoustic_ns
        out["cmd_to_click_ms"] = (out["click_ns"] - t_cmd) / 1e6
        out["click_snr"] = float(onset["snr"])
        if tr.get("break_ns") is not None:
            out["break_to_click_ms"] = (out["click_ns"] - tr["break_ns"]) / 1e6
    return out


def analyze_flash(ctx: Context, flash: dict, disp_res: dict | None) -> dict:
    H = ctx.smap.screen_h
    per = []
    for tr in flash["trials"]:
        row = {"trial": tr["trial"], "decision": tr["decision_frame"] is not None}
        t_draw = ctx.first_drawn_ns(tr["item"])
        if t_draw is None:
            row["error"] = "never drawn"
            per.append(row)
            continue
        d_ms = display_latency_ms(disp_res, tr["xy"][1], H)
        photon = t_draw + d_ms * 1e6 if d_ms is not None else None
        uv = _target_uv(ctx, tr["xy"])
        gate = _gate_px(ctx, tr["xy"], tr["r"])
        frames = _frames(ctx, tr["frames"])
        ref = photon if photon is not None else t_draw
        seen = next((f for f in frames if f["te"] + _exp_ns(f) / 2 >= ref), None)
        det = next((f for f in frames if f["te"] + _exp_ns(f) / 2 >= ref and _near_det(f, uv, gate) is not None), None)
        row["display_ms"] = d_ms
        if seen is not None and photon is not None:
            row["photon_to_exposure_end_ms"] = (seen["te"] + _exp_ns(seen) / 2 - photon) / 1e6
        if seen is not None:
            row["exposure_end_to_user_ms"] = (seen["tc"] - (seen["te"] + _exp_ns(seen) / 2)) / 1e6
            row["user_to_pipeline_ms"] = (_read_end_ns(seen) - seen["tc"]) / 1e6
        if det is not None:
            row["detect_ms"] = det["stage"]["detect"]
            row["frames_missed"] = det["i"] - seen["i"] if seen is not None else None
            row["first_det_frame"] = det["i"]
            if photon is not None:
                row["photon_to_detection_ms"] = (_read_end_ns(det) + det["stage"]["detect"] * 1e6 - photon) / 1e6
        if tr["decision_frame"] is not None:
            dec = ctx.frames[tr["decision_frame"]]
            if det is not None:
                row["track_wait_ms"] = (dec["te"] - det["te"]) / 1e6
                row["track_wait_frames"] = dec["i"] - det["i"]
            row["frame_age_at_decision_ms"] = (dec["now"] - dec["te"]) / 1e6
            row["decision_to_cmd_ms"] = (tr["t_cmd_ns"] - dec["now"]) / 1e6
            row["draw_to_cmd_ms"] = (tr["t_cmd_ns"] - t_draw) / 1e6
            if photon is not None:
                row["photon_to_decision_ms"] = (dec["now"] - photon) / 1e6
                row["photon_to_cmd_ms"] = (tr["t_cmd_ns"] - photon) / 1e6
            row["range_est_m"] = dec["dec"]["range"]
            act = _actuation(ctx, tr)
            row.update({k: v for k, v in act.items() if k != "click_ns"})
            if "click_ns" in act:
                row["draw_to_click_ms"] = (act["click_ns"] - t_draw) / 1e6
                if photon is not None:
                    row["photon_to_click_ms"] = (act["click_ns"] - photon) / 1e6
            if tr.get("break_ns") is not None:
                row["draw_to_break_ms"] = (tr["break_ns"] - t_draw) / 1e6
                if photon is not None:
                    row["photon_to_break_ms"] = (tr["break_ns"] - photon) / 1e6
        per.append(row)
    keys = ["display_ms", "photon_to_exposure_end_ms", "exposure_end_to_user_ms", "user_to_pipeline_ms",
            "detect_ms", "frames_missed", "photon_to_detection_ms", "track_wait_frames", "track_wait_ms",
            "frame_age_at_decision_ms", "photon_to_decision_ms", "decision_to_cmd_ms", "photon_to_cmd_ms",
            "cmd_to_break_ms", "cmd_to_click_ms", "break_to_click_ms", "photon_to_break_ms", "photon_to_click_ms",
            "draw_to_cmd_ms", "draw_to_break_ms", "draw_to_click_ms", "range_est_m"]
    stats = {k: summarize(r[k] for r in per if r.get(k) is not None) for k in keys}
    n = len(per)
    n_dec = sum(1 for r in per if r["decision"])
    return {"range_m": flash["range_m"], "n": n, "decisions": n_dec, "stats": stats, "trials": per}


# ----------------------------------------------------------------------------------------- sweep
def _truth_screen(tr: dict, t_ns: float, ctx: Context, d_ms: float, hold_ms: float) -> np.ndarray:
    """Screen position of the moving disc as the camera sees it at monotonic time t (sample-and-hold
    centred: the page draws pos(t_raf) and holds it for one refresh, photons arrive d_ms later)."""
    t_ms = (t_ns - ctx.epoch_ns) / 1e6 - d_ms - 0.5 * hold_ms
    dt = t_ms - tr["t0"]
    return np.array([tr["x0"] + tr["vx"] * dt, tr["y0"]])


def _clay_world(ctx: Context, R_wg: np.ndarray, xy, range_m: float) -> np.ndarray:
    ray_c = ctx.smap.to_ray([xy])[0]
    return range_m * (R_wg @ np.asarray(ctx.mount.cam_to_gun(ray_c)).reshape(3))


def _shot_miss(ctx: Context, tr: dict, R_wg: np.ndarray, t_exit_ns: float, range_m: float, d_ms: float,
               hold_ms: float) -> dict:
    """Miss of a shot leaving the muzzle at t_exit, against the TRUE clay (virtual, at range_m)."""
    muzzle = R_wg @ (ctx.mount.muzzle_pos_g - ctx.mount.cam_pos_g)
    bore = R_wg @ ctx.mount.bore_dir_g
    bore = bore / np.linalg.norm(bore)
    tof = float(ctx.shot.time_of_flight(range_m))
    p = None
    for _ in range(4):
        t_arr = t_exit_ns + tof * 1e9
        p = _clay_world(ctx, R_wg, _truth_screen(tr, t_arr, ctx, d_ms, hold_ms), range_m)
        dist = float(np.linalg.norm(p - muzzle))
        tof = float(ctx.shot.time_of_flight(dist))
    t_arr = t_exit_ns + tof * 1e9
    p = _clay_world(ctx, R_wg, _truth_screen(tr, t_arr, ctx, d_ms, hold_ms), range_m)
    p_next = _clay_world(ctx, R_wg, _truth_screen(tr, t_arr + 1e6, ctx, d_ms, hold_ms), range_m)
    to_clay = p - muzzle
    dist = float(np.linalg.norm(to_clay))
    drop = float(ctx.shot.drop(dist, math.asin(max(-1.0, min(1.0, float(bore[2]))))))
    to_shot = bore * dist - drop * _UP
    u_clay, u_shot = to_clay / dist, to_shot / np.linalg.norm(to_shot)
    miss = math.atan2(float(np.linalg.norm(np.cross(u_clay, u_shot))), float(u_clay @ u_shot))
    motion = (p_next - muzzle) / np.linalg.norm(p_next - muzzle) - u_clay
    rate = float(np.linalg.norm(motion)) * 1000.0  # rad/s
    m_hat = motion / max(np.linalg.norm(motion), 1e-12)
    err = u_clay - u_shot
    along = float(err @ m_hat)  # + = the clay is ahead of the shot (fired late / lead too short)
    cross = float(np.linalg.norm(err - along * m_hat))
    pattern = float(ctx.shot.pattern_radius(dist)) / dist
    return {"miss_deg": math.degrees(miss), "along_deg": math.degrees(along), "cross_deg": math.degrees(cross),
            "late_ms": along / rate * 1000.0 if rate > 0 else float("nan"), "pattern_deg": math.degrees(pattern),
            "hit": bool(miss <= pattern), "tof_ms": tof * 1e3, "rate_dps": math.degrees(rate)}


def analyze_sweep(ctx: Context, sweep: dict, disp_res: dict | None) -> dict:
    H = ctx.smap.screen_h
    hold = display_period_ms(ctx)
    hold = 0.0 if not math.isfinite(hold) else hold
    R = float(sweep["range_m"])
    gun = ctx.cfg.gun
    per = []
    for tr in sweep["trials"]:
        row = {"trial": tr["trial"], "speed_dps": tr["speed_dps"], "direction": tr["direction"],
               "decision": tr["decision_frame"] is not None}
        for key in ("runway_deg", "need_deg", "runway_short"):
            if key in tr:
                row[key] = tr[key]
        d_ms = display_latency_ms(disp_res, tr["y0"], H)
        row["display_ms"] = d_ms
        if tr["decision_frame"] is None:
            per.append(row)
            continue
        if d_ms is None:
            d_ms = 0.0
            row["note"] = "no display-latency measurement: truth not delayed (run the display test)"
        dec_rec = ctx.frames[tr["decision_frame"]]
        dec = dec_rec["dec"]
        R_wg = np.asarray(tr["R_wg"], dtype=np.float64)
        t_exit_pred = float(dec["t_exit"])
        row.update({"p_hit_pred": dec["p_hit"], "miss_pred_deg": math.degrees(dec["miss"]),
                    "range_est_m": dec["range"], "range_err_m": dec["range"] - R})
        row["pred"] = _shot_miss(ctx, tr, R_wg, t_exit_pred, R, d_ms, hold)
        act = _actuation(ctx, tr)
        if "cmd_to_break_ms" in act:
            t_exit = tr["break_ns"] + (gun.lock_time_ms + gun.barrel_time_ms) * 1e6
            row["timing_source"] = "break-sense + lock + barrel"
        elif "click_ns" in act:
            t_exit = act["click_ns"] + gun.barrel_time_ms * 1e6
            row["timing_source"] = "mic click + barrel"
        else:
            t_exit = None
            row["timing_source"] = "budget (dry run)"
        if t_exit is not None:
            row["exit_late_ms"] = (t_exit - t_exit_pred) / 1e6
            row["actual"] = _shot_miss(ctx, tr, R_wg, t_exit, R, d_ms, hold)
        per.append(row)
    res = {"range_m": R, "n": len(per), "decisions": sum(1 for r in per if r["decision"]), "trials": per,
           "by_speed": {}}
    for speed in sorted({r["speed_dps"] for r in per}):
        rows = [r for r in per if r["speed_dps"] == speed]
        best = [r.get("actual") or r.get("pred") for r in rows if r["decision"]]
        best = [b for b in best if b is not None]
        res["by_speed"][str(speed)] = {
            "n": len(rows), "decisions": sum(1 for r in rows if r["decision"]),
            "hits": sum(1 for b in best if b["hit"]),
            "runway_short": any(r.get("runway_deg") is not None and r.get("need_deg") is not None
                                and r["runway_deg"] < r["need_deg"] for r in rows),
            "range_err_m": summarize(r["range_err_m"] for r in rows if r.get("range_err_m") is not None),
            "range_est_m": summarize(r["range_est_m"] for r in rows if r.get("range_est_m") is not None),
            "miss_deg": summarize(b["miss_deg"] for b in best),
            "late_ms": summarize(b["late_ms"] for b in best),
            "cross_deg": summarize(b["cross_deg"] for b in best),
            "pred_miss_deg": summarize(r["pred"]["miss_deg"] for r in rows if r.get("pred")),
            "exit_late_ms": summarize(r["exit_late_ms"] for r in rows if r.get("exit_late_ms") is not None),
            "pattern_deg": summarize(b["pattern_deg"] for b in best),
        }
    return res


# ---------------------------------------------------------------------------------------- detect
def analyze_detect(ctx: Context, det: dict) -> dict:
    psf = float(ctx.cfg.detector.psf_px)
    clay_d = float(ctx.cfg.mount.clay_diameter_m)
    base = _frames(ctx, det["baseline"])
    fp_base = float(np.mean([len(f["dets"]) for f in base])) if base else float("nan")
    groups: dict[float, dict] = {}
    for tr in det["trials"]:
        uv = np.asarray(tr["uv"], dtype=np.float64)
        gate = _gate_px(ctx, tr["xy"], tr["r"])
        ifov = float(ctx.cam.ifov(np.array([uv]))[0])
        g = groups.setdefault(tr["range_m"], {"frames": 0, "hits": 0, "du": [], "dv": [], "d": [], "rng": [],
                                              "false": 0, "geo": [], "psf_sq": []})
        geo = clay_d / float(tr["range_m"]) / ifov
        g["geo"].append(geo)
        for f in _frames(ctx, tr["frames"]):
            g["frames"] += 1
            d = _near_det(f, uv, gate)
            g["false"] += len(f["dets"]) - (1 if d is not None else 0)
            if d is None:
                continue
            g["hits"] += 1
            g["du"].append(d[0] - uv[0])
            g["dv"].append(d[1] - uv[1])
            g["d"].append(d[2])
            g["psf_sq"].append(float(d[2]) ** 2 - geo ** 2)
            g["rng"].append(range_from_size(d[2], ifov, clay_d, psf))
    pooled = [v for g in groups.values() for v in g["psf_sq"]]
    out = {"false_per_frame_blank": fp_base, "psf_px": psf,
           "psf_fit_px": math.sqrt(max(float(np.mean(pooled)), 0.0)) if pooled else None, "by_range": {}}
    for R, g in sorted(groups.items()):
        err = np.hypot(g["du"], g["dv"]) if g["du"] else np.array([])
        out["by_range"][str(R)] = {
            "frames": g["frames"], "detect_rate": g["hits"] / g["frames"] if g["frames"] else float("nan"),
            "centroid_bias_px": [float(np.mean(g["du"])) if g["du"] else float("nan"),
                                 float(np.mean(g["dv"])) if g["dv"] else float("nan")],
            "centroid_rms_px": float(np.sqrt(np.mean(err**2))) if err.size else float("nan"),
            "diameter_px": summarize(g["d"]), "size_range_m": summarize(g["rng"]),
            "d_geo_px": float(np.mean(g["geo"])) if g["geo"] else None,
            "psf_fit_px": math.sqrt(max(float(np.mean(g["psf_sq"])), 0.0)) if g["psf_sq"] else None,
            "false_per_frame": g["false"] / g["frames"] if g["frames"] else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------------------- report
def analyze(raw: dict) -> dict:
    ctx = context_from_raw(raw)
    rep = {"meta": {k: raw["meta"].get(k) for k in ("created", "live", "actuator", "source", "tests", "sim_truth",
                                                    "mic", "mic_distance_m", "camera", "range_m")},
           "page": page_summary(ctx, (raw.get("page") or {}).get("status") or {}),
           "budget": {"exposure_to_user_ms": ctx.budget.exposure_to_user_ms,
                      "camera_timestamp_offset_ms": ctx.budget.camera_timestamp_offset_ns / 1e6,
                      "cmd_to_break_ms": ctx.budget.cmd_to_break_ms.get(str(ctx.cfg.actuator.type)),
                      "lock_time_ms": ctx.budget.lock_time_ms, "barrel_time_ms": ctx.budget.barrel_time_ms,
                      "fire_delay_ms": ctx.budget.fire_delay_s(str(ctx.cfg.actuator.type)) * 1e3,
                      "sources": ctx.budget.sources},
           "frames": _frame_summary(ctx)}
    if ctx.smap is not None:
        rep["geometry"] = ctx.smap.to_dict()
    disp = None
    if raw.get("display"):
        disp = analyze_display(ctx, raw["display"])
        rep["display"] = disp
    elif raw["meta"].get("display_model"):
        disp = raw["meta"]["display_model"]
        rep["display"] = {"reused": True, **disp}
    if raw.get("flash"):
        rep["flash"] = analyze_flash(ctx, raw["flash"], disp)
    if raw.get("sweep"):
        rep["sweep"] = analyze_sweep(ctx, raw["sweep"], disp)
    if raw.get("detect"):
        rep["detect"] = analyze_detect(ctx, raw["detect"])
    rep["checks"] = checks(rep)
    if raw["meta"].get("sim_truth"):
        rep["sim_check"] = sim_check(rep, raw["meta"]["sim_truth"], ctx)
    return rep


def _frame_summary(ctx: Context) -> dict:
    fr = ctx.frames
    if len(fr) < 2:
        return {"n": len(fr)}
    te = np.array([f["te"] for f in fr], dtype=np.float64)
    seq = np.array([f["seq"] for f in fr])
    gaps = np.diff(seq)
    return {"n": len(fr), "fps": 1e9 / float(np.median(np.diff(te))) if len(te) > 1 else float("nan"),
            "skipped_frames": int(np.sum(np.clip(gaps - 1, 0, None))),
            "exposure_end_to_user_ms": summarize((f["tc"] - f["te"] - _exp_ns(f) / 2) / 1e6 for f in fr),
            "stage_total_ms": summarize(f["stage"].get("total", float("nan")) for f in fr),
            "stage_detect_ms": summarize(f["stage"].get("detect", float("nan")) for f in fr),
            "stage_track_ms": summarize(f["stage"].get("track", float("nan")) for f in fr),
            "stage_solve_ms": summarize(f["stage"].get("solve", float("nan")) for f in fr)}


def checks(rep: dict) -> list[dict]:
    """Pass/warn/fail lines against docs/HITL.md criteria."""
    out = []

    def add(level: str, what: str, detail: str) -> None:
        out.append({"level": level, "what": what, "detail": detail})

    st = (rep.get("page") or {}).get("status") or {}
    if st:
        rtt = st.get("rtt_min_ms")
        if isinstance(rtt, (int, float)):
            add("PASS" if rtt <= 2.0 else "WARN", "clock sync", f"page<->rig min RTT {rtt:.2f} ms (bound on sync error: half)")
        if not st.get("fullscreen", True):
            add("WARN", "page", "not fullscreen")
    late = (rep.get("page") or {}).get("late_pct")
    if late is not None:
        add("PASS" if late < 1.0 else "WARN", "display cadence", f"{late:.1f} % of page frames late (>1.5x period)")
    g = rep.get("geometry")
    if g:
        add("PASS" if g["rms_px"] < 0.5 else ("WARN" if g["rms_px"] < 1.5 else "FAIL"), "screen map",
            f"rms {g['rms_px']:.2f} px over {g['n_points']} markers (large = intrinsics or focus)")
    fs = rep.get("frames") or {}
    if fs.get("skipped_frames"):
        add("WARN", "camera", f"{fs['skipped_frames']} frames skipped by the pipeline (processing slower than the camera)")
    d = rep.get("display")
    if d and d.get("rise", {}).get("n"):
        add("INFO", "display latency", f"rAF->photon {d['rise']['d0_ms']:.1f} ms at the top, scan-out "
            f"{d['rise']['scanout_ms']:.1f} ms, resid {d['rise']['resid_std_ms']:.2f} ms")
    if d:
        for w in d.get("warnings", []):
            add("WARN", "display latency", w)
    fl = rep.get("flash")
    if fl:
        s = fl["stats"]
        if fl["decisions"] < fl["n"]:
            add("FAIL", "flash", f"{fl['n'] - fl['decisions']} of {fl['n']} static targets never produced a decision")
        e2u = s["exposure_end_to_user_ms"]["mean"]
        bud = rep["budget"]["exposure_to_user_ms"]
        if math.isfinite(e2u):
            add("INFO", "camera delivery", f"exposure end -> user {e2u:.1f} ms (budget exposure_to_user {bud:.1f} ms "
                "includes the phase wait)")
        dc = s["decision_to_cmd_ms"]
        if dc["n"] and dc["mean"] > 1.0:
            add("WARN", "decision->command", f"{dc['mean']:.1f} ms between the solver's 'now' and the actuator "
                "call: the solver does not budget it")
        for key, budget_key in (("cmd_to_break_ms", "cmd_to_break_ms"),):
            st2 = s[key]
            if st2["n"]:
                b = rep["budget"][budget_key]
                add("PASS" if abs(st2["mean"] - b) <= max(5.0, 0.1 * b) else "WARN", "actuator",
                    f"cmd->break {st2['mean']:.1f} +- {st2['std']:.1f} ms vs budget {b:.1f} ms")
        cc = s["cmd_to_click_ms"]
        if cc["n"]:
            b = rep["budget"]["cmd_to_break_ms"] + rep["budget"]["lock_time_ms"]
            add("PASS" if abs(cc["mean"] - b) <= max(5.0, 0.1 * b) else "WARN", "actuator (mic)",
                f"cmd->click {cc['mean']:.1f} +- {cc['std']:.1f} ms vs budget cmd_to_break + lock {b:.1f} ms")
    sw = rep.get("sweep")
    if sw:
        for speed, g2 in sw["by_speed"].items():
            n = g2["decisions"]
            rows = [r for r in sw["trials"] if str(r["speed_dps"]) == speed]
            short = [r for r in rows if r.get("runway_short",
                     r.get("runway_deg") is not None and r.get("need_deg") is not None
                     and r["runway_deg"] < r["need_deg"])]
            if not n:
                lvl = "FAIL"
                detail = f"no fire decision in {g2['n']} trials"
            else:
                lvl = "PASS" if g2["hits"] == n else ("WARN" if g2["hits"] else "FAIL")
                if lvl == "FAIL" and len(short) == len(rows) and rows:
                    lvl = "WARN"
                detail = (f"{g2['hits']}/{n} virtual hits, miss "
                          f"{g2['miss_deg']['mean']:.2f} deg (pattern {g2['pattern_deg']['mean']:.2f} deg), late "
                          f"{g2['late_ms']['mean']:+.1f} ms")
            if short and lvl in ("FAIL", "WARN"):
                measured = [r for r in short if r.get("need_deg") is not None and r.get("runway_deg") is not None]
                if measured:
                    worst = max(measured, key=lambda r: r["need_deg"] - r["runway_deg"])
                    detail += f"; runway too short: need {worst['need_deg']:.2f} deg, have {worst['runway_deg']:.2f} deg"
            add(lvl, f"sweep {float(speed):g} deg/s", detail)
            err = g2["range_err_m"]
            if err["n"] and abs(err["mean"]) > 0.2 * sw["range_m"]:
                add("WARN", f"sweep {float(speed):g} deg/s range", f"mean range error {err['mean']:+.1f} m; "
                    "size-based ranging drives ToF/lead (see detect PSF fit)")
    dt = rep.get("detect")
    if dt:
        for R, g3 in dt["by_range"].items():
            rms = _number(g3["centroid_rms_px"], ".2f")
            size = _number(g3["size_range_m"]["mean"], ".1f")
            add("PASS" if g3["detect_rate"] >= 0.9 else "WARN", f"detect {float(R):g} m",
                f"rate {_number(g3['detect_rate'], '.2f')}, centroid rms {rms} px, size-range {size} m")
    return out


def sim_check(rep: dict, truth: dict, ctx: Context) -> list[dict]:
    """--sim self-test: the analysis must give back the constants the simulation injected."""
    out = []

    def cmp(what: str, got: float | None, want: float, tol: float) -> None:
        ok = got is not None and math.isfinite(got) and abs(got - want) <= tol
        out.append({"level": "PASS" if ok else "FAIL", "what": what,
                    "detail": f"measured {got if got is None else round(got, 2)} vs injected {want:g} (tol {tol:g})"})

    d = (rep.get("display") or {}).get("rise") or {}
    if d.get("n"):
        cmp("display latency (top row) ms", d.get("d0_ms"), truth["display_latency_ms"], 1.0)
        cmp("display scan-out ms", d.get("scanout_ms"), truth["scanout_ms"], 1.5)
    fl = rep.get("flash")
    if fl:
        s = fl["stats"]
        cmp("exposure end -> user ms", s["exposure_end_to_user_ms"]["mean"], truth["camera_delivery_ms"],
            1.0 + 2.0 * truth["camera_jitter_ms"])
        if s["break_to_click_ms"]["n"]:
            cmp("break -> click ms (lock time)", s["break_to_click_ms"]["mean"], truth["lock_time_ms"], 0.5)
        if s["cmd_to_break_ms"]["n"]:
            # The pipeline passes the actual command time to actuator.fire(), and the fixed-model SimActuator
            # breaks exactly delay_ms after the time it is given.
            want = float(ctx.cfg.actuator.sim.delay_ms)
            cmp("cmd -> break ms (sim delay)", s["cmd_to_break_ms"]["mean"], want, 1.0)
    sw = rep.get("sweep")
    if sw:
        hits = sum(g["hits"] for g in sw["by_speed"].values())
        n = sum(g["decisions"] for g in sw["by_speed"].values())
        out.append({"level": "PASS" if n and hits >= 0.8 * n else "FAIL", "what": "sweep virtual hits",
                    "detail": f"{hits}/{n} (a matched latency model must hit nearly all)"})
    return out


# ---------------------------------------------------------------------------------------- printing
def _number(value, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None and math.isfinite(value) else "-"


def _fmt(st: dict) -> str:
    return " ".join(f"{_number(st.get(k)):>8s}" for k in ("mean", "std", "p50", "p95")) + f" {st.get('n', 0):4d}"


FLASH_ROWS = [
    ("display_ms", "draw (rAF) -> photon [display test model]"),
    ("photon_to_exposure_end_ms", "photon -> end of first exposure (0..1 frame)"),
    ("exposure_end_to_user_ms", "exposure end -> frame in user space"),
    ("user_to_pipeline_ms", "user space -> pipeline picked it up"),
    ("detect_ms", "detector processing"),
    ("frames_missed", "frames seen but not detected [frames]"),
    ("photon_to_detection_ms", "photon -> first detection done"),
    ("track_wait_frames", "detection -> decision frame [frames]"),
    ("track_wait_ms", "detection -> decision frame (exposure to exposure)"),
    ("frame_age_at_decision_ms", "decision frame age at 'now' (mid-exposure -> now)"),
    ("photon_to_decision_ms", "photon -> decision ('now')"),
    ("decision_to_cmd_ms", "decision 'now' -> actuator command"),
    ("photon_to_cmd_ms", "photon -> actuator command"),
    ("cmd_to_break_ms", "command -> trigger break [break-sense]"),
    ("cmd_to_click_ms", "command -> dry-fire click [mic]"),
    ("break_to_click_ms", "break -> click (lock time + mic error)"),
    ("photon_to_break_ms", "photon -> break"),
    ("photon_to_click_ms", "photon -> click  (END TO END)"),
    ("draw_to_click_ms", "draw -> click (camera-independent end to end)"),
    ("range_est_m", "tracker range at decision [m]"),
]


def format_report(rep: dict) -> str:
    import textwrap

    L = []
    m = rep["meta"]
    L.append(f"TBOT HITL report  {m.get('created')}  source {m.get('source')}  actuator {m.get('actuator')}  "
             f"{'LIVE' if m.get('live') else 'dry run'}")
    pg = rep.get("page") or {}
    st = pg.get("status") or {}
    if st:
        L.append(f"page: {st.get('w')}x{st.get('h')} px @ {_number(pg.get('refresh_hz'), '.1f')} Hz, "
                 f"sync RTT {_number(st.get('rtt_min_ms'))} ms, drift {_number(st.get('drift_ppm'), '.1f')} ppm, "
                 f"late {_number(pg.get('late_pct'))} %")
    fs = rep.get("frames") or {}
    if fs.get("n", 0) > 1:
        L.append(f"camera: {_number(fs['fps'], '.1f')} fps, {fs['skipped_frames']} skipped frames, "
                 f"exposure end->user {_number(fs['exposure_end_to_user_ms']['mean'])} ms, "
                 f"pipeline p95 {_number(fs['stage_total_ms']['p95'])} ms")
    g = rep.get("geometry")
    if g:
        L.append(f"screen map: {g['n_points']} markers, rms {g['rms_px']:.2f} px, max {g['max_px']:.2f} px")
    d = rep.get("display")
    if d and d.get("rise", {}).get("n"):
        r, f = d["rise"], d.get("fall") or {}
        L.append(f"display: rise {r['d0_ms']:.2f} ms + {r['scanout_ms']:.2f} ms*row/H "
                 f"({r['n']} samples, {r['partial']} partial, resid {r['resid_std_ms']:.2f} ms)"
                 + (f"; fall {f['d0_ms']:.2f} ms" if f.get("n") else ""))
    fl = rep.get("flash")
    if fl:
        L.extend(["", f"FLASH (static clay at the bore, {fl['range_m']:g} m): "
                       f"{fl['decisions']}/{fl['n']} decisions"])
        b = rep["budget"]
        L.append(f"  END-TO-END (ms; mean +/- std, p95)   solver fire delay: {b['fire_delay_ms']:.1f} ms")
        for key, label in (("photon_to_decision_ms", "photon -> decision"),
                           ("photon_to_cmd_ms", "photon -> command"),
                           ("cmd_to_break_ms", "command -> break"),
                           ("cmd_to_click_ms", "command -> click"),
                           ("photon_to_click_ms", "photon -> click")):
            s = fl["stats"].get(key) or {}
            if s.get("n"):
                value = f"{_number(s.get('mean'))} +/- {_number(s.get('std'))} ms, p95 {_number(s.get('p95'))} ms"
            else:
                value = "-"
            L.append(f"    {label:20s} {value}")
        L.append(f"  {'stage':52s} {'unit':>6s} {'mean':>8s} {'std':>8s} {'p50':>8s} {'p95':>8s} {'n':>4s}")
        for key, label in FLASH_ROWS:
            s = fl["stats"].get(key)
            if s and s.get("n"):
                unit = "frames" if key.endswith("_frames") or key == "frames_missed" else ("m" if key.endswith("_m") else "ms")
                L.append(f"  {label:52s} {unit:>6s} {_fmt(s)}")
        L.append(f"  budget: exposure->user {b['exposure_to_user_ms']:.1f} ms, "
                 f"command->break {b['cmd_to_break_ms']:.1f} ms, lock {b['lock_time_ms']:.1f} ms, "
                 f"barrel {b['barrel_time_ms']:.1f} ms")
        sources = ", ".join(f"{k}={v}" for k, v in b["sources"].items() if "sim" not in k and "none" not in k)
        if sources:
            L.append(f"  budget sources: {sources}")
    sw = rep.get("sweep")
    if sw:
        L.extend(["", f"SWEEP (crossing clay through the bore, {sw['range_m']:g} m): "
                       f"{sw['decisions']}/{sw['n']} decisions"])
        L.append(f"  {'speed':>6s} {'hits':>7s} {'miss':>7s} {'pattern':>7s} {'late':>7s} {'cross':>7s} "
                 f"{'exit late':>9s} {'pred miss':>9s} {'range err':>9s}")
        L.append(f"  {'deg/s':>6s} {'count':>7s} {'deg':>7s} {'deg':>7s} {'ms':>7s} {'deg':>7s} "
                 f"{'ms':>9s} {'deg':>9s} {'m':>9s}")
        for speed, group in sw["by_speed"].items():
            vals = [_number(group[key]["mean"], spec) for key, spec in
                    (("miss_deg", ".2f"), ("pattern_deg", ".2f"), ("late_ms", "+.1f"),
                     ("cross_deg", ".2f"), ("exit_late_ms", "+.1f"), ("pred_miss_deg", ".2f"),
                     ("range_err_m", "+.1f"))]
            hits = f"{group['hits']}/{group['decisions']}"
            L.append(f"  {float(speed):6.1f} {hits:>7s} {vals[0]:>7s} {vals[1]:>7s} {vals[2]:>7s} "
                     f"{vals[3]:>7s} {vals[4]:>9s} {vals[5]:>9s} {vals[6]:>9s}")
        L.append("  Positive late means the shot passed behind the clay; exit late = actual exit - solver exit.")
    dt = rep.get("detect")
    if dt:
        L.extend(["", f"DETECT (blank-screen false detections {_number(dt['false_per_frame_blank'], '.3f')}/frame; "
                       f"configured PSF {dt['psf_px']:g} px; fitted PSF {_number(dt.get('psf_fit_px'))} px)"])
        L.append("  --set detector.psf_px=<fit> makes bench ranging consistent (bench defocus differs from the field).")
        L.append(f"  {'range':>6s} {'rate':>8s} {'rms':>7s} {'bias u,v':>15s} {'diam':>7s} {'geo':>7s} "
                 f"{'PSF fit':>7s} {'size range':>12s} {'false':>7s}")
        L.append(f"  {'m':>6s} {'fraction':>8s} {'px':>7s} {'px':>15s} {'px':>7s} {'px':>7s} "
                 f"{'px':>7s} {'m':>12s} {'/frame':>7s}")
        for R, group in dt["by_range"].items():
            bias = ",".join(_number(v, "+.2f") for v in group["centroid_bias_px"])
            size = group["size_range_m"]
            size_str = (f"{_number(size['mean'], '.1f')}+/-{_number(size['std'], '.1f')}"
                        if size["n"] else "-")
            L.append(f"  {float(R):6.1f} {_number(group['detect_rate']):>8s} "
                     f"{_number(group['centroid_rms_px']):>7s} {bias:>15s} "
                     f"{_number(group['diameter_px']['mean']):>7s} {_number(group.get('d_geo_px')):>7s} "
                     f"{_number(group.get('psf_fit_px')):>7s} {size_str:>12s} "
                     f"{_number(group['false_per_frame'], '.3f'):>7s}")
    L.extend(["", "CHECKS"])
    for c in rep.get("checks", []):
        L.append(f"{c['level']:5s} {c['what']}: {c['detail']}")
    if rep.get("sim_check"):
        L.extend(["", "SIM SELF-CHECK (analysis vs injected constants)"])
        for c in rep["sim_check"]:
            L.append(f"{c['level']:5s} {c['what']}: {c['detail']}")
    return "\n".join(part for line in L for part in
                     (textwrap.wrap(line, width=110, subsequent_indent="  ", replace_whitespace=False,
                                    break_long_words=False, break_on_hyphens=False) or [""]))
