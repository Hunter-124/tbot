"""Scores of a front end on a press: the pipeline's per-frame telemetry against the truth (`tbot vbench`).

Inputs of one run (one press, at most one injected clay):
  * tel: the telemetry dicts of the pipeline (tbot.app.run.Pipeline.step), each with "row" = the row of
    frames.csv it was made from (added by the harness);
  * clay: the injected clay's truth per row, {row: TruthRow-like} (u, v, major_px, exists, in_image, masked,
    bg_std), or None (a clean run: nothing was injected);
  * movers: REAL movers already in the footage, in the truth schema shared with bench_data/truth/<press>.json
    ({"id", "label", "frames": [[row, u, v, w_px, h_px], ...]}). Detections and tracks on them are counted as
    "real_mover", never as false;
  * project(az_deg, el_deg, R_wg) -> (u, v) | None: a track's pixel, its az/el through that frame's R_wg (as
    tbot.tools.score / tbot.app.overlay do);
  * outcome(rec) -> (miss_deg, hit) | None: the shot of a fire decision judged against the true clay (vclay's
    criterion: the bore at exit time from the whole IMU record, the clay where it really is at arrival).

"On truth" = within max(8 px, 0.75 x the truth's major size) of the truth pixel in the same frame.

score_run() returns raw counts; aggregate() turns any number of them into the rates reported.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

SIZE_BINS = ((0.0, 4.0, "<4"), (4.0, 8.0, "4-8"), (8.0, 16.0, "8-16"), (16.0, math.inf, ">16"))
SKY_STD_DN = 6.0
MIN_RADIUS_PX = 8.0
RADIUS_FRAC = 0.75
MAX_DT_S = 0.2            # a gap in the frames longer than this does not count as armed time


def radius(size_px: float) -> float:
    return max(MIN_RADIUS_PX, RADIUS_FRAC * size_px) if math.isfinite(size_px) else MIN_RADIUS_PX


def size_bin(major_px: float) -> str:
    for lo, hi, name in SIZE_BINS:
        if lo <= major_px < hi:
            return name
    return SIZE_BINS[-1][2]


def _get(t, key, default=float("nan")):
    if t is None:
        return default
    if isinstance(t, dict):
        return t.get(key, default)
    return getattr(t, key, default)


def mover_index(movers: Iterable[dict] | None) -> dict[int, list[tuple[float, float, float, str]]]:
    """row -> [(u, v, radius, id)] of the real movers."""
    out: dict[int, list] = {}
    for obj in movers or ():
        for fr in obj.get("frames") or ():
            row, u, v = int(fr[0]), float(fr[1]), float(fr[2])
            size = max(float(fr[3]), float(fr[4])) if len(fr) >= 5 else float("nan")
            out.setdefault(row, []).append((u, v, radius(size), str(obj.get("id"))))
    return out


def _near(u: float, v: float, pts) -> bool:
    return any(math.hypot(u - a, v - b) <= r for a, b, r, _ in pts)


def score_run(tel: list[dict], project: Callable, clay: dict | None = None, movers=None,
              outcome: Callable | None = None, polarity: str | None = None) -> dict:
    """Raw counts of one run (see the module docstring)."""
    mv = mover_index(movers)
    c = {
        "runs": 1, "injected": clay is not None, "frames": 0, "armed_frames": 0, "armed_s": 0.0,
        "vis_frames": 0, "vis_det": 0, "recall_bins": {},
        "clutter": [], "mover_dets": 0, "truth_dets": 0,
        "false_conf_track_frames": 0, "false_conf_ids": 0, "mover_conf_track_frames": 0,
        "armed_vis_frames": 0, "target_correct": 0, "target_false": 0, "target_mover": 0, "target_none": 0,
        "ttt_eligible": False, "ttt_ms": None,
        "fires": [],
    }
    false_ids: set[int] = set()
    t_first = None
    ttt_done = False
    tel = sorted(tel, key=lambda r: int(r["t_exposure_ns"]))
    for k, rec in enumerate(tel):
        row = int(rec.get("row", k))
        t = int(rec["t_exposure_ns"])
        armed = bool(rec.get("armed"))
        c["frames"] += 1
        if armed:
            c["armed_frames"] += 1
            if k + 1 < len(tel):
                c["armed_s"] += min(MAX_DT_S, max(0.0, (int(tel[k + 1]["t_exposure_ns"]) - t) * 1e-9))
        tr = clay.get(row) if clay is not None else None
        exists = bool(_get(tr, "exists", False))
        tu, tv = float(_get(tr, "u")), float(_get(tr, "v"))
        has_px = exists and math.isfinite(tu) and math.isfinite(tv)
        visible = has_px and bool(_get(tr, "in_image", False)) and not bool(_get(tr, "masked", False))
        r_truth = radius(float(_get(tr, "major_px")))
        movers_here = mv.get(row, [])

        def on_truth(u: float, v: float) -> bool:
            return has_px and math.hypot(u - tu, v - tv) <= r_truth

        # ---- detections
        hit = False
        clutter = 0
        for d in rec.get("dets") or ():
            u, v = float(d[0]), float(d[1])
            if on_truth(u, v):
                hit = True
                c["truth_dets"] += 1
            elif _near(u, v, movers_here):
                c["mover_dets"] += 1
            else:
                clutter += 1
        c["clutter"].append(clutter)
        if visible:
            c["vis_frames"] += 1
            c["vis_det"] += int(hit)
            bg = float(_get(tr, "bg_std"))
            keys = ["size:" + size_bin(float(_get(tr, "major_px"))),
                    "bg:" + ("sky" if math.isfinite(bg) and bg < SKY_STD_DN else "textured")]
            if polarity:
                keys.append("polarity:" + polarity)
            for key in keys:
                b = c["recall_bins"].setdefault(key, [0, 0])
                b[0] += 1
                b[1] += int(hit)

        # ---- tracks
        R = rec.get("R_wg")
        label: dict[int, str] = {}
        confirmed: dict[int, bool] = {}
        for trk in rec.get("tracks") or ():
            tid = int(trk["id"])
            confirmed[tid] = bool(trk.get("confirmed"))
            uv = project(float(trk["az_deg"]), float(trk["el_deg"]), R) if R is not None else None
            if uv is None or not all(math.isfinite(x) for x in uv):
                label[tid] = "false"
            elif on_truth(uv[0], uv[1]):
                label[tid] = "truth"
            elif _near(uv[0], uv[1], movers_here):
                label[tid] = "mover"
            else:
                label[tid] = "false"
        if armed:
            for tid, lab in label.items():
                if not confirmed[tid]:
                    continue
                if lab == "false":
                    c["false_conf_track_frames"] += 1
                    false_ids.add(tid)
                elif lab == "mover":
                    c["mover_conf_track_frames"] += 1
        # time to a confirmed track, from the first armed frame with the clay in view
        if armed and visible and t_first is None:
            t_first = t
            c["ttt_eligible"] = True
        if t_first is not None and not ttt_done and any(confirmed[i] and label[i] == "truth" for i in label):
            c["ttt_ms"] = (t - t_first) / 1e6
            ttt_done = True

        # ---- the solver's target
        dec = rec.get("decision") or {}
        tid = dec.get("track_id")
        lab = label.get(int(tid), "false") if tid is not None else None
        if armed and visible:
            c["armed_vis_frames"] += 1
            if lab is None:
                c["target_none"] += 1
            elif lab == "truth":
                c["target_correct"] += 1
            elif lab == "mover":
                c["target_mover"] += 1
            else:
                c["target_false"] += 1

        # ---- fire decisions
        if rec.get("fired"):
            f = {"row": row, "t_ns": int(rec.get("t_ns") or t), "track_id": tid, "class": lab or "false",
                 "p_hit": dec.get("p_hit"), "range_m": dec.get("range_m")}
            if f["class"] == "truth" and outcome is not None:
                res = outcome(rec)
                if res is not None:
                    f["miss_deg"], f["hit"] = float(res[0]), bool(res[1])
            c["fires"].append(f)
    c["false_conf_ids"] = len(false_ids)
    return c


# ------------------------------------------------------------------------------------------------ aggregate
def _pct(a: float, b: float) -> float | None:
    return None if not b else 100.0 * a / b


def _rate(a: float, b: float) -> float | None:
    return None if not b else a / b


def aggregate(runs: list[dict]) -> dict:
    """Rates over any number of score_run() counts."""
    out: dict = {"runs": len(runs)}
    if not runs:
        return out
    inj = [r for r in runs if r.get("injected")]
    frames = sum(r["frames"] for r in runs)
    armed_s = sum(r["armed_s"] for r in runs)
    clutter = np.array([x for r in runs for x in r["clutter"]], dtype=np.float64)
    fires = [f for r in runs for f in r["fires"]]
    out.update(frames=frames, armed_frames=sum(r["armed_frames"] for r in runs), armed_s=round(armed_s, 3))
    out["clutter_per_frame_mean"] = float(clutter.mean()) if clutter.size else None
    out["clutter_per_frame_p95"] = float(np.percentile(clutter, 95)) if clutter.size else None
    out["real_mover_dets"] = sum(r["mover_dets"] for r in runs)
    fct = sum(r["false_conf_track_frames"] for r in runs)
    fid = sum(r["false_conf_ids"] for r in runs)
    out["false_conf_track_frames"] = fct
    out["false_conf_track_ids"] = fid
    out["false_conf_track_frames_per_armed_s"] = _rate(fct, armed_s)
    out["false_conf_track_ids_per_armed_s"] = _rate(fid, armed_s)
    out["real_mover_conf_track_frames"] = sum(r["mover_conf_track_frames"] for r in runs)
    false_fires = [f for f in fires if f["class"] == "false"]
    out["fires"] = len(fires)
    out["false_fires"] = len(false_fires)
    out["real_mover_fires"] = sum(1 for f in fires if f["class"] == "mover")
    out["false_fires_per_armed_min"] = _rate(len(false_fires), armed_s / 60.0)
    if inj:
        vis = sum(r["vis_frames"] for r in inj)
        out["clays"] = len(inj)
        out["vis_frames"] = vis
        out["det_recall"] = _rate(sum(r["vis_det"] for r in inj), vis)
        bins: dict[str, list[int]] = {}
        for r in inj:
            for k, (n, h) in r["recall_bins"].items():
                b = bins.setdefault(k, [0, 0])
                b[0] += n
                b[1] += h
        out["det_recall_by"] = {k: {"frames": n, "recall": _rate(h, n)} for k, (n, h) in sorted(bins.items())}
        elig = [r for r in inj if r["ttt_eligible"]]
        ttt = np.array([r["ttt_ms"] for r in elig if r["ttt_ms"] is not None], dtype=np.float64)
        out["ttt_clays"] = len(elig)
        out["ttt_ms_median"] = float(np.median(ttt)) if ttt.size else None
        out["ttt_ms_p90"] = float(np.percentile(ttt, 90)) if ttt.size else None
        out["never_tracked_frac"] = _rate(len(elig) - ttt.size, len(elig))
        avf = sum(r["armed_vis_frames"] for r in inj)
        out["armed_vis_frames"] = avf
        out["target_correct_frac"] = _rate(sum(r["target_correct"] for r in inj), avf)
        out["target_false_frac"] = _rate(sum(r["target_false"] for r in inj), avf)
        out["target_none_frac"] = _rate(sum(r["target_none"] for r in inj), avf)
        truth_fires = [f for r in inj for f in r["fires"] if f["class"] == "truth"]
        out["truth_fires"] = len(truth_fires)
        out["truth_fire_hits"] = sum(1 for f in truth_fires if f.get("hit"))
        out["clays_hit_pct"] = _pct(sum(1 for r in inj if any(f.get("hit") for f in r["fires"]
                                                               if f["class"] == "truth")), len(inj))
        out["clays_fired_at_pct"] = _pct(sum(1 for r in inj if any(f["class"] == "truth" for f in r["fires"])),
                                         len(inj))
        out["hits_per_truth_fire_pct"] = _pct(out["truth_fire_hits"], len(truth_fires))
    return out


def trim(counts: dict) -> dict:
    """A run's counts for the JSON: the per-frame clutter list summarised."""
    c = dict(counts)
    cl = np.asarray(c.pop("clutter"), dtype=np.float64)
    c["clutter_sum"] = int(cl.sum()) if cl.size else 0
    c["clutter_p95"] = float(np.percentile(cl, 95)) if cl.size else None
    return c


# ------------------------------------------------------------------------------------------------ text
ROWS = [  # (key, label, format)
    ("runs", "runs", "{:.0f}"),
    ("clays", "injected clays", "{:.0f}"),
    ("armed_s", "armed seconds", "{:.1f}"),
    ("det_recall", "detection recall (clay in view)", "{:.3f}"),
    ("det_recall_by.size:<4.recall", "  size < 4 px", "{:.3f}"),
    ("det_recall_by.size:4-8.recall", "  size 4-8 px", "{:.3f}"),
    ("det_recall_by.size:8-16.recall", "  size 8-16 px", "{:.3f}"),
    ("det_recall_by.size:>16.recall", "  size > 16 px", "{:.3f}"),
    ("det_recall_by.bg:sky.recall", "  on sky", "{:.3f}"),
    ("det_recall_by.bg:textured.recall", "  on texture (trees, ground)", "{:.3f}"),
    ("det_recall_by.polarity:dark.recall", "  dark clay", "{:.3f}"),
    ("det_recall_by.polarity:bright.recall", "  bright clay", "{:.3f}"),
    ("det_recall_by.polarity:twotone.recall", "  two-tone clay", "{:.3f}"),
    ("clutter_per_frame_mean", "clutter detections / frame, mean", "{:.2f}"),
    ("clutter_per_frame_p95", "clutter detections / frame, p95", "{:.1f}"),
    ("false_conf_track_frames_per_armed_s", "false confirmed track-frames / armed s", "{:.2f}"),
    ("false_conf_track_ids_per_armed_s", "false confirmed track ids / armed s", "{:.2f}"),
    ("ttt_ms_median", "time to confirmed track, median ms", "{:.0f}"),
    ("ttt_ms_p90", "time to confirmed track, p90 ms", "{:.0f}"),
    ("never_tracked_frac", "clays never tracked (confirmed)", "{:.3f}"),
    ("target_correct_frac", "solver target = the clay (armed, in view)", "{:.3f}"),
    ("target_false_frac", "solver target = a false track", "{:.3f}"),
    ("clays_fired_at_pct", "clays fired at %", "{:.1f}"),
    ("clays_hit_pct", "clays hit %", "{:.1f}"),
    ("hits_per_truth_fire_pct", "hits per shot at the clay %", "{:.1f}"),
    ("false_fires", "false fires (shot at scenery)", "{:.0f}"),
    ("false_fires_per_armed_min", "false fires / armed minute", "{:.2f}"),
    ("real_mover_fires", "fires at real movers", "{:.0f}"),
]


def lookup(d: dict, path: str):
    if path in d:
        return d[path]
    node = d
    for key in path.split("."):  # (no key has a dot in it; bin names have ':' and '<')
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _fmt(v, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return fmt.format(v)


def table(metrics: dict, title: str = "") -> str:
    lines = [title] if title else []
    for key, label, fmt in ROWS:
        if key.split(".")[0] not in metrics:
            continue
        v = lookup(metrics, key)
        lines.append(f"  {label:<44s} {_fmt(v, fmt):>10s}")
    return "\n".join(lines)


def compare_table(a: dict, b: dict, name_a: str = "A", name_b: str = "B") -> str:
    lines = [f"  {'':<44s} {name_a[:12]:>12s} {name_b[:12]:>12s} {'delta':>10s}"]
    for key, label, fmt in ROWS:
        va, vb = lookup(a, key), lookup(b, key)
        if va is None and vb is None:
            continue
        d = vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        ds = "-" if d is None else ("+" if d >= 0 else "") + fmt.format(d)
        lines.append(f"  {label:<44s} {_fmt(va, fmt):>12s} {_fmt(vb, fmt):>12s} {ds:>10s}")
    return "\n".join(lines)
