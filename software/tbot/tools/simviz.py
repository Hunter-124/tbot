"""`tbot simviz` - graphics of closed-loop simulator runs (`tbot sim`): a video, a summary still, a contact sheet.

Each run is the exact `tbot sim` run (tbot.tools.sim_eval.evaluate_run, same pipeline, same scoring); a
Recorder only watches it through evaluate_run's `observer` hook: the camera frame and telemetry of every
processed frame, the ground truth the run already computes, and each shot as the judge scored it. Nothing it
does feeds back into the run, so the numbers in the graphics are the numbers `tbot sim` prints.

Per scenario run (out/<scenario>_seed<N>.*):
- .mp4 (slow motion): left, the camera view with the pipeline's own overlay (tbot.app.overlay.draw_record on
  the frame's telemetry; corner brackets mark the TRUE clay); right, the field from above (PLAN, field frame:
  the gun at the origin, downrange up) and from the side (ELEVATION, height above ground along the clay's
  track) with each clay's true trajectory, the tracker's estimate, the predicted intercept, the true bore,
  the shot charge in flight, its pattern on arrival (hit / miss / debris) and the fragments of a broken
  clay; an INTERCEPT close-up (the solver's prediction, then the truth at the shot's arrival, to scale in
  the plane across the shot line); and a timeline (armed, clays, detections, confirmed track, P(hit), shots).
- .png: the summary still (all trajectories, shots and outcomes, decision reasons, the run's numbers).
contact_sheet.png: every summary of the call side by side.

Drawing is numpy + OpenCV only (anti-aliased Hershey text, mp4v video, MJPG/AVI if mp4v is unavailable).
"""

from __future__ import annotations

import argparse
import bisect
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from tbot.camera.synthetic import T0_NS
from tbot.config import add_common_args, load_from_args, resolve_path
from tbot.sim.scenario import SCENARIOS, SHOULDER_HEIGHT_M

log = logging.getLogger("tbot.simviz")

DEFAULT_OUT = "sessions/simviz"


# ============================================================================================ palette
def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))  # BGR


BG = _hex("0d1117")
PANEL = _hex("141a22")
PANEL_HI = _hex("1b232d")
GRID = _hex("232c37")
GRID_HI = _hex("313c4a")
TEXT = _hex("e6ebf1")
MUTED = _hex("8d99a8")
DIM = _hex("566170")
GROUND = _hex("1a1f1a")
CLAY = (_hex("ff8a3d"), _hex("c792ff"), _hex("f2e28c"))
TRACK = _hex("5ab8ff")
LEAD = _hex("ff4fd8")
BORE = _hex("eef6ff")
BORE_PRED = _hex("6ff3ff")
HIT = _hex("3ddc84")
MISS = _hex("ff4d5e")
DEBRIS = _hex("ffc145")
FIRE = _hex("ff6a3d")
ARMED = _hex("2fbf71")
SAFE = _hex("5d6673")
DET = _hex("4fd1c5")
OUTCOME = {"hit": HIT, "miss": MISS, "debris": DEBRIS, "none": MUTED}
OUTCOME_LABEL = {"hit": "HIT", "miss": "MISS", "debris": "DEBRIS", "none": "NO TARGET"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX
_SH = 4  # sub-pixel bits for cv2 drawing (smooth motion with LINE_AA)
_SF = float(1 << _SH)


def _clay_color(i: int) -> tuple[int, int, int]:
    return CLAY[int(i) % len(CLAY)]


def _mix(c, bg, a: float) -> tuple[int, int, int]:
    a = min(max(float(a), 0.0), 1.0)
    return tuple(int(round(float(x) * a + float(y) * (1.0 - a))) for x, y in zip(c, bg))


# ============================================================================================ primitives
def _ip(x: float, y: float) -> tuple[int, int]:
    x = min(max(float(x), -2e4), 2e4)
    y = min(max(float(y), -2e4), 2e4)
    return (int(round(x * _SF)), int(round(y * _SF)))


def _ok(*vals) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in vals)


def line(img, a, b, color, th: int = 1) -> None:
    if _ok(a[0], a[1], b[0], b[1]):
        cv2.line(img, _ip(*a), _ip(*b), color, th, cv2.LINE_AA, _SH)


def circle(img, c, r: float, color, th: int = 1) -> None:
    if _ok(c[0], c[1], r):
        cv2.circle(img, _ip(*c), max(int(round(float(r) * _SF)), 1), color, th, cv2.LINE_AA, _SH)


def polyline(img, pts: np.ndarray, color, th: int = 1, closed: bool = False) -> None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        return
    q = np.round(np.clip(pts, -2e4, 2e4) * _SF).astype(np.int32)
    cv2.polylines(img, [q], closed, color, th, cv2.LINE_AA, _SH)


def dashed(img, a, b, color, th: int = 1, dash: float = 6.0, gap: float = 4.0) -> None:
    if not _ok(a[0], a[1], b[0], b[1]):
        return
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    n = math.hypot(bx - ax, by - ay)
    if n < 1e-6 or n > 1e5:
        return
    ux, uy = (bx - ax) / n, (by - ay) / n
    s = 0.0
    while s < n:
        e = min(s + dash, n)
        line(img, (ax + ux * s, ay + uy * s), (ax + ux * e, ay + uy * e), color, th)
        s = e + gap


def dotted_poly(img, pts: np.ndarray, color, every: float = 7.0, r: float = 1.0) -> None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        return
    seg = np.hypot(*(np.diff(pts, axis=0).T))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    for d in np.arange(0.0, s[-1], every):
        x = float(np.interp(d, s, pts[:, 0]))
        y = float(np.interp(d, s, pts[:, 1]))
        circle(img, (x, y), r, color, -1)


def fill_rect(img, x0, y0, x1, y1, color, alpha: float = 1.0) -> None:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    if alpha >= 1.0:
        roi[:] = color
    else:
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)


def text(img, s: str, x, y, color=TEXT, scale: float = 0.4, font=FONT, th: int = 1, anchor: str = "l") -> int:
    (tw, _), _ = cv2.getTextSize(s, font, scale, th)
    if anchor == "r":
        x = x - tw
    elif anchor == "c":
        x = x - tw // 2
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, th, cv2.LINE_AA)
    return tw


def label(img, s: str, x, y, color=TEXT, bg=PANEL, scale: float = 0.36, alpha: float = 0.8, anchor: str = "l",
          border=None, font=FONT, pad: int = 3) -> tuple[int, int, int, int]:
    """Text with a translucent box behind it; (x, y) is the text baseline. Returns the box."""
    (tw, th_), base = cv2.getTextSize(s, font, scale, 1)
    if anchor == "r":
        x = x - tw
    elif anchor == "c":
        x = x - tw // 2
    x, y = int(x), int(y)
    box = (x - pad, y - th_ - pad, x + tw + pad, y + base + pad - 1)
    fill_rect(img, *box, bg, alpha)
    if border is not None:
        cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), border, 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, 1, cv2.LINE_AA)
    return box


def chip(img, s: str, x, y, color, scale: float = 0.4, anchor: str = "l", font=FONT, solid: bool = False) -> int:
    """A coloured status chip (tinted box, coloured border and text); returns its width."""
    bg = color if solid else _mix(color, PANEL, 0.22)
    fg = BG if solid else color
    box = label(img, s, x, y, fg, bg, scale, 1.0, anchor, border=color, font=font, pad=5)
    return box[2] - box[0]


def star(img, c, r: float, color, th: int = 1, n: int = 8) -> None:
    for k in range(n):
        a = math.pi * k / (n / 2.0) + 0.3
        rr = r if k % 2 == 0 else 0.55 * r
        line(img, (c[0] + 0.25 * rr * math.cos(a), c[1] + 0.25 * rr * math.sin(a)),
             (c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)), color, th)


def xmark(img, c, r: float, color, th: int = 2) -> None:
    line(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), color, th)
    line(img, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), color, th)


def diamond(img, c, r: float, color, filled: bool) -> None:
    pts = np.array([[c[0], c[1] - r], [c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]]], dtype=np.float64)
    if filled:
        cv2.fillPoly(img, [np.round(pts * _SF).astype(np.int32)], color, cv2.LINE_AA, _SH)
    polyline(img, pts, color if not filled else _mix(color, BG, 0.6), 1, closed=True)


def arrow(img, a, b, color, th: int = 1, tip: float = 6.0) -> None:
    if not _ok(a[0], a[1], b[0], b[1]):
        return
    line(img, a, b, color, th)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    if math.hypot(b[0] - a[0], b[1] - a[1]) < 3.0:
        return
    for s in (+1, -1):
        line(img, b, (b[0] - tip * math.cos(ang + s * 0.45), b[1] - tip * math.sin(ang + s * 0.45)), color, th)


def panel_title(img, title: str, sub: str = "") -> None:
    (tw, _), _ = cv2.getTextSize(title, FONT_B, 0.42, 1)
    (sw, _), _ = cv2.getTextSize(sub, FONT, 0.36, 1) if sub else ((0, 0), 0)
    fill_rect(img, 4, 3, 18 + tw + (sw + 8 if sub else 0), 24, PANEL, 0.75)
    w = text(img, title, 10, 17, TEXT, 0.42, FONT_B)
    if sub:
        text(img, sub, 18 + w, 17, MUTED, 0.36)


def _nice(target: float) -> float:
    for v in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100):
        if v >= target:
            return float(v)
    return 100.0


def scale_bar(img, x_right: int, y: int, px_per_m: float, target_px: float = 90.0, color=MUTED) -> None:
    if not (px_per_m > 0 and math.isfinite(px_per_m)):
        return
    m = _nice(target_px / px_per_m * 0.8)
    L = m * px_per_m
    x0 = x_right - L
    line(img, (x0, y), (x_right, y), color, 2)
    for xx in (x0, x_right):
        line(img, (xx, y - 4), (xx, y + 4), color, 1)
    s = f"{m:g} m"
    text(img, s, int(x0 + L / 2), y - 7, color, 0.34, anchor="c")


# ============================================================================================ recording
def _ts(t_ns) -> float:
    return (int(t_ns) - T0_NS) * 1e-9


@dataclass
class Step:
    """One processed frame: its mid-exposure time (s from scenario start), telemetry, image, true clay pixels."""

    t_s: float
    tele: dict
    image: np.ndarray | None
    truth_px: list[tuple[int, tuple[float, float]]]


@dataclass
class ShotGeom:
    """One shot as the judge scored it, plus the geometry to draw it (all world frame, camera origin)."""

    index: int
    step: int
    t_fire: float
    t_exit: float
    t_arrive: float
    outcome: str  # hit | miss | debris | none
    target: int | None
    range_m: float
    tof_s: float
    miss_m: float
    radius_m: float
    muzzle_w: np.ndarray
    bore_w: np.ndarray
    impact_w: np.ndarray
    clay_at_w: np.ndarray | None
    ds: np.ndarray  # distance down the shot line ...
    tofs: np.ndarray  # ... its time of flight ...
    drops: np.ndarray  # ... and gravity drop
    predicted: dict
    raw: dict

    def pellet(self, t: float) -> tuple[np.ndarray, float] | None:
        """(centre of the charge, distance flown) at time t while it is in flight, else None."""
        dt = float(t) - self.t_exit
        if dt < 0.0 or dt > float(self.tofs[-1]):
            return None
        d = float(np.interp(dt, self.tofs, self.ds))
        drop = float(np.interp(d, self.ds, self.drops))
        return self.muzzle_w + d * self.bore_w - np.array([0.0, 0.0, drop]), d


class Recorder:
    """The observer of one evaluate_run: keeps what it needs to draw the run, changes nothing in it."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.shots: list[ShotGeom] = []
        self.source = None
        self._frame = None

    # ---- evaluate_run hook
    def attach(self, source, pipe) -> None:
        self.source = source
        self.scenario = source.scenario
        self.cfg = source.cfg
        self.cam, self.mount, self.mask, self.bore_px = pipe.cam, pipe.mount, pipe.mask, pipe.bore_px
        self.shot_model = pipe.shot
        self.max_range_m = float(source.cfg.shot.max_range_m)
        read = source.read

        def _read(*args, **kwargs):  # pass-through that remembers the frame the pipeline is given
            f = read(*args, **kwargs)
            self._frame = f
            return f

        source.read = _read

    def step(self, tele: dict, shot: dict | None) -> None:
        f = self._frame
        img = None if f is None else np.array(f.image, copy=True)
        t_ns = int(tele["t_exposure_ns"])
        truth = self.source.truth(t_ns)  # read-only (the run asks the same question of every frame)
        px = [(int(c["index"]), (float(c["clay_px"][0]), float(c["clay_px"][1])))
              for c in truth["clays"] if c.get("clay_px") is not None]
        self.steps.append(Step(t_s=_ts(t_ns), tele=tele, image=img, truth_px=px))
        if shot is not None:
            self.shots.append(self._geometry(shot, len(self.steps) - 1))

    # ---- shot geometry (the judge's own formula: true bore at exit, ToF, drop)
    def _geometry(self, shot: dict, step_idx: int) -> ShotGeom:
        sc, mount, sm = self.scenario, self.mount, self.shot_model
        t_exit = _ts(shot["t_exit_ns"])
        R = np.asarray(sc.R_wg(t_exit), dtype=np.float64).reshape(3, 3)
        muzzle = R @ (np.asarray(mount.muzzle_pos_g, np.float64) - np.asarray(mount.cam_pos_g, np.float64))
        bore = R @ np.asarray(mount.bore_dir_g, np.float64).reshape(3)
        bore = bore / float(np.linalg.norm(bore))
        el = math.asin(max(-1.0, min(1.0, float(bore[2]))))
        rng = float(shot.get("range_m", float("nan")))
        target = shot.get("target_clay")
        if math.isfinite(rng):
            outcome = "hit" if shot.get("hit") else ("debris" if shot.get("debris_shot") else "miss")
            t_arrive = _ts(shot["t_arrive_ns"])
        else:
            outcome = "none"
            rng = min(35.0, self.max_range_m)
            t_arrive = t_exit + float(sm.time_of_flight(rng))
        ds = np.linspace(0.0, rng, 160)
        tofs = np.asarray(sm.time_of_flight(ds), dtype=np.float64)
        drops = np.asarray(sm.drop(ds, el), dtype=np.float64)
        impact = muzzle + rng * bore - np.array([0.0, 0.0, float(drops[-1])])
        clay_at = None
        if target is not None:
            p = sc.clay_pos_w(t_arrive, int(target))
            clay_at = None if p is None else np.asarray(p, dtype=np.float64)
        return ShotGeom(
            index=len(self.shots) + 1, step=step_idx, t_fire=_ts(shot["t_fire_ns"]), t_exit=t_exit,
            t_arrive=t_arrive, outcome=outcome, target=None if target is None else int(target), range_m=rng,
            tof_s=float(shot.get("tof_s", float("nan"))), miss_m=float(shot.get("miss_m", float("nan"))),
            radius_m=float(shot.get("pattern_radius_m", float("nan"))), muzzle_w=muzzle, bore_w=bore,
            impact_w=impact, clay_at_w=clay_at, ds=ds, tofs=tofs, drops=drops,
            predicted=dict(shot.get("predicted") or {}), raw=shot)


def record_run(cfg, calib, scenario: str, seed: int, **kwargs) -> tuple[Recorder, dict]:
    """evaluate_run (exactly `tbot sim`'s run) with a Recorder attached; returns (recorder, result)."""
    from tbot.tools.sim_eval import evaluate_run

    rec = Recorder()
    result = evaluate_run(cfg, calib, scenario, seed, observer=rec, **kwargs)
    return rec, result


# ============================================================================================ scene
def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def field_rotation(name: str) -> np.ndarray:
    """World -> field rotation (tbot.sim.scenario: the field frame is rotated about Z onto the hold point)."""
    try:
        from tbot.sim.scenario import _SPECS

        hp = _SPECS[name]["hold_point"]
        return _rot_z(math.atan2(float(hp[1]), float(hp[0])))
    except Exception:  # noqa: BLE001 - an unknown layout: draw in the world frame
        return np.eye(3)


@dataclass
class ClayPath:
    index: int
    label: str
    t: np.ndarray
    P: np.ndarray  # (n, 3) world, launch -> ground / end of flight / break
    t_break: float | None
    t_end: float


@dataclass
class Scene:
    """Everything the views draw, computed once after the run."""

    name: str
    seed: int
    description: str
    rec: Recorder
    result: dict
    duration: float
    Rf: np.ndarray
    clays: list[ClayPath]
    frag_paths: list[tuple[int, float, np.ndarray, np.ndarray]]  # (clay, t_break, t (n,), P (n, 3)) per fragment
    steps_t: list[float]
    track_pts: list[tuple[float, int, np.ndarray, np.ndarray | None, bool]]
    elev_dir: np.ndarray
    elev_axis_label: str
    t_end_video: float
    shots: list[ShotGeom] = field(default_factory=list)

    # ---- projections into the two views' metric coordinates (a: horizontal, b: up)
    def plan_ab(self, Pw) -> tuple[np.ndarray, np.ndarray]:
        f = np.asarray(Pw, dtype=np.float64).reshape(-1, 3) @ self.Rf.T
        return -f[:, 1], f[:, 0]  # shooter's right, downrange

    def elev_ab(self, Pw) -> tuple[np.ndarray, np.ndarray]:
        f = np.asarray(Pw, dtype=np.float64).reshape(-1, 3) @ self.Rf.T
        return f[:, :2] @ self.elev_dir, f[:, 2] + SHOULDER_HEIGHT_M

    def step_at(self, t: float) -> int:
        return bisect.bisect_right(self.steps_t, float(t) + 1e-9) - 1


def build_scene(rec: Recorder, result: dict) -> Scene:
    sc = rec.scenario
    name = str(result.get("scenario", sc.name))
    Rf = field_rotation(name)
    clays = []
    for c in sc.clays:
        t_break = sc.broken.get(c.index)
        t1 = float(c.curve.t_end) if t_break is None else float(t_break)
        t = np.arange(float(c.launch_t_s), t1 + 1e-9, 0.005)
        if t.size < 2:
            t = np.array([float(c.launch_t_s), t1])
        P = np.asarray(c.curve.value(t), dtype=np.float64).reshape(-1, 3)
        under = np.flatnonzero(P[:, 2] < -SHOULDER_HEIGHT_M)  # on the ground
        if under.size:
            k = max(int(under[0]), 2)
            t, P = t[:k], P[:k]
        lab = str(c.label)
        if lab == f"clay{c.index}":  # an unnamed clay
            lab = "clay" if len(sc.clays) == 1 else f"clay {c.index + 1}"
        clays.append(ClayPath(c.index, lab, t, P, None if t_break is None else float(t_break), float(t[-1])))
    frags = []
    for d in sc.debris.values():
        for k in range(d.pos0_w.shape[0]):
            tt = np.arange(d.t_break_s, d.t_break_s + float(d.life_s[k]), 0.01)
            if tt.size < 2:
                continue
            P = np.stack([d.positions(x)[k] for x in tt])
            frags.append((int(d.clay_index), float(d.t_break_s), tt, P))
    track_pts = []
    from tbot.optics import azel_to_dir

    for st in rec.steps:
        # the world views show the track the solver is working on (scenery tracks are in the camera view)
        tid = (st.tele.get("decision") or {}).get("track_id")
        for tr in st.tele.get("tracks") or ():
            if tid is None or int(tr["id"]) != int(tid):
                continue
            s = tr.get("s")
            if s is not None and len(s) >= 6:
                p, v = np.asarray(s[0:3], np.float64), np.asarray(s[3:6], np.float64)
            else:
                r = float(tr.get("range_m", float("nan")))
                if not math.isfinite(r):
                    continue
                p = np.asarray(azel_to_dir(math.radians(tr["az_deg"]), math.radians(tr["el_deg"])), np.float64) * r
                v = None
            on_clay = False
            n = float(np.linalg.norm(p))
            if n > 1e-6:
                for c in sc.clays:
                    if c.launch_t_s <= st.t_s <= c.curve.t_end and sc.clay_intact(st.t_s, c.index):
                        dc = np.asarray(sc.clay_dir_w(st.t_s, c.index), np.float64)
                        if float(p @ dc) / n > math.cos(math.radians(2.5)):
                            on_clay = True
            track_pts.append((st.t_s, int(tr["id"]), p, v, bool(tr.get("confirmed")) and on_clay))
    # elevation: the plane through the vertical and the clays' main horizontal direction of travel
    hv = []
    for c in sc.clays:
        v = Rf @ np.asarray(c.launch_vel_w, dtype=np.float64)
        n = float(np.hypot(v[0], v[1]))
        if n > 1e-6:
            hv.append(v[:2] / n)
    if hv:
        M = sum(np.outer(h, h) for h in hv)
        w, V = np.linalg.eigh(M)
        d = V[:, int(np.argmax(w))]
        if float(d @ hv[0]) < 0.0:
            d = -d
        if abs(d[0]) > 0.9 and d[0] < 0.0:  # a mostly downrange plane is always drawn downrange to the right
            d = -d
    else:
        d = np.array([1.0, 0.0])
    if abs(d[0]) > 0.9:
        axis_label = "downrange, m" if d[0] > 0 else "up-range, m"
    elif abs(d[1]) > 0.9:
        axis_label = "across the field, m  (shooter's right ->)" if d[1] < 0 else "across the field, m  (<- shooter's left)"
    else:
        axis_label = "along the clay's track, m"
    steps_t = [s.t_s for s in rec.steps]
    t_end = float(sc.duration_s)
    if rec.steps and steps_t[-1] < t_end - 0.1:  # a capped run (--max-frames)
        t_end = min(t_end, steps_t[-1] + 0.05)
    if rec.shots:  # let the last charge arrive and the pieces fly (the world is known past the last frame)
        t_end = max(t_end, min(max(s.t_arrive for s in rec.shots) + 0.45, t_end + 0.8))
    return Scene(name=name, seed=int(result.get("seed", sc.seed)), description=SCENARIOS.get(name, sc.description),
                 rec=rec, result=result, duration=float(sc.duration_s), Rf=Rf, clays=clays, frag_paths=frags,
                 steps_t=steps_t, track_pts=track_pts, elev_dir=d, elev_axis_label=axis_label, t_end_video=t_end,
                 shots=list(rec.shots))


# ============================================================================================ axes
class Axes:
    """Metric (a right, b up) -> panel pixels, bounds expanded to fill the drawing rectangle."""

    def __init__(self, w: int, h: int, a_rng, b_rng, pad=(12, 12, 28, 12), max_exag: float = 1.0):
        l, r, t, b = pad
        iw, ih = max(w - l - r, 10), max(h - t - b, 10)
        a0, a1 = float(a_rng[0]), float(a_rng[1])
        b0, b1 = float(b_rng[0]), float(b_rng[1])
        da, db = max(a1 - a0, 1e-3), max(b1 - b0, 1e-3)
        s = min(iw / da, ih / db)
        self.sx = s
        self.sy = min(ih / db, s * max_exag)
        self.exag = self.sy / self.sx
        ac, bc = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
        self.a0, self.a1 = ac - 0.5 * iw / self.sx, ac + 0.5 * iw / self.sx
        self.b0, self.b1 = bc - 0.5 * ih / self.sy, bc + 0.5 * ih / self.sy
        self.left, self.top, self.iw, self.ih = l, t, iw, ih

    def px(self, a, b) -> tuple[float, float]:
        return (self.left + (float(a) - self.a0) * self.sx, self.top + self.ih - (float(b) - self.b0) * self.sy)

    def pts(self, A, B) -> np.ndarray:
        A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
        return np.stack([self.left + (A - self.a0) * self.sx, self.top + self.ih - (B - self.b0) * self.sy], axis=1)


def _bounds(vals: list[np.ndarray], lo_hi_min: tuple[float, float], margin: float, min_span: float):
    v = np.concatenate([np.asarray(x, np.float64).ravel() for x in vals] + [np.asarray(lo_hi_min, np.float64)])
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()) - margin, float(v.max()) + margin
    if hi - lo < min_span:
        c = 0.5 * (lo + hi)
        lo, hi = c - 0.5 * min_span, c + 0.5 * min_span
    return lo, hi


def _view_points(scene: Scene) -> np.ndarray:
    pts = [np.zeros((1, 3))]
    pts += [c.P for c in scene.clays]
    pts += [s.impact_w.reshape(1, 3) for s in scene.shots]
    pts += [fp[3] for fp in scene.frag_paths]
    return np.concatenate(pts, axis=0)


def plan_axes(scene: Scene, w: int, h: int) -> Axes:
    A, B = scene.plan_ab(_view_points(scene))
    if not scene.clays:
        A, B = np.array([-12.0, 12.0]), np.array([0.0, 28.0])
    return Axes(w, h, _bounds([A], (0.0, 0.0), 3.0, 20.0), _bounds([B], (-2.0, 0.0), 3.0, 20.0), pad=(12, 12, 30, 14))


def elev_axes(scene: Scene, w: int, h: int) -> Axes:
    A, B = scene.elev_ab(_view_points(scene))
    if not scene.clays:
        A, B = np.array([0.0, 28.0]), np.array([0.0, 4.0])
    a = _bounds([A], (0.0, 0.0), 2.0, 16.0)
    b0, b1 = 0.0, max(float(np.nanmax(B)) + 1.0, SHOULDER_HEIGHT_M + 2.0)
    return Axes(w, h, a, (b0 - 0.6, b1), pad=(34, 12, 30, 22), max_exag=4.0)


# ============================================================================================ world views
def _proj(scene: Scene, ax: Axes, view: str) -> Callable[[np.ndarray], np.ndarray]:
    f = scene.plan_ab if view == "plan" else scene.elev_ab

    def P(Pw) -> np.ndarray:
        A, B = f(Pw)
        return ax.pts(A, B)

    return P


def _decision_at(scene: Scene, idx: int) -> dict | None:
    if idx < 0:
        return None
    return scene.rec.steps[idx].tele.get("decision")


def _intercept_w(dec: dict | None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(predicted clay position at arrival, predicted bore at exit, range) of a decision, or None."""
    if not dec or dec.get("lead_dir_w") is None or dec.get("bore_dir_w") is None:
        return None
    r = float(dec.get("range_m", float("nan")))
    if not math.isfinite(r) or r <= 0.0:
        return None
    lead = np.asarray(dec["lead_dir_w"], np.float64)
    bore = np.asarray(dec["bore_dir_w"], np.float64)
    return lead * r, bore, r


def draw_world(img: np.ndarray, scene: Scene, ax: Axes, view: str, t: float, summary: bool = False) -> None:
    """The plan or the elevation of the field at time t (summary: the whole run, no 'now')."""
    sc = scene.rec.scenario
    P = _proj(scene, ax, view)
    h, w = img.shape[:2]
    img[:] = PANEL
    g = P(np.zeros((1, 3)))[0]

    # ---- backdrop: range rings (plan) or ground + eye level (elevation)
    if view == "plan":
        r_max = math.hypot(max(abs(ax.a0), abs(ax.a1)), max(abs(ax.b0), abs(ax.b1)))
        step = 10.0 if r_max < 90 else 20.0
        r = step
        while r <= r_max:
            circle(img, g, r * ax.sx, GRID, 1)
            q = (g[0] + r * ax.sx, g[1])
            if 40 < q[0] < w - 30:
                text(img, f"{r:.0f} m", int(q[0]) + 3, int(q[1]) - 4, DIM, 0.32)
            r += step
        dashed(img, g, (g[0], ax.top - 30), GRID_HI, 1, 3, 5)
        title = ("PLAN", "top-down - gun at origin, downrange up")
    else:
        y_ground = ax.px(0.0, 0.0)[1]
        fill_rect(img, 0, int(y_ground), w, h, GROUND)
        for xx in range(-int(h), w, 9):
            line(img, (xx, h), (xx + (h - y_ground), y_ground), _mix(GRID, GROUND, 0.5), 1)
        line(img, (0, y_ground), (w, y_ground), GRID_HI, 1)
        y_eye = ax.px(0.0, SHOULDER_HEIGHT_M)[1]
        dashed(img, (0, y_eye), (w, y_eye), GRID, 1, 4, 5)
        text(img, "eye level", w - 8, int(y_eye) - 4, DIM, 0.32, anchor="r")
        # height ticks
        for hm in range(0, int(ax.b1) + 1, 2 if ax.b1 > 8 else 1):
            y = ax.px(0.0, hm)[1]
            if ax.top - 2 <= y <= ax.top + ax.ih + 2:
                line(img, (ax.left - 5, y), (ax.left - 1, y), MUTED, 1)
                text(img, f"{hm}", ax.left - 8, int(y) + 4, MUTED, 0.32, anchor="r")
        text(img, scene.elev_axis_label, ax.left + 4, h - 6, DIM, 0.32)
        sub = "height above ground, m"
        if ax.exag > 1.05:
            sub += f"  (vertical x{ax.exag:.1f})"
        title = ("ELEVATION", sub)

    idx = scene.step_at(t)

    # ---- the true bore now (under everything else)
    if not summary:
        b = np.asarray(sc.bore_dir_w(min(t, scene.duration)), np.float64)
        far = P((b * 60.0).reshape(1, 3))[0]
        line(img, g, far, _mix(BORE, PANEL, 0.10), 4)
        line(img, g, far, _mix(BORE, PANEL, 0.6), 1)
        d2 = np.array(far) - np.array(g)
        n = float(np.hypot(*d2))
        u = d2 / n if n > 1e-6 else np.array([0.0, -1.0])
    else:
        u = np.array([0.0, -1.0]) if view == "plan" else np.array([1.0, 0.0])

    # ---- clays: whole true trajectory (dotted), flown part (solid), now
    for c in scene.clays:
        col = _clay_color(c.index)
        pts = P(c.P)
        if summary:
            polyline(img, pts, col, 2)
        else:
            k = int(np.searchsorted(c.t, t, side="right"))
            if c.t_break is None or t < c.t_break:
                dotted_poly(img, pts[max(k - 1, 0):], _mix(col, PANEL, 0.45), 7.0, 0.9)
            if k >= 2:
                polyline(img, pts[:k], _mix(col, PANEL, 0.85), 2)
        p_launch = pts[0]
        cv2.rectangle(img, (int(p_launch[0]) - 3, int(p_launch[1]) - 3), (int(p_launch[0]) + 3, int(p_launch[1]) + 3),
                      _mix(col, PANEL, 0.7), 1, cv2.LINE_AA)
        if summary or t < c.t[0] + 0.15:
            label(img, c.label, p_launch[0] + 7, p_launch[1] + 4 + 14 * c.index, _mix(col, PANEL, 0.9), PANEL, 0.34,
                  0.7)
        if c.t_break is not None and (summary or t >= c.t_break):
            pb = pts[-1]
            age = 1.0 if summary else (t - c.t_break)
            star(img, pb, 5.0 + 9.0 * min(age / 0.12, 1.0) if age < 0.4 else 8.0, HIT, 1)
        if not summary and c.t[0] <= t <= c.t_end and (c.t_break is None or t < c.t_break):
            pos = np.asarray(sc.clay_pos_w(t, c.index), np.float64)
            q = P(pos.reshape(1, 3))[0]
            vis = sc.clay_visible(t, c.index)
            circle(img, q, 5.5, BG, -1)
            circle(img, q, 4.5, col, -1 if vis else 1)
            if view == "plan":
                s_ = f"{c.label}  {float(np.linalg.norm(pos)):.0f} m"
                (tw, _), _ = cv2.getTextSize(s_, FONT, 0.34, 1)
                if q[0] + 12 + tw < w - 6:
                    label(img, s_, q[0] + 9, q[1] - 7, col, PANEL, 0.34, 0.75)
                else:
                    label(img, s_, q[0] - 9, q[1] - 7, col, PANEL, 0.34, 0.75, anchor="r")

    # ---- debris
    for ci, tb, tt, FP in scene.frag_paths:
        col = _clay_color(ci)
        if summary:
            polyline(img, P(FP), _mix(col, PANEL, 0.35), 1)
            continue
        if t < tb or t > tt[-1]:
            continue
        k = int(np.searchsorted(tt, t, side="right"))
        fp = P(FP[:k])
        polyline(img, fp, _mix(col, PANEL, 0.3), 1)
        a = 1.0 - (t - tb) / max(tt[-1] - tb, 1e-3)
        circle(img, fp[-1], 2.0, _mix(col, PANEL, 0.35 + 0.65 * a), -1)

    # ---- tracker estimate: trail, now (diamond + velocity)
    cur_t = scene.steps_t[idx] if idx >= 0 else -1.0
    for (ts_, tid, p, v, conf) in scene.track_pts:
        if not conf or (not summary and ts_ > t):
            continue
        if not summary and ts_ == cur_t:
            continue
        q = P(p.reshape(1, 3))[0]
        circle(img, q, 1.3, _mix(TRACK, PANEL, 0.55), -1)
    if not summary and idx >= 0:
        for (ts_, tid, p, v, conf) in scene.track_pts:
            if ts_ != cur_t:
                continue
            q = P(p.reshape(1, 3))[0]
            if v is not None:
                q2 = P((p + 0.25 * v).reshape(1, 3))[0]
                arrow(img, q, q2, _mix(TRACK, PANEL, 0.9 if conf else 0.5), 1, 5)
            diamond(img, q, 5.0, TRACK if conf else _mix(TRACK, PANEL, 0.5), conf)
            if view == "plan":
                label(img, f"trk #{tid}", q[0] - 8, q[1] + 18, TRACK, PANEL, 0.32, 0.7, anchor="r")

    # ---- the solver's intercept (now) / at each shot (summary)
    preds = []
    if summary:
        for s in scene.shots:
            iw = _intercept_w(s.predicted)
            if iw is not None:
                preds.append((iw, f"#{s.index} pred"))
    else:
        iw = _intercept_w(_decision_at(scene, idx))
        if iw is not None and (t - cur_t) < 0.1:
            preds.append((iw, "intercept"))
    for (Lw, Bw, r), name in preds:
        qL = P(Lw.reshape(1, 3))[0]
        qB = P((Bw * r).reshape(1, 3))[0]
        dashed(img, g, qB, _mix(BORE_PRED, PANEL, 0.6), 1, 5, 4)
        line(img, qB, qL, _mix(LEAD, PANEL, 0.6), 1)
        xmark(img, qL, 4.5, LEAD, 2)
        if view == "plan":
            label(img, name, qL[0] + 8, qL[1] + 14, LEAD, PANEL, 0.32, 0.7)

    # ---- shots: trigger delay, charge in flight, pattern on arrival
    for s in scene.shots:
        col = OUTCOME[s.outcome]
        if not summary and t < s.t_fire:
            continue
        end = s.impact_w
        qI = P(end.reshape(1, 3))[0]
        if summary or t >= s.t_arrive:
            line(img, P(s.muzzle_w.reshape(1, 3))[0], qI, _mix(col, PANEL, 0.45), 1)
            r_px = max(s.radius_m * ax.sx if math.isfinite(s.radius_m) else 0.0, 4.5)
            ov = img.copy()
            circle(ov, qI, r_px, col, -1)
            cv2.addWeighted(ov, 0.28, img, 0.72, 0.0, dst=img)
            circle(img, qI, r_px, col, 1)
            if s.clay_at_w is not None and s.outcome != "hit":
                qc = P(s.clay_at_w.reshape(1, 3))[0]
                line(img, qI, qc, _mix(col, PANEL, 0.8), 1)
                circle(img, qc, 3.0, _mix(_clay_color(s.target or 0), PANEL, 0.9), 1)
            tag = f"#{s.index} {OUTCOME_LABEL[s.outcome]}"
            if s.outcome in ("miss", "debris") and math.isfinite(s.miss_m):
                tag += f" {s.miss_m:.2f} m"
            if view == "plan" or summary:
                label(img, tag, qI[0] + 9, qI[1] - 10 - 14 * (s.index - 1), col, PANEL, 0.36, 0.85, border=col)
        elif t >= s.t_exit:
            pl = s.pellet(t)
            if pl is not None:
                pos, d = pl
                q = P(pos.reshape(1, 3))[0]
                tail = s.muzzle_w + max(d - 4.0, 0.0) * s.bore_w
                line(img, P(tail.reshape(1, 3))[0], q, _mix(FIRE, PANEL, 0.5), 5)
                line(img, P(tail.reshape(1, 3))[0], q, FIRE, 2)
                circle(img, q, 6.0, _mix(FIRE, PANEL, 0.35), -1)
                circle(img, q, 3.2, _hex("fff1c7"), -1)
            if t - s.t_exit < 0.05:
                star(img, g, 12.0, FIRE, 2)
        else:  # the trigger is being pulled: the command is out, the shot is not
            k = (t - s.t_fire) / max(s.t_exit - s.t_fire, 1e-3)
            circle(img, g, 8.0 + 10.0 * k, _mix(FIRE, PANEL, 0.9 - 0.6 * k), 1)

    # ---- the gun
    tri = np.array([g + 11 * u, g - 6 * u + 6 * np.array([-u[1], u[0]]), g - 6 * u - 6 * np.array([-u[1], u[0]])])
    cv2.fillPoly(img, [np.round(tri * _SF).astype(np.int32)], BORE, cv2.LINE_AA, _SH)
    polyline(img, tri, BG, 1, closed=True)
    if view == "plan":
        label(img, "gun", g[0] + 10, g[1] + 12, MUTED, PANEL, 0.32, 0.6)
    scale_bar(img, w - 14, h - 12 if view == "plan" else h - 8, ax.sx)
    panel_title(img, *title)


# ============================================================================================ close-up
def _plane_basis(bore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = bore / max(float(np.linalg.norm(bore)), 1e-9)
    right = np.cross(b, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(right))
    right = right / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    up = np.cross(right, b)
    return right, up


def _pellets(seed: int, n: int = 140) -> np.ndarray:
    rng = np.random.default_rng([9137, int(seed)])  # cosmetic, its own stream
    return rng.standard_normal((n, 2)) * 0.62


def draw_closeup(img: np.ndarray, scene: Scene, t: float, summary_shot: ShotGeom | None = None,
                 summary_best: bool = False) -> None:
    """INTERCEPT: the plane across the shot line, to scale - the solver's prediction, or the truth at arrival."""
    h, w = img.shape[:2]
    img[:] = PANEL
    sc = scene.rec.scenario
    mode, shot, sol = None, None, None
    if summary_shot is not None:
        mode, shot = "truth", summary_shot
    elif summary_best:
        best = None
        for i, st in enumerate(scene.rec.steps):
            d = st.tele.get("decision") or {}
            p = float(d.get("p_hit", float("nan")))
            if math.isfinite(p) and _intercept_w(d) is not None and (best is None or p > best[0]):
                best = (p, i)
        if best is not None:
            st = scene.rec.steps[best[1]]
            mode, sol = "solver", (st.tele["decision"], f"best solution, t {st.t_s:.2f} s (not fired)")
    else:
        arrived = [s for s in scene.shots if s.t_arrive <= t]
        flying = [s for s in scene.shots if s.t_fire <= t < s.t_arrive and _intercept_w(s.predicted) is not None]
        idx = scene.step_at(t)
        dec = _decision_at(scene, idx)
        live = _intercept_w(dec) is not None and (t - scene.steps_t[idx]) < 0.1 if idx >= 0 else False
        if flying:
            mode, sol = "solver", (flying[-1].predicted, f"shot #{flying[-1].index} - as predicted at the command")
        elif arrived and (t - arrived[-1].t_arrive < 0.8 or not live):
            mode, shot = "truth", arrived[-1]
        elif live:
            mode, sol = "solver", (dec, "solver - predicted at exit")
        elif arrived:
            mode, shot = "truth", arrived[-1]

    if mode is None:
        panel_title(img, "INTERCEPT")
        text(img, "no firing solution yet", w // 2, h // 2, DIM, 0.4, anchor="c")
        return

    cx, cy = w / 2.0, (h + 24) / 2.0
    if mode == "truth":
        right, up = _plane_basis(shot.bore_w)
        rad = shot.radius_m if math.isfinite(shot.radius_m) else 0.3
        off = np.zeros(2)
        path = None
        if shot.clay_at_w is not None:
            rel = shot.clay_at_w - shot.impact_w
            off = np.array([rel @ right, rel @ up])
            tt = shot.t_arrive + np.linspace(-0.12, 0.12, 25)
            Pc = np.stack([np.asarray(sc.clay_pos_w(x, shot.target), np.float64) for x in tt]) - shot.impact_w
            path = np.stack([Pc @ right, Pc @ up], axis=1)
        sig = None
        col = OUTCOME[shot.outcome]
        title_sub = f"shot #{shot.index} - truth at arrival"
    else:
        dec, title_sub = sol
        Lw, Bw, r = _intercept_w(dec)
        right, up = _plane_basis(Bw)
        rel = Lw - Bw * r
        off = np.array([rel @ right, rel @ up])
        rad = r * math.tan(math.radians(float(dec.get("pattern_deg", 1.0))))
        sg = float(dec.get("sigma_deg", float("nan")))
        sig = r * math.tan(math.radians(sg)) if math.isfinite(sg) else None
        path = None
        col = BORE_PRED
        shot = None
    half = max(1.5 * rad, 1.3 * float(np.hypot(*off)) + 0.15, 0.5)
    s = (min(w, h - 40) / 2.0 - 8.0) / half

    def q(x, y):
        return (cx + x * s, cy - y * s)

    # grid + cross hairs on the shot line
    for gx in np.arange(-math.floor(half / 0.25) * 0.25, half + 1e-9, 0.25):
        line(img, q(gx, -half * 2), q(gx, half * 2), GRID, 1)
        line(img, q(-half * 2, gx), q(half * 2, gx), GRID, 1)
    # pattern
    ov = img.copy()
    circle(ov, q(0, 0), rad * s, col, -1)
    cv2.addWeighted(ov, 0.16, img, 0.84, 0.0, dst=img)
    if mode == "truth":
        for px_, py_ in _pellets(shot.index + 31 * scene.seed) * rad:
            circle(img, q(px_, py_), 1.1, _mix(col, PANEL, 0.7), -1)
    circle(img, q(0, 0), rad * s, col, 1)
    if sig is not None:
        for k in range(0, 360, 12):
            a0, a1 = math.radians(k), math.radians(k + 6)
            line(img, q(rad * 0 + (rad + sig) * math.cos(a0), (rad + sig) * math.sin(a0)),
                 q((rad + sig) * math.cos(a1), (rad + sig) * math.sin(a1)), _mix(col, PANEL, 0.5), 1)
    line(img, q(-0.06 * half, 0), q(0.06 * half, 0), col, 1)
    line(img, q(0, -0.06 * half), q(0, 0.06 * half), col, 1)
    # the clay
    cc = _clay_color(shot.target if shot is not None and shot.target is not None else 0)
    if path is not None:
        pp = np.array([q(x, y) for x, y in path])
        dotted_poly(img, pp, _mix(cc, PANEL, 0.6), 6.0, 0.9)
        k = len(pp) // 2  # the arrival sample: an arrow head just before it shows the clay's direction
        if float(np.hypot(*(pp[k] - pp[k - 2]))) > 3.0:
            v = (pp[k] - pp[k - 2]) / float(np.hypot(*(pp[k] - pp[k - 2])))
            a0 = pp[k] - v * (max(0.5 * float(sc.clay_diameter_m) * s, 2.5) + 22.0)
            arrow(img, tuple(a0), tuple(a0 + v * 14.0), _mix(cc, PANEL, 0.8), 1, 6)
    cpt = q(*off)
    clay_r = max(0.5 * float(sc.clay_diameter_m) * s, 2.5)
    if mode == "truth" and shot.outcome == "hit":
        star(img, cpt, max(clay_r * 2.2, 9.0), HIT, 1)
        circle(img, cpt, clay_r, cc, -1)
    else:
        circle(img, cpt, clay_r, cc, -1)
    if float(np.hypot(*off)) > 0.02:
        dist = float(np.hypot(*off))
        line(img, q(0, 0), cpt, _mix(TEXT, PANEL, 0.6), 1)
        mid = q(off[0] / 2, off[1] / 2)
        label(img, f"{dist:.2f} m", mid[0] + 6, mid[1] - 4, TEXT, PANEL, 0.32, 0.75)
    scale_bar(img, w - 10, 44, s, 50.0)
    panel_title(img, "INTERCEPT")
    text(img, title_sub, 10, 34, MUTED, 0.34)
    if mode == "truth":
        verdict = OUTCOME_LABEL[shot.outcome]
        clay_name = ""
        if shot.target is not None and shot.target < len(scene.clays):
            clay_name = f" '{scene.clays[shot.target].label}'"
        chip(img, verdict + clay_name, 10, h - 26, col, 0.4, font=FONT_B, solid=True)
        det = f"pattern r {rad:.2f} m  @ {shot.range_m:.1f} m" if math.isfinite(shot.range_m) else ""
        text(img, det, 10, h - 8, MUTED, 0.32)
    else:
        p = float(dec.get("p_hit", float("nan")))
        text(img, f"P(hit) {p:.2f}" if math.isfinite(p) else "P(hit) -", 10, h - 26, BORE_PRED, 0.42, FONT_B)
        text(img, f"miss {float(dec.get('miss_deg', float('nan'))):.2f} / pat "
                  f"{float(dec.get('pattern_deg', float('nan'))):.2f} deg @ {float(dec.get('range_m')):.1f} m",
             10, h - 8, MUTED, 0.32)


# ============================================================================================ timeline
_TL_LABEL_W = 74


def _tl_x(scene: Scene, w: int, t: float) -> float:
    return _TL_LABEL_W + (float(t) / max(scene.t_end_video, 1e-3)) * (w - _TL_LABEL_W - 14)


def timeline_base(scene: Scene, w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = PANEL
    steps = scene.rec.steps
    rows = {"armed": (8, 14), "clays": (20, 32), "detect": (38, 44), "track": (50, 56), "P(hit)": (61, 79),
            "shots": (84, 94)}
    X = lambda t: _tl_x(scene, w, t)  # noqa: E731
    for name, (y0, y1) in rows.items():
        text(img, name, _TL_LABEL_W - 8, (y0 + y1) // 2 + 4, MUTED, 0.33, anchor="r")
        fill_rect(img, int(X(0.0)), y0, int(X(scene.t_end_video)) + 1, y1 + 1, PANEL_HI)
    T = scene.steps_t + [scene.t_end_video]
    for i, st in enumerate(steps):
        x0, x1 = X(st.t_s), X(T[i + 1])
        tele = st.tele
        if tele.get("armed"):
            fill_rect(img, int(x0), rows["armed"][0], int(math.ceil(x1)), rows["armed"][1] + 1, _mix(ARMED, PANEL, 0.75))
        if int(tele.get("n_detections", 0)) > 0:
            line(img, (x0, rows["detect"][0]), (x0, rows["detect"][1]), DET, 1)
        if any(tr.get("confirmed") for tr in tele.get("tracks") or ()):
            fill_rect(img, int(x0), rows["track"][0], int(math.ceil(x1)), rows["track"][1] + 1, _mix(TRACK, PANEL, 0.8))
    # clays: in flight (dim), visible to the camera (bright), broken (star)
    sc = scene.rec.scenario
    nc = max(len(scene.clays), 1)
    y0, y1 = rows["clays"]
    hh = (y1 - y0 + 1) / nc
    for c in scene.clays:
        ya, yb = int(y0 + c.index * hh), int(y0 + (c.index + 1) * hh) - 1
        col = _clay_color(c.index)
        fill_rect(img, int(X(c.t[0])), ya, int(X(min(c.t_end, scene.t_end_video))), yb + 1, _mix(col, PANEL, 0.3))
        tt = np.arange(c.t[0], min(c.t_end, scene.t_end_video), 0.005)
        vis = np.array([sc.clay_visible(x, c.index) for x in tt], bool) if tt.size else np.zeros(0, bool)
        for k in np.flatnonzero(vis):
            fill_rect(img, int(X(tt[k])), ya, int(math.ceil(X(tt[k] + 0.005))), yb + 1, _mix(col, PANEL, 0.85))
        if c.t_break is not None:
            star(img, (X(c.t_break), (ya + yb) / 2), 5.0, HIT, 1)
    # P(hit)
    y0, y1 = rows["P(hit)"]
    dashed(img, (X(0.0), (y0 + y1) / 2), (X(scene.t_end_video), (y0 + y1) / 2), GRID_HI, 1, 2, 4)
    seg: list[tuple[float, float]] = []
    for st in steps + [None]:
        p = float((st.tele.get("decision") or {}).get("p_hit", float("nan"))) if st is not None else float("nan")
        if math.isfinite(p):
            seg.append((X(st.t_s), y1 - p * (y1 - y0)))
        else:
            if len(seg) >= 2:
                polyline(img, np.array(seg), BORE_PRED, 1)
            elif len(seg) == 1:
                circle(img, seg[0], 1.2, BORE_PRED, -1)
            seg = []
    # shots
    y0, y1 = rows["shots"]
    ym = (y0 + y1) / 2
    for s in scene.shots:
        col = OUTCOME[s.outcome]
        dashed(img, (X(s.t_fire), ym), (X(s.t_exit), ym), _mix(FIRE, PANEL, 0.7), 1, 3, 2)
        line(img, (X(s.t_exit), ym), (X(s.t_arrive), ym), FIRE, 2)
        tri = np.array([[X(s.t_fire), y1], [X(s.t_fire) - 4, y0], [X(s.t_fire) + 4, y0]])
        cv2.fillPoly(img, [np.round(tri * _SF).astype(np.int32)], FIRE, cv2.LINE_AA, _SH)
        circle(img, (X(s.t_arrive), ym), 3.5, col, -1)
        tag = f"#{s.index} {OUTCOME_LABEL[s.outcome]}"
        (tw, _), _ = cv2.getTextSize(tag, FONT, 0.32, 1)
        if X(s.t_arrive) + 8 + tw < w - 4:
            text(img, tag, int(X(s.t_arrive)) + 7, int(ym) + 4, col, 0.32)
        else:
            text(img, tag, int(X(s.t_fire)) - 8, int(ym) + 4, col, 0.32, anchor="r")
    if scene.t_end_video > scene.duration + 0.02:  # the world runs on past the last camera frame
        x = X(scene.duration)
        dashed(img, (x, rows["armed"][0] - 3), (x, rows["shots"][1] + 1), MUTED, 1, 2, 3)
        text(img, "end of frames", int(x) + 4, rows["armed"][1], DIM, 0.3)
    # time axis
    yax = h - 3
    tick = 0.25 if scene.t_end_video <= 1.5 else 0.5
    for k in range(0, int(scene.t_end_video / tick) + 1):
        tt = k * tick
        x = X(tt)
        line(img, (x, rows["armed"][0] - 3), (x, rows["shots"][1] + 1), _mix(GRID, PANEL, 0.6), 1)
        if k % (2 if tick == 0.25 else 1) == 0:
            text(img, f"{tt:.1f}s", int(x), yax, DIM, 0.3, anchor="c")
    return img


def timeline_at(base: np.ndarray, scene: Scene, t: float) -> np.ndarray:
    img = base.copy()
    w = img.shape[1]
    x = _tl_x(scene, w, t)
    xe = int(_tl_x(scene, w, scene.t_end_video)) + 1
    fill_rect(img, int(math.ceil(x)), 4, xe, img.shape[0] - 12, PANEL, 0.62)
    line(img, (x, 3), (x, img.shape[0] - 12), TEXT, 1)
    return img


# ============================================================================================ camera + info
def camera_view(scene: Scene, idx: int, size: tuple[int, int]) -> np.ndarray:
    from tbot.app import overlay

    w, h = size
    rec = scene.rec
    if idx < 0 or rec.steps[idx].image is None:
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = BG
        text(img, "waiting for the first frame", w // 2, h // 2, DIM, 0.45, anchor="c")
        return img
    st = rec.steps[idx]
    img = overlay.draw_record(st.image, st.tele, rec.cam, rec.mount, mask=rec.mask, bore_px=rec.bore_px,
                              banner=f"t {st.t_s:.3f} s")
    for ci, (u, v) in st.truth_px:  # the TRUE clay: thin corner brackets in its colour
        col = _clay_color(ci)
        r, k = 17.0, 6.0
        for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cx_, cy_ = u + sx_ * r, v + sy_ * r
            line(img, (cx_, cy_), (cx_ - sx_ * k, cy_), col, 2)
            line(img, (cx_, cy_), (cx_, cy_ - sy_ * k), col, 2)
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _legend(img, x: int, y: int) -> None:
    items = [
        ("clay (truth)", lambda p: circle(img, p, 4.0, CLAY[0], -1)),
        ("fragments", lambda p: [circle(img, (p[0] + dx, p[1] + dy), 1.4, CLAY[0], -1)
                                 for dx, dy in ((-4, 1), (0, -3), (4, 2))]),
        ("track est.", lambda p: diamond(img, p, 4.5, TRACK, True)),
        ("intercept", lambda p: xmark(img, p, 4.0, LEAD, 2)),
        ("bore (truth)", lambda p: line(img, (p[0] - 7, p[1]), (p[0] + 7, p[1]), BORE, 1)),
        ("pred. bore", lambda p: dashed(img, (p[0] - 8, p[1]), (p[0] + 8, p[1]), BORE_PRED, 1, 4, 3)),
        ("hit", lambda p: circle(img, p, 4.5, HIT, 1)),
        ("miss", lambda p: circle(img, p, 4.5, MISS, 1)),
    ]
    for name, glyph in items:
        glyph((x + 8, y - 4))
        x += 20 + text(img, name, x + 18, y, MUTED, 0.33) + 12


def info_panel(scene: Scene, idx: int, t: float, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = PANEL
    tele = scene.rec.steps[idx].tele if idx >= 0 else {}
    dec = tele.get("decision") or {}
    armed = bool(tele.get("armed"))
    x = 10
    x += chip(img, "ARMED" if armed else "SAFE", x, 22, ARMED if armed else SAFE, 0.42, font=FONT_B, solid=armed) + 10
    text(img, "decision", x, 22, MUTED, 0.36)
    reason = str(dec.get("reason", "-"))
    rc = FIRE if dec.get("fire") or tele.get("fired") else TEXT
    text(img, reason[:44], x + 62, 22, rc, 0.42, FONT_B)
    # P(hit) gauge
    p = float(dec.get("p_hit", float("nan")))
    gx0, gx1, gy = w - 170, w - 12, 16
    fill_rect(img, gx0, gy - 6, gx1, gy + 4, PANEL_HI)
    if math.isfinite(p):
        fill_rect(img, gx0, gy - 6, int(gx0 + (gx1 - gx0) * min(max(p, 0.0), 1.0)), gy + 4, _mix(BORE_PRED, PANEL, 0.85))
    text(img, f"P(hit) {p:.2f}" if math.isfinite(p) else "P(hit)  -", gx0 - 8, 22, MUTED, 0.36, anchor="r")

    def f(v, nd=2, unit=""):
        v = float("nan") if v is None else float(v)
        return "-" if not math.isfinite(v) else f"{v:.{nd}f}{unit}"

    text(img, f"solver   miss {f(dec.get('miss_deg'))} deg   pattern {f(dec.get('pattern_deg'))} deg   "
              f"sigma {f(dec.get('sigma_deg'))} deg   range {f(dec.get('range_m'), 1, ' m')}   "
              f"tof {f(dec.get('tof_ms'), 0, ' ms')}", 10, 42, TEXT, 0.36)
    trs = tele.get("tracks") or []
    text(img, f"pipeline  detections {int(tele.get('n_detections', 0))}   tracks {len(trs)} "
              f"({sum(1 for tr in trs if tr.get('confirmed'))} confirmed)   frame age {f(tele.get('age_ms'), 0, ' ms')}"
              f"   seq {tele.get('seq', '-')}", 10, 59, TEXT, 0.36)
    y = 78
    shown = [s for s in scene.shots if s.t_fire <= t][-2:]
    if not shown:
        text(img, "no shot fired yet", 10, y, DIM, 0.36)
    for s in shown:
        col = OUTCOME[s.outcome]
        x = 10
        x += text(img, f"#{s.index}", x, y, FIRE, 0.38, FONT_B) + 8
        x += text(img, f"cmd {s.t_fire:.3f} s  ->  exit +{1e3 * (s.t_exit - s.t_fire):.0f} ms  ->  arrive "
                       f"{s.t_arrive:.3f} s", x, y, TEXT, 0.35) + 10
        if t >= s.t_arrive:
            x += chip(img, OUTCOME_LABEL[s.outcome], x, y, col, 0.36) + 8
            if math.isfinite(s.miss_m):
                text(img, f"miss {s.miss_m:.2f} m / pattern {s.radius_m:.2f} m @ {s.range_m:.1f} m", x, y, MUTED, 0.34)
        else:
            text(img, "in flight" if t >= s.t_exit else "trigger delay", x, y, FIRE, 0.36)
        y += 18
    line(img, (8, h - 22), (w - 8, h - 22), GRID, 1)
    _legend(img, 6, h - 7)
    return img


def _result_line(scene: Scene) -> tuple[str, tuple[int, int, int]]:
    r = scene.result
    thrown, broken, fires = int(r.get("clays_thrown", 0)), int(r.get("clays_broken", 0)), int(r.get("fires", 0))
    if thrown == 0:
        return ("NO FALSE FIRE" if fires == 0 else f"FALSE FIRE x{fires}"), (HIT if fires == 0 else MISS)
    if fires == 0:
        return f"NO SHOT  0/{thrown}", SAFE
    col = HIT if broken == thrown else (DEBRIS if broken else MISS)
    return f"{broken}/{thrown} BROKEN  ({fires} shot{'s' if fires != 1 else ''})", col


def header(scene: Scene, w: int, h: int, t: float | None, slowmo: float | None, final: bool) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    y = h // 2 + 7
    x = 12
    x += text(img, "TBOT", x, y, TEXT, 0.62, FONT_B, 1) + 6
    x += text(img, "SIM", x, y, CLAY[0], 0.62, FONT_B, 1) + 18
    x += text(img, scene.name, x, y, TEXT, 0.55, FONT_B) + 10
    x += text(img, f"seed {scene.seed}", x, y, MUTED, 0.42) + 14
    desc = scene.description
    if len(desc) > 70:
        desc = desc[:67] + "..."
    text(img, desc, x, y, DIM, 0.36)
    xr = w - 12
    if final or t is None:
        s, col = _result_line(scene)
        chip(img, s, xr, y, col, 0.45, anchor="r", font=FONT_B, solid=True)
    else:
        if slowmo:
            xr -= text(img, f"x{slowmo:g} speed", xr, y, DIM, 0.36, anchor="r") + 12
        text(img, f"t {t:6.3f} s", xr, y, TEXT, 0.52, FONT_B, anchor="r")
    line(img, (0, h - 1), (w, h - 1), GRID, 1)
    return img


# ============================================================================================ composition
VIDEO_W, VIDEO_H = 1280, 720
_M, _G = 8, 6


def _layout(cam_w: int, cam_h: int):
    """Panel rectangles (x, y, w, h) of the video frame."""
    head = (0, 0, VIDEO_W, 40)
    cam = (_M, 44, cam_w, cam_h + 18)
    info = (_M, cam[1] + cam[3] + _G, cam_w, 596 - (cam[1] + cam[3] + _G))
    rx = _M + cam_w + _G
    rw = VIDEO_W - _M - rx
    plan = (rx, 44, rw, 338)
    ew = int(round(rw * 0.61))
    elev = (rx, 44 + 338 + _G, ew, 596 - (44 + 338 + _G))
    close = (rx + ew + _G, elev[1], rw - ew - _G, elev[3])
    tl = (_M, 602, VIDEO_W - 2 * _M, VIDEO_H - 602 - _M)
    return head, cam, info, plan, elev, close, tl


class _Renderer:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        rec = scene.rec
        cw, ch = int(rec.cam.width), int(rec.cam.height)
        if cw > 640 or ch > 400:  # keep the right column: a bigger camera is shown scaled to 640 wide
            f = min(640.0 / cw, 400.0 / ch)
            cw, ch = int(cw * f), int(ch * f)
        self.cam_size = (cw, ch)
        self.L = _layout(cw, ch)
        _, _, _, plan, elev, _, tl = self.L
        self.plan_ax = plan_axes(scene, plan[2], plan[3])
        self.elev_ax = elev_axes(scene, elev[2], elev[3])
        self.tl_base = timeline_base(scene, tl[2], tl[3])
        self._cam_cache: dict[int, np.ndarray] = {}

    def _cam(self, idx: int) -> np.ndarray:
        if idx not in self._cam_cache:
            self._cam_cache = {idx: camera_view(self.scene, idx, self.cam_size)}
        return self._cam_cache[idx]

    def frame(self, t: float, slowmo: float | None, final: bool = False) -> np.ndarray:
        sc = self.scene
        head, cam, info, plan, elev, close, tl = self.L
        out = np.empty((VIDEO_H, VIDEO_W, 3), np.uint8)
        out[:] = BG

        def put(rect, img):
            x, y, w, h = rect
            out[y:y + h, x:x + w] = img[:h, :w]

        idx = sc.step_at(t)
        put(head, header(sc, head[2], head[3], t, slowmo, final))
        cp = np.empty((cam[3], cam[2], 3), np.uint8)
        cp[:] = PANEL
        panel_title(cp, "CAMERA", "pipeline view + overlay - brackets: true clay")
        cp[18:18 + self.cam_size[1], :self.cam_size[0]] = self._cam(idx)
        if t > sc.duration + 1e-6:  # the world view runs on; the camera has no more frames
            roi = cp[18:18 + self.cam_size[1], :self.cam_size[0]]
            roi[:] = (roi.astype(np.float32) * 0.45).astype(np.uint8)
            label(roi, "end of the scenario's frames", self.cam_size[0] // 2, self.cam_size[1] // 2, TEXT, BG, 0.5,
                  0.85, anchor="c", pad=8)
        put(cam, cp)
        put(info, info_panel(sc, idx, t, (info[2], info[3])))
        for rect, ax, view in ((plan, self.plan_ax, "plan"), (elev, self.elev_ax, "elev")):
            p = np.empty((rect[3], rect[2], 3), np.uint8)
            draw_world(p, sc, ax, view, t)
            if final and view == "plan":
                s, col = _result_line(sc)
                (tw, _), _ = cv2.getTextSize(s, FONT_B, 0.8, 1)
                bx, by = (rect[2] - tw) // 2, 58
                fill_rect(p, bx - 16, by - 26, bx + tw + 16, by + 12, BG, 0.8)
                cv2.rectangle(p, (bx - 16, by - 26), (bx + tw + 15, by + 11), col, 1, cv2.LINE_AA)
                text(p, s, bx, by, col, 0.8, FONT_B)
            put(rect, p)
        cl = np.empty((close[3], close[2], 3), np.uint8)
        draw_closeup(cl, sc, t)
        put(close, cl)
        tlp = timeline_at(self.tl_base, sc, t)
        put(tl, tlp)
        return out


class _Video:
    def __init__(self, stem: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = None
        for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
            p = stem.with_suffix(ext)
            wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
            if wr.isOpened():
                self.w, self.path = wr, p
                return
            wr.release()
        raise RuntimeError("OpenCV can write neither mp4v nor MJPG video here")

    def write(self, img: np.ndarray) -> None:
        self.w.write(img)

    def close(self) -> Path:
        self.w.release()
        return self.path


def render_video(scene: Scene, stem: Path, fps: float = 30.0, slowmo: float = 0.25, hold_s: float = 1.5,
                 max_video_frames: int | None = None) -> Path:
    """Slow-motion video of the run: sim time advances slowmo / fps per video frame."""
    r = _Renderer(scene)
    dt = float(slowmo) / float(fps)
    n = int(math.floor(scene.t_end_video / dt)) + 1
    n_hold = int(round(hold_s * fps))
    if max_video_frames is not None:
        n = min(n, int(max_video_frames))
        n_hold = min(n_hold, max(int(max_video_frames) - n, 0))
    vid = _Video(stem, fps, (VIDEO_W, VIDEO_H))
    try:
        for k in range(n):
            vid.write(r.frame(k * dt, slowmo))
        if n_hold:
            last = r.frame(min((n - 1) * dt, scene.t_end_video), slowmo, final=True)
            for _ in range(n_hold):
                vid.write(last)
    finally:
        path = vid.close()
    return path


def video_frame(scene: Scene, t: float, slowmo: float | None = None, final: bool = False) -> np.ndarray:
    """One video frame at sim time t (for stills and tests)."""
    return _Renderer(scene).frame(t, slowmo, final)


# ============================================================================================ summary still
SUM_W, SUM_H = 1280, 720


def reason_kind(reason: str) -> str:
    """A solver verdict without its numbers ('p_hit 0.10 < 0.21' -> 'P(hit) below the bar')."""
    s = re.sub(r"[-+]?\d+(\.\d+)?", "#", str(reason)).strip()
    if s.startswith("p_hit") and "<" in s:
        return "P(hit) below the bar"
    return s or "-"


def _reasons_chart(img: np.ndarray, reasons: dict[str, int]) -> None:
    img[:] = PANEL
    panel_title(img, "DECISIONS", "frames per solver verdict")
    grouped: dict[str, int] = {}
    for k, n in reasons.items():
        grouped[reason_kind(k)] = grouped.get(reason_kind(k), 0) + int(n)
    reasons = grouped
    items = sorted(reasons.items(), key=lambda kv: -kv[1])[:7]
    if not items:
        return
    total = max(sum(reasons.values()), 1)
    top = items[0][1]
    h, w = img.shape[:2]
    y = 40
    bw = w - 190 - 50
    for name, n in items:
        text(img, name[:28], 180, y + 4, TEXT, 0.35, anchor="r")
        L = int(bw * n / top)
        col = FIRE if name.lower() == "fire" else _mix(BORE_PRED, PANEL, 0.6)
        fill_rect(img, 190, y - 6, 190 + max(L, 1), y + 5, col)
        text(img, f"{n}  ({100.0 * n / total:.0f}%)", 196 + L, y + 4, MUTED, 0.32)
        y += int(min(24, (h - 50) / max(len(items), 1)))


def _stats_table(img: np.ndarray, scene: Scene) -> None:
    img[:] = PANEL
    panel_title(img, "RUN", "as `tbot sim` scores it")
    r = scene.result

    def f(v, nd=2, unit=""):
        v = float("nan") if v is None else float(v)
        return "-" if not math.isfinite(v) else f"{v:.{nd}f}{unit}"

    rows = [
        ("frames processed", f"{r.get('frames', 0)}   (clay visible in {r.get('visible_frames', 0)})"),
        ("detection rate", f(100.0 * float(r.get('detection_rate', float('nan'))), 1, " %")),
        ("track latency", f(r.get("track_latency_ms"), 0, " ms")),
        ("lead error p50 / p95", f"{f(r['lead_error_deg']['p50'])} / {f(r['lead_error_deg']['p95'])} deg"),
        ("shots / hits", f"{r.get('fires', 0)} / {r.get('hits', 0)}"),
        ("clays broken", f"{r.get('clays_broken', 0)} / {r.get('clays_thrown', 0)}"),
        ("miss p50", f(r["miss_m"]["p50"], 2, " m")),
        ("debris shots", f"{r.get('debris_shots', 0)}"),
    ]
    y = 42
    for k, v in rows:
        text(img, k, 14, y, MUTED, 0.36)
        text(img, v, 200, y, TEXT, 0.38)
        y += 20


def render_summary(scene: Scene, path: Path) -> Path:
    out = np.empty((SUM_H, SUM_W, 3), np.uint8)
    out[:] = BG
    hd = header(scene, SUM_W, 44, None, None, True)
    out[0:44] = hd

    def panel(x, y, w, h, fn):
        p = np.empty((h, w, 3), np.uint8)
        fn(p)
        out[y:y + h, x:x + w] = p

    lw = 784
    pa = plan_axes(scene, lw, 448)
    panel(_M, 48, lw, 448, lambda p: draw_world(p, scene, pa, "plan", scene.t_end_video, summary=True))
    ea = elev_axes(scene, lw, 720 - 48 - 448 - _G - _M)
    panel(_M, 48 + 448 + _G, lw, 720 - 48 - 448 - _G - _M,
          lambda p: draw_world(p, scene, ea, "elev", scene.t_end_video, summary=True))
    rx = _M + lw + _G
    rw = SUM_W - _M - rx
    ch = 236
    shots = scene.shots[:2]
    if not shots:
        panel(rx, 48, rw, ch, lambda p: draw_closeup(p, scene, scene.t_end_video, summary_best=True))
    elif len(shots) == 1:
        panel(rx, 48, rw, ch, lambda p: draw_closeup(p, scene, scene.t_end_video, summary_shot=shots[0]))
    else:
        cw = (rw - _G) // 2
        panel(rx, 48, cw, ch, lambda p: draw_closeup(p, scene, scene.t_end_video, summary_shot=shots[0]))
        panel(rx + cw + _G, 48, rw - cw - _G, ch,
              lambda p: draw_closeup(p, scene, scene.t_end_video, summary_shot=shots[1]))
    ry = 48 + ch + _G
    panel(rx, ry, rw, 496 - ry, lambda p: _reasons_chart(p, dict(scene.result.get("reasons") or {})))
    panel(rx, 496 + _G, rw, 720 - 496 - _G - _M, lambda p: _stats_table(p, scene))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise RuntimeError(f"could not write {path}")
    return path


def contact_sheet(paths: Sequence[Path], path: Path, cols: int = 2, scale: float = 0.5) -> Path:
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise ValueError("contact sheet: no images")
    tw, th = int(imgs[0].shape[1] * scale), int(imgs[0].shape[0] * scale)
    cols = max(1, min(cols, len(imgs)))
    rows = int(math.ceil(len(imgs) / cols))
    head = 56
    W = cols * tw + (cols + 1) * _M
    H = head + rows * th + (rows + 1) * _M
    out = np.empty((H, W, 3), np.uint8)
    out[:] = BG
    x = 14
    x += text(out, "TBOT", x, 36, TEXT, 0.7, FONT_B) + 6
    x += text(out, "SIM", x, 36, CLAY[0], 0.7, FONT_B) + 16
    text(out, f"closed-loop scenarios - {len(imgs)} run{'s' if len(imgs) != 1 else ''}", x, 36, MUTED, 0.5)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        x0 = _M + c * (tw + _M)
        y0 = head + _M + r * (th + _M)
        out[y0:y0 + th, x0:x0 + tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), GRID, 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
    return path


def render_outputs(rec: Recorder, result: dict, out_dir: Path, *, video: bool = True, fps: float = 30.0,
                   slowmo: float = 0.25, hold_s: float = 1.5, max_video_frames: int | None = None) -> dict:
    """Summary PNG (and the video) of one recorded run -> {"png": path, "video": path or None}."""
    scene = build_scene(rec, result)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{scene.name}_seed{scene.seed}"
    png = render_summary(scene, stem.with_suffix(".png"))
    vid = render_video(scene, stem, fps, slowmo, hold_s, max_video_frames) if video else None
    return {"png": png, "video": vid}


# ============================================================================================ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbot simviz",
        description="graphics of closed-loop simulator runs: slow-motion video, summary still, contact sheet",
    )
    add_common_args(parser)
    parser.add_argument("--scenarios", nargs="+", default=None, metavar="NAME",
                        help=f"scenarios (default: all). available: {', '.join(SCENARIOS)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1], metavar="N", help="seed(s) to run (default 1)")
    parser.add_argument("--out", default=DEFAULT_OUT, metavar="DIR", help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--no-video", action="store_true", help="summary stills only (fast)")
    parser.add_argument("--fps", type=float, default=30.0, help="video frame rate (default 30)")
    parser.add_argument("--slowmo", type=float, default=0.25, help="sim seconds per video second (default 0.25)")
    parser.add_argument("--hold", type=float, default=1.5, metavar="S", help="final-frame hold, video seconds")
    parser.add_argument("--max-frames", type=int, default=None, metavar="N", help="cap the frames per run")
    parser.add_argument("--actuator-delay-ms", type=float, default=None, help="TRUE actuator delay (as tbot sim)")
    parser.add_argument("--not-armed", action="store_true", help="never press the arming button")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg, calib = load_from_args(args)
    names = list(args.scenarios) if args.scenarios else list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s) {unknown}; available: {', '.join(SCENARIOS)}")
    if args.slowmo <= 0 or args.fps <= 0:
        parser.error("--fps and --slowmo must be > 0")
    out_dir = resolve_path(args.out)
    pngs = []
    for name in names:
        for seed in args.seeds:
            rec, r = record_run(cfg, calib, name, int(seed), actuator_delay_ms=args.actuator_delay_ms,
                                armed=not args.not_armed, max_frames=args.max_frames)
            paths = render_outputs(rec, r, out_dir, video=not args.no_video, fps=args.fps, slowmo=args.slowmo,
                                   hold_s=args.hold)
            pngs.append(paths["png"])
            extra = f", {paths['video']}" if paths["video"] else ""
            print(f"{name:<18} seed {seed}: {r['fires']} fire(s), {r['clays_broken']}/{r['clays_thrown']} broken"
                  f" -> {paths['png']}{extra}", flush=True)
    if len(pngs) > 1:
        print(f"contact sheet -> {contact_sheet(pngs, out_dir / 'contact_sheet.png')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
