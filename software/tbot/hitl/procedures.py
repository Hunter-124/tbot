"""HITL test procedures (generators driven by `Rig.drive`). Raw evidence only; `analysis.py` computes results.

Every procedure appends its raw record to `bench.raw[<name>]` so a run can be re-analysed offline
(`tbot hitl analyze`). Order: geometry (screen map, needed by all others) -> display (photon timing of the
monitor) -> flash (static target, stage-by-stage latency) / sweep (moving target, lead accuracy) / detect.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from tbot.hitl.audio import detect_onset, write_wav
from tbot.hitl.geometry import ScreenMap
from tbot.hitl.rig import Rig
from tbot.hitl.scene import Disc, HitlStore, Rect

#: Geometry markers aim for this detected diameter (camera px): big enough to centroid well, far below max_area_px.
GEOMETRY_DIAMETER_PX = 6.0
#: A detection within this many camera px of a static artefact (seen with nothing drawn) is ignored.
ARTEFACT_RADIUS_PX = 3.0
#: Inset (screen px) of the detector's screen mask: keeps the screen bezel edge out of the search area.
SCREEN_MASK_INSET_PX = 12.0
#: Random lead time before a target appears: decorrelates the target from the camera/display frame phase.
SCHEDULE_MS = (300.0, 900.0)
#: How long a flash target stays up without a decision before the trial is a timeout.
FLASH_HOLD_MS = 1500.0
#: Audio kept before the fire command (noise floor) and after it (click search window margin).
MIC_PRE_MS = 150.0
MIC_POST_MS = 150.0


@dataclass
class Bench:
    rig: Rig
    store: HitlStore
    cfg: object
    cam: object
    mount: object
    live: bool
    bg: int
    level: int
    out_dir: Path
    rng: np.random.Generator
    log: Callable[[str], None] = print
    operator: object | None = None
    mic: object | None = None
    sense: object | None = None
    actuator: object | None = None
    smap: ScreenMap | None = None
    allow_windowed: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def clay_radius_m(self) -> float:
        return 0.5 * float(self.cfg.mount.clay_diameter_m)

    def bore_screen(self, range_m: float) -> np.ndarray:
        uv = np.asarray(self.mount.bore_pixel(self.cam, float(range_m)), dtype=np.float64)
        return self.smap.from_pixel(uv)[0]


def _near(u: float, v: float, pts, radius: float) -> bool:
    return any((u - a) ** 2 + (v - b) ** 2 <= radius * radius for a, b in pts)


def wait_page(b: Bench, timeout_s: float = 600.0):
    """Until the page is open, synced, posting its frame log and (unless allow_windowed) full screen:
    the start panel must not be in the camera's view while it measures."""
    def ready() -> bool:
        s = b.store
        return (s.connected() and s.screen_size() is not None
                and (b.allow_windowed or bool(s.page_status.get("fullscreen", False))))

    if not ready():
        b.log("  waiting for the target page: open the URL above on the monitor and click Start (full screen) ...")
    ok = yield from b.rig.wait_for(ready, timeout_s)
    if not ok:
        raise RuntimeError("the target page never connected in full screen (or pass --allow-windowed)")
    if not b.store.page_status.get("fullscreen", False):
        b.log("  WARNING: the page is not full screen (--allow-windowed)")


def _settle_ns(b: Bench, t_draw_ms: float, display_ms: float = 120.0) -> int:
    return b.store.ns(t_draw_ms) + int(display_ms * 1e6) + b.rig.period_ns


# ----------------------------------------------------------------------------------------- geometry
def _show_and_measure(b: Bench, xy, r: float, artefacts, frames: int = 6):
    """Draw a static disc, return the median detection {u, v, d, n} of its blob (None if unseen)."""
    store, rig = b.store, b.rig
    item = Disc(id=store.new_id(), t0=store.rig_ms() + 30.0, t1=None, x0=float(xy[0]), y0=float(xy[1]),
                r=float(r), level=b.level)
    store.add(item)
    t_draw = yield from rig.wait_drawn(item.id, 3.0)
    if t_draw is None:
        store.end(item.id)
        raise RuntimeError("the page is not drawing targets (is the tab visible and fullscreen?)")
    yield from rig.wait_until_ns(_settle_ns(b, t_draw))
    sign = 1.0 if b.level > b.bg else -1.0
    hits = []
    for _ in range(frames):
        yield
        cands = [d for d in rig.last["dets"]
                 if d[3] * sign > 0 and not _near(d[0], d[1], artefacts, ARTEFACT_RADIUS_PX)]
        if cands:
            hits.append(max(cands, key=lambda d: d[4]))
    store.end(item.id)
    yield from rig.wait_s(0.08)
    if len(hits) < max(2, frames // 2):
        return None
    h = np.asarray(hits, dtype=np.float64)
    return {"u": float(np.median(h[:, 0])), "v": float(np.median(h[:, 1])), "d": float(np.median(h[:, 2])),
            "n": len(hits)}


def geometry(b: Bench, cols: int = 5, rows: int = 4, margin: float = 0.15):
    """Flash a grid of discs one at a time and fit the screen -> camera map."""
    store, rig = b.store, b.rig
    yield from wait_page(b)
    W, H = b.store.screen_size()
    store.set_status(False)
    store.set_background(b.bg)
    store.clear()
    rig.set_screen_mask(None)
    yield from rig.wait_s(0.4)
    artefacts = []
    for _ in range(int(0.3 / max(rig.pipeline.frame_period_s, 1e-3)) + 3):
        yield
        artefacts += [(d[0], d[1]) for d in rig.last["dets"]]
    if artefacts:
        b.log(f"  {len(artefacts)} static detection(s) with nothing drawn: ignored within {ARTEFACT_RADIUS_PX:g} px")

    r = 0.012 * H
    probe = None
    for _ in range(4):
        probe = yield from _show_and_measure(b, (W / 2, H / 2), r, artefacts)
        if probe is not None:
            break
        r *= 2.0
    if probe is None:
        raise RuntimeError("no detection of a disc at the screen centre: aim the camera at the monitor, "
                           "check focus/exposure (tbot focus) and the background/target levels")
    r = float(np.clip(r * GEOMETRY_DIAMETER_PX / max(probe["d"], 1.0), 3.0, 0.06 * H))
    b.log(f"  marker radius {r:.1f} screen px (centre probe: {probe['d']:.1f} px blob at "
          f"({probe['u']:.1f}, {probe['v']:.1f}))")

    xs = np.linspace(margin * W, (1 - margin) * W, int(cols))
    ys = np.linspace(margin * H, (1 - margin) * H, int(rows))
    points = []
    for j, y in enumerate(ys):
        for x in (xs if j % 2 == 0 else xs[::-1]):
            obs = yield from _show_and_measure(b, (x, y), r, artefacts)
            points.append({"xy": [float(x), float(y)], "obs": obs})
            b.log(f"  marker ({x:6.0f}, {y:5.0f}) -> " + ("not seen" if obs is None else
                  f"({obs['u']:7.2f}, {obs['v']:7.2f}) d {obs['d']:.1f} px"))
    seen = [p for p in points if p["obs"] is not None]
    if len(seen) < 6:
        raise RuntimeError(f"only {len(seen)} of {len(points)} markers seen: the camera must see most of the screen")
    xy = np.array([p["xy"] for p in seen])
    uv = np.array([[p["obs"]["u"], p["obs"]["v"]] for p in seen])
    smap = ScreenMap.fit(xy, uv, b.cam, W, H)
    dropped = []
    while smap.max_px > max(3.0 * smap.rms_px, 1.0) and len(xy) > 6:
        worst = int(np.argmax(np.linalg.norm(smap.residuals(xy, uv), axis=1)))
        dropped.append(xy[worst].tolist())
        xy, uv = np.delete(xy, worst, axis=0), np.delete(uv, worst, axis=0)
        smap = ScreenMap.fit(xy, uv, b.cam, W, H)
    b.smap = smap
    rig.set_screen_mask(smap.camera_mask(b.cam.width, b.cam.height, SCREEN_MASK_INSET_PX))
    b.raw["geometry"] = {"points": points, "marker_r": r, "artefacts": artefacts, "dropped": dropped,
                         "map": smap.to_dict()}
    b.log(f"  screen map: {smap.n_points} markers, rms {smap.rms_px:.2f} px, max {smap.max_px:.2f} px"
          + (f", {len(dropped)} outlier(s) dropped" if dropped else ""))


def use_geometry(b: Bench, smap: ScreenMap):
    """Reuse a saved screen map (calib/hitl.json) after checking the page still has the same size."""
    yield from wait_page(b)
    size = b.store.screen_size()
    if size != (smap.screen_w, smap.screen_h):
        raise RuntimeError(f"saved screen map is for {smap.screen_w}x{smap.screen_h}, the page is {size}: rerun geometry")
    b.store.set_status(False)
    b.store.set_background(b.bg)
    b.store.clear()
    b.smap = smap
    b.rig.set_screen_mask(smap.camera_mask(b.cam.width, b.cam.height, SCREEN_MASK_INSET_PX))
    b.raw["geometry"] = {"reused": True, "map": smap.to_dict()}


def check_bore_on_screen(b: Bench, range_m: float) -> np.ndarray:
    xy = b.bore_screen(range_m)
    if not b.smap.visible(xy, margin_px=20):
        raise RuntimeError(f"the bore line at {range_m:g} m lands off the screen at ({xy[0]:.0f}, {xy[1]:.0f}) "
                           f"of {b.smap.screen_w}x{b.smap.screen_h}: aim the gun at the monitor")
    return xy


# ------------------------------------------------------------------------------------------ display
def display(b: Bench, trials: int = 30, range_m: float = 20.0, exposure_us: float | None = None):
    """Photon timing of the monitor: square patches at three rows, bg -> target level and back.

    `exposure_us` (sources with set_exposure_us) lengthens the exposure for this test only, so most
    transitions land inside an exposure and are timed to a fraction of it instead of a frame bracket."""
    store, rig, smap = b.store, b.rig, b.smap
    W, H = smap.screen_w, smap.screen_h
    bore = check_bore_on_screen(b, range_m)
    x = float(np.clip(bore[0], 0.1 * W, 0.9 * W))
    ys = sorted({round(0.12 * H), round(float(np.clip(bore[1], 0.12 * H, 0.88 * H))), round(0.88 * H)})
    side = 0.06 * H
    ifov = float(b.cam.ifov(np.array([[b.cam.cx, b.cam.cy]]))[0])
    side_cam = side * float(smap.px_angle([[x, H / 2]])[0]) / ifov
    if side_cam < 8.0:
        side *= 8.0 / side_cam
        side_cam = 8.0
    rows = []
    for j, y in enumerate(ys):
        u, v = smap.to_pixel([[x, y]])[0]
        half = max(1, int(0.3 * side_cam))
        roi = (int(round(u)) - half, int(round(v)) - half, 2 * half + 1, 2 * half + 1)
        if roi[0] < 0 or roi[1] < 0 or roi[0] + roi[2] > b.cam.width or roi[1] + roi[3] > b.cam.height:
            continue
        rows.append({"name": f"row{j}", "y": float(y), "roi": roi})
    if not rows:
        raise RuntimeError("no display-test patch lands inside the camera image")
    rig.probes = {r["name"]: r["roi"] for r in rows}
    store.clear()
    if exposure_us:
        yield from _set_exposure(b, exposure_us)
    yield from rig.wait_s(0.3)
    recs = []
    hold_ms = 250.0
    for k in range(int(trials)):
        row = rows[int(b.rng.integers(0, len(rows)))]
        t0 = store.rig_ms() + float(b.rng.uniform(150.0, 400.0))
        item = Rect(id=store.new_id(), t0=t0, t1=t0 + hold_ms, x=x - side / 2, y=row["y"] - side / 2,
                    w=side, h=side, level=b.level)
        i0 = len(rig.frames)
        store.add(item)
        yield from rig.wait_until_ns(store.ns(t0 + hold_ms) + 200_000_000)
        recs.append({"trial": k, "item": item.id, "row": row["name"], "y": row["y"], "t0": t0, "t1": t0 + hold_ms,
                     "frames": [i0, len(rig.frames)]})
        yield from rig.wait_s(float(b.rng.uniform(0.02, 0.12)))
    rig.probes = {}
    store.clear()
    if exposure_us:
        yield from _set_exposure(b, None)
    b.raw["display"] = {"rows": rows, "side_px": side, "x": x, "hold_ms": hold_ms, "trials": recs,
                        "exposure_us": exposure_us}
    b.log(f"  display: {len(recs)} patches on rows {[round(r['y']) for r in rows]}")


def _set_exposure(b: Bench, exposure_us: float | None):
    """Switch the camera exposure (None = back to camera.exposure_us) and wait until frames report it."""
    src = b.rig.pipeline.source
    want = float(b.cfg.camera.exposure_us if exposure_us is None else exposure_us)
    src.set_exposure_us(want)
    ok = yield from b.rig.wait_for(lambda: abs(b.rig.last["exp_us"] - want) <= max(20.0, 0.02 * want), 3.0)
    if not ok:
        b.log(f"  WARNING: the camera still reports {b.rig.last['exp_us']:.0f} us after asking for {want:.0f} us")
    yield from b.rig.wait_frames(3)


def supports_exposure(b: Bench) -> bool:
    return hasattr(b.rig.pipeline.source, "set_exposure_us")


# --------------------------------------------------------------------------------- fire-and-listen
def _click_window_ms(b: Bench) -> float:
    kind = str(b.cfg.actuator.type)
    if kind == "servo":
        return float(b.cfg.actuator.servo.hold_ms) + 400.0
    if kind == "pq12":
        return float(b.cfg.actuator.pq12.extend_ms) + 500.0
    return 4.0 * float(b.cfg.actuator.sim.delay_ms) + 300.0


def _decision_frame(b: Bench, i0: int, t_min_ns: int) -> int | None:
    """First frame since i0 (exposed after t_min) on which the pipeline fired (live) or would have (dry)."""
    for rec in b.rig.frames[i0:]:
        if rec["te"] < t_min_ns:
            continue
        if (b.live and rec["fired"]) or (not b.live and (rec["fired"] or rec["dec"]["would"])):
            return rec["i"]
    return None


def _fire_trial(b: Bench, item: Disc, i0: int, n_calls0: int, name: str, timeout_ms: float):
    """Wait for the decision, then for the break edge / click. Returns the timing part of the record."""
    store, rig = b.store, b.rig
    deadline_ns = store.ns(item.t0) + int(timeout_ms * 1e6)
    t_min = store.ns(item.t0)
    while _decision_frame(b, i0, t_min) is None and time.monotonic_ns() < deadline_ns:
        yield
    i_dec = _decision_frame(b, i0, t_min)
    out = {"decision_frame": i_dec, "t_cmd_ns": None, "cmd_source": None, "break_ns": None, "mic": None}
    if i_dec is None:
        return out
    rec = rig.frames[i_dec]
    calls = getattr(b.actuator, "calls", [])
    if b.live and len(calls) > n_calls0 and calls[-1][2]:
        out["t_cmd_ns"], out["cmd_source"] = int(calls[-1][0]), "actuator.fire() call"
    else:
        out["t_cmd_ns"] = int(rec["now"] + rec["stage"]["solve"] * 1e6)
        out["cmd_source"] = "decision (dry run)" if not b.live else "decision (fire refused)"
    out["R_wg"] = np.asarray(rig.pipeline.attitude.R_wg(int(rec["now"])), dtype=np.float64).tolist()
    if not b.live:
        return out
    t_cmd = out["t_cmd_ns"]
    window_ns = int(_click_window_ms(b) * 1e6)
    yield from rig.wait_until_ns(t_cmd + window_ns + int(MIC_POST_MS * 1e6))
    if b.sense is not None:
        out["break_ns"] = b.sense.wait_closed(0.0)
    if b.mic is not None:
        clip = b.mic.clip(t_cmd - int(MIC_PRE_MS * 1e6), t_cmd + window_ns)
        if clip is not None:
            samples, t_first, fs = clip
            onset = detect_onset(samples, fs, t_first, search_from_ns=t_cmd, search_to_ns=t_cmd + window_ns,
                                 noise_to_ns=t_cmd)
            wav = b.out_dir / "audio" / f"{name}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            write_wav(wav, samples, int(round(fs)))
            out["mic"] = {"onset": onset, "t_first_ns": int(t_first), "fs": float(fs), "wav": str(wav.name)}
    return out


def _arm_for_trial(b: Bench, what: str):
    if not b.live:
        return
    arming = b.rig.arming
    if b.sense is not None:
        b.sense.arm_trial()
    if arming.armed and arming.shots_this_press < int(b.cfg.fire.max_shots_per_arm):
        return
    if arming.armed:
        yield from b.operator.release(b.rig)
    yield from b.operator.press(b.rig, what)
    if b.sense is not None:
        b.sense.arm_trial()


def _after_trial(b: Bench):
    if b.live and b.rig.arming.armed and b.rig.arming.shots_this_press >= int(b.cfg.fire.max_shots_per_arm):
        yield from b.operator.release(b.rig)


def flash(b: Bench, trials: int = 20, range_m: float = 20.0):
    """Static clay-sized disc at the bore point, random onset: photon -> detection -> decision -> break."""
    store, rig = b.store, b.rig
    xy = check_bore_on_screen(b, range_m)
    r = b.smap.radius_for(xy, b.clay_radius_m / range_m)
    recs = []
    for k in range(int(trials)):
        yield from _arm_for_trial(b, f"(flash {k + 1}/{trials})")
        t0 = store.rig_ms() + float(b.rng.uniform(*SCHEDULE_MS))
        item = Disc(id=store.new_id(), t0=t0, t1=t0 + FLASH_HOLD_MS, x0=float(xy[0]), y0=float(xy[1]),
                    r=r, level=b.level)
        i0 = len(rig.frames)
        n_calls0 = len(getattr(b.actuator, "calls", []))
        store.add(item)
        res = yield from _fire_trial(b, item, i0, n_calls0, f"flash_{k:02d}", FLASH_HOLD_MS + 300.0)
        store.end(item.id)
        yield from rig.wait_s(0.15)
        yield from _after_trial(b)
        rec = {"trial": k, "item": item.id, "xy": [float(xy[0]), float(xy[1])], "r": r, "t0": t0,
               "frames": [i0, len(rig.frames)], **res}
        recs.append(rec)
        b.log(f"  flash {k + 1:2d}/{trials}: " + _trial_line(b, rec))
        store.prune(store.rig_ms() - 5000.0)
        yield from rig.wait_s(float(b.rng.uniform(0.1, 0.3)))
    b.raw["flash"] = {"range_m": range_m, "trials": recs}


def _trial_line(b: Bench, rec: dict) -> str:
    if rec["decision_frame"] is None:
        return "no fire decision (timeout)"
    t_draw = b.store.first_drawn(rec["item"])
    parts = []
    if t_draw is not None:
        parts.append(f"draw->decision {(b.rig.frames[rec['decision_frame']]['now'] - b.store.ns(t_draw)) / 1e6:6.1f} ms")
    if rec.get("break_ns") is not None:
        parts.append(f"cmd->break {(rec['break_ns'] - rec['t_cmd_ns']) / 1e6:6.1f} ms")
    mic = rec.get("mic")
    if mic is not None:
        on = mic.get("onset")
        parts.append("no click heard" if on is None else
                     f"cmd->click {(on['t_ns'] - rec['t_cmd_ns']) / 1e6:6.1f} ms (snr {on['snr']:.0f})")
    return "  ".join(parts)


# -------------------------------------------------------------------------------------------- sweep
def sweep(b: Bench, trials: int = 20, speeds_dps: tuple[float, ...] = (30.0, 60.0), range_m: float = 20.0,
          directions: str = "auto"):
    """Clay crossing the bore point at a known angular speed: does the shot land on it?"""
    store, rig, smap = b.store, b.rig, b.smap
    xy = check_bore_on_screen(b, range_m)
    W = smap.screen_w
    rad_per_px = float(smap.px_angle([xy])[0])
    r = smap.radius_for(xy, b.clay_radius_m / range_m)
    margin = r + 4.0
    available = {direction: abs(float(xy[0]) - (margin if direction > 0 else W - margin))
                 * math.degrees(rad_per_px) for direction in (1.0, -1.0)}
    # Tracker uncertainty must shrink before P(hit) passes: measured ~23 frames on the sim bench.
    required_s = (rig.pipeline.budget.fire_delay_s(str(b.cfg.actuator.type))
                  + rig.pipeline.shot.time_of_flight(range_m)
                  + (int(b.cfg.fire.min_track_hits) + 20) * rig.pipeline.frame_period_s)
    if directions not in ("auto", "both", "ltr", "rtl"):
        raise ValueError(f"invalid sweep directions: {directions}")
    choices = {}
    for speed in speeds_dps:
        need = float(speed) * required_s
        if directions == "auto":
            selected = tuple(d for d in (1.0, -1.0) if available[d] >= need)
            if not selected:
                selected = (max(available, key=available.get),)
        else:
            selected = {"both": (1.0, -1.0), "ltr": (1.0,), "rtl": (-1.0,)}[directions]
        choices[float(speed)] = selected
        names = ", ".join("L->R" if d > 0 else "R->L" for d in selected)
        have = ", ".join(f"{'L->R' if d > 0 else 'R->L'} {available[d]:.2f} deg" for d in (1.0, -1.0))
        b.log(f"  sweep {speed:g} deg/s: {names} ({directions}; need {need:.2f} deg, have {have})")
        for direction in selected:
            if available[direction] < need:
                b.log(f"  WARNING: sweep {speed:g} deg/s {'L->R' if direction > 0 else 'R->L'} runway "
                      f"too short: need {need:.2f} deg, have {available[direction]:.2f} deg; move the monitor "
                      "closer / aim the bore toward the far side of the screen / lower --speeds-dps")
    recs = []
    for k in range(int(trials)):
        speed = float(speeds_dps[k % len(speeds_dps)])
        selected = choices[speed]
        direction = selected[(k // len(speeds_dps)) % len(selected)]
        v = direction * math.radians(speed) / rad_per_px / 1000.0  # px/ms
        x0 = margin if direction > 0 else W - margin
        duration = (W - 2 * margin) / abs(v)
        yield from _arm_for_trial(b, f"(sweep {k + 1}/{trials})")
        t0 = store.rig_ms() + float(b.rng.uniform(*SCHEDULE_MS))
        item = Disc(id=store.new_id(), t0=t0, t1=t0 + duration, x0=x0, y0=float(xy[1]), r=r, level=b.level, vx=v)
        i0 = len(rig.frames)
        n_calls0 = len(getattr(b.actuator, "calls", []))
        store.add(item)
        res = yield from _fire_trial(b, item, i0, n_calls0, f"sweep_{k:02d}", duration + 300.0)
        yield from rig.wait_until_ns(store.ns(item.t1) + 100_000_000)
        yield from _after_trial(b)
        rec = {"trial": k, "item": item.id, "speed_dps": speed, "direction": direction, "x0": x0,
               "y0": float(xy[1]), "vx": v, "r": r, "t0": t0, "t1": item.t1, "t_cross": t0 + abs(xy[0] - x0) / abs(v),
               "runway_deg": available[direction], "need_deg": speed * required_s,
               "runway_short": available[direction] < speed * required_s,
               "frames": [i0, len(rig.frames)], **res}
        recs.append(rec)
        b.log(f"  sweep {k + 1:2d}/{trials} {speed:4.0f} deg/s {'L->R' if direction > 0 else 'R->L'}: "
              + _trial_line(b, rec))
        store.prune(store.rig_ms() - 5000.0)
        yield from rig.wait_s(float(b.rng.uniform(0.1, 0.3)))
    b.raw["sweep"] = {"range_m": range_m, "bore_xy": [float(xy[0]), float(xy[1])], "trials": recs}


# ------------------------------------------------------------------------------------------- detect
def detect(b: Bench, ranges_m: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0), positions: int = 5,
           frames_each: int = 30, range_m: float = 20.0):
    """Static clays at simulated ranges: detection rate, centroid error, size -> range."""
    store, rig, smap = b.store, b.rig, b.smap
    W, H = smap.screen_w, smap.screen_h
    pts = [check_bore_on_screen(b, range_m)]
    tries = 0
    while len(pts) < positions and tries < 200:
        tries += 1
        cand = np.array([b.rng.uniform(0.2 * W, 0.8 * W), b.rng.uniform(0.2 * H, 0.8 * H)])
        uv = smap.to_pixel([cand])[0]
        if (np.all(np.isfinite(uv)) and 10 <= uv[0] < b.cam.width - 10 and 10 <= uv[1] < b.cam.height - 10
                and rig.pipeline.mask[int(uv[1]), int(uv[0])] > 0):
            pts.append(cand)
    store.clear()
    yield from rig.wait_s(0.3)
    i_base = len(rig.frames)
    yield from rig.wait_frames(frames_each)
    baseline = [i_base, len(rig.frames)]
    recs = []
    for R in ranges_m:
        for p in pts:
            r = smap.radius_for(p, b.clay_radius_m / float(R))
            item = Disc(id=store.new_id(), t0=store.rig_ms() + 30.0, t1=None, x0=float(p[0]), y0=float(p[1]),
                        r=r, level=b.level)
            store.add(item)
            t_draw = yield from rig.wait_drawn(item.id, 3.0)
            if t_draw is None:
                raise RuntimeError("the page stopped drawing targets")
            yield from rig.wait_until_ns(_settle_ns(b, t_draw))
            i0 = len(rig.frames)
            yield from rig.wait_frames(frames_each)
            store.end(item.id)
            recs.append({"range_m": float(R), "xy": [float(p[0]), float(p[1])], "r": r,
                         "uv": smap.to_pixel([p])[0].tolist(), "frames": [i0, len(rig.frames)]})
            yield from rig.wait_s(0.1)
        b.log(f"  detect: {R:g} m done ({len(pts)} positions)")
    store.prune(store.rig_ms())
    b.raw["detect"] = {"baseline": baseline, "trials": recs}
