"""`tbot score <session> ...` - would each (dry or live) shot of a recorded press have hit?

The solver fires on a prediction: where the bore will point when the shot leaves (trigger delay ahead) and
where the target will be when the shot gets there (time of flight after that). A recorded press holds what
really happened afterwards, so each shot can be checked against it:

  * where the bore REALLY pointed at the exit time: the recorded IMU samples run through the attitude filter
    again, independent of what the live pipeline believed at the time;
  * where the target REALLY was at the arrival time: its detections in the frames either side of that
    moment, turned into world directions with the same attitude.

miss = the angle between the two (shot drop and the offset of the muzzle from the camera included), HIT when
it is inside the pattern the solver assumed. The two halves of the prediction are reported separately: `gun`
is the error in extrapolating the swing over the trigger delay, `target` the error in extrapolating the target.

The target need not be a clay (the dry-run test mode fires on birds): the range, and so the time of flight, is
then what the gun believed, which is the timing chain under test. Needs frames after the shot: the press must
be held (or the post-roll must last) until the shot would have arrived.

`--image` writes shot_<n>.png next to the clip: the frame at the arrival time with the pattern drawn where
the shot went.

`--every` asks the same question of every frame in which the solver had a firing solution for a settled track,
fired or not: "had the command gone out on this frame, would it have hit?". A press with no shot in it then
still says two things: whether there were chances the gun let go (frames that would have hit), and whether
P(hit) means what it says (of the frames it called 0.4, did four in ten hit?).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from tbot.app.record import TELEMETRY_JSONL
from tbot.camera.replay import EVENTS_JSONL, FRAMES_BIN, IMU_CSV, META_JSON
from tbot.config import Config, LatencyBudget, _build, load_config
from tbot.imu.attitude import AttitudeEstimator
from tbot.optics import CameraModel, MountModel
from tbot.types import ImuSample

G_MPS2 = 9.80665
MATCH_PX = 20.0        # a detection this close to the pixel of the track is its own (else the track is used)
MAX_GAP_NS = 60e6      # the target must be seen within this long of the arrival time


def _fill(have: dict, default: dict) -> dict:
    out = copy.deepcopy(default)
    for k, v in have.items():
        out[k] = _fill(v, default[k]) if isinstance(v, dict) and isinstance(default.get(k), dict) else v
    return out


def config_from_meta(meta: dict) -> Config:
    """The config the session was recorded with; keys added to the software since take the defaults of today."""
    try:
        return _build(Config, meta["config"], "")
    except Exception:  # noqa: BLE001
        return _build(Config, _fill(meta["config"], dataclasses.asdict(load_config())), "")


RIG_SECTIONS = ("camera", "optics", "mount", "shot", "gun", "imu", "attitude")


def rig_config(cfg: Config, press_dir: str | Path) -> Config:
    """Today's software settings on the gun a press was recorded with: the hardware sections (camera, lens,
    mount, load, IMU) come from the press, everything that decides (detector, tracker, fire, latency) from
    `cfg`."""
    meta = json.loads((Path(press_dir) / META_JSON).read_text())
    raw = dataclasses.asdict(cfg)
    have = meta.get("config") or {}
    for sec in RIG_SECTIONS:
        if isinstance(have.get(sec), dict):
            raw[sec] = _fill({k: v for k, v in have[sec].items() if k in raw[sec]}, raw[sec])
    return _build(Config, raw, "")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _angle(a, b) -> float:
    a, b = _unit(a), _unit(b)
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b))


def _azel_dir(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


class Session:
    """A recorded press with its attitude re-estimated from the stored IMU samples."""

    def __init__(self, path: str | Path, cfg: Config | None = None, telemetry: list[dict] | None = None) -> None:
        """telemetry: per-frame records to score instead of the session's own telemetry.jsonl (the press run
        again through other software: `tbot rescore`)."""
        self.path = Path(path)
        self.meta = json.loads((self.path / META_JSON).read_text())
        self.cfg = cfg if cfg is not None else config_from_meta(self.meta)
        w, h = int(self.meta["width"]), int(self.meta["height"])
        self.size = (w, h)
        cal = self.meta.get("calibration") or {}
        self.cam = CameraModel.from_calibration(cal.get("intrinsics"), self.cfg, w, h)
        self.mount = MountModel.from_config(self.cfg, cal.get("boresight"))
        self.budget = LatencyBudget.from_config(self.cfg, cal.get("latency"))
        self.R_gc = np.asarray(self.mount.gun_to_cam(np.eye(3)), dtype=np.float64).reshape(3, 3)
        self.tel = []
        if telemetry is not None:
            self.tel = [rec for rec in telemetry if "t_exposure_ns" in rec and rec.get("R_wg") is not None]
        else:
            with open(self.path / TELEMETRY_JSONL) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "t_exposure_ns" in rec and rec.get("R_wg") is not None:
                        self.tel.append(rec)
        self.tel.sort(key=lambda r: r["t_exposure_ns"])
        self.t_exp = np.array([r["t_exposure_ns"] for r in self.tel], dtype=np.int64)
        self.events = []
        p = self.path / EVENTS_JSONL
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        imu = np.loadtxt(self.path / IMU_CSV, delimiter=",", skiprows=1, ndmin=2)
        self.imu = imu
        # no IMU in the recording (a simulated or bench source): the gun did not move
        self.imu_span = (int(imu[0, 0]), int(imu[-1, 0])) if len(imu) else (0, 2**62)
        self.att = AttitudeEstimator(self.cfg)
        self.att.history_ns = int(3600e9)  # keep the whole press
        self.att.max_history = max(self.att.max_history, len(imu) + 16)
        self.att.update([ImuSample(t_ns=int(r[0]), gyro=np.array(r[1:4]), accel=np.array(r[4:7]),
                                   temp_c=float(r[7]) if len(r) > 7 else float("nan")) for r in imu])
        self._seqs: list[int] | None = None

    def R(self, t_ns: int) -> np.ndarray:
        return np.asarray(self.att.R_wg(int(t_ns)), dtype=np.float64).reshape(3, 3)

    def to_offline(self, rec: dict) -> np.ndarray:
        """Rotation taking a world vector of the LIVE pipeline (at this frame) into the world frame of this
        replay: the two differ by where heading zero happened to be, and by what the live attitude got wrong."""
        return self.R(rec["t_exposure_ns"]) @ np.asarray(rec["R_wg"], dtype=np.float64).reshape(3, 3).T

    def target_dir(self, rec: dict, track_id: int):
        """World direction (replay frame) of the track in this frame, from its detection when there is one.
        Returns (dir, pixel, "detection" | "track"), or None if the track is not in the frame."""
        tr = next((t for t in rec.get("tracks") or [] if t.get("id") == track_id), None)
        if tr is None:
            return None
        R_live = np.asarray(rec["R_wg"], dtype=np.float64).reshape(3, 3)
        ray_c = (R_live @ self.R_gc).T @ _azel_dir(tr["az_deg"], tr["el_deg"])
        uv = np.asarray(self.cam.ray_to_pixel(ray_c), dtype=np.float64).reshape(2)
        R_wc = self.R(rec["t_exposure_ns"]) @ self.R_gc
        dets = rec.get("dets") or []
        if dets and tr.get("misses", 0) == 0 and np.isfinite(uv).all():
            d = min(dets, key=lambda d: (d[0] - uv[0]) ** 2 + (d[1] - uv[1]) ** 2)
            if math.hypot(d[0] - uv[0], d[1] - uv[1]) <= max(MATCH_PX, 0.75 * max(d[7], d[8])):
                ray = np.asarray(self.cam.pixel_to_ray(np.array([d[0], d[1]], dtype=np.float64))).reshape(3)
                return _unit(R_wc @ ray), (float(d[0]), float(d[1])), "detection"
        return _unit(R_wc @ ray_c), (float(uv[0]), float(uv[1])), "track"

    def frame(self, index_in_tel: int) -> np.ndarray | None:
        """The stored image of telemetry record `index_in_tel`."""
        w, h = self.size
        ch = int(self.meta.get("channels", 1))
        if self._seqs is None:
            with open(self.path / "frames.csv") as f:
                f.readline()
                self._seqs = [int(line.split(",")[0]) for line in f if line.strip()]
        try:
            k = self._seqs.index(int(self.tel[index_in_tel]["seq"]))
        except ValueError:
            return None
        with open(self.path / FRAMES_BIN, "rb") as f:
            f.seek(k * w * h * ch)
            buf = f.read(w * h * ch)
        if len(buf) < w * h * ch:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape((h, w) if ch == 1 else (h, w, ch))


GATED = ("p_hit ", "better in", "would fire", "FIRE")   # reasons of a frame that passed every gate but confidence


def score(session, image: bool = False, every: bool = False) -> list[dict]:
    """One result per shot of the press; with `every`, one per frame that had a firing solution for a settled
    track (see the module docstring), each scored as if the command had gone out on that frame.
    `session` is a press directory or a Session."""
    s = session if isinstance(session, Session) else Session(session)
    atype = str(s.cfg.actuator.type)
    delay_ns = int(round(s.budget.fire_delay_s(atype) * 1e9))
    bore_g = _unit(s.mount.bore_dir_g)
    muzzle_g = np.asarray(s.mount.muzzle_pos_g, dtype=np.float64) - np.asarray(s.mount.cam_pos_g, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0])
    out = []
    if every:
        def solved(r):
            d = r.get("decision") or {}
            return (d.get("track_id") is not None and d.get("tof_ms") is not None and d.get("miss_deg") is not None
                    and math.isfinite(float(d["miss_deg"])) and str(d.get("reason", "")).startswith(GATED))
        shots = [(i, r) for i, r in enumerate(s.tel) if solved(r)]
    else:
        shots = [(i, r) for i, r in enumerate(s.tel) if r.get("fired")]
    for n, (i, rec) in enumerate(shots, 1):
        dec = rec["decision"]
        res = {"shot": n, "frame": i, "t_ns": int(rec["t_ns"]), "track_id": dec.get("track_id"),
               "fired": bool(rec.get("fired")), "sigma_deg": dec.get("sigma_deg"),
               "live": bool(rec.get("live")), "predicted_miss_deg": dec.get("miss_deg"),
               "pattern_deg": dec.get("pattern_deg"), "p_hit": dec.get("p_hit"), "range_m": dec.get("range_m"),
               "tof_ms": dec.get("tof_ms")}
        out.append(res)
        t_cmd = int(rec["cmd_t_ns"]) if rec.get("cmd_t_ns") else int(rec["t_ns"])
        if every and not rec.get("fired"):
            t_cmd = int(dec["fire_at_ns"]) if dec.get("fire_at_ns") else int(rec["t_ns"])
        t_exit = int(dec["t_exit_ns"]) if dec.get("t_exit_ns") else t_cmd + delay_ns
        t_arr = t_exit + int(round(float(dec["tof_ms"]) * 1e6))
        res.update(delay_ms=(t_exit - t_cmd) / 1e6, arrive_after_ms=(t_arr - int(rec["t_ns"])) / 1e6)
        if t_exit > s.imu_span[1]:
            res["verdict"] = "no data: the recording ends before the shot would have left"
            continue
        k = int(np.searchsorted(s.t_exp, t_arr))
        near = [j for j in (k - 1, k) if 0 <= j < len(s.tel)]
        seen = [(j, s.target_dir(s.tel[j], dec["track_id"])) for j in near]
        seen = [(j, x) for j, x in seen if x is not None]
        if not seen or min(abs(int(s.t_exp[j]) - t_arr) for j, _ in seen) > MAX_GAP_NS:
            res["verdict"] = ("no data: the target was not being tracked when the shot would have arrived"
                              if near and min(abs(int(s.t_exp[j]) - t_arr) for j in near) <= MAX_GAP_NS
                              else "no data: the recording ends before the shot would have arrived")
            continue
        if len(seen) == 2 and s.t_exp[seen[0][0]] <= t_arr <= s.t_exp[seen[1][0]]:
            (j0, a), (j1, b) = seen
            f = (t_arr - int(s.t_exp[j0])) / max(1, int(s.t_exp[j1]) - int(s.t_exp[j0]))
            tdir = _unit((1.0 - f) * a[0] + f * b[0])
            j_show, src = (j0, a[2]) if f < 0.5 else (j1, b[2])
            if a[2] != b[2]:
                src = "detection+track"
        else:
            j_show, (tdir, _px, src) = min(seen, key=lambda x: abs(int(s.t_exp[x[0]]) - t_arr))
        rng = float(dec["range_m"])
        R_exit = s.R(t_exit)
        bore_w = _unit(R_exit @ bore_g)
        to_target = rng * tdir - R_exit @ muzzle_g
        dist = float(np.linalg.norm(to_target))
        tof = float(dec["tof_ms"]) * 1e-3
        drop = 0.5 * G_MPS2 * tof * tof * math.cos(math.asin(max(-1.0, min(1.0, float(bore_w[2])))))
        to_shot = bore_w * dist - drop * up
        miss = _angle(to_target, to_shot)
        pattern = math.radians(float(dec["pattern_deg"]))
        M = s.to_offline(rec)
        gun_err = (_angle(M @ np.asarray(dec["bore_dir_w"], dtype=np.float64), bore_w)
                   if dec.get("bore_dir_w") else float("nan"))
        tgt_err = (_angle(M @ np.asarray(dec["lead_dir_w"], dtype=np.float64), to_target)
                   if dec.get("lead_dir_w") else float("nan"))
        swing = _angle(s.R(int(rec["t_ns"])) @ bore_g, bore_w)
        res.update(miss_deg=math.degrees(miss), hit=bool(miss <= pattern), gun_error_deg=math.degrees(gun_err),
                   target_error_deg=math.degrees(tgt_err), target_from=src,
                   swing_during_delay_deg=math.degrees(swing),
                   miss_cm_at_range=100.0 * dist * math.tan(miss), pattern_cm_at_range=100.0 * dist * math.tan(pattern),
                   verdict="HIT" if miss <= pattern else "MISS")
        if image and not every:
            img = s.frame(j_show)
            if img is not None:
                res["image"] = str(_draw(s, img, j_show, to_shot, tdir, pattern, res, n))
    return out


def _draw(s: Session, img: np.ndarray, j: int, to_shot: np.ndarray, tdir: np.ndarray, pattern: float,
          res: dict, n: int) -> Path:
    scale = 2
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    R_wc = s.R(s.tel[j]["t_exposure_ns"]) @ s.R_gc
    ifov = float(np.asarray(s.cam.ifov(np.array([[0.5 * s.size[0], 0.5 * s.size[1]]]))).ravel()[0])

    def px(v):
        uv = np.asarray(s.cam.ray_to_pixel(R_wc.T @ _unit(v)), dtype=np.float64).reshape(2)
        return None if not np.isfinite(uv).all() else (int(round(uv[0] * scale)), int(round(uv[1] * scale)))

    shot, tgt = px(to_shot), px(tdir)
    colour = (0, 220, 0) if res["hit"] else (0, 0, 255)
    if shot is not None:
        cv2.circle(bgr, shot, max(2, int(round(pattern / ifov * scale))), colour, 2, cv2.LINE_AA)
        cv2.drawMarker(bgr, shot, colour, cv2.MARKER_CROSS, 12, 1)
    if tgt is not None:
        cv2.drawMarker(bgr, tgt, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    if shot is not None and tgt is not None:
        cv2.line(bgr, shot, tgt, (255, 255, 0), 1, cv2.LINE_AA)
    lines = [f"shot {n}: {res['verdict']}  miss {res['miss_deg']:.2f} deg / pattern {res['pattern_deg']:.2f} deg",
             f"gun {res['gun_error_deg']:.2f} deg  target {res['target_error_deg']:.2f} deg  "
             f"swing in delay {res['swing_during_delay_deg']:.1f} deg  range {res['range_m']:.0f} m",
             "circle = where the shot went (pattern)   x = target at arrival"]
    for k, text in enumerate(lines):
        org = (8, 22 + 20 * k)
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = s.path / f"shot_{n}.png"
    cv2.imwrite(str(out), bgr)
    return out


def format_shot(r: dict) -> str:
    head = f"  shot {r['shot']}  track #{r['track_id']}  range {r['range_m']:.1f} m  P(hit) at fire {r['p_hit']:.2f}"
    if "miss_deg" not in r:
        return f"{head}\n    {r['verdict']}"
    return (f"{head}\n    {r['verdict']}: miss {r['miss_deg']:.2f} deg ({r['miss_cm_at_range']:.0f} cm) against a pattern "
            f"of {r['pattern_deg']:.2f} deg ({r['pattern_cm_at_range']:.0f} cm); predicted miss "
            f"{r['predicted_miss_deg']:.2f} deg\n"
            f"    gun extrapolation error {r['gun_error_deg']:.2f} deg (the bore swung "
            f"{r['swing_during_delay_deg']:.1f} deg during the {r['delay_ms']:.0f} ms delay), target extrapolation "
            f"error {r['target_error_deg']:.2f} deg (target from its {r['target_from']})")


P_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01)


def summarize_every(rows: list[dict]) -> dict:
    """Frames scored as if fired (score(..., every=True)): how often they would have hit, by the P(hit) the
    solver gave them, and how far off the two halves of the prediction were."""
    rows = [r for r in rows if "hit" in r and r.get("p_hit") is not None]
    out = {"frames": len(rows), "would_hit": int(sum(r["hit"] for r in rows)), "bins": []}
    if not rows:
        return out
    p = np.array([float(r["p_hit"]) for r in rows])
    hit = np.array([bool(r["hit"]) for r in rows])
    for lo, hi in zip(P_BINS, P_BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            out["bins"].append({"p_lo": lo, "p_hi": min(hi, 1.0), "frames": int(m.sum()),
                                "p_mean": float(p[m].mean()), "hit_frac": float(hit[m].mean())})
    for key in ("gun_error_deg", "target_error_deg", "miss_deg"):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            out[key] = {"p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}
    # the longest run of consecutive frames that would have hit: a chance the gun had, in seconds
    best = cur = 0.0
    start = None
    last = None
    for r in rows:
        if r["hit"] and start is not None and last is not None and r["frame"] == last + 1:
            cur = (r["t_ns"] - start) / 1e9
        elif r["hit"]:
            start, cur = r["t_ns"], 0.0
        else:
            start = None
        last = r["frame"]
        best = max(best, cur)
    out["longest_hit_window_s"] = best
    return out


def format_every(summary: dict) -> str:
    n = summary["frames"]
    if not n:
        return "  no frame had a firing solution for a settled track with frames after it"
    lines = [f"  {n} frames with a solution, scored as if fired: {summary['would_hit']} would have hit "
             f"({100.0 * summary['would_hit'] / n:.0f} %), longest unbroken chance "
             f"{summary['longest_hit_window_s'] * 1e3:.0f} ms",
             "    P(hit) given    frames   really hit"]
    for b in summary["bins"]:
        lines.append(f"    {b['p_lo']:.1f} - {b['p_hi']:.1f}     {b['frames']:6d}   {100.0 * b['hit_frac']:5.0f} %")
    for key, name in (("gun_error_deg", "gun extrapolation error"), ("target_error_deg", "target extrapolation error"),
                      ("miss_deg", "real miss")):
        if key in summary:
            lines.append(f"    {name}: p50 {summary[key]['p50']:.2f} deg, p90 {summary[key]['p90']:.2f} deg")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbot score", description="would each shot of a recorded press have hit?")
    parser.add_argument("sessions", nargs="+", metavar="SESSION")
    parser.add_argument("--image", action="store_true", help="write shot_<n>.png (the arrival frame) into the session")
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    parser.add_argument("--every", action="store_true",
                        help="score every frame that had a firing solution as if it had fired: chances the gun "
                             "let go, and whether P(hit) means what it says")
    args = parser.parse_args(argv)
    status = 0
    everything = []
    for session in args.sessions:
        try:
            shots = score(session, image=args.image, every=args.every)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the rest
            print(f"tbot score: {session}: {exc}", file=sys.stderr)
            status = 1
            continue
        everything.append({"session": str(session), "shots": shots})
        if args.every:
            everything[-1]["every"] = summarize_every(shots)
        if args.json:
            continue
        if args.every:
            print(f"{session}:")
            print(format_every(everything[-1]["every"]))
            continue
        print(f"{session}: {len(shots)} shot(s)")
        for r in shots:
            print(format_shot(r))
    if args.json:
        print(json.dumps(everything, indent=1))
    elif args.every:
        if len(everything) > 1:
            print("all sessions:")
            print(format_every(summarize_every([r for e in everything for r in e["shots"]])))
    else:
        scored = [r for e in everything for r in e["shots"] if "hit" in r]
        if scored:
            hits = sum(r["hit"] for r in scored)
            miss = np.array([r["miss_deg"] for r in scored])
            print(f"total: {hits}/{len(scored)} hits   miss p50 {np.median(miss):.2f} deg  max {miss.max():.2f} deg   "
                  f"gun error p50 {np.median([r['gun_error_deg'] for r in scored]):.2f} deg   "
                  f"target error p50 {np.median([r['target_error_deg'] for r in scored]):.2f} deg")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
