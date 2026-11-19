#!/usr/bin/env python3
"""Range-day media from recorded presses: press table, highlight reel, GIFs, and phone-clip sync.

Works on press folders as the Pi records them (sessions/<date>-<time>-press/: frames.bin, frames.csv,
telemetry.jsonl, events.jsonl, meta.json, summary.json, imu.csv). Picture source per press: the annotated
clip <clips>/<HHMMSS>-press.mp4 made by `tbot clip` (frame n of the clip = row n of frames.csv), or the raw
frames.bin when there is no clip. Needs numpy, opencv (cv2) and ffmpeg/ffprobe on PATH; does not import tbot.

    python scripts/range_media.py table SESSIONS                      # markdown table: who fired, recoil, boots
    python scripts/range_media.py reel  SESSIONS --clips DIR --out reel.mp4
    python scripts/range_media.py gif   SESSIONS/<press> --clips DIR --out shot.gif
    python scripts/range_media.py sync  SESSIONS --phone PHONE_DIR --out OUT_DIR --clips DIR \
        --window 2026-11-19T15:40-05:00/2026-11-19T16:30-05:00

Times. Every press carries the Pi's monotonic clock (ns since that boot) and the Pi's wall clock at the end of
the press (meta.json "created", also the folder name). The Pi has no clock battery: after a power cut its wall
clock restarts from a saved time and can jump BACKWARDS by minutes, so presses from different boots cannot be
ordered by name. Within one boot the monotonic clock is exact. `boot` below = wall clock minus uptime; presses
whose boot times agree within 3 s are one boot. The shot instant of a press is the IMU recoil event when there
is one (the gun really went off), else the fire command + `expected_ms` (the trigger delay the gun expected).

Phone sync (`sync`). Phone clips are checked against --window using their container creation time
(com.apple.quicktime.creationdate, which carries the phone's local offset, else creation_time in UTC; some
phones stamp the END of the recording, so a clip passes when any part of [t, t+duration] or [t-duration, t]
overlaps the window). Muzzle blasts are found in each clip's audio (sharp onsets far above the clip's noise
floor, 0.3 s refractory). Matching does not trust the Pi's wall clock: for each Pi boot it searches the clock
offset (up to --max-skew-s) that lines up the most of that boot's shots with blasts at once (shot SPACING inside
a boot is exact), boots with more shots first, one blast per shot, and reports the residuals. Single-shot boots
fall back to the nearest unclaimed blast in order and are marked low confidence. Override any pairing with
--pair IMG_1234.MOV=160357-press (optionally =160357-press@12.345 to name the blast time in the phone clip).
Each pair is rendered with the phone and the Pi picture side by side (--layout sbs) or the Pi picture inset
in the phone video (--layout pip), aligned on the shot, phone audio kept, then (with --slowmo N) the moment of
the shot again at 1/N speed. --dry-run prints the matching only.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("range_media.py needs opencv (pip install opencv-python)")

EST = dt.timezone(dt.timedelta(hours=-5))
SOUND_MPS = 343.0
W, H = 1280, 800  # Pi picture size used everywhere (tbot clip --scale 2)


# ----------------------------------------------------------------------------------------------- presses
@dataclass
class Press:
    path: Path
    name: str                 # e.g. 160357-press
    frames: int
    t_frames: list[int]       # exposure time of each recorded frame, Pi monotonic ns
    created: dt.datetime      # Pi wall clock when the press was written (end of the press)
    end_ns: int               # Pi monotonic ns of the release
    press_ns: tuple[int, int]
    fps: float
    live: bool
    fires: list[dict]
    recoils: list[dict]
    imu_shocks_ms: list[tuple[float, float]] = field(default_factory=list)  # (ms after fire cmd, g) >= 4 g
    p_hit_max: float | None = None
    detections: int = 0
    reasons: dict = field(default_factory=dict)
    raw_ok: bool = True
    boot_id: int = -1

    @property
    def boot_wall(self) -> dt.datetime:
        return self.created - dt.timedelta(seconds=self.end_ns / 1e9)

    @property
    def uptime_s(self) -> float:
        return self.press_ns[0] / 1e9

    @property
    def held_s(self) -> float:
        return (self.press_ns[1] - self.press_ns[0]) / 1e9

    @property
    def fired(self) -> bool:
        return bool(self.fires)

    @property
    def shot_ns(self) -> int | None:
        """Pi monotonic ns of the shot: IMU recoil if seen, else fire command + expected delay."""
        if self.recoils:
            return int(self.recoils[0]["t_ns"])
        if self.fires:
            f = self.fires[0]
            exp = next((e.get("expected_ms") for e in self.fire_events_all if e.get("expected_ms")), None)
            return int(f["cmd_t_ns"] + (exp if exp else f.get("fire_delay_ms", 45.0)) * 1e6)
        return None

    fire_events_all: list[dict] = field(default_factory=list)

    def wall_of(self, t_ns: int) -> dt.datetime:
        """Pi wall clock at monotonic t_ns (only as good as the Pi's clock)."""
        return self.created - dt.timedelta(seconds=(self.end_ns - t_ns) / 1e9)

    def frame_at(self, t_ns: int) -> int:
        i = bisect.bisect_right(self.t_frames, t_ns) - 1
        return max(0, min(i, len(self.t_frames) - 1))


def load_press(d: Path) -> Press:
    s = json.loads((d / "summary.json").read_text())
    m = json.loads((d / "meta.json").read_text())
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    with open(d / "frames.csv") as fh:
        t_frames = [int(r["t_exposure_ns"]) for r in csv.DictReader(fh)]
    tr = s["trigger"]
    name = d.name.split("-", 1)[1] if d.name.count("-") >= 2 else d.name
    p = Press(
        path=d, name=name, frames=s["frames"], t_frames=t_frames,
        created=dt.datetime.fromisoformat(m["created"]), end_ns=int(tr["t_ns"]),
        press_ns=(int(tr["press_ns"][0]), int(tr["press_ns"][1] or tr["t_ns"])),
        fps=float(s.get("fps") or 0), live=bool(s.get("live")),
        fires=[e for e in ev if e.get("type") == "fire"],
        recoils=[e for e in ev if e.get("type") == "recoil"],
        p_hit_max=s.get("max_p_hit"), detections=s.get("frames_with_detections", 0),
        reasons=s.get("decision_reasons") or {},
        raw_ok=(d / "frames.bin").exists() and (d / "frames.bin").stat().st_size
        >= len(t_frames) * m["width"] * m["height"] * m.get("channels", 1),
    )
    p.fire_events_all = ev
    if p.fires and (d / "imu.csv").exists():
        c = p.fires[0]["cmd_t_ns"]
        shocks: list[tuple[float, float]] = []
        with open(d / "imu.csv") as fh:
            for r in csv.DictReader(fh):
                g = math.sqrt(float(r["ax"]) ** 2 + float(r["ay"]) ** 2 + float(r["az"]) ** 2) / 9.81
                if g >= 4.0:
                    t_ms = (int(r["t_ns"]) - c) / 1e6
                    if shocks and t_ms - shocks[-1][0] < 25:
                        if g > shocks[-1][1]:
                            shocks[-1] = (shocks[-1][0], g)
                    else:
                        shocks.append((t_ms, g))
        p.imu_shocks_ms = shocks
    return p


def load_presses(root: Path, after: str | None = None) -> list[Press]:
    dirs = sorted(x for x in root.iterdir() if x.is_dir() and (x / "summary.json").exists())
    presses = []
    for x in dirs:
        try:
            presses.append(load_press(x))
        except (ValueError, KeyError, OSError) as e:  # zero-byte files after a power cut
            print(f"skip {x.name}: {type(e).__name__}", file=sys.stderr)
    if after:
        presses = [p for p in presses if p.path.name >= after]
    assign_boots(presses)
    return order_presses(presses)


def assign_boots(presses: list[Press], tol_s: float = 3.0) -> None:
    boots: list[dt.datetime] = []
    for p in sorted(presses, key=lambda p: p.boot_wall):
        if boots and abs((p.boot_wall - boots[-1]).total_seconds()) <= tol_s:
            p.boot_id = len(boots) - 1
        else:
            boots.append(p.boot_wall)
            p.boot_id = len(boots) - 1


def order_presses(presses: list[Press]) -> list[Press]:
    """Best-guess true order: boots by their (restored) wall clock, then uptime inside a boot."""
    return sorted(presses, key=lambda p: (p.boot_id, p.uptime_s))


def shot_note(p: Press) -> str:
    if not p.fired:
        return ""
    big = [t for t, g in p.imu_shocks_ms if g >= 8.0]
    if not p.recoils and not big:
        return "fire command; NO recoil on the IMU"
    extra = [t for t in big if t > (big[0] + 60)] if big else []
    s = "gun went off (recoil on the IMU)"
    if extra:
        s += ("; more recoil-sized jolts at " + ", ".join(f"{t / 1000:.2f}" for t in extra)
              + " s after the command (double?)")
    return s


# ----------------------------------------------------------------------------------------------- frames
class FrameSource:
    """Frames of a press as 1280x800 BGR, from the annotated clip or the raw frames.bin."""

    def __init__(self, p: Press, clips: Path | None):
        self.p = p
        self.clip = None
        if clips is not None:
            for cand in (clips / f"{p.name}.mp4", p.path / "clip.mp4"):
                if cand.exists() and cand.stat().st_size > 1000:
                    self.clip = cand
                    break
        self._cache: dict[int, np.ndarray] = {}
        self._mm = None
        if self.clip is None:
            if not p.raw_ok:
                raise FileNotFoundError(f"{p.name}: no clip and frames.bin is empty/short")
            m = json.loads((p.path / "meta.json").read_text())
            self._mm = np.memmap(p.path / "frames.bin", dtype=np.uint8, mode="r").reshape(
                -1, m["height"], m["width"])

    def load(self, i0: int, i1: int) -> None:
        i0, i1 = max(0, i0), min(self.p.frames - 1, i1)
        if self.clip is not None:
            cap = cv2.VideoCapture(str(self.clip))
            n = 0
            while n <= i1:
                ok, fr = cap.read()
                if not ok:
                    break
                if n >= i0:
                    if fr.shape[1] != W:
                        fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_NEAREST)
                    self._cache[n] = fr
                n += 1
            cap.release()
        else:
            for n in range(i0, i1 + 1):
                g = cv2.resize(np.asarray(self._mm[n]), (W, H), interpolation=cv2.INTER_NEAREST)
                self._cache[n] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def get(self, i: int) -> np.ndarray:
        i = max(0, min(i, self.p.frames - 1))
        if i not in self._cache:
            self.load(i, i + 120)
        if i not in self._cache:  # clip shorter than the csv: hold its last frame
            i = max(self._cache) if self._cache else i
        return self._cache[i]

    def at(self, t_ns: int) -> np.ndarray:
        """Frame showing at Pi monotonic time t_ns (the latest frame exposed at or before it)."""
        return self.get(self.p.frame_at(t_ns))


# ----------------------------------------------------------------------------------------------- output
def pick_encoder() -> list[str]:
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if " libx264 " in enc:
        return ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]
    if " libopenh264 " in enc:
        return ["-c:v", "libopenh264", "-b:v", "4M"]
    return ["-c:v", "mpeg4", "-q:v", "3"]


class Writer:
    """Pipe BGR frames into ffmpeg (H.264 mp4)."""

    def __init__(self, out: Path, w: int, h: int, fps: float, extra_in: list[str] | None = None,
                 extra_out: list[str] | None = None):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = w, h
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps}", "-i", "-"] + (extra_in or []) + pick_encoder() + [
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"] + (extra_out or []) + [str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img: np.ndarray) -> None:
        if img.shape[1] != self.w or img.shape[0] != self.h:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 1.0,
          color=(255, 255, 255), bg=(0, 0, 0), thick: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + base + 6), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def title_card(lines: list[str], w: int = W, h: int = H) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (24, 24, 24)
    y = h // 2 - 40 * (len(lines) - 1)
    for k, t in enumerate(lines):
        sc = 1.6 if k == 0 else 1.0
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 3 if k == 0 else 2)
        cv2.putText(img, t, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                    (255, 255, 255) if k == 0 else (200, 200, 200), 3 if k == 0 else 2, cv2.LINE_AA)
        y += 80 if k == 0 else 55
    return img


def press_segment(src: FrameSource, t0: int, t1: int, fps: float, slow: float = 1.0):
    """Yield frames on a constant output clock from Pi time t0 to t1 (ns), `slow`x slow motion."""
    step = 1e9 / fps / slow
    n = int((t1 - t0) / step)
    for k in range(n + 1):
        yield src.at(int(t0 + k * step)), int(t0 + k * step)


# ----------------------------------------------------------------------------------------------- table
def cmd_table(a) -> int:
    presses = load_presses(Path(a.sessions), a.after)
    print("| # | press (Pi clock) | boot | up | held | frames | fps | fired | P(hit) | range | IMU | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, p in enumerate(presses, 1):
        f = p.fires[0] if p.fires else None
        top = max(p.reasons.items(), key=lambda kv: kv[1])[0] if p.reasons else "-"
        notes = []
        if not p.raw_ok:
            notes.append("frames lost (power cut)")
        if not p.fired and p.detections == 0:
            notes.append("nothing detected")
        elif not p.fired:
            notes.append(f"held fire: {top}")
        print(f"| {k} | {p.name} | {p.boot_id + 1} ({p.boot_wall:%H:%M:%S}) | {p.uptime_s:.0f} s | "
              f"{p.held_s:.1f} s | {p.frames} | {p.fps:.0f} | {'**FIRE**' if f else '-'} | "
              f"{(f['p_hit'] if f else (p.p_hit_max or 0)):.2f} | "
              f"{(str(round(f['range_m'])) + ' m') if f else '-'} | {shot_note(p) or '-'} | "
              f"{'; '.join(notes) or '-'} |")
    fired = [p for p in presses if p.fired]
    rec = [p for p in fired if p.recoils]
    boots = len({p.boot_id for p in presses})
    print(f"\n{len(presses)} presses, {boots} boots, {len(fired)} fire commands, {len(rec)} with recoil on the IMU.")
    return 0


# ----------------------------------------------------------------------------------------------- reel / gif
def shot_caption(p: Press, idx: int | None = None) -> list[str]:
    f = p.fires[0]
    head = f"Shot {idx}" if idx else p.name
    return [head, f"Pi clock {p.name[:2]}:{p.name[2:4]}:{p.name[4:6]}  -  range {f['range_m']:.0f} m  -  "
                  f"P(hit) {f['p_hit']:.2f}", shot_note(p)]


def cmd_reel(a) -> int:
    presses = load_presses(Path(a.sessions), a.after)
    clips = Path(a.clips) if a.clips else None
    sel = [p for p in presses if p.fired and (not a.only or p.name in a.only or p.path.name in a.only)]
    if a.recoil_only:
        sel = [p for p in sel if p.recoils]
    if not sel:
        print("no fired presses selected")
        return 1
    fps = a.fps
    w = Writer(Path(a.out), W, H, fps)
    if a.title:
        for _ in range(int(fps * 2.5)):
            w.write(title_card(a.title.split("|")))
    for k, p in enumerate(sel, 1):
        src = FrameSource(p, clips)
        shot = p.shot_ns
        t0 = max(p.t_frames[0], shot - int(a.pre * 1e9))
        t1 = min(p.t_frames[-1], shot + int(a.post * 1e9))
        cap = shot_caption(p, k)
        card = title_card(cap)
        for _ in range(int(fps * a.card_s)):
            w.write(card)
        for img, t in press_segment(src, t0, t1, fps):
            img = img.copy()
            label(img, f"{cap[0]}  real time", (24, H - 70), 0.9)
            w.write(img)
        if a.slowmo > 1:
            s0 = max(p.t_frames[0], shot - int(a.slow_pre * 1e9))
            s1 = min(p.t_frames[-1], shot + int(a.slow_post * 1e9))
            for img, t in press_segment(src, s0, s1, fps, a.slowmo):
                img = img.copy()
                label(img, f"{cap[0]}  {a.slowmo:g}x slow  t{(t - shot) / 1e6:+.0f} ms from the shot",
                      (24, H - 70), 0.9, color=(80, 230, 255))
                w.write(img)
        print(f"reel: {p.name} ({shot_note(p)})")
    w.close()
    print(f"wrote {a.out}")
    return 0


def cmd_gif(a) -> int:
    p = load_press(Path(a.press))
    src = FrameSource(p, Path(a.clips) if a.clips else None)
    shot = p.shot_ns or p.t_frames[len(p.t_frames) * 2 // 3]
    t0 = max(p.t_frames[0], shot - int(a.pre * 1e9))
    t1 = min(p.t_frames[-1], shot + int(a.post * 1e9))
    tmp = Path(a.out).with_suffix(".tmp.mp4")
    w = Writer(tmp, W, H, a.fps)
    for img, t in press_segment(src, t0, t1, a.fps, a.slowmo):
        if a.caption:
            img = img.copy()
            label(img, a.caption, (24, H - 70), 1.0)
        w.write(img)
    w.close()
    vf = (f"scale={a.width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors={a.colors}:stats_mode=diff[p];"
          f"[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-vf", vf, "-loop", "0", a.out], check=True)
    tmp.unlink()
    mb = Path(a.out).stat().st_size / 1e6
    print(f"wrote {a.out} ({mb:.1f} MB)")
    return 0 if mb < a.max_mb else 2


# ----------------------------------------------------------------------------------------------- phone
@dataclass
class PhoneClip:
    path: Path
    start: dt.datetime | None        # aware datetime of the first frame (best guess)
    duration: float
    fps: float
    width: int
    height: int
    rotation: int
    has_audio: bool
    tag: str                          # which metadata the time came from
    blasts: list[float] = field(default_factory=list)  # seconds into the clip
    ok_window: bool = False


def ffprobe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        str(path)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def parse_time(s: str) -> dt.datetime | None:
    s = s.strip().replace("Z", "+00:00")
    for cand in (s, s[:19] + s[-6:] if len(s) > 25 else s):
        try:
            t = dt.datetime.fromisoformat(cand)
            return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    try:  # 2026-11-19T15:47:12-0500
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def probe_phone(path: Path) -> PhoneClip | None:
    j = ffprobe(path)
    if not j.get("streams"):
        return None
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        return None
    tags = {k.lower(): val for k, val in (j.get("format", {}).get("tags") or {}).items()}
    for s in j["streams"]:
        for k, val in (s.get("tags") or {}).items():
            tags.setdefault(k.lower(), val)
    start, tag = None, "none"
    for key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        if key in tags and (t := parse_time(tags[key])):
            start, tag = t, key
            break
    if start is None:  # last resort: file mtime (copying a file often keeps it)
        start = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        tag = "file mtime"
    num, den = (v.get("avg_frame_rate") or "30/1").split("/")
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int((v.get("tags") or {}).get("rotate", rot))
    return PhoneClip(path=path, start=start, duration=float(j["format"].get("duration") or 0),
                     fps=float(num) / float(den or 1) if float(den or 1) else 30.0,
                     width=int(v["width"]), height=int(v["height"]), rotation=rot,
                     has_audio=any(s.get("codec_type") == "audio" for s in j["streams"]), tag=tag)


def read_audio(path: Path, fs: int = 16000) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1",
                          "-ar", str(fs), "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def find_blasts(path: Path, fs: int = 16000, group_s: float = 0.6, k_floor: float = 25.0,
                min_dbfs: float = -16.0) -> list[tuple[float, float]]:
    """Gun shots in a clip's audio as (onset s, peak dBFS).

    A shot = the 2 ms envelope jumping above max(k_floor x the clip's median, min_dbfs). Phones limit a shot
    to about -10 dBFS, so the absolute floor does most of the work; the echo off the tree line ~0.3 s later
    is as loud as the shot itself, so onsets within group_s of a shot are folded into it (a real double
    inside group_s cannot be told from the echo by the audio alone: the IMU in the press tells)."""
    x = read_audio(path, fs)
    if x.size < fs // 10:
        return []
    x = x - np.convolve(x, np.ones(64) / 64, mode="same")       # drop rumble / wind below ~250 Hz
    win = int(fs * 0.002)
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same"))
    thr = max((np.median(env) + 1e-5) * k_floor, 10 ** (min_dbfs / 20))
    out: list[tuple[float, float]] = []
    for j in np.flatnonzero(env > thr):
        if out and j / fs - out[-1][0] < group_s:
            continue
        b = j
        while b > 0 and env[b - 1] > thr * 0.25 and j - b < win * 10:   # back to the start of the rise
            b -= 1
        pk = float(env[j:j + int(0.05 * fs)].max())
        out.append((b / fs, 20 * math.log10(pk + 1e-9)))
    return out


def in_window(c: PhoneClip, w0: dt.datetime, w1: dt.datetime) -> bool:
    if c.start is None:
        return False
    d = dt.timedelta(seconds=c.duration)
    spans = [(c.start, c.start + d), (c.start - d, c.start)]  # start-stamped or end-stamped
    return any(a < w1 and b > w0 for a, b in spans)


def match_shots(presses: list[Press], clips: list[PhoneClip], tol_s: float, max_skew_s: float,
                sound_s: float, end_stamped: bool, audible: str, boot_s: float) -> list[dict]:
    """Pair Pi shots with phone blasts. Returns [{press, clip, blast_s, offset_s, resid_ms, how}].

    1. Boots with 2+ audible shots: the offset (phone clock - Pi clock) that puts the most of them on blasts,
       within tol_s each (the spacing inside one boot is exact), smallest residual.
    2. Boots after that, in Pi-clock boot order: a boot cannot start before the previous boot's last press
       ended, and its shot comes `uptime` after that start (+ boot_s, the time to boot), so its shot is the
       first unclaimed blast after that bound. LOW confidence: check the pictures.
    """
    blasts = []  # (absolute time as epoch s, clip, t_in_clip)
    for c in clips:
        st = c.start - dt.timedelta(seconds=c.duration) if end_stamped else c.start
        for b, _db in c.blasts:
            blasts.append((st.timestamp() + b - sound_s, c, b))
    blasts.sort(key=lambda x: x[0])
    shots = [p for p in presses if p.fired and (audible == "fire" or p.recoils)]
    # one wall-clock reference per boot (+ exact monotonic time); per-press "created" stamps jitter by ~1 s
    ref: dict[int, float] = {}
    for p in presses:
        ref.setdefault(p.boot_id, []).append(p.boot_wall.timestamp())  # type: ignore[attr-defined]
    ref = {k: float(np.median(v)) for k, v in ref.items()}

    def real(p: Press, t_ns: int) -> float:
        return ref[p.boot_id] + t_ns / 1e9

    by_boot: dict[int, list[Press]] = {}
    for p in shots:
        by_boot.setdefault(p.boot_id, []).append(p)
    used: set[int] = set()
    res: list[dict] = []
    offsets: dict[int, float] = {}
    for bid, ps in sorted(by_boot.items(), key=lambda kv: -len(kv[1])):
        if len(ps) < 2:
            continue
        pi_t = [real(p, p.shot_ns) for p in ps]
        best = None
        for k, (bt, _, _) in enumerate(blasts):
            for t in pi_t:
                off = bt - t
                if abs(off) > max_skew_s:
                    continue
                pairs, resid = [], 0.0
                for t2 in pi_t:
                    cand = [(abs(blasts[m][0] - (t2 + off)), m) for m in range(len(blasts)) if m not in used]
                    cand = [c for c in cand if c[0] <= tol_s]
                    if cand:
                        e, m = min(cand)
                        pairs.append(m)
                        resid += e
                    else:
                        pairs.append(None)
                nmatch = sum(m is not None for m in pairs)
                if nmatch < 2 or len({m for m in pairs if m is not None}) < nmatch:
                    continue
                score = (nmatch, -resid)
                if best is None or score > best[0]:
                    best = (score, off, pairs)
        if best is None:
            continue
        _, off, pairs = best
        offsets[bid] = off
        for p, t, m in zip(ps, pi_t, pairs):
            if m is None:
                res.append(dict(press=p, clip=None, how="no blast at this boot's offset"))
                continue
            used.add(m)
            bt, c, b = blasts[m]
            res.append(dict(press=p, clip=c, blast_s=b, offset_s=off, resid_ms=(bt - (t + off)) * 1e3,
                            how=f"spacing ({sum(x is not None for x in pairs)} shots of this boot)"))
    # boots in Pi-clock order: bound each unresolved boot by the end of the previous one
    allp = sorted(presses, key=lambda p: (p.boot_id, p.uptime_s))
    boot_ids = sorted({p.boot_id for p in allp})
    last_end_real: float | None = None
    for bid in boot_ids:
        bp = [p for p in allp if p.boot_id == bid]
        if bid not in offsets and bid in by_boot and last_end_real is not None:
            for p in by_boot[bid]:
                bound = last_end_real + boot_s + p.shot_ns / 1e9
                cand = [m for m, x in enumerate(blasts) if m not in used and x[0] >= bound]
                if not cand:
                    res.append(dict(press=p, clip=None, how="no unclaimed blast after the previous boot"))
                    continue
                m = cand[0]
                used.add(m)
                bt, c, b = blasts[m]
                offsets[bid] = bt - real(p, p.shot_ns)
                res.append(dict(press=p, clip=c, blast_s=b, offset_s=offsets[bid], resid_ms=float("nan"),
                                how=f"first blast >= {bound - bt:+.0f} s bound after the previous boot "
                                    f"(LOW confidence: check the pictures)"))
        elif bid in by_boot and bid not in offsets:
            for p in by_boot[bid]:
                res.append(dict(press=p, clip=None, how="single shot and no earlier boot to anchor on"))
        if bid in offsets:
            last = bp[-1]
            last_end_real = real(last, last.end_ns) + offsets[bid]
    return res


def pi_frame_scaled(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])) // 2 * 2, h), interpolation=cv2.INTER_AREA)


def phone_frames(path: Path, t_a: float, t_b: float, height: int) -> list[tuple[float, np.ndarray]]:
    """(time s, frame) of a phone clip from t_a to t_b, rotated upright, `height` tall. OpenCV decodes
    (its wheels carry HEVC, which a distro ffmpeg often lacks)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_a - 0.5) * 1000)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t < t_a - 0.05:
            continue
        if t > t_b:
            break
        w = int(round(fr.shape[1] * height / fr.shape[0])) // 2 * 2
        out.append((t, cv2.resize(fr, (w, height), interpolation=cv2.INTER_AREA)))
    cap.release()
    return out


def render_pair(p: Press, c: PhoneClip, blast_s: float, src: FrameSource, out: Path, layout: str,
                pre: float, post: float, slowmo: float, sound_s: float, fps: float = 30.0,
                height: int = 720) -> None:
    """Phone + Pi, aligned on the shot: phone time blast_s - sound delay == Pi shot instant."""
    shot = p.shot_ns
    b = blast_s - sound_s
    t_a = max(0.0, b - pre)
    t_b = min(c.duration, b + post)
    ph = phone_frames(c.path, t_a, t_b, height)
    if not ph:
        raise RuntimeError(f"{c.path.name}: no frames decoded")
    pts = [t for t, _ in ph]
    pw = ph[0][1].shape[1]
    piw = int(round(W * height / H)) // 2 * 2
    ow, oh = (pw + piw, height) if layout == "sbs" else (pw, height)

    def phone_at(t: float) -> np.ndarray:
        return ph[max(0, bisect.bisect_right(pts, t) - 1)][1]

    def compose(t_phone: float, tag: str) -> np.ndarray:
        rel = t_phone - b
        phf = phone_at(t_phone)
        pi = src.at(int(shot + rel * 1e9))
        if layout == "sbs":
            canvas = np.zeros((oh, ow, 3), np.uint8)
            canvas[:, :phf.shape[1]] = phf
            pis = pi_frame_scaled(pi, oh)
            canvas[:, pw:pw + pis.shape[1]] = pis[:, :ow - pw]
        else:
            canvas = phf.copy()
            ih = oh // 4
            pis = pi_frame_scaled(pi, ih)
            pis = pis[:, :min(pis.shape[1], ow - 20)]
            x0, y0 = ow - pis.shape[1] - 8, 8
            canvas[y0:y0 + ih, x0:x0 + pis.shape[1]] = pis
        label(canvas, f"phone {c.path.name}  |  Pi {p.name}  |  t{rel * 1e3:+.0f} ms {tag}", (16, oh - 20), 0.7)
        return canvas

    tmpdir = out.parent / f".{out.stem}_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    vid = tmpdir / "video.mp4"
    w = Writer(vid, ow, oh, fps)
    n = int((t_b - t_a) * fps)
    for k in range(n):
        w.write(compose(t_a + k / fps, ""))
    if slowmo > 1:
        s_a, s_b = max(t_a, b - 0.5), min(t_b, b + 0.6)
        for k in range(int((s_b - s_a) * fps * slowmo)):
            w.write(compose(s_a + k / (fps * slowmo), f"({slowmo:g}x slow)"))
    w.close()
    if c.has_audio:
        # phone audio under the real-time part; the slow replay is silent
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-ss", f"{t_a:.3f}", "-t", f"{t_b - t_a:.3f}",
                        "-i", str(c.path), "-map", "0:v", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "160k", "-af", "apad", "-shortest", str(out)], check=True)
    else:
        shutil.copy(vid, out)
    shutil.rmtree(tmpdir, ignore_errors=True)


VIDEO_EXT = {".mov", ".mp4", ".m4v", ".3gp", ".mkv", ".avi", ".webm"}


def cmd_sync(a) -> int:
    presses = load_presses(Path(a.sessions), a.after)
    w0s, w1s = a.window.split("/")
    w0, w1 = parse_time(w0s), parse_time(w1s)
    files = sorted(f for f in Path(a.phone).rglob("*") if f.suffix.lower() in VIDEO_EXT and not f.name.startswith("."))
    clips: list[PhoneClip] = []
    print(f"phone clips in {a.phone}: {len(files)}")
    for f in files:
        c = probe_phone(f)
        if c is None:
            print(f"  skip {f.name}: no video stream")
            continue
        c.ok_window = in_window(c, w0, w1)
        loc = c.start.astimezone(EST) if c.start else None
        print(f"  {f.name}: {loc:%Y-%m-%d %H:%M:%S %Z} ({c.tag}), {c.duration:.1f} s, "
              f"{'IN window' if c.ok_window else 'OUTSIDE window -> skipped'}")
        if not c.ok_window:
            continue
        if c.has_audio:
            c.blasts = find_blasts(f, k_floor=a.blast_k, min_dbfs=a.blast_dbfs)
        print(f"      shots heard at: {', '.join(f'{b:.2f} s ({db:.0f} dBFS)' for b, db in c.blasts) or 'none'}"
              + ("" if c.has_audio else " (no audio track)"))
        clips.append(c)
    sound_s = a.distance_m / SOUND_MPS
    manual = {}
    for spec in a.pair or []:
        clip_name, rest = spec.split("=", 1)
        press_name, _, bt = rest.partition("@")
        manual[press_name.split("-", 1)[1] if press_name.count("-") >= 2 else press_name] = (clip_name, bt)
    res = match_shots(presses, clips, a.tol_s, a.max_skew_s, sound_s, a.end_stamped, a.audible, a.boot_s)
    for pname in manual:
        if not any(r["press"].name == pname for r in res):
            pp = next((p for p in presses if p.name == pname), None)
            if pp is None:
                print(f"--pair: no press {pname}")
                continue
            res.append(dict(press=pp, clip=None, how="--pair"))
    for r in res:
        if r["press"].name in manual:
            cn, bt = manual[r["press"].name]
            c = next((c for c in clips if c.path.name == cn), None) or (probe_phone(Path(a.phone) / cn))
            if c is None:
                print(f"--pair: {cn} not found")
                continue
            if not c.blasts and c.has_audio:
                c.blasts = find_blasts(c.path, k_floor=a.blast_k, min_dbfs=a.blast_dbfs)
            b = float(bt) if bt else (c.blasts[0][0] if c.blasts else None)
            if b is None:
                print(f"--pair {cn}: no blast found; give it as {cn}={r['press'].name}@<seconds>")
                continue
            r.update(clip=c, blast_s=b, offset_s=float("nan"), resid_ms=float("nan"), how="--pair (manual)")
    print("\nshot                  recoil  -> phone clip           blast    Pi-clock skew   resid   how")
    for r in sorted(res, key=lambda r: (r["press"].boot_id, r["press"].uptime_s)):
        p = r["press"]
        if r.get("clip") is None:
            print(f"  {p.name:18s}  {'yes' if p.recoils else 'NO ':3s}     -> (none)   {r['how']}")
            continue
        print(f"  {p.name:18s}  {'yes' if p.recoils else 'NO ':3s}     -> {r['clip'].path.name:20s} "
              f"{r['blast_s']:7.2f} s  {r['offset_s']:+9.1f} s  {r['resid_ms']:+6.0f} ms  {r['how']}")
    if a.dry_run:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for r in res:
        if r.get("clip") is None:
            continue
        p = r["press"]
        dst = out / f"{p.name}__{r['clip'].path.stem}.mp4"
        render_pair(p, r["clip"], r["blast_s"], FrameSource(p, Path(a.clips) if a.clips else None), dst,
                    a.layout, a.pre, a.post, a.slowmo, sound_s)
        made.append(dst)
        print(f"wrote {dst}")
    if made and a.reel:
        widths = [int(next(s for s in ffprobe(m)["streams"] if s["codec_type"] == "video")["width"]) for m in made]
        wmax = max(widths)
        ins, fc, cat = [], "", ""
        for k, m in enumerate(made):
            ins += ["-i", str(m)]
            fc += (f"[{k}:v]scale=-2:720,pad={wmax}:720:(ow-iw)/2:0,fps=30,setsar=1[v{k}];"
                   f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo[a{k}];")
            cat += f"[v{k}][a{k}]"
        fc += f"{cat}concat=n={len(made)}:v=1:a=1[v][a]"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                        *pick_encoder(), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                        "-movflags", "+faststart", str(out / a.reel)], check=True)
        print(f"wrote {out / a.reel}")
    return 0 if made else 1


# ----------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("table", help="markdown table of the presses (fired, recoil, boots)")
    t.add_argument("sessions")
    t.add_argument("--after", help="only press folders whose name sorts at/after this (e.g. 20261119-154600)")
    t.set_defaults(fn=cmd_table)

    r = sub.add_parser("reel", help="highlight reel of the presses that fired")
    r.add_argument("sessions")
    r.add_argument("--clips", help="folder of annotated clips <HHMMSS>-press.mp4 (tbot clip --scale 2)")
    r.add_argument("--out", required=True)
    r.add_argument("--after")
    r.add_argument("--only", nargs="*", help="press names to include (default: every press that fired)")
    r.add_argument("--recoil-only", action="store_true", help="only presses where the IMU saw the recoil")
    r.add_argument("--title", help="opening card, lines separated by |")
    r.add_argument("--fps", type=float, default=30.0)
    r.add_argument("--pre", type=float, default=1.5, help="real-time seconds before the shot")
    r.add_argument("--post", type=float, default=0.7)
    r.add_argument("--slowmo", type=float, default=4.0, help="replay the shot at 1/N speed (1 = off)")
    r.add_argument("--slow-pre", type=float, default=0.4)
    r.add_argument("--slow-post", type=float, default=0.3)
    r.add_argument("--card-s", type=float, default=1.6)
    r.set_defaults(fn=cmd_reel)

    g = sub.add_parser("gif", help="GIF of one press around the shot")
    g.add_argument("press")
    g.add_argument("--clips")
    g.add_argument("--out", required=True)
    g.add_argument("--pre", type=float, default=0.8)
    g.add_argument("--post", type=float, default=0.4)
    g.add_argument("--slowmo", type=float, default=2.0)
    g.add_argument("--fps", type=float, default=15.0)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--colors", type=int, default=96)
    g.add_argument("--caption")
    g.add_argument("--max-mb", type=float, default=8.0)
    g.set_defaults(fn=cmd_gif)

    s = sub.add_parser("sync", help="pair phone clips with Pi presses on the shot and render them together")
    s.add_argument("sessions")
    s.add_argument("--phone", required=True, help="folder of phone clips (searched recursively)")
    s.add_argument("--out", required=True)
    s.add_argument("--clips", help="annotated clips folder (else the raw frames are used)")
    s.add_argument("--after")
    s.add_argument("--window", required=True, help="ISO start/end with offset, e.g. "
                   "2026-11-19T15:40-05:00/2026-11-19T16:30-05:00")
    s.add_argument("--layout", choices=["sbs", "pip"], default="sbs")
    s.add_argument("--pre", type=float, default=2.5, help="seconds of phone video before the shot")
    s.add_argument("--post", type=float, default=1.5)
    s.add_argument("--slowmo", type=float, default=4.0)
    s.add_argument("--tol-s", type=float, default=0.35, help="max shot-to-blast mismatch inside one boot")
    s.add_argument("--max-skew-s", type=float, default=3600.0, help="max Pi-clock error to search")
    s.add_argument("--distance-m", type=float, default=0.0, help="phone to gun, for the sound delay")
    s.add_argument("--end-stamped", action="store_true", help="phone stamps the END of a clip, not the start")
    s.add_argument("--blast-k", type=float, default=25.0, help="blast threshold, x the clip's noise floor")
    s.add_argument("--blast-dbfs", type=float, default=-16.0, help="blast threshold, absolute floor (dBFS)")
    s.add_argument("--pair", action="append", help="force CLIP=PRESS[@seconds] (repeatable)")
    s.add_argument("--audible", choices=["recoil", "fire"], default="recoil",
                   help="which presses made a bang: only those with an IMU recoil (default), or every fire command")
    s.add_argument("--boot-s", type=float, default=5.0, help="power-on to uptime 0, for the boot-order bound")
    s.add_argument("--reel", help="also join every pair into this file name in --out")
    s.add_argument("--dry-run", action="store_true", help="only print the matching")
    s.set_defaults(fn=cmd_sync)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
