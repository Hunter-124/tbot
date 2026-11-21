"""Synthetic clays composited into the RAW frames of a recorded press, at pixel level (`tbot vbench`).

`tbot vclay` flies virtual clays through recorded gun motion but hands the tracker synthetic DETECTIONS: the
pixel stage (detector, scenery filter), where trees and concrete cracks are confused with clays, is skipped.
This module draws the clay into the frames themselves, so whatever front end is configured sees a real
background (foliage, sky, the barrel, the sensor's own noise) with a clay of exactly known position in it.

The clay
  * flies in WORLD coordinates. Two kinds:
      - "follow": built by `tbot.tools.vclay.Bench.make_clay` around where the bore really pointed, so the
        recorded swing looks like a shooter following that clay (gravity, drag and lift from
        `tbot.sim.scenario._clay_accel`);
      - "cross": NOT followed by the gun: launched from a random frame edge at a random moment around the press,
        at 5-60 deg/s across the sky (world), 12-45 m away (at most 30 m/s), and flown with the same physics
        while the gun does whatever it did (pans across trees, sits still). Acquisition is tested as well.
  * is rendered with the ray-cast model of `tbot.camera.synthetic` (`_disc_coverage`: an opaque short cylinder
    108 mm x 25 mm, true face-on / edge-on geometry, the disc nearly level (banked up to 25 deg, pitched up to
    15 deg); 4x4 supersampling; motion blur over the frame's real
    exposure_us, the camera's own rotation during the exposure included) through the press's camera model and
    attitude (`tbot.tools.score.Session`: the whole IMU record), plus what the sim lacks: an optical PSF
    (Gaussian, sigma 0.6 px at 640x400) and compositing over the LOCAL background,
    pixel = (1 - cov) * bg + cov * clay_level (+ cov * shot noise of 1-2 DN: the frame already has its own).
    Masked pixels (barrel, static mask: `tbot.optics.combined_mask`) are never drawn.
  * has a brightness drawn per clay ("polarity"): dark (0.25-0.6 x the reference level), bright sunlit
    (1.3-2.0 x, clipped at 255) or two-tone (the upper half bright, the lower half dark: a sunlit dome).

Measured on real clays (2026-11-20, telemetry `dets` + the frames; core = the inner half of the detection,
bg = the 31x31 patch around it outside the detection):
  * 2026-11-19 afternoon (exposure 497 us, gain 1.4-3.2), sunlit clays seen against dark foliage, presses
    160243 (14 m at the shot), 160333 (14 m), 160357 (24 m), the frames before "fire" on the solver's target:
    detector diameter 12-21 px; bbox 10-18 x 4-10 px (seen nearly edge-on: aspect 2-3:1); contrast
    +122..+193 DN, snr 10-28; core mean 230-250 DN (clipped at 255 in most frames); background median 50-98 DN
    with a local std of 8-42 DN (foliage). Core / background median 2.5-4.9.
  * 2026-11-20 morning, press 072712 frames 110-180 (exposure 9 us, gain 1.0), sky median 135-170 DN with a
    local std of only 1.7-3.4 DN: a small dark object crossing the sky (bbox 10 x 6-8 px, contrast -63..-79 DN,
    core mean 0.57-0.69 and core minimum 0.40-0.55 x the sky median, snr 40-50), and the ball (12 x 10 px at
    frame 130, contrast -107, sunlit top / dark bottom); once it is in front of dense foliage (background
    100-115 DN, std 15-30) the legacy detector reports 0-1 detections a frame and loses it.
So the brightness of a clay is not a fixed ratio to whatever is behind it: a sunlit clay is ~240 DN whether
the foliage behind it is at 50 or 100, and a dark one is ~0.5 x the SKY even when it is in front of trees. The
default reference level is therefore the scene's (`level_ref="scene"`: the 90th percentile of the unmasked
pixels of the original frame, which is the sky or the sunlit leaves the auto-exposure was set for);
`level_ref="local"` uses the median of the 31x31 patch around the clay instead (the clay then always keeps
its contrast to what is behind it, which flatters a detector in front of trees).

Truth: for every row of frames.csv, per clay, the true pixel (projection of the centre at mid-exposure), the
projected major / minor size in px, range, whether it is in the image / masked, the local background of the
ORIGINAL frame (31x31 median and std: sky below ~6 DN), and `pos(t_ns)` (world, camera at the origin, the
Session's world frame) for hit scoring. Deterministic from the seed; the press files are only read (frames.bin
is memory-mapped read-only and a frame is copied before anything is drawn on it).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from tbot.camera.replay import FRAMES_BIN, FRAMES_CSV, META_JSON, ReplaySource
from tbot.camera.synthetic import _disc_coverage, _unit_rows
from tbot.config import Config
from tbot.sim.scenario import CLAY_THICKNESS_M
from tbot.tools import vclay

UP = np.array([0.0, 0.0, 1.0])
PSF_SIGMA_PX = 0.6
SUPERSAMPLE = 4
MAX_SUBSTEPS = 6
POLARITIES = (("dark", 0.40), ("bright", 0.35), ("twotone", 0.25))
DARK_K = (0.25, 0.60)
BRIGHT_K = (1.3, 2.0)
NOISE_DN = (1.0, 2.0)
CROSS_RATE_DPS = (5.0, 60.0)
CROSS_RANGE_M = (12.0, 45.0)
CROSS_MAX_SPEED_MPS = 30.0
CROSS_MAX_CLIMB_DEG = 30.0   # flight path at most this steep, up or down (no elevation or ground rule: on
                             # several presses the IMU's world is tilted by ~10 deg against the true horizon)
BANK_DEG = 25.0              # the disc is banked about its heading by up to this much, either way
PITCH_DEG = 15.0             # ... and pitched nose up or down by up to this much
SKY_STD_DN = 6.0             # 31x31 std of the original frame below this: sky; else textured
PATCH = 31
MIN_VISIBLE_ARMED = 3        # a clay must be seen in at least this many armed frames to be used


# ------------------------------------------------------------------------------------------------ the press
def press_ok(path: str | Path) -> str | None:
    """None if the press can be used; else why not (incomplete, still being written, no IMU...)."""
    p = Path(path)
    try:
        meta = json.loads((p / META_JSON).read_text())
    except Exception as exc:  # noqa: BLE001
        return f"no readable meta.json ({exc.__class__.__name__})"
    if not meta.get("complete"):
        return "meta.json not complete"
    try:
        n, h, w = int(meta["frames"]), int(meta["height"]), int(meta["width"])
        ch = int(meta.get("channels", 1))
    except (KeyError, TypeError, ValueError):
        return "meta.json without frames/width/height"
    if ch != 1:
        return "not a mono press"
    b = p / FRAMES_BIN
    if not b.exists() or b.stat().st_size != n * h * w:
        return f"frames.bin is not {n} frames"
    for f in (FRAMES_CSV, "imu.csv", "events.jsonl", "telemetry.jsonl"):
        if not (p / f).exists() or (p / f).stat().st_size == 0:
            return f"no {f}"
    return None


def arming_intervals(events: list[dict], t_last: int) -> list[tuple[int, int]]:
    """(armed, released) source-clock times of each hold of the button in a press."""
    out, on = [], None
    for t, armed in sorted((int(e["t_ns"]), bool(e.get("armed"))) for e in events if e.get("type") == "arming"):
        if armed and on is None:
            on = t
        elif not armed and on is not None:
            out.append((on, t))
            on = None
    if on is not None:
        out.append((on, int(t_last)))
    return out


class Stage:
    """A recorded press as a stage for injected clays: `vclay.Press` (true attitude from the whole IMU record,
    camera, mount, barrel mask) with its armed window taken from the longest hold of the button, and the frame
    rows of frames.csv."""

    def __init__(self, path: str | Path, cfg: Config) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.press = vclay.Press(self.path, cfg)
        s = self.press.session
        self.cam, self.mount, self.R_gc = s.cam, s.mount, s.R_gc
        self.size = s.size
        w, h = self.size
        self.mask = (np.asarray(self.press.mask) if self.press.mask is not None
                     else np.full((h, w), 255, dtype=np.uint8))
        with open(self.path / FRAMES_CSV, newline="") as f:
            r = csv.reader(f)
            next(r)
            rows = [(int(x[1]), float(x[3])) for x in r if x]
        n_bin = (self.path / FRAMES_BIN).stat().st_size // (w * h)
        rows = rows[: int(n_bin)]
        self.t_exp = np.array([t for t, _ in rows], dtype=np.int64)
        self.exposure_us = np.array([e for _, e in rows], dtype=np.float64)
        self.n = len(rows)
        imu_end = int(s.imu_span[1])
        holds = arming_intervals(s.events, imu_end)
        recoil = [int(e["t_ns"]) for e in s.events if e.get("type") == "recoil"]
        if holds:
            t_arm, t_rel = max(holds, key=lambda x: x[1] - x[0])
        else:
            t_arm, t_rel = int(self.t_exp[0]), imu_end
        self.holds = holds
        self.t_arm, self.t_release = t_arm, t_rel
        # the swing after a real shot is recoil, not aiming (as vclay.Press)
        self.t_end = min([t_rel, imu_end] + [t - int(5e6) for t in recoil if t > t_arm])
        self.press.t_arm, self.press.t_release, self.press.t_end = self.t_arm, self.t_release, self.t_end

    @property
    def armed_s(self) -> float:
        return max(0.0, (self.t_end - self.t_arm) * 1e-9)

    def armed(self, t_ns: int) -> bool:
        return any(a <= t_ns <= b for a, b in self.holds)

    def R_wc(self, t_ns: int) -> np.ndarray:
        return self.press.R(int(t_ns)) @ self.R_gc

    def pixel(self, p_w: np.ndarray, t_ns: int) -> tuple[float, float] | None:
        ray_c = self.R_wc(t_ns).T @ np.asarray(p_w, dtype=np.float64)
        if ray_c[2] <= 1e-6 * max(1.0, float(np.linalg.norm(ray_c))):
            return None
        uv = np.asarray(self.cam.ray_to_pixel(ray_c / np.linalg.norm(ray_c)), dtype=np.float64).reshape(2)
        if not np.isfinite(uv).all():
            return None
        return float(uv[0]), float(uv[1])

    def in_image(self, uv) -> bool:
        w, h = self.size
        return uv is not None and 0.0 <= uv[0] <= w - 1.0 and 0.0 <= uv[1] <= h - 1.0

    def masked(self, uv) -> bool:
        return not self.mask[int(round(uv[1])), int(round(uv[0]))]


# ------------------------------------------------------------------------------------------------ the clay
@dataclass
class InjectedClay:
    """One injected clay: its world path (a vclay.VClay: `pos(t_ns)`), its disc and how it looks."""

    id: str
    kind: str                     # "follow" | "cross"
    cls: str                      # vclay speed class, or "cross"
    polarity: str                 # "dark" | "bright" | "twotone"
    path: vclay.VClay
    axis_w: np.ndarray
    k_top: float                  # brightness / reference level (upper half; the whole clay unless two-tone)
    k_bot: float
    noise_dn: float
    seed: tuple = ()
    diameter_m: float = 0.108
    thickness_m: float = CLAY_THICKNESS_M

    @property
    def visible_from_ns(self) -> int:
        return int(self.path.visible_from_ns)

    def pos(self, t_ns: int) -> np.ndarray | None:
        """World position (camera at the origin) once the clay is in the air; None before or after its path."""
        if int(t_ns) < self.visible_from_ns:
            return None
        return self.path.pos(int(t_ns))

    def meta(self) -> dict:
        return {"id": self.id, "kind": self.kind, "cls": self.cls, "polarity": self.polarity,
                "range_m": round(float(self.path.range_m), 2), "speed_mps": round(float(self.path.speed_mps), 2),
                "rate_dps": round(float(self.path.rate_dps), 2), "k_top": round(self.k_top, 3),
                "k_bot": round(self.k_bot, 3), "noise_dn": round(self.noise_dn, 2),
                "visible_from_ns": self.visible_from_ns}


def _disc_axis(vel: np.ndarray, bank_deg: float, pitch_deg: float) -> np.ndarray:
    """Disc normal: world up, banked about the clay's horizontal heading and pitched along it. A spinning clay
    keeps its attitude: it stays nearly level whether its path climbs or falls (not perpendicular to the
    velocity, as tbot.sim.scenario._disc_axis makes it at launch: on a steep path that shows the clay
    face-on, where the real clays of 2026-11-19 were seen edge-on, 2-3:1)."""
    h = np.array([float(vel[0]), float(vel[1]), 0.0])
    n = float(np.linalg.norm(h))
    h = h / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    side = np.cross(UP, h)
    axis = UP + math.tan(math.radians(bank_deg)) * side - math.tan(math.radians(pitch_deg)) * h
    return axis / np.linalg.norm(axis)


def _look(rng: np.random.Generator) -> tuple[str, float, float, float]:
    names = [p for p, _ in POLARITIES]
    probs = np.array([w for _, w in POLARITIES], dtype=np.float64)
    pol = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    dark, bright = float(rng.uniform(*DARK_K)), float(rng.uniform(*BRIGHT_K))
    if pol == "dark":
        top = bot = dark
    elif pol == "bright":
        top = bot = bright
    else:
        top, bot = bright, dark
    return pol, top, bot, float(rng.uniform(*NOISE_DN))


def _visible_armed(stage: Stage, path: vclay.VClay) -> int:
    n = 0
    for t in stage.t_exp:
        t = int(t)
        if t < path.visible_from_ns or not (stage.t_arm <= t <= stage.t_end):
            continue
        p = path.pos(t)
        if p is None:
            continue
        uv = stage.pixel(p, t)
        if stage.in_image(uv) and not stage.masked(uv):
            n += 1
    return n


def make_follow(stage: Stage, bench: vclay.Bench, rng: np.random.Generator, cls: str) -> vclay.VClay | None:
    """A clay the recorded swing follows (vclay.Bench.make_clay), seen in enough armed frames."""
    for _ in range(6):
        c = bench.make_clay(stage.press, rng, cls)
        if c is not None and _visible_armed(stage, c) >= MIN_VISIBLE_ARMED:
            return c
    return None


def make_cross(stage: Stage, rng: np.random.Generator) -> vclay.VClay | None:
    """A clay the gun does not follow: in from a random frame edge, 5-60 deg/s across the sky."""
    w, h = stage.size
    lo = max(int(stage.t_exp[0]), stage.t_arm - int(0.3e9))
    hi = stage.t_end - int(0.5e9)
    if hi <= lo:
        hi = stage.t_end - int(0.15e9)
    if hi <= lo:
        return None
    for _ in range(80):
        t0 = int(rng.integers(lo, hi))
        rate = float(rng.uniform(*CROSS_RATE_DPS))
        w_rad = math.radians(rate)
        r_hi = min(CROSS_RANGE_M[1], CROSS_MAX_SPEED_MPS / w_rad)
        r = float(rng.uniform(CROSS_RANGE_M[0], max(CROSS_RANGE_M[0] + 0.5, r_hi)))
        edge = int(rng.integers(4))
        f = float(rng.uniform(0.08, 0.92))
        inset = 1.0
        u, v, normal = {0: (inset, f * (h - 1), 0.0), 1: (w - 1 - inset, f * (h - 1), math.pi),
                        2: (f * (w - 1), inset, 0.5 * math.pi), 3: (f * (w - 1), h - 1 - inset, -0.5 * math.pi)}[edge]
        ang = normal + math.radians(float(rng.uniform(-60.0, 60.0)))
        R = stage.R_wc(t0)
        ray0 = np.asarray(stage.cam.pixel_to_ray(np.array([u, v], dtype=np.float64)), dtype=np.float64).reshape(3)
        ray1 = np.asarray(stage.cam.pixel_to_ray(np.array([u + 4.0 * math.cos(ang), v + 4.0 * math.sin(ang)],
                                                          dtype=np.float64)), dtype=np.float64).reshape(3)
        d0 = vclay._unit(R @ ray0)
        d1 = vclay._unit(R @ ray1)
        tan = d1 - float(d1 @ d0) * d0
        if float(np.linalg.norm(tan)) < 1e-9:
            continue
        tan = vclay._unit(tan)
        v_t = w_rad * r * tan
        radial = float(rng.uniform(-0.4, 0.4)) * float(np.linalg.norm(v_t))
        vel = v_t + radial * d0
        if abs(float(vel[2])) > math.sin(math.radians(CROSS_MAX_CLIMB_DEG)) * float(np.linalg.norm(vel)):
            continue  # clays fly across, not straight up or down
        p0 = r * d0
        n = int(math.ceil((stage.t_end + int(0.8e9) - t0) * 1e-9 / vclay.PATH_STEP_S)) + 1
        path = vclay.Bench._integrate(p0, vel, max(n, 2), vclay.PATH_STEP_S)
        c = vclay.VClay(t0_ns=t0, path=path, t_plan_exit_ns=t0, range_m=r, speed_mps=float(np.linalg.norm(vel)),
                        rate_dps=rate, offset_deg=float("nan"), cls="cross", visible_from_ns=t0)
        if _visible_armed(stage, c) >= MIN_VISIBLE_ARMED:
            return c
    return None


def make_clays(stage: Stage, bench: vclay.Bench, n: int, seed: int, press_key: int,
               cross_every: int = 2) -> list[InjectedClay]:
    """n clays for one press, each for a run of its own; deterministic from (seed, press_key, n).
    Every `cross_every`-th clay (offset by the press, so both kinds are spread over presses) is a crossing one;
    the rest are followed, cycling the vclay speed classes. A follow that cannot be built becomes a crossing."""
    out = []
    classes = tuple(vclay.SPEED_CLASSES)
    for k in range(int(n)):
        rng = np.random.default_rng([int(seed), int(press_key), k])
        kind = "cross" if cross_every > 0 and (k + press_key) % cross_every == 1 else "follow"
        cls = classes[(k + press_key) % len(classes)]
        path = make_follow(stage, bench, rng, cls) if kind == "follow" else None
        if path is None:
            kind, cls = "cross", "cross"
            path = make_cross(stage, rng)
        if path is None:
            continue
        p_a = path.pos(path.visible_from_ns)
        p_b = path.pos(path.visible_from_ns + int(0.02e9))
        vel = (p_b - p_a) / 0.02 if p_a is not None and p_b is not None else np.array([1.0, 0.0, 0.0])
        axis = _disc_axis(vel, float(rng.uniform(-BANK_DEG, BANK_DEG)), float(rng.uniform(-PITCH_DEG, PITCH_DEG)))
        pol, top, bot, noise = _look(rng)
        out.append(InjectedClay(id=f"inj{k}", kind=kind, cls=cls, polarity=pol, path=path, axis_w=axis,
                                k_top=top, k_bot=bot, noise_dn=noise, seed=(int(seed), int(press_key), k),
                                diameter_m=float(stage.cfg.mount.clay_diameter_m)))
    return out


# ------------------------------------------------------------------------------------------------ the source
@dataclass
class TruthRow:
    row: int
    t_ns: int
    exists: bool = False          # in the air (and on its path) at this frame
    u: float = float("nan")
    v: float = float("nan")
    in_image: bool = False
    masked: bool = False
    major_px: float = float("nan")
    minor_px: float = float("nan")
    range_m: float = float("nan")
    bg_med: float = float("nan")
    bg_std: float = float("nan")
    drawn: bool = False

    @property
    def visible(self) -> bool:
        return self.exists and self.in_image and not self.masked

    def as_list(self) -> list:
        f = lambda x: None if not math.isfinite(x) else round(float(x), 3)  # noqa: E731
        return [self.row, self.t_ns, int(self.exists), f(self.u), f(self.v), int(self.in_image), int(self.masked),
                f(self.major_px), f(self.minor_px), f(self.range_m), f(self.bg_med), f(self.bg_std)]


TRUTH_FIELDS = ["row", "t_ns", "exists", "u", "v", "in_image", "masked", "major_px", "minor_px", "range_m",
                "bg_med", "bg_std"]


def _patch_stats(img: np.ndarray, u: float, v: float) -> tuple[float, float]:
    h, w = img.shape[:2]
    r = PATCH // 2
    x, y = int(round(u)), int(round(v))
    p = img[max(0, y - r): min(h, y + r + 1), max(0, x - r): min(w, x + r + 1)]
    if p.size == 0:
        return float("nan"), float("nan")
    p = p.astype(np.float32)
    return float(np.median(p)), float(p.std())


class InjectedSource(ReplaySource):
    """ReplaySource of a press with clays composited into its frames (see the module docstring).

    `truth[clay.id]` is a list of TruthRow, one per row of frames.csv. `last_row` is the row of the frame the
    last read() returned."""

    def __init__(self, path: str | Path, clays: list[InjectedClay], stage: Stage, work_ms: float = 0.0,
                 psf_sigma: float = PSF_SIGMA_PX, level_ref: str = "scene", supersample: int = SUPERSAMPLE) -> None:
        super().__init__(path, realtime=False, work_ms=work_ms)
        if level_ref not in ("scene", "local"):
            raise ValueError(f"level_ref must be 'scene' or 'local', got {level_ref!r}")
        self.clays = list(clays)
        self.stage = stage
        self.psf_sigma = float(psf_sigma)
        self.level_ref = level_ref
        self.ss = int(supersample)
        self.last_row: int | None = None
        self._mask = stage.mask
        self.truth: dict[str, list[TruthRow]] = {c.id: self._truth(c) for c in self.clays}

    # ----------------------------------------------------------------------------- truth
    def _truth(self, c: InjectedClay) -> list[TruthRow]:
        st = self.stage
        out = []
        for i in range(self.n_frames):
            t = int(self._rows[i][1])
            tr = TruthRow(row=i, t_ns=t)
            out.append(tr)
            p = c.pos(t)
            if p is None:
                continue
            tr.exists = True
            rng_m = float(np.linalg.norm(p))
            tr.range_m = rng_m
            uv = st.pixel(p, t)
            if uv is None:
                continue
            tr.u, tr.v = uv
            tr.in_image = st.in_image(uv)
            if not tr.in_image:
                continue
            tr.masked = st.masked(uv)
            ifov = float(st.cam.ifov1(uv[0], uv[1]))
            los = p / rng_m
            ca = abs(float(c.axis_w @ los))
            tr.major_px = c.diameter_m / rng_m / ifov
            tr.minor_px = (c.diameter_m * ca + c.thickness_m * math.sqrt(max(0.0, 1.0 - ca * ca))) / rng_m / ifov
            tr.bg_med, tr.bg_std = _patch_stats(np.asarray(self._frames[i]), uv[0], uv[1])
        return out

    def truth_objects(self) -> list[dict]:
        """The truth in the schema shared with bench_data/truth/<press>.json: {"id", "label", "frames":
        [[row, u, v, w_px, h_px], ...]} over the frames in which the clay is visible (in the image, unmasked)."""
        out = []
        for c in self.clays:
            fr = [[t.row, round(t.u, 2), round(t.v, 2), round(t.major_px, 2), round(t.minor_px, 2)]
                  for t in self.truth[c.id] if t.visible]
            out.append({"id": c.id, "label": f"injected:{c.kind}/{c.polarity}", "frames": fr})
        return out

    # ----------------------------------------------------------------------------- frames
    def read(self, timeout_s: float = 1.0):
        i = self._index
        frame = super().read(timeout_s)
        if frame is None:
            return None
        self.last_row = i
        if any(self.truth[c.id][i].exists for c in self.clays):
            frame.image = self.render(i)
        return frame

    def original(self, i: int) -> np.ndarray:
        return np.asarray(self._frames[i])

    def render(self, i: int) -> np.ndarray:
        """Frame row i with every clay in the air drawn into it (a copy; the press is never written)."""
        orig = np.asarray(self._frames[i])
        img = orig.astype(np.float32)  # a copy
        ref = None
        for c in self.clays:
            if not self.truth[c.id][i].exists:
                continue
            if ref is None:
                sub = orig[::4, ::4][self._mask[::4, ::4] > 0]
                ref = float(np.percentile(sub, 90)) if sub.size else 128.0
            if self._draw(img, orig, i, c, ref):
                self.truth[c.id][i].drawn = True
        np.clip(np.rint(img), 0.0, 255.0, out=img)
        return img.astype(np.uint8)

    def _draw(self, img: np.ndarray, orig: np.ndarray, i: int, c: InjectedClay, scene_ref: float) -> bool:
        st = self.stage
        t_mid = int(self._rows[i][1])
        exp_ns = max(0, int(round(float(self._rows[i][3]) * 1e3)))
        h, w = img.shape[:2]
        radius_m = 0.5 * math.hypot(c.diameter_m, c.thickness_m)
        # motion over the exposure -> number of sub-steps (camera rotation included: pixel() uses R(t))
        ends = [st.pixel(p, t) for t in (t_mid - exp_ns // 2, t_mid + exp_ns // 2)
                for p in [c.pos(t)] if p is not None]
        motion = math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]) if len(ends) == 2 and None not in ends else 0.0
        n_sub = int(min(MAX_SUBSTEPS, max(1, math.ceil(motion / 0.5))))
        offs = [int(round(exp_ns * ((k + 0.5) / n_sub - 0.5))) for k in range(n_sub)]
        centres = []
        for dt in offs:
            p = c.pos(t_mid + dt)
            if p is None:
                continue
            uv = st.pixel(p, t_mid + dt)
            if uv is None:
                continue
            ifov = float(st.cam.ifov1(min(max(uv[0], 0.0), w - 1.0), min(max(uv[1], 0.0), h - 1.0)))
            if not math.isfinite(ifov) or ifov <= 0.0:
                continue
            centres.append((uv[0], uv[1], math.atan2(radius_m, float(np.linalg.norm(p))) / ifov))
        if not centres:
            return False
        m = int(math.ceil(3.0 * self.psf_sigma)) + 2
        x0 = max(int(math.floor(min(a - r for a, _, r in centres))) - m, 0)
        x1 = min(int(math.ceil(max(a + r for a, _, r in centres))) + m + 1, w)
        y0 = max(int(math.floor(min(b - r for _, b, r in centres))) - m, 0)
        y1 = min(int(math.ceil(max(b + r for _, b, r in centres))) + m + 1, h)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > 250000:
            return False
        pw, ph, s = x1 - x0, y1 - y0, self.ss
        us = x0 - 0.5 + (np.arange(pw * s, dtype=np.float64) + 0.5) / s
        vs = y0 - 0.5 + (np.arange(ph * s, dtype=np.float64) + 0.5) / s
        uu, vv = np.meshgrid(us, vs)
        rays_c = _unit_rows(st.cam.pixel_to_ray(np.stack([uu.ravel(), vv.ravel()], axis=1)))
        cov = np.zeros(rays_c.shape[0], dtype=np.float64)
        top = np.zeros(rays_c.shape[0], dtype=np.float64)
        used = 0
        for dt in offs:
            tk = t_mid + dt
            P = c.pos(tk)
            if P is None:
                continue
            rays_w = rays_c @ st.R_wc(tk).T
            hit = _disc_coverage(rays_w, P, c.axis_w, 0.5 * c.diameter_m, c.thickness_m)
            cov += hit
            if c.k_top != c.k_bot:
                # the upper half of the clay (in the world: above its centre as seen from the camera)
                top += hit * (rays_w[:, 2] > float(P[2]) / float(np.linalg.norm(P)))
            used += 1
        if used == 0 or not cov.any():
            return False
        cov = (cov / used).reshape(ph, s, pw, s).mean(axis=(1, 3))
        top = (top / used).reshape(ph, s, pw, s).mean(axis=(1, 3))
        if self.psf_sigma > 0.0:
            cov = cv2.GaussianBlur(cov, (0, 0), self.psf_sigma, borderType=cv2.BORDER_CONSTANT)
            top = cv2.GaussianBlur(top, (0, 0), self.psf_sigma, borderType=cv2.BORDER_CONSTANT)
        cov = np.clip(cov, 0.0, 1.0)
        top = np.clip(top, 0.0, cov)
        bg = img[y0:y1, x0:x1]
        if self.level_ref == "local":
            u_c, v_c = centres[len(centres) // 2][:2]
            ref = _patch_stats(orig, u_c, v_c)[0]
        else:
            ref = scene_ref
        l_top = min(255.0, c.k_top * ref)
        l_bot = min(255.0, c.k_bot * ref)
        clay = top * l_top + (cov - top) * l_bot
        rng = np.random.default_rng(list(c.seed) + [7, int(i)])
        noise = rng.standard_normal(cov.shape) * c.noise_dn * cov
        new = (1.0 - cov) * bg + clay + noise
        keep = self._mask[y0:y1, x0:x1] > 0
        bg[...] = np.where(keep, new, bg).astype(np.float32)
        return True


# ------------------------------------------------------------------------------------------------ helpers
def build(press: str | Path, cfg: Config, n_clays: int, seed: int, press_key: int, latency: dict | None = None,
          cross_every: int = 2) -> tuple[Stage, vclay.Bench, list[InjectedClay]]:
    """Stage, vclay Bench (for the hit criterion) and the clays of one press. `cfg` is the rig config of the
    press (tbot.tools.score.rig_config)."""
    stage = Stage(press, cfg)
    if latency is None:
        latency = (stage.press.session.meta.get("calibration") or {}).get("latency")
    bench = vclay.Bench(cfg, jitter_ms=0.0, latency=latency)
    return stage, bench, make_clays(stage, bench, n_clays, seed, press_key, cross_every=cross_every)
