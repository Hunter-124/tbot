"""Simulated HITL bench (`tbot hitl run --sim`): page emulator, a camera that films it, and the dry-fire click.

Everything runs in real time on `time.monotonic_ns()`, so the real pipeline's processing latency is part of
the result, and the analysis has to recover the injected constants (`SimTruth`) from the same evidence it
uses on hardware: the page's frame log, the camera frames and the microphone stream.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from tbot.camera.base import SourceBase
from tbot.hitl.analysis import SPEED_OF_SOUND_MPS
from tbot.hitl.geometry import ScreenMap
from tbot.hitl.scene import Disc, HitlStore, Rect, active
from tbot.types import Frame


@dataclass
class SimTruth:
    """The constants the simulated bench injects; a correct analysis reports these back."""

    refresh_hz: float = 180.0
    display_latency_ms: float = 12.0  # rAF timestamp -> photon at the top row
    scanout_ms: float = 4.0  # top row -> bottom row
    camera_delivery_ms: float = 6.0  # exposure end -> frame in user space
    camera_jitter_ms: float = 0.3
    lock_time_ms: float = 6.0  # trigger break -> firing-pin click
    mic_latency_ms: float = 2.0  # click -> sample arrival (minimum)
    screen_distance_m: float = 0.9
    screen_width_m: float = 0.53

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------ display
class SimDisplay:
    """Emulates the target page: one frame per refresh, the frame log posted to the store in batches."""

    def __init__(self, store: HitlStore, truth: SimTruth, width: int = 1920, height: int = 1080,
                 post_every_s: float = 0.25):
        self.store = store
        self.truth = truth
        self.width = int(width)
        self.height = int(height)
        self.period_ns = int(round(1e9 / truth.refresh_hz))
        self.latency_ns = int(round(truth.display_latency_ms * 1e6))
        self.scan_ns = truth.scanout_ms * 1e6
        self.post_every_ns = int(post_every_s * 1e9)
        self._lock = threading.Lock()
        self._frames: list[tuple[int, int, tuple]] = []  # (t_raf_ns, bg, primitives)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hitl-sim-display", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _primitives(self, t_ms: float) -> tuple:
        prims = []
        for it in self.store.items():
            if not active(it, t_ms):
                continue
            if isinstance(it, Disc):
                x, y = it.pos(t_ms)
                prims.append((it.id, "disc", x, y, it.r, it.level))
            elif isinstance(it, Rect):
                prims.append((it.id, "rect", it.x, it.y, it.w, it.h, it.level))
        return tuple(prims)

    def _status(self) -> dict:
        return {"w": self.width, "h": self.height, "dpr": 1.0, "fullscreen": True, "visible": True,
                "synced": True, "n_sync": 1, "rtt_min_ms": 0.0, "rtt_p50_ms": 0.0, "offset_ms": 0.0,
                "drift_ppm": 0.0, "refresh_hz": self.truth.refresh_hz, "frame_ms_p50": self.period_ns / 1e6,
                "frame_ms_max": self.period_ns / 1e6, "late_frames": 0, "ua": "tbot-sim-display"}

    def _run(self) -> None:
        t0 = time.monotonic_ns()
        k = 0
        batch: list[dict] = []
        next_post = t0
        while not self._stop.is_set():
            t_raf = t0 + k * self.period_ns
            wait = (t_raf - time.monotonic_ns()) / 1e9
            if wait > 0 and self._stop.wait(wait):
                break
            t_ms = self.store.rig_ms(t_raf)
            prims = self._primitives(t_ms)
            with self._lock:
                self._frames.append((t_raf, self.store.bg, prims))
                if len(self._frames) > 20000:
                    del self._frames[:10000]
            batch.append({"i": k, "t": t_ms, "ids": [p[0] for p in prims]})
            if t_raf >= next_post:
                self.store.post_log({"frames": batch, "status": self._status()}, time.monotonic_ns())
                batch = []
                next_post = t_raf + self.post_every_ns
            k += 1

    def photon_ns(self, t_raf_ns: int, y: float) -> float:
        return t_raf_ns + self.latency_ns + self.scan_ns * min(max(y / self.height, 0.0), 1.0)

    def exposure_segments(self, a_ns: int, b_ns: int) -> tuple[float, list[tuple[float, tuple]]]:
        """Time-weighted screen content during [a, b): (mean background level, [(weight, primitive)]).

        Each primitive's visibility uses the photon times at its own row (scan-out)."""
        with self._lock:
            frames = [f for f in self._frames
                      if f[0] + self.latency_ns <= b_ns and f[0] + self.latency_ns + self.scan_ns + 2 * self.period_ns >= a_ns]
            nxt = {i: (frames[i + 1][0] if i + 1 < len(frames) else None) for i in range(len(frames))}
        span = float(b_ns - a_ns)
        bg_w = 0.0
        bg_sum = 0.0
        out: list[tuple[float, tuple]] = []
        for i, (t_raf, bg, prims) in enumerate(frames):
            t_next = nxt[i]
            mid_lo = self.photon_ns(t_raf, self.height / 2)
            mid_hi = self.photon_ns(t_next, self.height / 2) if t_next is not None else math.inf
            w = max(0.0, min(b_ns, mid_hi) - max(a_ns, mid_lo))
            bg_sum += w * bg
            bg_w += w
            for p in prims:
                y = p[3] if p[1] == "disc" else p[3] + 0.5 * p[5]
                lo = self.photon_ns(t_raf, y)
                hi = self.photon_ns(t_next, y) if t_next is not None else math.inf
                wp = max(0.0, min(b_ns, hi) - max(a_ns, lo)) / span
                if wp > 0.0:
                    out.append((wp, p))
        bg_level = bg_sum / bg_w if bg_w > 0 else float(self.store.bg)
        return bg_level, out


# ------------------------------------------------------------------------------------------- camera
class SimScreenCamera(SourceBase):
    """Global-shutter camera filming the simulated screen: exposure-averaged, blurred, noisy, real time."""

    def __init__(self, cfg, display: SimDisplay, smap: ScreenMap, truth: SimTruth, *, seed: int = 0,
                 room_dn: float = 10.0, black_dn: float = 22.0, white_dn: float = 225.0, noise_sigma: float = 1.5,
                 blur_sigma_px: float = 0.6):
        super().__init__()
        self.width, self.height = int(smap.cam.width), int(smap.cam.height)
        self.is_color = False
        self.nominal_fps = float(cfg.camera.fps)
        self.name = "sim-screen"
        self.display = display
        self.smap = smap
        self.exposure_us = float(cfg.camera.exposure_us)
        self.period_ns = int(round(1e9 / self.nominal_fps))
        self.exposure_ns = int(round(self.exposure_us * 1e3))
        self.delivery_ns = int(round(truth.camera_delivery_ms * 1e6))
        self.jitter_ns = truth.camera_jitter_ms * 1e6
        self.room_dn, self.black_dn, self.white_dn = float(room_dn), float(black_dn), float(white_dn)
        self.blur = float(blur_sigma_px)
        self._rng = np.random.default_rng(seed)
        self._noise = [self._rng.normal(0.0, noise_sigma, (self.height, self.width)).astype(np.float32)
                       for _ in range(8)]
        uu, vv = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        rays = np.asarray(smap.cam.pixel_to_ray(np.column_stack([uu.ravel(), vv.ravel()])))
        xy = smap.from_ray(rays)
        xy[rays[:, 2] <= 0] = -1e6
        self._sx = xy[:, 0].reshape(self.height, self.width)
        self._sy = xy[:, 1].reshape(self.height, self.width)
        gx = np.hypot(*np.gradient(self._sx))
        gy = np.hypot(*np.gradient(self._sy))
        self._scale = np.clip(0.5 * (gx + gy), 1e-3, None)  # screen px per camera px
        sw, sh = smap.screen_w, smap.screen_h
        edge = np.minimum.reduce([self._sx, sw - self._sx, self._sy, sh - self._sy])
        self._screen_cov = np.clip(edge / self._scale + 0.5, 0.0, 1.0).astype(np.float32)
        self._cond = threading.Condition()
        self._slot: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0

    def _dn(self, level: float) -> float:
        return self.black_dn + (self.white_dn - self.black_dn) * float(level) / 255.0

    def _window(self, u: float, v: float, rad: float) -> tuple[slice, slice] | None:
        x0, x1 = int(math.floor(u - rad)), int(math.ceil(u + rad)) + 1
        y0, y1 = int(math.floor(v - rad)), int(math.ceil(v + rad)) + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.width, x1), min(self.height, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return slice(y0, y1), slice(x0, x1)

    def render(self, a_ns: int, b_ns: int) -> np.ndarray:
        bg_level, segs = self.display.exposure_segments(a_ns, b_ns)
        bg_dn = self._dn(bg_level)
        img = self.room_dn + self._screen_cov * np.float32(bg_dn - self.room_dn)
        for w, p in segs:
            if p[1] == "disc":
                _, _, x, y, r, level = p
                uv = self.smap.to_pixel([[x, y]])[0]
                if not np.all(np.isfinite(uv)):
                    continue
                s = float(self._scale[int(np.clip(uv[1], 0, self.height - 1)), int(np.clip(uv[0], 0, self.width - 1))])
                win = self._window(uv[0], uv[1], r / s + 3.0)
                if win is None:
                    continue
                d = np.hypot(self._sx[win] - x, self._sy[win] - y)
                cov = np.clip((r - d) / self._scale[win] + 0.5, 0.0, 1.0)
            else:
                _, _, x, y, rw, rh, level = p
                corners = self.smap.to_pixel([[x, y], [x + rw, y], [x, y + rh], [x + rw, y + rh]])
                if not np.all(np.isfinite(corners)):
                    continue
                lo, hi = corners.min(axis=0), corners.max(axis=0)
                pad = 0.1 * float(np.max(hi - lo)) + 3.0
                win = self._window(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * float(np.max(hi - lo)) + pad)
                if win is None:
                    continue
                sx, sy, sc = self._sx[win], self._sy[win], self._scale[win]
                cov = (np.clip(np.minimum(sx - x, x + rw - sx) / sc + 0.5, 0.0, 1.0)
                       * np.clip(np.minimum(sy - y, y + rh - sy) / sc + 0.5, 0.0, 1.0))
            img[win] += np.float32(w) * cov.astype(np.float32) * np.float32(self._dn(level) - bg_dn)
        if self.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur)
        img = img + self._noise[int(self._rng.integers(0, len(self._noise)))]
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hitl-sim-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_exposure_us(self, exposure_us: float) -> None:
        self.exposure_us = float(exposure_us)
        self.exposure_ns = int(round(self.exposure_us * 1e3))

    def _run(self) -> None:
        t0 = time.monotonic_ns() + 50_000_000
        k = 0
        while not self._stop.is_set():
            a = t0 + k * self.period_ns
            b = a + self.exposure_ns
            capture = b + self.delivery_ns + int(abs(self._rng.normal(0.0, self.jitter_ns)))
            # Render once the exposure has ended, then deliver at the simulated capture time, so
            # exposure end -> user space is the injected delivery latency, not delivery + render time.
            wait = (b + 500_000 - time.monotonic_ns()) / 1e9
            if wait > 0 and self._stop.wait(wait):
                break
            img = self.render(a, b)
            wait = (capture - time.monotonic_ns()) / 1e9
            if wait > 0 and self._stop.wait(wait):
                break
            frame = Frame(image=img, seq=k, t_exposure_ns=a + self.exposure_ns // 2,
                          t_capture_ns=time.monotonic_ns(), exposure_us=self.exposure_us, gain=1.0)
            with self._cond:
                if self._slot is not None:
                    self.dropped += 1
                self._slot = frame
                self._cond.notify_all()
            k += 1

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        with self._cond:
            if self._slot is None:
                self._cond.wait(timeout_s)
            frame, self._slot = self._slot, None
            return frame


# ----------------------------------------------------------------------------------------- mic link
class SimClickLink:
    """Turns the simulated break-sense edge into an audible click (and the servo whine before it)."""

    def __init__(self, gpio, pin: int, mic, truth: SimTruth, cmd_to_break_ms: float, mic_distance_m: float = 0.0):
        self.mic = mic
        self.lock_ns = int(round(truth.lock_time_ms * 1e6))
        self.whine_ns = int(round(cmd_to_break_ms * 1e6))
        self.flight_ns = int(round(float(mic_distance_m) / SPEED_OF_SOUND_MPS * 1e9))
        gpio.watch(int(pin), "falling", self._on_break)

    def _on_break(self, _pin: int, _level: int, t_ns: int) -> None:
        self.mic.schedule_noise(int(t_ns) - self.whine_ns + self.flight_ns, int(t_ns) + self.flight_ns, amplitude=0.03)
        self.mic.schedule_click(int(t_ns) + self.lock_ns + self.flight_ns, amplitude=0.6)
