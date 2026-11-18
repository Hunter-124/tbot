"""Synthetic camera + IMU source: closed-loop testing of the whole pipeline without hardware [sim slice].

The scene is rendered from the scenario ground truth (tbot.sim.scenario) through the real camera model:
sky gradient and clouds fixed in world azimuth/elevation, textured ground below the horizon, the barrel
silhouette from tbot.optics.barrel_mask drawn dark, and every intact clay as a physically sized tilted
disc with exposure motion blur. A clay reported hit (break_clay, called by the shot judge in closed loop)
is drawn from then on as a few small fading fragments (no motion blur). Mono uint8 by default (the OV9281
is monochrome), BGR when cfg.sim.render_color. Simulated timestamps start at 1_000_000_000 ns; scene and
truth are deterministic for a given seed and given breaks.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from tbot.camera.base import SourceBase
from tbot.config import Calibration, Config
from tbot.optics import CameraModel, MountModel, barrel_mask
from tbot.sim.scenario import Scenario, make_scenario
from tbot.types import Frame, ImuSample, SimClock

__all__ = ["SyntheticSource", "SyntheticImu", "T0_NS"]

T0_NS = 1_000_000_000

_BG_DOWNSCALE = 4  # the sky/ground field is smooth: render it small and upsample (keeps < 15 ms/frame)
_SUPERSAMPLE = 4  # clay patch supersampling per axis
_MAX_SUBSTEPS = 6  # exposure motion-blur sub-steps
_HORIZON_BLEND_RAD = 0.01  # smooth horizon transition
_GROUND_FRACTION = 0.30  # ground brightness relative to sky_level
_GROUND_TEXTURE = 0.30  # relative amplitude of the ground texture
_BARREL_FRACTION = 0.22  # barrel silhouette brightness relative to sky_level
_CLOUD_TERMS = 4
_SKY_BGR = (1.18, 1.00, 0.78)  # colour weights of the sky luminance (blue sky)
_GROUND_BGR = (0.62, 0.86, 1.00)  # colour weights of the ground luminance (brown)
_CLAY_BGR = (24.0, 104.0, 238.0)  # orange clay at the reference illumination
_REF_SKY_LEVEL = 170.0  # sky level the clay colour above was chosen for
_G_W = np.array([0.0, 0.0, 9.80665])  # gravity reaction measured by an accelerometer at rest
_ACCEL_NOISE_M_S2 = 0.02  # MPU-6050 noise density ~400 ug/sqrt(Hz) over ~100 Hz [INFERENCE]
_IMU_TEMP_C = 28.0
_IMU_MAX_BURST = 20000


def _unit_rows(v: np.ndarray) -> np.ndarray:
    """Finite unit rows; degenerate rows fall back to the optical axis."""
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    v = np.where(np.isfinite(v), v, 0.0)
    n = np.linalg.norm(v, axis=1)
    bad = n < 1e-9
    if bad.any():
        v = v.copy()
        v[bad] = (0.0, 0.0, 1.0)
        n = np.where(bad, 1.0, n)
    return v / n[:, None]


_LOOP_TAIL_NS = 2_000_000  # solver, telemetry and recording after the modelled detect + track work


class SyntheticImu:
    """Gun-frame IMU of the simulated gun, sampled at cfg.imu.sample_rate_hz (ImuSource protocol)."""

    def __init__(self, source: "SyntheticSource"):
        self._src = source
        self._rate = float(source.cfg.imu.sample_rate_hz)
        self._gyro_sigma = math.radians(float(source.cfg.sim.gyro_noise_dps))
        self._rng = np.random.default_rng([source.seed, 3])
        self._k = 0
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._k = 0
        self._rng = np.random.default_rng([self._src.seed, 3])

    def read(self) -> list[ImuSample]:
        """All samples whose time is at or before the newest delivered frame, each returned once."""
        if not self._running:
            return []
        sc = self._src.scenario
        limit = self._src.imu_limit_ns()
        period = 1e9 / self._rate
        out: list[ImuSample] = []
        while len(out) < _IMU_MAX_BURST:
            t_ns = T0_NS + int(round(self._k * period))
            if t_ns > limit:
                break
            t_s = (t_ns - T0_NS) * 1e-9
            R = sc.R_wg(t_s)
            gyro = sc.omega_g(t_s) + self._gyro_sigma * self._rng.standard_normal(3)
            accel = R.T @ _G_W + _ACCEL_NOISE_M_S2 * self._rng.standard_normal(3)
            out.append(ImuSample(t_ns=t_ns, gyro=gyro, accel=accel, temp_c=_IMU_TEMP_C))
            self._k += 1
        return out


class SyntheticSource(SourceBase):
    """Frame source rendering a scenario through the configured camera model."""

    def __init__(self, cfg: Config, calib: Calibration, scenario: str | None = None,
                 seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.seed = int(cfg.sim.seed if seed is None else seed)
        self.scenario_name = scenario or str(cfg.sim.scenario)
        self.scenario: Scenario = make_scenario(self.scenario_name, cfg, self.seed)
        self.width = int(cfg.camera.width)
        self.height = int(cfg.camera.height)
        self.is_color = bool(cfg.sim.render_color)
        self.nominal_fps = float(cfg.camera.fps)
        self.name = f"synthetic:{self.scenario_name}"
        # sim.process_ms > 0: the loop's own work time is modelled (see read), on a clock that only moves when
        # the simulation moves it, so a run repeats exactly. 0: the clock runs on with this computer's time.
        self._work_ns = int(round(float(getattr(cfg.sim, "process_ms", 0.0)) * 1e6))
        self.clock = SimClock(T0_NS, frozen=self._work_ns > 0)
        self.frame_t_ns = T0_NS
        self.cam = CameraModel.from_calibration(calib.intrinsics, cfg, self.width, self.height)
        self.mount = MountModel.from_config(cfg, calib.boresight)
        self.mask = np.asarray(barrel_mask(self.cam, self.mount, cfg))
        self._exposure_s = float(cfg.camera.exposure_us) * 1e-6
        self._imu: SyntheticImu | None = None
        self._seq = 0
        self._precompute()

    # ---------------------------------------------------------------- setup
    def _precompute(self) -> None:
        cfg = self.cfg
        ws = max(2, self.width // _BG_DOWNSCALE)
        hs = max(2, self.height // _BG_DOWNSCALE)
        self._bg_shape = (hs, ws)
        sx = self.width / ws
        sy = self.height / hs
        u = (np.arange(ws, dtype=np.float64) + 0.5) * sx - 0.5
        v = (np.arange(hs, dtype=np.float64) + 0.5) * sy - 0.5
        uu, vv = np.meshgrid(u, v)
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        rays_c = _unit_rows(self.cam.pixel_to_ray(uv))
        self._bg_rays_g = np.ascontiguousarray(_unit_rows(self.mount.cam_to_gun(rays_c)), dtype=np.float64)

        rng = np.random.default_rng([self.seed, 11])
        amps = np.array([1.0, 0.6, 0.4, 0.25][:_CLOUD_TERMS], dtype=np.float64)
        amps = amps / (0.5 * math.sqrt(float((amps * amps).sum())))  # unit standard deviation
        self._cloud = (
            amps,
            rng.uniform(1.2, 5.0, _CLOUD_TERMS),
            rng.uniform(1.2, 5.0, _CLOUD_TERMS),
            rng.uniform(0.0, 2.0 * math.pi, _CLOUD_TERMS),
            rng.uniform(0.0, 2.0 * math.pi, _CLOUD_TERMS),
        )
        self._ground_phase = rng.uniform(0.0, 2.0 * math.pi, 2)

        sky = float(cfg.sim.sky_level)
        self._barrel_zero = self.mask == 0
        rows = np.arange(self.height, dtype=np.float64)[:, None]
        fill = (sky * _BARREL_FRACTION * (0.85 + 0.3 * rows / max(self.height - 1, 1))
                * np.ones((1, self.width)))
        if self.is_color:  # blued steel: neutral, no colour cast
            fill = fill[:, :, None] * np.ones(3)
            self._barrel_where = self._barrel_zero[:, :, None]
        else:
            self._barrel_where = self._barrel_zero
        self._barrel_fill = np.ascontiguousarray(fill, dtype=np.float32)
        self._clay_bgr = np.array(_CLAY_BGR, dtype=np.float64) * (sky / _REF_SKY_LEVEL)

    # ---------------------------------------------------------------- source protocol
    def start(self) -> None:
        self._seq = 0
        self.frame_t_ns = T0_NS
        self.clock.set(T0_NS)
        self.dropped = 0
        self.skipped = 0
        self.scenario.reset_breaks()  # a fresh pass throws whole clays again

    def stop(self) -> None:
        pass

    def imu(self) -> SyntheticImu:
        if self._imu is None:
            self._imu = SyntheticImu(self)
        return self._imu

    @property
    def n_frames(self) -> int:
        return self.scenario.n_frames

    def imu_limit_ns(self) -> int:
        """IMU samples up to this time have been taken: the newest frame's arrival, or (modelled work time)
        the simulated present, so a second read just before the solver gets the samples of the work time."""
        return max(self.frame_t_ns, int(self.clock.now_ns())) if self._work_ns > 0 else self.frame_t_ns

    def processed(self) -> None:
        if self._work_ns > 0:
            self.clock.advance(self._work_ns)

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        """Next scenario frame; None once the scenario has run out.

        sim.process_ms = 0: every frame, the simulator never drops. Otherwise the loop behaves as on the gun:
        it comes back for a frame when its work on the last one is done (the clock, moved on by processed()
        and by any wait for the fire command), takes the NEWEST frame that has arrived by then, or waits for
        the next one, and the ones in between are never seen (counted in `skipped`)."""
        i = self._seq
        latency = int(round(self._exposure_s * 0.5e9)) + int(round(self.cfg.latency.exposure_to_user_ms * 1e6))
        period = 1e9 / self.nominal_fps
        t_start = None
        if self._work_ns > 0 and i > 0:
            ready = int(self.clock.now_ns()) + _LOOP_TAIL_NS
            newest = int(math.floor((ready - T0_NS - latency) / period))
            if newest > i:
                self.skipped += newest - i
                i = newest
            t_start = ready
        t_s = i / self.nominal_fps
        if t_s > self.scenario.duration_s + 1e-9:
            return None
        t_exp = T0_NS + int(round(i * period))
        t_cap = t_exp + latency
        image = self._render(t_s, i)
        self._seq = i + 1
        self.frame_t_ns = t_cap
        self.clock.set(t_cap if t_start is None else max(t_start, t_cap))
        return Frame(image=image, seq=i, t_exposure_ns=t_exp, t_capture_ns=t_cap,
                     exposure_us=float(self.cfg.camera.exposure_us), gain=float(self.cfg.camera.analogue_gain))

    # ---------------------------------------------------------------- truth
    def truth(self, t_ns: int) -> dict:
        """Ground truth at a frame's mid-exposure time (keys per docs/ARCHITECTURE.md)."""
        t_s = (int(t_ns) - T0_NS) * 1e-9
        out = self.scenario.truth(t_s)
        out["clay_px"] = None
        for c in out["clays"]:
            c["clay_px"] = self._visible_pixel(t_s, c["index"]) if c["visible"] else None
            if c["index"] == out["clay_index"] and out["visible"]:
                out["clay_px"] = c["clay_px"]
        return out

    def break_clay(self, index: int, t_ns: int) -> bool:
        """Closed-loop hook: clay `index` was hit by a shot arriving at t_ns (source clock).

        From then on it is rendered as fading debris instead of a whole clay and reported not visible.
        Returns False when there is nothing to break (see Scenario.break_clay).
        """
        return self.scenario.break_clay(int(index), (int(t_ns) - T0_NS) * 1e-9)

    def _visible_pixel(self, t_s: float, i: int) -> tuple[float, float] | None:
        px = self._clay_pixel(t_s, i)
        if px is None:
            return None
        u, v = px
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < self.width and 0 <= vi < self.height and self.mask[vi, ui] != 0:
            return (u, v)
        return None

    # ---------------------------------------------------------------- rendering
    def _project(self, dir_w: np.ndarray, R: np.ndarray) -> tuple[float, float] | None:
        v_g = R.T @ np.asarray(dir_w, dtype=np.float64)
        v_c = np.asarray(self.mount.gun_to_cam(v_g), dtype=np.float64).reshape(1, 3)
        uv = np.asarray(self.cam.ray_to_pixel(v_c), dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(uv[0]).all():
            return None
        return float(uv[0, 0]), float(uv[0, 1])

    def _clay_pixel(self, t_s: float, i: int | None = None) -> tuple[float, float] | None:
        d = self.scenario.clay_dir_w(t_s, i)
        if d is None:
            return None
        return self._project(d, self.scenario.R_wg(t_s))

    def _background(self, R: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        rays_w = self._bg_rays_g @ R.T
        el = np.arcsin(np.clip(rays_w[:, 2], -1.0, 1.0))
        az = np.arctan2(rays_w[:, 1], rays_w[:, 0])
        amps, faz, fel, paz, pel = self._cloud
        cloud = np.zeros_like(el)
        for k in range(len(amps)):
            cloud += amps[k] * np.sin(faz[k] * az + paz[k]) * np.sin(fel[k] * el + pel[k])
        sky_level = float(cfg.sim.sky_level)
        sky = (sky_level + float(cfg.sim.sky_gradient) * (1.0 - (2.0 / math.pi) * el)
               + float(cfg.sim.cloud_amount) * float(cfg.sim.sky_gradient) * cloud)
        gp = self._ground_phase
        tex = np.sin(17.0 * az + gp[0]) * np.sin(23.0 * el + gp[1])
        ground = sky_level * _GROUND_FRACTION * (1.0 + _GROUND_TEXTURE * tex)
        w = np.clip(0.5 + el / (2.0 * _HORIZON_BLEND_RAD), 0.0, 1.0)
        lum = ground + (sky - ground) * w
        hs, ws = self._bg_shape
        if self.is_color:
            weights = np.array(_GROUND_BGR) + (np.array(_SKY_BGR) - np.array(_GROUND_BGR)) * w[:, None]
            small = (lum[:, None] * weights).reshape(hs, ws, 3)
        else:
            small = lum.reshape(hs, ws)
        img = cv2.resize(small.astype(np.float32), (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return np.asarray(img, dtype=np.float32)

    def _render(self, t_s: float, seq: int) -> np.ndarray:
        cfg = self.cfg
        R = self.scenario.R_wg(t_s)
        img = self._background(R)
        for c in self.scenario.clays:
            if self.scenario.clay_visible(t_s, c.index):
                self._draw_clay(img, t_s, c.index)
        if self.scenario.debris:
            self._draw_fragments(img, t_s, R)
        np.copyto(img, self._barrel_fill, where=self._barrel_where)
        sigma = float(cfg.sim.noise_sigma)
        if sigma > 0.0:
            rng = np.random.default_rng([self.seed, 7, int(seq)])
            img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(sigma)
        np.clip(img, 0.0, 255.0, out=img)
        return img.astype(np.uint8)

    def _draw_fragments(self, img: np.ndarray, t_s: float, R: np.ndarray) -> None:
        """Debris of broken clays: small fading discs, area-weighted (sub-pixel pieces still dim)."""
        pos, size, alpha = self.scenario.fragments_w(t_s)
        s = _SUPERSAMPLE
        for p, sz, a in zip(pos, size, alpha):
            rng_m = float(np.linalg.norm(p))
            if rng_m < 1e-3 or a <= 0.0:
                continue
            px = self._project(p / rng_m, R)
            if px is None:
                continue
            ifov = float(np.asarray(self.cam.ifov(np.array([[px[0], px[1]]], dtype=np.float64))).ravel()[0])
            if not math.isfinite(ifov) or ifov <= 0.0:
                continue
            r_px = math.atan2(0.5 * float(sz), rng_m) / ifov
            x0 = max(int(math.floor(px[0] - r_px)) - 1, 0)
            x1 = min(int(math.ceil(px[0] + r_px)) + 2, self.width)
            y0 = max(int(math.floor(px[1] - r_px)) - 1, 0)
            y1 = min(int(math.ceil(px[1] + r_px)) + 2, self.height)
            if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > 40000:
                continue
            pw, ph = x1 - x0, y1 - y0
            us = x0 - 0.5 + (np.arange(pw * s, dtype=np.float64) + 0.5) / s
            vs = y0 - 0.5 + (np.arange(ph * s, dtype=np.float64) + 0.5) / s
            inside = ((us[None, :] - px[0]) ** 2 + (vs[:, None] - px[1]) ** 2) <= max(r_px, 0.5 / s) ** 2
            cov = inside.astype(np.float64).reshape(ph, s, pw, s).mean(axis=(1, 3)) * float(a)
            patch = img[y0:y1, x0:x1]
            if img.ndim == 3:
                c = cov[:, :, None]
                patch *= np.float32(1.0) - c.astype(np.float32)
                patch += (c * self._clay_bgr).astype(np.float32)
            else:
                patch += (cov * float(self.cfg.sim.clay_contrast)).astype(np.float32)

    def _draw_clay(self, img: np.ndarray, t_s: float, i: int = 0) -> None:
        sc = self.scenario
        exp = self._exposure_s
        p_first = self._clay_pixel(t_s - 0.5 * exp, i)
        p_last = self._clay_pixel(t_s + 0.5 * exp, i)
        p_mid = self._clay_pixel(t_s, i)
        if p_mid is None:
            return
        motion = 0.0
        if p_first is not None and p_last is not None:
            motion = math.hypot(p_last[0] - p_first[0], p_last[1] - p_first[1])
        n_sub = int(min(_MAX_SUBSTEPS, max(1, math.ceil(motion / 0.5))))
        offs = exp * ((np.arange(n_sub, dtype=np.float64) + 0.5) / n_sub - 0.5)

        radius_m = 0.5 * math.hypot(sc.clay_diameter_m, sc.clay_thickness_m)
        centres: list[tuple[float, float, float]] = []
        for dt in offs:
            tk = t_s + float(dt)
            px = self._clay_pixel(tk, i)
            if px is None:
                continue
            rng_m = sc.clay_range_m(tk, i)
            ifov = float(np.asarray(self.cam.ifov(np.array([[px[0], px[1]]], dtype=np.float64))).ravel()[0])
            if not math.isfinite(ifov) or ifov <= 0.0:
                continue
            r_px = math.atan2(radius_m, max(rng_m, 1e-3)) / ifov
            centres.append((px[0], px[1], r_px))
        if not centres:
            return
        x0 = int(math.floor(min(c[0] - c[2] for c in centres))) - 1
        x1 = int(math.ceil(max(c[0] + c[2] for c in centres))) + 2
        y0 = int(math.floor(min(c[1] - c[2] for c in centres))) - 1
        y1 = int(math.ceil(max(c[1] + c[2] for c in centres))) + 2
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, self.width)
        y1 = min(y1, self.height)
        if x1 <= x0 or y1 <= y0:
            return
        pw, ph = x1 - x0, y1 - y0
        if pw * ph > 250000:  # a clay this big means a broken scenario; never allocate unbounded work
            return

        s = _SUPERSAMPLE
        us = x0 - 0.5 + (np.arange(pw * s, dtype=np.float64) + 0.5) / s
        vs = y0 - 0.5 + (np.arange(ph * s, dtype=np.float64) + 0.5) / s
        uu, vv = np.meshgrid(us, vs)
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        rays_g = _unit_rows(self.mount.cam_to_gun(_unit_rows(self.cam.pixel_to_ray(uv))))

        cov = np.zeros(rays_g.shape[0], dtype=np.float64)
        used = 0
        for dt in offs:
            tk = t_s + float(dt)
            if not sc.clay_visible(tk, i):
                continue
            P = sc.clay_pos_w(tk, i)
            rays_w = rays_g @ sc.R_wg(tk).T
            cov += _disc_coverage(rays_w, P, sc.clays[i].axis_w, 0.5 * sc.clay_diameter_m, sc.clay_thickness_m)
            used += 1
        if used == 0:
            return
        cov = (cov / used).reshape(ph, s, pw, s).mean(axis=(1, 3))

        patch = img[y0:y1, x0:x1]
        if img.ndim == 3:
            c = cov[:, :, None]
            patch *= np.float32(1.0) - c.astype(np.float32)
            patch += (c * self._clay_bgr).astype(np.float32)
        else:
            patch += (cov * float(self.cfg.sim.clay_contrast)).astype(np.float32)


def _disc_coverage(rays_w: np.ndarray, centre: np.ndarray, axis: np.ndarray, radius: float,
                   thickness: float) -> np.ndarray:
    """1.0 for rays from the camera (world origin) that hit the clay, modelled as a short cylinder."""
    n = np.asarray(axis, dtype=np.float64)
    P = np.asarray(centre, dtype=np.float64)
    half = 0.5 * float(thickness)
    r2 = float(radius) * float(radius)
    dn = rays_w @ n
    pn = float(P @ n)
    hit = np.zeros(rays_w.shape[0], dtype=bool)

    # side wall: |(t*d)_perp - P_perp| = radius with |axial| <= half
    d_perp = rays_w - dn[:, None] * n
    w_perp = -P + pn * n  # (origin - centre) perpendicular to the axis
    a = np.einsum("ij,ij->i", d_perp, d_perp)
    b = 2.0 * (d_perp @ w_perp)
    c = float(w_perp @ w_perp) - r2
    disc = b * b - 4.0 * a * c
    m = (disc > 0.0) & (a > 1e-12)
    if m.any():
        idx = np.flatnonzero(m)
        sq = np.sqrt(disc[m])
        for t in ((-b[m] - sq) / (2.0 * a[m]), (-b[m] + sq) / (2.0 * a[m])):
            axial = t * dn[m] - pn
            hit[idx[(t > 0.0) & (np.abs(axial) <= half)]] = True

    # end caps
    valid = np.abs(dn) > 1e-9
    for s in (half, -half):
        t = np.where(valid, (pn + s) / np.where(valid, dn, 1.0), -1.0)
        X = t[:, None] * rays_w
        rel = X - (P + s * n)
        hit |= valid & (t > 0.0) & (np.einsum("ij,ij->i", rel, rel) <= r2)
    return hit.astype(np.float64)
