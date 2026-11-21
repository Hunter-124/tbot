"""HITL rig runtime: the real `Pipeline`, tapped, driven frame by frame by test procedures.

Procedures are generators: every bare `yield` hands control back so the rig processes exactly one more
camera frame. Nothing here bypasses the safety model: the actuator (when `--live`) is the normal one behind
the arming interlock, the operator holds the arming button, and the pipeline's one-shot-per-press rule
applies.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator

import cv2
import numpy as np

from tbot.app.run import Pipeline
from tbot.hitl.scene import HitlStore

#: The rig gives up when the camera delivers nothing for this long.
NO_FRAME_TIMEOUT_S = 3.0


class TapSource:
    """Pass-through FrameSource that keeps the last frame the pipeline read."""

    def __init__(self, inner):
        self._inner = inner
        self.last = None

    def read(self, timeout_s: float = 1.0):
        frame = self._inner.read(timeout_s=timeout_s)
        if frame is not None:
            self.last = frame
        return frame

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TapActuator:
    """Pass-through actuator that timestamps every fire() call on the monotonic clock."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[int, int, bool]] = []  # (t_call_ns, t_arg_ns, accepted)

    def fire(self, t_ns: int) -> bool:
        t_call = time.monotonic_ns()
        ok = bool(self._inner.fire(t_ns))
        self.calls.append((t_call, int(t_ns), ok))
        return ok

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _det_row(d) -> list:
    return [round(float(d.u), 3), round(float(d.v), 3), round(float(d.diameter_px), 3),
            round(float(d.contrast), 2), round(float(d.snr), 2)]


class Rig:
    """Owns the pipeline and the per-frame log the analysis consumes."""

    def __init__(self, cfg, calib, source, store: HitlStore, *, gpio, arming, actuator=None, imu=None,
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.store = store
        self.log = log
        self.tap = TapSource(source)
        self.actuator = actuator
        # The bench procedures find a disc that STANDS on the monitor (geometry, flash latency) with a camera that
        # stands too: the motion front end, which only sees what moves against the scenery, sees nothing there.
        # The bench therefore runs the single-frame front end, whatever the gun is set to; what it measures (the
        # geometry, the camera and actuator latencies, the arming chain) does not depend on the front end.
        if str(getattr(getattr(cfg, "vision", None), "engine", "legacy")).strip().lower() != "legacy":
            log("hitl: the bench runs the single-frame front end (vision.engine = legacy for this run)")
            cfg.vision.engine = "legacy"
        self.pipeline = Pipeline(cfg, calib, self.tap, gpio=gpio, actuator=actuator, arming=arming, imu=imu)
        self.arming = self.pipeline.arming
        self.gpio = self.pipeline.gpio
        self.frames: list[dict] = []
        self.probes: dict[str, tuple[int, int, int, int]] = {}
        self._base_mask = None if self.pipeline.mask is None else self.pipeline.mask.copy()

    # ---------------------------------------------------------------- setup
    def start(self) -> None:
        self.pipeline.start()

    def close(self) -> None:
        self.pipeline.close()

    def set_screen_mask(self, screen_mask: np.ndarray | None) -> None:
        """Restrict the detector to the screen (AND with the pipeline's barrel/calibration mask)."""
        base = self._base_mask
        if screen_mask is None:
            mask = base
        elif base is None:
            mask = screen_mask
        else:
            mask = cv2.bitwise_and(base, screen_mask)
        self.pipeline.mask = mask
        self.pipeline.detector.set_mask(mask)

    @property
    def last(self) -> dict | None:
        return self.frames[-1] if self.frames else None

    @property
    def period_ns(self) -> int:
        return int(round(self.pipeline.frame_period_s * 1e9))

    # ---------------------------------------------------------------- one frame
    def step(self) -> dict | None:
        tele = self.pipeline.step()
        if tele is None:
            return None
        t_done = time.monotonic_ns()
        frame = self.tap.last
        p = self.pipeline
        st = tele["stage_ms"]
        dec = p.decision
        best = p.tracker.best()
        rec = {
            "i": len(self.frames),
            "seq": int(tele["seq"]),
            "te": int(tele["t_exposure_ns"]),
            "tc": int(frame.t_capture_ns),
            "exp_us": float(frame.exposure_us),
            "now": int(tele["t_ns"]),
            "done": t_done,
            "stage": {k: round(float(v), 4) for k, v in st.items()},
            "dropped": int(tele["dropped"]),
            "armed": bool(tele["armed"]),
            "fired": bool(tele["fired"]),
            "dets": [_det_row(d) for d in p.detections],
            "trk": None if best is None else {
                "id": int(best.track_id), "hits": int(best.hits), "confirmed": bool(best.confirmed),
                "range": float(best.range_m), "az": float(best.az), "el": float(best.el),
            },
            "dec": {
                "fire": bool(dec.fire),
                "would": bool(dec.fire) or "would fire" in str(dec.reason),
                "reason": str(dec.reason),
                "track": None if dec.track_id is None else int(dec.track_id),
                "p_hit": float(dec.p_hit), "miss": float(dec.miss_rad), "sigma": float(dec.sigma_rad),
                "pattern": float(dec.pattern_rad), "range": float(dec.range_m), "tof": float(dec.tof_s),
                "t_exit": None if dec.t_exit_ns is None else int(dec.t_exit_ns),
                "lead_w": None if dec.lead_dir_w is None else [float(v) for v in dec.lead_dir_w],
                "bore_w": None if dec.bore_dir_w is None else [float(v) for v in dec.bore_dir_w],
            },
        }
        if self.probes:
            img = frame.image if frame.image.ndim == 2 else cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            rec["probe"] = {name: float(img[y:y + h, x:x + w].mean()) for name, (x, y, w, h) in self.probes.items()}
        self.frames.append(rec)
        return rec

    def _advance(self) -> dict:
        deadline = time.monotonic() + NO_FRAME_TIMEOUT_S
        while True:
            rec = self.step()
            if rec is not None:
                return rec
            if time.monotonic() > deadline:
                raise RuntimeError(f"no camera frame for {NO_FRAME_TIMEOUT_S:.0f} s from {self.pipeline.source.name}")

    def drive(self, gen: Iterator):
        """Run a procedure generator to completion, one camera frame per `yield`; returns its value."""
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                return stop.value
            self._advance()

    # ---------------------------------------------------------------- waiting helpers (generators)
    def wait_s(self, seconds: float):
        deadline = time.monotonic_ns() + int(seconds * 1e9)
        while time.monotonic_ns() < deadline:
            yield

    def wait_until_ns(self, t_ns: int):
        while time.monotonic_ns() < t_ns:
            yield

    def wait_frames(self, n: int):
        for _ in range(int(n)):
            yield

    def wait_for(self, predicate: Callable[[], bool], timeout_s: float):
        """Yield frames until predicate() is true; returns False on timeout."""
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > deadline:
                return False
            yield
        return True

    def wait_drawn(self, item_id: int, timeout_s: float = 2.0):
        """Until the page has logged a frame that drew the item; returns its rig_ms (None on timeout)."""
        ok = yield from self.wait_for(lambda: self.store.first_drawn(item_id) is not None, timeout_s)
        return self.store.first_drawn(item_id) if ok else None


# --------------------------------------------------------------------------------------- operators
class HumanOperator:
    """Prompts the person at the bench to hold / release the arming button."""

    def __init__(self, arming, log: Callable[[str], None] = print, timeout_s: float = 300.0):
        self.arming = arming
        self.log = log
        self.timeout_s = timeout_s

    def press(self, rig: Rig, what: str = ""):
        if not self.arming.armed:
            self.log(f"  HOLD the arming button{(' ' + what) if what else ''} ...")
        ok = yield from rig.wait_for(lambda: self.arming.armed, self.timeout_s)
        if not ok:
            raise RuntimeError("the arming button was never held")

    def release(self, rig: Rig):
        if self.arming.armed:
            self.log("  release the arming button")
        ok = yield from rig.wait_for(lambda: not self.arming.armed, self.timeout_s)
        if not ok:
            raise RuntimeError("the arming button was never released")


class SimOperator:
    """Simulated hand on the simulated arming button (SimGpio only)."""

    def __init__(self, gpio, arming, pin: int):
        self.gpio = gpio
        self.arming = arming
        self.pin = int(pin)

    def press(self, rig: Rig, what: str = ""):
        self.gpio.inject(self.pin, 0)
        ok = yield from rig.wait_for(lambda: self.arming.armed, 1.0)
        if not ok:
            raise RuntimeError("simulated arming press did not arm the interlock")

    def release(self, rig: Rig):
        self.gpio.inject(self.pin, 1)
        yield from rig.wait_for(lambda: not self.arming.armed, 1.0)
