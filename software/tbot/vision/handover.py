"""Detection in the sensor process: what crosses between it and the pipeline (`camera.detect_in_process`).

On the Pi 3B a frame in front of trees costs detect 11 ms + track 5 ms + solve 2 ms in one loop, one core.
Threads do not help (the interpreter lock: two threads took longer than one loop), a second process does. The
sensor process (camera/proc.py) already has each frame first and the IMU stream, so it does the front of the
loop there:

    sensor process   frame -> attitude at the frame time -> scenery filter -> detect      (SensorDetector)
    pipeline         <- frame + detections + that attitude (DetectResult); track -> solve -> fire -> record
                     -> the tracks the scenery filter protects / looks for first (Hints), back to the sensor
                        process for the frames to come

The hints are the tracks' own filter states, and the sensor process predicts them to its frame's time with the
tracker's own code, so the only difference from the single loop is WHICH update they come from: the frame being
detected started while the pipeline was still tracking the one before, so its hints are one frame old.

Everything is packed as raw bytes (struct + arrays + marshal for the two small dicts): pickling per frame cost
milliseconds on the Pi (see camera/proc.py).
"""

from __future__ import annotations

import marshal
import math
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from tbot.types import Detection, Frame, Track

__all__ = ["DetectResult", "Hints", "SensorDetector", "pack_hints", "pack_result", "unpack_hints", "unpack_result"]

DET_COLS = 11  # u, v, area_px, diameter_px, contrast, snr, score, bbox x, y, w, h

# n detections, scenery filter armed, attitude ready, detector scale, noise sigma, attitude ms, detect ms (arming
# the filter + detect), monotonic ns the sensor process took the frame / sent the result, the hints used
# (pipeline frame count; 0 = none), scenery drift x / y (px/s), drift updates, length of the marshalled extras
_RESULT = struct.Struct("<qBBqdddqqqddqq")
# pipeline frame count, tracks, protect indices, focus indices
_HINTS = struct.Struct("<qqqq")
_TRACK = struct.Struct("<qB")  # t_ns, has a cartesian state


@dataclass(slots=True)
class DetectResult:
    """One frame's detections as the sensor process made them."""

    detections: list
    R_wg: np.ndarray            # the gun attitude the scenery filter used: the tracker must use the same
    filtered: bool = False      # the scenery filter was armed (a real attitude, ready)
    imu_ready: bool = False
    scale: int = 1
    noise_sigma: float = float("nan")
    timings_ms: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    static_px: list = field(default_factory=list)
    drift: tuple = (0.0, 0.0)
    drift_updates: int = 0
    attitude_ms: float = 0.0
    detect_ms: float = 0.0
    t_read_ns: int = 0          # CLOCK_MONOTONIC in the sensor process: frame taken / result sent
    t_sent_ns: int = 0
    hints_n: int = 0            # which Hints it had (the pipeline's frame count when they were made)


def _plain(v):
    """A value marshal takes (NumPy scalars and arrays become Python numbers and lists)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def pack_result(r: DetectResult) -> bytes:
    n = len(r.detections)
    flat = [x for d in r.detections
            for x in (d.u, d.v, d.area_px, d.diameter_px, d.contrast, d.snr, d.score, *d.bbox)]
    a = struct.pack(f"<{n * DET_COLS}d", *flat)
    # (_plain: marshal writes a NumPy scalar as its raw bytes without complaint)
    extras = marshal.dumps((_plain(r.timings_ms), _plain(r.stats), _plain(r.static_px)))
    head = _RESULT.pack(n, 1 if r.filtered else 0, 1 if r.imu_ready else 0, int(r.scale), float(r.noise_sigma),
                        float(r.attitude_ms), float(r.detect_ms), int(r.t_read_ns), int(r.t_sent_ns), int(r.hints_n),
                        float(r.drift[0]), float(r.drift[1]), int(r.drift_updates), len(extras))
    return b"".join((head, np.ascontiguousarray(r.R_wg, dtype=np.float64).tobytes(), a, extras))


def unpack_result(buf: bytes, t_ns: int, offset: int = 0) -> DetectResult:
    """t_ns: the frame's mid-exposure time (every Detection carries it)."""
    (n, filtered, ready, scale, noise, att_ms, det_ms, t_read, t_sent, hints_n, dx, dy, d_upd,
     n_extra) = _RESULT.unpack_from(buf, offset)
    o = offset + _RESULT.size
    R = np.frombuffer(buf, dtype=np.float64, count=9, offset=o).reshape(3, 3).copy()
    o += 72
    rows = np.frombuffer(buf, dtype=np.float64, count=n * DET_COLS, offset=o).reshape(n, DET_COLS).tolist()
    o += 8 * n * DET_COLS
    extras = marshal.loads(buf[o:o + n_extra])
    t = int(t_ns)
    dets = [Detection(t_ns=t, u=r[0], v=r[1], area_px=r[2], diameter_px=r[3], contrast=r[4], snr=r[5], score=r[6],
                      bbox=(int(r[7]), int(r[8]), int(r[9]), int(r[10]))) for r in rows]
    return DetectResult(detections=dets, R_wg=R, filtered=bool(filtered), imu_ready=bool(ready), scale=int(scale),
                        noise_sigma=float(noise), timings_ms=extras[0], stats=extras[1], static_px=extras[2],
                        drift=(float(dx), float(dy)), drift_updates=int(d_upd), attitude_ms=float(att_ms),
                        detect_ms=float(det_ms), t_read_ns=int(t_read), t_sent_ns=int(t_sent), hints_n=int(hints_n))


# ------------------------------------------------------------------------------------------------- hints
@dataclass(slots=True)
class Hints:
    """The tracks the scenery filter needs, as the pipeline had them after frame number `n` (its frame count).

    protect: tracks seen crossing the sky (candidates near them are never called scenery); focus: established
    targets, the main one first (the detector looks where they should be first). Both index into `tracks`."""

    n: int = 0
    tracks: list = field(default_factory=list)
    protect: list = field(default_factory=list)
    focus: list = field(default_factory=list)


def pack_hints(n: int, protect: list, focus: list) -> bytes:
    """protect / focus: lists of Track (the same Track may be in both)."""
    tracks: list = []
    index: dict[int, int] = {}
    for t in (*protect, *focus):
        if id(t) not in index:
            index[id(t)] = len(tracks)
            tracks.append(t)
    parts = [_HINTS.pack(int(n), len(tracks), len(protect), len(focus))]
    for t in tracks:
        cart = t.s is not None and t.S is not None
        parts.append(_TRACK.pack(int(t.t_ns), 1 if cart else 0))
        # the state the tracker predicts from: cartesian [p, v, a] and its covariance, else the angular one
        for a, size in ((t.s, 9), (t.S, 81)) if cart else ((t.x, 6), (t.P, 36)):
            a = np.ascontiguousarray(a, dtype=np.float64)
            if a.size != size:
                raise ValueError(f"track state of {a.size} values, expected {size}")
            parts.append(a.tobytes())
    parts.append(np.array([index[id(t)] for t in (*protect, *focus)], dtype=np.int64).tobytes())
    return b"".join(parts)


def unpack_hints(buf: bytes, offset: int = 0) -> Hints:
    n, n_tracks, n_protect, n_focus = _HINTS.unpack_from(buf, offset)
    o = offset + _HINTS.size

    def take(count: int, shape):
        nonlocal o
        a = np.frombuffer(buf, dtype=np.float64, count=count, offset=o).reshape(shape).copy()
        o += 8 * count
        return a

    tracks = []
    for _ in range(n_tracks):
        t_ns, cart = _TRACK.unpack_from(buf, o)
        o += _TRACK.size
        if cart:
            x, P, s, S = np.zeros(6), np.zeros((6, 6)), take(9, (9,)), take(81, (9, 9))
        else:
            x, P, s, S = take(6, (6,)), take(36, (6, 6)), None, None
        tracks.append(Track(track_id=-1, t_ns=int(t_ns), x=x, P=P, range_m=0.0, range_sigma_m=0.0, s=s, S=S))
    idx = np.frombuffer(buf, dtype=np.int64, count=n_protect + n_focus, offset=o).tolist()
    return Hints(n=int(n), tracks=tracks, protect=idx[:n_protect], focus=idx[n_protect:])


def scenery_dirs(tracker, t_ns: int, protect: list, focus: list) -> tuple[list, list, list]:
    """(protect directions, focus directions, focus radii) at t_ns for StaticFilter.begin, from Tracks.

    The one place this is worked out: the single loop and the sensor process both call it."""
    gate = float(getattr(tracker, "_gate", 0.0))
    gate_min = float(getattr(tracker, "_gate_min", 0.0))
    focus_dirs, focus_radius = [], []
    for t in focus:
        focus_dirs.append(tracker.predict_dir(t, t_ns))
        # as wide as the tracker's own association gate for this track in this frame
        focus_radius.append(math.hypot(gate * float(tracker.predict(t, t_ns)[2]), gate_min))
    return [tracker.predict_dir(t, t_ns) for t in protect], focus_dirs, focus_radius


# ------------------------------------------------------------------------------------------------- sensor side
class SensorDetector:
    """The front of the frame loop, as the sensor process runs it: attitude -> scenery filter -> detect.

    Built from the same config and calibration as the pipeline's own stages, so a frame gives the same
    detections here as there. Its tracker never tracks: it only predicts the hinted tracks to the frame time."""

    def __init__(self, cfg, calib, width: int, height: int, has_imu: bool) -> None:
        from tbot.imu.attitude import AttitudeEstimator
        from tbot.optics import CameraModel, MountModel, combined_mask
        from tbot.vision.clutter import StaticFilter
        from tbot.vision.detector import ClayDetector
        from tbot.vision.tracker3d import make_tracker

        self.cfg = cfg
        self.cam = CameraModel.from_calibration(calib.intrinsics, cfg, int(width), int(height))
        self.mount = MountModel.from_config(cfg, calib.boresight)
        self.detector = ClayDetector(cfg, combined_mask(self.cam, self.mount, cfg, calib))
        self.clutter = StaticFilter(cfg, self.cam, self.mount) if bool(cfg.detector.static_reject) else None
        self.attitude = AttitudeEstimator(cfg) if has_imu else None
        self.tracker = make_tracker(cfg, self.cam, self.mount)
        self.hints: Hints | None = None

    def imu(self, samples) -> None:
        if samples and self.attitude is not None:
            self.attitude.update(samples)

    def reset(self) -> None:
        """The camera was reopened: what the scenery filter remembers is from before the outage."""
        if self.clutter is not None:
            self.clutter.reset()
        self.hints = None

    def process(self, frame: Frame, t_read_ns: int = 0) -> DetectResult:
        t0 = time.perf_counter_ns()
        t_ns = int(frame.t_exposure_ns)
        att = self.attitude
        ready = att is not None and bool(att.ready)
        R_wg = np.array(att.R_wg9(t_ns), dtype=np.float64).reshape(3, 3) if att is not None else np.eye(3)
        t1 = time.perf_counter_ns()
        clutter, det = self.clutter, self.detector
        hints = self.hints
        filtered = clutter is not None and ready and bool(self.cfg.detector.static_reject)
        if filtered:
            if hints is not None and hints.tracks:
                protect, focus, radius = scenery_dirs(self.tracker, t_ns, [hints.tracks[i] for i in hints.protect],
                                                      [hints.tracks[i] for i in hints.focus])
            else:
                protect, focus, radius = [], [], []
            clutter.begin(t_ns, R_wg, protect_dirs_w=protect, focus_dirs_w=focus, focus_radius_rad=radius,
                          others_max=int(getattr(self.cfg.detector, "followed_others_max", 0)))
            det.candidate_filter = clutter.filter
        else:
            det.candidate_filter = None
        dets = det.detect(frame)
        t2 = time.perf_counter_ns()
        return DetectResult(
            detections=dets, R_wg=R_wg, filtered=filtered, imu_ready=ready, scale=int(det.scale),
            noise_sigma=float(det.noise_sigma), timings_ms=dict(det.timings_ms), stats=dict(det.stats),
            static_px=list(clutter.static_px) if filtered else [],
            drift=tuple(clutter.drift) if clutter is not None else (0.0, 0.0),
            drift_updates=int(clutter.drift_updates) if clutter is not None else 0,
            attitude_ms=(t1 - t0) / 1e6, detect_ms=(t2 - t1) / 1e6, t_read_ns=int(t_read_ns),
            hints_n=int(hints.n) if hints is not None else 0)
