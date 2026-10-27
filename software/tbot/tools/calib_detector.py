"""`tbot calib-detector` — tune the clay detector and build the static (barrel) mask.

Three modes:
  (default)            interactive trackbar tuner (needs --gui) over the configured source or --session
  --report             headless: run N frames, print detections/frame, stage timings and the noise sigma
  --auto-mask SESSION  per-pixel temporal std of a swinging-gun recording -> calib/mask.png

The session format (meta.json + frames.bin + frames.csv) is read directly here, so the tool works on a
machine where the camera stack is absent.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tbot.config import Calibration, Config, add_common_args, load_from_args, save_json, update_json
from tbot.stats import summarize
from tbot.types import Detection, Frame
from tbot.vision.detector import ClayDetector

WINDOW = "tbot calib-detector"
_POLARITIES = ("bright", "dark", "both")
# Trackbar ranges (the config value is the initial position, clamped into the range).
_K_SIGMA_MAX = 300  # x10
_MIN_CONTRAST_MAX = 128
_BG_KERNEL_MAX = 51
_MIN_AREA_MAX = 64
_MAX_AREA_MAX = 4000


# ------------------------------------------------------------------------------- static mask
def static_mask(frames, static_std: float = 3.0, min_area_px: int = 200,
                dilate_px: int = 4) -> np.ndarray:
    """Usable-pixel mask (uint8: 255 usable, 0 masked) from a stack of frames of a *swinging* gun.

    Pixels whose temporal standard deviation stays below `static_std` DN saw the same thing all
    recording long: on a swinging gun only the barrel/pod (and dead pixels) can do that. Only static
    regions that touch the image border and are larger than `min_area_px` are masked out, so a clay
    that happens to hover over a few pixels never creates a hole; the result is grown by `dilate_px`.

    `frames` is any sequence indexable as frames[i] -> (H,W) or (H,W,3) uint8 (a numpy stack or a
    memmap); it is read one frame at a time.
    """
    n = len(frames)
    if n < 2:
        raise ValueError(f"need at least 2 frames for a temporal std, got {n}")
    acc = None
    for i in range(n):
        f = np.asarray(frames[i])
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        x = f.astype(np.float64)
        if acc is None:
            acc = np.zeros_like(x)
            acc2 = np.zeros_like(x)
        elif x.shape != acc.shape:
            raise ValueError(f"frame {i} has shape {x.shape}, expected {acc.shape}")
        acc += x
        acc2 += x * x
    mean = acc / n
    var = acc2 / n - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    static = np.ascontiguousarray(std < float(static_std))
    h, w = static.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(static.view(np.uint8), connectivity=8)
    blocked = np.zeros((h, w), np.bool_)
    for lid in range(1, count):
        x0 = int(stats[lid, cv2.CC_STAT_LEFT])
        y0 = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        area = int(stats[lid, cv2.CC_STAT_AREA])
        touches_border = x0 == 0 or y0 == 0 or x0 + bw == w or y0 + bh == h
        if area > int(min_area_px) and touches_border:
            blocked |= labels == lid
    if int(dilate_px) > 0 and blocked.any():
        k = 2 * int(dilate_px) + 1
        # Square dilation without cv2.dilate (not in the API whitelist): a box blur is non-zero
        # exactly where some pixel inside the kernel was set.
        grown = cv2.blur(blocked.astype(np.float32), (k, k))
        blocked = grown > 1e-6
    mask = np.where(blocked, np.uint8(0), np.uint8(255))
    return np.ascontiguousarray(mask)


# ------------------------------------------------------------------------------- frame streams
class SessionFrames:
    """Random access to a recorded session directory (meta.json + frames.bin + frames.csv)."""

    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        meta_p = self.dir / "meta.json"
        if not meta_p.exists():
            raise FileNotFoundError(f"not a session directory (no meta.json): {self.dir}")
        self.meta = json.loads(meta_p.read_text())
        self.width = int(self.meta["width"])
        self.height = int(self.meta["height"])
        self.channels = int(self.meta.get("channels", 1))
        self.fps = float(self.meta.get("fps", 0.0) or 0.0)
        dtype = np.dtype(self.meta.get("dtype", "uint8"))
        shape = (self.height, self.width) if self.channels == 1 else (self.height, self.width, self.channels)
        frame_bytes = int(np.prod(shape)) * dtype.itemsize
        bin_p = self.dir / "frames.bin"
        n = bin_p.stat().st_size // frame_bytes
        if n == 0:
            raise ValueError(f"{bin_p} holds no complete {shape} frame")
        self.stack = np.memmap(bin_p, dtype=dtype, mode="r", shape=(n,) + shape)
        self.rows = self._read_index(self.dir / "frames.csv", n)

    @staticmethod
    def _read_index(path: Path, n: int) -> list[dict]:
        rows: list[dict] = []
        if not path.exists():
            return rows
        fields = ["seq", "t_exposure_ns", "t_capture_ns", "exposure_us", "gain"]
        with open(path, newline="") as f:
            for parts in csv.reader(f):
                if not parts:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue  # header line
                row = dict(zip(fields, parts))
                rows.append(row)
                if len(rows) >= n:
                    break
        return rows

    @property
    def is_color(self) -> bool:
        return self.channels == 3

    def __len__(self) -> int:
        return int(self.stack.shape[0])

    def frame(self, i: int) -> Frame:
        img = np.array(self.stack[i])
        period_ns = int(1e9 / self.fps) if self.fps > 0 else 10_000_000
        if i < len(self.rows):
            r = self.rows[i]
            seq = int(r["seq"])
            t_exp = int(r["t_exposure_ns"])
            t_cap = int(r.get("t_capture_ns") or t_exp)
            exposure_us = float(r.get("exposure_us") or 0.0)
            gain = float(r.get("gain") or 1.0)
        else:
            seq, t_exp, t_cap, exposure_us, gain = i, i * period_ns, i * period_ns, 0.0, 1.0
        return Frame(image=img, seq=seq, t_exposure_ns=t_exp, t_capture_ns=t_cap,
                     exposure_us=exposure_us, gain=gain)


class _SessionStream:
    """Sequential (rewindable) view of a SessionFrames."""

    def __init__(self, sess: SessionFrames) -> None:
        self.sess = sess
        self.width = sess.width
        self.height = sess.height
        self.is_color = sess.is_color
        self.name = str(sess.dir)
        self.total = len(sess)
        self.rewindable = True
        self._i = 0

    def next(self) -> Frame | None:
        if self._i >= self.total:
            return None
        f = self.sess.frame(self._i)
        self._i += 1
        return f

    def rewind(self) -> None:
        self._i = 0

    def close(self) -> None:
        pass


class _SourceStream:
    """Sequential view of a live/replay FrameSource."""

    def __init__(self, cfg: Config, calib: Calibration) -> None:
        from tbot.camera import open_source

        self.src = open_source(cfg.camera.source, cfg, calib)
        self.src.start()
        self.width = int(self.src.width)
        self.height = int(self.src.height)
        self.is_color = bool(self.src.is_color)
        self.name = getattr(self.src, "name", cfg.camera.source)
        self.total = None
        self.rewindable = False

    def next(self) -> Frame | None:
        return self.src.read()

    def rewind(self) -> None:
        pass

    def close(self) -> None:
        self.src.stop()


def _open_stream(args: argparse.Namespace, cfg: Config, calib: Calibration):
    if args.session:
        return _SessionStream(SessionFrames(args.session))
    return _SourceStream(cfg, calib)


def _runtime_mask(cfg: Config, calib: Calibration, width: int, height: int) -> np.ndarray:
    """Barrel mask (projected from the mount geometry) AND calib/mask.png."""
    from tbot.optics import CameraModel, MountModel, combined_mask

    cam = CameraModel.from_calibration(calib.intrinsics, cfg, width, height)
    mount = MountModel.from_config(cfg, calib.boresight)
    return combined_mask(cam, mount, cfg, calib)


# ------------------------------------------------------------------------------- report
def _report(stream, cfg: Config, mask: np.ndarray | None, n_frames: int) -> int:
    detector = ClayDetector(cfg, mask)
    counts: list[float] = []
    stage: dict[str, list[float]] = {"bg": [], "threshold": [], "components": [], "total": []}
    snrs: list[float] = []
    sigmas: list[float] = []
    diameters: list[float] = []
    frames = 0
    while n_frames <= 0 or frames < n_frames:
        f = stream.next()
        if f is None:
            break
        dets = detector.detect(f)
        frames += 1
        counts.append(len(dets))
        sigmas.append(detector.noise_sigma)
        for k, v in detector.timings_ms.items():
            stage.setdefault(k, []).append(v)
        for d in dets:
            snrs.append(d.snr)
            diameters.append(d.diameter_px)
    if frames == 0:
        print("calib-detector: no frames from the source", file=sys.stderr)
        return 1

    det = cfg.detector
    thr = max(det.k_sigma * sigmas[-1], det.min_contrast)
    masked = 0.0 if mask is None else float(np.count_nonzero(mask == 0)) / mask.size
    print(f"source        : {stream.name}  ({stream.width}x{stream.height}"
          f"{' colour' if stream.is_color else ' mono'})")
    print(f"frames        : {frames}")
    print(f"mode/polarity : {det.mode} / {det.polarity}   combine={det.color_combine}")
    print(f"noise sigma DN: last {sigmas[-1]:.2f}  {_fmt(summarize(sigmas))}")
    print(f"threshold DN  : {thr:.2f}  (k_sigma={det.k_sigma:g} x sigma vs min_contrast={det.min_contrast:g})")
    print(f"mask          : {masked * 100:.1f} % of pixels masked")
    print(f"detections/fr : {_fmt(summarize(counts))}")
    hit = float(np.count_nonzero(np.asarray(counts) > 0)) / frames
    print(f"frames w/ det : {hit * 100:.1f} %   total {int(sum(counts))}")
    if snrs:
        print(f"detection snr : {_fmt(summarize(snrs))}")
        print(f"diameter px   : {_fmt(summarize(diameters))}")
    print("timing ms     :")
    for name in ("bg", "threshold", "components", "total"):
        if stage.get(name):
            print(f"  {name:<10}: {_fmt(summarize(stage[name]))}")
    return 0


def _fmt(s: dict) -> str:
    return (f"mean {s['mean']:.3g}  p50 {s['p50']:.3g}  p95 {s['p95']:.3g}  "
            f"min {s['min']:.3g}  max {s['max']:.3g}  n {s['n']}")


# ------------------------------------------------------------------------------- psf calibration
def psf_from_diameters(diameters_px, true_diameter_px: float) -> dict:
    """System blur from observed blob sizes of a target of known angular size.

    Detection.diameter_px is the observed 4-sigma major axis, i.e. the object convolved with the
    system blur, so variances add: d_obs^2 = d_true^2 + psf^2. The median of the observations is used
    (a few frames always catch the target half-occluded or merged with noise).
    """
    d = np.asarray([float(x) for x in diameters_px], dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("no detections to calibrate the psf from")
    if not true_diameter_px > 0.0:
        raise ValueError(f"target diameter must be > 0 px, got {true_diameter_px}")
    d_obs = float(np.median(d))
    if d_obs <= true_diameter_px:
        raise ValueError(
            f"observed diameter {d_obs:.2f} px <= the target's {true_diameter_px:.2f} px: the blur "
            "cannot be negative, so --range-m or --diameter-mm is wrong (or the blob is not the target)")
    return {
        "psf_px": (d_obs * d_obs - true_diameter_px * true_diameter_px) ** 0.5,
        "d_obs_px": d_obs,
        "d_true_px": float(true_diameter_px),
        "n": int(d.size),
        "spread_px": float(np.percentile(d, 84) - np.percentile(d, 16)) / 2.0,
    }


def _psf(stream, cfg: Config, calib: Calibration, mask: np.ndarray | None,
         args: argparse.Namespace) -> int:
    from tbot.optics import CameraModel

    if not args.range_m or args.range_m <= 0.0:
        print("calib-detector: --psf needs --range-m <metres to the target>", file=sys.stderr)
        return 2
    detector = ClayDetector(cfg, mask)
    diameters: list[float] = []
    uvs: list[tuple[float, float]] = []
    frames = 0
    while args.frames <= 0 or frames < args.frames:
        f = stream.next()
        if f is None:
            break
        frames += 1
        dets = detector.detect(f)
        if dets:
            diameters.append(dets[0].diameter_px)
            uvs.append((dets[0].u, dets[0].v))
    if not diameters:
        print(f"calib-detector: no detections in {frames} frames - aim at the target, or tune first "
              "with --report/--gui", file=sys.stderr)
        return 1

    cam = CameraModel.from_calibration(calib.intrinsics, cfg, stream.width, stream.height)
    uv = np.asarray(uvs, dtype=np.float64)
    centre = np.median(uv, axis=0).reshape(1, 2)
    ifov = float(np.asarray(cam.ifov(centre)).reshape(-1)[0])
    true_px = (args.diameter_mm * 1e-3) / (args.range_m * ifov)
    try:
        res = psf_from_diameters(diameters, true_px)
    except ValueError as exc:
        print(f"calib-detector: {exc}", file=sys.stderr)
        return 1
    update_json(calib.path("detector.json"), {"psf_px": res["psf_px"]})
    print(f"frames        : {frames} ({len(diameters)} with a detection)")
    print(f"target        : {args.diameter_mm:g} mm at {args.range_m:g} m, "
          f"ifov {ifov * 1e3:.3f} mrad/px at ({centre[0, 0]:.1f}, {centre[0, 1]:.1f})")
    print(f"d_true        : {res['d_true_px']:.3f} px")
    print(f"d_obs (median): {res['d_obs_px']:.3f} px  +-{res['spread_px']:.3f} (1 sigma)")
    print(f"psf_px        : {res['psf_px']:.3f} px  -> {calib.path('detector.json')}")
    return 0


# ------------------------------------------------------------------------------- auto mask
def _auto_mask(args: argparse.Namespace, calib: Calibration) -> int:
    sess = SessionFrames(args.auto_mask)
    stack = sess.stack
    if args.frames and args.frames > 0:
        stack = stack[: args.frames]
    mask = static_mask(stack, static_std=args.static_std, min_area_px=args.min_area,
                       dilate_px=args.dilate)
    out = calib.path("mask.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), mask):
        print(f"calib-detector: could not write {out}", file=sys.stderr)
        return 1
    frac = float(np.count_nonzero(mask == 0)) / mask.size
    print(f"frames used   : {len(stack)} of {len(sess)} ({sess.width}x{sess.height})")
    print(f"static_std    : {args.static_std:g} DN   min_area {args.min_area} px   dilate {args.dilate} px")
    print(f"masked        : {frac * 100:.1f} % of pixels")
    print(f"written       : {out}")
    return 0


# ------------------------------------------------------------------------------- tuner (GUI)
def _clamp_config(cfg: Config) -> None:
    det = cfg.detector
    det.bg_kernel = max(1, int(det.bg_kernel) | 1)
    det.min_area_px = max(1, int(det.min_area_px))
    det.max_area_px = max(det.min_area_px, int(det.max_area_px))


def _make_trackbars(cfg: Config) -> None:
    det = cfg.detector
    nop = lambda _v: None  # noqa: E731 - OpenCV requires a callback
    cv2.createTrackbar("k_sigma x10", WINDOW, min(int(det.k_sigma * 10), _K_SIGMA_MAX), _K_SIGMA_MAX, nop)
    cv2.createTrackbar("min_contrast", WINDOW, min(int(det.min_contrast), _MIN_CONTRAST_MAX),
                       _MIN_CONTRAST_MAX, nop)
    cv2.createTrackbar("bg_kernel", WINDOW, min(int(det.bg_kernel), _BG_KERNEL_MAX), _BG_KERNEL_MAX, nop)
    cv2.createTrackbar("min_area_px", WINDOW, min(int(det.min_area_px), _MIN_AREA_MAX), _MIN_AREA_MAX, nop)
    cv2.createTrackbar("max_area_px", WINDOW, min(int(det.max_area_px), _MAX_AREA_MAX), _MAX_AREA_MAX, nop)
    cv2.createTrackbar("polarity", WINDOW, _POLARITIES.index(det.polarity), 2, nop)


def _read_trackbars(cfg: Config) -> None:
    det = cfg.detector
    det.k_sigma = cv2.getTrackbarPos("k_sigma x10", WINDOW) / 10.0
    det.min_contrast = float(cv2.getTrackbarPos("min_contrast", WINDOW))
    det.bg_kernel = cv2.getTrackbarPos("bg_kernel", WINDOW)
    det.min_area_px = cv2.getTrackbarPos("min_area_px", WINDOW)
    det.max_area_px = cv2.getTrackbarPos("max_area_px", WINDOW)
    det.polarity = _POLARITIES[cv2.getTrackbarPos("polarity", WINDOW)]
    _clamp_config(cfg)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _panes(frame: Frame, dets: list[Detection], detector: ClayDetector, thr: float) -> np.ndarray:
    img = frame.image
    view = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(dets):
        r = max(6, int(round(d.diameter_px)))
        cv2.circle(view, (int(round(d.u)), int(round(d.v))), r, (0, 255, 0) if i == 0 else (0, 200, 255), 1)
        cv2.putText(view, f"{d.score:.2f} s{d.snr:.0f} d{d.diameter_px:.1f}",
                    (int(d.u) + r + 2, int(d.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    _label(view, f"frame {frame.seq}  dets {len(dets)}")

    contrast = detector.debug["contrast"]
    gain = 127.0 / max(thr, 1.0)
    cmap = np.clip(128.0 + contrast.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_GRAY2BGR)
    _label(cmap, f"contrast x{gain:.1f}  sigma {detector.noise_sigma:.2f} thr {thr:.1f}")

    binary = cv2.cvtColor(detector.debug["binary"], cv2.COLOR_GRAY2BGR)
    _label(binary, f"binary  {detector.timings_ms['total']:.1f} ms")
    return np.hstack([view, cmap, binary])


def _tune(stream, cfg: Config, calib: Calibration, mask: np.ndarray | None) -> int:
    cfg.detector.debug = True
    _clamp_config(cfg)
    detector = ClayDetector(cfg, mask)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    _make_trackbars(cfg)
    print("keys: s = save calib/detector.json, space = pause, q = quit")
    paused = False
    frame: Frame | None = None
    while True:
        if not paused or frame is None:
            nxt = stream.next()
            if nxt is None:
                if stream.rewindable:
                    stream.rewind()
                    nxt = stream.next()
                if nxt is None:
                    print("end of stream")
                    break
            frame = nxt
        _read_trackbars(cfg)
        dets = detector.detect(frame)
        thr = max(cfg.detector.k_sigma * detector.noise_sigma, cfg.detector.min_contrast)
        cv2.imshow(WINDOW, _panes(frame, dets, detector, thr))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            path = _save(cfg, calib)
            print(f"saved {path}")
    cv2.destroyAllWindows()
    return 0


def _save(cfg: Config, calib: Calibration) -> Path:
    fields = dataclasses.asdict(cfg.detector)
    fields["debug"] = False  # debug images are a tuner-only cost
    return save_json(calib.path("detector.json"), fields)


# ------------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tbot calib-detector",
        description="Tune the clay detector, measure its psf, and build calib/mask.png.")
    add_common_args(ap)
    ap.add_argument("--session", metavar="DIR", help="run over a recorded session instead of a live source")
    ap.add_argument("--gui", action="store_true", help="open OpenCV windows (required by the tuner)")
    ap.add_argument("--report", action="store_true",
                    help="headless: detection/timing statistics over --frames frames")
    ap.add_argument("--frames", type=int, default=200, metavar="N",
                    help="frames to process (0 = all / until the stream ends)")
    ap.add_argument("--psf", action="store_true",
                    help="measure cfg.detector.psf_px from a target of known size at --range-m")
    ap.add_argument("--range-m", type=float, default=0.0, metavar="M",
                    help="measured distance to the calibration target (--psf)")
    ap.add_argument("--diameter-mm", type=float, default=108.0, metavar="MM",
                    help="true diameter of the calibration target (--psf; a standard clay is 108 mm)")
    ap.add_argument("--auto-mask", metavar="SESSION_DIR",
                    help="build calib/mask.png from a recording made while swinging the gun")
    ap.add_argument("--static-std", type=float, default=3.0, metavar="DN",
                    help="a pixel is static when its temporal std is below this (--auto-mask)")
    ap.add_argument("--min-area", type=int, default=200, metavar="PX",
                    help="smallest static border region that gets masked (--auto-mask)")
    ap.add_argument("--dilate", type=int, default=4, metavar="PX",
                    help="grow the masked region by this many pixels (--auto-mask)")
    ap.add_argument("--no-mask", action="store_true", help="ignore the barrel/calibration mask")
    args = ap.parse_args(argv)

    cfg, calib = load_from_args(args)
    if args.auto_mask:
        return _auto_mask(args, calib)
    if not args.report and not args.psf and not args.gui:
        print("calib-detector: the interactive tuner needs --gui (or use --report / --psf headless)",
              file=sys.stderr)
        return 2

    stream = _open_stream(args, cfg, calib)
    try:
        mask = None if args.no_mask else _runtime_mask(cfg, calib, stream.width, stream.height)
        if args.psf:
            return _psf(stream, cfg, calib, mask, args)
        if args.report:
            return _report(stream, cfg, mask, args.frames)
        return _tune(stream, cfg, calib, mask)
    finally:
        stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
