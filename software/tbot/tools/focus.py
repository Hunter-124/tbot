"""Headless focus and exposure aid; no actuator or GPIO access."""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from tbot.app.stream import DebugServer
from tbot.camera import linkcheck, open_live_source
from tbot.config import add_common_args, load_from_args
from tbot.optics import CameraModel, MountModel, combined_mask
from tbot.types import SimClock


def measure(image: np.ndarray, mask: np.ndarray, roi: float) -> dict:
    """Mean squared adjacent-pixel gradient (DN²) and exposure statistics on usable pixels.

    Both endpoints of each gradient must be unmasked, so the mask boundary cannot
    introduce artificial sharpness. Exposure statistics cover the full usable frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    valid = mask != 0
    height, width = gray.shape
    pixels = gray[valid]
    values = gray.astype(np.int16)

    def score(y0: int, y1: int, x0: int, x1: int) -> float | None:
        v = values[y0:y1, x0:x1]
        m = valid[y0:y1, x0:x1]
        dx = np.diff(v, axis=1).astype(np.float32)
        dy = np.diff(v, axis=0).astype(np.float32)
        mx = m[:, 1:] & m[:, :-1]
        my = m[1:, :] & m[:-1, :]
        count = int(mx.sum()) + int(my.sum())
        if not count:
            return None
        return float(((dx[mx] ** 2).sum(dtype=np.float64) +
                      (dy[my] ** 2).sum(dtype=np.float64)) / count)

    rw, rh = max(1, int(width * roi)), max(1, int(height * roi))
    x0, y0 = (width - rw) // 2, (height - rh) // 2
    mx, my = width // 2, height // 2
    return {
        "sharpness": score(y0, y0 + rh, x0, x0 + rw),
        "quadrants": {
            "top_left": score(0, my, 0, mx), "top_right": score(0, my, mx, width),
            "bottom_left": score(my, height, 0, mx), "bottom_right": score(my, height, mx, width),
        },
        "saturated_pct": float(np.count_nonzero(pixels == 255) * 100.0 / pixels.size) if pixels.size else None,
        "mean_dn": float(np.mean(pixels)) if pixels.size else None,
        "median_dn": float(np.median(pixels)) if pixels.size else None,
    }


def _number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _status(data: dict) -> str:
    q = data["quadrants"]
    return (f"sharp {_number(data['sharpness'])} peak {_number(data['peak'])} "
            f"({_number(data['peak_pct'])}%) "
            f"quad TL/TR/BL/BR {'/'.join(_number(q[k]) for k in ('top_left', 'top_right', 'bottom_left', 'bottom_right'))} "
            f"sat {_number(data['saturated_pct'], 2)}% mean/median {_number(data['mean_dn'])}/{_number(data['median_dn'])} DN "
            f"{data['fps']:.1f} fps exposure {data['exposure_us']:.0f} us gain {data['gain']:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbot focus", description="Focus and exposure aid (sharpness is mean squared gradient in DN²).")
    add_common_args(parser)
    parser.add_argument("--roi", type=float, default=0.5, metavar="FRAC", help="central ROI fraction of width and height (default 0.5)")
    parser.add_argument("--seconds", type=float, default=None, help="stop after this much frame time (default: end of stream or Ctrl-C)")
    parser.add_argument("--stream", action="store_true", help="serve annotated frames on the configured debug server")
    parser.add_argument("--json", action="store_true", help="write final JSON summary to stdout; live status goes to stderr")
    args = parser.parse_args(argv)
    if not np.isfinite(args.roi) or not 0 < args.roi <= 1:
        parser.error("--roi must be in (0, 1]")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive")
    cfg, calib = load_from_args(args)
    # live camera: a CSI stall reopens the camera instead of ending the session (RecoveringSource)
    source = open_live_source(cfg.camera.source, cfg, calib)
    live = bool(getattr(source, "endless", False))
    server = None
    frame = None
    mask = None
    data = None
    peak = None
    first_ns = last_ns = None
    frames = timeouts = 0
    next_report = 0.0
    status = 0
    down_since: float | None = None
    next_diag = 0.0
    out = sys.stderr if args.json else sys.stdout
    try:
        source.start()
        if args.stream:
            server = DebugServer(cfg).start()
            print(f"tbot focus: stream http://{server.host}:{server.port}/", file=out, flush=True)
        while True:
            next_frame = source.read(timeout_s=0.5)
            now = time.monotonic()
            if next_frame is None:
                if isinstance(source.clock, SimClock):
                    break
                timeouts += 1
                if not live:
                    if timeouts >= 20:
                        print("tbot focus: no frames for 10 s", file=sys.stderr)
                        status = 1
                        break
                    continue
                down_since = now if down_since is None else down_since
                if args.seconds is not None and now - down_since > max(10.0, args.seconds):
                    print("tbot focus: no frames, giving up", file=sys.stderr)
                    status = 1
                    break
                if now - down_since >= 5.0 and now >= next_diag and getattr(source, "streaming", True):
                    diag = linkcheck.diagnose(window_s=1.5, source=source)
                    print(f"tbot focus: camera down for {now - down_since:.0f} s, still retrying "
                          f"(restarts {getattr(source, 'restarts', 0)}) - {diag.layer}: {diag.message}",
                          file=sys.stderr, flush=True)
                    if server is not None:
                        server.publish(linkcheck.down_card(cfg.camera.width, cfg.camera.height,
                                                           f"CAMERA DOWN ({diag.layer})", diag.message),
                                       {"camera": "down", "since_s": round(now - down_since, 1), **diag.as_dict()})
                    next_diag = time.monotonic() + 10.0
                continue
            if down_since is not None:
                print(f"tbot focus: camera back after {now - down_since:.1f} s", file=sys.stderr, flush=True)
                down_since = None
                next_diag = 0.0
            frame = next_frame
            timeouts = 0
            frames += 1
            if first_ns is None:
                first_ns = frame.t_capture_ns
            last_ns = frame.t_capture_ns
            if mask is None or mask.shape != frame.image.shape[:2]:
                cam = CameraModel.from_calibration(calib.intrinsics, cfg, frame.width, frame.height)
                mount = MountModel.from_config(cfg, calib.boresight)
                mask = combined_mask(cam, mount, cfg, calib)
            elapsed = max(0.0, (last_ns - first_ns) * 1e-9)
            done = args.seconds is not None and elapsed >= args.seconds
            if now >= next_report or done:
                data = measure(frame.image, mask, args.roi)
                sharp = data['sharpness']
                if sharp is not None:
                    peak = sharp if peak is None else max(peak, sharp)
                data.update(peak=peak, peak_pct=(100.0 * sharp / peak if sharp is not None and peak else None),
                            fps=(frames - 1) / elapsed if elapsed > 0 else 0.0,
                            exposure_us=float(frame.exposure_us), gain=float(frame.gain),
                            frames=frames, elapsed_s=elapsed, source=cfg.camera.source, roi=args.roi)
                print(_status(data), file=out, flush=True)
                next_report = now + 0.5
            if server is not None and data is not None and server.due():
                bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR) if frame.image.ndim == 2 else frame.image.copy()
                cv2.putText(bgr, f"focus {_number(data['sharpness'])} / peak {_number(data['peak'])} ({_number(data['peak_pct'])}%)",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                server.publish(bgr, data)
            if done:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
        source.stop()
    if data is None:
        print("tbot focus: no frames captured", file=sys.stderr)
        return 1
    if frames != data["frames"]:
        data = measure(frame.image, mask, args.roi)
        sharp = data["sharpness"]
        if sharp is not None:
            peak = sharp if peak is None else max(peak, sharp)
        elapsed = max(0.0, (last_ns - first_ns) * 1e-9)
        data.update(peak=peak, peak_pct=(100.0 * sharp / peak if sharp is not None and peak else None),
                    fps=(frames - 1) / elapsed if elapsed > 0 else 0.0,
                    exposure_us=float(frame.exposure_us), gain=float(frame.gain),
                    frames=frames, elapsed_s=elapsed, source=cfg.camera.source, roi=args.roi)
        print(_status(data), file=out, flush=True)
    if args.json:
        print(json.dumps(data, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
