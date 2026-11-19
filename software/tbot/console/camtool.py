"""Camera helper process for the console (pipeline stopped): live picture, focus/exposure numbers, bore-laser
twist capture.

    python -m tbot.console.camtool --port 8081 [--set camera.width=1280 ...]

The picture is served as MJPEG by a DebugServer on 127.0.0.1:<port> (the console proxies it). Commands arrive
as JSON lines on stdin, results leave as JSON lines on stdout; the log goes to stderr. Nothing here touches
GPIO or the actuator.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from tbot.app.stream import DebugServer, jsonable
from tbot.camera import open_live_source
from tbot.config import add_common_args, load_from_args, resolve_path
from tbot.console.twist import find_dot, fit_ring
from tbot.optics import CameraModel, MountModel, combined_mask
from tbot.tools.focus import measure


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(jsonable(obj), allow_nan=False) + "\n")
    sys.stdout.flush()


def _stdin_reader(q: "queue.Queue[dict]") -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            q.put(json.loads(line))
        except ValueError:
            emit({"type": "error", "message": f"bad command: {line[:80]}"})
    q.put({"cmd": "quit"})


def _stored_rpy_deg(calib) -> list[float] | None:
    b = calib.boresight or {}
    rpy = b.get("rpy_deg")
    return [float(x) for x in rpy] if isinstance(rpy, list) and len(rpy) == 3 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot.console.camtool")
    add_common_args(ap)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--stream-fps", type=float, default=12.0)
    ap.add_argument("--runs-dir", default="calib_runs")
    args = ap.parse_args(argv)
    cfg, calib = load_from_args(args)
    if cfg.camera.auto_exposure:  # a still scene: long exposure, low gain (as tbot focus does)
        cfg.camera.ae_max_exposure_us = min(20000.0, 0.9e6 / max(float(cfg.camera.fps), 1.0))
        cfg.camera.ae_max_gain = min(float(cfg.camera.ae_max_gain), 4.0)
        cfg.camera.ae_hold_s, cfg.camera.ae_up_stops_per_s = 0.0, 8.0
    cfg.stream.host, cfg.stream.port = "127.0.0.1", int(args.port)
    cfg.stream.fps, cfg.stream.jpeg_quality = float(args.stream_fps), 80
    runs = resolve_path(args.runs_dir)

    cmds: "queue.Queue[dict]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(cmds,), daemon=True).start()

    source = open_live_source(cfg.camera.source, cfg, calib)
    server = DebugServer(cfg)
    try:
        source.start()
        server.start()
    except Exception as exc:  # noqa: BLE001 - report why the camera did not open, then leave
        emit({"type": "fatal", "message": f"camera did not open: {exc}"})
        return 1

    cam = mount = mask = None
    peak = None
    last_stats = 0.0
    frames = 0
    t0 = time.monotonic()
    fps_win: list[float] = []
    capture: dict | None = None
    bore_range: float | None = None
    last_dot_emit = 0.0
    ready_sent = False
    frame = None
    last_wait = 0.0
    try:
        while True:
            try:
                while True:
                    c = cmds.get_nowait()
                    kind = c.get("cmd")
                    if kind == "quit":
                        return 0
                    if kind == "bore":
                        bore_range = float(c["range_m"]) if c.get("range_m") else None
                        last_stats = 0.0
                    elif kind == "reset_peak":
                        peak = None
                    elif kind == "capture":
                        win = c.get("window")
                        capture = {"until": time.monotonic() + float(c.get("seconds", 10.0)), "rows": [],
                                   "window": tuple(float(v) for v in win) if win else None,
                                   "range_m": c.get("range_m"), "first": None, "t0": time.monotonic(),
                                   "seconds": float(c.get("seconds", 10.0))}
                        emit({"type": "capture_started", "seconds": capture["seconds"]})
                    elif kind == "cancel_capture":
                        capture = None
                        emit({"type": "capture_cancelled"})
                    elif kind == "find_dot" and frame is not None:
                        win = c.get("window")
                        d = find_dot(frame.image if frame.image.ndim == 2 else
                                     cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY),
                                     tuple(float(v) for v in win) if win else None)
                        emit({"type": "dot_probe", "dot": list(d) if d else None})
                    elif kind == "still" and frame is not None:
                        out = runs / "stills" / time.strftime("%Y%m%d-%H%M%S.png")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out), frame.image)
                        emit({"type": "still", "path": str(out)})
            except queue.Empty:
                pass

            f = source.read(timeout_s=0.5)
            if f is None:
                if not getattr(source, "endless", False):  # a recording or a synthetic scene ended: loop it
                    source.stop()
                    source = open_live_source(cfg.camera.source, cfg, calib)
                    source.start()
                    continue
                if time.monotonic() - last_wait > 5.0:
                    emit({"type": "waiting", "message": "no frame from the camera"})
                    last_wait = time.monotonic()
                continue
            frame = f
            frames += 1
            now = time.monotonic()
            fps_win.append(now)
            fps_win = [t for t in fps_win if now - t < 2.0]
            gray = f.image if f.image.ndim == 2 else cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
            if mask is None or mask.shape != gray.shape:
                cam = CameraModel.from_calibration(calib.intrinsics, cfg, f.width, f.height)
                mount = MountModel.from_config(cfg, calib.boresight)
                mask = combined_mask(cam, mount, cfg, calib)
            if not ready_sent:
                intr = calib.intrinsics or {}
                emit({"type": "ready", "width": f.width, "height": f.height,
                      "intrinsics_size": [int(intr.get("width", f.width)), int(intr.get("height", f.height))],
                      "rpy_deg": _stored_rpy_deg(calib), "auto_exposure": bool(cfg.camera.auto_exposure)})
                ready_sent = True

            if capture is not None:
                d = find_dot(gray, capture["window"])
                if capture["first"] is None:
                    capture["first"] = gray.copy()
                if d is not None:
                    capture["rows"].append(d)
                    if now - last_dot_emit > 0.06:
                        emit({"type": "dot", "x": d[0], "y": d[1], "snr": d[2],
                              "n": len(capture["rows"]), "left_s": max(0.0, capture["until"] - now)})
                        last_dot_emit = now
                if now >= capture["until"]:
                    _finish_capture(capture, runs, f.width, f.height)
                    capture = None

            if server.due():
                server.publish(gray if gray.ndim == 2 else gray, None)

            if now - last_stats >= 0.25:
                last_stats = now
                m = measure(gray, mask, 0.5)
                sharp = m["sharpness"]
                if sharp is not None:
                    peak = sharp if peak is None else max(peak, sharp)
                hist = np.bincount((gray[mask != 0] >> 2).ravel(), minlength=64)
                bore = None
                if bore_range and cam is not None:
                    rpy = _stored_rpy_deg(calib)
                    if rpy is not None:
                        from tbot.tools.calib_boresight import mount_for_rpy

                        mnt = mount_for_rpy(cfg, np.radians(rpy))
                        u, v = mnt.bore_pixel(cam, float(bore_range))
                        if math.isfinite(u) and math.isfinite(v):
                            bore = [float(u), float(v)]
                emit({"type": "stats", **m, "peak": peak,
                      "peak_pct": 100.0 * sharp / peak if sharp is not None and peak else None,
                      "fps": (len(fps_win) - 1) / max(1e-3, fps_win[-1] - fps_win[0]) if len(fps_win) > 1 else 0.0,
                      "exposure_us": float(f.exposure_us), "gain": float(f.gain), "frames": frames,
                      "width": f.width, "height": f.height, "hist": hist.tolist(), "bore_px": bore,
                      "bore_range_m": bore_range, "uptime_s": now - t0, "capturing": capture is not None})
    finally:
        server.stop()
        source.stop()


def _finish_capture(capture: dict, runs: Path, width: int, height: int) -> None:
    rows = capture["rows"]
    out_dir = runs / time.strftime("boresight-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "dots.csv", np.asarray(rows).reshape(-1, 4), delimiter=",", header="x,y,snr,snr2",
               fmt="%.2f")
    try:
        ring = fit_ring(rows)
    except ValueError as exc:
        emit({"type": "ring_error", "message": str(exc), "frames": len(rows), "dir": str(out_dir)})
        return
    if capture["first"] is not None:
        vis = cv2.cvtColor(capture["first"], cv2.COLOR_GRAY2BGR)
        for x, y, k in ring["points"]:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0) if k else (0, 0, 255), -1)
        c = (int(round(ring["cx"])), int(round(ring["cy"])))
        cv2.circle(vis, c, int(round(ring["r"])), (255, 200, 0), 1)
        cv2.drawMarker(vis, c, (255, 200, 0), cv2.MARKER_CROSS, 14, 1)
        cv2.imwrite(str(out_dir / "twist_fit.png"), vis)
    result = {**ring, "width": width, "height": height, "range_m": capture["range_m"], "dir": str(out_dir),
              "seconds": capture["seconds"]}
    (out_dir / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k != "points"}, indent=1))
    emit({"type": "ring", **result})


if __name__ == "__main__":
    raise SystemExit(main())
