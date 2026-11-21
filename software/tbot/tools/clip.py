"""`tbot clip <session> ...` - annotated video of a recorded press (clip.mp4 in the session directory).

`tbot run --record armed` saves every held press of the arming button as a session: the raw frames
(replayable), IMU, events, and telemetry.jsonl with what the pipeline computed on each frame (detections with
their boxes, tracks, the fire solution, timing, exposure). This tool draws that record back onto the frames:
the same overlay as the live stream, on EVERY recorded frame, with the time relative to the press in the
corner. It never re-runs the detector, so the clip shows what the gun saw and decided at the time; to try a
changed detector on the same frames use `tbot run --source replay:<session>`.

The pipeline starts it as a low-priority child process after each release (record.clip = "process"), so the
drawing and encoding never sit in the frame loop. `tbot clip --missing` renders every session that has no
clip yet (e.g. after the service was stopped mid-render).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tbot.app import overlay
from tbot.app.record import TELEMETRY_JSONL
from tbot.camera.replay import FRAMES_BIN, FRAMES_CSV, FRAMES_CSV_HEADER, META_JSON
from tbot.config import Config, _build, load_config, resolve_path
from tbot.optics import CameraModel, MountModel, barrel_mask

CLIP_NAME = "clip"
TMP_SUFFIX = ".part"


class VideoOut:
    """H.264 in MP4 (plays in a browser straight off the Pi's debug page) through PyAV when it is installed,
    else whatever OpenCV's own writer can do on this machine: H.264, MPEG-4, or Motion-JPEG in an AVI."""

    def __init__(self, stem: Path, width: int, height: int, fps: float) -> None:
        self.width, self.height = int(width), int(height)
        self.pad = (self.width % 2, self.height % 2)  # yuv420 needs even sizes
        self._av = self._cv = None
        self.path: Path | None = None
        w, h = self.width + self.pad[0], self.height + self.pad[1]
        try:
            import av  # noqa: PLC0415

            path = stem.with_suffix(".mp4")
            tmp = path.with_name(path.name + TMP_SUFFIX)
            container = av.open(str(tmp), "w", format="mp4", options={"movflags": "faststart"})
            from fractions import Fraction  # noqa: PLC0415

            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.options = {"preset": "ultrafast", "crf": "24", "threads": "2"}
            self._av = (av, container, stream)
            self.path, self._tmp = path, tmp
            return
        except Exception:  # noqa: BLE001 - no PyAV / no libx264: OpenCV's writer
            self._av = None
        tries = [("avc1", ".mp4", None), ("mp4v", ".mp4", None), ("MJPG", ".avi", None)]
        if sys.platform == "win32":
            # the laptop: OpenCV's ffmpeg there has no H.264 encoder (and mp4v does not play in a browser), but
            # Windows' own Media Foundation has one
            tries.insert(0, ("H264", ".mp4", cv2.CAP_MSMF))
        for fourcc, suffix, api in tries:
            path = stem.with_suffix(suffix)
            tmp = path.with_name(path.stem + TMP_SUFFIX + suffix)  # OpenCV picks the container from the suffix
            if api is None:
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
            else:
                writer = cv2.VideoWriter(str(tmp), api, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
            if writer.isOpened():
                self._cv, self.path, self._tmp = writer, path, tmp
                return
            writer.release()
            tmp.unlink(missing_ok=True)
        raise RuntimeError("no usable video encoder (PyAV/libx264 or OpenCV avc1/mp4v/MJPG)")

    def write(self, bgr: np.ndarray) -> None:
        if self.pad != (0, 0):
            bgr = cv2.copyMakeBorder(bgr, 0, self.pad[1], 0, self.pad[0], cv2.BORDER_REPLICATE)
        if self._av is not None:
            av, container, stream = self._av
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")):
                container.mux(packet)
        else:
            self._cv.write(bgr)

    def close(self, keep: bool = True) -> Path | None:
        if self._av is not None:
            _, container, stream = self._av
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
        elif self._cv is not None:
            self._cv.release()
        self._av = self._cv = None
        if not keep:
            self._tmp.unlink(missing_ok=True)
            return None
        self._tmp.replace(self.path)  # a clip that exists is a complete clip
        return self.path


def existing_clip(session: Path) -> Path | None:
    for suffix in (".mp4", ".avi"):
        p = session / (CLIP_NAME + suffix)
        if p.exists():
            return p
    return None


def _load_telemetry(session: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    p = session / TELEMETRY_JSONL
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line
            if "seq" in rec:
                out[int(rec["seq"])] = rec
    return out


def _frame_rows(session: Path) -> list[tuple[int, int, int]]:
    """(seq, t_exposure_ns, t_capture_ns) per stored frame."""
    rows = []
    with open(session / FRAMES_CSV, newline="") as f:
        head = f.readline().strip().split(",")
        if head != FRAMES_CSV_HEADER:
            raise ValueError(f"{session / FRAMES_CSV}: unexpected header {head}")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def render(session: str | Path, out: str | Path | None = None, fps: float | None = None,
           scale: int = 1, threads: int = 1, detail: str | None = None) -> Path:
    """Write the annotated clip of one session and return its path. detail: the overlay's "clean" | "debug"
    (None: stream.overlay_detail of the session's config, clean for a session that has none)."""
    session = Path(session)
    meta = json.loads((session / META_JSON).read_text())
    w, h, ch = int(meta["width"]), int(meta["height"]), int(meta.get("channels", 1))
    if threads > 0:
        cv2.setNumThreads(int(threads))  # share the Pi with the live pipeline
    try:
        cfg: Config = _build(Config, meta["config"], "")
    except Exception:  # noqa: BLE001 - a session from another software version: today's config still draws it
        cfg = load_config()
    calibration = meta.get("calibration") or {}
    cam = CameraModel.from_calibration(calibration.get("intrinsics"), cfg, w, h)
    mount = MountModel.from_config(cfg, calibration.get("boresight"))
    try:
        mask = np.asarray(barrel_mask(cam, mount, cfg))
    except Exception:  # noqa: BLE001
        mask = None
    bore_px = mount.bore_pixel(cam, float(cfg.tracker.range_prior_m))
    detail = overlay.norm_detail(detail or getattr(getattr(cfg, "stream", None), "overlay_detail", None))
    telemetry = _load_telemetry(session)
    rows = _frame_rows(session)
    if not rows:
        raise ValueError(f"{session}: no frames")
    trig = meta.get("trigger") or {}
    press = trig.get("press_ns") or [trig.get("t_ns", rows[0][2])] * 2
    t_zero = int(press[0])
    fps = float(fps or getattr(cfg.record, "clip_fps", 30.0) or 30.0)
    scale = max(1, int(scale))
    stem = Path(out).with_suffix("") if out else session / CLIP_NAME
    video = VideoOut(stem, w * scale, h * scale, fps)
    frame_bytes = w * h * ch
    shape = (h, w) if ch == 1 else (h, w, ch)
    ok = False
    try:
        with open(session / FRAMES_BIN, "rb") as f:
            for i, (seq, _t_exp, t_cap) in enumerate(rows):
                buf = f.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                image = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
                rel = (t_cap - t_zero) / 1e9
                held = int(press[0]) <= t_cap <= int(press[1])
                banner = f"{'HELD ' if held else ''}t{rel:+.3f}s  frame {i + 1}/{len(rows)}"
                rec = telemetry.get(seq)
                if rec is None:  # a frame the pipeline never processed (it was replaced by a newer one)
                    bgr = overlay.draw(image, mask=mask, bore_px=bore_px, telemetry={"seq": seq},
                                       detail=detail)
                    overlay._text(bgr, (max(150, w - 8 * len(banner) - 6), 15), banner)
                else:
                    bgr = overlay.draw_record(image, rec, cam, mount, mask=mask, bore_px=bore_px, banner=banner,
                                              detail=detail)
                if scale > 1:
                    bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
                video.write(bgr)
        ok = True
    finally:
        path = video.close(keep=ok)
    return path


def sessions_missing_clips(root: Path) -> list[Path]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / META_JSON).exists() and (d / FRAMES_BIN).exists() and (d / TELEMETRY_JSONL).exists() \
                and existing_clip(d) is None:
            out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbot clip", description="annotated video of a recorded press")
    parser.add_argument("sessions", nargs="*", metavar="SESSION", help="session directories (sessions/<name>)")
    parser.add_argument("--missing", action="store_true",
                        help="every session under record.dir (or --root) that has no clip yet")
    parser.add_argument("--root", default=None, help="sessions directory for --missing (default: record.dir)")
    parser.add_argument("--fps", type=float, default=None, help="playback rate (default: record.clip_fps)")
    parser.add_argument("--scale", type=int, default=1, help="integer upscale of the picture (default 1)")
    parser.add_argument("--out", default=None, help="output file (one session only; default <session>/clip.mp4)")
    parser.add_argument("--detail", choices=overlay.DETAILS, default=None,
                        help="overlay detail: clean (tracks, solution, bore) or debug (every detection, chip and "
                             "timing); default stream.overlay_detail of the session's config")
    parser.add_argument("--threads", type=int, default=1, help="OpenCV threads (default 1; 0 = all cores)")
    parser.add_argument("--nice", type=int, default=0, help="lower this process's priority by N (the Pi service uses 15)")
    args = parser.parse_args(argv)
    if args.nice and hasattr(os, "nice"):
        try:
            os.nice(int(args.nice))
        except OSError:
            pass
    targets = [Path(s) for s in args.sessions]
    if args.missing:
        root = Path(args.root) if args.root else resolve_path(load_config().record.dir)
        if root.is_dir():
            targets += sessions_missing_clips(root)
    if not targets:
        parser.error("give a session directory, or --missing")
    if args.out and len(targets) != 1:
        parser.error("--out needs exactly one session")
    status = 0
    for session in targets:
        t0 = time.monotonic()
        try:
            path = render(session, out=args.out, fps=args.fps, scale=args.scale, threads=args.threads,
                          detail=args.detail)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the rest
            print(f"tbot clip: {session}: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"tbot clip: {path}  ({path.stat().st_size / 1e6:.1f} MB, {time.monotonic() - t0:.1f} s)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
