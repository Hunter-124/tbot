"""Session previews for the console: thumb.jpg (the key moment of a press, drawn like the clip) + preview.json.

    python -m tbot.console.thumbs SESSION [SESSION ...]

The key moment is the shot when there was one, else the frame the solver liked best (highest P(hit)), else the
frame with the most confirmed tracks / detections, else the middle of the press. It is rendered from the raw
frames exactly as `tbot clip` draws that frame, scaled to THUMB_W, with the target magnified in an inset when there
is one (a clay is a few pixels in a 640x400 frame; at thumbnail size it would be invisible). preview.json says
which frame and where it sits in the clip (clip_t), so the list can play a hover preview from just before it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tbot.app import overlay
from tbot.camera.replay import FRAMES_BIN, META_JSON
from tbot.config import Config, _build, load_config
from tbot.optics import CameraModel, MountModel, barrel_mask
from tbot.tools.clip import _frame_rows, _load_telemetry

THUMB_W = 480
PREVIEW_VERSION = 2   # bump when previews should be drawn again (sessions.needs_thumb compares it)
INSET = 0.34          # inset width as a share of the thumbnail
INSET_CROP_PX = 56    # half-size of the region around the target that the inset magnifies (full-res pixels)


def key_frame(rows: list[tuple[int, int, int]], telemetry: dict[int, dict], press: list[int] | None) -> tuple[int, str]:
    """(row index into frames.csv, reason)."""
    index = {seq: i for i, (seq, _, _) in enumerate(rows)}
    stored = [(index[s], r) for s, r in telemetry.items() if s in index]
    stored.sort(key=lambda t: t[0])
    for i, r in stored:
        if r.get("fired"):
            return i, "shot"
    best = max(stored, key=lambda t: float((t[1].get("decision") or {}).get("p_hit") or 0.0), default=None)
    if best and float((best[1].get("decision") or {}).get("p_hit") or 0.0) > 0.0:
        return best[0], "best P(hit)"

    def tracked(r: dict) -> tuple[int, int]:
        tr = [t for t in (r.get("tracks") or []) if t.get("confirmed")]
        return len(tr), len(r.get("dets") or [])

    best = max(stored, key=lambda t: tracked(t[1]), default=None)
    if best and tracked(best[1])[0] > 0:
        return best[0], "target tracked"
    if best and tracked(best[1])[1] > 0:
        return best[0], "detection"
    if press and rows:
        mid = (int(press[0]) + int(press[1])) // 2
        return min(range(len(rows)), key=lambda i: abs(rows[i][2] - mid)), "press"
    return len(rows) // 2, "middle"


def _display_stretch(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Preview only (the recording is untouched): stretch the picture's own 1st-99.5th percentile range to full
    scale, so a press filmed in dim light (indoors, dusk) is still recognisable. Relative to the frame itself, no
    fixed brightness numbers. Returns (image, gain); gain 1 = left as it was."""
    lo, hi = np.percentile(image, (1.0, 99.5))
    if hi - lo < 4.0 or (lo < 8 and hi > 240):
        return image, 1.0
    gain = 255.0 / float(hi - lo)
    out = np.clip((image.astype(np.float32) - float(lo)) * gain, 0, 255).astype(np.uint8)
    return out, gain


def _target_px(rec: dict | None, cam) -> tuple[float, float] | None:
    """Where to magnify: the strongest detection on the key frame."""
    if not rec:
        return None
    dets = rec.get("dets") or []
    if dets:
        d = max(dets, key=lambda d: float(d[3]) if len(d) > 3 else 0.0)
        return float(d[0]), float(d[1])
    return None


def render_thumb(session: Path) -> dict:
    meta = json.loads((session / META_JSON).read_text())
    w, h, ch = int(meta["width"]), int(meta["height"]), int(meta.get("channels", 1))
    try:
        cfg: Config = _build(Config, meta["config"], "")
    except Exception:  # noqa: BLE001 - an older session: today's config still draws it
        cfg = load_config()
    calibration = meta.get("calibration") or {}
    cam = CameraModel.from_calibration(calibration.get("intrinsics"), cfg, w, h)
    mount = MountModel.from_config(cfg, calibration.get("boresight"))
    try:
        mask = np.asarray(barrel_mask(cam, mount, cfg))
    except Exception:  # noqa: BLE001
        mask = None
    bore_px = mount.bore_pixel(cam, float(cfg.tracker.range_prior_m))
    telemetry = _load_telemetry(session)
    rows = _frame_rows(session)
    if not rows:
        raise ValueError("no frames")
    trig = meta.get("trigger") or {}
    press = trig.get("press_ns") or None
    i, reason = key_frame(rows, telemetry, press)
    seq, _t_exp, t_cap = rows[i]
    frame_bytes = w * h * ch
    with open(session / FRAMES_BIN, "rb") as f:
        f.seek(i * frame_bytes)
        buf = f.read(frame_bytes)
    if len(buf) < frame_bytes:
        raise ValueError("frames.bin is shorter than frames.csv")
    image = np.frombuffer(buf, dtype=np.uint8).reshape((h, w) if ch == 1 else (h, w, ch))
    image, gain = _display_stretch(image)
    rec = telemetry.get(seq)
    t_zero = int(press[0]) if press else rows[0][2]
    banner = f"t{(t_cap - t_zero) / 1e9:+.3f}s"
    if rec is None:
        bgr = overlay.draw(image, mask=mask, bore_px=bore_px, telemetry={"seq": seq})
    else:
        bgr = overlay.draw_record(image, rec, cam, mount, mask=mask, bore_px=bore_px, banner=banner)
    # draw_record adds text bands above/below the picture: find the picture inside them for the inset
    oh = bgr.shape[0]
    top = max(0, (oh - h) // 2) if oh > h else 0
    scale = THUMB_W / bgr.shape[1]
    thumb = cv2.resize(bgr, (THUMB_W, int(round(oh * scale))), interpolation=cv2.INTER_AREA)
    tgt = _target_px(rec, cam)
    if tgt is not None:
        u, v = tgt
        c = INSET_CROP_PX
        x0, y0 = int(np.clip(u - c, 0, w - 2 * c)), int(np.clip(v - c, 0, h - 2 * c))
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crop = cv2.cvtColor(gray[y0:y0 + 2 * c, x0:x0 + 2 * c], cv2.COLOR_GRAY2BGR)
        side = int(THUMB_W * INSET)
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)
        cx, cy = int((u - x0) / (2 * c) * side), int((v - y0) / (2 * c) * side)
        cv2.circle(crop, (cx, cy), max(8, side // 9), (26, 122, 255), 2, cv2.LINE_AA)
        cv2.rectangle(crop, (0, 0), (side - 1, side - 1), (255, 255, 255), 2)
        oy = int(round(top * scale)) + 6
        ox = THUMB_W - side - 6
        if oy + side <= thumb.shape[0]:
            thumb[oy:oy + side, ox:ox + side] = crop
            # mark where the inset comes from
            cv2.rectangle(thumb, (int(x0 * scale), int((y0 + top) * scale)),
                          (int((x0 + 2 * c) * scale), int((y0 + 2 * c + top) * scale)), (255, 255, 255), 1)
    ok, jpg = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise ValueError("JPEG encoding failed")
    tmp = session / "thumb.jpg.part"
    tmp.write_bytes(jpg.tobytes())
    tmp.replace(session / "thumb.jpg")
    clip_fps = float(getattr(cfg.record, "clip_fps", 30.0) or 30.0)
    info = {"version": PREVIEW_VERSION, "frame": i, "seq": seq, "clip_t": i / clip_fps, "reason": reason,
            "frames": len(rows), "display_gain": round(gain, 2),
            "target_px": list(tgt) if tgt else None}
    (session / "preview.json").write_text(json.dumps(info, indent=1))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tbot.console.thumbs", description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+")
    args = ap.parse_args(argv)
    cv2.setNumThreads(1)
    status = 0
    for s in args.sessions:
        try:
            info = render_thumb(Path(s))
            print(f"{s}: {info['reason']} at frame {info['frame']} ({info['clip_t']:.2f} s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the rest
            print(f"{s}: no preview: {exc}", file=sys.stderr, flush=True)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
