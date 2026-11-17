"""Debug overlay: one BGR image showing what the pipeline saw and decided (contract: docs/ARCHITECTURE.md).

Pure drawing, no state: the caller supplies the per-frame objects and a `project_w` closure that maps a
WORLD direction to a pixel in the frame that is being drawn.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

from tbot.optics import azel_to_dir
from tbot.types import Detection, FireDecision, Track

# BGR palette
C_DET = (80, 255, 80)
C_TRACK = (0, 215, 255)
C_TRACK_DIM = (90, 150, 170)
C_VEL = (255, 190, 60)
C_LEAD = (255, 70, 255)
C_BORE_PRED = (255, 255, 90)
C_BORE_STATIC = (60, 60, 255)
C_ARMED = (40, 170, 40)
C_SAFE = (70, 70, 70)
C_FIRE = (40, 40, 255)
C_TEXT = (235, 235, 235)
C_STATIC = (150, 150, 150)

LEAD_PREVIEW_S = 0.1  # velocity arrow shows where the track is predicted to be this far ahead


def _as_bgr(image: np.ndarray) -> np.ndarray:
    import cv2

    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return np.ascontiguousarray(img.copy())
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"overlay.draw: expected (H,W) or (H,W,3), got shape {img.shape}")


def _ipt(uv) -> tuple[int, int] | None:
    """Pixel tuple -> integer point, or None when it is missing / not finite / absurdly far off frame."""
    if uv is None:
        return None
    u, v = float(uv[0]), float(uv[1])
    if not (math.isfinite(u) and math.isfinite(v)) or abs(u) > 1e5 or abs(v) > 1e5:
        return None
    return (int(round(u)), int(round(v)))


def _text(img: np.ndarray, org: tuple[int, int], s: str, color=C_TEXT, scale: float = 0.4) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, s, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, color, 1, cv2.LINE_AA)


def _tint_mask(img: np.ndarray, mask: np.ndarray) -> None:
    m = np.asarray(mask)
    if m.ndim != 2 or m.shape != img.shape[:2]:
        return
    sel = m == 0
    if not bool(sel.any()):
        return
    tint = np.array([110.0, 40.0, 10.0], dtype=np.float32)  # blue-ish wash over the barrel / dead pixels
    img[sel] = (img[sel].astype(np.float32) * 0.55 + tint * 0.45).astype(np.uint8)


def draw(
    image: np.ndarray,
    *,
    detections: Iterable[Detection] = (),
    tracks: Iterable[Track] = (),
    decision: FireDecision | None = None,
    mask: np.ndarray | None = None,
    bore_px: Sequence[float] | None = None,
    telemetry: dict | None = None,
    project_w: Callable[[np.ndarray], tuple[float, float] | None] | None = None,
    track_lead: dict[int, np.ndarray] | None = None,
    armed: bool | None = None,
    fired: bool = False,
) -> np.ndarray:
    """Render the debug overlay for one frame and return a new BGR uint8 image of the input size."""
    import cv2

    img = _as_bgr(image)
    h, w = img.shape[:2]
    tele = telemetry or {}
    if armed is None:
        armed = bool(tele.get("armed", False))
    fired = bool(fired or tele.get("fired", False))

    if mask is not None:
        _tint_mask(img, mask)

    # --- static bore point (where the shot goes at the prior range), drawn under everything else
    bp = _ipt(bore_px)
    if bp is not None:
        cv2.line(img, (bp[0] - 9, bp[1]), (bp[0] + 9, bp[1]), C_BORE_STATIC, 1, cv2.LINE_AA)
        cv2.line(img, (bp[0], bp[1] - 9), (bp[0], bp[1] + 9), C_BORE_STATIC, 1, cv2.LINE_AA)
        cv2.circle(img, bp, 3, C_BORE_STATIC, 1, cv2.LINE_AA)

    # --- detections
    for d in detections:
        p = _ipt((d.u, d.v))
        if p is None:
            continue
        r = max(4, int(round(float(d.diameter_px) * 0.5)) + 2)
        bbox = getattr(d, "bbox", None)
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:  # the detector's bounding box, 2 px clear of the blob
            x, y, bw, bh = (int(b) for b in bbox)
            cv2.rectangle(img, (x - 2, y - 2), (x + bw + 1, y + bh + 1), C_DET, 1)
            r = max(r, (max(bw, bh) + 1) // 2 + 2)
        else:
            cv2.circle(img, p, r, C_DET, 1, cv2.LINE_AA)
        _text(img, (p[0] - r, p[1] + r + 11), f"snr {float(d.snr):.0f}  {float(d.diameter_px):.1f} px", C_DET, 0.33)

    # --- tracks (world direction -> pixel) with a velocity arrow
    for t in tracks:
        if project_w is None:
            break
        p = _ipt(project_w(azel_to_dir(t.az, t.el)))
        if p is None:
            continue
        color = C_TRACK if t.confirmed else C_TRACK_DIM
        cv2.circle(img, p, 11, color, 2 if t.confirmed else 1, cv2.LINE_AA)
        _text(img, (p[0] + 13, p[1] - 8), f"#{t.track_id} {t.range_m:.0f}m", color, 0.38)
        lead_dir = None if track_lead is None else track_lead.get(t.track_id)
        q = _ipt(project_w(lead_dir)) if lead_dir is not None else None
        if q is not None and (abs(q[0] - p[0]) + abs(q[1] - p[1])) >= 2:
            cv2.arrowedLine(img, p, q, C_VEL, 2, cv2.LINE_AA, 0, 0.3)

    # --- fire solution: where the clay will be, and where the bore will point
    if decision is not None and project_w is not None:
        lead = _ipt(project_w(decision.lead_dir_w)) if decision.lead_dir_w is not None else None
        bore = _ipt(project_w(decision.bore_dir_w)) if decision.bore_dir_w is not None else None
        if lead is not None:
            cv2.line(img, (lead[0] - 7, lead[1] - 7), (lead[0] + 7, lead[1] + 7), C_LEAD, 2, cv2.LINE_AA)
            cv2.line(img, (lead[0] - 7, lead[1] + 7), (lead[0] + 7, lead[1] - 7), C_LEAD, 2, cv2.LINE_AA)
        if bore is not None:
            cv2.circle(img, bore, 6, C_BORE_PRED, 2, cv2.LINE_AA)
        if lead is not None and bore is not None:
            cv2.line(img, bore, lead, C_BORE_PRED, 1, cv2.LINE_AA)

    # --- arming banner
    label = "ARMED" if armed else "SAFE"
    cv2.rectangle(img, (0, 0), (74, 20), C_ARMED if armed else C_SAFE, -1)
    _text(img, (8, 15), label, (255, 255, 255), 0.5)
    shots = tele.get("shots")
    if shots is not None:
        _text(img, (80, 15), f"shots {int(shots)}", C_TEXT, 0.4)

    # --- fire flash
    if fired or (decision is not None and decision.fire):
        cv2.rectangle(img, (1, 1), (w - 2, h - 2), C_FIRE, 4)
        _text(img, (w // 2 - 28, 36), "FIRE", C_FIRE, 0.8)

    # --- telemetry text block
    lines: list[str] = []
    fps = tele.get("fps")
    age = tele.get("age_ms")
    if fps is not None or age is not None:
        lines.append(f"{float(fps or 0.0):5.1f} fps   age {float(age or 0.0):5.1f} ms   seq {tele.get('seq', '-')}")
    stage = tele.get("stage_ms") or {}
    if stage:
        lines.append("  ".join(f"{k} {float(v):.1f}" for k, v in stage.items()))
    det_n = tele.get("n_detections")
    if det_n is not None:
        lines.append(f"det {int(det_n)}  trk {len(tele.get('tracks') or [])}  drop {int(tele.get('dropped', 0))}")
    if decision is not None:
        bits = [decision.reason]
        if math.isfinite(decision.miss_rad) and math.isfinite(decision.pattern_rad):
            bits.append(
                f"miss {math.degrees(decision.miss_rad):.2f} / pat {math.degrees(decision.pattern_rad):.2f}"
                f" +-{math.degrees(decision.sigma_rad):.2f} deg"
            )
        if math.isfinite(decision.p_hit):
            bits.append(f"P(hit) {decision.p_hit:.2f}")
        lines.append("  ".join(bits))
    if lines:
        y0 = h - 6 - 14 * (len(lines) - 1)
        for i, s in enumerate(lines):
            _text(img, (6, y0 + 14 * i), s, C_TEXT, 0.4)
    return img


# ------------------------------------------------------------------------------------------ from a record
def projector(cam, mount, R_wg) -> Callable[[np.ndarray], tuple[float, float] | None]:
    """World direction -> pixel in the frame whose gun attitude is R_wg (None when it does not project)."""
    R_gw = np.asarray(R_wg, dtype=np.float64).reshape(3, 3).T

    def project(dir_w) -> tuple[float, float] | None:
        if dir_w is None:
            return None
        d_g = R_gw @ np.asarray(dir_w, dtype=np.float64).reshape(3)
        d_c = np.asarray(mount.gun_to_cam(d_g), dtype=np.float64).reshape(1, 3)
        uv = np.asarray(cam.ray_to_pixel(d_c), dtype=np.float64).reshape(-1, 2)
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return (u, v)

    return project


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def draw_record(image: np.ndarray, rec: dict, cam, mount, *, mask: np.ndarray | None = None,
                bore_px: Sequence[float] | None = None, banner: str | None = None) -> np.ndarray:
    """The overlay for one frame from its telemetry record alone (tbot.app.run: the dict written to a
    session's telemetry.jsonl). The live stream and the recorded clips are both drawn through here, off the
    pipeline thread, so what a clip shows is what the pipeline computed on that frame.
    `banner` is an extra first text line (the clip's clock)."""
    import cv2
    from types import SimpleNamespace as NS

    dets = [NS(u=d[0], v=d[1], diameter_px=d[2], snr=d[3], bbox=(d[5], d[6], d[7], d[8])) for d in rec.get("dets") or ()]
    tracks, lead = [], {}
    for t in rec.get("tracks") or ():
        tid = int(t["id"])
        tracks.append(NS(track_id=tid, az=math.radians(t["az_deg"]), el=math.radians(t["el_deg"]),
                         range_m=_nan(t.get("range_m")), confirmed=bool(t.get("confirmed"))))
        s = t.get("s")
        if s is not None and len(s) >= 6:  # cartesian state [p, v, a]: where the clay is LEAD_PREVIEW_S ahead
            s = np.asarray(s, dtype=np.float64)
            p = s[0:3] + s[3:6] * LEAD_PREVIEW_S + (0.5 * LEAD_PREVIEW_S**2 * s[6:9] if s.size >= 9 else 0.0)
            n = float(np.linalg.norm(p))
            if n > 0.0:
                lead[tid] = p / n
        elif t.get("x") is not None and len(t["x"]) >= 6:  # angular state [az, rate, acc, el, rate, acc]
            x = t["x"]
            lead[tid] = azel_to_dir(x[0] + x[1] * LEAD_PREVIEW_S, x[3] + x[4] * LEAD_PREVIEW_S)
    d = rec.get("decision")
    decision = None
    if d:
        decision = NS(fire=bool(d.get("fire")), reason=str(d.get("reason", "")),
                      miss_rad=math.radians(_nan(d.get("miss_deg"))), pattern_rad=math.radians(_nan(d.get("pattern_deg"))),
                      sigma_rad=math.radians(_nan(d.get("sigma_deg"))), p_hit=_nan(d.get("p_hit")),
                      lead_dir_w=d.get("lead_dir_w"), bore_dir_w=d.get("bore_dir_w"))
    R = rec.get("R_wg")
    project = projector(cam, mount, R) if R is not None else None
    img = draw(image, detections=dets, tracks=tracks, decision=decision, mask=mask, bore_px=bore_px, telemetry=rec,
               project_w=project, track_lead=lead, armed=bool(rec.get("armed")), fired=bool(rec.get("fired")))
    # what the detector discarded as scenery fixed in the world (grey crosses), and its exposure
    for c in rec.get("static") or ():
        q = _ipt(c)
        if q is not None:
            cv2.drawMarker(img, q, C_STATIC, cv2.MARKER_TILTED_CROSS, 5, 1)
    w = img.shape[1]
    notes = []
    if banner:
        notes.append(banner)
    if rec.get("exposure_us") is not None:
        ae = rec.get("ae") or {}
        notes.append(f"exp {float(rec['exposure_us']):.0f}us x{float(rec.get('gain') or 1.0):.1f} {ae.get('state', '')}".rstrip())
    for i, line in enumerate(notes):
        _text(img, (max(150, w - 8 * len(line) - 6), 15 + 14 * i), line, C_TEXT, 0.4)
    return img
