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
        cv2.circle(img, p, r, C_DET, 1, cv2.LINE_AA)
        _text(img, (p[0] + r + 2, p[1] - 2), f"{float(d.snr):.0f}", C_DET, 0.33)

    # --- tracks (world direction -> pixel) with a velocity arrow
    for t in tracks:
        if project_w is None:
            break
        p = _ipt(project_w(azel_to_dir(t.az, t.el)))
        if p is None:
            continue
        color = C_TRACK if t.confirmed else C_TRACK_DIM
        cv2.circle(img, p, 11, color, 2 if t.confirmed else 1, cv2.LINE_AA)
        _text(img, (p[0] + 13, p[1] + 4), f"#{t.track_id} {t.range_m:.0f}m", color, 0.38)
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
