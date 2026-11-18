"""Debug overlay: one BGR image showing what the pipeline saw and decided (contract: docs/ARCHITECTURE.md).

Pure drawing, no per-frame state: the caller supplies the per-frame objects and a `project_w` closure that maps
a WORLD direction to a pixel in the frame that is being drawn.

Layout: a semi-transparent status band on top (ARMED/SAFE, LIVE/DRY, shots, fps, frame age; Pi temperature,
power/exposure warnings, exposure and the clip clock on the right), the scene marks in the picture, and a band
at the bottom with the fire decision, a P(hit) meter and the stage timings. Legend of the marks:

    green corner brackets   detection (the detector's box), "snr 13 6.8px" chip under it
    grey x                  candidate dropped as scenery fixed in the world
    amber ring              confirmed track ("#id range" chip); a second ring: the track the solver aims at
    dim dashed amber ring   tentative track
    blue arrow              where the track will be LEAD_PREVIEW_S ahead
    magenta diamond         lead point: where the clay will be when the shot gets there (an arrow at the
                            frame edge when it is outside the picture)
    cyan ring               predicted bore at the shot; dashed cyan circle around it: the pattern radius
                            (solid red when firing); the cyan line from it to the diamond is the miss
    red gapped crosshair    static bore: where the barrel points now, at the prior range
    red frame + FIRE        a shot on this frame

Cost (measured on the Pi 3B, OpenCV 4.10, busy recorded frames): ~7-8.5 ms per frame against 31-39 ms for
the previous overlay; ~1.5 ms on a laptop. Rings and outlines are 1 px anti-aliased strokes (a thick AA
stroke is 5-10x dearer), many marks of one kind go into one polylines call, the bands are darkened with an
in-place shift of whole rows (no float blending), label chips sit on opaque plates, world directions are
projected in one batch, and text widths are memoised. What is left is mostly putText (~2 ms for ~16 labels)
and the gray->BGR conversion (~0.4 ms). The stream draws at stream.fps (10) on its own thread, only while
somebody watches. numpy + cv2 only, cv2.FONT_HERSHEY_* fonts (OpenCV 5 draws them smoother than the Pi's 4.x).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Sequence

import numpy as np

from tbot.types import Detection, FireDecision, Track

# ----------------------------------------------------------------------------------------------- palette
# BGR. Saturated mark colours, each over a dark outline so it reads on bright sky and on dark trees.
C_DET = (80, 235, 80)            # detections: green
C_TRACK = (0, 200, 255)          # confirmed track: amber
C_TRACK_DIM = (70, 150, 190)     # tentative track: dim amber
C_VEL = (255, 175, 40)           # predicted-motion arrow: blue
C_LEAD = (255, 80, 255)          # lead point (the clay at the shot): magenta
C_BORE_PRED = (255, 235, 60)     # predicted bore at the shot + pattern circle: cyan
C_BORE_STATIC = (70, 70, 255)    # static bore at the prior range: red
C_ARMED = (50, 165, 50)          # pill: armed
C_SAFE = (95, 95, 95)            # pill: safe
C_LIVE = (40, 40, 210)           # pill: live trigger
C_DRY = (170, 110, 40)           # pill: dry run
C_FIRE = (40, 40, 255)           # fire flash
C_WARN = (0, 165, 255)           # amber warnings (exposure, warm Pi, no IMU, old frame)
C_BAD = (50, 50, 235)            # red warnings (under-voltage, throttled, hot)
C_GOOD = (90, 220, 90)           # P(hit) over the bar
C_TEXT = (235, 235, 235)
C_TEXT_DIM = (175, 175, 175)
C_STATIC = (175, 175, 175)       # scenery-rejected candidates: neutral grey
C_SHADOW = (12, 12, 12)          # outline under marks and text
C_PLATE = (32, 32, 32)           # label chip backing
C_MASK_TINT = (110, 40, 10)      # wash over the barrel / dead pixels

LEAD_PREVIEW_S = 0.1  # velocity arrow shows where the track is predicted to be this far ahead

FONT = 0              # cv2.FONT_HERSHEY_SIMPLEX (numeric: importing this module does not import cv2)
TOP_H = 20            # status band height, px
LINE_H = 15           # bottom band line height, px
HOT_C, WARM_C = 75.0, 70.0  # Pi temperature colours (tbot.app.run.HOT_TEMP_C warns at 75 C)
OLD_FRAME_MS = 100.0  # frame age shown amber above this
DET_LABELS = 4        # detections that get an snr/size chip (the strongest; the rest only brackets)

_BAR_RE = re.compile(r"p_hit\s+[-+0-9.eE]+\s*<\s*([0-9.]+)")  # the solver's "p_hit 0.12 < 0.50"


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
        return cv2.cvtColor(np.ascontiguousarray(img[:, :, 0]), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"overlay.draw: expected (H,W) or (H,W,3), got shape {img.shape}")


def _ipt(uv) -> tuple[int, int] | None:
    """Pixel tuple -> integer point, or None when it is missing / not finite / absurdly far off frame."""
    if uv is None:
        return None
    try:
        u, v = float(uv[0]), float(uv[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(u) and math.isfinite(v)) or abs(u) > 1e5 or abs(v) > 1e5:
        return None
    return (int(round(u)), int(round(v)))


def _fnum(v) -> float | None:
    """A finite float or None (old records hold None / NaN / nothing for many fields)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ------------------------------------------------------------------------------------------- text
_TSIZE: dict[tuple[str, float], tuple[int, int]] = {}  # labels repeat from frame to frame


def _tsize(s: str, scale: float) -> tuple[int, int]:
    """cv2.getTextSize(s, FONT, scale, 1)[0], memoised (OpenCV 4 and 5 measure the same font differently,
    so the widths are always asked of the cv2 that draws them)."""
    key = (s, scale)
    v = _TSIZE.get(key)
    if v is None:
        import cv2

        if len(_TSIZE) > 1024:
            _TSIZE.clear()
        (tw, th), _ = cv2.getTextSize(s, FONT, scale, 1)
        v = _TSIZE[key] = (int(tw), int(th))
    return v


def _text(img: np.ndarray, org: tuple[int, int], s: str, color=C_TEXT, scale: float = 0.4) -> None:
    """Anti-aliased text with a dark outline (free-standing labels; tbot.tools.clip uses it for its clock)."""
    import cv2

    cv2.putText(img, s, org, FONT, scale, C_SHADOW, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def _darken(img: np.ndarray, y0: int, y1: int) -> None:
    """Darken full-width rows to a quarter in place: the semi-transparent bands. An in-place shift of
    contiguous rows (~40 us on a Pi 3B for a 640x35 band, where cv2.convertScaleAbs takes ~350 us)."""
    y0, y1 = max(0, int(y0)), min(img.shape[0], int(y1))
    if y1 > y0:
        band = img[y0:y1]
        np.right_shift(band, 2, out=band)


def _chip(img: np.ndarray, org: tuple[int, int], s: str, color, scale: float, lim: tuple[int, int],
          anchor: tuple[int, int]) -> None:
    """Text on a small dark plate. org is the baseline start (as cv2.putText); the chip is kept inside the
    frame and between the bands (lim = (top, bottom) of the free picture), and left out when the mark it
    labels (anchor) is not in that free picture. (An opaque plate: blending each chip costs 4x more on a
    Pi 3B than the whole rest of the label.)"""
    import cv2

    h, w = img.shape[:2]
    if not (0 <= anchor[0] < w and lim[0] <= anchor[1] < lim[1]):
        return
    tw, th = _tsize(s, scale)
    x = min(max(3, int(org[0])), max(3, w - tw - 4))
    y = min(max(lim[0] + th + 4, int(org[1])), max(lim[0] + th + 4, lim[1] - 5))
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 1, y + 3), C_PLATE, -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _pill(img: np.ndarray, x: int, y0: int, y1: int, s: str, bg, tw: int, scale: float) -> None:
    """Filled state label in the status band."""
    import cv2

    cv2.rectangle(img, (x, y0), (x + tw + 9, y1 - 1), bg, -1)
    th = _tsize(s, scale)[1]
    cv2.putText(img, s, (x + 5, y0 + (y1 - y0 + th) // 2 - 1), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------------------------------- marks
# Pi 3B costs (OpenCV 4.10) behind these choices: a 1 px anti-aliased circle r14 ~30 us, the same at 2 px
# ~195 us, a 4 px LINE_8 one ~80 us; 1 px AA polylines/lines ~15-50 us; every cv2 call ~2 us of overhead and
# every small numpy op ~5 us. So rings and outlines are built from 1 px strokes, and many small marks of one
# kind go into one polylines call.
def _ring(img, c, r: int, color, thick: int = 1) -> None:
    """Ring of 1-2 px with a dark outline inside and out, all 1 px anti-aliased circles."""
    import cv2

    aa = cv2.LINE_AA
    cv2.circle(img, c, r + 1, C_SHADOW, 1, aa)
    cv2.circle(img, c, max(1, r - thick), C_SHADOW, 1, aa)
    cv2.circle(img, c, r, color, 1, aa)
    if thick > 1:
        cv2.circle(img, c, max(1, r - 1), color, 1, aa)


_DASH: dict[tuple[int, int], np.ndarray] = {}  # (radius, dashes) -> int32 dash vertices around (0, 0)


def _dash_offsets(r: int, dashes: int) -> np.ndarray:
    key = (int(r), int(dashes))
    off = _DASH.get(key)
    if off is None:
        if len(_DASH) > 512:
            _DASH.clear()
        step = 2.0 * math.pi / dashes
        a = np.arange(dashes)[:, None] * step + np.linspace(0.0, 0.55 * step, max(3, min(9, r // 4 + 2)))[None, :]
        off = np.stack([np.rint(r * np.cos(a)), np.rint(r * np.sin(a))], axis=-1).astype(np.int32)
        off = _DASH[key] = off.reshape(dashes, -1, 1, 2)
    return off


def _dashed_ring(img, c, r: int, color, dashes: int = 12) -> None:
    """Dashed 1 px anti-aliased ring with a dark outline; the dash geometry is cached per radius."""
    import cv2

    cc = np.array(c, dtype=np.int32)
    for rr, col in ((r + 1, C_SHADOW), (max(1, r - 1), C_SHADOW), (r, color)):
        cv2.polylines(img, list(_dash_offsets(rr, dashes) + cc), False, col, 1, cv2.LINE_AA)


def _line(img, p, q, color) -> None:
    import cv2

    cv2.line(img, p, q, C_SHADOW, 3, cv2.LINE_8)
    cv2.line(img, p, q, color, 1, cv2.LINE_AA)


def _brackets(img, boxes: list[tuple[int, int, int, int]], color) -> None:
    """Corner brackets of every box (x0, y0, x1, y1): crisp axis-aligned strokes in the exact colour over a
    dark outline, two polylines calls for all of them."""
    import cv2

    if not boxes:
        return
    pl = []
    for x0, y0, x1, y1 in boxes:
        n = max(3, min(6, (min(x1 - x0, y1 - y0) + 1) // 3))
        pl += [((x0, y0 + n), (x0, y0), (x0 + n, y0)), ((x1 - n, y0), (x1, y0), (x1, y0 + n)),
               ((x0, y1 - n), (x0, y1), (x0 + n, y1)), ((x1 - n, y1), (x1, y1), (x1, y1 - n))]
    arr = list(np.array(pl, dtype=np.int32).reshape(-1, 3, 1, 2))
    cv2.polylines(img, arr, False, C_SHADOW, 3, cv2.LINE_8)
    cv2.polylines(img, arr, False, color, 1, cv2.LINE_8)


def _crosses(img, pts: list[tuple[int, int]], color, r: int = 3) -> None:
    """Small x at each point (scenery-rejected candidates): one outline and one colour call for all."""
    import cv2

    if not pts:
        return
    segs = []
    for x, y in pts:
        segs += [((x - r, y - r), (x + r, y + r)), ((x - r, y + r), (x + r, y - r))]
    arr = list(np.array(segs, dtype=np.int32).reshape(-1, 2, 1, 2))
    cv2.polylines(img, arr, False, C_SHADOW, 3, cv2.LINE_8)
    cv2.polylines(img, arr, False, color, 1, cv2.LINE_AA)


def _crosshair(img, c, color, r_in: int = 4, r_out: int = 13) -> None:
    """Gapped crosshair with a centre dot: the static bore reticle."""
    import cv2

    x, y = c
    arms = list(np.array(((x - r_out, y), (x - r_in, y), (x + r_in, y), (x + r_out, y),
                          (x, y - r_out), (x, y - r_in), (x, y + r_in), (x, y + r_out)),
                         dtype=np.int32).reshape(4, 2, 1, 2))
    cv2.polylines(img, arms, False, C_SHADOW, 3, cv2.LINE_8)
    cv2.rectangle(img, (x - 1, y - 1), (x + 1, y + 1), C_SHADOW, -1)
    cv2.polylines(img, arms, False, color, 1, cv2.LINE_8)
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        img[y, x] = color


def _diamond(img, c, color, r: int = 6) -> None:
    """Hollow diamond with a 2 px rim and a dark outline: three filled polygons (cheaper than thick strokes)."""
    import cv2

    x, y = c
    for rr, col in ((r + 2, C_SHADOW), (r, color), (r - 3, C_SHADOW)):
        pts = np.array(((x, y - rr), (x + rr, y), (x, y + rr), (x - rr, y)), dtype=np.int32)
        cv2.fillConvexPoly(img, pts, col, cv2.LINE_AA)


def _edge_arrow(img, src, dst, color, lim: tuple[int, int]) -> None:
    """dst is outside the picture: an arrowhead at the picture's edge on the way from src to it."""
    import cv2

    h, w = img.shape[:2]
    top, bot = lim
    ok, _p1, p2 = cv2.clipLine((2, top + 2, w - 4, max(1, bot - top - 4)), src, dst)
    if not ok:
        return
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    n = math.hypot(dx, dy)
    if n < 1.0:
        return
    tail = (int(round(p2[0] - 16 * dx / n)), int(round(p2[1] - 16 * dy / n)))
    cv2.arrowedLine(img, tail, p2, C_SHADOW, 4, cv2.LINE_8, 0, 0.5)
    cv2.arrowedLine(img, tail, p2, color, 2, cv2.LINE_AA, 0, 0.5)


_MASK_CACHE: list[tuple] = []  # (mask array, its shape, flat indices of its masked pixels), newest first


def _tint_mask(img: np.ndarray, mask: np.ndarray) -> None:
    """Blue wash over the masked pixels (barrel, dead pixels): integer maths on those pixels only; the pixel
    list is cached per mask array (the pipeline's mask never changes)."""
    m = np.asarray(mask)
    if m.ndim != 2 or m.shape != img.shape[:2]:
        return
    idx = next((e[2] for e in _MASK_CACHE if e[0] is m and e[1] == m.shape), None)
    if idx is None:
        idx = np.flatnonzero(m.reshape(-1) == 0)
        _MASK_CACHE.insert(0, (m, m.shape, idx))
        del _MASK_CACHE[4:]
    if idx.size == 0:
        return
    flat = img.reshape(-1, 3)
    px = flat[idx].astype(np.uint16)
    flat[idx] = ((px * 9 + np.array(C_MASK_TINT, dtype=np.uint16) * 7) >> 4).astype(np.uint8)  # 56/44 blend


def _azel(az: float, el: float) -> tuple[float, float, float]:
    """tbot.optics.azel_to_dir for one direction, in plain floats (a few numpy calls per track add up to
    milliseconds on a Pi 3B)."""
    ce = math.cos(el)
    return (ce * math.cos(az), ce * math.sin(az), math.sin(el))


def _cone_dir(center_dir, ang_rad: float | None) -> tuple[float, float, float] | None:
    """A direction ang_rad away from center_dir (its pixel distance is the cone's radius in the picture)."""
    if center_dir is None or ang_rad is None or not ang_rad > 0.0:
        return None
    try:
        x, y, z = (float(v) for v in np.asarray(center_dir, dtype=np.float64).reshape(3))
    except (TypeError, ValueError):
        return None
    n = math.sqrt(x * x + y * y + z * z)
    if not (math.isfinite(n) and n > 0.0):
        return None
    x, y, z = x / n, y / n, z / n
    if abs(z) < 0.9:   # a = d x (0, 0, 1)
        ax, ay, az = y, -x, 0.0
    else:              # a = d x (0, 1, 0)
        ax, ay, az = -z, 0.0, x
    m = math.sqrt(ax * ax + ay * ay + az * az)
    c, s = math.cos(ang_rad), math.sin(ang_rad) / m
    return (c * x + s * ax, c * y + s * ay, c * z + s * az)


def _project_all(project_w, dirs: list) -> list:
    """Integer pixels (or None) of a list of world directions (None entries stay None); one batched call
    when the projector has a `many` method (tbot.app.overlay.projector), else one call each."""
    out: list = [None] * len(dirs)
    idx = [i for i, d in enumerate(dirs) if d is not None]
    if not idx or project_w is None:
        return out
    many = getattr(project_w, "many", None)
    if many is not None:
        try:
            arr = np.array([dirs[i] for i in idx], dtype=np.float64)
            if arr.shape != (len(idx), 3):
                arr = np.array([np.asarray(dirs[i], dtype=np.float64).reshape(3) for i in idx])
            uv = np.asarray(many(arr), dtype=np.float64).reshape(-1, 2).tolist()
            for k, i in enumerate(idx):
                out[i] = _ipt(uv[k])
            return out
        except (TypeError, ValueError):
            pass
    for i in idx:
        try:
            out[i] = _ipt(project_w(dirs[i]))
        except (TypeError, ValueError):
            out[i] = None
    return out


# ------------------------------------------------------------------------------------------- HUD
def _status_band(img: np.ndarray, tele: dict, armed: bool, banner: str | None) -> None:
    """Top band: state pills and numbers on the left; warnings, temperature, exposure, clip clock right."""
    import cv2

    h, w = img.shape[:2]
    bh = min(h, TOP_H)
    _darken(img, 0, bh)
    scale = 0.4
    y0, y1 = 3, bh - 3
    base = (bh + _tsize("0", scale)[1]) // 2

    # (kind, text, colour, priority): a narrow frame drops the lowest priority first, never priority 9
    left: list[tuple] = [("pill", "ARMED" if armed else "SAFE", C_ARMED if armed else C_SAFE, 9)]
    live = tele.get("live")
    if live is not None:
        left.append(("pill", "LIVE" if live else "DRY", C_LIVE if live else C_DRY, 8))
    if tele.get("imu_ready") is False:
        left.append(("pill", "NO IMU", C_WARN, 7))
    shots = _fnum(tele.get("shots"))
    if shots is not None:
        left.append(("text", f"shots {int(shots)}", C_TEXT, 6))
    fps = _fnum(tele.get("fps"))
    if fps is not None:
        left.append(("text", f"{fps:.0f} fps", C_TEXT, 5))
    age = _fnum(tele.get("age_ms"))
    if age is not None:
        left.append(("text", f"age {age:.0f} ms", C_WARN if age > OLD_FRAME_MS else C_TEXT_DIM, 3))

    right: list[tuple] = []
    health = tele.get("health") if isinstance(tele.get("health"), dict) else {}
    flags = health.get("throttled") if isinstance(health.get("throttled"), dict) else {}
    if flags.get("under_voltage_now"):
        right.append(("pill", "UNDER-VOLT", C_BAD, 8))
    if flags.get("throttled_now") or flags.get("soft_temp_limit_now"):
        right.append(("pill", "THROTTLED", C_BAD, 8))
    ae = tele.get("ae") if isinstance(tele.get("ae"), dict) else {}
    ae_state = str(ae.get("state") or "")
    if ae_state in ("too_bright", "too_dark"):
        right.append(("pill", "TOO BRIGHT" if ae_state == "too_bright" else "TOO DARK", C_WARN, 7))
    temp = _fnum(health.get("cpu_temp_c"))
    if temp is not None:
        tcol = C_BAD if temp >= HOT_C else C_WARN if temp >= WARM_C else C_TEXT_DIM
        right.append(("text", f"{temp:.0f}C", tcol, 4))
    exp_us = _fnum(tele.get("exposure_us"))
    if exp_us is not None:
        gain = _fnum(tele.get("gain")) or 1.0
        right.append(("text", f"exp {exp_us:.0f}us x{gain:.1f}", C_TEXT_DIM, 2))
    if banner:
        right.append(("text", str(banner), C_TEXT, 9))

    gap, margin = 8, 8  # (margin: clear of the FIRE frame)
    widths = {id(i): _tsize(i[1], scale)[0] for i in left + right}

    def span(i) -> int:
        return widths[id(i)] + (10 if i[0] == "pill" else 0)

    while sum(span(i) + gap for i in left + right) - gap + 2 * margin + 12 > w:
        cands = [i for i in left + right if i[3] < 9]
        if not cands:
            break
        worst = min(cands, key=lambda i: i[3])
        (left if any(worst is i for i in left) else right).remove(worst)

    def put(item, x) -> None:
        kind, s, col, _ = item
        if kind == "pill":
            _pill(img, x, y0, y1, s, col, widths[id(item)], scale)
        else:
            cv2.putText(img, s, (x, base), FONT, scale, col, 1, cv2.LINE_AA)

    x = margin
    for item in left:
        put(item, x)
        x += span(item) + gap
    x = w - margin
    for item in reversed(right):
        x -= span(item)
        put(item, x)
        x -= gap


_STAGE_SHORT = {"attitude": "att", "overlay": "ovl"}
_METER_W = 64


def _decision_lines(tele: dict, decision) -> tuple[bool, list[str], list[str]]:
    """What the bottom band will show: (decision line?, counts, stage timings)."""
    stats: list[str] = []
    if tele.get("seq") is not None:
        stats.append(f"seq {tele.get('seq')}")
    det_n = _fnum(tele.get("n_detections"))
    if det_n is not None:
        stats.append(f"det {int(det_n)}")
        stats.append(f"trk {len(tele.get('tracks') or [])}")
        drop = _fnum(tele.get("dropped"))
        if drop is not None:
            stats.append(f"drop {int(drop)}")
    stage = tele.get("stage_ms") if isinstance(tele.get("stage_ms"), dict) else {}
    timing = []
    for k, v in stage.items():
        fv = _fnum(v)
        if fv is None or (k == "overlay" and fv == 0.0):
            continue
        timing.append(f"{_STAGE_SHORT.get(k, k)} {fv:.1f}")
    return decision is not None, stats, timing


def _band_height(lines: tuple[bool, list, list]) -> int:
    n = int(lines[0]) + int(bool(lines[1] or lines[2]))
    return 0 if n == 0 else 5 + LINE_H * n


def _pmeter(img: np.ndarray, x: int, y_base: int, p: float | None, bar: float | None) -> None:
    """'P(hit)' label, a bar with the firing bar as a white tick, and the value."""
    import cv2

    lw = _tsize("P(hit)", 0.36)[0]
    cv2.putText(img, "P(hit)", (x, y_base), FONT, 0.36, C_TEXT_DIM, 1, cv2.LINE_AA)
    bx, bh = x + lw + 5, 7
    by = y_base - bh + 1
    cv2.rectangle(img, (bx, by), (bx + _METER_W, by + bh), (60, 60, 60), -1)
    thr = bar if bar is not None else 0.5
    if p is not None:
        pc = min(1.0, max(0.0, p))
        col = C_GOOD if pc >= thr else C_WARN if pc >= 0.5 * thr else C_BAD
        fw = int(round(pc * _METER_W))
        if fw > 0:
            cv2.rectangle(img, (bx, by), (bx + fw, by + bh), col, -1)
    cv2.rectangle(img, (bx, by), (bx + _METER_W, by + bh), (150, 150, 150), 1)
    if bar is not None and 0.0 < bar < 1.0:
        tx = bx + int(round(bar * _METER_W))
        cv2.line(img, (tx, by - 2), (tx, by + bh + 2), (255, 255, 255), 1)
    cv2.putText(img, "--" if p is None else f"{p:.2f}", (bx + _METER_W + 5, y_base), FONT, 0.4, C_TEXT, 1,
                cv2.LINE_AA)


def _decision_band(img: np.ndarray, decision, lines: tuple[bool, list, list]) -> None:
    """Bottom band: the fire decision with its miss/pattern numbers and the P(hit) meter; then counts and
    stage timings (dim)."""
    import cv2

    h, w = img.shape[:2]
    bh = _band_height(lines)
    if bh == 0:
        return
    top = max(0, h - bh)
    _darken(img, top, h)
    y = top + LINE_H - 1
    has_decision, stats, timing = lines
    if has_decision:
        fire = bool(getattr(decision, "fire", False))
        reason = str(getattr(decision, "reason", "") or "-")
        meter_w = _tsize("P(hit)", 0.36)[0] + 5 + _METER_W + 5 + _tsize("0.00", 0.4)[0]
        mx = w - 10 - meter_w
        text_end = mx - 10 if mx > 60 else w - 10
        s = "FIRE" if fire else reason
        cv2.putText(img, s, (10, y), FONT, 0.42, C_FIRE if fire else C_TEXT, 1 + int(fire), cv2.LINE_AA)
        x = 10 + _tsize(s, 0.42)[0] + 12
        miss = _fnum(getattr(decision, "miss_rad", None))
        pat = _fnum(getattr(decision, "pattern_rad", None))
        sig = _fnum(getattr(decision, "sigma_rad", None))
        if miss is not None and pat is not None:
            s = f"miss {math.degrees(miss):.2f} / pattern {math.degrees(pat):.2f} deg"
            if sig is not None:
                s += f"  sd {math.degrees(sig):.2f}"
            if x + _tsize(s, 0.36)[0] > text_end:
                s = f"miss {math.degrees(miss):.2f}/{math.degrees(pat):.2f} deg"
            if x + _tsize(s, 0.36)[0] <= text_end:
                cv2.putText(img, s, (x, y), FONT, 0.36, C_TEXT_DIM, 1, cv2.LINE_AA)
        if mx > 60:
            cv2.rectangle(img, (mx - 6, y - 11), (w - 1, y + 3), C_PLATE, -1)  # a long reason never runs under it
            m = _BAR_RE.search(reason)
            _pmeter(img, mx, y, _fnum(getattr(decision, "p_hit", None)), _fnum(m.group(1)) if m else None)
        y += LINE_H
    if stats or timing:
        def compose() -> str:
            s = "  ".join(stats)
            if timing:
                s = (s + "   |   " if s else "") + "ms " + "  ".join(timing)
            return s

        s = compose()
        while timing and _tsize(s, 0.34)[0] > w - 20:  # too long: drop stages, 'total' (the last) goes last
            timing.pop(-2 if len(timing) > 1 else -1)
            s = compose()
        cv2.putText(img, s, (10, y), FONT, 0.34, C_TEXT_DIM, 1, cv2.LINE_AA)


def _fire_flash(img: np.ndarray) -> None:
    """Red frame round the picture and a big FIRE plate under the status band."""
    import cv2

    h, w = img.shape[:2]
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), C_FIRE, 5)
    s, scale = "FIRE", 1.0
    tw, th = _tsize(s, scale)
    tw += 2  # thickness 2
    cx, y0 = w // 2, min(h - 1, TOP_H + 6)
    x0, x1, y1 = cx - tw // 2 - 14, cx + tw // 2 + 14, y0 + th + 16
    cv2.rectangle(img, (x0, y0), (x1, y1), C_FIRE, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.putText(img, s, (cx - tw // 2 + 1, y0 + th + 8), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


# ------------------------------------------------------------------------------------------- the frame
def _render(img: np.ndarray, *, detections, tracks, decision, mask, bore_px, tele: dict, project_w, track_lead,
            armed: bool, fired: bool, static=None, banner: str | None = None) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    lines = _decision_lines(tele, decision)
    lim = (min(h, TOP_H), max(min(h, TOP_H), h - _band_height(lines)))  # free picture between the bands

    if mask is not None:
        _tint_mask(img, mask)

    # --- candidates the detector dropped as scenery fixed in the world: quiet grey x (not under the bands)
    _crosses(img, [q for q in map(_ipt, static or ()) if q is not None and lim[0] + 3 <= q[1] < lim[1] - 3],
             C_STATIC)

    # --- static bore (where the shot goes at the prior range), under everything else
    bp = _ipt(bore_px)
    if bp is not None:
        _crosshair(img, bp, C_BORE_STATIC)

    # --- everything that needs the projector, in one batch
    tracks = list(tracks or ())
    lead_w = getattr(decision, "lead_dir_w", None) if decision is not None else None
    bore_w = getattr(decision, "bore_dir_w", None) if decision is not None else None
    pat_rad = _fnum(getattr(decision, "pattern_rad", None)) if decision is not None else None
    pts: list = [None] * (2 * len(tracks) + 3)
    if project_w is not None:
        dirs: list = []
        for t in tracks:
            dirs.append(_azel(float(t.az), float(t.el)))
            dirs.append(None if track_lead is None else track_lead.get(t.track_id))
        dirs += [lead_w, bore_w, _cone_dir(bore_w, pat_rad)]
        pts = _project_all(project_w, dirs)

    # --- detections: corner brackets on the detector's box; snr/size chip under the strongest few (clear of a
    # track ring round the same clay)
    boxes, labels = [], []
    for d in detections or ():
        p = _ipt((d.u, d.v))
        if p is None:
            continue
        snr, dia = _fnum(d.snr), _fnum(d.diameter_px)
        r = max(4, int(round(dia * 0.5)) + 2) if dia is not None else 4
        bbox = getattr(d, "bbox", None)
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:  # 2 px clear of the blob
            x, y, bw, bh = (int(b) for b in bbox)
            box = (x - 2, y - 2, x + bw + 1, y + bh + 1)
        else:
            box = (p[0] - r, p[1] - r, p[0] + r, p[1] + r)
        boxes.append(box)
        label = " ".join(s for s in ((f"snr {snr:.0f}" if snr is not None else ""),
                                     (f"{dia:.1f}px" if dia is not None else "")) if s)
        if label:
            labels.append((-(snr or 0.0), (box[0], max(box[3] + 12, p[1] + 28)), label, p))
    _brackets(img, boxes, C_DET)
    for _, org, label, p in sorted(labels, key=lambda x: x[0])[:DET_LABELS]:
        _chip(img, org, label, C_DET, 0.32, lim, p)

    # --- tracks with the predicted-motion arrow
    target = getattr(decision, "track_id", None) if decision is not None else None
    for k, t in enumerate(tracks):
        p, q = pts[2 * k], pts[2 * k + 1]
        if p is None:
            continue
        if q is not None and (abs(q[0] - p[0]) + abs(q[1] - p[1])) >= 2:
            thick = 2 if t.confirmed else 1
            cv2.arrowedLine(img, p, q, C_SHADOW, thick + 2, cv2.LINE_8, 0, 0.25)
            cv2.arrowedLine(img, p, q, C_VEL, thick, cv2.LINE_AA, 0, 0.25)
        if t.confirmed:
            _ring(img, p, 14, C_TRACK, 2)
            try:
                if target is not None and int(t.track_id) == int(target):
                    _ring(img, p, 19, C_TRACK, 1)
            except (TypeError, ValueError):
                pass
            color = C_TRACK
        else:
            _dashed_ring(img, p, 13, C_TRACK_DIM, 10)
            color = C_TRACK_DIM
        rng = _fnum(t.range_m)
        _chip(img, (p[0] + 18, p[1] - 10), f"#{t.track_id}" + (f" {rng:.0f}m" if rng is not None else ""),
              color, 0.36, lim, p)

    # --- fire solution: predicted bore (ring + pattern circle), lead point (diamond), the miss between them
    lead, bore, pat = pts[-3], pts[-2], pts[-1]
    if decision is not None and project_w is not None:
        fire = bool(getattr(decision, "fire", False)) or fired
        lead_in = lead is not None and 0 <= lead[0] < w and lim[0] <= lead[1] < lim[1]
        if lead is not None and bore is not None:
            _line(img, bore, lead, C_BORE_PRED)
        if bore is not None:
            rp = None if pat is None else int(round(math.hypot(pat[0] - bore[0], pat[1] - bore[1])))
            if rp is not None and 3 <= rp <= 4 * max(h, w):
                if fire:
                    _ring(img, bore, rp, C_FIRE, 2)
                else:
                    _dashed_ring(img, bore, rp, C_BORE_PRED, 16)
            _ring(img, bore, 4, C_BORE_PRED, 2)
        if lead_in:
            _diamond(img, lead, C_LEAD)
        elif lead is not None:
            src = bore if bore is not None else (w // 2, (lim[0] + lim[1]) // 2)
            _edge_arrow(img, src, lead, C_LEAD, lim)

    # --- HUD
    _status_band(img, tele, armed, banner)
    _decision_band(img, decision, lines)
    if fired or (decision is not None and bool(getattr(decision, "fire", False))):
        _fire_flash(img)
    return img


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
    img = _as_bgr(image)
    tele = telemetry or {}
    if armed is None:
        armed = bool(tele.get("armed", False))
    fired = bool(fired or tele.get("fired", False))
    return _render(img, detections=detections, tracks=tracks, decision=decision, mask=mask, bore_px=bore_px,
                   tele=tele, project_w=project_w, track_lead=track_lead, armed=bool(armed), fired=fired)


# ------------------------------------------------------------------------------------------ from a record
def projector(cam, mount, R_wg) -> Callable[[np.ndarray], tuple[float, float] | None]:
    """World direction -> pixel in the frame whose gun attitude is R_wg (None when it does not project).
    The returned function also has `.many(dirs (N,3)) -> (N,2)` (NaN rows for what does not project)."""
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

    def many(dirs_w) -> np.ndarray:
        d_g = np.asarray(dirs_w, dtype=np.float64).reshape(-1, 3) @ R_gw.T
        d_c = np.asarray(mount.gun_to_cam(d_g), dtype=np.float64).reshape(-1, 3)
        return np.asarray(cam.ray_to_pixel(d_c), dtype=np.float64).reshape(-1, 2)

    project.many = many  # type: ignore[attr-defined]
    return project


def _nan(v) -> float:
    f = _fnum(v)
    return float("nan") if f is None else f


def _record_dets(rec: dict) -> list:
    from types import SimpleNamespace as NS

    out = []
    for d in rec.get("dets") or ():
        try:
            bbox = tuple(d[5:9]) if len(d) >= 9 and all(_fnum(b) is not None for b in d[5:9]) else None
            out.append(NS(u=_nan(d[0]), v=_nan(d[1]), diameter_px=_nan(d[2]),
                          snr=_nan(d[3]) if len(d) > 3 else float("nan"), bbox=bbox))
        except (TypeError, IndexError, ValueError):
            continue
    return out


def _record_tracks(rec: dict) -> tuple[list, dict]:
    from types import SimpleNamespace as NS

    tracks, lead = [], {}
    for t in rec.get("tracks") or ():
        try:
            tid = int(t["id"])
            az, el = _fnum(t.get("az_deg")), _fnum(t.get("el_deg"))
            if az is None or el is None:
                continue
            tracks.append(NS(track_id=tid, az=math.radians(az), el=math.radians(el),
                             range_m=_nan(t.get("range_m")), confirmed=bool(t.get("confirmed"))))
            s = t.get("s")
            if s is not None and len(s) >= 6:  # cartesian state [p, v, a]: where the clay is LEAD_PREVIEW_S ahead
                s = [float(v) for v in s]
                T, h2 = LEAD_PREVIEW_S, 0.5 * LEAD_PREVIEW_S**2
                p = [s[i] + s[3 + i] * T + (h2 * s[6 + i] if len(s) >= 9 else 0.0) for i in range(3)]
                n = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                if math.isfinite(n) and n > 0.0:
                    lead[tid] = (p[0] / n, p[1] / n, p[2] / n)
            elif t.get("x") is not None and len(t["x"]) >= 6:  # angular state [az, rate, acc, el, rate, acc]
                x = [float(v) for v in t["x"]]
                lead[tid] = _azel(x[0] + x[1] * LEAD_PREVIEW_S, x[3] + x[4] * LEAD_PREVIEW_S)
        except (KeyError, TypeError, ValueError):
            continue
    return tracks, lead


def draw_record(image: np.ndarray, rec: dict, cam, mount, *, mask: np.ndarray | None = None,
                bore_px: Sequence[float] | None = None, banner: str | None = None) -> np.ndarray:
    """The overlay for one frame from its telemetry record alone (tbot.app.run: the dict written to a
    session's telemetry.jsonl). The live stream and the recorded clips are both drawn through here, off the
    pipeline thread, so what a clip shows is what the pipeline computed on that frame.
    `banner` is extra text at the right of the status band (the clip's clock). Records of older software
    versions lack some fields; whatever is missing is not drawn."""
    from types import SimpleNamespace as NS

    rec = rec or {}
    tracks, lead = _record_tracks(rec)
    d = rec.get("decision")
    decision = None
    if isinstance(d, dict) and d:
        decision = NS(fire=bool(d.get("fire")), reason=str(d.get("reason") or ""), track_id=d.get("track_id"),
                      miss_rad=math.radians(_nan(d.get("miss_deg"))),
                      pattern_rad=math.radians(_nan(d.get("pattern_deg"))),
                      sigma_rad=math.radians(_nan(d.get("sigma_deg"))), p_hit=_nan(d.get("p_hit")),
                      lead_dir_w=d.get("lead_dir_w"), bore_dir_w=d.get("bore_dir_w"))
    project = None
    if rec.get("R_wg") is not None:
        try:
            project = projector(cam, mount, rec["R_wg"])
        except (TypeError, ValueError):
            project = None
    return _render(_as_bgr(image), detections=_record_dets(rec), tracks=tracks, decision=decision, mask=mask,
                   bore_px=bore_px, tele=rec, project_w=project, track_lead=lead, armed=bool(rec.get("armed")),
                   fired=bool(rec.get("fired")), static=rec.get("static"), banner=banner)
