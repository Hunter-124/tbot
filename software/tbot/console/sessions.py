"""Recorded presses (sessions/<name>/) for the console: list, details, shot times inside the clip, notes, previews.

Light on purpose (standard library only): summary.json and the small preview.json are read per session;
telemetry.jsonl and frames.csv are only scanned when a session is opened, to place each shot in the clip.

Clip frame numbers are rows of frames.csv (`tbot clip` writes every stored frame, processed or not), so a shot is
placed by the `seq` of its fired telemetry record, never by its line number in telemetry.jsonl (frames the pipeline
skipped have no telemetry line).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
#: Files the console serves from a session (frames.bin is hundreds of MB: download it with scp if ever needed).
SERVABLE = {
    "clip.mp4": "video/mp4", "clip.avi": "video/x-msvideo", "summary.json": "application/json",
    "telemetry.jsonl": "text/plain; charset=utf-8", "events.jsonl": "text/plain; charset=utf-8",
    "meta.json": "application/json", "imu.csv": "text/csv", "frames.csv": "text/csv",
    "console.json": "application/json", "score.json": "application/json",
    "thumb.jpg": "image/jpeg", "preview.json": "application/json",
}
_SHOT_PNG = re.compile(r"^shot_\d{1,3}\.png$")
_NOTES = "console.json"
PREVIEW = "preview.json"   # written by tbot.console.thumbs next to thumb.jpg
THUMB = "thumb.jpg"
PREVIEW_VERSION = 2        # = tbot.console.thumbs.PREVIEW_VERSION (kept here: this module must not import cv2)
_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")
_SEQ = re.compile(rb'"seq": (\d+)')


def servable(name: str) -> str | None:
    if name in SERVABLE:
        return SERVABLE[name]
    if _SHOT_PNG.match(name):
        return "image/png"
    return None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _when(name: str, path: Path) -> float:
    m = _DATE.match(name)
    if m:
        try:
            return time.mktime(time.strptime(m.group(0), "%Y%m%d-%H%M%S"))
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.iterdir():
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def notes(path: Path) -> dict:
    d = _read_json(path / _NOTES)
    return d if isinstance(d, dict) else {}


def save_notes(path: Path, patch: dict) -> dict:
    d = notes(path)
    allowed = {"outcome", "note", "starred"}
    for k, v in patch.items():
        if k not in allowed:
            raise ValueError(f"unknown field {k!r}")
        if k == "outcome" and v not in (None, "hit", "miss", "no_shot"):
            raise ValueError("outcome must be hit, miss, no_shot or null")
        if k == "note" and v is not None:
            v = str(v)[:2000]
        if k == "starred":
            v = bool(v)
        d[k] = v
    tmp = path / (_NOTES + ".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(path / _NOTES)
    return d


def frame_seqs(path: Path) -> list[int]:
    """seq of every stored frame, in clip order (frames.csv rows)."""
    out: list[int] = []
    try:
        with open(path / "frames.csv") as f:
            f.readline()
            for line in f:
                head = line.split(",", 1)[0].strip()
                if head.isdigit():
                    out.append(int(head))
    except OSError:
        pass
    return out


def fired_seqs(path: Path) -> list[int]:
    out: list[int] = []
    try:
        with open(path / "telemetry.jsonl", "rb") as f:
            for line in f:
                if b'"fired": true' in line:
                    m = _SEQ.search(line)
                    if m:
                        out.append(int(m.group(1)))
    except OSError:
        pass
    return out


_SHOT_CACHE: dict[str, tuple[float, list[int]]] = {}


def shot_frames(path: Path) -> list[int]:
    """Clip frame numbers (0-based rows of frames.csv) of the frames on which a fire decision was taken."""
    tele = path / "telemetry.jsonl"
    mtime = _mtime(tele)
    if mtime is None:
        return []
    hit = _SHOT_CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return list(hit[1])
    fired = fired_seqs(path)
    seqs = frame_seqs(path)
    index = {s: i for i, s in enumerate(seqs)}
    frames = []
    for s in fired:
        if s in index:
            frames.append(index[s])
        elif seqs:  # not stored (cannot happen for a recorded press): the nearest stored frame after it
            frames.append(next((i for i, q in enumerate(seqs) if q >= s), len(seqs) - 1))
    _SHOT_CACHE[str(path)] = (mtime, frames)
    return frames


def _score_verdicts(path: Path) -> list[dict]:
    s = _read_json(path / "score.json")
    if isinstance(s, list) and s:
        s = s[0]
    if isinstance(s, dict):
        return [r for r in (s.get("shots") or []) if isinstance(r, dict)]
    return []


def summarize(path: Path, clip_fps: float) -> dict:
    name = path.name
    summ = _read_json(path / "summary.json") or {}
    meta_live = summ.get("live")
    events = summ.get("events") or {}
    shots = int(events.get("fire", 0) or 0)
    clip = next((c for c in ("clip.mp4", "clip.avi") if (path / c).exists()), None)
    n = notes(path)
    scored = _score_verdicts(path)
    verdicts = [str(r.get("verdict", "")) for r in scored]
    predicted_hits = sum(1 for r in scored if r.get("hit") is True)
    prev = _read_json(path / PREVIEW) or {}
    thumb_m = _mtime(path / THUMB)
    return {
        "name": name,
        "time": _when(name, path),
        "span_s": summ.get("span_s"),
        "frames": summ.get("frames"),
        "fps": summ.get("fps"),
        "live": bool(meta_live) if meta_live is not None else None,
        "shots": shots,
        "max_p_hit": summ.get("max_p_hit"),
        "detections": summ.get("frames_with_detections"),
        "tracked": summ.get("frames_with_confirmed_track"),
        "clip": clip,
        "clip_fps": clip_fps,
        "complete": (path / "summary.json").exists(),
        "renderable": (path / "frames.bin").exists() and (path / "summary.json").exists(),
        "outcome": n.get("outcome"),
        "note": n.get("note"),
        "starred": bool(n.get("starred")),
        "scored": bool(scored) or (path / "score.json").exists(),
        "predicted_hits": predicted_hits,
        "verdicts": verdicts,
        "shot_images": sorted(p.name for p in path.glob("shot_*.png")),
        # preview: thumb.jpg is the key moment (the shot, else the best tracked frame); key_t is where it sits in
        # the clip, so a hover preview can start just before it. thumb_v changes when the picture does (cache key)
        "thumb": thumb_m is not None,
        "thumb_v": int(thumb_m or 0),
        "key_t": prev.get("clip_t"),
        "key_reason": prev.get("reason"),
    }


def list_sessions(root: Path, clip_fps: float, limit: int = 500) -> list[dict]:
    if not root.is_dir():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and SESSION_NAME.match(d.name)]
    dirs.sort(key=lambda d: _when(d.name, d), reverse=True)
    return [summarize(d, clip_fps) for d in dirs[:limit]]


def needs_thumb(path: Path) -> bool:
    """A finished press whose preview is missing (or older than its telemetry: a re-recorded session)."""
    if not ((path / "summary.json").exists() and (path / "frames.bin").exists()):
        return False
    t = _mtime(path / THUMB)
    if t is None or t < (_mtime(path / "summary.json") or 0):
        return True
    prev = _read_json(path / PREVIEW) or {}
    return int(prev.get("version", 1)) < PREVIEW_VERSION


def needs_clip(path: Path) -> bool:
    return ((path / "summary.json").exists() and (path / "frames.bin").exists() and (path / "meta.json").exists()
            and (path / "telemetry.jsonl").exists()
            and not any((path / c).exists() for c in ("clip.mp4", "clip.avi")))


def detail(path: Path, clip_fps: float) -> dict:
    out = summarize(path, clip_fps)
    summ = _read_json(path / "summary.json") or {}
    out["summary"] = summ
    out["size_bytes"] = _dir_size(path)
    frames = shot_frames(path) if out["shots"] else []
    fires = [e for e in (summ.get("fire_events") or []) if e.get("type") == "fire"]
    scored = _score_verdicts(path)
    shots = []
    for i, fr in enumerate(frames):
        ev = fires[i] if i < len(fires) else {}
        sc = next((r for r in scored if r.get("shot") == i + 1), {})
        shots.append({
            "n": i + 1, "frame": fr, "clip_t": fr / clip_fps if clip_fps > 0 else None,
            "p_hit": ev.get("p_hit"), "range_m": ev.get("range_m"), "miss_deg": ev.get("miss_deg"),
            "live": ev.get("live"), "verdict": sc.get("verdict"), "hit": sc.get("hit"),
            "score_miss_deg": sc.get("miss_deg"), "pattern_deg": sc.get("pattern_deg"),
            "image": f"shot_{i + 1}.png" if (path / f"shot_{i + 1}.png").exists() else None,
        })
    out["shot_list"] = shots
    out["files"] = sorted(p.name for p in path.iterdir() if p.is_file() and servable(p.name))
    return out


def delete(path: Path) -> None:
    shutil.rmtree(path)
