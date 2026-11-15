"""Engagement: which tracks may still be shot during one press of the arming button.

One press may fire up to `fire.max_shots_per_arm` shots (doubles, triples). Between shots:

- spacing: no fire command within `fire.min_shot_interval_ms` of the previous one;
- engaged: a track that was shot at is never targeted again, nor is any track that sat within
  `fire.duplicate_deg` of it at the shot (a second track on the same clay);
- debris: a track born after a shot (from `fire.debris_birth_margin_ms` before the predicted break on) whose
  direction lies inside a cone around the broken clay's predicted path is a fragment. Fragments carry the
  clay's momentum, so the cone follows the clay's extrapolated path and widens at `fire.debris_spread_deg_s`
  from `fire.debris_cone_deg`, for `fire.debris_window_ms` after the break;
- recoil: after a shot, a target needs `fire.min_track_hits` fresh detections later than
  `fire.recoil_settle_ms` past the shot's exit (recoil shakes the gun and the attitude estimate; a track
  that keeps being re-detected consistently afterwards is trustworthy again).

Tracks that already existed before a shot (a real second clay) stay eligible unless engaged.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from ..types import FireDecision, Track

__all__ = ["Engagement", "ShotRecord"]


@dataclass
class ShotRecord:
    cmd_ns: int            # trigger command time
    exit_ns: int           # predicted shot exit
    break_ns: int          # predicted arrival at the clay (where it breaks, if hit)
    track_id: int
    target: Track          # frozen copy of the target track at the shot (its path carries the debris)
    engaged_ids: frozenset[int]


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), ax * bx + ay * by + az * bz)


class Engagement:
    """Per-press shot bookkeeping and target eligibility (see module docstring)."""

    def __init__(self, cfg, tracker):
        self.cfg = cfg
        self.tracker = tracker
        self.shots: list[ShotRecord] = []
        self._press: int | None = None
        self._fresh: dict[int, int] = {}  # track_id -> detections since the last shot's recoil settled

    # ------------------------------------------------------------------ state
    def new_press(self, press_count: int) -> None:
        """Forget the previous press's shots when the button is pressed again."""
        if press_count != self._press:
            self._press = press_count
            self.shots.clear()
            self._fresh.clear()

    def observe(self, tracks: list[Track], t_ns: int) -> None:
        """Count, per track, the detections it received after the last shot's recoil settled."""
        if not self.shots:
            return
        settle_ns = self.shots[-1].exit_ns + int(float(self.cfg.fire.recoil_settle_ms) * 1e6)
        if int(t_ns) < settle_ns:
            return
        live = set()
        for tr in tracks:
            live.add(tr.track_id)
            if tr.misses == 0:
                self._fresh[tr.track_id] = self._fresh.get(tr.track_id, 0) + 1
        for tid in list(self._fresh):
            if tid not in live:
                del self._fresh[tid]

    def record(self, decision: FireDecision, cmd_ns: int, target: Track | None) -> None:
        """Book a shot that was really commanded (or would have been, in a dry run)."""
        exit_ns = int(decision.t_exit_ns) if decision.t_exit_ns is not None else int(cmd_ns)
        tof = float(decision.tof_s) if decision.tof_s is not None and math.isfinite(float(decision.tof_s)) else 0.0
        break_ns = exit_ns + int(round(tof * 1e9))
        engaged = {int(decision.track_id)} if decision.track_id is not None else set()
        frozen = copy.deepcopy(target) if target is not None else None
        if frozen is not None:
            d_t = self.tracker.predict_dir(frozen, int(cmd_ns))
            dup = math.radians(float(self.cfg.fire.duplicate_deg))
            for tr in self.tracker.tracks:
                if tr.track_id != frozen.track_id and _angle(self.tracker.predict_dir(tr, int(cmd_ns)), d_t) <= dup:
                    engaged.add(tr.track_id)
        self.shots.append(ShotRecord(int(cmd_ns), exit_ns, break_ns, int(decision.track_id or -1), frozen,
                                     frozenset(engaged)))
        self._fresh.clear()

    # ------------------------------------------------------------------ gates
    def spacing_ok(self, now_ns: int) -> bool:
        if not self.shots:
            return True
        return int(now_ns) - self.shots[-1].cmd_ns >= int(float(self.cfg.fire.min_shot_interval_ms) * 1e6)

    def why_not(self, tr: Track, now_ns: int) -> str | None:
        """None when `tr` may be targeted at now_ns, else the reason it may not."""
        if not self.shots:
            return None
        fc = self.cfg.fire
        for shot in self.shots:
            if tr.track_id in shot.engaged_ids:
                return "already engaged"
        now_ns = int(now_ns)
        d = None
        for shot in self.shots:
            if shot.target is None:
                continue
            since = now_ns - shot.break_ns
            if since > int(float(fc.debris_window_ms) * 1e6):
                continue
            if tr.first_t_ns < shot.break_ns - int(float(fc.debris_birth_margin_ms) * 1e6):
                continue  # existed before this shot: a real clay, not one of its fragments
            if d is None:
                d = self.tracker.predict_dir(tr, now_ns)
            centre = self.tracker.predict_dir(shot.target, now_ns)
            cone = math.radians(float(fc.debris_cone_deg) + float(fc.debris_spread_deg_s) * max(0.0, since * 1e-9))
            if _angle(d, centre) <= cone:
                return "debris"
        if self._fresh.get(tr.track_id, 0) < int(fc.min_track_hits):
            return "settling after shot"
        return None

    def accept(self, now_ns: int):
        """Predicate for tracker.best(): tracks this press may still shoot at now_ns."""
        return lambda tr: self.why_not(tr, now_ns) is None
