"""Engagement: multi-shot spacing, engaged/duplicate tracks, debris lockout, recoil settling."""

from __future__ import annotations

import math

import numpy as np

from tbot.config import load_config
from tbot.fire.engagement import Engagement
from tbot.types import FireDecision, Track
from tbot.vision.tracker import ClayTracker

MS = 1_000_000


class FakeTracker:
    def __init__(self, tracks):
        self.tracks = tracks

    def predict_dir(self, track, t_ns):
        return ClayTracker.predict_dir(None, track, t_ns)


def trk(tid, az_deg, rate_deg_s=0.0, first_ms=0, el_deg=10.0):
    x = np.array([math.radians(az_deg), math.radians(rate_deg_s), 0.0, math.radians(el_deg), 0.0, 0.0])
    return Track(track_id=tid, t_ns=first_ms * MS, x=x, P=np.eye(6), range_m=20.0, range_sigma_m=2.0,
                 hits=10, misses=0, age=10, confirmed=True, first_t_ns=first_ms * MS)


def shoot(eng, track, cmd_ms, tof_ms=60.0, delay_ms=90.0):
    d = FireDecision(fire=True, t_ns=cmd_ms * MS, reason="FIRE", track_id=track.track_id,
                     t_exit_ns=int((cmd_ms + delay_ms) * MS), tof_s=tof_ms / 1000.0)
    eng.record(d, cmd_ms * MS, track)


def settle(eng, tracker, t_ms, frames=5):
    for k in range(frames):
        eng.observe(tracker.tracks, (t_ms + 20 * k) * MS)


def test_spacing_and_engaged_target():
    cfg = load_config()
    a, b = trk(1, 0.0, 20.0), trk(2, 30.0, -20.0)
    tr = FakeTracker([a, b])
    eng = Engagement(cfg, tr)
    eng.new_press(1)
    assert eng.spacing_ok(0) and eng.why_not(a, 0) is None
    shoot(eng, a, 0)
    assert not eng.spacing_ok(249 * MS)
    assert eng.spacing_ok(250 * MS)
    assert eng.why_not(a, 400 * MS) == "already engaged"
    # the second clay existed before the shot: eligible once it has been re-detected after the recoil
    assert eng.why_not(b, 300 * MS) == "settling after shot"
    settle(eng, tr, 260)
    assert eng.why_not(b, 400 * MS) is None
    assert tr is eng.tracker


def test_duplicate_track_of_the_same_clay_is_engaged_too():
    cfg = load_config()
    a, a2 = trk(1, 0.0, 20.0), trk(2, 0.5, 20.0)
    tr = FakeTracker([a, a2])
    eng = Engagement(cfg, tr)
    eng.new_press(1)
    shoot(eng, a, 0)
    settle(eng, tr, 260)
    assert eng.why_not(a2, 400 * MS) == "already engaged"


def test_fragments_born_after_the_break_are_debris_and_far_new_clays_are_not():
    cfg = load_config()
    a = trk(1, 0.0, 20.0)
    tr = FakeTracker([a])
    eng = Engagement(cfg, tr)
    eng.new_press(1)
    shoot(eng, a, 0)                          # break at 0 + 90 + 60 = 150 ms, clay moving at 20 deg/s
    # a fragment appears at 200 ms on the clay's extrapolated path (4 deg along at 400 ms) and drifts a bit
    frag = trk(5, 3.0, 22.0, first_ms=200)
    far = trk(6, 25.0, -15.0, first_ms=200)   # a new clay well away from the break
    tr.tracks = [a, frag, far]
    settle(eng, tr, 260)
    assert eng.why_not(frag, 400 * MS) == "debris"
    assert eng.why_not(far, 400 * MS) is None
    # past the debris window the lockout lapses (the fragment would long have fallen out of the sky)
    late = 150 + int(cfg.fire.debris_window_ms) + 10
    assert eng.why_not(frag, late * MS) is None


def test_a_new_press_starts_clean():
    cfg = load_config()
    a = trk(1, 0.0, 20.0)
    tr = FakeTracker([a])
    eng = Engagement(cfg, tr)
    eng.new_press(1)
    shoot(eng, a, 0)
    assert eng.why_not(a, 400 * MS) == "already engaged"
    eng.new_press(1)  # same press: nothing changes
    assert eng.why_not(a, 400 * MS) == "already engaged"
    eng.new_press(2)
    assert eng.why_not(a, 400 * MS) is None and eng.spacing_ok(1)
