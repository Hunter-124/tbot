"""SensorProcess replaces a sensor child that dies, hangs or stops delivering frames (camera/proc.py).

The live camera is a picam on the Pi; here the child runs the synthetic source, which exercises the same
parent-side supervision: frames resume, `seq` keeps rising, the first frame after the outage carries
gap_after_restart, and the counters survive the replacement.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from tbot.camera.proc import SensorProcess
from tbot.config import Calibration, load_config


def _cfg(tmp_path):
    cfg = load_config(sets=["sim.process_ms=0.0", "camera.fps=50.0"])
    cfg.paths.calib_dir = str(tmp_path / "calib")
    return cfg


def _frames(proc, n, timeout_s=20.0):
    out, deadline = [], time.monotonic() + timeout_s
    while len(out) < n and time.monotonic() < deadline:
        f = proc.read(timeout_s=0.2)
        if f is not None:
            out.append((f, proc.gap_after_restart))
            proc.gap_after_restart = False
    assert len(out) == n, f"only {len(out)} of {n} frames within {timeout_s} s"
    return out


def _wait_respawn(proc, respawns, timeout_s=30.0):
    """Read until the proxy has replaced its child `respawns` times and frames flow again."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc.read(timeout_s=0.1)
        if proc.respawns >= respawns and proc.streaming:
            return
    raise AssertionError(f"no respawn within {timeout_s} s (respawns {proc.respawns}, {proc.last_error})")


def test_dead_child_is_replaced_and_frames_resume(tmp_path):
    cfg = _cfg(tmp_path)
    proc = SensorProcess(cfg, Calibration(dir=tmp_path / "calib"), "synthetic:static_gun", respawn_s=5.0)
    try:
        before = _frames(proc, 5)
        old_pid = proc.child_pid
        proc._proc.kill()
        _wait_respawn(proc, 1)
        assert proc.child_pid != old_pid and "died" in (proc.last_error or "")
        after = _frames(proc, 5)
        seqs = [f.seq for f, _ in before + after]
        assert seqs == sorted(set(seqs)), seqs               # strictly rising across the two children
        assert any(gap for _, gap in after[:2])               # the outage is flagged to the pipeline
        assert proc.restarts >= 1 and proc.respawns == 1
        imu = proc.imu()
        assert imu is not None and imu.read() is not None    # the IMU proxy follows the new child
    finally:
        proc.stop()
    assert not proc._proc.is_alive()


def test_child_without_frames_is_replaced(tmp_path):
    """Alive and talking (heartbeats) but no frame for respawn_s: the in-child reopen is not working."""
    cfg = _cfg(tmp_path)
    proc = SensorProcess(cfg, Calibration(dir=tmp_path / "calib"), "synthetic:static_gun", respawn_s=1.0)
    try:
        _frames(proc, 3)
        proc._request_stop()  # the child stops reading its camera ... and here exits, which is also covered
        _wait_respawn(proc, 1)
        _frames(proc, 3)
    finally:
        proc.stop()


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGSTOP")
def test_hung_child_is_killed_and_replaced(tmp_path):
    cfg = _cfg(tmp_path)
    proc = SensorProcess(cfg, Calibration(dir=tmp_path / "calib"), "synthetic:static_gun", respawn_s=30.0,
                         hang_s=1.0)
    try:
        _frames(proc, 3)
        old = proc._proc
        os.kill(proc.child_pid, signal.SIGSTOP)   # wedged: alive, holding everything, silent
        t0 = time.monotonic()
        _wait_respawn(proc, 1)
        assert time.monotonic() - t0 < 20.0
        assert "hung" in (proc.last_error or "") and not old.is_alive()
        _frames(proc, 3)
    finally:
        proc.stop()


def test_without_respawn_a_dead_child_still_raises(tmp_path):
    cfg = _cfg(tmp_path)
    proc = SensorProcess(cfg, Calibration(dir=tmp_path / "calib"), "synthetic:static_gun")
    try:
        assert proc.respawn_s == 0.0   # not a live camera: the old contract
        _frames(proc, 2)
        proc._proc.kill()
        proc._proc.join(5.0)
        with pytest.raises(RuntimeError):
            for _ in range(200):
                proc.read(timeout_s=0.05)
    finally:
        proc.stop()
