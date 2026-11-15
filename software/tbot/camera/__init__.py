"""Frame sources. open_source() maps a source spec to an implementation (imported lazily)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tbot.camera.base import FrameSource, SourceBase

if TYPE_CHECKING:
    from tbot.config import Calibration, Config

__all__ = ["FrameSource", "SourceBase", "open_live_source", "open_source"]


def open_live_source(spec: str, cfg: "Config", calib: "Calibration") -> FrameSource:
    """open_source for long-running live use (tbot run / record): a picam is wrapped in RecoveringSource so a
    CSI stall or crashed capture thread reopens the camera instead of ending the stream."""
    if spec.partition(":")[0] != "picam":
        return open_source(spec, cfg, calib)
    from tbot.camera.recover import RecoveringSource

    return RecoveringSource(lambda: open_source(spec, cfg, calib), stall_s=float(cfg.camera.stall_s),
                            open_timeout_s=float(cfg.camera.reopen_timeout_s))


def open_source(spec: str, cfg: "Config", calib: "Calibration") -> FrameSource:
    """picam | cv:<index> | video:<path> | replay:<session_dir> | synthetic[:<scenario>]"""
    kind, _, arg = spec.partition(":")
    if kind == "picam":
        from tbot.camera.picam import PicamSource

        return PicamSource(cfg, calib)
    if kind in ("cv", "video"):
        from tbot.camera.cv_source import CvSource

        return CvSource(cfg, spec)
    if kind == "replay":
        if not arg:
            raise ValueError("replay source needs a session directory: replay:<dir>")
        from tbot.camera.replay import ReplaySource

        return ReplaySource(arg, realtime=False)
    if kind == "synthetic":
        from tbot.camera.synthetic import SyntheticSource

        return SyntheticSource(cfg, calib, scenario=arg or None)
    raise ValueError(f"unknown source spec {spec!r} (picam | cv:<i> | video:<path> | replay:<dir> | synthetic[:<name>])")
