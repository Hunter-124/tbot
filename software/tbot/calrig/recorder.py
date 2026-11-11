"""Run recorder: one directory per run, laid out exactly as docs/CALRIG_PROTOCOL.md section 4.

meta.json, rig.csv, cmd.csv, hall.csv, events.csv, imu.csv. Firmware-timed rows (rig, hall, events) are kept as
(t_us unwrapped, receive time); the CSVs carry the raw uint32 t_us exactly as the firmware sent it while the run is going and converted to the host monotonic clock with the FINAL
clock fit at finalize(), so pings from after the run improve every row. imu.csv is streamed to disk as it
arrives (its timestamps are already host clock); cmd.csv is written at the end from the device's command log.
"""

from __future__ import annotations

import csv
import datetime as _dt
import platform
import socket
from pathlib import Path
from typing import Any

from tbot import __version__
from tbot.calrig import protocol as P
from tbot.calrig.clocksync import ClockFit
from tbot.config import now_iso, save_json

__all__ = ["RunRecorder", "SIGN_CONVENTION", "POSES", "EXTRA_POSES", "ALL_POSES"]

POSES = ("upright", "left_side_down", "right_side_down", "inverted")  # the four rig poses of docs/CALIBRATION
EXTRA_POSES = ("muzzle_up", "muzzle_down")                            # accepted by the analysis too
ALL_POSES = POSES + EXTRA_POSES
SIGN_CONVENTION = "table angle positive = counter-clockwise seen from above (right-hand rotation about +Z up)"


def _fmt(x: float) -> str:
    return f"{x:.9g}"


class RunRecorder:
    def __init__(self, root: Path, procedure: str, geometry: P.Geometry, now: _dt.datetime | None = None):
        self.geometry = geometry
        self.procedure = procedure
        stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{stamp}_{procedure}"
        n = 1
        while True:  # two runs in the same second must not share a directory
            try:
                path.mkdir()
                break
            except FileExistsError:
                n += 1
                path = root / f"{stamp}_{procedure}-{n}"
        self.dir = path
        self.start_iso = now_iso()
        self._rig: list[tuple[int, int, int, int]] = []     # t_rx, t_us, pos, vel
        self._hall: list[tuple[int, int, int, int]] = []    # t_rx, t_us, pos, level
        self._events: list[tuple[int, int, str, str]] = []  # t_rx, t_us, code, arg
        self._cmds: list[tuple[int, str]] = []
        self._imu_f = open(path / "imu.csv", "w", newline="")
        self._imu = csv.writer(self._imu_f)
        self._imu.writerow(["t_ns", "gx", "gy", "gz", "ax", "ay", "az", "temp_c"])
        self.imu_samples = 0
        self._closed = False

    # ---------------------------------------------------------------- feeding
    def add_telemetry(self, items) -> None:
        self._rig.extend((rx, u, m.pos, m.vel) for rx, u, m in items)

    def add_hall(self, items) -> None:
        self._hall.extend((rx, u, m.pos, m.level) for rx, u, m in items)

    def add_events(self, items) -> None:
        self._events.extend((rx, u, m.code, "" if m.arg is None else str(m.arg)) for rx, u, m in items)

    def add_commands(self, items) -> None:
        self._cmds.extend(items)

    def add_imu(self, samples) -> None:
        """ImuSample list; gyro rad/s and accel m/s^2 in the RAW chip frame (spec section 4)."""
        w = self._imu
        for s in samples:
            g, a = s.gyro, s.accel
            w.writerow([int(s.t_ns), _fmt(g[0]), _fmt(g[1]), _fmt(g[2]), _fmt(a[0]), _fmt(a[1]), _fmt(a[2]),
                        _fmt(s.temp_c)])
        self.imu_samples += len(samples)

    # ---------------------------------------------------------------- output
    def finalize(self, meta: dict[str, Any], fit: ClockFit | None) -> Path:
        """Write the firmware-timed CSVs with the final clock fit, then meta.json. Idempotent."""
        if self._closed:
            return self.dir
        self._closed = True
        self._imu_f.close()
        g = self.geometry

        def host_ns(rx: int, t_us: int) -> int:
            return fit.to_host_ns(t_us) if fit is not None else rx

        self._write("rig.csv", ["t_ns", "t_us", "pos_ustep", "angle_deg", "vel_deg_s"],
                    ((host_ns(rx, u), u % P.UINT32, pos, _fmt(g.to_deg(pos)), _fmt(g.to_deg(vel)))
                     for rx, u, pos, vel in self._rig))
        self._write("cmd.csv", ["t_ns", "command"], self._cmds)
        self._write("hall.csv", ["t_ns", "t_us", "pos_ustep", "angle_deg", "level"],
                    ((host_ns(rx, u), u % P.UINT32, pos, _fmt(g.to_deg(pos)), level) for rx, u, pos, level in self._hall))
        self._write("events.csv", ["t_ns", "t_us", "code", "arg"],
                    ((host_ns(rx, u), u % P.UINT32, code, arg) for rx, u, code, arg in self._events))

        meta = dict(meta)
        meta.setdefault("procedure", self.procedure)
        meta["deg_per_ustep"] = g.deg_per_ustep
        meta["usteps_per_deg"] = g.usteps_per_deg
        meta["sign_convention"] = SIGN_CONVENTION
        meta["clock_fit"] = ({"offset_ns": fit.offset_ns, "drift_ppm": fit.drift_ppm, "resid_us": fit.resid_us,
                          "n": fit.n, "n_total": fit.n_total, "rtt_med_us": fit.rtt_med_us,
                          "rtt_min_us": fit.rtt_min_us}
                         if fit is not None else
                         {"offset_ns": None, "drift_ppm": None, "resid_us": None, "n": 0,
                          "note": "no PING round trips: rig times are host receive times"})
        meta["host"] = {"hostname": socket.gethostname(), "platform": platform.platform(),
                        "python": platform.python_version()}
        meta["software_version"] = __version__
        meta["start"] = self.start_iso
        meta["end"] = now_iso()
        meta["counts"] = {"rig": len(self._rig), "hall": len(self._hall), "events": len(self._events),
                          "cmd": len(self._cmds), "imu": self.imu_samples}
        save_json(self.dir / "meta.json", meta)
        return self.dir

    def _write(self, name: str, header: list[str], rows) -> None:
        with open(self.dir / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def abort(self) -> None:
        """Release the open file when the run never reached finalize()."""
        if not self._closed:
            self._imu_f.close()

