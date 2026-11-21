"""The gun's own lens for tests of the motion front end.

An empty calibration directory gives CameraModel.nominal: the 2.8 mm lens the gun was first built with (fx 467 px,
8 px per degree). Through it the simulator's clay at 20 m is under 3 px across, one pixel at the motion detector's
half resolution, and the pixel-valued [vision] thresholds are set for the 12 mm lens the gun has carried since
2026-11-16 (fx 2009 px, 35 px per degree; calib/intrinsics.json and boresight.json of that day, copied here).
"""

from __future__ import annotations

import json
from pathlib import Path

INTRINSICS = {"model": "pinhole", "width": 640, "height": 400,
              "K": [[2009.0, 0, 319.5], [0, 2009.0, 199.5], [0, 0, 1]], "D": [0, 0, 0, 0, 0]}
BORESIGHT = {"rpy_deg": [0.0, -1.891405500774123, -1.6673685443944335], "range_m": 13.75,
             "bore_pixel": [522.49, 516.7], "width": 1280, "height": 800}


def write_rig_calibration(calib_dir) -> None:
    d = Path(calib_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "intrinsics.json").write_text(json.dumps(INTRINSICS))
    (d / "boresight.json").write_text(json.dumps(BORESIGHT))
