"""CalRig B: host software for the motorised yaw table (docs/CALRIG.md, contract docs/CALRIG_PROTOCOL.md).

Modules: protocol (pure line parsing), timebase (real / virtual clocks), device (serial driver), clocksync
(firmware -> host clock fit), sim (in-process firmware + synthetic IMU), recorder (run directory), session
(safety + orchestration), procedures (named calibration procedures), analysis (run directory -> results).
Nothing here touches the firing / actuator code.
"""
