# Integration status (handoff)

Living notes on bench integration of the real rig: what is calibrated, what is still open, and where the
per-rig values live. Last updated 2026-11-15 (IMU calibrated, multi-shot, speed-ups). The Pi keeps its own
`config/local.toml` and `calib/` (git-ignored; `scripts/deploy.sh push` never overwrites them), so the values
below are a record, not something a checkout restores.

## Access and setup

- Pi: `ssh -o UserKnownHostsFile=~/.ssh/known_hosts_tbot pi@tbot.local` (192.168.0.168 on the home LAN; when
  mDNS does not resolve use the IP with `-o HostKeyAlias=tbot.local`). The host key changes on every reflash:
  re-run `ssh-keyscan -t ed25519 tbot.local > ~/.ssh/known_hosts_tbot` instead of editing `~/.ssh/known_hosts`.
  Credentials: `software/pi/README.md`.
- **Wi-Fi is weak on the bench** (up to 75 % packet loss seen 2026-11-15): copies and SSH sessions drop. Push
  file by file with retries and verify checksums; run long jobs detached (`setsid nohup … > log`) and poll.
  At the range, do not depend on Wi-Fi for anything in the firing path (it is not in it); the debug stream may
  stutter.
- The Pi has no git; push code with `scripts/deploy.sh push`, or `scp` the changed files.
- Run anything that opens the camera **detached**: a command holding the camera can hang an SSH session.
- Blue Snowball mic: address it as `plughw:CARD=Snowball,DEV=0` (its card number changes across reboots).

## Per-rig configuration on the Pi (`config/local.toml`)

| key | value | source |
|---|---|---|
| `actuator.servo.rest_us` / `fire_us` | 1420 / 1530 | bench; lever has some spline play |
| `actuator.servo.min_us` / `max_us` | 1400 / 1540 | valid lever travel (the actuator refuses to fire outside it) |
| `camera.rotate_180` | false | the board is upright (now also the default) |
| `camera.fps` | 50 | see "Frame rate" |

## Calibration on the Pi (`calib/`)

| item | value | how | redo with the final lens? |
|---|---|---|---|
| camera timestamp offset | +0.03 ms | `latency-camera`, 4000 us exposure (a later 500 us run could not measure it and kept this value) | no (sensor) |
| exposure end -> user space | ~9.2 ms at 500 us | `latency-camera` | no |
| servo command -> break | **95 ms** (std 2.5 assumed) | video: LED at the fire command, lever motion centre 92 +/- 10 ms; mic 161 ms incl. USB latency | no |
| IMU gyro bias | +0.82 / +6.45 / +0.48 deg/s | `calib-imu bias`; drifts with temperature (z 0.48 -> 0.84 between 24 and 22 C) | no, but redo at the range |
| IMU axis map | `R_gun_imu` = [[0,-1,0],[-1,0,0],[0,0,-1]] (chip face-down, X/Y swapped) | `calib-imu axes --auto`, 3 hand-held poses, fit within 0.8 deg; also the default now | no |
| camera-IMU time offset | -6.31 ms (added to IMU stamps) | `calib-imu sync` on a 15 s hand-held swing, correlation 0.971 | no |
| boresight | pitch -1.71, yaw -4.03 deg | barrel silhouette, nominal lens model | **yes** |
| lens intrinsics | none (nominal model) | `tbot board` + `calib-intrinsics` | **yes** |
| detector blur `detector.psf_px` | 1.2 (guess) | `calib-detector --psf` on clay footage | **yes** |

## Software state (main)

- Multi-shot (`tbot/fire/engagement.py`): one press fires up to `fire.max_shots_per_arm = 3` shots, each at its
  own clay; >= `fire.min_shot_interval_ms = 250` between commands; a shot-at clay (and any duplicate track
  within 1 deg) is never engaged again; tracks born after a shot inside a cone around the broken clay's
  predicted path (2 deg, +15 deg/s, 1.5 s) are debris; after a shot a target needs `min_track_hits` fresh
  detections later than `recoil_settle_ms = 100` past exit. The pipeline does not spend a shot while the
  servo is still cycling (hold 150 + return 250 ms = 400 ms, so doubles are >= ~400 ms apart until the return
  time is measured shorter on the bench). Solver refuses targets slower than 6 m/s (cartesian tracker).
- Camera: `tbot run` / `tbot record` reopen a stalled camera (`camera.stall_s` = 0.5 s) instead of ending.
- Heat: `tbot run` warns at 75 C (firmware throttles at 80 C) and on the soft temperature limit.
- Tracker: 3-D cartesian EKF (`tracker.model = "cartesian"`). Detector: matched filter + hysteresis + speck
  and local-contrast tests. Solver: extrapolates the gun swing with its angular acceleration.
- Servo: limp when idle, short bursts, refuses to fire outside calibrated travel.

## Frame rate and CPU

- Camera/driver: capture-only 50-200 fps with zero drops; the limit is processing.
- Pi 3B pipeline with a clay tracked (synthetic crossing): detect 13.0 + track 2.6 + solve 1.7 = **17.5 ms
  p50, 23 ms p95** (was ~25 ms before 2026-11-15's speed-ups: detector 8-bit threshold / cv2.moments /
  vectorised rejection, tracker scalar ray path). Live camera, no clay: 7.0 ms p50. Keep **50 fps**.
- OpenCV uses 4 threads: 1 thread is much slower (live 12.4 vs 7.0 ms, synthetic 23 vs 17.6 ms), so the
  ~150-190 % process CPU (all four cores ~50 %) is buying speed, not wasted.
- Heat: 3 min live at 50 fps took the bare Pi from 56 to 66 C and still rising slowly (no throttling). In a
  closed pod in the sun this can reach the 80 C throttle: give the Pi a heatsink and airflow/shade, and watch
  for the 75 C warning.
- Further speed would need pipelining across processes (detect in one, track/solve/fire in another): more
  throughput, same latency. Not done before the first trip.

## Simulator notes

- `tbot sim` is not repeatable run to run (SimClock adds real processing time): ~1/3 of shots differ between
  runs of the same code, so +/-2 hits out of ~20 is noise. Compare code changes frame by frame instead.
- Doubles (`skeet_doubles`, `trap_doubles`), clay breaks with fragments and debris-shot counting exist. Latest
  (95-100 ms delay, 6 seeds): singles outgoing 6/6, trap 6/6, incoming 3/3 of shots taken; **0 debris shots**;
  doubles mostly get one shot (second clay not in the solution before the run ends).
- Scenarios last 2.5 s and the truth freezes after that: late crossing shots (the simulated swing fires ~2 s
  after launch) are judged against a frozen clay, so `skeet_crossing` "misses" are mostly artefacts.
- Fused range from clay size lags radial motion (trap ~0.77-0.82 of true at shot, incoming ~0.84); per-frame
  size range is ~0.91 with psf 1.2 and ~0.98 with psf 1.6 in the sim. Calibrate psf on real footage.

## Before the range (checklist)

1. Install the **final lens with the IR filter/protector**, set focus (`tbot focus`), then: lens intrinsics
   (`tbot board`, `calib-intrinsics`, chessboard), then `calib-boresight --barrel` on a full 1280x800 still,
   then re-check the static/barrel mask. These depend on the lens; the IMU, latency and servo values do not.
2. Secure and strain-relieve the camera ribbon at both ends; prove it with a 60 s hard-swing recording.
3. Heatsink + airflow/shade for the Pi in the pod.
4. At the range: `calib-imu bias` with the gun still (temperature), then record sky + clay footage
   (`tbot record`) before trying live shots: detector thresholds, mask and `psf_px` need real footage
   (`calib-detector`). Exposure may need raising with the IR filter fitted.
5. Optional, with the user present and agreeing: measure the servo return time so doubles can be faster than
   ~400 ms; servo wake-up from limp (~115 ms) could be cut by holding rest while armed (needs re-measurement).
6. Power: under-voltage/throttling has been seen when servo bursts sag the shared battery.

## Known test failure

`tests/test_hitl.py::test_sim_live_geometry_and_flash_end_to_end` fails on both the Pi and a PC on code that
predates this work.
