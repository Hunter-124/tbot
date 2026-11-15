# Integration status (handoff)

Living notes on bench integration of the real rig: what is calibrated, what is still open, and where the
per-rig values live. Last updated 2026-11-15 (IMU calibrated). The Pi keeps its own `config/local.toml` and `calib/`
(git-ignored; `scripts/deploy.sh push` never overwrites them), so the values below are a record, not
something a checkout restores.

## Access and setup

- Pi: `ssh -o UserKnownHostsFile=~/.ssh/known_hosts_tbot pi@tbot.local` (192.168.0.168 on the home LAN).
  The host key changes on every reflash: re-run `ssh-keyscan -t ed25519 tbot.local > ~/.ssh/known_hosts_tbot`
  instead of editing `~/.ssh/known_hosts`. Credentials: `software/pi/README.md`.
- The Pi has no git; push code with `scripts/deploy.sh push` (or tar over ssh).
- Run anything that opens the camera **detached** (`setsid nohup … > log`): SSH sessions drop, and a command
  holding the camera can otherwise hang the session.
- Blue Snowball mic: address it as `plughw:CARD=Snowball,DEV=0` (its card number changes across reboots).

## Per-rig configuration on the Pi (`config/local.toml`)

| key | value | source |
|---|---|---|
| `actuator.servo.rest_us` / `fire_us` | 1420 / 1530 | bench; lever has some spline play |
| `actuator.servo.min_us` / `max_us` | 1400 / 1540 | valid lever travel (the actuator refuses to fire outside it) |
| `camera.rotate_180` | false | the board is upright (now also the default) |
| `camera.fps` | 50 | Pi 3B holds 50 fps live (0.5-0.7 % drops); 40 fps measurably worse in sim |

## Calibration on the Pi (`calib/`)

| item | value | how |
|---|---|---|
| camera timestamp offset | +0.03 ms (sigma 0.04) | `latency-camera`, 4000 us exposure |
| exposure end -> user space | ~9.2 ms at 500 us | `latency-camera` |
| servo command -> break | **80 ms** (estimate; written as `method: manual estimate`) | mic upper bound 160 ms; operator measurement ~80-90 ms |
| IMU gyro bias | +0.82 / +6.45 / +0.48 deg/s | `calib-imu bias`, quiet runs only |
| IMU axis map | `R_gun_imu` = [[0,-1,0],[-1,0,0],[0,0,-1]] (chip face-down, X/Y swapped) | `calib-imu axes --auto`, 3 hand-held poses, fit within 0.8 deg; also the default now |
| camera-IMU time offset | -6.31 ms (added to IMU stamps) | `calib-imu sync` on a 15 s hand-held swing, correlation 0.971 |
| boresight | pitch -1.71, yaw -4.03 deg | barrel silhouette, nominal lens; `calib-boresight --barrel` gives pitch -2.68, yaw -3.91 |
| processing (bench) | ~24 ms p95 | `tbot bench` on the Pi |

The measured servo delay came out at roughly 80-90 ms; 80 ms is in `latency.json`. If a better number is
measured, write it with the same structure (`actuator.servo.cmd_to_break_ms.mean/std`).

## Software state (main)

- Tracker: 3-D cartesian EKF is the default (`tracker.model = "cartesian"`); `angular` remains available.
- Detector: matched filter + local hysteresis + speck and local-contrast tests (`detector.*` keys).
- Solver: extrapolates the gun swing with its angular acceleration (`fire.gun_accel_sigma_frac`).
- Servo: limp when idle, short bursts, refuses to fire outside calibrated travel; `tbot jog servo` creeps
  (max 50 us per move) until `lo`/`hi` are marked.
- `tbot latency-mic`: actuator latency from a microphone, one shot per arming press (`--led` lights the
  latency LED at each fire command as a video reference).
- Sim (50 fps, 80 ms, 6 seeds x 4 scenarios, P(hit) >= 0.5): 17/23 shots hit, incoming 5/5. Simulator
  evidence only; nothing has been tested on real clays.

## Frame rate

The camera and driver are not the limit: capture-only on the Pi 3B delivers 50/100/150/200 fps with zero
drops (250 fps requested times out: above the selected 10-bit mode's 247 fps). The limit is processing.
Frames the pipeline is too slow for are skipped (newest frame wins); the trackers use real timestamps, so a
skip is only a longer step.

Detector speed-up (same detections frame for frame as before; centroids within 1e-7 px median): 8-bit
threshold, cv2.moments, and the candidate rejection tests vectorised when a frame has >= 8 candidates.
Pi 3B detect p50 / p95: room footage 10.1 / 12.7 -> 6.4 / 9.7 ms; synthetic crossing 18.2 / 22.1 ->
12.7 / 15.5 ms; synthetic incoming p95 48 -> 37 ms.

Whole pipeline on the Pi (`tbot bench`): live camera, no clay in view: 7.0 ms p50 at 50 fps, 7.7 at 75,
10.5 at 100 (the capture thread costs more at higher rates). Synthetic crossing with a clay tracked:
detect 13.3 + track 4.7 + solve 1.9 = 20.3 ms p50 (25 ms p95) -> still ~50 fps with a clay in view.
The synthetic sky is textured (~30 candidates per frame); real-sky cost is unknown until range footage.
Keep 50 fps for the first trip; next targets are the tracker (4.7 ms) and solver (1.9 ms).
`tbot sim` is not repeatable run to run (SimClock adds real processing time): the same code differs in
~1/3 of shots between runs, so +/-2 hits out of 21 is noise.

## Open items

1. **Camera connector:** during the hand-held sync swing the camera stopped ~15 s in (libcamera "Camera
   frontend has timed out ... check that your camera sensor connector is attached securely"); the sensor still
   answered on I2C but no frames came, also with stock `rpicam-vid`, across a power cycle. It streamed again
   (20 s at 50.0 fps) after the rig was handled; secure and strain-relieve the ribbon and prove it with a long
   hard-swing recording before the range. `tbot run` / `tbot record` now reopen a stalled camera
   (`camera.stall_s`, `RecoveringSource`) instead of ending; frames during an outage are lost and tracks are
   reset. Recovery is unit-tested; not yet exercised on a real mid-stream stall. Redo `calib-imu bias` at the
   range temperature (z drifted 0.48 -> 0.84 deg/s between 24 C and 22 C).
2. Lens intrinsics (`tbot board`, `calib-intrinsics`), then re-run `calib-boresight --barrel` on a full
   1280x800 still: the pitch disagreement above is the nominal lens model.
3. Detector thresholds and the static mask on real sky footage (`tbot record`, `calib-detector`): sky
   texture, hot pixels and clay contrast decide `k_sigma`, `min_contrast`, `polarity`.
4. Servo wake-up: the MG996R takes ~115 ms to start moving from limp. Holding `rest` while the button is held
   would cut most of the delay, but needs a re-measurement.
5. Power: under-voltage/throttling has been seen (servo bursts sagging the shared battery).
6. Pre-existing: `tests/test_hitl.py::test_sim_live_geometry_and_flash_end_to_end` fails on both the Pi and a
   PC on code that predates this work.
