# Integration status (handoff)

Living notes on bench integration of the real rig: what is calibrated, what is still open, and where the
per-rig values live. Last updated 2026-11-15. The Pi keeps its own `config/local.toml` and `calib/`
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

## Open items

1. IMU: axis map (`calib-imu axes`, three poses) and camera-IMU time offset (`calib-imu sync`, needs a
   recorded swing). `R_gun_imu` is the layout's identity until then.
2. Lens intrinsics (`tbot board`, `calib-intrinsics`), then re-run `calib-boresight --barrel` on a full
   1280x800 still: the pitch disagreement above is the nominal lens model.
3. Detector thresholds and the static mask on real sky footage (`tbot record`, `calib-detector`): sky
   texture, hot pixels and clay contrast decide `k_sigma`, `min_contrast`, `polarity`.
4. Servo wake-up: the MG996R takes ~115 ms to start moving from limp. Holding `rest` while the button is held
   would cut most of the delay, but needs a re-measurement.
5. Power: under-voltage/throttling has been seen (servo bursts sagging the shared battery).
6. Pre-existing: `tests/test_hitl.py::test_sim_live_geometry_and_flash_end_to_end` fails on both the Pi and a
   PC on code that predates this work.
