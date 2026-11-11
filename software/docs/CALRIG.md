# CalRig B — operator guide

CalRig B is a motorised yaw table (NEMA 17 direct drive, 1:1, 0.1125 deg per microstep at 16 microsteps) that swings
the **unloaded** gun so the gun-mounted MPU-6050 can be calibrated: gyro bias, gyro scale per axis, axis map and sign
(`R_gun_imu`), IMU time offset, and a lost-step check of the rig itself. Mechanics: `design/calibration_rig.md`
section 7. Interface contract (pins, serial protocol, clock sync, run directory format):
[CALRIG_PROTOCOL.md](CALRIG_PROTOCOL.md). Firmware, wiring, Vref and flashing:
[../firmware/calrig_uno/README.md](../firmware/calrig_uno/README.md) — start there; this guide assumes the firmware
is flashed and `tbot calrig info` answers.

The rig never fires and never touches the trigger actuator: no firing or actuator code is imported by `tbot.calrig`.

## Wiring summary

Arduino Uno + CNC Shield V3, A4988 (or DRV8825 / TMC2209) driver in the **X** socket, motor 12 V on the shield
terminal; STEP D2, DIR D5, EN D8; A3144 hall switch on D9 (home); normally-closed e-stop button on A0. Microstep
jumpers must match the firmware `MICROSTEPS` (default 16) and `[calrig] microsteps` in `config/default.toml`;
`tbot calrig` refuses to start if the firmware's `INFO` disagrees. Full table, Vref formulas and the first power-up
checklist are in the firmware README.

## Bench safety

- **The gun is UNLOADED**: chamber checked, action open, no ammunition in the room. Muzzle in a safe direction.
- `home`, `jog` and `run` will not energise the motor until you type `UNLOADED` at the prompt (no flag skips it; only
  `--sim`, which has no hardware, does).
- Clamp or screw the base down. Candidate B has **no mechanical stops**: only the software soft limits (+-90 deg) and the
  firmware hard limit (+-100 deg). Keep the swing arc (the whole 1.2 m gun, +-100 deg) clear of people and objects.
- Balance the gun over the shaft (hose clamp position) before powering; an unbalanced gun loads the shaft sideways.
- The e-stop (A0) latches the firmware and disables the driver within a millisecond; it does **not** cut motor
  power. Keep the 12 V supply switch within reach: that is the real emergency stop.
- Watchdog: while moving, the firmware stops the table if it hears nothing for 1 s. The host sends a heartbeat every
  250 ms from a separate thread; if the host program hangs, crashes or you pull USB, the table decelerates and stops.
- Ctrl-C (or SIGTERM) sends `STOP` (a controlled deceleration, the gun does not slam), waits for rest while still
  heartbeating, then `EN 0`. A second Ctrl-C skips the wait and disables the driver at once. Any exception, e-stop,
  watchdog event, lost telemetry or silent IMU does the same and still writes the run directory (`meta.json`
  `status` says why it ended).
- The rig is slow on purpose: defaults 45 deg/s and 20 deg/s^2 (caps 90 and 60). A 0.4 kg m^2 gun on a ~0.35 N m motor
  loses steps if pushed harder; check with `home_check` (below).

## Install and try it without hardware

```bash
pip install pyserial            # laptop; on the Pi: sudo apt install python3-serial   (--sim needs nothing)
tbot calrig run turns --sim     # whole stack against an in-process fake firmware and synthetic IMU, ~1 s
tbot calrig analyze runs/calrig/*_turns
```

`--sim` skips the UNLOADED prompt (there is nothing to protect) and runs on a virtual clock: a 100 s procedure costs
a second of CPU. The simulated truth can be set with `--sim-opt KEY=VALUE` (repeatable), e.g.
`--sim-opt gyro_scale=0.02,0.01,-0.015` (fractional error per chip axis), `gyro_bias_dps=0.3,-0.2,0.1`,
`imu_offset_ms=12` (IMU stamps late), `mount_err_deg=2,0,0` (mis-mounted IMU on top of the config map),
`r_gun_imu=<9 numbers>`, `start_deg=-12` (where the table sits at power-up), `drift_ppm`, `latency_ms`, `seed`.
The truth is stored in `meta.json` as `sim_truth`, so you can see whether the analysis recovered it.

## Commands

| Command | What it does |
|---|---|
| `tbot calrig info [--port P]` | connect, print firmware `INFO`, geometry, limits, state and the ping round trip; motor stays off |
| `tbot calrig home [--dir +-1] [--rate dps] [--no-return]` | seek the hall edge, set it as zero, move onto it (asks `UNLOADED`) |
| `tbot calrig jog [--deg X \| --to X]` | move by hand; without `--deg/--to` an interactive prompt: `+`, `-`, `+5`, `to 30`, `step 2`, `rate 10`, `home`, `zero`, `stat`, `q` (asks `UNLOADED`) |
| `tbot calrig run <procedure> [--pose P] [-p KEY=VALUE ...] [--no-home] [--no-imu] [--out DIR]` | record a procedure into `runs/calrig/<time>_<procedure>/` (asks `UNLOADED`). `tbot calrig run --list` shows procedures and parameters |
| `tbot calrig estop` | `ESTOP` then `EN 0`. Opening the port also resets an Uno (driver off at boot), but for a real emergency cut the motor power |
| `tbot calrig analyze RUN_DIR [RUN_DIR ...] [--apply]` | analyse one run, or combine several poses; writes `report.md`/`report.json` per run and prints the proposed `calib/imu.json` keys. Dry run unless `--apply` |

Port, baud, geometry, limits, home direction/rate and the run directory root live in `[calrig]` in
`config/default.toml` (override in `config/local.toml` or with `--set calrig.port=/dev/ttyUSB0`).

**Opening the serial port resets the Uno.** Every `tbot calrig` invocation therefore starts with the driver
disabled and the position counter at 0 *wherever the table happens to be*. That is why `run` homes first (zero =
the hall edge) for every motion procedure; `--no-home` takes the power-up position as zero instead.
Between invocations the table is not held: the motor is disabled when the command ends.

## Procedures

| Procedure | Does | Default parameters |
|---|---|---|
| `hold` | stand still with the driver enabled: gyro bias and noise | `duration_s=120` |
| `turns` | 0 -> +A -> -A -> 0 at a fixed rate, repeated, 3.5 s still between moves: gyro scale per axis, axis map | `amplitude_deg=90 rate_dps=30 repeats=2 settle_s=3.5` |
| `sine` | enveloped sinusoid about 0 (first and last cycle ramp the amplitude): time offset | `amplitude_deg=15 period_s=6.5 cycles=6` (peak accel 14 of 20 deg/s^2) |
| `rate` | constant-rate sweeps at several rates over +-80 deg | `rates_dps=10,20,30,45 span_deg=160 repeats=1` |
| `home_check` | home, +-60 deg excursion, home again in the same direction: the second `HOMED` offset is the lost-step count | `excursion_deg=60 tolerance_deg=0.5` |

Everything is validated against the limits **before** the motor is enabled: a sine whose peak acceleration
`A (2 pi / T)^2` exceeds the amax in force, a rate that never reaches a cruise within the sweep, an amplitude beyond
the soft limits and so on are refused with the reason (`tbot calrig run turns -p amplitude_deg=95` fails at once).
Pauses between moves are at least 3 s because the analysis needs a still segment of 2 s after trimming 0.5 s on
each side.

The recorded window starts **after** the initial homing: `HOME` re-zeroes the position counter, a jump that would
corrupt `rig.csv`. The homing result is in `meta.json` under `result.initial_home`. For the same reason
`home_check` (which re-homes inside the recording) is only meaningful for its hall / `HOMED` check; ignore the other
rows of its report.

## Poses

`--pose` records how the gun lies on the rig so every IMU axis sees the yaw rate in turn. The yaw axis is world up:

| `--pose` | Gun on the rig | World-up in the gun frame (+X muzzle, +Y left, +Z up) |
|---|---|---|
| `upright` | rib up, normal | +Z |
| `left_side_down` | rolled 90 deg onto its left side | -Y |
| `right_side_down` | rolled 90 deg onto its right side | +Y |
| `inverted` | upside down | -Z |

`muzzle_up` / `muzzle_down` are also accepted (the analysis knows them) but are not reachable on the standard saddle.
Re-seat and re-strap the gun for every pose; the hose clamp position must stay where it balances.

## Calibration session, step by step

All with the gun **unloaded**, base clamped, arc clear. Replace the port if it is not `/dev/ttyACM0`.

1. **Check the link and the geometry.** `tbot calrig info`. Expect `usteps=16 steps_rev=200`, state `IDLE en=0`,
   `estop_pin=0`, a ping round trip of 1-3 ms and a fit residual of a few hundred microseconds. A mismatch between
   firmware and `[calrig]` stops here: fix the jumpers / `MICROSTEPS` or the config.
2. **Direction and home magnet.** With the gun mounted upright and balanced, `tbot calrig jog --deg 10`: the muzzle
   must swing **counter-clockwise seen from above** (positive). If it goes the other way, flip `DIR_INVERT` in the
   firmware. Rotate by hand until the home magnet is within +-90 deg of where the table sits at power-up (the seek is
   limited to the soft limits), then `tbot calrig home`: it moves at 5 deg/s to the hall edge, zeroes there and
   returns onto it. If the sensor is already active it backs off 10 deg first. The edge approached from the + side is
   one edge of the magnet window, from the - side the other (`--dir -1`): keep the same direction for the whole
   session (`calrig.home_dir`).
3. **Lost-step check.** `tbot calrig run home_check` — pass: `result.lost_deg` within 0.5 deg (`ok: true`). More means
   the motor loses steps: lower `calrig.amax_dps2` / `vmax_dps`, raise the driver current, PTFE-tape the flange.
4. **Bias.** Gun upright: `tbot calrig run hold` (120 s, do not touch anything, let the IMU warm up first).
5. **Turns in all four poses.** For each of `upright`, `left_side_down`, `right_side_down`, `inverted`: seat the gun,
   `tbot calrig run turns --pose <pose>` (about a minute after the homing: +-90 deg at 30 deg/s, twice, 3.5 s still between moves).
6. **Sine.** Gun upright: `tbot calrig run sine` (time offset and dynamics).
7. **Analyse.** `tbot calrig analyze runs/calrig/*_turns` combines the four poses (axis map, per-axis scale, bias,
   time offset) and prints the proposed `calib/imu.json` keys; add `runs/calrig/*_sine` and the `hold` directory
   separately. Read each `report.md`: every number has a sigma and a PASS/FAIL against a stated limit.
8. **Apply** (only after reading the reports):
   ```bash
   tbot calrig analyze runs/calrig/*_turns --apply     # R_gun_imu, gyro_scale, bias, time offset
   tbot calrig analyze runs/calrig/*_sine --apply      # time offset from the sine
   tbot calrig analyze runs/calrig/*_hold --apply      # last: the long hold's bias wins
   ```
   `--apply` merges only keys the analysis considered trustworthy into `calib/imu.json` (`update_json`), so each
   call only touches what it measured; without `--apply` nothing is written.

### What the results mean for the main software

- `gyro_bias_rad_s`: CHIP-frame bias from the stills; replaces the one from `tbot calib-imu bias`.
- `gyro_scale`: per chip axis multiplier, new in `calib/imu.json`; the IMU applies `(raw - bias) * gyro_scale`
  in the chip frame, then `R_gun_imu`. Absent = `[1,1,1]`, nothing changes for existing calibrations.
- `R_gun_imu`: needs >= 2 poses. If it differs from the config map by more than a mount tolerance, the pod is not
  mounted as `design/layout.md` says; do not apply a matrix that is not a signed permutation.
- `time_offset_ns`: the rig measures how late the IMU **timestamps** are against the true table motion; the value
  is added to IMU timestamps. `tbot calib-imu sync` measures the camera-IMU offset (which already contains it): use
  the rig value as an independent check of that number rather than stacking both.

## Clocks

The IMU stamps samples with `time.monotonic_ns()` (`tbot.types.MonotonicClock`, used by `Mpu6050`); the rig host
converts firmware `micros()` to the same clock with a least-squares line over PING round trips (before, during at
1 Hz, and after each run; lowest-RTT half kept). The fit (`offset_ns`, `drift_ppm`, `resid_us`, `n`) is in
`meta.json` under `clock_fit`; a residual above ~1 ms means a bad link (long USB hubs, a busy Pi): use a short cable.
On the Pi the IMU is read raw (`ImuCalibration` neutral): `imu.csv` holds chip-frame values before `R_gun_imu`, bias,
scale and time offset, which the analysis must see; the main app's `Mpu6050` behaviour is unchanged.

## Run directory

`runs/calrig/<YYYYmmdd-HHMMSS>_<procedure>/` with `meta.json`, `rig.csv`, `cmd.csv`, `hall.csv`, `events.csv`,
`imu.csv`, plus `report.md` / `report.json` after `analyze` — exactly as specified in
[CALRIG_PROTOCOL.md](CALRIG_PROTOCOL.md) section 4. `runs/` is not tracked by git; copy runs off the Pi with rsync
(`rsync -a pi@tbot.local:~/TBOT/software/runs/ runs/`).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `no answer from the rig firmware` | wrong `--port`, firmware not flashed, another program (serial monitor) holds the port, user not in the `dialout` group |
| `firmware usteps=... but config says ...` | microstep jumpers / `MICROSTEPS` and `[calrig] microsteps` disagree |
| `homing failed (... LIMIT)` | the magnet is more than 90 deg from the power-up position in the seek direction: hand-rotate the gun closer, or use `--dir -1` |
| `the e-stop input (A0) is asserted` | button pressed, wire broken, or no jumper A0-GND |
| `no telemetry for 1 s` / `no IMU samples` | link or firmware stalled / IMU I2C trouble (`tbot hwcheck`); the table was stopped |
| `motion ended by the firmware: WATCHDOG` | the host stalled for > 1 s (heavy CPU load on the Pi) |
| Table shakes / stalls at 30 deg/s | amax too high for the gun's inertia: lower `calrig.amax_dps2`, check Vref and the balance |
