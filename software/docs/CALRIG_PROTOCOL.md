# CalRig B software contract (firmware <-> host <-> analysis)

Single source of truth for the interfaces between the Arduino firmware (`software/firmware/calrig_uno/`),
the host driver/recorder (`software/tbot/calrig/`) and the analysis (`software/tbot/calrig/analysis.py`).
Mechanical numbers come from `design/calibration_rig.md` section 7; change that first.

## 1. Hardware

- Arduino Uno R3 + CNC Shield V3 (GRBL pin-out). The yaw motor is on the **X** driver socket.
- Driver: A4988 (default), DRV8825 or TMC2209 standalone in the Pololu footprint. Microstep jumpers set the
  microstep count; the firmware constant `MICROSTEPS` (default 16) must match and is reported by `INFO`.
- Motor: the user's NEMA 17 (1.8 deg, 200 full steps/rev, short 37 mm body, factory pinion kept on the shaft).
  Direct drive: the yaw arm clamps the pinion, ratio 1:1.
  One table degree = 200 * MICROSTEPS / 360 microsteps (16 -> 8.888.. usteps/deg, 0.1125 deg/ustep).

| Function | Uno pin | Shield label | Notes |
|---|---|---|---|
| STEP | D2 | X.STEP | |
| DIR | D5 | X.DIR | `DIR_INVERT` compile option; + steps = yaw CCW seen from above |
| ENABLE | D8 | EN | active LOW, shared by all sockets |
| Home sensor | D9 | X- / X+ endstop | A3144 (or equivalent unipolar open-collector) hall switch, INPUT_PULLUP, LOW = magnet present |
| E-stop | A0 | Abort | normally-closed button to GND, INPUT_PULLUP; HIGH = e-stop (broken wire also stops). Jumper A0-GND to run without a button (not recommended) |
| Hall 5 V / GND | 5V / GND | any 5V/GND header pin | |

Motor power: 12 V on the shield screw terminal. Set the driver current limit (Vref) to the motor's rated current
before connecting the motor.

## 2. Serial protocol

USB serial, 115200 8N1, ASCII lines terminated `\n` (firmware accepts `\r\n`). Host -> firmware: commands.
Firmware -> host: one reply per command (`OK <CMD> [key=value ...]` or `ERR <CMD> <code> <text>`) plus
asynchronous lines that start with a single upper-case letter and a space. Integers are decimal; `t_us` is the
firmware `micros()` (uint32, wraps every 71.6 min; the host unwraps it). Positions are int32 microsteps,
velocities microsteps/s, accelerations microsteps/s^2.

### Commands

| Command | Reply | Behaviour |
|---|---|---|
| `PING <seq>` | `PONG <seq> <t_us>` | `t_us` sampled while handling the command; used for clock sync |
| `INFO` | `OK INFO fw=calrig-uno/<ver> usteps=<n> steps_rev=200 ratio=1 vmax_cap=<sps> amax_cap=<sps2> hard_lim=<steps> telem_hz=<hz>` | |
| `EN <0/1>` | `OK EN <0/1>` | enable/disable the driver. Refused (`ERR EN ESTOP`) while e-stopped |
| `ESTOP` | `OK ESTOP` | immediate driver disable, motion cleared, latched |
| `RESET` | `OK RESET` / `ERR RESET ESTOP_PIN` | clears the latch only if the A0 input is not asserted; driver stays disabled |
| `CFG <vmax> <amax>` | `OK CFG vmax=<v> amax=<a>` | clamps to the compile-time caps, replies with the values in force |
| `LIM <min> <max>` | `OK LIM min=<a> max=<b>` | soft limits in steps, clamped inside the hard limit (+-100 deg) |
| `ZERO` | `OK ZERO` / `ERR ZERO MOVING` | position := 0, only when stopped |
| `MOVE <target>` | `OK MOVE` | trapezoidal move to absolute target (clamped to soft limits -> `ERR MOVE LIMIT`) |
| `VEL <v>` | `OK VEL` | ramped constant-velocity jog; decelerates to stop at the soft limit (`E LIMIT`) |
| `SINE <amp> <period_ms> <cycles>` | `OK SINE` / `ERR SINE ACCEL` | sinusoid about the current position, amplitude ramped in over the first cycle and out over the last; refused if `amp*(2*pi/period)^2 > amax` or the swing exceeds the soft limits |
| `HOME <dir> <v>` | `OK HOME` | seek the hall edge at `<v>` in direction `dir` (+1/-1); on the edge where the sensor becomes active, latch the position, stop, then set that edge = 0; emits `E HOMED <t_us> <offset>` where `offset` is the step count at the edge before re-zeroing (lost-step check) |
| `STOP` | `OK STOP` | controlled decel at amax |
| `STREAM <0/1>` | `OK STREAM <0/1>` | telemetry on/off |
| `HB` | `OK HB` | heartbeat; any received line also counts |
| `STAT` | `OK STAT t=<t_us> pos=<p> vel=<v> state=<IDLE/MOVE/VEL/SINE/HOME/ESTOP> en=<0/1> home=<0/1> estop_pin=<0/1>` | |

Any motion command while e-stopped or disabled -> `ERR <CMD> NOT_READY`. A new motion command while moving
replaces the current motion with a smooth transition (no velocity step).

Firmware details the host may rely on (clarifications, formats above unchanged):
- Error codes beyond the table: `ERR <CMD> ARG` (bad/missing argument), `ERR ? UNKNOWN` (unknown command),
  `ERR ? OVERFLOW` (line > 48 characters, discarded), `ERR MOVE LIMIT` (target outside the soft limits, nothing
  moves), `ERR SINE SPEED` (`amp*2*pi/period > vmax`), `ERR SINE LIMIT` (`pos +- amp` outside the soft limits),
  `ERR HOME ACTIVE` (sensor already active at the start: move off the magnet first).
- `PING <seq>`: `seq` is echoed verbatim (digits only); `t_us` is stamped when the line's `\n` is received.
- Every motion ends with exactly one terminal event: `DONE` (MOVE reached its target exactly, VEL 0 / STOP /
  SINE finished, also after `WATCHDOG`), `LIMIT` (VEL/HOME/SINE stopped at a limit), `HOMED` then `DONE` (HOME),
  or `ESTOP`. A replaced motion emits no terminal event of its own.
- `E HOMED <t_us> <offset>`: `t_us` is the capture time of the hall edge. The edge is re-zeroed once the table has
  stopped (`HOMED` is sent after the deceleration, followed by `DONE`); the position then jumps by `-offset`.
  The seek is bounded by the +-100 deg hard limit around the current position (`E LIMIT` if no edge found).
- `E ESTOP` carries `PIN` or `CMD`. `EN 0` or an e-stop mid-motion cancels the motion without a terminal event.
- SINE: if the table is moving the firmware first brakes to rest and uses that rest position as the centre. The
  envelope ramps can push the peak acceleration to at most 1.125 x `amp*(2*pi/period)^2` (the planner never
  commands more than 1.25 x `amax`, nor more than `AMAX_CAP`).

### Asynchronous lines

| Line | Meaning |
|---|---|
| `T <t_us> <pos> <vel>` | telemetry at `telem_hz` (default 200 Hz) while streaming; `t_us` and `pos` sampled coherently (interrupts off) |
| `H <t_us> <pos> <level>` | home-sensor edge captured in the pin-change ISR; `level` 1 = became active (magnet present) |
| `E <t_us> <code> [arg]` | events: `DONE` (motion finished), `LIMIT`, `HOMED <offset>`, `ESTOP` (pin or command), `WATCHDOG`, `BOOT` |

### Safety rules (firmware)

- Boots with the driver disabled and emits `E <t> BOOT`.
- E-stop pin is polled every control tick (>= 1 kHz) and latches `ESTOP`: driver disabled within 1 ms.
- Watchdog: while any motion is active, no received line for 1000 ms -> controlled `STOP` and `E WATCHDOG`.
- Hard limit +-100 deg is compiled in; soft limits default +-90 deg; any step that would cross a soft limit is
  not issued.
- Compile-time caps: `VMAX_CAP` 90 deg/s, `AMAX_CAP` 60 deg/s^2. Defaults: 45 deg/s, 20 deg/s^2.
  The gun on a direct-drive NEMA 17 has about 70 000x the rotor inertia; long ramps are mandatory.

## 3. Clock sync

The host converts firmware `t_us` to host `time.monotonic_ns()` (the clock used by `ImuSample.t_ns` on the Pi)
with a least-squares line (offset + drift) over `PING` round trips, using the round-trip midpoint and keeping
the lowest-RTT 50 % of samples. Pings run before, during (1 Hz) and after each run. The fit and its residual go
in `meta.json`.

## 4. Run recording (host -> analysis)

One directory per run: `runs/calrig/<YYYYmmdd-HHMMSS>_<procedure>/`

| File | Columns / content |
|---|---|
| `meta.json` | procedure name and parameters, firmware INFO, `deg_per_ustep`, sign convention, clock fit under the key `clock_fit` = `{offset_ns, drift_ppm, resid_us, n}` (`host_ns = offset_ns + (1 + drift_ppm*1e-6) * unwrapped_t_us * 1000`), IMU config (sample rate, ranges, `R_gun_imu` in use) under the key `imu`, `pose` (`upright`, `left_side_down`, `right_side_down`, `inverted`; the analysis also accepts `muzzle_up`, `muzzle_down`), host, software version, `start`/`end` ISO times, operator confirmation that the gun is unloaded |
| `rig.csv` | `t_ns,t_us,pos_ustep,angle_deg,vel_deg_s` (from `T` lines, t_ns = host clock) |
| `cmd.csv` | `t_ns,command` every command sent |
| `hall.csv` | `t_ns,t_us,pos_ustep,angle_deg,level` |
| `events.csv` | `t_ns,t_us,code,arg` |
| `imu.csv` | `t_ns,gx,gy,gz,ax,ay,az,temp_c` RAW chip-frame values: gyro rad/s, accel m/s^2, BEFORE `R_gun_imu` and bias are applied (the analysis must see the raw axes) |

`t_us` columns hold the raw uint32 firmware value (the readers unwrap). The recorded window starts after the initial
homing (HOME re-zeroes the position counter, a jump that would corrupt `rig.csv`); `cmd.csv` still lists every
command, including the homing ones, verbatim.

Yaw sign: table angle positive = counter-clockwise seen from above (right-hand about +Z up).
