# CalRig B firmware (Arduino Uno + CNC Shield V3)

Motorised yaw table that swings an **unloaded** gun so the gun-mounted MPU-6050 can be calibrated (gyro
bias/scale, axis map, IMU time offset, lost-step check). Interface contract (pins, serial protocol, clock sync,
run recording): [`../../docs/CALRIG_PROTOCOL.md`](../../docs/CALRIG_PROTOCOL.md). Host side: `software/tbot/calrig/`.

This firmware never touches any firing/trigger actuator code and must never be wired to one.

## Files

| File | Purpose |
|---|---|
| `calrig_uno.ino` | the whole firmware; compile-time options are the block at the top |
| `platformio.ini` | `uno` env, `src_dir = .` so `pio run -d software/firmware/calrig_uno` builds it |
| `test/` | host-side (PC) simulation harness with a tiny Arduino stub; not part of the firmware build |

## Wiring

| Function | Uno pin | CNC shield label | Notes |
|---|---|---|---|
| STEP | D2 | X.STEP | |
| DIR | D5 | X.DIR | `DIR_INVERT`: + steps = yaw CCW seen from above |
| ENABLE | D8 | EN | active LOW, shared by all sockets; pulled HIGH (driver off) at boot |
| Home sensor | D9 | X- (or X+) endstop signal | A3144 hall switch, `INPUT_PULLUP`, LOW = magnet present. Hall VCC -> 5 V, GND -> GND, OUT -> D9 |
| E-stop | A0 | Abort | normally-closed button to GND, `INPUT_PULLUP`; HIGH = stop, a broken wire also stops. `ESTOP_NC 0` for a normally-open button. To run without a button jumper A0-GND (not recommended) |
| Motor | | X driver socket, 2B 2A 1A 1B | NEMA 17, direct drive, 1:1 |
| Motor power | | shield screw terminal | 12 V. Set Vref **before** connecting the motor |

Rules of thumb: driver in the **X** socket only; put a 100 uF electrolytic across the 12 V terminals (the shield
has pads); never hot-plug the motor. The CNC shield microstep jumpers are labelled M0/M1/M2 (= MS1/MS2/MS3).

## Driver current limit (Vref) and microstep jumpers

`MICROSTEPS` in the sketch (default **16**) must match the jumpers; `INFO` reports it and the host derives
`deg/ustep = 360 / (200 * MICROSTEPS)` from it. Measure Vref between the driver's trimpot wiper (screw) and GND
with the board powered and the motor **disconnected**. Start at 60-70 % of the motor's rated phase current: the
rig is low-duty and a cooler motor is better than a missed step. Heat-sink the driver; the shield has no fan.

| Driver | Current formula (Vref in V) | 16 microsteps | 32 microsteps |
|---|---|---|---|
| **A4988** | `I_max = Vref / (8 * Rs)`; Rs = 0.068 ohm (Pololu) or 0.1 / 0.05 ohm on clones (read the sense resistor: R100 = 0.1, R050 = 0.05) | M0 M1 M2 all jumpered (H H H) | not supported (max 1/16) |
| **DRV8825** | `I_max = 2 * Vref` (Rs = 0.1 ohm); with the 0.2 ohm clones `I_max = Vref` | only M2 jumpered (L L H) | M0+M2 or M1+M2 or all three (any H on M2 with M0/M1 not both low) -> 1/32; set `-DMICROSTEPS=32` |
| **TMC2209** standalone (BTT / SilentStepStick style, 0.11 ohm sense) | `I_rms ~= 0.71 * Vref` (check your module's datasheet, sense resistors differ) | M0 + M1 jumpered (MS1=MS2=H) -> 1/16 step input, the chip interpolates to 1/256 | MS1=H, MS2=L -> 1/32; set `-DMICROSTEPS=32` |

Notes: A4988 full table (MS1 MS2 MS3): LLL full, HLL 1/2, LHL 1/4, HHL 1/8, HHH 1/16. DRV8825 (M0 M1 M2): LLL full,
HLL 1/2, LHL 1/4, HHL 1/8, LLH 1/16, others 1/32. TMC2209 (MS2 MS1): LL 1/8, HL 1/64, LH 1/32, HH 1/16. TMC2209 StealthChop
modules are quiet: do not rely on the sound to tell whether the motor is energised. Whatever the driver, the
firmware's maximum step rate is 1600 steps/s at the 90 deg/s cap with 32 microsteps; the 20 kHz DDA supports up
to 5 ksteps/s.

## Build and flash

Arduino IDE (1.8 / 2.x): open `calrig_uno.ino` (folder name = sketch name), board **Arduino Uno**, pick the port,
Upload. No libraries needed. To change options without editing the sketch use the top-of-file `#define`s.

PlatformIO:

```
pio run -d software/firmware/calrig_uno                    # build
pio run -d software/firmware/calrig_uno -t upload          # flash   (--upload-port /dev/ttyACM0 if autodetect fails)
pio device monitor -d software/firmware/calrig_uno         # 115200 8N1
```

Compile-time options (top of `calrig_uno.ino`, overridable with `build_flags = -D...` in `platformio.ini`):
`MICROSTEPS` (16), `DIR_INVERT` (0), `ESTOP_NC` (1), `VMAX_CAP_DEG` (90), `AMAX_CAP_DEG` (60), `VMAX_DEF_DEG` (45),
`AMAX_DEF_DEG` (20), `TELEM_HZ` (200, must divide 1000), `WATCHDOG_MS` (1000), `HARD_LIM_DEG` (100),
`SOFT_LIM_DEG` (90). Deg values are converted to microsteps at compile time.

First power-up checklist: (1) motor disconnected, set Vref; (2) connect motor, **gun not mounted**; (3) `INFO`
shows the right `usteps`; (4) `EN 1`, `MOVE 100` and check the direction: + must be counter-clockwise seen
from above, otherwise set `DIR_INVERT 1` (or swap one coil pair); (5) press the e-stop during a `VEL` jog.

## Serial quick reference (115200 8N1, `\n` terminated, integers; positions in microsteps)

| Command | Reply | |
|---|---|---|
| `PING <seq>` | `PONG <seq> <t_us>` | clock sync |
| `INFO` | `OK INFO fw=calrig-uno/1.0 usteps=16 steps_rev=200 ratio=1 vmax_cap=<sps> amax_cap=<sps2> hard_lim=<steps> telem_hz=200` | |
| `EN <0/1>` | `OK EN <n>` / `ERR EN ESTOP` | driver on/off (boots off) |
| `ESTOP` / `RESET` | `OK ESTOP` / `OK RESET`, `ERR RESET ESTOP_PIN` | latch / clear (driver stays off, `EN 1` again) |
| `CFG <vmax> <amax>` | `OK CFG vmax= amax=` | clamped to caps (steps/s, steps/s^2) |
| `LIM <min> <max>` | `OK LIM min= max=` | soft limits, clamped inside +-100 deg |
| `ZERO` | `OK ZERO` / `ERR ZERO MOVING` | position := 0 |
| `MOVE <target>` | `OK MOVE` / `ERR MOVE LIMIT` | trapezoid, lands exactly on target |
| `VEL <v>` | `OK VEL` | ramped jog, stops at soft limit (`E LIMIT`) |
| `SINE <amp> <period_ms> <cycles>` | `OK SINE` / `ERR SINE ACCEL|SPEED|LIMIT` | enveloped sine about the current position |
| `HOME <dir> <v>` | `OK HOME` / `ERR HOME ACTIVE` | seek hall edge, `E HOMED <t_us> <offset>` |
| `STOP`, `HB`, `STREAM <0/1>`, `STAT` | `OK ...` | `STAT`: `t pos vel state en home estop_pin` |

Async lines: `T <t_us> <pos> <vel>` (telemetry), `H <t_us> <pos> <level>` (hall edge), `E <t_us> <code> [arg]`
(`BOOT DONE LIMIT HOMED ESTOP WATCHDOG`). Any motion command while e-stopped or disabled -> `ERR <CMD> NOT_READY`.
Error code details and the terminal-event rules are in the protocol document.

Typical session: `EN 1`, `STREAM 1`, `HB` at >= 2 Hz while moving, `MOVE 800`, wait for `E ... DONE`.

## Safety rules and how they are implemented

- **Boots disabled**: `setup()` writes EN HIGH before configuring anything else; the first output is `E <t> BOOT`.
  If the e-stop input is asserted at boot the latch is set immediately.
- **E-stop**: A0 is read (`PINC`) on every `loop()` pass and again at the start of every 1 kHz control tick. On
  assertion the EN pin is driven HIGH first, then the DDA increment is zeroed, the motion is dropped and `ESTOP`
  latches (`E <t> ESTOP PIN`). `EN 1` and all motions are refused until `RESET`, which itself is refused while the pin
  is still asserted; the driver remains disabled after `RESET`. Worst-case latency is one `loop()` pass
  (sub-millisecond; the longest tick is a few hundred us of float maths).
- **Watchdog**: while a motion is active, no received line for `WATCHDOG_MS` -> controlled `STOP` and `E WATCHDOG`
  (then `E DONE`). Any complete line, including `HB`, feeds it; the host should send `HB` at >= 2 Hz.
- **Limits**: hard +-100 deg is compiled in, soft limits default +-90 deg and are clamped inside the hard limit.
  They are enforced twice: the planner brakes to a stop at the limit (`E LIMIT`) and, as a last line, the step ISR
  refuses to issue any step that would cross the effective limit (hard while seeking home, else soft).
  `MOVE` outside the soft limits is refused, `SINE` is refused if the swing leaves them.
- **No velocity steps**: every motion mode only sets a desired velocity; one slew limiter (<= `amax`) turns it into
  the commanded velocity, so a new command while moving blends smoothly. Caps `VMAX_CAP_DEG`/`AMAX_CAP_DEG` clamp
  `CFG`. The gun inertia is ~70 000x the rotor's: keep `amax` low.
- **Serial robustness**: lines are parsed non-blockingly into a 48 byte buffer; longer lines are discarded and
  answered with `ERR ? OVERFLOW`. Output goes through a 512 byte ring drained into the UART only when it has room,
  so a slow host can never stall the control loop (telemetry is dropped first, replies/events are kept).
- The rig never touches firing or actuator code.

## Implementation notes

- Step generation: Timer1 CTC interrupt at 20 kHz, 32-bit phase accumulator (DDA). STEP is raised on the carry tick
  and lowered on the next tick (50 us pulse); a DIR change waits one tick before stepping. The ISR is ~100-160 cycles
  including register save (6-10 us at 16 MHz) and holds no loop-dependent state. Max step rate allowed at
  compile time is `ISR_HZ/4` = 5 ksteps/s, far above `VMAX_CAP` at 1/16 or 1/32.
- Position is an `int32_t` changed only in the ISR (plus ZERO / HOME re-zero with interrupts off); reads for `T`/`STAT`
  take `pos`, the DDA phase and `micros()` together with interrupts off.
- Home: PCINT0 on D9 captures `micros()` and the position in the ISR and queues an `H` line; during `HOME` the first
  "became active" edge also latches the position, the table brakes at `amax` and the edge is re-zeroed after standstill.
- Trapezoid `MOVE` uses `v = sqrt(2*0.9*amax*remaining)` from the continuous position estimate and finishes with at
  most a few single-step trims, so `DONE` means `pos == target`.
- SINE evaluates `A*env(t)*sin(2*pi*t/T)` directly each tick (raised-cosine envelope over the first and the last
  cycle) and finishes with a `MOVE` back to the exact centre.

## Known limitations

- Limits are relative to the current zero: `ZERO`/`HOME` move them with it. The `HOME` seek is bounded by the hard
  limit around the *current* position.
- Open loop: no step-loss detection other than the `HOME` offset. A stalled motor is not noticed by the firmware.
- `ESTOP_NC 1` treats a missing button as a stop (fail-safe); jumper A0 to GND when testing without a button.
- Timing resolution of the step edges is 50 us (20 kHz DDA); at 800 steps/s this is +-4 % of a step period, which
  the motor and the gun inertia filter.

## Host-side simulation harness (optional, PC only)

`test/sim_host.cpp` compiles the sketch against a tiny Arduino stub (`test/stub/`) and runs scenarios (protocol,
trapezoid/replace/exact landing, VEL limit, SINE, HOME, e-stop latency, watchdog, telemetry rate) with a simulated
20 kHz ISR, hall magnet and e-stop input:

```
cd software/firmware/calrig_uno/test
g++ -std=c++17 -O1 -Istub sim_host.cpp -o /tmp/sim_host && /tmp/sim_host      # exit code 0 = all scenarios OK
```
