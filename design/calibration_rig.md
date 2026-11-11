# IMU / swing calibration rig (CalRig)

Status: candidate B rev B2 built in CAD; software written (section 7), not yet run on hardware.
Owner of the mechanical side: Hunter. Owner of the software side: Alex (`software/`).
Units mm unless stated. Rig frame: origin on the turntable axis at the top face of the base plate, +Z up,
+X toward the motor, yaw angle positive counter-clockwise seen from above.

## 1. Purpose

TBOT's tracker extrapolates the bore from IMU attitude and swing rate over the actuator delay
(`software/docs/CALIBRATION.md` step 4 and step 6). A wrong gyro scale, a wrong axis map or a wrong time offset
becomes a lead error proportional to swing rate, and the IMU is the one sensor that is never cross-checked on the
range. CalRig is a motorised yaw table that swings the **whole unloaded gun** (pod fitted) through angles and rates
the software commands, while a second, independent angle reference reports what the gun really did.

It exists to answer four questions:

1. **Is the IMU right?** Gyro bias, gyro scale factor (deg of IMU vs deg of table), axis map and sign, and the
   gravity vector / accelerometer scale.
2. **Is the attitude estimate right?** Command a known angle, compare the filtered yaw the TBOT software reports
   against the table angle. Drift over a long hold and after a fast swing.
3. **How late is it?** IMU-to-truth time offset, and the table's own command-to-motion delay, so the camera-IMU
   offset in `calib/imu.json` (`time_offset_ns`) has an independent check rather than only the optical-flow estimate.
4. **Can the tracker follow a controlled swing?** Constant-rate and sinusoidal sweeps at clay-like angular rates,
   repeatable run after run, with the gun resting in the same place every time.

Non-goals: the rig never fires, never carries ammunition, and does not drive the trigger actuator. Servo and trigger
work stays out of this document. The gun is **unloaded, bolt open, muzzle in a safe direction** for every run.

## 2. Requirements

### 2.1 Functional

| ID | Requirement | Target |
|---|---|---|
| F1 | Yaw the gun about a vertical axis | usable +-90 deg, hard stops at +-100 deg |
| F2 | Commanded angle resolution at the table | <= 0.02 deg per microstep |
| F3 | Peak swing rate | >= 90 deg/s (clay crossing at 20 m/s, 20 m is 57 deg/s; margin above that) |
| F4 | Acceleration | >= 200 deg/s^2 without lost steps with the gun on |
| F5 | Independent truth angle, not derived from step counts | absolute encoder on the table axle, <= 0.1 deg resolution |
| F6 | Hardware timing mark at a fixed angle (home flag) | one edge per crossing, repeatable, usable as a time reference |
| F7 | Repeatable zero | optical home plus hard stops; zero recovered within the encoder resolution |
| F8 | Gun sits in the same place every run | V saddles with straps, saddle positions on a scaled rail |
| F9 | Gun can be balanced over the axis | saddles slide along a 2020 rail to put the centre of mass over the axle |
| F10 | Gun can lie upright or on its side (roll 90 deg) so every IMU axis sees the yaw rate in turn | saddles are V blocks, strap slots on both faces |
| F11 | Pod stays unobstructed | saddles sit clear of the forend-nose clamp (gun X 431..515) and the pod's carriers |

### 2.2 Mechanical

- Printed on an Ender 3 (220 x 220 bed), PLA+, 0.2 mm layers, M3 heat-set inserts for plastic-to-plastic joints,
  M2 only where a board needs it. Every part prints without support inside circular holes. Prefer parts that print
  flat with no overhang.
- The motor is the user's NEMA 17 with its factory pinion; the reference model is built from caliper
  measurements (`refs/stepper_nema17_pinion.step+`). The pinion tooth count is **not yet confirmed**, so the
  gear stage has an adjustable centre distance (slotted motor holes).
- Axle: M8 bolt through two 608 bearings; the table is clamped to the inner races only. Load path is the axle,
  not the motor.
- Base is bolted or clamped to a bench; it does not rely on its own mass to resist the swinging gun.
- Mechanical hard stops at +-100 deg so a software fault cannot spin the gun into the base or the motor.
- No cable-management or zip-tie features (harnessing is handled separately).
- Gun contact surfaces are compliant (adhesive EPDM or felt on the saddles). The rig never grips the barrel or the pod.

### 2.3 Sensors and electronics (mounting provisions only)

| Item | Purpose | Mount |
|---|---|---|
| AS5600 magnetic encoder breakout + 6 x 2.5 diametric magnet | truth angle, I2C | under the gear, centred on the axle, 2 mm gap to the magnet bonded on the M8 bolt tip |
| Slot-type photo-interrupter module (FC-03 style) | home / timing mark | bracket on the base plate, flag tab on the table rim |
| Stepper driver (TMC2209 or similar) + controller | motion | off the rig, 4-wire cable to the motor |

Which controller generates the step pulses (a Pico/ESP32 next to the rig vs the Pi) is a software decision; see 3.

## 3. Software that needs to be written (not started)

None of this is part of the mechanical task. It is listed so the mechanical design and the software stay consistent.

1. **Motor control**: step/dir generation with trapezoidal and S-curve ramps, constant-rate sweeps, sinusoidal
   sweeps, move-to-angle, home-then-zero. Microstepping and gear ratio come from `design/calibration_rig.md`
   (table below), not hard-coded. Preferably a microcontroller with hardware timers, so step timing does not
   depend on Linux scheduling.
2. **Truth angle**: AS5600 reader with unwrap and offset, timestamped on the same clock as the IMU samples.
3. **Home / timing mark**: interrupt-driven edge timestamps from the photo-interrupter.
4. **Run recorder**: a single log per run holding commanded angle, encoder angle, home-flag edges, IMU raw gyro and
   accel, fused attitude, camera frame timestamps.
5. **Analysis** (writes `calib/imu.json`, `calib/latency.json` fields; read-only to the rest of the pipeline):
   - gyro bias from a long hold;
   - gyro scale per axis from a known angle (table angle vs integrated gyro);
   - axis map from the 4 lie positions (F10), compared with `R_gun_imu` in `design/layout.md`;
   - IMU-to-encoder time offset from a sinusoidal sweep (cross-correlation), cross-checked with the home-flag edge
     times, and compared with the optical-flow offset from `tbot calib-imu sync`;
   - command-to-motion delay of the table itself (step command edge to first encoder change);
   - fused yaw error versus encoder over constant-rate and sinusoidal runs, and drift over a hold;
   - backlash and gear-ratio check: encoder degrees per commanded degree, both directions.
6. **Report**: one pass/fail summary per run, with the limits taken from this document.
7. **Safety layer**: refuse to run unless the user confirms "unloaded"; honour the end stops (software soft limits
   inside +-90 deg); stop on loss of encoder readings; a keyboard / button stop that disables the driver. No firing code
   path is reachable from the rig tools.

### Gear and step numbers the software needs (single source of truth)

| Quantity | Value | Basis |
|---|---|---|
| Pinion teeth z1 / module | 19 / 0.8 | **ASSUMED** from measured OD 16.69 (OD = m(z+2)); count the teeth and confirm |
| Driven gear teeth z2 | 114 | design choice |
| Ratio | 6.00 : 1 | z2 / z1 |
| Centre distance | 53.2 + 0.4 backlash allowance = 53.6 | m (z1+z2)/2 |
| Motor step angle | 1.8 deg (200 steps/rev) | **ASSUMED**, check the label; 0.9 deg doubles resolution |
| Table angle per full step | 0.30 deg | 1.8 / 6 |
| Table angle per 1/16 microstep | 0.01875 deg | |
| Encoder resolution | 0.088 deg | AS5600 12 bit |

## 4. Candidate A: gear-reduced yaw table (parked, over-built for a first rig)

Kept as a potential candidate. Candidate B (simple, direct drive, 2 printed parts) is in section 7. Everything below
in section 4 describes candidate A only; its assembly is `asm/calrig_A.step+`.

CAD: parts in `parts/calrig/`, motor reference in `refs/stepper_nema17_pinion.step+`, hardware references in
`refs/hw/` (M8 x 70 bolt, M8 nut, 608 bearing, 500 mm 2020 extrusion), assembly `asm/calrig.step+`. Rig frame:
origin on the axle axis at the top face of the base plate, +Z up, +X toward the motor.

### 4.1 Concept

A single-axis yaw table. The gun lies along a 500 mm 2020 rail on two V saddles; the rail bolts to a printed
turntable that spins on an M8 axle in two 608 bearings. The stepper sits in a pocket in the base plate with its shaft
pointing down, and drives a 114-tooth printed spur gear under the base plate (6:1). The gear, the axle and the table
are one rotating unit (the bolt head locks into a hex pocket in the table, the M8 nut into a hex trap in the gear).
An AS5600 under the axle reads a magnet bonded on the bolt tip. A slot-type interrupter on the base plate reads a
flag blade hanging from the table. Two stop posts and a lug on the table limit travel to +-100 deg.

Why the motor hangs the "wrong" way: the motor face is 3 mm above the pinion teeth, so a plate-top mount would put
the teeth inside the plate. Recessing the motor 7 mm into a pocket (3 mm floor) lets the pinion teeth sit just under
the plate, so the gear runs under the plate, the plate stays 10 mm thick and stiff, and every printed part prints
flat with no overhang other than nut-trap ceilings.

### 4.2 Z stack (rig frame, mm)

| Item | Z |
|---|---|
| Bench plane (underside of feet) | -45 |
| Encoder PCB chip face | -32 (magnet -30..-27.5 on the M8 tip) |
| Gear (teeth and disc) | -18 .. -10.5, hub boss D17 up to -3 |
| Base plate | -10 .. 0 (172 x 164, R18 corners); motor pocket floor at -7 |
| Motor | face at -7, body to +30.15, shaft and pinion pointing down |
| Lower / upper 608 | -3 .. 4 / 27 .. 34 (23 mm spacer between them) |
| Hub tube (base) | 0 .. 34, OD 34 |
| Table plate | 40 .. 48, R50 (neck D26 36..40, bearing pin D14 34..36) |
| 2020 rail | 48 .. 68 |
| Saddles | 68 .. 132 |

### 4.3 Parts

| Part | File | PLA mass | Print orientation |
|---|---|---|---|
| Base plate + hub + stop posts | `rig_base` | 359 g | plate bottom on the bed |
| Turntable (plate, neck, stop arm and lug, home flag) | `rig_turntable` | 83 g | top face on the bed |
| Drive gear z=114 m0.8, 7.5 thick | `drive_gear` | 61 g | bottom face (the one with the nut trap) on the bed |
| Foot x4 | `rig_foot` | 29 g each | pad down |
| V saddle x2 | `saddle` | 170 g each | bottom on the bed |
| Axle spacer / bearing spacer ring | `axle_spacer`, `bearing_spacer_ring` | 1 g / 3 g | on end |
| Home sensor bracket | `sensor_bracket` | 16 g | bottom on the bed |
| Encoder holder | `encoder_holder` | 26 g | its modelled orientation (post up), bottom of the pad and arm on the bed |

Purchased: M8 x 70 A2 hex bolt (stainless so the magnet is not disturbed), M8 nut, 2 x 608 bearing, 500 mm 2020
extrusion + 4 M5 drop-in T-nuts, 2 x M5 x 8 (rail to table, from below; check against your T-nut), 4 x M5 x 20 + 4 M5
washers (saddles; the saddle's head wells are deeper than the contract), 4 x M3 x 6 countersunk (motor), M3 x 16
button head + insert x4 (feet), M3 x 14 button head + insert x2 (encoder holder), M3 screws and inserts for the
sensor bracket, AS5600 breakout, 6 x 2.5 diametric magnet, FC-03 slot module, 25 mm hook-and-loop straps, 2 mm
felt or EPDM for the saddles, bench screws for the feet.

### 4.4 Behaviour and limits

- Hard stops: the table lug meets the two base posts at exactly +-100.0 deg (computed from the geometry).
  Soft limits in software stay inside +-90 deg. The stops are sized to catch a slow fault, not a full-speed
  swing of the gun; with this much inertia the right protection is the soft limit plus a current-limited driver.
- Rotation is clear of the motor because the lug, the flag and the stop arm sweep only the +-100 deg window on the
  motor-free side (-X); the table and rail clear the motor top by 10 mm.
- Gear stage estimate (hand calculation, not a tool result): Lewis bending stress at an assumed 0.4 N m motor
  torque is about 28 MPa at the pinion and 22 MPa at the driven gear for 7.5 mm face width, against roughly 50 MPa
  for PLA. Treat as a marginal screen. The torque is an assumption.
- Backlash: nominal 0.4 mm added to the centre distance (53.2 -> 53.6); the CAD mesh showed 0.145 mm minimum
  flank clearance. Real backlash will be larger after printing. The encoder, not the step count, is the angle truth.

### 4.5 What was checked, and what was not

- `check_interference` on the 19-component assembly (base, turntable, gear, motor, M8 bolt and nut, two 608s, two
  spacers, rail, two saddles, four feet, sensor bracket, encoder holder): no overlaps after the bolt and nut were
  rotated to match the hex pockets; a 0.1 mm^3 sliver between motor face and pocket floor is the nominal face
  contact. Placement is by coordinates from the design frame, not by mates.
- `verify_battery`: see 4.6.
- `dfm_check` (FDM, Ender 3, in the print orientation above): remaining items are listed in 4.6. The 45 degree bed
  chamfers and the hex nut-trap ceilings (bridges up to 12.5 mm across the hex, 2.5 mm unsupported between the
  hex wall and the bolt hole) are flagged by the tool; they are normal practice but are not clean DFM passes.
- Not modelled: the gun, M3/M5 screws, heat-set inserts, T-nuts, the magnet, the AS5600 and FC-03 boards, the
  motor wires. The saddle fit on the real gun, the balance point and the sensor module dimensions are open items
  (section 5), not results.

### 4.6 Final results

Placement of the two sensor parts: the sensor bracket sits at (-44, 0) rotated 90 deg about Z, so its two M3 insert
bores in the base are at (-55.8, 0) and (-32.2, 0) and the module's slot gap runs radially; the home flag blade hangs
from the table at X -45..-43 and is 12 wide, bottom at Z 28, with 7 mm free above the bracket seat (Z 21). The
encoder holder is rotated 90 deg about the axle, so its post is on the -Y side with its two M3 screws at (-8, -56)
and (8, -56). The pad is 28.5 x 28.5 centred on the axle at Z -32.4 (PCB seat Z -33.6 floor, chip face -32).

Per-part results, after the last edit of each part (`verify_battery` = no FAIL on every part and on the
assembly; DFM statuses are `dfm_check` with the Ender 3 profile in the print orientation of 4.3):

| Part | PLA mass | `dfm_check` items still not passing |
|---|---|---|
| rig_base | 359 g | sharp concave edges (counterbore floors, bore ledges); min-wall warn (3 mm pocket floor) |
| rig_turntable | 83 g | 45 deg top-edge chamfer reads as 45.2 deg; hex bolt-head pocket ceiling (12.5 mm bridge, ring only 2.5 mm unsupported); sharp concave edges at the hex floor and counterbores |
| drive_gear | 61 g | nut-trap ceiling (same bridge); tooth roots have no fillet (smaller than the 0.4 mm nozzle); polyline tooth segments read as 0.31 mm features |
| saddle x2 | 170 g each | one sharp edge in the strap tunnel roof (rounding it needs support) |
| rig_foot x4 | 29 g each | countersink seat edge, kernel would not fillet it |
| sensor_bracket | 16 g | 16 sharp concave edges at the board seat and flares |
| encoder_holder | 26 g | 10 sharp concave edges; two D2.4 holes flagged small |
| axle_spacer / bearing_spacer_ring | 1 g / 3 g | passes (tall thin tubes, print on end, warn only) |

`find_stress_risers` was run on the base, turntable and gear. It lists counterbore floors, bore ledges and the 240
gear tooth roots as sharp concave edges and some hole-ligament candidates; none was fixed beyond the fillets already
applied. The agents' saddle, foot, bracket and holder runs report the same kind of items.

Total printed mass is about 1.0 kg (base 359, saddles 340, turntable 83, gear 61, feet 115, small parts 48).


## 5. Open items and assumptions

| # | Item | Why it matters |
|---|---|---|
| 1 | Count pinion teeth, confirm module | wrong module means the printed driven gear will not mesh; slots absorb only about +-2.5 mm |
| 2 | Weigh the gun (assumed 3.2 .. 3.6 kg) and find its balance point | sets where the saddles go and the torque margin |
| 3 | Confirm motor step angle, rated current and body length | body length measured as 37.15 mm from the calliper photo (digits partly unreadable); flange measured 42.0 .. 42.2 |
| 4 | Pinion height above the flange | photos show about 12 mm of shaft + pinion above the flange face; confirm with a calliper |
| 5 | Torque margin | NEMA 17 holding torque unknown; the 6:1 stage plus estimated gun inertia (about 0.4 kg m^2 for a 1.2 m gun about its middle) needs checking against the motor's real curve |
| 6 | Gear is printed 0.8 module (tooth 1.26 mm thick at the pitch line) | adequate for this light, slow duty; if teeth shear or the print is poor, change to a coarser module (needs a new pinion) or a belt stage |
| 7 | 2020 rail length 500 mm and saddle spacing | validate against the real balance point before cutting the rail |
| 8 | Saddle fit on the real gun | not checked: the gun model is not in the assembly. Both V saddles are identical and sit at one height, but the forend and the stock/receiver underside are at different heights; expect to shim one saddle with the felt/EPDM liner. The pod's forend-nose clamp (gun X 431..515) must stay clear of both saddles |
| 9 | FC-03 and AS5600 module dimensions | pocket sizes and hole positions are assumed (32 x 14 and 24 x 24 boards); check with the real boards before printing the two small brackets |
| 10 | Home flag blade length | blade bottom is at Z 28, 7 mm above the bracket seat; the real slot depth is unknown, trim the blade with a file if it bottoms out |
| 11 | Bearing and nut-trap fits | the 608 bore is D22.15, the bolt bore D8.4, the hex pockets are 13.4 across flats, the printed gear/turntable axle holes are plain; print the base alone first and test the fit before printing the rest |
| 12 | Motor and gear height | the pinion teeth sit at -18..-10 and the gear at -18..-10.5; if the real pinion is higher or lower than modelled (open item 4), shim under the nut trap or the gear hub boss by 0.5 to 2 mm |

## 6. Safety

- Gun unloaded, action open, muzzle direction safe, and the chamber checked before it goes on the rig.
- The table is driven by a motor with a 6:1 reduction under a 1.2 m, 3 kg lever. Keep hands and face clear of the
  swing arc. The rig has hard stops but a software or power fault can still move the gun a few degrees.
- Clamp or bolt the base down before powering the motor.


## 7. Candidate B: direct-drive rig (2 printed parts) — rev B2, 2026-11-09

Simple rig chosen by Hunter: motor screwed under a printed arch, gun straight on the motor, no gear, no bearings.
Rev B2 replaces the first B layout (pinion pulled off, barrel cradle). Candidate A (section 4) stays parked.

CAD: `parts/calrig_B/motor_mount_arch.step+`, `parts/calrig_B/yaw_arm.step+`, motor reference
`refs/stepper_nema17_pinion.step+` (the user's motor, pinion ON), assembly `asm/calrig_B_final.step+`.
`refs/stepper_nema17_bare.step+` is no longer used. Frame: origin on the shaft axis at the motor face, +Z up,
motor body Z -37.15..0, bench plane Z -40.
Software: `software/firmware/calrig_uno/` (Arduino Uno + CNC Shield V3), host and analysis in
`software/tbot/calrig/`, CLI `tbot calrig`, operator guide `software/docs/CALRIG.md`, interface contract
`software/docs/CALRIG_PROTOCOL.md`.

### 7.1 Why rev B2 (problems found in the first B layout)

| Problem in first B | Rev B2 |
|---|---|
| The pinion had to be pulled off; hub pinched a plain D5 shaft with one M3 (weak, slips) | Pinion stays on. The arm's split hub clamps the pinion tooth tips in a D17.1 socket (pinion OD 16.69) with an M3x16 + nut: ~3x the grip radius, teeth bite into the PLA. Tooth count does not matter |
| R13 barrel cradle could only grip the barrel ahead of the pod (X > 530), about 300 mm from the gun's balance point (behind X 300): ~10 N m pitch moment on one hose clamp and the motor bearings | 100 deg V saddle on the FOREND underside (forend X 178..484), 80 mm long, slides to the balance point; two strap tunnels |
| Barrel 115 mm above the bench: stock toe (gun Z -98.6) and grip hit the bench during a swing | V apex at Z 90 = 130 mm above the bench; forend bottom ~137 mm up, stock toe ~32 mm above the bench |
| 3 mm plate (countersinks left 1.3 mm), 5 mm legs loaded across layers | 5 mm plate, 8 mm legs, R6 leg/flare fillets, R3 plate/leg fillets; feet flare at 54.7 deg |
| D30 flange: tips at ~0.5 N m imbalance | D54 flange: tips at ~0.9 N m (3.4 kg on r 27) |
| No home, no lost-step check | Hall home: A3144 lying in a plate-top pocket at (0, +22), 6x3 magnet in the flange underside; feeds the CNC-shield X endstop |
| Wrong motor model (bare shaft, no connector) | Motor ref corrected from the 2026-11-09 photo: pinion teeth Z 5.5..13.3 (7.8 face, OD 16.69 caliper), shaft to Z 13.3, side connector 16 x 6 x 10 near the rear cap. Assembly rotates the motor 90 deg so the connector faces +Y (open side of the arch) |

### 7.2 Parts

| Part | What | PLA (solid / slicer est.) | Print orientation |
|---|---|---|---|
| `motor_mount_arch` | 104 x 56 x 45 arch. 5 mm plate on the motor face (D22.4 pilot hole, 4 x M3 countersunk on 31 mm square for M3x8 FHCS), 8 mm legs, flared feet with 4 head wells (D9.4, 45 deg seats for countersunk/bugle wood screws, D4.6 shank) at (+-44, +-18). Hall pocket 4.6 x 3.4 x 1.9 + lead groove to the +Y edge | 110 g solid / ~47 g at default infill | plate top face on the bed |
| `yaw_arm` | 80 x 76 x 107. D54 x 5 flange rides on the plate (PTFE tape), D40 clamp hub with 1.2 slit, M3x16 SHCS + M3 nut, D17.1 pinion socket with a 50 deg cone roof, magnet pocket D6.2 x 3.1 at (0, +22) underside, D32 column, lofted neck and 50 deg head flare (support-free), V saddle 100 deg incl. with 8 mm flat floor at Z 90, two 20 mm strap tunnels with peaked roofs | 369 g solid / ~110 g at default infill | flange on the bed, upright |

Hardware: 4 x M3x8 FHCS (motor), 1 x M3x16 SHCS + M3 nut (clamp), 4 x countersunk wood/deck screws ~4.5 shank,
2 x 20 mm hook-and-loop or ratchet straps (or a 12.7 mm hose band), 1-2 mm EPDM/felt liner in the V, PTFE tape
under the flange, A3144 hall switch + 6x3 N35 magnet (S pole down, glued), Arduino Uno + CNC Shield V3 + A4988/
DRV8825/TMC2209, 12 V supply, NC e-stop button on A0.

Load path: gun weight -> V saddle -> column -> flange -> plate -> legs -> bench. The motor face carries none of the
weight (the flange sits on the plate). The pinion clamp carries only the yaw torque.

### 7.3 Gun attachment

- Forend in the V on a 1-2 mm liner, both straps over the forend/barrel through the tunnels.
- Slide the gun until it balances over the shaft (expect around gun X 200..300 with the pod fitted); keep the
  saddle clear of the pod clamp (gun X 431..515). Weigh/balance the real gun: the balance point is not measured.
- Upright pose: forend in the V. Side poses (roll 90 deg): the forend side lies in the V; check strap tension.

### 7.4 Feasibility numbers (hand estimates, not tested)

| Quantity | Value | Basis |
|---|---|---|
| Angle per 1/16 microstep | 0.1125 deg | direct drive, 1.8 deg motor (check label) |
| Gun yaw inertia | ~0.4 kg m^2 | ASSUMED 3.4 kg, 1.2 m |
| Motor torque | ~0.3-0.4 N m holding | ASSUMED short-body NEMA 17; read the label |
| Firmware defaults | 45 deg/s, 20 deg/s^2 (caps 90 deg/s, 60 deg/s^2) | 20 deg/s^2 x 0.4 kg m^2 = 0.14 N m + flange friction ~0.07 N m with PTFE |
| Pinion clamp torque capacity | ~3 N m | M3 preload ~600 N, mu 0.3, r 8.3; PLA creep lowers it: re-tighten before a session |
| Column bending at a 50 N shove on the head | ~1.5 MPa | D32, 100 mm lever; interlayer PLA ~15-25 MPa |
| Legs at the same shove | ~6 MPa across layers | 8 x 56 legs, 2 legs |

What the rig can and cannot do: gyro bias, gyro scale and time offset on the axis that is vertical in a pose, axis
map/sign from several poses, attitude drift, slow sine sweeps (e.g. 15 deg, 6.5 s period). Not fast swings: clay-rate
(57-90 deg/s) moves need ~0.4 N m+ and long ramps. The four barrel-roll poses never put the chip axis that runs along
the barrel under the yaw rate, so its gyro scale needs the muzzle-up/down poses (not possible on this saddle) —
the analysis reports it as missing rather than guessing. Requirement F10 in section 2 is therefore only partly met.
Truth angle is the step count; the hall home detects lost steps (HOMED offset, hall-edge repeatability).

### 7.5 Checks run (2026-11-09, tool results)

- `check_interference` on `asm/calrig_B_final`: no interference. Contacts at 0.00 mm are the designed face contacts
  (motor face/plate, flange/plate). Pinion tips to socket 0.158 mm.
- `verify_battery`: no FAIL on `motor_mount_arch`, `yaw_arm`, `stepper_nema17_pinion` and the assembly.
- `dfm_check` (FDM, Ender 3, orientations above): `yaw_arm` support envelope 0.08 cm^3; remaining overhang rows
  are the bridged ceilings of the screw-head seat, nut trap and slit top (~12 mm^2, spans <= 9.4 mm). `motor_mount_arch`
  flags the 45.0 deg screw-head seats (8.7 mm^2 each) and passes support. Both report sharp concave edges at pockets
  and holes (normal for these features) and knife-edge min-wall warnings where pockets break the surface.
- `find_stress_risers`: hole-ligament candidates between the motor pilot hole and the countersinks (standard NEMA
  pattern, >= 7.7 mm), the clamp screw crossing the slit (by design), pocket corners. The hub/column step and the
  flange root are filleted.
- Not checked: real gun balance point and forend fit in the V, hall switch trigger distance through the PTFE tape,
  motor torque, the pinion's real tooth profile and the clamp's real holding torque, print accuracy of the D17.1 socket.
- Software: firmware builds for the Uno (15.4 kB flash, 970 B RAM); host + analysis test suite passes; full run +
  analysis verified in `--sim` only. Nothing has run on hardware.
