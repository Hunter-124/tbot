# TBOT — print list + hardware BOM

Geometry, placement and clearances: [layout.md](layout.md). Build order: [assembly_sequence.md](assembly_sequence.md).
Assembly: `asm/tbot_full.step+`. Units mm; gun frame +X muzzle, +Y gun-left, +Z up, origin at the trigger face.

## Printed parts

All PLA. Files: `print/creality_ender_3_0_4_mm_nozzle/<part>.gcode` (+ `.3mf`, `.print.json`). STEP: `export/<part>.step`
(gun frame — re-orient to the build direction below if you slice yourself).

Slicer: PrusaSlicer 2.9.6, Creality Ender-3 0.4, PLA+ 215 °C nozzle / 60 °C bed, 0.2 mm layers, 4 walls, gyroid infill
(25 % structural / 15 % covers), scarf seams, complete-individual-objects plates. Saved defaults: `print/print_defaults.json`; G-code is named `<HH>h<MM>m_<plate>`. Don't leave the parts in a hot car — PLA softens at ~55–60 °C.

### Front module

No supports intended, except under carrier_R's MI-1 section (rev U, see below). 8 parts (`camera_cover` added rev W),
231.0 g solid-PLA model mass without it (rev U; +4.5 g for the cover). The g / h figures and G-code in `print/` below are from rev R: every front-module part changed in rev S, five
changed in rev T and four again in rev U, so re-slice all seven from `export/<part>.step` before printing. The nose clamps changed again in rev W (FPC slot opened
through the nose front): re-slice them too. `chin_lid`
is obsolete (no chin since rev S); do not print it.

| Part | Build (up) | g / h (rev R) | Check (rev R) |
|---|---|---|---|
| nose_clamp_L | −Y, flank down | 37.0 / 3.7 | G-code pass; not print_ready only for the MI-1/IMU insert-bore teardrops (8.7 mm² model overhang) |
| nose_clamp_R | +Y, flank down | 38.4 / 4.0 | same (16.8 mm²) |
| carrier_L | +X, rear face down | 27.9 / 4.1 | G-code warn (9.2 mm air run, < 10); model overhang on the Pi-standoff chins |
| carrier_R | +X, **buck-bay rear face (X 396) down** (rev U) | 24.3 / 3.7 | re-slice: support under the MI-1 block's X 422.45 face (rev Z; ~26 mm up) and, since rev Y, under the battery hold-down A post, B boss and C block (~194 mm², C's tree support stands ~92 mm inside the open battery cavity) |
| cover_L | −Y, flank down | 15.2 / 1.6 | G-code warn (8.1 mm air run); counterbore roofs flagged |
| cover_R | +Y, flank down | 17.8 / 1.9 (rev R) | rev Y: flank moved out 1.6 (whole length) and front +7.6; not re-sliced |
| belly_cover | +Z, floor down | 21.5 / 2.3 | G-code pass; 0.3 mm skin under the blind buck #1 pilots is intended |
| batt_retainer | −Y, pack-contact face (Y −47.4) down | ~6.0 g model (rev Y) | new rev Y: 3-point battery hold-down plate, 2.2 thick. DFM at this orientation: only the 0.8 × 45° bed-edge chamfers flag as borderline overhang; no supports |
| camera_cover | +X, plate face down (X 522) | 4.5 / 0.6 (rev W) | **not print_ready**: 0.9 mm ligament between each M2 clearance hole and the plate edge (set by the camera PCB's own hole positions; 2 perimeters fit). G-code warn: 5.4 mm unanchored run, 6.5 mm bridge over the ear pockets. No supports. Print 1 |

### Trigger set

No G-code in `print/` for these three: the earlier files were sliced from older geometry and have been removed. Slice
them from `export/<part>.step` with the settings above.

| Part | Build (up) | Supports | Model mass |
|---|---|---|---|
| servo_lever | −Z, top face on the bed | none | ~3.2 g |
| servo_bracket_L | +Z, ear-block bottoms (Z −21.7) on the bed (rev AA: was +Y; clamp-screw axis now lies in the layer plane) | build plate only, organic | ~10.8 g |
| trig_clamp_R | +Z | none (D7.4 switch-hole and D7 access-bore crowns bridge; ream 7.4 if the switch hole sags) | 8.7 g |
| u18_spacer (rev AG) | standing on an end (axis up), 5 mm brim | none | 0.3 g |

### Lens lock nut

`lens_lock_nut` (split clamp nut, M12 x 0.5, pinched with 1 x M2 x 8 pan head self-tapping) locks the lens focus
against the lens-holder face; `lens_lock_nut_plain` is a plain jam-nut fallback. Front face on the bed, 0.12 mm
layers, no supports, under 1 g each. Not sliced or printed yet. Dimensions, the caliper checks to make first and
fitting: [lens_lock_nut.md](lens_lock_nut.md).

## Purchased hardware

### Electronics and actuator

| Qty | Item | Where / note |
|---|---|---|
| 1 | Raspberry Pi 3 Model B v1.2 | carrier_L, right side up: solder side to the inner wall on 4 rooted standoff blocks, components toward cover_L |
| 1 | innomaker CAM-MIPIOV9281 V2 (OV9281 mono global shutter) | nose, under `camera_cover`, 4 × M2 × 8 |
| 1 | JETSECAM 12 mm M12 lens, 1/1.8", IR-cut filter, "no distortion" (Amazon, ordered 2026-10-25) | replaces the camera kit's 2.8 mm lens; focus at ~15 m and lock with `lens_lock_nut`. Optics: software/docs/OPTICS.md §3 |
| 1 | Camera FFC 15-pin, **200 mm** (the 150 mm kit cable still does not reach) | route ≈ 165 mm (rev U, sideways from the camera into the Pi lobe); Z-fold the ~35 mm excess flat behind the Pi |
| 1 | GY-521 (MPU-6050) | nose_clamp_R floor pocket, 2 × M3 × 6 SHCS |
| 2 | XL4015 HW-083 buck module | screw terminals removed on both, leads soldered to the IN/OUT pads. #1 → Pi, **5.1 V**, crosswise in the belly. #2 → servo, **7.0 V**, standing (long axis vertical) in the carrier_R buck bay behind the pack, solder side to the inner wall, components outboard, IN pads at the bottom and OUT pads at the top. Set each voltage on a bench supply before anything is connected |
| 1 | 1000 µF / ≥ 16 V radial electrolytic, ≤ Ø10 × 20 | vertical in a wall snap clip above the charge jack, leads up and over the bulkhead to buck #2's OUT pads |
| 1 | 3S 2600 mAh Li-ion pack, 70 × 58 × 20 (~140 g) + its charger | battery lobe (gun-right) |
| 1 | 2-circuit lever connector | battery disconnect in the belly — the only power cut |
| 1 | DC5521 inline female jack | carrier_R, vertical, mouth DOWN at (436.5, −33.5); charge from below |
| 1 | TowerPro MG996R + stock single-arm horn | servo_bracket_L; stock horn screw and rubber grommets not used |
| 1 | 7 mm momentary NO pushbutton (yellow-cap E429-type, M7 bushing) + its hex nut | trig_clamp_R panel; washer not used. Pressed on its bare cap |

### Fasteners

Inserts: brass heat-set, Ruthex-style M3 × 5.7 (OD ≈ 4.6, hole Ø4.0) and M2 × 3.0 (OD ≈ 3.6, hole Ø3.2). Iron
220–240 °C. Rule: every plastic-to-plastic joint is an M3 screw into an M3 insert; M2 only where an electronics board's
holes are too small for M3 (Pi Ø2.75, camera Ø2.2, XL4015 Ø2.3). If your M2 inserts need a different hole, the four
Pi bores are one feature each in carrier_L (`PiM2Insert_*`).

| Qty | Item | Where |
|---|---|---|
| 23 | M3 heat-set insert | nose_clamp_L 2 (cross screws, Y = 0 face); nose clamps 4 (MI-1, rear face X 426.3, 2 per half) + 2 (front carrier screw, 1 per half, in the flank at X 495.5, Z 3.2); nose_clamp_R 2 (IMU); carrier_L 3 (1 belly + 2 cover_L flank posts); carrier_R 8 (2 belly + 3 cover_R: bulkhead top + bottom, bay bottom-rear + 3 battery hold-down A/B/C along +Y from the Y −46.2 boss faces); servo_bracket_L 4 (ears RI/RO/FI/FO only; the guard-clamp screws now use nuts) |
| 4 | M2 heat-set insert | carrier_L Pi standoff blocks |
| 2 | M3 × 20 ISO 10642 countersunk | nose-clamp cross screws along Y at (X 456, Z 4.5) and (X 505, Z 2.6), from gun-right |
| 4 | M3 × 10 ISO 10642 countersunk | carriers → clamp MI-1, driven +X from the carrier blocks' X 422.45 face (rev Z: was M3 × 18; the clamp rear face is now X 426.3 and the insert bore ends at X 432.8) |
| 13 | M3 × 8 ISO 10642 countersunk | carrier_L / carrier_R → nose_clamp_L / R front screw, 1 per side, along ∓Y at (X 495.5, Z 3.2), head flush in the carrier inner wall (Pi / battery side; remove Pi or pack to reach it), tip 5.0 into the clamp insert; batt_retainer → carrier_R along +Y at A (447.4, Z 35.0), B (447.4, Z −13.0), C (525.55, Z 11.6), heads flush in the plate; belly → carriers from below at (433, ±21.5), (518.9, −21.5); cover_R → carrier_R vertical at (433.4, −43.5) top + bottom and (401.4, −43.75) bottom (bay); cover_L → carrier_L through the flank along −Y at (X 433.4, Z 30.0 / −11.0) |
| 2 | M3 × 6 SHCS | GY-521 → nose_clamp_R (M3 × 10 does not fit) |
| 3 | M3 × 30 ISO 10642 countersunk | trig_clamp_R_v2 → servo_bracket_L: U18, U28, L40 through D3.5 holes into trapped M3 hex nuts (seats Y 12.9 / 12.9 / 13.3); heads flush with the clamp hole pad at Y −13.9, tips Y 16.1 (rev AA: was M3 × 25 into inserts, heads 2.4 deep in a counterbore) |
| 1 | M3 × 30 ISO 10642 countersunk | servo_bracket_L → trig_clamp_R_v2 rear (X −15.5, Z −17.0), from the bracket back face through the rear ear block and the clamp's rear spacer boss into a trapped M3 hex nut (side slot from −X, nut Y −2.5..−0.1); head flush at Y 27.5, tip Y −2.5 (rev AG: was M3 × 35) |
| 1 | M3 × 12 ISO 10642 countersunk | rev AG front jack: up and back into the clamp's front-bottom face, threads itself into the D2.8 bore, tip on the guard's front slope |
| 4 | M3 hex nut (ISO 4032, 5.5 AF × 2.4) | servo_bracket_L: U18 and L40 slide in from the side slots, U28 drops into its top slot (vertex down); trig_clamp_R_v2: rear nut slides into the cradle side slot from −X. Slots stop the nuts turning |
| 4 | M3 × 10 ISO 10642 countersunk | servo ears RI / RO / FI / FO, up into the bracket |
| 1 | M3 × 12 ISO 10642 countersunk | lever + horn → servo output shaft (M3 × 14 bottoms out) |
| 4 | M2 × 5 machine screw, pan head (ISO 7045) or SHCS | Pi 3B → M2 inserts, from the component side |
| 4 | M2 × 5 pan head self-tapping | electronics into plastic pilots: buck #1 2, buck #2 2 (top bay-wall posts at X 400.4 / 423.1, Z 38.25) |
| 4 | M2 × 8 pan head self-tapping | camera_cover + camera PCB → nose_clamp corner pilots (1.6 cover + 1.6 PCB, ≈ 4.8 thread engagement in the D1.7 × 6.0 pilots) |
| 1 | M5 flat washer ISO 7089 (Ø5.3 / Ø10 × 1.0) | sling-swivel stud, between the clamp front wall and the stud eye |
| 2 | SAE #44 worm-drive hose clamp, 1/2" (12.7) band, nominal ~59–83 mm (check the brand's range) | wrap the nose clamp directly in its grooves: B1 X 428.4–441.1 (rev Z, moved 11.9 mm back so it sits in the measured rib window, 4.0 mm behind the post; path 210.8 mm = 67.1 mm eq. ID), B2 X 502.15–514.85 (226.0 mm = 71.9 mm) — trim the tail after tightening, deburr. The #56 clamps no longer fit (too large) |

### Soft goods and wire

| Qty | Item | Where |
|---|---|---|
| 2 | 0.5 mm silicone/PTFE tape strip, ~40 × 14 | barrel under each band (band → rib plate 0.23 mm) |
| 2 | 1 mm adhesive EVA/neoprene foam, ~70 × 17 | inside each nose-clamp jaw (fills the 1.2 mm forend clearance) |
| — | 0.5–1 mm foam (optional) | battery: between batt_retainer and the pack only if the pack is thinner than 18.8 mm (the plate then bottoms on the bosses) |
| 2 | 1 mm neoprene/rubber liner, ~45 × 25 (0.85 compressed) | between each trigger plate and the guard bow |
| — | silicone wire 20 / 22 / 26 AWG, heat-shrink 3 mm + 6 mm | wire list below; splices S1, SPL, B_R |

No cable ties or tie points anywhere: every lead goes through a pass, harnessing is done separately. GPIO connections
are soldered — there is no room for DuPont housings (header pin tips are 8 mm from cover_L).

### Tools

2.0 mm hex (all ISO 10642 M3; long shaft for the MI-1 screws), 2.5 mm hex (M3 × 6 SHCS), PH1 or 1.5 mm hex (M2 × 5
machine screws), PH0 (M2 self-tappers), 9.5 mm (3/8") open-end wrench (button nut), soldering iron with M3 / M2 insert
tips, bench supply for setting the bucks.

## Wire list

Cut = modelled + 30 mm. Colours suggested.

| Circuit | Wire | From → to | Cut |
|---|---|---|---|
| Battery | 2 × 20 AWG | pack lead block → SPL above the jack (R lobe rear) → forward at Z 11.6 → down at X 450 / 448 → rear lobe pass (X 442.5..451) as singles → belly → lever +Y end | 147 / 147 |
| Charge jack | 2 × 20 AWG (jack pigtail) | jack boot → SPL, local in the R lobe | 35 each |
| 12 V | 2 × 20 AWG | lever −Y end → S1 (470, −23, −10.5) | 47 each |
| Buck #1 IN | 2 × 20 AWG | S1 → buck #1 IN pads | 46 / 42 |
| Buck #2 IN | 2 × 20 AWG | S1 → aft along the belly side of the carrier_R wall → rear lobe pass → outboard, aft at Y −46.8 through the bulkhead lower pass → under buck #2 → IN pads (bottom) | 133 (+) / 129 (−) |
| Pi 5 V | 2 × 20 AWG | buck #1 OUT pads → belly → carrier_L GPIO pass → under the Pi's bottom edge → up the component side → pins 2 / 6 | 108 / 99 |
| Servo 7 V | 2 × 22 AWG | buck #2 OUT pads (top) → aft above the board's top edge → bay rear-wall exit (+ external run to the MG996R) | 51 (+) / 54 (−) in module |
| Cap leads | 2 × 22 AWG extensions | 1000 µF (vertical, above the jack) → bulkhead upper pass → buck #2 OUT pads | 58 (+) / 48 (−) |
| IMU | 4 × 26 AWG | pins 1 (3V3) / 3 (SDA) / 5 (SCL) / 9 (GND) → GPIO bundle → carrier_L pass → B_R → raised clamp trough (above B1) → GY-521 | 126 / 132 / 129 / 130 |
| Arming + PWM | 3 × 26 AWG | pins 11 (GPIO 17, arming) / 12 (GPIO 18, servo PWM) / 14 (GND) → GPIO bundle → B_R → 3-wire rear exit (+ to the trigger set) | 106 / 105 / 107 in module |

SPL (charge jack onto the pack leads) is on the pack side of the lever connector, so charging works with the system
disconnected. Two cables leave the module: the 3-wire signal exit from B_R across the belly rear wall and the servo
pair through the buck-bay rear wall (Ø4.8 hole at X 396, Y −38.0, Z 38.4). Both are hardwired, no in-line connector; the runs
to the trigger set are routed by hand. Buck #2 IN/OUT pad positions and polarity in the model are inferred from the
terminal footprint: check on the board before wiring.

There is **no hardware arming interlock**: the servo rail is live whenever the lever connector is closed. Open the lever
connector to make the actuator safe.
