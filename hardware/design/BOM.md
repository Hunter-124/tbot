# TBOT — print list + hardware BOM (REV H)

Geometry and placement: hardware/design/layout.md. Verified Ender 3 G-code (PQ12 actuator set): hardware/print/ender3_0.4mm/
tbot_<MATERIAL>_pN plates (one .gcode + .3mf + .print.json per plate, all print_ready, no supports, "Complete individual
objects"). The material is in every file name — load that filament before starting the plate. STEP:
hardware/step/parts/*.step (units mm, gun frame — re-orient in the slicer to the build direction below if you slice yourself).

### Print plates (PQ12 set, 2026-10-27 R002) — "strength" (4 walls, 35 % cubic) on every plate
PETG: Generic PETG @CREALITY 240 °C / 70 °C. PLA: Generic PLA @CREALITY at the user's 215 °C nozzle / 60 °C bed.
Each plate prints its objects one at a time (shortest first). PETG 164 g / 18.0 h; PLA 122 g / 12.5 h.

| Plate / file | Filament | Objects (print order) | Mass / time |
|---|---|---|---|
| tbot_PETG_p1_pod_body.gcode | PETG | pod_sled, cam_mount_lower | 143.7 g / 15.5 h |
| tbot_PETG_p2_pod_cap.gcode | PETG | pod_cap | 20.6 g / 2.6 h |
| tbot_PLA_p3_trig_bracket.gcode | PLA | batt_lid, trig_bracket_L | 45.7 g / 4.5 h |
| tbot_PLA_p4_trig_clamp.gcode | PLA | batt_box, trig_clamp_R | 71.6 g / 7.3 h |
| tbot_PLA_p5_trig_finger.gcode | PLA | arm_pad, trig_finger | 4.3 g / 0.8 h |

Not sliced (MG996R set, kept for comparison): servo_bracket_L, servo_lever (their per-part G-code is still in the print folder).

## Printed parts

| Part | Material | Build direction (up) | Supports | Sliced mass / time | Notes |
|---|---|---|---|---|---|
| cam_mount_lower (pod body, open front, rails) | **PETG** (ASA preferred; user has no ASA-capable printer) | **+X — stands on the rear bulkhead face** | none | see plates | barrel contact via fiberglass insulation; REV F orientation |
| pod_cap (front cap: camera + IMU) | **PETG** (ASA preferred) | +Z (Z=−8 face on bed) | none | see plates | |
| pod_sled (Pi + XL4015 sled) | PETG (rides inside the barrel-mounted pod) | +Z (plate on bed) | none | PETG p1 | slides into the pod |
| batt_box (battery tub, left side of buttstock, REV E.3) | PLA | base down (gun +Y up) | none | PLA p4 | |
| batt_lid (outer lid with strap channels, REV E.3) | PLA | inner face down (gun +Y up) | none | PLA p3 | |
| trig_bracket_L (PQ12 left plate + cradle + carriage shroud, REV H) | PLA | +Z | none | PLA p3 | PQ12 set |
| trig_finger (PQ12 carriage/finger) | PLA | −X (rear face X=+26.7 on bed) | none | PLA p5 | PQ12 set |
| trig_clamp_R (right counter-plate + arming-trigger housing, REV G; REV H.1 shorter housing) | PLA | +Z | none | PLA p4 | shared by both trigger sets |
| arm_pad (finger pad glued on the arming-button cap, REV H.1) | PLA | +X (rear socket face on bed) | none | PLA p5 | shared by both trigger sets |
| servo_bracket_L (MG996R bracket) | PETG | +Z | none | 20.8 g / 2.5 h | servo set |
| servo_lever (MG996R lever/finger) | PETG | −Y (outboard plate face on bed) | none | 3.0 g / 0.4 h | servo set |

Not printed: cam_mount_upper, imu_clamp_upper (REV C split caps), imu_clamp_lower (REV D muzzle IMU — blocked the
camera), pendant_body / pendant_cap (REV F — the arming switch moved onto trig_clamp_R).
Slicer settings: PrusaSlicer Creality Ender-3 presets, 0.4 nozzle, "strength" on every plate. Material split (user,
2026-10-27): PLA for the trigger set and battery box/lid (off the barrel, prints better); PETG for everything mounted
on the barrel (pod body, pod cap, pod sled). Don't leave the PLA parts in a hot car — PLA softens at ~55–60 °C.
Pod body and cap: the user prints PETG only (stock open-frame Ender 3, no ASA). PETG (Tg ~80 °C) is better than PLA
on a hot barrel but still below ASA, so insulate every barrel contact with fiberglass sheet/tape between the barrel
and the print, and check the pod for creep/looseness after a long session. PLA is not acceptable for the pod body or cap.

## Hardware

### Barrel mounting (pod)
- 2× stainless worm-drive hose clamp, **9 mm band, 70–90 mm range** (DIN 3017-1 W4/A2; closed Ø ≈ 81 mm). They wrap the OUTSIDE of the pod (over the barrel top, down the pod's side-wall recesses, under the floor).
- 1.5 mm silicone rubber sheet ~60 × 160 mm (barrel liner over X 490–640), or 1 mm rubber tape, 2 wraps
- 7 mm nut driver or flat screwdriver (worm screws point straight down on the gun-right side)
- Before printing, check on the gun that a 9 mm band slips under the vent-rib plate at about 510 and 616 mm forward of the trigger face.

### Pod body, sled, cap
- 4× M3 brass heat-set insert Ø4.0 × 6 (pod cap bosses) + 4× M3 × 12 SHCS (cap → pod)
- 4× M2.5 brass heat-set insert (hole Ø3.6 × 4.5, sled standoffs) + 4× M2.5 × 6 pan/SHCS (Pi)
- 2× M2 × 6 self-tapping pan head (camera top edge; bottom edge drops into the cap's capture slots)
- 2× M2.5 × 10 SHCS + 2× M2.5 nut + 2× M2.5 washer (IMU on the cap)
- 2× silicone pad 2.2 × 5.1 × 1.5 (cap pusher tabs → sled preload)
- ~8× cable tie 2.5 mm (XL4015 ×2, sled lead anchors, pod rear anchor)
- Camera ribbon: kit 15-pin ~150 mm (Pi 3B); a 15→22-pin "Pi 5 camera cable" when you swap to the Pi 5

### Power
- 1× XL4015 5 A buck module (54 × 23 × 18), set to 5.1 V
- 1× 12 V (3S) 2600 mAh Li-ion pack, 70 × 58 × 20 (Xeanerol or equivalent) + its charger
- ~1.1 m 2-core 20 AWG lead (battery → pod along the gun's left side and forend underside, + short branch to the trigger actuator), DC5521 mating jack, adhesive cable clips/tape

### Battery box
- 2× hook-and-loop / webbing cinch strap, 25 mm wide × ~450 mm (around the stock + box, in the lid channels)
- 4× M3 brass heat-set insert Ø4.0 × 5–6 + 4× M3 × 8 countersunk flat head ISO 10642 (lid → box; heads sit flush, REV E.3)
- 1 mm closed-cell foam ~200 × 150 (cavity lining); 2 mm closed-cell neoprene ~100 × 70 (box-base-to-stock pad)
- 1–2× cable tie 2.5 mm (lead strain relief through the tongue slots)

### Trigger guard clamp + arming trigger (identical for both trigger sets)
- 4× M3 brass heat-set insert (hole Ø4.0; 5.0–5.2 long)
- 2× M3 × 20 countersunk, ISO 10642 (upper cross screws, head in trig_clamp_R) — REV G: countersunk so no heads sit under the trigger finger
- 2× M3 × 16 countersunk, ISO 10642 (lower cross screws)
- 2× 1 mm rubber shim ~30 × 8 (between plates and guard bow sides)
- 1× 7 mm momentary NO pushbutton (the user's yellow-cap E429-type, M7 bushing) + its hex nut (on the housing's front face, visible; the washer is not used)
- CA glue or 5-min epoxy (arm_pad onto the button cap — glue only the cap, keep it off the bushing)
- 9.5 mm (3/8") open-end wrench for that nut
- ~0.5 m 2-core lead (switch → pod rear cable exit), heat-shrink, adhesive cable clips

### PQ12 set
- 1× Actuonix PQ12-100-12-S
- 1× Ø3 × 12 stainless dowel pin (carriage ↔ rod-end)
- 1× Ø3 pin, ~14 long (PQ12 rear clevis in the slotted bracket ear; slot gives ±3 mm fire-position tuning) [length INFERENCE: ear outer face to clevis far face = 13.1 mm]
- 1× 2 mm TPU/silicone pad 7.8 × 3.9 (contact face)

### MG996R set
- 1× TowerPro MG996R + stock single-arm horn
- 4× M3 brass heat-set insert (hole Ø4.0 × 6) + 4× M3 × 8 SHCS (servo flange; omit the rubber grommets)
- 1× servo centre screw (stock) — or M3 × 16 countersunk if the shaft is tapped M3
- 2× M2 × 10 SHCS + 2× M2 washer + 2× M2 nyloc (lever ↔ horn; drill the horn arm Ø2.1 at r = 9.0 / 12.5 mm if no stock hole lands there)
- 1× 2 mm TPU pad 5.9 × 3.5 (crown pocket)

### Safety pendant — superseded (REV F), no parts needed
