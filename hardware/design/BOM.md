# TBOT — print list + hardware BOM (REV G)

Geometry and placement: hardware/design/layout.md. Verified Ender 3 G-code: hardware/print/ender3_0.4mm/
(one .gcode + .3mf + .print.json per part, all print_ready, no supports). STEP: hardware/step/parts/*.step (units mm, gun
frame — re-orient in the slicer to the build direction below if you slice yourself).

## Printed parts

| Part | Material | Build direction (up) | Supports | Sliced mass / time | Notes |
|---|---|---|---|---|---|
| cam_mount_lower (pod body, open front, rails) | **ASA** | **+X — stands on the rear bulkhead face** | none | 93.7 g / 12.4 h | barrel contact; REV F orientation |
| pod_cap (front cap: camera + IMU) | **ASA** | +Z (Z=−8 face on bed) | none | 13.4 g / 1.9 h | |
| pod_sled (Pi + XL4015 sled) | PETG | +Z (plate on bed) | none | 25.3 g / 2.7 h | slides into the pod |
| batt_box (battery tub, left side of buttstock) | PETG | base down (gun +Y up) | none | 41.9 g / 4.0 h | |
| batt_lid (outer lid with strap channels) | PETG | inner face down (gun +Y up) | none | 18.9 g / 1.7 h | |
| trig_bracket_L (PQ12 left plate + cradle) | PETG | +Z | none | 18.4 g / 1.9 h | PQ12 set |
| trig_finger (PQ12 carriage/finger) | PETG | −X (rear face X=+26.7 on bed) | none | 3.9 g / 0.7 h | PQ12 set |
| trig_clamp_R (right counter-plate + arming-trigger housing, REV G) | PETG | +Z | none | 14.1 g / 1.6 h | shared by both trigger sets |
| servo_bracket_L (MG996R bracket) | PETG | +Z | none | 20.8 g / 2.5 h | servo set |
| servo_lever (MG996R lever/finger) | PETG | −Y (outboard plate face on bed) | none | 3.0 g / 0.4 h | servo set |

Not printed: cam_mount_upper, imu_clamp_upper (REV C split caps), imu_clamp_lower (REV D muzzle IMU — blocked the
camera), pendant_body / pendant_cap (REV F — the arming switch moved onto trig_clamp_R).
Slicer settings: PrusaSlicer Creality Ender-3 presets, 0.4 nozzle; pod, brackets, finger, lever, clamp at "strength"
(4 walls, 35 % infill), cap/sled/battery box/lid at "balanced" (3 walls, 15 %). ASA: there is no Ender-3 ASA preset,
so those two files use Generic ABS at 255 °C / 100 °C — print them inside an enclosure or draft shield (open-frame
Ender 3 + ASA warps). PLA is not acceptable for the pod body or cap.

ASA: ≥4 perimeters, enclosure, no part cooling at first layers. PLA is not acceptable for the pod body or cap.

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
- 4× M3 brass heat-set insert Ø4.0 × 5–6 + 4× M3 × 8 SHCS (lid → box)
- 1 mm closed-cell foam ~200 × 150 (cavity lining); 2 mm closed-cell neoprene ~100 × 70 (box-base-to-stock pad)
- 1–2× cable tie 2.5 mm (lead strain relief at the nose shelf)

### Trigger guard clamp + arming trigger (identical for both trigger sets)
- 4× M3 brass heat-set insert (hole Ø4.0; 5.0–5.2 long)
- 2× M3 × 20 countersunk, ISO 10642 (upper cross screws, head in trig_clamp_R) — REV G: countersunk so no heads sit under the trigger finger
- 2× M3 × 16 countersunk, ISO 10642 (lower cross screws)
- 2× 1 mm rubber shim ~30 × 8 (between plates and guard bow sides)
- 1× 7 mm momentary NO pushbutton (the user's yellow-cap E429-type, M7 bushing) + its hex nut (on the housing's front face, visible; the washer is not used)
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
