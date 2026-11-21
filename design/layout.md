# TBOT — current layout contract

All dimensions are in millimetres and angles in degrees unless stated otherwise.
The single assembly is `cad/asm/tbot_full.step+`; its front module, trigger set,
references and harness are shown at their installed positions. Moving parts are
specified at rest unless a motion angle is stated. Dimensions marked
[INFERENCE] are estimates, not measurements or physical validation.

## Gun frame and datum

- +X points along the bore toward the muzzle, +Y toward gun-left, and +Z up
  toward the rib. The frame is right-handed.
- Origin: the trigger finger-face contact point at rest, on the gun mid-plane
  Y = 0. All installed positions and axes below use this datum.
- The gun model uses a Weatherby Element Synthetic 12-gauge, 26-inch-barrel
  reference: length of pull 14 5/8 in (371.5), comb drop 1 5/8 in (41.3),
  heel drop 2 1/4 in (57.2), and nominal overall length 46 3/4 in (1187).
- The reference side image was calibrated at 1.35 px/mm; the 28-inch reference
  image's overall length and length of pull agree within 0.3%. The model's
  visible edges are approximately ±1.5 mm; other photo-derived features may be
  ±2 mm because of perspective. Guard-specific uncertainty is stated below.
- Do not mistake the reference model or a simulated clearance for a measurement
  of the actual firearm. Gun-contact interfaces use compliant material and/or
  adjustment; measure the real part before relying on fit.

## Gun reference geometry

### `cad/refs/gun/gun_body.step+`

- Receiver: X -35..178.5, Y ±19, Z 16..76.6; its bottom is Z 16 and its rear
  top slopes to Z 68 at X -35. Preserve the loading-port access keep-out
  X 70..165, |Y| < 12, Z 1..16, and the crossbolt safety keep-out X -20..-3,
  Z 3..19 at any Y. The crossbolt is D8 at X -11.5, Z 11.
- Barrel axis is Y 0, Z 62.0. Outside diameter is 27.0 at X 178.5 (not
  measured), 21.5 at X 500 and 22.5 at X 819, the muzzle. The 21.5 / 22.5
  values are caliper readings: just ahead of the forend, and at the tip. The
  IMG_0462 scan puts the muzzle at X 819 ±2, 8 longer than the earlier
  model's 811. The forend-nose barrel underside is Z 51.25. Vent-rib plate
  Y ±3.5 is Z 74.6..76.6 and runs to the muzzle. The fibre sight is X 807..817,
  Z 80.2.
- Vent-rib posts (rev Z, measured from the user's ruler photo of 2026-11-09):
  posts 17.3 long, windows 56.1, pitch 73.4. Scale is 147 px/in from the steel
  rule. Position is anchored on the forend-nose top edge, and the magazine-cap
  face cross-checks within 0.4. In this (module/layout) frame the posts start at
  X 224.9, 298.3, 371.7, 445.1, 518.5, 591.9, 665.3, 738.7 and 812.1 (last one
  to the muzzle). In `gun_body` / the assembly, which sit +10 in X from this
  frame, they are `RibPost1..9` at 234.9, 308.3, ... 822.1. Only the three
  posts near the forend (371.7, 445.1, 518.5 here) are in the photo; the
  others repeat the pitch. Estimate ±2. The window ends are rounded, so at the
  barrel surface a post can be up to about 2 longer than modelled. The
  photo also suggests the window is 3..4 tall rather than the modelled 1.4;
  the plate height was kept (worst case for the band).
- Forend nominal span X 178.5..484. Its side-profile points (X,Z) are
  (178.5,10.5), (260,9), (440,8.5), (462,9.5), (469.5,10), then the sloping
  nose to (484,55), (330,58.5), (178.5,62). Plan half-widths are 19.0 at
  X 178.5, 22.5 at X 260, 21.5 at 420, 19.75 at 440, 19.25 at 458..472, and
  approximately 18.75 at 476, where the nose face cuts it.
- The actual contact model is a ruled loft of teardrop sections. Each section
  is a lower semicircle R about (Y 0, Zc) with straight sides converging to
  (±Wt, Zt). The stations (X: R, Zc, Wt, Zt) are:
  - 178.5: 19, 29.5, 15, 62;
  - 260: 22.5, 31.5, 15.5, 60.5;
  - 420: 21.5, 30, 15, 57;
  - 440: 19.75, 28.25, 14, 56.5;
  - 458: 19.25, 28.75, 13, 56;
  - 492: 18.3, 28.3, 11, 55.
- The barrel-channel cone is centred on Z 62, radius 14.5 at X 175 and 12.5
  at X 495. The magazine cap is D28, axis (Y 0, Z 30.5), X 466..503.5,
  with R5 front edge; its tube is D19, X 178..466. The sling stud has a D5
  shank X 503..508.5 and D6.5 eye X 508.5..511.5, on that same axis.
  Its exposed 5.0 shank length is adjustable by backing out the stud.
- Scan source (IMG_0462 Gaussian-splat session `re_9509f878`). Scale comes
  from the ChArUco mat, 169.37 mm/unit, 0.38 mm RMS. The gun shape comes from
  a silhouette visual hull of the 101 posed frames (`scratch/gunscan/`). The
  hull is eroded by mask-edge blur, so its surface is offset outward 1.6. That
  offset is calibrated against the two barrel caliper readings, with ±0.2
  residual at both. Widths are reliable to about ±0.5. The gun lay rib-down
  and no view was below 23° elevation, so the forend underside, cap underside
  and the region between cap top and barrel are upper bounds only. The bottom
  profile above is kept from the calibrated side photo. Below Z ~20 at X
  440..466, the hull is up to 2–4 narrower than the semicircle sections, so
  the real underside may be more V-shaped: liner gap there is a maximum, not
  a fit. Cap face X 503.5 and stud tip X 511.5 are ±2. Both are confirmed by
  the model silhouette projected into the low side frames (old model: cap 7
  long).
- The upper forend section above Z approximately 40 and cap flute depth remain
  uncertain [INFERENCE]. The stud axis may be up to 1.5 above the cap axis:
  the scan puts its centre at Z 31..32.

### `cad/refs/gun/guard_measured.step+`

This is the fit reference for the trigger set. Since rev AH the guard inside
`gun_body` (f171/f6/f175) uses the same profile, so the two overlap exactly;
their overlap in the assembly is two models of one guard, not two parts.

- Profile measured 2026-11-18 from a taped side photo (10.6 px/mm, fitted to
  the outer edge) and cross-checked against the 2026-11-05 scan mesh to about
  1 mm. Widths are unchanged from the earlier caliper readings.
- Width: 9.3 at the narrowest, 15.7 at the widest. Half-width h(X) = 4.65 for
  X ≤ 12, rising linearly to 7.85 at X 50, then 7.85. The taper location is
  [INFERENCE].
- Opening: floor Z −13.2 (X 3..10), ceiling Z 12.2, so 25.4 high. Rear wall
  vertical at X −7.2 from the ceiling down to Z −1, then curving into the
  floor. Front wall leans forward: X 27.9 at Z −10.1, 33.9 at −4.5, 36.5 at
  0.9, 38.2 at 6.3, 39.6 at the ceiling.
- Outer profile: bottom Z −16.5 at X 6.5..13, then Z −14.6 at X 26, −12.3 at
  33, −9.4 at 40, −7.0 at 45, −3.4 at 51, 0.0 at 57, 3.8 at 63, 9.8 at 72,
  reaching the receiver (Z 16) at X 81. The rear outer profile is unchanged.
- A trigger slot X −10..13, Y ±4, Z 11..17 is cut through the guard top
  [INFERENCE: not visible in the photos].
- The old 2 mm underside bevel no longer builds on this profile and is
  suppressed. It was never caliper-confirmed. Without it the model is fuller
  than the real guard, so clearances measured against it are conservative.
- Guard-side liner is 1 mm neoprene, compressed to 0.85 on each side; the
  intended land face follows h(X) + 0.85.

### `cad/refs/gun/gun_trigger.step+`

A deep crescent, not a straight blade (measured 2026-11-18 from the same
photo). Moving reference, Y ±3.5. Front face (X,Z), tip to root: (4.6,−11.5),
(3.4,−10.2), (2.0,−8.5), (0.2,−4.8), belly (−0.5,−1.0), (0.3,2.8), (2.9,6.6),
(6.7,10.3), (9.1,11.7), (11.5,12.6). Rear edge: (3.6,−11.6), (−0.4,−8.5),
(−3.0,−4.8), (−3.5,−1.0), (−3.2,2.8), (−1.0,9.0), (1.5,12.2). The tip hooks
5 forward and ends 1.7 above the floor. Above the ceiling the outline to the
pivot is [INFERENCE]. Y-axis pivot at (X −4, Z 26) [INFERENCE]; positive
rotation about +Y is pull. Model break 7.6°, hard stop 9.4°
[INFERENCE, carried over]; pull approximately 27 N, design load 35 N.

## Front module

### Envelope and load path

The printed module occupies X 396.0..531.6 and Y -51.8..+50.4 (102.2 wide;
rev Y moved the battery-lobe flank out 1.6 and its front forward 7.6 for the
battery hold-down, the Pi lobe is unchanged). The Pi
lobe starts at X 428 and the nose clamps and carrier MI-1 blocks at X 422.45
(rev Z); only the battery lobe continues rearward
to X 396.0 as the buck bay (rev U). The lobes are Z -21.2..42.8, with the
belly keel reaching Z -22.4 (65.2 overall). Rev Y: only the battery lobe stands
ahead of the clamp nose (cover_R X 531.6, carrier_R lobe X 530.0, still behind the
camera_cover shroud front X 532.3); cover_L X 524.0, carrier_L / belly X 522.4,
clamp nose X 524.4. Only the camera itself (lens tip X 544) stands ahead of the box.
The solid-PLA model of the seven printed parts is 231.0 g at 1.24 g/cm³
(rev U); the print files have not been regenerated since rev R.

The two clamping halves grip the lined forend and cap, capture the sling stud,
and provide the datum for the camera, the IMU and both carriers. The two hose
clamps sit in grooves cut into the clamp halves and wrap the clamp directly
(barrel → clamp top corners → flank grooves → underside grooves), so band
tension squeezes the halves together at Y 0 and pulls the clamp up onto the
forend; B2 wraps the nose root right behind the camera. The bands no longer
pass through the carriers or the belly. Carriers hang on MI-1 (4 × M3) and are
tied across by the belly; servicing the electronics never loosens a band, so
the camera zero is not disturbed. Rev U puts the buck bay behind X 428 on the
gun-right side only (X 396..428, Y -25.8..-50.2, beside the forend, at least
1.2 from the gun model); the gun-left side behind X 428 stays free. Recoil load is reacted by the forend taper,
cap/front wall and captured stud; jaw contact and the bands resist roll.
The clamp has at least approximately 11 mm air to the hot barrel. Keep PLA
away from the barrel; the bands use the specified sleeve, not bare PLA contact.

The lobes have 1.6 walls: inner wall |Y| 25.8..27.4, 0.2 off the clamp flank;
outer flanks Y ±48.6..±50.2 on both lobes; top Z 41.2..42.8; bottom
Z -21.2..-19.6; Pi-lobe rear wall X 428..429.6; battery-lobe bulkhead
X 428..429.6 and bay rear wall X 396.0..397.6; cover front walls
X 522.4..524.0 on both sides. Carrier/cover split is at Y ±40.0/±40.15.
The Pi connector stack sets the Pi lobe (17.4 above the 3.0 standoff gap,
which the FFC riser uses); the 20-thick pack against the inner wall sets the
battery lobe. The bands are SAE #44 (sized to the new, shorter wrap; the case
was not enlarged for band size).

### `nose_clamp_L` and `nose_clamp_R`

- Split at Y 0; L is gun-left and R gun-right. Main body begins X 426.3 (rev Z:
  a 13.7 rear collar so B1 sits in the rib window behind the post at X 445.1
  and has a rear retaining lip), ends at the camera nose
  X 524.2, top Z 40 and nose bottom Z -10.2. Outside flank is |Y| 25.6 (24.2 in
  the band grooves). The rear corners are R1.6 (the other corners R3) so the lip
  keeps its thickness at the bottom. Side wall over the forend is at least 3.9
  including the 0.3 gun gap; 2.7 remains from the MI-1 jaw-insert bore to the
  B1 groove flank and 1.6 from the floor-insert bore to the B1 groove floor.
  Where B1 crosses the clamp top (Z 38..40), the wall between the gun cavity
  and the band tangent line is only 0.72. The forend is wider there, but that
  wall is backed by the liner, so it is in compression only.
- Band seats (rev Z): all clamp material outside the band inner path is removed
  over X 428.1..441.4 (B1; a 1.8 lip X 426.3..428.1 closes the rear and the
  body closes the front, so the channel has a wall on both sides) and
  X 501.85..515.15 (B2). Grooves are 1.4 deep on the flanks (|Y| 24.2) and
  B1 underside Z -1.6 (B2 underside kept at Z -9.2, 1.0 deep, to leave a skin
  under the camera pocket). Both bottom corners are R8 and the top corners R6
  (they were R3: a 0.6 steel band cannot follow R3 at a 90° turn, bridged the
  corner and bowed out, and at R3 the band pressure on the PLA is about
  T/(R·w), roughly 2.7 times that at R8). The top corner follows the band's
  tangent line (see "Hose clamps"). The FPC slot has R0.5 lower exit edges.
- B2 saddle rib: solid X 501.65..514.5, |Y| ≤ 25.6, Z -10.2..-3.0 on both
  halves under the nose root, so B2 wraps the camera end of the clamp; it fills
  the rear of the FPC slot, leaving the FFC drop open at X 514.5..520.4.
- Nose walls are thickened to |Y| 16.3..20.4 over X 514.5..524.2 (4.1 walls,
  previously 2.1) around the camera pocket.
- Each gun cavity is the forend loft offset 0.8 in-plane (rev Z; was 1.2: the
  printed clamp showed visible gaping). Sections are the rev V sections moved
  0.4 inward along their normals, plus a first section at X 425.3 (linear
  extension of the 431/440 sections) for the rear collar. Offset section
  (X: Ro, Zc), approximate: 435: 20.99, 28.69; 440: 20.55, 28.25; 445: 20.05,
  28.75; 492: 19.10, 28.30. Cradle-floor low points are Z 7.7 at X 440 (the
  forend underside is kept, raised 0.4). The nose bulkhead rear plane is
  X = 467.449 + 0.301333 × (Z + 0.346).
- Gun-to-cavity gap over the gripped forend (X 426.3..476, flanks) is 0.76..0.80
  by construction (rev V: 1.16..1.20 against the rev V gun_body; not re-measured
  against the real forend). Rev U's cavity gave a median 1.95 at
  X 440..452, too loose for the liner. At X 464..476 it went down to 0.03,
  where the halves could not close over the liner.
- A 1.0 adhesive neoprene/EPDM liner against the nominal 0.8 cavity gap is
  compressed 0.2 (20%) with the halves butted at Y 0 (the 1.5 sheet of rev V is
  now too thick). Check the real forend;
  0.5 sheet or tape shims may be needed for actual dimensions. Approximate
  liner pieces per side: upper 40 × 14, lower wrap 40 × 30, nose pad 26 × 20;
  cap cradle strip approximately 48 × 42. The underside below Z ~20 at
  X 440..466 may be narrower than modelled (see gun_body), so expect a
  thicker shim there rather than at the flanks.
- The cap cradle is D30.4, X 469.5..504.0. The front wall is X 504.0..507.5,
  3.5 thick, and the cap face (X 503.5) is 0.500 behind it. Rev U had the
  wall at 511.0..514.5 for a cap the scan shows does not exist. The stud bore
  is a D5.6 obround along +X, axes (Y 0, Z 30.5) and (Y 0, Z 32.0). It gives
  0.300 radial clearance to the D5 shank anywhere over Z 30.5..32.0, because
  the scan places the stud axis up to 1.5 above the cap axis. Use the M5
  washer, D5.3/D10 × 1.0, between wall and eye. The 5.0 shank accommodates
  the 3.5 wall, 1.0 washer and approximately 0.5 stud adjustment.
- The eye pocket is open to the top and front: X 507.5..513.0, |Y| < 9,
  Z ≥ 24.8, then X 513.0..514.6, Z ≥ 26.0, joining the open nose above the
  camera-pocket roof (Z 24.3..26.0). The roof plate keeps 1.5 of solid
  behind it. Over |Y| 9..25.6 the old wall material remains to X 514.5, so the
  B2 top-corner seats are unchanged.
- Two M3 × 20 ISO 10642 cross screws run along Y at (X 456.0, Z 4.5) and
  (X 505, Z 2.6). R has D7 head pockets to Y -14 and D3.4 through holes;
  L has two D4.0 × 6.5 M3 insert bores opening at the split face. Screw #1 is
  clear of B1 (7.0 from the B1 groove edge to the bore); screw #2 sits under B2 (loosen B2
  to re-torque it).
- MI-1 is the rear face at X 426.3 (rev Z; was X 440): four M3 insert bores
  D4.0 × 6.5 (7.5 deep from X 425.3) along X, at (Y ±12.5, Z 2.0) and
  (Y ±19.5, Z 12.0). Four M3 × 10 countersunk screws enter from the carriers'
  X 422.45 face (both carrier MI-1 blocks are X 422.45..426.3, 3.85 thick; the
  insert bore ends at X 432.8). Behind the clamp the carrier_L block joins its
  lobe through `MI1RearWeb` (X 422.45..429, Y 25.8..28.5, Z -2..37, R1.5 root
  fillet). The carrier_R block runs into the buck-bay inner wall (profile to
  Y -26.8), and `ClampCollarRelief` clears the old block region
  X 426.3..431.9.
- Front carrier screw (one per side): M3 × 8 ISO 10642 countersunk along ∓Y at
  (X 495.5, Z 3.2), between the IMU window (Z 8..12) and the underside, pocket edge 2.45 clear of
  the B2 groove edge (X 501.85). The carrier inner wall carries a D7.0 × 1.0 spigot
  (R0.3 root, R0.4 tip) on its clamp face, which seats in a D7.8 × 1.2 pocket in the
  clamp flank (|Y| 25.6 → 24.4); the spigot locates the carrier and gives the
  countersink (M3, D6.7 × 1.65) 2.6 of wall. The D4.0 × 6.5 M3 insert bore runs
  |Y| 24.4..17.9, so the tip stops 5.0 into the insert. The head is flush on the
  Pi (L) / battery (R) face of the inner wall: take the Pi or pack out to reach it.
  Features: carrier `FrontClampSpigot` + `FrontClampCountersunkM3`; clamp
  `FrontCarrierSpigotSeat` + `FrontCarrierM3Insert`. Checked: assembly interference
  clear; verify_battery pass on all four parts. Not tested: screw pull-out or recoil load.
- The underside trough is |Y| ≤ 8.5. Its roof is Z 1.6 over X 441.4..459.0
  (rev Z: it starts at the front edge of the B1 groove and no longer runs over
  B1 or out of the rear face, so the collar and B1 lip are solid). The IMU
  bundle comes up from the belly cavity into the trough ahead of B1 (0.9 web to
  the screw #1 insert bore). The trough is Z -3..-0.8 at X 500..514.5. The floor pocket between
  X 459..500 accommodates the IMU and leads. There are no integral wire-tie
  mounting points.

### Camera and IMU seats

The innomaker OV9281 camera PCB is 32 × 32 × 1.6, with four D2.2 holes on a
28 × 28 square. The lens-holder screw centres are Y ±10 at the board centre;
the J1 connector is on the back bottom edge, 23.1 wide. Back components extend
up to approximately 2 and the lens-screw heads are approximately D3.8.

- Camera ref is at (520.4, 0, 8), no rotation: PCB X 520.4..522.0, Y ±16,
  Z -8..24. The pocket is X 514.5..524.1, |Y| ≤ 16.3, Z -8.3..24.3,
  open to the front and accommodating the connector and rear components.
- Four D4 corner bosses run X 514.5..520.4 at (Y ±14, Z 22) and
  (Y ±14, Z -6), with D1.7 × 6.0 pilots. Four M2 × 8 pan self-tappers
  pass through `camera_cover` and the PCB corner holes, approximately 4.8 of
  thread engagement.
  The FPC exit is X 510.5..524.3 (open through the nose front so a clipped-in
  FFC slides in with the camera), |Y| ≤ 9, Z -13..-3; R0.5 on the rear lower
  exit edge (X 514.5, Z -9.2).
- Lens holder (innomaker manual V1.4 + measured): 17 × 17 square, 4.85 above
  the PCB front, then a D14 round boss 4.85 tall; lens screws at Y ±10.
- `camera_cover` (cad/parts/camera_cover.step+): 32 × 32 × 1.6 plate on the PCB
  front (X 522.0..523.6, 0.3 to the pocket walls) with a shroud over the
  holder (Y ±14.5, Z -2.4..18.4, front X 532.3). Holder pocket 17.6 square ×
  5.1 plus ear pockets to Y ±12.9; the roof tapers 43° from vertical so it
  prints plate-down without support; D14.6 hole for the boss/lens. Remove the
  lens to fit it if the lens will not pass the D14.6 hole.
- Assembly: `camera_cover` is component k80 in `cad/asm/tbot_full.step+`, placed with
  the same +10 X offset as the camera (`cam_inno9281`, k30); the four camera
  screws are now `m2x8_cam1..4` (k81..k84, `cad/refs/fasteners/screw_m2x8_pan.step+`),
  head on the cover front X 533.6, tip X 525.6. Their 4.18 mm³ overlap each with
  the nose pilots (D1.7 vs D2) is the intended self-tap.
- The GY-521 PCB is 21.2 × 16.4 × 1.6, X 472.5..493.7, Y -8.2..8.2,
  Z 0..1.6, components up. It seats on the R half under the magazine cap.
  Two M3 × 6 SHCS enter up through D3 board holes at (475.0,-5.7) and
  (491.2,-5.7), into D4.0 × 6.5 insert bores from Z 1.6 to 8.1. Screw heads
  occupy Z -3..0. The L header relief is X 473.0..493.2, Y 0..8.5,
  Z 5..11; a fitted header has 0.8 clearance, tails down to Z -3.

### `carrier_L` and `cover_L`

- The left carrier supports the Pi 3 Model B right side up: its solder side
  faces the carrier inner wall and its components face the cover, with
  USB/Ethernet forward. The ref is rotated -90° about X and translated to
  (436.85, 30.4, 40.0): PCB Y 30.4..31.8, components reach Y 47.4 (1.2 to the
  cover), microSD (on the solder side) Y 29.0..30.4. Board Z -16..40. GPIO is
  on the bottom edge (pin 1 at X 445.2, odd row Z -11.27, even row Z -13.81,
  pin tips Y 40.3); HDMI, micro-USB and audio are on the top edge; CSI is at
  X 479.9..485.4, Z 16..38.5, opening +Y; microSD is at the rear.
- Standoffs: four solid blocks rooted in BOTH the inner wall and the adjacent
  top or bottom skin (not cantilevered pegs). Blocks run Y 27.4..29.9 with a
  0.5 high D6 pad to Y 30.4 under each hole (the Pi D6.2 keep-out), X from
  hole -5.7 (45° print chamfer on the -X side) to hole +3.2; top blocks
  Z 33.3..41.2, bottom blocks Z -19.6..-9.3. Holes at X 440.35/498.35,
  Z 36.5/-12.5 take M2 heat-set inserts (bore D3.2 × 4.0, Y 30.4..26.4, 0.6
  skin left in the wall); four M2 × 5 machine screws from the component side.
  Clearances: insert bore to M2 shank 0.6 radial; bottom-front block ends
  1.2 ahead of the FFC pass; blocks clear the GPIO tails (X 445.2..493.5)
  by 0.35 on each side [INFERENCE: real tail positions].
- The 3.6 gap between the board's bottom edge (Z -16) and the floor (Z -19.6)
  is the wire channel to the component side. Inner-wall passes: GPIO /
  Pi 5 V pass X 444..462, Z -19.6..-15.6, R1; camera FFC pass (rev U, moved
  up to the camera's level) X 502.25..519.35, Z -12.2..-10.6, R0.5 on its four
  long edges. The old combined opening, capacitor cradle and B_L junction are
  gone from this lobe.
- Top-skin port notches, open to the cover split (Y 41): micro-USB
  X 441.5..453.5, Y 30.4..41; HDMI X 458.85..478.85, Y 29.8..41; audio
  X 486.85..493.85, Y 31.3..41 (all R1). The rear-wall microSD slot is
  X 427.9..429.7, Y 27.5..31.9, Z 4..20. Confirm plug overmolds fit.
- The front-hooked cover attaches with two M3 × 8 ISO 10642 countersunk
  screws along -Y through the flank at (X 433.4, Z 30.0) and (X 433.4,
  Z -11.0) into M3 heat-sets (D4.0 × 7.0) in two posts grown off the rear wall
  (X 429.6..437.4, Y 41.5..48.6, Z 26..34 and -15..-7). Countersinks 90°,
  D6.3 × 1.45 deep in the 1.6 flank, as on cover_R. The hook is now one
  continuous finger Z -10..36 (the window print-relief cuts are gone).
- The front window profiles (cover_L front wall, R1 corners) follow the flipped
  board: Ethernet Y 31.0..46.2, Z 20.7..38.3; USB1 Y 31.0..48.2, Z 3.7..18.4;
  USB2 Y 31.0..48.2, Z -14.3..0.4. The window roofs bridge 12.7..17.6 in the
  -Y flank-down print orientation.

### `carrier_R` and `cover_R`

- Battery hold-down (rev Y): a 3-point friction clamp. `batt_retainer`
  (cad/parts/batt_retainer.step+, 2.2 PLA plate, Y -49.6..-47.4, lightly
  skeletonised Y-frame, 0.8 chamfer on the pack face) lies flat on the
  pack's outboard face, touching only the cell crowns (nothing enters the
  valleys between the shrink-wrapped cells). 3 x M3 x 8 ISO 10642 pull it
  along +Y into M3 heat-sets in carrier_R bosses whose outboard faces are at
  Y -46.2 (bores D4.0 x 6.5): A rear-top (X 447.4, Z 35.0) and B rear-bottom
  (X 447.4, Z -13.0) in the outboard pocket between the jack/cap and the cell
  ends, C front-middle (X 525.55, Z 11.6) ahead of the cells. A is a D7.2
  post rooted in the inner wall; B is a D7.2-topped boss on the floor with a
  web to Y -30.3 (top Z -14, kept out of the rear lobe pass where wire s60
  runs at Z -14; bore B is 6.1 deep with a 0.7 floor so wire s26 keeps 0.35);
  C is a 6.9 x 7.2 block X 522.25..529.15, Z 8.0..15.2 cantilevered from the
  inner wall (R2.8 long-edge rounds, R0.35 wall root at the rear so the pack
  keeps its gap). C's rear face is the pack's forward/recoil stop (0.4 gap,
  middle cell). No full-height front bulkhead: in the +X print orientation
  it was a 672 mm2 ceiling. With a 20.0 pack against the inner wall the plate
  clears the boss faces by 1.2, so the screws set the clamp force; packs down
  to 18.8 thick clamp without shims. Plate to cover_R inner flank 0.6 (0.25
  at the corner arcs). M3 x 8 tips stop at Y -41.4, 4.8 of thread in the
  insert. Print supports: carrier_R (+X up) now also needs small supports
  under the A post, the B boss and the C block -X faces (about 194 mm2 of
  overhang from the new features; the C block's support stands about 92 mm
  on the bulkhead inside the open battery cavity).
- The 3S 2600 mAh pack is back against the inner wall (rev U): vertical,
  approximately 70 × 58 × 20 for the cell pack, with the lead block included
  in X 441.8..521.9, Y -48..-28, Z -17.4..40.6 (ref at (486.85, -38, 11.6)).
  Leads exit rearward at Z 11.6. The R11 saddle is Y -30.4..-38 with its 45°
  lead-in at the inner wall. Rev Y: the pack is clamped against the inner
  wall by batt_retainer (assembly places it 0.55 inboard of the rev U
  position); no cover-side foam.
- Buck bay (rev U): the lobe continues 32 rearward to a new rear wall at
  X 396.0..397.6 (`BuckBaySection` / `BuckBayBlank`, `BuckBayCavity`). The old
  rear wall X 428..429.6 stays as a bulkhead that roots the mid cover bosses.
  Its top and bottom are cut for the cover lips and its outboard edge is cut
  back to Y -47.2 so the cover flank and its inner chamfers pass
  (`BulkheadCoverFlankRelief`). Bulkhead passes: upper Y -36.5..-29.5,
  Z 33..40 (capacitor leads); lower Y -48.6..-44.4, Z -11.5..-4.5 (buck #2
  feed).
- Buck #2 (HW-083, terminal blocks omitted, leads soldered) stands in the bay
  with its long axis vertical, solder side to the inner wall and components
  outboard: transform [0,-1,0,424.9; 0,0,-1,-30.4; 1,0,0,-11.0], PCB
  X 398.6..424.9, Y -30.4..-32.0, Z -11.0..40.0; components to Y -45.0 (3.6 to
  the cover), solder envelope to Y -28.9, 1.0 to the bay rear wall. The
  trimpots face forward to X 427.5 (0.5 to the bulkhead) at Z 3.5..14.5; turn
  them from outboard with the cover off. Four D4.0 × 3.0 posts come off the
  wall at X 400.4/423.1, Z -9.25/38.25. The two top posts take M2 × 5
  self-tappers from the component side (D1.7 × 4.4 pilots); the bottom two are
  plain pads. The reference's 1.5 solder envelope box overlaps the post seats
  by 2.4 mm³: check the real board's solder around the holes. IN pads are at
  the bottom: IN+ (X 406.4, Z -7.35), IN- (X 411.5, Z -7.35). OUT pads are at
  the top: OUT+ (X 415.0, Z 35.95), OUT- (X 420.1, Z 35.95). Pad polarity is
  inferred from the terminal footprint.
- The 1000 µF / 16 V servo capacitor stands vertically above the jack, in
  front of the bulkhead: can X 431.5..441.5, Y -38.2..-28.2, Z 14..34, leads
  up (+ at X 439.0, − at X 434.0). It sits in a snap clip rooted on the inner
  wall: X 431.1..441.9, Y -35.6..-27.4, Z 20..28, bore D10.4, arm tips 9.2
  apart (0.4 interference per side). Its leads cross the bulkhead's upper
  pass to the OUT pads.
- Inner-wall passes: rear lobe pass X 442.5..451, Z -19.4..-11.0, R1, between
  the jack and the cells. It carries the battery singles at Z -13 and the
  buck #2 feed at Z -16.3/-18.3. The cross-screw access openings at
  (X 456.0, Z 4.5) and (X 505, Z 2.6) are unchanged and sit behind the pack:
  take the pack out to reach them. Servo exit: D4.8 hole through the bay rear
  wall at (Y -38.0, Z 38.4).
- The DC5521 inline charge jack is back at (X 436.5, Y -33.5): vertical, mouth
  down, body Z -19.7..8.3. The D11.2 bore has a 0.5 × 45° lead-in; mouth is
  recessed 1.5. A 1.6 shoulder at Z -5.55..-3.95 resists plug insertion; the
  1.3 cover lip (Y -40.15..-38.65), 0.2 below the mouth, resists removal.
- cover_R spans X 397.75..524.0. Rev Y: its flank is Y -50.2..-51.8 over the
  whole length and its front wall is X 530.0..531.6. It has a front hook at
  X 527.6..531.6 behind the inner wall's front end (as on cover_L) and three M3 × 8 countersunk
  screws into carrier heat-sets: two vertical at (X 433.4, Y -43.5), top and
  bottom, into the bulkhead bosses, and one bottom screw at (X 401.4,
  Y -43.75) into the bay boss X 397.6..405.2. There is no top screw at the bay
  rear because buck #2's output capacitor fills that corner. Install: hold
  2.4 forward, push inboard, slide back.

### `belly_cover`

- Belly floor is Z -22.4..-20.8, 1.6 thick, X 428.0..522.4, with a front wall
  X 520.8..522.4, Y -17.5..25.6, Z -20.8..-10.6 (carrier_R's front tray boss
  closes the Y -25.6..-17.5 corner). Its three M3 × 8 countersunk screws
  enter upward at (433.0, ±21.5) and (518.9, -21.5), into D4.0 × 6.5 carrier
  heat-sets. The FFC belly ramp is deleted (rev U): the camera ribbon now
  crosses under the clamp at Z -11.6..-11.0 and no longer runs on the belly
  floor. The rev S wire pocket and the suppressed chin features are deleted.
- Buck screws enter 2.2 posts and blind D1.7 pilots; nominal tips at Z -22.0
  leave approximately 0.3 skin under the holes. Tray bosses and lever have
  1.2 raised seats. The rear-wall outer top edge is R1 for the harness.
- The lever connector sits at (462.0, 0, -12.1), rotated 120° about
  (0.5774, 0.5774, 0.5774); approximately X 452.5..471.5, |Y| ≤ 20.5,
  Z -19.6..-4.6. End rails are notched X 456..462 to Z -18 for lower
  entries at Z -15.95. It is the removable battery disconnect.
- Both HW-083 XL4015 boards use the accurate reference: PCB 51 × 26.3 × 1.6,
  D2.3 holes on 47.5 × 22.7 centres, capacitors D10.4 × 13, terminals
  9.9 × 12.7 × 10 (omitted on both boards), trimpots overhang 2.6, 14.6
  maximum component height above the board, and an approximate 1.5 solder
  envelope below.
- Buck #1 supplies the Pi at 5.1 V. It is crosswise in the belly, rotated
  +90° about Z, with its terminal blocks omitted and leads soldered to its
  pads. It stays within |Y| 25.5 and is retained with two M2 × 5 screws at
  (X 473.6, Y ±23.75). Pilot-wall minimum is 1.0. The other two holes rest
  on plain pads; the inaccessible terminal/part area is not a fastener point.
- Buck #2 (7.0 V servo supply) is in the battery lobe; see `carrier_R`.
  `chin_lid` is obsolete (since rev S), no longer in the assembly, and has been removed from `cad/parts/`.
- The covers' approximately 2.4 hook stroke does not collide with the belly;
  the belly can stay installed while a cover is serviced.

### Hose clamps B1 and B2

- Two SAE #44 worm-drive clamps (1/2" band, nominal 59..83 working diameter;
  check the actual brand's range), 12.7 × 0.6 band. B1 band X 428.4..441.1
  (rev Z: 11.9 back from the original X 440.3..453.0, which landed on the real
  rib post at X 445.1..462.4; 0.25" back was not enough, see "gun_body");
  B2 band X 502.15..514.85. Worm housings (17.0 × 7.0 + flanges) are on the
  gun-right upper straight run, 7.4 down the run from the barrel tangent;
  housing to lobe tops (Z 42.8) at least 2.5. Trim and deburr the tail after
  tightening.
- Inner path (both bands, rev Z): barrel arc R 11.764 about (Y 0, Z 62); tangent
  lines from (±10.822, 66.613) to (±23.720, 36.353); R6 top-corner arcs about
  (±18.2, 34.0); vertical inner faces |Y| 24.2; R8 bottom-corner arcs; bottom
  inner face Z -1.6 (B1) / -9.2 (B2). Inner length 210.8 (eq. D 67.1, B1) and
  226.0 (eq. D 71.9, B2), mid-range for #44. The bands seat in the clamp's
  grooves (1.4 deep on the flanks and B1 underside; outer face 0.8 below the
  flank/underside) and touch only
  the clamp, the barrel sleeve and nothing in the carriers or belly; the
  carrier inner walls sit 1.0 outside the bands. The `band_B1` / `band_B2`
  reference models follow this path.
- Use a 0.5 sleeve over the barrel under the rib; band-to-rib plate clearance
  is 0.236 in the model. With the calipered barrel, sleeve radius is about
  11.7 at B1 and 11.26 at B2. That leaves B2's R 11.764 barrel arc about 0.5
  slack radially, which band tension takes up, so no band change is needed.
  Rib windows (rev Z, from the measured posts): B1 sits in the window
  X 389.0..445.1 with 4.0 to the post ahead. B2 sits in the window
  X 462.4..518.5 with 3.65 to the post ahead and 39.8 behind. Groove play
  (0.3 each side) leaves at least 3.7 (B1) / 3.35 (B2) in the worst groove
  position.
  B1 forward (1.25") would centre it in B2's window. It was rejected because
  B1 would pass through the IMU header pins (X 474.2..492.0) and buck #1, which
  sit 1 apart in Z under the clamp. If the real post edge at the barrel turns
  out closer than 2, move B1 further back. That means moving the collar,
  carrier MI-1 blocks and band ref together (same features as rev Z).

### Electrical architecture and safety

```text
3S pack -- SPL (+ charge jack) -- lever connector -- S1 -- buck #1 (5.1 V) -- Pi
                                                       +-- buck #2 (7.0 V) -+- 1000 µF (at OUT pads)
                                                                            +- servo pair -> R rear exit -> servo
Pi 3V3 / SDA / SCL / GND ---- B_R ----------------------------------------- GY-521
Pi GPIO18 PWM / GPIO17 / GND - B_R -- 3-wire exit -------------------------- servo signal, arming switch
```

All grounds are common: XL4015 IN- and OUT- are common, as are pack negative,
Pi ground and servo ground. The jack branches at SPL on the PACK side of the
lever connector, so charge access remains live when the system is disconnected.
Set buck outputs before connecting their loads; confirm terminal pitch and
polarity on the actual boards. Pi power goes to physical pins 2 (5 V) and 6
(GND). IMU VCC/SDA/SCL/GND use pins 1/3/5/9. Servo PWM is BCM18, pin 12;
the normally-open arming switch is BCM17, pin 11, to ground pin 14, active
low with pull-up. Signal acceptance at 3.3 V by a servo powered at 7.0 V is
[INFERENCE] and needs a bench check.

**The arming interlock is software only. There is no hardware interruption of
the servo supply when the button is released. Open the battery lever connector
to make the actuator electrically safe.** The trigger harness is hardwired,
with no inline disconnect. Two cables leave the module: the 3-wire signal
exit from B_R across the belly rear wall (modelled to X 413) and the servo
power pair through the buck-bay rear wall at (Y -38.0, Z 38.4), modelled to
X 381. Both runs to the trigger set are routed separately.

### Harness model and wire routing

`cad/refs/harness/` models each straight segment as a cylinder and each bend as a
sphere, rather than a fused sweep. Bundle diameters: 20 AWG single 1.9, servo
22 AWG pair 3.2, cap leads 1.6, 26 AWG single 1.1, four-wire IMU 2.7,
seven-wire GPIO 3.3, three-wire exit 3.0. Wires that pass under the Pi's
bottom edge (Pi 5 V) are modelled as singles because the channel is 3.6 high.
Minimum intended bend radii: 1 at the header stubs, 2.5..3 singles, 5 pairs
and bundles; no sharp corners. Actual wire outside diameters and bundled fit
remain to be checked.

Junctions: B_R (435.5, 12.0, -6.0) in the belly rear (GPIO 7 in → IMU 4 +
exit 3 out); SPL (438.5, -31.0, 11.6) in the battery lobe rear; S1
(470.0, -23.0, -10.5) in the belly; GPIOBreakout (449.0, 41.5, -16.3) at the
header. The B_L junction and the nine-conductor trunk are gone. The jack pair
joins SPL locally. Header stubs touch their neighbours at the 2.54 pitch in
the model; that is the soldered header, not a routing clash.

Cut lengths below are modelled module paths plus 30 service length, in mm;
add the separately routed external trigger-set runs where indicated.

| Circuit | Conductors | Route | Cut lengths |
|---|---|---|---|
| Battery | 2 × 20 AWG | Pack → SPL over the jack → forward at Y -29.3/-31.3, Z 11.6 → down at X 450/448 → rear lobe pass (Z -13) → belly (along X 449..451) → lever +Y end | 147 / 147 |
| Charge jack | 2 × 20 AWG | Jack boot → local SPL | 35 each |
| Supply | 2 × 20 AWG | Lever -Y end → S1 | 47 each |
| Buck #1 input | 2 × 20 AWG | S1 → buck #1 input pads | 46 / 42 |
| Buck #2 input | 2 × 20 AWG | S1 → aft along the belly wall at Y -24.4 (Z -18.3 / -16.3, under the battery singles) → rear lobe pass (X 445.7/443.7) → outboard and up → aft at Y -46.8 through the bulkhead lower pass → bay floor → IN pads | 133 (+) / 129 (−) |
| Pi 5 V | 2 × 20 AWG | Buck #1 OUT pads → up over the board end → belly at Y 23.6/24.5 → L pass → under the Pi edge (Z -17.6) → up the component side at Y 44.3/45.4 → pins 2 / 6 | 108 / 99 |
| Servo 7 V | 2 × 22 AWG | Buck #2 OUT pads (top) → outboard → bundle at (413.5, -35.6, 38.4) → aft above the board edge → bay rear-wall exit | 51 (+) / 54 (−) in module, + external run to the MG996R from (X 396, Z 38.4) |
| Capacitor | 2 × 22 AWG lead extensions | Capacitor leads (top, Z 37) → bulkhead upper pass → OUT pads | 58 (+) / 48 (−) |
| IMU | 4 × 26 AWG | Pins 1 / 3 / 5 / 9 → breakout → GPIO bundle → L pass → belly → B_R → raised trough above B1 → IMU | 126 / 132 / 129 / 130 |
| Arming + PWM | 3 × 26 AWG | Pins 11 / 12 / 14 → breakout → GPIO bundle → B_R → 3-wire exit | 106 / 105 / 107 in module, + external run |

### Camera ribbon

Rev U takes the FFC sideways out of the camera straight into the Pi lobe; it
no longer drops into the belly. From camera J1 (back bottom edge, X 519) it
drops 3.5 to Z -11.3 inside the clamp's FPC exit. It then bends aft flat
under the clamp nose (Z -11.3..-11.0, 0.6 under the clamp floor, 1.2 under B2)
and 45° creases to +Y over X 502.75..518.75. It runs straight across to the Pi
lobe at Z -11.6..-11.3 through the carrier_L FFC pass
(X 502.25..519.35, Z -12.2..-10.6).

Inside the lobe it has to go round the Pi board. The CSI faces the cover, and
the USB / Ethernet block fills the board's front end (Y 31..47.4,
Z -14..38.3), so the only way to the component side is over the board's top
edge. The ribbon stands up flat behind the board (Y 27.6..27.9,
X 502.75..518.75), folds over the top edge in the 0.6 gap under the top skin,
and comes down outboard of the Ethernet jack (Y 45.0..46.5). It turns aft with
a 45° fold at Z 19.25..35.25 and plugs into the CSI along -Y at X 482.6. These
last segments are unchanged from rev T.

The modelled centreline is approximately 165, down from 181: the belly
descent, floor crossing, belly ramp and the 60-high riser from the floor are
gone. That is still too long for the 150 kit cable. **Use a 200 FFC** and
Z-fold the approximately 35 excess flat in the riser gap behind the board
(1.6 between the inner wall and the solder tails). Model checks: no FFC
contact with printed parts except the two connector entries. Physical folds,
contacts and connector seating remain open.

## Software-relevant poses and optics

- **As-built camera:** `cam_inno9281` translation (520.4, 0, 8), no board
  rotation; optical axis +X through (Y 0, Z 8). J1 is at the bottom edge.
  Lens tip approximately X 546.2, derived from the reference's 25.8 projection
  from the board datum (lens length estimated from photos; focus-dependent). The entrance pupil / optical centre is assumed at approximately
  (540, 0, 8) [INFERENCE]; calibrate its real location. Since rev T nothing
  printed is ahead of X 524.4, so the housing no longer depends on it.
  Fisheye nominal focal length 2.8, sensor 3.84 × 2.4, diagonal field of view
  148° and horizontal 118°. The barrel is visible and must be masked.
  Software uses a 1280 × 800 nominal sensor at 3.0 µm pitch and a default
  640 × 400, 120 fps capture mode.
- **As-built IMU:** translation (483.1, 0, 0). The model has no board rotation
  (board axes = gun axes), but the sensor chip as installed is NOT: measured
  2026-11-15 with `tbot calib-imu axes` (three hand-held poses, fit within
  0.8 deg of an exact axis swap), gun +X = chip -Y, gun +Y = chip -X,
  gun +Z = chip -Z. Chip-to-gun matrix R_gun_imu = [[0,-1,0],[-1,0,0],[0,0,-1]].
- **Software nominal camera frame:** camera +x = gun -Y, camera +y = gun -Z,
  camera +z = gun +X. Thus nominal `R_gun_cam` has rows [0,0,1], [-1,0,0],
  [0,-1,0]. Mount error roll/pitch/yaw prior is [0,0,0]°; camera image
  orientation still needs confirmation from a real frame.
- **Physical barrel-mask geometry:** model axis Y 0, Z 62.0, muzzle
  (819, 0, 62.0), outside diameter 21.5 at X 500 and 22.5 at X 819
  (calipered; muzzle position from the IMG_0462 scan, ±2).
  It differs from the current software defaults; do not treat those defaults
  as the geometry of this assembly. Mask margin remains 6 pixels in config.
  `optics.py` samples the barrel in 2 mm axial steps with at least 24 rings
  and 72 angular samples, discarding incidence angles of 89° or greater.
- The configured 115 ms servo command-to-break value is a loaded 4.8 V
  estimate [INFERENCE], not a measurement of the current 7.0 V setup. Measure
  command-to-break on the actual assembly before using it as a timing model.
- Software servo defaults are 50 Hz, rest 1670 µs and fire 1330 µs; these
  pulse values are not a measured mapping to this assembly's rest-to-+31°
  lever motion. Verify pulse direction and endpoints on an unloaded bench
  setup before relying on them.

## MG996R trigger set

### Servo, horn and motion

- `cad/refs/actuators/mg996r` is photo-derived, approximately ±0.7, cross-checked against
  the TowerPro side-view drawing (ear underside 26.6 above the case bottom).
  The case is X -6.3..34.4, Y 5.65..25.35, Z -51.3..-16.0. Ears span X
  -12.95..41.05, Z -24.7..-21.7; D4.6 ear holes are X -10.7/38.8, Y 10.5/20.5.
  The gear cover is Z -16.0..-13.6: rectangular from the shaft to X 28.9, with
  an R9.85 round end concentric with the output. The output has a stepped
  collar, as on the real servo: D13.6 × 1.2 (Z -13.6..-12.4), then a D9 ring
  to Z -10.4. The D5.9 spline is Z -10.4..-6.3 at (X 3.6, Y 15.5), pointing
  +Z. Its M3 tap is modelled D3 × 6. The rear cable stub is D3.6.
- The measured stock horn hub is D11.2 × 3.5, Z -9.2..-5.7; its arm is
  8.0 wide × 2.5 thick, Z -9.2..-6.7, with R4 tip 19.5 from the axis.
  D1.6 holes are at radii 8.2/11.2/14.3/17.3; spline bore D6.0 spans
  Z -9.25..-6.2 with 2.9 engagement. Arm bearing at rest is 190°.
- Lever hinge axis is +Z through (3.6,15.5); lever motion is in the horizontal
  plane. Contact occurs at approximately +13.3° from rest, with the trigger
  held at -9.07° (4.15 pull) at +31°; guard rear leg is met near +34°.
  Modelled crown is 2.77 ahead of the trigger at rest. The approximately
  15.5 arm and 35 N design load give 0.54 N·m / 5.5 kg·cm; estimated torque
  margin 1.7× at 4.8 V and 2.0× at 6 V. The 31° travel is estimated at
  88 ms / 72 ms no-load at those voltages [INFERENCE].

### `servo_bracket_L`

- PLA bracket, gun-left of the guard, printed on its inboard face (+Y).
  The top frame is X -16..44.5, Y 5.5..27.5, Z -12.8..-9.7, tying the
  front and rear servo ear blocks. The blocks finish at X -7.1 and 35.2,
  nominally 0.8 from the case ends. Gear-step clearance is 0.8. Root
  reinforcement radii are R3 front and R0.8 rear.
- Output clearance is a plain round D16.4 through-hole (no counterbore, no
  ceiling over the output collar): radial clearance is 1.4 to the D13.6
  collar and 3.7 to the D9 bearing ring, so the collar can never bear on the
  frame and bow it when the ear screws are tightened. The earlier D15.6 ×
  1.2 counterbore (0.8 axial) still let the real collar bear on the frame.
- Arm relief: the frame top is lowered from Z -9.7 to Z -10.9 inside an R21
  arc about the output axis, clipped at X 11.5 (clear of the upper block and
  guard recess). The horn arm (R4 tip at 19.5) keeps 1.7 at its modelled
  seat and 0.5 even if pressed down onto the D9 bearing ring (Z -10.4).
  Frame thickness there is 1.9; the front ligament beside the hole is 1.8.
  The guard taper recess goes from Y 5.5
  to h(X) + 0.85, with floor profiled to the measured guard and approximately
  0.85 measured gap. The L40 region has floor Z -10.4.
- Rev AA guard-clamp joints are through-bolted, no inserts: D3.5 holes on the
  U18, U28 and L40 axes run from the clamp face to trapped M3 hex nuts (5.7 AF
  pockets, flats horizontal). U18: nut seat Y 12.9, side slot open to −X into
  the arm relief, exit channel cut through the cove (X 13..21.8, Y 15..22.5,
  above the frame top). U28: seat Y 12.9, drop-in slot open to the top.
  L40: seat Y 13.3 (0.9 from the front-inboard ear insert), side slot open to
  +X. Rev AB: the L40 bore now runs through the whole bracket (to the back
  face Y 27.5, through the seat shim) so a longer screw can pass; a 5 × 7.5 rib
  (X 37.5..42.5, Y 20..27.5, top Z -8.5) caps the front ear block where the
  bore crown (Z -9.65) would otherwise break its top (Z -9.7). The old
  insert bores are filled from Y 8 (D4 counterbore remains below the recess
  floor). The plastic between the guard face and each nut is in compression.
- Rev AB rear joint (X −15.5, Z −17.0): the rev AA rear spacer nub (the R4.7
  boss, the spacer to Y −1.5 and its nut slot), which snapped on the range after
  two shots, is gone. A D3.5 hole now runs through the whole rear ear block,
  Y 5.5..27.5, with a 90° countersink in the back face. An M3 × 35 countersunk
  screw goes in from the back (head flush at Y 27.5) into a nut in the
  trig_clamp_R_v2 guard cradle (tip Y −7.5). The bolt clamps 22 mm of solid
  bracket, so there is no cantilevered pad left to break. Thinnest web: 0.52
  between this bore and the rear ear inserts (X −10.9, Y 10.5 / 20.5); after
  ironing those inserts, run a 3.5 mm drill through the bore. Fillets f100,
  f116, f117 are suppressed: f100/f116 no longer rebuild on this tree.
- Four M3 × 10 countersunk screws still hold the servo ears into heat-set
  inserts, one in each ear hole.

### `servo_lever`

- PLA, printed -Z. The horn hub seats at Z -5.7 in a D11.8 pocket, 0.3 per
  side. The straight arm slot is 8.6 wide, 0.3 per side, with roof Z -6.5,
  0.2 above the horn arm. Its tail is open; the pocket walls remain at least
  2.0. Body hub radius R7.6, tail arc R6.3 centred 9.04 from the axis.
- A single M3 × 12 countersunk centre screw through lever and horn enters
  the output shaft; head is flush at Z 1, tip Z -11, approximately 4.7
  engagement. M3 × 14 can bottom. The horn arm in the pocket carries torque.

### `trig_clamp_R` and arming button

- Right-side PLA plate carries the measured-guard land and the circular button
  extension. Its Y plate faces are -11.5..-8.5; the inboard lip reaches Y -2.
  The front guard land follows the plate slope below its top, through XZ
  (34.10,-1.15) to (41.6,-3.93), then to the front face at Z -7.5.
- Four countersunk screws pass from the right plate to the bracket nuts:
  M3 × 30 at U18 (X 18.5, Z -7.75), U28 (28.0,-3.0) and L40 (40.0,-11.4),
  heads flush with the hole pad at Y -13.9 (rev AA removed the 2.4-deep D6.8
  head pockets that left ~1.3 of plastic under each head; now ~3.7), and
  (rev AB) the rear M3 × 35 now comes from the bracket side (see
  `servo_bracket_L`). Rev AC (history rebuild) removed the old X −10.5 rear
  hole and its D7 rear-head access teardrop through the button web, which had
  no bolt since rev AB, along with the rev AA spacer block and the buried lip,
  bottom-rib, back-rim-rib and rear-boss bodies.
  The U18/U28 countersink-to-countersink web is 4.3; U28 hole edge is 1.62
  above the opening.
- Rev AB guard cradle (anti-slide lock): a solid block from the plate's inner
  face (Y −8.5) to Y 0.5, which is 5.0 short of the bracket face (rev AF;
  it was Y 5.0, 0.5 short, through rev AD). It spans
  X −19..44.5, from the bed (Z −22.3) up to the outer surface of the guard bow.
  - Its top follows the bow's outer spline with about 0.5 clearance (it was
    measured at 0.27 against `guard_measured`) and is capped at Z −9 at both
    ends.
  - The bow therefore sits in a fitted saddle. The front face of the saddle,
    which rises from X 13 to 40, bears on the bow, so the trigger set cannot
    slide back toward the trigger even if the friction fit loosens. It does not
    rely on clamp friction.
  - The cradle carries:
    - the rear M3 nut in a side slot (open to −X; nut Y −6.6..−4.0, bearing at
      −4.0; D3.5 bore Y −4..0.5);
    - the L40 bore, which leaves only about 0.2 of skin over its crown at the
      cradle top. That skin is not load-bearing.
  - Lever plane (Z ≥ −8.2) and trigger clear the cradle top by ≥ 0.8 and ≥ 2.1.
- The bottom-front corner is trimmed by an R25.4 arc centred at XZ
  (27, 2.5), from the bed at X 32.5 to the front face at Z -15.9. The arc
  is 3.0 from the L40 countersink. Near the bed the arc is about 12° from
  horizontal, so its first millimetre prints as a shallow overhang; it is a
  cosmetic edge.
- The E429 cap is pressed directly; the switch is normally open. Cap D5.5 ×
  4.2, M7 bushing 8.6, measured flange D9.4, body D8 × 9.3, hex nut about
  9.5 across flats. Travel is estimated 1.5..2.0 [INFERENCE].
- Cap tip at rest is (X -4, Y -21.5, Z -7). Its axis points forward and
  down 10°, u = (cos 10°, 0, -sin 10°); press back and up. The panel is 3.0
  thick normal to the axis, with a round D7.4 hole and concentric R8.3 collar;
  wall is 4.6 at the side and 4.3 above the hole. The switch fits thread-first
  from behind, with the nut on the front.
- The panel root has a full-height R6 concave structural web from the bed to
  Z 1.0, and an R3 outside corner on the other side. A coaxial D12 × 2.5
  relief in the web clears the nut; the R6 web is trimmed to the collar and
  45° foot. The guard rail ends at approximately Z 2, leaving manual finger
  space above it. No integral cable-management anchors are present.
- Rev AC feature tree (rebuilt from 86 features to 46, no heal, no transform
  or Boolean tool bodies). In order:
  - plate (trigger scallop included in its sketch), guard land plus taper cut,
    guard cradle, then one R25.4 front-bottom trim cut for all three;
  - plate-outline R4/R3/R2 fillets;
  - the button module, sketched on `DatumButtonFace` (origin on the button
    axis, normal 10° down);
  - the bed flat, hole pad, holes, rear bore and nut slot;
  - the R6 web;
  - the bottom and scallop ribs (`rib_h` = 2.4, same height as the pad);
  - the trigger-edge R0.75 and the hood dress-ups.
  The scallop rib's ends are buried in the button flange and the hole pad. The
  old rib-end cap that collided with the R6 web (and its 0.1 step to the pad)
  is gone. The trigger edge is R0.75: R1.5 no longer builds on this tree. The
  flange-front R1 is dropped because it forces the R6 web smaller. Equations:
  `plate_t`, `rib_h`, `flange_t`.
- Rev AD (solid/mesh fix plus blends). Rev AC's scallop rib came after the
  holes and filled the U18 countersink. Its hole pad also ran 0.001 mm inside
  the top rail, and the rib footprint sat exactly on the R25.4 trim; both left
  sliver faces that would not mesh, so STL export failed or produced an open
  mesh that slicers reported as non-solid. Now:
  - all bosses (pad, bottom rib, scallop rib) come before the bed flat, the
    R25.4 front-bottom trim and the holes;
  - the bottom rib is drawn oversize (R26.5, down to Z −23) from
    `DatumY_PadRoot` and is cut to shape by the bed flat and the trim;
  - the pad meets the top rail with flush flats, non-tangent corners and a
    1 mm inset along the slanted rail;
  - root fillets: R1 on the scallop rib and bottom rib, R0.95 on the pad.
  Rib tops stay crisp: the kernel can't close those corners against the
  flange. The result is one closed solid with no sliver faces or tiny edges,
  and Chudware's STL export passes its facet check. Kernel hazard: exactly
  coincident tangent arcs between the pad and the plate fillets hang the
  boolean indefinitely.
- Rev AE (clamp gap for a thinner guard). The real trigger guard looks
  slightly thinner than `guard_measured`, so the guard cradle (f13 depth
  13.5 to 12.5) now stops at Y 4.0, 1.5 short of the bracket face at Y 5.5.
  The screws can then pull the plate up to 1.5 closer before the cradle
  bottoms on the bracket, so a thinner guard is still gripped rather than
  the cradle carrying the clamp load. Volume 15640.6 mm³, about 19.4 g. FEA
  (PLA, M3 bores fixed, 50 N on the button panel along the press direction,
  2.5 mm tet10): 4.7 MPa peak away from the fixtures (fatigue limit
  18 MPa), 0.027 deflection; the 18.8 MPa raw peak is the clamped rear-bore
  singularity.
- Rev AF (outer-face cleanup and rounded ribs).
  - Cradle: the hand-made 3.5 cut behind the cradle is folded into f13
    (depth 9.0). The cradle now stops at Y 0.5, 5.0 short of the bracket
    face, so the clamp can close up to 5 on a thin guard. The rear nut has
    4.5 of plastic behind it (Y −4.0..0.5).
  - Hole pad lower edge: drawn oversize (R26.2, tangent line Z −23.7) so the
    bed flat and the R25.4 trim cut it. This removes the 1.6 ledge and the
    stub step between the pad and the trim near the L40 hole; the pad's
    front edge is still the exact X 44.5 line. The bottom rib is back to
    its 303° end, now buried in the pad.
  - Rounded rib tops: the pad, the bottom rib and the scallop rib are built
    as their own bodies (merge off), pad+bottom rib fused and rounded
    R0.6 on the whole top rim (f56), the scallop rib rounded R0.6 (f57),
    then fused together (f58) and into the plate (f59) before the bed flat.
    Rounding after the fuse fails where the rib tops meet the flange wall.
    The R6 web (f43) now comes before the pad.
  - Scallop rib outer radius 26 to 25.2 (4.7 wide) so it crosses the pad's
    R5.6 notch 2.0 from the notch's tangent point instead of 1.2.
  - One root fillet R0.8 (f49, `RibPadRootR0_8`) for the scallop rib, pad
    and bottom rib floor edges; the separate f51/f52 root fillets collided
    with it. Vertical rib-to-pad junction edges are left sharp (rounding
    them ran into the R0.6 tops and left slivers).
  - Front-bottom corner round f42 is R0.9 (R2 no longer fits against the
    rounded pad edge).
  - Result: one solid, 144 faces, no slivers or tiny edges, every face
    triangulates, closed STL; 13815 mm³, PrusaSlicer 18.9 g with no mesh
    repairs. FEA as rev AE with a finer local mesh (130k tet10): 40 MPa
    at the sharp floor corner of the button nut relief (f33), where the
    load face ends. That corner was not changed; the coarser rev AE mesh
    did not resolve it. It is a 50 N case; a 10 N press scales to about
    8 MPa.
  - Follow-up: R0.3 on the nut-relief floor edge (f60,
    `NutReliefFloorR0_3`, last in the tree). It only occupies R5.6..5.9
    next to the floor; the 9.5 AF nut's corners reach R5.5. The same FEA
    now peaks at 13.7 MPa on the flange face (FoS 3.6 on yield, under the
    18 MPa endurance limit at 50 N). Slice: 18.8 g, no mesh repairs; the
    G-code floating-extrusion check went from warn to fail (one 7.4
    unanchored outer-wall run where the new round meets the unsupported
    pocket roof); look at the nut pocket roof in the PrusaSlicer preview.

### Rev AG — front jack, rigid spacers, M3 × 30 rear (anti-slide)

Problem: on the gun the trigger set crept back toward the trigger. The plates
squeeze a guard that narrows toward the rear (9.3 → 15.7), so the clamp force
itself pushes the set rearward. The cradle's front face sits under the
bottom-front slope of the guard, which is only about 22° from horizontal, so
the set could ride down and back off it.

- Front jack (`trig_clamp_R_v2` f65, `RevAG_FrontJack_D2_8`): a D2.8
  thread-forming bore through the cradle. Its axis is normal to the guard's
  outer front slope at (X 34, Y −4, Z −11.5), direction (0.378, 0, −0.926).
  An M3 × 12 countersunk screw goes in from the clamp's curved front-bottom
  face (2 mm hex, from under the gun). The thread it cuts in the PLA is the
  lock. The tip pushes back and up into the guard's forward-facing slope, so
  the reaction drives the whole set forward and down, away from the trigger.
  The set stops when U28 meets the front inside wall of the guard opening
  and the U18 spacer meets the opening floor. The bottom-front of the guard
  ends up clamped between those two (inside) and the screw tip (outside). To
  move rearward, the set would have to push the screw into the guard. The
  bore clears the L40 hole by 2.4. The head stands about 2
  proud of the face.
- `u18_spacer` (new part): a D5 × 11.6 tube, bore D3.3, on the U18 bolt
  between the two taper lands (X 18.5, Z −7.75). It fixes the
  plate-to-bracket gap at the front, so tightening the bolts no longer
  squeezes the tapered guard there. The 1 mm liners take up side play. Lever at
  rest: 2.4 clear.
- Rear joint M3 × 35 → M3 × 30, done in the bracket only: a D6.4
  counterbore from the back face (Y 27.5) down to Y 23.4, then a new 90°
  countersink seat (`servo_bracket_L` f145/f146). The head sits 4.1 deeper,
  so the M3 × 30 tip lands at Y −6.6, through the unchanged nut in the
  clamp's guard cradle (nut Y −6.4..−4.0). The clamp's rear joint is
  unchanged. The thinnest web around the new counterbore is 0.55, at the
  bracket's rear face edge, the same as the old flush countersink. It stays
  more than 0.5 clear of the RI/RO ear insert bores.
- `u18_spacer` is the only rigid spacer. The rear joint keeps the rev AF
  5 mm cradle-to-bracket gap.
- The servo and lever move with the set (about 0.4 forward, 1 down once
  jacked). Re-run the servo REST/FIRE endpoint calibration after installing.
- Assembly: `fhcs_m3x12_front_jack` (tip 0.06 off the modelled slope),
  `u18_spacer`, `fhcs_m3x30_rear` (head at Y 23.4), `nut_m3_rear` (original
  seat). The only overlaps are the
  jack screw in its D2.8 bore (8.5 mm³, the thread it cuts) and the existing
  `gun_body`/`guard_measured` pair. Clamp 13.8 cm³; slice 13.6 g / 1.7 h. Bracket re-sliced: 11.7 g / 1.8 h,
  build-plate organic supports as before. The
  G-code check flags a 13.6 mm anchored overhang perimeter at 0.4 mm on the
  existing shallow R25.4 front-bottom arc, a cosmetic edge. Spacer 0.3 g,
  printed standing up with a brim.

### Rev AH — servo output inside the guard, finger cam, plug-and-saddle mount

Replaces `servo_bracket_L`, `servo_lever`, `trig_clamp_R_v2` and `u18_spacer`
(rev AG and earlier stay in the repo for reference, suppressed in the assembly).
New parts: `servo_mount_L`, `btn_mount_R`, `trig_cam`, `jack_shoe`.
`cad/asm/tbot_full.step+` shows rev AH with all hardware; `cad/asm/trigger_set_AH.step+`
is the four parts plus the servo.

Guard and trigger geometry: see `cad/refs/gun/guard_measured` and `cad/refs/gun/gun_trigger`
above (both measured 2026-11-18).

- Safety button: in the guard's rear web, about X −16.4..−7.9, Z 2.3..12.2.
  Nothing in rev AH is behind X −7.4 above Z 0.3 on the left, or behind X −7.0
  above Z 1.5 on the right.

Servo and cam:

- MG996R on gun-left, output axis along −Y, shaft at (X 10.5, Z 1.2). Pose of
  `cad/refs/actuators/mg996r`: rotate +90° about X, then translate (6.9, −3.6, −14.3). Case
  X 0.6..41.3, Z −8.65..11.05, Y 12.4..47.7; gear cover top Y 10.0; ear seat
  Y 18.1; spline tip Y 2.7. Case top is 1.15 below the receiver underside.
  Rev AH-3 moved the servo 2.0 forward (+X) so the upper rear ear screw fits
  (all four ears are now screwed); the servo's rear ear is at X −6.05.
- `trig_cam`: Y ±4.6, hub R6.9, finger nose R2.0 centred 12.0 from the shaft
  (rev AH-3: was R3.0 at 10.3; the longer finger recovers most of the
  leverage the forward shift costs). The stock single-arm horn sits in a
  pocket on the +Y face: hub seat R5.75 × 3.3, boss bore R4.6 × 4.3, arm slot
  8.8 wide out to 9.6 from the centre, M3 × 12 centre screw, one M2 × 8 into
  the horn's first hole (7.5). **Cut the horn arm 9.3 from the centre, just
  past the first hole** (the arm is 8 wide; longer will not fit). The slot
  leads the finger by 20° on the side away from the trigger, inside a R5.9
  lobe; thinnest wall 0.94 at the slot's outer corners.
- Angles are measured at the shaft from the rearward horizontal, negative
  below it. Modelled at −72.5° (finger 17.5° behind straight down, nose 0.6
  clear of the trigger hook). First contact −66.75°, on the lower hook at about
  (4.2,−11.1); the nose then slides up the blade as it sweeps it back.
  Modelled break (7.6°) at −2.5°, trigger stop (9.4°) at +3.25°. Past the stop
  the cam would only load the trigger against its stop. Lowest point of the
  sweep is 1.05 above the floor; highest 8.1.
- Rest to break is about 64° of servo travel from first contact. The rest
  angle is set in firmware, so put REST just before contact and dead travel
  costs nothing. Peak servo torque in the model is 0.27 N·m at the 35 N design
  load before friction (0.20 at the old shaft position; the MG996R is about
  0.9–1.1). Forward of the trigger, every millimetre costs leverage, but the
  4th ear screw needs the 2 mm. **Re-measure the trigger delay** and set it
  in the config before live fire.
- Set the FIRE endpoint just past the break (about 0°), never at the trigger
  stop.

`servo_mount_L` (24.0 cm³, 25 g sliced):

- Chamfer along the inner top edge (Y 5..9.5, Z 9.8..12.5) clears the
  receiver's V underside by about 0.5.

- Slab Y 5.5..18.1 with the ear seats on its outer face, servo pocket floor at
  Y 9.7 (web 5.5..9.7), D16.4 hole for the output collar. Guard land tapered
  to h(X) + 0.85.
- Plug, Y 5.5..−4.0, fills the front of the opening from X 18.7 to the front
  wall with 0.5 clearance, top Z 9.5. Recoil bears on it directly.
- Saddle, Y 5.5..−4.0, runs under the bow and front slope with 0.7 clearance
  down to Z −24.4. The bow is enclosed between plug and saddle, so the set
  cannot come off even if every screw loosens.
- Jack: M3 × 10 countersunk through a trapped M3 nut (side slot from the −Y
  end), axis normal to the outer slope at (33.0,−12.3), direction
  (−0.353, 0.936). It lifts `jack_shoe` (9.2 × 8.0 × 3.0) against the bow, which
  pulls the plug down and forward into the corner between the floor and the
  front wall. The screw never touches the gun.
- Inserts (M3 × 5.7, D4.0 bores), all installed from +Y: three in the pocket
  floor for P1 (31.5,1.0), P2 (22,−7.5), P3 (22.5,6.0); four in D7 recesses
  (floor Y 13.5) for U1 (8,−20.3), U2 (24,−19.0), F1 (46,−10.8), F2 (59,−3.2);
  four in the ear seats at (−3.8,−3.8), (−3.8,6.2), (45.7,6.2), (45.7,−3.8).
  The upper rear insert has 1.6 of wall to the mount's rear edge (X −7.4),
  which stays clear of the safety button.

`btn_mount_R` (17.0 cm³, 20 g sliced): plate Y −5.5..−12.5, land tapered to
−(h(X) + 0.85), seven D3.4 countersunk holes on the same seven axes, D7 access
hole on the shaft axis (10.5, 1.2) for the cam screw, with a teardrop top for
printing. Arming button unchanged in position
and angle (hole centre (−4.556,−21.5,−10.15), axis (0.985,0,−0.174), panel
2.8 net, D10.4 body bore 13.3 long). The plate floats on its screws, so the
squeeze adapts to the real guard width; it closes up to 1.5 before touching
the plug.

Screws: P1..P3 are M3 × 20, U1, U2, F1, F2 are M3 × 25, heads in the right
plate at Y −12.5, tips at Y 7.5 and 12.5.

Checks in `cad/asm/tbot_full.step+` (trigger area, 24 members, rev AH-3): no
interference between printed parts, gun and hardware except the expected ones
(the two guard models; horn spline teeth on the smooth modelled servo spline,
14 mm³; M2 self-tapper in the horn hole; jack thread in its nut). Clearances:
mount to guard 0.50, plate to guard 0.85, cam to trigger at rest 0.61, cam to
guard 0.95, cam to each side 0.9, servo to `gun_body` 1.8, shoe to bow 0.075
(seats when the jack is turned), mount to `gun_body` 0.10 at the safety
crossbolt (D8 at X −11.5, Z 11, modelled to Y 12): the mount's rear-top
corner is right beside the safety head. The servo's rear ear (X −6.05) is
now 2.6 ahead of the safety head (was 0.6). Check that the safety can still be
pushed. Not yet printed or fitted.

## Printing and materials

All ten printed parts are PLA (`chin_lid` is obsolete). Do not leave them in a hot car: PLA can
soften or creep around 55–60 °C. The front-module slices use PrusaSlicer
2.9.6, Creality Ender-3 with 0.4 nozzle, Generic PLA, 0.20 layers, strength
settings of 4 perimeters and 35% cubic infill, and no supports. Their recorded
print metadata specifies 210 °C first-layer nozzle, 205 °C thereafter, and
60 °C bed. Slicing is not evidence of physical strength or fit.

The table below is the rev R slice record. All seven front-module parts changed
in rev S; carrier_L, carrier_R, cover_L, cover_R and belly_cover changed in
rev T, and carrier_R, cover_R, carrier_L and belly_cover changed again in
rev U. Re-slice and re-check every front-module print before printing.
**Rev U print orientation: carrier_R's two rear faces are no longer one
plane.** Print it +X with the buck-bay rear face (X 396) on the bed. The
inboard MI-1 section (MI-1 block, rear tray boss and chin; rev Z: the block
is now X 422.45..426.3) then starts about 26 above the bed: add support under
its X 422.45 face only
(approximately 400 mm², including the two MI-1 countersinks). The bulkhead
bridges three sides. In that orientation the buck #2 posts are 3.0
horizontal cantilevers and the capacitor clip is a horizontal C-ring rooted
on the wall. cover_R is an X-constant C-section and still prints flank down.
Other rev T notes stand: the Pi standoff blocks carry 45° chamfers on their
-X faces, the cover_L posts grow straight off the rear wall, and the M2/M3
insert bores in the carriers are horizontal D3.2/D4.0 holes.

| Part | Build direction / bed face | Slice g / h (rev R) | Recorded result (rev R) |
|---|---|---:|---|
| nose_clamp_L | -Y, outer flank down | 39.6 / 4.2 (rev V) | Rev Z changed this part (not re-sliced); rev V, build-plate supports: G-code pass. Not print_ready: printability rows score the 1.0-deep B1/B2 groove floors (1 above the bed, 19.1 span, now plate-supported) and 0.2 slivers where the B2 saddle rib (X 501.65) meets the B2 groove (501.85) |
| nose_clamp_R | +Y, outer flank down | 41.2 / 4.6 (rev V) | Rev Z changed this part (not re-sliced); rev V, build-plate supports: same as L |
| carrier_L | +X, rear face down | 27.9 / 4.1 | G-code warn: 9.2 air run; 199 mm² model overhang, mainly standoff chins |
| carrier_R | +X, rear face down | 24.3 / 3.7 | G-code pass; not print_ready: 18 mm² countersinks/chamfer and 1.8 notch bridge |
| cover_L | -Y, flank down | 15.2 / 1.6 | G-code warn: 8.1 air run; 14 mm² counterbore roofs/facet |
| cover_R | -X, front face down (rev X; was +Y flank down) | 17.8 / 1.9 (rev R, flank down) | Not re-sliced. Rev X DFM at -X: 55 mm² overhang, 201 mm³ support, vs 212 mm² / 2419 mm³ flank down |
| belly_cover | +Z, floor down | 21.5 / 2.3 | G-code pass; not print_ready: intended 0.3 pilot skin |
| camera_cover | +X, plate face down | 4.5 / 0.6 (rev W) | G-code warn (5.4 air run, 6.5 bridge); not print_ready: 0.9 corner ligament at the M2 holes |

Trigger build directions: `servo_bracket_L` +Z (rev AA), `servo_lever` -Z,
`trig_clamp_R` +Z. The bracket requires support on the build plate for its
guard-recess floor, round output-hole crown and insert-bore ends; the clamp's
round switch-hole and rear-access-hole crowns are short bridges. The trigger
print set has not been regenerated and verified for the current geometry.

Rev V re-sliced only the two nose halves (`cad/print/creality_ender_3_0_4_mm_nozzle/nose_clamp_L|R.*`).
The other front-module print files predate rev U and must be re-sliced before printing.

## Current-geometry verification

- Rev Z (hose-clamp fit, from the first print): the printed bands bowed out of
  their channels at the bottom corners (R3 path corners), B1 landed on a real
  rib post, and the clamp-to-forend gap showed. Changes:
  - `nose_clamp_L/R`: rear collar to X 426.3 (the body was X 440..); B1 seat
    X 428.1..441.4 with a 1.8 rear lip; flank grooves 1.4 deep (|Y| 24.2), B1
    underside Z -1.6, B2 underside kept at Z -9.2; R8 bottom / R6 top band-path
    corners on both seats (`Band1SeatProfile` / `Band2SeatProfile`, still fully
    defined); cavity sections moved 0.4 inward (offset 1.2 -> 0.8) with a new
    first section at X 425.3; MI-1 bores, teardrops and top/band chamfers moved
    to the new rear face; trough now starts at X 441.4; rear corners R1.6. The obsolete `RearFlankEdgeR0p8`
    (rear-rim round of the old open groove) was deleted.
  - `carrier_L/R`: MI-1 block and belly shoulder pad moved behind the clamp
    (X 422.45..426.3); carrier_L `MI1RearWeb` ties the block to the lobe,
    carrier_R `ClampCollarRelief` removes the old block region; a
    `TrayBossClampRelief` cut lowers the tray boss top to Z -3.4 for X > 426.3
    so it stays clear of the clamp collar and B1; MI-1 block fillet targets
    moved. MI-1 screws are now M3 × 10.
  - `belly_cover`: the rear-closure tongue (|Y| ≤ 8.6 up to Z 5.3) is gone, since
    the clamp collar now fills that space. The closure wall is Y ±17.3,
    Z -12..-3.4, and `SignalExitHoleProfile` was rebuilt without its stale
    external reference.
  - `cad/refs/gun/gun_body`: `RibPost1..9` placed from the measured photo (see
    "gun_body"); three placeholder posts deleted.
  - `cad/refs/fasteners/hose_clamp_rear` / `hose_clamp_front` (`band_B1` / `band_B2`): band
    path and worm housing rebuilt on the new path; B1 sits at X 428.4..441.1.
  - Checks run (final): `verify_battery` passes on nose_clamp_L/R, carrier_L/R,
    belly_cover and band_B1. `gun_body` fails only feature_attachment, on its
    pre-existing `SafetyBolt` cylinder. Front-module threshold interference
    (19 components incl. gun_body and the IMU) has no band or clamp overlaps.
    The only pair is carrier_R <-> xl4015_servo (bay wall, X 402..421): the bay
    geometry is identical to the pre-rev-Z backup, so this predates rev Z. Exact
    clearances: gun_body to bands 0.236 (rib plate), bands to carriers 0.9..1.0,
    band_B1 to belly 1.2, band_B2 to belly 6.10, B1 / B2 to the IMU 31.4 / 11.6,
    gun_body to clamps 0.50.
  - `dfm_check` (FDM, Ender 3), clamp L / R: min wall 0.722 / 0.772 (was
    0.898 / 0.906). The new minimum is the cavity-to-band sliver where B1
    crosses the clamp top. Knife edges 0.616 sit at the B1 rear lip by the
    floor-insert bore, area L 154 / R 120 mm². The existing fail rows (min
    wall, overhang, bridges, min feature, sharp internal corners, min pin)
    remain.
  - Not done: print files were not regenerated; the 0.8 gap and the band fit
    were not tested on the real gun. Rib posts come from one photo (±2), and
    the harness study was not re-routed for the IMU bundle's new entry ahead
    of B1.

- Rev X (housing polish, no layout or envelope change): added edge rounds and
  root fillets on carrier_L, carrier_R, cover_L, cover_R, belly_cover and
  camera_cover. Main changes:
  - carrier_L: the scratch `carrier_L3` fillet set was ported in. The inner-wall
    to top-skin root fillet is left off over X 502..522 so the FFC fold keeps
    its 0.2 gap.
  - carrier_R: R1.4 bay rear-wall perimeter; R0.5 seam; inner-wall R1.3/R1.37
    long edges; root fillets on the MI-1 block, rear tray boss, cover bosses,
    buck #2 posts (R0.6), cap clip (R0.6), bay floor (R1.5) and the
    bulkhead/inner wall (R1.2). The bay top corners are limited to R0.47/R0.37
    by the buck posts.
  - Covers: R1.4 rear perimeter; front perimeter R1.59 (the wall-thickness
    limit); R0.5 on the cover_L port-window edges.
  - camera_cover: shroud corners R2 to R5, front edge R1.5, root R1.0, plate
    corners R2.
  - belly: front-wall top R0.6.
  The front-module exact interference (20 bodies) found no new overlaps. Two
  touching pairs (nose_clamp_L / carrier_L, carrier_L / cover_L) now fail the
  OCCT Boolean. With carrier_L rolled back before f108/f110 they pass, and
  those two features are convex rounds that only remove material (-52 / -15 mm³).
  All six parts pass `verify_battery` (carrier_R's SaddleLeadInProfile had a
  redundant dimension, now set to driven). Open risers: the 1.6-long square
  steps at the J-hook ends on both covers (the kernel rejects fillets there).
- Harness (rev Y): study `tbot_front` in `cad/asm/tbot_full.step+` is now routed with the
  harness tools (35 connectors, 25 wires, 0 unrouted; `harness_check` 0 issues at
  0.3 clearance). The old solid bodies k56/k57/k63/k64/k65 are removed from the
  assembly; `ffc_cam_15p` (k59) stays because the ribbon is not in the study. The
  built component `Harness_tbot_front` (k86) is not a sound B-rep (`brep_valid`
  false on 3 sweep bodies), so exact interference cannot use it; clearance evidence
  is the mesh-based `harness_check`. Bundle bend radii were overridden to 1.5..2.5
  (layout intent for pairs/bundles is >= 5, not met). See `design/harness_report.md`.
- Rev Y (housing polish, no layout change): larger edge/root rounds on both
  carriers, covers, belly, camera_cover and nose clamps; cover_R/cover_L lobe
  corner radius R5/R6 with matching carrier_R boss-corner trims; nose clamp B2
  groove start moved 501.85 -> 501.75 (groove 13.4 wide) and nose front 524.2 -> 524.0.
  Covers are intended to print front-face-down; cover_L at that orientation still
  shows 120 mm2 overhang / 53 mm3 support in `dfm_check`. Print files not regenerated.
- Rev Y rear closure: `belly_cover` gets a rear closure wall (X 428..429.6, |Y| <= 17.3,
  Z -12..-3.4; its tongue |Y| <= 8.6 up to Z 5.3 was removed in rev Z) with one
  D5.2 wire hole at (Y 14.3, Z -8.2) for the 3-wire signal exit; `carrier_L/R` get
  `RearGapFill` blocks (X 428..433, |Y| 17.5..26.3, Z -11..-1.8) over the tray bosses. The servo
  pair still leaves through the D4.8 bay hole. Convex rim rounds on the new wall's top edge
  would not build (kernel); the corners are R2 / R1.2 only.

- Rev V (gun_body from the IMG_0462 scan; nose clamp refit): exact
  `gun_body` checks against 34 front-module and trigger-set components found
  only the clamp-to-stud overlap (2.1 mm³ per half, the rev U wall over the
  real stud eye). After the refit, gun ↔ nose_clamp_L/R clearance is 0.300
  (stud bore), carriers 1.30, trig_clamp_R 1.00, servo_bracket_L 1.00,
  mg996r 1.15, fhcs_m3x25_U18 0.54, bands 0.236 (rib plate). The exact
  20-body front-module check (190/190 pairs) found only the rev U overlaps
  listed below: M2 threads, buck #2 solder box, CSI FFC entry, IMU pins.
  `nose_clamp_L` / `nose_clamp_R` pass `verify_battery`. `gun_body` fails
  only `feature_attachment` on f10 SafetyBolt, the deliberate crossbolt
  keep-out envelope wider than the nominal guard slab.
- Rev U front-module threshold interference (37 components incl. gun)
  completed 666/666 pairs. Overlaps: M2 self-tapper threads in pilots
  (2.96 mm³ each: camera 4, buck #1 2, buck #2 2); the reference's 1.5
  solder-envelope box into buck #1 belly posts (0.88 mm³) and buck #2 wall
  posts (2.42 mm³); and terminations (J1 and CSI FFC entries, IMU pins, jack
  boot, lever, buck #1 pads, pack lead block, capacitor leads, B_R junction).
  Exact clearance from carrier_R / cover_R to the gun model: 1.2 / 15.7.
- Whole-assembly threshold interference (55 components, after the rear-pad
  change) checked 1429 of 1431 pairs. The trigger set has no overlaps
  except the two guard representations. Front-module overlaps are the
  known M2 threads, buck solder envelopes, FFC/IMU/jack terminations and
  harness stubs. Two harness pairs (`harness_pi5v` ↔ `harness_trigger`,
  `harness_power` ↔ `harness_jack`) fail the engine Boolean.
- `harness_power` / `harness_trigger` cannot be booleaned by the engine; an
  analytic capsule check of their segments finds no wire crossing.
  Pi 5 V / header stubs touch at the 2.54 header pitch (soldered pins).
- Trigger-set exact interference (18 components incl. gun) completed
  153/153 pairs. Its only overlap is `gun_body` with `guard_measured`, the
  two guard representations. Contacts are the switch flange and the four
  ear screw heads on the servo ears. Tightest gaps: screws to clamp 0.049,
  bracket to U28 tip 0.19, bracket to rear M3 × 18 0.415, gun/guard to
  clamp 0.563, gun/guard to bracket 0.847, rear screw to the rear-inboard
  ear screw 2.5, rear screw to servo 2.7. Ten motion frames driving the
  lever -5°..31° and trigger 0°..-9.07° all converged. Their only
  trigger-set overlap was the guard pair. The rear pad top (Z -12.8) is
  4.6 below the lever's lowest face (Z -8.2) and its plane of motion.
- Clamp DFM (+Z): minimum wall passed; the D7.4 switch-hole crown has
  9.4 mm² overhang and D7 access crown 1.2 mm². Bracket +Y DFM requires
  build-plate supports (approximately 433 mm² overhang). The bracket/plate
  attachment advisory is 0.9 for the top-frame span/button-panel features;
  sketches were fully defined. Rev U: carrier_R, cover_R, carrier_L and
  belly_cover pass `verify_battery`, as does the assembly. The FFC reference
  model fails only `feature_attachment` (a folded 0.3 sheet of touching
  boxes, not a printed part). The nose halves (unchanged) retain the
  intentional nose-root attachment advisory.

## Open checks

- Physically fit the forend liner, stud/washer, both bands in the clamp
  grooves and rib windows, covers, battery foam, Pi port overmolds in the
  top-skin notches, camera rear components and both connector entries.
  Confirm the 200 FFC crease under the clamp nose, the Z-fold behind the Pi,
  the wire channel under the Pi edge (3.6) with real insulation, and the M2
  insert size you have (bore modelled D3.2 × 4.0 for an M2 × 3 insert).
- No FEA or recoil test covers the carriers: they hang on MI-1 (4 × M3) and
  the belly; check for rattle or creep, especially the 32-long buck bay that
  cantilevers behind MI-1 on the gun-right side.
- Measure the actual guard taper at X ~20 and ~37. Confirm the 2 mm underside
  bevel: at 25–29 ahead of the trigger, if the full 11.3+ width reaches the
  bottom corner, the servo case may touch. Check button reach and click from
  the firing grip and ensure a manual finger can enter and pull with the
  actuator at rest. Inspect servo case length between bracket blocks.
- Verify the servo ear-top to horn-top distance (model 16.0), actual horn
  engagement and gear-cover step height 2.4. Confirm that the centre screw
  clamps without bottoming. Check the real M2 pilot/screw fit and board holes
  (the XL4015 reference has D2.3 holes; if your boards take M3, switch them).
- Confirm both XL4015 boards' terminal pitch (model 5.08 [INFERENCE]), polarity,
  solder clearance and trimpot access; confirm 1000 µF body size and real wire
  outside diameters. Route and secure both external runs (3-wire signal from
  X 413, servo pair from the carrier_R rear wall) without sharp bends.
- Bench-test 3.3 V PWM acceptance at the 7.0 V servo rail, common grounds,
  switch normally-open/pull-up behaviour and fail-safe disarming; measure
  actual command-to-break at 7.0 V. Open the battery lever connector before
  handling the actuator; the switch does not cut servo power.
- No FEA or recoil test has established the 1.6 walls, carrier stiffness,
  button-collar web, left-cover hook or jack shoulder under an estimated
  20 N plug load. The physical gun-contact and sustained-heat behaviour remain
  unverified.
- Regenerate and verify trigger-set print files for the current parts before
  printing; assess the bracket's build-plate supports and ream the D7.4 switch
  hole if its bridge sags. Model overhang and G-code warnings in the front
  print table are not a print-ready pass.
- Calibrate camera intrinsics, boresight, image orientation, entrance pupil,
  actual barrel mask and IMU axes. The existing software geometry defaults do
  not yet match the installed front module and must not be used as if they do.

## Files

- `cad/asm/tbot_full.step+`: the only current master assembly, including the gun
  references, front module, trigger set and modelled harness.
- `cad/parts/`: `nose_clamp_L`, `nose_clamp_R`, `carrier_L`, `carrier_R`,
  `cover_L`, `cover_R`, `belly_cover`, `camera_cover`, `servo_bracket_L`, `servo_lever` and
  `trig_clamp_R` native `.step+` parts. `chin_lid.step+` (and its print files)
  are obsolete since rev S and not in the assembly.
- `cad/refs/`: reference models by kind: `gun/` (gun, guard, trigger), `electronics/`,
  `actuators/`, `fasteners/` (screws, nuts, hose clamps) and `hardware/`; `cad/refs/harness/` contains the six modelled harness/FFC parts.
- `cad/export/`: part STEP exports, `refs/` reference exports, and
  `asm/tbot_full.step` assembly export.
- `cad/print/creality_ender_3_0_4_mm_nozzle/`: front-module `.gcode`, `.3mf`
  and `.print.json` files; see the print table for their actual check status.
- `design/BOM.md` and `design/assembly_sequence.md`: hardware, wiring cuts
  and build/service order. `design/renders/` contains the rev R, rev S, rev T
  and rev U views (`rev_u_top.png`, `rev_u_front_left.png`,
  `rev_u_ffc_route.png`, `rev_u_carrier_R_bay.png`).
route.png`, `rev_u_carrier_R_bay.png`).
