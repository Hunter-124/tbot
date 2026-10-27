# TBOT — trigger actuator + camera mounts: layout contract (mm)

Units mm, degrees. Every gun-mounted part is modelled IN THE GUN FRAME at its
installed position (assembly placement = identity). Moving parts are modelled at REST.

## Gun frame (shared datum)
- +X = along bore toward muzzle, +Z = up (rib side), +Y = gun's LEFT (right-handed).
- Origin = trigger finger-face contact point at rest, on the gun mid-plane (Y=0).
- Sources: Weatherby spec (12 ga Element, 26" bbl, LOP 14-3/8" = 365, OAL 47.5");
  user photo 1 (rulers, ~64 px/in at floor, x1.07 perspective for parts ~30 mm above
  floor; bore tilted 16 deg in image); photo 2 (~4.67 px/mm at floor, x1.05).
  Photo-derived numbers are +/-2 mm (iPhone wide lens; perspective dominates distortion).
  Every interface that touches the gun gets a compliant liner or adjustment.

## Shotgun reference — REV B (refs/gun_body.step+, refs/gun_trigger.step+)
Re-measured from photo 1 with bore tilt 13.5 deg in the image (ejection port, receiver
top/bottom edges), 2.7 px/mm at the gun surface (x1.02 for the stock so the butt lands at
LOP 365), origin re-picked on the trigger finger face mid-blade. +/-1.5 mm near the guard,
+/-4 mm on the stock. Rev A numbers are superseded.
- Receiver: X -35..+190, Y +/-19, Z +16..+70 (R6 long edges). Bottom plane Z=+16.
  Loading port pocket in bottom: X +70..+165, Y +/-11, 15 deep. KEEP-OUT: nothing in
  X +70..+165, |Y|<12, Z +1..+16 (thumb access for loading).
  Trigger slot in receiver bottom: X -10..+3, Y +/-4, Z +16..+32.
- Trigger guard, XZ profiles (X,Z), extruded Y +/-4.5 (bow 9 wide, [INFERENCE]):
  outer (-27,28) (-23,18) (-18.4,14.6) (-13.7,6.2) (-9.9,-1.6) (-5.9,-7.1) (0,-11.6)
        (7.4,-13.8) (19.5,-13.8) (29.5,-10.6) (41.6,-5.2) (54.1,1.6) (63.1,8.3) (68,16)
        closed along the receiver bottom Z=16 back to (-27,16) and up the stock front.
  opening (smooth spline) (-3.1,0) (0.6,-7.5) (8.6,-10.5) (18.7,-9.8) (28.4,-5.9)
        (35.2,0.8) (37.1,8.9) (31.5,14) (21.6,16) ... top = receiver bottom Z=16 ...
        (0,16) (-3.1,10) back to (-3.1,0).
  => bow bar inner Z=-10.5 / outer Z=-13.8 at X 7..20 (3.3 thick); opening X -3.1..+37.1,
  Z -10.5..+16. Front web X 37..68 (solid, tapering up into the receiver).
  Trigger clearance slot through the guard rear: X -3.1..-8, Y +/-4, Z -11..+16.
- Crossbolt safety: cyl D8 along Y at X=-11.5, Z=+11, Y +/-12 (photo puts its face centre
  at X-7.6; moved back so it clears the trigger's swept tail). KEEP-OUT for our parts:
  X -20..-3, Z +3..+19, any Y.
- Trigger (separate part, rotates): blade Y +/-3.5, 4 thick (X) behind the finger face,
  face polyline (X,Z): (5.4,-9.2) (2.8,-6.0) (0.6,-1.8) (-0.6,2.7) (-1.0,7.4) (-0.2,10.4)
  (-0.5,16) -> continues into the receiver to a hub at the pivot; tip rounded R2 at Z~-10
  (tip curls FORWARD). Face passes through the origin.
  Pivot axis along Y through (X=-4, Z=+26), radius to origin 26.3. +rotation about +Y =
  pull. Break 3.5 mm at contact (7.6 deg); hard stop 4.3 mm (9.4 deg). Pull ~27 N
  [INFERENCE]; design load 35 N.
- Stock (side profile, X,Z), extrude/loft: tang top (-37,68) wrist (-120,48) (-153,39)
  comb (-193,43) (-251,45) (-328,42) heel (-365,41) butt line to toe (-360,-77);
  toe line (-248,-51) (-145,-25); pistol grip back/palm to grip-cap corner (-93,-38);
  grip cap (-94,-28); grip front (finger side) (-91,-8) (-83,7) (-70,17) (-56,24)
  (-38,29) (-31,31) then up the stock front face X~-33 to the tang top.
  => butt 117 tall (heel Z+41, toe Z-77), drop at comb ~27, at heel ~29 below receiver
  top. Width Y: butt +/-21, comb +/-19, grip +/-16, stock front +/-19 (round all long
  edges R8-12). Recoil pad X -365..-340.
- Forend X +190..+440 (Y +/-22, Z +11..+65). Forend cap D34 along X, X +440..+480,
  axis Z=+15 (bore - 34, photo 2), sling stud D6 X 480..488.
- Barrel: axis Y=0, Z=+49 (rib top meets receiver top Z=70); X +190..+841.
  OD 27.0 @X190, 24.0 @X500, 20.5 @X841. Photo 2: 24-25.5 near forend: design for 24.0,
  accept 23-25.5.
- Vent rib: Y +/-3.5, plate 2 thick, top 7 above local barrel top (Z 66..68 at X500),
  posts 4 long every 45 from X=200.

## Actuonix PQ12-100-12-S (datasheet Rev D 2019)
- Body 36.5 (X) x 15 (Y) x 21.5 (Z); 100:1, 12 V; stroke 20; 50 N max, 40 N @ 6 mm/s,
  10 mm/s no-load; side load max 10 N; back-drive 35 N (holds position unpowered);
  limit switches stop within 1 mm of each end; stall 210 mA @ 12 V; 19 g.
- Installed (gun frame), rod pointing -X, REST = fully retracted:
  body X +27..+63.5, Y +9..+24, Z -16..+5.5. Rod axis Y=+16.5, Z=0, along X.
  Rod-end boss D5.8 (5.8 thick in Y, Y 13.6..19.4), cross hole D3 axis along Y:
  hole centre X=+24 at REST, X=+4 at FIRED (extended). Rod tip at X=+21.5 rest.
  Rear clevis tab (5 thick in Y, 6 tall in Z) hole D3 axis Y at X=+66.5, Y=16.5, Z=0.
  FPC (6 wide, 125 long) exits the BOTTOM face (Z=-16) at X=+32.2.
  No carriage material at X>+26.5 inside the body footprint (Y 9..24, Z -16..5.5).

## Trigger actuator set (parts/trig_bracket_L, trig_clamp_R, trig_finger)
- Left plate between guard (Y=+4.5) and actuator; right counter-plate at Y -4.5..-8.5.
  Clamp the guard bow with M3 cross screws ABOVE the bow inside the opening (shank on
  the bow top: Z=-8.8 at X=8, Z=-7.6 at X=20 — the opening floor curves up toward the
  front) and BELOW it (lip top Z=-14.0, screw Z~-16.5), +/-1 mm pad allowance on bow sides.
  REV B guard: bow inner Z=-10.5, outer Z=-13.8; opening X -3.1..+37.1; receiver bottom
  Z=+16 (more room above the PQ12 than Rev A).
- Finger: crosses the opening laterally at Z -3.5..+3.5, reaches Y=-4.5 (past the
  trigger's far side). Contact face (rear) X=+15.8 at REST, X=-4.2 at FIRED.
  2 mm TPU pad on the contact face (compliance so the -S limit switch, not a stall,
  ends the stroke). Carriage rides a guide on the left plate (moments on the guide,
  not the PQ12 rod); pinned to the rod end with an M3/D3 pin along Y.
- +/-3 mm X adjustment of the actuator in its cradle to tune the fired position.
- Actuator on gun LEFT side: shells eject to the right.

## Trigger carriage print/DFM (parts/trig_finger) — REV B
- PRINT: PETG, build_dir [-1,0,0] — the part's -X axis up, i.e. the carriage stands on its
  REAR face (the X=+26.5 plane) on the bed. 208.4 mm2 first layer, 26.5 tall, tip ratio
  1.84, COM inside the contact hull. ZERO support, 3.89 g, ~0.19 h, 133 layers.
  Printing up the bore axis is what makes it support-free: the channel roof, the finger-bar
  underside, the yoke underside and the pad-pocket ceiling are all 90 deg ceilings in any
  Z-up orientation (292.4 mm2 unsupported, 5521 mm3 support); up -X they are vertical walls.
  It is also the right layer direction for load: the finger bar is a cantilever along Y
  loaded along -X, so its root bending stress lies IN the YZ layer plane, and the pin->crown
  path is compression ACROSS layers. Do NOT print this part up +/-Y (538-718 mm2 overhang
  AND the weak interlayer plane lands on the bar's cantilever root).
- The D3 pin hole is horizontal in this orientation and self-supporting; the rod slot roof
  is a 6.0 mm anchored bridge (clean limit 5, fail 10) — acceptable, it only needs to hold
  the 0.8 mm clearance to the rod tip at X=+21.5.
- Hook tip blunted: the carriage's lowest point is now Z=-24.0 (was -25.85). The 45 deg
  retention hook used to feather to a 0 mm knife point where it met the Y=8.7 open face;
  it now ends in a 1.55 wide flat (1.30 perpendicular). Engagement face unchanged
  (Z = Y-31.155, 0.25 clearance to the rail wedge at Z = Y-30.80).
- The rail-wedge groove's far end was extended 0.35 to Y=14.0 / Z=-17.16 so the cut corner
  no longer lands exactly on the Y=8.7 face (it was producing a zero-area sliver face);
  this only adds clearance behind the rail's own wedge corner.
- FROZEN INTERFACES (verified unchanged, do not move): contact crescent R4.5 about axis
  (X=20.3, Z=0) -> contact apex X=+15.800 at Z=0; pad pocket X 15.8..17.8, Z -2.4..+1.5,
  open in +/-Y; pin hole D3.0 along Y at X=+24.000, Z=0, blind inboard at Y=+10, open at
  Y=+22.4; rod slot X 20.7..27.0, Y 13.4..19.6; guide channel Y 8.7..13.65, Z -17.50..-4.90.
- Installed fit at REST (4-body check at identity, 0 interferences): carriage<->bracket
  0.20, carriage<->PQ12 rod 0.20, carriage<->trigger 5.20 (the 2 mm pad closes this to 3.2
  of free travel before the 3.5 mm break).
- HARDWARE (trig_finger): 1x D3.0 x 12 stainless dowel pin (light press in the printed
  legs, free in the steel rod-end boss, bottoms on the blind hole at Y=+10 — insert from
  the outboard +Y side); 1x 2.0 mm TPU/silicone pad 7.8 (Y) x 3.9 (Z), slid into the pocket
  from -X. No screws, inserts or nuts on this part.
- parts/trig_bracket_L passes FDM DfM at build_dir [0,0,1] (REV H: 1151.8 mm2 bed, 0 overhang; was 928.8 before the shroud).
  Its two warns are benign: min_wall 1.051 (the 45 deg gable apexes of f22-f25) and a
  9.5 mm bridge over the f19 PinSlot — 0.5 mm under the fail limit, so do not widen that
  slot without gabling it.

## REV H — trigger hardware clean-up (user, 2026-10-26: "make it much cleaner")
- trig_bracket_L now reads as ONE flush slab from the gun-left side: the cradle's outboard wall continues rearward as
  ShroudWall X -20..27, Y 24.3..27.1, Z -22.3..8.6, tied to the cradle along its full height and stiffened by
  ShroudFloor (Y 16.85..24.35, Z -22.3..-19.8, on the bed). The carriage, rod end and dowel pin are no longer exposed
  from the left or below-outboard (pinch point covered). Carriage clearances: 1.9 to the wall (carriage max Y 22.4),
  0.8 to the floor flange (carriage hook max Y 16.05 below Z -17.5). The rear (X -20) stays OPEN: the carriage still
  slides onto the rail from the rear before the bracket goes on the guard.
- PinAccess D4.2 along Y at (X 24, Z 0) through the shroud: lines up with the carriage pin hole at REST, so the
  D3 x 12 dowel still goes in from +Y.
- FPCGroove on the cradle underside: X 35.4..62.1, Y 13.2..19.8, 1.0 deep — the PQ12 flex leaves the FPCSlot and lies
  flush forward instead of hanging.
- Outline: R8 rear-top corner, R4 ear front-top, R1.5 round the whole outboard face boundary, 1.0 x 0.8 bed chamfer
  on the outboard / rear / ear-front bottom edges (steeper than 45 deg so the R1.5 blends are not overhangs).
- trig_clamp_R: front end of PlateR rounded R4 (top) + 1.5 x 1.2 bed chamfer; button pose, housing and screws unchanged.
  An R1 on the plate's outboard top edge was tried and dropped: it runs back into the CskU17 rim (0.93 mm sliver).
- Verified: verify_battery pass (attachment 1.0) and dfm_check(ender 3, +Z) pass on both; wall_thickness red only at
  the pre-existing gable apexes (bracket) and housing rear edges (clamp). Rigs: 190 pairs clear (PQ12 rig), 21
  (servo rig); PQ12 sweeps m6 3->16.8 and m6 16.8->21.1 with trigger m4 0->-9.4 clear on all 10 frames.
  Sliced PETG strength, print_ready: bracket 24.3 g / 2.5 h (gcode warn: 8.25 mm air run < 10 limit), clamp 14.0 g / 1.6 h.
- Not changed: trig_finger (frozen interfaces), servo_bracket_L / servo_lever (the MG996R set still shows its mechanism).

## MG996R trigger actuator set (parts/servo_bracket_L, servo_lever; refs/mg996r) — ALTERNATIVE to the PQ12 set
Same guard-clamp interface as trig_bracket_L/trig_clamp_R: left plate Y +5.5..+8.5, clamp lip Y +2..+5.5 (X 0..30, Z -24.0..-14.8, 45 deg gable under it), M3 cross screws along Y at (X8,Z-8.8) (X17,Z-8.2) above the bow and (X8,Z-18) (X17,Z-18) below it, M3 heat-set inserts in the left plate, trig_clamp_R unchanged.
- TowerPro MG996R (catalog record sourced from towerpro.com.tw): case 40.9 (X) x 20 x 37, flange ears span 54 with a 49.5 x 10 hole pattern (D3.2), tab bottom 26.8 from the case base x 3 thick, output boss D12 x 3, 25T shaft D5.9 x 2.9, shaft 10.0 from the case end, overall 42.9. 9.4 kg·cm / 0.17 s per 60 deg @4.8 V; 11 kg·cm / 0.14 s @6 V; ~55 g.
- Installed (gun frame), output shaft along -Y, pivot axis X=0 Z=-14, case long axis along X with the long end FORWARD (keeps it off the firing hand; cable exits at X~+31):
  case X -10.0..+30.9, Y +28.7..+65.7, Z -24..-4; flange ears X -16.55..+37.45, Y +35.9..+38.9; tab holes X -14.3 / +35.2 at Z -9 / -19; boss D12 Y +25.7..+28.7; shaft Y +22.8..+25.7.
  Stock single-arm horn [INFERENCE 9.0 wide x 3.0 thick, hub D9 x 5.5, D2.0 holes at r 6.5/9.0/12.5/16.0]: hub Y +20.2..+25.7, arm Y +17.2..+20.2, pointing +17 deg from +Z at REST.
- servo_bracket_L (PETG, print +Z up): clamp plate X -20..+41, Y +5.5..+8.5, Z -28.5..-5.2 with a stop wing X +15.8..+41 up to Z +4.5 (its rear edge is the lever's rest stop, 0.61 mm gap); insert boss Y +8.5..+11.5 (X 3..22, Z -15.5..-5.2, 45 deg gable) so the upper M3 inserts get 5.2 mm depth; one L-section extrusion over X -20..+41 forms the base shelf (Y +8.5..+35.9, Z -28.5..-24.5, also the print bed face) and the servo wall (Y +29.9..+35.9, Z -24.5..+0.5) with an open-top window X -10.4..+31.3; 2 gusset ribs at X 16.5..20.5 and 31..35. 4x M3 heat-set inserts D4.0 x 6.0 in the wall carry the servo flange (M3 x 8).
- servo_lever (PETG, print with the plate's outboard face on the bed, bar up): plate Y +13.0..+17.2 (4.2 thick), outline = r 0..20 x t -9..+11 about the servo axis plus an R9 rear arc; the horn seats on the flat Y=+17.2 face (no pocket — a pocket floor is an unprintable 20 mm bridge in this orientation), held by the centre shaft screw plus 2x M2 through the horn holes at r 9.0 / 12.5 (D4.2 x 2.4 counterbores inboard, nuts outboard).
  Finger bar hangs off the plate's inboard face: shank 6 (fore-aft) x 5 (tall) running Y +13.0..-3.9, crown 3.5 deeper over Y -3.9..+2.0 with a 50 deg ramp back to the shank (keeps everything at Y>+4 clear of the guard's rear leg, X > -3.1). 2 mm TPU pad 5.9 x 3.5 flush in a 2.0 deep pocket on the crown's rear face, open at the bottom so TPU makes first contact.
- Motion: lever radius 14, sweep +17 deg (REST) to -17 deg (FIRED) = 34 deg of servo travel. Pad face sits 4.5 mm in front of the trigger face at rest (no preload — the unpowered, non-back-driveable servo never touches the trigger). Contact at lever -2 deg, BREAK (3.5 mm) at -14.5 deg, FIRED pose = 4.15 mm pull, 0.15 mm short of the 4.3 mm hard stop.
- Torque: 35 N design load at Z +0.4 gives a 14.4 mm arm about the servo axis = 0.50 N·m = 5.14 kg·cm -> 1.83x margin at 4.8 V, 2.14x at 6 V. Predicted rest-to-break 89 ms @4.8 V / 74 ms @6 V no-load, ~115 / ~95 ms loaded [INFERENCE].
- Hardware: 2x M3x20 + 2x M3x16 countersunk ISO 10642 (REV G) and 4x M3 inserts (guard clamp, as the PQ12 set) + 1 mm rubber shims; 4x M3x8 + 4x M3 inserts (servo flange); servo centre screw (or M3x16 CSK) + 2x M2x10 + M2 washers/nylocs (horn); 2 mm TPU pad.
- Safety interlock: the arming trigger on trig_clamp_R (REV G) gates the actuator — the servo may only be commanded to the fire angle while that button is held.


## REV C vision + electronics (supersedes the udevqal stereo pair and the bar IMU seat)
The rig is SHOULDERED by the shooter: keep the rib sight plane (above Z=70 on the
barrel) clear, keep the support hand's forend (X 190..440) clear, mount everything
under the barrel.

### Camera: 1x innomaker CAM-MIPIOV9281 V2 (user manual V1.2, §3.1-3.2)
- PCB 32 x 32 x 1.6 square, corner pads R2.0 with D2.2 holes (M2) on a 28.0 x 28.0
  square (2.0 from each edge). Lens-holder screws: 2x D2.0 on the board's vertical
  centreline, 20.0 apart (6.0 from top/bottom edges). Lens on the board centre.
- M12 lens holder ~16 x 16 base, ears to +/-12 along the 20 screw line, ~12 tall
  [INFERENCE from photos]; M12 fisheye lens D14 front, front face 22 from the PCB front
  (drawing: lens 21.71 long). FOV D148 / H118, f=2.8, TV distortion -17%.
- J1 15-pin 1.0 FPC connector on the BACK side along one edge (~21 x 5.5 x 2.5)
  [INFERENCE from manual photo]; J2/J3 2-pin strobe/ext-trigger headers on the back.
- Installed: PCB back face at X=+604, board in the YZ plane, lens axis along +X at
  (Y=0, Z=+17) (board top edge Z=33 = 4.5 under the barrel bottom), J1 edge DOWN.
  The barrel will be visible in the top of the 118 deg image: mask it in software.

### Raspberry Pi 3 Model B v1.2 (official mechanical drawing)
- PCB 85 x 56 x 1.4, corner radius 3; 4x D2.75 holes (M2.5) at (3.5,3.5) (61.5,3.5)
  (3.5,52.5) (61.5,52.5) from the SD-card short edge / power long edge. Component side
  envelope 17 tall (USB stack); Ethernet+2x2 USB on the far short edge (protrude 2.1);
  micro-USB power X=10.6, HDMI X=32, audio X=53.5 on the power long edge; 2x20 GPIO
  along the opposite long edge (pins 8.5 tall); CSI connector X~45, near the HDMI edge;
  microSD under the board at the SD end (protrudes 2). 45 g.
- Installed horizontal under the barrel, component side UP: PCB bottom at Z=+6,
  X +497..+582 (SD end at the rear X=497, USB/Ethernet end forward toward the camera),
  Y -28..+28 with the GPIO header on the gun-LEFT (+Y) edge and power/HDMI on the
  right (-Y) edge. Top of components Z<=23.4; >= 6 air gap from any part to the barrel.
  Camera FPC (kit ~150 [INFERENCE]) runs from J1 down/back to the CSI connector.

## REV E — electronics pod rework (SUPERSEDES the REV D pod section and the muzzle IMU saddle below)
User findings on REV D: (1) the thin saddle rim strip at the front station (top face Z=44 near
(582,-20.6), ButtressFront edge at Y=-17.4, Z=42.2) would break off; (2) the muzzle IMU
(X 746..783, Z 24.6..48) sits dead ahead of the camera lens and blocks the view; (3) the Pi
cannot be installed after printing (the saddles cover the Pi screw positions); (4) add power:
XL4015 buck in the pod, 12 V Li-ion pack in a strap-on box under the buttstock.
Decisions (user, 2026-10-26): battery under the buttstock; buck in the pod; the Pi slides in on
rails and is held by an end cap. The OPEN END IS THE FRONT (user: "you pick... make sure we can
still access the ports"): the sled slides in with the pod still clamped to the gun (a rear
opening is 6 mm from the forend cap), and taking off the front cap exposes the USB/Ethernet end.
parts/imu_clamp_lower is RETIRED; the IMU rides on the front cap next to the camera, so camera
and IMU share one rigid part (fixed camera-IMU extrinsics).

### Bought-part envelopes (commit; [INFERENCE] where marked)
- XL4015 5 A buck module: 54 (X) x 23 (Y) x 18 (Z) overall incl. PCB 1.6, 2x D3 holes at
  diagonally opposite corners ~3 from both edges [INFERENCE] (listing hit.ps: "54 x 23 x 18 mm,
  2 x 3 mm screw holes"). Screw-terminal/pad input on one short end, output on the other;
  bottom-side solder 1.5. Model refs/xl4015.step+ as this envelope (PCB + component block +
  holes).
- Battery: 12 V (3S1P 18650) 2600 mAh pack 70 x 58 x 20 (listing for the same 12 V 2600 mAh
  18650 class pack: "20*58*70mm") [INFERENCE for the Xeanerol pack: design 72 x 60 x 22
  internal + foam], ~140 g, leads with a DC5521 plug out of one 58 x 20 end.

### Pod body — parts/cam_mount_lower (ASA, prints floor-down +Z)
- Extent X 486..640 (front face X=640, OPEN at the front), floor Z 0..3, nothing below Z=0,
  nothing at X<486. Side walls inner |Y|=32.0, outer |Y|=35.5.
- Rails: grooves in both wall inner faces |Y| 32.0..33.5, Z 3.0..6.2, X 490..640, open at the
  front. Floor top Z=3.0 is the sliding surface. Rear bulkhead X 486..490: its front face X=490
  is the sled's rear stop.
- Sled-swept envelope that MUST stay clear inside the pod for X 490..640: |Y|<=29.0 for
  Z 3.0..27.5, plus |Y|<=31.0 for Z 9.0..20.0 (Pi -Y connectors stick out to Y=-30.5).
  => every saddle/station underside >= Z 29.0 over |Y|<=29.
- Stations: rear band channel X 501..510 (rib gap 474..515), front band channel X 620..629
  (rib gap 609..650); saddles = local barrel R + 1.5 liner, below Z=49, support-free 45 deg V
  undersides, worm-housing seats on gun-right. Barrel r(X) = (24.0 - (X-500)*3.5/341)/2.
  Minimum wall anywhere in the saddles/load path 2.4 (no knife-edge rims).
- Cap bosses: 4x M3 heat-set inserts D4.0 x 6 deep, axis along X, opening on the X=640 plane,
  at (Y +/-38.0, Z 6.0) and (Y +/-38.0, Z 26.0), bosses outside the walls (|Y| up to 42.0),
  45 deg undersides for printing.
- Port windows in the -Y (gun-right) wall for the installed Pi: micro-USB X 553..568,
  HDMI X 572..592, audio X 597..610, all Z 9.5..20.0 (Pi stays visible/usable on the gun).
- Cable exit: grommet hole/slot in the gun-right wall at the rear, X 491..499, Z 8..16 (battery
  12 V in, actuator/pendant GPIO leads out), cable-tie anchor next to it.

### Sled — parts/pod_sled (PETG), slides in from the front
- Plate X 490.0..637.0, |Y|<=33.2, Z 3.0..5.8 (2.8 thick); edges ride in the rail grooves.
- Pi 3B on 4 standoffs to Z 9.0 (PCB bottom), M2.5 heat-set inserts D3.6 x 4.5, holes at
  (553.5, +/-24.5) and (611.5, +/-24.5); Pi PCB X 550..635, Y -28..+28, SD end at the REAR,
  USB/Ethernet at the FRONT (protrude to X 637.1), GPIO on +Y, power/HDMI/audio on -Y.
  Pi top envelope Z <= 26.4.
- XL4015 at X 491.5..545.5, Y -26..-3, board bottom Z 9.0 (top Z 27.0), input end at the rear.
  Retention tolerant of the [INFERENCE] hole positions (fence/pocket + cable-tie slots, and/or
  M3 through its holes into slotted captive-nut channels).
- Pull tab/finger grip at the front edge for withdrawal (must stay inside X<=637.0, below
  Z=9 or outside |Y|>28).

### Front cap — parts/pod_cap (ASA), camera + IMU carrier
- Rear face X=640.0 against the pod front face; 4x M3 x 12 SHCS through D3.4 holes into the
  pod inserts at (Y +/-38.0, Z 6.0 / 26.0).
- Two pusher tabs reach back to X=637.3 at |Y| 29.5..31.7, Z 3.4..8.5 (1 mm silicone pad on
  each face preloads the sled against the bulkhead). Nothing else of the cap behind X=640.
- Camera (innomaker OV9281 V2) ROTATED 180 deg about its lens axis vs REV D (J1 FPC connector
  edge UP, J2/J3 down; flip the image in software): PCB back face X=652.0, lens axis along +X
  at (Y=0, Z=17), PCB Y -16..16, Z 1..33, lens tip X~675.6. 4 pads with faces on X=652 and
  D1.8 x 6 pilots at (Y +/-14, Z 3 / 31) for M2 x 6 self-tappers; reliefs behind the board for
  J1 (now top edge, ~21 x 5.5 x 2.5) and J2/J3 (bottom).
- FPC slot through the cap Y -10.5..+10.5, Z 26.5..31.5: J1 -> straight back -> over the Pi
  USB stack (Z 26.4..29) -> down to the CSI connector (~X 595, -Y half).
- IMU GY-521 rigidly on the cap, behind the lens front plane (X < 675) and outside the camera
  FOV, component side protected, header wired back to the Pi GPIO (+Y edge) through the cap.
  Report its exact pose (axes) for software.

### Battery box — parts/batt_box + parts/batt_lid (PETG), under the buttstock  — SUPERSEDED by REV E.2 (left side of the stock)
- Under the stock between the grip cap and the toe, X -240..-120 window (keep X <= -110: the
  firing hand), base contoured to the stock underside with a rubber pad, held by 2x 25 mm
  hook-and-loop / webbing straps round the stock (no hose clamps on the stock). Keep it as flat
  as possible (pack lies flat, 58 across Y). Cable exit forward with strain relief.

### REV E.1 — Raspberry Pi 5 + AI HAT+ provision (user, 2026-10-26) — OVERRIDES the REV E Z numbers above
The Pi 3B will later be swapped for a Pi 5 + Active Cooler + official AI HAT+ (Hailo) on the
supplied 16 mm stacking header (raspberrypi.com AI HAT+ docs / product page: "16mm stacking
header, spacers and screws to fit on Raspberry Pi 5 with Active Cooler"; HAT 65 x 56.5).
Stack envelope committed = Pi PCB bottom + 27.0 (PCB 1.4 + 16 spacer + HAT 1.6 + 3.9 parts +
4.1 heatsink allowance [INFERENCE]) and +2 clearance. That cannot fit under the barrel at the
REV E heights, so the whole electronics bay drops 8 mm:
- Pod floor Z -8.0..-5.0, slide surface Z -5.0; grooves |Y| 32.0..33.5 x Z -5.0..-1.8;
  nothing below Z -8.0 ("nothing below Z=0" withdrawn).
- Sled plate Z -5.0..-2.2 (parts/pod_sled.step+ shifted by its final BayDrop8 transform);
  XL4015 Z 1..19 (refs/xl4015.step+ shifted). Pi PCB bottom Z 1.0: refs/rpi3b.step+ and
  refs/rpi5_aihat.step+ both place at=[550,-28,1] (same 85 x 56 PCB and 58 x 49 hole pattern).
- refs/rpi5_aihat.step+ = clearance envelope (local frame like rpi3b): PCB, cooler/header block
  X 0..65 Z 1.4..17.4, AI HAT+ + heatsink allowance X -0.5..65 Y -0.25..56.25 Z 17.4..27.0,
  PCIe FFC loop X -3..0, USB/Ethernet block X 66..87.1, USB-C and 2x micro-HDMI on the Y=0 edge,
  microSD to X -2.5. World: X 547..637.1, Y -29.2..28.25, Z -0.4..28.0.
- Sled-swept envelope (X 490..640): |Y|<=29.0 for Z -5.0..29.0 plus |Y|<=31.0 for Z 1.0..12.0.
  Station/saddle undersides >= Z 29.0 over |Y|<=29.
- -Y wall port windows (Pi 3B + Pi 5): X 553..567, 569..595, 597..610 at Z 1.5..12.0.
  Ventilation slots in both side walls, Z 14..26, between stations (Active Cooler air).
- Cap insert bosses (Y +/-38, Z -2.0) and (Y +/-38, Z 26.0); cap pusher tabs |Y| 29.5..31.7,
  Z -4.6..0.5 to X 637.3; rear cable exit X 491..499, Z 0..8. Camera pose unchanged.
- Pi 5 swap notes: camera needs a 15-pin -> 22-pin (0.5 pitch) "Pi 5 camera cable"; the Pi 5
  wants 5 V / 5 A for full USB current (XL4015 is rated 5 A; set 5.1 V); expect a hotter bay.

### Pod body — parts/cam_mount_lower (ASA, prints floor-down +Z) — REV E.1 AS BUILT  — print orientation, band path, stations and windows SUPERSEDED by REV F (below)
- Extent X 486..640 (front face X=640, OPEN), floor Z -8.0..-5.0, nothing below Z=-8.0.
  Side walls inner |Y|=32.0, outer |Y|=35.5, Z -5..29. Rear bulkhead X 486..490 (full
  section, front face X=490 = sled stop) with a sling-stud relief |Y|<=4, Z 11..19
  (the gun's D6 sling stud, X 480..488, would otherwise clash).
- Rails: grooves |Y| 32.0..33.5, Z -5.0..-1.8, X 490..640, open at the front; floor top
  Z=-5.0 is the sliding surface. Above Z=-1.8 the groove roof is a 60 deg ramp up to
  (|Y|=32, Z=+0.8) so it prints without support; rail capture is unchanged.
- Sled-swept envelope kept clear for X 490..640: |Y|<=29 for Z -5..29 and |Y|<=31 for
  Z 1..12. Measured station underside minimum over |Y|<=29 = Z 30.55 (1.55 margin).
- Stations (X 492..519 and 611..638, 27 long, full-height webs into both side walls):
  OPEN-BOTTOM V-BLOCK. Each station is two arms whose inner faces are the cradle cone
  (local barrel r + 1.5 liner: 13.5718 @X486 -> 12.7810 @X640), contacting the barrel from
  51 to 86 deg off bottom dead centre on both flanks, trimmed at Z=48.0; 45.5 deg V faces
  below. Arm section at the band slice 9.6 / 9.9 mm; zero-support print. Contact ribs
  X 492-497, 514-519, 611-616, 633-638; relief +0.6 r between them. (A bottom-wrapping
  saddle cannot print support-free with the underside pinned at Z>=29.)
- Band channels: 9.0 x 1.2 recesses in the V faces at X 501..510 (rib gap 474..515) and
  X 620..629 (rib gap 609..650). Closed band path: rear 110.3 mm (equiv dia 35.1), front
  108.1 mm (34.4) -> 2x 8 mm worm-drive clamps, DIN 3017-1 W4/A2, range 25-40 mm. The band
  passes UNDER the barrel through the open bottom (3.1 / 3.7 mm below the lined barrel,
  above Z 32.4, clear of the sled envelope). Worm housing rides on the barrel's gun-right
  upper flank between rib posts (no pod material above Z=48).
- Cap bosses: 4x M3 heat-set inserts D4.0 x 6, axis X, mouths on X=640, at (Y +/-38.0,
  Z -2.0) and (Y +/-38.0, Z 26.0); bosses |Y| 35.5..42.5, Z -8..30.
- -Y (gun-right) port windows X 553.0..566.8, 569.2..594.8, 597.2..610.0, Z 1.5..14.0
  (50 deg gable tops). Vent slots 9 wide, Z 17..26: +Y wall at X 524/538/552/566/580/594,
  -Y wall at X 524/538. Floor windows |Y|<=26 at X 496..542, 546..592, 596..632.
- Rear cable exit 8x8 through the -Y wall at X 491..499, Z 0..8, zip-tie anchor frame
  X 499.5..508.5, Y -39..-35.5.
- Print: ASA (registry `abs` stand-in), +Z, 56 mm tall, NO supports; dfm_pass, min wall
  1.86, max bridge 9.2; verify_battery pass. 95.1 cm^3, ~101 g in ASA.
- Verified: seated clearances Pi 5 + AI HAT+ 2.32, Pi 3B 1.78, XL4015 1.50, gun 1.00
  (sling-stud relief; barrel gap = 1.5 liner). Slide-in sweep clear (asm/pod_slide_study).

### Sled — parts/pod_sled (PETG, prints plate-down +Z) — REV E.1 AS BUILT
- Plate X 490.0..637.0, |Y|<=33.2, Z -5.0..-2.2 (bottom slides on the pod floor, edges in the
  grooves). Above the plate everything stays inside |Y|<=29.0; sled bbox Z -5.0..12.0.
  (Modelled at the REV E heights then moved by its final BayDrop8 transform, dZ -8.)
- Pi standoffs to Z 1.0 with M2.5 heat-set bores D3.6 x 4.5 at (553.5 / 611.5, +/-24.5);
  Pi 3B or Pi 5 PCB X 550..635, SD end rearward. microSD notch and a PCIe-FFC relief at the
  Pi's rear edge (X 545.9..548.5) so a Pi 5 + AI HAT+ ribbon loop clears.
- XL4015 pocket/fence X 491.5..545.9, Y -26.5..-2.5, board on Z 1.0; 2 tie slots through
  the plate; cable-tie anchor slot pairs at |Y|=30 (X 493.8-501.2, 518.3-525.7,
  596.3-603.7 on -Y; 566.3-573.7, 616.3-623.7 on +Y). Pull fins at the front corners
  X 635.4..637.0, |Y| 27.0..29.0, Z 1..12.
- Front face X=637.0 continuous over |Y| 0..33.2 (cap pusher band). 31.5 g. dfm_pass,
  verify_battery pass; no interference with refs/rpi3b, refs/rpi5_aihat (both at
  [550,-28,1]) or refs/xl4015.

### Front cap — parts/pod_cap (ASA) — AS BUILT (REV E.1)
One printed part, ASA (document material key `abs` = registry stand-in), modelled in the gun frame at
X 638.2..655.5, Y +/-42, Z -8.0..33.5. 19.1 cm^3, 19.9 g in ABS (~20.4 g in ASA). Prints floor-down
in +Z (the Z=-8.0 face on the bed), NO supports: dfm_check(fdm, build_dir +Z) dfm_pass true,
0.0 mm^2 of unsupported overhang, largest bridge 2.75 mm, min wall 1.600 mm, first layer 575.9 mm^2.
- Plate X 640.0..645.0 (5 thick), Y +/-36 over Z -8.0..33.5, ears out to Y +/-42 over Z -8.0..30.0,
  outer vertical corners R2. Rear face X=640.0 bears on the pod front face.
- 4x D3.4 through-holes, axis X, at (Y +/-38.0, Z -2.0) and (Y +/-38.0, Z 26.0) for M3 x 12 SHCS into
  the pod's M3 inserts (5 mm plate + 7 mm engagement).
- Pusher tabs: 2x prisms |Y| 29.5..31.7, Z -4.6..+0.5, rear face FLAT at X=638.2 (no pocket), underside
  ramped 48 deg up to X=640 @ Z=-2.6 so it prints support-free. A 2.2 x 5.1 x 1.5 silicone pad on each
  face works to X=636.7 = 0.3 mm preload on the sled front face (X=637.0); bearing 2.2 x 2.4 per tab on
  the sled plate band (Z -5.0..-2.2). [Supersedes the 637.3 + 0.7-deep-pocket + 1 mm pad call-out: 0.7 is
  under the FDM 0.8 minimum and both the pocket roof and the flat tab underside needed support.]
- Nothing of the cap is behind X=640 except those two tabs. Gaps measured at the installed pose:
  cap->Pi 3B 2.90, cap->Pi 5 + AI HAT+ 2.90, cap->sled 1.20 (pad), cap->barrel 7.44 (>= 4 required).
- Camera (innomaker OV9281 V2, refs/cam_inno9281.step+ at=[652,0,17], rotate=[1,0,0], angle=180):
  J1 CONFIRMED on the TOP edge (connector back face at X=649.5, Y 0, Z 27.5..33), J2/J3 at the bottom
  (X 649..652, |Y| 9..13, Z 2..5), lens tip X=675.6.
  * Top: 2 pads with faces on X=652.0 carrying D1.8 x 6 pilots at (Y +/-14, Z 31) for M2 x 6
    self-tappers; each pad is backed by a 47 deg ramp down to (|Y| 16, Z 25.0) and a vertical fin
    |Y| 16.0..17.8 that runs to the bed, so the pads print support-free.
  * Bottom: the PCB's bottom edge drops into a 1.9 mm capture slot - ledge Z=1.0, seat face X=652,
    retaining lip X 653.9..655.5 - at |Y| 13.5..17, on a foot X 645..655.5, |Y| 12..18, Z -8..1.0.
    [Supersedes the two lower D1.8 pilots: the ref's J2/J3 envelope leaves <=0.1 mm of wall around a
    pilot at (|Y| 14, Z 3).] Assembly: drop the board into the slots, then 2x M2 x 6 at the top.
  * Reliefs for J1/J2/J3 are inherent - nothing of the cap lies between X 645 and 652 except the two
    fins/pads (|Y| >= 11) and the lower seats (|Y| >= 13.5).
- FPC passage: Y -10.5..+10.5, Z 26.5 UP THROUGH the plate top (33.5), X 639..646 - an open-topped
  notch, not a closed window (a closed 21 mm slot is an unsupported bridge; the bay is already vented).
  The cable is laid into the notch from above during assembly.
- IMU GY-521 (refs/imu_gy521.step+ at=[649.6, 26.2, 15.0], rotate=[1,0,-1], angle=180), rigid on the cap:
  PCB X 648.0 (component side, on the pads) .. 649.6, Y 18.0..34.4, Z 4.4..25.6; holes at (Y 31.9,
  Z 6.9) and (Y 31.9, Z 23.1); header line Y=19.5, Z 6.11..23.89.
  AXIS MAP for software: board +X (hole N -> hole P) = gun -Z; board +Y (toward the header) = gun -Y;
  board +Z (component-side normal, MPU-6050 +Z) = gun -X. Inverse: gun +X = -Z_imu, +Y = -Y_imu,
  +Z = -X_imu.
  * Carried by an inboard column (X 645..648, Y 17..23, Z -8..5.7) and an outboard rail (X 645..648,
    Y 28.8..34.9, Z -8..27.1), both grounded to the bed so nothing overhangs; a face relief pocket
    (X 647..648, Z 10.9..~17, sloped ceiling) clears the board's edge passive. 3.0 mm component gap,
    components face the cap.
  * 2x M2.5 x 10 SHCS from the REAR through 2.8 x 4.8 slots (+/-1.0 mm of Z adjustment for the
    [INFERENCE] +/-0.5 hole pitch) into plain M2.5 nuts fitted on the board's solder side; the heads sit
    in 5.0 x 7.0 x 2.6 deep counterbore slots in the rear face so nothing protrudes behind X=640 into
    the pod wall zone.
  * 5-wire lead: 1.8 x 18.6 slot through the plate at Y 18.6..20.4, Z 5.7..24.3, directly behind the
    header, into the bay on the +Y side (and it clears the supplied 8-pin header's pins, which end at
    X=639.5). Gaps: IMU->Pi 2.40, IMU->sled 5.05, IMU->camera 3.12.
- FOV: the whole carrier is behind the lens front plane (cap max X 655.5, IMU max X 652.6 vs lens tip
  675.6), so nothing lies inside the 120 deg full cone that opens forward from the lens tip.
- Hardware: 4x M3 x 12 SHCS, 2x M2 x 6 self-tappers, 2x M2.5 x 10 SHCS + 2x M2.5 nuts (+2 washers),
  2x 2.2 x 5.1 x 1.5 silicone pads.

### Battery box — parts/batt_box + parts/batt_lid (PETG) — REV E.2: LEFT SIDE OF THE BUTTSTOCK
User decision 2026-10-26: strap the battery to the SIDE of the stock, gun-LEFT (user shoots
right-handed and chose the left side). Placed in the rear part of the stock, behind the cheek
zone: box X -324..-224 (rear edge 41 mm from the butt face, X -365), Z -39..+29 — its top is
15 mm below the comb top (Z 44), so the upper comb stays clear for the cheek. Pack 70 (X) x 58
(Z) x 20 (Y) lies flat against the stock; refs/batt_3s2600.step+ placed rotate [1,0,0] 90 deg,
at=[-276, 36.2, -5] (world X -311..-231, Y 26.2..46.2, Z -34..24).
- Stock left face (refs/gun_body WidthSkL): Y 20.8 @X-324 -> 19.4 @X-236, R8 long edges; flat
  zone Z from (toe line + 8) to (comb - 8).
- batt_box (tub, opening faces OUTBOARD +Y): X -316..-236, Z -39..29, base Y 22.2..25.2
  (3.0) against a 2 mm closed-cell neoprene pad on the stock (measured min gap 1.43 mm at the
  rear, ~2.8 at the front -> the pad takes up the taper), walls 4.0, cavity 72 x 60 x 22
  (X -312..-240, Z -35..25, Y 25.2..47.2; pack + 1 mm foam all round). Rim Y=47.2.
  4 screw ears on the X ends (X -324..-316 / -236..-228, Z -30..-20 and 10..20, full height)
  with M3 heat-set inserts D4.0 x 6 down from the rim at (X -320 / -232, Z -25 / 15).
  Cable notch through the front (grip-side) wall X -241..-235, Z -10..0, Y 32.2..rim (closed by
  the lid); nose tab X -236..-224, Z -14..4, Y 22.2..26.2 with two 3 x 4 tie slots for strain
  relief. Print base-down, build_dir [0,1,0] (gun +Y up), no supports. 62 g PETG.
- batt_lid: plate X -324..-228, Z -39..29, Y 47.2..51.2 (4.0), 4x D3.4 holes for M3 x 8 SHCS,
  two strap channels 26 wide x 1.2 deep across the outer face at X -310..-284 and -268..-242,
  1.0 chamfer on the outer edges. Print inner face down, build_dir [0,1,0], no supports. 27 g.
- Retention: two 25 mm hook-and-loop/webbing cinch straps (~450 mm) lie in the lid channels and
  go round the stock (over the comb, under the toe line) at X ~-297 and ~-255 - both behind the
  cheek zone. Closure on the right side or underneath, never on top of the comb.
- Verified (throwaway asm + asm/tbot_trigger_cam): no interference; box->stock 1.43, pack->stock
  5.67, pack->box/lid 1.0 (foam). verify_battery pass, dfm_pass true (no warnings) for both parts.
- Lead: out of the front notch, forward along the stock's left side and the receiver's left
  side (past the trigger actuator, which is also on the left: short 12 V branch), then along
  the forend underside to the pod's rear cable exit.
- REV E.3 ergonomics (user, 2026-10-27: "improve the ergonomics ... remove the sharp edges/corners") - SUPERSEDES the
  ear/tab/lid geometry above; cavity, pack placement, rim Y 47.2 and strap channels unchanged.
  - batt_box: ears merged into one rounded slab X -324..-228, Z -39..29, Y 22.2..47.2, R12 outer corners (matches lid),
    1.5 x 45 deg chamfer round the whole stock-side (bed) edge incl. the tongue. M3 heat-set inserts moved to
    (X -318 / -234, Z -21 / 11) - 4 mm wall to outside and to the cavity. Cable notch now runs straight out through
    the front wall X -241..-226.5, Z -10..0, floor Y 26.2 flush with the tongue top (no step for the lead), exit
    edges R1.5. Nose tongue X -228..-214, Z -16..6, 4 thick: R5 tip corners, R3 root, R1.2 top edges; tie slots
    3 x 4 at X -222..-219, Z -13..-9 / -1..3. 63.0 cm^3 (solid; was 48.8 - the filled ears, mostly infill).
  - batt_lid: R12 corners, R3 outer rim, 4x M3 flat-head countersinks (90 deg, D7.0 at the face, 1.8 deep) at the
    new insert positions -> heads flush; hardware is now 4x M3 x 8 ISO 10642 countersunk (was SHCS). Strap-channel
    lips R0.8. 20.7 cm^3.
  - Verified: verify_battery pass both; dfm_check(ender 3, build_dir +Y) lid pass; box fails only on the 4 corner
    + tongue faces of the nominal 45 deg bed chamfer measured at 45.6-46.0 deg (tessellation of the conical
    corner patches, 1-9 mm^2 each at Z 0-1 mm) and warns the notch as a "horizontal hole" (it is open to the
    top in print orientation); wall_thickness min 1.24 / 1.38 mm, no red. Assembly (tbot_trigger_cam7): clear,
    box->stock 1.45, pack->box/lid 1.0, pack->stock 5.67.
  - 2026-10-27 print prep: bed chamfer changed to 1.5 x 1.2 (steeper than 45 deg) -> dfm_check(ender 3, +Y) PASS on
    batt_box (the 45.6-46 deg corner faces are gone). Sliced on plate tbot_p4 (see design/BOM.md).

## REV F — arming switch, external hose clamps, verified print set (user, 2026-10-26)
User: "make a place for this momentary switch too since it's going to act as an arming trigger, model in the
hose clamps ... verify manufacturing". Decisions: arming button on the right side of the trigger guard on
trig_clamp_R under the index finger (user pick); printer = Creality Ender 3 (user). The REV E.1 hose-clamp path
(bands through the pod's open-bottom V-blocks) could not be closed: the band had no path from the barrel top to
the V-face channels without passing through the station arms. REV F wraps the bands round the OUTSIDE of the pod.

### Arming trigger — REV G (refs/btn_e429 + parts/trig_clamp_R) — SUPERSEDES the REV F paddle/flexure
User (REV G): "put the momentary switch into a better place ... lets me still have my hand in the correct grip so i
dont have to take it off and put it forward to press the button". The REV F paddle (X 22..54.5) sat 22-55 mm ahead of
the trigger face — out of reach from a firing grip — and hid the button behind a flexure. Paddle, flexure, leaf anchor,
guard post, spine and the X 62..96 housing are deleted.
- Concept: the button itself is a second trigger just outboard of the guard. The right hand stays on the pistol grip in
  the normal firing grip; the index finger curls as for the trigger but stops outboard of the guard and pulls the
  yellow cap REARWARD (-X) with its pad. Cap faces FORWARD (+X); cap, bushing and hex nut are visible on the housing
  face (stock panel-button look).
- Button (measured REV F from the user's photo, 7.78 px/mm): cap D5.5 x 4.2, M7 bushing 8.6, flange D9.3 x 3.0,
  body D8 x 9.3, heat-shrink D6 x 10.9, twisted pair; hex nut ~9.5 AF x 2. Travel not published - [INFERENCE] 1.5-2 mm.
- Pose (gun frame): axis along X at (Y -15.0, Z -7.0); cap tip X -4.0 at rest. The gun's trigger face at that height
  is X +2.8, so the reach is 6.8 mm SHORTER than the trigger's; 15 mm right of the trigger centreline, 7 mm below the
  trigger contact point (origin). btn_e429 placement (row-major, local -> gun): [0,0,1,-13.8; 1,0,0,-15; 0,1,0,-7]
  (local +Z -> +X, nut flats horizontal), lock-mated flange-top <-> panel back face.
- Housing (trig_clamp_R base feature): X -41.0..-13.8 (REV H.1; was -44.5), Y -22.5..-5.5, Z -22.3..+1.0 — top stays under the crossbolt
  keep-out (Z +3; 6.5 mm below the bolt). Outboard underside 41 deg from vertical (Y -22.5, Z -13 -> Y -14.5, Z -22.3).
  Front face X -13.8 is the panel face: hex nut sits ON it (9.5 mm / 3/8" open-end wrench from outboard), panel
  X -16.8..-13.8 with a D7.4 50-deg teardrop, flange seats on the panel back X -16.8, D10 truncated 50-deg teardrop bore
  X -41.0..-16.8 (flat top Z -1.2, 3.3 wide), open at the rear: the switch goes in cap-first from the rear.
- Finger pocket in front of the cap: inboard wall Y -8.5..-5.5 = ArmTie (X -14.5..-2.5, up to Z +1) + FingerShield
  (X -3..20, Z -4.5..+8, 4 mm top-front chamfer), 6.5 mm from the cap axis. It is also the pinch barrier: the
  fingertip cannot enter the guard opening where the PQ12 carriage / MG996R crown sweep (Y >= -4.5; 1.0 mm to
  trig_finger, 1.6 to servo_lever). ArmChin under the cap: X -14.5..+2, Y -13.2..-5.5, Z -22.3..-16.5.
- Cross screws are now COUNTERSUNK (4x M3 ISO 10642) in 94-deg seats, r 3.1 at Y -8.6: heads ~0.3 mm proud, so the
  finger zone carries no screw heads. Lengths unchanged (M3x20 upper, M3x16 lower); thread engagement -0.3 mm.
- Lead: out of the open rear of the bore, down the rear face into the 3.0 x 2.7 underside channel (Y -11.5..-8.5,
  Z -22.4..-19.7, X -41.1..+2.1), out under the chin, along the underside of the plate/lip, then receiver/forend underside to the pod's
  rear cable exit (~450 mm, tape with the 12 V lead).
- Safety: the cap faces +X, so no flat surface the gun is laid on (either side, butt-down, upside-down) can press it;
  only a rearward pull arms. NO contact; firmware treats open as disarmed.
- Structure: the housing cantilevers 27.5 mm behind the plate (REV H.1; the stresses below were computed for REV G's 31 mm and are now conservative). Joint section at X -6 (tie + chin) 97 mm2, plate + lip at
  X 4 80 mm2: 4.4 / 4.8 MPa peak at a 35 N axial load (finger abuse, or recoil inertia [INFERENCE ~12 g x ~300 g]) vs
  PETG ~45 MPa; the stress lies in the layer plane.
- Verified: verify_battery pass (attachment 1.0); dfm_check(ender 3, +Z) pass, 0 support; wall_thickness red only on
  the R1/R1.5 rounded end edges (0.75-0.94, walls behind them 2.2/2.5; DfM perimeter min 1.50); PETG "strength" slice
  14.1 g / 1.6 h, print_ready, gcode_check pass. Assemblies: no interference; button seated on the panel (contact);
  6.8 mm to the gun; housing rear 26.9 mm from the grip. PQ12 sweeps (m6 3->16.8; m6 16.8->21.1 with trigger m4
  0->-9.4) and MG996R sweeps (m4 -17->+2; +2->+17 with trigger m5 0->-9.07) clear on every frame. The 0.48 mm
  gun_body <-> clamp minimum is the unchanged LipR top vs the guard-bow spline sag at X 13.7 (shim/pad zone).
- Fit check on the gun before final use: with the hand in the firing grip, the pad should land on the cap without
  moving the hand; cap travel must click before the cap bottoms.

### Arming trigger comfort — REV H.1 (user, 2026-10-26: "optimize the ergonomics more for the comfort of the shooter")
- The "free-floating" button the user saw was a stale tab (doc-cpp00000031) still carrying the REV F button mate m22,
  whose clamp face no longer exists; m22 was replaced by the REV G lock (flange top <-> panel back), button now seated.
- Finger pad (parts/arm_pad, PETG): the switch is a dead-man that is HELD the whole time the system is armed, and a bare
  D5.5 cap concentrates that hold force on the fingertip. The pad is an 11 (Y) x 13 (Z) shoe glued (CA/epoxy) on the cap:
  Y -20.5..-9.5, Z -13.5..-0.5, rear face X -5.5, D5.8 x 1.6 socket over the cap (0.1 glue gap at the cap tip),
  concave R22 finger face (axis along Y, centre X 19.6 / Z -7) -> face X -2.4 at the centre, -1.7 at the rounded rims;
  R2.5 corners, R1 face rim. ~4.5x the cap's contact area; it also covers the brass nut from the finger.
  Reach: 5.2 mm shorter than the trigger at that height (was 6.8 with the bare cap).
  Travel: 2.7 mm before the pad meets the bushing front (X -8.2) — above the 1.5-2 mm [INFERENCE] switch travel, so the
  switch, not the pad, ends the stroke. 1.0 mm to the ArmTie wall on the inboard side (it guides the pad), 3.0 above the
  chin. Safety unchanged: the housing outboard face (Y -22.5) and the chin (Z -22.3) stand proud of the pad, so a flat
  surface still cannot press it; only a rearward pull arms.
  Print +X up (rear face on the bed), 1.0 x 0.8 bed chamfer, no supports: 0.6 g / 0.1 h.
- Housing 3.5 mm shorter (rear X -41.0): the heat-shrink ends at X -40.0, so the bore still covers it; rear end is 3.5 mm
  further from the middle finger. Lead now modelled as routed (refs/btn_e429 LeadOut/LeadDown/LeadAcross/LeadChannelRun):
  1.4 mm out of the rear, down the rear face, across to Y -10, forward in the channel (centre Z -21.0, flush with the
  bed face) and out under the chin — nothing points back at the hand.
- Verified: arm_pad verify_battery pass, dfm_check(ender 3, +X) pass, wall_thickness clean; trig_clamp_R verify_battery /
  dfm pass, wall_thickness red only at the housing rear edges (0.75 / 0.94, as REV G) and 0.994 at the chin-nose corner
  over the channel (convex chamfer corner, 1.2 wall behind). Rigs: PQ12 210 pairs clear, servo 28 pairs clear; PQ12 sweeps
  (m6 3->16.8; 16.8->21.1 with m4 0->-9.4) and MG996R sweeps (m4 -17->+2; +2->+17 with m5 0->-9.07) clear on every frame.
  Sliced PETG strength, print_ready, gcode pass: trig_clamp_R 13.1 g / 1.5 h, arm_pad 0.6 g / 0.1 h.

### Hose clamps (refs/hose_clamp_rear, refs/hose_clamp_front)
- 2x DIN 3017 W4 stainless worm-drive clamp, 9 mm band, range 70-90 mm. Modelled closed path 254.7 / 254.1 mm
  (equivalent dia 81.1 / 80.9 = mid-range).
- Path: over the lined barrel top UNDER the vent-rib plate (2.9 mm to the plate), straight to the station's
  top-outer 3 mm chamfer, down the outer wall in a 0.8 mm recess (Z 1.5..46.5, 9.4 wide), round the 1.6 x 2.0
  bottom chamfer and across under the floor. Nothing of the band is inside the pod - the sled and Pi are free.
- Rear band X 505.5..514.5 sits 0.5 mm behind rib post X 515 (recoil stop); front band X 611.5..620.5 between the
  posts at 605-609 / 650-654.
- Worm housing on the gun-right (-Y) outer wall at Z 14..28; screw head points DOWN at (Y -41.8, Z 9..14). A 7 mm
  nut driver / flat blade goes straight up from below: the D10 access envelope Z -60..+8.8 clears the pod by 1.3 mm.
- Snug the clamps (~1-1.5 N.m), do not crank: the band squeezes the pod walls.

### Pod REV F (parts/cam_mount_lower) - now prints STANDING ON THE REAR BULKHEAD (+X up)
- Why: the open-bottom V-block arm tips are downward-pointing ridges; printed +Z the slicer laid them as 27 mm
  unanchored strands (verify_gcode floating_extrusion fail). Standing on X 486 every arm is a vertical prism.
- Rear station extended back to X 486 (merges with the bulkhead, sits on the bed). Front station profile now runs
  from X 584.5 with a 44 deg lead-in in plan (|Y| from 35.5 at X 584.5 to 9.3 at X 611) and a 42 deg rise above the
  wall top (Z 29 at X 584.4 -> Z 48.6 at X 606.3); full profile from X 611. Small 39 deg wall-strip ramps
  X 582.4..584.6 at |Y| 30.4..32, Z 27.5..29.1. All above Z 29, so the sled envelope is untouched.
- Internal band channels removed; outer band recesses + 1.1 mm 39 deg end ramps; top-outer station chamfers 3 mm;
  bottom-outer edge chamfer 1.6 (Y) x 2.0 (Z).
- Cap bosses X 630..640 (|Y| 35.5..42.5) with 43 deg ramps from X 622.5; inserts unchanged (Y +/-38, Z -2 / 26).
- -Y windows: ONE port window X 553..610 (+X apex to 616.6), Z 2.4..14 - covers Pi 3B micro-USB/HDMI/audio and Pi 5
  USB-C/micro-HDMI; cable exit X 491..499 (+X apex 502.2), Z 2.4..8; vents (both walls) 9 x 9 with 45 deg +X points.
- Floor windows are pentagons |Y| <= 26 with 45 deg +X points (X 496..542, 546..592, 596..632).
- Tie anchor X 522..531 (window 524.5..528.5) with a 40 deg ramp; cap-tab floor notches X 637.9..640, |Y| 29.2..32.
- Measured: pod<->Pi 5 envelope 2.15, Pi 3B 1.5, XL4015 1.5; slide-in clear on all 17 frames.
- Print: X-up, no supports, 4 walls / 35 %: 93.7 g, 12.4 h (CAD 117.8 g at 100 %).

### Manufacturing verification (Ender 3, PrusaSlicer vendor presets)
- Every printed part: verify_battery pass (feature attachment 1.0 on every part since REV G), dfm_check(printer="ender 3") pass at its
  print direction, slice_part print_ready with verify_gcode pass/warn, NO supports. ASA has no Ender 3 vendor
  filament: the two ASA parts are sliced with Generic ABS at 255/100 C - print them in an enclosure/draft shield.
- Fixes made for the print set: pod_cap pusher tabs now run to the bed (Z -8..0.5, no overhang); pod_sled SD-notch
  0.2 mm sliver removed; trig_finger rear face X 26.5 -> 26.7 (1.0 -> 1.2 skin over the pin hole; 0.3 to the PQ12
  body at rest); trig_bracket_L finger slot opened to the top (the 32 mm bridge over it could not print);
  servo_bracket_L clamp lip run to the bed (0.4 mm wall under the lower inserts removed); servo_lever crown moved
  0.5 mm along its face plane (0.5 mm ledge removed), ramp draft -55 deg.
- wall_thickness (inscribed sphere) still marks red only at convex edges and ramp lead-ins, not walls: the 45 deg
  lip tips of trig_bracket_L, the crescent edge of trig_finger (frozen contact interface), the servo_lever crown
  ramp edge, the rounded end edges of the trig_clamp_R switch housing (REV G) and the pod ramp lead-ins/feature
  seams at X 584.6 and 611. dfm_check's perimeter wall check passes on all of them (pod min 1.86).

### Assembly sequence — PROVEN, see design/assembly_sequence.md (REV G study; supersedes the REV E.1 summary below)
Bench: Pi + XL4015 onto the sled, wire buck -> Pi 5 V, plug the camera FPC into the Pi CSI,
IMU lead onto GPIO. Gun: liner + 2 hose clamps -> pod on barrel. Slide the sled in from the
front (FPC trailing out the front), camera onto the cap (FPC into J1), cap onto the pod (4x M3).
Battery box strapped under the stock, 12 V lead forward along the gun-right side into the
pod's rear cable exit.

### Camera/Pi pod carrier (parts/cam_mount_lower) - REV D, hose-clamp saddles  — SUPERSEDED by REV E
ONE printed part, ASA (document material key `abs` = registry stand-in), modelled in the gun
frame at X 486..604, Y +/-34.5, Z 0..44. 63.5 cm^3, 66 g in ABS (~68 g in ASA).
- Tray: floor Z 0..3 (X 486..604, Y +/-34.5, lightened by three floor windows), side walls
  Y +/-31.5..34.5 x Z 2.5..30 with lightening windows, rear bulkhead X 489.5..497 and
  camera backplate X 596.5..600.5 (Y +/-19, up to Z 44) tying the two saddles together.
- Saddles: two stations on the bore axis (Y=0, Z=49), outer r18.7, cradle contact cone
  R = local barrel radius + 1.5 (13.568 at X486 -> 12.956 at X606, i.e. 13.489 at X501.5 and
  13.045 at X588.5) with relief pockets between the contact ribs. Top edges cut at Z=44, so
  the cradle rim stops at Y +/-12.53 / Z 44 - entirely below the barrel centreline; the bands
  pull the pod on. 1.5 mm silicone sheet (or 2 wraps of 1 mm rubber tape) lines the barrel
  under both cradle and bands.
- Support-free saddle profile (prints floor-down, no supports): the saddle underside is cut by
  44 deg gable planes meeting at Y=+/-3 (cradle station) / the band-channel V (apex Z=31.03,
  shoulders Z=41.90 at Y=+/-16.5), so every downward face is <=44 deg from vertical. At the
  band stations those V faces are chords of the r17.5 channel circle between phi=29.74 deg and
  62.26 deg from bottom dead centre, so the band lies flat on them.
- Hose clamps: band channels 9.0 wide x 1.2 deep, floor r17.5 about the bore axis, at
  X 497..506 (rear) and X 584..593 (front), between r18.7 shoulders. Required closed inner
  circumference 98.4 mm rear (= 31.33 mm equivalent diameter) and 97.4 mm front (= 31.01 mm):
  use 2x 8 mm band stainless worm-drive clamps, nominal range 25-40 mm (DIN 3017-1, W4/A2) -
  the closed diameter sits just under mid-range. Band wrap 146.8 deg on the saddle +
  134.1 deg (rear) / 129.6 deg (front) on the lined barrel, joined by 11.1 / 11.7 mm tangents.
  The band crosses the barrel top 3.5 mm under the vent-rib plate and runs between the rib
  posts (474 / 515 for the rear band, 564 / 605 for the front): the forward post of each pair
  is the recoil stop. Worm-housing seats: r9 radial dishes 30 deg down on the gun-RIGHT flank,
  floor 17.6 from the bore axis, at X 501.5 / 588.5 - away from the shooter's eye; the
  14 x 10 x 20 housing nests on the flat V chord and stays above Z=26.8, clear of the Pi.
- Raspberry Pi 3B: 4x M2.5 heat-set insert bosses, tops at Z=6, at (500.5, +/-24.5) and
  (558.5, +/-24.5), bores D3.6 x 4.5 deep; board PCB bottom Z=6, X 497..582, component side
  up, 8.8 mm from the barrel. SD card slot through the rear bulkhead.
- Camera (innomaker CAM-MIPIOV9281 V2): PCB back face seats on FOUR pads whose faces are on
  X=604 - lower pads Y +/-(11..18) x Z 0..6.5, upper pads Y +/-(11..18) x Z 24..34.5 on the
  backplate with a 45 deg under-chamfer; D1.8 x 6 pilots at (Y +/-14, Z 3) and (Y +/-14, Z 31)
  = the board's 28 x 28 M2 pattern. 4x M2 x 6 self-tapping screws. The floor is notched
  X 600.9..604.2 x Y +/-11 (full depth) for the J1 FPC connector and the cable drop, and the
  upper pads are relieved to Y +/-13.25 over Z 28.4..34.6 for the J2/J3 headers. The FPC window
  through the backplate is Y +/-11, Z 5..33 with a 41.3 deg gable roof.
- Print: ASA, build direction +Z (floor on the bed), 44 mm tall, first layer 5375.8 mm^2,
  0.4 nozzle, NO supports. dfm_pass true (0 mm^2 overhang, 0 mm^3 support, max bridge 0.0,
  min wall 1.50). verify_battery pass.
- Hardware: 2x 8 mm worm-drive clamp 25-40 mm; 1.5 mm silicone liner over X 490..600;
  4x M2.5 heat-set inserts + 4x M2.5 x 6 screws (Pi); 4x M2 x 6 self-tappers (camera).
- Interference-checked against refs/gun_body, refs/rpi3b and refs/cam_inno9281 at their
  installed poses: no overlaps; barrel gap 1.497 (the liner), Pi and camera seat at 0.000.

### IMU: 1x MPU-6050 GY-521 near the muzzle (user: the stock stays planted on the  — RETIRED in REV E (IMU moved to the pod cap)
shooter, so the IMU goes toward the end of the gun)
- GY-521 (common breakout, [INFERENCE] +/-0.5): PCB 21.2 x 16.4 x 1.6, parts 1.5 tall on
  the top side; 8-pin 2.54 header row along one 21.2 (LONG) edge, pin centres 1.5 from
  that edge (VCC GND SCL SDA XDA XCL AD0 INT); two D3.0 holes at the corners of the
  opposite long edge, 2.5 from that edge and from each short edge (16.2 apart).
  Chip axes: X along the 21.2 side. Header pins protrude 3.0 through the solder side.
- parts/imu_clamp_lower (REV D, hose-clamp saddle; parts/imu_clamp_upper is SUPERSEDED
  and no longer used). ONE printed part, ASA, modelled in the gun frame at X 746..783,
  Y +/-16.71, Z 24.6..48.27 (37 x 33.4 x 23.7, 8.2 cm^3, 8.7 g in ASA).
  * Saddle: outer cyl R17.5 about the bore axis (Y=0, Z=49); cradle R12.25 = local
    barrel radius (21.47..21.10 OD over X 746..783) + 1.5 silicone liner. Top edges cut
    by the 45 deg planes Y+Z=60.5 / Z-Y=60.5, so the cradle rim stops at Y=+/-12.23,
    Z=48.27: 173.2 deg wrap, entirely below the barrel centreline Z=49, band pulls it on.
    Barrel contact is TWO ribs (X 746..753 and 771..783); X 753..771 is relieved to
    R13.05 - 51% contact area, for heat.
  * Hose clamp: band channel 9.0 wide x 1.2 deep (floor R16.3) at X 772..781, with a
    square aft wall and a 31 deg forward retaining ramp (X 781..783). The band runs under
    the rib plate (5.0 gap over the barrel top at X 776.5) between the posts at 740-744
    and 785-789: 28 mm aft margin, 4.0 mm forward margin - the FORWARD post is the recoil
    stop. Required closed inner circumference 91.0 mm (= 29.0 mm equivalent diameter):
    use an 8 mm band stainless worm-drive clamp, nominal range 20-32 mm (W4/A2).
    Worm-housing seat: 15 dia x 2.5 deep dish on the gun-RIGHT lower flank (axis 45 deg
    down-right through (777.5, 0, 49), floor 15.0 from the bore axis, 2.75 wall), away
    from the shooter's view; the housing nests in it and projects outward.
  * Liner: 1.5 mm silicone sheet (or 2 wraps of 1 mm rubber tape) wrapped round the
    barrel over X 746..783, under both the cradle and the band.
  * IMU tray: X 746..772, Y +/-10.0, Z 24.6..35.0, fused straight into the saddle wall
    (no cantilever). Board cavity X 748.1..772.0, Y +/-8.5, Z 24.6..29.5, open DOWNWARD
    (board goes in from below, last) and forward. Board horizontal, component side DOWN,
    PCB top face seated FLAT on the cavity ceiling Z=29.5 for rigid coupling; PCB
    27.9..29.5, components 26.4..27.9, skirt to 24.6 (1.8 below the parts). Header end
    toward the REAR on the gun-RIGHT long edge (Y=-6.7); solder-tail groove Y -5.0..-8.5,
    Z 29.5..33.2. Installed GY-521 pose = Rx(180 deg) + (759.0, 0, 29.5); board occupies
    X 748.4..769.6 => 71.4 behind the muzzle (X 841); PCB-to-barrel air gap 8.8
    (6.97 to the protruding header pins) - both >= 5.
  * Retention: 2x M2.5 x 6 screws up through the board holes (X 750.9 / 767.1, Y +5.7)
    into captive M2.5 hex nuts (5.0 AF x 2.0) in slotted channels 7.8 (X) x 5.2 (Y) x 2.2
    at Z 30.9..33.1 - nuts load from the gun-LEFT face through 5.9 x 2.0 slots, are held
    against rotation by the 5.2 Y-walls and float +/-1.0 in X; the screw clearance slots
    (5.2 x 3.2) and a 2.1 deep tip relief above each nut give the same +/-1.0. That covers
    the +/-0.5 hole-spacing tolerance of the breakout.
  * Cable: 5-wire lead leaves the header downward, turns aft and exits a 3.0 x 1.6 tunnel
    through the tray's aft web (Y -2..-5, Z 26.0..27.6); a 2.5 mm cable tie through the
    two flanking slots (Y +1.4..-0.2 and -6.6..-8.2) is the strain relief. ~180 run back
    to the Pi GPIO along the barrel underside.
  * Print: ASA, build direction +X (stands on its aft face, bore axis vertical), NO
    supports, 0.4 nozzle. dfm_pass true (0 mm2 overhang, max bridge 3.25, min wall 1.0).


## Safety pendant (parts/pendant_body, pendant_cap) — SUPERSEDED in REV F by the trigger-guard arming switch (not printed)
- Gebildet E429 7 mm momentary NO (listing): thread D7 (M7x0.75 [INFERENCE]),
  total length ~27, nut ~9.5 AF x 2 [INFERENCE], 1 A. Panel hole D7.2, panel 2.5-4.
- Hand-held dead-man grip: button on top under a guard ring; cable exits bottom.

## Materials + barrel heat (REV D)
- Barrel-adjacent parts (cam_mount_lower, pod_cap; REV D also imu_clamp_lower) print in ASA (Tg ~100 C; PLA and PETG
  soften/creep on a barrel that is hot after a day of 5-stand). The CAD registry has no ASA identity,
  so those documents carry ABS as the stand-in (density/stiffness within ~3%). Upgrade path with an
  enclosed, hardened printer: PC or PA-CF. All barrel contact goes through a 1.5 mm silicone liner
  on relieved contact ribs. Non-barrel parts (both trigger sets, pendant) stay PETG.
- 2026-10-27 (user): no ASA-capable printer (stock open-frame Ender 3). cam_mount_lower and pod_cap are now stored and
  sliced as PETG, with fiberglass insulation between the barrel and the print in place of the silicone-only liner.
  ASA/PC remain the upgrade path; inspect the pod for creep after long hot sessions.
- 2026-10-27 (user): PLA for every part off the barrel — trig_bracket_L, trig_finger, trig_clamp_R, arm_pad, batt_box,
  batt_lid (stored material set to PLA). Barrel-mounted parts stay PETG: cam_mount_lower, pod_cap, pod_sled
  (user confirmed the cap and sled stay PETG). Print files carry the material in the name (tbot_PLA_* / tbot_PETG_*).
- Barrel mounting = stainless worm-drive hose clamps passing UNDER the vent-rib plate between rib
  posts (posts 4 long every 45 from X=200 [INFERENCE from photos]); the posts locate the band
  axially (recoil) and in rotation. CHECK ON THE REAL GUN that a 9 mm band slips under the rib
  plate at X~510 and ~616 before printing (REV F band positions).

## Files
- refs/*.step+: reference geometry, not printed (gun, trigger, PQ12, MG996R, Pi 3B, Pi 5 + AI HAT+ envelope,
  innomaker camera, GY-521, XL4015, 3S 2600 mAh pack, E429 arming button, hose_clamp_rear/front).
- parts/*.step+ printed (2026-10-27 material split). PETG (barrel-mounted): cam_mount_lower (pod body), pod_cap,
  pod_sled. PLA (off the barrel): batt_box, batt_lid, trig_bracket_L, trig_clamp_R (with the arming switch),
  trig_finger, arm_pad. MG996R set (not sliced in the plates): servo_bracket_L, servo_lever.
  NOT PRINTED (superseded/retired): cam_mount_upper, imu_clamp_upper (REV C split caps), imu_clamp_lower
  (REV D muzzle IMU saddle — it blocked the camera; the IMU now rides on pod_cap), pendant_body/pendant_cap
  (REV F: the arming switch lives on trig_clamp_R).
- asm/tbot_trigger_cam.step+: full rig (PQ12 trigger variant, pod, sled, Pi 3B, buck, cap, camera, IMU, battery
  box, arming button + arm_pad, both hose clamps), all mated. asm/pod_slide_study.step+: sled slide-in study (Pi 3B + Pi 5
  envelopes). asm/tbot_trigger_servo.step+: MG996R trigger variant (+ arming button). Root tbot_trigger_cam.step+ /
  imu_fitcheck.step+ are pre-REV-D files (stale).
- export/*.step: ISO STEP of every printed part; print/creality_ender_3_0_4_mm_nozzle/: verified G-code, 3MF and
  .print.json per plate (tbot_<MATERIAL>_pN); design/BOM.md: print + hardware list; design/assembly_sequence.md: assemblability study +
  build/service sequence.
