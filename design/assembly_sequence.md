# TBOT — assembly sequence

Assembly: `asm/tbot_full.step+` (front module + MG996R trigger set + harness). Fit evidence: [layout.md](layout.md)
"Verification". Hardware and wire cut lengths: [BOM.md](BOM.md). Gun frame: +X muzzle, +Y gun-left, +Z up.

**Gun unloaded, muzzle in a safe direction, battery lever connector OPEN throughout.**

## Front module

### Bench

F1. Heat-set inserts (iron square to the bore, flush or 0.2 mm below):
    - nose_clamp_L: 2 × M3 on the Y = 0 split face (cross screws).
    - nose clamps: 4 × M3 in the X 426.3 rear face (MI-1, 2 per half: floor (|Y| 12.5, Z 2.0), jaw (|Y| 19.5, Z 12.0)).
    - nose_clamp_R: 2 × M3 for the IMU.
    - carrier_L: 3 × M3 (1 belly at (433, +21.5); 2 in the rear-wall cover posts, along −Y at (X 433.4, Z 30.0 / −11.0))
      and 4 × M2 in the Pi standoff blocks (along −Y at X 440.35 / 498.35, Z 36.5 / −12.5).
    - carrier_R: 8 × M3 (2 belly; 3 cover_R: bulkhead top + bottom at (433.4, −43.5), bay bottom-rear at (401.4, −43.75);
      3 battery hold-down along +Y into the Y −46.2 boss faces: A (X 447.4, Z 35.0), B (X 447.4, Z −13.0), C (X 525.55, Z 11.6)).
F2. Buck #1 (Pi): terminals off; set 5.1 V on the bench supply first. Solder the 12 V feed to its IN pads and the Pi 5 V
    pair to its OUT pads.
    Buck #2 (servo): terminals off too; set **7.0 V** on the bench supply before anything touches its output. Solder the
    feed pair (cut 133 / 129) to its IN pads (bottom end), the servo pair (51 / 54) and the 1000 µF lead extensions
    (58 / 48, mind polarity) to its OUT pads (top end). Check real pad polarity: the model infers it.
F3. Pi: solder pins 2 / 6 (5 V pair), 1 / 3 / 5 / 9 (IMU), 11 / 14 (arming), 12 (servo PWM); keep the joints low (pin
    tips sit 8 mm from cover_L). Gather the seven signal wires into one bundle just below the header.
F4. Splices, heat-shrunk: SPL = jack pigtail onto the pack leads (pack side of the lever connector); S1 = 12 V from the
    lever −Y end to both buck feeds; B_R = signal bundle → IMU 4 + 3-wire exit.
F5. IMU: GY-521 flat in the nose_clamp_R floor pocket, components up, header edge to +Y; 2 × M3 × 6 SHCS. Board axes =
    gun axes (layout.md "IMU"). Lay the IMU bundle aft along the raised trough (roof Z 1.6) and down into the belly cavity at the trough's rear end (X 441.4, just ahead of B1; rev Z: the trough no longer runs over B1 or out of the rear
    face) before the bands go on.

### On the gun

F6. Back the sling-swivel stud out of the magazine cap, slide the M5 flat washer onto the shank, run the stud most of
    the way back in. The washer must be on before the clamp closes — the split bore captures the shank.
F7. Stick the 1 mm foam into both clamp jaws. Close nose_clamp_L (+Y) and nose_clamp_R round the forend nose and
    magazine cap so the front wall drops between the cap face and the stud eye/washer. 2 × M3 × 20 cross screws from
    gun-right (2 mm hex), (X 456, Z 4.5) and (X 505, Z 2.6), until the halves butt at Y = 0 with no gap. Back the stud
    out up to 0.5 mm to take the fore-aft slack. Check: no rock, no gap at the butt.
F8. Bands (SAE #44), directly on the clamp, before the carriers: 0.5 mm tape on the barrel at X 440.3 and 508.5. Open
    each clamp, thread the free end over the barrel under the vent rib, down the gun-left clamp groove, under the clamp
    in its underside groove (B1 under the trough with the IMU bundle inside; B2 under the nose saddle rib), up the
    gun-right groove and back over the barrel into the housing; housing on the gun-right upper run. B1 sits flush with
    the rear groove behind the 1.8 lip (X 428.4..441.1, 2.1 ahead of the rear face; it must pass through the rib window behind the post, not over a post). Snug both, check the bands lie fully in the grooves and the halves still butt
    at Y = 0, tighten, trim the tail and deburr. Cross screw #1 stays reachable; loosen B2 to re-torque cross screw #2.
F9. carrier_L / carrier_R onto the clamp rear face over the banded clamp: 4 × M3 × 10 (MI-1), driven +X from behind
    under the forend. The carrier inner walls sit 0.6 outside the bands and do not touch them.
F10. Camera: 200 mm FFC into J1 first, then the camera (camera_cover over the lens holder) into the nose (4 × M2 × 8
    through cover + PCB). The FPC slot is open to the nose front (X 514.5..524.2, |Y| ≤ 9), so the clipped-in FFC slides
    in with the camera. FFC: down only to just below the
    camera inside the FPC opening, bend it flat aft under the clamp nose, 45° crease to +Y at X 502.75..518.75 and
    run it straight across to the Pi lobe through the carrier_L FFC pass (X 502.25..519.35, Z −12.2..−10.6). Then stand
    it up flat against the inner wall (X 502.75..518.75), Z-fold the ~35 mm excess flat there, and leave the free end
    hanging above the lobe for F11.
F11. Pi (components facing out): Pi 5 V pair and the signal bundle through the carrier_L GPIO pass (X 444..462, floor
    level) and under the board's bottom edge; seat the board on the four standoff blocks, 4 × M2 × 5 into the M2
    inserts from the component side. Fold the FFC over the board's top edge (it lies in the 0.6 mm gap under the top
    skin), down outboard of the Ethernet jack, 45° fold aft, and into the CSI from outboard. Connect the IMU bundle and
    3-wire exit at B_R.
F12. R lobe, rear to front: buck #2 components-outboard onto the four bay-wall posts (X 400.4 / 423.1), 2 × M2 × 5 into
    the top posts; feed pair from the IN pads along the bay floor, out through the bulkhead's lower pass (outboard,
    low), forward outboard of the jack and in through the rear lobe pass (X 442.5..451) into the belly; servo pair from
    the OUT pads aft above the board's top edge and out the bay rear-wall hole (Y −38.0, Z 38.4). Then the jack up
    through the floor bore until its barrel stops on the shoulder. Capacitor into its wall clip above the jack (snap,
    leads up) and its leads over the bulkhead's upper pass to the OUT pads. Battery pair over the jack to SPL, forward
    at Z 11.6, down at X 450 / 448 and as two singles through the rear lobe pass into the belly; battery into its
    saddle, pushed against the inner wall and forward to the C anchor block. batt_retainer flat on the pack's outboard
    face (chamfered face to the pack), 3 × M3 × 8 countersunk into A, B, C; snug them in turn until the pack cannot be
    slid by hand. The plate must not touch the boss faces (1.2 mm gap on a 20 mm pack); if it does, the pack is thin:
    add 0.5–1 mm foam under the plate.
F13. Covers: cover_R — hold 2.4 mm forward, push inboard, slide back (hook at the front, X 527.6..531.6), 3 × M3 × 8
    vertical (bulkhead top + bottom, bay bottom-rear, into inserts); cover_L — same hook motion, then 2 × M3 × 8
    through the flank (countersunk, into the rear-wall post inserts).
F14. Belly, bench-wired (lever connector in its seat, buck #1 on its posts with 2 × M2 × 5): bring the Pi 5 V pair to
    the carrier_L GPIO pass and join the buck #2 feed at S1 (it comes out of the carrier_R rear lobe pass and runs
    along the wall under the battery singles) as the belly rises, 3 × M3 × 8 from below.
    Check no wire is pinched at either pass.

## Trigger set

T1. Iron 4 M3 inserts into servo_bracket_L ears RI, RO, FI, FO from the Z −21.7 face (remove the print supports first).
    Then run a 3.5 mm drill through the rear clamp bore (X −15.5) to clear any plastic the RI/RO inserts pushed in.
    The guard-clamp joints have no inserts: drop/slide 3 M3 hex nuts into the bracket's U18 (side slot from −X), U28
    (top slot, vertex down) and L40 (side slot from +X) pockets, and 1 into trig_clamp_R_v2's guard-cradle rear slot
    (from −X); a dab of tape holds them for assembly.
T2. Servo into the bracket from below, 4 × M3 × 10 up through the ears. Cable grommet points rearward.
T3. Arming button, on the bench: nut off, push the button thread-first from behind the trig_clamp_R panel through the
    D7.4 hole until the flange seats on the panel back; nut on the panel front, flats horizontal, tighten with a 9.5 mm
    open-end wrench from outboard.
T4. Bracket on the gun-left side of the guard, trig_clamp_R on the right, a 1 mm liner on each plate's guard face.
    Drop an M3 nut into trig_plug's top pocket and start the M3 × 18 jack from the plug's rear face until the tip is
    flush with the front face. Set the plug in the front of the guard opening, jack pointing forward. The guard bow must
    sit down in the clamp's cradle. 3 × M3 × 30 from the right (U18 and U28 go through the plug, L40 below the guard;
    heads flush in the hole pad), then 1 × M3 × 35 from the LEFT at the rear (X −15.5, Z −17.0: in through the
    bracket's back-face countersink, through the whole rear ear block, into the nut in the cradle). Snug the four bolts.
T4a. Jack: 2 mm hex key through the finger hole, short arm in the screw head, long arm out sideways. Screw in until the
    tip meets the guard's front inside wall and the whole set pulls back onto the cradle, then about a quarter turn
    more. Firm, not cranked: an M3 can make about 500 N. Now tighten the four through-bolts fully.
T4b. SAFETY CHECK before connecting the lever: with the servo at REST, there must be visible daylight between the lever
    and the trigger. If they touch, back the jack off and fix the fit (or move the firmware REST angle away from the
    trigger). The plug stops the lever about 7.7° past REST, the other way from firing.
T5. Leads: button lead straight out of the tail, bend down/forward (R ≥ 6); servo cable rearward out of the grommet,
    bend up (R ≥ 8). The printed parts carry no tie points; harness both leads separately to the module: servo power
    to the pair from the carrier_R rear-wall exit, servo signal (PWM, GND) and button to the 3-wire exit from B_R (BOM
    wire list). No sharp bends.
T6. Power-up with the horn OFF the servo: close the lever connector, check 5.1 V at the Pi header and 7.0 V at the servo
    lead. Centre the servo (firmware REST), press the horn on with the arm at 190° (gun frame; nearest spline tooth,
    trim the rest in firmware), seat servo_lever over it, M3 × 12 countersunk down into the shaft.

No hardware interlock: open the lever connector to make the servo safe.

## Service

- Buck #1, lever connector, splices: belly (3 × M3 × 8).
- Buck #2 and its capacitor: cover_R off (3 × M3); buck #2 sits in the rear bay and comes out with 2 × M2 × 5 on the
  top bay-wall posts without taking the battery out; the cap snaps out of its clip above the jack.
- Battery: cover_R 3 × M3, then batt_retainer 3 × M3.
- Pi, battery, jack: cover_R 3 × M3, cover_L 2 × M3 (the belly does not have to come down). The Pi lifts straight out
  after 4 × M2 (its top-edge ports leave through the open-ended top-skin notches); unplug the FFC at the CSI first.
- Camera: 4 × M2 from the front.
- Electronics service never touches the bands, so the camera/IMU zero on the clamp is undisturbed.
- Clamp cross screws: through the carrier_R access holes with the cover off and the battery out; #1 (X 456) is clear of
  the bands, loosen B2 for #2 (X 505).
- Charge: DC5521 from below at the rear of the battery lobe.
- Ports with everything installed: Ethernet + 2 × USB through cover_L's front windows; HDMI, micro-USB and audio
  through the top-skin notches with cover_L off; microSD from the rear (carrier_L rear slot).
- Module off the gun: belly, covers and carriers (MI-1) first, then bands, then 2 cross screws + back the stud off.
