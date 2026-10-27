# TBOT REV G — assemblability study + assembly sequence

Evidence files: `asm/pod_slide_study.step+` (pod + sled + Pi 3B + Pi 5/AI HAT+ envelope + XL4015, slide joint),
`asm/tbot_trigger_cam.step+` (full rig incl. arming button and both hose clamps, every component mated),
`asm/tbot_trigger_servo.step+` (MG996R variant + arming button). Gun frame: +X to muzzle, +Z up, +Y gun-left.

## 1. What was checked and the result

| Check | Method | Result |
|---|---|---|
| Can the Pi go in after printing? | Sled (`parts/pod_sled`) slides into the pod's rail grooves from the open front. `simulate_motion` on the slide joint, 160 mm out → seated, 17 frames, interference on every frame | **Clear on all 17 frames** (checks complete) for both the Pi 3B and the Pi 5 + Active Cooler + AI HAT+ envelope |
| Seated clearances | `check_interference`, seated pose | Pi 5 + AI HAT+ → pod 2.15 mm; Pi 3B → pod 1.5; XL4015 → pod 1.5; sled ↔ floor/grooves = sliding contact |
| Hose clamps fit | Both clamps modelled closed (`refs/hose_clamp_rear/front`) on the pod + gun | No interference. Band ↔ rib post 0.5 mm (rear band rides against the post = recoil stop); band 2.9 mm under the vent-rib plate; band ↔ sled 2.3 mm (the band never enters the pod) |
| Hose clamps installable | Band path is entirely outside the pod: barrel top (under the rib) → station top chamfers → side-wall recesses → under the floor | Fit with the pod already on the barrel; no part of the band passes the Pi/sled. Order vs. sled does not matter |
| Clamp screw access | Ø10 nut-driver envelope from the screw head (Y −41.8, Z 9) straight down to Z −60 | Clear: 1.3 mm to the pod, nothing else in the way |
| Arming trigger (REV G) | `refs/btn_e429` lock-mated in the trig_clamp_R housing (flange on the panel back, nut on the front face), both rig assemblies | Seated (contact only), no interference with the gun (6.8 mm), trigger (9.4), PQ12 set or MG996R set; housing rear 26.9 mm ahead of the grip; 6.5 mm under the crossbolt |
| Full rig static fit | `check_interference` on all 20 components of `asm/tbot_trigger_cam` | No interference (190 pairs) |
| Trigger actuators still work | PQ12: rod 3 → 16.8 mm, then 16.8 → 21.1 mm with the trigger driven 0 → 9.4° (5 + 5 frames). MG996R: lever −17 → +2°, then +2 → +17° with the trigger driven 0 → 9.07° (5 + 5 frames) | Clear on every frame, checks complete; the finger shield stays 1.0 mm from trig_finger and 1.6 mm from servo_lever |
| Camera view | Everything on the cap/IMU is behind the lens front plane | Nothing in the 120° cone. The barrel bottom still shows at the top of the image (mask in software) |
| Pi ports on the gun | One −Y window X 553–610 (+X point to 616.6), Z 2.4–14 | Covers Pi 3B micro-USB/HDMI/audio and Pi 5 USB-C/micro-HDMI ×2 |
| USB-A / Ethernet, microSD | Front end behind the cap / rear end on the sled | Cap off (4× M3) for USB/Ethernet; cap off + slide the sled out for the SD card |
| Battery | Box on the LEFT side of the buttstock | Unchanged from REV E.2: no interference, box → stock 1.43 mm |
| Manufacturing | `dfm_check(printer="ender 3")` + `slice_part` + `verify_gcode` for all 10 printed parts | All `print_ready`, no supports (see BOM.md for orientation, grams and hours) |

## 2. Open issues
1. **Arming-switch travel is not published.** The cap is pressed directly (no paddle, no hard stop of ours). Bench-check: the switch must click within a comfortable pull and open again when you let go.
2. **Firmware owns the interlock.** The switch is normally-open on a GPIO; treat an open circuit as "not armed" (a cut lead then disarms, it never arms).
3. **Pi 5 + AI HAT+ GPIO wiring.** With the HAT fitted, standard DuPont housings will not fit under the barrel; solder the GPIO leads (IMU, arming switch, actuator signal) or use low-profile connectors.
4. **Pi 5 CSI connector access** is tight between the HAT edge and the USB stack: plug the camera ribbon in on the bench first. [INFERENCE: connector position from the Pi 5 layout]
5. **Leads.** 12 V lead from the battery (left side of the stock, ~1.1 m) and the arming-switch lead (~0.5 m, out of the trig_clamp_R underside channel below the trigger guard) both run along the forend underside to the pod's rear −Y cable exit; tape/clip them under your palm.
6. **Heat.** Pi 5 + Hailo in an ASA pod under a hot barrel will throttle on hot days; watch `vcgencmd measure_temp`.
7. **ASA on an Ender 3.** No vendor ASA preset exists; the files use Generic ABS at 255/100 °C. Print the pod and cap inside an enclosure or draft shield.
8. **Rib clearance** for a 9 mm band at ~510 and ~616 mm forward of the trigger face is from photos — check on the gun before printing.
9. **Ergonomic fit is a hand check.** The cap sits where the index-finger pad lands in the normal firing grip, 15 mm right of the trigger and 6.8 mm shorter reach than the trigger [hand posture is INFERENCE, not measured]. Dry-fit with the gun unloaded before trusting it.

## 3. Assembly sequence

### Bench (no gun)
1. Press heat-set inserts: pod 4× M3 (cap bosses, X 640 face); sled 4× M2.5 (Pi standoffs); battery box 4× M3; trigger-left bracket 4× M3 (per BOM.md).
2. Sled: Pi on 4× M2.5×6. XL4015 into its fence, zip-tie through the 2 plate slots. Set the buck to 5.1 V **before** connecting the Pi; wire buck OUT → Pi 5 V/GND; tie the 12 V input lead to the rear anchor.
3. Camera ribbon into the Pi CSI (Pi 5: 22-pin end). IMU lead onto the GPIO (Pi 3B) or soldered (Pi 5 + HAT).
4. Cap: camera bottom edge into the two capture slots, 2× M2×6 at the top. IMU with 2× M2.5×10 + nuts. Stick the 2 silicone pads on the pusher tabs. Ribbon into camera J1 (top edge).
5. Battery box: foam lining, pack in, lead zip-tied at the nose tab, outer lid on with 4× M3×8, 2 mm neoprene on the base.
6. Arming switch: feed it cap-first into the OPEN REAR of trig_clamp_R's housing (leads trailing) until the flange seats on the panel and the bushing comes through the front face. Spin the hex nut onto the bushing from the front and snug it with a 9.5 mm (3/8") open-end wrench from the outboard side while holding the lead pair so the switch does not turn. Bend the leads down the rear face into the underside channel; they come out under the chin. Check: pull the cap rearward — it clicks and releases cleanly. Heat-shrink the lead where it leaves the channel.

### On the gun
7. Trigger actuator set (PQ12 or MG996R) + trig_clamp_R onto the guard per layout.md. The 4 cross screws are M3 countersunk (2 mm hex key, straight in from the right side; nothing sits outboard of them).
8. Wrap the 1.5 mm silicone liner round the barrel over X 490–640.
9. Lift the pod onto the barrel (rear bulkhead just ahead of the forend cap / sling stud).
10. Hose clamps (9 mm band, 70–90 mm), one per station, rear at ~510 mm and front at ~616 mm from the trigger face: unscrew each clamp until the band comes out of its housing. Feed the band tail over the barrel top UNDER the vent-rib plate between the rib posts, down into the side-wall recess on the gun-left (+Y) side, under the pod floor, up the gun-right (−Y) recess and back into the housing, with the housing on the −Y wall and the screw head pointing DOWN. The rear band sits just behind the rib post. Snug with a 7 mm nut driver from below (~1–1.5 N·m) — don't crush the walls.
11. Feed the 12 V lead, actuator lead and arming-switch lead in through the pod's open front and out the rear −Y cable exit (reach in through the floor windows to guide them). Slide the sled in until it stops on the rear bulkhead; tie the leads to the anchor.
12. Camera ribbon into the cap's top notch, cap onto the pod (IMU lead through its slot), 4× M3×12. The pusher pads preload the sled ~0.3 mm.
13. Strap the battery box to the LEFT side of the buttstock (straps in the lid channels, closures on the right or underneath, never on the comb). Run the leads forward along the forend underside.

### Service
- Pi ports: through the gun-right window with everything installed. USB-A/Ethernet: cap off (4 screws).
- SD card or Pi swap: cap off → slide the sled out. The pod and clamps stay on the gun.
- Pod off: loosen both clamps from below, open the bands, lift the pod down.
