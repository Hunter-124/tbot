# TBOT

Vision-assisted trigger release for clay shooting (skeet / trap, sporting use only), built around a
Weatherby Element shotgun. A monochrome global-shutter camera under the barrel finds orange clays against the
sky, an IMU-referenced tracker predicts the clay, and an actuator presses the trigger only while the shooter
holds the arming button.

## Sections and owners

| Path | Owner | Contents |
|---|---|---|
| `design/`, `parts/`, `refs/`, `asm/`, `export/`, `print/` | Hunter (mechanical) | CAD, STEP exports, print files, BOM, layout and assembly sequence |
| `software/` | Alex (software / vision) | Python package `tbot`: camera, IMU, clay detector, tracker, fire solver, actuator + arming interlock, calibration and latency tools |

Cross-section rule: every dimension the software uses comes from `design/layout.md`. Change the layout first,
then the config (`software/config/default.toml`).

## Hardware files

Units mm. Gun frame: +X muzzle, +Y gun-left, +Z up, origin at the trigger face.

| Path | Contents |
|---|---|
| `design/` | `BOM.md` (print list + purchased hardware), `layout.md` (every dimension, with source and clearance), `assembly_sequence.md` |
| `parts/` | Native `.step+` (editable feature trees), current parts only. Front module: `nose_clamp_L/R`, `carrier_L/R`, `cover_L/R`, `belly_cover`, `camera_cover`, `batt_retainer`. Trigger set: `servo_bracket_L`, `servo_lever`, `trig_clamp_R_v2` (the current `trig_clamp_R`, also the arming-button housing). Obsolete parts (`chin_lid`, first-gen `trig_clamp_R`) are not kept here |
| `refs/` | Native `.step+` reference models used by the assembly: gun body/trigger/measured guard, Pi 3B, OV9281 camera, GY-521, MG996R + horn, XL4015 ×2, 1000 µF cap, battery, lever connector, DC5521 jack, hose clamps, arming button, screws, modelled harness (`refs/harness/`) |
| `asm/` | `tbot_full.step+` — the assembly: gun, front module, trigger set, harness |
| `export/` | ISO STEP (AP214) of the parts (`export/*.step`), refs (`export/refs/`) and assembly (`export/asm/tbot_full.step`) — open these in any CAD tool |
| `print/creality_ender_3_0_4_mm_nozzle/` | PrusaSlicer output for a Creality Ender 3, 0.4 mm nozzle, PLA: one `<HH>h<MM>m_<plate>` set per plate (`.gcode`, `.3mf`, `.print.json`; the leading `.slice` folders are slicer sidecars). Slicer defaults: `print/print_defaults.json` |
| `tbot_front_harness_<id>.step+` (repo root) | Modelled front harness. Must stay at the root: `asm/tbot_full.step+` links it by path (see below) |

- `asm/tbot_full.step+` references its components by absolute path on the CAD workstation, so the files it links
  (`parts/`, `refs/`, `refs/hw/`, `refs/harness/ffc_cam_15p.step+` and the root harness file) must not be moved or renamed
  without re-linking the assembly. Use `export/asm/` anywhere else. `.aicadvault/` and `.plm/` are local CAD-tool
  state and are not tracked.
- `scratch/` is a local, untracked work area (scan pipeline, experiments, `archive_pre_v1.1/`); nothing in it is part of
  the release.
- Every printed part is PLA.

## Safety invariants (both sections)

- The trigger actuator may be commanded to fire only while the momentary arming button is **held**. This is a
  **software interlock only**: the button is a GPIO input (BCM 17, pin 11, normally open to GND, open = not armed). The
  servo rail (second XL4015, 7.0 V) is powered whenever the battery lever connector is closed — there is no hardware
  cut-off, so open the lever connector to make the actuator safe.
- Software defaults to dry run (`fire.live = false`). One shot per press.
- Bench work, latency tests and assembly happen with the gun **unloaded**, muzzle in a safe direction.
- Live fire is for clay targets on a range, under range rules only.

See `software/README.md` for the software safety layers.

## Hardware at a glance

- Raspberry Pi 3 Model B v1.2, mounted right side up on four rooted standoffs (M2 into M2 heat-set inserts)
- innomaker CAM-MIPIOV9281 (OV9281 mono global shutter), on 4 × M2 in the nose
- MPU-6050 (GY-521) IMU in the nose clamp trough
- Trigger actuator: TowerPro MG996R servo (`actuator.type = "servo"`). The software also supports an Actuonix PQ12
  (`actuator.type = "pq12"`); this build has no PQ12 hardware
- 7 mm momentary NO arming button in the trigger-guard clamp, pressed on its bare cap
- XL4015 buck #1 (5.1 V) for the Pi and XL4015 buck #2 (7.0 V, 1000 µF across its output) for the servo, both from the
  3S 2600 mAh pack in the front module through one lever-connector disconnect; DC5521 charge jack on the battery
  (gun-right) side, plugged in from below
- Front module (100.7 mm wide, 1.6 mm PLA walls, Pi lobe gun-left, battery lobe gun-right with the pack against the
  inner wall and buck #2 in a buck bay behind it; nothing printed ahead of the clamp nose) held by the forend-nose clamp, which two SAE #44
  (1/2" band) worm-drive hose clamps wrap directly in grooves cut into the clamp; the carriers bolt to the clamp and
  never carry band load
- Fasteners: every plastic-to-plastic joint is M3 into a brass heat-set insert; M2 only for boards whose holes are too
  small for M3 (Pi, camera, bucks)
- Harness and camera ribbon modelled in `asm/tbot_full` (wire list in `design/BOM.md`); the trigger harness is
  hardwired, no in-line connector

Full list: [design/BOM.md](design/BOM.md).

## Start here

- Build it: [design/assembly_sequence.md](design/assembly_sequence.md)
- Run it: [software/README.md](software/README.md)
- Vision work boundary: [software/docs/VISION_HANDOFF.md](software/docs/VISION_HANDOFF.md)
