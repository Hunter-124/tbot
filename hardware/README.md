# TBOT hardware

Units mm. Gun frame: +X muzzle, +Y gun-left, +Z up, origin at the trigger face.

| Path | Contents |
|---|---|
| `design/` | `BOM.md` (print list + purchased hardware), `layout.md` (every dimension, with source and clearance), `assembly_sequence.md` |
| `step/parts/` | ISO STEP (AP214) of each designed part — open these in any CAD tool |
| `step/refs/` | ISO STEP of the reference models: gun body/trigger, Pi 3B, Pi 5 + AI HAT+, OV9281 boards, GY-521, PQ12, MG996R, XL4015, battery, hose clamps, arming button |
| `step/asm/` | ISO STEP of the assemblies: `tbot_trigger_cam` (full rig, PQ12 set), `tbot_trigger_servo` (MG996R set), `pod_slide_study` |
| `cad/` | Chudware native `.step+` (editable feature trees) for parts, refs, assemblies and fit `studies/` |
| `print/ender3_0.4mm/` | Sliced PrusaSlicer output for a Creality Ender 3, 0.4 mm nozzle: `.gcode`, `.3mf`, `.print.json` per printed part |

## Notes

- `cad/asm/` and `cad/studies/` reference their components by absolute path on the CAD workstation. Use
  `step/asm/` anywhere else.
- Not printed (superseded, kept for history): `cam_mount_upper`, `imu_clamp_lower`, `imu_clamp_upper`,
  `pendant_body`, `pendant_cap`. See the end of the print list in `design/BOM.md`.
- PQ12 set: `trig_bracket_L` + `trig_finger`. MG996R set: `servo_bracket_L` + `servo_lever`. Both use
  `trig_clamp_R` (also the arming-button housing).
- Barrel-contact parts (`cam_mount_lower`, `pod_cap`) are ASA. Do not print them in PLA.
