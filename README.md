# TBOT

Vision-assisted trigger release for clay shooting (skeet / trap, sporting use only), built around a
Weatherby Element shotgun. A monochrome global-shutter camera under the barrel finds orange clays against the
sky, an IMU-referenced tracker predicts the clay, and an actuator presses the trigger only while the shooter
holds the arming button.

## Sections and owners

| Path | Owner | Contents |
|---|---|---|
| `hardware/` | Hunter (mechanical) | CAD, STEP exports, print files, BOM, layout and assembly sequence |
| `software/` | Alex (software / vision) | Python package `tbot`: camera, IMU, clay detector, tracker, fire solver, actuator + arming interlock, calibration and latency tools |

Cross-section rule: every dimension the software uses comes from `hardware/design/layout.md`. Change the
layout first, then the config (`software/config/default.toml`).

## Safety invariants (both sections)

- The trigger actuator may move only while the momentary arming button is **held**. Releasing it returns the
  actuator to rest. Wire the button in series with the actuator driver enable, not only into a GPIO.
- Software defaults to dry run (`fire.live = false`). One shot per press.
- Bench work, latency tests and assembly happen with the gun **unloaded**, muzzle in a safe direction.
- Live fire is for clay targets on a range, under range rules only.

See `software/README.md` for the full five-layer safety model.

## Hardware at a glance (BOM REV H)

- Raspberry Pi 3 Model B v1.2 (clearance kept for a Pi 5 + AI HAT+)
- innomaker CAM-MIPIOV9281 (OV9281 mono global shutter)
- MPU-6050 (GY-521) IMU on the pod cap
- Trigger actuator, two interchangeable sets: Actuonix PQ12-100-12-S **or** TowerPro MG996R
- 7 mm momentary NO arming button in the trigger-guard clamp, with a glued-on finger pad (`arm_pad`)
- XL4015 buck (5.1 V) from a 3S 2600 mAh pack strapped to the stock
- Barrel pod held by two 70–90 mm worm-drive hose clamps; PETG (fiberglass-insulated) for barrel-mounted parts, PLA elsewhere

Full list: [hardware/design/BOM.md](hardware/design/BOM.md).

## Start here

- Build it: [hardware/README.md](hardware/README.md)
- Run it: [software/README.md](software/README.md)
- Vision work boundary: [software/docs/VISION_HANDOFF.md](software/docs/VISION_HANDOFF.md)
