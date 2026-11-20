# TBOT

**A shotgun that decides when to pull its own trigger.** A camera under the barrel finds the clay, a tracker
predicts where it will be when the shot arrives, and a servo presses the trigger at the instant the two line up.
The shooter only swings the gun and holds a button.

Designed, built and calibrated by one person in 23 days (2026-10-25 to 2026-11-17).
First live range day: **two shots fired by the machine, two clays broken.** Second range day (2026-11-19):
rebuilt trigger, two trips, a lot of power cuts. The afternoon was somewhat successful.

![The phone's view of a hit beside the gun camera's own view, lined up on the bang](docs/media/range_2026-11-19/shot_phone_and_gun_view_hit.gif)

*2026-11-19, press 160243, a hit at 14 m. Left: phone video. Right: the gun camera with the tracker's overlay.
The two are lined up on the bang, real time then 4x slow. The red FIRE frame is the command; the recoil shows on
the IMU 70 ms later.*

![The assembled gun: trigger servo at the guard, camera and compute module clamped under the barrel](docs/media/hero_assembled.jpg)

| | |
|---|---|
| ![Front module: camera, Raspberry Pi, battery, clamped to the barrel](docs/media/front_module.jpg) | ![At the range, 2026-11-17](docs/media/range_day.jpg) |
| Front module: global-shutter camera, Raspberry Pi 3B, 3S battery, two hose clamps | First live range day, 2026-11-17 |

> For clay targets on a range, under range rules, with a person holding the arming button for every shot.
> See [Safety](#safety).

## What it does

Hitting a crossing clay is a timing problem. The target moves about 20 m/s, the shot takes around 80 ms to get
there, and the trigger mechanism needs 160 ms to respond. A human solves it by swinging through the target and
learning when to pull. TBOT solves it by measuring: where the clay is, how the gun is moving, how long every
delay in the chain takes, and then firing at the moment the predicted shot cloud covers the predicted clay.

```mermaid
flowchart LR
    CAM["OV9281 mono<br/>global shutter<br/>640x400 @ 100 fps"] --> DET["Detector<br/>matched filter +<br/>hysteresis"]
    IMU["MPU-6050<br/>gyro + accel"] --> ATT["Attitude filter<br/>gun pose in<br/>world frame"]
    DET --> TRK["Tracker<br/>3-D Kalman filter<br/>in world coordinates"]
    ATT --> TRK
    TRK --> SOL["Fire solver<br/>lead, P(hit),<br/>fire / hold"]
    ATT --> SOL
    BAL["Ballistics<br/>pellet drag, drop,<br/>pattern spread"] --> SOL
    BTN["Arming button<br/>held by shooter"] --> ACT
    SOL --> ACT["Servo<br/>presses trigger"]
```

1. **See.** A monochrome global-shutter camera with a 12 mm lens looks along the bore. There is no colour, so an
   orange clay is found as a small blob of local contrast against the sky, at 9 µs exposure in daylight.
2. **Know where the gun points.** An IMU tracks the gun's rotation so the clay's path can be separated from the
   shooter's swing. The clay is tracked in world coordinates, not image coordinates.
3. **Predict.** A Kalman filter estimates the clay's 3-D position and velocity (range comes from its apparent
   size). The solver extrapolates the clay and the gun swing forward by the full delay: processing, servo
   travel, lock time and shot flight.
4. **Decide.** For each frame the solver computes the probability that the pattern covers the clay if it fires
   now. Above the bar, with the button held, it fires. One shot per press.

## Results

### 2026-11-17: first live day

First live day, 2026-11-17 ([full data and analysis](software/field_data/2026-11-17-range/README.md)):

| | |
|---|---|
| Presses of the arming button | 9 |
| Fire commands issued | 7 |
| Shots that left the gun | 2 (then the printed servo mount broke) |
| Clays broken | **2 of 2** |
| Engagement range | 13 to 30 m |
| Trigger delay, command to ignition | 160.2 and 162.0 ms, measured from recoil on the IMU |

![The gun camera's frame at shot arrival: green circle is the pattern, yellow cross is the clay](docs/media/shot_scored.png)

*The gun camera's own view at the moment the shot arrived, drawn by `tbot score`. Circle: where the pattern
went. Cross: where the clay was. This shot at 27.8 m missed centre by 0.62° against a 0.75° pattern, and broke
the clay.*

What the day also showed, stated plainly:

- **The trigger delay in the config was wrong by 60 ms** (95 ms assumed, 161 ms real). A bench microphone
  measurement two days earlier had read 161.3 ms and I discounted it in favour of a video estimate. The
  microphone was right. Both clays broke despite the error: judged at the moment the charge really left,
  both shots passed 1.04° from the clay, fringe hits on targets where the extra 60 ms moved the bore little.
  The delay is corrected in the software since, along with what it exposed (see
  [software/docs/INTEGRATION_STATUS.md](software/docs/INTEGRATION_STATUS.md), last section).
- **Frame rate under load was 19 to 43 fps, not 100.** The detector takes 15 to 20 ms per frame on the Pi 3B.
- **The servo mount failed after two shots.** It has since been redesigned twice (through-bolted with trapped
  nuts, then an anti-slide lock on the guard).

Every press saves the raw frames, IMU samples, per-frame decisions and events, so each one can be replayed
through the same code on a laptop.

### 2026-11-19: second range day, two trips

**Morning.** About 25 clays were tried, about 5 were fired at, and 1 was hit. The Pi recorded 22 presses (four
lost to zero-byte files). They contain 3 fire commands: one set the gun off, and the other two gave no recoil.
The Pi kept losing power: at least 6 boots in three minutes, with no crash or shutdown in its log
([what went wrong and what changed](software/docs/RANGE_2026-11-19.md)).

**Fixed between the trips** (sections 1 to 8 of that file):

- Presses are synced to the SD card as they are written, and the log reports uptime and under-voltage at each start.
- A camera process that died no longer leaves the service unable to restart it.
- Clip rendering moved off the gun, to the laptop console.
- Tracking in trees: the gun shoots at the moving track nearest the bore instead of the one with the most hits.
  Speck-born tracks no longer steal the clay. A target missing for 2 frames is never shot.
- A clay too small to range is solved at an assumed 25 m instead of being refused. The shot goes at once when
  P(hit) reaches 0.9.
- A ready light on GPIO26 shows whether the gun is armed and live.
- No shot at a target first seen less than 200 ms ago, so the gun stops shooting scenery the swing passes over.
- `tbot find-break` finds the trigger's break point on the IMU. That afternoon the break had moved by one spline
  tooth, and the pre-travel was tripping the hammer.

**Afternoon.** Hunter's verdict: *somewhat successful*. The Pi kept 5 presses from the range. Most shots took the
power with them before their press was written. All 3 fire commands that were kept set the gun off, with
recoil on the IMU 62 to 70 ms after the command. It double-fired on the first one: press 160333 shows a second
full recoil 0.36 s after the first, with no second command. Near the end it died after every shot, and the
boots got slower.

| 2026-11-19 afternoon | |
|---|---|
| Presses kept by the Pi at the range | 5 (more were lost to power cuts) |
| Fire commands kept | 3, all of which set the gun off |
| Clays broken | **at least 5** of about 16 thrown (Hunter's count; the rest: no shot, Pi off, or a miss) |
| Engagement range | 14 to 24 m |
| Trigger delay, command to ignition | 62, 66 and 70 ms (44.5 ms assumed; recalibrated to 60.4 ms since) |
| Frame rate in front of trees | 30 to 33 fps |

![The gun camera tracking clays in front of the tree line and firing](docs/media/range_2026-11-19/gun_view_track_and_fire.gif)

*Press 160243, from the gun's camera. Yellow rings are tracks. The magenta X is where the clay will be when the
shot arrives, and the cyan circle is where the bore will point. The frame turns red at the fire command.*

Phone clips and gun recordings are paired automatically by `software/scripts/range_media.py sync`. It uses the
bang in the phone audio and the shot spacing inside each Pi boot, not the Pi clock, which has no battery
([RANGE_MEDIA.md](software/docs/RANGE_MEDIA.md)).

## The engineering

### Mechanical

![CAD render of the full assembly, rev AH, from gun-right and below](docs/media/cad_full.jpg)

| | |
|---|---|
| ![Front module, gun-left cover removed](docs/media/cad_open_L.jpg) | ![Front module, gun-right cover removed](docs/media/cad_open_R.jpg) |
| Gun-left: Raspberry Pi 3B, camera ribbon, camera in the nose | Gun-right: 3S pack, buck converter bay, power harness |

All parts are PLA, printed on an Ender 3. Nothing on the gun is modified: the front module hangs from two
worm-drive hose clamps seated in grooves, and the trigger set clamps to the guard.

- **Front module** (8 parts, 231 g): camera in the nose, Pi on one side, 3S 2600 mAh battery and two buck
  converters on the other, IMU in the clamp. M3 into brass heat-set inserts throughout.
- **Trigger set** (4 parts, rev AH): MG996R servo mounted beside the guard with its output inside the guard; a printed cam presses the trigger's lower hook; a plug-and-saddle mount around the bow and a right-side plate that houses the arming button.
![Trigger set rev AH: the MG996R in its printed mount beside the guard](docs/media/cad_trigger.jpg)

*Trigger set, rev AH (2026-11-18), from below on the servo side. The servo sits beside the guard and its output
shaft reaches inside it, where the printed cam presses the trigger's lower hook.*

- **Design revisions G through AH** in three and a half weeks, each recorded with its reason in
  [design/layout.md](design/layout.md). Every dimension the software uses comes from that file.
- A modelled wiring harness and camera ribbon in the assembly, and an M12 × 0.5 split clamp nut to lock lens
  focus against recoil.

CAD: `parts/` (native), `export/STEP/` and `export/STL/` (open in anything), [BOM](design/BOM.md),
[assembly sequence](design/assembly_sequence.md).

### Reference model by Gaussian splatting

The forend and trigger guard are compound curves that calipers cannot capture, so the gun itself was scanned
from phone video ([refs/scan/](refs/scan/)):

1. Walk-around video with printed ChArUco boards in frame.
2. Structure from motion for camera poses (407 frames, 1.23 px mean reprojection error).
3. A Gaussian splat trained on those poses (567,000 Gaussians).
4. Splat to mesh by fusing rendered depth into a TSDF at 0.54 mm voxels.
5. Absolute scale from the boards (0.64 mm RMS), then registration into the gun frame used by the CAD and the
   software.

![Scanned mesh (white) registered onto the CAD gun body (red)](docs/media/scan_vs_cad.png)

*Scan in white over the CAD gun body in red, side and plan views, millimetres. The clamps were modelled against
this surface.*

### Calibration

The solver is only as good as what it knows about its own geometry and timing, so each quantity is measured
with its own procedure and uncertainty ([CALIBRATION.md](software/docs/CALIBRATION.md)):

| Quantity | Value | Method |
|---|---|---|
| Focal length | 4018 px ± 0.3 % at 1280×800 | A sheet of US Letter paper photographed at six taped distances, 20 to 45 ft |
| Boresight (camera to bore) | pitch −1.89°, yaw −1.67° | Bore laser, rotated in the chamber to cancel its own wobble; verified to 0.067° at a second distance |
| Camera to IMU clock offset | −6.31 ms | Cross-correlating gyro rate with image motion over a 15 s swing |
| IMU axis map and gyro bias | fit within 0.8° | Three hand-held poses; bias re-zeroed whenever the gun rests for 1 s |
| Trigger delay | 161 ms | Microphone on the bench, then recoil on the IMU in live fire |

### Techniques

- **Detection without colour.** The sensor is monochrome, so the clay is found with a matched filter sized to
  it, local hysteresis, and speck and contrast tests that reject noise and scenery. The barrel is masked using
  the camera model.
- **Range from one camera.** A clay is 110 mm across; its apparent size, corrected for lens blur, gives range.
- **Attitude.** A complementary filter fuses gyro and accelerometer, and re-zeroes gyro bias whenever the gun
  rests for a second.
- **Tracking.** An extended Kalman filter holds 3-D position and velocity in a gravity-aligned world frame, so
  the shooter's swing does not look like target motion.
- **Fire decision.** The solver extrapolates the clay and the gun swing (including angular acceleration) across
  the full delay, applies a pellet drag, drop and pattern model for the load and choke, and converts miss
  distance and its uncertainty into a hit probability.

### Simulation and bench testing

- **Closed-loop simulator.** `tbot sim` flies synthetic skeet and trap targets with gravity and drag, renders
  them through the calibrated camera model with motion blur and noise, swings a simulated shooter, and judges
  each shot against ground truth.
- **Monitor in the loop.** The gun's camera watches a screen showing a moving target while the real servo fires
  against a microphone, timing the chain from photon to trigger break without ammunition.
- **Replay.** Every live press records raw frames, IMU and decisions, and replays through the same pipeline on
  a laptop.
- **Yaw table.** A stepper turntable for IMU scale and timing is designed and coded but has not yet been run on
  hardware.

The software lives in [software/](software/) and runs with no hardware attached; start with
[software/README.md](software/README.md) and [ARCHITECTURE.md](software/docs/ARCHITECTURE.md).

## Safety

The trigger can only move while a person holds the arming button.

1. **Dry run by default.** Without an explicit live flag the pipeline logs decisions and never touches the
   actuator.
2. **The interlock lives in the actuator layer.** `Actuator.fire()` refuses unless the button reads held. No
   caller can bypass it.
3. **Release means rest.** Letting go drives the servo back within one control step. The button is normally
   open to ground with a pull-up, so a cut wire or loose connector reads as not armed.
4. **One shot per press.**
5. **Watchdog.** No processed frame for 250 ms and the servo returns to rest.

These are software layers. There is no hardware cut in the servo supply; opening the battery connector is what
makes the actuator safe. All bench work is done with the gun unloaded.

## Build log

| | | |
|---|---|---|
| ![](docs/media/01_measuring.jpg) | ![](docs/media/03_first_print.jpg) | ![](docs/media/04_trigger_prototype.jpg) |
| Oct 25: measuring the gun | Oct 28: first printed trigger bracket | Nov 1: servo and arming button prototype |
| ![](docs/media/05_test_fit.jpg) | ![](docs/media/06_bench_bringup.jpg) | ![](docs/media/07_trigger_clamp.jpg) |
| Nov 1: front module test fit | Nov 1: camera and Pi bring-up | Nov 8: guard clamp, second generation |
| ![](docs/media/08_wiring.jpg) | ![](docs/media/10_boresight.jpg) | ![](docs/media/11_backyard_test.jpg) |
| Nov 14: wiring the front module | Nov 16: focal length and boresight in a hallway | Nov 17: first outdoor tracking test |

| Date | Milestone |
|---|---|
| Oct 25 | Gun measured, parts ordered |
| Oct 26 | First full CAD (rev G) and first software package |
| Nov 5 | Gaussian-splat scan of the gun; CAD rebuilt against it |
| Nov 12 | Full assembly on the gun |
| Nov 14 to 15 | Latency measured, IMU calibrated, 3-D tracker, simulator with doubles |
| Nov 16 | 12 mm lens fitted; focal length and laser boresight |
| Nov 17 | Backyard tracking on birds in the morning, live fire on clays by noon |
| Nov 18 | Trigger set rev AH: servo output inside the guard, finger cam on the trigger's lower hook |
| Nov 19 | Second range day: morning trip, fixes at home, afternoon trip with phone video synced to the gun |

![A hawk tracked by hand for 8 seconds, as the gun camera saw it](docs/media/hawk_sequence.jpg)

*The first moving target: a hawk over the yard, followed by hand for 8 seconds on the morning of the range day.*

## Repository

| Path | Contents |
|---|---|
| `design/` | BOM, layout (every dimension with its source), assembly sequence, harness report |
| `parts/`, `refs/`, `asm/` | Native CAD: printed parts, reference models of purchased parts, full assembly |
| `export/` | STEP and STL of every part and the assembly |
| `print/` | Slicer settings and G-code for an Ender 3 |
| `software/` | The `tbot` Python package, tests, calibration data, field data, Pi provisioning |
| `docs/media/` | Photos, CAD renders and range GIFs used in this file |

Units are millimetres. Gun frame: +X toward the muzzle, +Y gun-left, +Z up, origin at the trigger face.

## Try the software

No hardware needed:

```bash
cd software
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[laptop,test]'
python3 -m pytest
tbot sim
tbot run --source synthetic --sim-armed --stream --port 8080
```

Then open `http://localhost:8080` to watch the detector, tracker and solver work on a synthetic clay. To replay
a real press from the range:

```bash
tbot run --source replay:field_data/2026-11-17-range/presses/20261117-104806-press --record off --set gpio.backend=sim --sim-armed --set actuator.type=sim
```

The raw frames are stored with Git LFS.

## License

[MIT](LICENSE). Built by Hunter Edmonds.
