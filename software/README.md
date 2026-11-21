# TBOT software

Vision-assisted trigger release for clay shooting (skeet / trap, sporting use only). A monochrome global-shutter
camera under the barrel of a Weatherby Element finds the clay by what moves against the world (the scenery is
registered and cancelled frame to frame: [docs/VISION_MOTION.md](docs/VISION_MOTION.md)), a Kalman tracker in world
coordinates (IMU-referenced) predicts where it will be when the shot arrives, and a fire solver decides whether
pressing the trigger *now* is the best moment — then, and only then, drives the trigger actuator. The same package
runs on the gun (Raspberry Pi 3 Model B v1.2) and on a laptop, where every stage can be
exercised against synthetic scenes, a webcam or a recorded session. The binding module/API contract is
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); it wins over this file.

The build uses the TowerPro MG996R servo (`actuator.type = "servo"`). Its estimated command-to-break time is
~0.1 s versus ~0.7–1.1 s for the supported PQ12 fallback (`actuator.type = "pq12"`): at 20 m/s the clay travels
~2 m versus 14–22 m during that delay. Measure the installed servo's latency rather than relying on these estimates.
At 50 Hz, servo PWM frame quantisation is the expected dominant jitter source: a command can wait 0–20 ms for
the next frame, corresponding to 5.8 ms one-sigma for uniformly distributed phase.

## Safety model

The gun fires only when a human is holding the arming button. Five layers, all in force by default:

| # | Layer | Where |
|---|---|---|
| 1 | **Dry run by default** — `fire.live = false`: the pipeline logs "would fire" decisions and never touches the actuator. `tbot run --live` and `tbot hitl run --live` explicitly enable their live paths. | `config/default.toml` `[fire] live`, `tbot/app/run.py` |
| 2 | **The actuator layer enforces the interlock itself.** `Actuator.fire()` returns `False` unless the arming switch reads armed; no caller can bypass it, including the calibration tools. | `tbot/hw/actuator.py`, `tbot/hw/arming.py` |
| 3 | **Release = immediate rest.** Every actuator subscribes to `ArmingSwitch.on_change` and `abort()`s to the rest position within one control step when the button opens. The switch is normally-open to GND with a pull-up, so a cut lead, an unplugged connector or a dead pull-up reads *not armed*. | `tbot/hw/actuator.py` |
| 4 | **One shot per press** (`fire.max_shots_per_arm = 1`): the button must be released and pressed again before a second release is possible. | `tbot/hw/arming.py` `try_consume_shot()` |
| 5 | **Watchdog**: no processed frame for `fire.watchdog_ms` (250 ms) and the actuator is driven back to rest. | `tbot/app/run.py` |

The BCM17 arming button provides a **software interlock only**; there is no hardware cut in the servo supply.
The MG996R has its own 7.0 V XL4015 supply; do not power it from the Pi. Open the battery lever connector
to make the actuator safe. Software layers 1–5 are not a substitute for a hardware power cut; a series
hardware interlock is recommended but not installed (see [docs/PROTOTYPING.md](docs/PROTOTYPING.md)).

Bench rules that no amount of code can enforce: the gun is **unloaded** for every latency, actuator and assembly
test, and the muzzle points somewhere safe whenever power is on.

## Install

### Laptop (prototyping, tests, replay)

```bash
cd software
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[laptop,test]'     # numpy + opencv-python + pytest
tbot --help
```

### Raspberry Pi

`scripts/pi_setup.sh` installs the apt packages (OpenCV, numpy, Picamera2, lgpio, smbus2, i2c-tools, pytest),
enables the OV9281 overlay / I2C / hardware PWM / UART in `config.txt`, adds you to the `video`, `gpio` and `i2c`
groups, and creates `~/tbot-venv` (with `--system-site-packages`, so apt's OpenCV is used) with TBOT installed
editable:

```bash
cd ~/TBOT/software
sudo ./scripts/pi_setup.sh
sudo reboot            # the config.txt changes need it
~/tbot-venv/bin/tbot run --stream
```

`pip install opencv-python` on a Pi 3B is a trap (hours of build or a wheel that fights apt's numpy) — the apt
packages plus `--system-site-packages` is the supported path.

## TBOT Console (the app)

Plug the gun's Ethernet into the laptop, power it on and open **TBOT Console** (Desktop shortcut, made once with
`desktop/install_shortcut.ps1`). Live view, recorded presses with the shot marked (mark hits, score, render clips),
camera focus, boresight wizard, trigger servo setup, every setting, diagnostics, modes and power. No internet needed;
any browser also works at `http://169.254.20.20/`. [docs/CONSOLE.md](docs/CONSOLE.md).

Or just run **`launch_console.bat`** (Windows, double-click) / **`./launch_console.sh`** (Linux, macOS) in this
folder. Add `dev` (`launch_console.bat dev`) to open it on a stand-in gun on this computer, with no gun connected.

## Quick start on the laptop

No hardware at all — synthetic scene, simulated gun swing, simulated IMU and GPIO:

```bash
tbot sim                                              # closed-loop score over all scenarios
tbot run --source synthetic --sim-armed --stream --port 8080   # watch it work: http://localhost:8080
```

`--sim-armed` holds the *simulated* arming button down; it is refused by the real GPIO backend.

Webcam, pretending to be the mono sensor behind an orange long-pass filter (the red channel is close to what an
OV9281 sees through such a filter — see [docs/OPTICS.md](docs/OPTICS.md)):

```bash
tbot run --source cv:0 --set camera.emulate_mono=red --stream
```

Replay a session recorded on the gun (copy it off the Pi first, e.g.
`rsync -a pi@tbot.local:~/TBOT/software/sessions/20261026-141230-rec sessions/`):

```bash
tbot run --source replay:sessions/20261026-141230-rec --stream
tbot calib-detector --session sessions/20261026-141230-rec --gui    # tune on real frames, on a real screen
```

## Commands

`tbot <command> -h` prints the real flag list; common configuration flags apply to the commands that accept them
(for HITL, use `tbot hitl run -h`).

| Command | What it does |
|---|---|
| `tbot run` | Live pipeline: camera → detect → track → fire solver. Dry run unless `--live`. `--stream`/`--port` HTTP overlay, `--display` local window, `--record off\|armed\|continuous`, `--log-jsonl`, `--max-frames`, `--sim-armed`. |
| `tbot sim` | Closed-loop evaluation on the synthetic scenarios (`--scenarios`, `--seeds N`, `--not-armed`, `--out JSON`): detection rate, track latency, shots, predicted miss, hit rate. Runs on the gun's time (`sim.process_ms`) and repeats exactly. |
| `tbot vclay` | Virtual clays through recorded gun motion: the real swing and frame times of recorded presses, a synthetic target, the real tracker and solver, judged at the true trigger delay. Clays hit per firing rule (`--policies`), per frame rate (`--fps`), per delay error (`--delay-error-ms`); `--calibration` checks P(hit). |
| `tbot vbench` | The vision front end scored on recorded presses: `inject` composites virtual clays with exact truth into real footage and runs the whole pipeline (recall by size / background / polarity, scenery detections, false tracks, time to a track, shots and hits), `clean` runs the footage as recorded, `compare A.json B.json` puts two runs side by side. Pin the engine: `--set vision.engine=motion\|legacy`. [docs/VISION_MOTION.md](docs/VISION_MOTION.md) |
| `tbot rescore` | Recorded presses through today's software (real frames, real IMU, the button as it was held), scored like `tbot score`. The regression test for the decision chain. |
| `tbot score` | Would each shot of a recorded press have hit? Bore at the exit time against the target at the arrival time, from the frames after the shot; `--every` scores every frame with a solution, `--image` draws the arrival frame. |
| `tbot doctor` | Offline configuration, calibration and layout checks; fix FAILs before assembly. |
| `tbot sensitivity` | MG996R error budget, isolated tolerance table and measurement-priority ranking. |
| `tbot explain` | Per-frame decision funnel, reasons, sigma budget and biggest blocker for a synthetic or recorded source. |
| `tbot tune` | Synthetic parameter search and Pareto front; optionally write winning overrides with `--write-best`. Validate on sessions and HITL. |
| `tbot record` | Record a session (frames + timestamps + IMU + events) from any source for replay (`--seconds`, `--out DIR`, `--decimate N`). |
| `tbot board` | Write a printable chessboard PNG (`--cols --rows --square-mm --dpi --out`). |
| `tbot focus` | Lens focus + exposure aid, headless: live sharpness (centre + quadrants, barrel masked), % of peak, saturation, mean/median DN, fps, exposure/gain; `--roi`, `--seconds`, `--stream`, `--json`. |
| `tbot calib-intrinsics` | Fisheye/pinhole intrinsic calibration from live views, `--from-dir` or `--session` → `calib/intrinsics.json` (+ FOV report). |
| `tbot calib-boresight` | Camera-to-bore alignment from a known aim point (`--range-m --pixel u,v` or `--gui`), plus patterning offset `--poi-offset-cm` → `calib/boresight.json`. |
| `tbot calib-imu` | `bias` (static gyro/accel bias), `axes` (guided 3-pose check of `R_gun_imu`), `sync` (camera–IMU time offset from a recorded swing) → `calib/imu.json`. |
| `tbot calib-detector` | Detector tuning: GUI trackbar tuner, `--report` headless stats, `--auto-mask SESSION` → `calib/mask.png`, `--psf --range-m` → blur size → `calib/detector.json`. |
| `tbot latency-camera` | LED test: exposure → userspace latency and the camera timestamp offset → `calib/latency.json`. |
| `tbot latency-actuator` | Fire command → trigger break latency, arming button held by the operator, break-sense microswitch → `calib/latency.json`. |
| `tbot latency-report` | Print the whole latency budget (config default vs measured, per actuator) and the resulting lead at a reference range. |
| `tbot bench` | Per-stage processing times and frame-period verdict → `calib/bench.json`; `--profile`/`--profile-out`, `--no-overlay`, and `--compare FILE` support profiling and cross-machine comparison. |
| `tbot hitl run` | Monitor-and-camera end-to-end bench: geometry, display timing, flash, moving-target sweep and optional detection tests; waits for fullscreen, dry by default, `--live` requires the held arming button and mic or break-sense. `--sim` self-tests without hardware. Writes text, JSON and self-contained HTML reports. |
| `tbot hitl analyze` | Re-analyse a saved HITL run and regenerate text, JSON and HTML reports. |
| `tbot hitl mics` | List ALSA capture devices and optionally check microphone level and capture-clock statistics. |
| `tbot hwcheck` | Check every fitted device on the Pi (boot overlays, camera fps, IMU WHO_AM_I/rate, arming pin, PWM/H-bridge, under-voltage/throttle, temperature, calibration status); `--interactive` tests the arming button, `--json`. Never moves the trigger. |
| `tbot console` | The TBOT Console web app and API (port 80 on the gun, `tbot-console.service`); `--dev` runs a stand-in gun on a laptop. [docs/CONSOLE.md](docs/CONSOLE.md). |
| `tbot calrig` | CalRig B motorised yaw table (UNLOADED gun): `info`, `home`, `jog`, `run hold\|turns\|sine\|rate\|home_check --pose ...`, `estop`, `analyze [--apply]`; `--sim` runs the whole stack with no hardware. See [docs/CALRIG.md](docs/CALRIG.md). |
| `tbot jog` | Bench setup of the trigger actuator with the gun unloaded: find and save MG996R `rest_us`/`fire_us`, check PQ12 rod direction (`swap`); moves only while the arming button is held; saves to `config/local.toml`. |

## Repo layout

```
software/
  config/default.toml   every key, with units and the source of each number ([INFERENCE] = not yet measured)
  tbot/
    config.py types.py optics.py stats.py cli.py
    camera/    picam (Picamera2) | cv_source (webcam/video) | replay | synthetic + base/open_source
    vision/    motion.py (registered frame differencing: what moves against the world) + tracker_motion.py (tracks
               born from a path of detections): the default front end, docs/VISION_MOTION.md. The front end of
               2026-11-19 (vision.engine = "legacy"): detector.py (contrast blob detector), clutter.py (IMU scenery
               filter), tracker.py / tracker3d.py (per-track Kalman filters in the world frame; tracker3d's EKF is
               the one both engines use)
    bench/     tbot vbench: clay injection into recorded presses, metrics, the offline truth oracle
    fire/      ballistics.py (pellet drag, ToF, drop, pattern), solver.py (lead, P(hit), fire/no-fire)
    imu/       mpu6050.py (I2C FIFO reader), attitude.py (complementary filter → R_wg, omega)
    hw/        gpio.py (lgpio|sim), arming.py (debounced interlock), actuator.py (servo|pq12|sim|none)
    calrig/    CalRig B yaw table: serial driver, FakeRig simulator, recorder, procedures, analysis (docs/CALRIG.md)
    hitl/      monitor target page, timing, rig procedures, analysis, reports and simulated bench
    app/       run.py (pipeline + CLI), record.py (sessions), stream.py (MJPEG debug server), overlay.py
    tools/     calibration, latency, bench, doctor, sensitivity, tune, explain and sim-eval commands
  tests/                pytest suite (runs with no hardware)
  calib/                calibration outputs: intrinsics/boresight/imu/latency/detector .json, mask.png
  sessions/             recordings (created on demand)
  scripts/pi_setup.sh   Raspberry Pi provisioning
  scripts/deploy.sh     laptop <-> Pi over Ethernet: push code, pull sessions/calib (docs/PROTOTYPING.md §1)
  pi-pulls/             data pulled off the Pi by deploy.sh pull (git-ignored)
  firmware/calrig_uno/  CalRig B Arduino Uno firmware (+ wiring README)
  runs/                 CalRig run directories (git-ignored, created on demand)
  docs/                 PREFLIGHT.md, ARCHITECTURE.md (contract), OPTICS.md, CALIBRATION.md, PROTOTYPING.md, HITL.md
```

## Tests

```bash
cd software
python3 -m pytest              # laptop (venv) or Pi (apt python3-pytest); no hardware needed
python3 -m pytest tests/test_solver.py -q
```

The suite is hardware-free: GPIO, PWM, camera and IMU all have simulated backends. After changing anything in
the detect → track → fire chain, run the three benches on top of it; each answers a different question and
all three repeat exactly:

```bash
tbot sim --seeds 6                                            # invented shooter, rendered frames: the whole pipeline
tbot rescore field_data/2026-11-17-range/presses/*            # the real frames of the range day through today's code
tbot vclay field_data/2026-11-17-range/presses/* --policies   # hundreds of virtual clays through the real swings
```

Reference results (2026-11-17, trigger delay 164 ms): `sim` 17 of 54 clays, `rescore` 6 shots / 3 planned hits,
`vclay` 63 % of 1500 clays. What they mean and what each lever is worth:
[docs/INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md), last section.

## Documentation

- [docs/CONSOLE.md](docs/CONSOLE.md) — the TBOT Console app: pages, safety, how it is built, working on it without the gun.
- [docs/INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md) — bench integration handoff: per-rig values on the Pi, calibration state, open items.
- [docs/PREFLIGHT.md](docs/PREFLIGHT.md) — ordered offline optimisation, diagnostics, profiling and bench workflow before assembly.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module/API contract, frames, clock, calibration file formats, safety model.
- [docs/OPTICS.md](docs/OPTICS.md) — the mono camera and lens, what "orange" means without colour, range vs clay size in pixels, filters, exposure and focus, camera pose.
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — the ordered calibration procedure, command by command, with pass criteria.
- [docs/CALRIG.md](docs/CALRIG.md) — CalRig B yaw table operator guide (bench safety, calibration session, applying results); contract in [docs/CALRIG_PROTOCOL.md](docs/CALRIG_PROTOCOL.md).
- [docs/HITL.md](docs/HITL.md) — monitor-and-camera bench setup, end-to-end measurements, interpretation and IMU verification proposals.
- [docs/PROTOTYPING.md](docs/PROTOTYPING.md) — Pi ↔ laptop (Ethernet / serial / HDMI), GPIO wiring table, hardware the software needs that is not yet on the BOM.
- [docs/VISION_HANDOFF.md](docs/VISION_HANDOFF.md) — boundary for the vision developer: what the hardware side guarantees, plug-in points, measured open problems.
- `../design/layout.md`, `../design/BOM.md`, `../design/assembly_sequence.md` — the mechanical side: every dimension the config quotes comes from there.

Per-rig settings (pins, servo pulses, anything in `config/default.toml`) go in `config/local.toml`, applied after the
defaults and before `--config` files; the `save` command inside `tbot jog servo` writes it for you.
