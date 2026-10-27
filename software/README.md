# TBOT software

Vision-assisted trigger release for clay shooting (skeet / trap, sporting use only). A monochrome global-shutter
camera under the barrel of a Weatherby Element finds the clay against the sky, a Kalman tracker in world
coordinates (IMU-referenced) predicts where it will be when the shot arrives, and a fire solver decides whether
pressing the trigger *now* is the best moment — then, and only then, drives the trigger actuator. The same package
runs on the gun (Raspberry Pi 3 Model B v1.2 now, Pi 5 + AI HAT+ later) and on a laptop, where every stage can be
exercised against synthetic scenes, a webcam or a recorded session. The binding module/API contract is
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); it wins over this file.

## Safety model

The gun fires only when a human is holding the arming button. Five layers, all in force by default:

| # | Layer | Where |
|---|---|---|
| 1 | **Dry run by default** — `fire.live = false`: the pipeline logs "would fire" decisions and never touches the actuator. `tbot run --live` is the only way to arm the software path. | `config/default.toml` `[fire] live`, `tbot/app/run.py` |
| 2 | **The actuator layer enforces the interlock itself.** `Actuator.fire()` returns `False` unless the arming switch reads armed; no caller can bypass it, including the calibration tools. | `tbot/hw/actuator.py`, `tbot/hw/arming.py` |
| 3 | **Release = immediate rest.** Every actuator subscribes to `ArmingSwitch.on_change` and `abort()`s to the rest position within one control step when the button opens. The switch is normally-open to GND with a pull-up, so a cut lead, an unplugged connector or a dead pull-up reads *not armed*. | `tbot/hw/actuator.py` |
| 4 | **One shot per press** (`fire.max_shots_per_arm = 1`): the button must be released and pressed again before a second release is possible. | `tbot/hw/arming.py` `try_consume_shot()` |
| 5 | **Watchdog**: no processed frame for `fire.watchdog_ms` (250 ms) and the actuator is driven back to rest. | `tbot/app/run.py` |

**Wire the arming switch in series with the actuator driver as well** — the H-bridge `STBY`/enable pin for the PQ12,
or the servo's 5–6 V supply for the MG996R (see [docs/PROTOTYPING.md](docs/PROTOTYPING.md)). Then a software fault,
a hung process or a stuck GPIO cannot move the trigger while the button is open. Software layers 1–5 are not a
substitute for that one wire.

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
cd ~/tbot/software
sudo ./scripts/pi_setup.sh
sudo reboot            # the config.txt changes need it
~/tbot-venv/bin/tbot run --stream
```

`pip install opencv-python` on a Pi 3B is a trap (hours of build or a wheel that fights apt's numpy) — the apt
packages plus `--system-site-packages` is the supported path.

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
`rsync -a pi@tbot.local:~/tbot/software/sessions/20261026-141230-rec sessions/`):

```bash
tbot run --source replay:sessions/20261026-141230-rec --stream
tbot calib-detector --session sessions/20261026-141230-rec --gui    # tune on real frames, on a real screen
```

## Commands

`tbot <command> -h` prints the real flag list; `--config FILE.toml`, `--set key=value`, `--source SPEC` and
`--calib-dir DIR` work on every command except `board`.

| Command | What it does |
|---|---|
| `tbot run` | Live pipeline: camera → detect → track → fire solver. Dry run unless `--live`. `--stream`/`--port` HTTP overlay, `--display` local window, `--record off\|armed\|continuous`, `--log-jsonl`, `--max-frames`, `--sim-armed`. |
| `tbot sim` | Closed-loop evaluation on the synthetic scenarios (`--scenarios`, `--seeds N`, `--not-armed`, `--out JSON`): detection rate, track latency, shots, predicted miss, hit rate. |
| `tbot record` | Record a session (frames + timestamps + IMU + events) from any source for replay (`--seconds`, `--out DIR`, `--decimate N`). |
| `tbot board` | Write a printable chessboard PNG (`--cols --rows --square-mm --dpi --out`). |
| `tbot calib-intrinsics` | Fisheye/pinhole intrinsic calibration from live views, `--from-dir` or `--session` → `calib/intrinsics.json` (+ FOV report). |
| `tbot calib-boresight` | Camera-to-bore alignment from a known aim point (`--range-m --pixel u,v` or `--gui`), plus patterning offset `--poi-offset-cm` → `calib/boresight.json`. |
| `tbot calib-imu` | `bias` (static gyro/accel bias), `axes` (guided 3-pose check of `R_gun_imu`), `sync` (camera–IMU time offset from a recorded swing) → `calib/imu.json`. |
| `tbot calib-detector` | Detector tuning: GUI trackbar tuner, `--report` headless stats, `--auto-mask SESSION` → `calib/mask.png`, `--psf --range-m` → blur size → `calib/detector.json`. |
| `tbot latency-camera` | LED test: exposure → userspace latency and the camera timestamp offset → `calib/latency.json`. |
| `tbot latency-actuator` | Fire command → trigger break latency, arming button held by the operator, break-sense microswitch → `calib/latency.json`. |
| `tbot latency-report` | Print the whole latency budget (config default vs measured, per actuator) and the resulting lead at a reference range. |
| `tbot bench` | Per-stage processing time at the configured resolution, percentiles, verdict against the frame period → `calib/bench.json`. |
| `tbot hwcheck` | Check every fitted device on the Pi (boot overlays, camera fps, IMU WHO_AM_I/rate, arming pin, PWM/H-bridge, under-voltage/throttle, temperature, calibration status); `--interactive` tests the arming button, `--json`. Never moves the trigger. |
| `tbot jog` | Bench setup of the trigger actuator with the gun unloaded: find and save MG996R `rest_us`/`fire_us`, check PQ12 rod direction (`swap`); moves only while the arming button is held; saves to `config/local.toml`. |

## Repo layout

```
software/
  config/default.toml   every key, with units and the source of each number ([INFERENCE] = not yet measured)
  tbot/
    config.py types.py optics.py stats.py cli.py
    camera/    picam (Picamera2) | cv_source (webcam/video) | replay | synthetic + base/open_source
    vision/    detector.py (contrast/HSV blob detector), tracker.py (per-track CA Kalman in world az/el)
    fire/      ballistics.py (pellet drag, ToF, drop, pattern), solver.py (lead, P(hit), fire/no-fire)
    imu/       mpu6050.py (I2C FIFO reader), attitude.py (complementary filter → R_wg, omega)
    hw/        gpio.py (lgpio|sim), arming.py (debounced interlock), actuator.py (servo|pq12|sim|none)
    app/       run.py (pipeline + CLI), record.py (sessions), stream.py (MJPEG debug server), overlay.py
    tools/     the calibration, latency, bench and sim-eval commands
  tests/                pytest suite (runs with no hardware)
  calib/                calibration outputs: intrinsics/boresight/imu/latency/detector .json, mask.png
  sessions/             recordings (created on demand)
  scripts/pi_setup.sh   Raspberry Pi provisioning
  docs/                 ARCHITECTURE.md (contract), OPTICS.md, CALIBRATION.md, PROTOTYPING.md
```

## Tests

```bash
cd software
python3 -m pytest              # laptop (venv) or Pi (apt python3-pytest); no hardware needed
python3 -m pytest tests/test_solver.py -q
```

The suite is hardware-free: GPIO, PWM, camera and IMU all have simulated backends. `tbot sim` is the end-to-end
regression on top of it — run both after changing anything in the detect → track → fire chain.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module/API contract, frames, clock, calibration file formats, safety model.
- [docs/OPTICS.md](docs/OPTICS.md) — the mono camera and lens, what "orange" means without colour, range vs clay size in pixels, filters, exposure and focus, camera pose.
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — the ordered calibration procedure, command by command, with pass criteria.
- [docs/PROTOTYPING.md](docs/PROTOTYPING.md) — Pi ↔ laptop (Ethernet / serial / HDMI), GPIO wiring table, hardware the software needs that is not yet on the BOM.
- [docs/VISION_HANDOFF.md](docs/VISION_HANDOFF.md) — boundary for the vision developer: what the hardware side guarantees, plug-in points, measured open problems.
- `../hardware/design/layout.md`, `../hardware/design/BOM.md`, `../hardware/design/assembly_sequence.md` — the mechanical side: every dimension the config quotes comes from there.

Per-rig settings (pins, servo pulses, anything in `config/default.toml`) go in `config/local.toml`, applied after the
defaults and before `--config` files; `tbot jog save` writes it for you.
