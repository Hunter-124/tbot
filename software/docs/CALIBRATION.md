# Calibration procedure

Do these in order: each step consumes the output of the previous one. Everything lands in `calib/`
(`paths.calib_dir`, override with `--calib-dir`); every file is plain JSON you can read and edit. Nothing here
fires the gun — the **gun stays unloaded** for all of it, including the actuator latency run in step 6.

| # | Step | Writes | Needs the gun? |
|---|---|---|---|
| 0 | Hardware bring-up (`tbot hwcheck`, `tbot jog servo`) | `config/local.toml` (servo pulses) | trigger set fitted (unloaded) |
| 1 | Focus + exposure check | — | camera only |
| 2 | Intrinsics | `calib/intrinsics.json` | camera only |
| 3 | Detector + mask | `calib/detector.json`, `calib/mask.png` | pod on the gun |
| 4 | IMU | `calib/imu.json` | pod on the gun |
| 5 | Boresight | `calib/boresight.json` | pod on the gun |
| 6 | Latency | `calib/latency.json`, `calib/bench.json` | trigger set fitted (unloaded) |
| 6e | End-to-end HITL bench | `calib/hitl.json`, run reports; optionally `calib/latency.json` | trigger set fitted for live test (unloaded) |
| 7 | Dry run on the range | session recordings | everything |

Re-run step 2 if the lens is touched, step 3 if the camera or pod moves, step 5 if anything in the camera/barrel
chain is disturbed, step 6 if the actuator or the frame rate changes.

---

## 0. Hardware bring-up — `tbot hwcheck`, `tbot jog`

**Purpose.** Prove every device answers before calibrating anything, and set the MG996R servo's end positions.
The servo uses its own 7.0 V XL4015 supply and a common ground with the Pi. The BCM17 interlock is software
only; no hardware supply cut is installed. Open the battery lever connector to make the actuator safe.
Before connecting hardware, run `tbot doctor` for offline configuration, layout and calibration checks; fix
FAILs and review WARNs. See [PREFLIGHT.md](PREFLIGHT.md) for the complete pre-assembly workflow.

```bash
tbot hwcheck                       # boot overlays, camera fps, IMU, GPIO, PWM/H-bridge, power, temperature, calib status
tbot hwcheck --interactive         # also asks you to press and release the arming button
tbot jog servo                     # MG996R: <us>, +/-, rest, fire, test, save, q  (moves only while the button is held)
tbot jog pq12                      # fallback only: e <ms>, r <ms>, home, test, swap, save, q
```

**Pass.** `hwcheck` shows no FAIL (camera at >= 90 % of `camera.fps`, IMU at >= 90 % of `imu.sample_rate_hz`,
no under-voltage now). In `jog`, the lever/rod reaches the trigger break at `fire` and clears it at `rest`, the
button release returns it to rest instantly, and `save` wrote `config/local.toml`. Gun unloaded throughout.

---

## 1. Focus and exposure

**Purpose.** A soft or saturated image invalidates everything downstream: blurred corners wreck the intrinsics,
inflated blur wrecks size-based ranging.

**Setup.** Camera on `pod_cap`, Pi booted, anything with detail at > 5 m in the field of view (a treeline is ideal).
Daylight.

```bash
tbot focus                            # headless over SSH: live sharpness, % of peak, saturation, fps, exposure/gain
tbot focus --stream                   # same, with the score overlaid on http://<pi>:8080 from the laptop
tbot run --stream                     # the full pipeline view
tbot record --seconds 10 --out sessions/focus-check   # keep a frame set to pixel-peep later
```

**Do.** Loosen the M12 lens ring, focus at infinity (the distant detail, not the barrel — the barrel is supposed to
be soft, and `tbot focus` excludes the barrel mask), turn the ring slowly through the peak until `sharp` reads
~100 % of `peak` again, and lock the ring. All four quadrant scores should rise and fall together; one corner
peaking at a different ring position means a tilted lens or board. `tbot focus` also prints the saturated-pixel
percentage, mean/median DN, frame rate and the `exposure_us`/gain from the frame metadata; `--roi 0.3` narrows
the central window, `--json` prints a final summary. The score is relative (mean squared gradient, DN²): compare it
only within one session and scene. `exposure_us` should be the configured 500 µs and the image
should not be clipping. In open daylight with no filter, raise `camera.analogue_gain` before `exposure_us`
(exposure buys motion blur, see docs/OPTICS.md §6):

```bash
tbot run --stream --set camera.analogue_gain=2.0
tbot run --stream --set camera.auto_exposure=true      # diagnostic only — never run the pipeline on AE
```

**Pass criteria.**
- Distant edges are crisp; the barrel at the top of the frame is the only soft thing.
- No large saturated (255 DN) region in the sky.
- Frame rate at the status line is within ~10 % of `camera.fps`.
- Record the gain/exposure you settled on into your own override TOML (`--config mine.toml`) — do not hand-edit
  `config/default.toml`.

---

## 2. Intrinsics — `tbot board`, `tbot calib-intrinsics`

**Purpose.** Replace the nominal prior (`fx = 466.7 px` at 640 × 400, equidistant, D = 0) with the real focal
length and distortion. Every pixel → ray conversion, the barrel mask, size-based ranging and the boresight depend
on it.

**Setup.** Print a chessboard, glued **flat** to rigid board (foam core, MDF — paper on a wall ripples and biases
the fit). Even, indirect light; no glare. Hold the board at 0.5–2 m and cover the whole frame, including the
corners, at several tilts.

```bash
tbot board --cols 9 --rows 6 --square-mm 25 --dpi 300        # -> calib/board_9x6_25mm.png, print at 100 % scale
# measure a printed square with calipers; use the measured size below

tbot calib-intrinsics --views 25 --square-mm 25.0            # live, auto-accepts diverse sharp views
tbot calib-intrinsics --views 25 --gui                       # same, with the live coverage window (needs a display)
tbot calib-intrinsics --session sessions/board-sweep --every 5   # offline, from a recording
tbot calib-intrinsics --from-dir calib/views                 # offline, from saved stills
```

Useful flags: `--model fisheye|pinhole` (default `optics.model` = fisheye), `--cols/--rows`, `--sharp-min`
(Laplacian-variance floor), `--min-pose-dist` (how different a new view must be), `--timeout-s`, `--out`,
`--no-save-views`, `--force`.

**Writes.** `calib/intrinsics.json`: `model`, `width`, `height`, `K`, `D`, `rms_px`, `n_views`, `per_view_rms`,
`used_views`, `rejected_views`, `fov_deg`, `pattern`, `square_mm`, `created`, `notes`. Accepted views are saved
next to it unless `--no-save-views`.

**Pass criteria.**
- `rms_px` **< ~0.5 px**. The tool refuses to write above `MAX_RMS_PX = 1.0` or below `MIN_VIEWS_OK = 8` views
  unless you pass `--force` (and it stamps "WRITTEN WITH --force" into `notes` when you do). Anything over 0.5 px
  means soft views, a wrong `--square-mm`, or a board that is not flat.
- The printed FOV line is sane: H 70–80° at this sensor, i.e. roughly 60–70 % of the vendor's 118° claim
  (docs/OPTICS.md §3). A huge value means the board was too close or the model is wrong.
- The measured `fx` is within ~10 % of the nominal prior the tool prints next to it.
- Coverage: the per-view RMS list has no outlier left (the tool drops views above 3 × the median automatically),
  and the coverage map is filled near the edges — distortion is only constrained where you put the board.

---

## 3. Detector and mask — `tbot record`, `tbot calib-detector`

**Purpose.** Set the contrast threshold, polarity and size limits to the real sky, and remove the barrel and
everything else that never moves from the search area. This is the step that decides whether a clay is seen at all.

**Setup.** Pod on the gun, intrinsics done. Go to the range (or anywhere with sky and a thrower). Shoulder the gun
and swing through the arcs you actually shoot; have someone throw clays.

### 3a. Record a swing session

```bash
tbot record --seconds 60 --out sessions/swing-01
```

Record *while swinging*: the static-pixel mask needs the background to move. Copy the session to the laptop for
tuning (`rsync -a pi@tbot.local:~/TBOT/software/sessions/swing-01 sessions/`).

### 3b. Barrel / static mask

```bash
tbot calib-detector --auto-mask sessions/swing-01
tbot calib-detector --auto-mask sessions/swing-01 --static-std 3.0 --min-area 200 --dilate 4
```

Pixels whose temporal standard deviation stays below `--static-std` DN over the recording, in regions of at least
`--min-area` px touching the border, are masked and grown by `--dilate` px.

**Writes.** `calib/mask.png` (uint8, 255 = usable sky). It is ANDed with the geometric `barrel_mask()` at runtime.

**Pass criteria.** The barrel, the vent rib and any cable in frame are black; the sky is white; the mask does not
eat into the sky where clays fly. If it masks half the frame, the recording had too little swing.

### 3c. Tune the thresholds

```bash
tbot calib-detector --session sessions/swing-01 --gui        # trackbar tuner (needs a display: laptop or Pi+HDMI)
tbot calib-detector --session sessions/swing-01 --report --frames 0   # headless stats over the whole session
tbot calib-detector --report --frames 500                    # headless, live source on the Pi
tbot calib-detector --session sessions/swing-01 --report --no-mask    # how much the mask is doing for you
```

The tuner shows the contrast image, the binary image and the detections, and writes the fields you change.
`--frames 0` means "all frames / until the stream ends".

**Writes.** `calib/detector.json` — only the tuned `DetectorConfig` fields, applied over `config/default.toml`
at load time.

**Pass criteria.**
- The clay is detected on **most frames of its visible arc**, and the detection count per frame on clay-free
  frames is ~0 (the `--report` output gives both). A couple of false blobs per second is survivable — the tracker
  gates them out — but a stream of them wastes the Pi's budget and can steal a track.
- `polarity` matches reality (`dark` for an unfiltered mono sensor against a bright sky, `bright` behind an orange
  long-pass filter, `both` if you are unsure).
- `k_sigma`/`min_contrast` leave margin: lower `k_sigma` until false positives appear, then back off one step.
- Per-frame detector time from `--report` leaves room in the frame period (cross-check with `tbot bench`, step 6).

### 3d. Point-spread function (size-based ranging)

**Purpose.** `psf_px` is subtracted from every measured diameter before a range is derived; the default 1.2 is an
inference, and at 30–40 m a 0.2 px error there is a ~15 % range error (docs/OPTICS.md §4).

**Setup.** Put a clay (or any disc of accurately known diameter) at a **measured** range — 20–30 m, tape-measure
or laser rangefinder, against the sky, in focus, stationary.

```bash
tbot calib-detector --psf --range-m 25 --diameter-mm 108 --frames 200
```

**Writes.** `psf_px` into `calib/detector.json` (other tuned fields are preserved).

**Pass criteria.** It writes a value in roughly 1.0–2.5 px. The tool errors out if the observed diameter is not
larger than the geometric one — that means the range or the diameter you gave it is wrong, or the target was not
actually resolved. Re-run at a second range and check the two agree within ~0.2 px.

---

## 4. IMU — `tbot calib-imu`

**Purpose.** The IMU gives the world attitude `R_wg(t)` and the swing rate `ω` used to extrapolate the bore over
the actuator delay. Bias → drift; wrong axes → the track flies the wrong way; wrong time offset → a lead error
proportional to the swing rate.

**Setup.** Pod on the gun, GY-521 wired to I2C bus 1 (BCM 2/3), `sudo i2cdetect -y 1` shows `68`.

### 4a. Bias

```bash
tbot calib-imu bias --seconds 10            # gun completely still, resting on something solid
tbot calib-imu bias --seconds 10 --no-write # measure only
```

**Writes.** `gyro_bias_rad_s`, `accel_bias_m_s2`, `accel_scale`, `temp_c` in `calib/imu.json`.

**Pass criteria.** Gyro bias of a few deg/s at most and *stable* between two consecutive runs (re-run it; if the
second differs by more than ~0.5 deg/s, the gun was not still or the chip had not warmed up). Accel magnitude
within ~2 % of 9.81 m/s².

### 4b. Axis map

```bash
tbot calib-imu axes --seconds 2             # check only, prints the fitted matrix against R_gun_imu
tbot calib-imu axes --seconds 2 --write     # store it — only once the fit matches the CAD map
```

Follow the prompts (muzzle up / roll left / swing left). The CAD axis map from layout.md is
`R_gun_imu = [[0,0,-1],[0,-1,0],[-1,0,0]]`.

**Pass criteria.** The fitted matrix is a signed permutation (entries ≈ 0 or ±1) and **equals** the configured
map. If it does, you do not need `--write` at all. If it differs by a sign or a swapped axis, the board is mounted
differently than the CAD says — fix the mount or `--write` the fit, then note it in `design/layout.md`. Never
`--write` a matrix with intermediate values (0.3, 0.7 …): that is a bad fit, not a rotated board.

### 4c. Camera–IMU time offset

```bash
tbot record --seconds 30 --out sessions/sync-01     # swing the gun steadily left-right, clays optional
tbot calib-imu sync --session sessions/sync-01
tbot calib-imu sync --session sessions/sync-01 --step 2 --max-lag-ms 150 --lag-step-ms 1 --min-score 0.5
```

Image rotation rate (optical flow through the camera model) is cross-correlated against the gyro rate.
`--no-mask` disables the barrel/static mask for the flow; `--no-write` prints only.

**Writes.** `time_offset_ns` in `calib/imu.json` (added to IMU timestamps).

**Pass criteria.** Correlation score ≥ `--min-score` (0.5; the tool refuses to write below it) and a clearly
peaked lag curve. Expect a few ms, tens at worst. A flat curve means the swing was too slow or too short — swing
harder, longer, with texture in the frame.

---

## 5. Boresight — `tbot calib-boresight`

**Purpose.** Where the bore points in the image. This is the single calibration that decides whether the shot goes
where the solver thinks it goes. It solves the *rotation* error of the camera mount (`cam_rpy_deg`); the
translation (camera 32 mm below the bore, 171 mm behind the muzzle) comes from the CAD and is handled as parallax.

**Setup.** Gun **unloaded**, clamped or sandbagged so it cannot move. Either:
- a bore laser in the chamber/muzzle projecting onto a target at a measured range, or
- the gun aimed at a small, unambiguous point of aim at a measured range (10–50 m; longer is better, the
  correction scales with range).

Capture a frame and read off the pixel the bore target lands on (the stream snapshot
`http://<pi>:8080/snapshot.jpg` is the easy way), or click it:

```bash
tbot calib-boresight --range-m 25 --pixel 318.5,214.0
tbot calib-boresight --range-m 25 --gui                       # click the point in a live frame (needs a display)
tbot calib-boresight --range-m 25 --gui --image snap.png      # click it on a saved still
```

Roll (rotation about the bore) is not observable from one point. Add a plumb line — hang a weighted string behind
the target — and give a second pixel on it:

```bash
tbot calib-boresight --range-m 25 --pixel 318.5,214.0 --pixel2 318.0,120.0
tbot calib-boresight --range-m 25 --gui --plumb               # click both
tbot calib-boresight --range-m 25 --pixel 318.5,214.0 --roll-deg 0.0   # or fix it explicitly
```

**Writes.** `calib/boresight.json`: `rpy_deg`, `range_m`, `bore_pixel`, `width`, `height`, `created`.

**Pass criteria.**
- The correction is small: a few degrees. The tool refuses anything over 15° without `--force` — a large value
  means the pixel, the range or the mount numbers are wrong, not that the camera is crooked.
- Repeat at a different range and the solved `rpy_deg` should agree within a few tenths of a degree.
- Sanity check in the overlay: `tbot run --stream` draws the bore marker; aimed at the same target it must sit on
  it.

### 5b. Patterning correction (after shooting paper)

Shoot a patterning board at a known range with the load you use, measure the pattern **centre** relative to the
point of aim, and feed it back as the mean shot direction:

```bash
tbot calib-boresight --poi-offset-cm 0,5@22.9      # pattern centre 5 cm HIGH at 22.9 m (+dy gun-left, +dz up)
```

**Writes.** `poi_offset_rad` (`[yaw, pitch]`) into the same file; `MountModel.bore_dir_g` uses it. This can be
combined with the pixel solve in one call or run on its own.

**Pass criteria.** The offset matches what you measured on paper, in radians: `atan(0.05 / 22.9) = 2.2 mrad`
for the example. A shotgun pattern centre is a statistic — use several patterns, not one.

---

## 6. Latency — `tbot latency-camera`, `tbot latency-actuator`, `tbot bench`, `tbot latency-report`

**Purpose.** The solver fires at `now + fire_delay`, so the whole photon → shot chain has to be measured, not
guessed. The defaults in `[latency]` are marked `[INFERENCE]` for a reason.
Run `tbot sensitivity` to compare latency, range and PWM jitter against the pattern and inspect the isolated
tolerance table and measurement ranking. Its assumptions are synthetic, not a substitute for these measurements.

### 6a. Camera: exposure → userspace, and the timestamp offset

**Setup.** An LED from **BCM 27** (`gpio.latency_led_pin`, header pin 13) through a **330 Ω** resistor to GND,
positioned in the camera's field of view, bright enough to show clearly but not to saturate half the frame.

```bash
tbot latency-camera --trials 100
tbot latency-camera --trials 100 --set camera.exposure_us=4000   # for the timestamp-offset part (see below)
tbot latency-camera --sim                                        # self-test, no hardware
```

Options: `--roi X,Y,W,H` (skip the LED search), `--pin`, `--toggles`, `--min-wait-ms/--max-wait-ms`, `--seed`,
`--no-write`, `--json`.

The timestamp offset is recovered from *partially lit* frames — frames whose exposure started before the LED came
on. At 500 µs out of an 8.33 ms frame period almost no toggle lands inside an exposure, so the tool warns and
tells you to raise `camera.exposure_us` to about half the frame period (**~4000 µs at 120 fps**). Run it twice:
once at the operational exposure for the latency statistics, once at ~4000 µs for the offset.

**Writes.** `calib/latency.json` → `camera.timestamp_offset_ns`, `camera.exposure_to_user_ms` (mean/std/p50/p95/
min/max/n).

**Pass criteria.** No `WARNING:` lines left; ≥ the tool's minimum of partially lit frames for the offset; p95 of
`exposure→user` within a frame period or two of the median (a long tail means the Pi is starved — check the
frame-drop count it prints).

### 6b. Actuator: fire command → trigger break

**Setup — read this twice.**
- **The gun is UNLOADED.** Better: do this on a bench fixture (the trigger set clamped to a jig) rather than on the
  gun at all.
- A **break-sense microswitch** wired from **BCM 22** (`gpio.break_sense_pin`, header pin 15) to GND, positioned so
  the trigger breaking closes it (internal pull-up: break = falling edge, sear reset = rising edge).
- The **operator holds the arming button** (BCM 17, header pin 11) for the whole run. The tool waits for it, pauses
  the moment it is released, and never bypasses the interlock.
- MG996R powered from its separate 7.0 V XL4015, with ground common with the Pi — see docs/PROTOTYPING.md.
  The arming interlock is software only, with no hardware supply cut; open the battery lever connector to
  make the actuator safe. The PQ12 fallback instead needs 12 V through an H-bridge.

```bash
tbot latency-actuator --type servo --trials 20
tbot latency-actuator --type servo --trials 20 --notes "MG996R, 6 V BEC, lever arm 12.5 mm"
tbot latency-actuator --type pq12 --trials 20 --timeout-s 3.0   # fallback only
tbot latency-actuator --sim                                    # self-test with simulated GPIO/actuator
```

The tool prints the wiring procedure before it starts and inserts a random pause between trials so the operator
cannot anticipate the shot.

**Writes.** `calib/latency.json` → `actuator.<type>.cmd_to_break_ms` and `.cmd_to_rest_ms` (full stats).

**Pass criteria.** `timeouts: 0 of N`; inspect the measured distribution rather than requiring a p95 − p50 of
only a few ms. The expected dominant servo jitter is 50 Hz PWM frame quantisation: 0–20 ms command wait, or
about 5.8 ms one-sigma assuming uniform phase. Compare the measured servo delay against the inferred
`servo_cmd_to_break_ms = 115` default. If no break edge is ever seen, the fixture or the actuator direction is
wrong (`actuator.servo.rest_us`/`fire_us` may be swapped — confirm on the bench).

### 6c. Processing time

```bash
tbot bench --frames 300
tbot bench --frames 300 --threads 2 --scenario skeet_crossing --json
```

**Writes.** `calib/bench.json` (per-stage mean/p50/p95, verdict against the frame period).

**Pass criteria.** Total p95 under the frame period (8.33 ms at 120 fps). If not: drop `camera.fps`, keep
640 × 400, raise `detector.bg_downscale`, or reduce `tracker.max_tracks` — and re-run, because the solver's
`horizon_frames` look-ahead assumes the configured frame period.

### 6d. The latency budget

```bash
tbot latency-report
tbot latency-report --range-m 20 --speed-mps 20 --json
```

It prints every component (measured vs config default, with its source), the shot time of flight to the reference
range, and per actuator: `cmd→break`, jitter, decision→muzzle, photon→arrival, and how far the reference clay
moves in that time — in metres and degrees. That last column is the lead the solver has to predict. The selected
MG996R's estimated ~0.1 s command-to-break means ~2 m at 20 m/s, versus 14–22 m for the PQ12 fallback's
~0.7–1.1 s. Read the report after every re-measurement.

**Pass criteria.** The components measured for the fitted rig say `measured`, not `default`. Defaults for the
unfitted fallback need not be measured.

### 6e. End-to-end check on the HITL bench

With the calibrated camera, detector, boresight and latency budget, point the rig at a monitor and run the
[HITL bench procedure](HITL.md). It measures display and pipeline timing, tests detection and tracking, and
compares moving-target lead against the known screen trajectory. Aim the bore toward the targets' exit side
for a longer sweep runway; the rig waits for the page to enter fullscreen. Start dry; for live actuator timing
the gun must be **unloaded** and the operator must hold the arming button. The bench writes `calib/hitl.json`
and run reports including self-contained `report.html`; `--write-latency` explicitly merges measured actuator
latency into `calib/latency.json`.

---

## 7. Dry run on the range

**Purpose.** Confirm, with the gun on your shoulder and clays in the air, that the system would have fired at the
right moments — before `fire.live` is ever set.

**Setup.** Everything calibrated. Gun **unloaded** for the first passes. `fire.live` stays `false` (the default);
the pipeline logs decisions and never drives the actuator.

```bash
tbot run --stream --record armed
tbot run --stream --record armed --log-jsonl sessions/dryrun-01.jsonl
```

`--record armed` dumps a session around each arming press (`record.preroll_s = 2`, `postroll_s = 1`) to
`sessions/YYYYmmdd-HHMMSS-<tag>/`, with the telemetry events in `events.jsonl`. Shoulder the gun, press and hold
the arming button as you would to shoot, swing and track the clay, release.

**Review.** Replay each session on the laptop and step through the decisions:

```bash
tbot run --source replay:sessions/20261026-141230-armed --stream
```

**Pass criteria.**
- The "would fire" instants land where you would have pulled the trigger — not 200 ms late, not on a false blob.
- `FireDecision.reason` on the no-fire frames is a sensible one (`not armed`, track not confirmed, range gate,
  `p_hit` below `fire.min_hit_probability = 0.5`, "not the best moment"), never a configuration error.
- No fire decision while the arming button is open — the reason must contain `not armed`.
- Track confirmation happens early in the clay's arc, not at the end of it.

**Regression check.** After any change to the detector, tracker, solver or calibration:

```bash
tbot sim                          # all scenarios, 1 seed
tbot sim --seeds 5 --out sim.json # the number to compare against last time
tbot sim --not-armed              # interlock check: must produce zero shots
python3 -m pytest                 # the unit suite
```

Only after all of this does `tbot run --live` make sense — on a range, with the muzzle downrange, your hand on
the gun and your finger on the arming button. The interlock is software only; no hardware cut is installed.
Open the battery lever connector to make the actuator safe. An additional series hardware cut is recommended
but not installed (README, safety model).
