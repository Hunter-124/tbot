# Integration status (handoff)

Living notes on bench integration of the real rig: what is calibrated, what is still open, and where the
per-rig values live. Last updated 2026-11-15 (IMU calibrated, multi-shot, speed-ups). The Pi keeps its own
`config/local.toml` and `calib/` (git-ignored; `scripts/deploy.sh push` never overwrites them), so the values
below are a record, not something a checkout restores.

## Access and setup

- Pi: `ssh -o UserKnownHostsFile=~/.ssh/known_hosts_tbot pi@tbot.local` (192.168.0.168 on the home LAN; when
  mDNS does not resolve use the IP with `-o HostKeyAlias=tbot.local`). The host key changes on every reflash:
  re-run `ssh-keyscan -t ed25519 tbot.local > ~/.ssh/known_hosts_tbot` instead of editing `~/.ssh/known_hosts`.
  Credentials: `software/pi/README.md`.
- **Wi-Fi is weak on the bench** (up to 75 % packet loss seen 2026-11-15): copies and SSH sessions drop. Push
  file by file with retries and verify checksums; run long jobs detached (`setsid nohup … > log`) and poll.
  At the range, do not depend on Wi-Fi for anything in the firing path (it is not in it); the debug stream may
  stutter.
- The Pi has no git; push code with `scripts/deploy.sh push`, or `scp` the changed files.
- Run anything that opens the camera **detached**: a command holding the camera can hang an SSH session.
- Blue Snowball mic: address it as `plughw:CARD=Snowball,DEV=0` (its card number changes across reboots).

## Per-rig configuration on the Pi (`config/local.toml`)

| key | value | source |
|---|---|---|
| `actuator.servo.rest_us` / `fire_us` | 1420 / 1530 | bench; lever has some spline play |
| `actuator.servo.min_us` / `max_us` | 1400 / 1540 | valid lever travel (the actuator refuses to fire outside it) |
| `camera.rotate_180` | false | the board is upright (now also the default) |
| `camera.fps` | 100 (was 50 until 2026-11-17) | see "Frame rate" |

## Calibration on the Pi (`calib/`)

| item | value | how | redo with the final lens? |
|---|---|---|---|
| camera timestamp offset | +0.03 ms | `latency-camera`, 4000 us exposure (a later 500 us run could not measure it and kept this value) | no (sensor) |
| exposure end -> user space | ~9.2 ms at 500 us | `latency-camera` | no |
| servo command -> break | **95 ms** (std 2.5 assumed) | video: LED at the fire command, lever motion centre 92 +/- 10 ms; mic 161 ms incl. USB latency | no |
| IMU gyro bias | +0.82 / +6.45 / +0.48 deg/s | `calib-imu bias`; drifts with temperature (z 0.48 -> 0.84 between 24 and 22 C) | no, but redo at the range |
| IMU axis map | `R_gun_imu` = [[0,-1,0],[-1,0,0],[0,0,-1]] (chip face-down, X/Y swapped) | `calib-imu axes --auto`, 3 hand-held poses, fit within 0.8 deg; also the default now | no |
| camera-IMU time offset | -6.31 ms (added to IMU stamps) | `calib-imu sync` on a 15 s hand-held swing, correlation 0.971 | no |
| boresight | pitch -1.71, yaw -4.03 deg | barrel silhouette, nominal lens model | **yes** |
| lens intrinsics | none (nominal model) | `tbot board` + `calib-intrinsics` | **yes** |
| detector blur `detector.psf_px` | 1.2 (guess) | `calib-detector --psf` on clay footage | **yes** |

## Software state (main)

- Multi-shot (`tbot/fire/engagement.py`): one press fires up to `fire.max_shots_per_arm` shots (default 1; was 3 until 2026-11-16), each at its
  own clay; >= `fire.min_shot_interval_ms = 250` between commands; a shot-at clay (and any duplicate track
  within 1 deg) is never engaged again; tracks born after a shot inside a cone around the broken clay's
  predicted path (2 deg, +15 deg/s, 1.5 s) are debris; after a shot a target needs `min_track_hits` fresh
  detections later than `recoil_settle_ms = 100` past exit. The pipeline does not spend a shot while the
  servo is still cycling (hold 150 + return 250 ms = 400 ms, so doubles are >= ~400 ms apart until the return
  time is measured shorter on the bench). Solver refuses targets slower than 6 m/s (cartesian tracker).
- Camera: `tbot run` / `tbot record` reopen a stalled camera (`camera.stall_s` = 0.5 s) instead of ending.
- Heat: `tbot run` warns at 75 C (firmware throttles at 80 C) and on the soft temperature limit.
- Tracker: 3-D cartesian EKF (`tracker.model = "cartesian"`). Detector: matched filter + hysteresis + speck
  and local-contrast tests. Solver: extrapolates the gun swing with its angular acceleration.
- Servo: limp when idle, short bursts, refuses to fire outside calibrated travel.

## Added 2026-11-16 (range preparation; walk-through in [RANGE_DAY.md](RANGE_DAY.md))

- `calib-boresight --laser [--laser-turns N] [--laser-bg] [--check]`: headless bore-laser boresight. Tested on
  synthetic images and, on the rig, only for the no-dot case (dark room: refuses, writes nothing). **Not yet
  run with a real laser.**
- Recoil timing: `tbot run --live` logs `recoil` / `no_recoil` events per fire command from the IMU
  (`fire.recoil_detect_g = 3`, held for 3 samples); `tbot latency-recoil [--write]` turns them into
  `actuator.servo.cmd_to_break`. **Thresholds are untested on real recoil** (first live shots will show).
- Gyro: the attitude filter re-zeroes the gyro whenever the gun rests for 1 s (`attitude.auto_bias`). Measured
  on the rig at 63 C: 2.2 / 0.0 / -0.9 deg/s left over from the 24 C calibration, removed within 1 s.
  The MPU-6050 start-up transient (first ~7 samples up to 70 deg/s) is now discarded.
- `scripts/tbotctl` + `pi/systemd/tbot-run.service` (installed on the Pi, **not** enabled at boot):
  `tbotctl start [dry|live] | stop | status | log | boot on|off | preflight`. Live mode is kept in `/run`, so
  a reboot always returns to dry run.
- Real-footage check: `sessions/sync1` (15 s indoor hand-held swing) replayed armed in dry run: 0 fire
  decisions out of 749 frames (max P(hit) 0.05) despite ~8 clutter detections per frame.
- Live dry run on the rig after these changes: 50 fps, 18.3 ms p50 / 25.9 ms p95 per frame, 3 drops in 400.
- Range access: the Pi is on Ethernet only (wlan0 is not configured). At the range use a direct cable.

## Added 2026-11-17 (vision speed, detection with the 12 mm lens, press recording)

Measured on the rig's Pi 3B with the real camera and IMU (dark room, so no clay in view), and on the
simulator rendered through the rig's lens calibration. Not yet seen outdoors: the first backyard presses are
the test, and every one of them is recorded ([PRESS_CLIPS.md](PRESS_CLIPS.md)).

- **Press recording.** Every held press of the arming button is one session: raw frames, IMU, events,
  `telemetry.jsonl` (per frame: detections with boxes, scenery dropped, tracks, fire solution, timing,
  exposure), `summary.json` and an annotated `clip.mp4`. Browse at `http://<pi>:8080/clips`. The service no
  longer writes `logs/run-*.jsonl`.
- **Sensor process** (`camera.process`, `tbot/camera/proc.py`): camera + IMU are acquired in a child process and
  handed over through shared memory. In one process their threads held the interpreter lock: at 100 fps the
  capture thread used 33 % of a core, the IMU thread 22 %, picamera2's own threads 20 %, and every detector
  step ran 2-3x slower live than on stored frames. Live, nothing in view, camera at 100 fps:

  | | frames processed | detect p50 | frame work p50 / p95 | frame age p50 |
  |---|---|---|---|---|
  | before (threads, full-res detector) | 50 /s | 13.0 ms | 15.5 / 38.8 ms | 49 ms |
  | sensor process + half-res detector | 86 /s (100 /s once warm, service log) | 4.7 ms | 6.5 / 13.5 ms | 25 ms |

  Also removed from the frame loop: a `vcgencmd` child process once a second (20-40 ms stall each; now a
  background thread), overlay drawing and JPEG encoding (own thread, and only while a browser is watching),
  the per-frame deep copy of the health dict, full garbage collections over the whole heap (`gc.freeze`).
- With a clay tracked and the button held (simulator frames on the Pi): detect 6.6 + track 2.7 + solve 2.6 ms,
  about 14 ms a frame with IMU and bookkeeping, so ~70 frames/s where it matters. Next levers: the per-blob
  measurement (3.4 ms for one clay, all small NumPy calls), tracker and solver.
- **Idle rate** (`camera.idle_fps = 30`): while the button is up the pipeline processes 30 frames/s; the moment
  it is held, every frame. Heat is why: flat out the bare Pi passed 72 C in 8 minutes indoors. The sensor
  process itself is ~90 % of a core at 100 fps whatever the pipeline does (capture, auto-exposure metering,
  IMU), and holds the board near 70 C indoors; if `health: Pi at 75 C` shows up outdoors, lower `camera.fps`
  or fit the heatsink.
- **Detector for the 12 mm lens.** A clay is now 5-27 px across (8-40 m), not 1.5-8 px:
  - `detector.downscale` (0 = auto -> 2 with this lens): the detector runs on the frame averaged 2x2 and
    reports full-resolution pixels. At full resolution a face-on clay at 8 m (572 px) was over `max_area_px`
    and never detected, and sizes were off by -6..+5 %; now 5-27 px discs measure within 3 %.
  - `detector.bg_mode` (auto -> morph when reduced): edge-preserving background. The box blur mixed sky and
    ground for 36 px around the horizon and lost the clay there: synthetic recall (all scenarios, 2 seeds)
    817/829 with 0.08 false detections a frame -> **829/829 with 0.01**.
  - `detector.local_noise_cell`: busy regions (trees) raise their own threshold; the frame-wide noise is read
    at a low quantile so a picture that is mostly trees does not desensitise the sky.
  - Scenery filter (`detector.static_reject`, `tbot/vision/clutter.py`): candidates that keep their world
    direction while the gun turns (IMU), allowing for parallax of anything beyond `static_min_range_m`, are
    dropped before they are measured; grey crosses in the clips. On real indoor stills swung through the
    camera model with a pasted clay: ~50 scenery candidates dropped per frame, detector time in clutter down
    ~3x, the clay kept 2-3x more often (an indoor room is far busier than sky and a tree line; that test is
    a stress case, not a prediction). **Never run with real attitude outdoors yet**; `static_reject = false`
    turns it off. A clay with under ~1.7 deg/s of angular motion that never crossed the sky before looks
    like scenery to it.
- **Found, not fixed: size reads 12-17 % large on the simulator's clay**, at any resolution. The diameter is a
  second-moment width, exact for an ellipse; an edge-on clay's silhouette is nearer a rectangle (x1.155).
  Range from size then reads ~13 % short. Needs a real clay at a taped distance, edge-on and face-on, before
  choosing a correction (`tbot calib-detector --psf --range-m`).
- Closed-loop simulator, rig lens, 3 seeds x 8 scenarios: 14 shots / 6 hits after, 14 / 7 before (run-to-run
  noise is +-2, see "Simulator notes").

## Added 2026-11-17, outdoors (first daylight session, backyard: trees, sky, birds, a jet)

Everything here was found on presses recorded that morning and is checked against them. Photos:
[field/2026-11-17](field/2026-11-17/README.md).

**Attitude filter was dragged by the gun's own motion (fixed).** `attitude.accel_gain` was 0.02 per sample: the
tilt re-levelled itself from the accelerometer with a 0.1 s time constant, and a swung gun's acceleration (the
pod is ~0.9 m from the shoulder; the magnitude gate lets 0.5 g sideways through) was read as tilt. Test: predict
how far the picture shifts between two frames from the attitude, compare with the shift measured in the picture
(phase correlation, press `074450`, 169 frame pairs): gyro alone 1 px off (p50), the filter as it was **43 px**
at the 90th percentile, now **6 px** (the same as the gyro alone). Now `accel_gain = 0.001` (2 s) and the
accelerometer is only used while the gun turns slower than `attitude.accel_max_rate_dps` (10). This fed the
tracker (world-frame clay motion), the scenery filter and the fire solution.

**Detector went blind with trees in most of the frame (fixed).** The noise sigma was one estimate over the whole
frame (a low quantile, right until texture covers ~65 % of it): 1.2 DN over sky, 26-37 DN with trees filling
the frame, i.e. a threshold nothing can pass, for 5 s of an 8 s press. Now the sigma is the 25th percentile of
24 tiles' own estimates (`NOISE_TILES`): the quietest part of the picture. Same frames: 1.7-3.0 DN. Busy
regions are still raised locally (`local_noise_cell`).

**Tree fringe against sky: 100-150 candidates a frame, 5-10 fps (improved, not solved).**
- `detector.clutter_ring_max` (0.05): in a frame with 8+ candidates a candidate must stand alone (at most that
  share of the ring around its box may be other blobs). 130 -> ~15 on the recorded frames.
- At most 6 candidates are measured per frame when a candidate filter is set (was 24; each costs 4-7 ms on the
  Pi 3B), solid and sizeable ones first.
- `detector.min_size_frac` (0.4): detections narrower than that fraction of a clay at `fire.max_range_m` are
  dropped (2.2 px with the 12 mm lens). They were half of all detections and each started a track.
- Scenery filter: same result, one stacked pass (5.7 -> 2.6 ms for one candidate on the Pi).
- Measured on the gun after the first three: open sky ~100 fps, a tree edge in view 16-25 fps (was 5-10).
  **Still open:** the scenery filter drops only ~2 of ~10 fringe blobs a frame even with the corrected attitude
  (they flicker from frame to frame); one target in clear sky still costs ~10 ms detect + 4 ms track + 2.6 ms
  solve on top of the 8 ms base (~37 fps with a bird in view).

**Auto-exposure** needed no change: highlights held at the target, nothing clipped, settles within a few frames
of a swing from trees to sky. Against open blue sky it sits at the sensor's shortest exposure (9 us, gain 1)
with the 98th percentile at the target: anything brighter (white cloud, haze near the sun) will clip, and only
a darker lens / ND filter would change that.

**Why a still clay on the deck rail got no dry-fire:** 0.5 m from the lens (150 px, out of focus: not a small
blob), not moving (`fire.min_clay_speed_mps`, and the scenery filter drops what is fixed in the world), inside
`fire.min_range_m`. By design.

**Test mode and shot scoring (new).** `tbotctl start test` = dry run that fires on paper at anything it tracks
(no speed floor, minimum range 0.5 m, 6 shots a press; things fixed in the world are still ignored). `tbot score sessions/<name>
--image` checks each shot against the frames after it: bore at the exit time (IMU replayed) against the target
at the arrival time (its detections), split into gun-extrapolation and target-extrapolation error, and writes
`shot_<n>.png`. First real target (a hawk, hand-held, press `080902`, ordinary dry mode): miss 0.78 deg against
a 0.75 deg pattern, gun error 1.01 deg, target error 0.49 deg.

**How well the gun predicts its own bore over the trigger delay** (6 presses of loose hand-held sweeping, 3578
moments, 104 ms): error p50 0.32 deg, p90 1.04 deg, p99 2.0 deg against a 0.75 deg pattern. The solver's method
(latest rate + acceleration from a 30 ms slope) was the best of those tried; longer windows and a jerk term
were worse (p90 1.3-2.0 deg). The error is predictable from the motion itself: the smoothest 30 % of moments
(small acceleration and little change of rate over the last 100 ms) have p50 0.14 / p90 0.36 deg, the roughest
30 % p50 0.68 / p90 1.49 deg.

**Firing rule, tested on virtual clays through that recorded motion** (a clay flown at constant angular velocity
so that the real bore path passes within 0.5 deg of it; the decision rule replayed frame by frame; scored
against the real bore at exit; 315 clays per speed class; "clays hit"):

| rule | slow 3-15 deg/s | fast 15-40 | very fast 40-90 |
|---|---|---|---|
| on frames, P >= 0.5, 100 fps (as it was) | 45 % | 40 % | 30 % |
| on frames, P >= 0.5, 30 fps | 44 % | 36 % | 15 % |
| on frames, P >= 0.5, 15 fps | 39 % | 25 % | 7 % |
| between frames, P >= 0.3, 100 fps | 49 % | 46 % | 48 % |
| between frames, P >= 0.3, 30 fps | 41 % | 45 % | 43 % |
| between frames, P >= 0.3, 15 fps | 34 % | 42 % | 37 % |

Firing as soon as P clears the bar instead of at the peak was worse everywhere (23-37 %). What was changed:
- `fire.subframe`: the command goes out at the instant of closest approach inside the frame period (one extra
  solution per frame, from the solutions for "now" and "one frame later"); the look-ahead period is the real
  loop period, not the camera's. Timing no longer depends on the frame rate.
- `fire.min_hit_probability` 0.5 -> 0.4. Not 0.3: the test above assumes a settled track, and `tbot sim`
  (10 seeds x 3 scenarios, +/-2 is noise) broke 18/30 with the old rule, 20/30 with subframe + the new sigma
  at 0.5, 21/30 at 0.4 (crossing 5 -> 8 of 10), but at 0.3 going-away clays fell from 6/6 to 2/6: it fires
  before the track has settled.
- The swing uncertainty is fitted to the data: `gun_rate_sigma_dps` 0.7 (a floor), `gun_accel_sigma_frac`
  0.3 -> 0.12, `gun_rate_change_sigma_frac` 0.18 of the change of rate over the last 100 ms. Error / sigma on
  the recorded motion: p90 2.20 and p99 3.9 (an honest model: 2.15, 3.03) against 2.76 and 6.3 before. Worth
  1-3 points of clays hit on its own.
- Open: the track's own uncertainty is too small while it is young (the reason 0.3 fails on going-away clays).
  Real shots at real targets so far (2): target extrapolation error 0.49 and 0.89 deg, as large as the gun's.

Shot travel time (it is in the solution, with drag: #8 at 366 m/s): 40 ft 39.8 ms, 60 ft 64.0 ms, 75 ft
83.9 ms, 100 ft 120.9 ms; at constant box speed those would read 33, 50, 63 and 83 ms.

## Added 2026-11-17, outdoors, second round (frame time around trees; every frame scored)

**Presses of 09:00 (a hawk past a tree line, a jet).** No dry shot in either.
- Hawk (`090017`): followed for 2 s at 10-11 m apparent range, bore 0.5-1.2 deg off it throughout; P(hit)
  peaked at 0.38 against the bar of 0.40. Scored frame by frame (`tbot score --every`, below): of 33 frames
  with a solution 4 would have hit, the longest unbroken chance 129 ms; real miss p50 0.87 deg against the
  0.75 deg pattern. The gun was right not to fire most of the time, and marginal for one eighth of a second.
- Jet (`090035`, and the first half of `090013`): no track. It crosses the sky at well under the 1.7 deg/s
  below which the scenery filter calls a thing fixed in the world; the jet that was "hit" at 08:21 was fired
  on while test mode had that filter switched off. Test mode is for things that move across the sky.
- Loop time in that press: 40-100 ms a frame with the tree line in view (10-25 fps), and two frames of
  340 ms. That is what the rest of this section is about.

**Every frame scored: `tbot score --every`.** Each frame in which the solver had a solution for a settled
track is scored as if the command had gone out on it (same method as for a real shot). Over the six presses
of today that have a target (192 frames):

| P(hit) the solver gave | frames | really would have hit |
|---|---|---|
| 0.0 - 0.1 | 145 | 5 % |
| 0.1 - 0.2 | 15 | 13 % |
| 0.2 - 0.3 | 9 | 22 % |
| 0.3 - 0.4 | 8 | 62 % |
| 0.4 - 0.5 | 13 | 100 % |
| 0.5 - 0.7 | 2 | 50 % |

Small numbers, and the frames of one press are not independent (the 13 at 0.4-0.5 are all the jet), but the
ordering is right and nothing says the probability is optimistic. Gun extrapolation error p50 0.57 deg, target
extrapolation error p50 0.77 deg (birds: wings, turns, and 15-25 fps): the target half is the larger one.

**Where the frame time went (Pi 3B, the hawk press replayed through the live loop, `tbot run --source
replay:<press>`; ms per frame):**

| | detect | track | solve | all work |
|---|---|---|---|---|
| before | 34-35 | 13-14 | 6.0 | 56-58 |
| after | 35 | 4.7 | 3.8 | 47 |

The same replay, old code against new, run alternately with the gun's own service running beside it (the Pi
was hot and partly frequency-capped during these runs, so read the ratios, not the milliseconds): hawk near
trees 55.8 -> 46.8 ms (-16 %), tree fringe (`075514`) 66.5 -> 54.7 (-18 %), bird in open sky with trees below
(`082831`) 47.0 -> 38.0 (-19 %), hawk in open sky (`080902`) 26.4 -> 23.7 (-10 %). What changed:
- **Per-detection and per-track arithmetic in plain floats** (`optics.CameraModel.ray_xyz` / `ifov1`,
  `tracker._measure_few`, `tracker3d`: gating, EKF update on the 3 position columns, new track, views,
  prediction). Every NumPy call on a 3-vector or a 3x3 costs 15-50 us on the Pi whatever it computes and one
  detection used to cost ~150 of them. Tracker update with 4 tracks and 4 detections: 13.8 -> 3.9 ms; measuring
  4 detections 3.9 -> 0.24 ms; one prediction 0.54 -> 0.08 ms. Same numbers: tracks agree to 1e-10 deg on
  replays, `tests/test_fast_paths.py` compares each piece with the matrix form.
- **Solver**: the look one frame ahead is the aim only (no uncertainty, no probability), and the full
  solution at the instant of closest approach is worked out only when that approach can come within 4 sigma
  of the pattern. 7.4 -> 4.4 ms. Checked against the full computation on identical inputs: 4724 solved
  frames (6 presses x 4 look-ahead periods), largest difference in P(hit) 4e-7, no decision changed.
- **Established targets are measured first** (`StaticFilter.begin(focus_dirs_w=...)`): the candidates where
  tracks with `fire.min_track_hits` hits should be (a zone as wide as the tracker's own gate, at least 25 px)
  go to the head of the list, so the target always gets one of the six measurements a cluttered frame is
  allowed.
- **`detector.followed_others_max` (off, 0)**: with N = 2, a frame in which the main target was found where
  expected measures only two candidates besides the established targets. A swing past trees keeps bringing
  new specks into view and each measurement costs ~6 ms. Measured: another 6-8 % of the frame time on the bird
  presses, and 5-11 % fewer detections of the followed bird (one that leaves its zone is skipped until it is
  searched for again). Not worth it as it stands; the switch is there for when the tracker holds a target better.
- **A press being saved no longer stalls the next one.** The two 340 ms frames came 0.4 and 1.6 s into a
  press started right after the previous one was released: the dump (IMU rows, one JSON line per frame) is
  pure-Python work in a thread of the same process, and the frame loop waited for the interpreter lock at
  every OpenCV call. The dump now goes out in small pieces with a pause between them and the switch interval
  is 0.5 ms instead of 5. Not yet seen on the gun (it needs two presses back to back).

**What is left, measured** (same replay): detection is now three quarters of the work. Per frame: background
1.7 ms, threshold 4.3 (of which the local threshold 1.4 and the noise estimate 0.9: 12 ms once every ~13
frames), connected components 1.2, scenery filter 2.5, and **6.1 ms per measured candidate** (4.3 ms of window
arithmetic, 1.8 ms adaptive moments), three candidates a frame on average. One firing solution 2.5 ms, of which
the hit probability quadrature 0.9. The tracker also refuses a jinking bird at 15 fps now and then (its gate
is sized for something that flies like a clay): 3 of ~35 detections of the small birds.

## Field mode (2026-11-17: first range day without a laptop)

Set up with `tbotctl boot live`: the gun comes up **LIVE at every power-up** until `tbotctl boot dry`.
- Mode: `/run/tbot/mode` (RAM, set by `tbotctl start <mode>`) wins over `/etc/tbot/boot-mode` (set by
  `tbotctl boot live|dry|test`), which wins over dry. `tbotctl status` shows both.
- The unit restarts on anything (`Restart=always`, no start limit) and has `WatchdogSec=60`: the frame loop
  reports in every 2 s, a hang is killed and restarted. Tested on the gun: see the commit message.
- Presses are still recorded in live mode; their clips are not rendered on the gun (`record.clip=off` in
  live: a render is a second Python + OpenCV process for 10-20 s after every press). `tbot clip sessions/<name>`
  renders one later.
- `record.min_free_mb = 1500`: a press is not saved when the disk has less than that free; the pipeline keeps
  running. A 29 GB card with 22 GB free holds ~150 presses of 4 s.
- A press cut off by a power loss is lost or incomplete (no fsync per frame); everything before it is intact.
  The root file system is ext4 with `fsck.repair=yes`. Session names stay unique when the clock jumps back
  (no RTC).
- Heat is the limit to watch: with the service alone the Pi sat at 68-71 C on a cool morning; with an hour of
  benchmarks on top it reached 81 C and capped its clock, which roughly doubles the frame time. Shade and air
  for the pod.

## Frame rate and CPU

*(2026-11-15 numbers, before the sensor process and the half-resolution detector; kept for the record.)*

- Camera/driver: capture-only 50-200 fps with zero drops; the limit is processing.
- Pi 3B pipeline with a clay tracked (synthetic crossing): detect 13.0 + track 2.6 + solve 1.7 = **17.5 ms
  p50, 23 ms p95** (was ~25 ms before 2026-11-15's speed-ups: detector 8-bit threshold / cv2.moments /
  vectorised rejection, tracker scalar ray path). Live camera, no clay: 7.0 ms p50. Keep **50 fps**.
- OpenCV uses 4 threads: 1 thread is much slower (live 12.4 vs 7.0 ms, synthetic 23 vs 17.6 ms), so the
  ~150-190 % process CPU (all four cores ~50 %) is buying speed, not wasted.
- Heat: 3 min live at 50 fps took the bare Pi from 56 to 66 C and still rising slowly (no throttling). In a
  closed pod in the sun this can reach the 80 C throttle: give the Pi a heatsink and airflow/shade, and watch
  for the 75 C warning.
- Further speed would need pipelining across processes (detect in one, track/solve/fire in another): more
  throughput, same latency. Not done before the first trip.

## Simulator notes

- `tbot sim` is not repeatable run to run (SimClock adds real processing time): ~1/3 of shots differ between
  runs of the same code, so +/-2 hits out of ~20 is noise. Compare code changes frame by frame instead.
- Doubles (`skeet_doubles`, `trap_doubles`), clay breaks with fragments and debris-shot counting exist. Latest
  (95-100 ms delay, 6 seeds): singles outgoing 6/6, trap 6/6, incoming 3/3 of shots taken; **0 debris shots**;
  doubles mostly get one shot (second clay not in the solution before the run ends).
- Scenarios last 2.5 s and the truth freezes after that: late crossing shots (the simulated swing fires ~2 s
  after launch) are judged against a frozen clay, so `skeet_crossing` "misses" are mostly artefacts.
- Fused range from clay size lags radial motion (trap ~0.77-0.82 of true at shot, incoming ~0.84); per-frame
  size range is ~0.91 with psf 1.2 and ~0.98 with psf 1.6 in the sim. Calibrate psf on real footage.

## Before the range (checklist)

1. Install the **final lens with the IR filter/protector**, set focus (`tbot focus`), then: lens intrinsics
   (`tbot board`, `calib-intrinsics`, chessboard), then `calib-boresight --barrel` on a full 1280x800 still,
   then re-check the static/barrel mask. These depend on the lens; the IMU, latency and servo values do not.
2. Secure and strain-relieve the camera ribbon at both ends; prove it with a 60 s hard-swing recording.
3. Heatsink + airflow/shade for the Pi in the pod.
4. At the range: rest the gun still for a few seconds after start (the gyro re-zeroes itself), then record sky + clay footage
   (`tbot record`) before trying live shots: detector thresholds, mask and `psf_px` need real footage
   (`calib-detector`). Exposure may need raising with the IR filter fitted.
5. Optional, with the user present and agreeing: measure the servo return time so doubles can be faster than
   ~400 ms; servo wake-up from limp (~115 ms) could be cut by holding rest while armed (needs re-measurement).
6. Power: under-voltage/throttling has been seen when servo bursts sag the shared battery.

## Known test failure

`tests/test_hitl.py::test_sim_live_geometry_and_flash_end_to_end` fails on both the Pi and a PC on code that
predates this work.
