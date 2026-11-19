# Integration status (handoff)

Living notes on bench integration of the real rig: what is calibrated, what is still open, and where the
per-rig values live. Last updated 2026-11-17 (after the first range day: see the last section). The Pi keeps its own
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
| servo command -> break | **155 ms** (the file still says 95, an estimate; its own microphone measurement, 161.3 ms to the bang, is what is used: see the last section) | microphone, 10 shots; confirmed by the recoil of 2 live shots (160.2, 162.0 ms) | **yes, after the mount rebuild** (`tbot latency-recoil`) |
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

- `tbot sim` repeats exactly since 2026-11-17 evening (`sim.process_ms`: the gun's work time is modelled, the
  clock no longer runs on this computer's). Before that ~1/3 of shots differed between runs of the same code.
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

## First live range day, 2026-11-17 (data: [field_data/2026-11-17-range](../field_data/2026-11-17-range/README.md))

Nine presses, seven fire commands, two real shots (both clays broke), then the servo mount broke. The
recordings, clips, scores and the gun's calibration of the day are in that folder. To act on before the next
live day:
- **Trigger delay**: recoil says 161 ms command to ignition (two shots, 1.4 ms apart), the same as the bench
  microphone said; the gun used 95 + 6 ms. Set `cmd_to_break` to ~155 ms once the new servo mount is on, and
  confirm with `tbot latency-recoil`.
- **Armed frame rate** was 19-43 fps, not 100.
- **Gun extrapolation error** (p50 0.82 deg) is now larger than the pattern radius (0.75 deg).

## After the first range day (2026-11-17, evening): delay, swing model, firing rule, test benches

Everything below was worked out on the laptop from the range-day recordings; the gun was off. **Nothing here
has been on the gun yet.** `scripts/deploy.sh push` brings it there; the first presses should be dry.

### Trigger delay: 155 ms command to break (164 ms to the muzzle)

- `latency.servo_cmd_to_break_ms` 115 -> **155** (recoil of the two live shots, 160.2 and 162.0 ms to ignition,
  and the ten bench shots by microphone, 161.3 +/- 1.0 ms; minus the 6 ms lock time). `latency.jitter_ms`
  6 -> 2 (measured 1.0 and 1.4 ms: the PWM starts with the command, the 50 Hz frame does not quantise it).
- **The gun's own `calib/latency.json` needs no edit.** It holds the 95 ms video estimate and, beside it, the
  microphone measurement that was set aside. A measurement kept beside an estimate now wins over the estimate
  (`config.LatencyBudget`): that file reads 155.3 ms as it is. `tbot latency-report` shows the source as
  `measured (microphone)`.
- The servo mount has been rebuilt since, which can move the delay. Every live shot is still timed by its
  recoil, and a shot more than 15 ms from the assumed delay now logs `TRIGGER DELAY is off by ...`. After a
  few shots with the new mount: `tbot latency-recoil --write` (`--min-shots 2` if they agree).
- What a wrong delay costs (virtual clays, below): 10 ms off 0.4 points of clays hit, 20 ms 1 point, 60 ms
  (the range day) 6 points. Within 10 ms is good enough.
- The two real shots judged at the time the charge really left (recoil, not the plan): both passed **1.04 deg**
  from the clay, against a 0.75 deg break radius, and both clays broke. On those two targets the extra 60 ms
  moved the bore little against the clay (1.14 -> 1.04 and 0.78 -> 1.04 deg), which is why the wrong delay did
  not show. Two breaks at 1.0 deg say the break radius for Modified may be nearer 1 deg than 0.75; pattern
  the gun ([PATTERN.md](PATTERN.md)) before changing `shot.pattern_radius_ref_m`.

### The swing over the delay: the acceleration term was hurting

Over 164 ms the solver's constant-acceleration extrapolation of the swing is **worse than no acceleration term at
all**. Range-day IMU, the 135 frames of 9 presses that had a target (gun turning 10-30 deg/s), error of the
predicted bore at exit:

| extrapolation | p50 | p90 | rms |
|---|---|---|---|
| rate + constant acceleration (as it was) | 1.10 deg | 2.98 deg | 2.19 deg |
| rate only | 0.69 | 1.64 | 1.03 |
| rate + acceleration fading with 0.02 s (`fire.gun_accel_decay_s`, now) | 0.70 | 1.60 | 1.00 |

(The pattern is 0.75 deg. At the 104 ms the gun believed it had, the fading term is also the better one:
0.41 / 1.23 / 0.84 -> 0.31 / 0.78 / 0.48 deg.) A hand does not keep an acceleration for 160 ms. A decay of the
rate as well fits the whole press better, but that is the gun being mounted and lowered (100-190 deg/s at the
ends of a press); on a target it is within the noise, so it is not in.

The uncertainty that goes with it is refitted. The error grows with the **square** of the time ahead (p50 0.31
deg at 104 ms, 0.70 at 164, 0.85 at 190), so every term now does: a floor (`gun_accel_floor_dps2` 20), a share
of the swing rate (`gun_rate_drift_per_s` 1.0), 0.6 of the acceleration term, 0.1 of the mean acceleration of
the last 100 ms. Error / sigma on those frames: p50 1.11, p90 2.29, p99 3.27 at 164 ms (a perfect model reads
1.18 / 2.15 / 3.03), the same at 104 and 190 ms.

Consequence worth knowing: **the trigger delay is the largest lever on the whole gun.** Virtual clays, clays
hit, at a trigger delay (command to muzzle) of 220 ms 42 %, 164 ms (now) 62 %, 100 ms 81 %, 60 ms 88 %. The
bench notes say part of the 155 ms is the servo waking from limp. Holding the rest position while the button
is held would remove that part, but it is a change to how the servo is driven, against the standing rule that
no pulse is ever held, and it needs a measurement with the lever on the gun. Not done, not prepared in code:
a decision for the bench.

### Three errors in the timing of a shot (fixed)

1. **Every fire command of the range day went out 7-11 ms later than the solver had planned.** The solver
   decided for "now + the average time a solution takes" (1.4-3.7 ms), and the frame that fires is the dearest
   of all (look-ahead and all: 10-15 ms on the Pi). It now plans for the worst case (`FireSolver.worst_case_work`
   x the measured cost of one solution) and the command waits for the planned instant
   (`clock.sleep_until`); telemetry has `cmd_t_ns` and `fire_at_ns` to check it.
2. **The solver extrapolated the swing from an IMU sample 20-25 ms old** (read at the start of the frame,
   before detection and tracking). While the button is held the IMU is now read again just before the solver
   (`stage_ms.imu2` is what that costs). The two together: 58.7 -> 62.3 % of virtual clays.
3. **Waiting always looked worse than it would be.** The look-ahead ("is a later frame better?") gave later
   frames the uncertainty of an extrapolation from now, which grows with the square of the wait; a later
   frame will have its own fresh look. The look-ahead now uses the uncertainty of a decision made then.

Also: P(hit) is a closed series instead of a 16 x 32-point quadrature (same value to 2e-8, 18x faster here;
it was 0.9 ms a call on the Pi, three or four calls on a firing frame).

### P(hit) that means what it says, and a bar that follows the engagement

**P(hit) was far too pessimistic near the top of a pass.** Frames the solver gave 0.1-0.2 really hit 44 %, and
0.3-0.4 hit 65 % (virtual clays); the range-day frames said the same (0.2-0.4: 7 of 7). The predicted miss is
the true miss plus an error, and the true miss is not just anything: a solution exists because a shooter is
swinging through the target. `fire.aim_prior_deg` (1.0) is that knowledge, as a prior on the true miss; the
predicted miss is shrunk accordingly. A prior is not a measurement, so it only counts as far as the
measurement supports a shot by itself (`fire.aim_prior_evidence` 0.2: below that plain P(hit), the result is
scaled down). Without that guard a track three frames old read 0.36 with nothing known, and going-away
clays in `tbot sim` fell from 4/6 to 0/6. Now, 58 000 virtual-clay frames scored as if fired:

| P(hit) given | 0.05-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.8 |
|---|---|---|---|---|---|---|---|
| really hit | 27 % | 37 % | 46 % | 52 % | 62 % | 72 % | 81 % |

(a little careful at the low end, right at the top). The 81 real frames of the range day with a solution,
replayed: under 0.1: 12 % (57 frames), 0.1-0.3: 31 % (13), 0.3-0.7: 55 % (11).

**The bar** (`tbot/fire/policy.py`, `fire.adaptive`). With the right delay the old bars were far out of
reach: on the real frames of the range day, replayed with 164 ms, the old rule (0.40, 0.30 after 0.7 s, old
P) fires on **1 press of 9**. The bar now starts at `min_hit_probability` (0.40) and is lowered to
- `adaptive_reach` (0.85) x what a perfect aim would be worth right now: a rough swing cannot give 0.40,
- `adaptive_regret` (0.9) x a chance this target already offered and lost,
- the settled bar (0.25) as the track ages from 0.2 to 0.7 s,
and never below `min_hit_probability_floor` (0.20). Clays hit, 1500 virtual clays:

| rule | clays hit | shot at | hits per shot |
|---|---|---|---|
| fixed 0.50 | 38 % | 48 % | 79 % |
| fixed 0.40 | 57 % | 78 % | 73 % |
| 0.40, then 0.30 from 0.7 s | 58 % | 79 % | 73 % |
| fixed 0.30 | 62 % | 86 % | 72 % |
| fixed 0.25 | 64 % | 89 % | 72 % |
| adaptive from 0.40 (as shipped) | 63 % | 87 % | 72 % |
| ... reach only / regret only / relax only | 61 / 61 / 60 % | | |

Read it honestly: the adaptive bar is not better than the best fixed bar, it **is** the best fixed bar without
knowing which one that is. Started at 0.35, 0.40, 0.50 or 0.60 it lands on 63, 63, 62 and 61 %; a fixed bar of
0.40 loses 6 points to it and a fixed 0.50 loses 25. `reach` does most of it. The same change in `tbot sim` (6 seeds x 8 scenarios, Pi timing):
old rule 11 shots, 6 of 54 clays; now 24 shots, **17 of 54** (hit rate per shot 55 -> 71 %).

### Two test benches on real data, and a simulator that repeats

- **`tbot vclay <presses>`: virtual clays through recorded gun motion.** Keeps the real swing (IMU) and the
  real frame times of a press and replaces only the target: a clay built to be where a charge of some moment
  of the press would arrive, flying at clay speed, as fast across the sky as the gun was really swinging
  times a swing-through factor (set from the range day: bore and clay moved against each other at 2.3 / 7 /
  20 deg/s, p10 / p50 / p90). Its detections go through the real tracker and solver, which see the attitude
  only as the live loop does; every shot is judged against the true bore at the TRUE delay. Hundreds of clays
  a minute. `--policies` (rules on the same clays), `--calibration` (is P honest), `--delay-error-ms`,
  `--cmd-late-ms`, `--imu-lag-ms`, `--fps N --age-ms M` (what a faster loop is worth). It is a relative test:
  the clays are built around the path the bore took, so a chance always exists.
- **`tbot rescore <presses>`: the recorded frames through today's software.** Real frames, real IMU, the
  button held when the shooter held it, the gun's lens / mount / load / calibration from the press itself;
  scored like `tbot score` (planned shot against where bore and target really were), `--every` for all frames
  with a solution. 5 s for the nine presses. Range day at 164 ms: **6 shots, 3 planned hits** (old rule: 1
  shot, 0). This is the regression test for anything that touches the decision.
- **`tbot sim` now runs on the gun's time and repeats exactly.** `sim.process_ms` (22) is the Pi's work per
  frame: the simulated loop takes the newest frame each time round (42 of the camera's 100-120 a second),
  the solver runs on a frame ~40 ms old, on a clock that only moves when the simulation moves it. Before, it
  ran at this computer's speed (every frame, 3 ms old: 19 of 54 clays against 17 on the gun's time) and a
  third of the shots changed from run to run. `sim.process_ms = 0` is the old behaviour. `actuator.sim.delay_ms`
  and `latency.sim_cmd_to_break_ms` are 155.
- Tests: `tests/test_policy.py`, `tests/test_vclay.py` (needs the range-day press; the frames are LFS),
  `tests/test_timing.py`.

### What each lever is worth (1500 virtual clays, clays hit; 62.3 % as the software is now)

| | clays hit |
|---|---|
| IMU read once per frame, as on the range day | 60.0 % |
| ... and the commands 9 ms late, as on the range day | 58.7 % |
| trigger delay wrong by 60 ms, as on the range day | 56.6 % |
| loop at 25 frames/s, frames 55 ms old (trees in view) | 54.1 % |
| loop at 40 frames/s, 40 ms old (about the range day) | 63.9 % |
| loop at 60 frames/s, 25 ms old | 67.3 % |
| loop at 100 frames/s, 20 ms old | 70.6 % |
| trigger delay 100 ms instead of 164 | 81.0 % |
| trigger delay 60 ms | 88.5 % |

### Would a rewrite in Rust / C / C++ / Bend help?

Measured, not guessed (range-day telemetry, Pi 3B, button held, a clay in view): 18-25 ms of work a frame, of
which detection 15-20 ms. Inside detection: OpenCV calls (background, threshold, labelling) ~5.6 ms, already
native code; the scenery filter 2.5 ms and the measurement of each candidate ~6 ms (4.3 ms of window
arithmetic + 1.8 ms adaptive moments), in NumPy, where every call costs 15-50 us on this CPU whatever it
computes. Tracker 1.5, attitude 1.3, solver 0.2 (10-15 on the frame that fires, before today), everything else
~2 ms.

- **The garbage collector is not in it.** `gc.freeze()` at start already takes the whole heap out of its
  sight; no full collection runs in the frame loop. The cost is the interpreter's per-call overhead in the
  parts that loop over small arrays.
- **Bend: no.** It is a research language for massively parallel pure functions; it has no ARM camera, GPIO
  or OpenCV story, and no real-time one. Nothing here is that kind of problem.
- **A full rewrite (Rust or C++, libcamera + OpenCV bindings): ~6-8 ms a frame is plausible, so 100 frames/s.**
  Worth at most the table's last loop row: 62 -> 71 % of clays, +8 points. Cost: 30 000 lines of working,
  tested Python, the calibration tools, the recorder, the simulators, and every hour of field debugging so
  far; weeks without a gun that shoots better. It would hurt now.
- **One native kernel gets about half of that for a few percent of the work.** The per-candidate measurement
  (`ClayDetector._measure`'s loop and `_adaptive_moments`, ~250 lines) is 6 ms x 1-3 candidates of every
  frame with a target. In C behind `ctypes` (one file, built by `scripts/pi_setup.sh` with the Pi's own gcc,
  the Python kept as the fallback and as the reference in `tests/test_fast_paths.py`) it is tens of
  microseconds: by that arithmetic ~22 -> ~13 ms a frame, 40 -> 60+ frames/s, about +4-5 points (an estimate
  until it is timed on the Pi). That is the next software step for speed. The scenery filter is the one after it.
- **And both are small beside the trigger delay** (+19 points for the first 60 ms), which no language touches.

### The native measurement kernel (measured on the Pi 2026-11-18)

The step above is in: `tbot/vision/_measure.c` (one file, no Python headers, called through `ctypes` by
`tbot/vision/_native.py`) is the loop of `ClayDetector._measure` and `_adaptive_moments`, statement for
statement. The Python loop is untouched: it runs whenever the library is not built, does not load, is from
another version of the C file, or is switched off (`detector.native = false`, or `TBOT_NATIVE=0` in the
environment), and it is the reference in `tests/test_native_measure.py`.

- **Build:** `scripts/build_native.sh` (`cc -O2 -shared -fPIC -ffp-contract=off`; `clang` builds a DLL on the
  laptop). `scripts/pi_setup.sh` installs `gcc` and calls it. The binary is not in git. `tbot bench` prints
  whether the kernel is in use, or why not.
- **Same detections.** Built scenes (both polarities, edge-on, image border, mask, clutter, colour, tiny
  frames), all eight simulator scenarios and all nine range presses of 2026-11-17, native against Python:
  the same candidates accepted and rejected, area / bounding box / peak contrast equal, positions and
  diameters equal to rounding (~1e-8 relative; the test allows 1e-3 px and 1e-3). They are not bit for bit:
  the C library's `expf` and NumPy's float32 `exp` round differently in the last place.
- **Downstream:** `tbot sim --seeds 6` and `tbot rescore` on the nine presses print the same report with the
  kernel on and off (24 shots / 17 hits; 6 shots / 3 hits). In the full-precision JSON the numbers agree to
  ~1e-8 (the worst is 3e-6, on a minimum over a run), and one fire time of 48 runs differs by 1 ns.
- **One trap worth knowing:** when the Gaussian weight wanders off a faint blob (a speck on the horizon),
  the weighted flux drops under ~2e-16 and `cv2.moments` then returns the raw second moments as the central
  ones; where that sets in depends on the OpenCV build (5.0 sums such an image twice, so half of
  `DBL_EPSILON`). The detector measures it once (`_moments_m00_floor`) and the kernel follows, so such a
  window gives the same junk either way. It is junk: a guard in `_adaptive_moments` would be the real fix.
- **Laptop timing** (AMD64, Python 3.13, the loop only, per candidate handed to it): Python 0.30 ms ->
  native 0.022 ms on the simulator scenarios (x13), 0.38 -> 0.044 ms on the range presses (x9). `tbot bench`
  (skeet_crossing, no overlay): detect 3.68 -> 2.49 ms median, 5.45 -> 3.52 ms p95.
- **Pi 3B (2026-11-18, OpenCV 4.10, NumPy 2.2):** `tests/test_native_measure.py` passes there too. The whole
  pipeline over three recorded range presses (299 armed frames, replay with their IMU, scenery filter on,
  warm page cache), kernel off -> on: frame p50 19.4 -> 15.8 ms, p90 30.7 -> 19.4 ms; detector components
  stage p50 10.4 -> 6.3 ms, p90 19.5 -> 7.5 ms. `tbot bench` on the synthetic crossing (a clean sky, few
  candidates) moves less: total p95 11.0 -> 8.9 ms. (A first replay run read 8 ms more in `downscale`:
  that is the recorded frames coming off the SD card on first touch, not the kernel.)

## Night before the second range day (2026-11-18): camera recovery, restarts, sensor load, graphics

Done on the gun (Pi 3B, real camera and IMU, dry run, dark room) unless it says otherwise.

### The camera comes back by itself

Two layers, both automatic, both logged:
1. **A stalled or failed camera is reopened inside the sensor process** (`RecoveringSource`, as before): no
   frame for `camera.stall_s` (0.5 s), or the capture thread fails -> close, reopen, back off up to 5 s. A
   frame or two missing does nothing at all: the tracker just sees a longer gap.
2. **New: the whole sensor process is replaced** (`camera/proc.py`) when it dies, when it stops talking (it
   sends a heartbeat every 0.25 s; 3 s of silence = wedged in a driver call, killed at once), or when it is
   alive but has delivered no frame for `camera.respawn_s` (8 s: the in-process reopen is not working). A
   fresh process starts libcamera from nothing, which before meant a service restart and a 15-20 s cold
   start of the pipeline. Measured on the gun: sensor process killed -> streaming again in **7 s**; frozen
   (SIGSTOP) -> noticed after 3 s, streaming again ~7-10 s after the freeze. Respawns back off 1, 2, 4, 8,
   15 s while they keep failing; the pipeline keeps running (stream page up, status line every second).

While the camera is away the pipeline's watchdog keeps the actuator safe (`watchdog: no processed frame`),
the first frame afterwards resets the tracks (`camera restarted (N so far): tracks reset`), and the status
line shows `CAMERA RESTARTS N`. Telemetry: `cam_restarts`, and **`lost`** = frames the sensor sent that never
arrived (gaps in the sensor timestamps; `dropped` also counts frames merely skipped because a newer one was
waiting, e.g. 85 a second while the button is up, so it never said whether anything was wrong).

**No shot on a stale IMU** (`fire.max_imu_age_ms` = 100): when the IMU stream stops (I2C fault, the sensor
process being replaced) the attitude and the swing the solver extrapolates are frozen; the decision is then
`IMU stale (N ms)`. On the range presses replayed armed: p50 7 ms, max 21 ms, so it never blocks a real shot.

### The sensor process was losing frames (fixed)

At 100 fps the sensor process used **107 %** of a core (camera 65 %, IMU polling ~30 %, auto-exposure ~15 %:
three threads behind one interpreter lock) and lost ~15 frames every 20 s. Now:
- IMU: 2 I2C transactions a poll instead of 4 (the FIFO count as one 2-byte read; the overflow flag only read
  when the FIFO is nearly full, the only time it can be set) and the FIFO bytes copied straight out of the
  ctypes buffer (iterating smbus2's message cost 0.6 ms a poll): 3.0 -> 2.05 ms a poll; the rest is I2C bus
  time (400 kHz already).
- Auto-exposure meters every 2nd frame (`camera.ae_every = 2`, 50 Hz control).
- The Y plane is copied once, straight out of the mapped camera buffer (`make_array` copied the whole YUV
  buffer first).
- **86 %** of a core (was 107 %). Lost frames in service, 40 minutes logged: ~2-3 a minute (0.05 %); one
  5-minute stretch earlier in the night lost ~2 a second for no reason found (no throttling, 66 C, nothing
  else running) and stopped with the service restart. Frames are still lost when something
  else loads the Pi (a few while `top` and two SSH sessions ran): don't run tools on the gun while shooting.

### Restarts and power cycles

- **Power-up to first frame: 48 -> 38.5 s** (measured: reboot to the first status line). The service no longer
  waits for `multi-user.target` (~5 s), and `tbot run` spawns the sensor process before importing the
  pipeline, so the two load OpenCV in parallel (~4 s; `import cv2` alone is 5 s on the Pi 3B, apt's build
  links half the system). What is left is the OS's own boot (NetworkManager, cloud-init) before the service
  starts at ~20 s.
- `systemctl stop` / `tbotctl stop` is clean now (it printed five Python tracebacks every time:
  multiprocessing semaphores racing their cleanup; the stop request goes over the pipe now, and
  `KillMode=mixed` lets the pipeline stop its sensor process itself).
- **The log survives a power cycle:** `logs/tbot-run.log` (5 MB x 4, rotated) gets every log line and one
  status line every 10 s. The journal (`tbotctl log`) is in RAM and starts empty after every boot.
- The debug stream failing to start (port taken) no longer stops the gun.
- `scripts/deploy.sh push` works from Windows without rsync (tar over scp, LF line endings, uncommitted edits
  included unless `TBOT_DEPLOY_REV=HEAD`), precompiles the Python and builds the native kernel on the Pi.
  After pushing a changed unit file: `tbotctl install`.

### Graphics

- **Overlay** (stream page and press clips), restyled: status band (ARMED/SAFE, LIVE/DRY, NO IMU, health and
  exposure warnings), decision band with a P(hit) meter and the firing bar, corner brackets for detections,
  grey x for scenery, rings for tracks (dashed while tentative), the lead point joined to the predicted bore
  with the pattern circle, an edge arrow when the lead point is off screen, a FIRE flash. Draws a busy frame
  in 7-8.5 ms on the Pi (was 31-39 ms); old recordings still draw. Samples: `docs/overlay_samples/`
  (`after_pi_opencv4_*.png` are drawn on the Pi itself).
- **Simulator graphics:** `tbot simviz` (or `tbot sim --render DIR`): per run an MP4 (camera view with the
  overlay, plan and elevation views of the true and tracked paths, intercept close-up to scale, the shot in
  flight, breaks and fragments, a timeline) and a summary PNG, plus a contact sheet. Watching a run changes
  nothing in it (`tbot sim` prints the same report; a test checks it). Samples: `docs/sim_samples/` (the
  MP4s are not in git).

### Still open

- Armed frame rate on the range presses is ~60 fps (15.8 ms p50), not 100: what is left is the scenery
  filter (2.3 ms, ~30 small NumPy calls: the next candidate for C), the rest of the detector (bg 1.3,
  threshold 2.1, components bookkeeping ~2 ms), attitude 1.1, tracker 1.3-1.9 ms.
- Not seen outdoors: the recovery has only met a killed and a frozen process, not a real CSI fault.

## Rev AH trigger set, re-zero and trigger delay (2026-11-18, day)

On the gun (Pi `config/local.toml`, `calib/`), all bench, gun unloaded:
- **Camera zero**: the pod was dropped lens-first; refocused, and the boresight had moved 0.84 deg. Re-solved
  (pitch -1.379, yaw -2.334) and checked at 35 ft: 0.027 deg. Data and method: `calib_data/bs20ft_2026-11-18/README.md`.
- **Servo** (new MG996R, rev AH cam inside the guard, horn fitted at 1500 us): a LOWER pulse pulls the trigger.
  Touch 1320 us, break ~1095 us (1100 does not break). `rest_us 1340`, `fire_us 1075`, `min_us 1070`,
  `max_us 1500`, `hold_ms 400` (with 150 ms the servo was sent back before the cam pulled through).
- **Trigger delay**: `tbot latency-mic`, 10 trials: click (hammer fall) 197.7 +/- 1.4 ms after the command, so
  `cmd_to_break` **191.7 ms** (was 155 with the old servo and lever). Confirm with `tbot latency-recoil` on the
  first live shots.
- **`tbot latency-mic` fix**: it wrote the click time as `cmd_to_break_ms`, and the budget adds
  `gun.lock_time_ms` on top, so the lock time was counted twice (6 ms late). It now writes
  `cmd_to_break_ms = cmd_to_bang_ms - gun.lock_time_ms` and keeps `cmd_to_bang_ms` beside it. Today's file was
  corrected by hand the same way before the fix.
