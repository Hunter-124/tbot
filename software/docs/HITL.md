# Hardware-in-the-loop bench

**Purpose.** Measure the photon → trigger-break chain on the real rig, and exercise detection, tracking and
lead end to end against targets with known screen positions and timing. This is a bench measurement, not a
live-ammunition hit test. Dry run is the default; a live bench run drives the actuator only while the operator
holds the arming button. **Gun UNLOADED, chamber and magazine checked, muzzle pointed in a safe direction.**
Release the button to stop. The default software interlock permits one shot per arming press; release and press
again for each trial. There is no hardware supply cut: open the battery lever connector to make the actuator
safe. A series hardware interlock is recommended but not installed ([PROTOTYPING.md](PROTOTYPING.md)).

## Signal chain and clocks

```mermaid
sequenceDiagram
    participant Page as Browser / monitor
    participant Rig as Rig monotonic clock / pipeline
    participant Cam as Camera
    participant Act as Actuator / trigger
    participant GPIO as Break-sense GPIO
    participant Mic as USB mic / sample clock
    Page->>Rig: NTP-style sync ping (performance.now send/receive)
    Rig-->>Page: rig_ms; fit low-RTT offset and drift
    Rig-->>Page: scene program (rig_ms, device pixels)
    Page->>Page: rAF draw and frame log (synced rig_ms, item IDs)
    Page->>Cam: display scan-out → photons → exposure
    Cam->>Rig: mid-exposure and capture timestamps, frame
    Rig->>Rig: detect → track → solver decision (forecast command time)
    Rig->>Act: actuator.fire() call timestamp
    Act->>GPIO: trigger break → GPIO edge timestamp
    Act->>Mic: lock time → dry-fire click + sound flight
    Mic->>Rig: sample stream; fit sample index to monotonic arrival time
    Page-->>Rig: batched rAF frame log and status
```

The page's `requestAnimationFrame` (rAF) timestamps are on the browser's clock, not photon timestamps.
The page uses ping-pong midpoint samples, preferring low RTT and fitting clock drift, to translate them to rig
monotonic time. Minimum RTT/2 bounds the sync error only under the usual nonnegative-delay assumption; path
asymmetry and browser scheduling still matter. Camera exposure midpoint and capture timestamps, actuator call
and GPIO edges use the rig clock. Audio samples have their own oscillator; their sample-index clock is fitted
to rig-clock arrivals. In simulation these components are simulated, but processing runs in real time.

## Bench setup

1. Complete the applicable steps in [CALIBRATION.md](CALIBRATION.md), especially intrinsics, boresight,
   detector and `tbot latency-camera` timestamp offset. Aim the camera and bore line at a flat monitor roughly
   1–2 m away; the camera should see most of the screen. Aim the bore toward the side where sweep targets
   exit to give them a long runway. Clamp the rig securely and keep camera, gun and monitor fixed after fitting
   geometry. The bore point at the chosen virtual range must be on screen.
2. Use full monitor brightness to avoid PWM dimming flicker under short exposures; keep room light steady and
   avoid glare and clipping. Focus for the intended field use first. The nominal 2.8 mm F2.2 lens has an
   approximate hyperfocal distance of 0.6 m [INFERENCE: depends on circle of confusion and the actual lens],
   but verify sharpness at the monitor and do not assume the bench PSF equals the field PSF.
3. If the rig is the Pi, connect it by **wired Ethernet** to the laptop driving the monitor. Run the command on
   the rig, open its printed URL on that monitor, click **Start** for fullscreen, leave the tab visible and
   unobscured, and allow clock sync to settle. `F` toggles fullscreen. The rig waits until the page is connected
   and fullscreen before measuring; `--allow-windowed` overrides this for headless or remote browsers. A local
   browser may also drive a monitor attached to the rig. Port 8090 is the default; use a trusted network.
4. For an unloaded live test fit the actuator and hold/release the arming button as prompted. Optionally wire
   the normally-open break contact to `gpio.break_sense_pin` (default BCM 22) and GND, closing at trigger break.
   Connect a USB microphone to the **rig**, not the laptop; the Pi 3B has no audio input. Measure its distance
   from the trigger group. Check microphone levels and eliminate other clicks or servo interference.

## Quick start: first 10 minutes

Run these from `software/`. First check the simulated bench without hardware:

```bash
python3 -m tbot.cli hitl run --sim --live --trials 4 --display-trials 16 --out /tmp/hitl-doc2
```

For the real rig, start the dry run, then open the printed URL on the monitor and click **Start** for
fullscreen. The rig waits for the page; leave it visible and unobscured.

```bash
python3 -m tbot.cli hitl run
```

For a live measurement, connect the microphone to the rig, check the gun is **unloaded**, and hold and
release the arming button as prompted. Replace the example ALSA device and distance with your own:

```bash
python3 -m tbot.cli hitl run --live --mic plughw:1,0 --mic-distance-m 0.3
```

Both commands print the run directory and the path to its self-contained `report.html`; open that file in a
browser (copy it to the laptop first if necessary). For the simulated run, for example:

```bash
xdg-open /tmp/hitl-doc2/report.html
```

Use `python3 -m tbot.cli hitl mics` to list devices. Calibrate `--mic-latency-ms` as below before treating mic
timing as absolute; `--break-sense` can be used alone or alongside `--mic`. The default tests are
`geometry,display,flash,sweep`; add `detect` with `--tests geometry,display,flash,sweep,detect`. Reuse saved
geometry only while the setup is unchanged, for example with `--tests flash,sweep`. Use `--no-write` to leave
`calib/hitl.json` unchanged; simulation never writes calibration. `python3 -m tbot.cli hitl analyze RUN_DIR`
regenerates the reports. Use `hitl run -h`, `hitl analyze -h`, and `hitl mics -h` for complete flag lists.

## Tests and criteria

| Test | Stimulus and measurement | Report criteria / interpretation |
|---|---|---|
| geometry | One disc at a time in a 5 × 4 grid; detect centroids and fit screen pixels → undistorted camera rays by homography. Distorted camera-pixel residuals expose mapping error. At least six markers must be seen; outliers may be removed. | RMS < 0.5 px PASS, 0.5–<1.5 WARN, ≥1.5 FAIL. Check maximum error and coverage too. A saved map validates dimensions, **not** that the setup stayed fixed. |
| display | Randomly timed grey square steps at up to three rows, both transitions; probe camera brightness, estimate photon time and fit `d0 + scanout × row/H` separately for rise and fall. | INFO model; WARN if fewer than five partial rise frames, brightness is saturated (≥250 DN) or contrast <8 DN. Check residuals, samples and bracket uncertainty; a model is not proof of an accurate clock offset. |
| flash | Clay-sized static disc at the bore point, random appearance; log first exposure/detection, decision, command, optional break edge and click. | Every trial should yield a decision; any missing decision is FAIL. Decision→command >1 ms is WARN. Measured command→break should agree with budget within max(5 ms, 10%); mic compares command→click with command→break + lock under the same tolerance. |
| sweep | Clay-sized disc crosses the bore at requested speeds and selected directions. Compute virtual shot miss against known trajectory, range, shot model, and predicted exit; live break or click permits an actual-timing estimate. | All decisions at each speed virtually hit → PASS; some hit → WARN; none → FAIL, except all-short-runway misses warn. No decision is FAIL. Look at miss, cross-track, range bias, runway and exit-late, not just the label. |
| detect | Blank screen, then static clay-sized discs at multiple positions and simulated ranges; measure detection rate, centroid bias/RMS, observed diameter, size-derived range and extra detections per frame. | ≥90% detected at a range is PASS; lower is WARN. Inspect false detections, centroid error and size/range bias as well. |

Page minimum RTT ≤2 ms and late-frame fraction <1% are PASS; otherwise WARN. Skipped camera frames warn.
A `--sim` report additionally checks recovered values against its injected constants. PASS/WARN/FAIL apply to
these checks, not to a guarantee of real-world shooting performance.

### Display transition method

For a transition within an exposure, normalised brightness gives the fraction of exposure after the step;
photon time is exposure end minus that fraction times exposure duration. If there is no usable partial frame,
the estimate is the midpoint of the bracket between the last fully old and first fully new exposures, with
half-bracket uncertainty. Both methods assume a global shutter and sufficiently linear, unsaturated brightness;
panel response, flicker and sync errors remain. Partial timing is not a separate calibrated photodiode.

At 120 fps, the usual 500 µs exposure occupies little of the frame period. When the source supports runtime
exposure change the display test defaults to **0.8 frame period**, then restores operational exposure: this
makes partial frames much more likely at any phase. A 180 Hz monitor and 120 fps camera are commensurate;
short-exposure frame-bracket timings can remain phase-locked and systematically biased rather than averaging
away. `--display-exposure-us 0` keeps the operational exposure; sources that cannot switch use it anyway.
Inspect the partial count and uncertainty, and rerun with a suitable exposure if needed.

### Flash interval definitions

The table names below match the printed `FLASH` rows. Times are milliseconds unless marked otherwise; `n` is
sample count. `draw` is the first logged rAF drawing the item, `photon` is draw plus the row-specific display
model; the first exposure is the first logged frame whose exposure end is at or after that photon estimate.
The decision `now` is the solver's forecast command timestamp: the pipeline adds an EMA of solve duration
to solve start and evaluates at that future time. A live call waits until `now` if solving finishes early,
but can occur later on an overrun. `cmd_t_ns` records the actual attempted call; `decide_ahead_ms` records the
forecast. In dry mode HITL infers command time from decision `now` plus solve duration, not a physical call;
this is a reporting estimate and can overstate decision→command now that `now` already includes the forecast.
Detection uses a spatial gate around the mapped target.

| Printed row | Exact endpoints / meaning |
|---|---|
| draw (rAF) → photon [display test model] | First rAF drawing target → modelled photon at its screen row. |
| photon → end of first exposure (0..1 frame) | Modelled photon → end of first eligible exposure; skipped frames can extend it. |
| exposure end → frame in user space | That exposure's midpoint + half exposure → frame capture/dequeue timestamp. |
| user space → pipeline picked it up | Capture timestamp → estimated read end (decision `now` less IMU, attitude, detection and tracking stage times). |
| detector processing | Detector stage duration on the first frame with a matching detection. |
| frames seen but not detected [frames] | First matched detection frame index minus first eligible exposure frame index. |
| photon → first detection done | Modelled photon → estimated read end of first matching frame + detector stage duration. |
| detection → decision frame [frames] | Decision frame index minus first matched detection frame index. |
| detection → decision frame (exposure to exposure) | First matched detection frame's exposure midpoint → decision frame's exposure midpoint. |
| decision frame age at 'now' (mid-exposure → now) | Decision frame's exposure midpoint → solver `now`. |
| photon → decision ('now') | Modelled photon → solver `now`. |
| decision 'now' → actuator command | Solver `now` → timestamp of accepted `fire()` call, or inferred dry-run command. |
| photon → actuator command | Modelled photon → command timestamp. |
| command → trigger break [break-sense] | Command timestamp → break contact edge. |
| command → dry-fire click [mic] | Command timestamp → detected click, corrected for capture latency and sound flight. |
| break → click (lock time + mic error) | Break edge → corrected click; includes gun lock time and remaining mic/onset error. |
| photon → break | Modelled photon → break edge. |
| photon → click (END TO END) | Modelled photon → corrected click. |
| draw → click (camera-independent end to end) | First logged rAF → corrected click. |
| tracker range at decision [m] | Solver's range estimate at the decision, not a time interval. |

The raw/report JSON also contains draw→command and draw→break when available. Intervals overlap: do not sum
all rows. `draw → click` and `draw → break` do not depend on camera timestamps, though they still depend on
page sync and, for click, mic calibration. Splitting draw→click into display and photon→click **does** depend
on camera timestamp offset: a camera timestamp error appears in the display latency and shifts the split.
Measure the offset with `tbot latency-camera` before interpreting absolute photon-based intervals.

## Microphone and break timing

`tbot hitl mics` lists ALSA devices (`arecord`, from `alsa-utils`); `--device` also performs a level check.
Capture is mono 48 kHz signed PCM. The sample clock fit uses the lower envelope of chunk arrival times to
estimate rate/drift and map sample indices to rig monotonic time. It does not remove fixed capture latency.
The onset detector finds a strong high-frequency transient relative to the pre-command noise floor; inspect the
saved WAV if servo noise or other impacts win. Overruns lose samples and invalidate the clock fit.

Set `--mic-distance-m` to the measured trigger-to-mic distance: analysis subtracts distance / 343 m/s from the
click time (0.3 m is about 0.87 ms). To calibrate `--mic-latency-ms`, collect repeated live trials with both
`--mic` and `--break-sense`, the distance correction set, and initial mic latency zero. The mean reported
break→click equals actual gun lock time **plus residual microphone/onset error**. If an independently measured
lock time is available, subtract it from that mean to estimate the mic correction, then pass that value on
subsequent runs. Without an independently known lock time the two are confounded: do not call their difference
a measured capture latency. Repeat after changing device, capture path or position.

## Outputs and persistence

| Path | Content |
|---|---|
| `sessions/hitl-<date>/raw.json.gz` | Raw page log, per-frame pipeline data, trial records, configuration and calibration snapshot; simulation uses an `hitl-sim-` prefix. `--out` selects another directory. |
| `report.txt`, `report.json`, `report.html` | Human-readable, structured and self-contained browser reports in the run directory; HTML includes a latency waterfall, display-latency fit and sweep accuracy plot. Both `run` and `analyze` write them. |
| `audio/*.wav` | Captured live-trial clips when a microphone clip was available. |
| `calib/hitl.json` | Merged screen homography and fit residuals, screen/camera dimensions, display rise/fall models and exposure, page dimensions/refresh/user agent, creation time. See [ARCHITECTURE.md](ARCHITECTURE.md). |
| `calib/latency.json` | Only updated with explicit `--write-latency` on a non-sim run with flash timing: break-sense command→break statistics take precedence; otherwise corrected command→click minus configured `gun.lock_time_ms` is written, with a note that lock was not measured. With no measurement it is untouched. Other entries are merged, not replaced wholesale. |

**Example, simulated, not a hardware measurement.** Trimmed verbatim from the quick-start `--sim --live`
run above; timings vary with host load.

```text
screen map: 19 markers, rms 0.02 px, max 0.04 px
display: rise 12.28 ms + 3.73 ms*row/H (16 samples, 11 partial, resid 0.41 ms); fall 12.28 ms

FLASH (static clay at the bore, 20 m): 4/4 decisions
  END-TO-END (ms; mean +/- std, p95)   solver fire delay: 109.0 ms
    photon -> decision   199.99 +/- 5.98 ms, p95 205.04 ms
    photon -> command    201.83 +/- 5.52 ms, p95 206.28 ms
    command -> break     98.16 +/- 1.06 ms, p95 99.11 ms
    command -> click     104.22 +/- 1.05 ms, p95 105.16 ms
    photon -> click      306.05 +/- 5.98 ms, p95 311.10 ms
  stage                                                  unit     mean      std      p50      p95    n
  exposure end -> frame in user space                      ms     6.36     0.26     6.30     6.66    4
  detector processing                                      ms     1.33     0.18     1.42     1.44    4
  command -> trigger break [break-sense]                   ms    98.16     1.06    98.14    99.11    4
  break -> click (lock time + mic error)                   ms     6.06     0.01     6.06     6.07    4

SWEEP (crossing clay through the bore, 20 m): 4/4 decisions
   speed    hits    miss pattern    late   cross exit late pred miss range err
   deg/s   count     deg     deg      ms     deg        ms       deg         m
    20.0     2/2    0.30    0.95   -14.2    0.07      +0.0      0.30      -1.7
    40.0     2/2    0.27    0.95    +6.3    0.07      +0.0      0.27      -2.7

CHECKS
PASS  screen map: rms 0.02 px over 19 markers (large = intrinsics or focus)
WARN  camera: 10 frames skipped by the pipeline (processing slower than the camera)
PASS  actuator: cmd->break 98.2 +- 1.1 ms vs budget 100.0 ms
PASS  sweep 20 deg/s: 2/2 virtual hits, miss 0.30 deg (pattern 0.95 deg), late -14.2 ms
PASS  sweep 40 deg/s: 2/2 virtual hits, miss 0.27 deg (pattern 0.95 deg), late +6.3 ms

SIM SELF-CHECK (analysis vs injected constants)
PASS  display latency (top row) ms: measured 12.28 vs injected 12 (tol 1)
PASS  display scan-out ms: measured 3.73 vs injected 4 (tol 1.5)
PASS  exposure end -> user ms: measured 6.36 vs injected 6 (tol 1.6)
PASS  break -> click ms (lock time): measured 6.06 vs injected 6 (tol 0.5)
PASS  cmd -> break ms (sim delay - solve): measured 98.16 vs injected 98.1574 (tol 1)
PASS  sweep virtual hits: 4/4 (a matched latency model must hit nearly all)
```

## Limits and troubleshooting

- **Sweep runway.** Available angular runway from the starting edge to the bore must be at least
  `speed_deg_per_s × (fire_delay_s + shot_time_of_flight_s + (min_track_hits + 20) × frame_period_s)`.
  The estimate uses local screen-pixel angular scale and a margin for the disc. The CLI defaults to
  `--speeds-dps 20,40` and `--sweep-dir auto`: for each speed it selects every direction with sufficient
  runway, or the longer direction with a warning if neither suffices. `--sweep-dir both`, `ltr`, or `rtl`
  forces the chosen direction(s); selected directions and runway are logged. Move the monitor closer, aim the
  bore toward the target's exit side, or lower the speed if runway is short.
- **Bench defocus and ranging.** The screen is physically near even when the disc subtends the size of a distant
  clay. Defocus can inflate measured diameter and bias size-based range. Detect reports a pooled and per-range
  bench PSF fit `sqrt(max(mean(observed_diameter² − geometric_diameter²), 0))`; compare ranges/positions and
  use the fitted `detector.psf_px` as a bench setting when appropriate. Do not silently substitute it for a
  separately measured field PSF. Monitor pixels, anti-aliasing and contrast also affect the fit.
- **Moving target truth.** The page samples position at each rAF and holds it until the next refresh. Sweep
  analysis approximates the displayed trajectory by the programmed linear path delayed by row-specific display
  latency plus half a refresh period (centred sample-and-hold), not a continuously moving physical clay. It
  computes virtual miss at shot arrival including muzzle parallax, pellet flight and drop; it is a model, not a
  measured impact. Without a display model truth uses zero display delay and the report notes that limitation.
- **Clock and camera.** Display latency includes camera timestamp error; verify with the LED test. Laptop webcams
  may have rolling shutters, auto-exposure, opaque buffering and uncalibrated timestamps: useful for development,
  not a substitute for the global-shutter rig's absolute timing.

| Symptom | What to check |
|---|---|
| Page never connects or draws | Open the printed rig URL on the monitor, click Start, keep the tab visible; check Ethernet, port and firewall. |
| Marker missing / geometry RMS high | Camera sees most of screen, bore on screen, steady focus, correct intrinsics and detector polarity/contrast; avoid glare, clipping and bezel. Rerun geometry after movement. |
| No display partial frames / implausible delay | Enable runtime exposure switching, increase display exposure toward a frame period, reduce gain to avoid saturation, check sync RTT and camera timestamp offset. |
| No flash or sweep decision | Check detector, tracker, bore and virtual range; inspect per-trial reasons and skipped frames. Use `tbot explain --source replay:sessions/your-session` on a suitable recorded session, or a synthetic scenario, to inspect the funnel, reasons and sigma budget; it is not a direct HITL report analyser. For sweep, check runway and lower speed. |
| No break edge | Check contact closes to GND at actual break, configured GPIO, fixture and actuator direction; gun unloaded and arming held. |
| No or wrong click | Check `tbot hitl mics`, device and level, acoustic distance, capture latency, WAV and overruns; remove other impacts. |
| Reused results no longer match | Refit geometry after moving camera/monitor or changing intrinsics/resolution; rerun display after refresh, monitor, exposure or timing changes. |

## Next: IMU verification

**Proposals, not implemented bench hardware.** Verify gyro bias, scale, axis map and cross-axis sensitivity;
accelerometer bias/scale and gravity response; camera–IMU time offset; attitude-estimator drift; and `R_wg`
during representative swings up to roughly 200 deg/s [INFERENCE: target test envelope, not a measured rig
limit]. The static monitor bench alone does not validate dynamic gun attitude or a real clay trajectory.

1. **No new hardware:** repeat static bias and guided axis prompts already in `tbot calib-imu`; compare warm-up
   and repeatability. Put a static high-texture pattern on the monitor, rotate the camera by hand, and extend
   `calib-imu sync`'s existing optical-flow/gyro cross-correlation to compare optical rotation and gyro scale
   as well as time offset. This monitor-pattern measurement is a proposed extension, not an existing HITL test.
2. **Single-axis rate table:** propose a Pi-driven servo or stepper table with encoder feedback, or camera
   observation of a fiducial as independent angular ground truth. Test both signs and multiple rates up to
   the chosen 200 deg/s envelope, with known axis alignment, repeatability and timestamp uncertainty.
3. **Dynamic attitude:** propose a pendulum or turntable with independent pose/time reference; compare `R_wg`
   and drift during accelerations, reversals and multi-axis motion. Separate gravity/linear-acceleration effects,
   camera/IMU synchronisation and mechanical flex before claiming dynamic accuracy.
