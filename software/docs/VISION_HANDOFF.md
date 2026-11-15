# Vision handoff (detector, tracker, fire solver)

The computer-vision side is owned by a separate developer. This page is the boundary: what the hardware/integration
side guarantees, where vision code plugs in, how to measure it, and what is known to be unfinished. Binding API
details are in [ARCHITECTURE.md](ARCHITECTURE.md); keep those signatures and `tbot.app.run.Pipeline` keeps working.

## What the integration side guarantees
| Item | Guarantee |
|---|---|
| `Frame` | Upright image (the 180 deg board rotation is undone), mono `uint8` from the OV9281 (BGR only from webcams/synthetic colour). `t_exposure_ns` = mid-exposure on `time.monotonic_ns()`, corrected by `calib/latency.json` `camera.timestamp_offset_ns` (measured by `tbot latency-camera`). |
| Attitude | `R_wg(t_ns)` / `omega_g(t_ns)` from the MPU-6050 (FIFO, regression-timestamped, `calib/imu.json` bias + axis map + camera-IMU time offset from `tbot calib-imu sync`). Measured on synthetic data: median 0.07-0.11 deg vs truth after gravity init (~3 deg transient in the first frames). |
| Geometry | `tbot.optics.CameraModel` (calibrated fisheye or nominal prior) and `MountModel` (boresight + patterning POI, camera-to-muzzle parallax). Barrel mask from geometry AND `calib/mask.png`. |
| Latency | `calib.budget` (`LatencyBudget`): decision -> muzzle delay and jitter per actuator, measured by `tbot latency-actuator`; `tbot latency-report` prints it. |
| Safety | The solver only *decides*. The actuator layer refuses to move unless the arming button is held, aborts on release, one shot per press, watchdog, dry run unless `fire.live`. Vision code never touches GPIO. |

## Plug-in points
- `tbot/vision/detector.py` `ClayDetector.detect(frame) -> list[Detection]`. `Detection.diameter_px` must be the
  observed, unthresholded 4-sigma major axis (clay convolved with blur); the tracker subtracts `detector.psf_px`.
- `tbot/vision/tracker.py` `ClayTracker` (`update`, `predict`, `predict_dir`, `best`, `reset`, `tracks`).
- `tbot/fire/solver.py` `FireSolver.evaluate(...) -> FireDecision` (`p_hit` gate, `min_hit_probability`, "now is
  the best moment" over `horizon_frames`). `FireDecision.reason` must contain "not armed" when disarmed.

## Data and measurement
- Record on the gun: `tbot record --seconds 30` (continuous) or `tbot run --record armed` (ring buffer dumped around
  each arming press). Copy `sessions/<name>/` to a laptop and replay with `--source replay:<dir>`.
- Synthetic closed loop: `tbot sim [--scenarios ...] [--seeds N] [--out result.json]` reports detection rate, time to
  confirmed track, lead error at +100 ms, fire count and the TRUE shot outcome (miss vs pattern) per scenario
  (`skeet_crossing`, `skeet_incoming`, `skeet_outgoing`, `trap_straightaway`, `static_gun`, `no_clay`), and for
  the pairs (`skeet_doubles`, `trap_doubles`) clays broken / thrown, debris shots, shots per run and launch-to-break time.
- Pi 3B time budget: `tbot bench` (per stage p50/p95 vs the frame period; writes `calib/bench.json`).
- Detector tuning: `tbot calib-detector` (`--gui`, `--report`, `--auto-mask`, `--psf --range-m`).

## Known state and open problems (measured, 640x400, 120 fps, defaults)
1. **Mono sensor.** The OV9281 has no colour; "orange" is local contrast against the sky. See
   [OPTICS.md](OPTICS.md) for the optional orange/red long-pass filter idea.
2. **Clay size / range.** Fixed bias: size is now measured with adaptive moments on an unthresholded window.
   Recovered diameter after PSF subtraction is within ~7 % for 2-10 px discs (pixel-box blur), was -25 % before
   (which made range +60 % and the lead wrong). `psf_px` (default 1.2) must be calibrated on the real lens:
   `tbot calib-detector --psf --range-m R` with a clay at a measured range. Range remains the dominant lead error:
   the shot ToF slope is ~3.3 ms/m near 20 m, so a 2 m range error at a 1 rad/s crossing is ~7 mrad.
3. **Tracker lag (main blocker).** The shipped tracker is a constant-acceleration Kalman filter on world az/el.
   On an analytic 20 m / 20 m/s crossing the +150 ms prediction error is median ~0.25 deg, max 0.51 deg, and still
   0.32 deg with noise-free detections (model lag; ideal CA truncation would be 0.06 deg). `jerk_psd_deg = 2000` is
   near optimal for this model (sweep: 600 -> 0.55, 200 -> 0.64, 20 -> 1.08 deg max). Its predicted sigma at the
   shot-arrival horizon (~0.2-0.3 s) is 15-25 mrad, larger than the 16.6 mrad pattern radius, so **with defaults
   `p_hit` stays below `min_hit_probability = 0.5` on synthetic crossings and the system does not fire.** The
   solver logic itself is covered by `tests/test_solver.py` with tracker overrides that make sigma honest.
4. **Prototype that was measured and removed:** a scale-free bearings-only model `q(tau) = u0 + w tau + b tau^2/2`,
   `|u0| = 1`, 0.4 s sliding window, Gauss-Newton (constant-velocity target bearings are range-invariant). At +250 ms
   over four scenarios: median 4.8 mrad, p95 18.7 mrad, predicted sigma ~2x the actual RMS (conservative), update
   p95 1.5 ms/track on a laptop (Pi 3B ~5x slower). The instantaneous fit error was only 0.3-1.8 mrad, so the
   error is extrapolation: a longer window or explicit drag/lift terms are the next levers.
5. **Measurement noise is over-stated.** Real detector centroid residuals were 0.12-0.22 px while
   `tracker.meas_sigma_px = 0.7`, so the tracker distrusts good measurements.
6. **`trap_straightaway` detection rate is low**: detections on 98 of 301 frames and no track reached 20 hits while
   the clay was visible at +250 ms. A going-away clay is small and nearly stationary in the image; investigate the
   detector thresholds/background kernel for it.

## Constraints to keep
- Python >= 3.11, numpy 1.24 and 2.x, OpenCV 4.6 (Raspberry Pi OS apt) and 5.x; no `cv2.aruco`; no scipy at runtime.
- Pi 3B: the whole pipeline must fit the frame period at the configured fps (`tbot bench` verdict).
- Run the suite on both environments before handing back (`python3 -m pytest` from `software/`).
