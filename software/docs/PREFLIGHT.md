# Optimise and debug before first assembly

Run from `software/` after installing TBOT as described in the [README](../README.md). These offline and simulated results are not evidence of safe real-world operation. For physical testing, keep the gun **unloaded**, chamber and magazine checked, and muzzle safe. The arming interlock is software only; there is no hardware supply cut. Open the battery lever connector to make the actuator safe. See [HITL.md](HITL.md) and [CALIBRATION.md](CALIBRATION.md).

## Ordered workflow

1. **Check configuration and layout offline.**

   ```bash
   tbot doctor
   ```

   Fix every FAIL before proceeding; inspect WARNs and missing calibrations rather than treating inferred defaults as measured. This does not replace `tbot hwcheck` on the assembled Pi.

2. **Prioritise measurements.**

   ```bash
   tbot sensitivity
   ```

   Read the isolated P(hit) ≥ 0.90 tolerance table for the geometry and speed of interest, then the what-to-measure-first ranking. Tolerances vary one error at a time with tracker noise, not all errors together. This run's ranking is model-dependent, not measured hardware performance:

   ```text
   What to measure first (mean fraction of pattern for stated representative errors):
     1. range error +/-20%: 103.3% (24 cases)
     2. latency error 10 ms: 53.6% (24 cases)
     3. PWM jitter 1 sigma: 30.9% (24 cases)
     4. tracker noise 1 sigma: 9.0% (24 cases)
     5. camera timestamp residual 1 ms: 5.4% (24 cases)
   ```

3. **Score scenarios and explain a decision funnel.**

   ```bash
   tbot sim --seeds 3 && tbot explain --source synthetic:skeet_crossing --seed 1
   ```

   Inspect detection, tracking, false shots, hits and latency across seeds; then follow the same-frame cumulative funnel from visibility to fire. Read decision reasons, sigma budget and biggest blocker rather than adjusting thresholds blindly. Trimmed output from the seed-1 crossing run on the current synthetic configuration:

   ```text
     visible              265  first=36
     detected             232  first=39
     confirmed            225  first=57
     p_hit_ok               0  first=None
     best_moment            0  first=None
     fire                   0  first=None
     best geometry [deg]: miss 0.19, pattern 0.95, sigma 0.91; P(hit) 0.42
   Biggest blocker: P(hit) never reached 0.50: best 0.42 at frame 102 (miss 0.19 deg, pattern 0.95 deg, sigma 0.91 deg; sigma dominated by track term 52 %)
   ```

   The funnel is truth-matched for synthetic scenes; counts and timing-dependent values can vary with configuration and host load. Replace the source scenario with another from `tbot sim -h` or use a recorded source supported by `tbot explain -h`.

4. **Explore parameter trade-offs.**

   ```bash
   tbot tune --param tracker.jerk_psd_deg=250,500,2000 --seeds 3 --write-best tuned.toml
   ```

   The table orders candidates by false shots ascending, hits descending, then mean visible-to-first-fire time ascending; hit rate breaks ties. The Pareto front contains candidates not dominated on the first three objectives, so compare trade-offs rather than just the first row. `--write-best` writes an override TOML; try it with `--config tuned.toml`. In a synthetic search of five scenarios × three seeds, jerk PSD 2000 produced zero shots, 500 produced 4 hits in 5 shots, and 250 produced 7 hits in 8 shots; tracker NEES at 2000 was 0.2–0.9 versus an ideal 2. However, the fast-crossing tracker prediction test at 1 rad/s and 150 ms prediction failed its 0.3° median error limit at 500 and 750; 1000 passed that test but produced 0 hits in 2 shots in the tune. This is a trade-off, not a demonstrated improvement across conditions: the default remains 2000 until real recorded sessions resolve it. Results depend on scenarios, seeds and configuration. This is **synthetic-only**: never edit `config/default.toml` from a tune without confirming on recorded sessions and HITL, and do not infer safety from the synthetic ranking.

5. **Measure processing on the Pi and compare machines.**

   ```bash
   tbot bench --profile 25 --no-overlay
   ```

   Run on the Pi and inspect per-stage p95 against the frame period and the top 25 profile functions; `--no-overlay` excludes drawing and JPEG encoding. To compare, first run the same command on the laptop, copy its `calib/bench.json` to the Pi as `laptop-bench.json`, then run `tbot bench --profile 25 --no-overlay --compare laptop-bench.json` on the Pi. Keep resolution, scenario, settings and overlay mode comparable. `--profile-out bench.pstats` also saves the profile; the normal run writes `calib/bench.json` unless `--no-write` is used.

6. **Self-test the bench, then measure the real rig.**

   ```bash
   tbot hitl run --sim --live
   ```

   Inspect the simulation self-check, funnel of decisions, timing and sweep reports, then follow [HITL.md](HITL.md) for the real dry bench and unloaded live bench with the arming button held and break-sense or microphone. The simulation does not validate hardware timing.

7. **Explore PWM frame jitter.**

   ```bash
   tbot latency-actuator --sim --set actuator.sim.model=servo --no-write
   ```

   Read the simulated command-to-break distribution: the 50 Hz model adds a wait for the next PWM frame before simulated travel. Uniform phase implies 0–20 ms wait and about 5.8 ms standard deviation, not a hardware measurement. Measure the installed servo and break contact as in [CALIBRATION.md](CALIBRATION.md#6b-actuator-fire-command--trigger-break).

## Known findings from the synthetic bench

These are seed-1 synthetic results at the current 640 × 400 binned configuration, not real rig measurements. `tbot explain --source synthetic:skeet_crossing --seed 1` reports a 0.95° pattern and best-frame 0.91° predicted sigma, with track and range contributing 52.1% and 38.3% of sigma²; P(hit) never reached the 0.50 threshold and no fire occurred. `tbot explain --source synthetic:trap_straightaway --seed 1` reports 18/120 in-window frames detected, a 1.7 px median diameter upper bound, and 215 local threshold-area and 150 local contrast miss factors. At best P(hit) was 0.21 with 1.40° sigma against the 0.95° pattern. The small/far synthetic clay can fall below `min_area_px` or contrast thresholds; the reported factors overlap and are observations, not proven causes. Confirm detection, PSF and uncertainty against real sessions and the HITL bench.
