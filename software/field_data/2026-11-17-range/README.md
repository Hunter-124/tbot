# First live range day, 2026-11-17

First time the gun was armed live on real clays (build `6f59293`, field mode, adaptive P(hit) bar 0.40 -> 0.30).
Nine presses, seven fire commands, **two real shots, both seen to break the clay** (shooter's account). After the
second shot the servo's mount broke, so the remaining five fire commands moved no trigger.

Session times are the Pi's clock (no RTC, about 3 h behind).

| press | fire cmd | gun fired (recoil) | range | P(hit) at fire | score as planned | note |
|---|---|---|---|---|---|---|
| `104601` | no | - | - | - | - | track never older than "young"; AE still settling (exposure 19 -> 9 us) |
| `104603` | yes | **no** | 27.7 m | 0.35 | hit, 0.52 deg | fired on the relaxed bar (0.30); no recoil seen |
| `104751` | yes | **yes, 162.0 ms after the command** | 22.8 m | 0.42 | miss, 1.00 deg | clay seen to break |
| `104806` | yes | **yes, 160.2 ms after the command** | 27.8 m | 0.59 | hit, 0.62 deg | clay seen to break |
| `104829` | yes | no (mount broken) | 28.1 m | 0.57 | miss, 0.84 deg | |
| `105308` | yes | no | 30.2 m | 0.73 | hit, 0.41 deg | |
| `105335` | no | - | - | - | - | 0.36 s press |
| `105523` | yes | no | 13.2 m | 0.43 | miss, 0.84 deg | |
| `105639` | yes | no | 25.0 m | 0.58 | hit, 0.13 deg | |

"Score as planned" is `tbot score`: where the charge would have gone had the shot left when the solver planned
it, against where the clay really was in the later frames. It is **not** what happened for the two real
shots, which left about 60 ms later than planned (below), and the other five never left at all.

## What this data says

1. **The trigger delay is wrong by about 60 ms.** Command to ignition from the recoil of the two real shots:
   160.2 and 162.0 ms (`latency_recoil.txt`). The gun assumed 95 ms command-to-break (+6 ms lock time = 101).
   The bench microphone measurement of 2026-11-15 said 161.3 +/- 1.0 ms and was set aside as "includes
   uncalibrated USB mic latency" in favour of a 10 fps video estimate; the microphone was right.
   `tbot latency-recoil` wants 5 shots before it writes; with two shots agreeing with ten bench shots to
   within 1 ms the value to use is `cmd_to_break` = 155 ms. Not changed yet: the servo mount is being
   rebuilt, which can move it. A 20 m/s crossing clay moves 1.2 m in 60 ms, so both clays broke despite
   this, not because of it; work out why (going-away presentation? swing carried the lead?) before trusting it.
2. **Frame rate while armed was 19-43 fps, not 100** (`summary.json`: `fps`, `camera_frames_skipped`
   160-306 per press, frame interval p50 20-60 ms). The detector took 15-20 ms p50 per frame, frame age
   40-48 ms p50.
3. **The scenery filter and the bar let the gun fire on 7 of 9 presses**, 0.3-1.5 s after the button went down. Reasons
   it held before that: mostly `no confirmed track`, then `track young`, `p_hit < bar`, and on two presses
   `too slow for a clay`.
4. **P(hit) versus what would have happened** (`score_every.txt`, every frame with a firing solution scored
   as if fired, 54 frames): below 0.1 none would have hit; 0.2-0.4 all 7 would have; 0.4-0.7 2 of 5; the one frame above 0.7 would have. Too
   few frames to call it calibrated, and all of it with the wrong trigger delay.
5. **Gun extrapolation error is now the larger term**: p50 0.82 deg over the 7 fire commands against 0.76 deg
   for the target, with a pattern of 0.75 deg. Real swings on clays are faster than the bird presses were.
6. Heat was not a problem: 43-63 C, never throttled. Exposure sat at the 9 us floor with AE reporting
   `too_bright` on four presses (level 222-234 DN).

## What is here

- `presses/<name>/`: everything the gun saved for each press: `frames.bin` (raw 640x400 8-bit frames, git
  LFS), `frames.csv`, `imu.csv`, `telemetry.jsonl` (one line per frame: detections, tracks, decision),
  `events.jsonl` (arming, fire, recoil), `meta.json` (config and calibration in force), `summary.json`.
  Added on the laptop afterwards: `clip.mp4` (annotated, 2x, git LFS) and `shot_1.png` (the arrival frame
  with the planned pattern drawn).
- `gun/`: the Pi's `calib/` and `config/local.toml` as they were on the day.
- `score.txt`, `score_every.txt`, `latency_recoil.txt`: output of `tbot score --image`, `tbot score --every`
  and `tbot latency-recoil` on these presses.

Replay a press through the live loop:

    tbot run --source replay:field_data/2026-11-17-range/presses/20261117-104806-press --record off \
        --set gpio.backend=sim --sim-armed --set actuator.type=sim

The journal of the service from the range was not kept (the Pi's journal did not survive the power cycle).
