# Recorded presses: clips and debug data

`tbot run --record armed` (what the service runs) saves **every held press of the arming button** as one
session under `sessions/<date>-<time>-press/`: from `record.preroll_s` (1 s) before the button went down to
`record.postroll_s` (0.7 s) after it came up. Nothing else is needed: hold the button, let go, it is on the SD
card. Dry run and live behave the same.

## What is in a session

| File | What |
|---|---|
| `clip.mp4` | Annotated video, H.264. **Every** recorded frame, played at `record.clip_fps` (30): a 100 fps press is 3.3x slow motion. Top-right: real time from the press (`HELD t+1.234s frame n/N`) and the exposure. |
| `summary.json` | The press at a glance: frames, achieved fps and frame-interval percentiles, camera frames skipped, frames with a detection / with a confirmed track, why the solver did not fire (reasons with counts), max P(hit), stage times p50/p95, exposure range, Pi temperature and throttle flags, shots. |
| `telemetry.jsonl` | One line per processed frame: everything the pipeline computed (see below). |
| `frames.bin` + `frames.csv` | The raw frames, bit exact, with timestamps and exposure: `tbot run --source replay:<session>` runs today's code on them. |
| `imu.csv`, `events.jsonl` | IMU samples (gun frame); arming / fire / recoil events. |
| `meta.json` | Full config and calibration in force, the press window. |

What the clip draws: green box = detection (label: SNR and measured width in px); grey crosses = candidates the
scenery filter dropped (fixed in the world); yellow ring = track (`#id range`), dim until confirmed, with its
velocity arrow; magenta X = where the clay will be when the shot arrives; cyan circle = where the bore will
point, joined by a line (that line is the predicted miss); red cross = bore at the prior range; FIRE frame in
red; bottom: fps, frame age, per-stage milliseconds, counts, and the decision with its reason.

`telemetry.jsonl` per frame: `seq`, `t_exposure_ns`, `age_ms`, `fps`, `armed`, `fired`, `shots`, `dropped`,
`exposure_us`, `gain`, `ae`, `health`, `stage_ms`, `detector_ms`, `detector` (scale, noise, components,
candidates, kept), `dets` (`[u, v, diameter_px, snr, contrast, bbox x, y, w, h, area]`, full-res pixels),
`static` (dropped scenery, pixels), `R_wg` (gun attitude), `tracks` (az/el, range and sigma, hits, misses,
confirmed, filter state), `decision` (fire, reason, miss / pattern / sigma in degrees, P(hit), range, time of
flight, lead and bore directions).

## Looking at them

- Browser, straight off the Pi: `http://169.254.20.20:8080/clips` (newest first; the three newest play in the
  page, each with links to its summary and telemetry). The clip appears a few seconds after the release: it
  is rendered by a low-priority child process (measured: 469 frames in 20 s on the Pi 3B while the pipeline
  kept running).
- Copy a session: `scp -r pi@169.254.20.20:TBOT/software/sessions/<name> .`
- Re-render or render later: `tbot clip sessions/<name>` (`--scale 2` for a bigger picture, `--fps`),
  `tbot clip --missing` for every session without a clip (e.g. the service was stopped mid-render).
- Replay through the pipeline: `tbot run --source replay:sessions/<name> --sim-armed --set gpio.backend=sim`.

## Did the shot hit? (`tbot score`)

`tbot score sessions/<name> --image` checks every shot of a press (dry or live) against the frames recorded
after it: where the bore really pointed when the shot left (the recorded IMU samples replayed through the
attitude filter) against where the target really was when the shot arrived (its detections in the frames
around that moment). It prints the miss against the pattern, splits it into the error in extrapolating the
swing over the trigger delay and the error in extrapolating the target, and writes `shot_<n>.png`: the arrival
frame with the pattern circle where the shot went and a cross on the target. The press must last (or the
post-roll run) until the shot would have arrived, about 0.2 s after the fire decision.

To get shots without clays: `tbotctl start test` is a dry run that fires on paper at ANY moving target it tracks (a
bird, a plane), up to 6 times a press. `tbotctl start` goes back to the normal dry
run. Test mode can never drive the servo.

## Limits

- Frames wait in RAM until the release: `record.max_ram_mb` (150) is about 6 s at 100 fps including the
  preroll. A longer press keeps its **last** 6 s (`summary.json: evicted_frames`).
- Storage: 25 MB per second of press at 640x400 / 100 fps (a 4 s press = ~140 MB with pre/postroll). `tbotctl
  status` shows the free space; delete or copy off `sessions/` between outings.
- `record.clip = "off"` skips the video (the raw data is still saved); `"inline"` renders in the recorder
  thread (laptops, tests).
- While the live page is open the pipeline loses ~10 % of its frame rate (drawing and JPEG run in the same
  process). Close it for the runs that matter; the clip has the same picture for every frame anyway.
