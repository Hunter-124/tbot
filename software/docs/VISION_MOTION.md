# The motion front end (`vision.engine = "motion"`)

From 2026-11-20 the gun finds its target by what MOVES against the world, not by what has contrast. This
replaces the single-frame blob detector, the IMU scenery filter and the track births of `tracker3d` as the
default; the filter (9-state EKF per clay), the fire solver, the arming chain and the actuator are unchanged.
The software of 2026-11-19 is still there: `vision.engine = "legacy"`.

## Why

Replays of 2026-11-17 to 2026-11-20 showed tracks on cracks in concrete, foliage and other things that do not
move, and clays in front of trees that were never seen. The cause was structural, not a threshold:

- the detector marked everything with local contrast in ONE frame, so a twig against a sky gap and a clay look
  the same to it;
- telling them apart was left to a filter working on candidate lists (was a candidate at this world direction
  100 ms ago?), with a 2 px tolerance on gyro-predicted positions and a cap of 6 candidates a frame;
- any 3 detections inside a 1 degree gate made a confirmed track.

Measured on 68 recorded presses (218 armed seconds), the old front end kept 2.7 scenery detections a frame and
made 13.5 confirmed tracks on them per armed second.

## How it works

Per frame, at half resolution (320 x 200, `vision.scale = 2`):

1. **Register.** Two earlier frames (A about 40 ms ago, B about 100 ms ago) are laid over the current one. The
   gun's rotation since then (IMU attitude through the camera model, a full homography) predicts where the
   scenery went; an integer SAD search on 1/8 and 1/4 size copies corrects that prediction. The correction is
   only believed where the picture can carry it:
   - at least 3 % of the overlap must be textured (never the target alone on open sky);
   - the texture must fix the shift in both directions (gradient structure tensor). A horizon or a roof line
     fixes it across the edge only: then only that component is applied;
   - otherwise the gyro's prediction stands, plus a running correction learnt from earlier registrations
     (the gyro's residual bias).
2. **Difference against what the scenery could look like.** For each earlier frame, the min..max of its
   registered picture over a 3 x 3 neighbourhood is the range the scenery can take at that pixel with a pixel of
   registration error. A pixel is a candidate only where the current frame leaves that range in BOTH earlier
   frames: a clay differs from both, the place it left differs from one only (no ghost), a leaf, a crack or a
   cloud edge differs from neither. Brightness changes between frames are fitted from the picture pair first.
3. **Threshold locally.** The bar is `max(t_min_dn, noise_k x sensor noise)` plus `cfar_gain` times the mean
   residual around the pixel plus `texture_gain` times what texture leaves when it does not register to the
   pixel: fluttering leaves raise their own bar, open sky keeps the low one.
4. **Measure.** Connected pixels become a candidate with centroid, size, summed residual ("mass") and its
   direction in the background-stabilised frame (`Detection.stab_dir`): a frame in which scenery stands still.
   (`Detection.stab` is the same thing as pixels in the plane of the first frame; that plane stretches without
   bound as the gun turns 60 to 90 degrees from where it pointed when the software started, so the tracker
   lays the directions out about a reference of its own that follows the target.)

The lags adapt: they shrink while the gun swings (the strip of picture that came into view since frame B has
nothing to be compared with, and that strip is where the clay being swung onto is), and they stretch by
`SKY_LAG_GAIN` on open sky, where nothing can be left behind by a longer lag and a slow clay needs one.

**Tracks** (`tracker_motion.py`) are born only from a PATH: `init_hits` (5) detections whose stabilised
positions lie on one smooth path in time (constant acceleration up to `init_max_accel_dps2`; a clay going away
slows down in angle, a near one curves under gravity), of consistent size and mass, travelling at least
`init_min_travel_px` at `init_min_speed_dps` or more against the scenery. Faster than `init_max_speed_dps`
against the scenery only counts while it is slow in the PICTURE: a target the gun is following (a ball thrown
from beside the shooter turned at 130 to 230 deg/s with the gun on it). Weak paths (`init_weak_mass`: the size
of foliage flicker) need two detections more. Strong ones (`init_strong_mass`, 2000: a clay at 30 m is about
3000) need one fewer (`init_strong_hits`, 4), but only when they stood alone: at most one other detection within
`init_strong_radius_px` (48) of their points over their frames. Chance lines of strong specks are as likely as the
specks are crowded; in a patch of strong flicker (sim, 12 a frame in 150 x 100 px) lines of 4 made 170 tracks in
1500 frames, lines of 5 made 9, and with the isolation test it is 9 again. A track is born confirmed with velocity
and range from the whole path, credited with `post_birth_hits` (2) detections fewer than the solver's
`fire.min_track_hits`: the solver cannot fire on the frame a path is first believed. A live track takes a
detection only inside its filter gate, near the path of its own last detections, and of its size, mass and sign.

The solver's age gate is `vision.track.fire_min_age_ms` (120 ms) under this front end, not
`fire.min_track_age_ms` (200 ms): that one gives the single-frame detector's scenery filter time to call a speck
scenery, and a track here is born from a path already.

The detector's hot loops are C (`_measure.c`, `tbot_motion_*`), a statement-for-statement port of the integer
NumPy reference in `motion.py`; `tests/test_native_motion.py` holds them equal bit for bit.

## Results

68 presses recorded on the gun from 2026-11-17 to 2026-11-20 (the range, the yard, a bench; 218 armed
seconds), through the whole pipeline as `tbot rescore` runs it, with 202 virtual clays composited into them
(three runs of each press, one clay each: `tbot vbench inject`, seed 1) and as recorded (`tbot vbench clean`).
Old = the software of 2026-11-19 (`vision.engine=legacy`), new = this front end as shipped on 2026-11-20, now =
with the births and the age gate of 2026-11-21 (strong lone paths born at 4 detections, the solver's age gate
120 ms; the detector is unchanged).

| | old | new | now |
|---|---|---|---|
| detections a frame that are not the target | 2.70 | 0.99 | 0.99 |
| confirmed tracks per armed second on something that is not a labelled target | 13.5 | 0.27 | 0.31 |
| frames in which the solver's target was such a track | 39.5 % | 8.8 % | 9.1 % |
| virtual clay detected, of the frames it is in view | 45 % | 60 % | 60 % |
| ... in front of trees or ground | 24 % | 52 % | 52 % |
| ... on sky | 74 % | 71 % | 71 % |
| ... bright / dark / lit from above | 56 / 41 / 37 % | 82 / 44 / 58 % | 82 / 44 / 58 % |
| ... under 4 px across | 44 % | 27 % | 27 % |
| virtual clays that never got a track | 44.6 % | 36.1 % | 33.7 % |
| first sight to track, median / 90th percentile | 80 / 389 ms | 160 / 390 ms | 140 / 347 ms |
| frames in which the solver's target was the virtual clay | 16.8 % | 21.7 % | 21.1 % |
| virtual clays shot at / hit | 11.4 / 11.4 % | 17.8 / 17.3 % | 19.3 / 18.8 % |
| fire decisions on the footage as recorded | 23 | 15 | 15 |

What the 2026-11-21 column changed, measured one lever at a time on the same 202 virtual clays (hits: 35 new):
the age gate at 120 ms +2, at 100 ms +3 (not taken: tests/test_solver.py's incoming clay, the bore on it, was
then shot on a filter that had not settled, its predicted miss at the pattern's edge and P(hit) 0.56, where at
120 ms it waits two frames for 0.85); births at 4 detections for every path +3 but 0.38 tracks a second
(the extra ones on the hand, the person and the ball of the yard presses, which nobody labelled); for strong
paths only (2000) the same +3 at 0.36; with the isolation test 0.31. What did not help: lower detection bars
(`noise_k` 3: recall 60 -> 63 %, 35 % more scenery detections, no hit more; `cfar_gain` 4: none either), the
gyro-only speed floor at 2 or 4 deg/s (-1 hit, 3 false fires more), firing one detection after the birth
instead of two (5 false fires more, no hit), a narrower border margin (time to a track -20 ms, no hit). The
15 fire decisions on the footage as recorded are the same targets as before: 1 or 2 frames earlier in five
presses, one frame later in one (155119, at P(hit) 0.96 instead of 0.72), and in 071755 the same foam glider on
the same frame as track #2 instead of #1.

What the remaining numbers are:

- **The 15 fire decisions of the new front end were each looked at** (the frame, the frames around it and a
  zoom on the target): the clays of 2026-11-17 (presses 104806, 104829, 105308, 105523, 105639), the clays
  of 2026-11-19 (092512, 092733, 160243, 160303, 160413), a thrown ball (155033, 155119, 20261120-072637),
  a foam glider (20261120-071755) and a small dark object flying across the sky (154955). None is at
  scenery. The old front end's 23 were not all looked at; of those that were, one is at a house wall with
  nothing in front of it (141512), one at dark trees (153543) and one at empty sky (092521), and two are birds
  (104601, 104603).
- **The 0.27 (now 0.31) tracks a second are mostly real movers nobody labelled**: the thrower's hand and the
  person in the yard presses of 2026-11-20 (072706, 072604, 071755), the ball, birds, a hand waved in front of
  the lens indoors (20261118-220236). Only 6 presses have reviewed labels
  (`bench_data/truth`), so every unlabelled real mover counts against the front end here.
- **Time to a track is 60 ms longer than the old front end's.** A track needs 4 or 5 detections on one path,
  and the solver two more after its birth; before it was any 3 detections inside a degree. On the hit of
  2026-11-19 16:02:43 the fire decision comes at row 159 of 172, where the old front end's comes when replayed;
  the shot recorded that day is at row 160.
- **Where the virtual clays the gun follows are lost** (89 of the 202; the 113 crossing ones are not followed and
  are almost never hit by either front end): 37 hit, 1 shot at and missed; 20 lose the press's one shot to the real clay, ball or
  glider of the recording (the same shots as on the footage as recorded: right, and counted against the
  virtual clay); 19 never get a track (4 of them are drawn into presses of 2026-11-19 15:46-15:47 whose frames
  are black, 6 DN median: a clay there is a few DN from its background, which no detector sees; 11 of the other
  15 are dark clays, nearly all in front of trees); 12 are tracked but the swing never brings P(hit) over the
  bar in the frames they are tracked (in press 104601 the bore crosses the lead point on the frame the track is
  born, at P(hit) 0.4).
- The virtual clays are drawn into footage whose gun was not following them, so "hit" is low for both; it
  compares the front ends, it does not predict a score.

On the Pi 3B (C kernel, as deployed 2026-11-20, commit 112198a): detection takes 11.7 ms a frame at the
median and 13.7 ms at the 95th percentile on a recorded press with its IMU (register 6.1, difference kernel
3.8, pyramid 1.2, components and measurement 0.6), measured while the gun's own service ran beside it; the
tracker 0.6 ms, the solver 0.4 ms. Live on the gun (idle, in the dark) the telemetry shows 10 to 17 ms for
detection. The front end of 2026-11-19 was budgeted at 22 ms a frame with a target in view (`sim.process_ms`)
and logged 11 to 12 ms idle. `tbot bench` gives the detector no attitude, which is the slower search without
an IMU: it reads 22 ms for detection on the same Pi and is not the gun's number.

Replayed on the Pi itself (`tbot rescore`), the hit of 2026-11-19 16:02:43 is one shot on track #1, scored a
hit (0.35 deg against a 0.75 deg pattern); the old front end hits it too, on track #357 of that press.

The simulator (`tbot sim`, a clean synthetic sky, the gun's lens, 10 runs) is the old detector's best case and
says less: 8 shots / 5 hits old, 7 / 3 new.

## What it does not do

- **Something that does not move against the world is not seen.** A clay at the top of a lob, or one flying
  straight at the muzzle, has no angular rate for a moment: no detections until it moves again. Below
  `init_min_speed_dps` (2 deg/s) nothing gets a track.
- **A slow mover cancels itself.** A target has to move more than about its own size against the scenery
  between reference A (40 ms back) and now. On sky, virtual clays that move less than half of that are seen in
  52 % of their frames, faster ones in about 90 % (2026-11-21, 68 presses): a clay going away at 2-5 deg/s once
  the gun steadies. Longer lags for every frame do not help (100 / 250 ms: clays hit 18.8 -> 14.9 %, 44 % never
  tracked). `vision.slow_gain` (off) adds a second comparison, against B and a frame about 2.5 x B's lag back
  laid over through the stabilised plane, with its own CFAR bar, an edge veto and nothing added next to what the
  A/B test already has: virtual clay seen 60 -> 62 % of its frames (sky 71 -> 74 %, under 4 px 27 -> 38 %),
  scenery detections 0.99 -> 1.16 a frame, the same clays hit and the same 15 shots on the footage as recorded.
  The Pi 3B pays 10 ms a frame for it (detection 12.1 -> 22.5 ms at the median), so it stays off. Without the
  "nothing next to the A/B candidates" rule it also moved tracks the A/B test already had: on the footage as
  recorded it then fired on the thrown ball of 155103 (a real target the A/B test does not shoot) but also on a
  bird (104601), and lost the foam glider of 071755.
- **Everything that moves is seen**: birds, a thrown ball, a hand, a person walking through the picture. The
  recordings of 2026-11-20 have tracks on the thrower's hand. What keeps a shot off such a track is what kept
  it off before: the solver's gates (a clay's speed, elevation, P(hit)) and the shooter, who holds the arming
  button. Nobody downrange, as always.
- **Open sky rests on the gyro.** With no texture to register on, the stabilised plane is the gyro's. A gyro
  bias the detector has not yet learnt (it learns it whenever it sees texture) makes a lone tree top at the
  picture's edge drift at that bias: press 20261119-092521 has such a track. The yaw gyro read about -2.5 deg/s
  at rest in every press of 2026-11-17 to 20 (the stored bias is from 2026-11-14 at 24 C).
- Far clays under 4 px (beyond about 50 m with the 12 mm lens) are mostly lost at half resolution
  (`vision.scale = 1` sees them, at 3 to 4 times the processing time).
- A dark clay in front of dark trees is seen in a quarter of its frames (the old front end: the same). Sunlit
  clays in front of trees are bright in this camera and are the easy case.
- `tbot hitl` (the monitor bench) runs the single-frame front end whatever `vision.engine` says: its targets
  stand still on a screen. It still measures the geometry, the latencies and the arming chain; it does not
  exercise this front end.

## Checking it

```
tbot vbench inject --presses DIR --clays-per-press 3 --seed 1 --set vision.engine=motion --out new.json
tbot vbench clean  --presses DIR --set vision.engine=motion --out new_clean.json
tbot vbench compare old.json new.json
```

`inject` composites virtual clays (the trajectories of `tbot vclay`, drawn through the lens model with the
recorded exposure, noise and blur) into recorded presses and runs the whole pipeline on them as `tbot rescore`
does, so the truth is exact: recall by size, background and polarity, scenery detections, false tracks, time to
a track, what the solver aimed at, shots and hits. `clean` runs the footage as recorded. Real movers in the
footage are taken from `bench_data/truth/<press>.json` (written by `python -m tbot.bench.oracle`, counted only
when a person has reviewed the file); a real clay in footage nobody has reviewed counts as "false", so every
"false fire" has to be looked at before it is believed.

`tbot rescore PRESS --set vision.engine=legacy|motion`, `tbot clip PRESS --detail clean|debug` and
`python -m tbot.vision.motion PRESS --sheet OUT.png` show one press.

## Configuration

`[vision]` and `[vision.track]` in `config/default.toml` (every key is commented there). The ones that matter
at the range:

| key | default | meaning |
|---|---|---|
| `vision.engine` | `motion` | `legacy` = the software of 2026-11-19 |
| `vision.scale` | 2 | 1 = full resolution (far clays; slower) |
| `vision.slow_gain` | 0 | 2.5 = the slow-mover test (sky 71 -> 74 %; +10 ms a frame on the Pi 3B) |
| `vision.noise_k`, `vision.cfar_gain`, `vision.texture_gain` | 4, 6, 2 | the detection bar: lower sees fainter, and more flicker |
| `vision.track.init_hits` | 5 | detections on one path before a track exists |
| `vision.track.init_strong_hits`, `init_strong_mass` | 4, 2000 | ... when they are this strong and alone (0 = off) |
| `vision.track.fire_min_age_ms` | 120 | the solver's age gate under this front end (negative: `fire.min_track_age_ms`) |
| `vision.track.init_min_speed_dps` | 2 | slower against the scenery is not a target |
| `stream.overlay_detail` | `clean` | `debug` draws every candidate as before |
