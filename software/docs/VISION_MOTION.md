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
of foliage flicker) need two detections more. A track is born confirmed with velocity and range from the whole
path, credited with two detections fewer than the solver's `fire.min_track_hits`: the solver cannot fire on the
frame a path is first believed. A live track takes a detection only inside its filter gate, near the path of its
own last detections, and of its size, mass and sign.

The detector's hot loops are C (`_measure.c`, `tbot_motion_*`), a statement-for-statement port of the integer
NumPy reference in `motion.py`; `tests/test_native_motion.py` holds them equal bit for bit.

## Results

68 presses recorded on the gun from 2026-11-17 to 2026-11-20 (the range, the yard, a bench; 218 armed
seconds), through the whole pipeline as `tbot rescore` runs it, with 202 virtual clays composited into them
(three runs of each press, one clay each: `tbot vbench inject`, seed 1) and as recorded (`tbot vbench clean`).
Old = the software of 2026-11-19 (`vision.engine=legacy`), new = this front end as shipped.

| | old | new |
|---|---|---|
| detections a frame that are not the target | 2.70 | 0.99 |
| confirmed tracks per armed second on something that is not a labelled target | 13.5 | 0.27 |
| frames in which the solver's target was such a track | 39.5 % | 8.8 % |
| virtual clay detected, of the frames it is in view | 45 % | 60 % |
| ... in front of trees or ground | 24 % | 52 % |
| ... on sky | 74 % | 71 % |
| ... bright / dark / lit from above | 56 / 41 / 37 % | 82 / 44 / 58 % |
| ... under 4 px across | 44 % | 27 % |
| virtual clays that never got a track | 44.6 % | 36.1 % |
| first sight to track, median / 90th percentile | 80 / 389 ms | 160 / 390 ms |
| frames in which the solver's target was the virtual clay | 16.8 % | 21.7 % |
| virtual clays shot at / hit | 11.4 / 11.4 % | 17.8 / 17.3 % |
| fire decisions on the footage as recorded | 23 | 15 |

What the remaining numbers are:

- **The 15 fire decisions of the new front end were each looked at** (the frame, the frames around it and a
  zoom on the target): the clays of 2026-11-17 (presses 104806, 104829, 105308, 105523, 105639), the clays
  of 2026-11-19 (092512, 092733, 160243, 160303, 160413), a thrown ball (155033, 155119, 20261120-072637),
  a foam glider (20261120-071755) and a small dark object flying across the sky (154955). None is at
  scenery. The old front end's 23 were not all looked at; of those that were, one is at a house wall with
  nothing in front of it (141512), one at dark trees (153543) and one at empty sky (092521), and two are birds
  (104601, 104603).
- **The 0.27 tracks a second are mostly real movers nobody labelled**: the thrower's hand and the person in
  the yard presses of 2026-11-20 (072706, 072604), the ball, birds. Only 6 presses have reviewed labels
  (`bench_data/truth`), so every unlabelled real mover counts against the front end here.
- **Time to a track is 80 ms longer.** A track now needs 5 detections on one path, and the solver two more
  after its birth; before it was any 3 detections inside a degree. On the hit of 2026-11-19 16:02:43 the fire
  decision comes at row 159 of 172, where the old front end's comes when replayed; the shot recorded that day
  is at row 160.
- The virtual clays are drawn into footage whose gun was not following them, so "hit" is low for both; it
  compares the front ends, it does not predict a score.

On the Pi 3B (C kernel, one core, with `tbot-run` and the console running beside it; measured before the
closed-form prior and with the local field still on, each worth about 2 ms): detection 17.8 ms a frame at the
median (register 11.0, difference kernel 5.0, the rest 1.8), tracker 0.5 ms, the whole loop 18.5 ms (p95 21.6).
The front end of 2026-11-19 was budgeted at 22 ms a frame with a target in view (`sim.process_ms`).

The simulator (`tbot sim`, a clean synthetic sky, the gun's lens, 10 runs) is the old detector's best case and
says less: 8 shots / 5 hits old, 7 / 3 new.

## What it does not do

- **Something that does not move against the world is not seen.** A clay at the top of a lob, or one flying
  straight at the muzzle, has no angular rate for a moment: no detections until it moves again. Below
  `init_min_speed_dps` (2 deg/s) nothing gets a track.
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
| `vision.noise_k`, `vision.cfar_gain`, `vision.texture_gain` | 4, 6, 2 | the detection bar: lower sees fainter, and more flicker |
| `vision.track.init_hits` | 5 | detections on one path before a track exists |
| `vision.track.init_min_speed_dps` | 2 | slower against the scenery is not a target |
| `stream.overlay_detail` | `clean` | `debug` draws every candidate as before |
