# Shot pattern for this gun and load (research, 2026-11-16)

Gun: Weatherby Element 12 ga, **26" barrel, Modified choke**. Load: **#8 lead target load** (1 oz or 1 1/8 oz,
~1200 fps box velocity). This replaces the skeet-choke guess the solver started with. It is a model built from
published pattern figures, **not a measurement of this gun**: confirm it on paper at the range (CALIBRATION.md §5b).

## What the solver needs

`[shot]` in `config/*.toml`. The solver fires only when P(hit) >= `fire.min_hit_probability`, and P(hit) uses an
angular "pattern radius" that grows linearly with range:

```
pattern_radius(range) = max(pattern_min_radius_m, pattern_radius_ref_m * range / pattern_ref_range_m)
```

So the one number that matters is the **angle** (radius / range) inside which a clay reliably breaks.

## Published figures

| Quantity | Value | Source |
|---|---|---|
| Modified choke, % of pellets in a 30" circle at 40 yd | ~60 % (55–67 % depending on source and constriction) | Wildfowl chart: Modified 100 / 83 / 60 % at 20 / 30 / 40 yd; Chuck Hawks: Modified .020", 67 %; Orvis: light modified 55 % |
| Modified "effective pattern diameter" | 12 / 20 / 32 / 46 in at 10 / 20 / 30 / 40 yd | Wildfowl chart 4 |
| #8 lead pellets | ~410 per oz (~461 in 1 1/8 oz) | Chuck Hawks |
| Velocity vs barrel length | +4 to +15 fps per inch; 26" vs the 30" test barrel ≈ 15–35 fps slower | BC Outdoors, Fox/Blazek (~5 fps/in), Field & Stream (10–15 fps/in), AccurateShooter cut-down (~8 fps/in) |

## Model

Pellets spread as a 2-D Gaussian with sigma proportional to range. Fitting "60 % in a 30-inch (0.381 m radius)
circle at 40 yd" gives **sigma = 7.69 mrad**. That fit reproduces the other chart points: 80 % at 30 yd (chart
83 %) and 97 % at 20 yd (chart 100 %).

A clay "breaks" where the expected number of hits is at least 3 #8 pellets. The clay is 108 mm across; it is
counted as 0.6 of its face area, because a flying clay is seen tilted. The break radius where that holds:

| Range | 1 oz | 1 1/8 oz |
|---|---|---|
| 8 m | 0.16 m (20.2 mrad) | 0.16 m (20.6 mrad) |
| 10 m | 0.19 m (18.9 mrad) | 0.19 m (19.2 mrad) |
| 15 m | 0.24 m (16.1 mrad) | 0.25 m (16.5 mrad) |
| 20 m | 0.28 m (13.8 mrad) | 0.29 m (14.3 mrad) |
| 25 yd (22.9 m) | 0.29 m (12.6 mrad) | 0.30 m (13.2 mrad) |
| 30 m | 0.29 m (9.8 mrad) | 0.31 m (10.5 mrad) |
| 35 m | 0.27 m (7.7 mrad) | 0.30 m (8.5 mrad) |

With the clay face-on the radius is 10–20 % larger. The real break angle shrinks with range, as the pattern
thins out, while the solver's model keeps it constant. So one constant angle is generous far out and strict close
in.

## Values set (Pi `config/local.toml`, 2026-11-16)

```toml
[shot]
muzzle_velocity_mps = 360.0   # 1200 fps box (30" test barrel) - ~20 fps for 26"
pattern_radius_ref_m = 0.30   # 13 mrad: the >= 3-hit radius at 25 yd, Modified, #8
pattern_ref_range_m = 22.9
```

What these values mean in practice:

- **Previous value:** 0.38 m @ 22.9 m (16.6 mrad, a skeet-choke guess). That is about 25 % too wide for Modified at
  20–30 m, so the gun would have fired on aims that leave the clay at the pattern's thin edge.
- **Close in (backyard, 8–15 m):** 13 mrad is strict there; the model's own number is 16–20 mrad. The solver
  waits for a better aim, which makes it fire less often, not miss more. Beyond ~30 m it is generous;
  `fire.max_range_m = 40` caps that.
- **Velocity:** a 20 fps (1.7 %) error moves the lead by about 1 ms of flight time at 20 m, roughly 2 cm on a
  crossing clay. Chronograph the actual load when convenient.

## Confirming it

At the range, shoot paper at a measured 25 yd with this load (CALIBRATION.md §5b):

- **Count:** the pellets inside 30" and inside 20" circles around the densest point. Fewer than ~50 % inside 30"
  means a more open pattern than modelled; raise `pattern_radius_ref_m` and expect fewer hits per pellet.
- **Centre offset:** the pattern centre against the point of aim goes in with `tbot calib-boresight --poi-offset-cm`.
- **Repeat:** use several sheets; a single pattern is noisy.

Sources: [Wildfowl — shotgun tips & choke chart](https://www.wildfowlmag.com/editorial/tips_strategies_shotgun_062304/281145),
[Chuck Hawks — shotgun chokes](https://chuckhawks.com/shotgun_chokes.htm),
[Orvis — what is choke](https://www.orvis.com/what-is-choke-and-how-much-of-it-do-i-need-in-my-shotgun.html),
[BC Outdoors — barrel length and velocity](https://www.bcoutdoorsmagazine.com/every-inch-of-barrel-will-give-you-an-extra-50-fps),
[Fox News — shotgun barrel length](https://www.foxnews.com/great-outdoors/shotgun-debate-does-length-matter),
[Field & Stream — barrel length](https://www.fieldandstream.com/guns/shotgun-barrel-length-tips),
[AccurateShooter — barrel cut-down tests](https://bulletin.accurateshooter.com/?p=4896),
[Delta Waterfowl — barrel length ballistics](https://deltawaterfowl.org/whats-new/news/shotgunning-barrel-length-ballistics/).
