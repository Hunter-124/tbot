# Re-zero after the lens drop (2026-11-18)

The camera pod was dropped lens-first on a table. Focus was knocked out and the boresight moved.

## Focus

`s0_5ms.png` / `s0_auto.png`: just after the drop, the 20.5 ft scene was soft everywhere (compare `../bs20ft/s0.png`).
Refocused by hand on `tbot focus --stream` with exposure locked for the indoor light
(`--set camera.fps=30 --set camera.auto_exposure=false --set camera.exposure_us=15000 --set camera.analogue_gain=1.0`;
a one-run override, production auto exposure unchanged). `s1_15ms.png` / `s1_5ms.png`: after refocus.

## Boresight at 20.5 ft (this folder)

20.5 ft lens front -> target = 5.975 m from the muzzle. Laser turned in the bore, 15 fps, 5 ms / gain 2.

| Run | Result |
|---|---|
| `run1/` (`twist.mjpeg`) | rejected: laser loose in the bore, dot cloud instead of a ring (scatter 19 px), dot lost at ~25 s |
| `twist2.mjpeg` -> `twist_dots.csv`, `twist_fit.png`, `twist_fit.txt` | bore pixel **(475.74, 461.52)**, r 22.1 px, scatter 4.3 px, sd 0.3 px, 551 frames, 12/12 sectors |

`tbot calib-boresight --pixel 475.74,461.52 --range-m 5.975 --check` against the 2026-11-16 zero:
0.667 deg left, 0.512 deg above, **0.840 deg FAIL** (29 cm at 20 m). Re-solved:

| | pitch | yaw |
|---|---|---|
| 2026-11-16 (`boresight_2026-11-16_predrop.json`) | -1.891 | -1.667 |
| **2026-11-18** (`boresight_2026-11-18.json`, now on the Pi) | **-1.379** | **-2.334** |

## Second distance: 35 ft (`../bs35ft_2026-11-18/`)

35 ft lens front = 10.395 m from the muzzle, 10 s twist (`twist35.mjpeg`, not committed).
The dot is faint at this range and the window blinds outshine it in a wide window, so the fit uses a
narrow one: `python twist_fit.py twist35.mjpeg 99999 475 465 70`.
Bore pixel **(476.49, 474.20)**, r 33.1 px, scatter 3.8 px, sd 0.5 px, 145/149 frames, 12/12 sectors.
`--check` with the new zero: 0.011 deg right, 0.024 deg above, **0.027 deg PASS** (0.9 cm at 20 m).

Not redone: fx (intrinsics) after the refocus. The camera shakes with the gun while the laser is twisted; that does
not matter, the laser and camera move together and only the twist moves the dot.

`twist_fit.py`: `python twist_fit.py clip.mjpeg [last_frame] [cx cy R]` -> dot per frame (DoG peak), robust circle fit.
Videos (`*.mjpeg`) are gitignored; they are on the Pi under `~/TBOT/software/calib_data/`.
