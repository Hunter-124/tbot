# Camera calibration, 2026-11-16 (12 mm lens): focal length + laser boresight

Result of the session that finished [FOCAL_LENGTH_HANDOFF.md](FOCAL_LENGTH_HANDOFF.md). Everything measured is in
`software/calib_data/`; the files the gun runs on are copied in `calib_data/pi_calib_2026-11-16/`.

## Results (what the Pi runs now)

| File (Pi `calib/`) | Value | Uncertainty |
|---|---|---|
| `intrinsics.json` | pinhole, **fx = fy = 4018 px at 1280x800** (2009 px at 640x400), centred principal point, D = 0 | ±0.3 % (±12 px) |
| `boresight.json` | **pitch −1.891°, yaw −1.667°**, roll 0 (bore lands on pixel 522.49, 516.70 at 1280x800, 13.75 m) | ±0.05° yaw, ±0.11° pitch (1 sd of three bore estimates) |

`tbot doctor --config config/local.toml`: intrinsics, boresight, projection all PASS (remaining WARNs are the
detector/mask, servo travel and frame-budget steps, not the camera).

Nominal was 4000 px: the lens is 0.45 % longer than its 12 mm marking, i.e. 12.05 mm. The earlier attempts
(4243 ± 190 and 3916 px, see the handoff) were limited by the reference sizes, not by the camera.

## Setup

- Gun on the floor in the hallway, pointing at a carton (the "MEDIUM"/Home Depot box) at the end. Lens focused on
  the far end (~50 ft) and locked with the lock nut before anything below; nothing touched the lens afterwards.
- Laptop wired straight to the Pi (no router). The Pi's NetworkManager profile was changed so the link stays up
  (see [pi/README.md](../pi/README.md)); stream and SSH at `169.254.20.20`.
- All pictures are `rpicam-still` 1280x800 PNG, fixed exposure (normally 20 ms, gain 4), taken over SSH (no
  camera-mode button presses). Distances are tape, **front of the lens to the target face**.

## 1. Focal length (`calib_data/fx_2026-11-16b/`)

### Reference: a US Letter sheet

The box's orange logo and the tissue boxes were tried first (sets `d45`, `e45`) and gave focal lengths between
3770 and 4620 px from the *same* pictures: the stated sizes were the problem (a tissue box turned 10° shows
8.5·cos + 12·sin ≈ 10.5 cm instead of 8.5 cm; the logo's printed edge and the box's angle are uncertain). A
sheet of **US Letter paper (279.4 x 215.9 mm)** taped flat on the box face fixed it: exact size, bright against the
carton, four straight edges. It was taped landscape, so the **215.9 mm height** is the measured size — a turn of the
box about the vertical shortens the width but not the height (the measured width/height ratio was 1.286 instead
of 1.294 at every distance: the box face is turned ~6°).

### Measurement

Each edge is located at the 50 % crossing between the paper level and the box level, averaged over the middle half
of the edge (`tools/paper.py`). The three shots at each distance repeat to ≤ 0.05 px.

| Distance (lens front -> sheet) | Height px (215.9 mm) | Width px (279.4 mm) | f = h·Z/S (z0 = 0) |
|---|---|---|---|
| 45 ft (13.716 m) | 63.77 | 81.86 | 4051 |
| 40 ft (12.192 m) | 71.44 | 91.81 | 4034 |
| 35 ft (10.668 m) | 81.33 | 104.47 | 4019 |
| 30 ft (9.144 m) | 94.87 | 122.14 | 4018 |
| 25 ft (7.620 m) | 113.87 | 146.62 | 4019 |
| 20 ft (6.096 m) | 142.49 | 183.22 | 4023 |

Fit of `1/h = (Z + z0)/(f·S)` (`tools/fit.py`):

- all six distances, free z0: f 4069, z0 +9.6 cm, residuals up to 0.44 % — the free offset jumped 40 -> 25 -> 16 -> 10 cm
  as distances were added, which is tape noise, not a real offset;
- **35–20 ft, free z0: f 4012.6 px, z0 −1.4 cm** (physical: the projection centre is just behind the lens front);
- z0 = 0, per distance: 35/30/25/20 ft agree at 4018–4023; 45 and 40 ft read 0.8 % and 0.4 % high, which is a tape
  reading ~4 in and ~2 in long at the far end.

**Adopted f = 4018 px at 1280x800**, ±0.3 %. Width-based values are ~0.6 % lower because of the box turn, which is
consistent.

Not done: shots with the sheet near the frame edges (distortion check). The lens images only the centre of a
1/1.8" circle, and D = 0 stays an assumption.

## 2. Laser boresight (`calib_data/bs46ft/`)

Gun aimed at the box at **46 ft from the lens front = 13.75 m from the muzzle** (`--range-m` is from the muzzle,
273 mm ahead of the lens front; `config/default.toml [mount]`). Bore laser in the barrel.

A bore laser is never on the bore axis: when it is turned **in the bore** the dot traces a circle whose centre is
the bore. (Rolling the whole gun does nothing: laser and camera turn together and the dot stays on its pixel —
`turn1b_*` proved it.) Gun nudges between turns do not matter for the same reason; only the target range does.

| Method | Bore pixel (1280x800) | Notes |
|---|---|---|
| single orientation (`scene_2ms.png`) | 543.65, 539.89 | 0.45° off the final answer; never use one reading |
| 4 hands-off stills at quarter turns (`t0..t3_20ms`) | 524.09, 508.47 | circle r 39.8 px (0.57°); 4 points barely over-determine a circle |
| twist video 1 (`twist_dots.csv`, `twist_fit.png`) | 524.59, 522.83 | fingers pushed the laser sideways, 11 px scatter |
| twist video 2 (`twist2_dots.csv`, `twist2_crop.mp4`, `twist2_fit.png`) | 518.79, 518.81 | 290 frames, full ring, fingertips, r 29.9 px, 7.5 px scatter |
| **mean of the last three (adopted)** | **522.49, 516.70** | sd 3.2 / 7.4 px = 0.05° / 0.11° |

The laser is ~0.5° off-axis and its seating is not repeatable (rest positions after handling do not lie on one
circle), so a continuous slow twist with fingertips, filmed, is the best method; anything that pushes it sideways
ruins it.

Solved with `tbot calib-boresight --image ... --pixel 522.49,516.70 --range-m 13.75` after `intrinsics.json` was
written (fx 4000 -> 4018 changed the angles by < 0.01°). Older solves are kept: `boresight.prev.json` (same pixel,
nominal fx) and `boresight_46ft_single.json`.

Video clips (MJPEG, 118–147 MB each) are too big for GitHub and are not committed (`calib_data/.gitignore`); they
are on the Pi in `~/TBOT/software/calib_data/bs46ft/`. The per-frame dot tracks (`*_dots.csv`) and a 400x300 crop
around the bore (`twist_crop.mp4`, `twist2_crop.mp4`) are.

## Still to do

1. **Second-distance boresight check** (RANGE_DAY §A step 4): a twist video at ~20 ft, `--check`. `PASS` within
   0.15° confirms the mount geometry (camera 54 mm under the bore).
2. Patterning on paper (CALIBRATION.md §5b) for the ballistic offset.
3. Detector + mask, servo endpoints, `tbot bench` (doctor WARNs).
4. Redo intrinsics + boresight if the lens is ever touched.

## Files

| Path | What |
|---|---|
| `calib_data/pi_calib_2026-11-16/` | copies of the Pi's `intrinsics.json`, `boresight.json` and the superseded solves |
| `calib_data/fx_2026-11-16b/` | `p45..p20_*.png` paper sets, `d45`/`e45` logo + tissue-box attempts, `log.md`, `tools/` (`fxset.sh` capture, `paper.py` edges, `fit.py`) |
| `calib_data/bs46ft/` | laser stills, dot tracks, fit overlays, `tools/` (`shot.sh` capture + dot, `vid_dots.py`, `shift.py`) |
