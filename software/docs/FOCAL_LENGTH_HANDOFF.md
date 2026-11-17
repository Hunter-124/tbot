# Focal-length calibration handoff (12 mm lens, no chessboard)

> **Finished 2026-11-16 — see [CALIBRATION_2026-11-16.md](CALIBRATION_2026-11-16.md):** fx = 4018 px at 1280x800
> (US Letter sheet, 6 distances), laser boresight pitch −1.891°, yaw −1.667°. The rest of this file is the
> state before that session.

State as of 2026-11-16 evening. Everything needed to continue on another computer is in the repo:
this file, the pictures and the measurement scripts in `software/calib_data/fx_2026-11-16/`.

## 1. Why there is no chessboard / ChArUco calibration

The camera now has a **JETSECAM 12 mm M12 lens** (1/1.8", IR-cut filter, sold as "no distortion"; BOM and
`docs/OPTICS.md` §3). At the ranges it is focused for, a printed board is useless:

- Hyperfocal distance is about 12–15 m. Focused for clays (40–70 ft), the board is only sharp beyond ~25 ft,
  where 25 mm squares are ~7 px wide: too small for corner detection.
- A board at arm's length is completely blurred at the correct focus.

So intrinsics are reduced to what matters for this lens:

| Parameter | How it is handled |
|---|---|
| Distortion | Assumed zero (pinhole, D = 0). The lens images a 1/1.8" circle (~9 mm) and the sensor only uses the middle 4.5 mm, where a "no distortion" lens is straight to well under 1 px `[INFERENCE]`. |
| Principal point (cx, cy) | Assumed at the image centre. Any offset is absorbed by the boresight calibration (laser), which measures where the bore points in pixels. |
| Focal length f | **The one number to measure.** Sets the angle per pixel (lead angles) and size-based range. Printed "12 mm" on hobby lenses is good to a few percent. |

Nominal: f = 12 mm / 3 µm = **4000 px at 1280x800**, **2000 px at 640x400** (the runtime size). HFOV 18.2°,
VFOV 11.4°. A 108 mm clay is ~14 px at 15 m and ~7 px at 30 m (640x400).

## 2. Current configuration (what the gun runs today)

- `config/local.toml` on the Pi (not in git): `[optics] model = "pinhole"`, `focal_length_mm = 12.0`.
  Backup of the old file: `config/local.toml.bak-2p8mm`.
- `config/default.toml` still describes the old 2.8 mm fisheye on purpose: the simulator and tracker tests
  are built on that geometry (switching it fails 12 tracker tests).
- **No `calib/intrinsics.json`** exists, so `CameraModel.nominal()` is used (f from `[optics]`, centred, D = 0).
  `tbot doctor` warns about the missing file; that warning is expected until a measurement is written.
- `calib/boresight.json` is **stale**: it was measured with the 2.8 mm lens. Redo it (laser, `docs/RANGE_DAY.md` §A)
  after the lens is focused and locked.

## 3. The method: known-size target at measured distances

Pinhole projection: `size_px = f * S / (Z + z0)`.

- `S` = real size of a feature (m), `Z` = measured distance (m), `z0` = unknown offset between where the tape
  was zeroed and the camera's projection centre (and between the target face and the feature).
- Rewrite as `1 / size_px = Z / (f*S) + z0 / (f*S)`. A straight-line fit of `1/size_px` against `Z` over
  several distances gives **slope = 1/(f*S)** and therefore f, **independent of z0**. Do not compute f from a
  single distance unless the distance is known to better than 1 %.
- Measure sizes from **edges**, not corners. The edge finder (`fx_edges.py`) averages a band of rows,
  takes the gradient and fits a parabola to its peak (sub-pixel). For blurred pictures, use the **50 % crossing
  between the inside level and the outside level** instead (`fx3_measure.py`). A symmetric blur does not move
  the 50 % point, but it needs an even background ~1 blur-width on each side of the edge.
- Draw every detected edge on the picture and look at it before trusting the number (`fx_more.py` writes
  `ov_*.png` overlays). Patterned boxes and busy backgrounds produce false edges.

Precision seen so far: the **pictures repeat to ~0.2 %** within a distance group. **Distances are the limit:**
residuals of 2–5 % per distance, i.e. tape readings to the nearest foot and targets a few cm off the measured
point. The fit with a free z0 partly absorbs this; better distances help more than more pictures.

## 4. Reference material in the pictures

All pictures: 1280x800 lossless PNG from camera mode (`tbot snap`), `snaps/snaps.jsonl` has exposure, gain,
brightness and sharpness per shot. Group shots by `wall_time`: each distance is a burst ~2 s apart, with
30–60 s gaps between distances. Fewer pictures than button presses were saved: presses during a save are
dropped.

### Set 1: tissue boxes, lens focused at ~10 ft (snaps 0002–0020)

| Snaps | Distance (to the camera) |
|---|---|
| 0002–0004 | 28 ft |
| 0005–0007 | 20 ft |
| 0008–0009 | 15 ft |
| 0010 | 10 ft — bad data, skip |
| 0011–0012 | 10 ft |
| 0013–0015 | 40 ft |
| 0016–0017 | 35 ft |
| 0018–0020 | 47 ft |

Targets: tissue boxes, stated **8.5 x 12 x 22.5 cm**, stacked in front of a "MEDIUM" carton (size unknown):

- **Right upright box** (dark, oval opening facing the camera): shows its 12 cm x 22.5 cm face. Cleanest
  feature: edges against the carton at 10–35 ft.
- **Left upright box** (diamond pattern): the pattern makes the edges ambiguous.
- **Lying box** ("Ultra Soft", Great Value): shows a 22.5 x 12 cm face at 10–28 ft.
- At 40 and 47 ft the arrangement changed (boxes turned, ovals squashed, dark background): not usable for
  widths. The **oval openings** are bright and crisp at every range, but their real size must first be
  calibrated against a box width at short range.
- Boxes lean back against the carton, so **heights are foreshortened** (~3 %): use widths only.
- The three boxes are different brands and may not all be 12 x 22.5 cm.

Result (right upright box, 12 cm, distances 10/15/20/28/35 ft, free z0):
**f = 4243 ± 190 px** at 1280x800 (z0 = +11 cm). Lying box 4653 ± 269, left box 4673 ± 201
(sizes uncertain). Leave-one-out on the right box: 4122–4538. Inconclusive versus the nominal 4000.

Focus check (10–90 % edge width, right box edge): 1.4 px at 10 ft, 2.4 px at 20 ft, 4.6 px at 35 ft → the
lens was focused near 10 ft.

### Set 2: Home Depot box, lens past infinity focus (snaps 0021–0037)

**Shot FAR → NEAR** (the box grows through the sequence), 5 ft steps. The operator recalled 10 → 45 ft; the
pictures say otherwise.

| Snaps | Distance |
|---|---|
| 0021–0023 | 45 ft |
| 0024–0025 | 40 ft |
| 0026–0027 | 35 ft |
| 0028–0029 | 30 ft |
| 0030–0031 | 25 ft |
| 0032–0033 | 20 ft |
| 0034–0035 | 15 ft |
| 0036–0037 | 10 ft |

Distance is measured **from the front of the camera lens to the front face of the Home Depot box** (the face
with the 7x7" logo; the other boxes are pressed against it).

Targets:

- **Home Depot box, 51.5 cm across** (the note said "51.5 mm"; the pictures show 51.5 cm: ~660 px at 10 ft).
- **7 x 7" (177.8 mm) orange Home Depot logo** on the front face: meant as the most precise reference, but it
  is **invisible to this camera**. Orange on orange-brown cardboard has no contrast in mono behind an IR-cut
  filter. Use a black-on-white or white-on-black target instead (see §5).
- The round bright spot on the box is a reflection blurred into a ~60–80 px disc, not a target.

Every picture in this set is out of focus by **~46–52 px** (10–90 % edge width), slightly more at near range →
the lens had been screwed past infinity focus (~0.3 mm, estimated half a turn or more). Only the 10 ft pair
measures cleanly: box width **661.6 / 661.9 px → f ≈ 3916 px**, but valid only at that wrong focus. Turning
the lens back changes f by up to ~2.5 %, which puts the in-focus value at or near 4000. 15–20 ft scatter by
4–10 % between shots (blur mixes in door frames and a person walking); 25–45 ft have no usable contrast
(box in front of a dark doorway).

**Decision so far: keep the nominal 12 mm (4000 / 2000 px), uncertainty ±2–3 %. No `intrinsics.json` written.**
A ±2.5 % scale error moves a clay near the bore line by under a pixel; it mostly affects size-based range.

## 5. How to finish it (next shoot)

Do these in order: the lens position changes f, so measure only after the final focus.

1. **Focus, then lock.** `tbot focus --stream` (now caps auto-exposure at 20 ms / gain 4 for a clean score).
   Aim at detail ~50–60 ft away, turn the lens a few degrees at a time, find the peak `focus` number, lock the
   nut, confirm the number holds. Screw **in** = focus farther; ~1/13 turn moves focus from 10 ft to 55 ft.
2. **Target with real contrast in mono**: two strips of black tape on a white wall (or white on black)
   exactly **1.000 m apart**, outer edge to outer edge, or a printed black square of known size (≥ 30 cm).
   Even, flat background ≥ 10 cm around each edge. Avoid orange/brown targets and shiny tape.
3. **Distances**: 3–4 distances spanning ~20 to 60 ft (e.g. 20, 30, 45, 60 ft), measured as precisely as
   possible (laser measure ideal), always to the same points: front of the lens to the target face.
   Write down which snap numbers belong to which distance (or just keep ~30 s between distances).
4. **Pictures**: camera mode (`tbotctl camera on`, arming button = picture). 3 shots per distance with the
   target centred, plus a pair with it near the left and right edges (checks the zero-distortion assumption).
   Wait ~1 s between presses.
5. **Fit**: measure the gap / square width in each picture with `fx_edges.edge` (sharp pictures), fit
   `1/width` against distance (§3), f = slope⁻¹ / S. Compare the edge-of-frame shots with the centre shots: a
   difference above ~0.5 % means distortion is not negligible.
6. **Write it** if the fit is tight (residuals < 1 %, f uncertainty < 1 %): `calib/intrinsics.json` on the Pi,
   at the runtime size:

   ```json
   {"model": "pinhole", "width": 640, "height": 400,
    "K": [[F_HALF, 0, 319.5], [0, F_HALF, 199.5], [0, 0, 1]],
    "D": [0, 0, 0, 0, 0],
    "source": "focal-length fit, <date>, <target>, distances ...", "fx_full_res_px": F_FULL}
   ```

   with `F_HALF = F_FULL / 2`. `CameraModel.from_calibration` rescales it to any other size with the same
   aspect. Then `tbot doctor --config config/local.toml` (the intrinsics warning disappears).
7. **Then the laser boresight** (`docs/RANGE_DAY.md` §A) and `tbotctl preflight`.

## 6. Tools and where things are

| What | Where |
|---|---|
| Camera mode (button = picture, power-up service) | `tbot snap`, `tbotctl camera on|off`; pictures in `~/TBOT/software/snaps/` on the Pi |
| Fetch pictures | `scp "pi@tbot.local:TBOT/software/snaps/*" <dir>` (SSH: dedicated known_hosts, see `pi/README.md`) |
| Focus aid | `tbot focus --stream` → `http://<pi>:8080/` (stop camera mode first: `tbotctl camera off` or `sudo systemctl stop tbot-camera`) |
| Measurement scripts | `calib_data/fx_2026-11-16/`: `fx_edges.py` (sub-pixel edge), `fx_measure.py` and `fx_more.py` (set 1, run from that folder), `fx3_measure.py` (set 2, 50 % crossings) |
| Camera model | `tbot/optics.py` `CameraModel.nominal / from_calibration` |
| Optics notes | `docs/OPTICS.md` §3 (12 mm numbers) |

Pi state at handoff: camera mode **disabled** at power-up (range state: nothing auto-starts), `tbot focus
--stream` running for refocusing, auto-exposure on by default (pipeline cap 500 µs, then gain to 8).
