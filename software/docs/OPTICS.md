# Optics: what this camera can actually see

The camera decides what the rest of the software can do, and this one is monochrome. Everything below is either
quoted from a source (innomaker CAM-MIPIOV9281 V2 user manual V1.4, `design/layout.md`, a datasheet) or marked
`[INFERENCE]` — an estimate to be replaced by a calibration measurement.

## 1. The hardware on the BOM

| Item | Value | Source |
|---|---|---|
| Sensor | OmniVision **OV9281, MONOCHROME**, global shutter | innomaker manual V1.4 |
| Resolution / pixel | 1280 × 800, 3.0 µm, 1/4" format (3.84 × 2.40 mm active) | innomaker manual V1.4 |
| Modes | 1280×800, 1280×720, 640×400 (2×2 binned) | innomaker manual V1.4 §2 |
| Frame rate | up to ~144 fps at 1280×800, 210–309 fps at 640×400 (vendor; Pi 3B achieved rate unverified) | innomaker manual V1.4 |
| Interface | 2-lane MIPI CSI-2, RAW8/RAW10, `dtoverlay=ov9281`, libcamera/Picamera2 | innomaker manual V1.4 |
| Lens | M12, f = 2.8 mm, F2.2, vendor "FOV D148 / H118", TV distortion < −17 % | innomaker manual V1.4 §3.1 |
| Mount | pod_cap, lens axis gun (Y 0, Z 17), lens tip X 675.6, board rotated 180° | layout.md "Front cap — pod_cap AS BUILT" |

Runtime default is **640 × 400 @ 120 fps, 500 µs exposure, gain 1.0** (`config/default.toml [camera]`): the binned
mode halves the data rate and doubles the per-pixel light at the cost of angular resolution. 1280 × 800 is a
`--set camera.width=1280 --set camera.height=800` away, but check `tbot bench` first — the Pi 3B is the budget.

## 2. Mono means orange is not a hue

There is no Bayer filter on an OV9281. An orange clay, a grey clay and a bird of the same luminance are the same
pixel value. The detector therefore works on **local contrast against the sky**, not colour:

1. background = heavily box-blurred image (`detector.bg_downscale = 4`, `bg_kernel = 9`), i.e. the sky gradient;
2. signed contrast = image − background;
3. threshold at `max(k_sigma × σ_noise, min_contrast)` = `max(6σ, 10 DN)` with the robust noise σ re-estimated
   every `noise_every = 10` frames;
4. `detector.polarity` picks the sign: `dark` (clay darker than sky — the usual case for an unfiltered mono sensor
   looking at a bright sky), `bright`, or `both` (the default, and what you want until a range session says
   otherwise);
5. connected components, area/aspect filtering (`min_area_px = 2`, `max_area_px = 400`, `max_aspect = 5.0` — a
   108 × 25 mm clay seen edge-on is 4.3 : 1), contrast-weighted centroid.

The HSV path (`detector.hsv_lo/hsv_hi`, `color_combine`) exists **only for colour sources** — a laptop webcam, a
colour video file. `detector.mode = auto` selects it when the frame has three channels, so on the Pi it never runs.
Do not tune the HSV numbers against webcam footage and expect them to matter on the gun; tune
`k_sigma`, `min_contrast`, `polarity` and the area limits, which do.

To rehearse the mono path on a colour webcam, set `camera.emulate_mono`:
`none` (colour), `luma` (grey) or `red` (R channel only — the closest cheap stand-in for a mono sensor behind an
orange long-pass filter, §5).

## 3. Field of view: the vendor number is for a different image circle

A 2.8 mm lens over a 3.84 mm-wide sensor cannot give 118° horizontally:

```
pinhole:       HFOV = 2 atan( (W/2) / f ) = 2 atan(1.92 / 2.8)  = 68.9°
equidistant:   HFOV = 2 * (W/2) / f       = 2 * 1.92 / 2.8 rad  = 78.6°
diagonal (equidistant, 4.53 mm):                                = 92.7°   (vendor claims 148°)
```

The lens is specified for a larger sensor (its image circle covers roughly a 1/2.5"–1/2" format); on this 1/4"
sensor you use the middle of it. Expect **70–80° horizontal**, not 118°, and note that the useful *sky* field is
smaller still because the barrel eats the top of the frame (§7).

Until `calib/intrinsics.json` exists the code uses a nominal prior (`[optics]`, `CameraModel.nominal`):

```
fx = fy = f / pixel_pitch = 2.8 mm / 3.0 µm = 933.3 px   at 1280 × 800
                          = 466.7 px                     at 640 × 400 (2×2 binned, 6 µm effective)
model = "fisheye" (equidistant base), D = 0
```

`tbot calib-intrinsics` prints the measured FOV next to the vendor claim and the measured fx next to this prior —
that report, not the datasheet, is the number to believe. The distortion spec (TV < −17 %) is why the default model
is `cv2.fisheye` rather than pinhole.

## 4. How big is a clay, and how far can this see?

Standard clay: **108 mm** (`mount.clay_diameter_m = 0.108`, ISSF/ATA). Apparent diameter is `d_px ≈ f_px · D / R`.
Face-on area is `π/4 · d_px²`; seen edge-on the disc is 108 × 25 mm, so the blob area collapses to about
`0.23 ×` that. `d_obs` is what the detector actually measures: the clay convolved with the system blur,
`d_obs = sqrt(d_px² + psf_px²)` with the default `detector.psf_px = 1.2`.

**640 × 400, fx = 466.7 px (ifov 2.14 mrad/px = 0.123°/px)**

| Range | d_px | d_obs | face-on area | edge-on area | size-range error per 0.2 px of diameter error |
|---|---|---|---|---|---|
| 8 m | 6.30 | 6.41 | 31 px² | 7.2 px² | 3 % |
| 15 m | 3.36 | 3.57 | 8.9 px² | 2.1 px² | 6 % |
| 20 m | 2.52 | 2.79 | 5.0 px² | 1.2 px² | 9 % |
| 30 m | 1.68 | 2.06 | 2.2 px² | 0.5 px² | 15 % |
| 40 m | 1.26 | 1.74 | 1.3 px² | 0.3 px² | 22 % |

**1280 × 800, fx = 933.3 px (ifov 1.07 mrad/px)**

| Range | d_px | d_obs | face-on area | edge-on area | size-range error per 0.2 px |
|---|---|---|---|---|---|
| 8 m | 12.60 | 12.66 | 125 px² | 29 px² | 2 % |
| 15 m | 6.72 | 6.83 | 35 px² | 8.2 px² | 3 % |
| 20 m | 5.04 | 5.18 | 20 px² | 4.6 px² | 4 % |
| 30 m | 3.36 | 3.57 | 8.9 px² | 2.1 px² | 6 % |
| 40 m | 2.52 | 2.79 | 5.0 px² | 1.2 px² | 9 % |

(Geometry only; `[INFERENCE]` for the error column, which is `dR/R = (d_obs/d_eff²)·δd` from
`range_from_size()` in `tbot/optics.py`.)

**SNR.** Detection needs the *peak* contrast of the blob to clear `max(6σ_noise, 10 DN)` and at least
`min_area_px = 2` pixels to survive. With the blur treated as a Gaussian of σ = `psf_px`/4 = 0.3 px, a disc of
diameter `d` keeps `1 − exp(−d²/8σ²)` of its full contrast: 89 % at d = 1.26 px, ~100 % above 2 px. So the
sensor-side limit at these ranges is **not** contrast dilution, it is **area**: at 640 × 400 the clay is a 1–2 px²
smudge beyond ~30 m, right on `min_area_px`, and edge-on it is below it. Practical reading:

- 8–20 m (skeet distances, `fire.min_range_m = 8`, `max_range_m = 40`): comfortable at 640 × 400.
- 25–40 m: switch to 1280 × 800 if `tbot bench` says the Pi keeps up, otherwise accept that the track confirms late.
- Real-world contrast is the unknown. The simulator assumes 45 DN of clay-vs-sky contrast with 2 DN of noise
  (`[sim] clay_contrast`, `noise_sigma`) — a per-pixel SNR of ~20. Measure the real value with
  `tbot calib-detector --report` on a recorded range session before trusting any of the ranges above.

**Size-based ranging.** The tracker fuses a size-derived range with the `range_prior_m = 20 ± 8 m` prior
(`tbot/vision/tracker.py`, `optics.range_from_size`):

```
d_eff = sqrt(d_obs² − psf_px²)        range = clay_diameter_m / (d_eff · ifov)
```

This is only as good as `psf_px`. The default 1.2 is `[INFERENCE]` (the pixel box alone is 1.15 px); at 40 m the
subtraction removes most of the measured diameter, so a 0.2 px error in `psf_px` is a ~15 % range error. Measure it:

```
tbot calib-detector --psf --range-m 25 --diameter-mm 108      # a clay (or any disc of known size) at a measured range
```

It writes `psf_px` into `calib/detector.json`. `size_range_sigma_frac = 0.35` (per-sample noise) and
`size_range_bias_frac = 0.10` (floor on the fused sigma, because diameter errors are frame-correlated) are the
knobs that say how much the tracker is allowed to believe it.

**Why range is the dominant lead error.** The lead the solver must predict is `ω_clay × (latency + time of flight)`,
and only the ToF part depends on range:

```
dToF/dR = 1 / v_pellet(R)
```

With the config defaults (#8 lead, 2.29 mm, 366 m/s muzzle, the drag model in `tbot/fire/ballistics.py`) the pellet
is down to ≈ 230 m/s at 20 m, so **dToF/dR ≈ 4.3 ms per metre** and ToF(20 m) ≈ 71 ms
`[INFERENCE: hand-integration of the documented model; print the model's own figure with
`tbot latency-report --range-m 20`]`. For a 20 m/s crossing clay at 20 m (ω = 1 rad/s):

| Error source | 1σ | Angular lead error |
|---|---|---|
| Range from the prior alone (σ = 8 m) | 34 ms of ToF | **34 mrad ≈ 1.9°** |
| Range from size, floored at 10 % (σ = 2 m) | 8.6 ms | 8.6 mrad ≈ 0.49° |
| Actuator jitter (`latency.jitter_ms = 6`) | 6 ms | 6.0 mrad ≈ 0.34° |
| Centroid noise (`tracker.meas_sigma_px = 0.7` at 640×400) | — | 1.5 mrad ≈ 0.09° |

Pattern core radius at 20 m is 0.33 m = 16.6 mrad (`shot.pattern_radius_ref_m 0.38 @ 22.9 m`). The prior alone is
twice the pattern — the shot is a coin toss. Size-based ranging brings it under the pattern, which is why `psf_px`
and the detector's diameter estimate matter more than any other calibration except the boresight.

## 5. Making a mono sensor "see orange": optical filters

Classic black-and-white photography trick. The clay is orange (reflects ~580–620 nm, absorbs blue); the sky is blue
(Rayleigh, peaking short). Put a **long-pass filter at ~570–600 nm** (an orange/red #21/#25 filter) in front of the
lens and the sky loses most of its signal while the clay keeps most of its own: contrast goes from "orange blob on
bright sky" (weak, and the wrong sign) to "bright blob on a dark sky" — a large, consistent `bright` polarity.

Costs and caveats:

- It is a 1.5–2 stop light loss. At 500 µs / F2.2 / gain 1.0 in daylight there is headroom; on an overcast evening
  there is not. Raise `camera.analogue_gain` before `camera.exposure_us` (§6: exposure buys motion blur).
- Foliage and ground clutter are *bright* in red/IR, so a long-pass filter also brightens the treeline. Adding an
  **IR-cut (short-pass ~650–700 nm)** on top keeps the orange band and kills the near-IR that makes leaves glow —
  i.e. a 580–650 nm bandpass. Whether the supplied M12 lens already contains an IR-cut filter is **not stated in
  the innomaker manual** `[INFERENCE: many M12 lenses sold for mono/NIR sensors ship without one]`. Check before
  buying: point the camera at a TV remote; if the LED shows up, there is no IR cut.
- The OV9281 is sensitive well into the NIR, which is exactly why an unfiltered mono image of a blue sky is
  *brighter* than your eye expects and the clay often ends up the darker object (hence `polarity = "dark"`).

**Test before cutting plastic.** Run the colour-webcam comparison first:

```bash
tbot run --source cv:0 --set camera.emulate_mono=none --stream   # what colour sees
tbot run --source cv:0 --set camera.emulate_mono=luma --stream   # what unfiltered mono sees
tbot run --source cv:0 --set camera.emulate_mono=red  --stream   # what mono behind an orange long-pass sees
```

`red` is the cheap simulation of the filtered sensor: the red channel of a Bayer camera is itself a long-pass-ish
measurement. If `red` is the only one where the clay separates cleanly from the sky, buy the filter.

**Mechanical status: there is no seat for a filter.** `pod_cap` holds the camera by its PCB (capture slots at the
bottom edge, 2 × M2 at the top); nothing of the cap lies in front of the lens — "the whole carrier is behind the
lens front plane (cap max X 655.5 vs lens tip 675.6)" (layout.md). A filter needs either an M12 screw-in disc
between the lens and the sensor (lens re-focus required) or a new ring feature on `pod_cap` in front of X 675.6,
which must stay outside the ~80° cone and out of the hose-clamp/barrel zone. **That CAD change has not been made.**

## 6. Exposure and focus

**Exposure — 500 µs default.** Motion blur in pixels is `blur = ω · t_exp · f_px`:

```
ω = 1 rad/s (20 m/s clay at 20 m), t_exp = 500 µs:
  640 × 400:   1 × 500e-6 × 466.7 = 0.23 px
  1280 × 800:  1 × 500e-6 × 933.3 = 0.47 px
```

Under a quarter of a pixel at the running resolution — blur stays below the centroid noise
(`meas_sigma_px = 0.7`) and below `psf_px`, so it does not corrupt size-based ranging. The global shutter means
there is no rolling-shutter skew to correct during a swing, which is the whole reason for this sensor. If you need
light, raise `analogue_gain` first; every doubling of `exposure_us` doubles the blur and inflates `psf_px`.

**Focus — set it at infinity.** Hyperfocal distance for the binned sensor:

```
H ≈ f² / (N · c) + f,   f = 2.8 mm, N = 2.2, c = circle of confusion = 1 pixel
  c = 6 µm (640 × 400, binned):  H = 7.84 / (2.2 × 0.006) + 2.8 = 0.60 m
  c = 3 µm (1280 × 800):         H = 7.84 / (2.2 × 0.003) + 2.8 = 1.19 m
```

Focused at infinity, everything beyond H/2 (0.3 m binned, 0.6 m full-res) is sharp. There is no reason to focus
anywhere else: the near field is the barrel, which is masked. Lock the M12 ring (thread-lock or the lens-holder
grub screw) once set — a lens that creeps changes `psf_px` and therefore every size-derived range.

## 7. Camera pose, the barrel, and parallax

From `design/layout.md` (pod_cap AS BUILT) and `config/default.toml [mount]`:

| Fact | Value |
|---|---|
| Optical centre, gun frame | X 670 (lens tip 675.6 − ~5.6 `[INFERENCE ±5]`), Y 0, **Z 17** |
| Bore axis | Y 0, **Z 49** → the lens sits **32 mm below the bore** |
| Muzzle | X 841, Z 49 |
| Barrel OD | 24.0 mm @ X 500 → 20.5 mm @ X 841 (linear) |
| Board orientation | rotated 180° about the lens axis (J1 edge up) → `camera.rotate_180 = true` |

**Rotation.** The PCB is mounted upside down, so the raw sensor image is upside down. The *source* fixes it
(`PicamSource` applies `cv2.rotate` when `camera.rotate_180`); replay and synthetic data are stored already
upright. Everything downstream — pixels, masks, boresight, overlays — is in the upright image. If you ever see the
clay falling upward, that flag is wrong, not the tracker.

**The barrel is in the picture.** Sitting 32 mm below the bore and looking forward past 171 mm of barrel, the
underside of the barrel occupies the top of the frame — roughly the top third at 640 × 400
`[INFERENCE: barrel underside 21.8 mm above the lens axis at the muzzle (Z 49 − 20.5/2 − 17), 171 mm ahead
→ 7.3° above the axis → ~59 px above the image centre, i.e. rows 0–141 of 400]`. Two mechanisms remove it:

- `optics.barrel_mask(cam, mount, cfg)` projects the barrel cylinder from the mount geometry and dilates it by
  `mount.barrel_mask_margin_px = 6`. It is automatic, it needs no recording, and it moves correctly when the
  intrinsics or the mount numbers change.
- `tbot calib-detector --auto-mask <session>` finds pixels that never change while the gun swings and writes
  `calib/mask.png`. That catches the real barrel edge, the vent rib, a stray cable tie and the FPC — whatever the
  geometry model got wrong. `optics.combined_mask()` ANDs the two.

Run the auto-mask once the pod is actually on the gun; the geometric mask alone is the fallback, not the goal.

**Parallax is handled, not ignored.** The camera is 171 mm behind and 32 mm below the muzzle, so the bore line does
not sit at a fixed pixel: `MountModel.bore_pixel(cam, range_m)` moves with range, and the solver computes the miss
angle as seen **from the muzzle**, including the shot drop. At 20 m the 32 mm offset is 1.6 mrad ≈ 0.09° — small
against the pattern, but at 8 m it is 4 mrad and the sign matters. Do not try to "zero" it with the boresight: the
boresight calibration solves for the camera's *rotation* error (`cam_rpy_deg`), and the translation comes from the
CAD numbers above. If the pod is re-printed or shimmed, update `mount.cam_pos_mm` rather than absorbing the shift
into the boresight angles.
