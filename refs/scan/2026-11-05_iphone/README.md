# 2026-11-05 iPhone capture: Weatherby Element on the ChArUco mat

Copied from the phone over AFC (`afc://00008150-00013d5226db401c/DCIM/100APPLE/`); byte sizes match the device.

| File | Recorded (local) | Length | Frames | Views |
|---|---|---|---|---|
| `IMG_0466.MOV` | 2026-11-05 15:36:18 | 23.0 s | 1378 | overhead and oblique orbit (supplementary; not named in the request, recorded one minute before 0467 in the same setup) |
| `IMG_0467.MOV` | 2026-11-05 15:37:18 | 5.1 s | 307 | short pass along the receiver and stock |
| `IMG_0468.MOV` | 2026-11-05 15:37:32 | 27.9 s | 1671 | side passes with close-ups of the receiver and trigger guard |

All three: iPhone 17, iOS 26.6.1, main camera `iPhone 17 back camera 5.96mm f/1.6`, 3840x2160 landscape (no rotation),
~60 fps, HEVC Main10 Dolby Vision 8.4 / HLG (BT.2020). 35 mm-equivalent focal length 26 mm (0467) or 27 mm (0466, 0468),
which suggests about 2650-2750 px at 3840x2160. SfM solved 3450-3510 px instead: the electronic stabilisation crops
the frame, so treat the metadata only as a loose prior.

Scale reference in frame: the Chudware ChArUco mat, 7x10 squares of 25.0 mm, 18.0 mm DICT_5X5_100 markers.

- `metadata_summary.json`: per-clip container, stream and lens metadata
- `contact_*.jpg`: evenly spaced frames with frame numbers
- `raw_metadata/`: `ffprobe` JSON, the QuickTime `moov` atoms, AFC file info and the raw timed-metadata tracks
  (track 4 is Apple's cinematic/maker data, track 6 is a per-frame 144-byte record, track 7 per-frame scene illuminance)

The gun lay on its gun-left side with the receiver top facing away from the camera, so the gun-right side, the side
profile and the trigger guard bow are well covered; the gun-left side, the receiver/rib top, the forend ahead of
X ~250, the barrel and the butt end are not.

## Mesh (`mesh/`)

Made with Chudware's overhauled `reveng_splat_to_mesh` (Gaussian median-depth TSDF) from one reconstruction of all
three clips: 407 frames, all registered (pycolmap, 1.23 px mean reprojection), Brush splat 30 000 iterations
(567k Gaussians), ChArUco scale 55.834 mm/unit (41 corners from 323 views, 0.64 mm RMS fit). Conversion: 120 views at
960 px, 0.54 mm voxels, 248 s.

| File | Contents |
|---|---|
| `gun_scan_gunframe_mm.ply` | colored mesh in the TBOT gun frame (mm), registered to `refs/gun_body` (690k triangles) |
| `gun_scan_gunframe_mm_400k.stl` | the same, decimated to 400k triangles for CAD import |
| `gun_scan_matframe_mm.ply` | the mesh in the ChArUco mat frame (mm), before registration |
| `gun_scan_report.json` | scale, registration transform, scan-vs-`gun_body` deviation per 25 mm station, side profile per 10 mm, coverage |
| `gun_scan_vs_gun_body.png` | side (X-Z) and plan (X-Y) overlay: scan white, `gun_body` red |

Accuracy evidence:

- Two independent reconstructions (A = IMG_0468, B = IMG_0466 + IMG_0467, each with its own SfM, splat and ChArUco
  scale; scales agree within 0.32 %) placed by the mat alone agree to a median of 1.00 mm in their overlap, 50 % of the
  surface within 1 mm and 76 % within 2 mm. The previous converter, on the same splats, gave 1.55 mm, 34 % and 61 %.
  Agreement is best near the mat (0.9 mm) and falls off toward the ends of the gun (2-2.5 mm). Smoothing did not
  change it, so it is low-frequency shape difference, most likely SfM drift between the captures, not meshing noise.
- Bottom of the side profile vs `gun_body`: within about ±1 mm over the stock (X -240..-190) and the trigger guard
  (X -10..50); the guard bow bottom is Z -16.4 against the measured -16.5.
- Gun-right surface vs `gun_body`: receiver (X -50..175) within ±0.9 mm. Differences worth checking with calipers:
  rear stock X -250..-175 is 1.7-2.3 mm inside the model, the grip X -125..-50 is 1.8-2.2 mm outside it, the receiver
  bottom X 80-140 is about 2 mm lower than the model, and the forend start X 190-220 is about 4 mm lower.
- Not verifiable here: widths (one side unseen), the receiver/rib top, and the guard width. The registration is weakly
  constrained in Y because one side is missing; allow 1-2 mm there.

The SfM datasets, splats and scripts to rerun or re-register are in `scratch/gunscan2/` (`recon/C` is this mesh;
`recon/A`, `recon/B` the independent pair).
