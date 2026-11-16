# Lens lock nut (M12 x 0.5 jam ring)

Locks the focus of the camera's S-mount lens. The nut runs on the exposed lens thread and jams against the front
face of the lens holder boss; the split version is then pinched onto the thread with one M2 screw, so it holds even
if the printed 0.5 mm thread is poor. Units mm. Nothing here depends on the lens barrel shape, only on its
M12 x 0.5 thread.

| File | What |
|---|---|
| `parts/lens_lock_nut.step+` | primary: split clamp nut with pinch screw |
| `parts/lens_lock_nut_plain.step+` | fallback: plain knurled jam nut |
| `export/STEP/lens_lock_nut*.step`, `export/STL/lens_lock_nut*.stl` | exports, modelled in print orientation (front face on Z 0) |
| `design/renders/lens_lock_nut.png`, `lens_lock_nut_plain.png`, `lens_lock_nut_thread_section.png` | renders |

## Dimensions

Part frame: lens axis = Z, front face (toward the lens head) at Z 0, seat face (against the holder) at the top.

| | Clamp nut | Plain nut |
|---|---|---|
| Thread | M12 x 0.5 RH internal, oversize: minor D11.76, major D12.30 | same, tighter: minor D11.66, major D12.20 |
| Threaded length | 3.4 (6.8 turns), 0.45 x 45 deg lead-in at the front face | 2.2 (4.4 turns), same lead-in |
| Body | D17.6 x 4.0, 12 grip grooves R0.6 | D18.0 x 3.0, 18 grip grooves R0.6 |
| Seat nose | D14.0 OD / D12.4 ID x 1.2 (unthreaded) | same |
| Overall thickness | 5.2 | 4.2 |
| Pinch ear | 9.0 wide, out to 12.7 from the axis; swing radius 13.9 | none |
| Slit | 1.0 wide, through ear, ring and nose | none |
| Pinch screw | axis 10.3 from the lens axis, 2.0 behind the front face; D2.3 clearance x 3.0 (head side), D1.7 pilot x 5.0 | none |
| Hinge | R1.5 relief opposite the slit, 1.45 wall left over the thread | none |
| Model volume / PLA mass | 630 mm3 / 0.8 g | 440 mm3 / 0.5 g |

The ISO basic M12 x 0.5 internal thread is minor D11.459 / major D12.000; the lens (external) is at most D12.0 over
the crests and about D11.39 at the roots. The clamp nut is modelled 0.30 oversize so that it spins freely on a
printed hole that comes out 0.1 to 0.2 small; the 1.0 slit can close the bore by up to about 0.3 on diameter when
pinched. The plain nut is 0.20 oversize, so it should run stiff and be cut to size by the metal lens.

## Fit against `camera_cover`

Checked with both parts in one document (cover in its own frame, nut seat face on the holder boss face at
X 531.7): no interference, tightest gap 0.30.

- The cover's front face is X 532.3 and its boss hole is D14.6 with a 0.6 lead-in chamfer. The holder boss (D14)
  ends at X 531.7 in the model, so the face the nut has to reach is 0.6 below the cover front, inside the hole.
- That is why both nuts have the D14.0 nose: 0.30 radial clearance in the D14.6 hole, 1.2 tall, which leaves
  0.4 between the nut body and the cover front. The body and ear are entirely in front of the cover, so their
  outside diameter is not limited by it.
- Lens side: the camera reference has 7.9 of thread between the holder face and the D14 lens head (an estimate,
  and it changes with focus). The clamp nut uses 5.2 of it (2.7 spare), the plain nut 4.2 (3.7 spare).
- The camera screw heads are 17.9 or more from the lens axis and about 7 behind the nut body, so the ear clears
  them at any angle.

## Measure before printing

1. Depth from the cover front face down to the holder boss face, with the cover fitted. Modelled 0.6. The nose
   works from flush up to about 1.0; deeper than 1.2 and the nut lands on the cover instead of the holder.
2. Printed diameter of the cover's boss hole: needs to be at least about D14.3 for the D14.0 nose.
3. Exposed thread between the holder face and the lens head at the focus position: at least 5.5 for the clamp
   nut, 4.5 for the plain nut. Repeat for the IR lens.
4. Lens thread outside diameter: 11.85 to 12.00 for M12 x 0.5.
5. The thread runs all the way to the rear end of the lens (the nut goes on from the rear).
6. Head of the M2 self-tapper: D4.0 or less, so it stays inside the 4.0 thick ear.

## Printing

- PLA+ on the Ender-3, 0.4 nozzle, as for the other parts, but with **0.12 layers** (0.5 pitch is only 2.5 layers
  at 0.2). 3 walls, 100 % infill, outer walls slow (about 20 mm/s).
- Orientation as modelled: front face on the bed, nose up. No supports. The two pinch-screw holes print
  horizontally and bridge on their own.
- Elephant's-foot compensation about 0.15, or scrape the first layer of the bore: the thread starts at the bed.
- Not sliced: PrusaSlicer is not installed on this machine, so the printability rows of the DFM check did not
  run. Print two or three; they weigh under a gram.
- If the clamp nut is too tight or too loose on the lens, scale X/Y by 1 % steps in the slicer (0.12 on the
  thread diameter per step), leaving Z alone.

## Hardware

| Qty | Item | Note |
|---|---|---|
| 1 | M2 x 8 pan head self-tapping | same screw as the camera corner screws; about 4 of engagement in the D1.7 pilot |

## Fitting

1. Unscrew the lens. Thread the nut on from the rear of the lens, flat face toward the lens head, nose toward
   the camera. Run it up near the head. If it starts hard,
   work it back and forth a turn at a time; the metal thread cleans up the plastic one.
2. Clamp nut: start the M2 screw in the ear a few turns, not tight.
3. Screw the lens back into the holder and focus (`tbot focus`).
4. Hold the lens by its head and run the nut down until the nose seats on the holder face. Tighten it finger
   tight against the holder while holding the lens, and check the focus did not move.
5. Clamp nut: tighten the pinch screw until the slit visibly closes a little and the nut no longer turns by
   hand. Do not strip the plastic pilot.
6. Check focus again, then do the lens calibration. Removing the lens to fit the nut changes focus, so
   intrinsics and boresight come after this, not before.

To refocus: slacken the pinch screw, back the nut off a quarter turn, focus, and repeat from step 4.

## Checks run

- Both parts: closed manifold solids, valid B-rep, no self-intersection (geometry health); `verify_battery`
  `is_manifold` and `unit_sanity` pass. The full default battery timed out on the threaded bodies and was not
  completed.
- Thread: the crest faces are cylinders at R5.88 (clamp) and R5.83 (plain); the major diameter is the thread
  feature's parameter and the section through the axis shows the 60 deg grooves at the expected depth.
- DFM (FDM): flags the 0.45 lead-in as below the 0.8 minimum feature size (intended) and the sharp thread roots
  and steps as internal corners; overhang, bridge, wall and bed rows not evaluated (no slicer).
- Not tested: a print, the fit on the real lens, or holding under recoil.
