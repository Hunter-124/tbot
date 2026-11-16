# Range day: laser boresight, then clays

Short, ordered, and everything runs headless over SSH (`ssh pi@tbot.local`; at the range plug an Ethernet
cable straight from the laptop into the Pi, `tbot.local` resolves over the link). Details behind each step
are in [CALIBRATION.md](CALIBRATION.md); state of the rig is in [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md).

Gun **unloaded** for everything in part A. The arming interlock is software only: open the battery lever
connector whenever the actuator must be safe.

## A. Laser boresight (at home, 10 minutes)

The camera must be free: stop anything using it (`tbotctl stop`, Ctrl-C a `tbot focus --stream`).

1. **Lock the lens first.** Focus (`tbot focus --stream`), fit the lock nut, then do not touch the lens
   again: boresight, intrinsics and the mask all move with it.
2. Rest the gun so it cannot move (bags, vice with padding). Put a plain, matt target 10-25 m away — a
   sheet of paper on a wall. Longer is better. Measure muzzle-to-target distance to within ~5 %.
3. Fit the bore laser and run, from `~/TBOT/software` on the Pi:

   ```bash
   tbot calib-boresight --laser --laser-turns 4 --range-m 15
   ```

   It finds the dot by itself. Between measurements it asks you to turn the laser a quarter turn in the bore:
   a bore laser is never exactly on the bore axis, and the centre of the circle the dot traces is the bore.
   The printed "dot circle radius" is how wrong a single reading would have been.
   - "no laser dot": the room is too bright or the exposure too short — add
     `--set camera.exposure_us=4000 --set camera.analogue_gain=4`, dim the lights.
   - "ambiguous" / "glare": add `--laser-bg` (it asks you to block the beam for a background frame).
   - A lens with an IR-pass filter cannot see a red or green laser: boresight before fitting such a filter,
     or use an IR laser.
   - Look at `calib/boresight_laser.png` (`scp pi@tbot.local:TBOT/software/calib/boresight_laser.png .`):
     green circles are the dots, the magenta cross is the solved bore, cyan is the previous calibration.
4. **Check at a second distance** (move the target or the gun; e.g. 8 m and 20 m):

   ```bash
   tbot calib-boresight --laser --laser-turns 4 --range-m 8 --check
   ```

   `PASS` (within 0.15 deg) means the mount geometry (camera 54 mm under the bore) and the calibration
   agree. A `FAIL` that grows as the target gets closer points at the mount numbers in
   `config/default.toml [mount]`, not at the boresight.
5. Undo, if ever needed: `cp calib/boresight.prev.json calib/boresight.json`.

Roll is not measured by the laser (it stays at its previous value, 0). It matters little for a camera on
the bore's vertical plane; a plumb line (`--pixel2`) adds it.

## B. Before leaving

```bash
tbotctl install        # once
tbotctl preflight      # tbot doctor + hardware check: fix every FAIL
```

Bring: laptop + Ethernet cable, charged pack + spare, the mic is not needed. Lens cap off.

## C. At the range

1. Power up, gun on the rack, **still**. `tbotctl start` (dry run). Leave it still for ~5 s: the gyro
   re-zeroes itself whenever the gun rests (the log does not mention it; `calib-imu bias` is no longer
   needed per session).
2. Open `http://tbot.local:8080/` on the laptop: the bore marker, detections and tracks are drawn on the
   stream. Check exposure against the real sky (`camera.exposure_us` in `config/local.toml`).
3. **Dry passes first** — unloaded gun, real clays. Hold the arming button, swing through the clay, release.
   Each press is saved under `sessions/`. The log (`tbotctl log`) prints `FIRE (dry run)` with the miss angle
   and P(hit) when it would have shot. Do a dozen. If it never says FIRE, `tbot explain --source
   replay:sessions/<one>` says which gate stopped it.
4. **Live**: `tbotctl start live` (asks you to type LIVE). One clay, one shell in the gun at first.
   After each shot the log prints
   `shot N: recoil 104.2 ms after the command (budget 101.0 ms, +3.2)` — the measured trigger delay — or
   `NO RECOIL` if the command did not fire the gun (lever slipped, safety on).
5. After ~5 live shots: `tbotctl stop`, then

   ```bash
   tbot latency-recoil            # shows the measured delay vs the calibration
   tbot latency-recoil --write    # stores it (needs 5 shots); then tbotctl start live again
   ```

   10 ms of delay error is 20 cm on a crossing clay, so do this early.
6. `tbotctl stop` before powering off (lets the last session finish writing). A power cycle always comes
   back in dry run; live mode never survives a reboot.

Watch the status line for heat (`!THR`, warning at 75 C) and `UV` (pack sagging under the servo).

## D. Bring the data home

```bash
scripts/deploy.sh pull         # sessions, calib and local.toml into software/pi-pulls/
```

The recorded sessions are what the detector thresholds, the mask and `psf_px` get tuned on
(`tbot calib-detector`), and they replay through the whole pipeline on the laptop.
