# TBOT Console

The app for setting up, calibrating and reviewing the gun. It runs **on the gun** (`tbot console`, port 80, started
at power-up) and opens on a laptop as a desktop app. **No internet** is needed, ever: plug the Pi's Ethernet into
the laptop, power the gun on, open **TBOT Console**.

## Using it

1. Once per laptop: install Python for Windows (python.org) and run
   `powershell -ExecutionPolicy Bypass -File software\desktop\install_shortcut.ps1`. It puts **TBOT Console** on the
   Desktop and in the Start menu.
2. Plug the cable, power the gun (about 50 s to boot), open the app. It shows "Connect your TBOT" until the gun
   answers, then finds it by itself at `169.254.20.20` (or `tbot.local`). Windows needs up to a minute to give a
   freshly plugged cable its own `169.254.x.x` address.
3. Any browser on the same cable or network works too: `http://169.254.20.20/` (also a phone on the gun's network).

The gun's clock (no battery, no NTP at the range) is set from the laptop when the app connects (System page,
"Sync on connect").

| Page | What it does |
|---|---|
| Overview | Pipeline state and mode, live picture, temperature / storage / memory / power, calibration summary, latest presses |
| Live view | The pipeline's own overlay stream, armed/safe, exposure and AE state, targets, decision, P(hit) |
| Sessions | Every recorded press. Player with the shot marked on the timeline, *Jump to shot*, 0.25-2x speed, frame stepping, loop around the shot. Mark **Hit / Miss / No shot**, star, notes (saved on the gun in `console.json`). Filters: Shots, Hits, Starred, To review. *Score* runs `tbot score --image` (arrival frame per shot); *Render clip* runs `tbot clip` (live-mode presses have none). Download clip / telemetry / summary; delete |
| Sessions: previews | Every press gets a **preview** (thumb.jpg): its key moment (the shot, else the best-aimed frame, else the best tracked frame) drawn like the clip, with the target magnified in a corner and a SHOT / BEST AIM label. The preview picture is brightness-stretched to its own range so presses filmed in dim light stay recognisable (display only; the recording is untouched). Hover a preview to play the second around that moment; list or grid view |
| Calibration | Status of intrinsics, boresight, servo, trigger delay, IMU with dates |
| Camera & focus | Raw camera at 1280x800 or 640x400 (pipeline stopped): sharpness with % of peak, quadrants, histogram, clipping, auto or manual exposure/gain, save a still |
| Boresight | Wizard: distance (ft or m from the lens front; converted to the muzzle range), click the laser dot, 6-15 s twist capture, live ring fit with grade, then **Check zero** (`calib-boresight --check`) or **Save as new zero** (old `boresight.json` backed up in `calib/history/`) |
| Trigger servo | `tbot jog servo` with buttons: big armed indicator, ±5/10/50 us, go to pulse, mark low/high/rest/fire, one test fire cycle, hold time, save |
| Settings | Every key of `config/default.toml` with its comment as help, units, estimate badges, search, changed-only; writes `config/local.toml` |
| Diagnostics | `tbot doctor`, `tbot hwcheck`, `tbot latency-report` with coloured output; the pipeline log, following |
| System | Start / stop in dry, test or live; power-up mode (field mode); clock; restart / shut down; theme (light for sunlight) |

## Clips are always made, and kept

- **On the gun (Auto clips, on by default; switch in the Sessions footer):** live mode does not render clips while
  shooting (`record.clip = off`), so the console catches up in the background: previews first (a fraction of a
  second each), then clips, newest press first, one at a time at the lowest CPU priority. It only works while the
  arming button is **up**, no calibration tool is open and the Pi is below 76 C (79 C with the pipeline
  stopped; resumes below 73 / 76 C). The moment the button is pressed, or the Pi gets hot, the render is **frozen**
  (SIGSTOP: no CPU at all) and continued where it stopped afterwards; one frozen for 15 min is dropped and redone
  (`tbot clip` writes a `.part` file and renames it when done, so nothing half-made is ever left). A bare Pi idles
  at 73-75 C with the live pipeline indoors, so clips then come slowly; with the pipeline stopped 12 clips took
  about 2.5 min. A press is left alone for 45 s after it ends. Code:
  `tbot/console/autoclips.py`, previews `tbot/console/thumbs.py`.
- **On this computer:** while connected, the desktop app copies every press that has a clip to
  `Videos\TBOT\<press>\` (clip.mp4, thumb.jpg, notes, score, shot images, summary), and again whenever its notes or
  score change. They stay there after the gun's SD card is cleaned. "Saved on this PC · open folder" in the player,
  and the footer button opens the folder. `TBOT_ARCHIVE` in the environment moves it.

## Safety

Nothing here weakens the safety model in the [README](../README.md#safety-model):

- **LIVE** (start or power-up) is one click in the app (no typed confirmation, by the owner's choice; `tbotctl`
  on the command line still asks). A red banner shows on every page while the gun is live, with a one-click
  switch to dry run.
- The servo page *is* `tbot jog servo` (same `ServoJog` object): anything but the rest pulse moves only while the
  arming button is held, release goes limp at once, moves are short bursts, travel limits creep until marked.
  The page tells you to unload first.
- Camera and servo tools only run with the pipeline stopped (the app asks before stopping it) and are closed when
  you leave their page; the pipeline then stays stopped until you start it.
- No login: the console is meant for a direct cable or the gun's own network. Every action that can move the
  trigger still needs a human holding the arming button.

## How it is built

```
software/tbot/console/
  server.py     HTTP server + JSON API (standard library only, Nice=5 below the pipeline)
  system.py     tbotctl / systemd / health / clock / power (PiSystem), and a laptop stand-in (DevSystem)
  tomledit.py   default.toml -> schema with docs; local.toml edits that keep every comment
  sessions.py   press list, shot frame -> clip time, notes
  tools.py      helper processes (JSON lines) and background jobs
  camtool.py    camera helper: MJPEG on 127.0.0.1:8081, focus numbers, twist capture
  twist.py      laser dot (DoG) + robust circle fit (method of calib_data/bs20ft_2026-11-18/twist_fit.py)
  servotool.py  servo helper around tbot.tools.jog.ServoJog
  web/          the app (HTML/CSS/JS, no frameworks, no CDNs)
software/desktop/
  tbot_console.pyw     Windows app: serves web/ on 127.0.0.1:47811, finds the gun, forwards /api, opens an Edge app window
  install_shortcut.ps1 Desktop + Start menu shortcut
  tbot.ico
software/pi/systemd/tbot-console.service
```

Settings are validated by the real loader (`tbot.config.load_config` in a child process) before `local.toml` is
replaced; the previous file goes to `config/backups/` (last 30 kept). Calibration-tool runs (dots.csv, twist_fit.png,
result.json) go to `calib_runs/`.

On the Pi: `tbotctl install` installs and enables the unit; `tbotctl console [on|off|restart]`.
Deploy changes with the tar recipe and `tbotctl console restart`.

### Working on it without the gun

```bash
cd software
tbot console --dev --port 8765 --sessions field_data/2026-11-17-range/presses   # synthetic pipeline, simulated servo
python desktop/tbot_console.pyw --pi 127.0.0.1:8765                              # the app window against it
```

`--dev-camera video:calib_data/bs35ft_2026-11-18/twist35.mjpeg` plays a recorded laser twist as the camera, so the
boresight wizard can be run end to end (it checks the 2026-11-18 zero at 0.04 deg, PASS). Dev mode writes notes and
clips into the sessions folder you give it; copy recordings to a scratch folder first if they are under git.
