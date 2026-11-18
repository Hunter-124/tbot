#!/usr/bin/env bash
# Service entry point. The mode is, in this order:
#   /run/tbot/mode       what `tbotctl start <mode>` asked for; lives in RAM, gone after a reboot or power cycle
#   /etc/tbot/boot-mode  what `tbotctl boot live|dry|test` set for power-up; stays until changed
#   dry                  otherwise: decisions logged and recorded, the actuator never driven
# So a gun that was only ever started with `tbotctl start live` is back in DRY RUN after a power cycle, and a
# gun set up with `tbotctl boot live` (a day in the field without a laptop) comes back LIVE.
set -u
mode=dry
[[ -r /etc/tbot/boot-mode ]] && mode=$(tr -d '[:space:]' < /etc/tbot/boot-mode)
[[ -r /run/tbot/mode ]] && mode=$(tr -d '[:space:]' < /run/tbot/mode)
case "$mode" in dry|live|test) ;; *) echo "tbot-run: unknown mode '$mode' -> dry"; mode=dry ;; esac
cd "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs sessions
# the log (and a status line every 10 s) also goes to logs/tbot-run.log: the Pi's journal is lost at every
# power cycle, this file is not (5 MB x 4, rotated)
export TBOT_LOG_FILE=${TBOT_LOG_FILE:-logs/tbot-run.log}
# Every held press of the arming button is saved under sessions/ (raw frames, IMU, per-frame telemetry, summary
# and an annotated clip.mp4; browse them at http://<pi>:8080/clips). No continuous per-frame log: at 100 fps it
# cost a millisecond a frame and 350 MB an hour for data the press recordings already hold.
args=(run --stream --record armed)
# live: the presses are still saved, but their clips are not rendered on the gun (a render is a second Python
# process with OpenCV for 10-20 s after every press: memory and a core the next clay may need). Render them
# afterwards: tbot clip sessions/<name>
[[ $mode == live ]] && args+=(--live --set record.clip=off)
# test: DRY RUN that fires (on paper) at anything MOVING it tracks, clay or not: birds, a plane.
# No speed floor, any range the tracker believes, several shots a press. The scenery filter stays on (things
# fixed in the world are still ignored). `tbot score sessions/<name> --image` then says whether each shot
# would have hit. Never with --live.
[[ $mode == test ]] && args+=(--set fire.min_clay_speed_mps=0 --set fire.min_range_m=0.5
                              --set fire.min_elevation_deg=-90 --set fire.max_shots_per_arm=6)
echo "tbot-run: mode=$mode"
exec "${TBOT_BIN:-$HOME/tbot-venv/bin/tbot}" "${args[@]}"
