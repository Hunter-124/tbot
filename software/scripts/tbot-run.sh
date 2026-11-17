#!/usr/bin/env bash
# Service entry point. Mode comes from /run/tbot/mode, which lives in RAM: after any reboot or power cycle
# the pipeline is back in DRY RUN (decisions logged, actuator never driven) until someone runs
# `tbotctl start live` again.
set -u
mode=dry
[[ -r /run/tbot/mode ]] && mode=$(tr -d '[:space:]' < /run/tbot/mode)
cd "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs sessions
# Every held press of the arming button is saved under sessions/ (raw frames, IMU, per-frame telemetry, summary
# and an annotated clip.mp4; browse them at http://<pi>:8080/clips). No continuous per-frame log: at 100 fps it
# cost a millisecond a frame and 350 MB an hour for data the press recordings already hold.
args=(run --stream --record armed)
[[ $mode == live ]] && args+=(--live)
echo "tbot-run: mode=$mode"
exec "${TBOT_BIN:-$HOME/tbot-venv/bin/tbot}" "${args[@]}"
