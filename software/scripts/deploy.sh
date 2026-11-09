#!/usr/bin/env bash
# Laptop-side deployment; never delete or overwrite the Pi's per-rig data.
set -euo pipefail

usage() { echo "Usage: $0 {push|pull|ssh} [host]" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
cmd=$1
host=${2:-${TBOT_HOST:-pi@tbot.local}}
software_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
remote="${host}:TBOT/software"
# Local test mode: this directory represents the Pi's ~/TBOT/software.
if [[ -n ${TBOT_DEPLOY_LOCAL_ROOT:-} ]]; then
    remote=${TBOT_DEPLOY_LOCAL_ROOT%/}
fi
rsync_opts=(-av)
if [[ ${TBOT_RSYNC_DRY_RUN:-0} == 1 ]]; then
    rsync_opts+=(--dry-run)
fi

case "$cmd" in
    push)
        echo "Pushing ${software_dir}/ to ${remote}/ (per-rig data excluded)"
        rsync "${rsync_opts[@]}" --mkpath --exclude='/.venv/' --exclude='__pycache__/' \
            --exclude='.pytest_cache/' --exclude='/sessions/' --exclude='/calib/' \
            --exclude='/pi-pulls/' --exclude='/config/local.toml' "${software_dir}/" "${remote}/"
        ;;
    pull)
        mkdir -p -- "${software_dir}/pi-pulls"
        dest="$(mktemp -d "${software_dir}/pi-pulls/$(date +%Y%m%d-%H%M%S)-XXXXXX")"
        echo "Pulling ${remote} sessions, calibration and local config into ${dest}/"
        rsync "${rsync_opts[@]}" --ignore-missing-args "${remote}/sessions" "${remote}/calib" "$dest/"
        rsync "${rsync_opts[@]}" --ignore-missing-args "${remote}/config/local.toml" "$dest/local.pi.toml"
        ;;
    ssh)
        [[ -z ${TBOT_DEPLOY_LOCAL_ROOT:-} ]] || { echo 'ssh is unavailable in local test mode' >&2; exit 2; }
        echo "Connecting to ${host}"
        exec ssh "$host"
        ;;
    *) usage ;;
esac
