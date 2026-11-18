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
# extra ssh/scp options for the tar deploy, e.g. TBOT_SSH_OPTS="-o UserKnownHostsFile=~/.ssh/known_hosts_tbot"
read -r -a TBOT_SSH_OPTS <<< "${TBOT_SSH_OPTS:-}"
if [[ ${TBOT_RSYNC_DRY_RUN:-0} == 1 ]]; then
    rsync_opts+=(--dry-run)
fi

case "$cmd" in
    push)
        if command -v rsync > /dev/null && [[ ${TBOT_DEPLOY_TAR:-0} != 1 ]]; then
            echo "Pushing ${software_dir}/ to ${remote}/ (per-rig data excluded)"
            rsync "${rsync_opts[@]}" --mkpath --exclude='/.venv/' --exclude='__pycache__/' \
                --exclude='.pytest_cache/' --exclude='/sessions/' --exclude='/calib/' \
                --exclude='/pi-pulls/' --exclude='/config/local.toml' --exclude='/field_data/' \
                --exclude='/calib_data/' --exclude='/runs/' --exclude='/tbot/vision/_measure*.so' \
                "${software_dir}/" "${remote}/"
        else
            # No rsync (Git Bash on Windows): the tracked files as they are in the working tree (uncommitted
            # edits included, via a stash commit that touches nothing), with LF line endings, through scp.
            # Untracked files are NOT sent. Files deleted here stay on the Pi. Per-rig data is never in git.
            [[ -z ${TBOT_DEPLOY_LOCAL_ROOT:-} ]] || { echo 'tar deploy needs a host' >&2; exit 2; }
            cd -- "$software_dir"
            rev=$(git stash create 2> /dev/null || true)
            rev=${rev:-HEAD}
            desc="$(git rev-parse --short HEAD)$( [[ $rev == HEAD ]] || echo '+uncommitted')"
            untracked=$(git ls-files --others --exclude-standard -- tbot scripts pi config tests | head -20)
            [[ -z $untracked ]] || printf 'NOT sent (untracked):\n%s\n' "$untracked" >&2
            part="${software_dir}/.deploy-$$.tar"
            trap 'rm -f -- "$part"' EXIT
            git -c core.autocrlf=false -c core.eol=lf archive --format=tar -o "$part" "$rev" -- \
                tbot scripts pi config/default.toml tests docs firmware pyproject.toml README.md .gitignore
            echo "Pushing $desc to ${host}:TBOT/software (tar over scp; per-rig data untouched)"
            scp -q "${TBOT_SSH_OPTS[@]}" "$part" "${host}:/tmp/tbot-deploy.tar"
            ssh "${TBOT_SSH_OPTS[@]}" "$host" "mkdir -p TBOT/software && tar -xf /tmp/tbot-deploy.tar -C TBOT/software \
                && rm -f /tmp/tbot-deploy.tar && find TBOT/software/tbot -name __pycache__ -prune -exec rm -rf {} + \
                && echo '$desc, deployed $(date -u +%Y-%m-%dT%H:%MZ)' > TBOT/software/DEPLOYED.txt \
                && (\$HOME/tbot-venv/bin/python -m compileall -q TBOT/software/tbot > /dev/null || true) \
                && (cd TBOT/software && ./scripts/build_native.sh > /dev/null && echo 'native kernel built' \
                    || echo 'native kernel NOT built (the Python path runs)')"
        fi
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
