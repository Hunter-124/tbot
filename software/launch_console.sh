#!/usr/bin/env bash
# TBOT Console launcher (Linux / macOS / Git Bash).
#
#   ./launch_console.sh          open the console for the gun on the Ethernet cable (169.254.20.20 / tbot.local)
#   ./launch_console.sh dev      no gun: start a stand-in gun on this computer (synthetic camera, simulated servo)
#                                and open the console on it; Ctrl-C stops both
#
# Needs Python 3.10+. Connecting to the gun needs no packages; "dev" needs the software's own dependencies
# (pip install -e .). On the gun itself the console is already running as tbot-console.service on port 80:
# just browse to http://tbot.local/ instead.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python was not found: install Python 3.10+." >&2
    exit 1
fi

if [ "${1:-}" = "dev" ]; then
    shift
    echo "Starting the stand-in gun on http://127.0.0.1:8765/ ..."
    "$PY" -m tbot.console.server --dev --host 127.0.0.1 --port 8765 &
    DEV_PID=$!
    trap 'kill "$DEV_PID" 2>/dev/null || true' EXIT INT TERM
    "$PY" desktop/tbot_console.pyw --pi 127.0.0.1:8765 "$@"
else
    exec "$PY" desktop/tbot_console.pyw "$@"
fi
