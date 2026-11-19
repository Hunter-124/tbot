"""TBOT Console: the gun's own configuration and calibration app.

`tbot console` runs on the Pi (systemd unit tbot-console.service, port 80) and serves the web UI in
console/web/ plus a JSON API. The desktop launcher (software/desktop/tbot_console.pyw) serves the same UI on the
laptop and forwards /api to the gun, so the app opens even before the cable is plugged in.

Everything here is standard library only (the server must stay light next to the 100 fps pipeline). Camera and
servo work runs in separate helper processes (camtool.py, servotool.py) that import numpy/OpenCV themselves.
Docs: docs/CONSOLE.md.
"""

API_VERSION = 1
