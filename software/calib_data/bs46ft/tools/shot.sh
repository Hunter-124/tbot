#!/usr/bin/env bash
# usage: shot.sh <name> [shutter_us] [gain]  -> takes 1280x800 still on Pi, fetches it, prints laser dot
set -e
K=${TBOT_KEY:-$HOME/.ssh/tbot_pi_ed25519}; H=${TBOT_HOST:-pi@169.254.20.20}; D=$(cd "$(dirname "$0")/.." && pwd)
n=$1; s=${2:-2000}; g=${3:-2}
ssh -i $K -o BatchMode=yes $H "cd ~/TBOT/software && rpicam-still -n -t 1200 --width 1280 --height 800 --shutter $s --gain $g -e png -o calib_data/bs46ft/$n.png >/dev/null 2>&1"
scp -q -i $K $H:TBOT/software/calib_data/bs46ft/$n.png $D/
cd "$D" && python -I - "$n.png" <<'PY'
import sys, cv2, numpy as np
im=cv2.imread(sys.argv[1],0).astype(float); bg=cv2.medianBlur(im.astype(np.uint8),21).astype(float)
d=im-bg; y,x=np.unravel_index(np.argmax(d),d.shape); w=d[y-6:y+7,x-6:x+7].clip(0); yy,xx=np.mgrid[-6:7,-6:7]
print(f"{sys.argv[1].split('/')[-1]}: dot ({x+(w*xx).sum()/w.sum():.2f}, {y+(w*yy).sum()/w.sum():.2f})  {d[y,x]:.0f} DN above bg")
PY
