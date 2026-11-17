#!/usr/bin/env bash
# usage: fxset.sh <tag> [n]   -> n 1280x800 stills (20 ms, gain 4) on the Pi, fetched to calib_data/fx_2026-11-16b/
set -e
K=${TBOT_KEY:-$HOME/.ssh/tbot_pi_ed25519}; H=${TBOT_HOST:-pi@169.254.20.20}; R=calib_data/fx_2026-11-16b; L=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; n=${2:-3}; mkdir -p $L
ssh -i $K -o BatchMode=yes $H "cd ~/TBOT/software && mkdir -p $R && for i in \$(seq 1 $n); do rpicam-still -n -t 1200 --width 1280 --height 800 --shutter ${SH:-20000} --gain ${GN:-4} -e png -o $R/${tag}_\$i.png >/dev/null 2>&1; done; ls $R | grep ^${tag}_"
scp -q -i $K "$H:TBOT/software/$R/${tag}_*.png" $L/
cd $L && python -I -c "
import cv2,sys; im=cv2.imread('${tag}_1.png',0); cv2.imwrite('${tag}_preview.png',cv2.resize(im,(640,400)))"
