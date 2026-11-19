# Range media: press table, highlight reels, GIFs, phone sync

`scripts/range_media.py` turns a folder of recorded presses (see [PRESS_CLIPS.md](PRESS_CLIPS.md)) into review
and promo media. It needs numpy, OpenCV and ffmpeg/ffprobe, and does not import `tbot`. OpenCV decodes the phone
video because Fedora's ffmpeg has no HEVC decoder. The pip OpenCV wheel has one.

```bash
cd software
S=~/.cache/tbs/pi/range2/sessions                 # the copied press folders
V=pi-pulls/range-2026-11-19-pm/videos

# 1. annotated clips (frame n of a clip = row n of frames.csv); on a laptop without a hardware H.264 encoder
#    OpenCV falls back to MPEG-4, which works but is large
for d in $S/*-press*; do n=$(basename $d); tbot clip $d --scale 2 --out $V/annotated/${n#*-}.mp4; done

# 2. who fired, whether the gun went off (IMU recoil), boots, as a markdown table
python scripts/range_media.py table $S

# 3. highlight reel: every press that fired, real time then 4x slow (--recoil-only: only shots that went off)
python scripts/range_media.py reel $S --clips $V/annotated --out $V/highlight_reel.mp4 --title "TBOT|range day"

# 4. a README GIF of one press around the shot (exits 2 if it is over --max-mb, default 8)
python scripts/range_media.py gif $S/20261119-160243-press --clips $V/annotated --out shot.gif --pre 1.2 --post 0.35

# 5. phone clips beside the gun camera, lined up on the bang
python scripts/range_media.py sync $S --phone pi-pulls/range-2026-11-19-pm/phone --clips $V/annotated \
    --out $V/synced --window 2026-11-19T15:40-05:00/2026-11-19T17:00-05:00 --reel synced_reel.mp4 [--dry-run]
```

## How `sync` pairs clips with presses

- **Right trip.** A phone clip is used only if its creation time is inside `--window`. It reads
  `com.apple.quicktime.creationdate` (local, with offset) first, then `creation_time` (UTC), then the file mtime.
  `--end-stamped` is for phones that stamp the end of a recording rather than the start.
- **Bangs.** Each clip's audio is searched for shots: the envelope jumps above `--blast-dbfs` (-16) and above 25x
  the clip's noise floor. An iPhone limits a shot to about -10 dBFS, and the echo off the trees about 0.3 s later is
  just as loud, so onsets within 0.6 s of a shot are counted as that shot. A double inside 0.6 s shows in the
  press's IMU, not in the audio count.
- **Which presses can be heard.** By default only presses with an IMU `recoil` event. A fire command that did not
  set the gun off makes no bang (`--audible fire` to include them anyway).
- **No trust in the Pi clock.** The Pi has no clock battery and its clock jumps backwards after a power cut.
  On 2026-11-19 it was also about 33 minutes behind the phone. Presses are grouped into boots (wall clock minus
  uptime); inside a boot the monotonic clock is exact. For a boot with 2 or more shots, the script searches the
  clock offset (within `--max-skew-s`, default 1 h) that puts the most shots on bangs within `--tol-s`, so the
  shot spacing decides the match. A boot with one shot gets the first unclaimed bang after the previous boot's last
  press plus that shot's uptime. These matches are printed as LOW confidence: check the pictures, and override with
  `--pair IMG_0569.MOV=160243-press[@0.38]`.
- **Rendering.** The phone frame at phone time `t` sits next to the gun-camera frame exposed at
  `shot + (t - bang)`. The shot is the IMU recoil (the fire command + `expected_ms` when there is no recoil). The
  phone audio plays under the real-time part, followed by a silent `--slowmo` replay. `--distance-m` removes the
  sound's travel time from the gun to the phone. `--layout pip` insets the gun view instead.

## 2026-11-19 afternoon result

`sync` paired 160333 and 160357 with IMG_0565.MOV at 14.89 s and 39.06 s. The bangs are 24.17 s apart and the
recoils 24.177 s apart, a 5 ms match. It paired 160243 with IMG_0569.MOV at 0.38 s at low confidence. In the
morning trip only one shot went off (092749). It has no second shot in its boot to anchor on, so it needs `--pair`
after looking at the clips.
