# Prototyping: Pi ↔ laptop, wiring, and the parts the software needs

How to work on the thing before it is a finished gun-mounted box: getting a shell and the debug stream off the
Pi 3B, what is wired where, and which electrical parts the software assumes exist but `design/BOM.md` does not
list yet.

## 1. Connecting the Pi 3 Model B to a laptop

### Ethernet, direct cable (recommended)

The Pi 3B's Ethernet port auto-MDI/MDI-X, so a plain patch cable between laptop and Pi works. Let NetworkManager
on the laptop run a shared (NAT + DHCP) connection, so the Pi also gets internet for `apt`:

```bash
nmcli device status                                    # find the laptop's wired interface, e.g. enp0s31f6
nmcli connection add type ethernet ifname enp0s31f6 con-name tbot-link ipv4.method shared
nmcli connection up tbot-link
```

The laptop becomes `10.42.0.1` and hands the Pi an address in `10.42.0.0/24`. Find it:

```bash
nmcli connection show tbot-link | grep -i address      # or:
ip neigh show dev enp0s31f6
ssh pi@tbot.local                                      # mDNS, if avahi-daemon is running on the Pi (default on Pi OS)
```

Static IPs instead, if you prefer no DHCP (no internet for the Pi this way):

```bash
# laptop
nmcli connection add type ethernet ifname enp0s31f6 con-name tbot-static \
      ipv4.method manual ipv4.addresses 192.168.50.1/24
# Pi (/etc/dhcpcd.conf, or NetworkManager if you use it)
#   interface eth0
#   static ip_address=192.168.50.2/24
```

With the link up, everything else is over SSH plus the debug stream:

```bash
ssh pi@tbot.local '~/tbot-venv/bin/tbot run --stream'
# then on the laptop:  http://tbot.local:8080/   (page, /stream.mjpg, /snapshot.jpg, /telemetry)
```

`stream.host` defaults to `0.0.0.0` and `stream.port` to `8080`; `tbot run --port 8081` implies `--stream`.
The stream is rate-limited to `stream.fps = 10` at `jpeg_quality = 70` so it does not eat the pipeline's budget —
it is a debug view, not a recording. For data, use `tbot record` and copy the session:

```bash
rsync -a pi@tbot.local:~/TBOT/software/sessions/20261026-141230-rec sessions/
scp -r pi@tbot.local:~/TBOT/software/calib calib-from-pi      # calibration files, to compare
```

For the [HITL bench](HITL.md), run the rig on the Pi and connect the laptop by wired Ethernet; the laptop
drives the monitor and opens the target page served by the Pi on port 8090. Point the camera at the monitor
1–2 m away, with the bore line on screen toward the targets' exit side for a longer sweep runway. Use full
brightness; the rig waits for the browser to enter fullscreen before measuring (`--allow-windowed` overrides
this). Keep the monitor and camera fixed after geometry calibration.

From the laptop, deploy and retrieve rig data without overwriting the Pi's calibration:

```bash
software/scripts/deploy.sh push                    # code -> ~/TBOT/software
software/scripts/deploy.sh ssh                     # interactive shell
software/scripts/deploy.sh pull                    # sessions, calib, local.pi.toml -> software/pi-pulls/<date>-<id>/
TBOT_HOST=pi@192.168.50.2 software/scripts/deploy.sh push
software/scripts/deploy.sh pull pi@192.168.50.2    # explicit host overrides TBOT_HOST
```

`push` excludes `.venv`, Python caches, sessions, calibration, `config/local.toml`, and prior pulls;
it does not delete remote files. Both machines need rsync 3.2 or newer. For a local dry-run test,
create an empty temporary directory and run
`TBOT_DEPLOY_LOCAL_ROOT=/path/to/temp TBOT_RSYNC_DRY_RUN=1 software/scripts/deploy.sh push`.
The local root stands in for the Pi's `~/TBOT/software`; SSH is unavailable in this test mode.

### "USB" — read this before buying a cable

**The Raspberry Pi 3 Model B has no USB device/gadget (OTG) mode.** Its four USB-A ports hang off an onboard
LAN9514 hub/Ethernet chip and are host-only, and the micro-USB connector is **power only**. The `dwc2`/`g_ether`
"USB networking" trick that works on a Pi Zero, Pi 4 (USB-C) or Pi 5 does **not** work here. So "connect over USB"
means one of two different things:

1. **USB-to-TTL serial console** (a CP2102/CH340/FT232 cable) onto the GPIO header — a real console that works
   with no network, no display and even when the boot fails:
   - cable **GND** → any Pi GND (header pin 6), cable **RX** → Pi **TXD, BCM 14** (pin 8), cable **TX** → Pi
     **RXD, BCM 15** (pin 10). **Do not connect the cable's 5 V line.**
   - **3.3 V logic levels only.** A 5 V TTL cable will damage the Pi.
   - `enable_uart=1` in `config.txt` (`scripts/pi_setup.sh` adds it) and the console enabled in
     `raspi-config` → Interface Options → Serial Port (login shell: yes, hardware: yes).
   - `screen /dev/ttyUSB0 115200` or `picocom -b 115200 /dev/ttyUSB0`.
   - Note the Pi 3B's primary UART is the mini-UART (the PL011 is on Bluetooth); `enable_uart=1` pins the core
     clock so the baud rate stays stable. Bluetooth is not needed here.
2. **A USB-Ethernet adapter on the laptop side**, when the laptop has no RJ45. Electrically it is case 1 of §1 —
   NetworkManager treats it as just another wired interface.

Serial is a console, not a data link: it is fine for `tbot bench`, log lines and fixing a broken boot, useless for
MJPEG. Use Ethernet for anything with frames in it.

### HDMI

A monitor, keyboard and mouse on the Pi gives a local desktop. The only thing it adds over SSH is **OpenCV
highgui**: `tbot run --display`, `tbot calib-detector --gui`, `tbot calib-intrinsics --gui` and
`tbot calib-boresight --gui` all need an X/Wayland session on the machine running them (`cv2.imshow`). Over SSH
they fail unless you forward X11, which is slower than useless for 120 fps video.

Costs: the desktop session and the display pipeline take CPU and memory from a 1 GB Pi 3B that is already tight
(`tbot bench` before and after is the honest comparison). The normal pattern is:

- **GUI tools on the laptop**, against a *replayed session*: `tbot calib-detector --session sessions/swing-01 --gui`.
- **Headless on the Pi**: `--report`, `--stream`, `--log-jsonl`.
- HDMI only when you need to see the live camera on the gun itself, or when the network is down.

## 2. Laptop-only workflows

No Pi, no camera, no GPIO — all three source types work with `--source` or `camera.source`:

```bash
# fully synthetic: rendered clay, simulated swing, simulated IMU, simulated GPIO/actuator
tbot run --source synthetic --sim-armed --stream --port 8080
tbot run --source synthetic:skeet_incoming --sim-armed --max-frames 600 --log-jsonl /tmp/t.jsonl
tbot sim --scenarios skeet_crossing trap_straightaway --seeds 5 --out sim.json

# webcam, pretending to be the mono sensor (none | luma | red — see docs/OPTICS.md §5)
tbot run --source cv:0 --set camera.emulate_mono=red --stream
tbot record --source cv:0 --seconds 20 --out sessions/desk-test

# a video file
tbot run --source video:clips/clay-throws.mp4 --stream

# replay a session recorded on the gun
tbot run --source replay:sessions/20261026-141230-rec --stream
tbot calib-detector --session sessions/20261026-141230-rec --gui
tbot calib-imu sync --session sessions/20261026-141230-rec
```

`--sim-armed` only works with the simulated GPIO backend (`gpio.backend = auto` resolves to `sim` off a Pi); it
cannot arm real hardware. The simulated clock reproduces real processing latency on simulated timestamps, so
timing results from replay/synthetic runs are meaningful.

## 3. GPIO wiring

Defaults from `config/default.toml`. BCM numbering is what the config uses; the header column is the physical pin
on the 40-pin connector.

| Signal | BCM | Header pin | Direction | Notes |
|---|---|---|---|---|
| Arming switch | **17** | 11 | input, pull-**up** | Momentary NO to **GND**. Pressed = LOW = armed (`arming.active_low = true`, `pull = "up"`, `debounce_ms = 5`). Open circuit = not armed, fail-safe. |
| Servo PWM (MG996R) | **18** | 12 | hardware PWM0 | `dtoverlay=pwm,pin=18,func=2`; sysfs `/sys/class/pwm/pwmchip0/pwm0`; 50 Hz frame, `rest_us = 1670`, `fire_us = 1330`. Signal only — **not** servo power. |
| PQ12 extend | **23** | 16 | output | To an H-bridge input (§4). |
| PQ12 retract | **24** | 18 | output | To the other H-bridge input. |
| Ready light (status LED) | **27** | 13 | output | `hw/status_led.py`: LED + **220–330 Ω** to GND (pin 14), see the wiring below. Same pin as the latency LED (that tool runs with the pipeline stopped). |
| Latency LED | **27** | 13 | output | `tbot latency-camera`: LED + **330 Ω** to GND, in the camera's view. |
| Break sense | **22** | 15 | input, pull-up | `tbot latency-actuator`: microswitch to **GND**, closes at trigger break (falling edge = break, rising = sear reset). |
| IMU SDA | **2** | 3 | I2C1 | GY-521 (MPU-6050), address **0x68**, `dtparam=i2c_arm=on`, 400 kHz. |
| IMU SCL | **3** | 5 | I2C1 | |
| IMU power | — | 1 (3V3) + 9 (GND) | — | **3.3 V**. The GY-521's regulator accepts 5 V, but then its logic lines are 5 V-tolerant-by-luck; run the board at 3.3 V. |
| Serial console TXD | **14** | 8 | UART out | To the USB-TTL cable's RX, 3.3 V, 115200 (`enable_uart=1`). |
| Serial console RXD | **15** | 10 | UART in | To the cable's TX. |

Conflicts to know about: **PWM0 on GPIO18 is shared with the analogue audio output**, so `dtparam=audio=off` is
required for a stable servo pulse (`scripts/pi_setup.sh` sets it). GPIO 2/3 have fixed 1.8 kΩ pull-ups to 3.3 V on
the board — fine for I2C, unusable as general inputs.

### 3.1 Pi 3 B header pinout (from the design docs and config; not checked against the physical board)

What is **planned/wired** on the gun, from `design/BOM.md` / `design/layout.md` (wire table) plus the ready light.
**Finding pin 1.** The header runs along one long edge of the board; the USB/Ethernet ports are on a short edge.
Pin 1 is at the end of the header **farthest from the USB/Ethernet ports** (the SD-card end), on the row **toward
the middle of the board**, and its solder pad on the underside is the only **square** one. Odd pins (1, 3, 5 ...)
are that inner row; even pins (2, 4, 6 ...) are the row along the board edge. Pin 2 (5 V) is the corner pin.

The diagram below is the board held **component side up, USB/Ethernet ports pointing down at you**: the header is
then a vertical strip along the **right-hand edge**, pin 1 at the top of the left (inner) column, pin 2 at the top
of the right (edge) column. The harness photo shows DuPont housings, not the soldered joints the BOM describes, so
verify against the board.

```
                                 inner  edge
   IMU VCC (3V3)  <-- 3V3         (1)   (2)  5V          --> Pi power + (buck #1, 5.1 V)
   IMU SDA        <-- GPIO2 SDA   (3)   (4)  5V
   IMU SCL        <-- GPIO3 SCL   (5)   (6)  GND         --> Pi power - (buck #1)
                      GPIO4       (7)   (8)  GPIO14 TXD
   IMU GND        <-- GND         (9)  (10)  GPIO15 RXD
   arming switch  <-- GPIO17     (11)  (12)  GPIO18 PWM0 --> servo signal
   READY LED +    <-- GPIO27     (13)  (14)  GND         --> arming GND, servo-signal GND, READY LED -
                      GPIO22     (15)  (16)  GPIO23
                      3V3        (17)  (18)  GPIO24
                      GPIO10     (19)  (20)  GND
                      GPIO9      (21)  (22)  GPIO25
                      GPIO11     (23)  (24)  GPIO8
                      GND        (25)  (26)  GPIO7
                      ID_SD      (27)  (28)  ID_SC
                      GPIO5      (29)  (30)  GND
                      GPIO6      (31)  (32)  GPIO12
                      GPIO13     (33)  (34)  GND
                      GPIO19     (35)  (36)  GPIO16
                      GPIO26     (37)  (38)  GPIO20
                      GND        (39)  (40)  GPIO21
                                  ^ USB/Ethernet end
```

Ready light: **pin 13 is the 7th pin down the inner row, pin 14 is right beside it on the edge row** (the pin
across from it). Count from the pin-1 end, not the USB end.

| Pin | Name | Wired to | Wire |
|---|---|---|---|
| 1 | 3V3 | GY-521 VCC | IMU 4 x 26 AWG |
| 2 | 5V | Buck #1 OUT+ (5.1 V) | Pi 5 V pair, 2 x 20 AWG |
| 3 | GPIO2 / SDA1 | GY-521 SDA | IMU |
| 5 | GPIO3 / SCL1 | GY-521 SCL | IMU |
| 6 | GND | Buck #1 OUT- | Pi 5 V pair |
| 9 | GND | GY-521 GND | IMU |
| 11 | GPIO17 | Arming switch (other side to GND) | Arming + PWM, 3 x 26 AWG |
| 12 | GPIO18 / PWM0 | MG996R signal | Arming + PWM |
| 14 | GND | Arming switch return **and** servo-signal ground | Arming + PWM |
| 13 | GPIO27 | Ready LED anode through 220-330 ohm | new |
| 14 | GND | Ready LED cathode, same pin as above | new |

Not wired on this build: break sense (GPIO22, pin 15, only for the `latency-actuator` bench fixture), PQ12
extend/retract (GPIO23/24, pins 16/18, fallback actuator only), the serial-console pins (GPIO14/15, pins 8/10, only
when a USB-TTL cable is attached for bring-up) and the SPI / ID pins.

Pin 5 V and GND are the only power pins used. GPIO pins are **3.3 V logic, not 5 V tolerant**, and each should source
only a few mA (keep an LED to ~5 mA). Pins 13 and 14 sit side by side, and pin 14 already carries the arming/servo
ground, so the ready light adds one wire at pin 13 and a second wire onto pin 14 (the BOM notes GPIO connections are
soldered, with no room for DuPont housings: header pin tips are 8 mm from cover_L).

### 3.2 Ready light wiring

Defaults: `gpio.status_led_pin = 27`, `gpio.status_led_active_high = true`.

```
 Pi pin 13 (GPIO27) ---[ 220-330 ohm ]---->|--- Pi pin 14 (GND)
                                        LED  (long leg / anode toward the resistor,
                                              short leg / flat side / cathode to GND)
```

- Pins 13 and 14 are adjacent, so the whole light is one LED and one resistor across two neighbouring pins.
- The resistor can be on either side of the LED; it must be in series. With 330 ohm a red/green LED draws about
  (3.3 V - ~2 V) / 330 = ~4 mA, plenty bright, well inside the pin's limit.
- Wired the other way (LED between 3.3 V on pin 1 or 17 and GPIO27) set `status_led_active_high = false`.
- Behaviour: solid = ready and live; slow blink = ready but dry run; fast blink = running but not ready;
  off = pipeline not running.

**Grounds.** The Pi, the actuator driver, the actuator supply and the arming switch share one ground. The PQ12
fallback needs 12 V; the installed servo uses a separate 7.0 V XL4015, never the Pi's 5 V pin.

## 4. Electrical hardware and calibration fixtures

`design/BOM.md` covers the printed parts, fasteners, power supplies and trigger sets. The table below gives
electrical details and calibration fixtures; the PQ12 driver is needed only for the fallback:

| Item | Why | Notes |
|---|---|---|
| **H-bridge / DC motor driver for the PQ12 fallback**, rated ≥ 12 V | Not needed for the servo build. `actuator.pq12` drives two GPIOs (extend/retract); the PQ12-100-12-S is a bare 12 V DC motor with internal limit switches and no driver of its own | **TB6612FNG** (15 V max, 1.2 A continuous / 3 A peak) fits: PQ12 stall is 210 mA @ 12 V (layout.md, datasheet Rev D). **DRV8833 is only 10.8 V max — do not use it on 12 V.** L298N works but wastes ~2 V and a lot of board space. Wire `AIN1/AIN2` to BCM 23/24; a hardware-gated **STBY/enable** is an uninstalled recommendation (§5). |
| **Separate servo supply for the MG996R** | The servo's stall current is far beyond what the Pi's 5 V rail can source; brown-outs reboot the Pi mid-shot | The installed second XL4015 is set to 7.0 V. ~2.5 A stall `[INFERENCE: typical for an MG996R-class metal-gear servo at 6 V; not published by TowerPro]`. Common ground with the Pi; **never** from the Pi 5 V pin. At 6 V the stated speed is 0.14 s/60° vs 0.17 s/60° at 4.8 V (TowerPro). |
| **Latency LED + 330 Ω resistor** | `tbot latency-camera` needs a light the camera can see, switched from BCM 27 | Any 3 mm/5 mm LED. 330 Ω keeps the pin at ~5 mA. Position it in frame, not pointing into the lens. |
| **Break-sense microswitch + bench fixture** | `tbot latency-actuator` times command → trigger break from a real contact on BCM 22 | A lever microswitch (NO contact to GND) on a jig that holds the trigger set with the sear travel reproduced. Cheap, and the only way to get a real `cmd_to_break_ms`. |
| **USB microphone / USB audio capture** | `tbot hitl run --live --mic` records the unloaded dry-fire click | The Pi 3B has no audio input; connect the USB microphone to the rig, not the laptop driving the display. Install `alsa-utils`, select a capture device with `tbot hitl mics`, measure mic-to-trigger distance, and calibrate capture latency against break-sense. |
| **HITL monitor and wired Ethernet** | Known, clock-synchronised targets for end-to-end camera-to-trigger timing | Laptop drives a bright fullscreen monitor; connect it to the Pi via Ethernet (§1). See [HITL.md](HITL.md) for setup and limitations. |
| **USB-to-TTL serial cable (3.3 V)** — optional | Console when the network is down or the boot fails | CP2102/CH340/FT232 with 3.3 V levels. See §1. |
| Dupont/JST leads, heat-shrink, a 2-pole connector for the actuator | Field-serviceability: the trigger set has to come off the gun | Use connectors that fit the installed assembly. |

### 5. Arming interlock and power isolation

The installed interlock is **software only**: `Actuator.fire()` refuses unless the BCM17 arming switch reads armed,
disarm aborts to rest within one control step, and `fire.live` is false by default. The normally-open switch goes
to GND; an open circuit reads disarmed. There is **no hardware cut** in the servo supply. Open the battery lever
connector to make the actuator safe. A powered servo can move on a glitched signal; software is not a power cut.

An additional hardware interlock is recommended but **not installed**: for the MG996R, use a switch or
MOSFET/relay in series with its supply, not its signal line. For the PQ12 fallback, gate the H-bridge
`STBY`/enable with a driver-side pull-down so open means standby. Retain the switch's GPIO connection so the
software knows when it is held.

## 6. Actuator decision: MG996R servo

The build uses the MG996R (`actuator.type = "servo"`). The PQ12 remains supported in software as a fallback
(`actuator.type = "pq12"`). The estimates behind the decision are:

| | MG996R servo | Actuonix PQ12-100-12-S |
|---|---|---|
| Command → trigger break | ~0.1 s (`latency.servo_cmd_to_break_ms = 115` @ 4.8 V, loaded) `[INFERENCE]` | **~0.7–1.1 s** (`pq12_cmd_to_break_ms = 900`) `[INFERENCE: ~6.7 mm of travel to break at the datasheet 6–10 mm/s, layout.md]` |
| Source of the delay | gear train + 50 Hz frame quantisation | screw drive speed |
| Clay travel at 20 m/s during that delay | ~2 m | ~14–22 m |
| Holding force unpowered | none (needs holding torque) | back-drive 35 N — holds position unpowered |
| Power | 5–6 V, ~2.5 A stall `[INFERENCE]` | 12 V, 210 mA stall |

The servo requires substantially less prediction over the actuator delay. Its 50 Hz PWM frame quantisation is
the expected dominant jitter source: commands wait 0–20 ms for the next frame, or 20/√12 ≈ 5.8 ms one-sigma
for uniformly distributed phase. These timings are estimates, not measurements of the installed rig. Measure
the selected servo with `tbot latency-actuator --type servo`, then use `tbot latency-report` to see the lead at
the reference range. Measure the PQ12 separately only if the fallback is fitted.

## 7. Bring-up order on the Pi

1. `sudo ./scripts/pi_setup.sh`, reboot.
2. `tbot hwcheck` until there is no FAIL; `tbot hwcheck --interactive` to prove the arming button both ways.
3. Gun UNLOADED, servo trigger set fitted: `tbot jog servo` to set and `save` the end positions.
4. Break-sense fixture: `tbot latency-actuator --type servo`, then `tbot latency-report`.
5. `tbot run --stream` from the laptop browser (`http://<pi>:8080`), dry run; watch `health` in `/telemetry` for
   under-voltage or throttling while the actuator moves.
6. Calibration in the order of [CALIBRATION.md](CALIBRATION.md).
7. [HITL bench](HITL.md): dry run first; if timing the actuator, gun unloaded and arming held for each press.

### Measured on the first Pi 3B (2026-10-27)

Raspberry Pi OS Lite 64-bit trixie (2026-10-06 image, kernel 6.18, libcamera 0.7.2), Python 3.13.5, numpy 2.2.4,
OpenCV 4.10.0. The SD card was prepared headless with the image's cloud-init files on the boot partition:
`user-data` (hostname `tbot`, user `pi` with an SSH key, `enable_ssh: true`) and `network-config` (eth0 DHCP).
The `rpi: {i2c, serial}` keys from the Raspberry Pi blog example were rejected by this image's cloud-init
("Unsupported key"); `pi_setup.sh` enables those interfaces anyway. The Pi's clock was ~3 weeks behind on first boot
until NTP synced — apt refuses repositories while the clock is wrong (`sudo date -s` fixes it).

- `pi_setup.sh`: completed; it now also loads `i2c-dev` (without it there is no `/dev/i2c-1`).
- `python3 -m pytest`: all tests pass on the Pi.
- `tbot hwcheck` with nothing but power and Ethernet connected: platform, boot, gpio PASS; camera FAIL (no camera
  listed), imu FAIL (bus 1 empty) — expected with those devices not wired; **power FAIL: under-voltage and
  throttled to 600 MHz** on a micro-USB supply. Use a 5.1 V / 2.5 A supply (or the XL4015 at 5.1 V) before any
  timing measurement.
- `tbot bench --source synthetic` while throttled at 600 MHz: detect 32 ms, track 11 ms, solve 6.5 ms p50
  (overlay 48 ms, encode 13 ms are stream-only) → ~8.5 fps with the overlay, far below `camera.fps = 120`.
  Re-run `tbot bench` on a proper supply before deciding the frame rate; these numbers are not representative.
