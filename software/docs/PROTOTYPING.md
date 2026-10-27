# Prototyping: Pi ↔ laptop, wiring, and the parts the software needs

How to work on the thing before it is a finished gun-mounted box: getting a shell and the debug stream off the
Pi 3B, what is wired where, and which electrical parts the software assumes exist but `hardware/design/BOM.md` does not
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
rsync -a pi@tbot.local:~/tbot/software/sessions/20261026-141230-rec sessions/
scp -r pi@tbot.local:~/tbot/software/calib calib-from-pi      # calibration files, to compare
```

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
| Latency LED | **27** | 13 | output | `tbot latency-camera`: LED + **330 Ω** to GND, in the camera's view. |
| Break sense | **22** | 15 | input, pull-up | `tbot latency-actuator`: microswitch to **GND**, closes at trigger break (falling edge = break, rising = sear reset). |
| IMU SDA | **2** | 3 | I2C1 | GY-521 (MPU-6050), address **0x68**, `dtparam=i2c_arm=on`, 400 kHz. |
| IMU SCL | **3** | 5 | I2C1 | |
| IMU power | — | 1 (3V3) + 6 (GND) | — | **3.3 V**. The GY-521's regulator accepts 5 V, but then its logic lines are 5 V-tolerant-by-luck; run the board at 3.3 V. |
| Serial console TXD | **14** | 8 | UART out | To the USB-TTL cable's RX, 3.3 V, 115200 (`enable_uart=1`). |
| Serial console RXD | **15** | 10 | UART in | To the cable's TX. |

Conflicts to know about: **PWM0 on GPIO18 is shared with the analogue audio output**, so `dtparam=audio=off` is
required for a stable servo pulse (`scripts/pi_setup.sh` sets it). GPIO 2/3 have fixed 1.8 kΩ pull-ups to 3.3 V on
the board — fine for I2C, unusable as general inputs.

**Grounds.** The Pi, the actuator driver, the actuator supply and the arming switch share one ground. The PQ12's
12 V and the servo's 5–6 V come from the battery/BEC, never from the Pi's 5 V pin.

## 4. Hardware the software needs that is not on `hardware/design/BOM.md`

`hardware/design/BOM.md` covers the printed parts, fasteners, battery, buck module and the two trigger actuators. The
electrical pieces below are assumed by the code and the calibration tools, and are missing from it:

| Item | Why | Notes |
|---|---|---|
| **H-bridge / DC motor driver for the PQ12**, rated ≥ 12 V | `actuator.pq12` drives two GPIOs (extend/retract); the PQ12-100-12-S is a bare 12 V DC motor with internal limit switches and no driver of its own | **TB6612FNG** (15 V max, 1.2 A continuous / 3 A peak) fits: PQ12 stall is 210 mA @ 12 V (layout.md, datasheet Rev D). **DRV8833 is only 10.8 V max — do not use it on 12 V.** L298N works but wastes ~2 V and a lot of board space. Wire `AIN1/AIN2` to BCM 23/24 and the driver's **STBY/enable through the arming switch** (§5). |
| **Separate 5–6 V supply (BEC) for the MG996R** | The servo's stall current is far beyond what the Pi's 5 V rail can source; brown-outs reboot the Pi mid-shot | ~2.5 A stall `[INFERENCE: typical for an MG996R-class metal-gear servo at 6 V; not published by TowerPro]`. A 3 A UBEC off the 12 V pack, or a second XL4015 set to 6.0 V. Common ground with the Pi; **never** from the Pi 5 V pin. 6 V also buys speed: 0.14 s/60° vs 0.17 s/60° at 4.8 V (TowerPro). |
| **Latency LED + 330 Ω resistor** | `tbot latency-camera` needs a light the camera can see, switched from BCM 27 | Any 3 mm/5 mm LED. 330 Ω keeps the pin at ~5 mA. Position it in frame, not pointing into the lens. |
| **Break-sense microswitch + bench fixture** | `tbot latency-actuator` times command → trigger break from a real contact on BCM 22 | A lever microswitch (NO contact to GND) on a jig that holds the trigger set with the sear travel reproduced. Cheap, and the only way to get a real `cmd_to_break_ms`. |
| **USB-to-TTL serial cable (3.3 V)** — optional | Console when the network is down or the boot fails | CP2102/CH340/FT232 with 3.3 V levels. See §1. |
| Dupont/JST leads, heat-shrink, a 2-pole connector for the actuator | Field-serviceability: the trigger set has to come off the gun | With a Pi 5 + AI HAT+, standard Dupont housings will not fit under the barrel — solder or use low-profile connectors (`hardware/design/assembly_sequence.md`, open issue 3). |

### 5. The hardware interlock (do this)

The software interlock is real — `Actuator.fire()` refuses unless the arming switch reads armed, disarm aborts to
rest within one control step, `fire.live` is false by default — but it is still software. Wire the arming switch
so it also **gates the driver**:

- **PQ12:** switch in series with the H-bridge `STBY`/enable input (pull-down on the driver side, so open = standby).
- **MG996R:** switch (or a small MOSFET/relay it drives) in series with the **servo's 5–6 V supply**, not its
  signal line. A servo with no power cannot move; a servo with power and a glitched signal can.

Keep the switch's GPIO connection as well — the software needs to *know* it is held, and the open-circuit =
disarmed behaviour is what makes a cut lead safe. One switch, two jobs: it tells the software, and it feeds the
driver.

## 6. Which actuator: latency is the whole argument

Both trigger sets are supported (`actuator.type = servo | pq12`) and both are modelled in CAD. They are not
equivalent in the only dimension that matters for a moving target:

| | MG996R servo | Actuonix PQ12-100-12-S |
|---|---|---|
| Command → trigger break | ~0.1 s (`latency.servo_cmd_to_break_ms = 115` @ 4.8 V, loaded) `[INFERENCE]` | **~0.7–1.1 s** (`pq12_cmd_to_break_ms = 900`) `[INFERENCE: ~6.7 mm of travel to break at the datasheet 6–10 mm/s, layout.md]` |
| Source of the delay | gear train + 50 Hz frame quantisation (`jitter_ms = 6`, = 20 ms/√12) | screw drive speed; it is simply slow |
| Clay travel at 20 m/s during that delay | ~2 m | ~14–22 m |
| Holding force unpowered | none (needs holding torque) | back-drive 35 N — holds position unpowered |
| Power | 5–6 V, ~2.5 A stall `[INFERENCE]` | 12 V, 210 mA stall |

A full second of lead on a crossing clay is far more prediction than the tracker can honestly provide. Measure
both before choosing — that is exactly what `tbot latency-actuator --type servo` / `--type pq12` and then
`tbot latency-report` are for: the report's "clay moves [m] / [deg]" column turns each measured delay into the
lead the solver must predict at a reference range, and the configured actuator is marked with `*`.

## 7. Bring-up order on the Pi

1. `sudo ./scripts/pi_setup.sh`, reboot.
2. `tbot hwcheck` until there is no FAIL; `tbot hwcheck --interactive` to prove the arming button both ways.
3. Gun UNLOADED, trigger set fitted: `tbot jog servo` (or `tbot jog pq12`) to set and `save` the end positions.
4. Break-sense fixture: `tbot latency-actuator`, then `tbot latency-report`.
5. `tbot run --stream` from the laptop browser (`http://<pi>:8080`), dry run; watch `health` in `/telemetry` for
   under-voltage or throttling while the actuator moves.
6. Calibration in the order of [CALIBRATION.md](CALIBRATION.md).
