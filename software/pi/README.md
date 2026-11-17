# TBOT Pi access

Credentials for the gun's Pi 3B. Kept in this private repo on purpose (the Pi is only networked occasionally).

| | |
|---|---|
| Hostname | `tbot` (`tbot.local` over mDNS; on the home LAN it was 192.168.0.168) |
| User | `pi` |
| Password | `dTXXu2mkqZMi` (console and SSH; sudo needs no password) |
| SSH key | `tbot_pi_ed25519` in this folder |

```bash
ssh -i software/pi/tbot_pi_ed25519 pi@tbot.local
```

The laptop's `~/.ssh/id_ed25519` and hunter-pc's key are also authorized, so plain `ssh pi@tbot.local` works from those.
On Windows, OpenSSH rejects a private key other users can read; if it complains, copy the key to `~/.ssh/` first.

## Reflashing the SD card

1. Download the Raspberry Pi OS Lite arm64 image (2026-10-06 trixie was used) and decompress the `.img.xz`.
2. `git archive --prefix=software/ -o tbot-software.tar.gz HEAD:software` (unpacked into `~/TBOT` on first boot).
3. From an elevated PowerShell, after checking the card's disk number with `Get-Disk`:
   `powershell -ExecutionPolicy Bypass -File software\pi\flash.ps1 -DiskNumber <n> -Image <img> -Tarball tbot-software.tar.gz`
4. `boot/user-data` and `boot/network-config` are the cloud-init files it copies onto the boot partition.
5. First boot takes a few minutes (cloud-init, then a reboot). Then run `sudo ./scripts/pi_setup.sh` (docs/PROTOTYPING.md).

## Direct ethernet to the laptop (no router)

With nothing handing out DHCP, NetworkManager used to give up after 45 s, mark the wired connection failed and
bounce eth0 (the link LEDs blink, SSH drops; it looks like the Pi resetting but it is not). Fixed on the SD card
2026-11-16 (not in `boot/network-config`, so redo it after a reflash):

```bash
sudo nmcli con mod "Wired connection 1" ipv4.method auto ipv4.addresses 169.254.20.20/16 \
    ipv4.dhcp-timeout infinity ipv4.may-fail yes
sudo nmcli con up "Wired connection 1"
```

DHCP still works on a LAN; on a direct cable the Pi is always `169.254.20.20` (`ssh pi@169.254.20.20`, stream at
`http://169.254.20.20:8080/`). The laptop gets a 169.254.x.x address on its own.
