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
