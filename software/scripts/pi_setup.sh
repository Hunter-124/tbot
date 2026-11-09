#!/usr/bin/env bash
# TBOT Raspberry Pi provisioning. Idempotent: run it as often as you like.
#
#   cd ~/TBOT/software && sudo ./scripts/pi_setup.sh
#
# Installs the apt packages the Pi path needs, enables the OV9281 camera overlay, I2C (400 kHz),
# hardware PWM0 on GPIO18 and the serial console in config.txt, puts the invoking user in the
# video/gpio/i2c groups, and creates ~/tbot-venv (--system-site-packages, so apt's OpenCV/numpy/
# picamera2/lgpio are used) with TBOT installed editable.
set -euo pipefail

BLOCK_BEGIN="# --- tbot ---"
BLOCK_END="# --- end tbot ---"

# Lines that must exist exactly once, inside our block, in config.txt.
CONFIG_LINES=(
	"camera_auto_detect=0"            # the OV9281 is not auto-detectable: load it explicitly
	"dtoverlay=ov9281"                # no parameter needed on Pi 3B: https://www.inno-maker.com/product/cam-mipi9281raw-v2/
	"dtparam=i2c_arm=on"              # GY-521 / MPU-6050 on bus 1 (BCM 2/3)
	"dtparam=i2c_arm_baudrate=400000" # imu.sample_rate_hz = 500 needs fast-mode I2C
	"dtoverlay=pwm,pin=18,func=2"     # PWM0 on BCM 18 for the MG996R servo
	"dtparam=audio=off"               # analogue audio shares PWM0: it must be off
	"enable_uart=1"                   # serial console on BCM 14/15, 115200 (docs/PROTOTYPING.md)
)

# Any pre-existing setting of these keys outside our block gets commented out, so each line above
# is in force exactly once.
CONFLICT_RE='^[[:space:]]*(camera_auto_detect|enable_uart)[[:space:]]*='
CONFLICT_RE+='|^[[:space:]]*dtparam[[:space:]]*=[[:space:]]*(audio|i2c_arm|i2c_arm_baudrate)[[:space:]]*='
CONFLICT_RE+='|^[[:space:]]*dtoverlay[[:space:]]*=[[:space:]]*(ov9281|pwm)([,=]|[[:space:]]*$)'

APT_PACKAGES=(
	python3-opencv
	python3-numpy
	python3-picamera2
	python3-lgpio
	python3-smbus                     # available on Bookworm and Trixie; smbus2 is not in Trixie
	i2c-tools
	python3-venv
	python3-pytest
)

info() { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() {
	printf 'pi_setup: %s\n' "$*" >&2
	exit 1
}

REBOOT_NEEDED=0

# --------------------------------------------------------------------------- preconditions
[[ ${EUID} -eq 0 ]] || die "run me with sudo: sudo ./scripts/pi_setup.sh"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOFTWARE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
[[ -f "${SOFTWARE_DIR}/pyproject.toml" && -d "${SOFTWARE_DIR}/tbot" ]] ||
	die "${SOFTWARE_DIR} is not the TBOT software directory (no pyproject.toml / tbot/)"

MODEL_FILE=/proc/device-tree/model
[[ -r ${MODEL_FILE} ]] || die "no ${MODEL_FILE}: this is not a Raspberry Pi"
MODEL="$(tr -d '\0' <"${MODEL_FILE}")"
case "${MODEL}" in
*[Rr]aspberry\ [Pp]i*) ;;
*) die "this is not a Raspberry Pi (${MODEL})" ;;
esac

TARGET_USER="${SUDO_USER:-}"
if [[ -z ${TARGET_USER} || ${TARGET_USER} == root ]]; then
	TARGET_USER="$(stat -c '%U' "${SOFTWARE_DIR}")"
fi
[[ -n ${TARGET_USER} && ${TARGET_USER} != root ]] ||
	die "cannot determine the non-root user to set up; run via sudo from your own login"
TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
[[ -d ${TARGET_HOME} ]] || die "no home directory for ${TARGET_USER}"
VENV_DIR="${TARGET_HOME}/tbot-venv"

info "${MODEL}"
note "software : ${SOFTWARE_DIR}"
note "user     : ${TARGET_USER} (${TARGET_HOME})"

# --------------------------------------------------------------------------- apt packages
missing=()
for pkg in "${APT_PACKAGES[@]}"; do
	if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q '^install ok installed$'; then
		missing+=("${pkg}")
	fi
done
if ((${#missing[@]})); then
	info "installing ${#missing[@]} package(s): ${missing[*]}"
	export DEBIAN_FRONTEND=noninteractive
	apt-get update
	apt-get install -y --no-install-recommends "${missing[@]}"
else
	info "apt packages already installed"
fi

# --------------------------------------------------------------------------- config.txt
CONFIG_TXT=""
for candidate in /boot/firmware/config.txt /boot/config.txt; do
	if [[ -f ${candidate} ]]; then
		CONFIG_TXT="${candidate}"
		break
	fi
done
[[ -n ${CONFIG_TXT} ]] || die "no config.txt at /boot/firmware/config.txt or /boot/config.txt"
# Raspberry Pi OS Bookworm and later mount the firmware partition at /boot/firmware:
# https://www.raspberrypi.com/documentation/computers/config_txt.html
info "firmware config: ${CONFIG_TXT}"

BACKUP="${CONFIG_TXT}.tbot-backup"
if [[ -e ${BACKUP} ]]; then
	note "backup already exists: ${BACKUP}"
else
	cp -a -- "${CONFIG_TXT}" "${BACKUP}"
	note "backed up to ${BACKUP}"
fi

NEW_CONFIG="$(mktemp)"
STRIPPED="$(mktemp)"
trap 'rm -f "${NEW_CONFIG}" "${STRIPPED}"' EXIT

# Refuse a damaged managed block rather than silently deleting the rest of config.txt.
awk -v b="${BLOCK_BEGIN}" -v e="${BLOCK_END}" '
	$0 == b { if (inside) exit 1; inside = 1; next }
	$0 == e { if (!inside) exit 1; inside = 0 }
	END { if (inside) exit 1 }
' "${CONFIG_TXT}" || die "unbalanced tbot block in ${CONFIG_TXT}; repair it before rerunning"
# Drop the previous tbot block and comment out conflicting settings elsewhere.
awk -v b="${BLOCK_BEGIN}" -v e="${BLOCK_END}" '
	$0 == b { skip = 1; next }
	$0 == e { skip = 0; next }
	!skip   { print }
' "${CONFIG_TXT}" |
	awk -v re="${CONFLICT_RE}" '
		$0 ~ re { print "#tbot# " $0; next }
		        { print }
	' >"${STRIPPED}"

# Trim trailing blank lines, then append the block.
awk 'BEGIN { blanks = 0 }
	/^[[:space:]]*$/ { blanks++; next }
	{ while (blanks-- > 0) print ""; blanks = 0; print }
' "${STRIPPED}" >"${NEW_CONFIG}"
{
	echo ""
	echo "${BLOCK_BEGIN}"
	echo "# managed by software/scripts/pi_setup.sh - edit there, not here"
	echo "[all]" # reset any conditional filter ([pi4], [cm5], ...) left open above
	printf '%s\n' "${CONFIG_LINES[@]}"
	echo "${BLOCK_END}"
} >>"${NEW_CONFIG}"

if cmp -s "${NEW_CONFIG}" "${CONFIG_TXT}"; then
	info "config.txt already has the tbot block"
else
	cat -- "${NEW_CONFIG}" >"${CONFIG_TXT}" # rewrite in place: keeps owner and mode
	info "config.txt updated (${#CONFIG_LINES[@]} settings)"
	REBOOT_NEEDED=1
fi

# --------------------------------------------------------------------------- i2c-dev
# dtparam=i2c_arm=on only binds the controller; /dev/i2c-1 (smbus, i2cdetect) needs the i2c-dev
# module, which raspi-config normally adds. Measured on Trixie 2026-10-06 Lite: no /dev/i2c-* without it.
I2C_DEV_CONF=/etc/modules-load.d/tbot-i2c-dev.conf
if [[ -f ${I2C_DEV_CONF} ]] || grep -qsx 'i2c-dev' /etc/modules /etc/modules-load.d/*.conf; then
	note "i2c-dev already loaded at boot"
else
	echo "i2c-dev" >"${I2C_DEV_CONF}"
	info "i2c-dev added to ${I2C_DEV_CONF}"
fi
modprobe i2c-dev || note "modprobe i2c-dev failed; it loads at the next boot"

# --------------------------------------------------------------------------- groups
for grp in video gpio i2c; do
	if ! getent group "${grp}" >/dev/null; then
		note "group ${grp} does not exist on this system, skipped"
	elif id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx "${grp}"; then
		note "${TARGET_USER} already in ${grp}"
	else
		usermod -aG "${grp}" "${TARGET_USER}"
		info "added ${TARGET_USER} to ${grp}"
		REBOOT_NEEDED=1
	fi
done

# --------------------------------------------------------------------------- venv
if [[ -x "${VENV_DIR}/bin/python" ]]; then
	info "venv already present: ${VENV_DIR}"
	"${VENV_DIR}/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix; import cv2, numpy, picamera2, lgpio, smbus' ||
		die "existing venv cannot import the Pi apt packages; recreate ${VENV_DIR} with --system-site-packages"
else
	info "creating venv ${VENV_DIR} (--system-site-packages)"
	runuser -u "${TARGET_USER}" -- python3 -m venv --system-site-packages "${VENV_DIR}"
fi

info "installing tbot (editable, --no-deps) into ${VENV_DIR}"
if ! runuser -u "${TARGET_USER}" -- "${VENV_DIR}/bin/pip" install --no-deps -e "${SOFTWARE_DIR}"; then
	note "retrying without build isolation (no network / no PyPI)"
	runuser -u "${TARGET_USER}" -- "${VENV_DIR}/bin/pip" install \
		--no-deps --no-build-isolation -e "${SOFTWARE_DIR}" ||
		die "editable install failed; check pip output above"
fi

# --------------------------------------------------------------------------- working directories
for d in calib sessions; do
	if [[ -d "${SOFTWARE_DIR}/${d}" ]]; then
		note "${d}/ exists"
	else
		install -d -o "${TARGET_USER}" -g "$(id -gn "${TARGET_USER}")" "${SOFTWARE_DIR}/${d}"
		info "created ${SOFTWARE_DIR}/${d}"
	fi
done

# --------------------------------------------------------------------------- next steps
cat <<EOF

Done.

Next steps:
EOF
if ((REBOOT_NEEDED)); then
	cat <<EOF
  1. sudo reboot                        # config.txt / group changes need it
  2. rpicam-hello --list-cameras        # the OV9281 must appear (1280x800 mono)
  3. sudo i2cdetect -y 1                # the MPU-6050 must answer at 0x68
  4. ${VENV_DIR}/bin/tbot run --stream
EOF
else
	cat <<EOF
  1. rpicam-hello --list-cameras        # the OV9281 must appear (1280x800 mono)
  2. sudo i2cdetect -y 1                # the MPU-6050 must answer at 0x68
  3. ${VENV_DIR}/bin/tbot run --stream
EOF
fi
cat <<EOF

Then open http://\$(hostname).local:8080/ from the laptop (docs/PROTOTYPING.md), and work through
docs/CALIBRATION.md. The pipeline is a dry run until fire.live is set: it never drives the trigger.
Add ${VENV_DIR}/bin to PATH if you want to type just 'tbot'.
EOF
