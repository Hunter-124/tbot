#!/usr/bin/env bash
# Build the detector's native measurement kernel: tbot/vision/_measure.c -> _measure.so (.dll on Windows,
# .dylib on macOS), next to the C file, where tbot/vision/_native.py looks for it.
#
#   ./scripts/build_native.sh            # uses $CC, else cc, gcc or clang, whichever is found first
#
# The kernel is optional: without it (or with TBOT_NATIVE=0, or detector.native = false) the detector runs
# its Python loop and gives the same detections, more slowly. The binary is not committed: build it on the
# machine that runs it (pi_setup.sh does).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/../tbot/vision/_measure.c"
[[ -f ${SRC} ]] || { echo "build_native: ${SRC} not found" >&2; exit 1; }

CC_BIN="${CC:-}"
if [[ -z ${CC_BIN} ]]; then
	for c in cc gcc clang; do
		if command -v "$c" >/dev/null 2>&1; then
			CC_BIN="$c"
			break
		fi
	done
fi
[[ -n ${CC_BIN} ]] || { echo "build_native: no C compiler found (cc, gcc, clang; or set CC)" >&2; exit 1; }

# -ffp-contract=off: no fused multiply-add, so ARM and x86 round the same way as the Python reference.
# -O3: the whole-frame loops (box means, cell statistics, labelling) are only vectorised at -O3 (Pi 3B: the
# 4x4 box mean 1.8 ms at -O2, 0.2 at -O3). No -ffast-math: results must not depend on the optimiser.
# One C file, so there is nothing for link-time optimisation to join (-flto measured: no change).
FLAGS=(-O3 -shared -ffp-contract=off -Wall -Wextra)
# Tuned for the processor it is built on (the library is never copied between machines). TBOT_NATIVE_CPU=0
# builds for the generic one.
if [[ ${TBOT_NATIVE_CPU:-1} != 0 ]]; then
	case "$(uname -m)" in
	aarch64 | arm64 | armv7l) CPU_FLAG=-mcpu=native ;;
	x86_64 | amd64) CPU_FLAG=-march=native ;;
	*) CPU_FLAG= ;;
	esac
	# (not every compiler knows "native" on every machine: use it only if a trial compile accepts it)
	if [[ -n ${CPU_FLAG} ]] && echo 'int main(void){return 0;}' | "${CC_BIN}" "${CPU_FLAG}" -x c -fsyntax-only - >/dev/null 2>&1; then
		FLAGS+=("${CPU_FLAG}")
	fi
fi
LIBS=()
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN* | Windows_NT)
	EXT=dll
	# a native Windows clang cannot use the POSIX-style TMPDIR of Git Bash ("unable to make temporary file")
	if [[ -n ${TMPDIR:-} ]] && command -v cygpath >/dev/null 2>&1; then
		TMPDIR="$(cygpath -w "${TMPDIR}")"
		export TMPDIR
	fi
	;;
Darwin) EXT=dylib; FLAGS+=(-fPIC) ;;
*) EXT=so; FLAGS+=(-fPIC); LIBS+=(-lm) ;;
esac
OUT="${SRC%.c}.${EXT}"
# (not TMP: that name is the temp directory of a Windows compiler)
PART="${OUT%.*}.part$$.${EXT}"
trap 'rm -f "${PART}"' EXIT

# Build to a temporary name and move into place: a failed build leaves the previous library alone.
"${CC_BIN}" "${FLAGS[@]}" -o "${PART}" "${SRC}" "${LIBS[@]}"
mv -f "${PART}" "${OUT}"
rm -f "${OUT%.*}.lib" "${OUT%.*}.exp" "${PART%.*}.lib" "${PART%.*}.exp" 2>/dev/null || true
echo "built ${OUT} with ${CC_BIN}"
