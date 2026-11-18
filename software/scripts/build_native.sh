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
FLAGS=(-O2 -shared -ffp-contract=off -Wall -Wextra)
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
