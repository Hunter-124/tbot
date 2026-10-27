"""`tbot board` — write a printable calibration chessboard (contract: docs/ARCHITECTURE.md [calib]).

Config-free on purpose: the board is a physical artefact, not a runtime setting. --cols/--rows are the
INNER corner counts OpenCV wants (a 9x6 board has 10x7 squares), so the printed file name and the
`tbot calib-intrinsics --cols/--rows` arguments always agree.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from tbot.config import resolve_path

MM_PER_INCH = 25.4
SCALE_BAR_MM = 50.0  # printed reference length: measure it to confirm the printer did not rescale
MARGIN_SQUARES = 1.0  # white quiet zone around the board, in squares (findChessboardCornersSB needs one)
FOOTER_MIN_MM = 16.0  # band under the board holding the scale bar and the identification line
BLACK = 20  # not 0: ink coverage of a solid black square curls paper and blooms into the white squares
WHITE = 245  # not 255: keeps the printed contrast symmetric around mid grey


def square_px(square_mm: float, dpi: float) -> int:
    """Printed size of one square in pixels (rounded to a whole pixel)."""
    return int(round(square_mm * dpi / MM_PER_INCH))


def board_layout(cols: int, rows: int, square_mm: float, dpi: float) -> dict:
    """Pixel layout of the sheet: square size, margins, footer band, image size, board origin."""
    if cols < 3 or rows < 3:
        raise ValueError(f"pattern too small: {cols}x{rows} inner corners (need >= 3x3)")
    if square_mm <= 0 or dpi <= 0:
        raise ValueError("--square-mm and --dpi must be positive")
    sq = square_px(square_mm, dpi)
    if sq < 8:
        raise ValueError(f"{square_mm} mm at {dpi} dpi is only {sq} px per square: raise --dpi or --square-mm")
    margin = int(round(MARGIN_SQUARES * sq))
    footer = max(sq, int(round(FOOTER_MIN_MM * dpi / MM_PER_INCH)))
    board_w = (cols + 1) * sq
    board_h = (rows + 1) * sq
    bar_px = int(round(SCALE_BAR_MM * dpi / MM_PER_INCH))
    width = max(board_w + 2 * margin, bar_px + 2 * margin + int(round(20.0 * dpi / MM_PER_INCH)))
    height = 2 * margin + board_h + footer
    return {
        "square_px": sq,
        "margin_px": margin,
        "footer_px": footer,
        "board_px": (board_w, board_h),
        "width": width,
        "height": height,
        "origin": ((width - board_w) // 2, margin),
        "bar_px": bar_px,
    }


def render_board(cols: int, rows: int, square_mm: float, dpi: float) -> np.ndarray:
    """The full sheet as a uint8 grayscale image: chessboard + quiet zone + scale bar + legend."""
    lay = board_layout(cols, rows, square_mm, dpi)
    sq = lay["square_px"]
    w, h = lay["width"], lay["height"]
    ox, oy = lay["origin"]
    img = np.full((h, w), WHITE, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = oy + r * sq, ox + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = BLACK

    footer_top = 2 * lay["margin_px"] + lay["board_px"][1]
    footer = lay["footer_px"]
    bar = lay["bar_px"]
    thick = max(1, int(round(dpi / 300.0)))
    bar_x = lay["margin_px"]
    bar_y = footer_top + int(round(0.35 * footer))
    tick = max(2, int(round(0.12 * footer)))
    cv2.line(img, (bar_x, bar_y), (bar_x + bar, bar_y), BLACK, thick)
    for x in (bar_x, bar_x + bar):
        cv2.line(img, (x, bar_y - tick), (x, bar_y + tick), BLACK, thick)
    mid = bar_x + bar // 2
    cv2.line(img, (mid, bar_y - tick // 2), (mid, bar_y + tick // 2), BLACK, thick)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"{SCALE_BAR_MM:g} mm"
    info = (f"TBOT board  {cols}x{rows} inner corners  {square_mm:g} mm squares  {dpi:g} dpi  "
            f"print at 100% (no fit-to-page)")
    scale_lab = _fit_text_scale(label, font, thick, max_w=w - (bar_x + bar) - 2 * lay["margin_px"],
                               target_h=0.22 * footer)
    cv2.putText(img, label, (bar_x + bar + tick, bar_y + tick), font, scale_lab, BLACK, thick, cv2.LINE_AA)
    scale_info = _fit_text_scale(info, font, thick, max_w=w - 2 * lay["margin_px"], target_h=0.20 * footer)
    cv2.putText(img, info, (bar_x, footer_top + int(round(0.85 * footer))), font, scale_info, BLACK,
                thick, cv2.LINE_AA)
    return img


def _fit_text_scale(text: str, font: int, thickness: int, max_w: float, target_h: float) -> float:
    """Largest font scale with glyph height ~target_h that still fits in max_w pixels."""
    (ref_w, ref_h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = max(0.3, target_h / max(ref_h, 1))
    if ref_w * scale > max_w:
        scale = max_w / max(ref_w, 1)
    return float(scale)


def _write_png_dpi(path: Path, dpi: float) -> None:
    """Stamp a pHYs chunk so viewers/printers know the physical size (PNG has no DPI otherwise)."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return
    idat = data.find(b"IDAT")
    if idat < 4:
        return
    ppm = int(round(dpi * 1000.0 / MM_PER_INCH))
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(body) - 4) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    at = idat - 4
    path.write_bytes(data[:at] + chunk + data[at:])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot board",
                                description="write a printable calibration chessboard (PNG)")
    p.add_argument("--cols", type=int, default=9, help="INNER corners across (default 9 -> 10 squares)")
    p.add_argument("--rows", type=int, default=6, help="INNER corners down (default 6 -> 7 squares)")
    p.add_argument("--square-mm", type=float, default=25.0, help="printed square size, mm (default 25)")
    p.add_argument("--dpi", type=float, default=300.0, help="printer resolution (default 300)")
    p.add_argument("--out", default=None, help="output PNG (default calib/board_<cols>x<rows>_<sq>mm.png)")
    args = p.parse_args(argv)

    try:
        img = render_board(args.cols, args.rows, args.square_mm, args.dpi)
    except ValueError as e:
        p.error(str(e))
        return 2  # pragma: no cover - argparse exits

    out = Path(args.out) if args.out else Path("calib") / f"board_{args.cols}x{args.rows}_{args.square_mm:g}mm.png"
    out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        print(f"tbot board: cannot write {out}")
        return 1
    if out.suffix.lower() == ".png":
        _write_png_dpi(out, args.dpi)

    h, w = img.shape
    board_mm = ((args.cols + 1) * args.square_mm, (args.rows + 1) * args.square_mm)
    sheet_mm = (w * MM_PER_INCH / args.dpi, h * MM_PER_INCH / args.dpi)
    print(f"board : {args.cols}x{args.rows} inner corners = {args.cols + 1}x{args.rows + 1} squares of "
          f"{args.square_mm:g} mm = {board_mm[0]:.1f} x {board_mm[1]:.1f} mm")
    print(f"sheet : {w} x {h} px at {args.dpi:g} dpi = {sheet_mm[0]:.1f} x {sheet_mm[1]:.1f} mm "
          f"({square_px(args.square_mm, args.dpi)} px per square)")
    print(f"wrote : {out}")
    print(f"print at 100 % (no 'fit to page' / 'shrink oversized pages'), then measure the "
          f"{SCALE_BAR_MM:g} mm bar; glue the sheet to something rigid and flat before calibrating.")
    print(f"then  : tbot calib-intrinsics --cols {args.cols} --rows {args.rows} --square-mm {args.square_mm:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
