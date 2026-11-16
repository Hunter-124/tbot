"""`tbot calib-boresight` — camera-to-bore alignment (contract: docs/ARCHITECTURE.md [calib]).

Aim the gun at a target at a known range with a bore laser (or any mark the bore is on), tell the tool
which pixel that target lands on, and it solves the small rotation R_corr = Rz(yaw) Ry(pitch) Rx(roll)
that makes the camera model agree with the barrel:

    MountModel(rpy).cam_to_gun(ray(bore_pixel))  ==  normalize(muzzle + R*(1,0,0) - cam_pos)

Roll is unobservable from one point; give `--pixel2` on a plumb line through the target (a hanging
string) and the solver adds the constraint that the plane through both rays stays vertical.

`--laser` finds the bore-laser dot by itself (headless: no display, no pixel to read off). It averages
`--laser-frames` frames, masks the barrel, and with `--laser-turns N` asks you to turn the laser in the bore
N times and uses the centre of the circle the dot traces (a bore laser is never exactly on the bore axis).
`--laser-bg` first takes a frame with the beam blocked, so lamps and windows cancel. `--check` solves nothing:
it reports how far the dot is from where the stored calibration puts the bore line (use it at a second range).

`--barrel IMAGE` needs no target at all: the barrel is in the camera's view, and its two silhouette edges
are generators of the (tapered) barrel cone, so they meet in the image at the projection of the cone's apex
on the bore axis - the bore direction itself for a straight barrel. The tool fits both edges in the
undistorted ray plane, intersects them and solves pitch/yaw against that apex direction (roll stays fixed).
Take the still with the gun pointed at something plain and lit (a wall or sky) so the barrel stands out.

`--poi-offset-cm dy,dz@range_m` records a patterning result (where the pattern centre lands relative to
the point of aim) as `poi_offset_rad`; it is written independently of the rpy solve, so the two can be
measured in separate runs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from tbot.config import Calibration, Config, add_common_args, load_from_args, now_iso, update_json
from tbot.optics import CameraModel, MountModel, barrel_mask

GN_ITERS = 40
GN_TOL_RAD = 1e-12
_JAC_STEP_RAD = 1e-6
MAX_CORRECTION_DEG = 15.0  # a bigger "mount error" means the wrong pixel was given, not a tilted camera


def parse_pixel(text: str) -> tuple[float, float]:
    """"u,v" -> (u, v) floats."""
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected 'u,v', got {text!r}")
    return float(parts[0]), float(parts[1])


def parse_poi(text: str) -> tuple[float, float, float]:
    """"dy,dz@range_m" in cm -> (dy_m, dz_m, range_m): pattern centre relative to the point of aim,
    +dy gun-left, +dz up."""
    body, sep, rng = text.partition("@")
    if not sep:
        raise ValueError(f"expected 'dy,dz@range_m', got {text!r}")
    dy_cm, dz_cm = parse_pixel(body)
    range_m = float(rng)
    if range_m <= 0.0:
        raise ValueError(f"patterning range must be positive, got {range_m}")
    return dy_cm * 0.01, dz_cm * 0.01, range_m


def poi_offset_rad(dy_m: float, dz_m: float, range_m: float) -> list[float]:
    """[yaw, pitch] of the mean shot direction relative to the bore line."""
    return [math.atan2(dy_m, range_m), math.atan2(dz_m, range_m)]


def mount_for_rpy(cfg: Config, rpy_rad: np.ndarray) -> MountModel:
    """MountModel with this mount correction and no patterning offset (pure bore geometry)."""
    rpy_deg = [float(math.degrees(v)) for v in np.asarray(rpy_rad, dtype=np.float64).reshape(3)]
    return MountModel.from_config(cfg, {"rpy_deg": rpy_deg, "poi_offset_rad": [0.0, 0.0]})


def bore_target_dir(mount: MountModel, range_m: float) -> np.ndarray:
    """Unit gun-frame direction from the optical centre to the bore line at range_m.

    The bore line is the mechanical axis (1,0,0) from the muzzle: the patterning offset is a ballistic
    correction and must not enter the geometric alignment.
    """
    target = mount.muzzle_pos_g + float(range_m) * np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d = target - mount.cam_pos_g
    n = float(np.linalg.norm(d))
    if n <= 0.0:
        raise ValueError("camera and bore target coincide")
    return d / n


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _residuals(cam: CameraModel, cfg: Config, rpy: np.ndarray, bore_px: tuple[float, float],
               range_m: float, plumb_px: tuple[float, float] | None,
               target_dir_g: np.ndarray | None = None) -> np.ndarray:
    mount = mount_for_rpy(cfg, rpy)
    target = bore_target_dir(mount, range_m) if target_dir_g is None else target_dir_g
    d = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    res = list(np.cross(d, target))  # |cross| = sin(miss angle), zero only when the rays coincide
    if plumb_px is not None:
        d2 = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(plumb_px, dtype=np.float64))))
        n = _unit(np.cross(d, d2))
        res.append(float(n[2]))  # the plane through two points of a plumb line is vertical
    return np.array(res, dtype=np.float64)


def solve_mount_rpy(cam: CameraModel, cfg: Config, bore_px: tuple[float, float], range_m: float,
                    plumb_px: tuple[float, float] | None = None,
                    roll_rad: float = 0.0, target_dir_g: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Solve the camera mount correction (roll, pitch, yaw) in radians.

    Pitch and yaw come from the bore pixel; roll stays at `roll_rad` unless a plumb-line pixel is given.
    `target_dir_g` (unit, gun frame, from the optical centre) replaces "the bore line at range_m" as the
    direction the pixel must map to (used by --barrel). Small-angle closed form first, then Gauss-Newton.
    """
    if target_dir_g is None and range_m <= 0.0:
        raise ValueError(f"--range-m must be positive, got {range_m}")
    if target_dir_g is not None:
        target_dir_g = _unit(np.asarray(target_dir_g, dtype=np.float64).reshape(3))
    rpy = np.array([float(roll_rad), 0.0, 0.0], dtype=np.float64)

    # closed-form start: the rotation vector that takes the uncorrected ray onto the bore direction is
    # (roll, pitch, yaw) to first order, and only its pitch/yaw parts are observable from one point.
    mount0 = mount_for_rpy(cfg, rpy)
    d0 = _unit(mount0.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    target = bore_target_dir(mount0, range_m) if target_dir_g is None else target_dir_g
    axis = np.cross(d0, target)
    s = float(np.linalg.norm(axis))
    if s > 1e-15:
        w = (axis / s) * math.atan2(s, float(np.dot(d0, target)))
        rpy[1] += float(w[1])
        rpy[2] += float(w[2])

    free = (0, 1, 2) if plumb_px is not None else (1, 2)
    iters = 0
    for iters in range(1, GN_ITERS + 1):
        r = _residuals(cam, cfg, rpy, bore_px, range_m, plumb_px, target_dir_g)
        J = np.zeros((r.size, len(free)), dtype=np.float64)
        for col, k in enumerate(free):
            plus, minus = rpy.copy(), rpy.copy()
            plus[k] += _JAC_STEP_RAD
            minus[k] -= _JAC_STEP_RAD
            J[:, col] = (_residuals(cam, cfg, plus, bore_px, range_m, plumb_px, target_dir_g)
                         - _residuals(cam, cfg, minus, bore_px, range_m, plumb_px, target_dir_g)) / (2.0 * _JAC_STEP_RAD)
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        for col, k in enumerate(free):
            rpy[k] += float(step[col])
        if float(np.max(np.abs(step))) < GN_TOL_RAD:
            break

    mount = mount_for_rpy(cfg, rpy)
    d = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    target = bore_target_dir(mount, range_m) if target_dir_g is None else target_dir_g
    miss = float(math.atan2(float(np.linalg.norm(np.cross(d, target))), float(np.dot(d, target))))
    info = {
        "iterations": iters,
        "residual_rad": miss,
        "residual_deg": math.degrees(miss),
        "residual_px": miss / float(cam.ifov(np.array(bore_px, dtype=np.float64))),
        "correction_deg": float(np.max(np.abs(np.degrees(rpy)))),
    }
    if plumb_px is not None:
        d2 = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(plumb_px, dtype=np.float64))))
        info["plumb_tilt_deg"] = math.degrees(math.asin(abs(float(_unit(np.cross(d, d2))[2]))))
    return rpy, info


# ----------------------------------------------------------------------------------- barrel method
BARREL_SEARCH_FRAC = 0.12  # search this fraction of the width either side of the nominal barrel silhouette
BARREL_MIN_ROWS = 12


def barrel_apex_dir_g(cfg: Config) -> np.ndarray:
    """Gun-frame unit direction (from the optical centre) whose image is where the barrel edges meet.

    The silhouette edges of a cone are lines through its apex, so their images meet at the apex's image. For a
    barrel that narrows toward the muzzle the apex is ahead on the bore axis; for one that widens it is behind
    the camera and the pinhole image is that of the opposite (forward) direction; a straight barrel gives +X.
    """
    m = cfg.mount
    (x1, od1), (x2, od2) = (tuple(map(float, p)) for p in (m.barrel_od_mm[0], m.barrel_od_mm[-1]))
    cam = np.asarray(m.cam_pos_mm, dtype=np.float64)
    k = (od2 - od1) / 2.0 / (x2 - x1)  # radius slope (mm/mm)
    if abs(k) < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    apex = np.array([x1 - (od1 / 2.0) / k, 0.0, float(m.barrel_axis_z_mm)])
    v = apex - cam
    if v[0] < 0.0:
        v = -v
    return _unit(v)


def _fit_line(points: np.ndarray) -> tuple[float, float, float]:
    """x = a*y + b, iteratively trimmed least squares; returns (a, b, rms)."""
    keep = np.ones(len(points), dtype=bool)
    a = b = 0.0
    for _ in range(4):
        a, b = np.polyfit(points[keep, 1], points[keep, 0], 1)
        r = points[:, 0] - (a * points[:, 1] + b)
        sd = float(np.std(r[keep]))
        new = np.abs(r) <= max(2.5 * sd, 1e-4)
        if new.sum() < BARREL_MIN_ROWS or np.array_equal(new, keep):
            break
        keep = new
    r = points[keep, 0] - (a * points[keep, 1] + b)
    return float(a), float(b), float(np.sqrt(np.mean(r * r)))


def barrel_vanishing_pixel(image: np.ndarray, cfg: Config, cam: CameraModel) -> tuple[tuple[float, float], dict]:
    """Fit the barrel's two silhouette edges in `image` (at cam's resolution) and intersect them.

    Edges are found row by row as the strongest dark->bright pair of horizontal gradients near the nominal
    silhouette, converted to undistorted rays, and fitted as straight lines in the z=1 plane (a fisheye bends
    them in pixels). Returns the intersection pixel and fit statistics.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    h, w = gray.shape
    nominal = barrel_mask(cam, MountModel.from_config(cfg, None), cfg) == 0  # True = barrel (nominal mount)
    rows = np.flatnonzero(nominal.any(axis=1))
    if len(rows) < BARREL_MIN_ROWS:
        raise ValueError("the configured barrel geometry puts no barrel in this image")
    sm = cv2.GaussianBlur(gray, (0, 0), max(1.0, w / 640.0 * 1.5))
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=5)
    pad = int(BARREL_SEARCH_FRAC * w)
    y_end = int(rows.max() - 0.1 * (rows.max() - rows.min()))  # stay clear of the rounded muzzle end
    left, right = [], []
    for y in range(int(rows.min()), max(int(rows.min()) + 1, y_end), max(1, h // 200)):
        cols = np.flatnonzero(nominal[y])
        if not len(cols):
            continue
        lo, hi = max(0, int(cols.min()) - pad), min(w, int(cols.max()) + pad)
        g = gx[y, lo:hi]
        if len(g) < 8:
            continue
        il, ir = int(np.argmin(g)), int(np.argmax(g))  # barrel is darker: falling edge left, rising edge right
        if ir - il < 4 or g[il] > -0.25 * float(np.abs(gx[y]).max()) or g[ir] < 0.25 * float(np.abs(gx[y]).max()):
            continue
        left.append((lo + il, y))
        right.append((lo + ir, y))
    if len(left) < BARREL_MIN_ROWS:
        raise ValueError(f"found the barrel edges on only {len(left)} rows (need {BARREL_MIN_ROWS}); "
                         "point the gun at something plain and lit so the barrel stands out")

    def plane(pts):  # pixels -> undistorted (x/z, y/z)
        rays = cam.pixel_to_ray(np.asarray(pts, dtype=np.float64))
        return np.column_stack([rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]])

    al, bl, rl = _fit_line(plane(left))
    ar, br, rr = _fit_line(plane(right))
    if abs(al - ar) < 1e-6:
        raise ValueError("the barrel edges are parallel in the image: cannot intersect them")
    yv = (br - bl) / (al - ar)
    xv = al * yv + bl
    uv = cam.ray_to_pixel(np.array([[xv, yv, 1.0]]))[0]
    px_per_rad = 1.0 / float(cam.ifov(np.asarray(uv, dtype=np.float64)))
    return (float(uv[0]), float(uv[1])), {
        "rows": len(left), "edge_rms_px": [rl * px_per_rad, rr * px_per_rad],
        "left_px": left, "right_px": right,
    }


# --------------------------------------------------------------------------------- frame handling
def _frame_from_source(cfg: Config, calib: Calibration, gui: bool) -> np.ndarray:
    """Grab one frame from the configured source (GUI: live preview until SPACE/ENTER freezes it)."""
    from tbot.camera import open_source

    source = open_source(cfg.camera.source, cfg, calib)
    source.start()
    window = "tbot calib-boresight"
    try:
        if not gui:
            for _ in range(10):
                frame = source.read(timeout_s=1.0)
                if frame is not None:
                    return np.array(frame.image, copy=True)
            raise RuntimeError(f"no frame from {cfg.camera.source}")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        last = None
        while True:
            frame = source.read(timeout_s=1.0)
            if frame is not None:
                last = np.array(frame.image, copy=True)
            if last is None:
                continue
            vis = _bgr(last)
            cv2.putText(vis, "[space] freeze  [q] abort", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (32, 13, 10):
                return last
            if key in (27, ord("q")):
                raise KeyboardInterrupt
    finally:
        source.stop()
        if gui:
            cv2.destroyWindow(window)


def _bgr(image: np.ndarray) -> np.ndarray:
    return image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _click_pixel(image: np.ndarray, prompt: str) -> tuple[float, float]:
    """Let the operator click one point (GUI only); ENTER confirms, q aborts."""
    window = "tbot calib-boresight"
    picked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):  # pragma: no cover - interactive
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            vis = _bgr(image)
            if picked:
                u, v = picked[-1]
                cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 25, 1)
            cv2.putText(vis, prompt + "  [enter] accept  [q] abort", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and picked:
                return picked[-1]
            if key in (27, ord("q")):
                raise KeyboardInterrupt
    finally:
        cv2.destroyWindow(window)


# --------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot calib-boresight",
                                description="camera-to-bore alignment -> calib/boresight.json")
    add_common_args(p)
    g = p.add_argument_group("bore point")
    g.add_argument("--range-m", type=float, default=None, help="distance to the boresight target (m)")
    g.add_argument("--pixel", default=None, metavar="U,V", help="pixel the bore target lands on")
    g.add_argument("--pixel2", default=None, metavar="U,V",
                   help="second pixel on a plumb line through the target (gives roll)")
    g.add_argument("--image", default=None, metavar="PATH", help="still image instead of a live source")
    g.add_argument("--barrel", default=None, metavar="IMAGE",
                   help="solve pitch/yaw from the barrel silhouette in this still (no target needed; any "
                        "resolution with the operating field of view, e.g. a full 1280x800 rpicam-still)")
    g.add_argument("--barrel-debug", default=None, metavar="PNG", help="with --barrel: save the fitted edges")
    g.add_argument("--laser", action="store_true",
                   help="find the bore-laser dot automatically (live camera, or --image); headless")
    g.add_argument("--laser-frames", type=int, default=20, metavar="N", help="frames averaged per measurement")
    g.add_argument("--laser-turns", type=int, default=1, metavar="N",
                   help="measure N times, turning the laser in the bore between them (removes its wobble; 4 is good)")
    g.add_argument("--laser-bg", action="store_true",
                   help="first take a background with the beam blocked (lamps, windows and glare cancel)")
    g.add_argument("--laser-bg-image", default=None, metavar="PATH", help="with --image: the beam-blocked still")
    g.add_argument("--laser-debug", default=None, metavar="PNG",
                   help="annotated picture of the result (default <calib-dir>/boresight_laser.png)")
    g.add_argument("--check", action="store_true",
                   help="with --laser/--pixel: do not write; report the error of the stored calibration at this range")
    g.add_argument("--gui", action="store_true", help="click the point(s) in a window (needs a display)")
    g.add_argument("--plumb", action="store_true",
                   help="with --gui: also click a second point on a plumb line (gives roll)")
    g.add_argument("--roll-deg", type=float, default=None,
                   help="fix roll (default: the roll already in boresight.json, else mount.cam_rpy_deg)")
    o = p.add_argument_group("patterning / output")
    o.add_argument("--poi-offset-cm", default=None, metavar="DY,DZ@RANGE_M",
                   help="pattern centre relative to the point of aim: +dy gun-left, +dz up, in cm")
    o.add_argument("--out", default=None, help="output JSON (default <calib-dir>/boresight.json)")
    o.add_argument("--force", action="store_true", help="accept a correction larger than "
                   f"{MAX_CORRECTION_DEG:g} deg")
    args = p.parse_args(argv)

    cfg, calib = load_from_args(args)
    out = Path(args.out) if args.out else calib.path("boresight.json")
    patch: dict = {}

    if args.poi_offset_cm:
        try:
            dy, dz, rng = parse_poi(args.poi_offset_cm)
        except ValueError as e:
            print(f"tbot calib-boresight: --poi-offset-cm: {e}", file=sys.stderr)
            return 2
        patch["poi_offset_rad"] = poi_offset_rad(dy, dz, rng)
        print(f"poi    : pattern centre {dy * 100:+.1f} cm gun-left, {dz * 100:+.1f} cm up at {rng:g} m "
              f"-> poi_offset_rad [{patch['poi_offset_rad'][0]:+.6f}, {patch['poi_offset_rad'][1]:+.6f}] rad")

    if args.barrel:
        rc = _barrel_solve(args, cfg, calib, patch)
        if rc:
            return rc

    want_solve = bool(args.pixel or args.gui or args.image or args.laser)
    if want_solve:
        image: np.ndarray | None = None
        laser_info: dict | None = None
        if not args.pixel and not args.gui and not args.laser:
            print("tbot calib-boresight: need --pixel u,v (or --laser to find the dot, or --gui to click it)",
                  file=sys.stderr)
            return 2
        if args.laser and args.range_m is None:
            print("tbot calib-boresight: --laser needs --range-m (muzzle to the target, metres)", file=sys.stderr)
            return 2
        if args.laser and not args.pixel:
            try:
                image, laser_info = _laser_measure(args, cfg, calib)
            except KeyboardInterrupt:
                print("aborted", file=sys.stderr)
                return 1
            except (RuntimeError, ValueError, OSError) as e:
                print(f"tbot calib-boresight: --laser: {e}", file=sys.stderr)
                return 1
        elif args.image:
            image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"tbot calib-boresight: cannot read {args.image}", file=sys.stderr)
                return 1
        elif args.gui:
            try:
                image = _frame_from_source(cfg, calib, args.gui)
            except KeyboardInterrupt:
                print("aborted", file=sys.stderr)
                return 1
            except (RuntimeError, ValueError, OSError) as e:
                print(f"tbot calib-boresight: {e}", file=sys.stderr)
                return 1

        if image is not None:
            height, width = int(image.shape[0]), int(image.shape[1])
        elif calib.intrinsics:
            width = int(calib.intrinsics.get("width", cfg.camera.width))
            height = int(calib.intrinsics.get("height", cfg.camera.height))
        else:
            width, height = int(cfg.camera.width), int(cfg.camera.height)

        try:
            if args.pixel:
                bore_px = parse_pixel(args.pixel)
                plumb_px = parse_pixel(args.pixel2) if args.pixel2 else None
            elif laser_info is not None:
                bore_px = (float(laser_info["pixel"][0]), float(laser_info["pixel"][1]))
                plumb_px = parse_pixel(args.pixel2) if args.pixel2 else None
            else:  # --gui, with a frame from --image or the live source
                bore_px = _click_pixel(image, "click the BORE target")
                if args.pixel2:
                    plumb_px = parse_pixel(args.pixel2)
                elif args.plumb:
                    plumb_px = _click_pixel(image, "click a second point on the PLUMB LINE")
                else:
                    plumb_px = None
        except ValueError as e:
            print(f"tbot calib-boresight: {e}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("aborted", file=sys.stderr)
            return 1

        if args.range_m is None:
            print("tbot calib-boresight: --range-m is required to solve the mount rotation",
                  file=sys.stderr)
            return 2
        for name, px in (("--pixel", bore_px), ("--pixel2", plumb_px)):
            if px is not None and not (0 <= px[0] < width and 0 <= px[1] < height):
                print(f"tbot calib-boresight: {name} {px[0]:g},{px[1]:g} is outside the "
                      f"{width}x{height} image", file=sys.stderr)
                return 2

        cam = CameraModel.from_calibration(calib.intrinsics, cfg, width, height)
        if args.check:
            return _check(cfg, calib, cam, bore_px, float(args.range_m), image, laser_info, args)
        roll_rad = math.radians(args.roll_deg) if args.roll_deg is not None else _existing_roll_rad(cfg, calib)
        try:
            rpy, info = solve_mount_rpy(cam, cfg, bore_px, float(args.range_m), plumb_px, roll_rad)
        except (ValueError, RuntimeError) as e:
            print(f"tbot calib-boresight: {e}", file=sys.stderr)
            return 1

        nominal = MountModel.from_config(cfg, None).bore_pixel(cam, float(args.range_m))
        mount = mount_for_rpy(cfg, rpy)
        check = mount.bore_pixel(cam, float(args.range_m))
        rpy_deg = [float(math.degrees(v)) for v in rpy]
        print(f"camera : {width}x{height} {cam.model}  fx {cam.fx:.2f}  cx {cam.cx:.2f}  cy {cam.cy:.2f}"
              + ("  (nominal prior — run calib-intrinsics first)" if not calib.intrinsics else ""))
        print(f"target : pixel ({bore_px[0]:.2f}, {bore_px[1]:.2f}) at {args.range_m:g} m"
              + (f"; plumb pixel ({plumb_px[0]:.2f}, {plumb_px[1]:.2f})" if plumb_px else ""))
        print(f"before : the bore line at {args.range_m:g} m projected to ({nominal[0]:.2f}, {nominal[1]:.2f})")
        print(f"solved : rpy_deg roll {rpy_deg[0]:+.4f}  pitch {rpy_deg[1]:+.4f}  yaw {rpy_deg[2]:+.4f}"
              f"   ({info['iterations']} iterations, residual {info['residual_deg']:.2e} deg "
              f"= {info['residual_px']:.3f} px)")
        print(f"after  : the bore line at {args.range_m:g} m projects to ({check[0]:.2f}, {check[1]:.2f})")
        if plumb_px is not None:
            print(f"plumb  : residual tilt {info['plumb_tilt_deg']:.2e} deg")
        if info["correction_deg"] > MAX_CORRECTION_DEG and not args.force:
            print(f"tbot calib-boresight: correction of {info['correction_deg']:.1f} deg exceeds "
                  f"{MAX_CORRECTION_DEG:g} deg — check the pixel and the range (or --force)",
                  file=sys.stderr)
            return 1
        patch.update({
            "rpy_deg": rpy_deg,
            "range_m": float(args.range_m),
            "bore_pixel": [float(bore_px[0]), float(bore_px[1])],
            "width": width,
            "height": height,
            "method": "laser" if laser_info is not None else "pixel",
        })
        if laser_info is not None:
            ifov_deg = math.degrees(float(cam.ifov(np.array(bore_px, dtype=np.float64))))
            laser_info["wobble_deg"] = laser_info["wobble_px"] * ifov_deg
            patch["laser"] = laser_info
            prev = _stored_rpy_deg(calib)
            if prev is not None:
                print(f"change : pitch {rpy_deg[1] - prev[1]:+.3f}  yaw {rpy_deg[2] - prev[2]:+.3f} deg "
                      "from the stored calibration")
            if laser_info["turns"] > 1:
                print(f"laser  : dot circle radius {laser_info['wobble_px']:.2f} px = "
                      f"{laser_info['wobble_deg']:.3f} deg (the laser's own error, removed by the turns)")
            else:
                print("laser  : one orientation only - a bore laser is typically 0.1-0.3 deg off the bore axis; "
                      "rerun with --laser-turns 4 to remove that")
            _laser_debug(args, calib, image, laser_info, bore_px, cfg, cam)

    if not patch:
        print("tbot calib-boresight: nothing to do — give --pixel/--gui/--image (mount rotation) "
              "and/or --poi-offset-cm (patterning)", file=sys.stderr)
        return 2

    patch["created"] = now_iso()
    if "rpy_deg" in patch and out.exists():
        prev_path = out.with_name(out.stem + ".prev" + out.suffix)
        prev_path.write_bytes(out.read_bytes())  # one step of undo: copy it back to restore
        print(f"backup : {prev_path}")
    update_json(out, patch)
    print(f"wrote  : {out}")
    return 0


# --------------------------------------------------------------------------------- laser method
LASER_WARMUP_FRAMES = 8
CHECK_PASS_DEG = 0.15  # stored calibration vs a fresh laser dot: within this, leave it alone


def _stored_rpy_deg(calib: Calibration) -> list[float] | None:
    if calib.boresight and calib.boresight.get("rpy_deg") is not None:
        return [float(v) for v in calib.boresight["rpy_deg"]]
    return None


def _grab(source, n: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    misses = 0
    while len(frames) < n and misses < 30:
        frame = source.read(timeout_s=1.0)
        if frame is None:
            misses += 1
            continue
        frames.append(np.array(frame.image, copy=True))
    if not frames:
        raise RuntimeError("the camera delivered no frame (is tbot run / focus / the range service using it?)")
    return frames


def _ask(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        raise RuntimeError("this step needs a terminal (run it in an interactive SSH session)") from None


def _laser_measure(args, cfg: Config, calib: Calibration) -> tuple[np.ndarray, dict]:
    """Dot position from the live camera (or --image). Returns (a frame, info with 'pixel')."""
    from tbot.tools import laser_dot

    turns = max(1, int(args.laser_turns))
    n = max(1, int(args.laser_frames))
    background = None
    source = None
    still = None
    if args.image:
        still = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
        if still is None:
            raise RuntimeError(f"cannot read {args.image}")
        if args.laser_bg_image:
            background = cv2.imread(str(args.laser_bg_image), cv2.IMREAD_UNCHANGED)
            if background is None:
                raise RuntimeError(f"cannot read {args.laser_bg_image}")
        turns = 1
    else:
        from tbot.camera import open_source

        source = open_source(cfg.camera.source, cfg, calib)
        source.start()
    try:
        if source is not None:
            _grab(source, LASER_WARMUP_FRAMES)
            first = _grab(source, 1)[0]
        else:
            first = still
        h, w = int(first.shape[0]), int(first.shape[1])
        cam = CameraModel.from_calibration(calib.intrinsics, cfg, w, h)
        mask = barrel_mask(cam, MountModel.from_config(cfg, calib.boresight), cfg)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(2, w // 80))  # widen the barrel

        if args.laser_bg and source is not None:
            _ask("BLOCK the laser beam (hand in front of the muzzle) and press Enter ... ")
            _grab(source, 3)
            background = np.median(np.stack(_grab(source, min(n, 10))).astype(np.float32), axis=0)
            _ask("UNBLOCK the beam and press Enter ... ")
            _grab(source, 3)

        dots: list[tuple[float, float]] = []
        measures = []
        image = first
        for k in range(turns):
            if k and source is not None:
                _ask(f"turn the laser about 1/{turns} of a turn in the bore, let the gun settle, press Enter "
                     f"({k + 1}/{turns}) ... ")
                _grab(source, 3)
            frames = _grab(source, n) if source is not None else [still]
            m = laser_dot.measure_dot(frames, background, mask)
            image = frames[-1]
            measures.append(m)
            dots.append((m.u, m.v))
            print(f"dot {k + 1}/{turns}: ({m.u:.2f}, {m.v:.2f})  {m.peak_dn:.0f} DN above background, "
                  f"{m.diameter_px:.1f} px wide, scatter {m.scatter_px:.2f} px over {m.n_found} frames")
            for wmsg in m.warnings:
                print(f"  WARNING: {wmsg}")
    finally:
        if source is not None:
            source.stop()

    centre, radius = laser_dot.circle_centre(np.array(dots))
    return image, {
        "pixel": [float(centre[0]), float(centre[1])],
        "dots": [[float(u), float(v)] for u, v in dots],
        "turns": turns,
        "frames": n,
        "wobble_px": float(radius),
        "scatter_px": float(max(m.scatter_px for m in measures)),
        "background": background is not None,
        "warnings": sorted({wmsg for m in measures for wmsg in m.warnings}),
    }


def _laser_debug(args, calib: Calibration, image, laser_info: dict | None, bore_px, cfg: Config,
                 cam: CameraModel) -> None:
    if image is None:
        return
    from tbot.tools import laser_dot

    path = Path(args.laser_debug) if args.laser_debug else calib.path("boresight_laser.png")
    predicted = None
    prev = _stored_rpy_deg(calib)
    if prev is not None and args.range_m:
        predicted = mount_for_rpy(cfg, np.radians(prev)).bore_pixel(cam, float(args.range_m))
    dots = [tuple(d) for d in laser_info["dots"]] if laser_info else [tuple(bore_px)]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), laser_dot.annotate(image, dots, tuple(bore_px), predicted))
        print(f"debug  : {path}")
    except (OSError, cv2.error) as e:  # the picture is a convenience, never a reason to lose the solve
        print(f"debug  : not written ({e})")


def _check(cfg: Config, calib: Calibration, cam: CameraModel, bore_px, range_m: float, image,
           laser_info: dict | None, args) -> int:
    """--check: how far is the observed bore point from where the stored calibration puts it? Writes nothing."""
    prev = _stored_rpy_deg(calib)
    if prev is None:
        print("tbot calib-boresight: --check: no stored calibration (calib/boresight.json) to check",
              file=sys.stderr)
        return 2
    mount = mount_for_rpy(cfg, np.radians(prev))
    predicted = mount.bore_pixel(cam, range_m)
    d_obs = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    d_exp = bore_target_dir(mount, range_m)
    err = math.atan2(float(np.linalg.norm(np.cross(d_obs, d_exp))), float(np.dot(d_obs, d_exp)))
    # +left / +up as seen from behind the gun (gun frame +Y left, +Z up)
    left_deg = math.degrees(math.atan2(float(d_obs[1]), float(d_obs[0]))
                            - math.atan2(float(d_exp[1]), float(d_exp[0])))
    up_deg = math.degrees(math.asin(float(d_obs[2])) - math.asin(float(d_exp[2])))
    print(f"stored : rpy_deg roll {prev[0]:+.4f}  pitch {prev[1]:+.4f}  yaw {prev[2]:+.4f} puts the bore line at "
          f"{range_m:g} m on ({predicted[0]:.2f}, {predicted[1]:.2f})")
    print(f"seen   : ({bore_px[0]:.2f}, {bore_px[1]:.2f})  ->  the bore is {abs(left_deg):.3f} deg "
          f"{'left' if left_deg >= 0 else 'right'} and {abs(up_deg):.3f} deg {'above' if up_deg >= 0 else 'below'} "
          "where the calibration says")
    print(f"error  : {math.degrees(err):.3f} deg = {err * range_m * 100.0:.1f} cm at {range_m:g} m "
          f"({err * 20.0 * 100.0:.1f} cm at 20 m; the pattern is ~76 cm wide there)")
    _laser_debug(args, calib, image, laser_info, bore_px, cfg, cam)
    ok = math.degrees(err) <= CHECK_PASS_DEG
    print(f"{'PASS' if ok else 'FAIL'}   : limit {CHECK_PASS_DEG:g} deg"
          + ("" if ok else " - rerun without --check to recalibrate"))
    return 0 if ok else 1


def _barrel_solve(args, cfg: Config, calib: Calibration, patch: dict) -> int:
    """--barrel: fit the barrel edges, solve pitch/yaw against the cone-apex direction, fill `patch`."""
    image = cv2.imread(str(args.barrel), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"tbot calib-boresight: cannot read {args.barrel}", file=sys.stderr)
        return 1
    if cfg.camera.rotate_180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    width, height = int(cfg.camera.width), int(cfg.camera.height)
    if calib.intrinsics:
        width = int(calib.intrinsics.get("width", width))
        height = int(calib.intrinsics.get("height", height))
    h_img, w_img = image.shape
    if abs(w_img / h_img - width / height) > 0.02:
        print(f"tbot calib-boresight: {w_img}x{h_img} does not have the {width}x{height} field of view",
              file=sys.stderr)
        return 2
    cam_img = CameraModel.from_calibration(calib.intrinsics, cfg, w_img, h_img)
    try:
        px_img, fit = barrel_vanishing_pixel(image, cfg, cam_img)
    except ValueError as e:
        print(f"tbot calib-boresight: --barrel: {e}", file=sys.stderr)
        return 1
    s = width / w_img
    bore_px = ((px_img[0] + 0.5) * s - 0.5, (px_img[1] + 0.5) * s - 0.5)
    cam = CameraModel.from_calibration(calib.intrinsics, cfg, width, height)
    target = barrel_apex_dir_g(cfg)
    roll_rad = math.radians(args.roll_deg) if args.roll_deg is not None else _existing_roll_rad(cfg, calib)
    rpy, info = solve_mount_rpy(cam, cfg, bore_px, 0.0, None, roll_rad, target_dir_g=target)
    rpy_deg = [float(math.degrees(v)) for v in rpy]
    taper_deg = math.degrees(math.atan2(float(target[2]), float(target[0])))
    print(f"barrel : edges on {fit['rows']} rows, rms {fit['edge_rms_px'][0]:.2f} / {fit['edge_rms_px'][1]:.2f} px; "
          f"meet at ({px_img[0]:.1f}, {px_img[1]:.1f}) in {w_img}x{h_img} -> ({bore_px[0]:.2f}, {bore_px[1]:.2f})")
    print(f"taper  : barrel OD profile puts that point {taper_deg:+.2f} deg in pitch off the bore direction")
    print(f"camera : {width}x{height} {cam.model}  fx {cam.fx:.2f}"
          + ("  (nominal prior — run calib-intrinsics first)" if not calib.intrinsics else ""))
    print(f"solved : rpy_deg roll {rpy_deg[0]:+.4f}  pitch {rpy_deg[1]:+.4f}  yaw {rpy_deg[2]:+.4f}"
          f"   ({info['iterations']} iterations, residual {info['residual_deg']:.2e} deg)")
    if args.barrel_debug:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for pts, col in ((fit["left_px"], (0, 0, 255)), (fit["right_px"], (0, 255, 0))):
            for x, y in pts:
                cv2.circle(vis, (int(x), int(y)), 2, col, -1)
        cv2.drawMarker(vis, (int(round(px_img[0])), int(round(px_img[1]))), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.imwrite(str(args.barrel_debug), vis)
        print(f"debug  : {args.barrel_debug}")
    if info["correction_deg"] > MAX_CORRECTION_DEG and not args.force:
        print(f"tbot calib-boresight: correction of {info['correction_deg']:.1f} deg exceeds "
              f"{MAX_CORRECTION_DEG:g} deg — check the image (or --force)", file=sys.stderr)
        return 1
    patch.update({
        "rpy_deg": rpy_deg,
        "method": "barrel",
        "barrel_pixel": [float(bore_px[0]), float(bore_px[1])],
        "barrel_apex_dir_g": [float(v) for v in target],
        "width": width,
        "height": height,
    })
    return 0


def _existing_roll_rad(cfg: Config, calib: Calibration) -> float:
    """Roll to keep when no plumb line is given: previous calibration, else the config prior."""
    if calib.boresight and calib.boresight.get("rpy_deg") is not None:
        return math.radians(float(calib.boresight["rpy_deg"][0]))
    return math.radians(float(cfg.mount.cam_rpy_deg[0]))


if __name__ == "__main__":
    raise SystemExit(main())
