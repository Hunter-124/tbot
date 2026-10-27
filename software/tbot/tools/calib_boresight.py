"""`tbot calib-boresight` — camera-to-bore alignment (contract: docs/ARCHITECTURE.md [calib]).

Aim the gun at a target at a known range with a bore laser (or any mark the bore is on), tell the tool
which pixel that target lands on, and it solves the small rotation R_corr = Rz(yaw) Ry(pitch) Rx(roll)
that makes the camera model agree with the barrel:

    MountModel(rpy).cam_to_gun(ray(bore_pixel))  ==  normalize(muzzle + R*(1,0,0) - cam_pos)

Roll is unobservable from one point; give `--pixel2` on a plumb line through the target (a hanging
string) and the solver adds the constraint that the plane through both rays stays vertical.

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
from tbot.optics import CameraModel, MountModel

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
               range_m: float, plumb_px: tuple[float, float] | None) -> np.ndarray:
    mount = mount_for_rpy(cfg, rpy)
    target = bore_target_dir(mount, range_m)
    d = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    res = list(np.cross(d, target))  # |cross| = sin(miss angle), zero only when the rays coincide
    if plumb_px is not None:
        d2 = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(plumb_px, dtype=np.float64))))
        n = _unit(np.cross(d, d2))
        res.append(float(n[2]))  # the plane through two points of a plumb line is vertical
    return np.array(res, dtype=np.float64)


def solve_mount_rpy(cam: CameraModel, cfg: Config, bore_px: tuple[float, float], range_m: float,
                    plumb_px: tuple[float, float] | None = None,
                    roll_rad: float = 0.0) -> tuple[np.ndarray, dict]:
    """Solve the camera mount correction (roll, pitch, yaw) in radians.

    Pitch and yaw come from the bore pixel; roll stays at `roll_rad` unless a plumb-line pixel is given.
    Small-angle closed form first (the correction rotates the observed ray onto the bore direction),
    then Gauss-Newton on the exact model.
    """
    if range_m <= 0.0:
        raise ValueError(f"--range-m must be positive, got {range_m}")
    rpy = np.array([float(roll_rad), 0.0, 0.0], dtype=np.float64)

    # closed-form start: the rotation vector that takes the uncorrected ray onto the bore direction is
    # (roll, pitch, yaw) to first order, and only its pitch/yaw parts are observable from one point.
    mount0 = mount_for_rpy(cfg, rpy)
    d0 = _unit(mount0.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    target = bore_target_dir(mount0, range_m)
    axis = np.cross(d0, target)
    s = float(np.linalg.norm(axis))
    if s > 1e-15:
        w = (axis / s) * math.atan2(s, float(np.dot(d0, target)))
        rpy[1] += float(w[1])
        rpy[2] += float(w[2])

    free = (0, 1, 2) if plumb_px is not None else (1, 2)
    iters = 0
    for iters in range(1, GN_ITERS + 1):
        r = _residuals(cam, cfg, rpy, bore_px, range_m, plumb_px)
        J = np.zeros((r.size, len(free)), dtype=np.float64)
        for col, k in enumerate(free):
            plus, minus = rpy.copy(), rpy.copy()
            plus[k] += _JAC_STEP_RAD
            minus[k] -= _JAC_STEP_RAD
            J[:, col] = (_residuals(cam, cfg, plus, bore_px, range_m, plumb_px)
                         - _residuals(cam, cfg, minus, bore_px, range_m, plumb_px)) / (2.0 * _JAC_STEP_RAD)
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        for col, k in enumerate(free):
            rpy[k] += float(step[col])
        if float(np.max(np.abs(step))) < GN_TOL_RAD:
            break

    mount = mount_for_rpy(cfg, rpy)
    d = _unit(mount.cam_to_gun(cam.pixel_to_ray(np.array(bore_px, dtype=np.float64))))
    target = bore_target_dir(mount, range_m)
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

    want_solve = bool(args.pixel or args.gui or args.image)
    if want_solve:
        image: np.ndarray | None = None
        if not args.pixel and not args.gui:
            print("tbot calib-boresight: need --pixel u,v (or --gui to click it)", file=sys.stderr)
            return 2
        if args.image:
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
        })

    if not patch:
        print("tbot calib-boresight: nothing to do — give --pixel/--gui/--image (mount rotation) "
              "and/or --poi-offset-cm (patterning)", file=sys.stderr)
        return 2

    patch["created"] = now_iso()
    update_json(out, patch)
    print(f"wrote  : {out}")
    return 0


def _existing_roll_rad(cfg: Config, calib: Calibration) -> float:
    """Roll to keep when no plumb line is given: previous calibration, else the config prior."""
    if calib.boresight and calib.boresight.get("rpy_deg") is not None:
        return math.radians(float(calib.boresight["rpy_deg"][0]))
    return math.radians(float(cfg.mount.cam_rpy_deg[0]))


if __name__ == "__main__":
    raise SystemExit(main())
