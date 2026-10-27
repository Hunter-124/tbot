"""Behaviour tests for the clay detector and the auto-mask core.

Scenes are built here (gradient sky + Gaussian noise + supersampled discs) so the detector is tested
against known ground truth without depending on the synthetic camera source.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tbot.config import load_config
from tbot.tools.calib_detector import psf_from_diameters, static_mask
from tbot.types import Frame
from tbot.vision.detector import ClayDetector

W, H = 640, 400


# --------------------------------------------------------------------------------- scene builders
def sky(rng, h=H, w=W, level=170.0, gradient=40.0, noise=2.0, clouds=0.0) -> np.ndarray:
    """Float32 sky: vertical brightness gradient (+ optional smooth cloud texture) + Gaussian noise."""
    img = np.empty((h, w), np.float32)
    img[:] = level + gradient * (np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] - 0.5)
    if clouds > 0.0:
        coarse = rng.normal(0.0, clouds, (7, 10)).astype(np.float32)
        img += cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    if noise > 0.0:
        img += rng.normal(0.0, noise, (h, w)).astype(np.float32)
    return img


def add_ellipse(img, cx, cy, major, minor, contrast, angle_deg=0.0, ss=8) -> None:
    """Add a supersampled (area-exact) filled ellipse of signed `contrast` DN, in place."""
    a, b = major / 2.0, minor / 2.0
    r = int(np.ceil(max(a, b))) + 2
    x0 = max(0, int(np.floor(cx)) - r)
    x1 = min(img.shape[1], int(np.ceil(cx)) + r + 1)
    y0 = max(0, int(np.floor(cy)) - r)
    y1 = min(img.shape[0], int(np.ceil(cy)) + r + 1)
    nx, ny = x1 - x0, y1 - y0
    sx = x0 - 0.5 + (np.arange(nx * ss) + 0.5) / ss - cx
    sy = y0 - 0.5 + (np.arange(ny * ss) + 0.5) / ss - cy
    gx, gy = np.meshgrid(sx, sy)
    th = np.radians(angle_deg)
    xr = gx * np.cos(th) + gy * np.sin(th)
    yr = -gx * np.sin(th) + gy * np.cos(th)
    inside = ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0).astype(np.float32)
    cov = inside.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    img[y0:y1, x0:x1] += contrast * cov


def add_disc(img, cx, cy, diameter, contrast, ss=8) -> None:
    add_ellipse(img, cx, cy, diameter, diameter, contrast, ss=ss)


def frame(img, seq=0, t_ns=1_000_000_000) -> Frame:
    u8 = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    return Frame(image=u8, seq=seq, t_exposure_ns=t_ns, t_capture_ns=t_ns + 5_000_000,
                 exposure_us=500.0, gain=1.0)


def cfg_default():
    return load_config()


def nearest(dets, u, v):
    return min(dets, key=lambda d: (d.u - u) ** 2 + (d.v - v) ** 2) if dets else None


# --------------------------------------------------------------------------------- mono detection
@pytest.mark.parametrize("cx,cy", [(320.0, 200.0), (320.3, 200.7), (101.25, 57.5)])
def test_small_dark_disc_found_with_subpixel_centroid(cx, cy):
    rng = np.random.default_rng(7)
    img = sky(rng)
    add_disc(img, cx, cy, 3.0, -30.0)
    cfg = cfg_default()
    dets = ClayDetector(cfg).detect(frame(img))
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.u - cx) <= 0.3 and abs(d.v - cy) <= 0.3
    assert d.contrast < -15.0
    assert d.snr > 5.0
    assert 0.0 <= d.score <= 1.0
    x, y, bw, bh = d.bbox
    assert x <= cx <= x + bw and y <= cy <= y + bh


def test_polarity_selects_bright_or_dark():
    rng = np.random.default_rng(11)
    img = sky(rng)
    add_disc(img, 400.0, 150.0, 3.0, +30.0)
    f = frame(img)
    cfg = cfg_default()

    cfg.detector.polarity = "bright"
    bright = ClayDetector(cfg).detect(f)
    assert len(bright) == 1 and bright[0].contrast > 15.0
    assert abs(bright[0].u - 400.0) <= 0.3 and abs(bright[0].v - 150.0) <= 0.3

    cfg.detector.polarity = "both"
    assert len(ClayDetector(cfg).detect(f)) == 1

    cfg.detector.polarity = "dark"
    assert ClayDetector(cfg).detect(f) == []


def test_pure_noise_sequence_stays_quiet():
    rng = np.random.default_rng(3)
    cfg = cfg_default()
    det = ClayDetector(cfg)
    false_positives = 0
    for i in range(50):
        img = sky(rng, clouds=8.0)
        false_positives += len(det.detect(frame(img, seq=i)))
    assert false_positives <= 1
    assert 1.5 < det.noise_sigma < 4.0


def test_mask_suppresses_detections_inside_it():
    rng = np.random.default_rng(5)
    img = sky(rng)
    add_disc(img, 200.0, 100.0, 3.0, -30.0)   # inside the masked band
    add_disc(img, 500.0, 300.0, 3.0, -30.0)   # outside it
    f = frame(img)
    cfg = cfg_default()

    unmasked = ClayDetector(cfg).detect(f)
    assert len(unmasked) == 2

    mask = np.full((H, W), 255, np.uint8)
    mask[0:160, :] = 0
    dets = ClayDetector(cfg, mask).detect(f)
    assert len(dets) == 1
    assert abs(dets[0].u - 500.0) <= 0.3 and abs(dets[0].v - 300.0) <= 0.3


def test_mask_is_resized_to_the_frame():
    rng = np.random.default_rng(5)
    img = sky(rng)
    add_disc(img, 200.0, 100.0, 3.0, -30.0)
    half = np.full((H // 2, W // 2), 255, np.uint8)
    half[0:80, :] = 0  # top 40 % of the image, at half resolution
    cfg = cfg_default()
    assert ClayDetector(cfg, half).detect(frame(img)) == []


def test_area_filters_reject_blobs_and_salt():
    rng = np.random.default_rng(13)
    img = sky(rng)
    img[90:120, 90:120] -= 80.0          # 30x30 = 900 px blob
    img[300, 400] -= 60.0                # single-pixel salt
    f = frame(img)
    cfg = cfg_default()
    cfg.detector.bg_kernel = 25          # keep a 900 px blob out of its own background estimate

    assert ClayDetector(cfg).detect(f) == []  # max_area_px=400 and min_area_px=2 reject both

    cfg.detector.max_area_px = 2000
    big = ClayDetector(cfg).detect(f)
    assert len(big) == 1 and 800.0 <= big[0].area_px <= 1000.0

    cfg.detector.max_area_px = 400
    cfg.detector.min_area_px = 1
    salt = nearest(ClayDetector(cfg).detect(f), 400.0, 300.0)
    assert salt is not None
    assert (round(salt.u), round(salt.v)) == (400, 300)
    assert salt.area_px == 1.0 and salt.contrast < -50.0


def measure_disc(D, sigma_g=0.0, contrast=-30.0, noise=2.0, frames_n=9, seed=17, h=240, w=320):
    """Observed diameters and centroid errors of a disc of true diameter D px at sub-pixel offsets.

    The disc is area-integrated (8x supersampling = the pixel box) and optionally blurred further, so
    the true observed size is sqrt(D^2 + psf_true^2) with psf_true = 4*sqrt(1/12 + sigma_g^2).
    """
    rng = np.random.default_rng(seed)
    diameters: list[float] = []
    centroid_err: list[float] = []
    for i in range(frames_n):
        cx, cy = w / 2 + (i % 4) * 0.25, h / 2 + (i % 3) * 0.33
        base = sky(rng, h=h, w=w, noise=0.0)
        disc = np.zeros_like(base)
        add_disc(disc, cx, cy, D, contrast)
        if sigma_g > 0.0:
            disc = cv2.GaussianBlur(disc, (0, 0), sigma_g)
        img = base + disc + rng.normal(0.0, noise, base.shape).astype(np.float32)
        dets = ClayDetector(cfg_default()).detect(frame(img))
        assert dets, f"no detection for D={D} sigma_g={sigma_g} frame {i}"
        diameters.append(dets[0].diameter_px)
        centroid_err.append(max(abs(dets[0].u - cx), abs(dets[0].v - cy)))
    return np.array(diameters), np.array(centroid_err)


def psf_true_px(sigma_g: float) -> float:
    return 4.0 * (1.0 / 12.0 + sigma_g * sigma_g) ** 0.5


@pytest.mark.parametrize("D,sigma_g", [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (6.0, 0.0), (10.0, 0.0),
                                       (3.0, 0.6), (6.0, 0.6)])
def test_observed_diameter_is_the_object_convolved_with_the_blur(D, sigma_g):
    """diameter_px must be unbiased for d_obs^2 = D^2 + psf^2 - the tracker undoes the quadrature."""
    diameters, centroid_err = measure_disc(D, sigma_g)
    recovered = (max(float(np.median(diameters)) ** 2 - psf_true_px(sigma_g) ** 2, 0.0)) ** 0.5
    tol = 0.20 if D <= 2.0 else 0.10
    assert abs(recovered - D) <= tol * D, f"recovered {recovered:.2f} px for a {D} px disc"
    assert centroid_err.max() <= 0.3


def test_diameter_grows_monotonically_with_the_disc():
    medians = [float(np.median(measure_disc(D)[0])) for D in (2.0, 4.0, 8.0)]
    assert medians[0] < medians[1] < medians[2]


def test_edge_on_clay_reports_its_major_axis():
    rng = np.random.default_rng(19)
    img = sky(rng)
    add_ellipse(img, 300.0, 220.0, 12.0, 3.0, -40.0)
    cfg = cfg_default()
    dets = ClayDetector(cfg).detect(frame(img))
    assert len(dets) == 1
    assert abs(dets[0].diameter_px - 12.0) <= 3.0
    assert dets[0].bbox[2] >= 4 * dets[0].bbox[3]


@pytest.mark.parametrize("D,sigma_g", [(3.0, 0.4), (3.0, 0.6), (4.0, 0.9)])
def test_psf_calibration_recovers_the_blur(D, sigma_g):
    """`tbot calib-detector --psf` core: a target of known size gives back the system blur."""
    diameters, _ = measure_disc(D, sigma_g)
    res = psf_from_diameters(diameters, D)
    assert abs(res["psf_px"] - psf_true_px(sigma_g)) <= 0.15
    assert res["n"] == diameters.size
    assert res["d_obs_px"] > res["d_true_px"]


def test_psf_calibration_rejects_an_impossible_target():
    with pytest.raises(ValueError, match="blur cannot be negative"):
        psf_from_diameters([3.0, 3.1, 2.9], 6.0)
    with pytest.raises(ValueError):
        psf_from_diameters([], 3.0)



def test_detections_are_capped_and_sorted_and_timestamped():
    rng = np.random.default_rng(23)
    img = sky(rng)
    for i in range(12):
        add_disc(img, 60.0 + 45.0 * i, 120.0 + 10.0 * i, 4.0, -20.0 - 3.0 * i)
    cfg = cfg_default()
    cfg.detector.max_detections = 5
    det = ClayDetector(cfg)
    dets = det.detect(frame(img, seq=4, t_ns=1_234_000_000))
    assert len(dets) == 5
    assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
    assert all(d.t_ns == 1_234_000_000 for d in dets)
    assert dets[0].snr > dets[-1].snr          # the strongest contrast wins
    assert set(det.timings_ms) >= {"bg", "threshold", "components", "total"}
    assert det.timings_ms["total"] > 0.0


def test_debug_images_only_when_enabled():
    rng = np.random.default_rng(29)
    img = sky(rng)
    add_disc(img, 320.0, 200.0, 4.0, -30.0)
    f = frame(img)
    cfg = cfg_default()
    det = ClayDetector(cfg)
    det.detect(f)
    assert det.debug == {}

    cfg.detector.debug = True
    det.detect(f)
    assert det.debug["contrast"].shape == (H, W)
    assert det.debug["binary"].shape == (H, W)
    assert set(np.unique(det.debug["binary"])) <= {0, 255}
    assert det.debug["contrast"].min() < -20

    cfg.detector.debug = False
    det.detect(f)
    assert det.debug == {}


def test_noise_sigma_tracks_the_image_noise():
    rng = np.random.default_rng(31)
    cfg = cfg_default()
    quiet = ClayDetector(cfg)
    quiet.detect(frame(sky(rng, noise=1.0)))
    loud = ClayDetector(cfg)
    loud.detect(frame(sky(rng, noise=6.0)))
    assert 0.7 < quiet.noise_sigma < 1.4
    assert 5.0 < loud.noise_sigma < 7.5
    # A 20 DN disc is above 6 sigma of the quiet noise but inside the loud noise.
    img = sky(rng, noise=1.0)
    add_disc(img, 320.0, 200.0, 4.0, -20.0)
    assert len(quiet.detect(frame(img))) == 1
    assert loud.detect(frame(img)) == []


def test_noise_sigma_refresh_interval():
    rng = np.random.default_rng(37)
    cfg = cfg_default()
    cfg.detector.noise_every = 5
    det = ClayDetector(cfg)
    det.detect(frame(sky(rng, noise=1.0)))
    first = det.noise_sigma
    loud = frame(sky(rng, noise=6.0))
    for _ in range(4):
        det.detect(loud)
        assert det.noise_sigma == first      # cached between refreshes
    det.detect(loud)
    assert det.noise_sigma > 4.0


def test_bad_config_values_are_rejected():
    rng = np.random.default_rng(41)
    f = frame(sky(rng))
    cfg = cfg_default()
    cfg.detector.polarity = "sideways"
    with pytest.raises(ValueError):
        ClayDetector(cfg).detect(f)
    cfg.detector.polarity = "both"
    cfg.detector.mode = "infrared"
    with pytest.raises(ValueError):
        ClayDetector(cfg).detect(f)


# --------------------------------------------------------------------------------- colour frames
def color_sky(rng, h=200, w=320, bgr=(200, 160, 120), noise=2.0) -> np.ndarray:
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.asarray(bgr, np.float32)
    img += rng.normal(0.0, noise, (h, w, 3)).astype(np.float32)
    return img


def add_color_disc(img, cx, cy, diameter, bgr) -> None:
    cov = np.zeros(img.shape[:2], np.float32)
    add_disc(cov, cx, cy, diameter, 1.0)
    delta = np.asarray(bgr, np.float32) - img
    img += cov[:, :, None] * delta


def test_orange_clay_found_on_a_colour_frame():
    rng = np.random.default_rng(43)
    img = color_sky(rng)
    add_color_disc(img, 160.0, 100.0, 6.0, (30, 160, 255))
    cfg = cfg_default()                       # mode auto, color_combine "or"
    dets = ClayDetector(cfg).detect(frame(img))
    assert len(dets) == 1
    assert abs(dets[0].u - 160.0) <= 0.6 and abs(dets[0].v - 100.0) <= 0.6


def test_color_and_combine_rejects_a_non_orange_blob():
    rng = np.random.default_rng(47)
    img = color_sky(rng)
    add_color_disc(img, 100.0, 60.0, 6.0, (30, 160, 255))    # orange clay
    add_color_disc(img, 220.0, 140.0, 8.0, (120, 96, 72))    # darker blue patch, strong luma contrast
    f = frame(img)
    cfg = cfg_default()

    both = ClayDetector(cfg).detect(f)                        # "or": hue or contrast
    assert len(both) == 2

    cfg.detector.color_combine = "and"
    only_orange = ClayDetector(cfg).detect(f)
    assert len(only_orange) == 1
    assert abs(only_orange[0].u - 100.0) <= 0.6 and abs(only_orange[0].v - 60.0) <= 0.6


def test_mono_mode_ignores_hue_on_a_colour_frame():
    rng = np.random.default_rng(53)
    img = color_sky(rng)
    # Same luma as the sky (152.6 DN), pure orange hue: visible only to the HSV path.
    add_color_disc(img, 160.0, 100.0, 6.0, (0, 143, 231))
    f = frame(img)
    cfg = cfg_default()
    assert len(ClayDetector(cfg).detect(f)) == 1
    cfg.detector.mode = "mono"
    assert ClayDetector(cfg).detect(f) == []


# --------------------------------------------------------------------------------- auto mask
def swinging_stack(n=24, h=120, w=160, seed=61) -> np.ndarray:
    """Frames of a swinging gun: scrolling textured sky, a static dark band at the top,
    and a small static patch in the middle of the sky (too small / not touching the border)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    stack = np.empty((n, h, w), np.uint8)
    for i in range(n):
        texture = 150.0 + 25.0 * np.cos((xx + 4.0 * i) / 7.0) * np.cos((yy + 2.0 * i) / 9.0)
        texture += rng.normal(0.0, 2.0, (h, w)).astype(np.float32)
        texture[0:20, :] = 30.0 + rng.normal(0.0, 1.0, (20, w))   # barrel: static, touches the border
        texture[60:65, 80:85] = 40.0 + rng.normal(0.0, 1.0, (5, 5))  # static speck in the sky
        stack[i] = np.clip(texture, 0, 255).astype(np.uint8)
    return stack


def test_static_mask_blocks_the_barrel_band_only():
    mask = static_mask(swinging_stack(), static_std=3.0, min_area_px=200, dilate_px=4)
    assert mask.shape == (120, 160)
    assert set(np.unique(mask)) <= {0, 255}
    assert np.all(mask[0:20, :] == 0)          # the static band is masked
    assert np.all(mask[30:, :] == 255)         # moving sky stays usable
    assert mask[62, 82] == 255                 # small interior static speck is kept
    assert 0.12 < np.count_nonzero(mask == 0) / mask.size < 0.25


def test_static_mask_dilation_grows_the_blocked_region():
    stack = swinging_stack()
    tight = static_mask(stack, static_std=3.0, min_area_px=200, dilate_px=0)
    grown = static_mask(stack, static_std=3.0, min_area_px=200, dilate_px=6)
    assert np.count_nonzero(grown == 0) > np.count_nonzero(tight == 0)
    assert np.all(tight[0:20, :] == 0) and np.all(tight[21:, :] == 255)
    assert np.all(grown[0:26, :] == 0)


def test_static_mask_needs_two_frames():
    with pytest.raises(ValueError):
        static_mask(swinging_stack(n=1))


def test_masked_detector_consumes_the_auto_mask():
    """A clay that lands on the barrel band must not be reported once the mask is applied."""
    mask = static_mask(swinging_stack(), static_std=3.0, min_area_px=200, dilate_px=4)
    rng = np.random.default_rng(67)
    img = sky(rng, h=120, w=160)
    add_disc(img, 80.0, 10.0, 3.0, -30.0)
    cfg = cfg_default()
    assert len(ClayDetector(cfg).detect(frame(img))) == 1
    assert ClayDetector(cfg, mask).detect(frame(img)) == []
