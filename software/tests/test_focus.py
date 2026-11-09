import cv2
import numpy as np
import pytest

from tbot.tools.focus import measure


def test_blur_reduces_sharpness_in_centre_and_quadrants():
    y, x = np.indices((100, 100))
    image = (((x // 5 + y // 5) % 2) * 200 + 20).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    mask = np.full(image.shape, 255, dtype=np.uint8)
    sharp = measure(image, mask, 0.5)
    soft = measure(blurred, mask, 0.5)
    assert sharp['sharpness'] > soft['sharpness']
    for key in sharp['quadrants']:
        assert sharp['quadrants'][key] > soft['quadrants'][key]


def test_exposure_and_mask_exclude_saturated_and_boundary_pixels():
    image = np.array([[255, 255, 10, 10], [255, 0, 10, 10],
                      [10, 10, 10, 10], [10, 10, 10, 10]], dtype=np.uint8)
    mask = np.full(image.shape, 255, dtype=np.uint8)
    mask[0, 0] = 0
    result = measure(image, mask, 1.0)
    assert result['saturated_pct'] == pytest.approx(200 / 15)
    assert result['mean_dn'] == pytest.approx((2 * 255 + 12 * 10) / 15)
    assert result['median_dn'] == 10
    mask[:, :2] = 0
    result = measure(image, mask, 1.0)
    assert result['saturated_pct'] == 0
    assert result['sharpness'] == 0
    assert result['quadrants']['top_left'] is None
