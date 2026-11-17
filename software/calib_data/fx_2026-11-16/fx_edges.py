"""Sub-pixel edge refinement for the box-size focal length measurement (scratch, runs/)."""
import cv2, numpy as np, sys
def profile(im, axis_rows, x0, x1, smooth=1.0):
    """mean intensity over rows axis_rows (y0,y1) for columns x0..x1"""
    y0, y1 = axis_rows
    p = im[y0:y1, x0:x1].astype(np.float64).mean(axis=0)
    if smooth > 0:
        p = cv2.GaussianBlur(p.reshape(1, -1), (0, 0), smooth).ravel()
    return p
def edge(im, band, guess, win=6, vertical=True, smooth=1.0):
    """sub-pixel position of the strongest step near `guess`.
    vertical edge: band=(y0,y1) rows, search x. horizontal edge: band=(x0,x1) cols, search y."""
    a = im if vertical else im.T
    lo, hi = int(round(guess - win - 3)), int(round(guess + win + 4))
    p = profile(a, band, lo, hi, smooth)
    g = np.abs(np.gradient(p))
    k0 = 3; k1 = len(g) - 3
    k = k0 + int(np.argmax(g[k0:k1]))
    if 0 < k < len(g) - 1:
        d = g[k - 1] - 2 * g[k] + g[k + 1]
        off = 0.5 * (g[k - 1] - g[k + 1]) / d if d != 0 else 0.0
    else:
        off = 0.0
    return lo + k + off, g[k], float(np.median(g))
