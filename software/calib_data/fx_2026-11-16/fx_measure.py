import cv2, numpy as np, json, sys
sys.path.insert(0, '.')
from fx_edges import edge
FT = 0.3048
# per group: distance, shots, right upright box (xl, xr, top, bottom|None), oval roi (x0,x1,y0,y1)
G = {
 "28": (28, [2,3,4],    (551.7, 611.7, 493, None),  (556, 607, 508, 592)),
 "20": (20, [5,6,7],    (580, 659.3, 511.7, 658),   (590, 650, 535, 637)),
 "15": (15, [8,9],      (589, 696.5, 536.5, 734),   (603, 682, 568, 703)),
 "10": (10, [11,12],    (630, 795, 578, None),      None),
 "40": (40, [13,14,15], (540, 575, 481, 547),       (540, 572, 490, 544)),
 "35": (35, [16,17],    (540, 586, 486, 562),       (546, 580, 497, 558)),
 "47": (47, [18,19,20], (537, 570, 475, 535),       (540, 563, 481, 530)),
}
def soft_oval(im, roi):
    x0, x1, y0, y1 = roi
    a = cv2.GaussianBlur(im, (0, 0), 0.7)[y0:y1, x0:x1].astype(np.float64)
    lo = np.percentile(a, 15); hi = np.percentile(a, 97)
    # inside bright, outside dark box face: soft membership; fit on the main connected blob
    w = np.clip((a - lo) / (hi - lo), 0, 1)
    m = (w > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8))
    w = w * keep
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    A = w.sum(); cx = (w * xx).sum() / A; cy = (w * yy).sum() / A
    vx = (w * (xx - cx) ** 2).sum() / A; vy = (w * (yy - cy) ** 2).sum() / A
    # uniform ellipse: var = (semi-axis)^2 / 4  -> full axis = 4 sqrt(var)
    return 4 * np.sqrt(vx), 4 * np.sqrt(vy), A, x0 + cx, y0 + cy
rows = []
for name, (ft, shots, box, roi) in G.items():
    for i in shots:
        im = cv2.imread(f"snaps/snap_{i:04d}.png", 0)
        xl, xr, top, bot = box
        hgt = (bot - top) if bot else 60
        band = (int(top + 0.35 * hgt), int(top + 0.75 * hgt)) if bot else (int(top + 15), int(top + 60))
        win = max(3, 0.08 * (xr - xl))
        L = edge(im, band, xl, win); R = edge(im, band, xr, win)
        r = {"group": name, "ft": ft, "snap": i, "box_w": R[0] - L[0], "edge_snr": round(min(L[1] / L[2], R[1] / R[2]), 1)}
        if bot:
            cb = (int(xl + 0.2 * (xr - xl)), int(xr - 0.2 * (xr - xl)))
            T = edge(im, cb, top, win, vertical=False); B = edge(im, cb, bot, win, vertical=False)
            r["box_h"] = B[0] - T[0]
        if roi:
            ow, oh, A, cx, cy = soft_oval(im, roi)
            r.update(oval_w=ow, oval_h=oh, oval_area=A)
        rows.append(r)
for r in rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rows, open("fx_rows.json", "w"), indent=1)
