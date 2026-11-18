# usage: python twist_fit.py twist.mjpeg [last_frame]  -> twist_dots.csv, twist_fit.png, twist_fit.txt
# Laser dot per frame (DoG peak, centroid) in a window around the target, then a robust circle fit:
# the bore laser turned in the bore traces a circle whose centre is the bore pixel.
import sys, cv2, numpy as np

def frames(path):
    data = open(path, 'rb').read(); i = 0
    while True:
        s = data.find(b'\xff\xd8', i)
        if s < 0: return
        e = data.find(b'\xff\xd9', s)
        if e < 0: return
        yield cv2.imdecode(np.frombuffer(data[s:e + 2], np.uint8), cv2.IMREAD_GRAYSCALE); i = e + 2

path = sys.argv[1]; last = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
X0, Y0, R = (int(v) for v in (sys.argv[3:6] if len(sys.argv) > 5 else (480, 500, 220)))
rows = []; first = None
for k, im in enumerate(frames(path)):
    if im is None or k > last: continue
    if first is None: first = im
    roi = im[Y0 - R:Y0 + R, X0 - R:X0 + R].astype(np.float32)
    dog = cv2.GaussianBlur(roi, (0, 0), 1.5) - cv2.GaussianBlur(roi, (0, 0), 6)
    noise = np.median(np.abs(dog)) * 1.48 + 1e-3
    y, x = np.unravel_index(np.argmax(dog), dog.shape); snr = dog[y, x] / noise
    d2 = dog.copy(); d2[max(0, y - 12):y + 12, max(0, x - 12):x + 12] = -1e9; snr2 = d2.max() / noise
    ya, yb, xa, xb = max(0, y - 7), min(dog.shape[0], y + 8), max(0, x - 7), min(dog.shape[1], x + 8)
    w = dog[ya:yb, xa:xb].clip(0); yy, xx = np.mgrid[ya:yb, xa:xb]
    rows.append((k, (w * xx).sum() / w.sum() + X0 - R, (w * yy).sum() / w.sum() + Y0 - R, snr, snr2))
a = np.array(rows)
np.savetxt('twist_dots.csv', a, delimiter=',', header='frame,x,y,snr,snr2', fmt=['%d', '%.2f', '%.2f', '%.1f', '%.1f'])
# clean dot: strong and unambiguous (no second peak of similar strength); thresholds relative to the run itself
good = (a[:, 3] > 0.25 * np.percentile(a[:, 3], 90)) & (a[:, 4] < 0.5 * a[:, 3])
P = a[good][:, 1:3]

def fit(P):
    A = np.c_[2 * P, np.ones(len(P))]; b = (P ** 2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = c[:2]; r = np.sqrt(c[2] + cx * cx + cy * cy)
    return cx, cy, r

keep = np.ones(len(P), bool)
for _ in range(5):  # drop points far off the ring (hand-pushed laser, reflections)
    cx, cy, r = fit(P[keep]); res = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
    s = 1.4826 * np.median(np.abs(res[keep])); keep = np.abs(res) < max(3 * s, 2.0)
cx, cy, r = fit(P[keep]); res = np.hypot(P[keep, 0] - cx, P[keep, 1] - cy) - r
ang = np.degrees(np.arctan2(P[keep, 1] - cy, P[keep, 0] - cx)) % 360
sectors = np.unique((ang // 30).astype(int))
rng = np.random.default_rng(0); bs = []
Q = P[keep]
for _ in range(300):
    bs.append(fit(Q[rng.integers(0, len(Q), len(Q))])[:2])
sd = np.std(bs, axis=0)
txt = (f"bore pixel ({cx:.2f}, {cy:.2f})  r {r:.1f} px  scatter {np.std(res):.1f} px  bootstrap sd {sd[0]:.1f} / {sd[1]:.1f} px\n"
       f"frames: {len(a)} read, {good.sum()} clean dot, {keep.sum()} on the ring; 30-deg sectors covered {len(sectors)}/12")
print(txt); open('twist_fit.txt', 'w').write(txt + '\n')
vis = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
for x, y in P[~keep]: cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
for x, y in P[keep]: cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (255, 200, 0), 1)
cv2.drawMarker(vis, (int(round(cx)), int(round(cy))), (255, 200, 0), cv2.MARKER_CROSS, 14, 1)
cv2.drawMarker(vis, (522, 497), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 14, 1)  # 2026-11-16 calibration prediction at 5.975 m
cv2.imwrite('twist_fit.png', vis[Y0 - R:Y0 + R, X0 - R:X0 + R])
