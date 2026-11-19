/* Per-candidate measurement of the clay detector: the native form of the `for i in idx:` loop of
 * ClayDetector._measure and of _adaptive_moments (tbot/vision/detector.py), which stay as the reference
 * and the fallback. Called through ctypes (tbot/vision/_native.py); no Python headers.
 *
 * Why: on a Raspberry Pi 3B each NumPy call on a small window costs 15-50 us whatever it computes, and one
 * candidate takes a few hundred of them (~6 ms). The arithmetic itself is tens of microseconds.
 *
 * The code follows the Python statement by statement, including its number types, so that every accept /
 * reject decision is the same: the residual is an integer number of DN, comparisons against thresholds are
 * made in float32 where NumPy makes them in float32, medians are the mean of the two middle elements, and
 * every argmax takes the first maximum in row-major order. The moments differ from the Python only by
 * rounding (expf against NumPy's float32 exp, and the summation order of cv2.moments): ~1e-6 px.
 *
 * The same library also holds the scenery filter's per-frame pass (tbot_scenery, at the end of this file;
 * StaticFilter.filter in tbot/vision/clutter.py is its reference and fallback).
 *
 * Build: scripts/build_native.sh (cc -O2 -shared -fPIC -ffp-contract=off). Change TBOT_MEASURE_ABI (or
 * TBOT_SCENERY_ABI) here and in _native.py together whenever a signature or struct changes: a stale library
 * is then refused.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define TBOT_API __declspec(dllexport)
#else
#define TBOT_API __attribute__((visibility("default")))
#endif

#define TBOT_MEASURE_ABI 1

/* Largest half-window the kernel has buffers for (detector.ADAPTIVE_MAX_HALF). */
#define MAX_HALF 40
#define MAX_SIDE (2 * MAX_HALF + 1)
#define MAX_PIX (MAX_SIDE * MAX_SIDE)
#define OUT_COLS 12

/* Reject codes in out[k][0] (0 = measured). */
enum {
    R_OK = 0,
    R_GROW = 1,        /* small seed that does not grow to min_area_px */
    R_EMPTY = 2,       /* no pixel of the component inside its window */
    R_FLOOR = 3,       /* raw peak under the DN floor */
    R_SPECK = 4,       /* peak holds (nearly) all of its 3x3 neighbourhood */
    R_LOCAL = 5,       /* does not stand out from the ring around it */
    R_SHAPE = 6        /* raw footprint too large or too elongated */
};

typedef struct {
    /* image: h x w, C-contiguous. Either gray/bg (raw residual = gray - bg) or contrast16. */
    int32_t h, w;
    const uint8_t *gray;       /* NULL when contrast16 is given */
    const uint8_t *bg;
    const uint8_t *gsm;        /* matched-filter image (statistic = gsm - bg); NULL = matched filter off */
    const int16_t *contrast16; /* plain mode: the residual itself */
    const int32_t *labels;     /* h x w */
    const int32_t *stats;      /* n x 5: left, top, width, height, area (cv2.connectedComponentsWithStats) */
    const uint8_t *mask;       /* h x w, non-zero = usable; NULL = none */
    /* frame values */
    double sigma;              /* raw noise sigma, DN */
    double matched_sigma;      /* sigma of the matched statistic (matched mode) */
    double grow_low;           /* hysteresis level (has_grow) */
    int32_t has_grow;
    int32_t grown_done;        /* the vectorised pre-screen already did the growth test */
    /* detector config */
    double k_sigma;            /* max(detector.k_sigma, 1e-6) */
    double min_contrast;
    double max_peak_fraction;
    double max_aspect;
    int32_t min_area_px;
    int32_t max_area_px;
    int32_t moat_half;
    /* detector module constants (one source of truth: detector.py) */
    int32_t adaptive_min_half;
    int32_t grow_radius;
    int32_t local_ring_px;
    int32_t adaptive_iters;
    double window_sigmas;
    double alpha_half;         /* sqrt(ADAPTIVE_ALPHA) */
    double local_k_sigma;
    double meas_seed_sigmas;
    double moment_floor_sigmas;
    double pixel_box_var;
    double adaptive_alpha;
    double min_weight_var;
    double adaptive_tol;
    double adaptive_damping;
    double diameter_divisor;   /* PROFILE_FACTOR (x MORPH_SIZE_BIAS with the morphological background) */
    double moment_m00_min;     /* cv2.moments gives central moments only above this |m00| (see moments_of) */
} tbot_measure_params;

/* Python's round(): half to even. */
static int round_half_even(double x) {
    double f = floor(x);
    double d = x - f;
    int i = (int)f;
    if (d > 0.5) return i + 1;
    if (d < 0.5) return i;
    return (i & 1) ? i + 1 : i;
}

static void sort_floats(float *a, int n) {
    for (int i = 1; i < n; i++) {
        float v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

/* detector._median: the middle element, or the mean of the two middle ones (float32 arithmetic). */
static float median_floats(float *a, int n) {
    sort_floats(a, n);
    int lo = (n - 1) / 2, hi = n / 2;
    return lo == hi ? a[lo] : (a[lo] + a[hi]) / 2.0f;
}

static inline int residual(const tbot_measure_params *p, int y, int x) {
    size_t k = (size_t)y * (size_t)p->w + (size_t)x;
    return p->contrast16 ? (int)p->contrast16[k] : (int)p->gray[k] - (int)p->bg[k];
}

static inline int statistic(const tbot_measure_params *p, int y, int x) {
    size_t k = (size_t)y * (size_t)p->w + (size_t)x;
    return (int)p->gsm[k] - (int)p->bg[k];
}

static int window_half(const tbot_measure_params *p, int bw, int bh) {
    int m = bw > bh ? bw : bh;
    int half = (int)(p->window_sigmas * p->alpha_half * (double)m / 4.0 + 0.5);
    if (half < p->adaptive_min_half) half = p->adaptive_min_half;
    if (half > p->moat_half) half = p->moat_half;
    return half;
}

typedef struct {
    int px0, py0, px1, py1, wn, hn;
} window;

static window window_for(const tbot_measure_params *p, int x0, int y0, int bw, int bh) {
    window q;
    int half = window_half(p, bw, bh);
    int rcx = round_half_even(x0 + 0.5 * (bw - 1));
    int rcy = round_half_even(y0 + 0.5 * (bh - 1));
    q.px0 = rcx - half > 0 ? rcx - half : 0;
    q.py0 = rcy - half > 0 ? rcy - half : 0;
    q.px1 = rcx + half + 1 < p->w ? rcx + half + 1 : p->w;
    q.py1 = rcy + half + 1 < p->h ? rcy + half + 1 : p->h;
    q.wn = q.px1 - q.px0;
    q.hn = q.py1 - q.py0;
    return q;
}

/* ClayDetector._grown_enough */
static int grown_enough(const tbot_measure_params *p, int lid) {
    const int32_t *st = p->stats + 5 * (size_t)lid;
    int r = p->grow_radius;
    int x0 = st[0], y0 = st[1], x1 = st[0] + st[2], y1 = st[1] + st[3];
    int wx0 = x0 - r > 0 ? x0 - r : 0, wy0 = y0 - r > 0 ? y0 - r : 0;
    int wx1 = x1 + r < p->w ? x1 + r : p->w, wy1 = y1 + r < p->h ? y1 + r : p->h;
    long long sum = 0;
    for (int y = wy0; y < wy1; y++)
        for (int x = wx0; x < wx1; x++)
            if (p->labels[(size_t)y * p->w + x] == lid) sum += statistic(p, y, x);
    int sign = sum >= 0 ? 1 : -1;
    int count = 0;
    for (int y = wy0; y < wy1; y++) {
        for (int x = wx0; x < wx1; x++) {
            size_t k = (size_t)y * p->w + x;
            if (p->labels[k] == lid) {
                count++;
                continue;
            }
            if (!((double)(statistic(p, y, x) * sign) > p->grow_low)) continue;
            if (p->mask && !p->mask[k]) continue;
            int near = 0;
            for (int yy = y - r; yy <= y + r && !near; yy++) {
                if (yy < wy0 || yy >= wy1) continue;
                for (int xx = x - r; xx <= x + r; xx++) {
                    if (xx < wx0 || xx >= wx1) continue;
                    if (p->labels[(size_t)yy * p->w + xx] == lid) {
                        near = 1;
                        break;
                    }
                }
            }
            count += near;
        }
    }
    return count >= p->min_area_px;
}

/* The sums cv2.moments makes of a float32 image (double accumulation), as m00, m10, m01 and the
 * central second moments mu20, mu11, mu02. */
typedef struct {
    double m00, m10, m01, mu20, mu11, mu02;
} moments;

static moments moments_of(const float *img, int hn, int wn, double m00_min) {
    double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m11 = 0, m02 = 0;
    for (int y = 0; y < hn; y++) {
        const float *row = img + (size_t)y * wn;
        double x0 = 0, x1 = 0, x2 = 0;
        for (int x = 0; x < wn; x++) {
            double v = (double)row[x];
            double xp = x * v;
            x0 += v;
            x1 += xp;
            x2 += xp * x;
        }
        double py = y * x0;
        m00 += x0;
        m10 += x1;
        m01 += py;
        m20 += x2;
        m11 += x1 * y;
        m02 += py * y;
    }
    /* cv2's completeMomentState: with no flux to speak of (the Gaussian weight has wandered off the blob) the
     * "central" moments are the raw ones. Where that sets in depends on the OpenCV build (DBL_EPSILON, or half
     * of it in 5.0, which sums such an image twice), so the detector measures it and passes it in. What comes
     * out of such a window is junk either way; this only keeps it the same junk as the Python loop's. */
    double cx = 0.0, cy = 0.0;
    if (fabs(m00) > m00_min) {
        double inv = 1.0 / m00;
        cx = m10 * inv;
        cy = m01 * inv;
    }
    moments m = {m00, m10, m01, m20 - m10 * cx, m11 - m10 * cy, m02 - m01 * cy};
    return m;
}

/* detector._adaptive_moments */
static void adaptive_moments(const tbot_measure_params *p, const float *img, int hn, int wn, float *flux,
                             double *cx_, double *cy_, double *oxx_, double *oxy_, double *oyy_) {
    double cx = *cx_, cy = *cy_, oxx = *oxx_, oxy = *oxy_, oyy = *oyy_;
    float dxs[MAX_SIDE];
    for (int it = 0; it < p->adaptive_iters; it++) {
        double kxx = p->adaptive_alpha * oxx, kyy = p->adaptive_alpha * oyy, kxy = p->adaptive_alpha * oxy;
        if (kxx < p->min_weight_var) kxx = p->min_weight_var;
        if (kyy < p->min_weight_var) kyy = p->min_weight_var;
        double det_k = kxx * kyy - kxy * kxy;
        if (det_k <= 0.0) break;
        double ixx = kyy / det_k, ixy = -kxy / det_k, iyy = kxx / det_k;
        float fcx = (float)cx, fcy = (float)cy;
        float a_xy = (float)(-ixy), a_xx = (float)(-0.5 * ixx), a_yy = (float)(-0.5 * iyy);
        for (int x = 0; x < wn; x++) dxs[x] = (float)x - fcx;
        for (int y = 0; y < hn; y++) {
            float dy = (float)y - fcy;
            float ay = a_xy * dy;
            float eyy = (a_yy * dy) * dy;
            const float *src = img + (size_t)y * wn;
            float *dst = flux + (size_t)y * wn;
            for (int x = 0; x < wn; x++) {
                float dx = dxs[x];
                float e = ay * dx;
                e += (a_xx * dx) * dx;
                e += eyy;
                dst[x] = expf(e) * src[x];
            }
        }
        moments mo = moments_of(flux, hn, wn, p->moment_m00_min);
        double s = mo.m00;
        if (!(s > 0.0)) break;
        double mx = mo.m10 / s - cx, my = mo.m01 / s - cy;
        double mxx = mo.mu20 / s, myy = mo.mu02 / s, mxy = mo.mu11 / s;
        cx += mx;
        cy += my;
        double det_m = mxx * myy - mxy * mxy;
        if (det_m <= 0.0) break;
        double pxx = myy / det_m - ixx, pyy = mxx / det_m - iyy, pxy = -mxy / det_m - ixy;
        double det_p = pxx * pyy - pxy * pxy;
        if (det_p <= 0.0 || pxx <= 0.0 || pyy <= 0.0) {
            oxx *= 2.0;
            oyy *= 2.0;
            oxy *= 2.0;
            continue;
        }
        double nxx = pyy / det_p, nxy = -pxy / det_p, nyy = pxx / det_p;
        double scale = nxx > nyy ? nxx : nyy;
        if (scale < 1e-3) scale = 1e-3;
        int done = fabs(nxx - oxx) < p->adaptive_tol * scale && fabs(nyy - oyy) < p->adaptive_tol * scale &&
                   fabs(nxy - oxy) < p->adaptive_tol * scale;
        oxx += p->adaptive_damping * (nxx - oxx);
        oyy += p->adaptive_damping * (nyy - oyy);
        oxy += p->adaptive_damping * (nxy - oxy);
        if (done) break;
    }
    *cx_ = cx;
    *cy_ = cy;
    *oxx_ = oxx;
    *oxy_ = oxy;
    *oyy_ = oyy;
}

/* One candidate; returns the reject code and fills o[1..11]: u, v, area, diameter, contrast, snr, score,
 * bbox x, y, w, h. */
static int measure_one(const tbot_measure_params *p, int lid, double *o, int *cbuf, float *fa, float *fb,
                       float *fc) {
    const int32_t *st = p->stats + 5 * (size_t)lid;
    const int matched = p->gsm != NULL;
    if (!p->grown_done && p->has_grow && st[4] < p->min_area_px && !grown_enough(p, lid)) return R_GROW;

    int x0 = st[0], y0 = st[1], bw = st[2], bh = st[3];
    double area = (double)st[4];
    window q = window_for(p, x0, y0, bw, bh);
    int wn = q.wn, hn = q.hn;

    /* residual of the window, and the raw extreme over the component's own pixels (first in row-major order) */
    int have = 0, peak = 0, peak_abs = -1;
    for (int y = 0; y < hn; y++) {
        for (int x = 0; x < wn; x++) {
            int c = residual(p, q.py0 + y, q.px0 + x);
            cbuf[y * wn + x] = c;
            if (p->labels[(size_t)(q.py0 + y) * p->w + q.px0 + x] != lid) continue;
            have = 1;
            int a = c < 0 ? -c : c;
            if (a > peak_abs) {
                peak_abs = a;
                peak = c;
            }
        }
    }
    if (!have) return R_EMPTY;

    int m_peak = 0;
    float seed_thr = 0.0f;
    int gy = 0, gx = 0;
    if (matched) {
        if ((double)peak_abs < p->min_contrast) return R_FLOOR;
        int m_abs = -1;
        for (int y = 0; y < hn; y++)
            for (int x = 0; x < wn; x++) {
                if (p->labels[(size_t)(q.py0 + y) * p->w + q.px0 + x] != lid) continue;
                int m = statistic(p, q.py0 + y, q.px0 + x);
                int a = m < 0 ? -m : m;
                if (a > m_abs) {
                    m_abs = a;
                    m_peak = m;
                }
            }
        int peak_sign = m_peak >= 0 ? 1 : -1;
        if (peak * peak_sign < 0) { /* the raw extreme fights the matched polarity: use the matched sign */
            int best = 0, got = 0;
            for (int y = 0; y < hn; y++)
                for (int x = 0; x < wn; x++) {
                    if (p->labels[(size_t)(q.py0 + y) * p->w + q.px0 + x] != lid) continue;
                    int v = cbuf[y * wn + x] * peak_sign;
                    if (!got || v > best) {
                        got = 1;
                        best = v;
                        peak = cbuf[y * wn + x];
                    }
                }
        }
        int sp = peak >= 0 ? 1 : -1;
        /* the component's strongest pixel of that polarity */
        int ry = 0, rx = 0, pk = 0, got = 0;
        for (int y = 0; y < hn; y++)
            for (int x = 0; x < wn; x++) {
                if (p->labels[(size_t)(q.py0 + y) * p->w + q.px0 + x] != lid) continue;
                int v = sp * cbuf[y * wn + x];
                if (!got || v > pk) {
                    got = 1;
                    pk = v;
                    ry = y;
                    rx = x;
                }
            }
        /* speck filter */
        int nb = 0;
        for (int y = ry - 1; y <= ry + 1; y++) {
            if (y < 0 || y >= hn) continue;
            for (int x = rx - 1; x <= rx + 1; x++) {
                if (x < 0 || x >= wn) continue;
                int v = sp * cbuf[y * wn + x];
                if (v > 0) nb += v;
            }
        }
        if ((double)pk > p->max_peak_fraction * (double)nb) return R_SPECK;
        /* local-contrast ring test, where the ring fits in the window */
        int rr = p->local_ring_px + (st[2] > st[3] ? st[2] : st[3]) / 2;
        int ya = ry - rr, yb = ry + rr, xa = rx - rr, xb = rx + rr;
        if (ya >= 0 && xa >= 0 && yb < hn && xb < wn) {
            int n = 0;
            for (int x = xa; x <= xb; x++) fa[n++] = (float)(sp * cbuf[ya * wn + x]);
            for (int x = xa; x <= xb; x++) fa[n++] = (float)(sp * cbuf[yb * wn + x]);
            for (int y = ya + 1; y < yb; y++) fa[n++] = (float)(sp * cbuf[y * wn + xa]);
            for (int y = ya + 1; y < yb; y++) fa[n++] = (float)(sp * cbuf[y * wn + xb]);
            if ((double)pk - (double)median_floats(fa, n) < p->local_k_sigma * p->sigma) return R_LOCAL;
        }
        /* raw footprint: the component's pixels above the raw threshold, else above MEAS_SEED_SIGMAS */
        double raw_thr_d = p->k_sigma * p->sigma;
        if (raw_thr_d < p->min_contrast) raw_thr_d = p->min_contrast;
        seed_thr = (float)raw_thr_d;
        for (int pass = 0; pass < 2; pass++) {
            int any = 0;
            for (int y = 0; y < hn && !any; y++)
                for (int x = 0; x < wn; x++)
                    if (p->labels[(size_t)(q.py0 + y) * p->w + q.px0 + x] == lid &&
                        (float)(sp * cbuf[y * wn + x]) > seed_thr) {
                        any = 1;
                        break;
                    }
            if (any || pass == 1) break;
            seed_thr = (float)(p->meas_seed_sigmas * p->sigma);
        }
        int fx0 = rx, fx1 = rx, fy0 = ry, fy1 = ry, count = 0;
        for (int y = 0; y < hn; y++)
            for (int x = 0; x < wn; x++) {
                int in = (y == ry && x == rx) ||
                         (p->labels[(size_t)(q.py0 + y) * p->w + q.px0 + x] == lid &&
                          (float)(sp * cbuf[y * wn + x]) > seed_thr);
                if (!in) continue;
                count++;
                if (x < fx0) fx0 = x;
                if (x > fx1) fx1 = x;
                if (y < fy0) fy0 = y;
                if (y > fy1) fy1 = y;
            }
        x0 = q.px0 + fx0;
        y0 = q.py0 + fy0;
        bw = fx1 - fx0 + 1;
        bh = fy1 - fy0 + 1;
        area = (double)count;
        {
            int lg = bw > bh ? bw : bh, sm = bw < bh ? bw : bh;
            if (sm < 1) sm = 1;
            if (count > p->max_area_px || (double)lg / (double)sm > p->max_aspect) return R_SHAPE;
        }
        gy = q.py0 + ry;
        gx = q.px0 + rx;
        /* the window again, around the raw footprint */
        q = window_for(p, x0, y0, bw, bh);
        wn = q.wn;
        hn = q.hn;
        for (int y = 0; y < hn; y++)
            for (int x = 0; x < wn; x++) cbuf[y * wn + x] = residual(p, q.py0 + y, q.px0 + x);
    }

    const int sign = peak >= 0 ? 1 : -1;
    const float fsign = (float)sign;
    const float floor_f = (float)(p->moment_floor_sigmas * p->sigma);
    float *img_w = fa, *wimg = fb, *flux = fc;
    /* fa: polarity-flipped flux; fb: soft-thresholded seed weights (0 outside `own`); fc: the two pixel
     * classes (bit 0 = own, bit 1 = kept) until the moments need it as scratch */
    for (int y = 0; y < hn; y++) {
        for (int x = 0; x < wn; x++) {
            size_t k = (size_t)(q.py0 + y) * p->w + q.px0 + x;
            int i = y * wn + x;
            int lab = p->labels[k];
            float v = fsign * (float)cbuf[i];
            int own;
            if (matched)
                own = (lab == lid && v > seed_thr) || (q.py0 + y == gy && q.px0 + x == gx);
            else
                own = lab == lid;
            int keep = matched ? (lab == lid || lab == 0) : (own || lab == 0);
            if (p->mask && !p->mask[k]) keep = 0;
            img_w[i] = v;
            flux[i] = (float)((own ? 1 : 0) | (keep ? 2 : 0));
        }
    }
    /* median of the window's border ring: the moat a blob digs in its own background */
    float ring[4 * MAX_SIDE];
    int n = 0;
    for (int x = 0; x < wn; x++) ring[n++] = img_w[x];
    for (int x = 0; x < wn; x++) ring[n++] = img_w[(hn - 1) * wn + x];
    for (int y = 1; y < hn - 1; y++) ring[n++] = img_w[y * wn];
    for (int y = 1; y < hn - 1; y++) ring[n++] = img_w[y * wn + wn - 1];
    float med = median_floats(ring, n);
    for (int i = 0; i < hn * wn; i++) {
        int cls = (int)flux[i];
        float v = img_w[i];
        float s = v - floor_f;
        wimg[i] = (cls & 1) ? (s > 0.0f ? s : 0.0f) : 0.0f;
        img_w[i] = (cls & 2) ? v - med : 0.0f;
        flux[i] = (cls & 1) ? 1.0f : 0.0f;
    }
    moments mo = moments_of(wimg, hn, wn, p->moment_m00_min);
    if (!(mo.m00 > 0.0)) mo = moments_of(flux, hn, wn, p->moment_m00_min); /* hue-only blob: weight its pixels equally */
    double wsum = mo.m00;
    double cx = mo.m10 / wsum, cy = mo.m01 / wsum;
    double oxx = mo.mu20 / wsum + p->pixel_box_var;
    double oyy = mo.mu02 / wsum + p->pixel_box_var;
    double oxy = mo.mu11 / wsum;
    adaptive_moments(p, img_w, hn, wn, flux, &cx, &cy, &oxx, &oxy, &oyy);

    double mid = 0.5 * (oxx + oyy);
    double hd = 0.5 * (oxx - oyy);
    double rad = sqrt(hd * hd + oxy * oxy);
    double lam = mid + rad;
    if (lam < p->pixel_box_var) lam = p->pixel_box_var;
    double diameter = 4.0 * sqrt(lam) / p->diameter_divisor;
    double snr = matched ? fabs((double)m_peak) / p->matched_sigma : fabs((double)peak) / p->sigma;
    double s_snr = snr / (snr + p->k_sigma);
    double compact = area / (0.25 * 3.141592653589793 * diameter * diameter);
    if (compact > 1.0) compact = 1.0;
    o[1] = cx + q.px0;
    o[2] = cy + q.py0;
    o[3] = area;
    o[4] = diameter;
    o[5] = (double)peak;
    o[6] = snr;
    o[7] = s_snr * (0.5 + 0.5 * compact);
    o[8] = (double)x0;
    o[9] = (double)y0;
    o[10] = (double)bw;
    o[11] = (double)bh;
    return R_OK;
}

TBOT_API int tbot_measure_abi(void) { return TBOT_MEASURE_ABI; }

TBOT_API int tbot_measure_params_size(void) { return (int)sizeof(tbot_measure_params); }

/* Measure the candidates lids[0..n) (component labels, i.e. stats rows). out is n x 12 doubles: reject code,
 * u, v, area, diameter, contrast, snr, score, bbox x, y, w, h. Returns 0, or -1 when the parameters are
 * outside what the kernel has buffers for (the caller then takes the Python path). */
TBOT_API int tbot_measure(const tbot_measure_params *p, const int32_t *lids, int32_t n, double *out) {
    /* on the stack (~105 kB), so two detectors in two threads do not share them */
    int cbuf[MAX_PIX];
    float fa[MAX_PIX], fb[MAX_PIX], fc[MAX_PIX];
    if (!p || !lids || !out || p->h <= 0 || p->w <= 0 || !p->labels || !p->stats) return -1;
    if (p->moat_half > MAX_HALF || p->moat_half < 0 || p->adaptive_min_half > MAX_HALF) return -1;
    if (p->contrast16) {
        if (p->gsm) return -1;
    } else if (!p->gray || !p->bg) {
        return -1;
    }
    for (int32_t k = 0; k < n; k++) {
        double *o = out + (size_t)k * OUT_COLS;
        memset(o, 0, OUT_COLS * sizeof(double));
        o[0] = (double)measure_one(p, lids[k], o, cbuf, fa, fb, fc);
    }
    return 0;
}

/* ======================================================================= scenery filter (vision/clutter.py)
 * One frame of StaticFilter.filter after the comparison frames have been picked: pixel -> camera ray ->
 * world direction, the parallax-segment match against the old frames, the "was it in view then" test,
 * the protected tracks, the priority order and the focus split. Same statements as the Python; dot
 * products are summed left to right where NumPy may use BLAS, so a distance can differ from it in the last
 * bit (only a candidate exactly on a tolerance edge could be judged differently). */

#define TBOT_SCENERY_ABI 3
#define SCENERY_MAX_OLD 4    /* clutter.COMPARE_FRAMES */
#define SCENERY_LISTED 24    /* clutter.STATIC_LISTED */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { RAYS_GIVEN = -1, RAYS_FISHEYE = 0, RAYS_PINHOLE = 1, RAYS_PINHOLE_IDEAL = 2 };
enum { ARR_F64 = 0, ARR_I64 = 1, ARR_I32 = 2 };

typedef struct {
    int32_t n;                  /* candidates */
    int32_t model;              /* RAYS_*: how rays are made from uv (RAYS_GIVEN: read from rays) */
    const double *uv;           /* (n,2) full-resolution pixels */
    const double *rays;         /* (n,3) camera-frame unit rays when model == RAYS_GIVEN */
    double fx, fy, cx, cy, skew;
    double k[8];                /* fisheye k1..k4, or pinhole k1 k2 p1 p2 k3 k4 k5 k6 */
    int32_t newton_iters, undistort_iters;
    double newton_tol;
    const double *R_now;        /* 3x3 row-major, camera -> world, this frame */
    int32_t n_old;              /* comparison frames (each with at least one direction), oldest first */
    int32_t old_m[SCENERY_MAX_OLD];
    const double *old_R[SCENERY_MAX_OLD];
    const double *old_dirs[SCENERY_MAX_OLD];   /* (m,3) world directions */
    double tol0, turn_tol, lever, lam_max;
    double view[4];             /* x0, x1, y0, y1 on the tangent plane of the camera frame */
    const void *areas, *widths, *heights;      /* (n) or NULL, element types ARR_* */
    int32_t areas_t, widths_t, heights_t;
    double size_tol;            /* SIZE_TOL * ifov: extra tolerance per pixel of a candidate's extent */
    const double *protect;      /* (n_protect,3) unit world directions */
    int32_t n_protect;
    double protect_cos;
    const double *focus;        /* (n_focus,3) unit world directions, the main target first */
    const double *focus_cos;    /* (n_focus) */
    int32_t n_focus;
    int32_t others_max;
    /* scenery drift (StaticFilter._update_drift): estimated here from this frame's votes and applied */
    int32_t drift_on;
    double ifov;                /* rad per px at the centre */
    int64_t t_now;
    int64_t old_t[SCENERY_MAX_OLD];
    double drift_bin, drift_max, drift_peak_ratio, drift_gain;
    int32_t drift_box, drift_min_votes;
    double drift_x, drift_y;    /* in/out: px/s along the camera x / y */
    int32_t drift_updates;      /* in/out */
    int32_t drift_votes;        /* out: votes in this frame's peak, 0 = no estimate from this frame */
    int32_t *drift_grid;        /* scratch, side * side with side = 2 * round(drift_max / drift_bin) + 1 */
    int32_t *drift_ii;          /* scratch, (side + 1)^2 */
    int32_t drift_side;
    /* outputs */
    double *dirs;               /* (n,3) world directions */
    intptr_t *keep;             /* (n) candidates to keep, best first: keep[0..n_keep) */
    uint8_t *flags;             /* (n) bit 0 scenery, bit 1 new (not in view in the newest comparison frame) */
    double *static_px;          /* (SCENERY_LISTED,2) the first scenery candidates, uv rounded to 0.1 px */
    int32_t n_keep, n_static, n_new, n_focus_kept, n_listed;
} tbot_scenery_params;

static double arr_at(const void *a, int32_t t, intptr_t i) {
    switch (t) {
    case ARR_I64: return (double)((const int64_t *)a)[i];
    case ARR_I32: return (double)((const int32_t *)a)[i];
    default: return ((const double *)a)[i];
    }
}

/* CameraModel.ray_xyz */
static void ray_of(const tbot_scenery_params *p, double u, double v, double *r) {
    if (p->model == RAYS_FISHEYE) {
        double y = (v - p->cy) / p->fy;
        double x = (u - p->cx) / p->fx - p->skew * y;
        double theta_d = hypot(x, y);
        double k1 = p->k[0], k2 = p->k[1], k3 = p->k[2], k4 = p->k[3];
        double theta = fmin(fmax(theta_d, 0.0), M_PI);
        for (int it = 0; it < p->newton_iters; it++) {
            double t2 = theta * theta;
            double f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d;
            double df = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
            if (fabs(df) < 1e-12) df = 1e-12;
            double step = f / df;
            theta = fmin(fmax(theta - step, 0.0), M_PI);
            if (fabs(step) < p->newton_tol) break;
        }
        double scale = theta_d > 0.0 ? sin(theta) / theta_d : 0.0;
        r[0] = x * scale;
        r[1] = y * scale;
        r[2] = cos(theta);
        return;
    }
    double x0 = (u - p->cx) / p->fx, y0 = (v - p->cy) / p->fy;
    double x = x0, y = y0;
    if (p->model == RAYS_PINHOLE) {
        double k1 = p->k[0], k2 = p->k[1], p1 = p->k[2], p2 = p->k[3];
        double k3 = p->k[4], k4 = p->k[5], k5 = p->k[6], k6 = p->k[7];
        for (int it = 0; it < p->undistort_iters; it++) {
            double r2 = x * x + y * y;
            double icdist = (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2) / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
            if (icdist < 0.0) {
                x = x0;
                y = y0;
                break;
            }
            double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            x = (x0 - dx) * icdist;
            y = (y0 - dy) * icdist;
        }
    }
    double norm = sqrt(x * x + y * y + 1.0);
    r[0] = x / norm;
    r[1] = y / norm;
    r[2] = 1.0 / norm;
}

static inline double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/* order of np.lexsort((-fill, new)): new ones last, then the larger fill first; stable */
static int before(const uint8_t *flags, const double *fill, intptr_t a, intptr_t b) {
    int na = (flags[a] >> 1) & 1, nb = (flags[b] >> 1) & 1;
    if (na != nb) return na < nb;
    return -fill[a] < -fill[b];
}

static void merge_sort(intptr_t *idx, intptr_t *tmp, int32_t n, const uint8_t *flags, const double *fill) {
    for (int32_t width = 1; width < n; width *= 2) {
        for (int32_t lo = 0; lo < n; lo += 2 * width) {
            int32_t mid = lo + width < n ? lo + width : n;
            int32_t hi = lo + 2 * width < n ? lo + 2 * width : n;
            int32_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) tmp[k++] = before(flags, fill, idx[j], idx[i]) ? idx[j++] : idx[i++];
            while (i < mid) tmp[k++] = idx[i++];
            while (j < hi) tmp[k++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)n * sizeof(intptr_t));
    }
}

/* StaticFilter._update_drift: vote (c - d) / age of every pair of this frame and an old one on a grid of
 * drift_bin px/s, take the densest (2 box + 1)^2 window, fold its centroid into drift_x / drift_y.
 * Returns -1 when out of memory. */
static int update_drift(tbot_scenery_params *p) {
    const double *R = p->R_now;
    double X0 = R[0], X1 = R[3], X2 = R[6], Y0 = R[1], Y1 = R[4], Y2 = R[7];
    int32_t nb = (int32_t)nearbyint(p->drift_max / p->drift_bin);
    int32_t side = 2 * nb + 1;
    int64_t total = 0;
    p->drift_votes = 0;
    if (side != p->drift_side || !p->drift_grid || !p->drift_ii) return -1;
    int32_t *grid = p->drift_grid;
    memset(grid, 0, (size_t)side * side * sizeof(int32_t));
    double *cxy = (double *)malloc((size_t)p->n * 2 * sizeof(double));
    if (!cxy) return -1;
    for (int32_t i = 0; i < p->n; i++) {
        const double *c = p->dirs + 3 * (size_t)i;
        cxy[2 * i] = c[0] * X0 + c[1] * X1 + c[2] * X2;
        cxy[2 * i + 1] = c[0] * Y0 + c[1] * Y1 + c[2] * Y2;
    }
    for (int f = 0; f < p->n_old; f++) {
        double inv = 1.0 / ((double)(p->t_now - p->old_t[f]) * 1e-9 * p->ifov);
        const double *D = p->old_dirs[f];
        for (int32_t m = 0; m < p->old_m[f]; m++) {
            const double *d = D + 3 * (size_t)m;
            double dx = d[0] * X0 + d[1] * X1 + d[2] * X2;
            double dy = d[0] * Y0 + d[1] * Y1 + d[2] * Y2;
            for (int32_t i = 0; i < p->n; i++) {
                double rx = (cxy[2 * i] - dx) * inv, ry = (cxy[2 * i + 1] - dy) * inv;
                if (!(fabs(rx) < p->drift_max && fabs(ry) < p->drift_max)) continue;
                int64_t ix = (int64_t)floor(rx / p->drift_bin + 0.5) + nb;
                int64_t iy = (int64_t)floor(ry / p->drift_bin + 0.5) + nb;
                grid[iy * side + ix]++;
                total++;
            }
        }
    }
    free(cxy);
    if (total < p->drift_min_votes) return 0;
    int32_t b = p->drift_box, k = 2 * b + 1, nw = side - 2 * b, s1 = side + 1;
    int64_t peak = -1;
    int32_t wy = 0, wx = 0;
    int32_t *ii = p->drift_ii;   /* integral image of the grid */
    memset(ii, 0, (size_t)s1 * sizeof(int32_t));
    for (int32_t y = 0; y < side; y++) {
        ii[(y + 1) * s1] = 0;
        int32_t row = 0;
        for (int32_t x = 0; x < side; x++) {
            row += grid[y * side + x];
            ii[(y + 1) * s1 + x + 1] = ii[y * s1 + x + 1] + row;
        }
    }
    for (int32_t y = 0; y < nw; y++) {
        for (int32_t x = 0; x < nw; x++) {
            int64_t sum = (int64_t)ii[(y + k) * s1 + x + k] - ii[y * s1 + x + k] - ii[(y + k) * s1 + x] + ii[y * s1 + x];
            if (sum > peak) {
                peak = sum;
                wy = y;
                wx = x;
            }
        }
    }
    double even = (double)(total * k * k) / (double)(side * side);
    if (peak < p->drift_min_votes || (double)peak < p->drift_peak_ratio * even) return 0;
    double sy = 0.0, sx = 0.0;
    for (int32_t yy = 0; yy < k; yy++) {
        for (int32_t xx = 0; xx < k; xx++) {
            double v = (double)grid[(wy + yy) * side + (wx + xx)];
            sy += v * (double)(yy - b);
            sx += v * (double)(xx - b);
        }
    }
    double my = sy / (double)peak + (double)(wy + b - nb);
    double mx = sx / (double)peak + (double)(wx + b - nb);
    double ux = mx * p->drift_bin, uy = my * p->drift_bin;
    if (p->drift_updates == 0) {
        p->drift_x = ux;
        p->drift_y = uy;
    } else {
        p->drift_x = p->drift_x + p->drift_gain * (ux - p->drift_x);
        p->drift_y = p->drift_y + p->drift_gain * (uy - p->drift_y);
    }
    p->drift_updates++;
    p->drift_votes = (int32_t)peak;
    return 0;
}

TBOT_API int tbot_scenery_abi(void) { return TBOT_SCENERY_ABI; }

TBOT_API int tbot_scenery_params_size(void) { return (int)sizeof(tbot_scenery_params); }

/* Returns 0, or -1 for parameters the kernel does not take (the caller then runs the Python). */
TBOT_API int tbot_scenery(tbot_scenery_params *p) {
    if (!p) return -1;
    int32_t n = p->n;
    if (n < 0 || !p->R_now || !p->uv || !p->dirs || !p->keep || !p->flags || !p->static_px) return -1;
    if (p->n_old < 0 || p->n_old > SCENERY_MAX_OLD) return -1;
    if (p->model == RAYS_GIVEN && !p->rays) return -1;
    if ((p->n_protect > 0 && !p->protect) || (p->n_focus > 0 && (!p->focus || !p->focus_cos))) return -1;
    for (int f = 0; f < p->n_old; f++)
        if (!p->old_R[f] || !p->old_dirs[f] || p->old_m[f] <= 0) return -1;
    p->n_keep = p->n_static = p->n_new = p->n_focus_kept = p->n_listed = 0;
    if (n == 0) return 0;

    const double *R = p->R_now;
    double ray[3];
    for (int32_t i = 0; i < n; i++) {
        const double *r = ray;
        if (p->model == RAYS_GIVEN) r = p->rays + 3 * (size_t)i;
        else ray_of(p, p->uv[2 * i], p->uv[2 * i + 1], ray);
        double *d = p->dirs + 3 * (size_t)i;   /* rays @ R_now.T */
        d[0] = r[0] * R[0] + r[1] * R[1] + r[2] * R[2];
        d[1] = r[0] * R[3] + r[1] * R[4] + r[2] * R[5];
        d[2] = r[0] * R[6] + r[1] * R[7] + r[2] * R[8];
    }

    int sized = p->widths && p->heights;
    double *shifted = NULL;   /* the old directions moved by the drift, when there is one */
    const double *odirs[SCENERY_MAX_OLD];
    for (int f = 0; f < p->n_old; f++) odirs[f] = p->old_dirs[f];
    p->drift_votes = 0;
    if (p->n_old > 0 && p->drift_on) {
        if (update_drift(p) != 0) return -1;
        double ux = p->drift_x, uy = p->drift_y;
        if (ux != 0.0 || uy != 0.0) {
            size_t total = 0;
            for (int f = 0; f < p->n_old; f++) total += (size_t)p->old_m[f];
            shifted = (double *)malloc(total * 3 * sizeof(double));
            if (!shifted) return -1;
            double X[3] = {R[0], R[3], R[6]}, Y[3] = {R[1], R[4], R[7]};
            double *w = shifted;
            for (int f = 0; f < p->n_old; f++) {
                double a = (double)(p->t_now - p->old_t[f]) * 1e-9 * p->ifov;
                double sx = ux * a, sy = uy * a;
                double s0 = sx * X[0] + sy * Y[0], s1 = sx * X[1] + sy * Y[1], s2 = sx * X[2] + sy * Y[2];
                const double *D = p->old_dirs[f];
                odirs[f] = w;
                for (int32_t m = 0; m < p->old_m[f]; m++, w += 3) {
                    double d0 = D[3 * m] + s0, d1 = D[3 * m + 1] + s1, d2 = D[3 * m + 2] + s2;
                    double nrm = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                    w[0] = d0 / nrm;
                    w[1] = d1 / nrm;
                    w[2] = d2 / nrm;
                }
            }
        }
    }
    if (p->n_old > 0) {
        double g[SCENERY_MAX_OLD][3], gg[SCENERY_MAX_OLD], tol[SCENERY_MAX_OLD];
        double zx = R[2], zy = R[5], zz = R[8];   /* optical axis in the world */
        for (int f = 0; f < p->n_old; f++) {
            const double *F = p->old_R[f];
            double tr = 0.0;   /* trace(R_f^T R_now) */
            for (int q = 0; q < 9; q++) tr += F[q] * R[q];
            double turn = acos(fmin(1.0, fmax(-1.0, 0.5 * (tr - 1.0))));
            double dx = p->lever * (zx - F[2]), dy = p->lever * (zy - F[5]), dz = p->lever * (zz - F[8]);
            double along = dx * zx + dy * zy + dz * zz;
            g[f][0] = -(dx - along * zx);
            g[f][1] = -(dy - along * zy);
            g[f][2] = -(dz - along * zz);
            gg[f] = dot3(g[f], g[f]);
            tol[f] = p->tol0 + p->turn_tol * turn;
        }
        const double *Rt = p->old_R[p->n_old - 1];
        for (int32_t i = 0; i < n; i++) {
            const double *c = p->dirs + 3 * (size_t)i;
            double st = 0.0;
            if (sized) {
                double w = arr_at(p->widths, p->widths_t, i), h = arr_at(p->heights, p->heights_t, i);
                st = p->size_tol * (w > h ? w : h);
            }
            int is_static = 0;
            for (int f = 0; f < p->n_old && !is_static; f++) {
                double cg = dot3(c, g[f]), ggf = gg[f] > 1e-12 ? gg[f] : 1e-12;
                double lim = tol[f] + st;
                lim *= lim;
                const double *D = odirs[f];
                for (int32_t m = 0; m < p->old_m[f]; m++) {
                    const double *dd = D + 3 * (size_t)m;
                    double e2 = 2.0 - 2.0 * dot3(c, dd);   /* |c - d|^2 for unit vectors */
                    if (e2 < 0.0) e2 = 0.0;
                    double eg = cg - dot3(dd, g[f]);       /* (c - d) . g */
                    double lam = eg / ggf;
                    lam = lam < 0.0 ? 0.0 : (lam > p->lam_max ? p->lam_max : lam);
                    e2 += lam * (lam * gg[f] - 2.0 * eg);  /* |c - d - lam g|^2 */
                    if (e2 <= lim) {
                        is_static = 1;
                        break;
                    }
                }
            }
            /* the direction in the newest comparison frame's camera: then = c @ Rt */
            double t0 = c[0] * Rt[0] + c[1] * Rt[3] + c[2] * Rt[6];
            double t1 = c[0] * Rt[1] + c[1] * Rt[4] + c[2] * Rt[7];
            double t2 = c[0] * Rt[2] + c[1] * Rt[5] + c[2] * Rt[8];
            double tx = t0 / t2, ty = t1 / t2;
            int in_view = t2 > 1e-6 && tx >= p->view[0] && tx <= p->view[1] && ty >= p->view[2] && ty <= p->view[3];
            p->flags[i] = (uint8_t)(is_static | (in_view ? 0 : 2));
        }
    } else {
        for (int32_t i = 0; i < n; i++) p->flags[i] = 2;   /* no history: nothing is known yet */
    }
    free(shifted);

    if (p->n_protect > 0) {
        for (int32_t i = 0; i < n; i++) {
            if (!(p->flags[i] & 1)) continue;
            const double *c = p->dirs + 3 * (size_t)i;
            for (int32_t j = 0; j < p->n_protect; j++) {
                if (dot3(c, p->protect + 3 * (size_t)j) >= p->protect_cos) {
                    p->flags[i] &= (uint8_t)~1u;
                    break;
                }
            }
        }
    }

    int32_t nk = 0;
    for (int32_t i = 0; i < n; i++) {
        if (p->flags[i] & 1) {
            if (p->n_listed < SCENERY_LISTED) {   /* np.round(uv, 1): rint(x * 10) / 10 */
                p->static_px[2 * p->n_listed] = nearbyint(p->uv[2 * i] * 10.0) / 10.0;
                p->static_px[2 * p->n_listed + 1] = nearbyint(p->uv[2 * i + 1] * 10.0) / 10.0;
                p->n_listed++;
            }
            p->n_static++;
        } else {
            if (p->flags[i] & 2) p->n_new++;
            p->keep[nk++] = i;
        }
    }

    if (nk > 1) {
        double *fill = (double *)malloc((size_t)n * sizeof(double));
        intptr_t *tmp = (intptr_t *)malloc((size_t)nk * sizeof(intptr_t));
        if (!fill || !tmp) {
            free(fill);
            free(tmp);
            return -1;
        }
        int filled = p->areas && sized;
        for (int32_t k = 0; k < nk; k++) {
            intptr_t i = p->keep[k];
            if (filled) {
                double a = arr_at(p->areas, p->areas_t, i);
                double box = arr_at(p->widths, p->widths_t, i) * arr_at(p->heights, p->heights_t, i);
                if (box < 1.0) box = 1.0;
                fill[i] = a / box * sqrt(a);
            } else {
                fill[i] = 1.0;
            }
        }
        merge_sort(p->keep, tmp, nk, p->flags, fill);
        free(fill);
        free(tmp);
    }

    if (p->n_focus > 0 && nk > 0) {
        intptr_t *others = (intptr_t *)malloc((size_t)nk * sizeof(intptr_t));
        if (!others) return -1;
        int32_t nn = 0, no = 0, main_hit = 0;
        for (int32_t k = 0; k < nk; k++) {
            intptr_t i = p->keep[k];
            const double *c = p->dirs + 3 * (size_t)i;
            int near = 0;
            for (int32_t j = 0; j < p->n_focus; j++) {   /* the main target is j = 0, tried first */
                if (dot3(c, p->focus + 3 * (size_t)j) >= p->focus_cos[j]) {
                    near = 1;
                    if (j == 0) main_hit = 1;
                    break;
                }
            }
            if (near) p->keep[nn++] = i;
            else others[no++] = i;
        }
        if (p->others_max > 0 && main_hit && no > p->others_max) no = p->others_max;
        memcpy(p->keep + nn, others, (size_t)no * sizeof(intptr_t));
        free(others);
        p->n_focus_kept = nn;
        nk = nn + no;
    }
    p->n_keep = nk;
    return 0;
}

/* ============================================================================================ tracker
 * The 9-state filter of vision/tracker3d.py (world [p, v, a], metres): one track's time step and one
 * measurement update. In Python each costs a dozen to two dozen small NumPy calls (15-50 us apiece on a Pi 3B:
 * 0.2 ms a predict, 0.66 ms an update; range day 2026-11-19, six tracks on tree specks: 8.7 ms a frame of
 * tracking, 30 frames a second). The Python functions remain the reference (tests compare the two). */
#define TBOT_TRACK_ABI 1

TBOT_API int tbot_track_abi(void) { return TBOT_TRACK_ABI; }

/* s_out = F s, S_out = F S F^T + Q for a step dt (white-jerk PSD q): F and Q as tracker3d._transition9 /
 * _noise9. Row-major 9x9. The outputs may not alias the inputs. */
TBOT_API void tbot_track_predict(const double *s, const double *S, double dt, double q, double *s_out,
                                 double *S_out) {
    const double h = 0.5 * dt * dt;
    double FS[81];
    for (int i = 0; i < 3; i++) {
        s_out[i] = s[i] + dt * s[3 + i] + h * s[6 + i];
        s_out[3 + i] = s[3 + i] + dt * s[6 + i];
        s_out[6 + i] = s[6 + i];
        for (int k = 0; k < 9; k++) {
            FS[9 * i + k] = S[9 * i + k] + dt * S[9 * (3 + i) + k] + h * S[9 * (6 + i) + k];
            FS[9 * (3 + i) + k] = S[9 * (3 + i) + k] + dt * S[9 * (6 + i) + k];
            FS[9 * (6 + i) + k] = S[9 * (6 + i) + k];
        }
    }
    for (int r = 0; r < 9; r++) {
        for (int i = 0; i < 3; i++) {
            S_out[9 * r + i] = FS[9 * r + i] + dt * FS[9 * r + 3 + i] + h * FS[9 * r + 6 + i];
            S_out[9 * r + 3 + i] = FS[9 * r + 3 + i] + dt * FS[9 * r + 6 + i];
            S_out[9 * r + 6 + i] = FS[9 * r + 6 + i];
        }
    }
    const double dt2 = dt * dt, dt3 = dt2 * dt;
    const double b[3][3] = {{q * (dt3 * dt2 / 20.0), q * (dt2 * dt2 / 8.0), q * (dt3 / 6.0)},
                            {q * (dt2 * dt2 / 8.0), q * (dt3 / 3.0), q * (dt2 / 2.0)},
                            {q * (dt3 / 6.0), q * (dt2 / 2.0), q * dt}};
    for (int i = 0; i < 3; i++)
        for (int a = 0; a < 3; a++)
            for (int c = 0; c < 3; c++) S_out[9 * (3 * a + i) + 3 * c + i] += b[a][c];
}

/* Measurement update with m (2 or 3) rows that look at the position only: H = [H3 | 0], H3 m x 3 row-major,
 * innovation z, measurement variances r. Joseph form, symmetrised; the position is then held inside
 * [range_min, range_max] as in tracker3d._ekf_update. Returns 0, or -1 when the innovation covariance is
 * singular (nothing written: the caller takes the Python path). */
TBOT_API int tbot_track_update(const double *sp, const double *Sp, const double *H3, const double *z,
                               const double *r, int32_t m, double range_min, double range_max, double *s_out,
                               double *S_out) {
    if (m < 2 || m > 3) return -1;
    double PHt[27], Sm[9], inv[9], K[27], A[81], AS[81];
    for (int i = 0; i < 9; i++)
        for (int k = 0; k < m; k++)
            PHt[3 * i + k] = Sp[9 * i] * H3[3 * k] + Sp[9 * i + 1] * H3[3 * k + 1] + Sp[9 * i + 2] * H3[3 * k + 2];
    for (int a = 0; a < m; a++)
        for (int k = 0; k < m; k++)
            Sm[3 * a + k] = H3[3 * a] * PHt[k] + H3[3 * a + 1] * PHt[3 + k] + H3[3 * a + 2] * PHt[6 + k]
                            + (a == k ? r[a] : 0.0);
    if (m == 2) {
        const double a = Sm[0], b = Sm[1], c = Sm[3], d = Sm[4];
        const double det = a * d - b * c;
        if (det == 0.0 || !isfinite(det)) return -1;
        inv[0] = d / det; inv[1] = -b / det; inv[3] = -c / det; inv[4] = a / det;
    } else {
        const double a = Sm[0], b = Sm[1], c = Sm[2], d = Sm[3], e = Sm[4], f = Sm[5], g = Sm[6], hh = Sm[7],
                     i = Sm[8];
        const double c11 = e * i - f * hh, c12 = f * g - d * i, c13 = d * hh - e * g;
        const double det = a * c11 + b * c12 + c * c13;
        if (det == 0.0 || !isfinite(det)) return -1;
        inv[0] = c11 / det; inv[1] = (c * hh - b * i) / det; inv[2] = (b * f - c * e) / det;
        inv[3] = c12 / det; inv[4] = (a * i - c * g) / det;  inv[5] = (c * d - a * f) / det;
        inv[6] = c13 / det; inv[7] = (b * g - a * hh) / det; inv[8] = (a * e - b * d) / det;
    }
    for (int i = 0; i < 9; i++) {
        double ds = 0.0;
        for (int k = 0; k < m; k++) {
            double v = 0.0;
            for (int a = 0; a < m; a++) v += PHt[3 * i + a] * inv[3 * a + k];
            K[3 * i + k] = v;
            ds += v * z[k];
        }
        s_out[i] = sp[i] + ds;
    }
    for (int i = 0; i < 9; i++) {
        for (int k = 0; k < 9; k++) A[9 * i + k] = (i == k) ? 1.0 : 0.0;
        for (int c = 0; c < 3; c++) {
            double v = 0.0;
            for (int k = 0; k < m; k++) v += K[3 * i + k] * H3[3 * k + c];
            A[9 * i + c] -= v;
        }
    }
    for (int i = 0; i < 9; i++)
        for (int k = 0; k < 9; k++) {
            double v = 0.0;
            for (int j = 0; j < 9; j++) v += A[9 * i + j] * Sp[9 * j + k];
            AS[9 * i + k] = v;
        }
    for (int i = 0; i < 9; i++)
        for (int k = 0; k < 9; k++) {
            double v = 0.0;
            for (int j = 0; j < 9; j++) v += AS[9 * i + j] * A[9 * k + j];
            for (int a = 0; a < m; a++) v += K[3 * i + a] * r[a] * K[3 * k + a];
            S_out[9 * i + k] = v;
        }
    for (int i = 0; i < 9; i++)
        for (int k = i + 1; k < 9; k++) {
            const double v = 0.5 * (S_out[9 * i + k] + S_out[9 * k + i]);
            S_out[9 * i + k] = S_out[9 * k + i] = v;
        }
    const double rr = sqrt(s_out[0] * s_out[0] + s_out[1] * s_out[1] + s_out[2] * s_out[2]);
    if (!(range_min <= rr && rr <= range_max) && rr > 0.0) {
        const double want = rr < range_min ? range_min : (rr > range_max ? range_max : rr);
        const double g = want / rr;
        s_out[0] *= g; s_out[1] *= g; s_out[2] *= g;
    }
    return 0;
}
